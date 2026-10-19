//! Pull mode for transmit paths: the output asks, the source supplies; bursts with start
//! times.

use bytemuck::Pod;
use sdrk::Sample;
use sdrk_blocks::PortType;
use sdrk_device::{BlockFlags, BlockMeta, SampleIndex};

use crate::error::{Error, Result};
use crate::latency::ceil_delay;
use crate::model::{DropPolicy, UnderrunPolicy};
use crate::runner::{
    Burst, Dst, InputData, OutKind, Pipeline, at, at_mut, elems_of, internal, name_of, sample_port,
    samples_of,
};
use crate::runner_time::map_along;
use crate::time::{Event, EventKind, OutputId};

/// The error type of a [`PullSource`].
pub type PullError = crate::Error;

/// Supplies input data when a transmit path asks for it.
///
/// ```
/// use sdrk_pipeline::{InputData, PullError, PullSource};
/// struct Silence([f32; 64]);
/// impl PullSource for Silence {
///     fn provide(&mut self, _input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> {
///         let n = max_samples.min(self.0.len());
///         Ok(InputData::F32(&self.0[..n]))
///     }
/// }
/// let mut s = Silence([0.0; 64]);
/// assert_eq!(s.provide("audio", 10)?.len(), 10);
/// # Ok::<(), PullError>(())
/// ```
pub trait PullSource {
    /// Returns at most `max_samples` elements for the input `input` (bytes for a raw input).
    /// An empty result means that the source is exhausted.
    fn provide(&mut self, input: &str, max_samples: usize) -> Result<InputData<'_>, PullError>;
}

/// What a pull call delivers before it hits a boundary.
struct Delivered {
    written: usize,
    start: Option<u64>,
    flags: BlockFlags,
    tx_time_ns: Option<i64>,
}

pub(crate) fn zero_input(ty: PortType, samples: usize) -> ZeroInput {
    match ty {
        PortType::C32 => ZeroInput::C32(vec![sdrk::C32::default(); samples]),
        PortType::CQ15 => ZeroInput::CQ15(vec![sdrk::CQ15::default(); samples]),
        PortType::F32 => ZeroInput::F32(vec![0.0; samples]),
        PortType::Q15 => ZeroInput::Q15(vec![sdrk::Q15::default(); samples]),
        _ => ZeroInput::Raw(vec![0; samples]),
    }
}

pub(crate) enum ZeroInput {
    Raw(Vec<u8>),
    C32(Vec<sdrk::C32>),
    CQ15(Vec<sdrk::CQ15>),
    F32(Vec<f32>),
    Q15(Vec<sdrk::Q15>),
}

impl Default for ZeroInput {
    fn default() -> Self {
        Self::Raw(Vec::new())
    }
}

impl ZeroInput {
    fn data(&self, n: usize) -> InputData<'_> {
        match self {
            Self::Raw(v) => InputData::Raw(v.get(..n).unwrap_or(v)),
            Self::C32(v) => InputData::C32(v.get(..n).unwrap_or(v)),
            Self::CQ15(v) => InputData::CQ15(v.get(..n).unwrap_or(v)),
            Self::F32(v) => InputData::F32(v.get(..n).unwrap_or(v)),
            Self::Q15(v) => InputData::Q15(v.get(..n).unwrap_or(v)),
        }
    }
}

impl Pipeline {
    /// Produces `out.len()` bytes of a raw output (whole samples only), asking `source` for
    /// input as needed.
    ///
    /// Surplus production stays buffered for the next call. When the source is exhausted, the
    /// output setting `underrun` decides: `zeros` (the default for raw outputs) fills the rest
    /// with silence and reports `Underrun`, `short` returns fewer bytes. The returned
    /// [`BlockMeta`] starts at the output index of the first sample; a block never spans a
    /// burst boundary and carries `BURST_START` and `BURST_END` where a burst begins and ends.
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, InputData, PullError, PullSource, Registry};
    /// struct Silence;
    /// impl PullSource for Silence {
    ///     fn provide(&mut self, _input: &str, max: usize) -> Result<InputData<'_>, PullError> {
    ///         static ZEROS: [f32; 4800] = [0.0; 4800];
    ///         Ok(InputData::F32(&ZEROS[..max]))
    ///     }
    /// }
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_transmitter.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// let mut bytes = [0u8; 1000];
    /// let (n, meta) = p.pull("rf", &mut bytes, &mut Silence)?;
    /// assert_eq!((n, meta.start.0), (1000, 0));
    /// let (_, meta) = p.pull("rf", &mut bytes, &mut Silence)?;
    /// assert_eq!(meta.start.0, 500); // two bytes per sample
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownOutput`, `WrongType` (the output is not a raw stream), errors of the source and
    /// of `push`.
    pub fn pull(
        &mut self,
        output: &str,
        out: &mut [u8],
        source: &mut dyn PullSource,
    ) -> Result<(usize, BlockMeta)> {
        self.pull_core(output, out, source, PortType::Raw)
    }

    /// Like [`Pipeline::pull`] for an output of samples of type `T`.
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, InputData, PullError, PullSource, Registry};
    /// struct Steady;
    /// impl PullSource for Steady {
    ///     fn provide(&mut self, _input: &str, max: usize) -> Result<InputData<'_>, PullError> {
    ///         static DATA: [f32; 4096] = [0.5; 4096];
    ///         Ok(InputData::F32(&DATA[..max.min(4096)]))
    ///     }
    /// }
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/resample_audio.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// let mut out = [0f32; 441];
    /// let (n, _) = p.pull_samples::<f32>("audio_out", &mut out, &mut Steady)?;
    /// assert_eq!(n, 441);
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// As `pull`; `WrongType` if `T` differs from the type of the output.
    pub fn pull_samples<T: Sample>(
        &mut self,
        output: &str,
        out: &mut [T],
        source: &mut dyn PullSource,
    ) -> Result<(usize, BlockMeta)> {
        self.pull_core(output, out, source, sample_port(T::TYPE))
    }

    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn pull_core<T: Pod>(
        &mut self,
        output: &str,
        out: &mut [T],
        source: &mut dyn PullSource,
        wanted: PortType,
    ) -> Result<(usize, BlockMeta)> {
        let oi = self.output_pos(output)?;
        let (elems, ty) = match &at(&self.outputs, oi)?.kind {
            OutKind::Stream { ty, queue, .. } => (queue.elems, *ty),
            OutKind::Frames(_) => {
                return Err(Error::WrongType {
                    name: output.to_string(),
                    expected: "frames",
                    got: "a stream",
                });
            }
        };
        if ty != wanted {
            return Err(Error::WrongType {
                name: output.to_string(),
                expected: name_of(ty),
                got: name_of(wanted),
            });
        }
        let group = if ty == PortType::Raw {
            usize::try_from(elems.0).unwrap_or(1).max(1)
        } else {
            1
        };
        let want = out.len() - out.len() % group;
        let mut d = Delivered {
            written: 0,
            start: None,
            flags: BlockFlags::empty(),
            tx_time_ns: None,
        };
        let mut exhausted = false;
        while d.written < want {
            if at(&self.outputs, oi)?.burst == Burst::Idle {
                break;
            }
            if self.deliver(oi, out, want, &mut d)? {
                break;
            }
            if self.queued(oi)? > 0 {
                continue;
            }
            if matches!(at(&self.outputs, oi)?.burst, Burst::Ending { .. }) {
                // Everything of the burst has been delivered.
                self.finish_burst(oi)?;
                d.flags.insert(BlockFlags::BURST_END);
                break;
            }
            if !self.produce(oi, (want - d.written).div_ceil(group), source)? {
                exhausted = true;
                break;
            }
        }
        let first = match d.start {
            Some(s) => s,
            None => self.next_index_of(oi)?,
        };
        if exhausted && d.written < want {
            let o = at(&self.outputs, oi)?;
            if o.underrun == UnderrunPolicy::Zeros {
                let at_idx = first + samples_of(d.written as u64, elems);
                let silence = core::mem::take(&mut at_mut(&mut self.outputs, oi)?.silence);
                fill_silence(out, d.written, want, &silence);
                at_mut(&mut self.outputs, oi)?.silence = silence;
                d.written = want;
                d.flags.insert(BlockFlags::UNDERRUN);
                self.events.push(Event {
                    output: OutputId::new(oi),
                    index: at_idx,
                    kind: EventKind::Underrun,
                });
            }
        }
        let meta = BlockMeta {
            start: SampleIndex(first),
            flags: d.flags,
            anchor: None,
            tx_time_ns: d.tx_time_ns,
        };
        Ok((d.written, meta))
    }

    /// Elements waiting in the queue of a stream output.
    fn queued(&self, oi: usize) -> Result<usize> {
        match &at(&self.outputs, oi)?.kind {
            OutKind::Stream { queue, .. } => Ok(queue.len()),
            OutKind::Frames(_) => Ok(0),
        }
    }

    /// Index of the next sample to be read (the queue front, or the next to come).
    fn next_index_of(&self, oi: usize) -> Result<u64> {
        match &at(&self.outputs, oi)?.kind {
            OutKind::Stream { queue, .. } => Ok(queue.first_idx()),
            OutKind::Frames(_) => Ok(0),
        }
    }

    /// Index of the next sample that the output will receive from the graph.
    fn produced_index_of(&self, oi: usize) -> Result<u64> {
        match &at(&self.outputs, oi)?.kind {
            OutKind::Stream { src, .. } => Ok(at(&self.streams, *src)?.next_idx()),
            OutKind::Frames(_) => Ok(0),
        }
    }

    /// Copies queued elements into `out` up to the next boundary. Returns true if the block
    /// has to end here.
    fn deliver<T: Pod>(
        &mut self,
        oi: usize,
        out: &mut [T],
        want: usize,
        d: &mut Delivered,
    ) -> Result<bool> {
        let o = at_mut(&mut self.outputs, oi)?;
        let policy = o.policy;
        let burst = o.burst;
        let OutKind::Stream { queue, .. } = &mut o.kind else {
            return Err(internal("pull on a frames output"));
        };
        let avail = queue.len();
        if avail == 0 {
            return Ok(false);
        }
        let first = queue.first_idx();
        let elems = queue.elems;
        let mut limit = avail.min(want - d.written);
        let mut stop = false;
        let mut finished = false;
        match burst {
            Burst::Armed {
                at: start,
                tx_time_ns,
            } => {
                if first < start {
                    let before =
                        usize::try_from(elems_of(start - first, elems)).unwrap_or(usize::MAX);
                    limit = limit.min(before);
                    stop = true;
                } else if d.written == 0 {
                    d.flags.insert(BlockFlags::BURST_START);
                    d.tx_time_ns = tx_time_ns;
                    o.burst = Burst::Active;
                }
            }
            Burst::Ending { end } => {
                let until = usize::try_from(elems_of(end.saturating_sub(first), elems))
                    .unwrap_or(usize::MAX);
                if until <= limit {
                    limit = until;
                    finished = true;
                }
            }
            _ => {}
        }
        if limit > 0 {
            if d.start.is_none() {
                d.start = Some(first);
            }
            let dst = out
                .get_mut(d.written..d.written + limit)
                .ok_or_else(|| internal("pull output range"))?;
            if !queue.buf.copy_to(queue.start, limit, dst) {
                return Err(internal("output queue read"));
            }
            queue.advance(limit);
            d.written += limit;
        }
        if finished {
            d.flags.insert(BlockFlags::BURST_END);
            self.finish_burst(oi)?;
            return Ok(true);
        }
        if policy == DropPolicy::Error && limit > 0 {
            self.run()?;
        }
        Ok(stop)
    }

    /// Asks the source for input for `missing` more output elements and pushes it. Returns
    /// false if the source is exhausted.
    fn produce(&mut self, oi: usize, missing: usize, source: &mut dyn PullSource) -> Result<bool> {
        let ii = at(&self.outputs, oi)?.input;
        let inp = at(&self.inputs, ii)?;
        let (num, den) = inp
            .reach
            .iter()
            .find(|r| r.output == oi)
            .map_or((1, 1), |r| (r.num, r.den));
        let min_input = inp.min_input;
        let n = (missing as u128 * den)
            .div_ceil(num.max(1))
            .saturating_add(min_input as u128);
        let n = usize::try_from(n)
            .unwrap_or(usize::MAX)
            .min(inp.max_len)
            .max(1);
        let data = source.provide(&inp.name, n)?;
        if data.is_empty() {
            return Ok(false);
        }
        if data.len() > n {
            return Err(Error::ChunkTooLarge {
                len: data.len(),
                max: n,
            });
        }
        self.check_push(ii, &data)?;
        self.ingest(ii, &data)?;
        self.run()?;
        Ok(true)
    }

    /// Starts a burst: the next sample the output produces is its first sample.
    ///
    /// A block returned by `pull` starts at that sample and carries `BURST_START` with
    /// `tx_time_ns`. A `BurstStart` event is reported at the same index.
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, Error, EventKind, Registry};
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_transmitter.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// p.begin_burst("rf", Some(42))?;
    /// let event = p.poll_event().unwrap();
    /// assert_eq!((event.index, event.kind), (0, EventKind::BurstStart { tx_time_ns: Some(42) }));
    /// // A second burst cannot start while one is active.
    /// assert!(matches!(p.begin_burst("rf", None), Err(Error::InvalidState(_))));
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownOutput`, `InvalidState` (a burst is active, or its tail has not been pulled).
    pub fn begin_burst(&mut self, output: &str, tx_time_ns: Option<i64>) -> Result<()> {
        let oi = self.output_pos(output)?;
        match at(&self.outputs, oi)?.burst {
            Burst::Continuous | Burst::Idle => {}
            _ => {
                return Err(Error::InvalidState(format!(
                    "a burst of `{output}` is active; end it and pull its tail first"
                )));
            }
        }
        let start = self.produced_index_of(oi)?;
        at_mut(&mut self.outputs, oi)?.burst = Burst::Armed {
            at: start,
            tx_time_ns,
        };
        self.events.push(Event {
            output: OutputId::new(oi),
            index: start,
            kind: EventKind::BurstStart { tx_time_ns },
        });
        Ok(())
    }

    /// Ends the burst: feeds `ceil(D_in)` zero samples to the input (`D_in` is the group delay
    /// of the path in input samples) so that the filter tails come out, and marks the last
    /// sample of the burst. It lies `ceil(lead_time)` samples after the sample that the last
    /// real input sample leads to. `pull` returns the tail, the block that ends there carries
    /// `BURST_END`, and the nodes of the path are reset. Nothing is produced until the next
    /// `begin_burst`.
    ///
    /// Pull the data of the burst before calling this: the zero samples need room in the
    /// output.
    ///
    /// ```
    /// use sdrk_device::BlockFlags;
    /// use sdrk_pipeline::{BuildOptions, InputData, PullError, PullSource, Registry};
    /// struct Tone {
    ///     left: usize,
    ///     data: [f32; 4800],
    /// }
    /// impl PullSource for Tone {
    ///     fn provide(&mut self, _input: &str, max: usize) -> Result<InputData<'_>, PullError> {
    ///         let n = max.min(self.left);
    ///         self.left -= n;
    ///         Ok(InputData::F32(&self.data[..n]))
    ///     }
    /// }
    /// let mut p = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/fm_transmitter.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// let mut source = Tone { left: 200, data: [0.25; 4800] };
    /// p.begin_burst("rf", Some(1_700_000_000_000_000_000))?;
    /// let mut bytes = [0u8; 4000];
    /// let (_, meta) = p.pull("rf", &mut bytes, &mut source)?;
    /// assert!(meta.flags.contains(BlockFlags::BURST_START));
    /// assert_eq!(meta.tx_time_ns, Some(1_700_000_000_000_000_000));
    /// p.end_burst("rf")?;
    /// let (tail, meta) = p.pull("rf", &mut bytes, &mut source)?;
    /// assert!(tail > 0 && meta.flags.contains(BlockFlags::BURST_END));
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownOutput`, `InvalidState` (no burst is active), `OutputFull` and the errors of
    /// `push`.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    pub fn end_burst(&mut self, output: &str) -> Result<()> {
        let oi = self.output_pos(output)?;
        let o = at(&self.outputs, oi)?;
        match o.burst {
            Burst::Armed { .. } => {
                // Nothing was produced yet: cancel.
                at_mut(&mut self.outputs, oi)?.burst = Burst::Idle;
                return Ok(());
            }
            Burst::Active => {}
            _ => {
                return Err(Error::InvalidState(format!(
                    "`{output}` has no active burst"
                )));
            }
        }
        let (ii, lead, lead_in) = (o.input, o.lead, o.lead_in);
        let inp = at(&self.inputs, ii)?;
        let (ty, max_len, stream) = (inp.ty, inp.max_len, inp.stream);
        let elems = at(&self.streams, stream)?.elems;
        let in_end = at(&self.streams, stream)?.next_idx();
        let mut left = ceil_delay(lead_in);
        while left > 0 {
            let chunk = left.min(max_len as u64);
            let count = if ty == PortType::Raw {
                elems_of(chunk, elems)
            } else {
                chunk
            };
            let n = usize::try_from(count).unwrap_or(usize::MAX);
            // Preallocated in `build`; taken out so that `self` stays free for `ingest`.
            let zeros = core::mem::take(&mut at_mut(&mut self.inputs, ii)?.zeros);
            let data = zeros.data(n);
            let r = self
                .check_push(ii, &data)
                .and_then(|()| self.ingest(ii, &data))
                .and_then(|()| self.run());
            at_mut(&mut self.inputs, ii)?.zeros = zeros;
            r?;
            left -= chunk;
        }
        let o = at(&self.outputs, oi)?;
        let mapped = map_along(&self.nodes, &self.streams, &o.path, in_end);
        let end = mapped
            .saturating_add(ceil_delay(lead))
            .min(self.produced_index_of(oi)?);
        at_mut(&mut self.outputs, oi)?.burst = Burst::Ending { end };
        self.events.push(Event {
            output: OutputId::new(oi),
            index: end.saturating_sub(1),
            kind: EventKind::BurstEnd,
        });
        Ok(())
    }

    /// Discards what lies beyond the end of the burst and resets the nodes of the path. The
    /// indices continue.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn finish_burst(&mut self, oi: usize) -> Result<()> {
        let o = at_mut(&mut self.outputs, oi)?;
        o.burst = Burst::Idle;
        let src = match &mut o.kind {
            OutKind::Stream {
                src, slot, queue, ..
            } => {
                queue.clear();
                Some((*src, *slot))
            }
            OutKind::Frames(_) => None,
        };
        if let Some((sid, slot)) = src {
            let s = at_mut(&mut self.streams, sid)?;
            let end = s.end;
            *at_mut(&mut s.readers, slot)? = end;
        }
        let path_len = at(&self.outputs, oi)?.path.len();
        for k in 0..path_len {
            let ni = *at(&at(&self.outputs, oi)?.path, k)?;
            let (s, d) = {
                let n = at(&self.nodes, ni)?;
                (n.src, n.dst)
            };
            let n0_in = at(&self.streams, s)?.next_idx();
            let n0_out = match d {
                Dst::Stream(d) => at(&self.streams, d)?.next_idx(),
                Dst::Frames(_) => 0,
            };
            let n = at_mut(&mut self.nodes, ni)?;
            n.node.reset();
            n.consumed = 0;
            n.n0_in = n0_in;
            n.n0_out = n0_out;
        }
        Ok(())
    }
}

/// Fills `out[from..to]` with the silence pattern (zeros without one).
fn fill_silence<T: Pod>(out: &mut [T], from: usize, to: usize, silence: &[u8]) {
    let Some(rest) = out.get_mut(from..to) else {
        return;
    };
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(rest);
    if silence.is_empty() {
        bytes.fill(0);
    } else {
        for chunk in bytes.chunks_mut(silence.len()) {
            let n = chunk.len();
            chunk.copy_from_slice(silence.get(..n).unwrap_or_default());
        }
    }
}
