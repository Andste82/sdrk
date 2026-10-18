//! A minimal virtual receiver: a deterministic [`SignalSource`] encoded into
//! the raw format of a device, delivered in chunks, optionally with gaps.

use num_complex::Complex64;
use sdrk_core::Error;
use sdrk_device::{BlockFlags, BlockMeta, Rate, RxStream, SampleFormat, SampleIndex, TimeAnchor};

use crate::encode::encode;
use crate::r#gen;
use crate::invariance::{ChunkPlan, ChunkSizes};
use crate::rng::Pcg32;

/// A signal that is a pure function of the absolute sample index.
pub trait SignalSource {
    /// Writes the samples `start..start + out.len()`.
    fn fill(&mut self, start: u64, out: &mut [Complex64]);
}

/// Complex tone (see `r#gen::tone`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneSource {
    /// Frequency in cycles per sample.
    pub freq: f64,
    /// Amplitude.
    pub amp: f64,
    /// Start phase in radians.
    pub phase: f64,
}

impl SignalSource for ToneSource {
    fn fill(&mut self, start: u64, out: &mut [Complex64]) {
        r#gen::tone(out, self.freq, self.amp, self.phase, start);
    }
}

/// Complex Gaussian noise that is a pure function of the sample index: sample
/// `n` is drawn after skipping `8 n` draws of a PCG32 generator, so any chunking
/// (and any gap) yields the same samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSource {
    /// Seed of the generator.
    pub seed: u64,
    /// Noise power `E|x|^2`.
    pub power: f64,
}

/// `next_u32` draws per complex noise sample (two normals of four draws).
const DRAWS_PER_NOISE_SAMPLE: u64 = 8;

impl SignalSource for NoiseSource {
    fn fill(&mut self, start: u64, out: &mut [Complex64]) {
        let mut rng = Pcg32::new(self.seed, 0x6e6f_6973);
        rng.advance(start.wrapping_mul(DRAWS_PER_NOISE_SAMPLE));
        r#gen::noise(out, &mut rng, self.power);
    }
}

/// The sum of two sources.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumSource<A, B> {
    /// First summand.
    pub a: A,
    /// Second summand.
    pub b: B,
}

impl<A: SignalSource, B: SignalSource> SignalSource for SumSource<A, B> {
    fn fill(&mut self, start: u64, out: &mut [Complex64]) {
        const PIECE: usize = 64;
        let mut tmp = [Complex64::new(0.0, 0.0); PIECE];
        let mut at = start;
        for piece in out.chunks_mut(PIECE) {
            let t = tmp.get_mut(..piece.len()).unwrap_or_default();
            self.a.fill(at, piece);
            self.b.fill(at, t);
            for (o, x) in piece.iter_mut().zip(t.iter()) {
                *o += *x;
            }
            at = at.wrapping_add(piece.len() as u64);
        }
    }
}

/// Virtual receiver implementing [`RxStream`].
///
/// - `read` produces the next chunk (size from the plan, limited by the
///   output buffer, the scratch buffer and the next gap), encodes it and
///   returns `BlockMeta { start, flags, anchor }`.
/// - `gaps` lists `(at_index, missing)`: when the next index reaches
///   `at_index`, `missing` samples are skipped (the index advances without
///   emitting them) and the following block carries `GAP_BEFORE`.
/// - The first block carries the configured anchor.
///
/// ```
/// use sdrk_device::{formats, Rate, RxStream};
/// use sdrk_testkit::{ChunkPlan, ToneSource, VirtualRx};
/// use num_complex::Complex64;
/// let mut scratch = [Complex64::new(0.0, 0.0); 16];
/// let src = ToneSource { freq: 0.1, amp: 0.5, phase: 0.0 };
/// let mut rx = VirtualRx::new(
///     formats::CS16, Rate::hz(48_000)?, src, &mut scratch,
///     ChunkPlan::new(1, 8), &[], None,
/// )?;
/// let mut buf = [0u8; 64];
/// let (bytes, meta) = rx.read(&mut buf)?;
/// assert_eq!(meta.start.0, 0);
/// assert_eq!(bytes % 4, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct VirtualRx<'a, Src: SignalSource> {
    format: SampleFormat,
    rate: Rate,
    source: Src,
    scratch: &'a mut [Complex64],
    sizes: ChunkSizes,
    gaps: &'a [(u64, u64)],
    gap_cursor: usize,
    next_index: u64,
    anchor: Option<TimeAnchor>,
    pending_gap: bool,
    /// Smallest number of samples that fills whole packing groups.
    granule: usize,
}

impl<'a, Src: SignalSource> VirtualRx<'a, Src> {
    /// Creates a receiver.
    ///
    /// # Errors
    /// `InvalidParameter` if the format cannot be encoded (`Bits1`), the
    /// scratch buffer cannot hold one packing group, or a gap is not sorted
    /// or not a multiple of the packing granule.
    pub fn new(
        format: SampleFormat,
        rate: Rate,
        source: Src,
        scratch: &'a mut [Complex64],
        plan: ChunkPlan,
        gaps: &'a [(u64, u64)],
        anchor: Option<TimeAnchor>,
    ) -> Result<Self, Error> {
        let granule = [1usize, 2, 4, 8]
            .into_iter()
            .find(|&g| format.bytes_for_samples(g).is_ok())
            .ok_or(Error::InvalidParameter("no whole packing group"))?;
        let mut probe = [0u8; 16];
        let sample = [Complex64::new(0.0, 0.0); 8];
        let probe_samples = sample.get(..granule).unwrap_or_default();
        encode(&format, probe_samples, &mut probe)
            .map_err(|_| Error::InvalidParameter("format cannot be encoded"))?;
        if scratch.len() < granule {
            return Err(Error::InvalidParameter("scratch buffer too small"));
        }
        let g = granule as u64;
        let mut previous = 0u64;
        for &(at, missing) in gaps {
            if at < previous || at % g != 0 || missing % g != 0 {
                return Err(Error::InvalidParameter(
                    "gaps must be sorted and a multiple of the packing granule",
                ));
            }
            previous = at;
        }
        Ok(Self {
            format,
            rate,
            source,
            scratch,
            sizes: plan.sizes(),
            gaps,
            gap_cursor: 0,
            next_index: 0,
            anchor,
            pending_gap: false,
            granule,
        })
    }

    /// Index of the next sample that `read` will emit (before any gap that
    /// starts there is applied).
    pub fn next_index(&self) -> u64 {
        self.next_index
    }

    /// Applies every gap whose position has been reached.
    fn apply_gaps(&mut self) {
        while let Some(&(at, missing)) = self.gaps.get(self.gap_cursor) {
            if at > self.next_index {
                break;
            }
            self.next_index = self.next_index.saturating_add(missing);
            self.pending_gap = true;
            self.gap_cursor += 1;
        }
    }
}

impl<Src: SignalSource> RxStream for VirtualRx<'_, Src> {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Error> {
        self.apply_gaps();
        let planned = self.sizes.next().unwrap_or(1);
        let room = self.format.samples_in_bytes(buf.len());
        let mut n = planned.min(room).min(self.scratch.len());
        if let Some(&(at, _)) = self.gaps.get(self.gap_cursor) {
            n = n.min(usize::try_from(at.saturating_sub(self.next_index)).unwrap_or(usize::MAX));
        }
        n -= n % self.granule;
        if n == 0 {
            n = self.granule;
        }
        let needed = self.format.bytes_for_samples(n)?;
        if needed > buf.len() || n > self.scratch.len() {
            return Err(Error::BufferTooSmall {
                needed,
                got: buf.len(),
            });
        }
        let chunk = self
            .scratch
            .get_mut(..n)
            .ok_or(Error::Internal("scratch slice"))?;
        self.source.fill(self.next_index, chunk);
        let written = encode(&self.format, chunk, buf)
            .map_err(|_| Error::Internal("encoding a chunk failed"))?;
        let mut flags = BlockFlags::empty();
        if self.pending_gap {
            flags.insert(BlockFlags::GAP_BEFORE);
            self.pending_gap = false;
        }
        let meta = BlockMeta {
            start: SampleIndex(self.next_index),
            flags,
            anchor: self.anchor.take(),
            tx_time_ns: None,
        };
        self.next_index = self.next_index.saturating_add(n as u64);
        Ok((written, meta))
    }
}
