//! Reading: a dedicated thread per running dongle (REQ-IMP-7) keeps the bulk
//! transfers of the driver in flight and puts every completed transfer into
//! a bounded queue; the [`RtlRx`] stream takes the
//! samples out. When the queue is full the newest transfer is dropped; the
//! next block then carries `OVERFLOW` and `GAP_BEFORE`, and its start index
//! jumps by the samples that were lost. When the radio is reconfigured while
//! the stream runs, the reading pauses; the next block carries `GAP_BEFORE`
//! and its index jumps by the samples the pause took (at least one).

use crate::driver::{Cancel, DriverError, DriverErrorKind, Radio};
use sdrk_device::{
    BlockFlags, BlockMeta, Rate as SdrkRate, RxStream, SampleFormat, SampleIndex, TimeAnchor,
    TimeSource, formats,
};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Bytes of one `cu8` sample.
const SAMPLE_BYTES: usize = 2;

fn unix_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_nanos()).ok())
        .unwrap_or(0)
}

struct Chunk {
    /// Bytes missing right before this transfer.
    lost_before: u64,
    /// Whether some of them were dropped because the queue was full (and not
    /// only because the reading was paused).
    overflow: bool,
    data: Vec<u8>,
}

#[derive(Default)]
struct QueueState {
    chunks: VecDeque<Chunk>,
    /// Bytes of `chunks` not yet read.
    queued: usize,
    /// Read position in the first chunk.
    offset: usize,
    /// Bytes dropped since the last accepted transfer.
    lost_pending: u64,
    /// Bytes the hardware took while the reading was paused, not yet
    /// accounted to a transfer.
    paused_pending: u64,
    /// Bytes dropped in total.
    lost_total: u64,
    /// Arrival of the first transfer (unix ns) and its sample count.
    first_arrival: Option<(i64, u64)>,
    /// The stream was stopped: reads return zero bytes.
    stopped: bool,
    /// The reader thread ended with this error.
    error: Option<DriverError>,
}

/// The queue between the reader thread and the stream.
pub(crate) struct Queue {
    capacity: usize,
    state: Mutex<QueueState>,
    cond: Condvar,
}

impl Queue {
    pub(crate) fn new(capacity_bytes: usize) -> Queue {
        Queue {
            capacity: capacity_bytes,
            state: Mutex::new(QueueState::default()),
            cond: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A completed transfer (reader thread).
    pub(crate) fn push(&self, data: &[u8]) {
        let mut st = self.lock();
        if st.stopped {
            return;
        }
        if st.first_arrival.is_none() {
            st.first_arrival = Some((unix_ns(), (data.len() / SAMPLE_BYTES) as u64));
        }
        // Whole samples only; a transfer has an even length in practice.
        let data = data
            .get(..data.len() - data.len() % SAMPLE_BYTES)
            .unwrap_or_default();
        if st.queued + data.len() > self.capacity {
            st.lost_pending += data.len() as u64;
            st.lost_total += data.len() as u64;
            return;
        }
        let overflow = st.lost_pending > 0;
        let lost_before =
            std::mem::take(&mut st.lost_pending) + std::mem::take(&mut st.paused_pending);
        st.queued += data.len();
        st.chunks.push_back(Chunk {
            lost_before,
            overflow,
            data: data.to_vec(),
        });
        drop(st);
        self.cond.notify_all();
    }

    /// The reading was paused (the radio was reconfigured while the stream
    /// ran) for about `samples` samples: the next transfer carries
    /// `GAP_BEFORE`, and its start index jumps by them.
    pub(crate) fn paused(&self, samples: u64) {
        let mut st = self.lock();
        st.paused_pending = st
            .paused_pending
            .saturating_add(samples.saturating_mul(SAMPLE_BYTES as u64));
    }

    /// The reader thread ended with an error.
    pub(crate) fn fail(&self, e: DriverError) {
        self.lock().error = Some(e);
        self.cond.notify_all();
    }

    /// The stream was stopped.
    pub(crate) fn stop(&self) {
        let mut st = self.lock();
        st.stopped = true;
        st.chunks.clear();
        st.queued = 0;
        drop(st);
        self.cond.notify_all();
    }

    /// Whether the reader thread ended with an error.
    pub(crate) fn failed(&self) -> bool {
        self.lock().error.is_some()
    }

    /// Bytes the queue dropped because the stream did not read fast enough.
    pub(crate) fn lost_bytes(&self) -> u64 {
        self.lock().lost_total
    }
}

/// The reader thread of a running dongle.
pub(crate) struct Reader {
    cancel: Cancel,
    join: JoinHandle<Box<dyn Radio>>,
}

impl Reader {
    /// Moves the radio onto a new thread that streams into `queue`.
    ///
    /// # Errors
    /// `device_error` if the thread cannot be started; the radio is lost
    /// then and the device has to be opened again.
    pub(crate) fn spawn(
        name: String,
        mut radio: Box<dyn Radio>,
        queue: Arc<Queue>,
        buffers: usize,
        buffer_bytes: usize,
    ) -> Result<Reader, DriverError> {
        let cancel = Cancel::new();
        let c = cancel.clone();
        let join = std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                if let Err(e) = radio.reset_buffer() {
                    queue.fail(e);
                    return radio;
                }
                let result = radio.stream(buffers, buffer_bytes, &c, &mut |data| queue.push(data));
                if let Err(e) = result
                    && !c.is_cancelled()
                {
                    queue.fail(e);
                }
                radio
            })
            .map_err(|e| {
                DriverError::new(
                    DriverErrorKind::Other,
                    format!("cannot start the reader thread: {e}"),
                )
            })?;
        Ok(Reader { cancel, join })
    }

    /// Stops streaming and returns the radio; `None` if the thread panicked.
    pub(crate) fn stop(self) -> Option<Box<dyn Radio>> {
        self.cancel.cancel();
        self.join.join().ok()
    }
}

/// The RX stream of an RTL-SDR: `cu8` at the rate set when it started
/// ([`crate::RtlSdr::start_rx`]).
///
/// A read blocks until samples arrive, at most the read timeout of the
/// [`crate::StreamOptions`]. It returns zero bytes once the stream was
/// stopped ([`crate::RtlSdr::stop_rx`]) and the error of the driver once the
/// reading failed (`Disconnected` when the dongle was unplugged). The first
/// block carries a host-time anchor.
pub struct RtlRx {
    pub(crate) queue: Arc<Queue>,
    pub(crate) rate: SdrkRate,
    pub(crate) next_index: u64,
    pub(crate) anchored: bool,
    /// A read waits this long for samples before it fails with `timeout`.
    pub(crate) timeout: Duration,
}

impl core::fmt::Debug for RtlRx {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RtlRx")
            .field("rate", &self.rate)
            .field("next_index", &self.next_index)
            .finish_non_exhaustive()
    }
}

impl RxStream for RtlRx {
    type Error = DriverError;

    fn format(&self) -> SampleFormat {
        formats::CU8
    }

    fn sample_rate(&self) -> SdrkRate {
        self.rate
    }

    #[allow(clippy::too_many_lines)]
    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), DriverError> {
        let want = buf.len() - buf.len() % SAMPLE_BYTES;
        if want == 0 {
            return Err(DriverError::new(
                DriverErrorKind::InvalidArgument,
                "buffer is smaller than one sample",
            ));
        }
        let mut st = self.queue.lock();
        let started = Instant::now();
        loop {
            if st.stopped {
                return Ok((
                    0,
                    BlockMeta {
                        start: SampleIndex(self.next_index),
                        ..BlockMeta::default()
                    },
                ));
            }
            if !st.chunks.is_empty() {
                break;
            }
            if let Some(e) = &st.error {
                return Err(e.clone());
            }
            let waited = started.elapsed();
            if waited >= self.timeout {
                return Err(DriverError::new(
                    DriverErrorKind::Timeout,
                    format!("no samples from the RTL-SDR for {} ms", waited.as_millis()),
                ));
            }
            st = self
                .queue
                .cond
                .wait_timeout(
                    st,
                    self.timeout
                        .saturating_sub(waited)
                        .min(Duration::from_millis(50)),
                )
                .map_or_else(|e| e.into_inner().0, |(g, _)| g);
        }
        let mut flags = BlockFlags::empty();
        let mut copied = 0;
        while copied < want {
            let offset = st.offset;
            let Some(chunk) = st.chunks.front_mut() else {
                break;
            };
            if offset == 0 && chunk.lost_before > 0 {
                if copied > 0 {
                    // A block is contiguous: the gap starts the next one.
                    break;
                }
                let lost = std::mem::take(&mut chunk.lost_before);
                self.next_index += lost / SAMPLE_BYTES as u64;
                if chunk.overflow {
                    flags.insert(BlockFlags::OVERFLOW);
                }
                flags.insert(BlockFlags::GAP_BEFORE);
            }
            let available = chunk.data.len() - offset;
            let n = available.min(want - copied);
            if let (Some(dst), Some(src)) = (
                buf.get_mut(copied..copied + n),
                chunk.data.get(offset..offset + n),
            ) {
                dst.copy_from_slice(src);
            }
            copied += n;
            if n == available {
                st.chunks.pop_front();
                st.offset = 0;
            } else {
                st.offset += n;
            }
            st.queued -= n;
        }
        let start = self.next_index;
        self.next_index += (copied / SAMPLE_BYTES) as u64;
        let anchor = if self.anchored {
            None
        } else {
            self.anchored = true;
            // Host time: the first transfer arrived when its last sample was
            // taken.
            st.first_arrival.map(|(at, samples)| {
                let span = i64::try_from(
                    u128::from(samples) * 1_000_000_000 * u128::from(self.rate.den())
                        / u128::from(self.rate.num().max(1)),
                )
                .unwrap_or(0);
                TimeAnchor {
                    index: SampleIndex(0),
                    unix_ns: at.saturating_sub(span),
                    source: TimeSource::Host,
                }
            })
        };
        Ok((
            copied,
            BlockMeta {
                start: SampleIndex(start),
                flags,
                anchor,
                tx_time_ns: None,
            },
        ))
    }
}
