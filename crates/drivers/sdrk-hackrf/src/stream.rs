//! The sample paths. Each direction runs on a dedicated thread (REQ-IMP-7)
//! that keeps the bulk transfers of the driver in flight.
//!
//! - RX: every completed transfer goes into a bounded [`RxQueue`]; the
//!   [`HackRx`] stream that the source reads takes the samples out. When the
//!   queue is full the newest transfer is dropped; the next block then
//!   carries `OVERFLOW` and `GAP_BEFORE`, and its start index jumps by the
//!   samples that were lost.
//! - TX: [`HackTx::write`] puts the samples into a bounded [`TxQueue`]
//!   (waiting while it is full); the writer thread switches the transceiver
//!   to transmit when the first samples are there (RF is switched on by the
//!   first samples, REQ-TX-3) and fills the bulk transfers from it. A
//!   transfer is submitted only when it is full of samples, or at the end
//!   of a burst (then padded with zeros to the next 16384 bytes), so that no
//!   zeros are ever put between the samples of a transmission. A burst end
//!   is a write with `BURST_END`, also one with an empty buffer. Once the
//!   samples of a burst have left the queue, no transfer is in flight any
//!   more and the firmware buffer had time to play out,
//!   [`TxStream::poll_status`] reports `BURST_END` (the burst is done); the
//!   core ends a finite stream only then, so that stopping the stream cuts
//!   nothing off. TX underruns that the driver reports come out of
//!   [`TxStream::poll_status`] with `UNDERRUN`, one per call: two underruns
//!   of the firmware are two reports, never one.

use crate::driver::{Cancel, DriverError, DriverErrorKind, Radio, TransceiverMode, TxSource};
use sdrk_device::{
    BlockFlags, BlockMeta, Rate as SdrkRate, RxStream, SampleFormat, SampleIndex, TimeAnchor,
    TimeSource, TxStream, formats,
};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Bytes of one `cs8` sample.
pub const SAMPLE_BYTES: usize = 2;

/// How long the writer waits for samples before it looks at the state of the
/// transfers and the stop request again.
const FILL_WAIT: Duration = Duration::from_millis(10);

fn unix_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_nanos()).ok())
        .unwrap_or(0)
}

fn wait<'a, T>(cond: &Condvar, guard: MutexGuard<'a, T>, d: Duration) -> MutexGuard<'a, T> {
    cond.wait_timeout(guard, d)
        .map_or_else(|e| e.into_inner().0, |(g, _)| g)
}

// ---------------------------------------------------------------- RX ----

struct Chunk {
    /// Bytes dropped right before this transfer.
    lost_before: u64,
    data: Vec<u8>,
}

#[derive(Default)]
struct RxState {
    chunks: VecDeque<Chunk>,
    /// Bytes of `chunks` not yet read.
    queued: usize,
    /// Read position in the first chunk.
    offset: usize,
    /// Bytes dropped since the last accepted transfer.
    lost_pending: u64,
    /// Bytes dropped in total.
    lost_total: u64,
    /// Arrival of the first transfer (unix ns) and its sample count.
    first_arrival: Option<(i64, u64)>,
    /// The stream was stopped: reads return zero bytes.
    stopped: bool,
    /// The reader thread ended with this error.
    error: Option<DriverError>,
}

/// The queue between the RX thread and the stream.
pub struct RxQueue {
    capacity: usize,
    state: Mutex<RxState>,
    cond: Condvar,
}

impl RxQueue {
    /// A queue that holds up to `capacity_bytes` before it drops transfers.
    pub fn new(capacity_bytes: usize) -> RxQueue {
        RxQueue {
            capacity: capacity_bytes,
            state: Mutex::new(RxState::default()),
            cond: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, RxState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A completed transfer (RX thread).
    pub fn push(&self, data: &[u8]) {
        let mut st = self.lock();
        if st.stopped {
            return;
        }
        if st.first_arrival.is_none() {
            st.first_arrival = Some((unix_ns(), (data.len() / SAMPLE_BYTES) as u64));
        }
        let data = data
            .get(..data.len() - data.len() % SAMPLE_BYTES)
            .unwrap_or_default();
        if st.queued + data.len() > self.capacity {
            st.lost_pending += data.len() as u64;
            st.lost_total += data.len() as u64;
            return;
        }
        let lost_before = std::mem::take(&mut st.lost_pending);
        st.queued += data.len();
        st.chunks.push_back(Chunk {
            lost_before,
            data: data.to_vec(),
        });
        drop(st);
        self.cond.notify_all();
    }

    /// The RX thread ended with an error.
    pub fn fail(&self, e: DriverError) {
        self.lock().error = Some(e);
        self.cond.notify_all();
    }

    /// The stream was stopped.
    pub fn stop(&self) {
        let mut st = self.lock();
        st.stopped = true;
        st.chunks.clear();
        st.queued = 0;
        drop(st);
        self.cond.notify_all();
    }

    /// Whether the RX thread ended with an error.
    pub fn failed(&self) -> bool {
        self.lock().error.is_some()
    }

    /// Bytes the queue dropped because the stream did not read fast enough.
    pub fn lost_bytes(&self) -> u64 {
        self.lock().lost_total
    }
}

/// The RX thread of a running board.
pub struct Reader {
    cancel: Cancel,
    join: JoinHandle<()>,
}

impl Reader {
    /// Starts a thread that streams into `queue`.
    ///
    /// # Errors
    /// `device_error` if the thread cannot be started.
    pub fn spawn(
        name: String,
        radio: Arc<dyn Radio>,
        queue: Arc<RxQueue>,
        buffers: usize,
        buffer_bytes: usize,
    ) -> Result<Reader, DriverError> {
        let cancel = Cancel::new();
        let c = cancel.clone();
        let join = std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                let result = radio.rx_stream(buffers, buffer_bytes, &c, &mut |d| queue.push(d));
                if let Err(e) = result
                    && !c.is_cancelled()
                {
                    queue.fail(e);
                }
            })
            .map_err(|e| {
                DriverError::new(
                    DriverErrorKind::Other,
                    format!("cannot start the RX thread: {e}"),
                )
            })?;
        Ok(Reader { cancel, join })
    }

    /// Stops streaming and waits for the thread; `false` if it panicked.
    pub fn stop(self) -> bool {
        self.cancel.cancel();
        self.join.join().is_ok()
    }
}

/// The RX stream of a HackRF: `cs8` at the configured rate.
pub struct HackRx {
    queue: Arc<RxQueue>,
    rate: SdrkRate,
    next_index: u64,
    anchored: bool,
    /// A read waits this long for samples before it fails with `timeout`.
    timeout: Duration,
}

impl HackRx {
    /// The stream that reads `queue` (filled by a [`Reader`]) at `rate`; a
    /// read waits at most `timeout` for samples.
    pub fn new(queue: Arc<RxQueue>, rate: SdrkRate, timeout: Duration) -> HackRx {
        HackRx {
            queue,
            rate,
            next_index: 0,
            anchored: false,
            timeout,
        }
    }
}

impl core::fmt::Debug for HackRx {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HackRx")
            .field("rate", &self.rate)
            .field("next_index", &self.next_index)
            .finish_non_exhaustive()
    }
}

impl RxStream for HackRx {
    type Error = DriverError;

    fn format(&self) -> SampleFormat {
        formats::CS8
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
                    format!("no samples from the HackRF for {} ms", waited.as_millis()),
                ));
            }
            st = wait(
                &self.queue.cond,
                st,
                self.timeout
                    .saturating_sub(waited)
                    .min(Duration::from_millis(50)),
            );
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
                flags.insert(BlockFlags::OVERFLOW);
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

// ---------------------------------------------------------------- TX ----

#[derive(Default)]
struct TxState {
    data: VecDeque<u8>,
    /// The byte positions in `data` (from its front, ascending) at which a
    /// burst ends. The writer sends the samples up to the first of them
    /// without waiting for a full transfer, padded with zeros; the samples
    /// after it (the next burst) wait for full transfers again.
    burst_ends: VecDeque<usize>,
    /// The stream was stopped: writes fail, the writer ends.
    stopped: bool,
    /// The writer thread ended with this error.
    error: Option<DriverError>,
    /// A burst end was written and the burst is not played out yet.
    burst_end_pending: bool,
    /// Since when the queue is empty and no transfer is in flight, while a
    /// burst end is pending.
    drained_since: Option<Instant>,
    /// The burst is played out; not yet reported by `poll_status`.
    burst_done: bool,
    /// Underruns not yet reported by `poll_status`.
    underruns_pending: u32,
    /// Underruns in total.
    underruns_total: u64,
    /// Bytes handed to the driver.
    sent: u64,
}

/// The queue between the stream and the TX thread.
pub struct TxQueue {
    capacity: usize,
    /// How long the board needs to play out its own buffer after the last
    /// transfer completed.
    settle: Duration,
    state: Mutex<TxState>,
    cond: Condvar,
    /// The antenna power `tx0` wants (its `bias_tee`). `configure` changes
    /// it while the stream runs, and the writer reads it when it switches
    /// to transmit; both send the request to the board under this lock, so
    /// the last request the board gets is always the current value.
    antenna: Mutex<bool>,
}

impl TxQueue {
    /// A queue of `capacity_bytes`; `settle` is the time the board needs to
    /// play out its own buffer after the last transfer completed.
    pub fn new(capacity_bytes: usize, settle: Duration) -> TxQueue {
        TxQueue {
            capacity: capacity_bytes.max(SAMPLE_BYTES),
            settle,
            state: Mutex::new(TxState::default()),
            cond: Condvar::new(),
            antenna: Mutex::new(false),
        }
    }

    /// Runs `f` with the antenna power `tx0` wants, under its lock (see
    /// the field `antenna`).
    pub fn with_antenna<R>(&self, f: impl FnOnce(&mut bool) -> R) -> R {
        let mut on = self.antenna.lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut on)
    }

    fn lock(&self) -> MutexGuard<'_, TxState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Marks the end of a burst: what is queued is sent without waiting
    /// for a full transfer.
    fn mark_burst_end(st: &mut TxState) {
        let end = st.data.len();
        if end > 0 && st.burst_ends.back() != Some(&end) {
            st.burst_ends.push_back(end);
        }
        st.drained_since = None;
        if st.data.is_empty() && st.sent == 0 {
            // Nothing was ever sent: the burst is done already.
            st.burst_end_pending = false;
            st.burst_done = true;
        } else {
            st.burst_end_pending = true;
            st.burst_done = false;
        }
    }

    /// Queues as many whole samples of `buf` as fit, waiting up to `timeout`
    /// for space (stream side). An empty `buf` queues nothing; with
    /// `burst_end` it ends the burst.
    fn push(&self, buf: &[u8], burst_end: bool, timeout: Duration) -> Result<usize, DriverError> {
        let whole = buf.len() - buf.len() % SAMPLE_BYTES;
        if buf.is_empty() {
            let mut st = self.lock();
            if st.stopped {
                return Err(DriverError::new(
                    DriverErrorKind::InvalidState,
                    "the TX stream was stopped",
                ));
            }
            if let Some(e) = &st.error {
                return Err(e.clone());
            }
            if burst_end {
                Self::mark_burst_end(&mut st);
                drop(st);
                self.cond.notify_all();
            }
            return Ok(0);
        }
        if whole == 0 {
            return Err(DriverError::new(
                DriverErrorKind::InvalidArgument,
                "the buffer is smaller than one sample",
            ));
        }
        let started = Instant::now();
        let mut st = self.lock();
        loop {
            if st.stopped {
                return Err(DriverError::new(
                    DriverErrorKind::InvalidState,
                    "the TX stream was stopped",
                ));
            }
            if let Some(e) = &st.error {
                return Err(e.clone());
            }
            let space = self.capacity.saturating_sub(st.data.len());
            let n = whole.min(space - space % SAMPLE_BYTES);
            if n > 0 {
                st.data.extend(buf.get(..n).unwrap_or_default());
                st.drained_since = None;
                st.burst_done = false;
                if burst_end && n == whole {
                    Self::mark_burst_end(&mut st);
                }
                drop(st);
                self.cond.notify_all();
                return Ok(n);
            }
            let waited = started.elapsed();
            if waited >= timeout {
                return Err(DriverError::new(
                    DriverErrorKind::Timeout,
                    format!(
                        "the HackRF took no TX samples for {} ms",
                        waited.as_millis()
                    ),
                ));
            }
            st = wait(
                &self.cond,
                st,
                timeout
                    .saturating_sub(waited)
                    .min(Duration::from_millis(50)),
            );
        }
    }

    /// Waits until samples are queued; `false` if the stream was stopped or
    /// `cancel` was set first (TX thread).
    fn wait_first(&self, cancel: &Cancel) -> bool {
        let mut st = self.lock();
        loop {
            if st.stopped || cancel.is_cancelled() {
                return false;
            }
            if !st.data.is_empty() {
                return true;
            }
            st = wait(&self.cond, st, FILL_WAIT);
        }
    }

    /// Fills a transfer (TX thread, see [`TxSource::fill`]). A full transfer
    /// is sent when the samples for it are queued (or the queue is full and
    /// cannot take more); a partial one only at the end of a burst, and only
    /// up to that end: the samples of a burst queued behind it are not put
    /// into the padded transfer. The driver pads a partial transfer with
    /// zeros to the next 16384 bytes.
    ///
    /// Zeros are never put between the samples of a running transmission:
    /// when the client does not keep up, no transfer is submitted, the
    /// hardware runs dry and the firmware counts the underrun, which
    /// [`TxStream::poll_status`] reports.
    ///
    /// `in_flight` (the transfers not completed yet) tells when a pending
    /// burst end is played out: the queue is empty, nothing is in flight and
    /// the board had `settle` to play out its own buffer.
    fn fill(&self, buf: &mut [u8], in_flight: usize) -> Option<usize> {
        let started = Instant::now();
        let mut st = self.lock();
        loop {
            if st.stopped {
                return None;
            }
            if st.burst_end_pending && st.data.is_empty() {
                if in_flight == 0 {
                    let since = *st.drained_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= self.settle {
                        st.burst_end_pending = false;
                        st.drained_since = None;
                        st.burst_done = true;
                    }
                } else {
                    st.drained_since = None;
                }
            }
            let queued = st.data.len();
            // The samples up to the end of the current burst, if it is queued.
            let burst_end = st.burst_ends.front().copied();
            let full = queued >= buf.len() || self.capacity.saturating_sub(queued) < SAMPLE_BYTES;
            if queued > 0 && (full || burst_end.is_some()) {
                let n = queued.min(buf.len()).min(burst_end.unwrap_or(usize::MAX));
                let (a, b) = st.data.as_slices();
                let from_a = a.len().min(n);
                if let (Some(dst), Some(src)) = (buf.get_mut(..from_a), a.get(..from_a)) {
                    dst.copy_from_slice(src);
                }
                if let (Some(dst), Some(src)) = (buf.get_mut(from_a..n), b.get(..n - from_a)) {
                    dst.copy_from_slice(src);
                }
                if let Some(rest) = buf.get_mut(n..) {
                    rest.fill(0);
                }
                st.data.drain(..n);
                for end in &mut st.burst_ends {
                    *end = end.saturating_sub(n);
                }
                while st.burst_ends.front() == Some(&0) {
                    st.burst_ends.pop_front();
                }
                st.sent += n as u64;
                drop(st);
                self.cond.notify_all();
                return Some(n);
            }
            let waited = started.elapsed();
            if waited >= FILL_WAIT {
                return None;
            }
            st = wait(&self.cond, st, FILL_WAIT.saturating_sub(waited));
        }
    }

    fn underrun(&self, count: u32) {
        let mut st = self.lock();
        st.underruns_pending = st.underruns_pending.saturating_add(count);
        st.underruns_total += u64::from(count);
    }

    /// The TX thread ended with an error.
    pub fn fail(&self, e: DriverError) {
        self.lock().error = Some(e);
        self.cond.notify_all();
    }

    /// The stream was stopped.
    pub fn stop(&self) {
        let mut st = self.lock();
        st.stopped = true;
        st.data.clear();
        st.burst_ends.clear();
        drop(st);
        self.cond.notify_all();
    }

    /// Whether the TX thread ended with an error.
    pub fn failed(&self) -> bool {
        self.lock().error.is_some()
    }

    /// Underruns reported by the driver so far.
    pub fn underruns(&self) -> u64 {
        self.lock().underruns_total
    }
}

struct QueueSource(Arc<TxQueue>);

impl TxSource for QueueSource {
    fn fill(&mut self, buf: &mut [u8], in_flight: usize) -> Option<usize> {
        self.0.fill(buf, in_flight)
    }

    fn underrun(&mut self, count: u32) {
        if count > 0 {
            self.0.underrun(count);
        }
    }
}

/// The TX thread of a transmitting board.
pub struct Writer {
    cancel: Cancel,
    join: JoinHandle<()>,
}

impl Writer {
    /// Starts a thread that waits for the first samples in `queue`, switches
    /// the transceiver to transmit (and the antenna power on again if `tx0`
    /// wants it at that moment, `TxQueue::with_antenna`) and streams the
    /// queue to the board.
    ///
    /// # Errors
    /// `device_error` if the thread cannot be started.
    pub fn spawn(
        name: String,
        radio: Arc<dyn Radio>,
        queue: Arc<TxQueue>,
        buffers: usize,
        buffer_bytes: usize,
    ) -> Result<Writer, DriverError> {
        let cancel = Cancel::new();
        let c = cancel.clone();
        let join = std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                if !queue.wait_first(&c) {
                    return;
                }
                if let Err(e) = radio.set_transceiver_mode(TransceiverMode::Transmit) {
                    queue.fail(e);
                    return;
                }
                // Some firmware versions switch the antenna power off with
                // the mode. The value is the current one: `configure` may
                // have changed it since the stream started.
                let antenna = queue.with_antenna(|on| {
                    if *on {
                        radio.set_antenna_power(true)
                    } else {
                        Ok(())
                    }
                });
                // Not fatal: the transmission goes on without antenna power.
                drop(antenna);
                let mut source = QueueSource(Arc::clone(&queue));
                let result = radio.tx_stream(buffers, buffer_bytes, &c, &mut source);
                if let Err(e) = result
                    && !c.is_cancelled()
                {
                    queue.fail(e);
                }
            })
            .map_err(|e| {
                DriverError::new(
                    DriverErrorKind::Other,
                    format!("cannot start the TX thread: {e}"),
                )
            })?;
        Ok(Writer { cancel, join })
    }

    /// Stops streaming and waits for the thread; `false` if it panicked.
    pub fn stop(self) -> bool {
        self.cancel.cancel();
        self.join.join().is_ok()
    }
}

/// The TX stream of a HackRF: `cs8` at the configured rate.
pub struct HackTx {
    queue: Arc<TxQueue>,
    rate: SdrkRate,
    /// A write waits this long for space before it fails with `timeout`.
    timeout: Duration,
}

impl HackTx {
    /// The stream that writes into `queue` (sent by a [`Writer`]) at `rate`;
    /// a write waits at most `timeout` for space.
    pub fn new(queue: Arc<TxQueue>, rate: SdrkRate, timeout: Duration) -> HackTx {
        HackTx {
            queue,
            rate,
            timeout,
        }
    }
}

impl core::fmt::Debug for HackTx {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HackTx")
            .field("rate", &self.rate)
            .finish_non_exhaustive()
    }
}

impl TxStream for HackTx {
    type Error = DriverError;

    fn format(&self) -> SampleFormat {
        formats::CS8
    }

    fn sample_rate(&self) -> SdrkRate {
        self.rate
    }

    fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize, DriverError> {
        self.queue.push(
            buf,
            meta.flags.contains(BlockFlags::BURST_END),
            self.timeout,
        )
    }

    /// Reports one status per call: each underrun of the firmware as its own
    /// `UNDERRUN` (the core counts each report, protocol.md 14.4), then a
    /// played-out burst with
    /// `BURST_END`.
    fn poll_status(&mut self) -> Result<Option<BlockMeta>, DriverError> {
        let mut st = self.queue.lock();
        if let Some(e) = &st.error {
            return Err(e.clone());
        }
        let flags = if st.underruns_pending > 0 {
            st.underruns_pending -= 1;
            BlockFlags::UNDERRUN
        } else if st.burst_done {
            st.burst_done = false;
            BlockFlags::BURST_END
        } else {
            return Ok(None);
        };
        Ok(Some(BlockMeta {
            start: SampleIndex(st.sent / SAMPLE_BYTES as u64),
            flags,
            anchor: None,
            tx_time_ns: None,
        }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const TRANSFER: usize = 16384;
    const TIMEOUT: Duration = Duration::from_millis(50);

    fn fill(q: &TxQueue) -> Option<Vec<u8>> {
        let mut buf = vec![0xa5u8; TRANSFER];
        q.fill(&mut buf, 0).map(|n| {
            // The bytes after the samples are zero.
            assert!(buf[n..].iter().all(|b| *b == 0));
            buf.truncate(n);
            buf
        })
    }

    /// Q82: back-to-back bursts on a continuous stream. Burst 1 ends with
    /// `BURST_END` while more than one transfer of burst 2 is queued behind
    /// it: only the tail of burst 1 goes out in a padded transfer, burst 2
    /// in full transfers without zeros between its samples.
    #[test]
    fn req_tx_4_back_to_back_bursts_put_no_zeros_into_the_second_burst() {
        let q = TxQueue::new(65536, Duration::ZERO);
        assert_eq!(q.push(&[0x11; 3000], true, TIMEOUT).unwrap(), 3000);
        let second = TRANSFER + 3616;
        assert_eq!(q.push(&vec![0x22; second], false, TIMEOUT).unwrap(), second);
        // The tail of burst 1, padded.
        let a = fill(&q).expect("the end of burst 1 is sent");
        assert_eq!(a, vec![0x11; 3000]);
        // A whole transfer of burst 2.
        let b = fill(&q).expect("a full transfer of burst 2");
        assert_eq!(b, vec![0x22; TRANSFER]);
        // The rest of burst 2 waits for more samples or its end.
        assert!(fill(&q).is_none());
        assert_eq!(q.push(&[], true, TIMEOUT).unwrap(), 0);
        let c = fill(&q).expect("the end of burst 2 is sent");
        assert_eq!(c, vec![0x22; 3616]);
        assert!(fill(&q).is_none());
    }

    /// Two short bursts, both ended, queued before the writer runs: each
    /// goes out in its own padded transfer.
    #[test]
    fn req_tx_4_each_queued_burst_end_bounds_a_padded_transfer() {
        let q = TxQueue::new(65536, Duration::ZERO);
        q.push(&[0x11; 1000], true, TIMEOUT).unwrap();
        q.push(&[0x22; 2000], true, TIMEOUT).unwrap();
        assert_eq!(fill(&q).unwrap(), vec![0x11; 1000]);
        assert_eq!(fill(&q).unwrap(), vec![0x22; 2000]);
        assert!(fill(&q).is_none());
    }

    /// A burst end far behind the front: full transfers up to it, then the
    /// tail.
    #[test]
    fn req_tx_4_long_burst_ends_with_one_padded_transfer() {
        let q = TxQueue::new(65536, Duration::ZERO);
        let len = 2 * TRANSFER + 100;
        q.push(&vec![0x33; len], true, TIMEOUT).unwrap();
        assert_eq!(fill(&q).unwrap().len(), TRANSFER);
        assert_eq!(fill(&q).unwrap().len(), TRANSFER);
        assert_eq!(fill(&q).unwrap(), vec![0x33; 100]);
        assert!(fill(&q).is_none());
    }
}
