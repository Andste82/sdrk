//! Conformance checks for stream implementations. Drivers in other repositories call these
//! functions in their own tests; the checks run on any target (no heap).
//!
//! `check_rx` reads `n_blocks` blocks and verifies the contract of
//! [`RxStream`]:
//!
//! - every read returns a whole number of samples (whole packing groups) that fits the buffer,
//! - `start` of a block equals `start + samples` of the previous block, unless the block
//!   carries `GAP_BEFORE`, in which case it lies strictly above,
//! - the time derived from the anchors (`TimeAnchor::time_of` at the block start) never
//!   decreases,
//! - format and rate do not change, except on a block with `RETUNED`.
//!
//! A read that returns zero bytes ends the check (end of a file); the report says so.
//! `check_tx` writes zero samples in blocks and verifies that the stream accepts whole
//! samples, never more than offered, makes progress, keeps format and rate and reports only
//! well-formed statuses.

use sdrk_device::{
    BlockFlags, BlockMeta, RxStream, SampleFormat, SampleIndex, TimeAnchor, TxStream,
};

use crate::suite::Failure;

/// Largest buffer that `check_rx` and `check_tx` put on the stack. Use `check_rx_with` for
/// larger buffers.
pub const MAX_BUF: usize = 2048;

/// Result of a conformance run.
///
/// ```
/// use sdrk_testkit::conformance::Report;
/// assert_eq!(Report::default().blocks, 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Report {
    /// Blocks read or written.
    pub blocks: u32,
    /// Samples read or accepted.
    pub samples: u64,
    /// Bytes read or accepted.
    pub bytes: u64,
    /// RX: blocks with `GAP_BEFORE`.
    pub gaps: u32,
    /// RX: blocks that carried an anchor.
    pub anchors: u32,
    /// RX: blocks with `RETUNED`.
    pub retunes: u32,
    /// TX: statuses returned by `poll_status`.
    pub statuses: u32,
    /// RX: the stream ended (a read returned zero bytes) before `n_blocks`.
    pub ended: bool,
}

fn whole_samples(format: &SampleFormat, bytes: usize) -> Option<usize> {
    let samples = format.samples_in_bytes(bytes);
    (format.bytes_for_samples(samples) == Ok(bytes)).then_some(samples)
}

fn granule_bytes(format: &SampleFormat) -> Option<usize> {
    [1usize, 2, 4, 8]
        .into_iter()
        .find_map(|g| format.bytes_for_samples(g).ok())
}

/// Checks the receive contract with an internal buffer of `buf_size` bytes (at most
/// [`MAX_BUF`]).
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_device::{formats, Rate};
/// use sdrk_testkit::{conformance, ChunkPlan, ToneSource, VirtualRx};
/// let mut scratch = [Complex64::new(0.0, 0.0); 64];
/// let src = ToneSource { freq: 0.1, amp: 0.5, phase: 0.0 };
/// let gaps = [(128u64, 64u64)];
/// let mut rx = VirtualRx::new(
///     formats::CS16, Rate::hz(48_000)?, src, &mut scratch, ChunkPlan::new(1, 32), &gaps, None,
/// )?;
/// let report = conformance::check_rx(&mut rx, 20, 256)?;
/// assert_eq!(report.blocks, 20);
/// assert_eq!(report.gaps, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// A [`Failure`] naming the first violation.
pub fn check_rx<S: RxStream>(
    stream: &mut S,
    n_blocks: usize,
    buf_size: usize,
) -> Result<Report, Failure> {
    if buf_size > MAX_BUF {
        return Err(Failure::new("check_rx", "buf_size is larger than MAX_BUF")
            .detail(format_args!("{buf_size} > {MAX_BUF}")));
    }
    let mut storage = [0u8; MAX_BUF];
    let buf = storage.get_mut(..buf_size).unwrap_or_default();
    check_rx_with(stream, n_blocks, buf)
}

/// Like [`check_rx`] with a caller-provided buffer.
///
/// # Errors
/// A [`Failure`] naming the first violation.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy adds up the temporary of every early `Failure` return (200 bytes each); the real frame is the buffer plus small locals"
)]
pub fn check_rx_with<S: RxStream>(
    stream: &mut S,
    n_blocks: usize,
    buf: &mut [u8],
) -> Result<Report, Failure> {
    const CASE: &str = "check_rx";
    let mut report = Report::default();
    let mut format = stream.format();
    let mut rate = stream.sample_rate();
    let mut expected: Option<u64> = None;
    let mut anchor: Option<TimeAnchor> = None;
    let mut last_time: Option<i64> = None;
    for block in 0..n_blocks {
        let (n, meta) = stream.read(buf).map_err(|e| {
            Failure::new(CASE, "read failed").detail(format_args!("block {block}: {e}"))
        })?;
        if n > buf.len() {
            return Err(
                Failure::new(CASE, "read returned more bytes than the buffer holds")
                    .detail(format_args!("block {block}: {n} > {}", buf.len())),
            );
        }
        if n == 0 {
            report.ended = true;
            break;
        }
        let new_format = stream.format();
        let new_rate = stream.sample_rate();
        if (new_format != format || new_rate != rate) && !meta.flags.contains(BlockFlags::RETUNED) {
            return Err(Failure::new(CASE, "format or rate changed without RETUNED")
                .detail(format_args!("block {block}")));
        }
        format = new_format;
        rate = new_rate;
        let Some(samples) = whole_samples(&format, n) else {
            return Err(Failure::new(CASE, "read did not return whole samples")
                .detail(format_args!("block {block}: {n} bytes")));
        };
        let gap = meta.flags.contains(BlockFlags::GAP_BEFORE);
        if let Some(e) = expected {
            if gap && meta.start.0 <= e {
                return Err(
                    Failure::new(CASE, "GAP_BEFORE without a jump of the start index").detail(
                        format_args!("block {block}: start {} expected above {e}", meta.start.0),
                    ),
                );
            }
            if !gap && meta.start.0 != e {
                return Err(
                    Failure::new(CASE, "start index does not continue the previous block").detail(
                        format_args!("block {block}: start {} expected {e}", meta.start.0),
                    ),
                );
            }
        }
        if meta.flags.contains(BlockFlags::GAP_BEFORE) {
            report.gaps += 1;
        }
        if meta.flags.contains(BlockFlags::RETUNED) {
            report.retunes += 1;
        }
        if let Some(a) = meta.anchor {
            report.anchors += 1;
            anchor = Some(a);
        }
        if let Some(a) = anchor {
            let t = a.time_of(meta.start, rate).map_err(|e| {
                Failure::new(CASE, "time_of failed").detail(format_args!("block {block}: {e}"))
            })?;
            if last_time.is_some_and(|prev| t < prev) {
                return Err(Failure::new(CASE, "time of the block start decreased")
                    .detail(format_args!("block {block}: {t} after {last_time:?}")));
            }
            last_time = Some(t);
        }
        expected = Some(meta.start.0.saturating_add(samples as u64));
        report.blocks += 1;
        report.samples += samples as u64;
        report.bytes += n as u64;
    }
    Ok(report)
}

/// Checks the transmit contract: writes `n_blocks` blocks of `buf_size` bytes (rounded down
/// to whole samples, at most [`MAX_BUF`]) of zeros.
///
/// ```
/// use sdrk_device::{formats, Rate};
/// use sdrk_testkit::{conformance, VirtualTx};
/// let mut storage = [0u8; 1024];
/// let mut tx = VirtualTx::new(formats::CS16, Rate::hz(48_000)?, &mut storage, &[])?;
/// let report = conformance::check_tx(&mut tx, 4, 64)?;
/// assert_eq!((report.blocks, report.samples), (4, 64));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// A [`Failure`] naming the first violation.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy adds up the temporary of every early `Failure` return (200 bytes each); the real frame is the buffer plus small locals"
)]
pub fn check_tx<S: TxStream>(
    stream: &mut S,
    n_blocks: usize,
    buf_size: usize,
) -> Result<Report, Failure> {
    const CASE: &str = "check_tx";
    if buf_size > MAX_BUF {
        return Err(Failure::new(CASE, "buf_size is larger than MAX_BUF")
            .detail(format_args!("{buf_size} > {MAX_BUF}")));
    }
    let format = stream.format();
    let rate = stream.sample_rate();
    let granule = granule_bytes(&format)
        .ok_or_else(|| Failure::new(CASE, "format has no whole packing group"))?;
    let block_bytes = buf_size - buf_size % granule;
    if block_bytes == 0 {
        return Err(Failure::new(CASE, "buf_size is smaller than one sample")
            .detail(format_args!("{buf_size} < {granule}")));
    }
    let storage = [0u8; MAX_BUF];
    let block = storage.get(..block_bytes).unwrap_or_default();
    let mut report = Report::default();
    let mut next = 0u64;
    for index in 0..n_blocks {
        let mut offset = 0usize;
        let mut first = true;
        while offset < block.len() {
            let rest = block.get(offset..).unwrap_or_default();
            let meta = BlockMeta {
                start: SampleIndex(next),
                flags: if first && offset == 0 && index == 0 {
                    BlockFlags::BURST_START
                } else {
                    BlockFlags::empty()
                },
                anchor: None,
                tx_time_ns: None,
            };
            let accepted = stream.write(rest, &meta).map_err(|e| {
                Failure::new(CASE, "write failed").detail(format_args!("block {index}: {e}"))
            })?;
            first = false;
            if accepted > rest.len() {
                return Err(Failure::new(CASE, "write accepted more bytes than offered")
                    .detail(format_args!("block {index}: {accepted} > {}", rest.len())));
            }
            if accepted == 0 {
                return Err(Failure::new(CASE, "write made no progress")
                    .detail(format_args!("block {index}")));
            }
            let Some(samples) = whole_samples(&format, accepted) else {
                return Err(Failure::new(CASE, "write accepted a partial sample")
                    .detail(format_args!("block {index}: {accepted} bytes")));
            };
            offset += accepted;
            next += samples as u64;
            report.samples += samples as u64;
            report.bytes += accepted as u64;
        }
        if stream.format() != format || stream.sample_rate() != rate {
            return Err(Failure::new(CASE, "format or rate changed during the run")
                .detail(format_args!("block {index}")));
        }
        for _ in 0..8 {
            let status = stream.poll_status().map_err(|e| {
                Failure::new(CASE, "poll_status failed").detail(format_args!("block {index}: {e}"))
            })?;
            match status {
                None => break,
                Some(m) if m.flags == BlockFlags::empty() => {
                    return Err(Failure::new(CASE, "status without flags")
                        .detail(format_args!("block {index}")));
                }
                Some(_) => report.statuses += 1,
            }
        }
        report.blocks += 1;
    }
    Ok(report)
}
