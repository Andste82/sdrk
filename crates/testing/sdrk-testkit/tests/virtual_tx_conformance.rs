//! `VirtualTx` and the conformance checks, including streams that break the contract.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::result_large_err
)]

use num_complex::Complex64;
use sdrk_core::Error;
use sdrk_device::{
    BlockFlags, BlockMeta, Rate, RxStream, SampleFormat, SampleIndex, TimeAnchor, TimeSource,
    TxStream, formats,
};
use sdrk_testkit::conformance::{self, MAX_BUF};
use sdrk_testkit::{ChunkPlan, ToneSource, VirtualRx, VirtualTx};

fn rate() -> Rate {
    Rate::hz(48_000).unwrap()
}

fn assert_send<T: Send>() {}

#[test]
fn public_types_are_send() {
    assert_send::<VirtualTx<'static>>();
    assert_send::<conformance::Report>();
    assert_send::<sdrk_testkit::TxRecord>();
}

#[test]
fn virtual_tx_records_bytes_and_metas() {
    let mut storage = [0u8; 16];
    let mut tx = VirtualTx::new(formats::CS16, rate(), &mut storage, &[]).unwrap();
    let burst = BlockMeta {
        flags: BlockFlags::BURST_START,
        tx_time_ns: Some(5_000),
        ..BlockMeta::default()
    };
    assert_eq!(tx.write(&[1, 2, 3, 4], &burst).unwrap(), 4);
    let next = BlockMeta {
        start: SampleIndex(1),
        flags: BlockFlags::BURST_END,
        ..BlockMeta::default()
    };
    assert_eq!(tx.write(&[5, 6, 7, 8, 9, 10, 11, 12], &next).unwrap(), 8);
    assert_eq!(tx.written(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(tx.samples_written(), 3);
    let r = tx.records();
    assert_eq!(r.len(), 2);
    assert_eq!(r[0].tx_time_ns, Some(5_000));
    assert_eq!((r[0].samples, r[0].position), (1, 0));
    assert_eq!(r[1].flags, BlockFlags::BURST_END);
    assert_eq!(
        (r[1].samples, r[1].position, r[1].start),
        (2, 1, SampleIndex(1))
    );
}

#[test]
fn virtual_tx_accepts_less_when_full_and_fails_when_empty() {
    let mut storage = [0u8; 6];
    let mut tx = VirtualTx::new(formats::CS16, rate(), &mut storage, &[]).unwrap();
    // 12 bytes offered, room for 6: whole samples (4 bytes) -> 4 accepted.
    assert_eq!(tx.write(&[0; 12], &BlockMeta::default()).unwrap(), 4);
    let e = tx.write(&[0; 4], &BlockMeta::default()).unwrap_err();
    assert!(matches!(e, Error::BufferTooSmall { .. }));
    // an empty write is always fine
    assert_eq!(tx.write(&[], &BlockMeta::default()).unwrap(), 0);
}

#[test]
fn virtual_tx_rejects_partial_samples_and_bad_statuses() {
    let mut storage = [0u8; 16];
    let mut tx = VirtualTx::new(formats::CS16, rate(), &mut storage, &[]).unwrap();
    assert!(tx.write(&[0; 3], &BlockMeta::default()).is_err());
    let mut storage = [0u8; 16];
    let unsorted = [(5, BlockFlags::LATE), (2, BlockFlags::UNDERRUN)];
    assert!(VirtualTx::new(formats::CS8, rate(), &mut storage, &unsorted).is_err());
}

#[test]
fn virtual_tx_records_at_most_64_writes() {
    let mut storage = [0u8; 128];
    let mut tx = VirtualTx::new(formats::CS8, rate(), &mut storage, &[]).unwrap();
    for _ in 0..64 {
        tx.write(&[0, 0], &BlockMeta::default()).unwrap();
    }
    assert!(matches!(
        tx.write(&[0, 0], &BlockMeta::default()).unwrap_err(),
        Error::BufferTooSmall { .. }
    ));
    assert_eq!(tx.records().len(), 64);
}

#[test]
fn virtual_tx_injects_statuses_at_the_configured_indices() {
    let mut storage = [0u8; 64];
    let statuses = [
        (4, BlockFlags::UNDERRUN),
        (4, BlockFlags::LATE),
        (10, BlockFlags::UNDERRUN),
    ];
    let mut tx = VirtualTx::new(formats::CS8, rate(), &mut storage, &statuses).unwrap();
    assert_eq!(tx.poll_status().unwrap(), None);
    tx.write(&[0; 6], &BlockMeta::default()).unwrap(); // 3 samples
    assert_eq!(tx.poll_status().unwrap(), None);
    tx.write(&[0; 2], &BlockMeta::default()).unwrap(); // 4 samples
    let a = tx.poll_status().unwrap().unwrap();
    assert_eq!((a.start.0, a.flags), (4, BlockFlags::UNDERRUN));
    let b = tx.poll_status().unwrap().unwrap();
    assert_eq!((b.start.0, b.flags), (4, BlockFlags::LATE));
    assert_eq!(tx.poll_status().unwrap(), None);
    tx.write(&[0; 20], &BlockMeta::default()).unwrap();
    assert_eq!(tx.poll_status().unwrap().unwrap().start.0, 10);
    assert_eq!(tx.poll_status().unwrap(), None);
}

#[test]
fn virtual_rx_and_tx_pass_the_conformance_checks() {
    for format in [formats::CU8, formats::CS16, formats::CF32, formats::RS16] {
        let mut scratch = [Complex64::new(0.0, 0.0); 64];
        let src = ToneSource {
            freq: 0.05,
            amp: 0.5,
            phase: 0.0,
        };
        let gaps = [(64u64, 100u64), (640, 8)];
        let anchor = TimeAnchor {
            index: SampleIndex(0),
            unix_ns: 1_000,
            source: TimeSource::Host,
        };
        let mut rx = VirtualRx::new(
            format,
            rate(),
            src,
            &mut scratch,
            ChunkPlan::new(7, 40),
            &gaps,
            Some(anchor),
        )
        .unwrap();
        let report = conformance::check_rx(&mut rx, 40, 1024).unwrap();
        assert_eq!(report.blocks, 40);
        assert_eq!(report.gaps, 2);
        assert_eq!(report.anchors, 1);
        assert!(!report.ended);

        let mut storage = [0u8; 4096];
        let mut tx = VirtualTx::new(format, rate(), &mut storage, &[]).unwrap();
        let report = conformance::check_tx(&mut tx, 10, 100).unwrap();
        assert_eq!(report.blocks, 10);
        assert!(report.samples > 0);
    }
}

/// A scripted receiver for the negative tests.
struct Script {
    blocks: Vec<(usize, BlockMeta)>,
    format: SampleFormat,
    formats_after: Vec<SampleFormat>,
    next: usize,
}

impl Script {
    fn new(blocks: Vec<(usize, BlockMeta)>) -> Self {
        Self {
            blocks,
            format: formats::CS16,
            formats_after: Vec::new(),
            next: 0,
        }
    }
}

impl RxStream for Script {
    type Error = Error;
    fn format(&self) -> SampleFormat {
        self.formats_after
            .get(self.next.wrapping_sub(1))
            .copied()
            .unwrap_or(self.format)
    }
    fn sample_rate(&self) -> Rate {
        rate()
    }
    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Error> {
        let (n, m) = self
            .blocks
            .get(self.next)
            .copied()
            .unwrap_or((0, BlockMeta::default()));
        self.next += 1;
        let k = n.min(buf.len());
        buf[..k].fill(0);
        Ok((n, m))
    }
}

fn at(start: u64) -> BlockMeta {
    BlockMeta {
        start: SampleIndex(start),
        ..BlockMeta::default()
    }
}

fn with(start: u64, flags: BlockFlags) -> BlockMeta {
    BlockMeta {
        start: SampleIndex(start),
        flags,
        ..BlockMeta::default()
    }
}

fn rx_msg(blocks: Vec<(usize, BlockMeta)>) -> &'static str {
    let mut s = Script::new(blocks);
    conformance::check_rx(&mut s, 10, 64).unwrap_err().msg
}

#[test]
fn check_rx_detects_each_violation() {
    // good baseline
    let mut good = Script::new(vec![(8, at(0)), (8, at(2)), (4, at(4)), (0, at(5))]);
    let r = conformance::check_rx(&mut good, 10, 64).unwrap();
    assert!(r.ended);
    assert_eq!((r.blocks, r.samples, r.bytes), (3, 5, 20));

    assert_eq!(
        rx_msg(vec![(8, at(0)), (8, at(3))]),
        "start index does not continue the previous block"
    );
    assert_eq!(
        rx_msg(vec![(8, at(0)), (8, with(2, BlockFlags::GAP_BEFORE))]),
        "GAP_BEFORE without a jump of the start index"
    );
    assert_eq!(
        rx_msg(vec![(6, at(0))]),
        "read did not return whole samples"
    );
    assert_eq!(
        rx_msg(vec![(100, at(0))]),
        "read returned more bytes than the buffer holds"
    );
    // a jump with GAP_BEFORE is fine
    let mut gap = Script::new(vec![(8, at(0)), (8, with(10, BlockFlags::GAP_BEFORE))]);
    assert_eq!(conformance::check_rx(&mut gap, 5, 64).unwrap().gaps, 1);

    // a format change needs RETUNED
    let mut s = Script::new(vec![(8, at(0)), (8, at(2))]);
    s.formats_after = vec![formats::CS16, formats::CS8];
    assert_eq!(
        conformance::check_rx(&mut s, 5, 64).unwrap_err().msg,
        "format or rate changed without RETUNED"
    );
    let mut s = Script::new(vec![(8, at(0)), (8, with(2, BlockFlags::RETUNED))]);
    s.formats_after = vec![formats::CS16, formats::CS8];
    let r = conformance::check_rx(&mut s, 5, 64).unwrap();
    assert_eq!(r.retunes, 1);
}

#[test]
fn check_rx_checks_the_anchor_times() {
    let anchor = |index: u64, ns: i64| TimeAnchor {
        index: SampleIndex(index),
        unix_ns: ns,
        source: TimeSource::Host,
    };
    let with_anchor = |start: u64, a: TimeAnchor| BlockMeta {
        start: SampleIndex(start),
        anchor: Some(a),
        ..BlockMeta::default()
    };
    // The second anchor moves the time of block 2 behind block 1.
    let mut s = Script::new(vec![
        (8, with_anchor(0, anchor(0, 1_000_000))),
        (8, with_anchor(2, anchor(2, 500))),
    ]);
    assert_eq!(
        conformance::check_rx(&mut s, 5, 64).unwrap_err().msg,
        "time of the block start decreased"
    );
    let mut s = Script::new(vec![
        (8, with_anchor(0, anchor(0, 1_000_000))),
        (8, with_anchor(2, anchor(2, 1_000_000 + 41_667))),
    ]);
    assert_eq!(conformance::check_rx(&mut s, 5, 64).unwrap().anchors, 2);
}

#[test]
fn check_rx_rejects_a_buffer_larger_than_max_buf() {
    let mut s = Script::new(vec![(8, at(0))]);
    let e = conformance::check_rx(&mut s, 1, MAX_BUF + 1).unwrap_err();
    assert_eq!(e.msg, "buf_size is larger than MAX_BUF");
    let mut big = vec![0u8; MAX_BUF * 2];
    let mut s = Script::new(vec![(8, at(0))]);
    assert!(conformance::check_rx_with(&mut s, 1, &mut big).is_ok());
}

/// A transmitter with a scripted misbehaviour.
struct BadTx {
    mode: u8,
    status: Option<BlockMeta>,
}

impl TxStream for BadTx {
    type Error = Error;
    fn format(&self) -> SampleFormat {
        formats::CS16
    }
    fn sample_rate(&self) -> Rate {
        rate()
    }
    fn write(&mut self, buf: &[u8], _meta: &BlockMeta) -> Result<usize, Error> {
        Ok(match self.mode {
            0 => buf.len() + 4,
            1 => 0,
            2 => buf.len().saturating_sub(1).max(1),
            _ => buf.len(),
        })
    }
    fn poll_status(&mut self) -> Result<Option<BlockMeta>, Error> {
        Ok(self.status.take())
    }
}

#[test]
fn check_tx_detects_each_violation() {
    let msg = |mode: u8, status: Option<BlockMeta>| {
        let mut tx = BadTx { mode, status };
        conformance::check_tx(&mut tx, 3, 16).unwrap_err().msg
    };
    assert_eq!(msg(0, None), "write accepted more bytes than offered");
    assert_eq!(msg(1, None), "write made no progress");
    assert_eq!(msg(2, None), "write accepted a partial sample");
    assert_eq!(msg(3, Some(BlockMeta::default())), "status without flags");
    let mut ok = BadTx {
        mode: 3,
        status: Some(with(0, BlockFlags::UNDERRUN)),
    };
    let r = conformance::check_tx(&mut ok, 3, 16).unwrap();
    assert_eq!((r.blocks, r.statuses, r.samples), (3, 1, 12));
    let mut tx = BadTx {
        mode: 3,
        status: None,
    };
    assert_eq!(
        conformance::check_tx(&mut tx, 1, MAX_BUF + 1)
            .unwrap_err()
            .msg,
        "buf_size is larger than MAX_BUF"
    );
    assert_eq!(
        conformance::check_tx(&mut tx, 1, 3).unwrap_err().msg,
        "buf_size is smaller than one sample"
    );
}

#[test]
fn check_tx_handles_partial_writes() {
    // A transmitter that accepts at most one sample per call still conforms.
    struct Trickle(usize);
    impl TxStream for Trickle {
        type Error = Error;
        fn format(&self) -> SampleFormat {
            formats::CS16
        }
        fn sample_rate(&self) -> Rate {
            rate()
        }
        fn write(&mut self, buf: &[u8], _m: &BlockMeta) -> Result<usize, Error> {
            self.0 += 1;
            Ok(buf.len().min(4))
        }
        fn poll_status(&mut self) -> Result<Option<BlockMeta>, Error> {
            Ok(None)
        }
    }
    let mut t = Trickle(0);
    let r = conformance::check_tx(&mut t, 2, 16).unwrap();
    assert_eq!(r.samples, 8);
    assert_eq!(t.0, 8);
}
