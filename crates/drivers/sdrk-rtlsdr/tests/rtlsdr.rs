//! `RtlSdr` against the simulated bus: device traits, stream contract,
//! overflow, pause on reconfiguration, unplug and direct sampling.

#![allow(clippy::unwrap_used)]

use sdrk_device::{
    BlockFlags, Device, Direction, ErrorKind, GainControl, Rate, RateControl, RxStream, SdrkError,
    Tuner, formats,
};
use sdrk_rtlsdr::fake::{FakeDongle, FakeDriver};
use sdrk_rtlsdr::{DirectSampling, Driver, RtlSdr, StreamOptions};
use sdrk_testkit::conformance;
use std::time::Duration;

const TRANSFER: usize = 4096;

fn options() -> StreamOptions {
    StreamOptions {
        buffers: 2,
        buffer_bytes: TRANSFER,
        queue_transfers: 4,
        read_timeout: Duration::from_secs(2),
    }
}

fn open() -> (FakeDriver, RtlSdr) {
    let bus = FakeDriver::new(vec![FakeDongle::r820t(0, Some("00000001"), 1)]);
    let info = bus.list().unwrap().remove(0);
    let sdr = RtlSdr::open(&bus, &info, true, options()).unwrap();
    (bus, sdr)
}

#[test]
fn capabilities_and_defaults() {
    let (bus, sdr) = open();
    assert_eq!(sdr.driver(), "rtlsdr");
    assert_eq!(sdr.serial(), Some("00000001"));
    let caps = sdr.capabilities();
    assert_eq!((caps.rx_channels, caps.tx_channels), (1, 0));
    assert_eq!(caps.formats, &[formats::CU8]);
    let s = bus.dongle(0).unwrap().settings;
    assert_eq!(s.center_hz, Some(100_000_000));
    assert_eq!(s.rate_hz, Some(2_400_000));
    assert_eq!(sdr.frequency(Direction::Rx, 0).unwrap(), 100e6);
    let r = sdr.frequency_ranges(Direction::Rx, 0)[0];
    assert_eq!((r.min, r.max), (24e6, 1766e6));
    assert_eq!(sdr.frequency_ranges(Direction::Tx, 0).len(), 0);
    assert_eq!(sdr.sample_rates(Direction::Rx, 0).len(), 2);
    let stages = sdr.gain_stages(Direction::Rx, 0);
    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].0, "tuner");
    assert_eq!(stages[0].1.max, 49.6);
}

#[test]
fn setters_go_to_the_hardware() {
    let (bus, mut sdr) = open();
    assert_eq!(
        sdr.set_frequency(Direction::Rx, 0, 433.92e6).unwrap(),
        433.92e6
    );
    assert_eq!(bus.dongle(0).unwrap().settings.center_hz, Some(433_920_000));
    let e = sdr.set_frequency(Direction::Rx, 0, 10e6).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidArgument);
    let e = sdr.set_frequency(Direction::Tx, 0, 100e6).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidArgument);
    let rate = sdr
        .set_sample_rate(Direction::Rx, 0, Rate::hz(1_024_000).unwrap())
        .unwrap();
    assert!((rate.as_f64() - 1_024_000.0).abs() < 1.0);
    assert_eq!(bus.dongle(0).unwrap().settings.rate_hz, Some(1_024_000));
    let e = sdr
        .set_sample_rate(Direction::Rx, 0, Rate::hz(500_000).unwrap())
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidArgument);
    assert_eq!(
        sdr.set_gain(Direction::Rx, 0, Some("tuner"), 30.0).unwrap(),
        29.7
    );
    assert_eq!(sdr.set_gain(Direction::Rx, 0, None, 100.0).unwrap(), 49.6);
    assert!(sdr.set_gain(Direction::Rx, 0, Some("lna"), 10.0).is_err());
    sdr.set_agc(Direction::Rx, 0, true).unwrap();
    assert!(sdr.agc());
    sdr.set_ppm(5).unwrap();
    assert_eq!(bus.dongle(0).unwrap().settings.ppm, Some(5));
    assert_eq!(
        sdr.set_ppm(200).unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    sdr.set_bias_tee(true).unwrap();
    assert_eq!(bus.dongle(0).unwrap().settings.bias_tee, Some(true));
}

#[test]
fn stream_meets_the_rx_contract() {
    // The queue holds all fed transfers, so that a slow reader (an emulator
    // in CI) sees no overflow.
    let bus = FakeDriver::new(vec![FakeDongle::r820t(0, Some("00000001"), 1)]);
    let info = bus.list().unwrap().remove(0);
    let opts = StreamOptions {
        queue_transfers: 16,
        ..options()
    };
    let mut sdr = RtlSdr::open(&bus, &info, true, opts).unwrap();
    let mut rx = sdr.start_rx().unwrap();
    assert!(sdr.is_streaming());
    assert_eq!(sdr.start_rx().unwrap_err().kind(), ErrorKind::InvalidState);
    bus.feed(0, 8);
    let mut buf = vec![0u8; TRANSFER];
    let report = conformance::check_rx_with(&mut rx, 8, &mut buf).unwrap();
    assert_eq!(report.blocks, 8);
    assert_eq!(report.gaps, 0);
    sdr.stop_rx();
    let (n, _) = rx.read(&mut [0u8; 64]).unwrap();
    assert_eq!(n, 0);
    assert!(!sdr.is_streaming());
}

#[test]
fn first_block_has_a_host_anchor_and_the_sample_pattern() {
    let (bus, mut sdr) = open();
    let mut rx = sdr.start_rx().unwrap();
    bus.feed(0, 1);
    let mut buf = vec![0u8; TRANSFER];
    let (n, meta) = rx.read(&mut buf).unwrap();
    assert_eq!(n, TRANSFER);
    assert_eq!(meta.start.0, 0);
    assert!(meta.anchor.is_some());
    // Sample k is the byte pair (k mod 256, k / 256 mod 256).
    assert_eq!(&buf[..6], &[0, 0, 1, 0, 2, 0]);
    let (_, meta) = {
        bus.feed(0, 1);
        rx.read(&mut buf).unwrap()
    };
    assert!(meta.anchor.is_none());
    assert_eq!(meta.start.0, (TRANSFER / 2) as u64);
}

#[test]
fn a_full_queue_drops_transfers_and_flags_the_gap() {
    let (bus, mut sdr) = open();
    let mut rx = sdr.start_rx().unwrap();
    // The queue holds 4 transfers; 10 arrive before the first read.
    bus.feed(0, 10);
    assert!(bus.wait_drained(0, Duration::from_secs(5)));
    let mut buf = vec![0u8; TRANSFER];
    let mut index = 0;
    for _ in 0..4 {
        let (n, meta) = rx.read(&mut buf).unwrap();
        assert_eq!(meta.start.0, index);
        assert_eq!(meta.flags, BlockFlags::empty());
        index += (n / 2) as u64;
    }
    assert!(sdr.lost_bytes() >= 6 * TRANSFER as u64);
    bus.feed(0, 1);
    let (_, meta) = rx.read(&mut buf).unwrap();
    assert!(meta.flags.contains(BlockFlags::OVERFLOW));
    assert!(meta.flags.contains(BlockFlags::GAP_BEFORE));
    assert_eq!(meta.start.0, index + 6 * (TRANSFER / 2) as u64);
}

#[test]
fn a_setter_while_streaming_pauses_with_a_gap() {
    let (bus, mut sdr) = open();
    let mut rx = sdr.start_rx().unwrap();
    bus.feed(0, 1);
    let mut buf = vec![0u8; TRANSFER];
    let (n, _) = rx.read(&mut buf).unwrap();
    sdr.set_frequency(Direction::Rx, 0, 145e6).unwrap();
    assert!(sdr.is_streaming());
    assert!(bus.wait_streaming(0, Duration::from_secs(5)));
    bus.feed(0, 1);
    let (_, meta) = rx.read(&mut buf).unwrap();
    assert!(meta.flags.contains(BlockFlags::GAP_BEFORE));
    assert!(!meta.flags.contains(BlockFlags::OVERFLOW));
    assert!(meta.start.0 > (n / 2) as u64);
    assert_eq!(
        bus.dongle(0).unwrap().transfers_at.get(&145_000_000),
        Some(&1)
    );
}

#[test]
fn unplugging_fails_the_stream_with_disconnected() {
    let (bus, mut sdr) = open();
    let mut rx = sdr.start_rx().unwrap();
    assert!(bus.wait_streaming(0, Duration::from_secs(5)));
    bus.unplug(0);
    let e = rx.read(&mut [0u8; 64]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Disconnected);
    assert!(!sdr.is_streaming());
}

#[test]
fn direct_sampling_changes_the_range_and_clamps_the_centre() {
    let (bus, mut sdr) = open();
    let center = sdr.set_direct_sampling(DirectSampling::Q).unwrap();
    assert_eq!(center, 28_800_000);
    let r = sdr.frequency_ranges(Direction::Rx, 0)[0];
    assert_eq!((r.min, r.max), (0.0, 28.8e6));
    sdr.set_frequency(Direction::Rx, 0, 7.1e6).unwrap();
    let center = sdr.set_direct_sampling(DirectSampling::Off).unwrap();
    assert_eq!(center, 24_000_000);
    let s = bus.dongle(0).unwrap().settings;
    assert_eq!(s.rate_hz, Some(2_400_000));
    assert_eq!(s.center_hz, Some(24_000_000));
}
