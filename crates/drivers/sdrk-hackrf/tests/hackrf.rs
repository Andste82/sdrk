//! `HackRf` against the simulated bus: safety on open, device traits, the
//! half-duplex RF path, RX and TX streams, underruns and unplugging.

#![allow(clippy::unwrap_used)]

use sdrk_device::{
    BandwidthControl, BlockFlags, BlockMeta, Device, Direction, ErrorKind, GainControl, Rate,
    RateControl, RxStream, SdrkError, Tuner, TxStream, formats,
};
use sdrk_hackrf::fake::{FakeBoard, FakeCall, FakeDriver};
use sdrk_hackrf::{Driver, HackRf, StreamOptions, TransceiverMode};
use sdrk_testkit::conformance;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(5);

fn options() -> StreamOptions {
    StreamOptions {
        rx_buffers: 2,
        rx_buffer_bytes: 4096,
        queue_transfers: 4,
        tx_buffers: 2,
        tx_buffer_bytes: 16384,
        tx_queue_bytes: 65536,
        ..StreamOptions::default()
    }
}

fn open() -> (FakeDriver, HackRf) {
    let bus = FakeDriver::new(vec![FakeBoard::example()]);
    let info = bus.list().unwrap().remove(0);
    let sdr = HackRf::open(&bus, &info, options()).unwrap();
    (bus, sdr)
}

#[test]
fn opening_switches_the_transmitter_off() {
    let (bus, sdr) = open();
    let calls = bus.board(0).unwrap().calls;
    assert_eq!(
        &calls[..3],
        &[
            FakeCall::Mode(TransceiverMode::Off),
            FakeCall::Amp(false),
            FakeCall::AntennaPower(false)
        ]
    );
    assert_eq!(sdr.driver(), "hackrf");
    assert!(sdr.serial().is_some());
    let caps = sdr.capabilities();
    assert_eq!(
        (caps.rx_channels, caps.tx_channels, caps.full_duplex),
        (1, 1, false)
    );
    assert_eq!(caps.formats, &[formats::CS8]);
    assert_eq!(sdr.gain_stages(Direction::Rx, 0).len(), 3);
    assert_eq!(sdr.gain_stages(Direction::Tx, 0).len(), 2);
    assert_eq!(sdr.bandwidths(Direction::Rx, 0).len(), 16);
    assert_eq!(sdr.frequency(Direction::Rx, 0).unwrap(), 100e6);
    assert_eq!(
        sdr.frequency(Direction::Tx, 0).unwrap_err().kind(),
        ErrorKind::InvalidState
    );
}

#[test]
fn values_reach_the_hardware_only_when_the_channel_streams() {
    let (bus, mut sdr) = open();
    bus.clear_calls(0);
    sdr.set_frequency(Direction::Rx, 0, 433.92e6).unwrap();
    sdr.set_gain(Direction::Rx, 0, Some("lna"), 23.0).unwrap();
    assert_eq!(bus.board(0).unwrap().calls, Vec::new());
    let _rx = sdr.start_rx().unwrap();
    let calls = bus.board(0).unwrap().calls;
    assert!(calls.contains(&FakeCall::Freq(433_920_000)));
    assert!(calls.contains(&FakeCall::Lna(24)));
    assert!(calls.contains(&FakeCall::SampleRate(10_000_000, 1)));
    // The filter follows the rate: the widest not above 7.5 MHz.
    assert!(calls.contains(&FakeCall::Filter(7_000_000)));
    assert!(calls.contains(&FakeCall::Mode(TransceiverMode::Receive)));
    // While RX runs, a change goes out at once.
    bus.clear_calls(0);
    sdr.set_gain(Direction::Rx, 0, Some("vga"), 31.0).unwrap();
    assert_eq!(bus.board(0).unwrap().calls, vec![FakeCall::Vga(32)]);
    // ... but not one of TX.
    bus.clear_calls(0);
    sdr.set_frequency(Direction::Tx, 0, 2.4e9).unwrap();
    assert_eq!(bus.board(0).unwrap().calls, Vec::new());
}

#[test]
fn rates_filters_and_gains_map_to_hardware_steps() {
    let (_bus, mut sdr) = open();
    let r = sdr
        .set_sample_rate(Direction::Rx, 0, Rate::new(20_000_000, 3).unwrap())
        .unwrap();
    assert_eq!(r, Rate::new(20_000_000, 3).unwrap());
    assert!(
        sdr.set_sample_rate(Direction::Rx, 0, Rate::hz(1_000_000).unwrap())
            .is_err()
    );
    assert_eq!(sdr.set_bandwidth(Direction::Rx, 0, 5.2e6).unwrap(), 5e6);
    // The overall gain never switches the amplifier.
    assert_eq!(sdr.set_gain(Direction::Rx, 0, None, 30.0).unwrap(), 30.0);
    let s = sdr.settings(Direction::Rx);
    assert_eq!((s.lna_db, s.vga_db, s.amp), (24, 6, false));
    assert_eq!(sdr.set_gain(Direction::Rx, 0, None, 200.0).unwrap(), 102.0);
    assert!(!sdr.settings(Direction::Rx).amp);
    assert_eq!(sdr.set_gain(Direction::Tx, 0, None, 20.4).unwrap(), 20.0);
    assert_eq!(
        sdr.set_gain(Direction::Rx, 0, Some("amp"), 14.0).unwrap(),
        14.0
    );
    assert!(sdr.settings(Direction::Rx).amp);
    assert!(sdr.set_gain(Direction::Tx, 0, Some("lna"), 8.0).is_err());
    assert!(sdr.set_gain(Direction::Rx, 0, Some("vga"), 70.0).is_err());
    assert_eq!(
        sdr.set_agc(Direction::Rx, 0, true).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
}

#[test]
fn rx_stream_meets_the_contract() {
    // The queue holds all fed transfers, so that a slow reader (an emulator
    // in CI) sees no overflow.
    let bus = FakeDriver::new(vec![FakeBoard::example()]);
    let info = bus.list().unwrap().remove(0);
    let opts = StreamOptions {
        queue_transfers: 16,
        ..options()
    };
    let mut sdr = HackRf::open(&bus, &info, opts).unwrap();
    let mut rx = sdr.start_rx().unwrap();
    assert_eq!(rx.format(), formats::CS8);
    bus.feed(0, 6);
    let mut buf = vec![0u8; 4096];
    let report = conformance::check_rx_with(&mut rx, 6, &mut buf).unwrap();
    assert_eq!(report.blocks, 6);
    assert_eq!(report.gaps, 0);
    sdr.stop().unwrap();
    assert_eq!(rx.read(&mut buf).unwrap().0, 0);
    let board = bus.board(0).unwrap();
    assert_eq!(board.mode, TransceiverMode::Off);
}

#[test]
fn rx_and_tx_share_the_rf_path() {
    let (_bus, mut sdr) = open();
    let _rx = sdr.start_rx().unwrap();
    assert_eq!(sdr.running_direction(), Some(Direction::Rx));
    assert_eq!(sdr.start_tx().unwrap_err().kind(), ErrorKind::InvalidState);
    assert_eq!(sdr.start_rx().unwrap_err().kind(), ErrorKind::InvalidState);
    // A running channel keeps its rate.
    assert_eq!(
        sdr.set_sample_rate(Direction::Rx, 0, Rate::hz(8_000_000).unwrap())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidState
    );
    sdr.stop().unwrap();
    // TX needs a centre frequency.
    assert_eq!(sdr.start_tx().unwrap_err().kind(), ErrorKind::InvalidState);
    sdr.set_frequency(Direction::Tx, 0, 433.92e6).unwrap();
    let _tx = sdr.start_tx().unwrap();
    assert_eq!(sdr.running_direction(), Some(Direction::Tx));
}

#[test]
fn tx_switches_to_transmit_with_the_first_samples_and_reports_the_burst_end() {
    let (bus, mut sdr) = open();
    sdr.set_frequency(Direction::Tx, 0, 433.92e6).unwrap();
    let mut tx = sdr.start_tx().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    assert_ne!(bus.board(0).unwrap().mode, TransceiverMode::Transmit);
    let samples = vec![1u8; 20000];
    let n = tx
        .write(
            &samples,
            &BlockMeta {
                flags: BlockFlags::BURST_START.union(BlockFlags::BURST_END),
                ..BlockMeta::default()
            },
        )
        .unwrap();
    assert_eq!(n, samples.len());
    assert!(bus.wait_for(0, WAIT, |b| b.tx_bytes.len() >= 32768));
    let board = bus.board(0).unwrap();
    assert!(board.tx_mode_ok);
    assert!(board.tx_whole_transfers);
    // The burst is padded with zeros to whole transfers.
    assert!(board.tx_bytes[..20000].iter().all(|b| *b == 1));
    assert!(board.tx_bytes[20000..32768].iter().all(|b| *b == 0));
    let mut done = false;
    for _ in 0..500 {
        if let Some(meta) = tx.poll_status().unwrap()
            && meta.flags.contains(BlockFlags::BURST_END)
        {
            done = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(done);
    sdr.tx_off().unwrap();
    let board = bus.board(0).unwrap();
    assert_eq!(board.mode, TransceiverMode::Off);
    assert!(!board.amp);
}

#[test]
fn firmware_underruns_come_out_one_by_one() {
    let (bus, mut sdr) = open();
    sdr.set_frequency(Direction::Tx, 0, 433.92e6).unwrap();
    let mut tx = sdr.start_tx().unwrap();
    tx.write(&vec![1u8; 65536], &BlockMeta::default()).unwrap();
    assert!(bus.wait_for(0, WAIT, |b| b.tx_streaming));
    bus.tx_underrun(0, 2);
    let mut underruns = 0;
    for _ in 0..500 {
        match tx.poll_status().unwrap() {
            Some(meta) if meta.flags.contains(BlockFlags::UNDERRUN) => underruns += 1,
            _ => std::thread::sleep(Duration::from_millis(5)),
        }
        if underruns == 2 {
            break;
        }
    }
    assert_eq!(underruns, 2);
}

#[test]
fn unplugging_fails_the_rx_stream_with_disconnected() {
    let (bus, mut sdr) = open();
    let mut rx = sdr.start_rx().unwrap();
    assert!(bus.wait_for(0, WAIT, |b| b.rx_streaming));
    bus.unplug(0);
    let e = rx.read(&mut [0u8; 64]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Disconnected);
    assert_eq!(sdr.running_direction(), None);
}

#[test]
fn dropping_switches_the_board_off() {
    let (bus, mut sdr) = open();
    sdr.set_bias_tee(Direction::Rx, true).unwrap();
    let _rx = sdr.start_rx().unwrap();
    assert!(bus.board(0).unwrap().antenna_power);
    drop(sdr);
    let board = bus.board(0).unwrap();
    assert_eq!(board.mode, TransceiverMode::Off);
    assert!(!board.antenna_power);
}
