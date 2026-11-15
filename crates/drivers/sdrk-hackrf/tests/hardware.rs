//! Tests with a real HackRF One on the USB bus:
//! `cargo test -p sdrk-hackrf --features hw-tests -- --ignored --test-threads 1`.
//! They receive only; nothing is transmitted.

#![cfg(feature = "hw-tests")]
#![allow(clippy::unwrap_used)]

use sdrk_device::{Direction, Tuner};
use sdrk_hackrf::{Driver, HackRf, StreamOptions, UsbDriver};
use sdrk_testkit::conformance;

#[test]
#[ignore = "needs a HackRF One on the USB bus"]
fn the_first_board_streams_and_meets_the_rx_contract() {
    let bus = UsbDriver;
    let info = bus
        .list()
        .unwrap()
        .into_iter()
        .next()
        .expect("no HackRF found");
    let mut sdr = HackRf::open(&bus, &info, StreamOptions::default()).unwrap();
    assert_ne!(sdr.board().firmware, "");
    sdr.set_frequency(Direction::Rx, 0, 100e6).unwrap();
    let mut rx = sdr.start_rx().unwrap();
    let mut buf = vec![0u8; 65536];
    let report = conformance::check_rx_with(&mut rx, 200, &mut buf).unwrap();
    assert_eq!(report.blocks, 200);
    sdr.stop().unwrap();
}
