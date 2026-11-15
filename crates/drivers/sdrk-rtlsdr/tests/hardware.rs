//! Tests with a real dongle on the USB bus:
//! `cargo test -p sdrk-rtlsdr --features hw-tests -- --ignored`.

#![cfg(feature = "hw-tests")]
#![allow(clippy::unwrap_used)]

use sdrk_device::{Direction, Tuner};
use sdrk_rtlsdr::{Driver, RtlSdr, StreamOptions, UsbDriver};
use sdrk_testkit::conformance;

#[test]
#[ignore = "needs an RTL-SDR on the USB bus"]
fn the_first_dongle_streams_and_meets_the_rx_contract() {
    let bus = UsbDriver;
    let info = bus
        .list()
        .unwrap()
        .into_iter()
        .next()
        .expect("no RTL-SDR found");
    let mut sdr = RtlSdr::open(&bus, &info, false, StreamOptions::default()).unwrap();
    sdr.set_frequency(Direction::Rx, 0, 100e6).unwrap();
    let mut rx = sdr.start_rx().unwrap();
    let mut buf = vec![0u8; 16384];
    let report = conformance::check_rx_with(&mut rx, 200, &mut buf).unwrap();
    assert_eq!(report.blocks, 200);
    sdr.stop_rx();
}
