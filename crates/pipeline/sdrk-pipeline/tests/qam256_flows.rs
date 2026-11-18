//! M44.2 §5: the example flows `qam256_transmitter` and `qam256_receiver` validate without
//! warnings, with the rates of 256-QAM at 25 kBd (200 kbit/s). Test 7 of M44.2 (the flow loop
//! through multipath, noise and a carrier offset) is stopped: with the scrambler of the flows
//! (I44.2-6, D1529) the receiver decodes the frames at every rotation, but 930 of 1000 packets at
//! Eb/N0 26 dB, not 990, the packets lost to the equalizer (open item I44.2-8, D1534).
#![allow(clippy::unwrap_used, clippy::print_stderr)]

mod common;

use sdrk::Rate;
use sdrk_pipeline::{BuildOptions, Registry};

use common::flow;

#[test]
fn the_qam256_flows_validate_without_warnings() {
    for (name, kinds) in [
        (
            "qam256_transmitter",
            &["convert_out", "duc", "qam_frame_encode", "scale"][..],
        ),
        (
            "qam256_receiver",
            &["agc", "convert", "ddc", "qam_frame_decode"][..],
        ),
    ] {
        let report = Registry::standard().validate_json(&flow(name), &BuildOptions::default());
        assert!(report.is_ok(), "{name}: {}", report.to_json());
        assert!(report.warnings.is_empty(), "{name}: {}", report.to_json());
        let mut got: Vec<_> = report.nodes.iter().map(|n| n.kind.as_str()).collect();
        got.sort_unstable();
        assert_eq!(got, kinds, "{name}");
        let rate = |kind: &str| {
            report
                .nodes
                .iter()
                .find(|n| n.kind == kind)
                .map(|n| n.rate_out)
                .unwrap()
        };
        if name == "qam256_transmitter" {
            // 200 kbit/s, 8 samples per 256-QAM symbol of 8 bits: 200 kHz, then 1 MHz.
            assert_eq!(rate("qam_frame_encode"), Rate::hz(200_000).unwrap());
            assert_eq!(rate("duc"), Rate::hz(1_000_000).unwrap());
        } else {
            // 200 kHz into the demodulator.
            assert_eq!(rate("ddc"), Rate::hz(200_000).unwrap());
        }
        eprintln!(
            "{name}: cost {:.1} per input sample, {} bytes",
            report.cost_per_input_sample, report.memory_bytes
        );
    }
}
