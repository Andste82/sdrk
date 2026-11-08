//! `BandPulseDetectBlock` against the golden cases of
//! `tests/golden/sdrk_golden/cases/band_pulse_detect.py` (M45.1): OOK trains with the exact widths
//! and frequencies of an independent Python generator (D973).
//!
//! - `band_pulse_detect/ook_lane` (seed 629): one lane at 250 kHz (a lane view of width 1), 6 PWM,
//!   PPM and Manchester trains at offsets within +-40 kHz, 30 to 36 dB above the noise.
//! - `band_pulse_detect/ook_band` (seed 630): 2 MHz through `ChannelizeBlock` (16 channels, the
//!   defaults) into the detector, 4 PWM trains on two channel boundaries and at the two edges of the
//!   433 MHz band, 30 to 36 dB above the noise in a channel spacing.
//!
//! Every transmission comes out once, with its number of pairs, every width within the case's
//! tolerance (1 step), its start and its frequency within the case's limits.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::print_stdout,
    clippy::large_stack_frames
)]

use sdrk::{C32, Rate};
use sdrk_blocks::{
    BandPulseDetectBlock, BandPulseDetectConfig, Block, ChannelizeBlock, ChannelizeConfig,
    LanesMut, LanesRef, Packet,
};
use sdrk_proto::pulses::{Assembler, Header, Pair};
use sdrk_testkit::VecStorage;
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};

fn f64s(c: &GoldenCase, name: &str) -> Vec<f64> {
    match c.array(name).unwrap() {
        Array::F64(v) => v.clone(),
        other => panic!("{name}: f64 expected, got {}", other.dtype()),
    }
}

fn i32s(c: &GoldenCase, name: &str) -> Vec<i32> {
    match c.array(name).unwrap() {
        Array::I32(v) => v.clone(),
        other => panic!("{name}: i32 expected, got {}", other.dtype()),
    }
}

fn c32s(c: &GoldenCase, name: &str) -> Vec<C32> {
    match c.array(name).unwrap() {
        Array::CF32(v) => v.clone(),
        other => panic!("{name}: cf32 expected, got {}", other.dtype()),
    }
}

/// The lanes of the case's input: the input itself for one lane, else through a channelizer.
fn lanes_of(c: &GoldenCase, x: &[C32]) -> (Vec<C32>, usize) {
    let lanes = c.param_u64("lanes").unwrap() as usize;
    if lanes == 1 {
        return (x.to_vec(), 1);
    }
    let cfg = ChannelizeConfig {
        channels: lanes,
        taps_per_branch: c.param_u64("taps_per_branch").unwrap() as usize,
        ..ChannelizeConfig::default()
    };
    let hop = lanes / 2;
    let n = x.len() / hop * hop;
    let mut chan = ChannelizeBlock::new(&cfg, &mut VecStorage).unwrap();
    let mut out = vec![C32::default(); n / hop * lanes];
    let p = chan
        .process(&x[..n], LanesMut::new(&mut out, lanes).unwrap())
        .unwrap();
    assert_eq!(p.consumed, n);
    (out, lanes)
}

/// The trains of the detector (the defaults) on `lanes`.
fn trains(lanes: &[C32], width: usize, rate: f64) -> Vec<(u64, Header, Vec<Pair>)> {
    let cfg = BandPulseDetectConfig::default();
    let mut det =
        BandPulseDetectBlock::new(&cfg, Rate::hz(rate as u64).unwrap(), width, &mut VecStorage)
            .unwrap();
    let mut asm = Assembler::new(vec![Pair::default(); 4096]);
    let mut out = Vec::new();
    let mut at = 0usize;
    let steps = lanes.len() / width;
    loop {
        let mut buf = vec![Packet::default(); 16];
        let input = LanesRef::new(&lanes[at * width..steps * width], width).unwrap();
        let p = det.process(input, &mut buf).unwrap();
        for q in &buf[..p.produced] {
            if let Some(t) = asm.push(q).unwrap() {
                out.push((t.index, t.header, t.pairs.to_vec()));
            }
        }
        at += p.consumed;
        if at == steps && p.produced == 0 {
            break;
        }
    }
    out
}

fn check(name: &str) {
    let c = golden::load("band_pulse_detect", name).unwrap();
    let Tolerance::MaxAbsErr { max: tol, .. } = c.tolerance else {
        panic!("{name}: max_abs_err expected");
    };
    let fs = c.param_f64("fs").unwrap();
    let rate = c.param_f64("step_rate").unwrap();
    let max_freq = c.param_f64("max_freq_err_hz").unwrap();
    let max_start = c.param_f64("max_start_err_steps").unwrap();
    let x = c32s(&c, "input");
    let starts = f64s(&c, "starts");
    let freqs = f64s(&c, "freqs");
    let levels = f64s(&c, "levels_db");
    let counts = i32s(&c, "pair_counts");
    let pulses = f64s(&c, "pulses");
    let gaps = f64s(&c, "gaps");
    let (lanes, width) = lanes_of(&c, &x);
    let got = trains(&lanes, width, rate);
    assert_eq!(got.len(), starts.len(), "{name}: {got:?}");
    let mut first = 0usize;
    for (k, (index, header, pairs)) in got.iter().enumerate() {
        let n = counts[k] as usize;
        let start = starts[k] * rate / fs;
        let mut worst = 0f64;
        assert_eq!(pairs.len(), n, "{name} {k}: {pairs:?}");
        for (j, p) in pairs.iter().enumerate() {
            worst = worst.max((f64::from(p.pulse) - pulses[first + j]).abs());
            if j + 1 < n {
                worst = worst.max((f64::from(p.gap) - gaps[first + j]).abs());
            } else {
                assert_eq!(p.gap, 0);
            }
        }
        let df = f64::from(header.freq_offset_hz) - freqs[k];
        println!(
            "{name} {k}: {:.0} Hz at {:.1} dB, channel {}, worst width error {worst:.2} steps, \
             frequency error {df:.0} Hz, start {} against {start:.2}",
            freqs[k], levels[k], header.channel, index
        );
        assert!(worst <= tol, "{name} {k}: width error {worst}");
        assert!(df.abs() <= max_freq, "{name} {k}: frequency error {df}");
        assert!(
            (*index as f64 - start).abs() <= max_start,
            "{name} {k}: start {index} against {start}"
        );
        first += n;
    }
}

#[test]
fn ook_trains_on_one_lane_match_the_golden_widths() {
    check("ook_lane");
}

#[test]
fn ook_trains_through_the_channelizer_match_the_golden_widths() {
    check("ook_band");
}
