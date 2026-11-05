//! Golden tests of M39: the Gaussian pulse of `design::gaussian` and the blocks `fsk_mod` and
//! `fsk_demod` (test 1).
//!
//! The block cases are run in one piece (one call with room for every output) and in 20 chunk
//! plans (plan `k` of a case uses the seeds `seed 100 + k` for the input and `seed 100 + 50 + k`
//! for the output chunks): each call gets an input chunk of 1 to 700 samples and room for 1 to
//! 200 outputs, so calls also end with a full output and the caller passes the unconsumed input
//! again. Every run must match the golden file (`tolerance`: `fsk_mod` against the float64
//! chain, `fsk_demod` against the float64 chain with the float32 model of `symbol_sync`), produce
//! the stored output count, and the chunked runs must equal the whole run bit for bit.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation
)]

use sdrk::Rate;
use sdrk_blocks::design::gaussian;
use sdrk_blocks::{
    Block, FskDemodBlock, FskDemodConfig, FskModBlock, FskModConfig, StreamElem, TED_GAIN_2FSK,
};
use sdrk_core::kernels::symbol_sync::{SymbolSyncState, Ted};
use sdrk_core::{C32, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};
use sdrk_testkit::{ChunkPlan, VecStorage, bit_exact};

/// Chunk plans per case.
const PLANS: u64 = 20;
/// The largest input and output chunks of the plans.
const MAX_IN: usize = 700;
const MAX_OUT: usize = 200;

#[test]
fn the_gaussian_design_cases_are_present() {
    let names: Vec<String> = golden::list("design")
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with("gaussian_"))
        .collect();
    assert_eq!(names, ["gaussian_bt0p5_sps8_span4"]);
}

#[test]
fn the_fsk_block_cases_are_present() {
    let names: Vec<String> = golden::list("blocks")
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with("fsk_"))
        .collect();
    assert_eq!(names, ["fsk_demod_2fsk_f32", "fsk_mod_gfsk_c32"]);
}

#[test]
fn gaussian_matches_the_python_formula() {
    let case = golden::load("design", "gaussian_bt0p5_sps8_span4").unwrap();
    let bt = case.param_f64("bt").unwrap();
    let sps = usize::try_from(case.param_u64("sps").unwrap()).unwrap();
    let span = usize::try_from(case.param_u64("span").unwrap()).unwrap();
    let n = usize::try_from(case.param_u64("num_taps").unwrap()).unwrap();
    assert_eq!((bt, sps, span, n), (0.5, 8, 4, 33));
    let Array::F64(want) = case.array("expected").unwrap() else {
        panic!("expected must be f64");
    };
    let Tolerance::MaxAbsErr { max, .. } = case.tolerance else {
        panic!("design cases use max_abs_err");
    };
    assert_eq!(max, 1e-12);
    let mut got = vec![0.0; n];
    gaussian(bt, sps, span, &mut got).unwrap();
    assert_eq!(want.len(), n);
    let err = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(err <= max, "largest error {err}");
}

fn usize_param(case: &GoldenCase, key: &str) -> usize {
    usize::try_from(case.param_u64(key).unwrap()).unwrap()
}

/// Runs `block` over `input`: in one call with room for everything (`plan` `None`), or with
/// the input and output chunk sizes of plan `k`; resubmits what the block did not consume.
fn run<I: StreamElem, O: StreamElem + Default, B: Block<In = I, Out = O>>(
    block: &mut B,
    input: &[I],
    plan: Option<u64>,
    seed: u64,
) -> Vec<O> {
    let mut out = Vec::new();
    let mut room = vec![O::default(); input.len() * 16 + 16];
    let (mut ins, mut outs) = match plan {
        Some(k) => (
            Some(ChunkPlan::new(seed * 100 + k, MAX_IN).sizes()),
            Some(ChunkPlan::new(seed * 100 + 50 + k, MAX_OUT).sizes()),
        ),
        None => (None, None),
    };
    let mut pending: Vec<I> = Vec::new();
    let mut pos = 0;
    while pos < input.len() || !pending.is_empty() {
        if pos < input.len() {
            let want = ins
                .as_mut()
                .map_or(input.len(), |s| s.next().unwrap())
                .min(input.len() - pos);
            pending.extend_from_slice(&input[pos..pos + want]);
            pos += want;
        }
        let r = outs
            .as_mut()
            .map_or(room.len(), |s| s.next().unwrap())
            .min(room.len());
        let mut p = block.process(&pending, &mut room[..r]).unwrap();
        if p.consumed == 0 && p.produced == 0 && r < room.len() {
            // Too little room for one output step of the block: call again with all of it.
            p = block.process(&pending, &mut room).unwrap();
        }
        out.extend_from_slice(&room[..p.produced]);
        pending.drain(..p.consumed);
        if p.consumed == 0 && p.produced == 0 && pos == input.len() {
            // The rest is too short for the block; nothing more can come out.
            break;
        }
    }
    // Drain what the block still holds (fsk_demod keeps symbols that did not fit).
    loop {
        let p = block.process(&[], &mut room).unwrap();
        out.extend_from_slice(&room[..p.produced]);
        if p.produced == 0 {
            break;
        }
    }
    out
}

/// The whole run and the 20 chunked runs match the case and each other.
fn check<I: StreamElem, O: StreamElem + Default + Sample, B: Block<In = I, Out = O>>(
    case: &GoldenCase,
    input: &[I],
    make: impl Fn() -> B,
) {
    let expected = case.array("expected").unwrap();
    let whole = run(&mut make(), input, None, case.seed);
    let count = case
        .param_u64("produced")
        .or_else(|_| case.param_u64("outputs"));
    assert_eq!(
        whole.len() as u64,
        count.unwrap(),
        "{}: output count",
        case.case
    );
    golden::assert_matches(case, expected, &whole).unwrap();
    for k in 0..PLANS {
        let chunked = run(&mut make(), input, Some(k), case.seed);
        assert_eq!(
            bit_exact(&whole, &chunked),
            None,
            "{}: plan {k} changed the output",
            case.case
        );
    }
}

#[test]
fn fsk_mod_matches_the_float64_chain() {
    let case = golden::load("blocks", "fsk_mod_gfsk_c32").unwrap();
    let Array::F32(input) = case.array("input").unwrap() else {
        panic!("input must be f32");
    };
    let Array::F32(taps) = case.array("taps").unwrap() else {
        panic!("taps must be f32");
    };
    let cfg = FskModConfig {
        levels: usize_param(&case, "levels"),
        samples_per_symbol: usize_param(&case, "samples_per_symbol"),
        deviation_hz: case.param_f64("deviation_hz").unwrap(),
        bt: case.param_f64("bt").unwrap(),
        span: usize_param(&case, "span"),
    };
    let rate = Rate::hz(case.param_f64("bit_rate_hz").unwrap() as u64).unwrap();
    assert_eq!(
        cfg.rate_out(rate).unwrap(),
        Rate::hz(case.param_f64("rate_out_hz").unwrap() as u64).unwrap()
    );
    let make = || FskModBlock::<VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap();
    // The taps of the block are the generator's (float64 design, one rounding to f32) up to
    // the last bit of the float64 sums.
    let block = make();
    assert_eq!(block.taps().len(), usize_param(&case, "taps"));
    for (a, b) in block.taps().iter().zip(taps) {
        assert!((a - b).abs() <= f32::EPSILON * b.abs(), "tap {a} vs {b}");
    }
    check(&case, input, make);
}

#[test]
fn fsk_demod_matches_the_float64_chain_with_the_float32_timing_loop() {
    let case = golden::load("blocks", "fsk_demod_2fsk_f32").unwrap();
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!("input must be cf32");
    };
    let cfg = FskDemodConfig {
        levels: usize_param(&case, "levels"),
        symbol_rate_hz: case.param_f64("symbol_rate_hz").unwrap(),
        deviation_hz: case.param_f64("deviation_hz").unwrap(),
        loop_bw: case.param_f64("loop_bw").unwrap(),
    };
    let rate = Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap();
    let sync = cfg.symbol_sync().unwrap();
    assert_eq!(sync.ted_gain, TED_GAIN_2FSK);
    assert_eq!(sync.ted_gain, case.param_f64("ted_gain").unwrap());
    let sps = cfg.sps(rate).unwrap();
    // The JSON reader may round a decimal float to a neighbour, so the bits are stored too.
    assert_eq!(sps.to_bits(), case.param_u64("sps_bits").unwrap());
    // The loop constants of the inner symbol_sync are the generator's bits.
    let st = SymbolSyncState::<f32>::new(
        sps,
        Ted::Gardner,
        sync.loop_bw,
        sync.damping,
        sync.max_deviation,
        sync.ted_gain,
    )
    .unwrap();
    let g = st.gains();
    assert_eq!(g.kp.to_bits(), case.param_u64("kp_bits").unwrap());
    assert_eq!(g.ki.to_bits(), case.param_u64("ki_bits").unwrap());
    assert_eq!(
        g.max_integ.to_bits(),
        case.param_u64("max_integ_bits").unwrap()
    );
    let make = || FskDemodBlock::<VecStorage>::new(&cfg, rate, &mut VecStorage).unwrap();
    assert_eq!(make().num_taps(), usize_param(&case, "taps"));
    let input: &[C32] = input;
    check(&case, input, make);
}
