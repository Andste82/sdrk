//! Golden tests of the `squelch` kernel (M19) through the facade `Sdrk`.
//!
//! Every case is run in one piece and in chunks of random size (the plan seed is the case
//! seed); both runs must match the golden file and each other bit for bit. The retune case
//! calls `set_thresholds` at the listed sample, in both runs.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::squelch::{Squelch, SquelchState};
use sdrk_core::{C32, CQ15, ComplexSample, Rate};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// A threshold change applied before sample `at`.
struct Retune {
    at: usize,
    threshold_db: f64,
    hysteresis_db: f64,
}

fn state_of<T: ComplexSample>(case: &GoldenCase) -> SquelchState<T> {
    let rate = Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap();
    SquelchState::new(
        case.param_f64("threshold_db").unwrap(),
        case.param_f64("hysteresis_db").unwrap(),
        case.param_f64("tau_s").unwrap(),
        rate,
    )
    .unwrap()
}

fn retune_of(case: &GoldenCase) -> Option<Retune> {
    case.param_u64("retune_at").ok().map(|at| Retune {
        at: usize::try_from(at).unwrap(),
        threshold_db: case.param_f64("retune_threshold_db").unwrap(),
        hysteresis_db: case.param_f64("retune_hysteresis_db").unwrap(),
    })
}

/// Runs the kernel over `input`; `sizes` cuts the input into chunks (None: one piece per
/// segment). The retune, if any, happens exactly at its sample in both modes.
fn run<T: ComplexSample>(
    mut state: SquelchState<T>,
    input: &[T],
    retune: Option<&Retune>,
    plan: Option<ChunkPlan>,
) -> Vec<T>
where
    Sdrk: Squelch<T>,
{
    let mut out = vec![T::ZERO; input.len()];
    let mut cuts = vec![0, input.len()];
    if let Some(r) = retune {
        cuts.insert(1, r.at);
    }
    let mut sizes = plan.map(|p| p.sizes());
    for seg in cuts.windows(2) {
        let (mut pos, end) = (seg[0], seg[1]);
        if let Some(r) = retune
            && pos == r.at
            && pos != 0
        {
            state
                .set_thresholds(r.threshold_db, r.hysteresis_db)
                .unwrap();
        }
        while pos < end {
            let n = sizes
                .as_mut()
                .map_or(end - pos, |s| s.next().unwrap().min(end - pos));
            Sdrk.squelch(&mut state, &input[pos..pos + n], &mut out[pos..pos + n])
                .unwrap();
            pos += n;
        }
    }
    out
}

fn check_both<T: ComplexSample>(case: &GoldenCase, input: &[T], want: &Array)
where
    Sdrk: Squelch<T>,
{
    let retune = retune_of(case);
    let whole = run(state_of::<T>(case), input, retune.as_ref(), None);
    let plan = ChunkPlan::new(case.seed, 700);
    let chunked = run(state_of::<T>(case), input, retune.as_ref(), Some(plan));
    golden::assert_matches(case, want, &whole).unwrap();
    assert_eq!(
        bit_exact(&whole, &chunked),
        None,
        "{}: block size changed the output",
        case.case
    );
    golden::assert_matches(case, want, &chunked).unwrap();
}

#[test]
fn all_cases_are_present() {
    let names = golden::list("squelch").unwrap();
    assert_eq!(
        names,
        ["bursts_c32_24000", "bursts_cq15_24000", "retune_c32_8192"]
    );
}

#[test]
fn squelch_golden() {
    for name in golden::list("squelch").unwrap() {
        let case = golden::load("squelch", &name).unwrap();
        let want = case.array("expected").unwrap();
        match case.array("input").unwrap() {
            Array::CF32(x) => check_both::<C32>(&case, x, want),
            Array::CI16(x) => check_both::<CQ15>(&case, x, want),
            other => panic!("unexpected input type {}", other.dtype()),
        }
    }
}

#[test]
fn golden_gate_follows_the_documented_positions() {
    // The bursts open the gate at their first sample (the -20 dBFS tone is far above the
    // open level) and close it a few time constants after their end (tau = 96 samples); the
    // -41 dBFS burst stays below the open level and never opens.
    let case = golden::load("squelch", "bursts_c32_24000").unwrap();
    let Array::CF32(out) = case.array("expected").unwrap() else {
        panic!("expected cf32");
    };
    let open: Vec<bool> = out.iter().map(|s| *s != C32::ZERO).collect();
    assert!(!open[..4000].iter().any(|&o| o));
    assert!(open[4000..10_000].iter().all(|&o| o));
    assert!(open[10_000..10_400].iter().all(|&o| o));
    assert!(!open[10_600..15_000].iter().any(|&o| o));
    assert!(open[15_000]);
    assert!(!open[18_600..].iter().any(|&o| o));
}
