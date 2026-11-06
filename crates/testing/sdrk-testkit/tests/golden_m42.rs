//! Golden tests of the M42 kernels `rds_shape` and `stereo_encode_rds` through the facade `Sdrk`,
//! `Fallback` and `Reference`.
//!
//! Every case is run in one piece and in 20 chunk plans of random size (seeds derived from the
//! case seed, chunks of 1 to 700 samples); every run must match the golden file bit for bit and
//! leave the same final state. `rds_shape` runs on the stored table rounded to `f32`, and the
//! table of `sdrk-core` must equal the stored f64 table within 1e-12 (and give the same `f32`
//! values). `stereo_encode_rds` checks its `snr_db_guard` against the float64 formula with exact
//! sines (D34).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::manual_midpoint
)]

use sdrk::Sdrk;
use sdrk_core::kernels::rds::{
    RDS_TABLE_LEN, RdsShape, RdsShapeState, rds_pulse_peak, rds_pulse_table, rds_pulse_table_f64,
};
use sdrk_core::kernels::stereo::{StereoEncodeRds, StereoEncodeRdsState};
use sdrk_core::{C32, Fallback, Rate, Reference};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// Chunk plans per kernel and case.
const PLANS: u64 = 20;

/// The chunk bounds of a run: the whole signal at once, or the sizes of a plan.
fn chunks(len: usize, plan: Option<ChunkPlan>) -> Vec<(usize, usize)> {
    let mut sizes = plan.map(|p| p.sizes());
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < len {
        let n = sizes
            .as_mut()
            .map_or(len - pos, |s| s.next().unwrap())
            .min(len - pos);
        out.push((pos, pos + n));
        pos += n;
    }
    out
}

/// The plans of a case: the whole signal, then 20 plans with chunks of 1 to 700 samples.
fn plans(case: &GoldenCase) -> Vec<Option<ChunkPlan>> {
    let mut out = vec![None];
    out.extend((0..PLANS).map(|k| Some(ChunkPlan::new(case.seed * 100 + k, 700))));
    out
}

fn rate_of(case: &GoldenCase) -> Rate {
    Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap()
}

fn f32_array<'a>(case: &'a GoldenCase, name: &str) -> &'a [f32] {
    let Array::F32(v) = case.array(name).unwrap() else {
        panic!("{}: f32 array {name} expected", case.case);
    };
    v
}

#[test]
fn the_cases_of_m42_are_present() {
    assert_eq!(
        golden::list("rds_shape").unwrap(),
        ["table_and_bits_f32_16384"]
    );
    assert_eq!(golden::list("stereo_encode_rds").unwrap(), ["c32_16384"]);
}

// --------------------------------------------------------------------------------------------
// rds_shape
// --------------------------------------------------------------------------------------------

fn table_of(case: &GoldenCase) -> &[f64] {
    let Array::F64(t) = case.array("table").unwrap() else {
        panic!("f64 table expected");
    };
    t
}

fn bits_of(case: &GoldenCase) -> &[i8] {
    let Array::I8(b) = case.array("bits").unwrap() else {
        panic!("i8 bits expected");
    };
    b
}

#[test]
fn the_pulse_table_is_the_one_of_the_generator() {
    let case = golden::load("rds_shape", "table_and_bits_f32_16384").unwrap();
    let want = table_of(&case);
    assert_eq!(want.len(), RDS_TABLE_LEN);
    assert_eq!(case.param_u64("table_len").unwrap(), RDS_TABLE_LEN as u64);
    let mut t64 = vec![0.0f64; RDS_TABLE_LEN];
    rds_pulse_table_f64(&mut t64).unwrap();
    let worst = want
        .iter()
        .zip(&t64)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(worst <= 1e-12, "largest table difference {worst:e}");
    let peak = case.param_f64("grid_peak").unwrap();
    assert!((rds_pulse_peak() - peak).abs() <= 1e-12 * peak);
    let mut t32 = vec![0.0f32; RDS_TABLE_LEN];
    rds_pulse_table(&mut t32).unwrap();
    let golden32: Vec<f32> = want.iter().map(|&v| v as f32).collect();
    assert_eq!(bit_exact(&golden32, &t32), None, "f32 tables differ");
}

/// Runs `kernel` whole and in the 20 plans on `table`; every run must match `expected`, fetch
/// every stored bit once and end in the same state.
fn shape_check<K: RdsShape>(kernel: &K, case: &GoldenCase, table: &[f32]) -> Vec<f32> {
    let bits = bits_of(case);
    let want = case.array("expected").unwrap();
    let n = f32_array(case, "expected").len();
    let mut first: Option<(Vec<f32>, RdsShapeState<&[f32]>)> = None;
    for plan in plans(case) {
        let mut state = RdsShapeState::new(table, rate_of(case)).unwrap();
        let mut out = vec![0.0f32; n];
        let mut k = 0usize;
        let mut next = || {
            let b = bits
                .get(k)
                .copied()
                .expect("the kernel asked for too many bits");
            k += 1;
            b
        };
        for (a, b) in chunks(n, plan) {
            kernel
                .rds_shape(&mut state, &mut next, &mut out[a..b])
                .unwrap();
        }
        assert_eq!(k, bits.len(), "{}: bits fetched", case.case);
        assert_eq!(state.next_bit(), bits.len() as u64);
        assert_eq!(state.position(), n as u64);
        golden::assert_matches(case, want, &out).unwrap();
        match &first {
            None => first = Some((out, state)),
            Some((whole, end)) => {
                assert_eq!(
                    bit_exact(whole, &out),
                    None,
                    "block size changed the output"
                );
                assert_eq!(*end, state, "block size changed the state");
            }
        }
    }
    first.unwrap().0
}

#[test]
fn rds_shape_golden() {
    let case = golden::load("rds_shape", "table_and_bits_f32_16384").unwrap();
    let table: Vec<f32> = table_of(&case).iter().map(|&v| v as f32).collect();
    let facade = shape_check(&Sdrk, &case, &table);
    let fallback = shape_check(&Fallback, &case, &table);
    let reference = shape_check(&Reference, &case, &table);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
    // The table of sdrk-core gives the same output.
    let mut own = vec![0.0f32; RDS_TABLE_LEN];
    rds_pulse_table(&mut own).unwrap();
    let ours = shape_check(&Sdrk, &case, &own);
    assert_eq!(bit_exact(&facade, &ours), None);
    // The distance from the exact baseband is recorded in the case (D771).
    let err = case.param_f64("closed_form_max_abs_err").unwrap();
    assert!(err > 1e-5 && err < 1e-4, "{err:e}");
}

// --------------------------------------------------------------------------------------------
// stereo_encode_rds
// --------------------------------------------------------------------------------------------

fn encoder_state(case: &GoldenCase) -> StereoEncodeRdsState {
    let mut state = StereoEncodeRdsState::new(
        case.param_f64("pilot_hz").unwrap(),
        case.param_f64("main").unwrap() as f32,
        case.param_f64("pilot").unwrap() as f32,
        case.param_f64("rds").unwrap() as f32,
        rate_of(case),
    )
    .unwrap();
    assert_eq!(u64::from(state.increment()), case.param_u64("inc").unwrap());
    state.set_phase(u32::try_from(case.param_u64("phase0").unwrap()).unwrap());
    state
}

fn c32_input(case: &GoldenCase) -> &[C32] {
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!("{}: cf32 input expected", case.case);
    };
    input
}

fn encoder_check<K: StereoEncodeRds>(kernel: &K, case: &GoldenCase) -> Vec<f32> {
    let input = c32_input(case);
    let rds = f32_array(case, "rds");
    let want = case.array("expected").unwrap();
    let phase_end = case.param_u64("phase_end").unwrap();
    let mut first: Option<Vec<f32>> = None;
    for plan in plans(case) {
        let mut state = encoder_state(case);
        let mut out = vec![0.0f32; input.len()];
        for (a, b) in chunks(input.len(), plan) {
            kernel
                .stereo_encode_rds(&mut state, &input[a..b], &rds[a..b], &mut out[a..b])
                .unwrap();
        }
        golden::assert_matches(case, want, &out).unwrap();
        assert_eq!(u64::from(state.phase()), phase_end, "{}: phase", case.case);
        match &first {
            None => first = Some(out),
            Some(whole) => assert_eq!(
                bit_exact(whole, &out),
                None,
                "block size changed the output"
            ),
        }
    }
    first.unwrap()
}

#[test]
fn stereo_encode_rds_golden() {
    let case = golden::load("stereo_encode_rds", "c32_16384").unwrap();
    let facade = encoder_check(&Sdrk, &case);
    let fallback = encoder_check(&Fallback, &case);
    let reference = encoder_check(&Reference, &case);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
    // The accuracy guard (D34): the float64 formula with exact sines of the exact phase.
    let main = f64::from(case.param_f64("main").unwrap() as f32);
    let pilot = f64::from(case.param_f64("pilot").unwrap() as f32);
    let level = f64::from(case.param_f64("rds").unwrap() as f32);
    let inc = case.param_u64("inc").unwrap();
    let phase0 = case.param_u64("phase0").unwrap();
    let rds = f32_array(&case, "rds");
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, (x, &b)) in c32_input(&case).iter().zip(rds).enumerate() {
        let phase = (phase0 + i as u64 * inc) % (1u64 << 32);
        let phi = 2.0 * core::f64::consts::PI * (phase as f64 / 4_294_967_296.0);
        let (l, r) = (f64::from(x.re), f64::from(x.im));
        let (m, d) = ((l + r) * 0.5, (l - r) * 0.5);
        let want = main * m
            + main * d * libm::sin(2.0 * phi)
            + pilot * libm::sin(phi)
            + level * f64::from(b) * libm::sin(3.0 * phi);
        num += want * want;
        den += (want - f64::from(facade[i])).powi(2);
    }
    let snr = 10.0 * libm::log10(num / den);
    let guard = case.param_f64("snr_db_guard").unwrap();
    assert!(
        snr >= guard && snr < guard + 3.0,
        "SNR {snr:.2} dB, guard {guard}"
    );
    // The same pilot phase in `stereo_encode_rds` and `stereo_encode`: with the RDS level 0 the
    // output equals that of `stereo_encode` bit for bit.
    let mut st = encoder_state(&case);
    st.set_rds(0.0).unwrap();
    let mut out = vec![0.0f32; rds.len()];
    Sdrk.stereo_encode_rds(&mut st, c32_input(&case), rds, &mut out)
        .unwrap();
    let mut plain = *encoder_state(&case).encoder();
    let mut want = vec![0.0f32; rds.len()];
    sdrk_core::kernels::stereo::StereoEncode::stereo_encode(
        &Sdrk,
        &mut plain,
        c32_input(&case),
        &mut want,
    )
    .unwrap();
    assert_eq!(bit_exact(&want, &out), None);
}
