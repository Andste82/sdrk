//! Tests of the M22 kernels `biquad` and `biquad_real` that need the testkit: the suite cases
//! one by one, the Q15 limit cycle at silence, and block-size invariance of the reference.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::print_stderr
)]

use sdrk::Sdrk;
use sdrk_blocks::design::{BiquadResponse, butterworth_sos, to_q14};
use sdrk_core::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadState};
use sdrk_core::{CQ15, Progress, Q15, Reference};
#[cfg(feature = "std")]
use sdrk_testkit::{Arena, StdReporter, Tier, all_cases, run_suite};
use sdrk_testkit::{ChunkPlan, Pcg32, bit_exact, check_block_invariance};

#[cfg(feature = "std")]
#[test]
fn the_m22_suite_cases_pass_one_by_one() {
    let names = [
        "biquad_c32",
        "biquad_cq15",
        "biquad_real_f32",
        "biquad_real_q15",
        "extremes_m22",
    ];
    for name in names {
        let case = all_cases().iter().find(|c| c.name == name).unwrap();
        assert_eq!(case.tier == Tier::Full, name == "extremes_m22", "{name}");
        // The quick cases fit the arena of a Cortex-M4 target (64 KiB for the cases).
        let mut mem = vec![0u8; 64 * 1024];
        let mut arena = Arena::new(&mut mem);
        let summary = run_suite(
            std::slice::from_ref(case),
            Tier::Full,
            &mut arena,
            &mut StdReporter::new(),
        );
        assert_eq!((summary.passed, summary.failed), (1, 0), "{name}");
    }
}

fn butter4_q14(cutoff_hz: f64) -> Vec<[i16; 5]> {
    let mut sos = [[0.0f64; 5]; 2];
    butterworth_sos(BiquadResponse::Lowpass, 4, cutoff_hz, 1.0, &mut sos).unwrap();
    let mut q = [[0i16; 5]; 2];
    to_q14(&sos, &mut q).unwrap();
    q.to_vec()
}

fn q(v: f64) -> Q15 {
    Q15((v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
}

/// The excitations of the limit-cycle measurement, each `len` samples long.
fn excitations(len: usize) -> Vec<(&'static str, Vec<Q15>)> {
    let mut rng = Pcg32::new(2200, 1);
    let noise = (0..len).map(|_| q(rng.normal() * 0.3)).collect();
    let loud = (0..len).map(|_| q(rng.normal() * 0.9)).collect();
    let step = vec![q(0.9); len];
    let square = (0..len)
        .map(|i| if (i / 7) % 2 == 0 { q(0.95) } else { q(-0.95) })
        .collect();
    let mut impulse = vec![Q15(0); len];
    impulse[0] = Q15(32767);
    let tiny = (0..len).map(|i| Q15([3, -2, 1, -4][i % 4])).collect();
    vec![
        ("noise 0.3", noise),
        ("noise 0.9", loud),
        ("dc step", step),
        ("square", square),
        ("impulse", impulse),
        ("tiny", tiny),
    ]
}

/// Test 6: after the input becomes silent, the `butter4_lp` filter settles to a limit cycle or to
/// zero; the largest amplitude is at most 16 LSB. The measurement is printed for the record
/// in `docs/kernels.md`.
#[test]
fn butter4_lp_q15_limit_cycle_at_silence_is_at_most_16_lsb() {
    const ACTIVE: usize = 3000;
    const SILENT: usize = 20_000;
    // The cutoff of the golden case (0.1 fs), one half of it, and 0.02 fs (fs / 50, still above
    // the limit of fs / 500), which is only measured.
    let mut worst = 0i32;
    for cutoff in [0.1, 0.05, 0.02] {
        let coefs = butter4_q14(cutoff);
        for (name, active) in excitations(ACTIVE) {
            let mut x = active;
            x.resize(ACTIVE + SILENT, Q15(0));
            let mut y = vec![Q15(0); x.len()];
            let mut st = BiquadRealState::<Q15, _, _>::new(coefs.clone(), vec![Q15(0); 8]).unwrap();
            Sdrk.biquad_real(&mut st, &x, &mut y).unwrap();
            // The Reference gives the same bits.
            let mut y_ref = vec![Q15(0); x.len()];
            let mut st = BiquadRealState::<Q15, _, _>::new(coefs.clone(), vec![Q15(0); 8]).unwrap();
            Reference.biquad_real(&mut st, &x, &mut y_ref).unwrap();
            assert_eq!(bit_exact(&y, &y_ref), None);
            // The tail, long after the transient, is the limit cycle.
            let tail = y.iter().skip(ACTIVE + SILENT / 2);
            let amplitude = tail.map(|v| i32::from(v.0).abs()).max().unwrap();
            eprintln!("limit cycle butter4_lp cutoff {cutoff} fs, {name}: {amplitude} LSB");
            // Only the filters above fs / 20 are held to the limit; the lower cutoff is
            // measured for the record (docs/kernels.md).
            if cutoff >= 0.05 {
                worst = worst.max(amplitude);
            }
        }
    }
    eprintln!("largest limit cycle of butter4_lp (Q15): {worst} LSB");
    assert!(worst <= 16, "limit cycle of {worst} LSB");
}

/// The same measurement for the complex kernel (both components run the same recursion).
#[test]
fn butter4_lp_cq15_limit_cycle_at_silence_is_at_most_16_lsb() {
    let coefs = butter4_q14(0.1);
    let mut rng = Pcg32::new(2201, 1);
    let mut x: Vec<CQ15> = (0..3000)
        .map(|_| CQ15 {
            re: q(rng.normal() * 0.3),
            im: q(rng.normal() * 0.3),
        })
        .collect();
    x.resize(23_000, CQ15::default());
    let mut y = vec![CQ15::default(); x.len()];
    let mut st = BiquadState::<CQ15, _, _>::new(coefs, vec![CQ15::default(); 8]).unwrap();
    Sdrk.biquad(&mut st, &x, &mut y).unwrap();
    let amplitude = y
        .iter()
        .skip(13_000)
        .map(|v| i32::from(v.re.0).abs().max(i32::from(v.im.0).abs()))
        .max()
        .unwrap();
    eprintln!("limit cycle butter4_lp (CQ15): {amplitude} LSB");
    assert!(amplitude <= 16);
}

/// The reference keeps exact states, so it is block-size invariant too (20 plans, f32 and Q15).
#[test]
fn the_reference_is_block_size_invariant() {
    let mut rng = Pcg32::new(2202, 1);
    let x: Vec<f32> = (0..3000).map(|_| (rng.normal() * 0.3) as f32).collect();
    let xq: Vec<Q15> = x.iter().map(|v| q(f64::from(*v))).collect();
    let coefs_q = butter4_q14(0.1);
    let coefs_f: Vec<[f32; 5]> = coefs_q
        .iter()
        .map(|c| c.map(|v| f32::from(v) / 16384.0))
        .collect();
    let same_rate = |n: usize| Progress {
        consumed: n,
        produced: n,
    };
    for seed in 0..20u64 {
        let plan = || ChunkPlan::new(2300 + seed, 1 + (seed as usize * 53) % 500);
        let (mut whole, mut chunked) = (vec![0.0f32; x.len()], vec![0.0f32; x.len()]);
        check_block_invariance(
            &x,
            &mut whole,
            &mut chunked,
            plan(),
            || BiquadRealState::<f32, _, _>::new(coefs_f.clone(), vec![0.0f32; 8]).unwrap(),
            |st, i, o| {
                Reference.biquad_real(st, i, &mut o[..i.len()])?;
                Ok(same_rate(i.len()))
            },
        )
        .unwrap();
        let (mut whole, mut chunked) = (vec![Q15(0); x.len()], vec![Q15(0); x.len()]);
        check_block_invariance(
            &xq,
            &mut whole,
            &mut chunked,
            plan(),
            || BiquadRealState::<Q15, _, _>::new(coefs_q.clone(), vec![Q15(0); 8]).unwrap(),
            |st, i, o| {
                Reference.biquad_real(st, i, &mut o[..i.len()])?;
                Ok(same_rate(i.len()))
            },
        )
        .unwrap();
    }
}
