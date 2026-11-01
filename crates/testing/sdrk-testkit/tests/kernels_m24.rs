//! Tests of the M24 stereo kernels that need the testkit: `Fallback` against `Reference` for
//! several parameter sets and signals, and block-size invariance over 20 chunk plans for both
//! kernels in both types.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::large_stack_frames
)]

use sdrk::Sdrk;
use sdrk_core::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Rate, RealSample, Reference, Sample};
use sdrk_testkit::{ChunkPlan, Pcg32, bit_exact, check_block_invariance};

const N: usize = 6000;

fn signals<T: RealSample>(rng: &mut Pcg32, fs: f64, pilot_hz: f64, amp: f64) -> Vec<(T, T)> {
    (0..N)
        .map(|n| {
            let turns = pilot_hz * n as f64 / fs;
            let pil = (2.0 * core::f64::consts::PI * (turns - turns.floor())).sin();
            let noise = |r: &mut Pcg32| (r.next_f64() - 0.5) * 0.02;
            let l = (2.0 * core::f64::consts::PI * (1000.0 * n as f64 / fs).fract()).sin();
            let mpx = 0.4 * l + amp * pil + noise(rng);
            let pilot = amp * pil + noise(rng);
            (T::from_f64_round(mpx), T::from_f64_round(pilot))
        })
        .collect()
}

fn run<T: RealSample, K: StereoPll<T>>(
    k: &K,
    st: &mut StereoPllState<T>,
    i: &[(T, T)],
    o: &mut [T::Complex],
) -> sdrk_core::Result<Progress> {
    let n = i.len().min(o.len());
    let (mpx, pilot): (Vec<T>, Vec<T>) = i[..n].iter().copied().unzip();
    k.stereo_pll(st, &mpx, &pilot, &mut o[..n])?;
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

fn pll_cases<T: RealSample>(amp: f64)
where
    Sdrk: StereoPll<T>,
    Fallback: StereoPll<T>,
    Reference: StereoPll<T>,
{
    let mut rng = Pcg32::new(2401, 1);
    // (rate, pilot offset in Hz, natural frequency, level)
    for (fs, pilot_hz, natural, level) in [
        (192_000u64, 19_000.5, 50.0, amp),
        (480_000, 19_002.0, 20.0, amp),
        (96_000, 18_990.0, 100.0, amp),
        (80_000, 19_000.0, 1.0, amp),
        (480_000, 19_000.0, 190.0, amp),
    ] {
        let rate = Rate::hz(fs).unwrap();
        let make = || StereoPllState::<T>::new(19_000.0, natural, level, rate).unwrap();
        let input = signals::<T>(&mut rng, fs as f64, pilot_hz, amp);
        let (mpx, pilot): (Vec<T>, Vec<T>) = input.iter().copied().unzip();
        let zero = <T::Complex as Sample>::ZERO;
        let (mut a, mut b, mut c) = (make(), make(), make());
        let (mut oa, mut ob, mut oc) = (vec![zero; N], vec![zero; N], vec![zero; N]);
        Fallback.stereo_pll(&mut a, &mpx, &pilot, &mut oa).unwrap();
        Reference.stereo_pll(&mut b, &mpx, &pilot, &mut ob).unwrap();
        Sdrk.stereo_pll(&mut c, &mpx, &pilot, &mut oc).unwrap();
        assert_eq!(bit_exact(&oa, &ob), None, "fs {fs} natural {natural}");
        assert_eq!(bit_exact(&oa, &oc), None);
        assert_eq!(a, b);
        assert_eq!(a, c);
        // Invariance over 20 plans, with the state in the loop.
        for seed in 0..20u64 {
            let plan = ChunkPlan::new(seed, 1 + (seed as usize * 37) % 500);
            let (mut w, mut ch) = (vec![zero; N], vec![zero; N]);
            let r = check_block_invariance(&input, &mut w, &mut ch, plan, make, |st, i, o| {
                run(&Sdrk, st, i, o)
            })
            .unwrap();
            assert_eq!(r.outputs, N);
        }
    }
}

#[test]
fn stereo_pll_f32_matches_the_reference_and_ignores_the_block_size() {
    pll_cases::<f32>(0.1);
}

#[test]
fn stereo_pll_q15_matches_the_reference_and_ignores_the_block_size() {
    pll_cases::<Q15>(0.0625);
}

fn matrix_cases<T: sdrk_core::ComplexSample>(fill: impl Fn(&mut Pcg32) -> T)
where
    Sdrk: StereoMatrix<T>,
    Fallback: StereoMatrix<T>,
    Reference: StereoMatrix<T>,
{
    let mut rng = Pcg32::new(2402, 1);
    let input: Vec<T> = (0..N).map(|_| fill(&mut rng)).collect();
    let (mut a, mut b, mut c) = (vec![T::ZERO; N], vec![T::ZERO; N], vec![T::ZERO; N]);
    Fallback.stereo_matrix(&input, &mut a).unwrap();
    Reference.stereo_matrix(&input, &mut b).unwrap();
    Sdrk.stereo_matrix(&input, &mut c).unwrap();
    assert_eq!(bit_exact(&a, &b), None);
    assert_eq!(bit_exact(&a, &c), None);
    for seed in 0..20u64 {
        let plan = ChunkPlan::new(seed, 1 + (seed as usize * 29) % 400);
        let (mut w, mut ch) = (vec![T::ZERO; N], vec![T::ZERO; N]);
        check_block_invariance(
            &input,
            &mut w,
            &mut ch,
            plan,
            || (),
            |(), i, o| {
                let n = i.len().min(o.len());
                Sdrk.stereo_matrix(&i[..n], &mut o[..n])?;
                Ok(Progress {
                    consumed: n,
                    produced: n,
                })
            },
        )
        .unwrap();
    }
}

#[test]
fn stereo_matrix_matches_the_reference_and_ignores_the_block_size() {
    matrix_cases::<C32>(|r| {
        C32::new(
            ((r.next_f64() * 2.0 - 1.0) * 0.6) as f32,
            ((r.next_f64() * 2.0 - 1.0) * 0.6) as f32,
        )
    });
    matrix_cases::<CQ15>(|r| CQ15 {
        re: Q15(r.next_u32() as i16),
        im: Q15((r.next_u32() >> 7) as i16),
    });
}
