//! Tests of the M25 kernel `preemph_limit` that need the testkit: `Fallback`, `Reference` and the
//! facade `Sdrk` agree bit for bit for several parameter sets and signals (up to 20 times full
//! scale), and the output does not depend on the block size over 20 chunk plans.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::large_stack_frames
)]

use sdrk::Sdrk;
use sdrk_blocks::design::preemphasis_coefs;
use sdrk_core::kernels::preemph_limit::{PreemphLimit, PreemphLimitState};
use sdrk_core::{Fallback, Progress, Q15, Rate, RealSample, Reference};
use sdrk_testkit::{ChunkPlan, Pcg32, bit_exact, check_block_invariance};

const N: usize = 6000;

fn signal<T: RealSample>(rng: &mut Pcg32, scale: f64) -> Vec<T> {
    (0..N)
        .map(|n| {
            // A loud burst in the middle, quiet noise around it.
            let amp = if (2000..3500).contains(&n) {
                scale
            } else {
                0.05
            };
            let tone = (0.3 * n as f64).sin();
            T::from_f64_round(amp * (0.7 * tone + (rng.next_f64() * 2.0 - 1.0) * 0.3))
        })
        .collect()
}

fn run<T: RealSample, K: PreemphLimit<T>>(
    k: &K,
    s: &mut PreemphLimitState<T>,
    i: &[T],
    o: &mut [T],
) -> sdrk_core::Result<Progress> {
    let n = i.len().min(o.len());
    k.preemph_limit(s, &i[..n], &mut o[..n])?;
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

fn cases<T: RealSample>()
where
    Sdrk: PreemphLimit<T>,
    Fallback: PreemphLimit<T>,
    Reference: PreemphLimit<T>,
{
    let mut rng = Pcg32::new(2501, 1);
    // (rate, tau in us, limit, hold in s, release in s, burst scale)
    for (fs, tau_us, limit, hold, release, scale) in [
        (48_000u64, 50.0, 1.0, 0.02, 0.2, 0.2),
        (48_000, 50.0, 0.5, 0.02, 0.2, 1.5),
        (48_000, 530.0, 1.0, 0.02, 0.2, 0.2),
        (240_000, 50.0, 1.0, 0.0, 0.001, 20.0),
        (240_000, 1000.0, 0.85, 0.005, 0.02, 20.0),
        (96_000, 0.0, 0.3, 0.001, 0.01, 20.0),
    ] {
        let coefs = if tau_us == 0.0 {
            [1.0, 0.0, 0.0]
        } else {
            preemphasis_coefs(tau_us * 1e-6, fs as f64).unwrap()
        };
        let rate = Rate::hz(fs).unwrap();
        let make = || PreemphLimitState::<T>::new(coefs, limit, hold, release, rate).unwrap();
        let input = signal::<T>(&mut rng, scale);
        let (mut a, mut b, mut c) = (make(), make(), make());
        let (mut oa, mut ob, mut oc) = (vec![T::ZERO; N], vec![T::ZERO; N], vec![T::ZERO; N]);
        Fallback.preemph_limit(&mut a, &input, &mut oa).unwrap();
        Reference.preemph_limit(&mut b, &input, &mut ob).unwrap();
        Sdrk.preemph_limit(&mut c, &input, &mut oc).unwrap();
        assert_eq!(bit_exact(&oa, &ob), None, "fs {fs} tau {tau_us}");
        assert_eq!(bit_exact(&oa, &oc), None);
        assert_eq!(a, b);
        assert_eq!(a, c);
        for seed in 0..20u64 {
            let plan = ChunkPlan::new(seed, 1 + (seed as usize * 37) % 500);
            let (mut w, mut ch) = (vec![T::ZERO; N], vec![T::ZERO; N]);
            let r = check_block_invariance(&input, &mut w, &mut ch, plan, make, |st, i, o| {
                run(&Sdrk, st, i, o)
            })
            .unwrap();
            assert_eq!(r.outputs, N);
        }
    }
}

#[test]
fn preemph_limit_f32_matches_the_reference_and_ignores_the_block_size() {
    cases::<f32>();
}

#[test]
fn preemph_limit_q15_matches_the_reference_and_ignores_the_block_size() {
    cases::<Q15>();
}
