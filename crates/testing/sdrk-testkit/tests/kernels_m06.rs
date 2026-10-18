//! Tests of the receive kernels of M06 that need the testkit: block-size invariance of
//! every stateful kernel over 20 chunk plans and random comparisons with the reference.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_lossless,
    clippy::too_many_lines
)]

use proptest::prelude::*;
use sdrk::Sdrk;
use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::demod::{
    DcBlock, DcBlockState, Deemphasis, DeemphasisState, QuadDemod, QuadDemodState,
};
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Rate, Reference, Sample};
use sdrk_testkit::{ChunkPlan, Pcg32, bit_exact, check_block_invariance};

const N: usize = 2500;

fn same_rate<I: Copy, O>(
    input: &[I],
    output: &mut [O],
    kernel: impl FnOnce(&[I], &mut [O]) -> sdrk_core::Result<()>,
) -> sdrk_core::Result<Progress> {
    let n = input.len().min(output.len());
    kernel(&input[..n], &mut output[..n])?;
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

fn c32_input(rng: &mut Pcg32) -> Vec<C32> {
    (0..N)
        .map(|_| C32::new(rng.normal() as f32 * 0.3, rng.normal() as f32 * 0.3))
        .collect()
}

fn cq_input(rng: &mut Pcg32) -> Vec<CQ15> {
    (0..N)
        .map(|_| CQ15 {
            re: Q15((rng.next_u32() >> 16) as i16),
            im: Q15((rng.next_u32() >> 16) as i16),
        })
        .collect()
}

fn plan(seed: u64, k: usize) -> ChunkPlan {
    ChunkPlan::new(seed, 1 + (seed as usize * k) % 400)
}

#[test]
fn fir_decimate_is_block_size_invariant() {
    let mut rng = Pcg32::new(70, 1);
    let (c32, cq) = (c32_input(&mut rng), cq_input(&mut rng));
    for seed in 0..20u64 {
        let taps_n = 1 + (seed as usize * 7) % 80;
        let factor = 1 + (seed as usize * 3) % 12;
        let taps32: Vec<f32> = (0..taps_n).map(|_| rng.normal() as f32 * 0.1).collect();
        let tapsq: Vec<Q15> = (0..taps_n)
            .map(|_| Q15((rng.next_u32() >> 20) as i16 - 2048))
            .collect();
        let (mut a, mut b) = (vec![C32::ZERO; N], vec![C32::ZERO; N]);
        let r = check_block_invariance(
            &c32,
            &mut a,
            &mut b,
            plan(seed, 41),
            || FirDecimState::<C32, _, _>::new(taps32.clone(), vec![C32::ZERO; taps_n - 1], factor),
            |st, i, o| {
                let st = st.as_mut().map_err(|e| e.clone())?;
                FirDecimate::<C32>::fir_decimate(&Sdrk, st, i, o)
            },
        )
        .unwrap();
        assert_eq!(r.outputs, N.div_ceil(factor), "factor {factor}");
        let (mut a, mut b) = (vec![CQ15::ZERO; N], vec![CQ15::ZERO; N]);
        check_block_invariance(
            &cq,
            &mut a,
            &mut b,
            plan(seed + 100, 29),
            || {
                FirDecimState::<CQ15, _, _>::new(
                    tapsq.clone(),
                    vec![CQ15::ZERO; taps_n - 1],
                    factor,
                )
            },
            |st, i, o| {
                let st = st.as_mut().map_err(|e| e.clone())?;
                FirDecimate::<CQ15>::fir_decimate(&Sdrk, st, i, o)
            },
        )
        .unwrap();
    }
}

#[test]
fn resample_is_block_size_invariant() {
    let mut rng = Pcg32::new(71, 1);
    let (c32, cq) = (c32_input(&mut rng), cq_input(&mut rng));
    let ratios = [
        (1usize, 5usize),
        (3, 2),
        (2, 3),
        (4, 1),
        (1, 1),
        (7, 5),
        (147, 160),
        (5, 4),
    ];
    for seed in 0..20u64 {
        let (up, down) = ratios[seed as usize % ratios.len()];
        let taps_n = 1 + (seed as usize * 11) % 120;
        let hist = taps_n.div_ceil(up);
        let taps32: Vec<f32> = (0..taps_n).map(|_| rng.normal() as f32 * 0.1).collect();
        let tapsq: Vec<Q15> = (0..taps_n)
            .map(|_| Q15((rng.next_u32() >> 20) as i16 - 2048))
            .collect();
        let room = N * up / down + up + 2;
        let (mut a, mut b) = (vec![C32::ZERO; room], vec![C32::ZERO; room]);
        let r = check_block_invariance(
            &c32,
            &mut a,
            &mut b,
            plan(seed, 37),
            || ResampleState::<C32, _, _>::new(taps32.clone(), vec![C32::ZERO; hist], up, down),
            |st, i, o| {
                let st = st.as_mut().map_err(|e| e.clone())?;
                Resample::<C32>::resample(&Sdrk, st, i, o)
            },
        )
        .unwrap();
        assert_eq!(r.outputs, (N * up - 1) / down + 1, "L={up} M={down}");
        let (mut a, mut b) = (vec![CQ15::ZERO; room], vec![CQ15::ZERO; room]);
        check_block_invariance(
            &cq,
            &mut a,
            &mut b,
            plan(seed + 100, 23),
            || ResampleState::<CQ15, _, _>::new(tapsq.clone(), vec![CQ15::ZERO; hist], up, down),
            |st, i, o| {
                let st = st.as_mut().map_err(|e| e.clone())?;
                Resample::<CQ15>::resample(&Sdrk, st, i, o)
            },
        )
        .unwrap();
        // Real variants.
        let xr: Vec<f32> = c32.iter().map(|s| s.re).collect();
        let xq: Vec<Q15> = cq.iter().map(|s| s.re).collect();
        let (mut a, mut b) = (vec![0f32; room], vec![0f32; room]);
        check_block_invariance(
            &xr,
            &mut a,
            &mut b,
            plan(seed + 200, 19),
            || ResampleRealState::<f32, _, _>::new(taps32.clone(), vec![0f32; hist], up, down),
            |st, i, o| {
                let st = st.as_mut().map_err(|e| e.clone())?;
                ResampleReal::<f32>::resample_real(&Sdrk, st, i, o)
            },
        )
        .unwrap();
        let (mut a, mut b) = (vec![Q15(0); room], vec![Q15(0); room]);
        check_block_invariance(
            &xq,
            &mut a,
            &mut b,
            plan(seed + 300, 17),
            || ResampleRealState::<Q15, _, _>::new(tapsq.clone(), vec![Q15(0); hist], up, down),
            |st, i, o| {
                let st = st.as_mut().map_err(|e| e.clone())?;
                ResampleReal::<Q15>::resample_real(&Sdrk, st, i, o)
            },
        )
        .unwrap();
    }
}

#[test]
fn quad_demod_deemphasis_and_dc_block_are_block_size_invariant() {
    let mut rng = Pcg32::new(72, 1);
    let (c32, cq) = (c32_input(&mut rng), cq_input(&mut rng));
    let audio: Vec<f32> = c32.iter().map(|s| s.re).collect();
    let audio_q: Vec<Q15> = cq.iter().map(|s| s.re).collect();
    let rate = Rate::hz(48_000).unwrap();
    for seed in 0..20u64 {
        let gain = 0.5 + seed as f32 * 0.37;
        let gain_q = Q15(32767 - (seed as i16) * 1500);
        let tau = 20e-6 + seed as f64 * 10e-6;
        let r = 0.9 + seed as f32 * 0.004;
        let r_q = Q15(29000 + (seed as i16) * 150);

        let (mut a, mut b) = (vec![0f32; N], vec![0f32; N]);
        check_block_invariance(
            &c32,
            &mut a,
            &mut b,
            plan(seed, 41),
            QuadDemodState::<C32>::new,
            |st, i, o| {
                same_rate(i, o, |i, o| {
                    QuadDemod::<C32>::quad_demod(&Sdrk, st, i, o, gain)
                })
            },
        )
        .unwrap();
        let (mut a, mut b) = (vec![Q15(0); N], vec![Q15(0); N]);
        check_block_invariance(
            &cq,
            &mut a,
            &mut b,
            plan(seed + 100, 43),
            QuadDemodState::<CQ15>::new,
            |st, i, o| {
                same_rate(i, o, |i, o| {
                    QuadDemod::<CQ15>::quad_demod(&Sdrk, st, i, o, gain_q)
                })
            },
        )
        .unwrap();

        let (mut a, mut b) = (vec![0f32; N], vec![0f32; N]);
        check_block_invariance(
            &audio,
            &mut a,
            &mut b,
            plan(seed + 200, 47),
            || DeemphasisState::<f32>::new(tau, rate).unwrap(),
            |st, i, o| same_rate(i, o, |i, o| Deemphasis::<f32>::deemphasis(&Sdrk, st, i, o)),
        )
        .unwrap();
        let (mut a, mut b) = (vec![Q15(0); N], vec![Q15(0); N]);
        check_block_invariance(
            &audio_q,
            &mut a,
            &mut b,
            plan(seed + 300, 53),
            || DeemphasisState::<Q15>::new(tau, rate).unwrap(),
            |st, i, o| same_rate(i, o, |i, o| Deemphasis::<Q15>::deemphasis(&Sdrk, st, i, o)),
        )
        .unwrap();

        let (mut a, mut b) = (vec![C32::ZERO; N], vec![C32::ZERO; N]);
        check_block_invariance(
            &c32,
            &mut a,
            &mut b,
            plan(seed + 400, 59),
            || DcBlockState::<C32>::new(r).unwrap(),
            |st, i, o| same_rate(i, o, |i, o| DcBlock::<C32>::dc_block(&Sdrk, st, i, o)),
        )
        .unwrap();
        let (mut a, mut b) = (vec![CQ15::ZERO; N], vec![CQ15::ZERO; N]);
        check_block_invariance(
            &cq,
            &mut a,
            &mut b,
            plan(seed + 500, 61),
            || DcBlockState::<CQ15>::new(r_q).unwrap(),
            |st, i, o| same_rate(i, o, |i, o| DcBlock::<CQ15>::dc_block(&Sdrk, st, i, o)),
        )
        .unwrap();
    }
}

proptest! {
    /// The Q15 resampler is bit-exact to the reference for arbitrary taps, ratios and chunking.
    #[test]
    fn q15_resampler_matches_the_reference(
        up in 1usize..12,
        down in 1usize..12,
        taps_n in 1usize..60,
        n in 1usize..200,
        room in 1usize..40,
        seed in any::<u64>(),
    ) {
        let mut rng = Pcg32::new(seed, 5);
        let taps: Vec<Q15> = (0..taps_n).map(|_| Q15((rng.next_u32() >> 16) as i16)).collect();
        let x: Vec<CQ15> = (0..n)
            .map(|_| CQ15 { re: Q15((rng.next_u32() >> 16) as i16), im: Q15((rng.next_u32() >> 16) as i16) })
            .collect();
        let hist = taps_n.div_ceil(up);
        let mut a = ResampleState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::ZERO; hist], up, down).unwrap();
        let mut b = a.clone();
        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        // Both sides run with a small output room and resubmit the rest.
        for (state_is_fallback, out) in [(true, &mut out_a), (false, &mut out_b)] {
            let mut pos = 0;
            let mut buf = vec![CQ15::ZERO; room];
            loop {
                let p = if state_is_fallback {
                    Fallback.resample(&mut a, &x[pos..], &mut buf).unwrap()
                } else {
                    Reference.resample(&mut b, &x[pos..], &mut buf).unwrap()
                };
                out.extend_from_slice(&buf[..p.produced]);
                pos += p.consumed;
                if pos == x.len() && p.produced < room {
                    break;
                }
                prop_assert!(p.consumed + p.produced > 0);
            }
        }
        prop_assert_eq!(bit_exact(&out_a, &out_b), None);
        prop_assert_eq!(out_a.len(), (n * up - 1) / down + 1);
        prop_assert_eq!(a.history(), b.history());
    }

    /// The Q15 decimator is bit-exact to the reference for arbitrary factors.
    #[test]
    fn q15_decimator_matches_the_reference(
        factor in 1usize..20,
        taps_n in 1usize..50,
        n in 1usize..300,
        seed in any::<u64>(),
    ) {
        let mut rng = Pcg32::new(seed, 6);
        let taps: Vec<Q15> = (0..taps_n).map(|_| Q15((rng.next_u32() >> 16) as i16)).collect();
        let x: Vec<CQ15> = (0..n)
            .map(|_| CQ15 { re: Q15((rng.next_u32() >> 16) as i16), im: Q15((rng.next_u32() >> 16) as i16) })
            .collect();
        let mut a = FirDecimState::<CQ15, _, _>::new(taps, vec![CQ15::ZERO; taps_n - 1], factor).unwrap();
        let mut b = a.clone();
        let (mut ya, mut yb) = (vec![CQ15::ZERO; n], vec![CQ15::ZERO; n]);
        let pa = Fallback.fir_decimate(&mut a, &x, &mut ya).unwrap();
        let pb = Reference.fir_decimate(&mut b, &x, &mut yb).unwrap();
        prop_assert_eq!(pa, pb);
        prop_assert_eq!(pa.produced, n.div_ceil(factor));
        prop_assert_eq!(&ya[..pa.produced], &yb[..pb.produced]);
    }
}
