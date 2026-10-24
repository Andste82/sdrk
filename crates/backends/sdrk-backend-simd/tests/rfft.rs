//! `rfft` and `irfft` against the fallback: within the 110 dB of the contract at every size, and
//! the same length errors.
#![allow(clippy::unwrap_used)]

mod common;

use common::{Rng, snr_db};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::fft::{RealFft, RealFftPlan};
use sdrk_core::{C32, Error, Fallback};

fn plan(n: usize) -> RealFftPlan<Vec<C32>, Vec<C32>> {
    RealFftPlan::new(n, vec![C32::default(); n / 4], vec![C32::default(); n / 2]).unwrap()
}

#[test]
fn rfft_and_irfft_meet_the_contract_at_every_size() {
    let mut rng = Rng::new(80);
    for log in 5..=16 {
        let n = 1usize << log;
        let p = plan(n);
        let x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
        let mut scratch = vec![C32::default(); n / 2];
        let (mut want, mut got) = (
            vec![C32::default(); n / 2 + 1],
            vec![C32::default(); n / 2 + 1],
        );
        let ws = Fallback.rfft(&p, &x, &mut want, &mut scratch).unwrap();
        let gs = Simd.rfft(&p, &x, &mut got, &mut scratch).unwrap();
        assert_eq!(ws, gs);
        let snr = snr_db(&want, &got);
        assert!(snr > 110.0, "rfft n={n}: {snr:.1} dB");
        let (mut back_want, mut back_got) = (vec![0f32; n], vec![0f32; n]);
        let ws = Fallback
            .irfft(&p, &want, &mut back_want, &mut scratch)
            .unwrap();
        let gs = Simd.irfft(&p, &want, &mut back_got, &mut scratch).unwrap();
        assert_eq!(ws, gs);
        let pairs =
            |v: &[f32]| -> Vec<C32> { v.chunks_exact(2).map(|p| C32::new(p[0], p[1])).collect() };
        let snr = snr_db(&pairs(&back_want), &pairs(&back_got));
        assert!(snr > 110.0, "irfft n={n}: {snr:.1} dB");
    }
}

#[test]
fn rfft_length_errors_match_the_fallback() {
    let p = plan(64);
    let x = vec![0f32; 64];
    let mut scratch = vec![C32::default(); 31];
    let mut out = vec![C32::default(); 33];
    let want = Fallback.rfft(&p, &x, &mut out, &mut scratch);
    let got = Simd.rfft(&p, &x, &mut out, &mut scratch);
    assert!(matches!(got, Err(Error::LengthMismatch { .. })));
    assert_eq!(want, got);
}
