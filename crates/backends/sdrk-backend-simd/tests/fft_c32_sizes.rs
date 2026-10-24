//! The `C32` FFT at every size from 16 to 65 536, against an f64 transform: on noise the SIMD
//! backend must stay within 1 dB of the accuracy of the fallback (and far inside the 110 dB
//! contract) for both directions and for plans over their own table, over a subsampled
//! 4096-point table and over the static 4096-point table of `sdrk-core` (a table stride of
//! `4096 / n`); inputs that hit special cases of the passes must stay above 125 dB.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

mod common;

use common::{Rng, c32_noise, db, powers_exact, snr_db_exact as snr_db};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::{C32, Fallback};

/// The sizes of the plans: every power of two from 16 to 65 536.
const SIZES: [usize; 13] = [
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];

/// The transform of `x` in f64 (iterative radix-2, twiddles from `sin`/`cos` per stage), with the
/// conventions of the kernel: unnormalised forward, `1/N` inverse.
fn fft_f64(x: &[C32], inverse: bool) -> Vec<(f64, f64)> {
    let n = x.len();
    let bits = n.trailing_zeros();
    let mut a: Vec<(f64, f64)> = vec![(0.0, 0.0); n];
    for (i, v) in x.iter().enumerate() {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        a[j] = (f64::from(v.re), f64::from(v.im));
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut h = 1;
    while h < n {
        for k in 0..h {
            let (s, c) = (sign * std::f64::consts::PI * k as f64 / h as f64).sin_cos();
            for base in (0..n).step_by(2 * h) {
                let (u, v) = (a[base + k], a[base + k + h]);
                let t = (v.0 * c - v.1 * s, v.0 * s + v.1 * c);
                a[base + k] = (u.0 + t.0, u.1 + t.1);
                a[base + k + h] = (u.0 - t.0, u.1 - t.1);
            }
        }
        h *= 2;
    }
    if inverse {
        let g = 1.0 / n as f64;
        for v in &mut a {
            *v = (v.0 * g, v.1 * g);
        }
    }
    a
}

/// The SNR of the SIMD backend and of the fallback over `inputs` with `plan` (the energies of all
/// inputs summed, so that short transforms are judged on as many samples as long ones).
fn snrs<Tw: AsRef<[C32]>>(
    plan: &FftPlan<C32, Tw>,
    inputs: &[Vec<C32>],
    dir: Direction,
) -> (f64, f64) {
    let (mut ours_e, mut fallback_e) = ((0.0, 0.0), (0.0, 0.0));
    for x in inputs {
        let want = fft_f64(x, dir == Direction::Inverse);
        let mut ours = x.clone();
        let scale = Simd.fft(plan, &mut ours, dir, Scaling::Default).unwrap();
        let mut fallback = x.clone();
        let fallback_scale = Fallback
            .fft(plan, &mut fallback, dir, Scaling::Default)
            .unwrap();
        assert_eq!(scale, fallback_scale);
        let (o, f) = (powers_exact(&want, &ours), powers_exact(&want, &fallback));
        ours_e = (ours_e.0 + o.0, ours_e.1 + o.1);
        fallback_e = (fallback_e.0 + f.0, fallback_e.1 + f.1);
    }
    (db(ours_e.0, ours_e.1), db(fallback_e.0, fallback_e.1))
}

/// Random inputs of `n` points with `65536` samples in total.
fn noise_batches(rng: &mut Rng, n: usize, amp: f32) -> Vec<Vec<C32>> {
    (0..(65536 / n).max(1))
        .map(|_| c32_noise(rng, n, amp))
        .collect()
}

/// The backend is at most 1 dB less accurate than the fallback and above `floor`.
fn check(n: usize, what: &str, (ours, fallback): (f64, f64), floor: f64) {
    assert!(
        ours >= fallback - 1.0 && ours >= floor,
        "n={n} {what}: {ours:.2} dB, fallback {fallback:.2} dB"
    );
}

#[test]
fn fft_c32_every_size_within_1_db_of_the_fallback() {
    let mut rng = Rng::new(60);
    for n in SIZES {
        let plan = FftPlan::<C32, _>::new(n, vec![C32::default(); n / 2]).unwrap();
        for dir in [Direction::Forward, Direction::Inverse] {
            for amp in [1.0, 3.0e4, 1.0e-6] {
                let x = noise_batches(&mut rng, n, amp);
                check(
                    n,
                    &format!("{dir:?} amp {amp}"),
                    snrs(&plan, &x, dir),
                    125.0,
                );
            }
        }
    }
}

#[test]
fn fft_c32_every_size_of_a_subsampled_table() {
    // Plans over every `4096 / n`-th entry of the 4096-point table, copied into a table of their
    // own (`from_twiddles`, so the table stride is 1).
    let big = FftPlan::<C32, _>::new(4096, vec![C32::default(); 2048]).unwrap();
    let table = big.twiddles().to_vec();
    let mut rng = Rng::new(61);
    for n in SIZES.into_iter().filter(|&n| n <= 4096) {
        let plan = FftPlan::<C32, _>::from_twiddles(
            n,
            table.iter().step_by(4096 / n).copied().collect::<Vec<_>>(),
        )
        .unwrap();
        for dir in [Direction::Forward, Direction::Inverse] {
            let x = noise_batches(&mut rng, n, 1.0);
            check(
                n,
                &format!("{dir:?} subsampled"),
                snrs(&plan, &x, dir),
                125.0,
            );
        }
    }
}

#[test]
fn fft_c32_every_size_of_the_static_4096_plan() {
    // `from_static_4096` reads every `4096 / n`-th entry of the static table in place: the passes
    // run with a table stride above 1 (without the feature `fft-tables` every twiddle of the
    // passes comes from that table).
    let mut rng = Rng::new(63);
    for n in SIZES.into_iter().filter(|&n| n <= 4096) {
        let plan = FftPlan::<C32, &'static [C32]>::from_static_4096(n).unwrap();
        assert_eq!(plan.twiddle_stride(), 4096 / n);
        for dir in [Direction::Forward, Direction::Inverse] {
            let x = noise_batches(&mut rng, n, 1.0);
            check(n, &format!("{dir:?} static"), snrs(&plan, &x, dir), 125.0);
        }
    }
}

#[test]
fn fft_c32_every_size_special_inputs() {
    for n in SIZES {
        let plan = FftPlan::<C32, _>::new(n, vec![C32::default(); n / 2]).unwrap();
        let mut inputs = vec![
            // A unit impulse at the start, one at the end and one in the middle of a tile row.
            (0..n)
                .map(|i| C32::new(f32::from(u8::from(i == 0)), 0.0))
                .collect::<Vec<_>>(),
            (0..n)
                .map(|i| C32::new(0.0, f32::from(u8::from(i == n - 1))))
                .collect(),
            (0..n)
                .map(|i| C32::new(f32::from(u8::from(i == n / 2 + 5)), -1.0))
                .collect(),
            // DC and the Nyquist tone.
            vec![C32::new(1.0, -0.5); n],
            (0..n)
                .map(|i| C32::new(if i % 2 == 0 { 1.0 } else { -1.0 }, 0.25))
                .collect(),
        ];
        // A tone on every bin of a coarse grid, which puts the energy into single outputs.
        for bin in [1, 3, n / 4 + 1, n - 1] {
            inputs.push(
                (0..n)
                    .map(|i| {
                        let ph = 2.0 * std::f64::consts::PI * (bin * i % n) as f64 / n as f64;
                        C32::new(ph.cos() as f32, ph.sin() as f32)
                    })
                    .collect(),
            );
        }
        for x in &inputs {
            for dir in [Direction::Forward, Direction::Inverse] {
                let want = fft_f64(x, dir == Direction::Inverse);
                let mut got = x.clone();
                Simd.fft(&plan, &mut got, dir, Scaling::Default).unwrap();
                let snr = snr_db(&want, &got);
                assert!(snr > 125.0, "n={n} {dir:?} special: {snr:.2} dB");
            }
        }
        // Zeros stay zeros.
        let mut z = vec![C32::default(); n];
        Simd.fft(&plan, &mut z, Direction::Forward, Scaling::Default)
            .unwrap();
        assert!(z.iter().all(|v| v.re == 0.0 && v.im == 0.0), "n={n}");
    }
}

#[test]
fn fft_c32_round_trip_every_size() {
    let mut rng = Rng::new(62);
    for n in SIZES {
        let plan = FftPlan::<C32, _>::new(n, vec![C32::default(); n / 2]).unwrap();
        let x = c32_noise(&mut rng, n, 1.0);
        let mut y = x.clone();
        Simd.fft(&plan, &mut y, Direction::Forward, Scaling::Default)
            .unwrap();
        Simd.fft(&plan, &mut y, Direction::Inverse, Scaling::Default)
            .unwrap();
        let want: Vec<(f64, f64)> = x
            .iter()
            .map(|v| (f64::from(v.re), f64::from(v.im)))
            .collect();
        let snr = snr_db(&want, &y);
        assert!(snr > 120.0, "n={n}: round trip {snr:.2} dB");
    }
}

#[test]
fn fft_c32_misaligned_buffers_give_the_same_bits() {
    // The vector loads and stores need no alignment: the transform of a slice that starts one
    // sample (8 bytes) after an aligned one is bit for bit the same.
    let mut rng = Rng::new(63);
    for n in SIZES {
        let plan = FftPlan::<C32, _>::new(n, vec![C32::default(); n / 2]).unwrap();
        let x = c32_noise(&mut rng, n, 1.0);
        for dir in [Direction::Forward, Direction::Inverse] {
            let mut aligned = x.clone();
            Simd.fft(&plan, &mut aligned, dir, Scaling::Default)
                .unwrap();
            let mut buffer = vec![C32::default(); n + 1];
            buffer[1..].copy_from_slice(&x);
            Simd.fft(&plan, &mut buffer[1..], dir, Scaling::Default)
                .unwrap();
            assert!(
                aligned
                    .iter()
                    .zip(&buffer[1..])
                    .all(|(a, b)| a.re.to_bits() == b.re.to_bits()
                        && a.im.to_bits() == b.im.to_bits()),
                "n={n} {dir:?}"
            );
        }
    }
}
