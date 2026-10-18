//! Tests of `convert`, `rotate` and `fir` that need the testkit: the independent
//! encoder as oracle for `convert`, the zero-copy views and block invariance.
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

use num_complex::Complex64;
use proptest::prelude::*;
use sdrk::Sdrk;
use sdrk_core::kernels::convert::{Convert, ConvertReal, view_c32, view_cq15};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::num::q15_from_f64;
use sdrk_core::{C32, CQ15, Progress, Q15, Sample};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat, formats};
use sdrk_testkit::encode::{decode_reference, encode};
use sdrk_testkit::{ChunkPlan, Pcg32, bit_exact, check_block_invariance};

fn format_strategy() -> impl Strategy<Value = SampleFormat> {
    (
        0usize..7,
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        0usize..3,
        prop::option::of(1u32..=1_000_000),
        any::<u8>(),
    )
        .prop_filter_map(
            "valid format",
            |(c, msb, big, swap, layout, explicit, bits_seed)| {
                let (container, max_bits, signed) = match c {
                    0 => (Container::U8, 8u8, false),
                    1 => (Container::I8, 8, true),
                    2 => (Container::I16, 16, true),
                    3 => (Container::I32, 24, true),
                    4 => (Container::F32, 32, true),
                    5 => (Container::Packed12, 12, true),
                    _ => (Container::Packed10, 10, true),
                };
                let packed = matches!(container, Container::Packed12 | Container::Packed10);
                let bits = match container {
                    Container::U8 | Container::F32 | Container::Packed12 | Container::Packed10 => {
                        max_bits
                    }
                    _ => 1 + bits_seed % max_bits,
                };
                let layout = match layout {
                    0 => Layout::Interleaved,
                    1 if !packed => Layout::Planar,
                    _ => Layout::Real,
                };
                let align = if msb { Align::Msb } else { Align::Lsb };
                let endian = if big { Endian::Big } else { Endian::Little };
                let mut f =
                    SampleFormat::new(container, bits, signed, layout, align, endian).ok()?;
                f = f.with_swap_iq(swap);
                // Explicit full scales only where the oracle and the integer rules coincide.
                if let (Some(n), false) = (explicit, container == Container::F32) {
                    f = f.with_full_scale(FullScale::Explicit(n)).ok()?;
                }
                Some(f)
            },
        )
}

proptest! {
    #[test]
    fn convert_matches_the_encoder_oracle(format in format_strategy(), seed in any::<u64>(), groups in 1usize..16) {
        let n = groups * 8;
        let mut rng = Pcg32::new(seed, 11);
        let samples: Vec<Complex64> = (0..n)
            .map(|_| Complex64::new((rng.next_f64() * 2.0 - 1.0) * 1.1, (rng.next_f64() * 2.0 - 1.0) * 1.1))
            .collect();
        let mut bytes = vec![0u8; n * 8];
        let used = encode(&format, &samples, &mut bytes).unwrap();
        let raw = &bytes[..used];
        let mut oracle = vec![Complex64::new(0.0, 0.0); n];
        prop_assert_eq!(decode_reference(&format, raw, &mut oracle).unwrap(), n);
        if format.layout() == Layout::Real {
            let mut f = vec![0f32; n];
            let mut q = vec![Q15(0); n];
            prop_assert_eq!(ConvertReal::<f32>::convert_real(&Sdrk, &format, raw, &mut f).unwrap(), n);
            ConvertReal::<Q15>::convert_real(&Sdrk, &format, raw, &mut q).unwrap();
            for ((f, q), want) in f.iter().zip(&q).zip(&oracle) {
                prop_assert_eq!(f.to_bits(), (want.re as f32).to_bits());
                prop_assert_eq!(*q, q15_from_f64(want.re));
            }
        } else {
            let mut f = vec![C32::ZERO; n];
            let mut q = vec![CQ15::ZERO; n];
            prop_assert_eq!(Convert::<C32>::convert(&Sdrk, &format, raw, &mut f).unwrap(), n);
            Convert::<CQ15>::convert(&Sdrk, &format, raw, &mut q).unwrap();
            for ((f, q), want) in f.iter().zip(&q).zip(&oracle) {
                prop_assert_eq!(f.re.to_bits(), (want.re as f32).to_bits());
                prop_assert_eq!(f.im.to_bits(), (want.im as f32).to_bits());
                prop_assert_eq!(q.re, q15_from_f64(want.re));
                prop_assert_eq!(q.im, q15_from_f64(want.im));
            }
        }
    }
}

#[test]
fn views_exist_only_for_compatible_formats() {
    let cq: Vec<CQ15> = (0..64)
        .map(|i| CQ15 {
            re: Q15(i * 311 - 9000),
            im: Q15(-i * 97),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&cq);

    let view = view_cq15(&formats::CS16, bytes).expect("CS16 is compatible");
    let mut converted = vec![CQ15::ZERO; cq.len()];
    Convert::<CQ15>::convert(&Sdrk, &formats::CS16, bytes, &mut converted).unwrap();
    assert_eq!(bit_exact(view, &converted), None);
    assert_eq!(bit_exact(view, &cq), None);

    // Incompatible formats and options.
    let swapped = formats::CS16.with_swap_iq(true);
    let explicit = formats::CS16
        .with_full_scale(FullScale::Explicit(2048))
        .unwrap();
    let big = SampleFormat::new(
        Container::I16,
        16,
        true,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Big,
    )
    .unwrap();
    for f in [
        formats::CS8,
        formats::CS12_LSB16,
        formats::CS16_PLANAR,
        formats::RS16,
        formats::CF32,
        swapped,
        explicit,
        big,
    ] {
        assert!(view_cq15(&f, bytes).is_none(), "{f:?}");
    }
    // Misaligned or partial buffers.
    assert!(view_cq15(&formats::CS16, &bytes[1..bytes.len() - 3]).is_none());
    assert!(view_cq15(&formats::CS16, &bytes[..bytes.len() - 1]).is_none());
    assert_eq!(view_cq15(&formats::CS16, &[]).map(<[CQ15]>::len), Some(0));

    let c32: Vec<C32> = (0..64)
        .map(|i| C32::new(i as f32 * 0.25, -(i as f32)))
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&c32);
    let view = view_c32(&formats::CF32, bytes).expect("CF32 is compatible");
    let mut converted = vec![C32::ZERO; c32.len()];
    Convert::<C32>::convert(&Sdrk, &formats::CF32, bytes, &mut converted).unwrap();
    assert_eq!(bit_exact(view, &converted), None);
    let big = SampleFormat::new(
        Container::F32,
        32,
        true,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Big,
    )
    .unwrap();
    for f in [
        formats::CS16,
        formats::RF32,
        formats::CF32.with_swap_iq(true),
        big,
    ] {
        assert!(view_c32(&f, bytes).is_none(), "{f:?}");
    }
    assert!(view_c32(&formats::CF32, &bytes[1..bytes.len() - 3]).is_none());
    assert!(view_c32(&formats::CF32, &bytes[..bytes.len() - 3]).is_none());
}

fn same_rate<T: Copy>(
    input: &[T],
    output: &mut [T],
    kernel: impl FnOnce(&[T], &mut [T]) -> sdrk_core::Result<()>,
) -> sdrk_core::Result<Progress> {
    let n = input.len().min(output.len());
    kernel(&input[..n], &mut output[..n])?;
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

#[test]
fn rotate_is_block_size_invariant() {
    const N: usize = 3000;
    let mut rng = Pcg32::new(60, 1);
    let c32: Vec<C32> = (0..N)
        .map(|_| C32::new(rng.normal() as f32 * 0.3, rng.normal() as f32 * 0.3))
        .collect();
    let cq: Vec<CQ15> = (0..N)
        .map(|_| CQ15 {
            re: Q15((rng.next_u32() >> 16) as i16),
            im: Q15((rng.next_u32() >> 16) as i16),
        })
        .collect();
    for seed in 0..20u64 {
        let inc = rng.next_u32() | 1;
        let phase = rng.next_u32();
        let make = || {
            let mut n = Nco::from_increment(inc);
            n.set_phase(phase);
            n
        };
        let (mut a, mut b) = (vec![C32::ZERO; N], vec![C32::ZERO; N]);
        let r = check_block_invariance(
            &c32,
            &mut a,
            &mut b,
            ChunkPlan::new(seed, 1 + (seed as usize * 37) % 400),
            make,
            |nco, i, o| same_rate(i, o, |i, o| Rotate::<C32>::rotate(&Sdrk, nco, i, o)),
        )
        .unwrap();
        assert_eq!(r.outputs, N);
        let (mut a, mut b) = (vec![CQ15::ZERO; N], vec![CQ15::ZERO; N]);
        check_block_invariance(
            &cq,
            &mut a,
            &mut b,
            ChunkPlan::new(seed + 100, 1 + (seed as usize * 53) % 500),
            make,
            |nco, i, o| same_rate(i, o, |i, o| Rotate::<CQ15>::rotate(&Sdrk, nco, i, o)),
        )
        .unwrap();
    }
}

#[test]
fn fir_is_block_size_invariant() {
    const N: usize = 3000;
    let mut rng = Pcg32::new(61, 1);
    let c32: Vec<C32> = (0..N)
        .map(|_| C32::new(rng.normal() as f32 * 0.3, rng.normal() as f32 * 0.3))
        .collect();
    let cq: Vec<CQ15> = (0..N)
        .map(|_| CQ15 {
            re: Q15((rng.next_u32() >> 16) as i16),
            im: Q15((rng.next_u32() >> 16) as i16),
        })
        .collect();
    for seed in 0..20u64 {
        // Tap counts from a single tap up to longer than most chunks.
        let taps_n = 1 + (seed as usize * 7) % 90;
        let taps32: Vec<f32> = (0..taps_n).map(|_| rng.normal() as f32 * 0.1).collect();
        let tapsq: Vec<Q15> = (0..taps_n)
            .map(|_| Q15((rng.next_u32() >> 20) as i16 - 2048))
            .collect();
        let (mut a, mut b) = (vec![C32::ZERO; N], vec![C32::ZERO; N]);
        check_block_invariance(
            &c32,
            &mut a,
            &mut b,
            ChunkPlan::new(seed, 1 + (seed as usize * 41) % 300),
            || FirState::<C32, _, _>::new(taps32.clone(), vec![C32::ZERO; taps_n - 1]),
            |st, i, o| {
                let st = st.as_mut().map_err(|e| e.clone())?;
                same_rate(i, o, |i, o| Fir::<C32>::fir(&Sdrk, st, i, o))
            },
        )
        .unwrap();
        let (mut a, mut b) = (vec![CQ15::ZERO; N], vec![CQ15::ZERO; N]);
        check_block_invariance(
            &cq,
            &mut a,
            &mut b,
            ChunkPlan::new(seed + 100, 1 + (seed as usize * 29) % 250),
            || FirState::<CQ15, _, _>::new(tapsq.clone(), vec![CQ15::ZERO; taps_n - 1]),
            |st, i, o| {
                let st = st.as_mut().map_err(|e| e.clone())?;
                same_rate(i, o, |i, o| Fir::<CQ15>::fir(&Sdrk, st, i, o))
            },
        )
        .unwrap();
    }
}
