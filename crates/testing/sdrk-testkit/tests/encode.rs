//! Tests of the format encoder and the reference decoder (milestone test 4).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unnecessary_wraps,
    clippy::ignored_unit_patterns,
    clippy::items_after_statements,
    clippy::result_large_err,
    clippy::needless_pass_by_value,
    clippy::match_wildcard_for_single_variants,
    clippy::single_match_else
)]

use num_complex::Complex64;
use proptest::prelude::*;
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat, formats};
use sdrk_testkit::encode::{decode_reference, encode};
use sdrk_testkit::rng::Pcg32;

const ALL: [(&str, SampleFormat); 12] = [
    ("CU8", formats::CU8),
    ("CS8", formats::CS8),
    ("CS16", formats::CS16),
    ("CS12_LSB16", formats::CS12_LSB16),
    ("SC16_Q11", formats::SC16_Q11),
    ("CS16_PLANAR", formats::CS16_PLANAR),
    ("PACKED12", formats::PACKED12),
    ("PACKED10", formats::PACKED10),
    ("I2S32_MSB", formats::I2S32_MSB),
    ("CF32", formats::CF32),
    ("RF32", formats::RF32),
    ("RS16", formats::RS16),
];

fn enc(f: &SampleFormat, s: &[(f64, f64)]) -> Vec<u8> {
    let samples: Vec<Complex64> = s.iter().map(|&(re, im)| Complex64::new(re, im)).collect();
    let mut out = vec![0u8; samples.len() * 8 + 8];
    let n = encode(f, &samples, &mut out).unwrap();
    out.truncate(n);
    out
}

fn lsb(f: &SampleFormat) -> f64 {
    match f.container() {
        Container::F32 => 1e-7 * 2.0,
        _ => {
            let fs = match f.full_scale() {
                FullScale::Explicit(v) => f64::from(v),
                _ => 2f64.powi(i32::from(f.bits()) - 1),
            };
            1.0 / fs
        }
    }
}

fn random_samples(rng: &mut Pcg32, n: usize, amp: f64) -> Vec<Complex64> {
    (0..n)
        .map(|_| {
            Complex64::new(
                (rng.next_f64() * 2.0 - 1.0) * amp,
                (rng.next_f64() * 2.0 - 1.0) * amp,
            )
        })
        .collect()
}

fn roundtrip_error(f: &SampleFormat, x: &[Complex64]) -> f64 {
    let mut bytes = vec![0u8; x.len() * 8];
    let n = encode(f, x, &mut bytes).unwrap();
    let mut back = vec![Complex64::new(0.0, 0.0); x.len()];
    let count = decode_reference(f, &bytes[..n], &mut back).unwrap();
    assert_eq!(count, x.len());
    let real = f.layout() == Layout::Real;
    x.iter()
        .zip(&back)
        .map(|(a, b)| {
            let ai = if real { 0.0 } else { a.im };
            (a.re - b.re).abs().max((ai - b.im).abs())
        })
        .fold(0.0, f64::max)
}

#[test]
fn round_trip_error_is_at_most_half_an_lsb_for_every_format() {
    let mut rng = Pcg32::new(1234, 1);
    let x = random_samples(&mut rng, 2000, 0.99);
    for (name, f) in ALL {
        let err = roundtrip_error(&f, &x);
        assert!(err <= 0.5 * lsb(&f) + 1e-12, "{name}: error {err}");
    }
}

#[test]
fn round_trip_holds_for_swapped_big_endian_and_msb_variants() {
    let mut rng = Pcg32::new(77, 1);
    let x = random_samples(&mut rng, 512, 0.99);
    let mut variants = Vec::new();
    for (name, f) in ALL {
        variants.push((format!("{name}+swap"), f.with_swap_iq(true)));
    }
    for (c, bits, signed) in [
        (Container::I16, 16, true),
        (Container::I16, 12, true),
        (Container::I32, 24, true),
        (Container::I8, 8, true),
        (Container::U8, 8, false),
    ] {
        for align in [Align::Lsb, Align::Msb] {
            for endian in [Endian::Little, Endian::Big] {
                for layout in [Layout::Interleaved, Layout::Planar, Layout::Real] {
                    let f = SampleFormat::new(c, bits, signed, layout, align, endian).unwrap();
                    variants.push((format!("{c:?}/{bits}/{align:?}/{endian:?}/{layout:?}"), f));
                }
            }
        }
    }
    variants.push((
        "F32 big".into(),
        SampleFormat::new(
            Container::F32,
            32,
            true,
            Layout::Interleaved,
            Align::Lsb,
            Endian::Big,
        )
        .unwrap(),
    ));
    for (name, f) in variants {
        let err = roundtrip_error(&f, &x);
        assert!(err <= 0.5 * lsb(&f) + 1e-12, "{name}: error {err}");
    }
}

#[test]
fn spot_values_of_the_milestone() {
    // CU8: 0.0 -> 127.5 -> 128 (half up), -1.0 -> -0.5 -> 0, +1.0 -> 255.5 -> 255 (saturated)
    assert_eq!(
        enc(&formats::CU8, &[(0.0, -1.0), (1.0, 1.0)]),
        [128, 0, 255, 255]
    );
    // CS16: +1.0 -> 32768 -> 32767
    assert_eq!(
        enc(&formats::CS16, &[(1.0, -1.0)]),
        [0xFF, 0x7F, 0x00, 0x80]
    );
    // CS12_LSB16: +0.5 -> 1024 = 0x0400
    assert_eq!(
        enc(&formats::CS12_LSB16, &[(0.5, 0.0)]),
        [0x00, 0x04, 0x00, 0x00]
    );
    // SC16_Q11: -1.0 -> -2048 = 0xF800, +1.0 -> 2048 saturates to 2047 = 0x07FF
    assert_eq!(
        enc(&formats::SC16_Q11, &[(-1.0, 1.0)]),
        [0x00, 0xF8, 0xFF, 0x07]
    );
    // PACKED12: I = 1, Q = -1 (as 12-bit amplitudes 1/2048 and -1/2048):
    // v0 = 0x001, v1 = 0xFFF -> b0 = v0 & 0xFF = 0x01, b1 = (v0 >> 8) | (v1 & 0xF) << 4 = 0xF0,
    // b2 = v1 >> 4 = 0xFF
    assert_eq!(
        enc(&formats::PACKED12, &[(1.0 / 2048.0, -1.0 / 2048.0)]),
        [0x01, 0xF0, 0xFF]
    );
    // PACKED10: values (1, -1, 2, -2), LSB first, 10 bits each:
    //   v0 = 0x001, v1 = 0x3FF, v2 = 0x002, v3 = 0x3FE
    //   acc = v0 | v1 << 10 | v2 << 20 | v3 << 30
    //       = 0x0000000001 + 0x00000FFC00 + 0x0000200000 + 0xFF80000000 = 0xFF802FFC01
    //   bytes (little endian, 5 bytes): 01 FC 2F 80 FF
    let p10 = enc(
        &formats::PACKED10,
        &[(1.0 / 512.0, -1.0 / 512.0), (2.0 / 512.0, -2.0 / 512.0)],
    );
    assert_eq!(p10, [0x01, 0xFC, 0x2F, 0x80, 0xFF]);
}

#[test]
fn byte_layouts_of_special_formats() {
    // swap_iq: the first value of each pair is Q
    let swapped = formats::CS8.with_swap_iq(true);
    assert_eq!(enc(&swapped, &[(0.5, -0.5)]), [0xC0, 0x40]);
    // planar: all I, then all Q
    assert_eq!(
        enc(&formats::CS16_PLANAR, &[(0.5, -0.5), (0.25, 0.0)]),
        [0x00, 0x40, 0x00, 0x20, 0x00, 0xC0, 0x00, 0x00]
    );
    // MSB aligned 24 in 32: 0.5 -> 0x400000 << 8, little endian
    assert_eq!(
        enc(&formats::I2S32_MSB, &[(0.5, -0.5)]),
        [0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0xC0]
    );
    // big endian reverses the bytes of the container
    let big = SampleFormat::new(
        Container::I16,
        16,
        true,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Big,
    )
    .unwrap();
    assert_eq!(enc(&big, &[(0.5, -0.5)]), [0x40, 0x00, 0xC0, 0x00]);
    // real layout uses only the real part
    assert_eq!(
        enc(&formats::RS16, &[(0.5, 0.9), (-0.5, 0.9)]),
        [0x00, 0x40, 0x00, 0xC0]
    );
    // float containers are stored unscaled
    assert_eq!(
        enc(&formats::CF32, &[(0.5, -2.0)]),
        [0, 0, 0, 0x3F, 0, 0, 0, 0xC0]
    );
    // NaN encodes as zero amplitude
    assert_eq!(
        enc(&formats::CS16, &[(f64::NAN, f64::INFINITY)]),
        [0, 0, 0xFF, 0x7F]
    );
}

#[test]
fn saturation_at_both_ends() {
    assert_eq!(enc(&formats::CS8, &[(10.0, -10.0)]), [127, 0x80]);
    assert_eq!(enc(&formats::CU8, &[(-10.0, 10.0)]), [0, 255]);
    assert_eq!(
        enc(&formats::CS12_LSB16, &[(1.0, -1.0)]),
        [0xFF, 0x07, 0x00, 0xF8]
    );
}

#[test]
fn decode_reference_values() {
    let mut out = [Complex64::new(9.0, 9.0); 2];
    assert_eq!(
        decode_reference(&formats::CU8, &[0, 255, 128, 127], &mut out).unwrap(),
        2
    );
    assert_eq!(out[0], Complex64::new(-127.5 / 128.0, 127.5 / 128.0));
    assert_eq!(out[1], Complex64::new(0.5 / 128.0, -0.5 / 128.0));
    // sign extension of the low 12 bits for LSB aligned data
    let mut one = [Complex64::new(0.0, 0.0); 1];
    decode_reference(&formats::CS12_LSB16, &[0xFF, 0xFF, 0x00, 0x08], &mut one).unwrap();
    assert_eq!(one[0], Complex64::new(-1.0 / 2048.0, -1.0));
    // trailing bytes that do not make a sample are ignored
    assert_eq!(
        decode_reference(&formats::CS16, &[0; 7], &mut out).unwrap(),
        1
    );
    // PACKED12 byte pattern from the documentation
    decode_reference(&formats::PACKED12, &[0x01, 0xF0, 0xFF], &mut one).unwrap();
    assert_eq!(one[0], Complex64::new(1.0 / 2048.0, -1.0 / 2048.0));
}

#[test]
fn errors_are_reported_not_panicked() {
    let mut out = [0u8; 4];
    let s = [Complex64::new(0.0, 0.0); 3];
    // too small
    assert!(encode(&formats::CS16, &s, &mut out).is_err());
    // PACKED10 interleaved needs an even sample count
    assert!(encode(&formats::PACKED10, &s, &mut [0u8; 64]).is_err());
    // Bits1 is not supported
    let bits1 = SampleFormat::new(
        Container::Bits1,
        1,
        true,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Little,
    )
    .unwrap();
    assert!(encode(&bits1, &s, &mut [0u8; 64]).is_err());
    assert!(decode_reference(&bits1, &[0u8; 4], &mut [Complex64::new(0.0, 0.0); 4]).is_err());
    // decode output too small
    assert!(
        decode_reference(
            &formats::CS16,
            &[0u8; 16],
            &mut [Complex64::new(0.0, 0.0); 2]
        )
        .is_err()
    );
    // empty input is fine
    assert_eq!(encode(&formats::CS16, &[], &mut out).unwrap(), 0);
}

proptest! {
    #[test]
    fn prop_round_trip_for_random_samples(
        idx in 0usize..12,
        swap in any::<bool>(),
        seed in any::<u64>(),
        amp in 0.0f64..0.99,
    ) {
        let (name, f) = ALL[idx];
        let f = f.with_swap_iq(swap);
        let x = random_samples(&mut Pcg32::new(seed, 3), 64, amp);
        let err = roundtrip_error(&f, &x);
        prop_assert!(err <= 0.5 * lsb(&f) + 1e-12, "{} error {}", name, err);
    }

    #[test]
    fn prop_encoding_is_monotonic_for_signed_16_bit(a in -0.99f64..0.99, b in -0.99f64..0.99) {
        let ra = i16::from_le_bytes(enc(&formats::CS16, &[(a, 0.0)])[..2].try_into().unwrap());
        let rb = i16::from_le_bytes(enc(&formats::CS16, &[(b, 0.0)])[..2].try_into().unwrap());
        prop_assert_eq!(a <= b, ra <= rb || a == b);
    }
}
