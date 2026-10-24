//! Tests of the transmit and fused kernels of `sdrk-core` (M08): `convert_out`,
//! `fir_interpolate`, `fm_mod` and the fused `ddc`, `duc` and `fm_demod`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::cast_lossless
)]

use sdrk_core::fused::{ddc_chained, duc_chained, fm_demod_chained};
use sdrk_core::kernels::convert::{Convert, ConvertOut, ConvertOutReal, ConvertReal};
use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::demod::{Deemphasis, DeemphasisState, QuadDemod, QuadDemodState};
use sdrk_core::kernels::fused::{Ddc, Duc, FmDemod};
use sdrk_core::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::timing::{Delay, RatioKind, Timing};
use sdrk_core::{C32, CQ15, Error, Fallback, Progress, Q15, Rate, Ratio, Sample};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat, SdrkError, formats};

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn random_c32(n: usize, seed: u64, scale: f32) -> Vec<C32> {
    let mut s = seed;
    (0..n)
        .map(|_| C32::new(unit(&mut s) * scale, unit(&mut s) * scale))
        .collect()
}

fn random_cq15(n: usize, seed: u64) -> Vec<CQ15> {
    let mut s = seed;
    (0..n)
        .map(|_| CQ15 {
            re: Q15(lcg(&mut s) as i16),
            im: Q15(lcg(&mut s) as i16),
        })
        .collect()
}

fn random_f32(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| unit(&mut s) * scale).collect()
}

fn random_q15(n: usize, seed: u64) -> Vec<Q15> {
    let mut s = seed;
    (0..n).map(|_| Q15(lcg(&mut s) as i16)).collect()
}

fn fmt(
    container: Container,
    bits: u8,
    layout: Layout,
    align: Align,
    endian: Endian,
) -> SampleFormat {
    SampleFormat::new(
        container,
        bits,
        container != Container::U8,
        layout,
        align,
        endian,
    )
    .unwrap()
}

/// Every kind of format the encoder has to handle.
fn all_formats() -> Vec<SampleFormat> {
    vec![
        formats::CU8,
        formats::CS8,
        formats::CS16,
        formats::CS12_LSB16,
        formats::SC16_Q11,
        formats::CS16_PLANAR,
        formats::PACKED12,
        formats::PACKED10,
        formats::I2S32_MSB,
        formats::CF32,
        formats::RF32,
        formats::RS16,
        formats::CU8.with_swap_iq(true),
        formats::CU8
            .with_full_scale(FullScale::Explicit(200))
            .unwrap(),
        formats::CS16
            .with_full_scale(FullScale::Explicit(30000))
            .unwrap()
            .with_swap_iq(true),
        fmt(
            Container::I16,
            16,
            Layout::Interleaved,
            Align::Lsb,
            Endian::Big,
        ),
        fmt(Container::I16, 10, Layout::Planar, Align::Msb, Endian::Big),
        fmt(
            Container::I32,
            32,
            Layout::Interleaved,
            Align::Lsb,
            Endian::Big,
        ),
        fmt(Container::I32, 20, Layout::Real, Align::Lsb, Endian::Little),
        fmt(
            Container::F32,
            32,
            Layout::Interleaved,
            Align::Lsb,
            Endian::Big,
        )
        .with_swap_iq(true),
        fmt(
            Container::I8,
            6,
            Layout::Interleaved,
            Align::Msb,
            Endian::Little,
        ),
        fmt(Container::I8, 5, Layout::Planar, Align::Lsb, Endian::Little),
        fmt(Container::I16, 12, Layout::Real, Align::Msb, Endian::Little),
    ]
}

fn encode_c32(f: &SampleFormat, input: &[C32]) -> Vec<u8> {
    let mut out = vec![0u8; f.bytes_for_samples(input.len()).unwrap()];
    let n = Fallback.convert_out(f, input, &mut out).unwrap();
    assert_eq!(n, out.len());
    out
}

fn encode_cq15(f: &SampleFormat, input: &[CQ15]) -> Vec<u8> {
    let mut out = vec![0u8; f.bytes_for_samples(input.len()).unwrap()];
    let n = Fallback.convert_out(f, input, &mut out).unwrap();
    assert_eq!(n, out.len());
    out
}

#[test]
fn convert_out_spot_values() {
    let c = C32::new;
    // CU8: zero is at 127.5, so 0.0 rounds to the even neighbour 128; the range saturates.
    // -0.5 gives 63.5, which rounds to 64.
    assert_eq!(encode_c32(&formats::CU8, &[c(0.0, -1.0)]), [128, 0]);
    assert_eq!(encode_c32(&formats::CU8, &[c(1.0, 2.0)]), [255, 255]);
    assert_eq!(encode_c32(&formats::CU8, &[c(-0.5, f32::NAN)]), [64, 128]);
    // CS8: +1.0 saturates to 127, -1.0 is -128.
    assert_eq!(encode_c32(&formats::CS8, &[c(1.0, -1.0)]), [127, 128]);
    assert_eq!(
        encode_c32(&formats::CS8, &[c(0.5, f32::INFINITY)]),
        [64, 127]
    );
    // Ties to even at the half-LSB boundaries: 0.5 / 128 is exactly half an LSB.
    assert_eq!(
        encode_c32(
            &formats::CS8,
            &[c(0.5 / 128.0, -0.5 / 128.0), c(1.5 / 128.0, -1.5 / 128.0)]
        ),
        [0, 0, 2, 254]
    );
    // CS16 little endian.
    assert_eq!(
        encode_c32(&formats::CS16, &[c(1.0, -1.0)]),
        [0xFF, 0x7F, 0x00, 0x80]
    );
    // 12 bit in the low bits of a 16-bit word is sign-extended.
    assert_eq!(
        encode_c32(&formats::CS12_LSB16, &[c(0.5, -1.0)]),
        [0x00, 0x04, 0x00, 0xF8]
    );
    // bladeRF Q11: full scale 2048, 12 bit.
    assert_eq!(
        encode_c32(&formats::SC16_Q11, &[c(-1.0, 1.0)]),
        [0x00, 0xF8, 0xFF, 0x07]
    );
    // Planar: all I, then all Q.
    assert_eq!(
        encode_c32(&formats::CS16_PLANAR, &[c(0.5, -0.5), c(0.25, -0.25)]),
        [0x00, 0x40, 0x00, 0x20, 0x00, 0xC0, 0x00, 0xE0]
    );
    // Packed 12 bit: v0 | v1 << 12 in three bytes.
    let one = 1.0f32 / 2048.0;
    assert_eq!(
        encode_c32(&formats::PACKED12, &[c(one, -one)]),
        [0x01, 0xF0, 0xFF]
    );
    // MSB-aligned 24 bit in 32.
    assert_eq!(
        encode_c32(&formats::I2S32_MSB, &[c(0.5, -0.5)]),
        [0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0xC0]
    );
    // F32 containers copy the bits (also NaN and -0.0).
    let cf = encode_c32(&formats::CF32, &[c(f32::NAN, -0.0)]);
    assert_eq!(&cf[..4], &f32::NAN.to_bits().to_le_bytes());
    assert_eq!(&cf[4..], &(-0.0f32).to_bits().to_le_bytes());
    // I/Q swap and big endian.
    assert_eq!(
        encode_c32(&formats::CU8.with_swap_iq(true), &[c(-1.0, 1.0)]),
        [255, 0]
    );
    let be = fmt(
        Container::I16,
        16,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Big,
    );
    assert_eq!(encode_c32(&be, &[c(1.0, -1.0)]), [0x7F, 0xFF, 0x80, 0x00]);
}

#[test]
fn convert_out_q15_integer_rules() {
    let q = |re, im| CQ15 {
        re: Q15(re),
        im: Q15(im),
    };
    // U8: (q + 32640) / 256 rounded to the nearest integer, ties to even, clamped to 0..=255.
    assert_eq!(encode_cq15(&formats::CU8, &[q(0, 32767)]), [128, 255]);
    assert_eq!(encode_cq15(&formats::CU8, &[q(-32768, 32640)]), [0, 255]);
    assert_eq!(encode_cq15(&formats::CU8, &[q(-127, -128)]), [127, 127]);
    assert_eq!(encode_cq15(&formats::CU8, &[q(256, 512)]), [128, 130]);
    // Signed 8 bit: q / 256 rounded, ties to even, saturated.
    assert_eq!(encode_cq15(&formats::CS8, &[q(127, 128)]), [0, 0]);
    assert_eq!(encode_cq15(&formats::CS8, &[q(384, -384)]), [2, 254]);
    assert_eq!(encode_cq15(&formats::CS8, &[q(32767, -32768)]), [127, 128]);
    assert_eq!(encode_cq15(&formats::CS8, &[q(-128, -129)]), [0, 255]);
    // 16 bit: identity. 12 bit: q / 16 rounded, ties to even.
    assert_eq!(
        encode_cq15(&formats::CS16, &[q(-32768, 32767)]),
        [0x00, 0x80, 0xFF, 0x7F]
    );
    assert_eq!(
        encode_cq15(&formats::CS12_LSB16, &[q(7, 8)]),
        [0x00, 0x00, 0x00, 0x00]
    );
    assert_eq!(
        encode_cq15(&formats::CS12_LSB16, &[q(24, 40)]),
        [0x02, 0x00, 0x02, 0x00]
    );
    assert_eq!(
        encode_cq15(&formats::CS12_LSB16, &[q(32767, -32768)]),
        [0xFF, 0x07, 0x00, 0xF8]
    );
    // More than 16 bits: shifted left (24 bit, MSB-aligned in 32).
    assert_eq!(
        encode_cq15(&formats::I2S32_MSB, &[q(1, -1)]),
        [0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF]
    );
    // Explicit full scale 2048: q * 2048 / 32768 rounded, ties to even.
    assert_eq!(
        encode_cq15(&formats::SC16_Q11, &[q(16, -16)]),
        [0x01, 0x00, 0xFF, 0xFF]
    );
    assert_eq!(
        encode_cq15(&formats::SC16_Q11, &[q(8, -8)]),
        [0x00, 0x00, 0x00, 0x00]
    );
    assert_eq!(
        encode_cq15(&formats::SC16_Q11, &[q(24, -24)]),
        [0x02, 0x00, 0xFE, 0xFF]
    );
    // F32 containers get the exact f32 value.
    let cf = encode_cq15(&formats::CF32, &[q(16384, -32768)]);
    assert_eq!(&cf[..4], &0.5f32.to_le_bytes());
    assert_eq!(&cf[4..], &(-1.0f32).to_le_bytes());
    // Real layouts.
    let mut out = [0u8; 4];
    let n = Fallback
        .convert_out_real(&formats::RS16, &[Q15(16384), Q15(-16384)], &mut out)
        .unwrap();
    assert_eq!((n, out), (4, [0x00, 0x40, 0x00, 0xC0]));
    let n = Fallback
        .convert_out_real(&formats::RF32, &[-0.5f32], &mut out)
        .unwrap();
    assert_eq!((n, out), (4, (-0.5f32).to_le_bytes()));
}

#[test]
fn convert_out_saturates_and_handles_special_values() {
    let specials = [
        0.0f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        -1.0,
        1.0e-40,
        3.0,
        -3.0,
        f32::MAX,
        f32::MIN,
    ];
    let input: Vec<C32> = specials.chunks(2).map(|p| C32::new(p[0], p[1])).collect();
    for f in all_formats() {
        if f.layout() == Layout::Real {
            continue;
        }
        let bytes = encode_c32(&f, &input);
        // Decoding the output stays within the range of the format.
        let mut back = vec![C32::ZERO; input.len()];
        Fallback.convert(&f, &bytes, &mut back).unwrap();
        for (x, y) in input.iter().zip(&back) {
            for (a, b) in [(x.re, y.re), (x.im, y.im)] {
                if f.container() == Container::F32 {
                    assert!(a.to_bits() == b.to_bits(), "{f:?}");
                } else if a.is_finite() && a.abs() <= 0.99 {
                    assert!(
                        (a - b).abs() <= 0.51 / f.full_scale_value() as f32 + 1e-6,
                        "{f:?}"
                    );
                } else {
                    let limit = (1u64 << (f.bits() - 1)) as f32 / f.full_scale_value() as f32;
                    assert!(b.abs() <= limit + 1e-6, "{f:?}");
                }
            }
        }
    }
    // Saturation to the exact limits of 12 bit.
    assert_eq!(
        encode_c32(&formats::CS12_LSB16, &[C32::new(1.0e9, -1.0e9)]),
        [0xFF, 0x07, 0x00, 0xF8]
    );
}

#[test]
fn convert_out_errors() {
    let x = [C32::new(0.0, 0.0); 3];
    let mut out = [0u8; 12];
    // Too small: needs 12 bytes for three CS16 samples.
    let e = Fallback
        .convert_out(&formats::CS16, &x, &mut out[..11])
        .unwrap_err();
    assert_eq!(
        e,
        Error::BufferTooSmall {
            needed: 12,
            got: 11
        }
    );
    assert_eq!(e.kind(), sdrk_device::ErrorKind::BufferTooSmall);
    // Exactly enough is fine; a larger buffer is only written as far as needed.
    assert_eq!(Fallback.convert_out(&formats::CS16, &x, &mut out), Ok(12));
    let mut big = [0xAAu8; 16];
    assert_eq!(Fallback.convert_out(&formats::CS8, &x, &mut big), Ok(6));
    assert_eq!(&big[6..], &[0xAA; 10]);
    // Bits1 is unsupported (checked before the packing group).
    let bits1 = fmt(
        Container::Bits1,
        1,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Little,
    );
    assert!(matches!(
        Fallback.convert_out(&bits1, &x, &mut out),
        Err(Error::Unsupported(_))
    ));
    // Packing groups must be complete: PACKED10 holds two samples per five bytes.
    assert!(matches!(
        Fallback.convert_out(&formats::PACKED10, &x, &mut out),
        Err(Error::InvalidParameter(_))
    ));
    let rf = [0.0f32; 3];
    assert!(matches!(
        Fallback.convert_out_real(
            &fmt(
                Container::Packed12,
                12,
                Layout::Real,
                Align::Lsb,
                Endian::Little
            ),
            &rf,
            &mut out
        ),
        Err(Error::InvalidParameter(_))
    ));
    // Layout mismatch.
    assert!(matches!(
        Fallback.convert_out(&formats::RS16, &x, &mut out),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        Fallback.convert_out_real(&formats::CS16, &rf, &mut out),
        Err(Error::InvalidParameter(_))
    ));
    // Empty input writes nothing.
    assert_eq!(
        Fallback.convert_out(&formats::CS16, &[] as &[C32], &mut []),
        Ok(0)
    );
}

/// Raw bytes that survive decoding and encoding unchanged: the unused bits are what the
/// encoder writes (zero for MSB alignment, the sign extension for LSB alignment).
fn canonical_bytes(f: &SampleFormat, n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let bytes = f.bytes_for_samples(n).unwrap();
    let mut raw: Vec<u8> = (0..bytes).map(|_| (lcg(&mut s) >> 24) as u8).collect();
    let (cbytes, bits) = match f.container() {
        Container::U8 | Container::I8 => (1usize, u32::from(f.bits())),
        Container::I16 => (2, u32::from(f.bits())),
        Container::I32 => (4, u32::from(f.bits())),
        _ => return raw,
    };
    let cbits = cbytes as u32 * 8;
    let signed = f.container() != Container::U8;
    for chunk in raw.chunks_exact_mut(cbytes) {
        let mut w = 0u64;
        for (i, b) in chunk.iter().enumerate() {
            let pos = if f.endian() == Endian::Big {
                cbytes - 1 - i
            } else {
                i
            };
            w |= u64::from(*b) << (8 * pos);
        }
        let value = w & ((1u64 << cbits) - 1);
        let canonical = if f.align() == Align::Msb {
            value & !((1u64 << (cbits - bits)) - 1)
        } else if signed {
            let low = value & ((1u64 << bits) - 1);
            let ext = if low >> (bits - 1) == 1 {
                !0u64 << bits
            } else {
                0
            };
            (low | ext) & ((1u64 << cbits) - 1)
        } else {
            value & ((1u64 << bits) - 1)
        };
        for (i, b) in chunk.iter_mut().enumerate() {
            let pos = if f.endian() == Endian::Big {
                cbytes - 1 - i
            } else {
                i
            };
            *b = (canonical >> (8 * pos)) as u8;
        }
    }
    raw
}

#[test]
fn decoding_then_encoding_reproduces_the_bytes() {
    for (i, f) in all_formats().iter().enumerate() {
        if f.container() == Container::F32 || f.bits() > 24 {
            continue; // F32 is covered below; more than 24 bits do not fit an f32 mantissa
        }
        let n = 128;
        let bytes = canonical_bytes(f, n, 100 + i as u64);
        if f.layout() == Layout::Real {
            let mut x = vec![0f32; n];
            Fallback.convert_real(f, &bytes, &mut x).unwrap();
            let mut back = vec![0u8; bytes.len()];
            Fallback.convert_out_real(f, &x, &mut back).unwrap();
            assert_eq!(bytes, back, "f32, format {i}: {f:?}");
            if f.bits() <= 16 && f.full_scale() == FullScale::PowerOfTwo {
                let mut q = vec![Q15(0); n];
                Fallback.convert_real(f, &bytes, &mut q).unwrap();
                Fallback.convert_out_real(f, &q, &mut back).unwrap();
                assert_eq!(bytes, back, "q15, format {i}: {f:?}");
            }
        } else {
            let mut x = vec![C32::ZERO; n];
            Fallback.convert(f, &bytes, &mut x).unwrap();
            let mut back = vec![0u8; bytes.len()];
            Fallback.convert_out(f, &x, &mut back).unwrap();
            assert_eq!(bytes, back, "c32, format {i}: {f:?}");
            if f.bits() <= 16 && f.full_scale() == FullScale::PowerOfTwo {
                let mut q = vec![CQ15::ZERO; n];
                Fallback.convert(f, &bytes, &mut q).unwrap();
                Fallback.convert_out(f, &q, &mut back).unwrap();
                assert_eq!(bytes, back, "cq15, format {i}: {f:?}");
            }
        }
    }
    // F32 containers: a finite value round-trips bit for bit.
    let x = random_c32(64, 5, 4.0);
    for f in [
        formats::CF32,
        fmt(
            Container::F32,
            32,
            Layout::Interleaved,
            Align::Lsb,
            Endian::Big,
        ),
    ] {
        let bytes = encode_c32(&f, &x);
        let mut y = vec![C32::ZERO; x.len()];
        Fallback.convert(&f, &bytes, &mut y).unwrap();
        assert_eq!(x, y);
    }
}

#[test]
fn encoding_then_decoding_stays_within_half_an_lsb() {
    for (i, f) in all_formats().iter().enumerate() {
        if f.container() == Container::F32 || f.layout() == Layout::Real {
            continue;
        }
        let half_lsb = 0.5 / f.full_scale_value() as f32;
        // Stay inside the representable range (explicit full scales can make it smaller).
        let range = (1u64 << (f.bits() - 1)) as f32 / f.full_scale_value() as f32;
        let x = random_c32(256, 300 + i as u64, 0.95 * range.min(1.0) * 0.99);
        let bytes = encode_c32(f, &x);
        let mut y = vec![C32::ZERO; x.len()];
        Fallback.convert(f, &bytes, &mut y).unwrap();
        for (a, b) in x.iter().zip(&y) {
            // U8 has its zero at 127.5, which is not on the grid of the f32 sum, so allow
            // one part in a million on top.
            assert!(
                (a.re - b.re).abs() <= half_lsb * 1.0001 + 1e-7,
                "format {i}: {a:?} {b:?} {half_lsb}"
            );
            assert!(
                (a.im - b.im).abs() <= half_lsb * 1.0001 + 1e-7,
                "format {i}"
            );
        }
        if f.bits() <= 16 && f.full_scale() == FullScale::PowerOfTwo {
            let q = random_cq15(256, 400 + i as u64);
            let bytes = encode_cq15(f, &q);
            let mut y = vec![CQ15::ZERO; q.len()];
            Fallback.convert(f, &bytes, &mut y).unwrap();
            let lsb_q15 = 32768.0 / f.full_scale_value();
            for (a, b) in q.iter().zip(&y) {
                let clipped = |v: i16, w: i16| {
                    let err = (f64::from(v) - f64::from(w)).abs();
                    // Saturation at the top of the range costs at most one more LSB.
                    err <= lsb_q15 * 0.5 + 1.0
                        || (f64::from(v) >= 32768.0 - lsb_q15 * 1.5 && err <= lsb_q15 * 1.5 + 1.0)
                };
                assert!(
                    clipped(a.re.0, b.re.0) && clipped(a.im.0, b.im.0),
                    "format {i}"
                );
            }
        }
    }
}

#[test]
fn explicit_full_scale_and_u8_rounding_to_even() {
    // Explicit(3): q = 16384 is amplitude 0.5, times 3 = 1.5, which rounds to the even 2.
    let f = formats::CS16
        .with_full_scale(FullScale::Explicit(3))
        .unwrap();
    let q = |v| CQ15 {
        re: Q15(v),
        im: Q15(-v),
    };
    assert_eq!(encode_cq15(&f, &[q(16384)]), [2, 0, 0xFE, 0xFF]);
    // The mirrored value -1.5 rounds to the even -2.
    assert_eq!(encode_c32(&f, &[C32::new(0.5, -0.5)]), [2, 0, 0xFE, 0xFF]);
    // CU8 with explicit full scale: amplitude * n + 127.5; 177.5 and 77.5 round to even.
    let u = formats::CU8
        .with_full_scale(FullScale::Explicit(100))
        .unwrap();
    assert_eq!(encode_c32(&u, &[C32::new(0.5, -0.5)]), [178, 78]);
    assert_eq!(encode_cq15(&u, &[q(16384)]), [178, 78]);
    assert_eq!(encode_cq15(&u, &[q(32767)]), [227, 28]);
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<FmModState>();
    assert_sync::<FmModState>();
    assert_send::<FirInterpState<C32, Vec<f32>, Vec<C32>>>();
    assert_sync::<FirInterpState<CQ15, Vec<Q15>, Vec<CQ15>>>();
    assert_send::<FirInterpRealState<f32, Vec<f32>, Vec<f32>>>();
    assert_sync::<FirInterpRealState<Q15, Vec<Q15>, Vec<Q15>>>();
}

// --- fir_interpolate -------------------------------------------------------------------

fn taps_f32(n: usize, seed: u64, gain: f32) -> Vec<f32> {
    random_f32(n, seed, gain)
}

fn interp_state_c32(taps: &[f32], l: usize) -> FirInterpState<C32, Vec<f32>, Vec<C32>> {
    FirInterpState::new(taps.to_vec(), vec![C32::ZERO; taps.len().div_ceil(l)], l).unwrap()
}

#[test]
fn fir_interpolate_state_validation_and_timing() {
    assert!(FirInterpState::<C32, _, _>::new(Vec::<f32>::new(), vec![C32::ZERO; 0], 2).is_err());
    assert!(FirInterpState::<C32, _, _>::new(vec![1.0f32; 5], vec![C32::ZERO; 3], 0).is_err());
    assert_eq!(
        FirInterpState::<C32, _, _>::new(vec![1.0f32; 5], vec![C32::ZERO; 2], 2).unwrap_err(),
        Error::LengthMismatch {
            expected: 3,
            got: 2
        }
    );
    let state = interp_state_c32(&[0.5; 241], 5);
    assert_eq!(state.factor(), 5);
    assert_eq!(state.taps().len(), 241);
    assert_eq!(state.history().len(), 49);
    assert_eq!(state.ratio(), Ratio::new(5, 1).unwrap());
    assert_eq!(state.ratio_kind(), RatioKind::Constant);
    assert_eq!(state.group_delay(), Delay::new(240, 10).unwrap());
    assert_eq!(state.group_delay().reduced(), Delay::new(24, 1).unwrap());
    let real = FirInterpRealState::<Q15, _, _>::new(vec![Q15(0); 11], vec![Q15(0); 3], 4).unwrap();
    assert_eq!(real.group_delay(), Delay::new(10, 8).unwrap());
    assert_eq!(real.ratio(), Ratio::new(4, 1).unwrap());
}

#[test]
fn fir_interpolate_is_resample_with_down_one() {
    let h = taps_f32(37, 1, 0.3);
    let x = random_c32(200, 2, 0.5);
    for l in [1usize, 2, 5, 8] {
        let mut a = interp_state_c32(&h, l);
        let mut b =
            ResampleState::<C32, _, _>::new(h.clone(), vec![C32::ZERO; h.len().div_ceil(l)], l, 1)
                .unwrap();
        let mut ya = vec![C32::ZERO; x.len() * l];
        let mut yb = vec![C32::ZERO; x.len() * l];
        let pa = Fallback.fir_interpolate(&mut a, &x, &mut ya).unwrap();
        let pb = Fallback.resample(&mut b, &x, &mut yb).unwrap();
        assert_eq!(pa, pb);
        assert_eq!(
            pa,
            Progress {
                consumed: x.len(),
                produced: x.len() * l
            }
        );
        assert_eq!(ya, yb);
        assert_eq!(a.history(), b.history());
    }
}

#[test]
fn fir_interpolate_impulse_response_and_exact_output_count() {
    // An impulse produces the taps, the polyphase way: output j*L + p is h[p + L*m].
    let l = 4;
    let h: Vec<f32> = (1..=10).map(|v| v as f32).collect();
    let mut state = interp_state_c32(&h, l);
    let mut x = vec![C32::ZERO; 4];
    x[0] = C32::new(1.0, -2.0);
    let mut y = vec![C32::ZERO; 16];
    let p = Fallback.fir_interpolate(&mut state, &x, &mut y).unwrap();
    assert_eq!(
        p,
        Progress {
            consumed: 4,
            produced: 16
        }
    );
    let re: Vec<f32> = y.iter().map(|s| s.re).collect();
    let im: Vec<f32> = y.iter().map(|s| s.im).collect();
    let mut want = h.clone();
    want.resize(16, 0.0);
    assert_eq!(re, want);
    assert_eq!(im, want.iter().map(|v| -2.0 * v).collect::<Vec<_>>());
    // A call with no input produces nothing.
    let p = Fallback.fir_interpolate(&mut state, &[], &mut y).unwrap();
    assert_eq!(p, Progress::default());
}

#[test]
fn fir_interpolate_small_output_makes_progress() {
    let h = taps_f32(25, 3, 0.4);
    let x = random_c32(40, 4, 0.5);
    let l = 6;
    let mut whole = interp_state_c32(&h, l);
    let mut yw = vec![C32::ZERO; x.len() * l];
    Fallback.fir_interpolate(&mut whole, &x, &mut yw).unwrap();
    // Output room of 4 samples (less than L): the outputs of one input sample come in pieces.
    let mut state = interp_state_c32(&h, l);
    let (mut got, mut pos) = (Vec::new(), 0usize);
    while pos < x.len() || got.len() < yw.len() {
        let mut room = [C32::ZERO; 4];
        let p = Fallback
            .fir_interpolate(&mut state, &x[pos..], &mut room)
            .unwrap();
        got.extend_from_slice(&room[..p.produced]);
        pos += p.consumed;
        assert!(p.consumed + p.produced > 0, "no progress");
    }
    assert_eq!(got, yw);
}

#[test]
fn fir_interpolate_cq15_and_real_variants() {
    let h = random_q15(33, 6)
        .iter()
        .map(|q| Q15(q.0 / 8))
        .collect::<Vec<_>>();
    let x = random_cq15(120, 7);
    let l = 3;
    let mut a =
        FirInterpState::<CQ15, _, _>::new(h.clone(), vec![CQ15::ZERO; h.len().div_ceil(l)], l)
            .unwrap();
    let mut b =
        ResampleState::<CQ15, _, _>::new(h.clone(), vec![CQ15::ZERO; h.len().div_ceil(l)], l, 1)
            .unwrap();
    let (mut ya, mut yb) = (vec![CQ15::ZERO; 360], vec![CQ15::ZERO; 360]);
    Fallback.fir_interpolate(&mut a, &x, &mut ya).unwrap();
    Fallback.resample(&mut b, &x, &mut yb).unwrap();
    assert_eq!(ya, yb);
    // Real f32 and Q15.
    let hr = taps_f32(21, 8, 0.5);
    let xr = random_f32(80, 9, 0.5);
    let mut a =
        FirInterpRealState::<f32, _, _>::new(hr.clone(), vec![0f32; hr.len().div_ceil(5)], 5)
            .unwrap();
    let mut b =
        ResampleRealState::<f32, _, _>::new(hr.clone(), vec![0f32; hr.len().div_ceil(5)], 5, 1)
            .unwrap();
    let (mut ya, mut yb) = (vec![0f32; 400], vec![0f32; 400]);
    Fallback.fir_interpolate_real(&mut a, &xr, &mut ya).unwrap();
    Fallback.resample_real(&mut b, &xr, &mut yb).unwrap();
    assert_eq!(ya, yb);
    let hq: Vec<Q15> = random_q15(21, 10).iter().map(|q| Q15(q.0 / 4)).collect();
    let xq = random_q15(80, 11);
    let mut a = FirInterpRealState::<Q15, _, _>::new(hq.clone(), vec![Q15(0); 6], 4).unwrap();
    let mut b = ResampleRealState::<Q15, _, _>::new(hq.clone(), vec![Q15(0); 6], 4, 1).unwrap();
    let (mut ya, mut yb) = (vec![Q15(0); 320], vec![Q15(0); 320]);
    Fallback.fir_interpolate_real(&mut a, &xq, &mut ya).unwrap();
    Fallback.resample_real(&mut b, &xq, &mut yb).unwrap();
    assert_eq!(ya, yb);
    // Reset clears the history and the phase.
    a.reset();
    assert!(a.history().iter().all(|q| q.0 == 0));
    assert_eq!(a.phase(), 0);
}

// --- fm_mod ----------------------------------------------------------------------------

fn fm_state(dev: f64, fs: u32) -> FmModState {
    FmModState::new(dev, Rate::hz(u64::from(fs)).unwrap()).unwrap()
}

#[test]
fn fm_mod_state_validation_and_accessors() {
    let rate = Rate::hz(240_000).unwrap();
    assert!(FmModState::new(f64::NAN, rate).is_err());
    assert!(FmModState::new(f64::INFINITY, rate).is_err());
    assert!(FmModState::new(-1.0, rate).is_err());
    let mut s = FmModState::new(75_000.0, rate).unwrap();
    assert_eq!(s.scale(), 75_000.0 / 240_000.0 * 4_294_967_296.0);
    assert_eq!(s.phase(), 0);
    s.set_phase(12345);
    assert_eq!(s.phase(), 12345);
    s.reset();
    assert_eq!(s.phase(), 0);
    assert_eq!(s.ratio(), Ratio::ONE);
    assert_eq!(s.group_delay(), Delay::ZERO);
    assert_eq!(s.ratio_kind(), RatioKind::Constant);
    let mut out = [C32::ZERO; 2];
    assert_eq!(
        Fallback.fm_mod(&mut s, &[0.0f32; 3], &mut out),
        Err(Error::LengthMismatch {
            expected: 3,
            got: 2
        })
    );
}

#[test]
fn fm_mod_known_phases() {
    // Deviation fs / 8: an input of 1.0 advances the phase by 45 degrees.
    let mut s = fm_state(125.0, 1000);
    let mut y = [C32::ZERO; 9];
    Fallback.fm_mod(&mut s, &[1.0f32; 9], &mut y).unwrap();
    for (n, v) in y.iter().enumerate() {
        let a = core::f64::consts::PI / 4.0 * (n + 1) as f64;
        assert!((f64::from(v.re) - a.cos()).abs() < 1e-6, "n = {n}");
        assert!((f64::from(v.im) - a.sin()).abs() < 1e-6, "n = {n}");
    }
    assert_eq!(s.phase(), (9u64 * (1 << 29)) as u32);
    // Zero input keeps the phase: the output stays at the current phase.
    let before = s.phase();
    Fallback.fm_mod(&mut s, &[0.0f32; 4], &mut y[..4]).unwrap();
    assert_eq!(s.phase(), before);
    assert!(y[..4].windows(2).all(|w| w[0] == w[1]));
    // Negative input runs the phase backwards.
    let mut s = fm_state(125.0, 1000);
    Fallback.fm_mod(&mut s, &[-1.0f32; 2], &mut y[..2]).unwrap();
    assert_eq!(s.phase(), 0u32.wrapping_sub(1 << 30));
    // Q15 input is q / 32768; full scale is 32767 / 32768 of the deviation.
    let mut s = fm_state(125.0, 1000);
    let mut yq = [CQ15::ZERO; 1];
    Fallback.fm_mod(&mut s, &[Q15(16384)], &mut yq).unwrap();
    assert_eq!(s.phase(), 1 << 28);
    // The output is the cosine table itself: amplitude 1 - 2^-15.
    let mut s = fm_state(0.0, 1000);
    Fallback.fm_mod(&mut s, &[Q15(32767)], &mut yq).unwrap();
    assert_eq!(
        yq[0],
        CQ15 {
            re: Q15(32767),
            im: Q15(0)
        }
    );
}

#[test]
fn fm_mod_rounds_the_phase_step_half_up() {
    // k = 0.5 exactly (deviation 2^-33 Hz at 1 Hz), so x * k + 0.5 is easy to read.
    let rate = Rate::hz(1).unwrap();
    let cases: [(f32, u32); 8] = [
        (1.0, 1),          // 0.5 + 0.5 = 1.0 -> 1
        (0.99, 0),         // 0.495 + 0.5 -> 0
        (-1.0, 0),         // -0.5 + 0.5 = 0 -> 0
        (-1.01, u32::MAX), // -0.505 + 0.5 -> -1
        (3.0, 2),          // 1.5 + 0.5 -> 2
        (-3.0, u32::MAX),  // -1.5 + 0.5 = -1 -> -1
        (0.0, 0),
        (-0.0, 0),
    ];
    for (x, want) in cases {
        let mut s = FmModState::new(2f64.powi(-33), rate).unwrap();
        assert_eq!(s.scale(), 0.5);
        let mut y = [C32::ZERO];
        Fallback.fm_mod(&mut s, &[x], &mut y).unwrap();
        assert_eq!(s.phase(), want, "x = {x}");
    }
}

#[test]
fn fm_mod_handles_extreme_input() {
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        1e-40,
        0.0,
        -0.0,
    ];
    let mut s = fm_state(75_000.0, 240_000);
    let mut y = vec![C32::ZERO; specials.len()];
    Fallback.fm_mod(&mut s, &specials, &mut y).unwrap();
    assert!(
        y.iter()
            .all(|v| v.re.is_finite() && v.im.is_finite() && v.re.abs() <= 1.0001)
    );
    // Full-scale Q15 input stays on the table (|x| <= 1) and silence keeps the phase.
    let mut yq = vec![CQ15::ZERO; 4];
    let mut s = fm_state(75_000.0, 240_000);
    Fallback
        .fm_mod(&mut s, &[Q15::MIN, Q15::MAX, Q15(0), Q15::MIN], &mut yq)
        .unwrap();
    assert!(yq.iter().all(|v| {
        let (re, im) = (i32::from(v.re.0), i32::from(v.im.0));
        (re * re + im * im - 32767 * 32767).abs() < 4 * 32768
    }));
    let mut s = fm_state(75_000.0, 240_000);
    Fallback.fm_mod(&mut s, &[Q15(0); 4], &mut yq).unwrap();
    assert!(yq.iter().all(|v| *v
        == CQ15 {
            re: Q15(32767),
            im: Q15(0)
        }));
}

#[test]
fn fm_mod_then_quad_demod_recovers_the_input() {
    let fs = 240_000u32;
    let dev = 75_000.0;
    let x: Vec<f32> = random_f32(2000, 12, 0.9);
    let mut s = fm_state(dev, fs);
    let mut y = vec![C32::ZERO; x.len()];
    Fallback.fm_mod(&mut s, &x, &mut y).unwrap();
    let mut q = QuadDemodState::<C32>::new();
    let mut d = vec![0f32; x.len()];
    // Gain 1: the discriminator output is the phase step in radians.
    Fallback.quad_demod(&mut q, &y, &mut d, 1.0).unwrap();
    let scale = (2.0 * core::f64::consts::PI * dev / f64::from(fs)) as f32;
    // Sample 0 compares with the zero state; from the second sample on it is x * scale.
    for n in 1..x.len() {
        assert!(
            (d[n] - x[n] * scale).abs() < 1e-4,
            "n = {n}: {} vs {}",
            d[n],
            x[n] * scale
        );
    }
    // Q15: the output unit is phase / pi.
    let xq: Vec<Q15> = x.iter().map(|v| Q15::from_f32(*v * 0.9)).collect();
    let mut s = fm_state(dev, fs);
    let mut yq = vec![CQ15::ZERO; xq.len()];
    Fallback.fm_mod(&mut s, &xq, &mut yq).unwrap();
    let mut q = QuadDemodState::<CQ15>::new();
    let mut dq = vec![Q15(0); xq.len()];
    Fallback
        .quad_demod(&mut q, &yq, &mut dq, Q15(32767))
        .unwrap();
    let step = 2.0 * dev / f64::from(fs); // radians / pi per unit input
    for n in 1..xq.len() {
        let want = xq[n].to_f32() as f64 * step * 32768.0;
        assert!((f64::from(dq[n].0) - want).abs() < 60.0, "n = {n}");
    }
}

#[test]
fn fm_mod_is_block_size_invariant() {
    let x = random_f32(1000, 13, 0.8);
    let mut s = fm_state(50_000.0, 200_000);
    let mut whole = vec![C32::ZERO; x.len()];
    Fallback.fm_mod(&mut s, &x, &mut whole).unwrap();
    let mut s = fm_state(50_000.0, 200_000);
    let mut chunked = vec![C32::ZERO; x.len()];
    let mut pos = 0;
    for size in [1usize, 7, 64, 3, 200, 725].iter().cycle() {
        if pos >= x.len() {
            break;
        }
        let end = (pos + size).min(x.len());
        Fallback
            .fm_mod(&mut s, &x[pos..end], &mut chunked[pos..end])
            .unwrap();
        pos = end;
    }
    assert_eq!(whole, chunked);
}

// --- fused kernels ----------------------------------------------------------------------

fn decim_state_c32(h: &[f32], m: usize) -> FirDecimState<C32, Vec<f32>, Vec<C32>> {
    FirDecimState::new(h.to_vec(), vec![C32::ZERO; h.len() - 1], m).unwrap()
}

#[test]
fn max_input_for_output_matches_what_the_decimator_consumes() {
    let h = taps_f32(9, 14, 0.3);
    let x = random_c32(100, 15, 0.5);
    for m in [1usize, 2, 3, 10] {
        for warm in [0usize, 1, 2, 5, 11] {
            for out_len in [0usize, 1, 2, 3, 7] {
                let mut state = decim_state_c32(&h, m);
                let mut sink = vec![C32::ZERO; 64];
                Fallback
                    .fir_decimate(&mut state, &x[..warm], &mut sink)
                    .unwrap();
                let n = state.max_input_for_output(out_len);
                let mut out = vec![C32::ZERO; out_len];
                let p = Fallback.fir_decimate(&mut state, &x, &mut out).unwrap();
                // With plenty of input the decimator stops exactly at the predicted count.
                assert_eq!(
                    p.consumed,
                    n.min(x.len()),
                    "m {m} warm {warm} out {out_len}"
                );
                assert!(p.produced <= out_len);
            }
        }
    }
    let state = decim_state_c32(&h, 4);
    assert_eq!(state.max_input_for_output(usize::MAX), usize::MAX);
}

fn chain_ddc<K: Rotate<C32> + FirDecimate<C32>>(
    k: &K,
    nco: &mut Nco,
    fir: &mut FirDecimState<C32, Vec<f32>, Vec<C32>>,
    input: &[C32],
    output: &mut [C32],
) -> Progress {
    // The single kernels, without the fused wrapper.
    let mut rotated = vec![C32::ZERO; input.len()];
    k.rotate(nco, input, &mut rotated).unwrap();
    k.fir_decimate(fir, &rotated, output).unwrap()
}

#[test]
fn ddc_equals_the_chain_of_single_kernels() {
    let rate = Rate::hz(2_400_000).unwrap();
    let h = taps_f32(63, 16, 0.1);
    let x = random_c32(3000, 17, 0.5);
    let mut nco_a = Nco::new(-600_000.0, rate).unwrap();
    let mut nco_b = nco_a;
    let mut fir_a = decim_state_c32(&h, 10);
    let mut fir_b = decim_state_c32(&h, 10);
    let mut want = vec![C32::ZERO; 300];
    let pw = chain_ddc(&Fallback, &mut nco_a, &mut fir_a, &x, &mut want);
    let mut got = vec![C32::ZERO; 300];
    let mut scratch = vec![C32::ZERO; x.len()];
    let pg = Fallback
        .ddc(&mut nco_b, &mut fir_b, &x, &mut got, &mut scratch)
        .unwrap();
    assert_eq!(pw, pg);
    assert_eq!(want, got);
    assert_eq!(nco_a, nco_b);
    assert_eq!(fir_a.history(), fir_b.history());
    assert_eq!(fir_a.countdown(), fir_b.countdown());
}

#[test]
fn ddc_partial_output_keeps_the_oscillator_phase() {
    let rate = Rate::hz(1_000_000).unwrap();
    let h = taps_f32(21, 18, 0.2);
    let x = random_c32(2000, 19, 0.5);
    let mut nco_w = Nco::new(123_456.0, rate).unwrap();
    let mut fir_w = decim_state_c32(&h, 7);
    let mut whole = vec![C32::ZERO; 400];
    let mut scratch = vec![C32::ZERO; 2000];
    let pw = Fallback
        .ddc(&mut nco_w, &mut fir_w, &x, &mut whole, &mut scratch)
        .unwrap();
    assert_eq!(pw.consumed, 2000);
    whole.truncate(pw.produced);
    // The same through a tiny output buffer (3 outputs) and a small scratch (50 samples).
    let mut nco = Nco::new(123_456.0, rate).unwrap();
    let mut fir = decim_state_c32(&h, 7);
    let mut got = Vec::new();
    let mut pos = 0;
    let mut calls = 0;
    while pos < x.len() {
        let mut out = [C32::ZERO; 3];
        let mut small = [C32::ZERO; 50];
        let p = Fallback
            .ddc(&mut nco, &mut fir, &x[pos..], &mut out, &mut small)
            .unwrap();
        // The oscillator moved by exactly the consumed samples.
        got.extend_from_slice(&out[..p.produced]);
        pos += p.consumed;
        calls += 1;
        assert!(p.consumed > 0 || p.produced > 0, "no progress");
        assert!(calls < 10_000);
    }
    assert_eq!(got, whole);
    assert_eq!(nco, nco_w);
    // With no output space and no pending skip nothing is consumed and the phase stays.
    let mut fresh = Nco::new(1000.0, rate).unwrap();
    let mut fir = decim_state_c32(&h, 7);
    let p = Fallback
        .ddc(&mut fresh, &mut fir, &x, &mut [], &mut scratch)
        .unwrap();
    assert_eq!(p, Progress::default());
    assert_eq!(fresh.phase(), 0);
}

#[test]
fn ddc_cq15_equals_the_chain_and_is_chunk_invariant() {
    let h: Vec<Q15> = random_q15(31, 20).iter().map(|q| Q15(q.0 / 16)).collect();
    let x = random_cq15(1500, 21);
    let rate = Rate::hz(480_000).unwrap();
    let new_fir = || FirDecimState::<CQ15, _, _>::new(h.clone(), vec![CQ15::ZERO; 30], 5).unwrap();
    let mut nco = Nco::new(-60_000.0, rate).unwrap();
    let mut fir = new_fir();
    let mut rotated = vec![CQ15::ZERO; x.len()];
    Fallback.rotate(&mut nco, &x, &mut rotated).unwrap();
    let mut want = vec![CQ15::ZERO; 300];
    let pw = Fallback
        .fir_decimate(&mut fir, &rotated, &mut want)
        .unwrap();
    want.truncate(pw.produced);
    let mut nco = Nco::new(-60_000.0, rate).unwrap();
    let mut fir = new_fir();
    let (mut got, mut pos) = (Vec::new(), 0);
    for size in [13usize, 100, 1, 77, 400].iter().cycle() {
        if pos >= x.len() {
            break;
        }
        let end = (pos + size).min(x.len());
        let mut rest = &x[pos..end];
        while !rest.is_empty() {
            let mut out = [CQ15::ZERO; 8];
            let mut scratch = [CQ15::ZERO; 64];
            let p =
                ddc_chained(&Fallback, &mut nco, &mut fir, rest, &mut out, &mut scratch).unwrap();
            got.extend_from_slice(&out[..p.produced]);
            rest = &rest[p.consumed..];
            assert!(p.consumed > 0 || p.produced > 0);
        }
        pos = end;
    }
    assert_eq!(got, want);
}

#[test]
fn duc_equals_the_chain_of_single_kernels() {
    let rate = Rate::hz(2_400_000).unwrap();
    let l = 10;
    let h = taps_f32(61, 22, 0.3);
    let x = random_c32(150, 23, 0.5);
    let mut nco_a = Nco::new(600_000.0, rate).unwrap();
    let mut nco_b = nco_a;
    let mut fir_a = interp_state_c32(&h, l);
    let mut fir_b = interp_state_c32(&h, l);
    let mut up = vec![C32::ZERO; x.len() * l];
    let pi = Fallback.fir_interpolate(&mut fir_a, &x, &mut up).unwrap();
    let mut want = vec![C32::ZERO; up.len()];
    Fallback.rotate(&mut nco_a, &up, &mut want).unwrap();
    let mut got = vec![C32::ZERO; x.len() * l];
    let mut scratch = vec![C32::ZERO; x.len() * l];
    let pg = Fallback
        .duc(&mut fir_b, &mut nco_b, &x, &mut got, &mut scratch)
        .unwrap();
    assert_eq!(pi, pg);
    assert_eq!(want, got);
    assert_eq!(nco_a, nco_b);
    assert_eq!(fir_a.history(), fir_b.history());
}

#[test]
fn duc_limited_output_and_scratch_still_match() {
    let rate = Rate::hz(100_000).unwrap();
    let l = 4;
    let h = taps_f32(17, 24, 0.4);
    let x = random_c32(97, 25, 0.5);
    let mut nco = Nco::new(12_345.0, rate).unwrap();
    let mut fir = interp_state_c32(&h, l);
    let mut whole = vec![C32::ZERO; x.len() * l];
    let mut scratch = vec![C32::ZERO; x.len() * l];
    Fallback
        .duc(&mut fir, &mut nco, &x, &mut whole, &mut scratch)
        .unwrap();
    let mut nco2 = Nco::new(12_345.0, rate).unwrap();
    let mut fir2 = interp_state_c32(&h, l);
    let (mut got, mut pos) = (Vec::new(), 0);
    while pos < x.len() {
        let mut out = [C32::ZERO; 10]; // two whole input samples
        let mut small = [C32::ZERO; 9]; // two whole input samples as well
        let p = duc_chained(
            &Fallback,
            &mut fir2,
            &mut nco2,
            &x[pos..],
            &mut out,
            &mut small,
        )
        .unwrap();
        assert!(p.consumed <= 2 && p.produced == p.consumed * l);
        got.extend_from_slice(&out[..p.produced]);
        pos += p.consumed;
        assert!(p.consumed > 0);
    }
    assert_eq!(got, whole);
    assert_eq!(nco, nco2);
    // An output smaller than L takes no input at all.
    let p = Fallback
        .duc(&mut fir2, &mut nco2, &x, &mut [C32::ZERO; 3], &mut scratch)
        .unwrap();
    assert_eq!(p, Progress::default());
}

#[test]
fn fm_demod_equals_the_chain_and_ignores_the_scratch_size() {
    let fs = Rate::hz(240_000).unwrap();
    let x = random_c32(1000, 26, 0.5);
    let run_single = || {
        let mut q = QuadDemodState::<C32>::new();
        let mut d = DeemphasisState::<f32>::new(50e-6, fs).unwrap();
        let mut tmp = vec![0f32; x.len()];
        Fallback.quad_demod(&mut q, &x, &mut tmp, 2.5).unwrap();
        let mut out = vec![0f32; x.len()];
        Fallback.deemphasis(&mut d, &tmp, &mut out).unwrap();
        (out, q, d)
    };
    let (want, q_want, d_want) = run_single();
    for scratch_len in [1usize, 7, 128, 1000, 5000] {
        let mut q = QuadDemodState::<C32>::new();
        let mut d = DeemphasisState::<f32>::new(50e-6, fs).unwrap();
        let mut out = vec![0f32; x.len()];
        let mut scratch = vec![0f32; scratch_len];
        Fallback
            .fm_demod(&mut q, &mut d, &x, &mut out, &mut scratch, 2.5)
            .unwrap();
        assert_eq!(out, want, "scratch {scratch_len}");
        assert_eq!(q, q_want);
        assert_eq!(d, d_want);
    }
    // Errors.
    let mut q = QuadDemodState::<C32>::new();
    let mut d = DeemphasisState::<f32>::new(50e-6, fs).unwrap();
    let mut out = vec![0f32; 3];
    assert_eq!(
        fm_demod_chained(&Fallback, &mut q, &mut d, &x, &mut out, &mut [0.0; 4], 1.0),
        Err(Error::LengthMismatch {
            expected: 1000,
            got: 3
        })
    );
    assert!(matches!(
        fm_demod_chained(&Fallback, &mut q, &mut d, &x[..3], &mut out, &mut [], 1.0),
        Err(Error::BufferTooSmall { .. })
    ));
    assert_eq!(
        fm_demod_chained(&Fallback, &mut q, &mut d, &[], &mut [], &mut [], 1.0),
        Ok(())
    );
    // Q15.
    let xq = random_cq15(500, 27);
    let mut q1 = QuadDemodState::<CQ15>::new();
    let mut d1 = DeemphasisState::<Q15>::new(50e-6, fs).unwrap();
    let mut tmp = vec![Q15(0); xq.len()];
    Fallback
        .quad_demod(&mut q1, &xq, &mut tmp, Q15(32767))
        .unwrap();
    let mut want = vec![Q15(0); xq.len()];
    Fallback.deemphasis(&mut d1, &tmp, &mut want).unwrap();
    let mut q2 = QuadDemodState::<CQ15>::new();
    let mut d2 = DeemphasisState::<Q15>::new(50e-6, fs).unwrap();
    let mut got = vec![Q15(0); xq.len()];
    Fallback
        .fm_demod(
            &mut q2,
            &mut d2,
            &xq,
            &mut got,
            &mut vec![Q15(0); 33],
            Q15(32767),
        )
        .unwrap();
    assert_eq!(got, want);
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    fn bits(v: &[C32]) -> Vec<u32> {
        v.iter()
            .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
            .collect()
    }

    #[test]
    fn convert_out_agrees_bit_for_bit() {
        for (i, f) in all_formats().iter().enumerate() {
            let n = 64;
            if f.layout() == Layout::Real {
                let mut x = random_f32(n, 500 + i as u64, 1.3);
                x[0] = f32::NAN;
                x[1] = f32::INFINITY;
                x[2] = f32::NEG_INFINITY;
                x[3] = -0.0;
                x[4] = 1.0;
                x[5] = -1.0;
                let len = f.bytes_for_samples(n).unwrap();
                let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                assert_eq!(
                    Fallback.convert_out_real(f, &x, &mut a),
                    Reference.convert_out_real(f, &x, &mut b)
                );
                assert_eq!(a, b, "f32 real, format {i}");
                let q = random_q15(n, 600 + i as u64);
                assert_eq!(
                    Fallback.convert_out_real(f, &q, &mut a),
                    Reference.convert_out_real(f, &q, &mut b)
                );
                assert_eq!(a, b, "q15 real, format {i}");
            } else {
                let mut x = random_c32(n, 700 + i as u64, 1.3);
                x[0] = C32::new(f32::NAN, f32::INFINITY);
                x[1] = C32::new(f32::NEG_INFINITY, -0.0);
                x[2] = C32::new(1.0, -1.0);
                x[3] = C32::new(0.5 / 128.0, -0.5 / 128.0);
                let len = f.bytes_for_samples(n).unwrap();
                let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                if f.container() == Container::F32 {
                    // The reference canonicalises nothing either; NaN payloads are copied.
                }
                assert_eq!(
                    Fallback.convert_out(f, &x, &mut a),
                    Reference.convert_out(f, &x, &mut b)
                );
                assert_eq!(a, b, "c32, format {i}");
                let q = random_cq15(n, 800 + i as u64);
                assert_eq!(
                    Fallback.convert_out(f, &q, &mut a),
                    Reference.convert_out(f, &q, &mut b)
                );
                assert_eq!(a, b, "cq15, format {i}");
            }
        }
    }

    #[test]
    fn convert_out_is_exhaustive_for_q15_and_8_bit_formats() {
        // Every Q15 value through the 8-bit and 12-bit encoders and the f32 path.
        let all: Vec<Q15> = (i16::MIN..=i16::MAX).map(Q15).collect();
        for f in [
            fmt(Container::U8, 8, Layout::Real, Align::Lsb, Endian::Little),
            fmt(Container::I8, 8, Layout::Real, Align::Lsb, Endian::Little),
            fmt(Container::I16, 12, Layout::Real, Align::Lsb, Endian::Little),
            fmt(Container::I16, 16, Layout::Real, Align::Msb, Endian::Big),
            fmt(Container::I32, 24, Layout::Real, Align::Msb, Endian::Little),
            fmt(Container::U8, 8, Layout::Real, Align::Lsb, Endian::Little)
                .with_full_scale(FullScale::Explicit(77))
                .unwrap(),
            fmt(Container::I16, 12, Layout::Real, Align::Lsb, Endian::Little)
                .with_full_scale(FullScale::Explicit(2047))
                .unwrap(),
        ] {
            let len = f.bytes_for_samples(all.len()).unwrap();
            let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
            Fallback.convert_out_real(&f, &all, &mut a).unwrap();
            Reference.convert_out_real(&f, &all, &mut b).unwrap();
            assert_eq!(a, b, "{f:?}");
        }
        // Every f32 amplitude on a fine grid around the rounding boundaries of CS8.
        let grid: Vec<f32> = (-300..=300).map(|k| k as f32 / 256.0).collect();
        let f = formats::RS16;
        let (mut a, mut b) = (vec![0u8; grid.len() * 2], vec![0u8; grid.len() * 2]);
        Fallback.convert_out_real(&f, &grid, &mut a).unwrap();
        Reference.convert_out_real(&f, &grid, &mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn convert_out_errors_agree() {
        let x = [C32::ZERO; 3];
        let mut out = [0u8; 8];
        for f in [formats::CS16, formats::PACKED10, formats::RS16] {
            assert_eq!(
                Fallback.convert_out(&f, &x, &mut out),
                Reference.convert_out(&f, &x, &mut out)
            );
        }
    }

    #[test]
    fn fir_interpolate_matches_the_reference() {
        let h = taps_f32(241, 30, 0.04);
        let x = random_c32(400, 31, 0.5);
        let l = 5;
        let (mut a, mut b) = (interp_state_c32(&h, l), interp_state_c32(&h, l));
        let (mut ya, mut yb) = (vec![C32::ZERO; 2000], vec![C32::ZERO; 2000]);
        Fallback.fir_interpolate(&mut a, &x, &mut ya).unwrap();
        Reference.fir_interpolate(&mut b, &x, &mut yb).unwrap();
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for (p, q) in ya.iter().zip(&yb) {
            sig += f64::from(q.re).powi(2) + f64::from(q.im).powi(2);
            err += f64::from(p.re - q.re).powi(2) + f64::from(p.im - q.im).powi(2);
        }
        assert!(10.0 * (sig / err).log10() >= 110.0);
        // CQ15 is bit-exact.
        let hq: Vec<Q15> = random_q15(241, 32).iter().map(|q| Q15(q.0 / 64)).collect();
        let xq = random_cq15(300, 33);
        let new =
            || FirInterpState::<CQ15, _, _>::new(hq.clone(), vec![CQ15::ZERO; 49], 5).unwrap();
        let (mut a, mut b) = (new(), new());
        let (mut ya, mut yb) = (vec![CQ15::ZERO; 1500], vec![CQ15::ZERO; 1500]);
        Fallback.fir_interpolate(&mut a, &xq, &mut ya).unwrap();
        Reference.fir_interpolate(&mut b, &xq, &mut yb).unwrap();
        assert_eq!(ya, yb);
        // Real variants.
        let hr = taps_f32(61, 34, 0.1);
        let xr = random_f32(200, 35, 0.5);
        let new = || FirInterpRealState::<f32, _, _>::new(hr.clone(), vec![0f32; 13], 5).unwrap();
        let (mut a, mut b) = (new(), new());
        let (mut ya, mut yb) = (vec![0f32; 1000], vec![0f32; 1000]);
        Fallback.fir_interpolate_real(&mut a, &xr, &mut ya).unwrap();
        Reference
            .fir_interpolate_real(&mut b, &xr, &mut yb)
            .unwrap();
        let sig: f64 = yb.iter().map(|v| f64::from(*v).powi(2)).sum();
        let err: f64 = ya
            .iter()
            .zip(&yb)
            .map(|(p, q)| f64::from(p - q).powi(2))
            .sum();
        assert!(10.0 * (sig / err).log10() >= 110.0);
    }

    #[test]
    fn fm_mod_matches_the_reference() {
        let x = random_f32(8192, 36, 0.3);
        let (mut a, mut b) = (fm_state(75_000.0, 240_000), fm_state(75_000.0, 240_000));
        let (mut ya, mut yb) = (vec![C32::ZERO; x.len()], vec![C32::ZERO; x.len()]);
        Fallback.fm_mod(&mut a, &x, &mut ya).unwrap();
        Reference.fm_mod(&mut b, &x, &mut yb).unwrap();
        assert_eq!(a.phase(), b.phase());
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for (p, q) in ya.iter().zip(&yb) {
            sig += f64::from(q.re).powi(2) + f64::from(q.im).powi(2);
            err += f64::from(p.re - q.re).powi(2) + f64::from(p.im - q.im).powi(2);
        }
        assert!(
            10.0 * (sig / err).log10() >= 120.0,
            "{}",
            10.0 * (sig / err).log10()
        );
        let xq = random_q15(8192, 37);
        let (mut a, mut b) = (fm_state(75_000.0, 240_000), fm_state(75_000.0, 240_000));
        let (mut ya, mut yb) = (vec![CQ15::ZERO; xq.len()], vec![CQ15::ZERO; xq.len()]);
        Fallback.fm_mod(&mut a, &xq, &mut ya).unwrap();
        Reference.fm_mod(&mut b, &xq, &mut yb).unwrap();
        assert_eq!(a.phase(), b.phase());
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for (p, q) in ya.iter().zip(&yb) {
            let (pr, pi, qr, qi) = (
                f64::from(p.re.0),
                f64::from(p.im.0),
                f64::from(q.re.0),
                f64::from(q.im.0),
            );
            sig += qr * qr + qi * qi;
            err += (pr - qr).powi(2) + (pi - qi).powi(2);
        }
        assert!(
            10.0 * (sig / err).log10() >= 80.0,
            "{}",
            10.0 * (sig / err).log10()
        );
    }

    #[test]
    fn fused_kernels_of_the_reference_equal_their_chains() {
        let rate = Rate::hz(1_000_000).unwrap();
        let h = taps_f32(31, 38, 0.1);
        let x = random_c32(1000, 39, 0.5);
        let mut nco = Nco::new(-200_000.0, rate).unwrap();
        let mut nco2 = nco;
        let mut fir = decim_state_c32(&h, 5);
        let mut fir2 = decim_state_c32(&h, 5);
        let (mut a, mut b) = (vec![C32::ZERO; 200], vec![C32::ZERO; 200]);
        let mut scratch = vec![C32::ZERO; 1000];
        let pa = Reference
            .ddc(&mut nco, &mut fir, &x, &mut a, &mut scratch)
            .unwrap();
        let pb = ddc_chained(&Reference, &mut nco2, &mut fir2, &x, &mut b, &mut scratch).unwrap();
        assert_eq!(pa, pb);
        assert_eq!(bits(&a), bits(&b));
        let mut fir_i = interp_state_c32(&h, 3);
        let mut fir_j = interp_state_c32(&h, 3);
        let (mut a, mut b) = (vec![C32::ZERO; 3000], vec![C32::ZERO; 3000]);
        let mut scratch = vec![C32::ZERO; 3000];
        let pa = Reference
            .duc(&mut fir_i, &mut nco, &x, &mut a, &mut scratch)
            .unwrap();
        let pb = duc_chained(&Reference, &mut fir_j, &mut nco2, &x, &mut b, &mut scratch).unwrap();
        assert_eq!(pa, pb);
        assert_eq!(bits(&a), bits(&b));
        let mut q = QuadDemodState::<C32>::new();
        let mut d = DeemphasisState::<f32>::new(50e-6, rate).unwrap();
        let (mut q2, mut d2) = (q, d);
        let (mut a, mut b) = (vec![0f32; 1000], vec![0f32; 1000]);
        let mut sc = vec![0f32; 64];
        Reference
            .fm_demod(&mut q, &mut d, &x, &mut a, &mut sc, 1.0)
            .unwrap();
        fm_demod_chained(&Reference, &mut q2, &mut d2, &x, &mut b, &mut sc, 1.0).unwrap();
        assert_eq!(a, b);
    }
}
