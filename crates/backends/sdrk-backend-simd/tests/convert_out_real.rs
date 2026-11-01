//! `convert_out_real` of `f32` and `Q15` against the fallback, bit for bit: the real formats
//! with a vector path (`RS16`, real `I8` and `U8`) and some without one, float bit patterns of
//! every class, every length and offset around the vector blocks, and the identity with the
//! complex encoding of the same values.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation
)]

mod common;

use common::Rng;
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::convert::{ConvertOut, ConvertOutReal};
use sdrk_core::{C32, CQ15, Fallback, Q15};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat, formats};

fn real(
    container: Container,
    bits: u8,
    signed: bool,
    align: Align,
    endian: Endian,
) -> SampleFormat {
    SampleFormat::new(container, bits, signed, Layout::Real, align, endian).unwrap()
}

/// The real formats with a vector path.
fn fast() -> [SampleFormat; 3] {
    [
        formats::RS16,
        real(Container::I8, 8, true, Align::Lsb, Endian::Little),
        real(Container::U8, 8, false, Align::Lsb, Endian::Little),
    ]
}

/// Real formats that go to the fallback.
fn slow() -> [SampleFormat; 4] {
    [
        real(Container::I16, 16, true, Align::Lsb, Endian::Big),
        real(Container::I16, 12, true, Align::Msb, Endian::Little),
        real(Container::I16, 12, true, Align::Lsb, Endian::Little)
            .with_full_scale(FullScale::Explicit(2047))
            .unwrap(),
        formats::RF32,
    ]
}

/// Both encodings of `input`, with a guard byte after the output that must stay untouched.
fn encode<T>(format: &SampleFormat, input: &[T], skip_out: usize) -> (Vec<u8>, Vec<u8>)
where
    Fallback: ConvertOutReal<T>,
    Simd: ConvertOutReal<T>,
    T: sdrk_core::RealSample,
{
    let n = format.bytes_for_samples(input.len()).unwrap();
    let mut want = vec![0x5Au8; skip_out + n + 1];
    let mut got = vec![0x5Au8; skip_out + n + 1];
    assert_eq!(
        Fallback.convert_out_real(format, input, &mut want[skip_out..]),
        Ok(n)
    );
    assert_eq!(
        Simd.convert_out_real(format, input, &mut got[skip_out..]),
        Ok(n)
    );
    assert_eq!(got[skip_out + n], 0x5A, "{format:?}: byte after the output");
    (want, got)
}

#[test]
fn convert_out_real_strided_sweep_of_every_float_class() {
    // Every 4099th bit pattern (all exponents, both signs, NaN payloads, infinities,
    // subnormals) at several offsets of the vector loop.
    let input: Vec<f32> = (0..=u32::MAX / 4099)
        .map(|k| f32::from_bits(k.wrapping_mul(4099)))
        .collect();
    for format in fast().into_iter().chain(slow()) {
        for skip in [0usize, 1, 3] {
            let (want, got) = encode(&format, &input[skip..], 0);
            assert!(want == got, "{format:?} skip {skip}");
        }
    }
}

#[test]
fn convert_out_real_every_length_and_offset_around_the_vector_blocks() {
    let mut rng = Rng::new(11);
    let pool: Vec<f32> = (0..300)
        .map(|_| match rng.below(3) {
            0 => rng.unit() * 1.2,
            1 => (rng.below(512) as f32 - 256.0) / 256.0,
            _ => f32::from_bits(rng.next_u32()),
        })
        .collect();
    let q: Vec<Q15> = (0..300).map(|_| Q15(rng.next_u32() as i16)).collect();
    for format in fast().into_iter().chain(slow()) {
        for len in 0..=140 {
            for skip in [0usize, 1, 2, 5] {
                let (want, got) = encode(&format, &pool[skip..skip + len], skip % 2);
                assert!(want == got, "{format:?} f32 len {len} skip {skip}");
                let (want, got) = encode(&format, &q[skip..skip + len], skip % 2);
                assert!(want == got, "{format:?} q15 len {len} skip {skip}");
            }
        }
    }
}

#[test]
fn convert_out_real_equals_the_complex_encoding_of_the_same_values() {
    let mut rng = Rng::new(12);
    let floats: Vec<f32> = (0..202).map(|_| rng.unit() * 1.1).collect();
    let q15: Vec<Q15> = (0..202).map(|_| Q15(rng.next_u32() as i16)).collect();
    let c32: Vec<C32> = floats
        .chunks_exact(2)
        .map(|p| C32::new(p[0], p[1]))
        .collect();
    let cq15: Vec<CQ15> = q15
        .chunks_exact(2)
        .map(|p| CQ15 { re: p[0], im: p[1] })
        .collect();
    for (real, complex) in fast()
        .into_iter()
        .zip([formats::CS16, formats::CS8, formats::CU8])
    {
        let n = real.bytes_for_samples(floats.len()).unwrap();
        let (mut got, mut want) = (vec![0u8; n], vec![0u8; n]);
        Simd.convert_out_real(&real, &floats, &mut got).unwrap();
        Simd.convert_out(&complex, &c32, &mut want).unwrap();
        assert_eq!(got, want, "{real:?} f32");
        Simd.convert_out_real(&real, &q15, &mut got).unwrap();
        Simd.convert_out(&complex, &cq15, &mut want).unwrap();
        assert_eq!(got, want, "{real:?} q15");
    }
}

#[test]
fn convert_out_real_extremes_and_ties() {
    let lsb = 1.0 / 32768.0;
    let mut x = vec![
        0.0f32,
        -0.0,
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        -1.0,
        1.0 - lsb / 2.0,
        -1.0 - lsb,
        65536.0,
        -65536.0,
        f32::MAX,
        f32::MIN,
        1.0e-30,
        -1.0e-30,
    ];
    x.extend((-40..40).map(|k| (k as f32 + 0.5) * lsb));
    x.extend((-40..40).map(|k| (k as f32 + 0.5) / 128.0));
    x.extend((-40..40).map(|k| k as f32 / 128.0));
    for format in fast() {
        let (want, got) = encode(&format, &x, 0);
        assert_eq!(want, got, "{format:?}");
    }
    let rs16 = |v: &[u8], k: usize| i16::from_le_bytes([v[2 * k], v[2 * k + 1]]);
    let (_, got) = encode(&formats::RS16, &x, 0);
    let want: [i16; 14] = [
        0, 0, 0, 0, 32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768,
    ];
    for (k, w) in want.iter().enumerate() {
        assert_eq!(rs16(&got, k), *w, "value {k}");
    }
    let q: Vec<Q15> = [
        i16::MIN,
        -32767,
        -129,
        -128,
        -1,
        0,
        1,
        127,
        128,
        383,
        384,
        i16::MAX,
    ]
    .into_iter()
    .map(Q15)
    .collect();
    for format in fast() {
        let (want, got) = encode(&format, &q, 0);
        assert_eq!(want, got, "{format:?} q15");
    }
}

/// The values where the forms of the vector paths could differ from the formula: the bound of
/// `CU8` at `x = -2^-54` and the negative values above `-2^-25 / 128`, the ties and their
/// neighbours of every scale, the overflow of the conversion at `2^31`, NaN payloads,
/// infinities, signed zeros and subnormals; each with its neighbours one ulp up and down.
fn boundary_values() -> Vec<f32> {
    let mut base = vec![
        0.0f32,
        f32::from_bits(1),
        1.0e-40,
        2.0f32.powi(-54),
        2.0f32.powi(-47),
        2.0f32.powi(-32),
        2.0f32.powi(-25),
        2.0f32.powi(-24),
        0.5,
        1.0,
        127.5 / 128.0,
        2.0f32.powi(23),
        2.0f32.powi(24),
        2.0f32.powi(31) / 32768.0,
        1.0e30,
        f32::MAX,
        f32::INFINITY,
    ];
    for scale in [128.0f32, 32768.0] {
        for k in -132..=132 {
            base.push(k as f32 / scale);
            base.push((k as f32 + 0.5) / scale);
        }
    }
    let mut x = Vec::new();
    for v in base {
        for w in [v, -v] {
            x.push(w);
            if w.is_finite() {
                x.push(w.next_up());
                x.push(w.next_down());
            }
        }
    }
    x.extend(
        [
            0x7FC0_0000u32,
            0xFFC0_0000,
            0x7F80_0001,
            0xFFFF_FFFF,
            0x7FBF_FFFF,
        ]
        .map(f32::from_bits),
    );
    x
}

#[test]
fn convert_out_boundary_values_in_every_lane_of_the_vector_blocks() {
    // The 128-bit paths take a different form for each of the four vectors of a block of 16
    // values, the wider ones blocks of 32 and 64: every value goes through every position of a
    // block of 64 (and through the tails), as real values and as both parts of a complex one.
    let x = boundary_values();
    let n = x.len();
    for offset in 0..64 {
        let input: Vec<f32> = (0..n + 37).map(|k| x[(k + offset) % n]).collect();
        let c32: Vec<C32> = input
            .chunks_exact(2)
            .map(|p| C32::new(p[0], p[1]))
            .collect();
        for (real, complex) in fast()
            .into_iter()
            .zip([formats::CS16, formats::CS8, formats::CU8])
        {
            let (want, got) = encode(&real, &input, 0);
            assert!(want == got, "{real:?} offset {offset}");
            let m = complex.bytes_for_samples(c32.len()).unwrap();
            let (mut want, mut got) = (vec![0u8; m], vec![0u8; m]);
            Fallback.convert_out(&complex, &c32, &mut want).unwrap();
            Simd.convert_out(&complex, &c32, &mut got).unwrap();
            assert!(want == got, "{complex:?} offset {offset}");
        }
    }
}
