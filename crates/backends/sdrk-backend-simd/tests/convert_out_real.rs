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
