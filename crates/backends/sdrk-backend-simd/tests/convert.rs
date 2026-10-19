//! `convert` and `convert_out` against the fallback: bit for bit.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod common;

use common::{LENGTHS, Rng, bits};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::convert::{Convert, ConvertOut};
use sdrk_core::{C32, CQ15, Fallback, Q15};
use sdrk_device::{SampleFormat, formats};

const FAST: [SampleFormat; 3] = [formats::CU8, formats::CS8, formats::CS16];

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
        formats::CS8.with_swap_iq(true),
    ]
}

fn bytes_for(format: &SampleFormat, samples: usize, rng: &mut Rng) -> Vec<u8> {
    let n = format.bytes_for_samples(samples).unwrap();
    (0..n).map(|_| rng.next_u32() as u8).collect()
}

#[test]
fn convert_matches_the_fallback_for_every_format_and_length() {
    let mut rng = Rng::new(1);
    for format in all_formats() {
        for &len in &LENGTHS {
            // Whole packing groups only.
            let samples = len;
            let Ok(_) = format.bytes_for_samples(samples) else {
                continue;
            };
            let input = bytes_for(&format, samples, &mut rng);
            let mut want = vec![C32::default(); samples + 3];
            let mut got = vec![C32::default(); samples + 3];
            let rw = Fallback.convert(&format, &input, &mut want);
            let rg = Simd.convert(&format, &input, &mut got);
            assert_eq!(rw, rg, "{format:?} {samples}");
            assert_eq!(bits(&want), bits(&got), "{format:?} {samples}");
            let mut want = vec![CQ15::default(); samples + 3];
            let mut got = vec![CQ15::default(); samples + 3];
            let rw = Fallback.convert(&format, &input, &mut want);
            let rg = Simd.convert(&format, &input, &mut got);
            assert_eq!(rw, rg, "{format:?} {samples}");
            assert_eq!(want, got, "{format:?} {samples}");
        }
    }
}

#[test]
fn convert_covers_every_byte_value() {
    for format in FAST {
        let width = format.bytes_for_samples(1).unwrap();
        let input: Vec<u8> = (0..4096 * width).map(|i| (i * 7 + i / 256) as u8).collect();
        let n = format.samples_in_bytes(input.len());
        let mut want = vec![C32::default(); n];
        let mut got = vec![C32::default(); n];
        Fallback.convert(&format, &input, &mut want).unwrap();
        Simd.convert(&format, &input, &mut got).unwrap();
        assert_eq!(bits(&want), bits(&got));
        let mut want = vec![CQ15::default(); n];
        let mut got = vec![CQ15::default(); n];
        Fallback.convert(&format, &input, &mut want).unwrap();
        Simd.convert(&format, &input, &mut got).unwrap();
        assert_eq!(want, got);
    }
}

#[test]
fn convert_reports_the_same_errors() {
    for format in all_formats() {
        let mut out = [C32::default(); 4];
        // A partial sample, an output that is too small, and a real layout.
        for len in [1usize, 3, 5, 6, 7, 9] {
            let input = vec![0u8; len];
            assert_eq!(
                Fallback.convert(&format, &input, &mut out),
                Simd.convert(&format, &input, &mut out),
                "{format:?} {len}"
            );
        }
        let input = vec![0u8; 64];
        let mut tiny = [C32::default(); 1];
        assert_eq!(
            Fallback.convert(&format, &input, &mut tiny),
            Simd.convert(&format, &input, &mut tiny)
        );
        let mut tiny = [CQ15::default(); 1];
        assert_eq!(
            Fallback.convert(&format, &input, &mut tiny),
            Simd.convert(&format, &input, &mut tiny)
        );
    }
}

fn float_values(rng: &mut Rng, n: usize) -> Vec<C32> {
    let specials = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.0 - f32::EPSILON,
        1.000_01,
        -1.000_01,
        1.0e-30,
        -1.0e-30,
        1.0e-45,
        0.003_906_25,
        0.003_906_26,
        -0.003_906_25,
        127.0 / 128.0,
        127.5 / 128.0,
        -127.5 / 128.0,
        32767.5 / 32768.0,
        -32768.5 / 32768.0,
        3.4e38,
        -3.4e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
    ];
    (0..n)
        .map(|i| {
            let pick = |r: &mut Rng| match r.below(4) {
                0 => specials[r.below(specials.len())],
                1 => r.unit() * 1.5,
                2 => (r.below(65536) as f32 - 32768.0 + 0.5) / 32768.0,
                _ => (r.below(512) as f32 - 256.0 + 0.5) / 128.0,
            };
            C32::new(
                pick(rng),
                if i % 5 == 0 {
                    specials[i % specials.len()]
                } else {
                    pick(rng)
                },
            )
        })
        .collect()
}

#[test]
fn convert_out_matches_the_fallback_for_every_format_and_length() {
    let mut rng = Rng::new(2);
    for format in all_formats() {
        for &len in &LENGTHS {
            let Ok(nbytes) = format.bytes_for_samples(len) else {
                continue;
            };
            let input = float_values(&mut rng, len);
            let mut want = vec![0xAAu8; nbytes + 5];
            let mut got = vec![0xAAu8; nbytes + 5];
            let rw = Fallback.convert_out(&format, &input, &mut want);
            let rg = Simd.convert_out(&format, &input, &mut got);
            assert_eq!(rw, rg, "{format:?} {len}");
            assert_eq!(want, got, "{format:?} {len}");
            let q: Vec<CQ15> = common::cq15_noise(&mut rng, len);
            let mut want = vec![0xAAu8; nbytes + 5];
            let mut got = vec![0xAAu8; nbytes + 5];
            let rw = Fallback.convert_out(&format, &q, &mut want);
            let rg = Simd.convert_out(&format, &q, &mut got);
            assert_eq!(rw, rg, "{format:?} {len}");
            assert_eq!(want, got, "{format:?} {len}");
        }
    }
}

#[test]
fn convert_out_q15_extremes() {
    for format in FAST {
        let vals: Vec<i16> = (-32768i32..=32767).map(|v| v as i16).collect();
        let input: Vec<CQ15> = vals
            .chunks_exact(2)
            .map(|c| CQ15 {
                re: Q15(c[0]),
                im: Q15(c[1]),
            })
            .collect();
        let n = format.bytes_for_samples(input.len()).unwrap();
        let mut want = vec![0u8; n];
        let mut got = vec![0u8; n];
        Fallback.convert_out(&format, &input, &mut want).unwrap();
        Simd.convert_out(&format, &input, &mut got).unwrap();
        assert_eq!(want, got, "{format:?}");
    }
}

#[test]
fn convert_out_float_grid_around_every_rounding_boundary() {
    // Every half-integer neighbourhood of the 16-bit and 8-bit grids.
    for format in FAST {
        let scale = format.full_scale_value() as f32;
        let mut input = Vec::new();
        let steps = if scale > 1000.0 { 70_000 } else { 600 };
        for k in 0..steps {
            let base = (k as f32 - steps as f32 / 2.0 + 0.5) / scale;
            for d in [0, 1, -1] {
                let bits = base.to_bits().wrapping_add_signed(d);
                input.push(C32::new(
                    f32::from_bits(bits),
                    f32::from_bits(bits ^ 0x8000_0000),
                ));
            }
        }
        let n = format.bytes_for_samples(input.len()).unwrap();
        let mut want = vec![0u8; n];
        let mut got = vec![0u8; n];
        Fallback.convert_out(&format, &input, &mut want).unwrap();
        Simd.convert_out(&format, &input, &mut got).unwrap();
        if let Some(i) = want.iter().zip(&got).position(|(a, b)| a != b) {
            let w = format.bytes_for_samples(1).unwrap();
            let x = input[i / w];
            panic!(
                "{format:?} first mismatch at byte {i}: want {} got {} input {x:?} bits {:x} {:x}",
                want[i],
                got[i],
                x.re.to_bits(),
                x.im.to_bits()
            );
        }
    }
}
