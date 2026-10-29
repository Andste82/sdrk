//! `convert` of `CS16`, `CS8` and `CU8` to `C32` and `CQ15`, and `convert_real` of the real
//! formats with a vector path to `f32` and Q15, against the fallback, bit for bit: every input
//! value in every lane position, at every start alignment, with every number of values after
//! the last whole block of the vector paths, no `-0.0` and nothing written beyond the converted
//! samples. Runs at the level of the build (and, with run-time detection,
//! at the level of the CPU), so the test matrix covers every level-specific path.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, bits};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::convert::{Convert, ConvertReal};
use sdrk_core::{C32, CQ15, Fallback, Q15};
use sdrk_device::{Align, Container, Endian, Layout, SampleFormat, formats};

/// Bytes before the converted input, so that its start takes every alignment that matters.
const STARTS: usize = 8;
/// A value the kernels never produce, written beyond the converted samples.
const SENTINEL: f32 = 1234.5;

/// Converts `input` with both implementations and compares results, bits and the untouched
/// output beyond the samples. Also checks that no `-0.0` is produced.
fn check(format: &SampleFormat, input: &[u8], what: &str) {
    let samples = format.samples_in_bytes(input.len());
    let mut want = vec![C32::new(SENTINEL, SENTINEL); samples + 3];
    let mut got = want.clone();
    let rw = Fallback.convert(format, input, &mut want);
    let rg = Simd.convert(format, input, &mut got);
    assert_eq!(rw, rg, "{format:?} {what}");
    assert_eq!(rg, Ok(samples), "{format:?} {what}");
    // The first differing sample, for a readable failure.
    let (w, g) = (bits(&want), bits(&got));
    if let Some(first) = w.iter().zip(&g).position(|(a, b)| a != b) {
        assert_eq!(
            (want[first].re, want[first].im),
            (got[first].re, got[first].im),
            "{format:?} {what}: sample {first} of {samples}"
        );
        assert_eq!(
            w[first], g[first],
            "{format:?} {what}: sample {first} of {samples}"
        );
    }
    for c in &got[..samples] {
        assert!(
            c.re.to_bits() != (-0.0f32).to_bits() && c.im.to_bits() != (-0.0f32).to_bits(),
            "{format:?} {what}: -0.0"
        );
    }
    for c in &got[samples..] {
        assert_eq!(
            (c.re, c.im),
            (SENTINEL, SENTINEL),
            "{format:?} {what}: overrun"
        );
    }
}

/// Lanes of the widest vector of the paths (AVX-512); the positions of a value in its vector and
/// in its load repeat with this period.
const LANES: usize = 16;

/// Every one of the `values` (a multiple of `LANES`) patterns of `width` bytes, `LANES` times
/// rotated by one, so that each value takes every position in a vector.
fn all_values(width: usize, values: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * values * LANES);
    for round in 0..LANES {
        for i in 0..values {
            out.extend_from_slice(&((i + round) % values).to_le_bytes()[..width]);
        }
    }
    out
}

/// The exhaustive input (whole samples) at every start alignment.
fn every_value_at_every_start(format: &SampleFormat, width: usize, values: usize) {
    let data = all_values(width, values);
    let mut buffer = vec![0u8; STARTS + data.len()];
    for start in 0..STARTS {
        buffer[start..start + data.len()].copy_from_slice(&data);
        check(
            format,
            &buffer[start..start + data.len()],
            &format!("start {start}"),
        );
    }
}

/// Every length from 0 to beyond the largest block of the vector paths, at every start.
fn every_length_at_every_start(format: &SampleFormat, width: usize) {
    // The largest block of the paths holds 256 values (128 samples): several blocks, each
    // number of units after them and each number of values after the last unit.
    let max_samples = 4 * 128 + 40;
    let mut rng = Rng::new(5);
    let buffer: Vec<u8> = (0..STARTS + 2 * width * max_samples)
        .map(|_| rng.next_u32() as u8)
        .collect();
    for start in 0..STARTS {
        for samples in 0..=max_samples {
            let len = 2 * width * samples;
            check(
                format,
                &buffer[start..start + len],
                &format!("start {start}, {samples} samples"),
            );
        }
    }
}

#[test]
fn cs16_every_value_bit_exact() {
    every_value_at_every_start(&formats::CS16, 2, 1 << 16);
}

#[test]
fn cs8_every_value_bit_exact() {
    every_value_at_every_start(&formats::CS8, 1, 256);
}

#[test]
fn cu8_every_value_bit_exact() {
    every_value_at_every_start(&formats::CU8, 1, 256);
}

#[test]
fn cs16_every_length_bit_exact() {
    every_length_at_every_start(&formats::CS16, 2);
}

#[test]
fn cs8_every_length_bit_exact() {
    every_length_at_every_start(&formats::CS8, 1);
}

#[test]
fn cu8_every_length_bit_exact() {
    every_length_at_every_start(&formats::CU8, 1);
}

#[test]
fn the_8_bit_formats_with_a_big_endian_tag_take_the_same_path() {
    // One byte per value has no byte order: the tag must not change the result.
    let mut rng = Rng::new(9);
    let input: Vec<u8> = (0..2 * 300).map(|_| rng.next_u32() as u8).collect();
    for format in [formats::CS8, formats::CU8] {
        let big = SampleFormat::new(
            format.container(),
            format.bits(),
            format.signed(),
            format.layout(),
            format.align(),
            sdrk_device::Endian::Big,
        )
        .unwrap();
        check(&big, &input, "big-endian tag");
        let mut a = vec![C32::default(); 300];
        let mut b = vec![C32::default(); 300];
        Simd.convert(&format, &input, &mut a).unwrap();
        Simd.convert(&big, &input, &mut b).unwrap();
        assert_eq!(bits(&a), bits(&b), "{format:?}");
    }
}

// `CQ15` output and the real formats (`convert_real` to `f32` and Q15): the same paths over
// values in stream order, against the fallback bit for bit.

/// A Q15 value the kernels never produce for these inputs (odd, and neither `x << 8` nor
/// `(2u - 255) << 7`), written beyond the converted values.
const Q15_SENTINEL: i16 = 0x1235;

/// `CQ15` of `input` with both implementations: results, values and the untouched output.
fn check_cq15(format: &SampleFormat, input: &[u8], what: &str) {
    let samples = format.samples_in_bytes(input.len());
    let s = Q15(Q15_SENTINEL);
    let mut want = vec![CQ15 { re: s, im: s }; samples + 3];
    let mut got = want.clone();
    let rw = Fallback.convert(format, input, &mut want);
    let rg = Simd.convert(format, input, &mut got);
    assert_eq!(rw, rg, "{format:?} {what}");
    assert_eq!(rg, Ok(samples), "{format:?} {what}");
    assert_eq!(want, got, "{format:?} {what}");
    assert!(
        got[samples..].iter().all(|c| *c == CQ15 { re: s, im: s }),
        "{format:?} {what}: overrun"
    );
}

/// The bits of `x`.
fn f32_bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// `convert_real` of `input` to `f32` and Q15 with both implementations: results, bits, no
/// `-0.0` and the untouched output.
fn check_real(format: &SampleFormat, input: &[u8], what: &str) {
    let samples = format.samples_in_bytes(input.len());
    let mut want = vec![SENTINEL; samples + 3];
    let mut got = want.clone();
    let rw = Fallback.convert_real(format, input, &mut want);
    let rg = Simd.convert_real(format, input, &mut got);
    assert_eq!(rw, rg, "{format:?} {what}");
    assert_eq!(rg, Ok(samples), "{format:?} {what}");
    assert_eq!(f32_bits(&want), f32_bits(&got), "{format:?} {what}");
    assert!(
        got[..samples]
            .iter()
            .all(|v| v.to_bits() != (-0.0f32).to_bits()),
        "{format:?} {what}: -0.0"
    );
    assert!(
        got[samples..].iter().all(|v| *v == SENTINEL),
        "{format:?} {what}: overrun"
    );
    let mut want = vec![Q15(Q15_SENTINEL); samples + 3];
    let mut got = want.clone();
    let rw = Fallback.convert_real(format, input, &mut want);
    let rg = Simd.convert_real(format, input, &mut got);
    assert_eq!(rw, rg, "{format:?} {what}");
    assert_eq!(want, got, "{format:?} {what}");
}

/// A real format of one container word per value.
fn real(container: Container, bits: u8, signed: bool, endian: Endian) -> SampleFormat {
    SampleFormat::new(container, bits, signed, Layout::Real, Align::Lsb, endian).unwrap()
}

/// The real formats with a vector path and their bytes per value: `RS16` and the real 8-bit
/// formats.
fn real_formats() -> [(SampleFormat, usize); 3] {
    [
        (formats::RS16, 2),
        (real(Container::I8, 8, true, Endian::Little), 1),
        (real(Container::U8, 8, false, Endian::Little), 1),
    ]
}

/// Every length from 0 to beyond the largest block of the paths at every start, through
/// `check`.
fn every_length(width: usize, values_per_sample: usize, check: impl Fn(&[u8], &str)) {
    let max_samples = 4 * 128 + 40;
    let mut rng = Rng::new(7);
    let buffer: Vec<u8> = (0..STARTS + values_per_sample * width * max_samples)
        .map(|_| rng.next_u32() as u8)
        .collect();
    for start in 0..STARTS {
        for samples in 0..=max_samples {
            let len = values_per_sample * width * samples;
            check(
                &buffer[start..start + len],
                &format!("start {start}, {samples} samples"),
            );
        }
    }
}

/// The exhaustive input of `width`-byte values at every start, through `check`.
fn every_value(width: usize, values: usize, check: impl Fn(&[u8], &str)) {
    let data = all_values(width, values);
    let mut buffer = vec![0u8; STARTS + data.len()];
    for start in 0..STARTS {
        buffer[start..start + data.len()].copy_from_slice(&data);
        check(
            &buffer[start..start + data.len()],
            &format!("start {start}"),
        );
    }
}

#[test]
fn cq15_every_value_and_every_length_bit_exact() {
    for (format, width, values) in [
        (formats::CS16, 2, 1 << 16),
        (formats::CS8, 1, 256),
        (formats::CU8, 1, 256),
    ] {
        every_value(width, values, |input, what| {
            check_cq15(&format, input, what);
        });
        every_length(width, 2, |input, what| check_cq15(&format, input, what));
    }
}

#[test]
fn real_every_value_and_every_length_bit_exact() {
    for (format, width) in real_formats() {
        every_value(width, 1 << (8 * width), |input, what| {
            check_real(&format, input, what);
        });
        every_length(width, 1, |input, what| check_real(&format, input, what));
    }
}

#[test]
fn real_errors_and_other_formats_match_the_fallback() {
    let mut rng = Rng::new(11);
    let input: Vec<u8> = (0..4 * 301).map(|_| rng.next_u32() as u8).collect();
    let formats_without_path = [
        formats::RF32,
        real(Container::I16, 16, true, Endian::Big),
        real(Container::I16, 12, true, Endian::Little),
    ];
    for format in formats_without_path.into_iter().chain([formats::RS16]) {
        for len in [0, 1, 2, 3, 4, 601, 602, 1204] {
            let input = &input[..len];
            for room in [0, 1, 150, 301, 602] {
                let mut want = vec![SENTINEL; room];
                let mut got = want.clone();
                let rw = Fallback.convert_real(&format, input, &mut want);
                let rg = Simd.convert_real(&format, input, &mut got);
                assert_eq!(rw, rg, "{format:?} {len} {room}");
                assert_eq!(f32_bits(&want), f32_bits(&got), "{format:?} {len} {room}");
                let mut want = vec![Q15(Q15_SENTINEL); room];
                let mut got = want.clone();
                let rw = Fallback.convert_real(&format, input, &mut want);
                let rg = Simd.convert_real(&format, input, &mut got);
                assert_eq!(rw, rg, "{format:?} {len} {room}");
                assert_eq!(want, got, "{format:?} {len} {room}");
            }
        }
    }
    // A complex format is an error of `convert_real`, as in the fallback.
    let (mut a, mut b) = ([0.0f32; 4], [0.0f32; 4]);
    let rw = Fallback.convert_real(&formats::CS16, &input[..8], &mut a);
    assert!(rw.is_err());
    assert_eq!(Simd.convert_real(&formats::CS16, &input[..8], &mut b), rw);
}
