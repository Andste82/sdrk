//! `convert` of `CS16`, `CS8` and `CU8` to `C32` against the fallback, bit for bit: every input
//! value in every lane position, at every start alignment, with every number of values after
//! the last whole block of the vector paths, no `-0.0` and nothing written beyond the converted
//! samples. Runs at the level of the build (and, with run-time detection,
//! at the level of the CPU), so the test matrix covers every level-specific path.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, bits};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::convert::Convert;
use sdrk_core::{C32, Fallback};
use sdrk_device::{SampleFormat, formats};

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
