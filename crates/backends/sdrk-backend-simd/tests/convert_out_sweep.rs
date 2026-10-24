//! `convert_out` of `C32` and `CQ15` to `CU8`, `CS8` and `CS16` against the fallback, bit for
//! bit: float bit patterns of every class, every length and offset around the vector blocks,
//! the ties, and (ignored, run with `--release -- --ignored`; nightly in CI) every `f32`.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation
)]

mod common;

use common::Rng;
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::convert::ConvertOut;
use sdrk_core::{C32, CQ15, Fallback, Q15};
use sdrk_device::{SampleFormat, formats};

const FAST: [SampleFormat; 3] = [formats::CU8, formats::CS8, formats::CS16];

/// Both encodings of `input`, with a guard byte after the output that must stay untouched.
fn encode<T>(format: &SampleFormat, input: &[T], skip_out: usize) -> (Vec<u8>, Vec<u8>)
where
    Fallback: ConvertOut<T>,
    Simd: ConvertOut<T>,
    T: sdrk_core::ComplexSample,
{
    let n = format.bytes_for_samples(input.len()).unwrap();
    let mut want = vec![0x5Au8; skip_out + n + 1];
    let mut got = vec![0x5Au8; skip_out + n + 1];
    assert_eq!(
        Fallback.convert_out(format, input, &mut want[skip_out..]),
        Ok(n)
    );
    assert_eq!(Simd.convert_out(format, input, &mut got[skip_out..]), Ok(n));
    assert_eq!(got[skip_out + n], 0x5A, "{format:?}: byte after the output");
    (want, got)
}

#[test]
fn convert_out_strided_sweep_of_every_float_class() {
    // Every 4099th bit pattern (all exponents, both signs, NaN payloads, infinities,
    // subnormals), as both components and at several offsets of the vector loop.
    let input: Vec<C32> = (0..=u32::MAX / 4099)
        .map(|k| {
            let b = k.wrapping_mul(4099);
            C32::new(f32::from_bits(b), f32::from_bits(b.rotate_left(16)))
        })
        .collect();
    for format in FAST {
        for skip in [0usize, 1, 3] {
            let (want, got) = encode(&format, &input[skip..], 0);
            assert!(want == got, "{format:?} skip {skip}");
        }
    }
}

#[test]
fn convert_out_every_length_and_offset_around_the_vector_blocks() {
    let mut rng = Rng::new(7);
    let pool: Vec<C32> = (0..300)
        .map(|_| {
            let mut c = || match rng.below(3) {
                0 => rng.unit() * 1.2,
                1 => (rng.below(512) as f32 - 256.0) / 256.0,
                _ => f32::from_bits(rng.next_u32()),
            };
            C32::new(c(), c())
        })
        .collect();
    let q: Vec<CQ15> = common::cq15_noise(&mut rng, 300);
    for format in FAST {
        for len in 0..=140 {
            for skip in [0usize, 1, 2, 5] {
                let (want, got) = encode(&format, &pool[skip..skip + len], skip % 2);
                assert!(want == got, "{format:?} c32 len {len} skip {skip}");
                let (want, got) = encode(&format, &q[skip..skip + len], skip % 2);
                assert!(want == got, "{format:?} cq15 len {len} skip {skip}");
            }
        }
    }
}

#[test]
fn convert_out_ties_round_to_even() {
    // CS8 and CS16: k + 1/2 LSB goes to the even neighbour.
    for (format, lsb) in [(formats::CS8, 1.0 / 128.0), (formats::CS16, 1.0 / 32768.0)] {
        let input: Vec<C32> = (-6..6)
            .map(|k| C32::new((k as f32 + 0.5) * lsb, -(k as f32 + 0.5) * lsb))
            .collect();
        let (want, got) = encode(&format, &input, 0);
        assert_eq!(want, got);
        let w = format.bytes_for_samples(1).unwrap() / 2;
        for (k, c) in (-6..6).zip(got.chunks_exact(2 * w)) {
            let re = if w == 1 {
                i32::from(c[0] as i8)
            } else {
                i32::from(i16::from_le_bytes([c[0], c[1]]))
            };
            let even = if k % 2 == 0 { k } else { k + 1 };
            assert_eq!(re, even, "{format:?} {k}");
        }
    }
    // CU8: x * 128 integer is a tie of x * 128 + 127.5; 0 -> 128, 1/128 -> 128, 2/128 -> 130,
    // -1/128 -> 126; a negative amplitude above -2^-54 counts as zero like in f64.
    let input = [
        C32::new(0.0, 1.0 / 128.0),
        C32::new(2.0 / 128.0, -1.0 / 128.0),
        C32::new(-1.0e-17, -1.0e-15),
    ];
    let (want, got) = encode(&formats::CU8, &input, 0);
    assert_eq!(want, got);
    assert_eq!(got[..6], [128, 128, 130, 126, 128, 127]);
    // CQ15: q / 256 for CS8 and (q + 32640) / 256 for CU8.
    let q = |re, im| CQ15 {
        re: Q15(re),
        im: Q15(im),
    };
    let input = [q(128, 384), q(-128, -384), q(0, 256), q(512, 32767)];
    let (want, got) = encode(&formats::CS8, &input, 0);
    assert_eq!(want, got);
    assert_eq!(got[..8], [0, 2, 0, 254, 0, 1, 2, 127]);
    let (want, got) = encode(&formats::CU8, &input, 0);
    assert_eq!(want, got);
    assert_eq!(got[..8], [128, 129, 127, 126, 128, 128, 130, 255]);
}

/// The part `k` of `n` of the exhaustive test (`SDRK_SWEEP_PART=k/n`, default `0/1`, all of
/// it): the chunks whose index is `k` modulo `n`. Every part covers every exponent and both
/// signs, so the nightly AVX-512 run under SDE splits the work over several jobs.
fn sweep_part() -> (u32, u32) {
    let Ok(part) = std::env::var("SDRK_SWEEP_PART") else {
        return (0, 1);
    };
    let parsed = part
        .split_once('/')
        .and_then(|(k, n)| Some((k.parse::<u32>().ok()?, n.parse::<u32>().ok()?)))
        .filter(|&(k, n)| k < n);
    parsed.unwrap_or_else(|| panic!("SDRK_SWEEP_PART must be k/n with k < n, not {part:?}"))
}

#[test]
#[ignore = "every f32 bit pattern: minutes; run with --release -- --ignored"]
fn convert_out_exhaustive_every_f32() {
    const CHUNK: u32 = 1 << 20;
    let (part, parts) = sweep_part();
    let mut input = vec![C32::ZERO; CHUNK as usize / 2];
    for format in FAST {
        for index in (part..=u32::MAX / CHUNK).step_by(parts as usize) {
            let start = index * CHUNK;
            for (k, c) in input.iter_mut().enumerate() {
                let b = start + 2 * k as u32;
                *c = C32::new(f32::from_bits(b), f32::from_bits(b + 1));
            }
            let (want, got) = encode(&format, &input, 0);
            assert!(want == got, "{format:?} from bit pattern {start:#x}");
        }
    }
}
