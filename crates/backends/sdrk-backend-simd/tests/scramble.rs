//! `scramble` (M49 test 7, revised by M49.1 test 3) against the fallback at the level of the
//! running build: both types, every `bits` 1 to 8, both directions, every length from 0 to 67
//! and 4096, unaligned input and output slices, every phase byte 0 to 255 at every lane
//! position, the special values (`CQ15` components −32768 and 32767; `C32` `±0`, `±∞`,
//! subnormals and NaN payloads) and the golden cases.
//!
//! A `C32` sample at `bits` 1 and 2 (pure sign and swap operations) is compared by `to_bits()`
//! with NaN payloads included; a sample at `bits ≥ 3` (the product with the table phasor) by
//! `to_bits()` except that a NaN result is compared by class (D296). `CQ15` is compared bit for
//! bit everywhere.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

mod common;

use std::path::PathBuf;

use common::{Rng, chunks};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::scramble::{Scramble, phase_index, scramble_one_c32, split_index};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15};

/// The seeds of the inputs (the golden seeds of M49 §4).
const SEEDS: [u64; 7] = [690, 691, 692, 693, 694, 698, 699];

/// `C32` special values: signed zeros, infinities, subnormals, NaN payloads of both signs,
/// the extremes of the finite range and ordinary values.
fn c32_specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007F_FFFF),
        f32::MIN_POSITIVE,
        f32::from_bits(0x7FC0_0000),
        f32::from_bits(0xFFC0_0001),
        f32::from_bits(0x7F80_0001),
        f32::from_bits(0xFFA5_A5A5),
        f32::MAX,
        f32::MIN,
        1.0,
        -1.0,
        0.5,
        -0.75,
    ]
}

/// `n` `C32` samples: uniform in the unit square with one special value in every fourth sample.
fn c32_input(seed: u64, n: usize) -> Vec<C32> {
    let mut rng = Rng::new(seed);
    let sp = c32_specials();
    (0..n)
        .map(|i| {
            let mut v = C32::new(rng.unit(), rng.unit());
            if i % 4 == 1 {
                v.re = sp[rng.below(sp.len())];
            }
            if i % 4 == 3 {
                v.im = sp[rng.below(sp.len())];
            }
            v
        })
        .collect()
}

/// `n` `CQ15` samples: random raw values with the corners −32768, 32767, −32767 and 0 in every
/// fourth sample.
fn cq15_input(seed: u64, n: usize) -> Vec<CQ15> {
    let mut rng = Rng::new(seed);
    let corners = [-32768i16, 32767, -32767, 0, 1, -1];
    (0..n)
        .map(|i| {
            let mut v = CQ15 {
                re: Q15(rng.next_u32() as i16),
                im: Q15(rng.next_u32() as i16),
            };
            if i % 4 == 0 {
                v.re = Q15(corners[rng.below(corners.len())]);
                v.im = Q15(corners[rng.below(corners.len())]);
            }
            v
        })
        .collect()
}

/// `n` phase bytes: every byte value 0 to 255 in turn (at an offset from the seed), with random
/// bytes in every third position.
fn phases(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed ^ 0x5A5A);
    let off = rng.below(256);
    (0..n)
        .map(|i| {
            if i % 3 == 2 {
                rng.next_u32() as u8
            } else {
                ((i + off) % 256) as u8
            }
        })
        .collect()
}

/// `got` equals `want` under the rule of the module documentation.
fn assert_c32(want: &[C32], got: &[C32], p: &[u8], bits: u32, what: &str) {
    assert_eq!(want.len(), got.len(), "{what}");
    // Exact (no product) only at bits 1 and 2 since M49.1.
    let exact = bits <= 2;
    for (n, ((w, g), &pv)) in want.iter().zip(got).zip(p).enumerate() {
        for (a, b) in [(w.re, g.re), (w.im, g.im)] {
            let same = if !exact && a.is_nan() {
                b.is_nan()
            } else {
                a.to_bits() == b.to_bits()
            };
            assert!(
                same,
                "{what}: sample {n} phase {pv}: want ({:?}, {:?}) got ({:?}, {:?})",
                w.re, w.im, g.re, g.im
            );
        }
    }
}

fn assert_cq15(want: &[CQ15], got: &[CQ15], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}");
    for (n, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(w, g, "{what}: sample {n}");
    }
}

/// One call of each backend on `x[..n]` with the output at offset `out_off` of a larger buffer.
fn same_c32(x: &[C32], p: &[u8], bits: u32, inverse: bool, out_off: usize, what: &str) {
    let n = x.len();
    let mut want = vec![C32::default(); n];
    Fallback.scramble(x, p, bits, inverse, &mut want).unwrap();
    let mut buf = vec![C32::new(7.0, 7.0); n + out_off + 1];
    Simd.scramble(x, p, bits, inverse, &mut buf[out_off..out_off + n])
        .unwrap();
    assert_c32(&want, &buf[out_off..out_off + n], p, bits, what);
    assert_eq!(
        buf[out_off + n],
        C32::new(7.0, 7.0),
        "{what}: wrote past the end"
    );
    if out_off > 0 {
        assert_eq!(
            buf[out_off - 1],
            C32::new(7.0, 7.0),
            "{what}: wrote before the start"
        );
    }
}

fn same_cq15(x: &[CQ15], p: &[u8], bits: u32, inverse: bool, out_off: usize, what: &str) {
    let n = x.len();
    let mut want = vec![CQ15::default(); n];
    Fallback.scramble(x, p, bits, inverse, &mut want).unwrap();
    let fill = CQ15 {
        re: Q15(1234),
        im: Q15(-4321),
    };
    let mut buf = vec![fill; n + out_off + 1];
    Simd.scramble(x, p, bits, inverse, &mut buf[out_off..out_off + n])
        .unwrap();
    assert_cq15(&want, &buf[out_off..out_off + n], what);
    assert_eq!(buf[out_off + n], fill, "{what}: wrote past the end");
    if out_off > 0 {
        assert_eq!(buf[out_off - 1], fill, "{what}: wrote before the start");
    }
}

fn lengths() -> impl Iterator<Item = usize> {
    (0..=67).chain([4096])
}

#[test]
fn c32_gives_the_bits_of_the_fallback_at_every_bits_length_and_offset() {
    for (s, &seed) in SEEDS.iter().enumerate() {
        let x = c32_input(seed, 4096 + 8);
        let p = phases(seed, 4096 + 8);
        for bits in 1..=8 {
            for inverse in [false, true] {
                for n in lengths() {
                    // Input and output offsets in samples (the vectors then start unaligned).
                    let (ia, oa) = ((n + s) % 3, (n + bits as usize) % 4);
                    same_c32(
                        &x[ia..ia + n],
                        &p[ia..ia + n],
                        bits,
                        inverse,
                        oa,
                        &format!(
                            "seed {seed} bits {bits} inverse {inverse} n {n} in +{ia} out +{oa}"
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn cq15_gives_the_bits_of_the_fallback_at_every_bits_length_and_offset() {
    for (s, &seed) in SEEDS.iter().enumerate() {
        let x = cq15_input(seed, 4096 + 8);
        let p = phases(seed, 4096 + 8);
        for bits in 1..=8 {
            for inverse in [false, true] {
                for n in lengths() {
                    let (ia, oa) = ((n + s) % 3, (n + bits as usize) % 4);
                    same_cq15(
                        &x[ia..ia + n],
                        &p[ia..ia + n],
                        bits,
                        inverse,
                        oa,
                        &format!(
                            "seed {seed} bits {bits} inverse {inverse} n {n} in +{ia} out +{oa}"
                        ),
                    );
                }
            }
        }
    }
}

/// Every phase byte 0 to 255 at every lane position: the 256 bytes in order, shifted by every
/// offset 0 to 15, with the same sample repeated (full-scale and special values).
#[test]
fn every_phase_byte_at_every_lane_position() {
    let cq = [
        CQ15 {
            re: Q15(-32768),
            im: Q15(-32768),
        },
        CQ15 {
            re: Q15(32767),
            im: Q15(-32768),
        },
        CQ15 {
            re: Q15(-32768),
            im: Q15(32767),
        },
        CQ15 {
            re: Q15(12345),
            im: Q15(-3),
        },
    ];
    let cf = [
        C32::new(f32::INFINITY, -0.0),
        C32::new(f32::from_bits(0xFFC0_0001), 1.0),
        C32::new(f32::from_bits(1), f32::NEG_INFINITY),
        C32::new(0.25, -0.5),
    ];
    for off in 0..16 {
        let p: Vec<u8> = (0..256 + 16).map(|i| ((i + off) % 256) as u8).collect();
        for bits in 1..=8 {
            for inverse in [false, true] {
                for v in cq {
                    let x = vec![v; p.len()];
                    same_cq15(
                        &x,
                        &p,
                        bits,
                        inverse,
                        0,
                        &format!("{v:?} off {off} bits {bits}"),
                    );
                }
                for v in cf {
                    let x = vec![v; p.len()];
                    same_c32(
                        &x,
                        &p,
                        bits,
                        inverse,
                        0,
                        &format!("{v:?} off {off} bits {bits}"),
                    );
                }
            }
        }
    }
}

/// At `bits` 1 and 2 every sample equals the scalar quadrant map exactly, NaN payloads included,
/// and `±∞` stays `±∞` (shared conventions §A.2): checked on the SIMD output directly. At
/// `bits ≥ 3` the samples equal `scramble_one_c32` (NaN by class).
#[test]
fn c32_samples_keep_every_bit_and_infinity_at_bits_1_and_2() {
    use sdrk_core::num::c32_quadrant;
    let sp = c32_specials();
    let x: Vec<C32> = sp
        .iter()
        .flat_map(|&a| sp.iter().map(move |&b| C32::new(a, b)))
        .collect();
    for bits in 1..=8 {
        for inverse in [false, true] {
            let p: Vec<u8> = (0..x.len()).map(|i| (i * 37 % 256) as u8).collect();
            let mut y = vec![C32::default(); x.len()];
            Simd.scramble(&x, &p, bits, inverse, &mut y).unwrap();
            for ((&xv, &yv), &pv) in x.iter().zip(&y).zip(&p) {
                let k = phase_index(pv, bits, inverse);
                if bits <= 2 {
                    let (q, _) = split_index(k, bits);
                    let w = c32_quadrant(xv, q);
                    assert_eq!(
                        (w.re.to_bits(), w.im.to_bits()),
                        (yv.re.to_bits(), yv.im.to_bits()),
                        "bits {bits} phase {pv} x {xv:?}"
                    );
                    let inf = |v: C32| u8::from(v.re.is_infinite()) + u8::from(v.im.is_infinite());
                    assert_eq!(inf(xv), inf(yv), "bits {bits} phase {pv} x {xv:?}");
                } else {
                    let w = scramble_one_c32(xv, k, bits).unwrap();
                    for (a, b) in [(w.re, yv.re), (w.im, yv.im)] {
                        assert!(
                            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                            "bits {bits} phase {pv} x {xv:?}"
                        );
                    }
                }
            }
        }
    }
}

/// Chunked calls give the bits of one call (the kernel is stateless; the phase bytes are sliced
/// with the input).
#[test]
fn chunked_calls_give_the_bits_of_one_call() {
    let x = c32_input(695, 2000);
    let xq = cq15_input(695, 2000);
    let p = phases(695, 2000);
    let mut rng = Rng::new(695);
    for bits in [1, 2, 3, 5, 8] {
        let mut whole = vec![C32::default(); 2000];
        Simd.scramble(&x, &p, bits, false, &mut whole).unwrap();
        let mut wq = vec![CQ15::default(); 2000];
        Simd.scramble(&xq, &p, bits, true, &mut wq).unwrap();
        for _ in 0..20 {
            let mut parts = vec![C32::default(); 2000];
            let mut pq = vec![CQ15::default(); 2000];
            let mut at = 0;
            for len in chunks(&mut rng, 2000) {
                let s = at..at + len;
                Simd.scramble(
                    &x[s.clone()],
                    &p[s.clone()],
                    bits,
                    false,
                    &mut parts[s.clone()],
                )
                .unwrap();
                Simd.scramble(&xq[s.clone()], &p[s.clone()], bits, true, &mut pq[s])
                    .unwrap();
                at += len;
            }
            assert_c32(&whole, &parts, &p, bits, "chunks");
            assert_eq!(wq, pq);
        }
    }
}

/// `CQ15` slices that are only 2-byte aligned (a `CQ15` has the alignment of `i16`).
#[test]
fn cq15_slices_at_two_byte_alignment() {
    let x = cq15_input(694, 300);
    let p = phases(694, 300);
    let flat: Vec<i16> = x.iter().flat_map(|v| [v.re.0, v.im.0]).collect();
    let mut src = vec![0i16; flat.len() + 2];
    let mut dst = vec![0i16; flat.len() + 2];
    // One of the offsets 0 and 1 gives a slice that is not 4-byte aligned.
    for off in 0..2 {
        src[off..off + flat.len()].copy_from_slice(&flat);
        let xs: &[CQ15] = bytemuck::cast_slice(&src[off..off + flat.len()]);
        for doff in 0..2 {
            let ys: &mut [CQ15] = bytemuck::cast_slice_mut(&mut dst[doff..doff + flat.len()]);
            for bits in [2, 8] {
                Simd.scramble(xs, &p, bits, false, ys).unwrap();
                let mut want = vec![CQ15::default(); x.len()];
                Fallback.scramble(&x, &p, bits, false, &mut want).unwrap();
                assert_cq15(&want, ys, &format!("off {off} {doff} bits {bits}"));
            }
        }
    }
    let mut buf = [0i16; 3];
    let ys: &mut [CQ15] = bytemuck::cast_slice_mut(&mut buf[1..3]);
    let xs = [CQ15::default()];
    Simd.scramble(&xs, &[1], 2, true, ys).unwrap();
}

#[test]
fn the_parameter_rules_hold() {
    let x = vec![C32::default(); 8];
    let mut y = vec![C32::new(3.0, 3.0); 8];
    for bits in [0, 9, 32] {
        assert!(matches!(
            Simd.scramble(&x, &[0; 8], bits, false, &mut y),
            Err(Error::InvalidParameter(_))
        ));
    }
    assert!(matches!(
        Simd.scramble(&x, &[0; 7], 2, false, &mut y),
        Err(Error::LengthMismatch { .. })
    ));
    assert!(matches!(
        Simd.scramble(&x, &[0; 8], 2, false, &mut y[..7]),
        Err(Error::LengthMismatch { .. })
    ));
    // Nothing is written on an error.
    assert!(y.iter().all(|v| *v == C32::new(3.0, 3.0)));
    let xq = vec![CQ15::default(); 8];
    let mut yq = vec![CQ15::default(); 8];
    assert!(matches!(
        Simd.scramble(&xq, &[0; 8], 0, true, &mut yq),
        Err(Error::InvalidParameter(_))
    ));
    Simd.scramble(&[] as &[CQ15], &[], 8, true, &mut [])
        .unwrap();
}

/// The level the kernels run at, when the run names one (`SDRK_EXPECT_LEVEL`, as in `api.rs`):
/// the facade reports `simd` for `scramble` there (`sdrk/tests/facade.rs`), and this is the
/// level behind it.
#[test]
fn the_level_is_the_expected_one() {
    if let Ok(want) = std::env::var("SDRK_EXPECT_LEVEL") {
        assert_eq!(sdrk_backend_simd::level_name(), want);
    }
    assert!(
        [
            "sse2", "sse4.2", "avx2", "avx512", "neon", "simd128", "scalar"
        ]
        .contains(&sdrk_backend_simd::level_name())
    );
}

// ---------------------------------------------------------------------------------------------
// The golden cases of M49 §4.

fn golden(file: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("tests")
        .join("golden")
        .join("vectors")
        .join("scramble")
        .join(file);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The value of the JSON field `"name": value` (numbers and `true`/`false`).
fn field(json: &str, name: &str) -> String {
    let key = format!("\"{name}\": ");
    let at = json.find(&key).unwrap_or_else(|| panic!("no field {name}")) + key.len();
    json[at..]
        .chars()
        .take_while(|c| !matches!(c, ',' | '\n' | '}'))
        .collect::<String>()
        .trim()
        .to_owned()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i16s(b: &[u8]) -> Vec<i16> {
    b.chunks_exact(2)
        .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// SNR in dB of `got` (pairs) against `want` (pairs).
fn snr(want: &[f64], got: &[f64]) -> f64 {
    let (mut s, mut e) = (0.0, 0.0);
    for (w, g) in want.iter().zip(got) {
        s += w * w;
        e += (w - g) * (w - g);
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / e).log10()
    }
}

/// The case through `Simd` (checked against `Fallback`): the output as `f64` pairs and as bytes.
fn run_cq15(name: &str, input: &[u8], p: &[u8], bits: u32, inverse: bool) -> (Vec<f64>, Vec<u8>) {
    let x: Vec<CQ15> = i16s(input)
        .chunks_exact(2)
        .map(|c| CQ15 {
            re: Q15(c[0]),
            im: Q15(c[1]),
        })
        .collect();
    assert_eq!(x.len(), p.len(), "{name}");
    let mut y = vec![CQ15::default(); x.len()];
    Simd.scramble(&x, p, bits, inverse, &mut y).unwrap();
    let mut f = vec![CQ15::default(); x.len()];
    Fallback.scramble(&x, p, bits, inverse, &mut f).unwrap();
    assert_cq15(&f, &y, name);
    let bytes = y
        .iter()
        .flat_map(|v| [v.re.0.to_le_bytes(), v.im.0.to_le_bytes()].concat())
        .collect();
    let vals = y
        .iter()
        .flat_map(|v| [f64::from(v.re.0) / 32768.0, f64::from(v.im.0) / 32768.0])
        .collect();
    (vals, bytes)
}

fn run_c32(name: &str, input: &[u8], p: &[u8], bits: u32, inverse: bool) -> (Vec<f64>, Vec<u8>) {
    let x: Vec<C32> = f32s(input)
        .chunks_exact(2)
        .map(|c| C32::new(c[0], c[1]))
        .collect();
    assert_eq!(x.len(), p.len(), "{name}");
    let mut y = vec![C32::default(); x.len()];
    Simd.scramble(&x, p, bits, inverse, &mut y).unwrap();
    let mut f = vec![C32::default(); x.len()];
    Fallback.scramble(&x, p, bits, inverse, &mut f).unwrap();
    assert_c32(&f, &y, p, bits, name);
    let bytes = y
        .iter()
        .flat_map(|v| [v.re.to_le_bytes(), v.im.to_le_bytes()].concat())
        .collect();
    let vals = y
        .iter()
        .flat_map(|v| [f64::from(v.re), f64::from(v.im)])
        .collect();
    (vals, bytes)
}

fn golden_case(name: &str, seed: u64) {
    let json = String::from_utf8(golden(&format!("{name}.json"))).unwrap();
    assert_eq!(field(&json, "seed"), seed.to_string(), "{name}");
    let bits: u32 = field(&json, "bits").parse().unwrap();
    let inverse: bool = field(&json, "inverse").parse().unwrap();
    let class = field(&json, "class");
    let p = golden(&format!("{name}.phase.bin"));
    let input = golden(&format!("{name}.input.bin"));
    let expected = golden(&format!("{name}.expected.bin"));
    let (got, want_bits) = if name.contains("cq15") {
        run_cq15(name, &input, &p, bits, inverse)
    } else {
        run_c32(name, &input, &p, bits, inverse)
    };
    match class.trim_matches('"') {
        "bit_exact" => assert!(want_bits == expected, "{name}: bits"),
        "tolerance" => {
            let min: f64 = field(&json, "snr_db_min").parse().unwrap();
            let s = snr(&f64s(&expected), &got);
            assert!(s >= min, "{name}: SNR {s} dB < {min} dB");
        }
        other => panic!("{name}: class {other}"),
    }
}

#[test]
fn the_golden_cases_hold_at_this_level() {
    golden_case("bpsk_c32_2048", 690);
    golden_case("qpsk_c32_2048", 691);
    golden_case("oct_c32_2048", 692);
    golden_case("qpsk_inv_c32_2048", 693);
    golden_case("qpsk_cq15_2048", 694);
    golden_case("byte_cq15_2048", 698);
    golden_case("byte_inv_c32_2048", 699);
}
