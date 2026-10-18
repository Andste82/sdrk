//! Golden tests of `convert`, `rotate` and `fir` through the facade `Sdrk`.
//!
//! Every case is run in one piece and in chunks of random size; both runs must
//! match the golden file and each other bit for bit.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_core::kernels::convert::{Convert, ConvertReal};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::{C32, CQ15, Q15, Rate, Sample};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

fn format_of(case: &GoldenCase) -> SampleFormat {
    let p = |k: &str| case.param_str(k).unwrap();
    let container = match p("container") {
        "u8" => Container::U8,
        "i8" => Container::I8,
        "i16" => Container::I16,
        "i32" => Container::I32,
        "f32" => Container::F32,
        "packed12" => Container::Packed12,
        "packed10" => Container::Packed10,
        other => panic!("unknown container {other}"),
    };
    let layout = match p("layout") {
        "interleaved" => Layout::Interleaved,
        "planar" => Layout::Planar,
        "real" => Layout::Real,
        other => panic!("unknown layout {other}"),
    };
    let align = if p("align") == "msb" {
        Align::Msb
    } else {
        Align::Lsb
    };
    let endian = if p("endian") == "big" {
        Endian::Big
    } else {
        Endian::Little
    };
    let bits = u8::try_from(case.param_u64("bits").unwrap()).unwrap();
    let signed = case.params["signed"].as_bool().unwrap();
    let mut f = SampleFormat::new(container, bits, signed, layout, align, endian).unwrap();
    f = f.with_swap_iq(case.params["swap_iq"].as_bool().unwrap());
    if let Some(n) = case.params["full_scale"].as_u64() {
        f = f
            .with_full_scale(FullScale::Explicit(u32::try_from(n).unwrap()))
            .unwrap();
    }
    f
}

fn bytes_of(case: &GoldenCase) -> &[u8] {
    match case.array("input").unwrap() {
        Array::U8(v) => v,
        other => panic!("input must be u8, got {}", other.dtype()),
    }
}

/// Sample counts of the chunks (multiples of two samples so packing groups stay whole).
fn chunks(total: usize, seed: u64) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut left = total;
    for s in ChunkPlan::new(seed, 40).sizes() {
        if left == 0 {
            break;
        }
        let take = (s * 2).min(left);
        sizes.push(take);
        left -= take;
    }
    sizes
}

#[test]
fn all_cases_are_present() {
    let names = golden::list("convert").unwrap();
    for expected in [
        "cu8_256",
        "cs8_256",
        "cs16_256",
        "cs12_lsb16_256",
        "sc16_q11_256",
        "cs16_planar_256",
        "packed12_256",
        "packed10_256",
        "i2s32_msb_256",
        "rs16_256",
        "cf32_256",
        "rf32_256",
        "cu8_swapiq_256",
        "cs16_bigendian_256",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "missing convert/{expected}"
        );
    }
    assert_eq!(names.len(), 14);
    assert_eq!(golden::list("rotate").unwrap().len(), 5);
    assert_eq!(golden::list("fir").unwrap().len(), 5);
}

#[test]
fn convert_golden() {
    for name in golden::list("convert").unwrap() {
        let case = golden::load("convert", &name).unwrap();
        let format = format_of(&case);
        let input = bytes_of(&case);
        let samples = case.param_u64("samples").unwrap() as usize;
        let want_f = case.array("expected_f32").unwrap();
        let want_q = case.array("expected_q15").unwrap();
        if format.layout() == Layout::Real {
            let mut f = vec![0f32; samples];
            assert_eq!(
                ConvertReal::<f32>::convert_real(&Sdrk, &format, input, &mut f).unwrap(),
                samples
            );
            golden::assert_matches(&case, want_f, &f).unwrap();
            let mut q = vec![Q15(0); samples];
            assert_eq!(
                ConvertReal::<Q15>::convert_real(&Sdrk, &format, input, &mut q).unwrap(),
                samples
            );
            golden::assert_matches(&case, want_q, &q).unwrap();
            // chunked
            let (mut fc, mut qc) = (Vec::new(), Vec::new());
            let mut pos = 0;
            for n in chunks(samples, 1) {
                let bytes = format.bytes_for_samples(n).unwrap();
                let piece = &input[pos..pos + bytes];
                pos += bytes;
                let (mut fo, mut qo) = (vec![0f32; n], vec![Q15(0); n]);
                ConvertReal::<f32>::convert_real(&Sdrk, &format, piece, &mut fo).unwrap();
                ConvertReal::<Q15>::convert_real(&Sdrk, &format, piece, &mut qo).unwrap();
                fc.extend(fo);
                qc.extend(qo);
            }
            golden::assert_matches(&case, want_f, &fc).unwrap();
            golden::assert_matches(&case, want_q, &qc).unwrap();
        } else {
            let mut f = vec![C32::ZERO; samples];
            assert_eq!(
                Convert::<C32>::convert(&Sdrk, &format, input, &mut f).unwrap(),
                samples
            );
            golden::assert_matches(&case, want_f, &f).unwrap();
            let mut q = vec![CQ15::ZERO; samples];
            assert_eq!(
                Convert::<CQ15>::convert(&Sdrk, &format, input, &mut q).unwrap(),
                samples
            );
            golden::assert_matches(&case, want_q, &q).unwrap();
            if format.layout() == Layout::Planar {
                continue; // a planar buffer cannot be split into independent pieces
            }
            let (mut fc, mut qc) = (Vec::new(), Vec::new());
            let mut pos = 0;
            for n in chunks(samples, 2) {
                let bytes = format.bytes_for_samples(n).unwrap();
                let piece = &input[pos..pos + bytes];
                pos += bytes;
                let (mut fo, mut qo) = (vec![C32::ZERO; n], vec![CQ15::ZERO; n]);
                Convert::<C32>::convert(&Sdrk, &format, piece, &mut fo).unwrap();
                Convert::<CQ15>::convert(&Sdrk, &format, piece, &mut qo).unwrap();
                fc.extend(fo);
                qc.extend(qo);
            }
            golden::assert_matches(&case, want_f, &fc).unwrap();
            golden::assert_matches(&case, want_q, &qc).unwrap();
        }
    }
}

fn run_rotate<T: Copy + Default>(
    case: &GoldenCase,
    input: &[T],
    mut step: impl FnMut(&mut Nco, &[T], &mut [T]),
) -> (Vec<T>, Vec<T>) {
    let nco = || {
        let mut n = Nco::from_increment(u32::try_from(case.param_u64("inc").unwrap()).unwrap());
        n.set_phase(u32::try_from(case.param_u64("phase0").unwrap()).unwrap());
        n
    };
    // The increment computed from the frequency must equal the Python model.
    let from_freq = Nco::new(case.param_f64("freq").unwrap(), Rate::hz(1).unwrap()).unwrap();
    assert_eq!(
        u64::from(from_freq.increment()),
        case.param_u64("inc").unwrap(),
        "{}",
        case.case
    );
    let mut whole = vec![T::default(); input.len()];
    let mut n = nco();
    step(&mut n, input, &mut whole);
    let mut chunked = Vec::new();
    let mut n = nco();
    let mut pos = 0;
    for size in ChunkPlan::new(7, 500).sizes() {
        if pos >= input.len() {
            break;
        }
        let end = (pos + size).min(input.len());
        let mut out = vec![T::default(); end - pos];
        step(&mut n, &input[pos..end], &mut out);
        chunked.extend(out);
        pos = end;
    }
    (whole, chunked)
}

#[test]
fn rotate_golden() {
    for name in golden::list("rotate").unwrap() {
        let case = golden::load("rotate", &name).unwrap();
        let want = case.array("expected").unwrap();
        match case.array("input").unwrap() {
            Array::CF32(x) => {
                let (whole, chunked) = run_rotate(&case, x, |n, i, o| {
                    Rotate::<C32>::rotate(&Sdrk, n, i, o).unwrap();
                });
                golden::assert_matches(&case, want, &whole).unwrap();
                assert_eq!(
                    bit_exact(&whole, &chunked),
                    None,
                    "{name}: block size changed the output"
                );
            }
            Array::CI16(x) => {
                let (whole, chunked) = run_rotate(&case, x, |n, i, o| {
                    Rotate::<CQ15>::rotate(&Sdrk, n, i, o).unwrap();
                });
                golden::assert_matches(&case, want, &whole).unwrap();
                assert_eq!(
                    bit_exact(&whole, &chunked),
                    None,
                    "{name}: block size changed the output"
                );
            }
            other => panic!("unexpected input type {}", other.dtype()),
        }
    }
}

#[test]
fn fir_golden() {
    for name in golden::list("fir").unwrap() {
        let case = golden::load("fir", &name).unwrap();
        let want = case.array("expected").unwrap();
        match (case.array("input").unwrap(), case.array("taps").unwrap()) {
            (Array::CF32(x), Array::F32(h)) => {
                let new =
                    || FirState::<C32, _, _>::new(h.clone(), vec![C32::ZERO; h.len() - 1]).unwrap();
                let mut st = new();
                let mut whole = vec![C32::ZERO; x.len()];
                Fir::<C32>::fir(&Sdrk, &mut st, x, &mut whole).unwrap();
                golden::assert_matches(&case, want, &whole).unwrap();
                let mut st = new();
                let mut chunked = Vec::new();
                let mut pos = 0;
                for size in ChunkPlan::new(8, 300).sizes() {
                    if pos >= x.len() {
                        break;
                    }
                    let end = (pos + size).min(x.len());
                    let mut out = vec![C32::ZERO; end - pos];
                    Fir::<C32>::fir(&Sdrk, &mut st, &x[pos..end], &mut out).unwrap();
                    chunked.extend(out);
                    pos = end;
                }
                assert_eq!(
                    bit_exact(&whole, &chunked),
                    None,
                    "{name}: block size changed the output"
                );
                golden::assert_matches(&case, want, &chunked).unwrap();
            }
            (Array::CI16(x), Array::I16(_)) => {
                let h = case.array("taps").unwrap().as_q15().unwrap();
                let new = || {
                    FirState::<CQ15, _, _>::new(h.clone(), vec![CQ15::ZERO; h.len() - 1]).unwrap()
                };
                let mut st = new();
                let mut whole = vec![CQ15::ZERO; x.len()];
                Fir::<CQ15>::fir(&Sdrk, &mut st, x, &mut whole).unwrap();
                golden::assert_matches(&case, want, &whole).unwrap();
                let mut st = new();
                let mut chunked = Vec::new();
                let mut pos = 0;
                for size in ChunkPlan::new(9, 200).sizes() {
                    if pos >= x.len() {
                        break;
                    }
                    let end = (pos + size).min(x.len());
                    let mut out = vec![CQ15::ZERO; end - pos];
                    Fir::<CQ15>::fir(&Sdrk, &mut st, &x[pos..end], &mut out).unwrap();
                    chunked.extend(out);
                    pos = end;
                }
                golden::assert_matches(&case, want, &chunked).unwrap();
            }
            _ => panic!("unexpected array types in {name}"),
        }
    }
}
