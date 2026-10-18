//! Golden tests of the transmit kernels of M08 (`convert_out`, `fir_interpolate`, `fm_mod`)
//! through the facade `Sdrk`, and of the same cases through `Reference` where it applies.
//!
//! Every case is run in one piece and in chunks of random size (the interpolators with a small
//! output buffer, so the `Progress` handling is exercised too); both runs must match the golden
//! file and each other bit for bit.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use sdrk::Sdrk;
use sdrk_core::kernels::convert::{ConvertOut, ConvertOutReal};
use sdrk_core::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::{C32, CQ15, Progress, Q15, Rate, Reference, Sample};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};

/// Output buffer size of the chunked interpolator runs; deliberately not a multiple of `L`.
const SMALL_ROOM: usize = 13;

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

fn usize_param(case: &GoldenCase, name: &str) -> usize {
    usize::try_from(case.param_u64(name).unwrap()).unwrap()
}

fn expected_bytes(case: &GoldenCase) -> &[u8] {
    match case.array("expected").unwrap() {
        Array::U8(v) => v,
        other => panic!("expected must be u8, got {}", other.dtype()),
    }
}

/// Encodes `input` in one piece and in chunks of whole packing groups; returns both byte streams.
fn encode_both<T: Copy>(
    format: &SampleFormat,
    input: &[T],
    seed: u64,
    encode: &impl Fn(&SampleFormat, &[T], &mut [u8]) -> usize,
) -> (Vec<u8>, Option<Vec<u8>>) {
    let total = format.bytes_for_samples(input.len()).unwrap();
    let mut whole = vec![0u8; total + 5];
    let n = encode(format, input, &mut whole);
    assert_eq!(n, total);
    whole.truncate(n);
    if format.layout() == Layout::Planar {
        // The halves of a planar stream span the whole block: chunking is not defined.
        return (whole, None);
    }
    let mut chunked = Vec::new();
    let mut pos = 0;
    for size in ChunkPlan::new(seed, 40).sizes() {
        if pos >= input.len() {
            break;
        }
        // Multiples of four samples keep every packing group whole.
        let end = (pos + size * 4).min(input.len());
        let piece = &input[pos..end];
        let mut out = vec![0u8; format.bytes_for_samples(piece.len()).unwrap()];
        encode(format, piece, &mut out);
        chunked.extend(out);
        pos = end;
    }
    (whole, Some(chunked))
}

#[test]
fn all_cases_are_present() {
    let names = golden::list("convert_out").unwrap();
    for format in [
        "cu8",
        "cs8",
        "cs16",
        "cs12_lsb16",
        "sc16_q11",
        "cs16_planar",
        "packed12",
        "packed10",
        "i2s32_msb",
        "rs16",
        "cu8_swapiq",
        "cs16_bigendian",
    ] {
        for source in ["c32", "cq15"] {
            let name = format!("{format}_from_{source}_256");
            assert!(names.contains(&name), "missing convert_out/{name}");
        }
    }
    for name in [
        "cf32_from_c32_256",
        "rf32_from_c32_256",
        "cf32_from_cq15_256",
        "rf32_from_cq15_256",
    ] {
        assert!(
            names.contains(&name.to_string()),
            "missing convert_out/{name}"
        );
    }
    assert_eq!(names.len(), 28);
    assert_eq!(
        golden::list("fir_interpolate").unwrap(),
        ["l5_n241_c32_4000", "l5_n241_cq15_4000"]
    );
    assert_eq!(
        golden::list("fir_interpolate_real").unwrap(),
        ["l5_n241_f32_4000", "l5_n241_q15_4000"]
    );
    assert_eq!(
        golden::list("fm_mod").unwrap(),
        [
            "dev33k_fs192k_c32_8192",
            "dev33k_fs192k_cq15_8192",
            "dev75k_fs240k_c32_8192",
            "dev75k_fs240k_cq15_8192"
        ]
    );
}

#[test]
fn convert_out_golden() {
    for name in golden::list("convert_out").unwrap() {
        let case = golden::load("convert_out", &name).unwrap();
        let format = format_of(&case);
        let want = expected_bytes(&case);
        let samples = usize_param(&case, "samples");
        let (whole, chunked): (Vec<u8>, Option<Vec<u8>>) = match case.array("input").unwrap() {
            Array::CF32(x) => {
                assert_eq!(x.len(), samples);
                for backend in 0..2 {
                    let enc = |f: &SampleFormat, i: &[C32], o: &mut [u8]| {
                        if backend == 0 {
                            ConvertOut::<C32>::convert_out(&Sdrk, f, i, o).unwrap()
                        } else {
                            ConvertOut::<C32>::convert_out(&Reference, f, i, o).unwrap()
                        }
                    };
                    let (w, c) = encode_both(&format, x, case.seed, &enc);
                    assert_eq!(w, want, "{name} backend {backend}: bytes differ");
                    if let Some(c) = c {
                        assert_eq!(c, want, "{name} backend {backend}: chunking changed bytes");
                    }
                }
                continue;
            }
            Array::F32(x) => {
                assert_eq!(x.len(), samples);
                for backend in 0..2 {
                    let enc = |f: &SampleFormat, i: &[f32], o: &mut [u8]| {
                        if backend == 0 {
                            ConvertOutReal::<f32>::convert_out_real(&Sdrk, f, i, o).unwrap()
                        } else {
                            ConvertOutReal::<f32>::convert_out_real(&Reference, f, i, o).unwrap()
                        }
                    };
                    let (w, c) = encode_both(&format, x, case.seed, &enc);
                    assert_eq!(w, want, "{name} backend {backend}: bytes differ");
                    if let Some(c) = c {
                        assert_eq!(c, want, "{name} backend {backend}: chunking changed bytes");
                    }
                }
                continue;
            }
            Array::CI16(x) => {
                assert_eq!(x.len(), samples);
                let enc = |f: &SampleFormat, i: &[CQ15], o: &mut [u8]| {
                    let a = ConvertOut::<CQ15>::convert_out(&Sdrk, f, i, o).unwrap();
                    let mut other = vec![0u8; o.len()];
                    let b = ConvertOut::<CQ15>::convert_out(&Reference, f, i, &mut other).unwrap();
                    assert_eq!((a, &o[..a]), (b, &other[..b]), "reference differs");
                    a
                };
                encode_both(&format, x, case.seed, &enc)
            }
            Array::I16(x) => {
                assert_eq!(x.len(), samples);
                let q: Vec<Q15> = x.iter().map(|&r| Q15(r)).collect();
                let enc = |f: &SampleFormat, i: &[Q15], o: &mut [u8]| {
                    let a = ConvertOutReal::<Q15>::convert_out_real(&Sdrk, f, i, o).unwrap();
                    let mut other = vec![0u8; o.len()];
                    let b = ConvertOutReal::<Q15>::convert_out_real(&Reference, f, i, &mut other)
                        .unwrap();
                    assert_eq!((a, &o[..a]), (b, &other[..b]), "reference differs");
                    a
                };
                encode_both(&format, &q, case.seed, &enc)
            }
            other => panic!("unexpected input type {} in {name}", other.dtype()),
        };
        assert_eq!(whole, want, "{name}: bytes differ");
        if let Some(c) = chunked {
            assert_eq!(c, want, "{name}: chunking changed the bytes");
        }
    }
}

/// Runs an interpolator over `input` in one piece (room for everything).
fn run_whole<T: Copy + Default, S>(
    state: &mut S,
    input: &[T],
    room: usize,
    step: &mut impl FnMut(&mut S, &[T], &mut [T]) -> Progress,
) -> Vec<T> {
    let mut out = vec![T::default(); room];
    let p = step(state, input, &mut out);
    assert_eq!(p.consumed, input.len(), "the whole input must be consumed");
    out.truncate(p.produced);
    out
}

/// Runs an interpolator in chunks from `plan` with `SMALL_ROOM` outputs per call.
fn run_chunked<T: Copy + Default, S>(
    state: &mut S,
    input: &[T],
    plan: ChunkPlan,
    step: &mut impl FnMut(&mut S, &[T], &mut [T]) -> Progress,
) -> Vec<T> {
    let mut all = Vec::new();
    let mut pos = 0;
    for size in plan.sizes() {
        if pos >= input.len() {
            break;
        }
        let end = (pos + size).min(input.len());
        let mut rest = &input[pos..end];
        loop {
            let mut buf = [T::default(); SMALL_ROOM];
            let p = step(state, rest, &mut buf);
            all.extend_from_slice(&buf[..p.produced]);
            rest = &rest[p.consumed..];
            if rest.is_empty() && p.produced < SMALL_ROOM {
                break;
            }
            assert!(
                p.consumed + p.produced > 0,
                "no progress with {} samples left",
                rest.len()
            );
        }
        pos = end;
    }
    all
}

fn check_both<T: Sample>(case: &GoldenCase, want: &Array, whole: &[T], chunked: &[T]) {
    golden::assert_matches(case, want, whole).unwrap();
    assert_eq!(
        bit_exact(whole, chunked),
        None,
        "{}: block size changed the output",
        case.case
    );
    golden::assert_matches(case, want, chunked).unwrap();
}

#[test]
fn fir_interpolate_golden() {
    for name in golden::list("fir_interpolate").unwrap() {
        let case = golden::load("fir_interpolate", &name).unwrap();
        let want = case.array("expected").unwrap();
        let l = usize_param(&case, "factor");
        assert_eq!(want.len(), usize_param(&case, "outputs"));
        let plan = ChunkPlan::new(case.seed, 300);
        match (case.array("input").unwrap(), case.array("taps").unwrap()) {
            (Array::CF32(x), Array::F32(h)) => {
                let new = || {
                    FirInterpState::<C32, _, _>::new(
                        h.clone(),
                        vec![C32::ZERO; h.len().div_ceil(l)],
                        l,
                    )
                    .unwrap()
                };
                let mut step =
                    |s: &mut FirInterpState<C32, Vec<f32>, Vec<C32>>, i: &[C32], o: &mut [C32]| {
                        FirInterpolate::<C32>::fir_interpolate(&Sdrk, s, i, o).unwrap()
                    };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            (Array::CI16(x), Array::I16(_)) => {
                let h = case.array("taps").unwrap().as_q15().unwrap();
                let new = || {
                    FirInterpState::<CQ15, _, _>::new(
                        h.clone(),
                        vec![CQ15::ZERO; h.len().div_ceil(l)],
                        l,
                    )
                    .unwrap()
                };
                let mut step = |s: &mut FirInterpState<CQ15, Vec<Q15>, Vec<CQ15>>,
                                i: &[CQ15],
                                o: &mut [CQ15]| {
                    FirInterpolate::<CQ15>::fir_interpolate(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
                // The reference is bit-exact as well.
                let mut state = new();
                let mut y = vec![CQ15::ZERO; want.len()];
                FirInterpolate::<CQ15>::fir_interpolate(&Reference, &mut state, x, &mut y).unwrap();
                golden::assert_matches(&case, want, &y).unwrap();
            }
            _ => panic!("unexpected array types in {name}"),
        }
    }
}

#[test]
fn fir_interpolate_real_golden() {
    for name in golden::list("fir_interpolate_real").unwrap() {
        let case = golden::load("fir_interpolate_real", &name).unwrap();
        let want = case.array("expected").unwrap();
        let l = usize_param(&case, "factor");
        let plan = ChunkPlan::new(case.seed, 300);
        match (case.array("input").unwrap(), case.array("taps").unwrap()) {
            (Array::F32(x), Array::F32(h)) => {
                let new = || {
                    FirInterpRealState::<f32, _, _>::new(
                        h.clone(),
                        vec![0f32; h.len().div_ceil(l)],
                        l,
                    )
                    .unwrap()
                };
                let mut step = |s: &mut FirInterpRealState<f32, Vec<f32>, Vec<f32>>,
                                i: &[f32],
                                o: &mut [f32]| {
                    FirInterpolateReal::<f32>::fir_interpolate_real(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            (Array::I16(x), Array::I16(_)) => {
                let x: Vec<Q15> = x.iter().map(|&r| Q15(r)).collect();
                let h = case.array("taps").unwrap().as_q15().unwrap();
                let new = || {
                    FirInterpRealState::<Q15, _, _>::new(
                        h.clone(),
                        vec![Q15(0); h.len().div_ceil(l)],
                        l,
                    )
                    .unwrap()
                };
                let mut step = |s: &mut FirInterpRealState<Q15, Vec<Q15>, Vec<Q15>>,
                                i: &[Q15],
                                o: &mut [Q15]| {
                    FirInterpolateReal::<Q15>::fir_interpolate_real(&Sdrk, s, i, o).unwrap()
                };
                let whole = run_whole(&mut new(), &x, want.len() + 3, &mut step);
                let chunked = run_chunked(&mut new(), &x, plan, &mut step);
                check_both(&case, want, &whole, &chunked);
            }
            _ => panic!("unexpected array types in {name}"),
        }
    }
}

fn fm_state(case: &GoldenCase) -> FmModState {
    let rate = Rate::hz(case.param_f64("rate_hz").unwrap() as u64).unwrap();
    FmModState::new(case.param_f64("deviation_hz").unwrap(), rate).unwrap()
}

/// Runs `fm_mod` on all of `x` and again in random chunks.
fn fm_mod_both<I: Copy, O: Copy + Default>(
    case: &GoldenCase,
    x: &[I],
    run: &impl Fn(&mut FmModState, &[I], &mut [O]),
) -> (Vec<O>, Vec<O>) {
    let mut state = fm_state(case);
    let mut whole = vec![O::default(); x.len()];
    run(&mut state, x, &mut whole);
    let mut state = fm_state(case);
    let mut chunked = vec![O::default(); x.len()];
    let mut pos = 0;
    for size in ChunkPlan::new(case.seed, 500).sizes() {
        if pos >= x.len() {
            break;
        }
        let end = (pos + size).min(x.len());
        run(&mut state, &x[pos..end], &mut chunked[pos..end]);
        pos = end;
    }
    (whole, chunked)
}

#[test]
fn fm_mod_golden() {
    for name in golden::list("fm_mod").unwrap() {
        let case = golden::load("fm_mod", &name).unwrap();
        let want = case.array("expected").unwrap();
        match case.array("input").unwrap() {
            Array::F32(x) => {
                let (whole, chunked) = fm_mod_both(&case, x, &|s, i, o: &mut [C32]| {
                    FmMod::<C32>::fm_mod(&Sdrk, s, i, o).unwrap();
                });
                check_both(&case, want, &whole, &chunked);
                let mut state = fm_state(&case);
                let mut y = vec![C32::ZERO; x.len()];
                FmMod::<C32>::fm_mod(&Reference, &mut state, x, &mut y).unwrap();
                golden::assert_matches(&case, want, &y).unwrap();
            }
            Array::I16(raw) => {
                let x: Vec<Q15> = raw.iter().map(|&r| Q15(r)).collect();
                let (whole, chunked) = fm_mod_both(&case, &x, &|s, i, o: &mut [CQ15]| {
                    FmMod::<CQ15>::fm_mod(&Sdrk, s, i, o).unwrap();
                });
                check_both(&case, want, &whole, &chunked);
                let mut state = fm_state(&case);
                let mut y = vec![CQ15::ZERO; x.len()];
                FmMod::<CQ15>::fm_mod(&Reference, &mut state, &x, &mut y).unwrap();
                golden::assert_matches(&case, want, &y).unwrap();
            }
            other => panic!("unexpected input type {} in {name}", other.dtype()),
        }
    }
}
