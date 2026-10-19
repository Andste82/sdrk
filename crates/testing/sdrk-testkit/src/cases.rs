//! The test cases registered by [`all_cases`](crate::all_cases).
//!
//! Arena use per case (every case needs at most 256 KiB, quick cases at most
//! 64 KiB):
//!
//! | Case | Tier | Arena |
//! | --- | --- | --- |
//! | `rng_known_answer` | quick | 0 |
//! | `gen_tone_frequency` | quick | about 13 KiB |
//! | `encode_decode_roundtrip_all_formats` | quick | about 20 KiB |
//! | `scale_fallback_vs_reference` | quick | about 40 KiB |
//! | `scale_block_invariance` | full | about 100 KiB |
//! | `convert_all_formats_vs_encoder` | quick | about 7 KiB |
//! | `rotate_c32_vs_reference` | quick | about 48 KiB |
//! | `rotate_cq15_vs_reference` | quick | about 32 KiB |
//! | `rotate_block_invariance` | quick | about 24 KiB |
//! | `fir_c32_vs_reference` | quick | about 48 KiB |
//! | `fir_cq15_vs_reference` | quick | about 32 KiB |
//! | `fir_block_invariance` | quick | about 24 KiB |
//! | `fir_impulse_response` | quick | 0 (stack only) |
//! | `extremes_convert_rotate_fir` | full | about 20 KiB |
//! | `fir_decimate_c32_vs_reference` | quick | about 33 KiB |
//! | `fir_decimate_cq15_vs_reference` | quick | about 17 KiB |
//! | `fir_decimate_block_invariance` | quick | about 24 KiB |
//! | `resample_c32_vs_reference` | quick | about 15 KiB |
//! | `resample_cq15_vs_reference` | quick | about 10 KiB |
//! | `resample_real_vs_reference` | quick | about 10 KiB |
//! | `resample_block_invariance` | quick | about 25 KiB |
//! | `quad_demod_c32_vs_reference` | quick | about 20 KiB |
//! | `quad_demod_cq15_vs_reference` | quick | about 10 KiB |
//! | `quad_demod_block_invariance` | quick | about 20 KiB |
//! | `deemphasis_f32_vs_reference` | quick | about 32 KiB |
//! | `deemphasis_q15_vs_reference` | quick | about 16 KiB |
//! | `deemphasis_block_invariance` | quick | about 12 KiB |
//! | `dc_block_c32_vs_reference` | quick | about 32 KiB |
//! | `dc_block_cq15_vs_reference` | quick | about 32 KiB |
//! | `dc_block_block_invariance` | quick | about 24 KiB |
//! | `window_vs_reference` | quick | about 27 KiB |
//! | `window_generators` | quick | about 8 KiB |
//! | `mag_sq_vs_reference` | quick | about 30 KiB |
//! | `power_db_vs_reference` | quick | about 18 KiB |
//! | `extremes_rx_kernels` | full | about 30 KiB |
//! | `fft_c32_vs_reference_64` | quick | about 3 KiB |
//! | `fft_c32_vs_reference_1024` | quick | about 36 KiB |
//! | `fft_cq15_fixed_vs_reference_1024` | quick | about 18 KiB |
//! | `fft_cq15_bfp_vs_reference_1024` | quick | about 18 KiB |
//! | `fft_roundtrip_c32` | quick | about 20 KiB |
//! | `fft_roundtrip_cq15_bfp` | quick | about 10 KiB |
//! | `rfft_vs_reference_1024` | quick | about 26 KiB |
//! | `irfft_vs_reference_1024` | quick | about 30 KiB |
//! | `rfft_roundtrip_f32` | quick | about 22 KiB |
//! | `fft_weak_tone_bfp` | full | about 14 KiB |
//! | `fft_parseval_c32` | quick | about 21 KiB |
//! | `extremes_fft` | full | about 5 KiB |
//! | `convert_out_all_formats_roundtrip` | quick | about 6 KiB |
//! | `fir_interpolate_c32_vs_reference` | quick | about 19 KiB |
//! | `fir_interpolate_cq15_vs_reference` | quick | about 11 KiB |
//! | `fir_interpolate_real_vs_reference` | quick | about 9 KiB |
//! | `fir_interpolate_block_invariance` | quick | about 19 KiB |
//! | `fm_mod_c32_vs_reference` | quick | about 28 KiB |
//! | `fm_mod_cq15_vs_reference` | quick | about 14 KiB |
//! | `fm_mod_block_invariance` | quick | about 20 KiB |
//! | `ddc_equals_chain` | quick | about 12 KiB |
//! | `duc_equals_chain` | quick | about 16 KiB |
//! | `fm_demod_equals_chain` | quick | about 12 KiB |
//! | `ddc_partial_output_keeps_phase` | quick | about 10 KiB |
//! | `e2e_fm_rx_f32` | quick | about 43 KiB |
//! | `e2e_fm_rx_q15` | quick | about 37 KiB |
//! | `e2e_fm_txrx_f32` | quick | about 48 KiB |
//! | `e2e_fm_txrx_q15` | quick | about 37 KiB |
//! | `e2e_fm_rx_f32_1s`, `e2e_fm_rx_q15_1s` | full | about 211 KiB |
//! | `e2e_fm_txrx_f32_1s`, `e2e_fm_txrx_q15_1s` | full | about 217 KiB |
//! | `extremes_tx_kernels` | full | about 16 KiB |
//! | `dot_c32_vs_reference` | quick | about 6 KiB |
//! | `dot_cq15_vs_reference` | quick | about 2 KiB |
//! | `dot_real_vs_reference` | quick | about 5 KiB |
//! | `extremes_dot` | full | about 2 KiB |
//! | `block_ddc_vs_chain` | quick | about 30 KiB |
//! | `block_fm_demod_sinad` | quick | about 45 KiB |
//! | `block_spectrum_tone_level` | quick | about 34 KiB |
//! | `block_resampler_ppm_count` | quick | about 24 KiB |

// The stack-frame lint counts every `Result<_, Failure>` temporary of an
// unoptimised build separately (a `Failure` is about 200 bytes by
// specification), so check-heavy test bodies exceed 4 KiB in the lint's model
// although the optimised frames are small. See docs/decisions-log.md (M03).
#![allow(clippy::large_stack_frames)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::basic::Scale;
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Progress, Q15, Reference};
use sdrk_device::{FullScale, SampleFormat, formats};

use crate::arena::Arena;
use crate::encode::{decode_reference, encode};
use crate::r#gen;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::{bit_exact, compare_complex};
use crate::rng::Pcg32;
use crate::suite::{Failure, TestCase, Tier};

mod m05;
mod m06;
mod m07;
mod m08;
mod m09;

pub(crate) static CASES: [TestCase; 76] = [
    TestCase {
        name: "rng_known_answer",
        tier: Tier::Quick,
        run: rng_known_answer,
    },
    TestCase {
        name: "gen_tone_frequency",
        tier: Tier::Quick,
        run: gen_tone_frequency,
    },
    TestCase {
        name: "encode_decode_roundtrip_all_formats",
        tier: Tier::Quick,
        run: encode_decode_roundtrip_all_formats,
    },
    TestCase {
        name: "scale_fallback_vs_reference",
        tier: Tier::Quick,
        run: scale_fallback_vs_reference,
    },
    TestCase {
        name: "scale_block_invariance",
        tier: Tier::Full,
        run: scale_block_invariance,
    },
    TestCase {
        name: "convert_all_formats_vs_encoder",
        tier: Tier::Quick,
        run: m05::convert_all_formats_vs_encoder,
    },
    TestCase {
        name: "rotate_c32_vs_reference",
        tier: Tier::Quick,
        run: m05::rotate_c32_vs_reference,
    },
    TestCase {
        name: "rotate_cq15_vs_reference",
        tier: Tier::Quick,
        run: m05::rotate_cq15_vs_reference,
    },
    TestCase {
        name: "rotate_block_invariance",
        tier: Tier::Quick,
        run: m05::rotate_block_invariance,
    },
    TestCase {
        name: "fir_c32_vs_reference",
        tier: Tier::Quick,
        run: m05::fir_c32_vs_reference,
    },
    TestCase {
        name: "fir_cq15_vs_reference",
        tier: Tier::Quick,
        run: m05::fir_cq15_vs_reference,
    },
    TestCase {
        name: "fir_block_invariance",
        tier: Tier::Quick,
        run: m05::fir_block_invariance,
    },
    TestCase {
        name: "fir_impulse_response",
        tier: Tier::Quick,
        run: m05::fir_impulse_response,
    },
    TestCase {
        name: "extremes_convert_rotate_fir",
        tier: Tier::Full,
        run: m05::extremes_convert_rotate_fir,
    },
    TestCase {
        name: "fir_decimate_c32_vs_reference",
        tier: Tier::Quick,
        run: m06::fir_decimate_c32_vs_reference,
    },
    TestCase {
        name: "fir_decimate_cq15_vs_reference",
        tier: Tier::Quick,
        run: m06::fir_decimate_cq15_vs_reference,
    },
    TestCase {
        name: "fir_decimate_block_invariance",
        tier: Tier::Quick,
        run: m06::fir_decimate_block_invariance,
    },
    TestCase {
        name: "resample_c32_vs_reference",
        tier: Tier::Quick,
        run: m06::resample_c32_vs_reference,
    },
    TestCase {
        name: "resample_cq15_vs_reference",
        tier: Tier::Quick,
        run: m06::resample_cq15_vs_reference,
    },
    TestCase {
        name: "resample_real_vs_reference",
        tier: Tier::Quick,
        run: m06::resample_real_vs_reference,
    },
    TestCase {
        name: "resample_block_invariance",
        tier: Tier::Quick,
        run: m06::resample_block_invariance,
    },
    TestCase {
        name: "quad_demod_c32_vs_reference",
        tier: Tier::Quick,
        run: m06::quad_demod_c32_vs_reference,
    },
    TestCase {
        name: "quad_demod_cq15_vs_reference",
        tier: Tier::Quick,
        run: m06::quad_demod_cq15_vs_reference,
    },
    TestCase {
        name: "quad_demod_block_invariance",
        tier: Tier::Quick,
        run: m06::quad_demod_block_invariance,
    },
    TestCase {
        name: "deemphasis_f32_vs_reference",
        tier: Tier::Quick,
        run: m06::deemphasis_f32_vs_reference,
    },
    TestCase {
        name: "deemphasis_q15_vs_reference",
        tier: Tier::Quick,
        run: m06::deemphasis_q15_vs_reference,
    },
    TestCase {
        name: "deemphasis_block_invariance",
        tier: Tier::Quick,
        run: m06::deemphasis_block_invariance,
    },
    TestCase {
        name: "dc_block_c32_vs_reference",
        tier: Tier::Quick,
        run: m06::dc_block_c32_vs_reference,
    },
    TestCase {
        name: "dc_block_cq15_vs_reference",
        tier: Tier::Quick,
        run: m06::dc_block_cq15_vs_reference,
    },
    TestCase {
        name: "dc_block_block_invariance",
        tier: Tier::Quick,
        run: m06::dc_block_block_invariance,
    },
    TestCase {
        name: "window_vs_reference",
        tier: Tier::Quick,
        run: m06::window_vs_reference,
    },
    TestCase {
        name: "window_generators",
        tier: Tier::Quick,
        run: m06::window_generators,
    },
    TestCase {
        name: "mag_sq_vs_reference",
        tier: Tier::Quick,
        run: m06::mag_sq_vs_reference,
    },
    TestCase {
        name: "power_db_vs_reference",
        tier: Tier::Quick,
        run: m06::power_db_vs_reference,
    },
    TestCase {
        name: "extremes_rx_kernels",
        tier: Tier::Full,
        run: m06::extremes_rx_kernels,
    },
    TestCase {
        name: "fft_c32_vs_reference_64",
        tier: Tier::Quick,
        run: m07::fft_c32_vs_reference_64,
    },
    TestCase {
        name: "fft_c32_vs_reference_1024",
        tier: Tier::Quick,
        run: m07::fft_c32_vs_reference_1024,
    },
    TestCase {
        name: "fft_cq15_fixed_vs_reference_1024",
        tier: Tier::Quick,
        run: m07::fft_cq15_fixed_vs_reference_1024,
    },
    TestCase {
        name: "fft_cq15_bfp_vs_reference_1024",
        tier: Tier::Quick,
        run: m07::fft_cq15_bfp_vs_reference_1024,
    },
    TestCase {
        name: "fft_roundtrip_c32",
        tier: Tier::Quick,
        run: m07::fft_roundtrip_c32,
    },
    TestCase {
        name: "fft_roundtrip_cq15_bfp",
        tier: Tier::Quick,
        run: m07::fft_roundtrip_cq15_bfp,
    },
    TestCase {
        name: "rfft_vs_reference_1024",
        tier: Tier::Quick,
        run: m07::rfft_vs_reference_1024,
    },
    TestCase {
        name: "irfft_vs_reference_1024",
        tier: Tier::Quick,
        run: m07::irfft_vs_reference_1024,
    },
    TestCase {
        name: "rfft_roundtrip_f32",
        tier: Tier::Quick,
        run: m07::rfft_roundtrip_f32,
    },
    TestCase {
        name: "fft_weak_tone_bfp",
        tier: Tier::Full,
        run: m07::fft_weak_tone_bfp,
    },
    TestCase {
        name: "fft_parseval_c32",
        tier: Tier::Quick,
        run: m07::fft_parseval_c32,
    },
    TestCase {
        name: "extremes_fft",
        tier: Tier::Full,
        run: m07::extremes_fft,
    },
    TestCase {
        name: "convert_out_all_formats_roundtrip",
        tier: Tier::Quick,
        run: m08::convert_out_all_formats_roundtrip,
    },
    TestCase {
        name: "fir_interpolate_c32_vs_reference",
        tier: Tier::Quick,
        run: m08::fir_interpolate_c32_vs_reference,
    },
    TestCase {
        name: "fir_interpolate_cq15_vs_reference",
        tier: Tier::Quick,
        run: m08::fir_interpolate_cq15_vs_reference,
    },
    TestCase {
        name: "fir_interpolate_real_vs_reference",
        tier: Tier::Quick,
        run: m08::fir_interpolate_real_vs_reference,
    },
    TestCase {
        name: "fir_interpolate_block_invariance",
        tier: Tier::Quick,
        run: m08::fir_interpolate_block_invariance,
    },
    TestCase {
        name: "fm_mod_c32_vs_reference",
        tier: Tier::Quick,
        run: m08::fm_mod_c32_vs_reference,
    },
    TestCase {
        name: "fm_mod_cq15_vs_reference",
        tier: Tier::Quick,
        run: m08::fm_mod_cq15_vs_reference,
    },
    TestCase {
        name: "fm_mod_block_invariance",
        tier: Tier::Quick,
        run: m08::fm_mod_block_invariance,
    },
    TestCase {
        name: "ddc_equals_chain",
        tier: Tier::Quick,
        run: m08::ddc_equals_chain,
    },
    TestCase {
        name: "duc_equals_chain",
        tier: Tier::Quick,
        run: m08::duc_equals_chain,
    },
    TestCase {
        name: "fm_demod_equals_chain",
        tier: Tier::Quick,
        run: m08::fm_demod_equals_chain,
    },
    TestCase {
        name: "ddc_partial_output_keeps_phase",
        tier: Tier::Quick,
        run: m08::ddc_partial_output_keeps_phase,
    },
    TestCase {
        name: "e2e_fm_rx_f32",
        tier: Tier::Quick,
        run: m08::e2e_fm_rx_f32,
    },
    TestCase {
        name: "e2e_fm_rx_q15",
        tier: Tier::Quick,
        run: m08::e2e_fm_rx_q15,
    },
    TestCase {
        name: "e2e_fm_txrx_f32",
        tier: Tier::Quick,
        run: m08::e2e_fm_txrx_f32,
    },
    TestCase {
        name: "e2e_fm_txrx_q15",
        tier: Tier::Quick,
        run: m08::e2e_fm_txrx_q15,
    },
    TestCase {
        name: "e2e_fm_rx_f32_1s",
        tier: Tier::Full,
        run: m08::e2e_fm_rx_f32_1s,
    },
    TestCase {
        name: "e2e_fm_rx_q15_1s",
        tier: Tier::Full,
        run: m08::e2e_fm_rx_q15_1s,
    },
    TestCase {
        name: "e2e_fm_txrx_f32_1s",
        tier: Tier::Full,
        run: m08::e2e_fm_txrx_f32_1s,
    },
    TestCase {
        name: "e2e_fm_txrx_q15_1s",
        tier: Tier::Full,
        run: m08::e2e_fm_txrx_q15_1s,
    },
    TestCase {
        name: "extremes_tx_kernels",
        tier: Tier::Full,
        run: m08::extremes_tx_kernels,
    },
    TestCase {
        name: "dot_c32_vs_reference",
        tier: Tier::Quick,
        run: m09::dot_c32_vs_reference,
    },
    TestCase {
        name: "dot_cq15_vs_reference",
        tier: Tier::Quick,
        run: m09::dot_cq15_vs_reference,
    },
    TestCase {
        name: "dot_real_vs_reference",
        tier: Tier::Quick,
        run: m09::dot_real_vs_reference,
    },
    TestCase {
        name: "extremes_dot",
        tier: Tier::Full,
        run: m09::extremes_dot,
    },
    TestCase {
        name: "block_ddc_vs_chain",
        tier: Tier::Quick,
        run: m09::block_ddc_vs_chain,
    },
    TestCase {
        name: "block_fm_demod_sinad",
        tier: Tier::Quick,
        run: m09::block_fm_demod_sinad,
    },
    TestCase {
        name: "block_spectrum_tone_level",
        tier: Tier::Quick,
        run: m09::block_spectrum_tone_level,
    },
    TestCase {
        name: "block_resampler_ppm_count",
        tier: Tier::Quick,
        run: m09::block_resampler_ppm_count,
    },
];

fn ensure(cond: bool, f: impl FnOnce() -> Failure) -> Result<(), Failure> {
    if cond { Ok(()) } else { Err(f()) }
}

fn rng_known_answer(_arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "rng_known_answer";
    // Reference values of `pcg32-demo` (pcg_basic.c), seed 42, stream 54.
    const EXPECTED: [u32; 6] = [
        0xa15c_02b7,
        0x7b47_f409,
        0xba1d_3330,
        0x83d2_f293,
        0xbfa4_784b,
        0xcbed_606e,
    ];
    let mut rng = Pcg32::new(42, 54);
    for (i, want) in EXPECTED.iter().enumerate() {
        let got = rng.next_u32();
        ensure(got == *want, || {
            Failure::new(CASE, "wrong output")
                .detail(format_args!("draw {i}: got {got:#010x}, want {want:#010x}"))
        })?;
    }
    let mut rng = Pcg32::new(1, 1);
    for _ in 0..1000 {
        let x = rng.next_f64();
        ensure((0.0..1.0).contains(&x), || {
            Failure::new(CASE, "next_f64 out of range").detail(format_args!("{x}"))
        })?;
        let b = rng.below(7);
        ensure(b < 7, || {
            Failure::new(CASE, "below out of range").detail(format_args!("{b}"))
        })?;
    }
    let mut a = Pcg32::new(5, 6);
    let mut b = a.clone();
    for _ in 0..777 {
        a.next_u32();
    }
    b.advance(777);
    ensure(a == b, || {
        Failure::new(CASE, "advance differs from stepping")
    })
}

fn gen_tone_frequency(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "gen_tone_frequency";
    const N: usize = 256;
    const BIN: usize = 37;
    let mut c = arena.carver();
    let x = c.take::<Complex64>(N)?;
    let chunked = c.take::<Complex64>(N)?;
    let twiddle = c.take::<Complex64>(N)?;
    r#gen::tone(x, BIN as f64 / N as f64, 1.0, 0.0, 0);
    // Same signal in chunks of seven (the last one shorter).
    let mut start = 0usize;
    for piece in chunked.chunks_mut(7) {
        r#gen::tone(piece, BIN as f64 / N as f64, 1.0, 0.0, start as u64);
        start += piece.len();
    }
    let same = compare_complex(x, chunked.iter().copied(), 0);
    ensure(same.first_mismatch.is_none(), || {
        Failure::new(CASE, "chunked tone differs")
            .detail(format_args!("index {:?}", same.first_mismatch))
    })?;
    for (m, w) in twiddle.iter_mut().enumerate() {
        let a = -2.0 * PI * m as f64 / N as f64;
        *w = Complex64::new(libm::cos(a), libm::sin(a));
    }
    let mut peak = (0usize, 0.0f64);
    let mut mags = [0.0f64; N];
    for (k, mag) in mags.iter_mut().enumerate() {
        let mut acc = Complex64::new(0.0, 0.0);
        for (n, s) in x.iter().enumerate() {
            let w = twiddle.get(k * n % N).copied().unwrap_or_default();
            acc += s * w;
        }
        *mag = libm::sqrt(acc.re * acc.re + acc.im * acc.im);
        if *mag > peak.1 {
            peak = (k, *mag);
        }
    }
    ensure(peak.0 == BIN, || {
        Failure::new(CASE, "peak at the wrong bin").detail(format_args!("bin {}", peak.0))
    })?;
    for (k, mag) in mags.iter().enumerate().filter(|(k, _)| *k != BIN) {
        let db = 20.0 * libm::log10(peak.1 / mag.max(1e-300));
        ensure(db >= 100.0, || {
            Failure::new(CASE, "leakage too high").detail(format_args!("bin {k}: {db} dB"))
        })?;
    }
    Ok(())
}

/// Formats exercised by the round-trip case (everything but `Bits1`).
fn all_formats() -> [SampleFormat; 12] {
    [
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
    ]
}

/// Largest allowed round-trip error: half an LSB (f32 containers: rounding of f32).
fn half_lsb(f: &SampleFormat) -> f64 {
    if f.container() == sdrk_device::Container::F32 {
        return 1e-7;
    }
    let fs = match f.full_scale() {
        FullScale::Explicit(v) => f64::from(v),
        FullScale::PowerOfTwo => (1u64 << (f.bits() - 1)) as f64,
    };
    0.5 / fs + 1e-12
}

fn encode_decode_roundtrip_all_formats(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "encode_decode_roundtrip_all_formats";
    const N: usize = 512;
    let mut c = arena.carver();
    let samples = c.take::<Complex64>(N)?;
    let bytes = c.take::<u8>(N * 8)?;
    let decoded = c.take::<Complex64>(N)?;
    let mut rng = Pcg32::new(99, 1);
    for s in samples.iter_mut() {
        let re = (rng.next_f64() * 2.0 - 1.0) * 0.99;
        let im = (rng.next_f64() * 2.0 - 1.0) * 0.99;
        *s = Complex64::new(re, im);
    }
    for (i, format) in all_formats().iter().enumerate() {
        let real = format.layout() == sdrk_device::Layout::Real;
        let written = encode(format, samples, bytes)?;
        let used = bytes.get(..written).unwrap_or_default();
        let count = decode_reference(format, used, decoded)?;
        ensure(count == N, || {
            Failure::new(CASE, "wrong sample count").detail(format_args!("format {i}: {count}"))
        })?;
        let tol = half_lsb(format);
        for (k, (want, got)) in samples.iter().zip(decoded.iter()).enumerate() {
            let want_im = if real { 0.0 } else { want.im };
            let err = libm::fabs(want.re - got.re).max(libm::fabs(want_im - got.im));
            ensure(err <= tol, || {
                Failure::new(CASE, "round-trip error too large")
                    .detail(format_args!("format {i}, sample {k}: {err} > {tol}"))
            })?;
        }
    }
    // Spot values.
    let one = |format: &SampleFormat, re: f64, im: f64, want: &[u8]| -> Result<(), Failure> {
        let mut out = [0u8; 8];
        let n = encode(format, &[Complex64::new(re, im)], &mut out)?;
        ensure(out.get(..n) == Some(want), || {
            Failure::new(CASE, "spot value differs").detail(format_args!("{re} {im}"))
        })
    };
    one(&formats::CU8, 0.0, -1.0, &[128, 0])?;
    one(&formats::CU8, 1.0, 0.0, &[255, 128])?;
    one(&formats::CS16, 1.0, -1.0, &[0xFF, 0x7F, 0x00, 0x80])?;
    one(&formats::CS12_LSB16, 0.5, 0.0, &[0x00, 0x04, 0, 0])?;
    one(&formats::SC16_Q11, -1.0, 0.0, &[0x00, 0xF8, 0, 0])?;
    one(
        &formats::PACKED12,
        f64::from(1.0f32 / 2048.0),
        -1.0 / 2048.0,
        &[0x01, 0xF0, 0xFF],
    )
}

fn random_cq15(rng: &mut Pcg32) -> CQ15 {
    CQ15 {
        re: Q15((rng.next_u32() >> 16) as i16),
        im: Q15((rng.next_u32() >> 16) as i16),
    }
}

fn scale_fallback_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "scale_fallback_vs_reference";
    const N: usize = 1024;
    let mut rng = Pcg32::new(7, 3);
    {
        let mut c = arena.carver();
        let input = c.take::<CQ15>(N)?;
        let fb = c.take::<CQ15>(N)?;
        let rf = c.take::<CQ15>(N)?;
        let fac = c.take::<CQ15>(N)?;
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng);
        }
        for gain in [Q15(0), Q15(1), Q15(16384), Q15::MAX, Q15::MIN, Q15(-12345)] {
            for extremes in [false, true] {
                if extremes {
                    r#gen::extreme::full_scale_alternating(input);
                    if let Some(first) = input.first_mut() {
                        *first = CQ15 {
                            re: Q15::MIN,
                            im: Q15::MIN,
                        };
                    }
                }
                Scale::<CQ15>::scale(&Fallback, input, fb, gain)?;
                Scale::<CQ15>::scale(&Reference, input, rf, gain)?;
                Scale::<CQ15>::scale(&Sdrk, input, fac, gain)?;
                for (name, got) in [("fallback", &*fb), ("facade", &*fac)] {
                    if let Some(i) = bit_exact(rf, got) {
                        return Err(Failure::new(CASE, "not bit-exact to the reference")
                            .detail(format_args!("{name}, gain {}, index {i}", gain.0)));
                    }
                }
            }
        }
    }
    let mut c = arena.carver();
    let input = c.take::<C32>(N)?;
    let fb = c.take::<C32>(N)?;
    let rf = c.take::<C32>(N)?;
    let want = c.take::<Complex64>(N)?;
    for s in input.iter_mut() {
        let re = (rng.next_f64() * 2.0 - 1.0) as f32;
        let im = (rng.next_f64() * 2.0 - 1.0) as f32;
        *s = C32::new(re, im);
    }
    let gain = 0.37f32;
    Scale::<C32>::scale(&Fallback, input, fb, gain)?;
    Scale::<C32>::scale(&Reference, input, rf, gain)?;
    for (w, i) in want.iter_mut().zip(input.iter()) {
        *w = Complex64::new(
            f64::from(i.re) * f64::from(gain),
            f64::from(i.im) * f64::from(gain),
        );
    }
    for (name, got) in [("fallback", &*fb), ("reference", &*rf)] {
        let cmp = compare_complex(want, got.iter().map(|s| s.to_c64()), 0);
        ensure(cmp.snr_db >= 140.0, || {
            Failure::new(CASE, "scale C32 below 140 dB")
                .detail(format_args!("{name}: {} dB", cmp.snr_db))
        })?;
    }
    Ok(())
}

fn same_rate<T: ComplexSample>(
    input: &[T],
    output: &mut [T],
    kernel: impl FnOnce(&[T], &mut [T]) -> sdrk_core::Result<()>,
) -> sdrk_core::Result<Progress> {
    let n = input.len().min(output.len());
    let internal = || sdrk_core::Error::Internal("slice split");
    let (i, _) = input.split_at_checked(n).ok_or_else(internal)?;
    let (o, _) = output.split_at_mut_checked(n).ok_or_else(internal)?;
    kernel(i, o)?;
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

fn scale_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "scale_block_invariance";
    const N: usize = 4096;
    let mut rng = Pcg32::new(11, 5);
    {
        let mut c = arena.carver();
        let input = c.take::<CQ15>(N)?;
        let whole = c.take::<CQ15>(N)?;
        let chunked = c.take::<CQ15>(N)?;
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng);
        }
        let report = check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(1, 97),
            || (),
            |(), i, o| same_rate(i, o, |i, o| Scale::<CQ15>::scale(&Sdrk, i, o, Q15(23456))),
        )?;
        ensure(report.outputs == N, || {
            Failure::new(CASE, "wrong output count").detail(format_args!("{}", report.outputs))
        })?;
    }
    let mut c = arena.carver();
    let input = c.take::<C32>(N)?;
    let whole = c.take::<C32>(N)?;
    let chunked = c.take::<C32>(N)?;
    for s in input.iter_mut() {
        *s = C32::new(rng.next_f64() as f32, -(rng.next_f64() as f32));
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(2, 53),
        || (),
        |(), i, o| same_rate(i, o, |i, o| Scale::<C32>::scale(&Sdrk, i, o, 0.8)),
    )?;
    Ok(())
}
