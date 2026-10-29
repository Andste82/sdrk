//! The measured kernels: for every entry of the benchmark table, the buffers, the call of our
//! kernel, the calls of the comparisons and the accuracy against the f64 reference.
//!
//! The binary `count` drives them; `cargo xtask count` runs the binary under the instruction
//! counters and decides which comparison gates which level.

use std::hint::black_box;
use std::sync::Arc;

use cref_sys::cmsis_dsp::{CmsisCfftF32, CmsisCfftQ15, CmsisRfftFastF32};
use cref_sys::{
    CmsisBiquad, Fftw, FftwR2c, GrAgcCc, GrFmDeemph, GrFrequencyModulatorFc, GrPwrSquelchCc,
    GrRotator, KfrBiquad, KissFft16, LiquidDcBlocker, LiquidDcBlockerCrcf, Ne10Int16, cmsis_dsp,
};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use sdrk::kernels::agc::{Agc, AgcState};
use sdrk::kernels::basic::Scale;
use sdrk::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState};
use sdrk::kernels::convert::{Convert, ConvertOut, ConvertOutReal, ConvertReal};
use sdrk::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk::kernels::demod::{
    DcBlock, DcBlockReal, DcBlockRealState, DcBlockState, Deemphasis, DeemphasisState, QuadDemod,
    QuadDemodState,
};
use sdrk::kernels::dot::{Dot, DotReal};
use sdrk::kernels::fft::{Direction, Fft, FftPlan, RealFft, RealFftPlan, Scaling};
use sdrk::kernels::filter::{Fir, FirState};
use sdrk::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use sdrk::kernels::mix::{Nco, Rotate};
use sdrk::kernels::modulate::{FmMod, FmModState};
use sdrk::kernels::spectrum::{Mag, MagSq, PowerDb, Window};
use sdrk::kernels::squelch::{Squelch, SquelchState};
use sdrk::{C32, CQ15, ComplexSample, Fallback, Q15, Rate, Sdrk};
use sdrk_core::Reference;
use sdrk_device::{Align, Container, Endian, Layout as DataLayout, SampleFormat, formats};
use volk_sys::{lv_16sc_t, lv_32fc_t};

mod fir_family;

use fir_family::{Entry, FirFamK, Mode};

/// Samples per call unless overridden.
pub const N: usize = 4096;

/// An implementation the binary can run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Impl {
    /// Our kernel through the facade (the SIMD backend unless the build forces the fallback).
    Ours,
    /// Our portable fallback.
    Fallback,
    /// VOLK, called as GNU Radio calls it.
    Volk,
    /// The loop of a GNU Radio block that is not a VOLK kernel (`agc_cc`, `pwr_squelch_cc`),
    /// compiled as GNU Radio compiles it.
    GnuRadio,
    /// rustfft, `FftPlanner` (it picks the best planner of the CPU).
    RustFft,
    /// FFTW, single precision, `FFTW_ESTIMATE`, out of place (as GNU Radio's `fft_vcc`).
    Fftw,
    /// FFTW as [`Impl::Fftw`], but in place.
    FftwInPlace,
    /// KissFFT with `FIXED_POINT=16`.
    KissFft,
    /// Ne10's int16 NEON FFT.
    Ne10,
    /// KFR's biquad cascade (`kfr::iir`), compiled for the level of the build.
    Kfr,
    /// CMSIS-DSP's biquad cascades in generic C, compiled for the level of the build.
    Cmsis,
    /// liquid-dsp's DC blocker (`iirfilt_rrrf`/`iirfilt_crcf` `create_dc_blocker`, portable C).
    Liquid,
    /// Only the copy that restores the input of an in-place transform before every call. Its
    /// count is the baseline that xtask subtracts from every FFT comparison.
    Restore,
}

impl Impl {
    /// All implementations, in the order of `count info`.
    pub const ALL: [Impl; 13] = [
        Impl::Ours,
        Impl::Fallback,
        Impl::Volk,
        Impl::GnuRadio,
        Impl::RustFft,
        Impl::Fftw,
        Impl::FftwInPlace,
        Impl::KissFft,
        Impl::Ne10,
        Impl::Kfr,
        Impl::Cmsis,
        Impl::Liquid,
        Impl::Restore,
    ];

    /// The command-line name.
    pub fn name(self) -> &'static str {
        match self {
            Impl::Ours => "ours",
            Impl::Fallback => "fallback",
            Impl::Volk => "volk",
            Impl::GnuRadio => "gnuradio",
            Impl::RustFft => "rustfft",
            Impl::Fftw => "fftw",
            Impl::FftwInPlace => "fftw-in-place",
            Impl::KissFft => "kissfft",
            Impl::Ne10 => "ne10",
            Impl::Kfr => "kfr",
            Impl::Cmsis => "cmsis",
            Impl::Liquid => "liquid",
            Impl::Restore => "restore",
        }
    }

    /// Parses the command-line name.
    pub fn parse(s: &str) -> Option<Impl> {
        Impl::ALL.into_iter().find(|i| i.name() == s)
    }
}

/// Buffer layout of a measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Samples per call (ignored by the FFTs, whose size is part of the name).
    pub n: usize,
    /// Every buffer starts one complex sample after a 64-byte boundary instead of on it.
    pub misaligned: bool,
}

impl Default for Layout {
    fn default() -> Self {
        Layout {
            n: N,
            misaligned: false,
        }
    }
}

impl Layout {
    /// The byte offset from a 64-byte boundary of a buffer whose samples have `sample_bytes`
    /// bytes: 0, or one sample when misaligned.
    fn skew(self, sample_bytes: usize) -> usize {
        if self.misaligned { sample_bytes } else { 0 }
    }

    /// The skew of a buffer whose element `first` (of `sample_bytes` bytes) is where the
    /// kernel's input starts: that element, not the buffer, is aligned or one sample off.
    fn skew_before(self, sample_bytes: usize, first: usize) -> usize {
        (self.skew(sample_bytes) + 64 - first * sample_bytes % 64) % 64
    }
}

/// PCG32 (the generator of the testkit), seeded per kernel with `1000 + index`.
pub struct Pcg32(u64);

impl Pcg32 {
    /// A generator for `seed`.
    pub fn new(seed: u64) -> Self {
        let mut r = Pcg32(0);
        r.next_u32();
        r.0 = r.0.wrapping_add(seed);
        r.next_u32();
        r
    }

    /// The next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        (f64::from(self.next_u32()) / 2_147_483_648.0 - 1.0) as f32
    }
}

/// Bytes of readable padding in front of every [`Aligned`] buffer: enough for an access that
/// rounds a pointer into the buffer down to VOLK's alignment (at most 64 bytes).
const LEAD: usize = 64;

/// A vector whose data starts `skew` bytes after a 64-byte boundary (0: VOLK takes its aligned
/// kernels, and our loads do not straddle cache lines), with at least [`LEAD`] bytes of the
/// same allocation in front of it.
pub struct Aligned<T> {
    v: Vec<T>,
    off: usize,
    len: usize,
}

impl<T: Copy + Default> Aligned<T> {
    /// `len` default-initialised elements on a 64-byte boundary.
    pub fn new(len: usize) -> Self {
        Self::skewed(len, 0)
    }

    /// `len` default-initialised elements, `skew` bytes after a 64-byte boundary.
    ///
    /// # Panics
    /// If `skew` is not a multiple of the element size.
    pub fn skewed(len: usize, skew: usize) -> Self {
        let size = std::mem::size_of::<T>().max(1);
        assert_eq!(
            skew % size,
            0,
            "the skew must be a multiple of the element size"
        );
        let lead = LEAD.div_ceil(size);
        let pad = lead + 64 / size + 1;
        let v = vec![T::default(); len + pad];
        let off = (lead..pad)
            .find(|o| (v.as_ptr() as usize + o * size) % 64 == skew % 64)
            .unwrap_or(lead);
        Aligned { v, off, len }
    }
}

impl<T> std::ops::Deref for Aligned<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.v[self.off..self.off + self.len]
    }
}

impl<T> std::ops::DerefMut for Aligned<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.v[self.off..self.off + self.len]
    }
}

fn noise_c32(rng: &mut Pcg32, n: usize, amp: f32, skew: usize) -> Aligned<C32> {
    let mut v = Aligned::<C32>::skewed(n, skew);
    for s in v.iter_mut() {
        *s = C32::new(rng.unit() * amp, rng.unit() * amp);
    }
    v
}

fn noise_cq15(rng: &mut Pcg32, n: usize, amp: f32, skew: usize) -> Aligned<CQ15> {
    let mut v = Aligned::<CQ15>::skewed(n, skew);
    for s in v.iter_mut() {
        *s = CQ15 {
            re: Q15((rng.unit() * amp) as i16),
            im: Q15((rng.unit() * amp) as i16),
        };
    }
    v
}

/// Signal-to-noise ratio in dB of `got` against `want`; 999 for identical signals.
pub fn snr_db(want: &[f64], got: &[f64]) -> f64 {
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (w, g) in want.iter().zip(got) {
        sig += w * w;
        err += (w - g) * (w - g);
    }
    if err == 0.0 {
        999.0
    } else {
        10.0 * (sig / err).log10()
    }
}

fn f32s(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

fn c32s(x: &[C32]) -> Vec<f64> {
    x.iter()
        .flat_map(|v| [f64::from(v.re), f64::from(v.im)])
        .collect()
}

/// The components of `x` in units of `scale` (1.0: raw Q15 integers).
fn cq15s(x: &[CQ15], scale: f64) -> Vec<f64> {
    x.iter()
        .flat_map(|v| [f64::from(v.re.0) * scale, f64::from(v.im.0) * scale])
        .collect()
}

/// `x` as C32 in Q15 units (exact).
fn cq15_as_c32(x: &[CQ15]) -> Vec<C32> {
    x.iter()
        .map(|v| C32::new(f32::from(v.re.0), f32::from(v.im.0)))
        .collect()
}

/// A prepared measurement.
pub trait Kernel {
    /// Runs the kernel `k` times on the same buffers with implementation `which`, which must be
    /// one that [`Kernel::supports`].
    fn run(&mut self, which: Impl, k: usize);
    /// The SNR in dB of one run of `which` against the f64 reference.
    fn accuracy(&mut self, which: Impl) -> f64;
    /// Whether this build can run `which` for this kernel.
    fn supports(&self, which: Impl) -> bool;
    /// What `which` calls, for the reports.
    fn describe(&self, which: Impl) -> &'static str;
    /// The VOLK functions [`Impl::Volk`] calls.
    fn volk_functions(&self) -> &'static [&'static str] {
        &[]
    }
}

/// All kernel names. The position gives the seed of the input (`1000 + index`), so new kernels
/// are appended; a `<kernel>_fallback` entry takes the seed of `<kernel>`, so that both entries
/// filter the same input.
pub const KERNELS: [&str; 120] = [
    "convert_cs8",
    "convert_cs16",
    "convert_out_cs16",
    "rotate",
    "fir",
    "fir_decimate",
    "mag_sq",
    "quad_demod",
    "fm_mod",
    "fft_64",
    "fft_1024",
    "fft_4096",
    "fft_cq15_1024",
    "convert_cu8",
    "convert_out_cs8",
    "convert_out_cu8",
    "rotate_cq15",
    "mag_sq_q15",
    "fft_cq15_1024_default",
    "fir_interpolate",
    "resample",
    "fft_cq15_1024_full_scale",
    "rfft_1024",
    "quad_demod_cq15",
    "squelch",
    "squelch_cq15",
    "mag",
    "mag_q15",
    "agc",
    "agc_cq15",
    "dc_block_real",
    "dc_block_real_q15",
    "biquad",
    "biquad_cq15",
    "biquad_real",
    "biquad_real_q15",
    "biquad_fallback",
    "biquad_cq15_fallback",
    "biquad_real_fallback",
    "biquad_real_q15_fallback",
    "convert_out_real_rs16",
    "convert_out_real_rs8",
    "convert_out_real_ru8",
    "deemphasis",
    "deemphasis_q15",
    "dc_block",
    "dc_block_cq15",
    "window",
    "window_cq15",
    "power_db",
    "power_db_q15",
    "fir_interpolate_real",
    "resample_real",
    "fir_interpolate_real_q15",
    "resample_real_q15",
    "fm_mod_cq15",
    "convert_real_rs16",
    "convert_cq15_cs8",
    "convert_cq15_cu8",
    "dot",
    "dot_cq15",
    "dot_real",
    "dot_real_q15",
    "scale",
    "scale_cq15",
    "dc_block_real_fallback",
    "deemphasis_fallback",
    "dc_block_fallback",
    "fft_64_fallback",
    "fft_1024_fallback",
    "fft_4096_fallback",
    "fft_cq15_1024_fallback",
    "fft_cq15_1024_default_fallback",
    "rfft_1024_fallback",
    "convert_cs8_fallback",
    "convert_cs16_fallback",
    "convert_cu8_fallback",
    "convert_real_rs16_fallback",
    "convert_cq15_cs8_fallback",
    "convert_cq15_cu8_fallback",
    "convert_out_cs16_fallback",
    "convert_out_cs8_fallback",
    "convert_out_cu8_fallback",
    "convert_out_real_rs16_fallback",
    "convert_out_real_rs8_fallback",
    "convert_out_real_ru8_fallback",
    "dot_fallback",
    "dot_cq15_fallback",
    "dot_real_fallback",
    "dot_real_q15_fallback",
    "scale_fallback",
    "scale_cq15_fallback",
    "rotate_fallback",
    "rotate_cq15_fallback",
    "fm_mod_fallback",
    "fm_mod_cq15_fallback",
    "fir_cq15",
    "fir_decimate_cq15",
    "fir_interpolate_cq15",
    "resample_cq15",
    "fir_fallback",
    "fir_cq15_fallback",
    "fir_decimate_fallback",
    "fir_decimate_cq15_fallback",
    "fir_interpolate_fallback",
    "fir_interpolate_cq15_fallback",
    "resample_fallback",
    "resample_cq15_fallback",
    "fir_interpolate_real_fallback",
    "fir_interpolate_real_q15_fallback",
    "resample_real_fallback",
    "resample_real_q15_fallback",
    "mag_sq_fallback",
    "mag_sq_q15_fallback",
    "mag_fallback",
    "mag_q15_fallback",
    "window_fallback",
    "window_cq15_fallback",
    "power_db_fallback",
    "power_db_q15_fallback",
];

/// Prepares the measurement `name` with `layout` (the FFT sizes are fixed by the name).
pub fn prepare(name: &str, layout: Layout) -> Option<Box<dyn Kernel>> {
    KERNELS.iter().position(|k| *k == name)?;
    let seed_of = name.strip_suffix("_fallback").unwrap_or(name);
    let idx = KERNELS.iter().position(|k| *k == seed_of)? as u64;
    let (simd, fallback) = (BiquadEntry::Simd, BiquadEntry::Fallback);
    let entry = if name.ends_with("_fallback") {
        fallback
    } else {
        simd
    };
    let c32 = |c: C32| (c.re, c.im);
    let cq15 = |c: CQ15| (c.re.0, c.im.0);
    let mut rng = Pcg32::new(1000 + idx);
    let rng = &mut rng;
    Some(match name {
        "convert_cs8"
        | "convert_cs16"
        | "convert_cu8"
        | "convert_real_rs16"
        | "convert_cs8_fallback"
        | "convert_cs16_fallback"
        | "convert_cu8_fallback"
        | "convert_real_rs16_fallback" => {
            Box::new(ConvertIn::new(rng, layout, in_format(seed_of), entry))
        }
        "convert_cq15_cs8"
        | "convert_cq15_cu8"
        | "convert_cq15_cs8_fallback"
        | "convert_cq15_cu8_fallback" => {
            Box::new(ConvertQ15In::new(rng, layout, in_format(seed_of), entry))
        }
        "convert_out_cs16"
        | "convert_out_cs8"
        | "convert_out_cu8"
        | "convert_out_cs16_fallback"
        | "convert_out_cs8_fallback"
        | "convert_out_cu8_fallback" => {
            Box::new(ConvertOutK::new(rng, layout, out_format(seed_of), entry))
        }
        "convert_out_real_rs16"
        | "convert_out_real_rs8"
        | "convert_out_real_ru8"
        | "convert_out_real_rs16_fallback"
        | "convert_out_real_rs8_fallback"
        | "convert_out_real_ru8_fallback" => {
            Box::new(ConvertOutK::real(rng, layout, out_format(seed_of), entry))
        }
        "rotate" => Box::new(RotateK::new(rng, layout, simd)),
        "rotate_cq15" => Box::new(RotateQ15K::new(rng, layout, simd)),
        "rotate_fallback" => Box::new(RotateK::new(rng, layout, fallback)),
        "rotate_cq15_fallback" => Box::new(RotateQ15K::new(rng, layout, fallback)),
        "fir" => Box::new(FirK::new(rng, layout)),
        "fir_decimate" => Box::new(DecimateK::new(rng, layout)),
        "fir_interpolate" => Box::new(PolyK::new(rng, layout, INTERP_TAPS, INTERP, 1)),
        "resample" => Box::new(PolyK::new(
            rng,
            layout,
            RESAMPLE_TAPS,
            RESAMPLE_UP,
            RESAMPLE_DOWN,
        )),
        "fir_interpolate_real" => {
            Box::new(PolyRealK::<f32>::new(rng, layout, INTERP_TAPS, INTERP, 1))
        }
        "resample_real" => Box::new(PolyRealK::<f32>::new(
            rng,
            layout,
            RESAMPLE_TAPS,
            RESAMPLE_UP,
            RESAMPLE_DOWN,
        )),
        "fir_interpolate_real_q15" => {
            Box::new(PolyRealK::<Q15>::new(rng, layout, INTERP_TAPS, INTERP, 1))
        }
        "resample_real_q15" => Box::new(PolyRealK::<Q15>::new(
            rng,
            layout,
            RESAMPLE_TAPS,
            RESAMPLE_UP,
            RESAMPLE_DOWN,
        )),
        "mag_sq" => Box::new(MagSqK::new(rng, layout, simd)),
        "mag_sq_q15" => Box::new(MagSqQ15K::new(rng, layout, simd)),
        "mag_sq_fallback" => Box::new(MagSqK::new(rng, layout, fallback)),
        "mag_sq_q15_fallback" => Box::new(MagSqQ15K::new(rng, layout, fallback)),
        "quad_demod" => Box::new(QuadK::new(rng, layout)),
        "quad_demod_cq15" => Box::new(QuadQ15K::new(rng, layout)),
        "fm_mod" => Box::new(FmModK::new(rng, layout, simd)),
        "fm_mod_cq15" => Box::new(FmModQ15K::new(rng, layout, simd)),
        "fm_mod_fallback" => Box::new(FmModK::new(rng, layout, fallback)),
        "fm_mod_cq15_fallback" => Box::new(FmModQ15K::new(rng, layout, fallback)),
        "squelch" => Box::new(SquelchK::<C32>::new(rng, layout)),
        "squelch_cq15" => Box::new(SquelchK::<CQ15>::new(rng, layout)),
        "mag" => Box::new(MagK::new(rng, layout, simd)),
        "mag_q15" => Box::new(MagQ15K::new(rng, layout, simd)),
        "window" => Box::new(WindowK::new(rng, layout, simd)),
        "window_cq15" => Box::new(WindowQ15K::new(rng, layout, simd)),
        "power_db" => Box::new(PowerDbK::new(rng, layout, simd)),
        "power_db_q15" => Box::new(PowerDbQ15K::new(rng, layout, simd)),
        "mag_fallback" => Box::new(MagK::new(rng, layout, fallback)),
        "mag_q15_fallback" => Box::new(MagQ15K::new(rng, layout, fallback)),
        "window_fallback" => Box::new(WindowK::new(rng, layout, fallback)),
        "window_cq15_fallback" => Box::new(WindowQ15K::new(rng, layout, fallback)),
        "power_db_fallback" => Box::new(PowerDbK::new(rng, layout, fallback)),
        "power_db_q15_fallback" => Box::new(PowerDbQ15K::new(rng, layout, fallback)),
        "agc" => Box::new(AgcK::<C32>::new(rng, layout)),
        "agc_cq15" => Box::new(AgcK::<CQ15>::new(rng, layout)),
        "dc_block_real" => Box::new(DcBlockRealK::<f32>::new(rng, layout, simd)),
        "dc_block_real_q15" => Box::new(DcBlockRealK::<Q15>::new(rng, layout, simd)),
        "dc_block_real_fallback" => Box::new(DcBlockRealK::<f32>::new(rng, layout, fallback)),
        "deemphasis" => Box::new(DeemphasisK::<f32>::new(rng, layout, simd)),
        "deemphasis_q15" => Box::new(DeemphasisK::<Q15>::new(rng, layout, simd)),
        "deemphasis_fallback" => Box::new(DeemphasisK::<f32>::new(rng, layout, fallback)),
        "dc_block" => Box::new(DcBlockK::<C32>::new(rng, layout, simd)),
        "dc_block_cq15" => Box::new(DcBlockK::<CQ15>::new(rng, layout, simd)),
        "dc_block_fallback" => Box::new(DcBlockK::<C32>::new(rng, layout, fallback)),
        "biquad" => Box::new(BiquadK::new(rng, layout, simd, c32)),
        "biquad_cq15" => Box::new(BiquadK::new(rng, layout, simd, cq15)),
        "biquad_real" => Box::new(BiquadRealK::<f32>::new(rng, layout, simd)),
        "biquad_real_q15" => Box::new(BiquadRealK::<Q15>::new(rng, layout, simd)),
        "biquad_fallback" => Box::new(BiquadK::new(rng, layout, fallback, c32)),
        "biquad_cq15_fallback" => Box::new(BiquadK::new(rng, layout, fallback, cq15)),
        "biquad_real_fallback" => Box::new(BiquadRealK::<f32>::new(rng, layout, fallback)),
        "biquad_real_q15_fallback" => Box::new(BiquadRealK::<Q15>::new(rng, layout, fallback)),
        "fft_64" | "fft_64_fallback" => Box::new(FftK::new(rng, 64)),
        "fft_1024" | "fft_1024_fallback" => Box::new(FftK::new(rng, 1024)),
        "fft_4096" | "fft_4096_fallback" => Box::new(FftK::new(rng, 4096)),
        "fft_cq15_1024" | "fft_cq15_1024_fallback" => {
            Box::new(FftQ15K::new(rng, 1024, Scaling::BlockFloat, 8000.0))
        }
        "fft_cq15_1024_default" | "fft_cq15_1024_default_fallback" => {
            Box::new(FftQ15K::new(rng, 1024, Scaling::Default, 8000.0))
        }
        "rfft_1024" | "rfft_1024_fallback" => Box::new(RfftK::new(rng, 1024)),
        "dot" => Box::new(DotC32K::new(rng, layout, simd)),
        "dot_cq15" => Box::new(DotCq15K::new(rng, layout, simd)),
        "dot_real" => Box::new(DotF32K::new(rng, layout, simd)),
        "dot_real_q15" => Box::new(DotQ15K::new(rng, layout, simd)),
        "scale" => Box::new(ScaleK::new(rng, layout, simd)),
        "scale_cq15" => Box::new(ScaleCq15K::new(rng, layout, simd)),
        "dot_fallback" => Box::new(DotC32K::new(rng, layout, fallback)),
        "dot_cq15_fallback" => Box::new(DotCq15K::new(rng, layout, fallback)),
        "dot_real_fallback" => Box::new(DotF32K::new(rng, layout, fallback)),
        "dot_real_q15_fallback" => Box::new(DotQ15K::new(rng, layout, fallback)),
        "scale_fallback" => Box::new(ScaleK::new(rng, layout, fallback)),
        "scale_cq15_fallback" => Box::new(ScaleCq15K::new(rng, layout, fallback)),
        "fir_cq15" => Box::new(FirFamK::<CQ15>::new(rng, layout, Mode::Fir, Entry::Simd)),
        "fir_decimate_cq15" => Box::new(FirFamK::<CQ15>::new(
            rng,
            layout,
            Mode::Decimate,
            Entry::Simd,
        )),
        "fir_interpolate_cq15" => Box::new(FirFamK::<CQ15>::new(
            rng,
            layout,
            Mode::Interpolate,
            Entry::Simd,
        )),
        "resample_cq15" => Box::new(FirFamK::<CQ15>::new(
            rng,
            layout,
            Mode::Resample,
            Entry::Simd,
        )),
        "fir_fallback" => Box::new(FirFamK::<C32>::new(rng, layout, Mode::Fir, Entry::Fallback)),
        "fir_cq15_fallback" => Box::new(FirFamK::<CQ15>::new(
            rng,
            layout,
            Mode::Fir,
            Entry::Fallback,
        )),
        "fir_decimate_fallback" => Box::new(FirFamK::<C32>::new(
            rng,
            layout,
            Mode::Decimate,
            Entry::Fallback,
        )),
        "fir_decimate_cq15_fallback" => Box::new(FirFamK::<CQ15>::new(
            rng,
            layout,
            Mode::Decimate,
            Entry::Fallback,
        )),
        "fir_interpolate_fallback" => Box::new(FirFamK::<C32>::new(
            rng,
            layout,
            Mode::Interpolate,
            Entry::Fallback,
        )),
        "fir_interpolate_cq15_fallback" => Box::new(FirFamK::<CQ15>::new(
            rng,
            layout,
            Mode::Interpolate,
            Entry::Fallback,
        )),
        "resample_fallback" => Box::new(FirFamK::<C32>::new(
            rng,
            layout,
            Mode::Resample,
            Entry::Fallback,
        )),
        "resample_cq15_fallback" => Box::new(FirFamK::<CQ15>::new(
            rng,
            layout,
            Mode::Resample,
            Entry::Fallback,
        )),
        "fir_interpolate_real_fallback" => Box::new(FirFamK::<f32>::new(
            rng,
            layout,
            Mode::Interpolate,
            Entry::Fallback,
        )),
        "fir_interpolate_real_q15_fallback" => Box::new(FirFamK::<Q15>::new(
            rng,
            layout,
            Mode::Interpolate,
            Entry::Fallback,
        )),
        "resample_real_fallback" => Box::new(FirFamK::<f32>::new(
            rng,
            layout,
            Mode::Resample,
            Entry::Fallback,
        )),
        "resample_real_q15_fallback" => Box::new(FirFamK::<Q15>::new(
            rng,
            layout,
            Mode::Resample,
            Entry::Fallback,
        )),
        "fft_cq15_1024_full_scale" => Box::new(FftQ15K::new(rng, 1024, Scaling::Default, 32767.0)),
        _ => return None,
    })
}

/// SDE's SSC marker that starts the counted region (`count run` puts it after the warm-up).
pub const REGION_START: u32 = 0x5d4b_0001;
/// SDE's SSC marker that ends the counted region.
pub const REGION_STOP: u32 = 0x5d4b_0002;

/// Emits Intel SDE's SSC marker `$mark`: `mov ebx, imm32` followed by the prefixed `nop`
/// 64 67 90 (SDE recognises only this exact form, so the value must be an immediate).
macro_rules! ssc_mark {
    ($mark:expr) => {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: saves and restores `rbx` around the marker (LLVM reserves `rbx`, so the asm
        // must not clobber it); the prefixed `nop` has no effect; the stack is balanced again
        // at the end and no memory is touched otherwise.
        unsafe {
            std::arch::asm!(
                "push rbx",
                "mov ebx, {mark}",
                ".byte 0x64, 0x67, 0x90",
                "pop rbx",
                mark = const $mark,
            );
        }
    };
}

/// Starts the region that Intel SDE counts with `-start_ssc_mark` and `-stop_ssc_mark`, so that
/// the start-up of the process, whose instructions depend on the address layout, does not
/// enter its counts. Other counters execute the marker as four plain instructions, the same in
/// every run. No effect on other architectures.
#[inline(always)]
pub fn region_start() {
    ssc_mark!(REGION_START);
}

/// Ends the region of [`region_start`].
#[inline(always)]
pub fn region_stop() {
    ssc_mark!(REGION_STOP);
}

// ---------------------------------------------------------------------------------------------
// Conversions

/// The input formats of the `convert` measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InFormat {
    Cs8,
    Cs16,
    Cu8,
    /// Real `i16`, little endian (`ConvertReal<f32>`).
    Rs16,
}

impl InFormat {
    fn format(self) -> &'static SampleFormat {
        match self {
            InFormat::Cs8 => &formats::CS8,
            InFormat::Cs16 => &formats::CS16,
            InFormat::Cu8 => &formats::CU8,
            InFormat::Rs16 => &formats::RS16,
        }
    }

    fn component_bytes(self) -> usize {
        match self {
            InFormat::Cs8 | InFormat::Cu8 => 1,
            InFormat::Cs16 | InFormat::Rs16 => 2,
        }
    }

    /// Values per sample.
    fn components(self) -> usize {
        if self == InFormat::Rs16 { 1 } else { 2 }
    }

    /// The f64 value of the component at byte offset `2 * i` (CS16: little endian).
    fn value(self, raw: &[u8], i: usize) -> f64 {
        match self {
            InFormat::Cs8 => f64::from(raw[i] as i8) / 128.0,
            InFormat::Cu8 => (f64::from(raw[i]) - 127.5) / 128.0,
            InFormat::Cs16 | InFormat::Rs16 => {
                f64::from(i16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])) / 32768.0
            }
        }
    }
}

/// The input format of a `convert` entry (its name without `_fallback`).
fn in_format(name: &str) -> InFormat {
    match name {
        "convert_cs8" | "convert_cq15_cs8" => InFormat::Cs8,
        "convert_cs16" => InFormat::Cs16,
        "convert_cu8" | "convert_cq15_cu8" => InFormat::Cu8,
        "convert_real_rs16" => InFormat::Rs16,
        other => unreachable!("no convert entry {other}"),
    }
}

/// The text of [`Kernel::describe`] for a CMSIS-DSP conversion of a fallback entry.
fn describe_cmsis_convert(function: &str, format: InFormat) -> &'static str {
    describe_cmsis(&if format == InFormat::Cu8 {
        format!(
            "{function} on the bytes with the sign bit flipped (the CU8 conversion without its offset)"
        )
    } else {
        function.to_string()
    })
}

/// The implementations a `convert` or `convert_out` entry runs: the SIMD entry ours, our
/// fallback and VOLK (if `volk`); the fallback entry (D16) our fallback and CMSIS-DSP.
fn convert_supports(entry: BiquadEntry, which: Impl, volk: bool) -> bool {
    match (entry, which) {
        (_, Impl::Fallback) | (BiquadEntry::Simd, Impl::Ours) => true,
        (BiquadEntry::Simd, Impl::Volk) => volk,
        (BiquadEntry::Fallback, Impl::Cmsis) => cref_sys::cmsis_dsp::AVAILABLE,
        _ => false,
    }
}

/// `convert` from a raw interleaved format to C32, or `convert_real` from a raw real format to
/// f32 (in the first `n` floats of `out`). The entry `<name>_fallback` (D16) measures our
/// fallback against CMSIS-DSP's generic C.
struct ConvertIn {
    entry: BiquadEntry,
    format: InFormat,
    raw: Aligned<u8>,
    /// CU8 only: the bytes with the sign bit flipped (`u - 128` as `i8`), the input of the VOLK
    /// comparison.
    signed: Aligned<u8>,
    out: Aligned<C32>,
    n: usize,
}

impl ConvertIn {
    fn new(rng: &mut Pcg32, layout: Layout, format: InFormat, entry: BiquadEntry) -> Self {
        let (raw, signed) = convert_input(rng, layout, format);
        ConvertIn {
            entry,
            format,
            raw,
            signed,
            out: Aligned::skewed(layout.n, layout.skew(8)),
            n: layout.n,
        }
    }

    /// The `f32` output values: `2 n` of a complex format, `n` of a real one.
    fn values(&self) -> usize {
        self.format.components() * self.n
    }
}

/// The first `values` floats of `out` (at most `2 out.len()`).
fn floats(out: &mut [C32], values: usize) -> &mut [f32] {
    assert!(values <= 2 * out.len());
    // SAFETY: `C32` is two `f32` (`repr(C)`), so `out` holds `2 out.len() >= values` floats;
    // the slice borrows `out` mutably.
    unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f32>(), values) }
}

/// The raw input of `n` samples of `format`, and for `CU8` the bytes with the sign bit flipped
/// (`u - 128` as `i8`), the input of the VOLK comparisons.
fn convert_input(rng: &mut Pcg32, layout: Layout, format: InFormat) -> (Aligned<u8>, Aligned<u8>) {
    let sample = format.components() * format.component_bytes();
    let bytes = layout.n * sample;
    let mut raw = Aligned::<u8>::skewed(bytes, layout.skew(sample));
    match format {
        InFormat::Cs16 | InFormat::Rs16 => {
            let (pairs, _) = raw.as_chunks_mut::<2>();
            for b in pairs {
                *b = (rng.next_u32() as i16).to_le_bytes();
            }
        }
        InFormat::Cs8 | InFormat::Cu8 => {
            for b in raw.iter_mut() {
                *b = rng.next_u32() as u8;
            }
        }
    }
    let mut signed = Aligned::<u8>::skewed(bytes, layout.skew(sample));
    if format == InFormat::Cu8 {
        for (s, r) in signed.iter_mut().zip(raw.iter()) {
            *s = r ^ 0x80;
        }
    }
    (raw, signed)
}

// The calls of our kernels go through these wrappers, which are never inlined into the
// measurement loops: the kernel's code is then compiled the same however a loop around it
// changes (inlined into `FftK::run`, the C32 FFT once moved by 0.35 instructions per sample
// when only the loop changed), and each call is a real call, as the references' are.

#[inline(never)]
fn convert_in<B: Convert<C32>>(backend: &B, format: &SampleFormat, raw: &[u8], out: &mut [C32]) {
    let _ = black_box(backend.convert(format, black_box(raw), out));
}

#[inline(never)]
fn convert_real_in<B: ConvertReal<f32>>(
    backend: &B,
    format: &SampleFormat,
    raw: &[u8],
    out: &mut [f32],
) {
    let _ = black_box(backend.convert_real(format, black_box(raw), out));
}

#[inline(never)]
fn convert_cq15_in<B: Convert<CQ15>>(
    backend: &B,
    format: &SampleFormat,
    raw: &[u8],
    out: &mut [CQ15],
) {
    let _ = black_box(backend.convert(format, black_box(raw), out));
}

impl Kernel for ConvertIn {
    fn run(&mut self, which: Impl, k: usize) {
        let format = self.format.format();
        let values = self.values() as u32;
        let real = self.format == InFormat::Rs16;
        for _ in 0..k {
            match which {
                Impl::Ours if real => {
                    convert_real_in(&Sdrk, format, &self.raw, floats(&mut self.out, self.n));
                }
                Impl::Fallback if real => {
                    convert_real_in(&Fallback, format, &self.raw, floats(&mut self.out, self.n));
                }
                Impl::Ours => convert_in(&Sdrk, format, &self.raw, &mut self.out),
                Impl::Fallback => convert_in(&Fallback, format, &self.raw, &mut self.out),
                Impl::Volk => {
                    let out = self.out.as_mut_ptr().cast::<f32>();
                    match self.format {
                        // SAFETY: `raw` holds `2 n` bytes, `out` `2 n` floats.
                        InFormat::Cs8 => unsafe {
                            volk_sys::volk_8i_s32f_convert_32f(
                                out,
                                self.raw.as_ptr().cast(),
                                128.0,
                                values,
                            );
                        },
                        // SAFETY: `raw` holds `values` little-endian `i16` values (the
                        // alignment of an `i16` is kept by the even skew), `out` at least
                        // `values` floats.
                        InFormat::Cs16 | InFormat::Rs16 => unsafe {
                            volk_sys::volk_16i_s32f_convert_32f(
                                out,
                                self.raw.as_ptr().cast(),
                                32768.0,
                                values,
                            );
                        },
                        // VOLK converts signed bytes only. `(u - 128) / 128` from the flipped
                        // bytes is the same work as the CU8 conversion apart from its offset
                        // `0.5 / 128`, which `accuracy` adds outside the counted region.
                        // SAFETY: `signed` holds `2 n` bytes, `out` `2 n` floats.
                        InFormat::Cu8 => unsafe {
                            volk_sys::volk_8i_s32f_convert_32f(
                                out,
                                self.signed.as_ptr().cast(),
                                128.0,
                                values,
                            );
                        },
                    }
                }
                Impl::Cmsis => {
                    let out = self.out.as_mut_ptr().cast::<f32>();
                    match self.format {
                        // `(u - 128) / 128` from the flipped bytes for CU8, as for VOLK.
                        // SAFETY: `raw` and `signed` hold `values` bytes, `out` at least
                        // `values` floats.
                        InFormat::Cs8 | InFormat::Cu8 => unsafe {
                            let input = if self.format == InFormat::Cu8 {
                                &self.signed
                            } else {
                                &self.raw
                            };
                            cref_sys::cmsis_dsp::ffi::arm_q7_to_float(
                                input.as_ptr().cast(),
                                out,
                                values,
                            );
                        },
                        // SAFETY: `raw` holds `values` little-endian `i16` values (2-byte
                        // aligned by the even skew), `out` at least `values` floats.
                        InFormat::Cs16 | InFormat::Rs16 => unsafe {
                            cref_sys::cmsis_dsp::ffi::arm_q15_to_float(
                                self.raw.as_ptr().cast(),
                                out,
                                values,
                            );
                        },
                    }
                }
                other => unreachable!("convert has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        if matches!(which, Impl::Volk | Impl::Cmsis) && self.format == InFormat::Cu8 {
            for v in self.out.iter_mut() {
                *v = C32::new(v.re + 0.5 / 128.0, v.im + 0.5 / 128.0);
            }
        }
        let values = self.values();
        let want: Vec<f64> = (0..values)
            .map(|i| self.format.value(&self.raw, i))
            .collect();
        snr_db(&want, &c32s(&self.out)[..values])
    }

    fn supports(&self, which: Impl) -> bool {
        convert_supports(self.entry, which, volk_sys::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.format) {
            (Impl::Cmsis, InFormat::Cs8 | InFormat::Cu8) => {
                describe_cmsis_convert("arm_q7_to_float", self.format)
            }
            (Impl::Cmsis, _) => describe_cmsis_convert("arm_q15_to_float", self.format),
            (Impl::Volk, InFormat::Cs8) => "volk_8i_s32f_convert_32f",
            (Impl::Volk, InFormat::Cu8) => {
                "volk_8i_s32f_convert_32f on the bytes with the sign bit flipped (the CU8 conversion without its offset)"
            }
            (Impl::Volk, _) => "volk_16i_s32f_convert_32f",
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        match self.format {
            InFormat::Cs8 | InFormat::Cu8 => &["volk_8i_s32f_convert_32f"],
            InFormat::Cs16 | InFormat::Rs16 => &["volk_16i_s32f_convert_32f"],
        }
    }
}

/// `convert` from a raw interleaved 8-bit format to CQ15.
struct ConvertQ15In {
    entry: BiquadEntry,
    format: InFormat,
    raw: Aligned<u8>,
    /// CU8 only: the bytes with the sign bit flipped, the input of the VOLK comparison.
    signed: Aligned<u8>,
    out: Aligned<CQ15>,
    n: usize,
}

impl ConvertQ15In {
    fn new(rng: &mut Pcg32, layout: Layout, format: InFormat, entry: BiquadEntry) -> Self {
        let (raw, signed) = convert_input(rng, layout, format);
        ConvertQ15In {
            entry,
            format,
            raw,
            signed,
            out: Aligned::skewed(layout.n, layout.skew(4)),
            n: layout.n,
        }
    }
}

impl Kernel for ConvertQ15In {
    fn run(&mut self, which: Impl, k: usize) {
        let format = self.format.format();
        let values = 2 * self.n as u32;
        for _ in 0..k {
            match which {
                Impl::Ours => convert_cq15_in(&Sdrk, format, &self.raw, &mut self.out),
                Impl::Fallback => convert_cq15_in(&Fallback, format, &self.raw, &mut self.out),
                Impl::Volk => {
                    // VOLK converts signed bytes only. `(u - 128) << 8` from the flipped bytes
                    // is the same work as the CU8 conversion `(2u - 255) << 7` apart from its
                    // offset 128, which `accuracy` adds outside the counted region.
                    let input = if self.format == InFormat::Cu8 {
                        &self.signed
                    } else {
                        &self.raw
                    };
                    // SAFETY: `input` holds `2 n` bytes, `out` `2 n` `i16` values.
                    unsafe {
                        volk_sys::volk_8i_convert_16i(
                            self.out.as_mut_ptr().cast(),
                            input.as_ptr().cast(),
                            values,
                        );
                    }
                }
                Impl::Cmsis => {
                    // `(u - 128) << 8` from the flipped bytes for CU8, as for VOLK.
                    let input = if self.format == InFormat::Cu8 {
                        &self.signed
                    } else {
                        &self.raw
                    };
                    // SAFETY: `input` holds `2 n` bytes, `out` `2 n` `i16` values.
                    unsafe {
                        cref_sys::cmsis_dsp::ffi::arm_q7_to_q15(
                            input.as_ptr().cast(),
                            self.out.as_mut_ptr().cast(),
                            values,
                        );
                    }
                }
                other => unreachable!("convert has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        if matches!(which, Impl::Volk | Impl::Cmsis) && self.format == InFormat::Cu8 {
            for v in self.out.iter_mut() {
                *v = CQ15 {
                    re: Q15(v.re.0 + 128),
                    im: Q15(v.im.0 + 128),
                };
            }
        }
        let want: Vec<f64> = (0..2 * self.n)
            .map(|i| self.format.value(&self.raw, i))
            .collect();
        snr_db(&want, &cq15s(&self.out, 1.0 / 32768.0))
    }

    fn supports(&self, which: Impl) -> bool {
        convert_supports(self.entry, which, volk_sys::FULL)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.format) {
            (Impl::Cmsis, _) => describe_cmsis_convert("arm_q7_to_q15", self.format),
            (Impl::Volk, InFormat::Cu8) => {
                "volk_8i_convert_16i on the bytes with the sign bit flipped (the CU8 conversion without its offset)"
            }
            (Impl::Volk, _) => "volk_8i_convert_16i",
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &["volk_8i_convert_16i"]
    }
}

/// The output formats of the `convert_out` measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutFormat {
    Cs16,
    Cs8,
    Cu8,
}

impl OutFormat {
    fn format(self) -> &'static SampleFormat {
        match self {
            OutFormat::Cs16 => &formats::CS16,
            OutFormat::Cs8 => &formats::CS8,
            OutFormat::Cu8 => &formats::CU8,
        }
    }

    /// The real format with the same encoding of one value (`RS16`, and the real 8-bit
    /// formats, which have no constant).
    fn real_format(self) -> SampleFormat {
        let byte = |container, signed| {
            SampleFormat::new(
                container,
                8,
                signed,
                DataLayout::Real,
                Align::Lsb,
                Endian::Little,
            )
            .expect("a valid real 8-bit format")
        };
        match self {
            OutFormat::Cs16 => formats::RS16,
            OutFormat::Cs8 => byte(Container::I8, true),
            OutFormat::Cu8 => byte(Container::U8, false),
        }
    }

    fn component_bytes(self) -> usize {
        match self {
            OutFormat::Cs16 => 2,
            OutFormat::Cs8 | OutFormat::Cu8 => 1,
        }
    }

    /// Full scale and offset of the format: `e = x * scale + offset`.
    fn scale_offset(self) -> (f64, f64) {
        match self {
            OutFormat::Cs16 => (32768.0, 0.0),
            OutFormat::Cs8 => (128.0, 0.0),
            OutFormat::Cu8 => (128.0, 127.5),
        }
    }

    /// The integer at component `i` of the output bytes.
    fn value(self, raw: &[u8], i: usize) -> f64 {
        match self {
            OutFormat::Cs16 => f64::from(i16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])),
            OutFormat::Cs8 => f64::from(raw[i] as i8),
            OutFormat::Cu8 => f64::from(raw[i]),
        }
    }
}

/// The output format of a `convert_out` entry (its name without `_fallback`).
fn out_format(name: &str) -> OutFormat {
    match name {
        "convert_out_cs16" | "convert_out_real_rs16" => OutFormat::Cs16,
        "convert_out_cs8" | "convert_out_real_rs8" => OutFormat::Cs8,
        "convert_out_cu8" | "convert_out_real_ru8" => OutFormat::Cu8,
        other => unreachable!("no convert_out entry {other}"),
    }
}

/// `convert_out` from C32 to a raw interleaved format, or `convert_out_real` from f32 to the
/// real format of the same encoding (`real`; VOLK converts the same values either way). The
/// entry `<name>_fallback` (D16) measures our fallback against CMSIS-DSP's generic C.
struct ConvertOutK {
    entry: BiquadEntry,
    format: OutFormat,
    real: bool,
    input: Aligned<C32>,
    real_input: Aligned<f32>,
    out: Aligned<u8>,
    n: usize,
}

impl ConvertOutK {
    fn new(rng: &mut Pcg32, layout: Layout, format: OutFormat, entry: BiquadEntry) -> Self {
        let bytes = 2 * layout.n * format.component_bytes();
        ConvertOutK {
            entry,
            format,
            real: false,
            input: noise_c32(rng, layout.n, 0.9, layout.skew(8)),
            real_input: Aligned::new(0),
            out: Aligned::skewed(bytes, layout.skew(2 * format.component_bytes())),
            n: layout.n,
        }
    }

    fn real(rng: &mut Pcg32, layout: Layout, format: OutFormat, entry: BiquadEntry) -> Self {
        let mut real_input = Aligned::<f32>::skewed(layout.n, layout.skew(4));
        for v in real_input.iter_mut() {
            *v = rng.unit() * 0.9;
        }
        ConvertOutK {
            entry,
            format,
            real: true,
            input: Aligned::new(0),
            real_input,
            out: Aligned::skewed(
                layout.n * format.component_bytes(),
                layout.skew(format.component_bytes()),
            ),
            n: layout.n,
        }
    }

    /// The number of values converted per call.
    fn values(&self) -> usize {
        if self.real { self.n } else { 2 * self.n }
    }

    /// The input values.
    fn floats(&self) -> Vec<f64> {
        if self.real {
            f32s(&self.real_input)
        } else {
            c32s(&self.input)
        }
    }

    /// The first input value.
    fn input_ptr(&self) -> *const f32 {
        if self.real {
            self.real_input.as_ptr()
        } else {
            self.input.as_ptr().cast::<f32>()
        }
    }
}

#[inline(never)]
fn convert_out<B: ConvertOut<C32>>(
    backend: &B,
    format: &SampleFormat,
    input: &[C32],
    out: &mut [u8],
) {
    let _ = black_box(backend.convert_out(format, black_box(input), out));
}

#[inline(never)]
fn convert_out_real<B: ConvertOutReal<f32>>(
    backend: &B,
    format: &SampleFormat,
    input: &[f32],
    out: &mut [u8],
) {
    let _ = black_box(backend.convert_out_real(format, black_box(input), out));
}

impl Kernel for ConvertOutK {
    fn run(&mut self, which: Impl, k: usize) {
        let format = self.format.format();
        let real_format = &self.format.real_format();
        let (scale, offset) = self.format.scale_offset();
        let values = self.values() as u32;
        for _ in 0..k {
            match which {
                Impl::Ours if self.real => {
                    convert_out_real(&Sdrk, real_format, &self.real_input, &mut self.out);
                }
                Impl::Fallback if self.real => {
                    convert_out_real(&Fallback, real_format, &self.real_input, &mut self.out);
                }
                Impl::Ours => convert_out(&Sdrk, format, &self.input, &mut self.out),
                Impl::Fallback => convert_out(&Fallback, format, &self.input, &mut self.out),
                Impl::Volk => {
                    let input = self.input_ptr();
                    match self.format {
                        // SAFETY: the input holds `values` floats, the output `values` `i16` values
                        // (kept 2-byte aligned by the even skew).
                        OutFormat::Cs16 => unsafe {
                            volk_sys::volk_32f_s32f_convert_16i(
                                self.out.as_mut_ptr().cast(),
                                input,
                                scale as f32,
                                values,
                            );
                        },
                        // SAFETY: the input holds `values` floats, the output `values` bytes.
                        OutFormat::Cs8 => unsafe {
                            volk_sys::volk_32f_s32f_convert_8i(
                                self.out.as_mut_ptr().cast(),
                                input,
                                scale as f32,
                                values,
                            );
                        },
                        // SAFETY: the input holds `values` floats, the output `values` bytes.
                        OutFormat::Cu8 => unsafe {
                            volk_sys::volk_32f_s32f_x2_convert_8u(
                                self.out.as_mut_ptr(),
                                input,
                                scale as f32,
                                offset as f32,
                                values,
                            );
                        },
                    }
                }
                Impl::Cmsis => {
                    let input = self.input_ptr();
                    match self.format {
                        // SAFETY: the input holds `values` floats, the output `values` `i16`
                        // values (kept 2-byte aligned by the even skew).
                        OutFormat::Cs16 => unsafe {
                            cref_sys::cmsis_dsp::ffi::arm_float_to_q15(
                                input,
                                self.out.as_mut_ptr().cast(),
                                values,
                            );
                        },
                        // CMSIS-DSP has no offset binary output: CU8 is the Q7 conversion
                        // without its offset (`accuracy` reads the bytes as signed).
                        // SAFETY: the input holds `values` floats, the output `values` bytes.
                        OutFormat::Cs8 | OutFormat::Cu8 => unsafe {
                            cref_sys::cmsis_dsp::ffi::arm_float_to_q7(
                                input,
                                self.out.as_mut_ptr().cast(),
                                values,
                            );
                        },
                    }
                }
                other => unreachable!("convert_out has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        // The ideal output is the unquantised value; the SNR is the quantisation noise (the
        // offset of CU8 is removed from both, so that it does not count as signal).
        let (scale, offset) = self.format.scale_offset();
        let want: Vec<f64> = self.floats().iter().map(|x| x * scale).collect();
        let got: Vec<f64> = if which == Impl::Cmsis && self.format == OutFormat::Cu8 {
            (0..self.values())
                .map(|i| OutFormat::Cs8.value(&self.out, i))
                .collect()
        } else {
            (0..self.values())
                .map(|i| self.format.value(&self.out, i) - offset)
                .collect()
        };
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        convert_supports(self.entry, which, volk_sys::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis(match self.format {
                OutFormat::Cs16 => "arm_float_to_q15 (truncating, saturating)",
                OutFormat::Cs8 => "arm_float_to_q7 (truncating, saturating)",
                OutFormat::Cu8 => {
                    "arm_float_to_q7 (truncating, saturating; the CU8 conversion without its offset)"
                }
            }),
            Impl::Volk => self.volk_functions()[0],
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        match self.format {
            OutFormat::Cs16 => &["volk_32f_s32f_convert_16i"],
            OutFormat::Cs8 => &["volk_32f_s32f_convert_8i"],
            OutFormat::Cu8 => &["volk_32f_s32f_x2_convert_8u"],
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Mixing

/// The phase increment of the `rotate` measurements (about 0.0045 cycles per sample).
const NCO_INCREMENT: u32 = 0x0123_4567;

/// `rotate` (`C32`): our kernel against VOLK's `rotator2`; the entry `rotate_fallback` runs our
/// fallback against GNU Radio's scalar `gr::blocks::rotator`.
struct RotateK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    entry: BiquadEntry,
}

impl RotateK {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        RotateK {
            input: noise_c32(rng, layout.n, 0.9, layout.skew(8)),
            out: Aligned::skewed(layout.n, layout.skew(8)),
            entry,
        }
    }

    /// `e^{j theta}` of the phase increment, the step of VOLK's and GNU Radio's rotators.
    fn increment() -> [f32; 2] {
        let theta = 2.0 * std::f64::consts::PI * f64::from(NCO_INCREMENT) / 4_294_967_296.0;
        [theta.cos() as f32, theta.sin() as f32]
    }

    /// GNU Radio's rotator from the phase 0 with the increment of the measurement.
    fn gr() -> Option<GrRotator> {
        GrRotator::new([1.0, 0.0], Self::increment())
    }
}

#[inline(never)]
fn rotate<T: ComplexSample, B: Rotate<T>>(backend: &B, nco: &mut Nco, input: &[T], out: &mut [T]) {
    let _ = black_box(backend.rotate(nco, black_box(input), out));
}

impl Kernel for RotateK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut nco = Nco::from_increment(NCO_INCREMENT);
        let [re, im] = Self::increment();
        let inc = lv_32fc_t { re, im };
        let mut phase = lv_32fc_t { re: 1.0, im: 0.0 };
        let mut gr = (which == Impl::GnuRadio).then(Self::gr).flatten();
        for _ in 0..k {
            match which {
                Impl::Ours => rotate(&Sdrk, &mut nco, &self.input, &mut self.out),
                Impl::GnuRadio => gr
                    .as_mut()
                    .expect("gnuradio")
                    .rotate_n(c32_pairs_ref(&self.input), c32_pairs(&mut self.out)),
                Impl::Fallback => rotate(&Fallback, &mut nco, &self.input, &mut self.out),
                // SAFETY: input and output hold `n` samples.
                Impl::Volk => unsafe {
                    volk_sys::volk_32fc_s32fc_x2_rotator2_32fc(
                        self.out.as_mut_ptr().cast(),
                        self.input.as_ptr().cast(),
                        &inc,
                        &mut phase,
                        self.input.len() as u32,
                    );
                },
                other => unreachable!("rotate has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut nco = Nco::from_increment(NCO_INCREMENT);
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.rotate(&mut nco, &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match (self.entry, which) {
            (BiquadEntry::Simd, Impl::Ours) | (_, Impl::Fallback) => true,
            (BiquadEntry::Simd, Impl::Volk) => volk_sys::AVAILABLE,
            (BiquadEntry::Fallback, Impl::GnuRadio) => Self::gr().is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32fc_s32fc_x2_rotator2_32fc",
            Impl::GnuRadio => {
                "GNU Radio gr::blocks::rotator::rotate (scalar, renormalised every 512 samples; gr-rotator.c, generic compiler flags)"
            }
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &["volk_32fc_s32fc_x2_rotator2_32fc"]
    }
}

/// `rotate` of CQ15 samples. No library has a 16-bit complex oscillator: the SIMD entry is
/// compared with our fallback (information), CMSIS-DSP's building blocks as information too;
/// the entry `rotate_cq15_fallback` runs our fallback against those building blocks
/// ([`cmsis_rotate_q15`]).
struct RotateQ15K {
    input: Aligned<CQ15>,
    out: Aligned<CQ15>,
    entry: BiquadEntry,
}

impl RotateQ15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        RotateQ15K {
            input: noise_cq15(rng, layout.n, 23_000.0, layout.skew(4)),
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

/// Samples per run of the CMSIS-DSP compositions [`cmsis_rotate_q15`] and [`cmsis_fm_mod_q15`].
const CMSIS_RUN: usize = 64;

/// The oscillator of CMSIS-DSP at the 32-bit phase `p`: `(arm_cos_q15, arm_sin_q15)` of the
/// angle `p >> 17` (`Q15` of a full turn; 512-entry table, linear interpolation).
#[inline(always)]
fn cmsis_osc_q15(p: u32) -> [i16; 2] {
    let x = (p >> 17) as i16;
    // SAFETY: plain scalar functions of CMSIS-DSP; the callers check `cmsis_dsp::AVAILABLE`.
    unsafe {
        [
            cmsis_dsp::ffi::arm_cos_q15(x),
            cmsis_dsp::ffi::arm_sin_q15(x),
        ]
    }
}

/// `arm_shift_q15(.., 2)` in place: the `Q3.13` products of `arm_cmplx_mult_cmplx_q15` back to
/// `Q15`, saturating.
#[inline(always)]
fn cmsis_shift_q15_by_2(v: &mut [[i16; 2]]) {
    for s in v.iter_mut().flatten() {
        *s = (i32::from(*s) << 2).clamp(-32768, 32767) as i16;
    }
}

/// The `CQ15` rotation as a CMSIS-DSP user composes it (CMSIS-DSP has no NCO): per run of
/// [`CMSIS_RUN`] samples the oscillator from `arm_cos_q15`/`arm_sin_q15` of the running phase,
/// `arm_cmplx_mult_cmplx_q15` (`Q3.13`, truncating) and the shift back to `Q15`.
#[inline(never)]
fn cmsis_rotate_q15(phase: &mut u32, inc: u32, x: &[[i16; 2]], y: &mut [[i16; 2]]) {
    let mut osc = [[0i16; 2]; CMSIS_RUN];
    for (xr, yr) in x.chunks(CMSIS_RUN).zip(y.chunks_mut(CMSIS_RUN)) {
        let osc = &mut osc[..xr.len()];
        for o in osc.iter_mut() {
            *o = cmsis_osc_q15(*phase);
            *phase = phase.wrapping_add(inc);
        }
        cmsis_dsp::cmplx_mult_cmplx_q15(xr, osc, yr);
        cmsis_shift_q15_by_2(yr);
    }
}

impl Kernel for RotateQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        let mut nco = Nco::from_increment(NCO_INCREMENT);
        let mut phase = 0u32;
        for _ in 0..k {
            match which {
                Impl::Ours => rotate(&Sdrk, &mut nco, &self.input, &mut self.out),
                Impl::Fallback => rotate(&Fallback, &mut nco, &self.input, &mut self.out),
                Impl::Cmsis => cmsis_rotate_q15(
                    &mut phase,
                    NCO_INCREMENT,
                    cq15_pairs(&self.input),
                    cq15_pairs_mut(&mut self.out),
                ),
                other => unreachable!("rotate_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        // The C32 reference of the same input in Q15 units is exact to about 140 dB.
        let mut nco = Nco::from_increment(NCO_INCREMENT);
        let input = cq15_as_c32(&self.input);
        let mut want = vec![C32::default(); input.len()];
        let _ = Reference.rotate(&mut nco, &input, &mut want);
        snr_db(&c32s(&want), &cq15s(&self.out, 1.0))
    }

    fn supports(&self, which: Impl) -> bool {
        self.entry.supports(which, false, cmsis_dsp::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Cmsis => describe_cmsis(
                "arm_cos_q15 + arm_sin_q15 per sample, arm_cmplx_mult_cmplx_q15 and the shift to Q15 per 64 samples",
            ),
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Filters

const FIR_TAPS: usize = 64;
const DECIM_TAPS: usize = 127;
const DECIM: usize = 10;

fn taps(rng: &mut Pcg32, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.unit() / (n as f32).sqrt()).collect()
}

/// GNU Radio's `fir_filter_ccf` (`filterN` and `filterNdec`): VOLK's aligned dot product on the
/// input pointer rounded down to VOLK's alignment, with the tap copy that starts with as many
/// zeros as the pointer was rounded down by (`ntaps + al` taps), through the function pointer
/// without an alignment check per call; the single result goes through a separate aligned
/// variable, as in GNU Radio.
struct GrFir {
    /// `taps[al]`: `al` zeros, then the reversed taps.
    taps: Vec<Aligned<f32>>,
    ntaps: usize,
    /// VOLK's alignment in bytes (a power of two, at most [`LEAD`]).
    align: usize,
    result: Aligned<C32>,
}

impl GrFir {
    fn new(taps: &[f32]) -> Self {
        let align = volk_sys::alignment().clamp(1, LEAD);
        let copies = (align / std::mem::size_of::<C32>()).max(1);
        let reversed: Vec<f32> = taps.iter().rev().copied().collect();
        let taps = (0..copies)
            .map(|al| {
                let mut t = Aligned::<f32>::new(reversed.len() + al);
                t[al..].copy_from_slice(&reversed);
                t
            })
            .collect();
        GrFir {
            taps,
            ntaps: reversed.len(),
            align,
            result: Aligned::new(1),
        }
    }

    /// `out[j] = sum_i input[j * step + i] * taps[ntaps - 1 - i]`.
    ///
    /// # Panics
    /// If `input` is too short for `out.len()` outputs, or VOLK is not available.
    fn filter(&mut self, input: &Aligned<C32>, out: &mut [C32], step: usize) {
        assert!(
            out.is_empty() || (out.len() - 1) * step + self.ntaps <= input.len(),
            "GrFir: input too short"
        );
        let dot = volk_sys::dot_prod_a().expect("VOLK is not available");
        for (j, o) in out.iter_mut().enumerate() {
            // SAFETY: the window of output `j` lies in `input` (checked above).
            *o = unsafe { self.dot_at(dot, input.as_ptr().add(j * step)) };
        }
    }

    /// `sum_i window[i] * taps[ntaps - 1 - i]` through the dot product `dot`.
    ///
    /// # Safety
    /// `window` points at `ntaps` samples inside an [`Aligned`] buffer.
    #[inline(always)]
    unsafe fn dot_at(&mut self, dot: volk_sys::DotProd, window: *const C32) -> C32 {
        let result = self.result.as_mut_ptr();
        let p = window as usize;
        let al = (p & (self.align - 1)) / std::mem::size_of::<C32>();
        // SAFETY: the rounded-down pointer `p - al` lies at most `align - 8 < LEAD` bytes before
        // the window, inside the leading padding of the buffer's allocation; the call reads
        // `ntaps + al` samples from it, which end where the window ends (the caller's contract).
        // `al < taps.len()` because `al * 8 < align`, and each tap copy holds `ntaps + al`
        // values. `result` is one aligned sample.
        unsafe {
            dot(
                result.cast::<lv_32fc_t>(),
                window.sub(al).cast::<lv_32fc_t>(),
                self.taps.get_unchecked(al).as_ptr(),
                (self.ntaps + al) as u32,
            );
            *result
        }
    }
}

/// GNU Radio's polyphase filters `interp_fir_filter_ccf` and `rational_resampler_ccf`: one
/// [`GrFir`] per branch (the taps `h[p + L i]`), one aligned dot product per output with the
/// branch of its phase; the phase advances by `M` per output and the input by one sample per
/// `L` of it, as in `rational_resampler_ccf::general_work`.
struct GrPoly {
    branches: Vec<GrFir>,
    up: usize,
    down: usize,
}

impl GrPoly {
    fn new(taps: &[f32], up: usize, down: usize) -> Self {
        assert_eq!(
            taps.len() % up,
            0,
            "GrPoly: the branches must have equal lengths"
        );
        let branches = (0..up)
            .map(|p| {
                let b: Vec<f32> = taps.iter().skip(p).step_by(up).copied().collect();
                GrFir::new(&b)
            })
            .collect();
        GrPoly { branches, up, down }
    }

    /// The `out.len()` outputs from phase 0, the window of the first ending at sample
    /// `ntaps - 1` of `input`.
    ///
    /// # Panics
    /// If `input` is too short, or VOLK is not available.
    fn filter(&mut self, input: &Aligned<C32>, out: &mut [C32]) {
        let ntaps = self.branches[0].ntaps;
        let last = (out.len().saturating_sub(1) * self.down) / self.up;
        assert!(
            out.is_empty() || last + ntaps <= input.len(),
            "GrPoly: input too short"
        );
        let dot = volk_sys::dot_prod_a().expect("VOLK is not available");
        let (mut ctr, mut i) = (0usize, 0usize);
        for o in out.iter_mut() {
            // SAFETY: the window of this output, `ntaps` samples from `i <= last`, lies in
            // `input` (checked above); every branch has `ntaps` taps.
            *o = unsafe {
                self.branches
                    .get_unchecked_mut(ctr)
                    .dot_at(dot, input.as_ptr().add(i))
            };
            ctr += self.down;
            while ctr >= self.up {
                ctr -= self.up;
                i += 1;
            }
        }
    }
}

struct FirK {
    /// 63 history samples, then the input.
    buf: Aligned<C32>,
    taps: Vec<f32>,
    out: Aligned<C32>,
    state: FirState<C32, Vec<f32>, Vec<C32>>,
    gr: GrFir,
    n: usize,
}

impl FirK {
    fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let taps = taps(rng, FIR_TAPS);
        let buf = noise_c32(
            rng,
            layout.n + FIR_TAPS - 1,
            0.9,
            layout.skew_before(8, FIR_TAPS - 1),
        );
        let state = FirState::new(taps.clone(), vec![C32::default(); FIR_TAPS - 1]).expect("taps");
        FirK {
            buf,
            gr: GrFir::new(&taps),
            taps,
            out: Aligned::skewed(layout.n, layout.skew(8)),
            state,
            n: layout.n,
        }
    }
}

#[inline(never)]
fn fir<T: ComplexSample, B: Fir<T>>(
    backend: &B,
    state: &mut FirState<T, Vec<T::Real>, Vec<T>>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.fir(state, black_box(input), out));
}

impl Kernel for FirK {
    fn run(&mut self, which: Impl, k: usize) {
        let input = &self.buf[FIR_TAPS - 1..];
        for _ in 0..k {
            match which {
                Impl::Ours => fir(&Sdrk, &mut self.state, input, &mut self.out),
                Impl::Fallback => fir(&Fallback, &mut self.state, input, &mut self.out),
                Impl::Volk => self.gr.filter(&self.buf, &mut self.out, 1),
                other => unreachable!("fir has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.state.reset();
        self.run(which, 1);
        let mut s =
            FirState::<C32, _, _>::new(self.taps.clone(), vec![C32::default(); FIR_TAPS - 1])
                .expect("taps");
        let mut want = vec![C32::default(); self.n];
        let _ = Reference.fir(&mut s, &self.buf[FIR_TAPS - 1..], &mut want);
        // VOLK sees real history samples instead of zeros: compare only past the filter length.
        let skip = FIR_TAPS - 1;
        snr_db(&c32s(&want[skip..]), &c32s(&self.out[skip..]))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::Volk => volk_sys::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf calls it",
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &["volk_32fc_32f_dot_prod_32fc"]
    }
}

struct DecimateK {
    /// 126 history samples, then the input.
    buf: Aligned<C32>,
    taps: Vec<f32>,
    out: Aligned<C32>,
    state: FirDecimState<C32, Vec<f32>, Vec<C32>>,
    gr: GrFir,
    n: usize,
}

impl DecimateK {
    fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let taps = taps(rng, DECIM_TAPS);
        let buf = noise_c32(
            rng,
            layout.n + DECIM_TAPS - 1,
            0.9,
            layout.skew_before(8, DECIM_TAPS - 1),
        );
        DecimateK {
            buf,
            gr: GrFir::new(&taps),
            state: Self::state(&taps),
            taps,
            out: Aligned::skewed(layout.n.div_ceil(DECIM), layout.skew(8)),
            n: layout.n,
        }
    }

    fn state(taps: &[f32]) -> FirDecimState<C32, Vec<f32>, Vec<C32>> {
        FirDecimState::new(taps.to_vec(), vec![C32::default(); DECIM_TAPS - 1], DECIM)
            .expect("taps")
    }
}

#[inline(never)]
fn fir_decimate<T: ComplexSample, B: FirDecimate<T>>(
    backend: &B,
    state: &mut FirDecimState<T, Vec<T::Real>, Vec<T>>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.fir_decimate(state, black_box(input), out));
}

impl Kernel for DecimateK {
    fn run(&mut self, which: Impl, k: usize) {
        let input = &self.buf[DECIM_TAPS - 1..];
        for _ in 0..k {
            match which {
                Impl::Ours => fir_decimate(&Sdrk, &mut self.state, input, &mut self.out),
                Impl::Fallback => fir_decimate(&Fallback, &mut self.state, input, &mut self.out),
                Impl::Volk => self.gr.filter(&self.buf, &mut self.out, DECIM),
                other => unreachable!("fir_decimate has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.state = Self::state(&self.taps);
        self.run(which, 1);
        let mut s = Self::state(&self.taps);
        let mut want = vec![C32::default(); self.n.div_ceil(DECIM)];
        let p = Reference
            .fir_decimate(&mut s, &self.buf[DECIM_TAPS - 1..], &mut want)
            .expect("decimate");
        let skip = DECIM_TAPS.div_ceil(DECIM);
        let len = p.produced.min(self.n / DECIM);
        snr_db(&c32s(&want[skip..len]), &c32s(&self.out[skip..len]))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::Volk => volk_sys::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => {
                "volk_32fc_32f_dot_prod_32fc_a as GNU Radio's fir_filter_ccf (filterNdec) calls it"
            }
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &["volk_32fc_32f_dot_prod_32fc"]
    }
}

/// Taps of the `fir_interpolate` measurement (64 per branch).
const INTERP_TAPS: usize = 128;
const INTERP: usize = 2;
/// Taps of the `resample` measurement (64 per branch).
const RESAMPLE_TAPS: usize = 192;
const RESAMPLE_UP: usize = 3;
const RESAMPLE_DOWN: usize = 2;

/// `fir_interpolate` (`down = 1`) or `resample` by `up / down`. `n` is the number of outputs
/// per call, so the counts are per output sample like those of `fir`.
struct PolyK {
    /// `ntaps / up - 1` history samples, then the input.
    buf: Aligned<C32>,
    taps: Vec<f32>,
    out: Aligned<C32>,
    state: PolyState,
    gr: GrPoly,
    up: usize,
    down: usize,
}

/// The state of a [`PolyK`]: the interpolator for `down = 1`, else the resampler.
enum PolyState {
    Interpolate(FirInterpState<C32, Vec<f32>, Vec<C32>>),
    Resample(ResampleState<C32, Vec<f32>, Vec<C32>>),
}

impl PolyK {
    fn new(rng: &mut Pcg32, layout: Layout, ntaps: usize, up: usize, down: usize) -> Self {
        let taps = taps(rng, ntaps);
        let branch = ntaps / up;
        // Enough input for `n` outputs from any phase counter.
        let input = layout.n * down / up + 2;
        let buf = noise_c32(
            rng,
            input + branch - 1,
            0.9,
            layout.skew_before(8, branch - 1),
        );
        PolyK {
            buf,
            gr: GrPoly::new(&taps, up, down),
            state: Self::state(&taps, up, down),
            taps,
            out: Aligned::skewed(layout.n, layout.skew(8)),
            up,
            down,
        }
    }

    fn state(taps: &[f32], up: usize, down: usize) -> PolyState {
        if down == 1 {
            let state = FirInterpState::new(taps.to_vec(), Self::history(taps, up), up);
            PolyState::Interpolate(state.expect("taps"))
        } else {
            PolyState::Resample(Self::resampler(taps, up, down))
        }
    }

    fn resampler(taps: &[f32], up: usize, down: usize) -> ResampleState<C32, Vec<f32>, Vec<C32>> {
        ResampleState::new(taps.to_vec(), Self::history(taps, up), up, down).expect("taps")
    }

    /// The empty history of a polyphase filter: one sample per tap of the longest branch.
    fn history(taps: &[f32], up: usize) -> Vec<C32> {
        vec![C32::default(); taps.len().div_ceil(up)]
    }

    fn branch(&self) -> usize {
        self.taps.len() / self.up
    }
}

#[inline(never)]
fn resample<T: ComplexSample, B: Resample<T>>(
    backend: &B,
    state: &mut ResampleState<T, Vec<T::Real>, Vec<T>>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.resample(state, black_box(input), out));
}

#[inline(never)]
fn fir_interpolate<T: ComplexSample, B: FirInterpolate<T>>(
    backend: &B,
    state: &mut FirInterpState<T, Vec<T::Real>, Vec<T>>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.fir_interpolate(state, black_box(input), out));
}

impl Kernel for PolyK {
    fn run(&mut self, which: Impl, k: usize) {
        let input = &self.buf[self.branch() - 1..];
        for _ in 0..k {
            let out = &mut self.out;
            match (which, &mut self.state) {
                (Impl::Ours, PolyState::Interpolate(s)) => fir_interpolate(&Sdrk, s, input, out),
                (Impl::Fallback, PolyState::Interpolate(s)) => {
                    fir_interpolate(&Fallback, s, input, out);
                }
                (Impl::Ours, PolyState::Resample(s)) => resample(&Sdrk, s, input, out),
                (Impl::Fallback, PolyState::Resample(s)) => resample(&Fallback, s, input, out),
                (Impl::Volk, _) => self.gr.filter(&self.buf, out),
                (other, _) => unreachable!("the polyphase filters have no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.state = Self::state(&self.taps, self.up, self.down);
        self.run(which, 1);
        // The interpolator is the resampler with `down = 1`.
        let mut s = Self::resampler(&self.taps, self.up, self.down);
        let mut want = vec![C32::default(); self.out.len()];
        let p = Reference
            .resample(&mut s, &self.buf[self.branch() - 1..], &mut want)
            .expect("resample");
        // VOLK sees real history samples instead of zeros: compare only the outputs whose
        // windows lie in the input.
        let skip = self.branch() * self.up / self.down + 1;
        let len = p.produced;
        snr_db(&c32s(&want[skip..len]), &c32s(&self.out[skip..len]))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::Volk => volk_sys::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.down == 1) {
            (Impl::Volk, true) => {
                "volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's interp_fir_filter_ccf calls it"
            }
            (Impl::Volk, false) => {
                "volk_32fc_32f_dot_prod_32fc_a per output as GNU Radio's rational_resampler_ccf calls it"
            }
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &["volk_32fc_32f_dot_prod_32fc"]
    }
}

/// GNU Radio's `fir_filter_fff` (`filterN`): [`GrFir`] for real samples, with VOLK's aligned
/// `volk_32f_x2_dot_prod_32f_a` on the input pointer rounded down to VOLK's alignment and the
/// tap copy that starts with as many zeros as the pointer was rounded down by.
struct GrFirReal {
    /// `taps[al]`: `al` zeros, then the reversed taps.
    taps: Vec<Aligned<f32>>,
    ntaps: usize,
    /// VOLK's alignment in bytes (a power of two, at most [`LEAD`]).
    align: usize,
    result: Aligned<f32>,
}

impl GrFirReal {
    fn new(taps: &[f32]) -> Self {
        let align = volk_sys::alignment().clamp(1, LEAD);
        let copies = (align / std::mem::size_of::<f32>()).max(1);
        let reversed: Vec<f32> = taps.iter().rev().copied().collect();
        let taps = (0..copies)
            .map(|al| {
                let mut t = Aligned::<f32>::new(reversed.len() + al);
                t[al..].copy_from_slice(&reversed);
                t
            })
            .collect();
        GrFirReal {
            taps,
            ntaps: reversed.len(),
            align,
            result: Aligned::new(1),
        }
    }

    /// `sum_i window[i] * taps[ntaps - 1 - i]` through the dot product `dot`.
    ///
    /// # Safety
    /// `window` points at `ntaps` samples inside an [`Aligned`] buffer.
    #[inline(always)]
    unsafe fn dot_at(&mut self, dot: volk_sys::DotProdReal, window: *const f32) -> f32 {
        let result = self.result.as_mut_ptr();
        let p = window as usize;
        let al = (p & (self.align - 1)) / std::mem::size_of::<f32>();
        // SAFETY: as in `GrFir::dot_at`: the rounded-down pointer lies inside the leading padding
        // of the buffer's allocation, the call reads `ntaps + al` samples that end where the
        // window ends, `al < taps.len()` and each tap copy holds `ntaps + al` values.
        unsafe {
            dot(
                result,
                window.sub(al),
                self.taps.get_unchecked(al).as_ptr(),
                (self.ntaps + al) as u32,
            );
            *result
        }
    }
}

/// GNU Radio's polyphase filters `interp_fir_filter_fff` and `rational_resampler_fff`: one
/// [`GrFirReal`] per branch, one aligned dot product per output, the bookkeeping of [`GrPoly`].
struct GrPolyReal {
    branches: Vec<GrFirReal>,
    up: usize,
    down: usize,
}

impl GrPolyReal {
    fn new(taps: &[f32], up: usize, down: usize) -> Self {
        assert_eq!(
            taps.len() % up,
            0,
            "GrPolyReal: the branches must have equal lengths"
        );
        let branches = (0..up)
            .map(|p| {
                let b: Vec<f32> = taps.iter().skip(p).step_by(up).copied().collect();
                GrFirReal::new(&b)
            })
            .collect();
        GrPolyReal { branches, up, down }
    }

    /// The `out.len()` outputs from phase 0, the window of the first ending at sample
    /// `ntaps - 1` of `input`, which is the whole of an [`Aligned`] buffer (VOLK reads into its
    /// leading padding).
    ///
    /// # Panics
    /// If `input` is too short, or VOLK's real dot product is not available.
    fn filter(&mut self, input: &[f32], out: &mut [f32]) {
        let ntaps = self.branches[0].ntaps;
        let last = (out.len().saturating_sub(1) * self.down) / self.up;
        assert!(
            out.is_empty() || last + ntaps <= input.len(),
            "GrPolyReal: input too short"
        );
        let dot = volk_sys::dot_prod_real_a().expect("VOLK is not available");
        let (mut ctr, mut i) = (0usize, 0usize);
        for o in out.iter_mut() {
            // SAFETY: the window of this output, `ntaps` samples from `i <= last`, lies in
            // `input` (checked above); every branch has `ntaps` taps.
            *o = unsafe {
                self.branches
                    .get_unchecked_mut(ctr)
                    .dot_at(dot, input.as_ptr().add(i))
            };
            ctr += self.down;
            while ctr >= self.up {
                ctr -= self.up;
                i += 1;
            }
        }
    }
}

/// `fir_interpolate_real` (`down = 1`) or `resample_real` by `up / down` with real samples `T`
/// (`f32` or `Q15`), with the taps and ratios of [`PolyK`]. `n` is the number of outputs per
/// call. `f32` is compared with VOLK as GNU Radio's `fff` polyphase filters call it; `Q15` has no
/// established library (VOLK has no 16-bit real dot product) and is compared with our fallback.
struct PolyRealK<T: RealNoise> {
    /// `ntaps / up - 1` history samples, then the input.
    buf: Aligned<T>,
    taps: Vec<T>,
    out: Aligned<T>,
    state: PolyRealState<T>,
    /// GNU Radio's filter, for `f32` only.
    gr: Option<GrPolyReal>,
    up: usize,
    down: usize,
}

/// The state of a [`PolyRealK`]: the interpolator for `down = 1`, else the resampler.
enum PolyRealState<T: RealNoise> {
    Interpolate(FirInterpRealState<T, Vec<T>, Vec<T>>),
    Resample(ResampleRealState<T, Vec<T>, Vec<T>>),
}

impl<T: RealNoise> PolyRealK<T> {
    fn new(rng: &mut Pcg32, layout: Layout, ntaps: usize, up: usize, down: usize) -> Self {
        let scale = 1.0 / (ntaps as f32).sqrt();
        let taps: Vec<T> = (0..ntaps).map(|_| T::noise(rng, scale, 0.0)).collect();
        let branch = ntaps / up;
        let size = std::mem::size_of::<T>();
        // Enough input for `n` outputs from any phase counter.
        let input = layout.n * down / up + 2;
        let mut buf =
            Aligned::<T>::skewed(input + branch - 1, layout.skew_before(size, branch - 1));
        for s in buf.iter_mut() {
            *s = T::noise(rng, 0.9, 0.0);
        }
        let gr = T::as_f32(&taps).map(|f| GrPolyReal::new(f, up, down));
        PolyRealK {
            buf,
            state: Self::state(&taps, up, down),
            taps,
            gr,
            out: Aligned::skewed(layout.n, layout.skew(size)),
            up,
            down,
        }
    }

    fn state(taps: &[T], up: usize, down: usize) -> PolyRealState<T> {
        let hist = vec![T::default(); taps.len().div_ceil(up)];
        if down == 1 {
            let state = FirInterpRealState::new(taps.to_vec(), hist, up);
            PolyRealState::Interpolate(state.expect("taps"))
        } else {
            PolyRealState::Resample(Self::resampler(taps, up, down))
        }
    }

    fn resampler(taps: &[T], up: usize, down: usize) -> ResampleRealState<T, Vec<T>, Vec<T>> {
        let hist = vec![T::default(); taps.len().div_ceil(up)];
        ResampleRealState::new(taps.to_vec(), hist, up, down).expect("taps")
    }

    fn branch(&self) -> usize {
        self.taps.len() / self.up
    }
}

#[inline(never)]
fn resample_real<T: sdrk_core::RealSample, B: ResampleReal<T>>(
    backend: &B,
    state: &mut ResampleRealState<T, Vec<T>, Vec<T>>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.resample_real(state, black_box(input), out));
}

#[inline(never)]
fn fir_interpolate_real<T: sdrk_core::RealSample, B: FirInterpolateReal<T>>(
    backend: &B,
    state: &mut FirInterpRealState<T, Vec<T>, Vec<T>>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.fir_interpolate_real(state, black_box(input), out));
}

impl<T: RealNoise> Kernel for PolyRealK<T>
where
    Sdrk: ResampleReal<T> + FirInterpolateReal<T>,
    Fallback: ResampleReal<T> + FirInterpolateReal<T>,
    Reference: ResampleReal<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        let input = &self.buf[self.branch() - 1..];
        for _ in 0..k {
            let out = &mut self.out;
            match (which, &mut self.state) {
                (Impl::Ours, PolyRealState::Interpolate(s)) => {
                    fir_interpolate_real(&Sdrk, s, input, out);
                }
                (Impl::Fallback, PolyRealState::Interpolate(s)) => {
                    fir_interpolate_real(&Fallback, s, input, out);
                }
                (Impl::Ours, PolyRealState::Resample(s)) => resample_real(&Sdrk, s, input, out),
                (Impl::Fallback, PolyRealState::Resample(s)) => {
                    resample_real(&Fallback, s, input, out);
                }
                (Impl::Volk, _) => {
                    let (Some(gr), Some(x), Some(y)) =
                        (self.gr.as_mut(), T::as_f32(&self.buf), T::as_f32_mut(out))
                    else {
                        unreachable!("VOLK compares f32 only");
                    };
                    gr.filter(x, y);
                }
                (other, _) => unreachable!("the real polyphase filters have no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.state = Self::state(&self.taps, self.up, self.down);
        self.run(which, 1);
        // The interpolator is the resampler with `down = 1`.
        let mut s = Self::resampler(&self.taps, self.up, self.down);
        let mut want = vec![T::default(); self.out.len()];
        let p = Reference
            .resample_real(&mut s, &self.buf[self.branch() - 1..], &mut want)
            .expect("resample_real");
        // VOLK sees real history samples instead of zeros: compare only the outputs whose
        // windows lie in the input.
        let skip = self.branch() * self.up / self.down + 1;
        let len = p.produced;
        let f = |x: &[T]| x.iter().map(|v| v.as_f64()).collect::<Vec<f64>>();
        snr_db(&f(&want[skip..len]), &f(&self.out[skip..len]))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::Volk => self.gr.is_some() && volk_sys::FULL,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.down == 1) {
            (Impl::Volk, true) => {
                "volk_32f_x2_dot_prod_32f_a per output as GNU Radio's interp_fir_filter_fff calls it"
            }
            (Impl::Volk, false) => {
                "volk_32f_x2_dot_prod_32f_a per output as GNU Radio's rational_resampler_fff calls it"
            }
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if self.gr.is_some() {
            &["volk_32f_x2_dot_prod_32f"]
        } else {
            &[]
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Spectrum and demodulation

/// Whether a spectrum entry (`mag_sq`, `mag`, `window`, `power_db`) runs `which`: the SIMD
/// entry our kernel, our fallback and its library (`simd_ref`, VOLK), the `<kernel>_fallback`
/// entry our fallback and CMSIS-DSP's generic C.
fn spectrum_supports(entry: BiquadEntry, which: Impl, simd_ref: bool) -> bool {
    match (entry, which) {
        (_, Impl::Fallback) | (BiquadEntry::Simd, Impl::Ours) => true,
        (BiquadEntry::Simd, Impl::Volk) => simd_ref,
        (BiquadEntry::Fallback, Impl::Cmsis) => cref_sys::cmsis_dsp::AVAILABLE,
        _ => false,
    }
}

#[inline(never)]
fn mag_sq<T: ComplexSample, B: MagSq<T>>(backend: &B, input: &[T], out: &mut [T::Real]) {
    let _ = black_box(backend.mag_sq(black_box(input), out));
}

/// `mag_sq` of C32 samples: the SIMD entry against `volk_32fc_magnitude_squared_32f`, the
/// fallback entry against `arm_cmplx_mag_squared_f32` (the same work).
struct MagSqK {
    input: Aligned<C32>,
    out: Aligned<f32>,
    entry: BiquadEntry,
}

impl MagSqK {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        MagSqK {
            input: noise_c32(rng, layout.n, 0.9, layout.skew(8)),
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

impl Kernel for MagSqK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => mag_sq(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => mag_sq(&Fallback, &self.input, &mut self.out),
                // SAFETY: input and output hold `n` elements.
                Impl::Volk => unsafe {
                    volk_sys::volk_32fc_magnitude_squared_32f(
                        self.out.as_mut_ptr(),
                        self.input.as_ptr().cast(),
                        self.input.len() as u32,
                    );
                },
                Impl::Cmsis => cref_sys::cmsis_dsp::cmplx_mag_squared_f32(
                    c32_pairs_ref(&self.input),
                    &mut self.out,
                ),
                other => unreachable!("mag_sq has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .input
            .iter()
            .map(|v| f64::from(v.re).powi(2) + f64::from(v.im).powi(2))
            .collect();
        snr_db(&want, &f32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        spectrum_supports(self.entry, which, volk_sys::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32fc_magnitude_squared_32f",
            Impl::Cmsis => describe_cmsis("arm_cmplx_mag_squared_f32"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &["volk_32fc_magnitude_squared_32f"]
    }
}

/// `mag_sq` of CQ15 samples into Q15 (VOLK has no squared 16-bit magnitude: the SIMD entry is
/// compared with our fallback). The fallback entry is compared with `arm_cmplx_mag_squared_q15`,
/// which returns Q3.13 (each square shifted right by 17, truncating; scaled by 4 for the
/// accuracy).
struct MagSqQ15K {
    input: Aligned<CQ15>,
    out: Aligned<Q15>,
    entry: BiquadEntry,
}

impl MagSqQ15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        MagSqQ15K {
            input: noise_cq15(rng, layout.n, 23_000.0, layout.skew(4)),
            out: Aligned::skewed(layout.n, layout.skew(2)),
            entry,
        }
    }
}

impl Kernel for MagSqQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => mag_sq(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => mag_sq(&Fallback, &self.input, &mut self.out),
                Impl::Cmsis => cref_sys::cmsis_dsp::cmplx_mag_squared_q15(
                    cq15_pairs(&self.input),
                    q15_raw_mut(&mut self.out),
                ),
                other => unreachable!("mag_sq_q15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        // The unquantised power in Q15 units of the output.
        let want: Vec<f64> = self
            .input
            .iter()
            .map(|v| (f64::from(v.re.0).powi(2) + f64::from(v.im.0).powi(2)) / 32768.0)
            .collect();
        // CMSIS-DSP's Q3.13 in Q15 units.
        let unit = if which == Impl::Cmsis { 4.0 } else { 1.0 };
        let got: Vec<f64> = self.out.iter().map(|q| f64::from(q.0) * unit).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        spectrum_supports(self.entry, which, false)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis("arm_cmplx_mag_squared_q15 (Q3.13, truncating)"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

struct QuadK {
    /// One sample more than the output: VOLK conjugate-multiplies `input[i + 1]` by `input[i]`.
    input: Aligned<C32>,
    tmp: Aligned<C32>,
    out: Aligned<f32>,
}

impl QuadK {
    fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        // A frequency-modulated carrier, so that the arguments cover the whole range.
        let mut phase = 0.0f64;
        let mut input = Aligned::<C32>::skewed(layout.n + 1, layout.skew_before(8, 1));
        for s in input.iter_mut() {
            phase += 0.6 * f64::from(rng.unit());
            *s = C32::new((0.8 * phase.cos()) as f32, (0.8 * phase.sin()) as f32);
        }
        QuadK {
            input,
            tmp: Aligned::skewed(layout.n, layout.skew(8)),
            out: Aligned::skewed(layout.n, layout.skew(4)),
        }
    }
}

#[inline(never)]
fn quad_demod<B: QuadDemod<C32>>(
    backend: &B,
    state: &mut QuadDemodState<C32>,
    input: &[C32],
    out: &mut [f32],
) {
    let _ = black_box(backend.quad_demod(state, black_box(input), out, 1.0));
}

impl Kernel for QuadK {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.out.len();
        let mut state = QuadDemodState::<C32>::new();
        for _ in 0..k {
            match which {
                Impl::Ours => quad_demod(&Sdrk, &mut state, &self.input[1..], &mut self.out),
                Impl::Fallback => {
                    quad_demod(&Fallback, &mut state, &self.input[1..], &mut self.out);
                }
                // SAFETY: all buffers hold `n` samples (the input one more).
                Impl::Volk => unsafe {
                    volk_sys::volk_32fc_x2_multiply_conjugate_32fc(
                        self.tmp.as_mut_ptr().cast(),
                        self.input.as_ptr().add(1).cast(),
                        self.input.as_ptr().cast(),
                        n as u32,
                    );
                    volk_sys::volk_32fc_s32f_atan2_32f(
                        self.out.as_mut_ptr(),
                        self.tmp.as_ptr().cast(),
                        1.0,
                        n as u32,
                    );
                },
                other => unreachable!("quad_demod has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let n = self.out.len();
        // The first output has no predecessor in the kernel's state, so it is left out.
        let want: Vec<f64> = (0..n)
            .map(|i| {
                let (a, b) = (self.input[i + 1], self.input[i]);
                let re = f64::from(a.re) * f64::from(b.re) + f64::from(a.im) * f64::from(b.im);
                let im = f64::from(a.im) * f64::from(b.re) - f64::from(a.re) * f64::from(b.im);
                im.atan2(re)
            })
            .collect();
        snr_db(&want[1..], &f32s(&self.out)[1..])
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::Volk => volk_sys::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f",
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &[
            "volk_32fc_x2_multiply_conjugate_32fc",
            "volk_32fc_s32f_atan2_32f",
        ]
    }
}

/// `quad_demod` of CQ15 samples into Q15 (no established library: compared with our fallback).
/// The input is the carrier of [`QuadK`] at the same amplitude, quantised to Q15; the gain is
/// 1, so the output is the argument divided by pi.
struct QuadQ15K {
    input: Aligned<CQ15>,
    out: Aligned<Q15>,
}

impl QuadQ15K {
    fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let mut phase = 0.0f64;
        let mut input = Aligned::<CQ15>::skewed(layout.n, layout.skew(4));
        for s in input.iter_mut() {
            phase += 0.6 * f64::from(rng.unit());
            let q = |v: f64| Q15((0.8 * 32768.0 * v).round() as i16);
            *s = CQ15::new(q(phase.cos()), q(phase.sin()));
        }
        QuadQ15K {
            input,
            out: Aligned::skewed(layout.n, layout.skew(2)),
        }
    }
}

#[inline(never)]
fn quad_demod_q15<B: QuadDemod<CQ15>>(
    backend: &B,
    state: &mut QuadDemodState<CQ15>,
    input: &[CQ15],
    out: &mut [Q15],
) {
    let _ = black_box(backend.quad_demod(state, black_box(input), out, Q15::MAX));
}

impl Kernel for QuadQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = QuadDemodState::<CQ15>::new();
        for _ in 0..k {
            match which {
                Impl::Ours => quad_demod_q15(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => {
                    quad_demod_q15(&Fallback, &mut state, &self.input, &mut self.out);
                }
                other => unreachable!("quad_demod_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        // The argument of `x[i] conj(x[i - 1])` divided by pi, in Q15 units of the output (gain
        // `Q15::MAX`, 1 - 2^-15); the first output has no predecessor in the state.
        let gain = f64::from(Q15::MAX.0) / 32768.0;
        let want: Vec<f64> = self
            .input
            .windows(2)
            .map(|w| {
                let (b, a) = (w[0], w[1]);
                let (ar, ai) = (f64::from(a.re.0), f64::from(a.im.0));
                let (br, bi) = (f64::from(b.re.0), f64::from(b.im.0));
                let arg = (ai * br - ar * bi).atan2(ar * br + ai * bi);
                arg / std::f64::consts::PI * 32768.0 * gain
            })
            .collect();
        let got: Vec<f64> = self.out[1..].iter().map(|q| f64::from(q.0)).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

/// `fm_mod` (`C32`): no SIMD library has an FM modulator, so the SIMD entry is information
/// against GNU Radio's scalar `frequency_modulator_fc` and our fallback; the entry
/// `fm_mod_fallback` runs our fallback against GNU Radio's block.
struct FmModK {
    input: Aligned<f32>,
    out: Aligned<C32>,
    entry: BiquadEntry,
}

/// The deviation and the rate of the `fm_mod` measurements.
const FM_DEVIATION_HZ: f64 = 75_000.0;
const FM_RATE_HZ: u64 = 240_000;

impl FmModK {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut input = Aligned::<f32>::skewed(layout.n, layout.skew(4));
        for s in input.iter_mut() {
            *s = rng.unit();
        }
        FmModK {
            input,
            out: Aligned::skewed(layout.n, layout.skew(8)),
            entry,
        }
    }

    fn state() -> FmModState {
        FmModState::new(FM_DEVIATION_HZ, Rate::hz(FM_RATE_HZ).expect("rate")).expect("deviation")
    }

    /// GNU Radio's `frequency_modulator_fc` with the same deviation: the sensitivity
    /// `2 pi deviation / rate` in radians per sample per unit input.
    fn gr() -> Option<GrFrequencyModulatorFc> {
        let sensitivity = 2.0 * std::f64::consts::PI * FM_DEVIATION_HZ / FM_RATE_HZ as f64;
        GrFrequencyModulatorFc::new(sensitivity as f32)
    }
}

#[inline(never)]
fn fm_mod<B: FmMod<C32>>(backend: &B, state: &mut FmModState, input: &[f32], out: &mut [C32]) {
    let _ = black_box(backend.fm_mod(state, black_box(input), out));
}

impl Kernel for FmModK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let mut gr = (which == Impl::GnuRadio).then(Self::gr).flatten();
        for _ in 0..k {
            match which {
                Impl::Ours => fm_mod(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => fm_mod(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::GnuRadio => gr
                    .as_mut()
                    .expect("gnuradio")
                    .work(&self.input, c32_pairs(&mut self.out)),
                other => unreachable!("fm_mod has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.fm_mod(&mut Self::state(), &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::GnuRadio => Self::gr().is_some(),
            _ => self.entry.supports(which, false, false),
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio frequency_modulator_fc (f32 phase folded with fmodf, gr::fxpt sincos; gr-fm-mod.c, generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}

/// `fm_mod` of Q15 input to CQ15. No established library has a fixed-point FM modulator: the
/// SIMD entry is compared with our fallback (information), CMSIS-DSP's building blocks as
/// information too; the entry `fm_mod_cq15_fallback` runs our fallback against those building
/// blocks ([`cmsis_fm_mod_q15`]).
struct FmModQ15K {
    input: Aligned<Q15>,
    out: Aligned<CQ15>,
    entry: BiquadEntry,
}

impl FmModQ15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut input = Aligned::<Q15>::skewed(layout.n, layout.skew(2));
        for s in input.iter_mut() {
            *s = Q15((rng.unit() * 32767.0) as i16);
        }
        FmModQ15K {
            input,
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }

    /// The phase step per `Q15` unit of input, `deviation / rate * 2^32 / 2^15` (an integer for
    /// the measured deviation and rate: 40960).
    fn step_per_unit() -> i32 {
        (FmModK::state().scale() / 32768.0).round() as i32
    }
}

/// The `CQ15` FM modulator as a CMSIS-DSP user composes it (CMSIS-DSP has no NCO): per sample
/// the 32-bit phase advances by `x k` in integers and the oscillator is
/// `(arm_cos_q15, arm_sin_q15)` of its top 15 bits.
#[inline(never)]
fn cmsis_fm_mod_q15(phase: &mut u32, k: i32, x: &[i16], y: &mut [[i16; 2]]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *phase = phase.wrapping_add(i32::from(x).wrapping_mul(k) as u32);
        *y = cmsis_osc_q15(*phase);
    }
}

#[inline(never)]
fn fm_mod_cq15<B: FmMod<CQ15>>(
    backend: &B,
    state: &mut FmModState,
    input: &[Q15],
    out: &mut [CQ15],
) {
    let _ = black_box(backend.fm_mod(state, black_box(input), out));
}

impl Kernel for FmModQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = FmModK::state();
        let (mut phase, step) = (0u32, Self::step_per_unit());
        for _ in 0..k {
            match which {
                Impl::Ours => fm_mod_cq15(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => fm_mod_cq15(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::Cmsis => cmsis_fm_mod_q15(
                    &mut phase,
                    step,
                    q15_raw(&self.input),
                    cq15_pairs_mut(&mut self.out),
                ),
                other => unreachable!("fm_mod_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        // `q / 32768` is exact in f32, so the C32 reference has the same phases; its unit
        // vector is scaled to the CQ15 amplitude `1 - 2^-15` in Q15 units.
        let input: Vec<f32> = self
            .input
            .iter()
            .map(|q| f32::from(q.0) / 32768.0)
            .collect();
        let mut want = vec![C32::default(); input.len()];
        let _ = Reference.fm_mod(&mut FmModK::state(), &input, &mut want);
        let want: Vec<f64> = c32s(&want).iter().map(|v| v * 32767.0).collect();
        snr_db(&want, &cq15s(&self.out, 1.0))
    }

    fn supports(&self, which: Impl) -> bool {
        self.entry.supports(which, false, cmsis_dsp::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Cmsis => describe_cmsis(
                "arm_cos_q15 + arm_sin_q15 per sample of an integer phase accumulator",
            ),
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// squelch

/// `squelch` (not accelerated: the fallback at every level). `C32` is compared with the
/// `pwr_squelch_cc` loop of GNU Radio, `CQ15` (no counterpart in GNU Radio) with our fallback
/// (`info`). The input is noise with a loud burst in the middle third, so the gate opens and
/// closes once per call.
struct SquelchK<T: ComplexSample> {
    input: Aligned<T>,
    out: Aligned<T>,
}

trait SquelchSample: ComplexSample {
    /// Noise sample of the given amplitude as a fraction of full scale.
    fn noise(rng: &mut Pcg32, amplitude: f32) -> Self;
}

impl SquelchSample for C32 {
    fn noise(rng: &mut Pcg32, amplitude: f32) -> Self {
        C32::new(rng.unit() * amplitude, rng.unit() * amplitude)
    }
}

impl SquelchSample for CQ15 {
    fn noise(rng: &mut Pcg32, amplitude: f32) -> Self {
        CQ15 {
            re: Q15((rng.unit() * amplitude * 32768.0) as i16),
            im: Q15((rng.unit() * amplitude * 32768.0) as i16),
        }
    }
}

impl<T: SquelchSample> SquelchK<T> {
    fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let mut input = Aligned::<T>::skewed(layout.n, layout.skew(core::mem::size_of::<T>()));
        let n = layout.n;
        for (i, s) in input.iter_mut().enumerate() {
            let loud = i >= n / 3 && i < 2 * n / 3;
            *s = T::noise(rng, if loud { 0.1 } else { 0.0005 });
        }
        SquelchK {
            input,
            out: Aligned::skewed(layout.n, layout.skew(core::mem::size_of::<T>())),
        }
    }

    fn state() -> SquelchState<T> {
        SquelchState::new(-40.0, 3.0, 0.002, Rate::hz(48_000).expect("rate")).expect("squelch")
    }
}

#[inline(never)]
fn squelch<T: ComplexSample, B: Squelch<T>>(
    backend: &B,
    state: &mut SquelchState<T>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.squelch(state, black_box(input), out));
}

/// `alpha = 1 - exp(-1 / (fs tau))` of [`SquelchK::state`] (48 kHz, 2 ms) in f64, as GNU
/// Radio takes it.
fn gr_squelch_alpha() -> f64 {
    1.0 - (-1.0 / (48_000.0 * 0.002f64)).exp()
}

/// The `pwr_squelch_cc` of GNU Radio with the parameters of the measurement: the open level of
/// [`SquelchK::state`] as its threshold (it has no hysteresis), the same smoothing coefficient,
/// no ramp and no gating (zeros while muted, as ours).
fn gr_squelch() -> Option<GrPwrSquelchCc> {
    GrPwrSquelchCc::new(-40.0, gr_squelch_alpha(), 0, false)
}

/// The f64 model of the `pwr_squelch_cc` loop, the accuracy reference of [`Impl::GnuRadio`]:
/// the power in f64, the average in f64, open while it is at or above the threshold.
fn gr_squelch_model(input: &[C32]) -> Vec<C32> {
    let alpha = gr_squelch_alpha();
    let threshold = 10f64.powf(-40.0 / 10.0);
    let mut pwr = 0.0f64;
    input
        .iter()
        .map(|x| {
            let q = f64::from(x.re) * f64::from(x.re) + f64::from(x.im) * f64::from(x.im);
            pwr = alpha * q + (1.0 - alpha) * pwr;
            if pwr < threshold {
                C32::new(0.0, 0.0)
            } else {
                *x
            }
        })
        .collect()
}

impl Kernel for SquelchK<C32> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let mut gr = gr_squelch();
        for _ in 0..k {
            match which {
                Impl::Ours => squelch(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => squelch(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::GnuRadio => {
                    let written = gr
                        .as_mut()
                        .expect("gnuradio")
                        .work(c32_pairs_ref(&self.input), c32_pairs(&mut self.out));
                    debug_assert_eq!(written, self.input.len());
                }
                other => unreachable!("squelch has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = if which == Impl::GnuRadio {
            gr_squelch_model(&self.input)
        } else {
            let mut want = vec![C32::default(); self.input.len()];
            let _ = Reference.squelch(&mut Self::state(), &self.input, &mut want);
            want
        };
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => gr_squelch().is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio pwr_squelch_cc (squelch_base_cc_impl.cc of gr-analog, generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}

impl Kernel for SquelchK<CQ15> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        for _ in 0..k {
            match which {
                Impl::Ours => squelch(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => squelch(&Fallback, &mut state, &self.input, &mut self.out),
                other => unreachable!("squelch_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![CQ15::default(); self.input.len()];
        let _ = Reference.squelch(&mut Self::state(), &self.input, &mut want);
        snr_db(&cq15s(&want, 1.0), &cq15s(&self.out, 1.0))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// FFT

/// `x` viewed as interleaved pairs of `f32` (FFTW's `fftwf_complex`).
fn c32_pairs(x: &mut [C32]) -> &mut [[f32; 2]] {
    // SAFETY: `C32` is `repr(C)` of two `f32`, so the slice has the layout of `[[f32; 2]]`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

/// Shared form of [`c32_pairs`].
fn c32_pairs_ref(x: &[C32]) -> &[[f32; 2]] {
    // SAFETY: as in `c32_pairs`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

/// `x` viewed as interleaved pairs of `i16` (KissFFT's and Ne10's int16 complex type).
fn cq15_pairs(x: &[CQ15]) -> &[[i16; 2]] {
    // SAFETY: `CQ15` is `repr(C)` of two `repr(transparent)` `i16`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

/// Mutable form of [`cq15_pairs`].
fn cq15_pairs_mut(x: &mut [CQ15]) -> &mut [[i16; 2]] {
    // SAFETY: as in `cq15_pairs`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

#[inline(never)]
fn fft<T: ComplexSample, B: Fft<T>>(
    backend: &B,
    plan: &FftPlan<T, Vec<T>>,
    data: &mut [T],
    scaling: Scaling,
) {
    let _ = black_box(backend.fft(plan, data, Direction::Forward, scaling));
}

/// The C32 FFT, forward, unscaled. Every call first restores the input (the transform is in
/// place and unnormalised, so the values would otherwise grow until they overflow); xtask
/// subtracts the count of [`Impl::Restore`]. FFTW out of place reads the restored buffer too,
/// so that the subtraction is the same for every implementation.
struct FftK {
    input: Aligned<C32>,
    data: Aligned<C32>,
    /// The output of FFTW out of place.
    out: Aligned<C32>,
    scratch: Vec<Complex32>,
    plan: FftPlan<C32, Vec<C32>>,
    rustfft: Arc<dyn rustfft::Fft<f32>>,
    fftw: Option<Fftw>,
    fftw_in_place: Option<Fftw>,
    /// CMSIS-DSP's `arm_cfft_f32` (generic C), the scalar reference of the fallback entries.
    cmsis: Option<CmsisCfftF32>,
}

impl FftK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let input = noise_c32(rng, n, 0.5, 0);
        let mut data = Aligned::<C32>::new(n);
        data.copy_from_slice(&input);
        let rustfft = FftPlanner::<f32>::new().plan_fft_forward(n);
        let scratch = vec![Complex32::new(0.0, 0.0); rustfft.get_inplace_scratch_len()];
        let plan = FftPlan::new(n, vec![C32::default(); n / 2]).expect("plan");
        let mut out = Aligned::<C32>::new(n);
        let fftw = Fftw::out_of_place(c32_pairs(&mut data), c32_pairs(&mut out));
        let fftw_in_place = Fftw::in_place(c32_pairs(&mut data));
        FftK {
            input,
            data,
            out,
            scratch,
            plan,
            rustfft,
            fftw,
            fftw_in_place,
            cmsis: CmsisCfftF32::new(n),
        }
    }
}

impl Kernel for FftK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            self.data.copy_from_slice(&self.input);
            match which {
                Impl::Restore => {}
                Impl::Ours => fft(&Sdrk, &self.plan, &mut self.data, Scaling::Default),
                Impl::Fallback => fft(&Fallback, &self.plan, &mut self.data, Scaling::Default),
                // `C32` is rustfft's `Complex32` (both are num-complex).
                Impl::RustFft => self
                    .rustfft
                    .process_with_scratch(&mut self.data, &mut self.scratch),
                Impl::Fftw => {
                    if let Some(fftw) = &mut self.fftw {
                        // SAFETY: `data` and `out` are the buffers the plan was made for.
                        unsafe {
                            fftw.forward_to(c32_pairs(&mut self.data), c32_pairs(&mut self.out));
                        }
                    }
                }
                Impl::FftwInPlace => {
                    if let Some(fftw) = &mut self.fftw_in_place {
                        // SAFETY: `data` is the buffer the plan was made for.
                        unsafe { fftw.forward(c32_pairs(&mut self.data)) };
                    }
                }
                Impl::Cmsis => {
                    if let Some(cmsis) = &self.cmsis {
                        cmsis.forward(c32_pairs(&mut self.data));
                    }
                }
                other => unreachable!("fft has no {other:?}"),
            }
            black_box(&self.data);
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.dft(&self.input, &mut want, Direction::Forward, Scaling::Default);
        let got = if which == Impl::Fftw {
            &self.out
        } else {
            &self.data
        };
        snr_db(&c32s(&want), &c32s(got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback | Impl::RustFft | Impl::Restore => true,
            Impl::Fftw => self.fftw.is_some(),
            Impl::FftwInPlace => self.fftw_in_place.is_some(),
            Impl::Cmsis => self.cmsis.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::RustFft => "rustfft (FftPlanner, forward, in place)",
            Impl::Fftw => "FFTW (fftwf_plan_dft_1d, FFTW_ESTIMATE, forward, out of place)",
            Impl::FftwInPlace => "FFTW (fftwf_plan_dft_1d, FFTW_ESTIMATE, forward, in place)",
            Impl::Cmsis => describe_cmsis("arm_cfft_f32 (forward, in place, bit reversal)"),
            Impl::Restore => "restore of the input",
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

/// `rfft` of `N` real samples (`N / 2 + 1` bins) against rustfft's real-input planner
/// (`realfft`, forward, out of place) and FFTW's real-input transform (out of place, into `out`).
/// realfft uses its input as scratch, so every call of every implementation restores the input
/// buffer first, and xtask subtracts the count of [`Impl::Restore`].
struct RfftK {
    input: Aligned<f32>,
    data: Aligned<f32>,
    out: Aligned<C32>,
    scratch: Vec<C32>,
    plan: RealFftPlan<Vec<C32>, Vec<C32>>,
    realfft: Arc<dyn realfft::RealToComplex<f32>>,
    realfft_out: Vec<Complex32>,
    realfft_scratch: Vec<Complex32>,
    fftw: Option<FftwR2c>,
    /// CMSIS-DSP's `arm_rfft_fast_f32` (generic C) and its packed output.
    cmsis: Option<CmsisRfftFastF32>,
    cmsis_out: Vec<f32>,
}

impl RfftK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let mut input = Aligned::<f32>::new(n);
        for x in input.iter_mut() {
            *x = rng.unit() * 0.5;
        }
        let mut data = Aligned::<f32>::new(n);
        data.copy_from_slice(&input);
        let realfft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(n);
        let mut out = Aligned::<C32>::new(n / 2 + 1);
        let fftw = FftwR2c::new(&mut data, c32_pairs(&mut out));
        RfftK {
            input,
            data,
            out,
            scratch: vec![C32::default(); n / 2],
            plan: RealFftPlan::new(n, vec![C32::default(); n / 4], vec![C32::default(); n / 2])
                .expect("plan"),
            realfft_out: realfft.make_output_vec(),
            realfft_scratch: realfft.make_scratch_vec(),
            realfft,
            fftw,
            cmsis: CmsisRfftFastF32::new(n),
            cmsis_out: vec![0.0; n],
        }
    }
}

#[inline(never)]
fn rfft<B: RealFft>(
    backend: &B,
    plan: &RealFftPlan<Vec<C32>, Vec<C32>>,
    input: &[f32],
    out: &mut [C32],
    scratch: &mut [C32],
) {
    let _ = black_box(backend.rfft(plan, black_box(input), out, scratch));
}

impl Kernel for RfftK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            self.data.copy_from_slice(&self.input);
            match which {
                Impl::Restore => {}
                Impl::Ours => rfft(
                    &Sdrk,
                    &self.plan,
                    &self.data,
                    &mut self.out,
                    &mut self.scratch,
                ),
                Impl::Fallback => rfft(
                    &Fallback,
                    &self.plan,
                    &self.data,
                    &mut self.out,
                    &mut self.scratch,
                ),
                Impl::RustFft => {
                    let _ = self.realfft.process_with_scratch(
                        &mut self.data,
                        &mut self.realfft_out,
                        &mut self.realfft_scratch,
                    );
                }
                Impl::Fftw => {
                    if let Some(fftw) = &mut self.fftw {
                        // SAFETY: `data` and `out` are the buffers the plan was made for.
                        unsafe { fftw.forward(&mut self.data, c32_pairs(&mut self.out)) };
                    }
                }
                Impl::Cmsis => {
                    if let Some(cmsis) = &mut self.cmsis {
                        cmsis.forward(&mut self.data, &mut self.cmsis_out);
                    }
                }
                other => unreachable!("rfft has no {other:?}"),
            }
            black_box(&self.data);
            black_box(&self.out);
            black_box(&self.realfft_out);
            black_box(&self.cmsis_out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![C32::default(); self.out.len()];
        let mut scratch = vec![C32::default(); self.scratch.len()];
        let _ = Reference.rfft(&self.plan, &self.input, &mut want, &mut scratch);
        let got: Vec<C32> = if which == Impl::RustFft {
            self.realfft_out
                .iter()
                .map(|v| C32::new(v.re, v.im))
                .collect()
        } else if which == Impl::Cmsis {
            // CMSIS-DSP packs X[0] and X[N/2] (both real) into the first pair.
            let p = &self.cmsis_out;
            let mut bins = vec![C32::new(p[0], 0.0)];
            let (pairs, _) = p[2..].as_chunks::<2>();
            bins.extend(pairs.iter().map(|c| C32::new(c[0], c[1])));
            bins.push(C32::new(p[1], 0.0));
            bins
        } else {
            self.out.to_vec()
        };
        snr_db(&c32s(&want), &c32s(&got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback | Impl::RustFft | Impl::Restore => true,
            Impl::Fftw => self.fftw.is_some(),
            Impl::Cmsis => self.cmsis.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::RustFft => "realfft on rustfft (RealFftPlanner, forward, out of place)",
            Impl::Cmsis => describe_cmsis("arm_rfft_fast_f32 (forward, packed output)"),
            Impl::Fftw => "FFTW (fftwf_plan_dft_r2c_1d, FFTW_ESTIMATE, forward, out of place)",
            Impl::Restore => "restore of the input",
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

/// The CQ15 FFT, forward, with `BlockFloat` or `Default` (`X[k] / N`) scaling. As [`FftK`],
/// every call restores the input first. KissFFT and Ne10 transform out of place from the
/// restored buffer and always scale by 1/N.
struct FftQ15K {
    input: Aligned<CQ15>,
    data: Aligned<CQ15>,
    out: Aligned<CQ15>,
    scaling: Scaling,
    plan: FftPlan<CQ15, Vec<CQ15>>,
    kiss: Option<KissFft16>,
    ne10: Option<Ne10Int16>,
    /// CMSIS-DSP's `arm_cfft_q15` (generic C): fixed `1/N` scaling with truncating shifts.
    cmsis: Option<CmsisCfftQ15>,
}

impl FftQ15K {
    /// Noise of `(unit * amplitude) as i16` per component: 8000 as before the comparisons
    /// existed, or 32767 for full scale, whose corners reach a magnitude of `sqrt 2` (CS8, CU8
    /// and CS16 data converted at full scale), beyond the range of the plain plan of fixed
    /// scaling.
    fn new(rng: &mut Pcg32, n: usize, scaling: Scaling, amplitude: f32) -> Self {
        let mut input = Aligned::<CQ15>::new(n);
        for s in input.iter_mut() {
            *s = CQ15 {
                re: Q15((rng.unit() * amplitude) as i16),
                im: Q15((rng.unit() * amplitude) as i16),
            };
        }
        let mut data = Aligned::<CQ15>::new(n);
        data.copy_from_slice(&input);
        FftQ15K {
            input,
            data,
            out: Aligned::new(n),
            scaling,
            plan: FftPlan::new(n, vec![CQ15::default(); n / 2]).expect("plan"),
            kiss: KissFft16::new(n),
            ne10: Ne10Int16::new(n),
            cmsis: CmsisCfftQ15::new(n),
        }
    }

    /// One transform of `which` with the restored input; returns the result in Q15 units of
    /// the unscaled DFT.
    fn transform(&mut self, which: Impl) -> Vec<f64> {
        self.data.copy_from_slice(&self.input);
        let n = self.input.len() as f64;
        match which {
            Impl::Ours | Impl::Fallback => {
                let scale = if which == Impl::Ours {
                    Sdrk.fft(&self.plan, &mut self.data, Direction::Forward, self.scaling)
                } else {
                    Fallback.fft(&self.plan, &mut self.data, Direction::Forward, self.scaling)
                }
                .expect("fft");
                let unit = match self.scaling {
                    Scaling::BlockFloat => 2f64.powi(scale.exponent),
                    _ => n,
                };
                cq15s(&self.data, unit)
            }
            Impl::KissFft | Impl::Ne10 => {
                self.run(which, 1);
                cq15s(&self.out, n)
            }
            Impl::Cmsis => {
                self.run(which, 1);
                cq15s(&self.data, n)
            }
            other => unreachable!("fft_cq15 has no {other:?}"),
        }
    }
}

impl Kernel for FftQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            self.data.copy_from_slice(&self.input);
            match which {
                Impl::Restore => {}
                Impl::Ours => fft(&Sdrk, &self.plan, &mut self.data, self.scaling),
                Impl::Fallback => fft(&Fallback, &self.plan, &mut self.data, self.scaling),
                Impl::KissFft => {
                    if let Some(kiss) = &mut self.kiss {
                        kiss.forward(cq15_pairs(&self.data), cq15_pairs_mut(&mut self.out));
                    }
                }
                Impl::Ne10 => {
                    if let Some(ne10) = &mut self.ne10 {
                        ne10.forward(
                            cq15_pairs_mut(&mut self.data),
                            cq15_pairs_mut(&mut self.out),
                        );
                    }
                }
                Impl::Cmsis => {
                    if let Some(cmsis) = &self.cmsis {
                        cmsis.forward(cq15_pairs_mut(&mut self.data));
                    }
                }
                other => unreachable!("fft_cq15 has no {other:?}"),
            }
            black_box(&self.data);
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let got = self.transform(which);
        // The exact DFT: the integer input is exact in C32, and the f64 DFT rounded to f32 is
        // far more accurate than any 16-bit transform.
        let input = cq15_as_c32(&self.input);
        let mut want = vec![C32::default(); input.len()];
        let _ = Reference.dft(&input, &mut want, Direction::Forward, Scaling::Default);
        snr_db(&c32s(&want), &got)
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback | Impl::Restore => true,
            Impl::KissFft => self.kiss.is_some(),
            Impl::Ne10 => self.ne10.is_some(),
            Impl::Cmsis => self.cmsis.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::KissFft => "KissFFT FIXED_POINT=16 (kiss_fft, 1/N scaling, out of place)",
            Impl::Ne10 => "Ne10 ne10_fft_c2c_1d_int16_neon (1/N scaling, out of place)",
            Impl::Cmsis => describe_cmsis(
                "arm_cfft_q15 (forward, in place, bit reversal, 1/N scaling with truncating shifts)",
            ),
            Impl::Restore => "restore of the input",
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// mag (M20)

#[inline(never)]
fn mag<T: ComplexSample, B: Mag<T>>(backend: &B, input: &[T], out: &mut [T::Real]) {
    let _ = black_box(backend.mag(black_box(input), out));
}

/// `mag` of C32 samples: the SIMD entry against `volk_32fc_magnitude_32f`, the fallback entry
/// against `arm_cmplx_mag_f32` (`sqrtf` of the sum, the same work).
struct MagK {
    input: Aligned<C32>,
    out: Aligned<f32>,
    entry: BiquadEntry,
}

impl MagK {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        MagK {
            input: noise_c32(rng, layout.n, 0.9, layout.skew(8)),
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

impl Kernel for MagK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => mag(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => mag(&Fallback, &self.input, &mut self.out),
                // SAFETY: input and output hold `n` elements.
                Impl::Volk => unsafe {
                    volk_sys::volk_32fc_magnitude_32f(
                        self.out.as_mut_ptr(),
                        self.input.as_ptr().cast(),
                        self.input.len() as u32,
                    );
                },
                Impl::Cmsis => {
                    cref_sys::cmsis_dsp::cmplx_mag_f32(c32_pairs_ref(&self.input), &mut self.out);
                }
                other => unreachable!("mag has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .input
            .iter()
            .map(|v| f64::from(v.re).hypot(f64::from(v.im)))
            .collect();
        snr_db(&want, &f32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        spectrum_supports(self.entry, which, volk_sys::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32fc_magnitude_32f",
            Impl::Cmsis => describe_cmsis("arm_cmplx_mag_f32"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &["volk_32fc_magnitude_32f"]
    }
}

/// `mag` of CQ15 samples against `volk_16ic_magnitude_16i`. VOLK rounds differently (it scales
/// to float and back), so the accuracy rule compares the SNR of both against the unrounded
/// magnitude. The fallback entry is compared with `arm_cmplx_mag_q15`, which returns Q2.14 from
/// its own Q31 square root (scaled by 2 for the accuracy).
struct MagQ15K {
    input: Aligned<CQ15>,
    out: Aligned<Q15>,
    entry: BiquadEntry,
}

impl MagQ15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        MagQ15K {
            input: noise_cq15(rng, layout.n, 23_000.0, layout.skew(4)),
            out: Aligned::skewed(layout.n, layout.skew(2)),
            entry,
        }
    }
}

impl Kernel for MagQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => mag(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => mag(&Fallback, &self.input, &mut self.out),
                // SAFETY: input and output hold `n` elements; `CQ15` and `lv_16sc_t` are two
                // `i16` each, real part first.
                Impl::Volk => unsafe {
                    volk_sys::volk_16ic_magnitude_16i(
                        self.out.as_mut_ptr().cast(),
                        self.input.as_ptr().cast::<lv_16sc_t>(),
                        self.input.len() as u32,
                    );
                },
                Impl::Cmsis => cref_sys::cmsis_dsp::cmplx_mag_q15(
                    cq15_pairs(&self.input),
                    q15_raw_mut(&mut self.out),
                ),
                other => unreachable!("mag_q15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .input
            .iter()
            .map(|v| f64::from(v.re.0).hypot(f64::from(v.im.0)))
            .collect();
        // CMSIS-DSP's Q2.14 in Q15 units.
        let unit = if which == Impl::Cmsis { 2.0 } else { 1.0 };
        let got: Vec<f64> = self.out.iter().map(|q| f64::from(q.0) * unit).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        spectrum_supports(self.entry, which, volk_sys::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_16ic_magnitude_16i",
            Impl::Cmsis => describe_cmsis("arm_cmplx_mag_q15 (Q2.14, arm_sqrt_q31)"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        &["volk_16ic_magnitude_16i"]
    }
}

// ---------------------------------------------------------------------------------------------
// window and power_db

/// A periodic Hann window of `n` values, the typical weights of `window`.
fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

#[inline(never)]
fn window<T: ComplexSample, B: Window<T>>(backend: &B, input: &[T], out: &mut [T], w: &[T::Real]) {
    let _ = black_box(backend.window(black_box(input), out, black_box(w)));
}

/// `window` of C32 samples against `volk_32fc_32f_multiply_32fc` (a complex vector times a
/// real one, the same work; on wasm32 its generic protokernel).
/// The fallback entry is compared with `arm_cmplx_mult_real_f32` (the same work).
struct WindowK {
    input: Aligned<C32>,
    w: Aligned<f32>,
    out: Aligned<C32>,
    entry: BiquadEntry,
}

impl WindowK {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut w = Aligned::<f32>::skewed(layout.n, layout.skew(4));
        for (d, v) in w.iter_mut().zip(hann(layout.n)) {
            *d = v as f32;
        }
        WindowK {
            input: noise_c32(rng, layout.n, 0.9, layout.skew(8)),
            w,
            out: Aligned::skewed(layout.n, layout.skew(8)),
            entry,
        }
    }
}

impl Kernel for WindowK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => window(&Sdrk, &self.input, &mut self.out, &self.w),
                Impl::Fallback => window(&Fallback, &self.input, &mut self.out, &self.w),
                // SAFETY: input, weights and output hold `n` elements.
                Impl::Volk => unsafe {
                    volk_sys::native::volk_32fc_32f_multiply_32fc(
                        self.out.as_mut_ptr().cast(),
                        self.input.as_ptr().cast(),
                        self.w.as_ptr(),
                        self.input.len() as u32,
                    );
                },
                Impl::Cmsis => cref_sys::cmsis_dsp::cmplx_mult_real_f32(
                    c32_pairs_ref(&self.input),
                    &self.w,
                    c32_pairs(&mut self.out),
                ),
                other => unreachable!("window has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .input
            .iter()
            .zip(self.w.iter())
            .flat_map(|(x, &w)| {
                let w = f64::from(w);
                [f64::from(x.re) * w, f64::from(x.im) * w]
            })
            .collect();
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        spectrum_supports(self.entry, which, volk_sys::FULL)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32fc_32f_multiply_32fc",
            Impl::Cmsis => describe_cmsis("arm_cmplx_mult_real_f32"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if volk_sys::FULL {
            &["volk_32fc_32f_multiply_32fc"]
        } else {
            &[]
        }
    }
}

/// `window` of CQ15 samples with Q15 weights (no established library has a complex-by-real
/// Q15 multiply with the rounding of `q15_mul` on our levels: the SIMD entry is compared with
/// our fallback). The fallback entry is compared with `arm_cmplx_mult_real_q15` (the same
/// products, truncated instead of rounded).
struct WindowQ15K {
    input: Aligned<CQ15>,
    w: Aligned<Q15>,
    out: Aligned<CQ15>,
    entry: BiquadEntry,
}

impl WindowQ15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut w = Aligned::<Q15>::skewed(layout.n, layout.skew(2));
        for (d, v) in w.iter_mut().zip(hann(layout.n)) {
            *d = Q15((v * 32767.0).round() as i16);
        }
        WindowQ15K {
            input: noise_cq15(rng, layout.n, 23_000.0, layout.skew(4)),
            w,
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

impl Kernel for WindowQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => window(&Sdrk, &self.input, &mut self.out, &self.w),
                Impl::Fallback => window(&Fallback, &self.input, &mut self.out, &self.w),
                Impl::Cmsis => cref_sys::cmsis_dsp::cmplx_mult_real_q15(
                    cq15_pairs(&self.input),
                    q15_raw(&self.w),
                    cq15_pairs_mut(&mut self.out),
                ),
                other => unreachable!("window_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        // The unrounded products in Q15 units of the output.
        let want: Vec<f64> = self
            .input
            .iter()
            .zip(self.w.iter())
            .flat_map(|(x, w)| {
                let w = f64::from(w.0) / 32768.0;
                [f64::from(x.re.0) * w, f64::from(x.im.0) * w]
            })
            .collect();
        snr_db(&want, &cq15s(&self.out, 1.0))
    }

    fn supports(&self, which: Impl) -> bool {
        spectrum_supports(self.entry, which, false)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis("arm_cmplx_mult_real_q15 (truncating)"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

#[inline(never)]
fn power_db<T: sdrk::RealSample, B: PowerDb<T>>(backend: &B, input: &[T], out: &mut [f32]) {
    let _ = black_box(backend.power_db(black_box(input), out, black_box(POWER_DB_OFFSET)));
}

/// The offset of the `power_db` measurements in dB.
const POWER_DB_OFFSET: f32 = -3.0;

/// `x.len()` as the block length of a CMSIS-DSP call.
fn n_u32<T>(x: &[T]) -> u32 {
    u32::try_from(x.len()).expect("block too long")
}

/// The decibel steps after CMSIS-DSP's natural logarithm, in place: `arm_scale_f32` by
/// `10 / ln 10`, then the offset (a plain loop: the shim has no `arm_offset_f32`).
fn cmsis_db_scale(out: &mut [f32]) {
    let (input, dst) = (out.as_ptr(), out.as_mut_ptr());
    // SAFETY: `arm_scale_f32` is element-wise and works in place on the `n` values of `out`.
    unsafe {
        cref_sys::cmsis_dsp::ffi::arm_scale_f32(
            input,
            10.0 * std::f32::consts::LOG10_E,
            dst,
            n_u32(out),
        );
    }
    for v in out.iter_mut() {
        *v += POWER_DB_OFFSET;
    }
}

/// `power_db` with CMSIS-DSP (see [`PowerDbK`]): no floor (the measured powers are positive).
fn cmsis_power_db(input: &[f32], out: &mut [f32]) {
    cref_sys::cmsis_dsp::vlog_f32(input, out);
    cmsis_db_scale(out);
}

/// `10 log10(max(p, 1e-20)) + offset` in f64, the reference of the `power_db` accuracy.
fn power_db_model(p: f64) -> f64 {
    10.0 * p.max(1e-20).log10() + f64::from(POWER_DB_OFFSET)
}

/// `power_db` of f32 powers against VOLK's base-2 logarithm, scaled and offset with VOLK
/// (`volk_32f_log2_32f`, `volk_32f_s32f_multiply_32f` by `10 log10(2)`,
/// `volk_32f_s32f_add_32f`; the same work without the floor). The powers spread over 70 dB.
/// The fallback entry is compared with CMSIS-DSP's natural logarithm, scaled by `10 / ln 10`
/// (`arm_vlog_f32` in place, `arm_scale_f32`) and offset by a plain loop (the shim has no
/// `arm_offset_f32`); a polynomial of another accuracy class.
struct PowerDbK {
    input: Aligned<f32>,
    out: Aligned<f32>,
    entry: BiquadEntry,
}

impl PowerDbK {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut input = Aligned::<f32>::skewed(layout.n, layout.skew(4));
        for p in input.iter_mut() {
            *p = 10f32.powf(3.5 * rng.unit() - 2.5);
        }
        PowerDbK {
            input,
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

impl Kernel for PowerDbK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => power_db(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => power_db(&Fallback, &self.input, &mut self.out),
                // SAFETY: input and output hold `n` elements; the last two calls work in place,
                // which VOLK allows for element-wise kernels.
                Impl::Volk => unsafe {
                    let n = self.input.len() as u32;
                    let out = self.out.as_mut_ptr();
                    volk_sys::native::volk_32f_log2_32f(out, self.input.as_ptr(), n);
                    volk_sys::native::volk_32f_s32f_multiply_32f(
                        out,
                        out,
                        10.0 * std::f32::consts::LOG10_2,
                        n,
                    );
                    volk_sys::native::volk_32f_s32f_add_32f(out, out, POWER_DB_OFFSET, n);
                },
                Impl::Cmsis => cmsis_power_db(&self.input, &mut self.out),
                other => unreachable!("power_db has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .input
            .iter()
            .map(|&p| power_db_model(f64::from(p)))
            .collect();
        snr_db(&want, &f32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        spectrum_supports(self.entry, which, volk_sys::FULL)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => {
                "volk_32f_log2_32f + volk_32f_s32f_multiply_32f + volk_32f_s32f_add_32f"
            }
            Impl::Cmsis => describe_cmsis("arm_vlog_f32 + arm_scale_f32 (+ offset loop)"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if volk_sys::FULL {
            &[
                "volk_32f_log2_32f",
                "volk_32f_s32f_multiply_32f",
                "volk_32f_s32f_add_32f",
            ]
        } else {
            &[]
        }
    }
}

/// `power_db` of Q15 powers (no established library: the SIMD entry is compared with our
/// fallback). The raw powers spread over 45 dB below full scale. The fallback entry is compared
/// with `arm_q15_to_float` followed by the CMSIS-DSP steps of [`PowerDbK`] (information: no
/// library converts Q15 powers to decibels).
struct PowerDbQ15K {
    input: Aligned<Q15>,
    out: Aligned<f32>,
    entry: BiquadEntry,
}

impl PowerDbQ15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut input = Aligned::<Q15>::skewed(layout.n, layout.skew(2));
        for p in input.iter_mut() {
            *p = Q15((32767.0 * 10f32.powf(2.25 * rng.unit() - 2.25)) as i16);
        }
        PowerDbQ15K {
            input,
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

impl Kernel for PowerDbQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => power_db(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => power_db(&Fallback, &self.input, &mut self.out),
                Impl::Cmsis => {
                    cref_sys::cmsis_dsp::q15_to_float(q15_raw(&self.input), &mut self.out);
                    let (input, out) = (self.out.as_ptr(), self.out.as_mut_ptr());
                    // SAFETY: `arm_vlog_f32` is element-wise and works in place on the `n`
                    // values of the output.
                    unsafe { cref_sys::cmsis_dsp::ffi::arm_vlog_f32(input, out, n_u32(&self.out)) };
                    cmsis_db_scale(&mut self.out);
                }
                other => unreachable!("power_db_q15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .input
            .iter()
            .map(|p| power_db_model(f64::from(p.0) / 32768.0))
            .collect();
        snr_db(&want, &f32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        spectrum_supports(self.entry, which, false)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => {
                describe_cmsis("arm_q15_to_float + arm_vlog_f32 + arm_scale_f32 (+ offset loop)")
            }
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// agc and dc_block_real (M20)

/// `agc` (no established library: compared with our fallback; not accelerated, so the entries
/// are `info`). The input is noise whose amplitude steps from 0.01 to 0.3 to 0.003 in thirds,
/// so the gain moves in both directions within a call.
struct AgcK<T: ComplexSample> {
    input: Aligned<T>,
    out: Aligned<T>,
}

impl<T: SquelchSample> AgcK<T> {
    fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let mut input = Aligned::<T>::skewed(n, layout.skew(core::mem::size_of::<T>()));
        for (i, s) in input.iter_mut().enumerate() {
            let amplitude = [0.01, 0.3, 0.003][(3 * i / n).min(2)];
            *s = T::noise(rng, amplitude);
        }
        AgcK {
            input,
            out: Aligned::skewed(n, layout.skew(core::mem::size_of::<T>())),
        }
    }

    fn state() -> AgcState<T> {
        AgcState::new(
            0.5,
            0.005,
            0.05,
            -20.0,
            60.0,
            0.0,
            Rate::hz(48_000).expect("rate"),
        )
        .expect("agc")
    }
}

#[inline(never)]
fn agc<T: ComplexSample, B: Agc<T>>(
    backend: &B,
    state: &mut AgcState<T>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.agc(state, black_box(input), out));
}

/// The `agc_cc` of GNU Radio with the parameters of the measurement: rate, reference, initial
/// gain and maximum gain (additive, so the rate is not the one of our `AgcState`; the
/// instruction count of its loop does not depend on the values).
fn gr_agc() -> Option<GrAgcCc> {
    GrAgcCc::new(1e-3, 0.5, 1.0, 1000.0)
}

/// The f64 model of the `agc_cc` loop, the accuracy reference of [`Impl::GnuRadio`].
fn gr_agc_model(input: &[C32]) -> Vec<C32> {
    let (rate, reference, max_gain) = (1e-3f64, 0.5f64, 1000.0f64);
    let mut gain = 1.0f64;
    let mut out = Vec::with_capacity(input.len());
    for x in input {
        let (re, im) = (f64::from(x.re) * gain, f64::from(x.im) * gain);
        gain += rate * (reference - (re * re + im * im).sqrt());
        if gain > max_gain {
            gain = max_gain;
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a model value rounded to f32"
        )]
        out.push(C32::new(re as f32, im as f32));
    }
    out
}

impl Kernel for AgcK<C32> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let mut gr = gr_agc();
        for _ in 0..k {
            match which {
                Impl::Ours => agc(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => agc(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::GnuRadio => gr
                    .as_mut()
                    .expect("gnuradio")
                    .scale_n(c32_pairs_ref(&self.input), c32_pairs(&mut self.out)),
                other => unreachable!("agc has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = if which == Impl::GnuRadio {
            gr_agc_model(&self.input)
        } else {
            let mut want = vec![C32::default(); self.input.len()];
            let _ = Reference.agc(&mut Self::state(), &self.input, &mut want);
            want
        };
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => gr_agc().is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => "GNU Radio agc_cc (agc.h of gr-analog, generic compiler flags)",
            _ => "sdrk",
        }
    }
}

impl Kernel for AgcK<CQ15> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        for _ in 0..k {
            match which {
                Impl::Ours => agc(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => agc(&Fallback, &mut state, &self.input, &mut self.out),
                other => unreachable!("agc_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![CQ15::default(); self.input.len()];
        let _ = Reference.agc(&mut Self::state(), &self.input, &mut want);
        snr_db(&cq15s(&want, 1.0), &cq15s(&self.out, 1.0))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

/// A real sample for the `dc_block_real` measurements.
trait RealNoise: sdrk_core::RealSample {
    /// Noise of the given amplitude as a fraction of full scale, plus the offset `dc`.
    fn noise(rng: &mut Pcg32, amplitude: f32, dc: f32) -> Self;
    /// The pole `0.999`.
    fn pole() -> Self;
    /// The value as f64.
    fn as_f64(self) -> f64;
    /// `x` as `f32` samples if `Self` is `f32`.
    fn as_f32(x: &[Self]) -> Option<&[f32]>;
    /// `x` as `f32` samples if `Self` is `f32`.
    fn as_f32_mut(x: &mut [Self]) -> Option<&mut [f32]>;
}

impl RealNoise for f32 {
    fn noise(rng: &mut Pcg32, amplitude: f32, dc: f32) -> Self {
        rng.unit() * amplitude + dc
    }

    fn pole() -> Self {
        0.999
    }

    fn as_f64(self) -> f64 {
        f64::from(self)
    }

    fn as_f32(x: &[Self]) -> Option<&[f32]> {
        Some(x)
    }

    fn as_f32_mut(x: &mut [Self]) -> Option<&mut [f32]> {
        Some(x)
    }
}

impl RealNoise for Q15 {
    fn noise(rng: &mut Pcg32, amplitude: f32, dc: f32) -> Self {
        Q15(((rng.unit() * amplitude + dc) * 32768.0) as i16)
    }

    fn pole() -> Self {
        Q15(32_735)
    }

    fn as_f64(self) -> f64 {
        f64::from(self.0)
    }

    fn as_f32(_: &[Self]) -> Option<&[f32]> {
        None
    }

    fn as_f32_mut(_: &mut [Self]) -> Option<&mut [f32]> {
        None
    }
}

/// `dc_block_real` (`f32`: the `f64` block scan of the SIMD backend, for which no SIMD library
/// has an equivalent, so the entry is `info` against our fallback), `dc_block_real_q15` (the
/// fallback through the facade, `info`) and `dc_block_real_fallback` (our fallback against
/// liquid-dsp's DC blocker, the same pole and zero in direct form II, on the same input).
struct DcBlockRealK<T: RealNoise> {
    input: Aligned<T>,
    out: Aligned<T>,
    entry: BiquadEntry,
}

/// liquid-dsp's real DC blocker with SDRk's pole `r` (`alpha = 1 - r` is exact in `f32` for
/// `0.5 <= r < 1`) and gain 1.
fn liquid_dc_real(r: f32) -> Option<LiquidDcBlocker> {
    let mut q = LiquidDcBlocker::new(1.0 - r)?;
    q.set_scale(1.0);
    Some(q)
}

/// The complex variant of [`liquid_dc_real`].
fn liquid_dc_complex(r: f32) -> Option<LiquidDcBlockerCrcf> {
    let mut q = LiquidDcBlockerCrcf::new(1.0 - r)?;
    q.set_scale(1.0);
    Some(q)
}

/// The DC blocker `y[n] = x[n] - x[n-1] + r y[n-1]` in `f64` from rest (`stride` interleaved
/// components): the accuracy model of every `f32`/`C32` implementation of the `dc_block`
/// entries, as the golden of the accuracy policy.
fn dc_block_model(r: f32, x: &[f32], stride: usize) -> Vec<f64> {
    let r = f64::from(r);
    let mut y = vec![0.0f64; x.len()];
    for c in 0..stride {
        let (mut xp, mut yp) = (0.0f64, 0.0f64);
        for i in (c..x.len()).step_by(stride) {
            let v = f64::from(x[i]) - xp + r * yp;
            y[i] = v;
            xp = f64::from(x[i]);
            yp = v;
        }
    }
    y
}

/// The implementations of the `dc_block`/`deemphasis` entries: our kernel and fallback for the
/// facade entries, our fallback and `reference` (where this build has it) for a fallback entry.
fn recursion_supports(entry: BiquadEntry, which: Impl, reference: Impl, has: bool) -> bool {
    match (entry, which) {
        (BiquadEntry::Simd, Impl::Ours) | (_, Impl::Fallback) => true,
        (BiquadEntry::Fallback, w) if w == reference => has,
        _ => false,
    }
}

impl<T: RealNoise> DcBlockRealK<T> {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut input = Aligned::<T>::skewed(layout.n, layout.skew(core::mem::size_of::<T>()));
        for s in input.iter_mut() {
            *s = T::noise(rng, 0.2, 0.3);
        }
        DcBlockRealK {
            input,
            out: Aligned::skewed(layout.n, layout.skew(core::mem::size_of::<T>())),
            entry,
        }
    }

    fn state() -> DcBlockRealState<T> {
        DcBlockRealState::new(T::pole()).expect("dc_block_real")
    }
}

#[inline(never)]
fn dc_block_real<T: sdrk_core::RealSample, B: DcBlockReal<T>>(
    backend: &B,
    state: &mut DcBlockRealState<T>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.dc_block_real(state, black_box(input), out));
}

impl<T: RealNoise> Kernel for DcBlockRealK<T>
where
    Sdrk: DcBlockReal<T>,
    Fallback: DcBlockReal<T>,
    Reference: DcBlockReal<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let mut liquid = T::as_f32(&[T::pole()])
            .filter(|_| which == Impl::Liquid)
            .and_then(|r| liquid_dc_real(r[0]));
        for _ in 0..k {
            match which {
                Impl::Ours => dc_block_real(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => dc_block_real(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::Liquid => {
                    let input = T::as_f32(&self.input).expect("f32");
                    let out = T::as_f32_mut(&mut self.out).expect("f32");
                    liquid
                        .as_mut()
                        .expect("liquid")
                        .execute_block(black_box(input), out);
                }
                other => unreachable!("dc_block_real has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let got: Vec<f64> = self.out.iter().map(|v| v.as_f64()).collect();
        if let (Some(x), Some(r)) = (T::as_f32(&self.input), T::as_f32(&[T::pole()])) {
            return snr_db(&dc_block_model(r[0], x, 1), &got);
        }
        let mut want = vec![self.out[0]; self.out.len()];
        let _ = Reference.dc_block_real(&mut Self::state(), &self.input, &mut want);
        let want: Vec<f64> = want.iter().map(|v| v.as_f64()).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        let has = T::as_f32(&[T::pole()]).is_some_and(|r| liquid_dc_real(r[0]).is_some());
        recursion_supports(self.entry, which, Impl::Liquid, has)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Liquid => {
                "liquid-dsp 1.8.3 iirfilt_rrrf_create_dc_blocker (direct form II, portable dot product, -O3)"
            }
            _ => "sdrk",
        }
    }
}

/// `deemphasis` (`f32`: the `f64` block scan of the SIMD backend, `info` against our fallback;
/// `Q15`: the fallback through the facade, `info`) and `deemphasis_fallback` (our fallback
/// against the IIR loop of GNU Radio's `fm_deemph` with the same time constant: a first-order
/// filter with one more tap, the zero of the bilinear transform, so the work per sample is
/// comparable but the filter is not the same). The input is noise of 0.5, the time constant
/// 75 us at 48 kHz (`alpha` about 0.24).
struct DeemphasisK<T: RealNoise> {
    input: Aligned<T>,
    out: Aligned<T>,
    entry: BiquadEntry,
}

/// The sample rate and time constant of the `deemphasis` entries.
const DEEMPH_FS: f64 = 48_000.0;
/// See [`DEEMPH_FS`].
const DEEMPH_TAU: f64 = 75e-6;

/// GNU Radio's `fm_deemph` for the entries.
fn gr_deemph() -> Option<GrFmDeemph> {
    GrFmDeemph::new(DEEMPH_FS, DEEMPH_TAU)
}

/// The one-pole `y = y + alpha (x - y)` in `f64` from rest with the `f32` `alpha` of the state:
/// the accuracy model of our `f32` implementations.
fn deemphasis_model(alpha: f32, x: &[f32]) -> Vec<f64> {
    let a = f64::from(alpha);
    let mut y = 0.0f64;
    x.iter()
        .map(|&v| {
            y += a * (f64::from(v) - y);
            y
        })
        .collect()
}

/// GNU Radio's filter `y = b0 x + b1 x1 + a1 y1` in `f64` from rest with its own (double) taps:
/// the accuracy model of [`Impl::GnuRadio`].
fn gr_deemph_model(taps: [f64; 3], x: &[f32]) -> Vec<f64> {
    let [b0, b1, a1] = taps;
    let (mut x1, mut y1) = (0.0f64, 0.0f64);
    x.iter()
        .map(|&v| {
            let v = f64::from(v);
            let y = b0 * v + b1 * x1 + a1 * y1;
            x1 = v;
            y1 = y;
            y
        })
        .collect()
}

impl<T: RealNoise> DeemphasisK<T> {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut input = Aligned::<T>::skewed(layout.n, layout.skew(core::mem::size_of::<T>()));
        for s in input.iter_mut() {
            *s = T::noise(rng, 0.5, 0.0);
        }
        DeemphasisK {
            input,
            out: Aligned::skewed(layout.n, layout.skew(core::mem::size_of::<T>())),
            entry,
        }
    }

    fn state() -> DeemphasisState<T> {
        DeemphasisState::new(DEEMPH_TAU, Rate::hz(DEEMPH_FS as u64).expect("rate"))
            .expect("deemphasis")
    }
}

#[inline(never)]
fn deemphasis<T: sdrk_core::RealSample, B: Deemphasis<T>>(
    backend: &B,
    state: &mut DeemphasisState<T>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.deemphasis(state, black_box(input), out));
}

impl<T: RealNoise> Kernel for DeemphasisK<T>
where
    Sdrk: Deemphasis<T>,
    Fallback: Deemphasis<T>,
    Reference: Deemphasis<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let mut gr = (which == Impl::GnuRadio).then(gr_deemph).flatten();
        for _ in 0..k {
            match which {
                Impl::Ours => deemphasis(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => deemphasis(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::GnuRadio => {
                    let input = T::as_f32(&self.input).expect("f32");
                    let out = T::as_f32_mut(&mut self.out).expect("f32");
                    gr.as_mut().expect("gnuradio").work(black_box(input), out);
                }
                other => unreachable!("deemphasis has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let got: Vec<f64> = self.out.iter().map(|v| v.as_f64()).collect();
        if let Some(x) = T::as_f32(&self.input) {
            let want = if which == Impl::GnuRadio {
                gr_deemph_model(gr_deemph().expect("gnuradio").taps(), x)
            } else {
                let alpha = Self::state().alpha();
                deemphasis_model(
                    *T::as_f32(&[alpha]).expect("f32").first().expect("alpha"),
                    x,
                )
            };
            return snr_db(&want, &got);
        }
        let mut want = vec![self.out[0]; self.out.len()];
        let _ = Reference.deemphasis(&mut Self::state(), &self.input, &mut want);
        let want: Vec<f64> = want.iter().map(|v| v.as_f64()).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        let has = T::as_f32(&self.input).is_some() && gr_deemph().is_some();
        recursion_supports(self.entry, which, Impl::GnuRadio, has)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio fm_deemph: iir_filter_ffd with the bilinear taps of fm_emph.py (double taps and accumulator, generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}

/// A complex sample for the `dc_block` measurements.
trait DcNoise: ComplexSample {
    /// Noise of the given amplitude as a fraction of full scale, plus the offset `(0.3, -0.2)`.
    fn noise(rng: &mut Pcg32, amplitude: f32) -> Self;
    /// The pole `0.999`.
    fn pole() -> Self::Real;
    /// The components as f64.
    fn flat(x: &[Self]) -> Vec<f64>;
    /// The pole as `f32` (the value of [`DcNoise::pole`] for `C32`; unused for `CQ15`).
    fn pole_f32() -> f32;
    /// `x` as interleaved `f32` pairs if `Self` is `C32`.
    fn as_f32(x: &[Self]) -> Option<&[[f32; 2]]>;
    /// `x` as interleaved `f32` pairs if `Self` is `C32`.
    fn as_f32_mut(x: &mut [Self]) -> Option<&mut [[f32; 2]]>;
}

impl DcNoise for C32 {
    fn noise(rng: &mut Pcg32, amplitude: f32) -> Self {
        C32::new(rng.unit() * amplitude + 0.3, rng.unit() * amplitude - 0.2)
    }

    fn pole() -> f32 {
        0.999
    }

    fn flat(x: &[Self]) -> Vec<f64> {
        c32s(x)
    }

    fn pole_f32() -> f32 {
        0.999
    }

    fn as_f32(x: &[Self]) -> Option<&[[f32; 2]]> {
        Some(c32_pairs_ref(x))
    }

    fn as_f32_mut(x: &mut [Self]) -> Option<&mut [[f32; 2]]> {
        Some(c32_pairs(x))
    }
}

impl DcNoise for CQ15 {
    fn noise(rng: &mut Pcg32, amplitude: f32) -> Self {
        CQ15 {
            re: <Q15 as RealNoise>::noise(rng, amplitude, 0.3),
            im: <Q15 as RealNoise>::noise(rng, amplitude, -0.2),
        }
    }

    fn pole() -> Q15 {
        Q15(32_735)
    }

    fn flat(x: &[Self]) -> Vec<f64> {
        cq15s(x, 1.0)
    }

    fn pole_f32() -> f32 {
        0.999
    }

    fn as_f32(_: &[Self]) -> Option<&[[f32; 2]]> {
        None
    }

    fn as_f32_mut(_: &mut [Self]) -> Option<&mut [[f32; 2]]> {
        None
    }
}

/// `dc_block` (`C32`: the `f64` block scan of the SIMD backend, `info` against our fallback;
/// `CQ15`: the fallback through the facade, `info`) and `dc_block_fallback` (our fallback
/// against liquid-dsp's complex DC blocker on the same input).
struct DcBlockK<T: DcNoise> {
    input: Aligned<T>,
    out: Aligned<T>,
    entry: BiquadEntry,
}

impl<T: DcNoise> DcBlockK<T> {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut input = Aligned::<T>::skewed(layout.n, layout.skew(core::mem::size_of::<T>()));
        for s in input.iter_mut() {
            *s = T::noise(rng, 0.2);
        }
        DcBlockK {
            input,
            out: Aligned::skewed(layout.n, layout.skew(core::mem::size_of::<T>())),
            entry,
        }
    }

    fn state() -> DcBlockState<T> {
        DcBlockState::new(T::pole()).expect("dc_block")
    }
}

#[inline(never)]
fn dc_block<T: ComplexSample, B: DcBlock<T>>(
    backend: &B,
    state: &mut DcBlockState<T>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.dc_block(state, black_box(input), out));
}

impl<T: DcNoise> Kernel for DcBlockK<T>
where
    Sdrk: DcBlock<T>,
    Fallback: DcBlock<T>,
    Reference: DcBlock<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let mut liquid = T::as_f32(&self.input)
            .filter(|_| which == Impl::Liquid)
            .and_then(|_| liquid_dc_complex(T::pole_f32()));
        for _ in 0..k {
            match which {
                Impl::Ours => dc_block(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => dc_block(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::Liquid => {
                    let input = T::as_f32(&self.input).expect("cf32");
                    let out = T::as_f32_mut(&mut self.out).expect("cf32");
                    liquid
                        .as_mut()
                        .expect("liquid")
                        .execute_block(black_box(input), out);
                }
                other => unreachable!("dc_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        if let Some(x) = T::as_f32(&self.input) {
            let x: &[f32] = x.as_flattened();
            return snr_db(&dc_block_model(T::pole_f32(), x, 2), &T::flat(&self.out));
        }
        let mut want = vec![self.out[0]; self.out.len()];
        let _ = Reference.dc_block(&mut Self::state(), &self.input, &mut want);
        snr_db(&T::flat(&want), &T::flat(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        let has = T::as_f32(&self.input).is_some() && liquid_dc_complex(T::pole_f32()).is_some();
        recursion_supports(self.entry, which, Impl::Liquid, has)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Liquid => {
                "liquid-dsp 1.8.3 iirfilt_crcf_create_dc_blocker (direct form II, portable dot product, -O3)"
            }
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// biquad and biquad_real (M22; references D15, D16)

/// Sections of the measured cascade: the fourth-order Butterworth low-pass at 0.1 of the sample
/// rate (the filter of the golden cases), `[b0, b1, b2, a1, a2]` per section in `f64`.
fn butter4_lowpass() -> [[f64; 5]; 2] {
    let k = (std::f64::consts::PI * 0.1).tan();
    [0, 1].map(|j| {
        // Damping of the section: 2 sin((2j + 1) pi / 8).
        let d = 2.0 * (std::f64::consts::PI * f64::from(2 * j + 1) / 8.0).sin();
        let n = 1.0 / (1.0 + d * k + k * k);
        [
            k * k * n,
            2.0 * k * k * n,
            k * k * n,
            2.0 * (k * k - 1.0) * n,
            (1.0 - d * k + k * k) * n,
        ]
    })
}

/// A coefficient type of the measurement: `f32`, or the Q2.14 `i16`.
trait BiquadCoef: Copy {
    fn from_f64(v: f64) -> Self;
    /// The value the coefficient stands for, in `f64` (exact).
    fn to_f64(self) -> f64;
}

impl BiquadCoef for f32 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a coefficient rounded to f32"
    )]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl BiquadCoef for i16 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a coefficient within Q2.14"
    )]
    fn from_f64(v: f64) -> Self {
        (v * 16384.0).round() as i16
    }

    fn to_f64(self) -> f64 {
        f64::from(self) / 16384.0
    }
}

fn biquad_coefs<C: BiquadCoef>() -> Vec<[C; 5]> {
    butter4_lowpass()
        .iter()
        .map(|s| s.map(C::from_f64))
        .collect()
}

/// The f64 model of the cascade, the accuracy reference of every biquad comparison: direct
/// form I in `f64` with the values of the coefficients of type `C` (as the kernels see them),
/// without any rounding between the sections. The input is one real channel in any unit (Q15
/// samples as raw integers); the output is in the same unit.
fn biquad_model<C: BiquadCoef>(x: impl Iterator<Item = f64>) -> Vec<f64> {
    let coefs: Vec<[f64; 5]> = biquad_coefs::<C>()
        .iter()
        .map(|s| s.map(BiquadCoef::to_f64))
        .collect();
    let mut hist = vec![[0.0f64; 4]; coefs.len()];
    x.map(|mut v| {
        for (c, h) in coefs.iter().zip(hist.iter_mut()) {
            let [b0, b1, b2, a1, a2] = *c;
            let [x1, x2, y1, y2] = *h;
            let y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            *h = [v, x1, y, y1];
            v = y;
        }
        v
    })
    .collect()
}

/// The model of an interleaved complex signal: re and im filtered apart, interleaved again
/// (the layout of [`c32s`] and [`cq15s`]).
fn biquad_model_complex<C: BiquadCoef>(x: &[f64]) -> Vec<f64> {
    let re = biquad_model::<C>(x.iter().step_by(2).copied());
    let im = biquad_model::<C>(x.iter().skip(1).step_by(2).copied());
    re.into_iter().zip(im).flat_map(|(r, i)| [r, i]).collect()
}

/// Interleaves two planar channels into the layout of [`c32s`] and [`cq15s`].
fn interleave<T: Copy>(re: &[T], im: &[T], f: impl Fn(T) -> f64) -> Vec<f64> {
    re.iter()
        .zip(im)
        .flat_map(|(r, i)| [f(*r), f(*i)])
        .collect()
}

/// `x` viewed as raw `i16` (CMSIS-DSP's `q15_t`).
fn q15_raw(x: &[Q15]) -> &[i16] {
    // SAFETY: `Q15` is `repr(transparent)` over `i16`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

/// Mutable form of [`q15_raw`].
fn q15_raw_mut(x: &mut [Q15]) -> &mut [i16] {
    // SAFETY: as in `q15_raw`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

#[inline(never)]
fn biquad<T: ComplexSample + BiquadSample, B: Biquad<T>>(
    backend: &B,
    state: &mut BiquadState<T, Vec<[T::Coef; 5]>, Vec<T>>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.biquad(state, black_box(input), out));
}

#[inline(never)]
fn biquad_real<T: sdrk_core::RealSample + BiquadSample, B: BiquadReal<T>>(
    backend: &B,
    state: &mut BiquadRealState<T, Vec<[T::Coef; 5]>, Vec<T>>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.biquad_real(state, black_box(input), out));
}

/// Which of the two entries of a kernel with a `<kernel>_fallback` entry a measurement is (the
/// biquads of D15 and D16, the spectrum kernels, `dot`, `dot_real`, `scale`, `rotate`, `fm_mod`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BiquadEntry {
    /// `biquad`, `biquad_cq15`, `biquad_real`, `biquad_real_q15`: our kernel through the facade
    /// against KFR (f32, C32) or our fallback (Q15, CQ15), CMSIS-DSP as information.
    Simd,
    /// `<kernel>_fallback`: our fallback against CMSIS-DSP's generic C on the same input.
    Fallback,
}

impl BiquadEntry {
    /// The implementations the entry runs, given those this build has.
    fn supports(self, which: Impl, kfr: bool, cmsis: bool) -> bool {
        match (self, which) {
            (BiquadEntry::Simd, Impl::Ours) | (_, Impl::Fallback) => true,
            (BiquadEntry::Simd, Impl::Kfr) => kfr,
            (_, Impl::Cmsis) => cmsis,
            _ => false,
        }
    }
}

/// The text of [`Kernel::describe`] for KFR: the release, what the shim calls, the architecture
/// KFR was compiled for, and how the channels are arranged.
fn describe_kfr(layout: &str) -> &'static str {
    let arch = KfrBiquad::arch().unwrap_or("?");
    Box::leak(
        format!("KFR 7.2.0 kfr::iir, iir_state<float, 2> ({arch}, -ffp-contract=off){layout}")
            .into_boxed_str(),
    )
}

/// The text of [`Kernel::describe`] for CMSIS-DSP.
fn describe_cmsis(function: &str) -> &'static str {
    Box::leak(
        format!(
            "CMSIS-DSP 1.18.0 {function}, generic C (-O3 -ffp-contract=off, LOOPUNROLL, {})",
            cref_sys::BIQUAD_REF_LEVEL
        )
        .into_boxed_str(),
    )
}

/// The two channels of a complex buffer, apart: KFR has no complex type, and CMSIS-DSP's Q15
/// cascade has one channel, so they run twice on planar re and im (D15: `2 N` real values; the
/// deinterleaving is done once at preparation and not counted, the best case for them).
struct Planar<S> {
    re: Aligned<S>,
    im: Aligned<S>,
    out_re: Aligned<S>,
    out_im: Aligned<S>,
}

impl<S: Copy + Default> Planar<S> {
    fn new<T: Copy>(input: &[T], layout: Layout, split: impl Fn(T) -> (S, S)) -> Self {
        let skew = layout.skew(core::mem::size_of::<S>());
        let mut p = Planar {
            re: Aligned::skewed(input.len(), skew),
            im: Aligned::skewed(input.len(), skew),
            out_re: Aligned::skewed(input.len(), skew),
            out_im: Aligned::skewed(input.len(), skew),
        };
        for ((r, i), x) in p.re.iter_mut().zip(p.im.iter_mut()).zip(input) {
            (*r, *i) = split(*x);
        }
        p
    }
}

/// `biquad`: the order-4 Butterworth low-pass (two sections) on noise of the amplitude 0.3.
///
/// - `C32`, [`BiquadEntry::Simd`]: KFR on planar re and im, CMSIS-DSP's stereo cascade (the two
///   channels of the C32 layout) as information.
/// - `CQ15`, [`BiquadEntry::Simd`]: our fallback (no SIMD Q15 biquad exists on our levels),
///   CMSIS-DSP's Q15 cascade on planar re and im as information.
/// - [`BiquadEntry::Fallback`]: our fallback against the CMSIS-DSP cascade above.
struct BiquadK<T: SquelchSample + BiquadSample, S> {
    input: Aligned<T>,
    out: Aligned<T>,
    planar: Planar<S>,
    entry: BiquadEntry,
}

impl<T: SquelchSample + BiquadSample, S: Copy + Default> BiquadK<T, S>
where
    T::Coef: BiquadCoef,
{
    fn new(
        rng: &mut Pcg32,
        layout: Layout,
        entry: BiquadEntry,
        split: impl Fn(T) -> (S, S),
    ) -> Self {
        let mut input = Aligned::<T>::skewed(layout.n, layout.skew(core::mem::size_of::<T>()));
        for s in input.iter_mut() {
            *s = T::noise(rng, 0.3);
        }
        BiquadK {
            planar: Planar::new(&input, layout, split),
            input,
            out: Aligned::skewed(layout.n, layout.skew(core::mem::size_of::<T>())),
            entry,
        }
    }

    fn state() -> BiquadState<T, Vec<[T::Coef; 5]>, Vec<T>> {
        BiquadState::new(biquad_coefs::<T::Coef>(), vec![T::default(); 8]).expect("biquad")
    }
}

impl Kernel for BiquadK<C32, f32> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let coefs = biquad_coefs::<f32>();
        let mut kfr =
            (which == Impl::Kfr).then(|| (KfrBiquad::new(&coefs), KfrBiquad::new(&coefs)));
        let mut cmsis = (which == Impl::Cmsis).then(|| CmsisBiquad::stereo_df2t_f32(&coefs));
        for _ in 0..k {
            match which {
                Impl::Ours => biquad(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => biquad(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::Kfr => {
                    let (re, im) = kfr.as_mut().expect("kfr");
                    let p = &mut self.planar;
                    re.as_mut().expect("kfr").run(&p.re, &mut p.out_re);
                    im.as_mut().expect("kfr").run(&p.im, &mut p.out_im);
                    black_box(&p.out_re);
                    black_box(&p.out_im);
                }
                Impl::Cmsis => cmsis
                    .as_mut()
                    .and_then(Option::as_mut)
                    .expect("cmsis")
                    .run_stereo(c32_pairs_ref(&self.input), c32_pairs(&mut self.out)),
                other => unreachable!("biquad has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = biquad_model_complex::<f32>(&c32s(&self.input));
        let got = if which == Impl::Kfr {
            interleave(&self.planar.out_re, &self.planar.out_im, f64::from)
        } else {
            c32s(&self.out)
        };
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        let coefs = biquad_coefs::<f32>();
        self.entry.supports(
            which,
            KfrBiquad::new(&coefs).is_some(),
            CmsisBiquad::stereo_df2t_f32(&coefs).is_some(),
        )
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Kfr => {
                describe_kfr(", two calls on planar re and im (deinterleaving not counted)")
            }
            Impl::Cmsis => {
                describe_cmsis("arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels)")
            }
            _ => "sdrk",
        }
    }
}

impl Kernel for BiquadK<CQ15, i16> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let coefs = biquad_coefs::<i16>();
        let mut cmsis = (which == Impl::Cmsis)
            .then(|| (CmsisBiquad::df1_q15(&coefs), CmsisBiquad::df1_q15(&coefs)));
        for _ in 0..k {
            match which {
                Impl::Ours => biquad(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => biquad(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::Cmsis => {
                    let (re, im) = cmsis.as_mut().expect("cmsis");
                    let p = &mut self.planar;
                    re.as_mut().expect("cmsis").run_q15(&p.re, &mut p.out_re);
                    im.as_mut().expect("cmsis").run_q15(&p.im, &mut p.out_im);
                    black_box(&p.out_re);
                    black_box(&p.out_im);
                }
                other => unreachable!("biquad_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = biquad_model_complex::<i16>(&cq15s(&self.input, 1.0));
        let got = if which == Impl::Cmsis {
            interleave(&self.planar.out_re, &self.planar.out_im, f64::from)
        } else {
            cq15s(&self.out, 1.0)
        };
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        let cmsis = CmsisBiquad::df1_q15(&biquad_coefs::<i16>()).is_some();
        self.entry.supports(which, false, cmsis)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Cmsis => describe_cmsis(
                "arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted)",
            ),
            _ => "sdrk",
        }
    }
}

/// `biquad_real` on the same filter and noise as `biquad`.
///
/// - `f32`, [`BiquadEntry::Simd`]: KFR, CMSIS-DSP's `df2T_f32` as information.
/// - `Q15`, [`BiquadEntry::Simd`]: our fallback, CMSIS-DSP's `df1_q15` as information.
/// - [`BiquadEntry::Fallback`]: our fallback against the CMSIS-DSP cascade above.
struct BiquadRealK<T: RealNoise + BiquadSample> {
    input: Aligned<T>,
    out: Aligned<T>,
    entry: BiquadEntry,
}

impl<T: RealNoise + BiquadSample> BiquadRealK<T>
where
    T::Coef: BiquadCoef,
{
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut input = Aligned::<T>::skewed(layout.n, layout.skew(core::mem::size_of::<T>()));
        for s in input.iter_mut() {
            *s = T::noise(rng, 0.3, 0.0);
        }
        BiquadRealK {
            input,
            out: Aligned::skewed(layout.n, layout.skew(core::mem::size_of::<T>())),
            entry,
        }
    }

    fn state() -> BiquadRealState<T, Vec<[T::Coef; 5]>, Vec<T>> {
        BiquadRealState::new(biquad_coefs::<T::Coef>(), vec![T::default(); 8]).expect("biquad_real")
    }

    fn accuracy_of(&self) -> f64 {
        let want = biquad_model::<T::Coef>(self.input.iter().map(|v| v.as_f64()));
        let got: Vec<f64> = self.out.iter().map(|v| v.as_f64()).collect();
        snr_db(&want, &got)
    }
}

impl Kernel for BiquadRealK<f32> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let coefs = biquad_coefs::<f32>();
        let mut kfr = (which == Impl::Kfr).then(|| KfrBiquad::new(&coefs));
        let mut cmsis = (which == Impl::Cmsis).then(|| CmsisBiquad::df2t_f32(&coefs));
        for _ in 0..k {
            match which {
                Impl::Ours => biquad_real(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => biquad_real(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::Kfr => kfr
                    .as_mut()
                    .and_then(Option::as_mut)
                    .expect("kfr")
                    .run(&self.input, &mut self.out),
                Impl::Cmsis => cmsis
                    .as_mut()
                    .and_then(Option::as_mut)
                    .expect("cmsis")
                    .run_f32(&self.input, &mut self.out),
                other => unreachable!("biquad_real has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        self.accuracy_of()
    }

    fn supports(&self, which: Impl) -> bool {
        let coefs = biquad_coefs::<f32>();
        self.entry.supports(
            which,
            KfrBiquad::new(&coefs).is_some(),
            CmsisBiquad::df2t_f32(&coefs).is_some(),
        )
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Kfr => describe_kfr(""),
            Impl::Cmsis => describe_cmsis("arm_biquad_cascade_df2T_f32"),
            _ => "sdrk",
        }
    }
}

impl Kernel for BiquadRealK<Q15> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let mut cmsis =
            (which == Impl::Cmsis).then(|| CmsisBiquad::df1_q15(&biquad_coefs::<i16>()));
        for _ in 0..k {
            match which {
                Impl::Ours => biquad_real(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => biquad_real(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::Cmsis => cmsis
                    .as_mut()
                    .and_then(Option::as_mut)
                    .expect("cmsis")
                    .run_q15(q15_raw(&self.input), q15_raw_mut(&mut self.out)),
                other => unreachable!("biquad_real_q15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        self.accuracy_of()
    }

    fn supports(&self, which: Impl) -> bool {
        let cmsis = CmsisBiquad::df1_q15(&biquad_coefs::<i16>()).is_some();
        self.entry.supports(which, false, cmsis)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Cmsis => describe_cmsis("arm_biquad_cascade_df1_q15 (postShift 1, truncating)"),
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// dot, dot_real and scale (issue #20, group C)

/// The supported implementations of the `dot`, `dot_real` and `scale` entries: the kernel
/// entry runs ours, our fallback and VOLK (where `volk`), the fallback entry our fallback and
/// CMSIS-DSP's generic C (D16).
fn dot_scale_supports(entry: BiquadEntry, which: Impl, volk: bool) -> bool {
    match (entry, which) {
        (_, Impl::Fallback) | (BiquadEntry::Simd, Impl::Ours) => true,
        (BiquadEntry::Simd, Impl::Volk) => volk,
        (BiquadEntry::Fallback, Impl::Cmsis) => cmsis_dsp::AVAILABLE,
        _ => false,
    }
}

/// The planar copy of the components of `x` for CMSIS-DSP's real functions (made once at
/// preparation and not counted, the best case for CMSIS-DSP, as `Planar` of the biquads).
fn planar_of<S: Copy + Default>(
    n: usize,
    skew: usize,
    x: impl Iterator<Item = (S, S)>,
) -> (Aligned<S>, Aligned<S>) {
    let (mut re, mut im) = (Aligned::<S>::skewed(n, skew), Aligned::<S>::skewed(n, skew));
    for ((r, i), (a, b)) in re.iter_mut().zip(im.iter_mut()).zip(x) {
        *r = a;
        *i = b;
    }
    (re, im)
}

/// CMSIS-DSP's Q34.30 dot product as Q15, rounded half up and saturated as our kernel does;
/// for the accuracy only (a caller needs it to get a Q15 result), outside the counted call.
fn cmsis_q15_of(sum: i64) -> Q15 {
    Q15(((sum + (1 << 14)) >> 15).clamp(-32768, 32767) as i16)
}

/// Windows of the accuracy runs of the dot products: `DOT_WINDOWS` dot products of `n / 2`
/// samples each, at offsets `w * n / 32`, so that one rounding error does not decide the SNR.
const DOT_WINDOWS: usize = 16;

/// The windows `(start, len)` of the accuracy runs for a buffer of `n` samples.
fn dot_windows(n: usize) -> impl Iterator<Item = (usize, usize)> {
    let len = (n / 2).max(1).min(n);
    (0..DOT_WINDOWS).map(move |w| ((w * n / 32).min(n - len), len))
}

#[inline(never)]
fn dot_c32<B: Dot<C32>>(backend: &B, x: &[C32], taps: &[f32]) -> C32 {
    black_box(backend.dot(black_box(x), black_box(taps))).unwrap_or_default()
}

#[inline(never)]
fn dot_cq15<B: Dot<CQ15>>(backend: &B, x: &[CQ15], taps: &[Q15]) -> CQ15 {
    black_box(backend.dot(black_box(x), black_box(taps))).unwrap_or_default()
}

#[inline(never)]
fn dot_f32<B: DotReal<f32>>(backend: &B, x: &[f32], taps: &[f32]) -> f32 {
    black_box(backend.dot_real(black_box(x), black_box(taps))).unwrap_or_default()
}

#[inline(never)]
fn dot_q15<B: DotReal<Q15>>(backend: &B, x: &[Q15], taps: &[Q15]) -> Q15 {
    black_box(backend.dot_real(black_box(x), black_box(taps))).unwrap_or_default()
}

/// `dot` of `n` C32 samples with `n` real taps, one call per run (VOLK:
/// `volk_32fc_32f_dot_prod_32fc` through its dispatcher, which checks the alignment).
///
/// `dot_fallback`: our fallback against CMSIS-DSP's `arm_dot_prod_f32` on the planar real and
/// imaginary parts (two calls, the same `2 n` products; CMSIS-DSP has no complex dot product
/// with real taps, and `arm_cmplx_dot_prod_f32` with zero imaginary taps would do twice the
/// work).
struct DotC32K {
    x: Aligned<C32>,
    taps: Aligned<f32>,
    result: Aligned<C32>,
    entry: BiquadEntry,
    planar: (Aligned<f32>, Aligned<f32>),
}

impl DotC32K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let x = noise_c32(rng, layout.n, 0.9, layout.skew(8));
        let mut t = Aligned::<f32>::skewed(layout.n, layout.skew(4));
        t.copy_from_slice(&taps(rng, layout.n));
        let planar = planar_of(layout.n, layout.skew(4), x.iter().map(|v| (v.re, v.im)));
        DotC32K {
            x,
            taps: t,
            result: Aligned::new(1),
            entry,
            planar,
        }
    }

    fn call(&mut self, which: Impl, start: usize, len: usize) -> C32 {
        let (x, t) = (&self.x[start..start + len], &self.taps[start..start + len]);
        match which {
            Impl::Ours => dot_c32(&Sdrk, x, t),
            Impl::Fallback => dot_c32(&Fallback, x, t),
            // SAFETY: `x` and `t` hold `len` elements, `result` one.
            Impl::Volk => unsafe {
                volk_sys::volk_32fc_32f_dot_prod_32fc(
                    self.result.as_mut_ptr().cast(),
                    x.as_ptr().cast(),
                    t.as_ptr(),
                    len as u32,
                );
                self.result[0]
            },
            Impl::Cmsis => {
                let re = &self.planar.0[start..start + len];
                let im = &self.planar.1[start..start + len];
                let (mut r, mut i) = (0.0f32, 0.0f32);
                // SAFETY: `re`, `im` and `t` hold `len` elements; the results are locals.
                unsafe {
                    let n = len as u32;
                    cmsis_dsp::ffi::arm_dot_prod_f32(re.as_ptr(), t.as_ptr(), n, &raw mut r);
                    cmsis_dsp::ffi::arm_dot_prod_f32(im.as_ptr(), t.as_ptr(), n, &raw mut i);
                }
                black_box(C32::new(r, i))
            }
            other => unreachable!("dot has no {other:?}"),
        }
    }
}

impl Kernel for DotC32K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.x.len();
        for _ in 0..k {
            black_box(self.call(which, 0, n));
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let (mut want, mut got) = (Vec::new(), Vec::new());
        for (s, len) in dot_windows(self.x.len()) {
            let y = self.call(which, s, len);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (v, &h) in self.x[s..s + len].iter().zip(&self.taps[s..s + len]) {
                re += f64::from(v.re) * f64::from(h);
                im += f64::from(v.im) * f64::from(h);
            }
            want.extend([re, im]);
            got.extend([f64::from(y.re), f64::from(y.im)]);
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, volk_sys::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32fc_32f_dot_prod_32fc",
            Impl::Cmsis => describe_cmsis("arm_dot_prod_f32 twice, on the planar re and im"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if self.entry == BiquadEntry::Simd {
            &["volk_32fc_32f_dot_prod_32fc"]
        } else {
            &[]
        }
    }
}

/// `dot_real` of `n` f32 samples with `n` taps, one call per run (VOLK:
/// `volk_32f_x2_dot_prod_32f`, which VOLK's wasm32 build does not have).
///
/// `dot_real_fallback`: our fallback against CMSIS-DSP's `arm_dot_prod_f32`.
struct DotF32K {
    x: Aligned<f32>,
    taps: Aligned<f32>,
    result: Aligned<f32>,
    entry: BiquadEntry,
}

impl DotF32K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut x = Aligned::<f32>::skewed(layout.n, layout.skew(4));
        for v in x.iter_mut() {
            *v = rng.unit() * 0.9;
        }
        let mut t = Aligned::<f32>::skewed(layout.n, layout.skew(4));
        t.copy_from_slice(&taps(rng, layout.n));
        DotF32K {
            x,
            taps: t,
            result: Aligned::new(1),
            entry,
        }
    }

    fn call(&mut self, which: Impl, start: usize, len: usize) -> f32 {
        let (x, t) = (&self.x[start..start + len], &self.taps[start..start + len]);
        match which {
            Impl::Ours => dot_f32(&Sdrk, x, t),
            Impl::Fallback => dot_f32(&Fallback, x, t),
            // SAFETY: `x` and `t` hold `len` elements, `result` one.
            Impl::Volk => unsafe {
                volk_sys::volk_32f_x2_dot_prod_32f(
                    self.result.as_mut_ptr(),
                    x.as_ptr(),
                    t.as_ptr(),
                    len as u32,
                );
                self.result[0]
            },
            Impl::Cmsis => {
                let mut r = 0.0f32;
                // SAFETY: `x` and `t` hold `len` elements; the result is a local.
                unsafe {
                    cmsis_dsp::ffi::arm_dot_prod_f32(
                        x.as_ptr(),
                        t.as_ptr(),
                        len as u32,
                        &raw mut r,
                    );
                }
                black_box(r)
            }
            other => unreachable!("dot_real has no {other:?}"),
        }
    }
}

impl Kernel for DotF32K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.x.len();
        for _ in 0..k {
            black_box(self.call(which, 0, n));
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let (mut want, mut got) = (Vec::new(), Vec::new());
        for (s, len) in dot_windows(self.x.len()) {
            let y = self.call(which, s, len);
            let w: f64 = self.x[s..s + len]
                .iter()
                .zip(&self.taps[s..s + len])
                .map(|(&v, &h)| f64::from(v) * f64::from(h))
                .sum();
            want.push(w);
            got.push(f64::from(y));
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, volk_sys::NATIVE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32f_x2_dot_prod_32f",
            Impl::Cmsis => describe_cmsis("arm_dot_prod_f32"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if volk_sys::NATIVE && self.entry == BiquadEntry::Simd {
            &["volk_32f_x2_dot_prod_32f"]
        } else {
            &[]
        }
    }
}

/// Q15 taps whose dot product with `n` samples of amplitude 23000 stays far from saturation.
fn taps_q15(rng: &mut Pcg32, n: usize, skew: usize) -> Aligned<Q15> {
    let amp = 16384.0 / (n as f32).sqrt();
    let mut t = Aligned::<Q15>::skewed(n, skew);
    for v in t.iter_mut() {
        *v = Q15((rng.unit() * amp) as i16);
    }
    t
}

/// `dot` of `n` CQ15 samples with `n` Q15 taps, one call per run. VOLK's 16-bit dot products
/// do other work (`volk_16ic_x2_dot_prod_16ic`: complex taps, saturating 16-bit sums), so the
/// comparison is our fallback.
///
/// `dot_cq15_fallback`: our fallback against CMSIS-DSP's `arm_dot_prod_q15` on the planar real
/// and imaginary parts (two calls; its `i64` sums are exact like ours, without the final
/// rounding and saturation, which the harness applies outside the counted calls).
struct DotCq15K {
    x: Aligned<CQ15>,
    taps: Aligned<Q15>,
    entry: BiquadEntry,
    planar: (Aligned<i16>, Aligned<i16>),
}

impl DotCq15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let x = noise_cq15(rng, layout.n, 23_000.0, layout.skew(4));
        let planar = planar_of(layout.n, layout.skew(2), x.iter().map(|v| (v.re.0, v.im.0)));
        DotCq15K {
            x,
            taps: taps_q15(rng, layout.n, layout.skew(2)),
            entry,
            planar,
        }
    }

    fn call(&self, which: Impl, start: usize, len: usize) -> CQ15 {
        let (x, t) = (&self.x[start..start + len], &self.taps[start..start + len]);
        match which {
            Impl::Ours => dot_cq15(&Sdrk, x, t),
            Impl::Fallback => dot_cq15(&Fallback, x, t),
            Impl::Cmsis => {
                let re = &self.planar.0[start..start + len];
                let im = &self.planar.1[start..start + len];
                let t = q15_raw(t);
                let (mut r, mut i) = (0i64, 0i64);
                // SAFETY: `re`, `im` and `t` hold `len` elements; the results are locals.
                unsafe {
                    let n = len as u32;
                    cmsis_dsp::ffi::arm_dot_prod_q15(re.as_ptr(), t.as_ptr(), n, &raw mut r);
                    cmsis_dsp::ffi::arm_dot_prod_q15(im.as_ptr(), t.as_ptr(), n, &raw mut i);
                }
                let (r, i) = black_box((r, i));
                CQ15 {
                    re: cmsis_q15_of(r),
                    im: cmsis_q15_of(i),
                }
            }
            other => unreachable!("dot_cq15 has no {other:?}"),
        }
    }
}

impl Kernel for DotCq15K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.x.len();
        for _ in 0..k {
            black_box(self.call(which, 0, n));
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let (mut want, mut got) = (Vec::new(), Vec::new());
        for (s, len) in dot_windows(self.x.len()) {
            let y = self.call(which, s, len);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (v, h) in self.x[s..s + len].iter().zip(&self.taps[s..s + len]) {
                re += f64::from(v.re.0) * f64::from(h.0);
                im += f64::from(v.im.0) * f64::from(h.0);
            }
            want.extend([re / 32768.0, im / 32768.0]);
            got.extend([f64::from(y.re.0), f64::from(y.im.0)]);
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, false)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Cmsis => describe_cmsis("arm_dot_prod_q15 twice, on the planar re and im"),
            _ => "sdrk",
        }
    }
}

/// `dot_real` of `n` Q15 samples with `n` Q15 taps, one call per run (VOLK has no 16-bit real
/// dot product: compared with our fallback).
///
/// `dot_real_q15_fallback`: our fallback against CMSIS-DSP's `arm_dot_prod_q15` (an exact `i64`
/// sum, rounded to Q15 outside the counted call).
struct DotQ15K {
    x: Aligned<Q15>,
    taps: Aligned<Q15>,
    entry: BiquadEntry,
}

impl DotQ15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let mut x = Aligned::<Q15>::skewed(layout.n, layout.skew(2));
        for v in x.iter_mut() {
            *v = Q15((rng.unit() * 23_000.0) as i16);
        }
        DotQ15K {
            x,
            taps: taps_q15(rng, layout.n, layout.skew(2)),
            entry,
        }
    }

    fn call(&self, which: Impl, start: usize, len: usize) -> Q15 {
        let (x, t) = (&self.x[start..start + len], &self.taps[start..start + len]);
        match which {
            Impl::Ours => dot_q15(&Sdrk, x, t),
            Impl::Fallback => dot_q15(&Fallback, x, t),
            Impl::Cmsis => {
                let mut r = 0i64;
                // SAFETY: `x` and `t` hold `len` elements; the result is a local.
                unsafe {
                    cmsis_dsp::ffi::arm_dot_prod_q15(
                        q15_raw(x).as_ptr(),
                        q15_raw(t).as_ptr(),
                        len as u32,
                        &raw mut r,
                    );
                }
                cmsis_q15_of(black_box(r))
            }
            other => unreachable!("dot_real_q15 has no {other:?}"),
        }
    }
}

impl Kernel for DotQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.x.len();
        for _ in 0..k {
            black_box(self.call(which, 0, n));
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let (mut want, mut got) = (Vec::new(), Vec::new());
        for (s, len) in dot_windows(self.x.len()) {
            let y = self.call(which, s, len);
            let w: f64 = self.x[s..s + len]
                .iter()
                .zip(&self.taps[s..s + len])
                .map(|(v, h)| f64::from(v.0) * f64::from(h.0))
                .sum();
            want.push(w / 32768.0);
            got.push(f64::from(y.0));
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, false)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Cmsis => describe_cmsis("arm_dot_prod_q15"),
            _ => "sdrk",
        }
    }
}

#[inline(never)]
fn scale<T: ComplexSample, B: Scale<T>>(backend: &B, input: &[T], out: &mut [T], gain: T::Real) {
    let _ = black_box(backend.scale(black_box(input), out, gain));
}

/// The gain of the `scale` measurements.
const SCALE_GAIN: f32 = 0.734_375;

/// `scale` of `n` C32 samples by a real gain. VOLK: `volk_32f_s32f_multiply_32f` on the `2n`
/// floats, the same work (VOLK's wasm32 build does not have it).
///
/// `scale_fallback`: our fallback against CMSIS-DSP's `arm_scale_f32` on the `2 n` floats.
struct ScaleK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    entry: BiquadEntry,
}

impl ScaleK {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        ScaleK {
            input: noise_c32(rng, layout.n, 0.9, layout.skew(8)),
            out: Aligned::skewed(layout.n, layout.skew(8)),
            entry,
        }
    }
}

impl Kernel for ScaleK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => scale(&Sdrk, &self.input, &mut self.out, SCALE_GAIN),
                Impl::Fallback => scale(&Fallback, &self.input, &mut self.out, SCALE_GAIN),
                // SAFETY: input and output hold `n` complex samples, `2n` floats.
                Impl::Volk => unsafe {
                    volk_sys::native::volk_32f_s32f_multiply_32f(
                        self.out.as_mut_ptr().cast(),
                        self.input.as_ptr().cast(),
                        SCALE_GAIN,
                        2 * self.input.len() as u32,
                    );
                },
                // SAFETY: input and output hold `n` complex samples, `2n` floats.
                Impl::Cmsis => unsafe {
                    cmsis_dsp::ffi::arm_scale_f32(
                        self.input.as_ptr().cast(),
                        SCALE_GAIN,
                        self.out.as_mut_ptr().cast(),
                        2 * self.input.len() as u32,
                    );
                },
                other => unreachable!("scale has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let g = f64::from(SCALE_GAIN);
        let want: Vec<f64> = c32s(&self.input).iter().map(|v| v * g).collect();
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, volk_sys::NATIVE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32f_s32f_multiply_32f",
            Impl::Cmsis => describe_cmsis("arm_scale_f32 on the 2 n floats"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if volk_sys::NATIVE && self.entry == BiquadEntry::Simd {
            &["volk_32f_s32f_multiply_32f"]
        } else {
            &[]
        }
    }
}

/// `scale` of `n` CQ15 samples by a Q15 gain (`q15_mul`, rounded and saturated; VOLK has no
/// 16-bit scaling with this rounding: compared with our fallback).
///
/// `scale_cq15_fallback`: our fallback against CMSIS-DSP's `arm_scale_q15` with shift 0 on the
/// `2 n` values (the same product, truncated instead of rounded: a lower accuracy class).
struct ScaleCq15K {
    input: Aligned<CQ15>,
    out: Aligned<CQ15>,
    entry: BiquadEntry,
}

impl ScaleCq15K {
    fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        ScaleCq15K {
            input: noise_cq15(rng, layout.n, 32_767.0, layout.skew(4)),
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

impl Kernel for ScaleCq15K {
    fn run(&mut self, which: Impl, k: usize) {
        let gain = Q15::from_f32(SCALE_GAIN);
        for _ in 0..k {
            match which {
                Impl::Ours => scale(&Sdrk, &self.input, &mut self.out, gain),
                Impl::Fallback => scale(&Fallback, &self.input, &mut self.out, gain),
                // SAFETY: input and output hold `n` complex samples, `2n` values.
                Impl::Cmsis => unsafe {
                    cmsis_dsp::ffi::arm_scale_q15(
                        self.input.as_ptr().cast(),
                        gain.0,
                        0,
                        self.out.as_mut_ptr().cast(),
                        2 * self.input.len() as u32,
                    );
                },
                other => unreachable!("scale_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let g = f64::from(Q15::from_f32(SCALE_GAIN).0) / 32768.0;
        let want: Vec<f64> = cq15s(&self.input, 1.0).iter().map(|v| v * g).collect();
        snr_db(&want, &cq15s(&self.out, 1.0))
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, false)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Cmsis => describe_cmsis("arm_scale_q15 (shift 0, truncating) on the 2 n values"),
            _ => "sdrk",
        }
    }
}

#[cfg(test)]
mod biquad_tests {
    //! The references must compute the filter we compare them with: each is checked against
    //! our `Reference` kernel on noise, over several calls (the state must carry over), when its
    //! library is linked (`SDRK_REFS_DIR`; without it the tests pass without checking).

    use super::*;

    /// `input` filtered in three calls of uneven length by `f`.
    ///
    /// No call is shorter than the two sections: KFR 7.2.0 loses the state of its cascade on a
    /// call with fewer samples than sections (its look-ahead consumes the whole call in
    /// `begin_pass`, and `end_pass` then restores a state that was never saved; measured here:
    /// 38.8 dB against the reference with a call of one sample). The measurements call it with
    /// 64 samples and more.
    fn in_parts<T: Copy + Default>(input: &[T], mut f: impl FnMut(&[T], &mut [T])) -> Vec<T> {
        let mut out = vec![T::default(); input.len()];
        let cuts = [0, 7, 1000, input.len()];
        for w in cuts.windows(2) {
            f(&input[w[0]..w[1]], &mut out[w[0]..w[1]]);
        }
        out
    }

    fn noise_real(n: usize) -> Vec<f32> {
        let mut rng = Pcg32::new(7);
        (0..n).map(|_| rng.unit() * 0.3).collect()
    }

    fn reference_real_f32(x: &[f32]) -> Vec<f32> {
        let mut want = vec![0.0f32; x.len()];
        let mut s = BiquadRealState::new(biquad_coefs::<f32>(), vec![0.0f32; 8]).expect("state");
        Reference
            .biquad_real(&mut s, x, &mut want)
            .expect("reference");
        want
    }

    #[test]
    fn kfr_computes_the_cascade() {
        let coefs = biquad_coefs::<f32>();
        if KfrBiquad::new(&coefs).is_none() {
            eprintln!("KFR is not linked; skipped");
            return;
        }
        let x = noise_real(4096);
        let mut kfr = KfrBiquad::new(&coefs).expect("kfr");
        let got = in_parts(&x, |i, o| kfr.run(i, o));
        let snr = snr_db(&f32s(&reference_real_f32(&x)), &f32s(&got));
        assert!(snr > 120.0, "KFR against the reference: {snr} dB");
    }

    #[test]
    fn cmsis_f32_cascades_compute_the_cascade() {
        let coefs = biquad_coefs::<f32>();
        let Some(mut mono) = CmsisBiquad::df2t_f32(&coefs) else {
            eprintln!("CMSIS-DSP is not linked; skipped");
            return;
        };
        let x = noise_real(4096);
        let want = reference_real_f32(&x);
        let got = in_parts(&x, |i, o| mono.run_f32(i, o));
        let snr = snr_db(&f32s(&want), &f32s(&got));
        assert!(
            snr > 120.0,
            "CMSIS df2T_f32 against the reference: {snr} dB"
        );

        // The stereo cascade on C32 against our complex reference.
        let mut rng = Pcg32::new(8);
        let c: Vec<C32> = (0..4096)
            .map(|_| C32::new(rng.unit() * 0.3, rng.unit() * 0.3))
            .collect();
        let mut want = vec![C32::default(); c.len()];
        let mut s = BiquadState::new(coefs.clone(), vec![C32::default(); 8]).expect("state");
        Reference.biquad(&mut s, &c, &mut want).expect("reference");
        let mut stereo = CmsisBiquad::stereo_df2t_f32(&coefs).expect("stereo");
        let got = in_parts(&c, |i, o| stereo.run_stereo(c32_pairs_ref(i), c32_pairs(o)));
        let snr = snr_db(&c32s(&want), &c32s(&got));
        assert!(
            snr > 120.0,
            "CMSIS stereo_df2T_f32 against the reference: {snr} dB"
        );
    }

    #[test]
    fn cmsis_q15_cascade_computes_the_cascade_with_truncation() {
        let coefs = biquad_coefs::<i16>();
        let Some(mut cmsis) = CmsisBiquad::df1_q15(&coefs) else {
            eprintln!("CMSIS-DSP is not linked; skipped");
            return;
        };
        let mut rng = Pcg32::new(9);
        let x: Vec<Q15> = (0..4096)
            .map(|_| Q15((rng.unit() * 0.3 * 32768.0) as i16))
            .collect();
        let mut want = vec![Q15(0); x.len()];
        let mut s = BiquadRealState::new(coefs.clone(), vec![Q15(0); 8]).expect("state");
        Reference
            .biquad_real(&mut s, &x, &mut want)
            .expect("reference");
        let got = in_parts(&x, |i, o| cmsis.run_q15(q15_raw(i), q15_raw_mut(o)));
        // CMSIS-DSP truncates where we round half up: the outputs differ by small amounts that
        // the recursion carries on, but they are the same filter.
        let as_f64 = |v: &[Q15]| v.iter().map(|q| f64::from(q.0)).collect::<Vec<_>>();
        let snr = snr_db(&as_f64(&want), &as_f64(&got));
        assert!(snr > 50.0, "CMSIS df1_q15 against the reference: {snr} dB");
        // A negated -2.0 does not exist in Q15.
        let mut bad = coefs;
        bad[0][3] = i16::MIN;
        assert!(CmsisBiquad::df1_q15(&bad).is_none());
    }

    #[test]
    fn the_model_matches_the_reference_kernel() {
        let x = noise_real(4096);
        let want = reference_real_f32(&x);
        let model = biquad_model::<f32>(x.iter().map(|v| f64::from(*v)));
        let snr = snr_db(&model, &f32s(&want));
        assert!(snr > 120.0, "model against the reference: {snr} dB");
    }

    #[test]
    fn the_measurements_run_every_implementation_they_support() {
        for name in [
            "biquad",
            "biquad_cq15",
            "biquad_real",
            "biquad_real_q15",
            "biquad_fallback",
            "biquad_cq15_fallback",
            "biquad_real_fallback",
            "biquad_real_q15_fallback",
            "dc_block",
            "dc_block_real",
            "deemphasis",
            "dc_block_fallback",
            "dc_block_real_fallback",
            "deemphasis_fallback",
            "dot",
            "dot_cq15",
            "dot_real",
            "dot_real_q15",
            "scale",
            "scale_cq15",
            "dot_fallback",
            "dot_cq15_fallback",
            "dot_real_fallback",
            "dot_real_q15_fallback",
            "scale_fallback",
            "scale_cq15_fallback",
            "rotate",
            "rotate_cq15",
            "rotate_fallback",
            "rotate_cq15_fallback",
            "fm_mod",
            "fm_mod_cq15",
            "fm_mod_fallback",
            "fm_mod_cq15_fallback",
        ] {
            let mut kernel = prepare(
                name,
                Layout {
                    n: 512,
                    misaligned: true,
                },
            )
            .expect(name);
            for which in Impl::ALL {
                if kernel.supports(which) {
                    let snr = kernel.accuracy(which);
                    assert!(snr > 40.0, "{name} {}: {snr} dB", which.name());
                }
            }
        }
    }
}
