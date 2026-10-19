//! The measured kernels: for every pair of the benchmark table, the buffers, the call of our
//! kernel, the call of the comparison and the accuracy against the f64 reference.
//!
//! The binary `count` drives them; `cargo xtask count` runs the binary under the instruction
//! counters.

use std::hint::black_box;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use sdrk::kernels::convert::{Convert, ConvertOut};
use sdrk::kernels::decim::{FirDecimState, FirDecimate};
use sdrk::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk::kernels::filter::{Fir, FirState};
use sdrk::kernels::mix::{Nco, Rotate};
use sdrk::kernels::modulate::{FmMod, FmModState};
use sdrk::kernels::spectrum::MagSq;
use sdrk::{C32, CQ15, Fallback, Q15, Rate, Sdrk};
use sdrk_core::Reference;
use sdrk_device::formats;
use volk_sys::lv_32fc_t;

/// Samples per measurement unless overridden.
pub const N: usize = 4096;

/// Which implementation to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Impl {
    /// Our kernel through the facade (the SIMD backend unless the build forces the fallback).
    Ours,
    /// The comparison library of the pair (VOLK or rustfft).
    Library,
    /// Our portable fallback.
    Fallback,
}

impl Impl {
    /// Parses the command-line name.
    pub fn parse(s: &str) -> Option<Impl> {
        match s {
            "ours" => Some(Impl::Ours),
            "library" => Some(Impl::Library),
            "fallback" => Some(Impl::Fallback),
            _ => None,
        }
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

/// A vector whose data pointer is 64-byte aligned (VOLK then takes its aligned kernels, and
/// our loads do not straddle cache lines).
pub struct Aligned<T> {
    v: Vec<T>,
    off: usize,
    len: usize,
}

impl<T: Copy + Default> Aligned<T> {
    /// `len` default-initialised elements.
    pub fn new(len: usize) -> Self {
        let size = std::mem::size_of::<T>().max(1);
        let pad = 64 / size + 1;
        let v = vec![T::default(); len + pad];
        let off = (0..pad)
            .find(|o| (v.as_ptr() as usize + o * size) % 64 == 0)
            .unwrap_or(0);
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

fn noise_c32(rng: &mut Pcg32, n: usize, amp: f32) -> Aligned<C32> {
    let mut v = Aligned::<C32>::new(n);
    for s in v.iter_mut() {
        *s = C32::new(rng.unit() * amp, rng.unit() * amp);
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

fn cq15s(x: &[CQ15]) -> Vec<f64> {
    x.iter()
        .flat_map(|v| [f64::from(v.re.0), f64::from(v.im.0)])
        .collect()
}

/// A prepared measurement.
pub trait Kernel {
    /// Runs the kernel `k` times on the same buffers with implementation `which`.
    fn run(&mut self, which: Impl, k: usize);
    /// The SNR in dB of one run of `which` against the f64 reference.
    fn accuracy(&mut self, which: Impl) -> f64;
    /// Whether a comparison library exists for this kernel on this build.
    fn has_library(&self) -> bool;
    /// The name of the comparison, for the reports.
    fn library_name(&self) -> &'static str;
}

/// All kernel names in report order.
pub const KERNELS: [&str; 13] = [
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
];

/// Prepares the measurement `name` for `n` samples (the FFT sizes are fixed by the name).
pub fn prepare(name: &str, n: usize) -> Option<Box<dyn Kernel>> {
    let idx = KERNELS.iter().position(|k| *k == name)? as u64;
    let mut rng = Pcg32::new(1000 + idx);
    Some(match name {
        "convert_cs8" => Box::new(ConvertCs8::new(&mut rng, n)),
        "convert_cs16" => Box::new(ConvertCs16::new(&mut rng, n)),
        "convert_out_cs16" => Box::new(ConvertOutCs16::new(&mut rng, n)),
        "rotate" => Box::new(RotateK::new(&mut rng, n)),
        "fir" => Box::new(FirK::new(&mut rng, n)),
        "fir_decimate" => Box::new(DecimateK::new(&mut rng, n)),
        "mag_sq" => Box::new(MagSqK::new(&mut rng, n)),
        "quad_demod" => Box::new(QuadK::new(&mut rng, n)),
        "fm_mod" => Box::new(FmModK::new(&mut rng, n)),
        "fft_64" => Box::new(FftK::new(&mut rng, 64)),
        "fft_1024" => Box::new(FftK::new(&mut rng, 1024)),
        "fft_4096" => Box::new(FftK::new(&mut rng, 4096)),
        "fft_cq15_1024" => Box::new(FftQ15K::new(&mut rng, 1024)),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------------------------

struct ConvertCs8 {
    bytes: Aligned<i8>,
    out: Aligned<f32>,
    n: usize,
}

impl ConvertCs8 {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let mut bytes = Aligned::<i8>::new(2 * n);
        for b in bytes.iter_mut() {
            *b = rng.next_u32() as i8;
        }
        ConvertCs8 {
            bytes,
            out: Aligned::new(2 * n),
            n,
        }
    }
}

impl Kernel for ConvertCs8 {
    fn run(&mut self, which: Impl, k: usize) {
        let raw: &[u8] = bytemuck_i8(&self.bytes);
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ = black_box(Sdrk.convert(
                        &formats::CS8,
                        black_box(raw),
                        cast_c32(&mut self.out),
                    ));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.convert(
                        &formats::CS8,
                        black_box(raw),
                        cast_c32(&mut self.out),
                    ));
                }
                // SAFETY: both buffers hold `2 n` elements.
                Impl::Library => unsafe {
                    volk_sys::volk_8i_s32f_convert_32f(
                        self.out.as_mut_ptr(),
                        self.bytes.as_ptr(),
                        128.0,
                        2 * self.n as u32,
                    );
                },
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self.bytes.iter().map(|&b| f64::from(b) / 128.0).collect();
        snr_db(&want, &f32s(&self.out))
    }

    fn has_library(&self) -> bool {
        volk_sys::AVAILABLE
    }

    fn library_name(&self) -> &'static str {
        "volk_8i_s32f_convert_32f"
    }
}

fn bytemuck_i8(x: &[i8]) -> &[u8] {
    // SAFETY: `i8` and `u8` have the same size and alignment and every bit pattern is valid.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), x.len()) }
}

fn cast_c32(x: &mut [f32]) -> &mut [C32] {
    // SAFETY: `C32` is `repr(C)` of two `f32`; the slice length is even.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<C32>(), x.len() / 2) }
}

fn as_lv(x: &[C32]) -> *const lv_32fc_t {
    x.as_ptr().cast()
}

fn as_lv_mut(x: &mut [C32]) -> *mut lv_32fc_t {
    x.as_mut_ptr().cast()
}

struct ConvertCs16 {
    bytes: Aligned<i16>,
    out: Aligned<f32>,
    n: usize,
}

impl ConvertCs16 {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let mut bytes = Aligned::<i16>::new(2 * n);
        for b in bytes.iter_mut() {
            *b = rng.next_u32() as i16;
        }
        ConvertCs16 {
            bytes,
            out: Aligned::new(2 * n),
            n,
        }
    }
}

impl Kernel for ConvertCs16 {
    fn run(&mut self, which: Impl, k: usize) {
        // SAFETY: an `i16` slice viewed as bytes (little-endian target, as CS16 requires).
        let raw: &[u8] =
            unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast::<u8>(), 4 * self.n) };
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ = black_box(Sdrk.convert(
                        &formats::CS16,
                        black_box(raw),
                        cast_c32(&mut self.out),
                    ));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.convert(
                        &formats::CS16,
                        black_box(raw),
                        cast_c32(&mut self.out),
                    ));
                }
                // SAFETY: both buffers hold `2 n` elements.
                Impl::Library => unsafe {
                    volk_sys::volk_16i_s32f_convert_32f(
                        self.out.as_mut_ptr(),
                        self.bytes.as_ptr(),
                        32768.0,
                        2 * self.n as u32,
                    );
                },
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self.bytes.iter().map(|&b| f64::from(b) / 32768.0).collect();
        snr_db(&want, &f32s(&self.out))
    }

    fn has_library(&self) -> bool {
        volk_sys::AVAILABLE
    }

    fn library_name(&self) -> &'static str {
        "volk_16i_s32f_convert_32f"
    }
}

struct ConvertOutCs16 {
    input: Aligned<C32>,
    out: Aligned<i16>,
    n: usize,
}

impl ConvertOutCs16 {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        ConvertOutCs16 {
            input: noise_c32(rng, n, 0.9),
            out: Aligned::new(2 * n),
            n,
        }
    }
}

impl Kernel for ConvertOutCs16 {
    fn run(&mut self, which: Impl, k: usize) {
        // SAFETY: the `i16` buffer viewed as bytes.
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(self.out.as_mut_ptr().cast::<u8>(), 4 * self.n)
        };
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ =
                        black_box(Sdrk.convert_out(&formats::CS16, black_box(&self.input), bytes));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.convert_out(
                        &formats::CS16,
                        black_box(&self.input),
                        bytes,
                    ));
                }
                // SAFETY: the input holds `2 n` floats, the output `2 n` samples.
                Impl::Library => unsafe {
                    volk_sys::volk_32f_s32f_convert_16i(
                        self.out.as_mut_ptr(),
                        self.input.as_ptr().cast::<f32>(),
                        32768.0,
                        2 * self.n as u32,
                    );
                },
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        // The reference quantises to the nearest integer; the SNR is the quantisation noise.
        let want: Vec<f64> = self
            .input
            .iter()
            .flat_map(|v| [f64::from(v.re) * 32768.0, f64::from(v.im) * 32768.0])
            .collect();
        let got: Vec<f64> = self.out.iter().map(|&v| f64::from(v)).collect();
        snr_db(&want, &got)
    }

    fn has_library(&self) -> bool {
        volk_sys::AVAILABLE
    }

    fn library_name(&self) -> &'static str {
        "volk_32f_s32f_convert_16i"
    }
}

struct RotateK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    inc: u32,
}

impl RotateK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        RotateK {
            input: noise_c32(rng, n, 0.9),
            out: Aligned::new(n),
            inc: 0x0123_4567,
        }
    }
}

impl Kernel for RotateK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut nco = Nco::from_increment(self.inc);
        let theta = 2.0 * std::f64::consts::PI * f64::from(self.inc) / 4_294_967_296.0;
        let inc = lv_32fc_t {
            re: theta.cos() as f32,
            im: theta.sin() as f32,
        };
        let mut phase = lv_32fc_t { re: 1.0, im: 0.0 };
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ = black_box(Sdrk.rotate(&mut nco, black_box(&self.input), &mut self.out));
                }
                Impl::Fallback => {
                    let _ =
                        black_box(Fallback.rotate(&mut nco, black_box(&self.input), &mut self.out));
                }
                // SAFETY: input and output hold `n` samples.
                Impl::Library => unsafe {
                    volk_sys::volk_32fc_s32fc_x2_rotator2_32fc(
                        as_lv_mut(&mut self.out),
                        as_lv(&self.input),
                        &inc,
                        &mut phase,
                        self.input.len() as u32,
                    );
                },
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut nco = Nco::from_increment(self.inc);
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.rotate(&mut nco, &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn has_library(&self) -> bool {
        volk_sys::AVAILABLE
    }

    fn library_name(&self) -> &'static str {
        "volk_32fc_s32fc_x2_rotator2_32fc"
    }
}

const FIR_TAPS: usize = 64;
const DECIM_TAPS: usize = 127;
const DECIM: usize = 10;

fn taps(rng: &mut Pcg32, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.unit() / (n as f32).sqrt()).collect()
}

struct FirK {
    /// 63 history samples, then the input.
    buf: Aligned<C32>,
    taps: Vec<f32>,
    rev: Aligned<f32>,
    out: Aligned<C32>,
    state: FirState<C32, Vec<f32>, Vec<C32>>,
    n: usize,
}

impl FirK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let taps = taps(rng, FIR_TAPS);
        let mut rev = Aligned::<f32>::new(FIR_TAPS);
        for (r, t) in rev.iter_mut().zip(taps.iter().rev()) {
            *r = *t;
        }
        let buf = noise_c32(rng, n + FIR_TAPS - 1, 0.9);
        let state = FirState::new(taps.clone(), vec![C32::default(); FIR_TAPS - 1]).expect("taps");
        FirK {
            buf,
            taps,
            rev,
            out: Aligned::new(n),
            state,
            n,
        }
    }
}

impl Kernel for FirK {
    fn run(&mut self, which: Impl, k: usize) {
        let input = &self.buf[FIR_TAPS - 1..];
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ = black_box(Sdrk.fir(&mut self.state, black_box(input), &mut self.out));
                }
                Impl::Fallback => {
                    let _ =
                        black_box(Fallback.fir(&mut self.state, black_box(input), &mut self.out));
                }
                Impl::Library => {
                    for i in 0..self.n {
                        let mut r = lv_32fc_t::default();
                        // SAFETY: the window `i .. i + 64` lies inside `buf`.
                        unsafe {
                            volk_sys::volk_32fc_32f_dot_prod_32fc(
                                &mut r,
                                as_lv(&self.buf).add(i),
                                self.rev.as_ptr(),
                                FIR_TAPS as u32,
                            );
                        }
                        self.out[i] = C32::new(r.re, r.im);
                    }
                }
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
        // The library variant sees real history samples instead of zeros: compare only past
        // the filter length.
        let skip = FIR_TAPS - 1;
        snr_db(&c32s(&want[skip..]), &c32s(&self.out[skip..]))
    }

    fn has_library(&self) -> bool {
        volk_sys::AVAILABLE
    }

    fn library_name(&self) -> &'static str {
        "volk_32fc_32f_dot_prod_32fc (per output)"
    }
}

struct DecimateK {
    buf: Aligned<C32>,
    taps: Vec<f32>,
    rev: Aligned<f32>,
    out: Aligned<C32>,
    state: FirDecimState<C32, Vec<f32>, Vec<C32>>,
    n: usize,
}

impl DecimateK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let taps = taps(rng, DECIM_TAPS);
        let mut rev = Aligned::<f32>::new(DECIM_TAPS);
        for (r, t) in rev.iter_mut().zip(taps.iter().rev()) {
            *r = *t;
        }
        let buf = noise_c32(rng, n + DECIM_TAPS - 1, 0.9);
        let state = FirDecimState::new(taps.clone(), vec![C32::default(); DECIM_TAPS - 1], DECIM)
            .expect("taps");
        DecimateK {
            buf,
            taps,
            rev,
            out: Aligned::new(n / DECIM + 2),
            state,
            n,
        }
    }
}

impl Kernel for DecimateK {
    fn run(&mut self, which: Impl, k: usize) {
        let input = &self.buf[DECIM_TAPS - 1..];
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ = black_box(Sdrk.fir_decimate(
                        &mut self.state,
                        black_box(input),
                        &mut self.out,
                    ));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.fir_decimate(
                        &mut self.state,
                        black_box(input),
                        &mut self.out,
                    ));
                }
                Impl::Library => {
                    for j in 0..self.n.div_ceil(DECIM) {
                        let mut r = lv_32fc_t::default();
                        // SAFETY: the window starts at `j * DECIM` and has 127 samples.
                        unsafe {
                            volk_sys::volk_32fc_32f_dot_prod_32fc(
                                &mut r,
                                as_lv(&self.buf).add(j * DECIM),
                                self.rev.as_ptr(),
                                DECIM_TAPS as u32,
                            );
                        }
                        self.out[j] = C32::new(r.re, r.im);
                    }
                }
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.state = FirDecimState::new(
            self.taps.clone(),
            vec![C32::default(); DECIM_TAPS - 1],
            DECIM,
        )
        .expect("taps");
        self.run(which, 1);
        let mut s = FirDecimState::<C32, _, _>::new(
            self.taps.clone(),
            vec![C32::default(); DECIM_TAPS - 1],
            DECIM,
        )
        .expect("taps");
        let mut want = vec![C32::default(); self.n / DECIM + 2];
        let p = Reference
            .fir_decimate(&mut s, &self.buf[DECIM_TAPS - 1..], &mut want)
            .expect("decimate");
        let skip = DECIM_TAPS.div_ceil(DECIM);
        let len = p.produced.min(self.n / DECIM);
        snr_db(&c32s(&want[skip..len]), &c32s(&self.out[skip..len]))
    }

    fn has_library(&self) -> bool {
        volk_sys::AVAILABLE
    }

    fn library_name(&self) -> &'static str {
        "volk_32fc_32f_dot_prod_32fc (per output)"
    }
}

struct MagSqK {
    input: Aligned<C32>,
    out: Aligned<f32>,
}

impl MagSqK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        MagSqK {
            input: noise_c32(rng, n, 0.9),
            out: Aligned::new(n),
        }
    }
}

impl Kernel for MagSqK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ = black_box(Sdrk.mag_sq(black_box(&self.input), &mut self.out));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.mag_sq(black_box(&self.input), &mut self.out));
                }
                // SAFETY: input and output hold `n` elements.
                Impl::Library => unsafe {
                    volk_sys::volk_32fc_magnitude_squared_32f(
                        self.out.as_mut_ptr(),
                        as_lv(&self.input),
                        self.input.len() as u32,
                    );
                },
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

    fn has_library(&self) -> bool {
        volk_sys::AVAILABLE
    }

    fn library_name(&self) -> &'static str {
        "volk_32fc_magnitude_squared_32f"
    }
}

struct QuadK {
    input: Aligned<C32>,
    tmp: Aligned<C32>,
    out: Aligned<f32>,
}

impl QuadK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        // A frequency-modulated carrier, so that the arguments cover the whole range.
        let mut phase = 0.0f64;
        let mut input = Aligned::<C32>::new(n + 1);
        for s in input.iter_mut() {
            phase += 0.6 * f64::from(rng.unit());
            *s = C32::new((0.8 * phase.cos()) as f32, (0.8 * phase.sin()) as f32);
        }
        QuadK {
            input,
            tmp: Aligned::new(n),
            out: Aligned::new(n),
        }
    }
}

impl Kernel for QuadK {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.out.len();
        let mut state = QuadDemodState::<C32>::new();
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ = black_box(Sdrk.quad_demod(
                        &mut state,
                        black_box(&self.input[1..]),
                        &mut self.out,
                        1.0,
                    ));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.quad_demod(
                        &mut state,
                        black_box(&self.input[1..]),
                        &mut self.out,
                        1.0,
                    ));
                }
                // SAFETY: all buffers hold `n` samples (the input one more).
                Impl::Library => unsafe {
                    volk_sys::volk_32fc_x2_multiply_conjugate_32fc(
                        as_lv_mut(&mut self.tmp),
                        as_lv(&self.input).add(1),
                        as_lv(&self.input),
                        n as u32,
                    );
                    volk_sys::volk_32fc_s32f_atan2_32f(
                        self.out.as_mut_ptr(),
                        as_lv(&self.tmp),
                        1.0,
                        n as u32,
                    );
                },
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

    fn has_library(&self) -> bool {
        volk_sys::AVAILABLE
    }

    fn library_name(&self) -> &'static str {
        "volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f"
    }
}

struct FmModK {
    input: Aligned<f32>,
    out: Aligned<C32>,
}

impl FmModK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let mut input = Aligned::<f32>::new(n);
        for s in input.iter_mut() {
            *s = rng.unit();
        }
        FmModK {
            input,
            out: Aligned::new(n),
        }
    }
}

impl Kernel for FmModK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state =
            FmModState::new(75_000.0, Rate::hz(240_000).expect("rate")).expect("deviation");
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Library => {
                    let _ =
                        black_box(Sdrk.fm_mod(&mut state, black_box(&self.input), &mut self.out));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.fm_mod(
                        &mut state,
                        black_box(&self.input),
                        &mut self.out,
                    ));
                }
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut state =
            FmModState::new(75_000.0, Rate::hz(240_000).expect("rate")).expect("deviation");
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.fm_mod(&mut state, &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn has_library(&self) -> bool {
        false
    }

    fn library_name(&self) -> &'static str {
        "none (own fallback)"
    }
}

struct FftK {
    data: Aligned<C32>,
    scratch: Vec<Complex32>,
    n: usize,
    plan: FftPlan<C32, Vec<C32>>,
    rf: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl FftK {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let data = noise_c32(rng, n, 0.5);
        let rf = FftPlanner::<f32>::new().plan_fft_forward(n);
        let scratch = vec![Complex32::new(0.0, 0.0); rf.get_inplace_scratch_len()];
        let plan = FftPlan::new(n, vec![C32::default(); n / 2]).expect("plan");
        FftK {
            data,
            scratch,
            n,
            plan,
            rf,
        }
    }
}

impl Kernel for FftK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    let _ = black_box(Sdrk.fft(
                        &self.plan,
                        &mut self.data,
                        Direction::Forward,
                        Scaling::Default,
                    ));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.fft(
                        &self.plan,
                        &mut self.data,
                        Direction::Forward,
                        Scaling::Default,
                    ));
                }
                Impl::Library => {
                    // SAFETY: `C32` and rustfft's `Complex32` are the same `repr(C)` pair of f32.
                    let buf: &mut [Complex32] = unsafe {
                        std::slice::from_raw_parts_mut(self.data.as_mut_ptr().cast(), self.n)
                    };
                    self.rf.process_with_scratch(buf, &mut self.scratch);
                }
            }
            black_box(&self.data);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let saved: Vec<C32> = self.data.to_vec();
        self.run(which, 1);
        let got = c32s(&self.data);
        let mut want = vec![C32::default(); self.n];
        let _ = Reference.dft(&saved, &mut want, Direction::Forward, Scaling::Default);
        self.data.copy_from_slice(&saved);
        snr_db(&c32s(&want), &got)
    }

    fn has_library(&self) -> bool {
        true
    }

    fn library_name(&self) -> &'static str {
        "rustfft (FftPlanner, forward, in place)"
    }
}

struct FftQ15K {
    data: Vec<CQ15>,
    n: usize,
    plan: FftPlan<CQ15, Vec<CQ15>>,
}

impl FftQ15K {
    fn new(rng: &mut Pcg32, n: usize) -> Self {
        let data = (0..n)
            .map(|_| CQ15 {
                re: Q15((rng.unit() * 8000.0) as i16),
                im: Q15((rng.unit() * 8000.0) as i16),
            })
            .collect();
        FftQ15K {
            data,
            n,
            plan: FftPlan::new(n, vec![CQ15::default(); n / 2]).expect("plan"),
        }
    }
}

impl Kernel for FftQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Library => {
                    let _ = black_box(Sdrk.fft(
                        &self.plan,
                        &mut self.data,
                        Direction::Forward,
                        Scaling::BlockFloat,
                    ));
                }
                Impl::Fallback => {
                    let _ = black_box(Fallback.fft(
                        &self.plan,
                        &mut self.data,
                        Direction::Forward,
                        Scaling::BlockFloat,
                    ));
                }
            }
            black_box(&self.data);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let saved = self.data.clone();
        let got_scale = match which {
            Impl::Ours | Impl::Library => Sdrk.fft(
                &self.plan,
                &mut self.data,
                Direction::Forward,
                Scaling::BlockFloat,
            ),
            Impl::Fallback => Fallback.fft(
                &self.plan,
                &mut self.data,
                Direction::Forward,
                Scaling::BlockFloat,
            ),
        }
        .expect("fft");
        let got: Vec<f64> = cq15s(&self.data)
            .iter()
            .map(|v| v * 2f64.powi(got_scale.exponent))
            .collect();
        let mut want = vec![CQ15::default(); self.n];
        let want_scale = Reference
            .dft(&saved, &mut want, Direction::Forward, Scaling::BlockFloat)
            .expect("dft");
        let want: Vec<f64> = cq15s(&want)
            .iter()
            .map(|v| v * 2f64.powi(want_scale.exponent))
            .collect();
        self.data.copy_from_slice(&saved);
        snr_db(&want, &got)
    }

    fn has_library(&self) -> bool {
        false
    }

    fn library_name(&self) -> &'static str {
        "none (own fallback)"
    }
}
