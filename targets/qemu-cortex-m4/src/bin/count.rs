//! Instruction-count workload of the fallback kernels on the Cortex-M4F (`cargo xtask count
//! --platform cortex-m4`).
//!
//! QEMU passes `<kernel> <k>` as the semihosting command line; the program runs the kernel
//! once as a warm-up and then `k` times on the same buffers. The host counts the instructions
//! of the whole run with the QEMU `insn` plugin, and the differential method of the count
//! command (`k = 16` minus `k = 0`) removes boot, warm-up and printing.
//!
//! The biquad kernels also run CMSIS-DSP 1.18.0 (`<kernel>@cmsis`, with the feature
//! `cmsis-dsp`, which links the library that `benches/refs/build.sh cortex-m4` builds), and
//! take `snr` in place of `k`: one call from a fresh state, then the energies of the f64 model
//! and of the error are printed (`sig=` and `err=`), from which the host computes the SNR.
#![no_std]
#![no_main]

use core::fmt::Write;
use core::hint::black_box;
use core::panic::PanicInfo;

use cortex_m_rt::entry;
use cortex_m_semihosting::debug::{self, EXIT_FAILURE, EXIT_SUCCESS};
use cortex_m_semihosting::{hio, nr, syscall1};
use sdrk::kernels::agc::{Agc, AgcState};
use sdrk::kernels::biquad::{Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState};
use sdrk::kernels::convert::{Convert, ConvertOut};
use sdrk::kernels::decim::{FirDecimState, FirDecimate};
use sdrk::kernels::demod::{DcBlockReal, DcBlockRealState, QuadDemod, QuadDemodState};
use sdrk::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk::kernels::filter::{Fir, FirState};
use sdrk::kernels::mix::{Nco, Rotate};
use sdrk::kernels::modulate::{FmMod, FmModState};
use sdrk::kernels::spectrum::MagSq;
use sdrk::{C32, CQ15, Fallback, Q15, Rate};
use sdrk_device::formats;

/// Samples per call of the streaming kernels.
const N: usize = 1024;
const FIR_TAPS: usize = 64;
const DECIM_TAPS: usize = 127;
const DECIM: usize = 10;

/// The kernels of the report: the names are the keys of `benches/results/cortex-m4.json`.
const KERNELS: [&str; 19] = [
    "convert_cs8",
    "convert_cs16",
    "convert_out_cs16",
    "rotate",
    "fir",
    "fir_decimate",
    "mag_sq",
    "quad_demod",
    "fm_mod",
    "fft_1024",
    "fft_cq15_1024",
    "agc",
    "agc_cq15",
    "dc_block_real",
    "dc_block_real_q15",
    "biquad",
    "biquad_cq15",
    "biquad_real",
    "biquad_real_q15",
];

/// A small linear congruential generator; the values only have to be varied.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 8_388_608.0 - 1.0
    }
}

fn noise(rng: &mut Lcg, out: &mut [C32], amp: f32) {
    for s in out {
        *s = C32::new(rng.unit() * amp, rng.unit() * amp);
    }
}

/// The amplitude of sample `i` of the `agc` input: it steps from 0.01 to 0.3 to 0.003 in
/// thirds, so the gain moves in both directions within a call (as in `benches/kernels`).
fn agc_amplitude(i: usize) -> f32 {
    [0.01, 0.3, 0.003][(3 * i / N).min(2)]
}

/// The `agc` parameters of `benches/kernels`: reference 0.5, attack 0.005, decay 0.05, gain
/// limits -20 and 60 dB, initial gain 0 dB, 48 kHz.
fn agc_state<T: sdrk::ComplexSample>() -> Option<AgcState<T>> {
    AgcState::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, Rate::hz(48_000).ok()?).ok()
}

fn exit(ok: bool) -> ! {
    debug::exit(if ok { EXIT_SUCCESS } else { EXIT_FAILURE });
    loop {
        cortex_m::asm::wfi();
    }
}

/// What a run does after the warm-up call.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// `k` more calls on the same buffers.
    Count(usize),
    /// Nothing more: the output of the warm-up call (from a fresh state) is compared with the
    /// f64 model (biquad kernels only).
    Snr,
}

impl Mode {
    /// The calls after the warm-up.
    fn calls(self) -> usize {
        match self {
            Mode::Count(k) => k,
            Mode::Snr => 0,
        }
    }
}

/// The last two words of the semihosting command line: the kernel name and `k` (or `snr`).
fn arguments(buf: &mut [u8; 128]) -> Option<(&str, Mode)> {
    let mut block = [buf.as_mut_ptr() as usize, buf.len()];
    // SAFETY: SYS_GET_CMDLINE takes a pointer to a buffer and its size and writes the length
    // back into the second word. The block is passed by a mutable pointer: the host writes it,
    // so passing a shared reference (`syscall(.., &block)`) would let the compiler keep the old
    // length (seen at `opt-level = "s"`).
    let status = unsafe { syscall1(nr::GET_CMDLINE, block.as_mut_ptr() as usize) };
    if status != 0 {
        return None;
    }
    let len = block[1].min(buf.len());
    let text = core::str::from_utf8(buf.get(..len)?).ok()?;
    let mut words = text.split_whitespace().rev();
    let k = match words.next()? {
        "snr" => Mode::Snr,
        k => Mode::Count(k.parse().ok()?),
    };
    let name = words.next()?;
    Some((name, k))
}

#[inline(never)]
fn run_convert_cs8(rng: &mut Lcg, k: usize) -> bool {
    let mut bytes = [0u8; 2 * N];
    for b in &mut bytes {
        *b = rng.next() as u8;
    }
    let mut out = [C32::default(); N];
    for _ in 0..=k {
        let _ = black_box(Fallback.convert(&formats::CS8, black_box(&bytes), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_convert_cs16(rng: &mut Lcg, k: usize) -> bool {
    let mut bytes = [0u8; 4 * N];
    for b in &mut bytes {
        *b = rng.next() as u8;
    }
    let mut out = [C32::default(); N];
    for _ in 0..=k {
        let _ = black_box(Fallback.convert(&formats::CS16, black_box(&bytes), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_convert_out_cs16(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [0u8; 4 * N];
    for _ in 0..=k {
        let _ = black_box(Fallback.convert_out(&formats::CS16, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_rotate(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [C32::default(); N];
    let mut nco = Nco::from_increment(0x0123_4567);
    for _ in 0..=k {
        let _ = black_box(Fallback.rotate(&mut nco, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_fir(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [C32::default(); N];
    let Ok(mut state) = FirState::new(
        [1.0 / FIR_TAPS as f32; FIR_TAPS],
        [C32::default(); FIR_TAPS - 1],
    ) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fir(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_fir_decimate(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [C32::default(); N / DECIM + 2];
    let Ok(mut state) = FirDecimState::new(
        [1.0 / DECIM_TAPS as f32; DECIM_TAPS],
        [C32::default(); DECIM_TAPS - 1],
        DECIM,
    ) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fir_decimate(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_mag_sq(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [0f32; N];
    for _ in 0..=k {
        let _ = black_box(Fallback.mag_sq(black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_quad_demod(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.8);
    let mut out = [0f32; N];
    let mut state = QuadDemodState::<C32>::new();
    for _ in 0..=k {
        let _ = black_box(Fallback.quad_demod(&mut state, black_box(&input), &mut out, 1.0));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_fm_mod(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [0f32; N];
    for s in &mut input {
        *s = rng.unit();
    }
    let mut out = [C32::default(); N];
    let Ok(rate) = Rate::hz(240_000) else {
        return false;
    };
    let Ok(mut state) = FmModState::new(75_000.0, rate) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fm_mod(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_fft_1024(rng: &mut Lcg, k: usize) -> bool {
    let mut data = [C32::default(); 1024];
    noise(rng, &mut data, 0.5);
    let Ok(plan) = FftPlan::new(1024, [C32::default(); 512]) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::Default));
        black_box(&data);
    }
    true
}

#[inline(never)]
fn run_fft_cq15_1024(rng: &mut Lcg, k: usize) -> bool {
    let mut data = [CQ15::default(); 1024];
    for s in &mut data {
        s.re = Q15((rng.unit() * 8000.0) as i16);
        s.im = Q15((rng.unit() * 8000.0) as i16);
    }
    let Ok(plan) = FftPlan::new(1024, [CQ15::default(); 512]) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::BlockFloat));
        black_box(&data);
    }
    true
}

#[inline(never)]
fn run_agc(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    for (i, s) in input.iter_mut().enumerate() {
        let amp = agc_amplitude(i);
        *s = C32::new(rng.unit() * amp, rng.unit() * amp);
    }
    let mut out = [C32::default(); N];
    let Some(mut state) = agc_state::<C32>() else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.agc(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_agc_cq15(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [CQ15::default(); N];
    for (i, s) in input.iter_mut().enumerate() {
        let amp = agc_amplitude(i) * 32768.0;
        s.re = Q15((rng.unit() * amp) as i16);
        s.im = Q15((rng.unit() * amp) as i16);
    }
    let mut out = [CQ15::default(); N];
    let Some(mut state) = agc_state::<CQ15>() else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.agc(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_dc_block_real(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [0f32; N];
    for s in &mut input {
        *s = rng.unit() * 0.2 + 0.3;
    }
    let mut out = [0f32; N];
    let Ok(mut state) = DcBlockRealState::new(0.999_f32) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.dc_block_real(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_dc_block_real_q15(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [Q15::default(); N];
    for s in &mut input {
        *s = Q15(((rng.unit() * 0.2 + 0.3) * 32768.0) as i16);
    }
    let mut out = [Q15::default(); N];
    let Ok(mut state) = DcBlockRealState::new(Q15(32_735)) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.dc_block_real(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

// ---------------------------------------------------------------------------------------------
// biquad (M22): the input, the filter and the samples per call of the host counts
// (`benches/kernels`); our fallback and CMSIS-DSP 1.18.0 built for the M4F.

/// Samples per call of the biquad kernels: `N` of the host counts.
const BIQUAD_N: usize = 4096;

/// PCG32 of `benches/kernels` (the generator of the testkit), so that the biquad kernels filter
/// the input of the host counts.
struct Pcg32(u64);

impl Pcg32 {
    /// The generator of the host kernel at position `index` of the kernel list of
    /// `benches/kernels` (seed `1000 + index`).
    fn of_host_kernel(index: u64) -> Self {
        let mut r = Pcg32(0);
        r.next_u32();
        r.0 = r.0.wrapping_add(1000 + index);
        r.next_u32();
        r
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (f64::from(self.next_u32()) / 2_147_483_648.0 - 1.0) as f32
    }
}

/// The positions of `biquad`, `biquad_cq15`, `biquad_real` and `biquad_real_q15` in the kernel
/// list of `benches/kernels` (their seeds there; the `_fallback` entries share them).
const HOST_BIQUAD: u64 = 32;
const HOST_BIQUAD_CQ15: u64 = 33;
const HOST_BIQUAD_REAL: u64 = 34;
const HOST_BIQUAD_REAL_Q15: u64 = 35;

/// The measured cascade: the fourth-order Butterworth low-pass at 0.1 of the sample rate (two
/// sections, `[b0, b1, b2, a1, a2]`), as `benches/kernels` rounds it to f32. The f64 design
/// there needs `tan` and `sin`, which `core` lacks, so the rounded values are written out; the
/// outputs of our fallback, bit-exact across platforms, give the SNR of the host counts only
/// with these coefficients and that input.
const BIQUAD_F32: [[f32; 5]; 2] = [
    [
        f32::from_bits(0x3d9f_a793),
        f32::from_bits(0x3e1f_a793),
        f32::from_bits(0x3d9f_a793),
        f32::from_bits(0xbfa9_13b1),
        f32::from_bits(0x3f21_fb2b),
    ],
    [
        f32::from_bits(0x3d7d_7b55),
        f32::from_bits(0x3dfd_7b55),
        f32::from_bits(0x3d7d_7b55),
        f32::from_bits(0xbf86_3883),
        f32::from_bits(0x3e97_9fb5),
    ],
];

/// The same cascade in Q2.14, rounded as `benches/kernels` rounds it.
const BIQUAD_Q14: [[i16; 5]; 2] = [
    [1277, 2554, 1277, -21642, 10367],
    [1014, 2028, 1014, -17180, 4852],
];

/// The energies of the f64 model and of the error of one output (the SNR is
/// `10 log10(sig / err)`, computed by the host).
#[derive(Clone, Copy)]
struct Energies {
    sig: f64,
    err: f64,
}

/// The f64 model of `benches/kernels`: direct form I in f64 with the values of the
/// coefficients, no rounding between the sections; `channels` interleaved channels filtered
/// apart. `x` and `got` are in the same unit (raw integers for Q15). The sums run in the
/// interleaved order of the host, so that the energies are those of the host.
fn model_energies(
    coefs: &[[f64; 5]; 2],
    channels: usize,
    x: impl Iterator<Item = f64>,
    got: impl Iterator<Item = f64>,
) -> Energies {
    let mut hist = [[[0.0f64; 4]; 2]; 2];
    let mut e = Energies { sig: 0.0, err: 0.0 };
    for (i, (mut v, g)) in x.zip(got).enumerate() {
        let h = &mut hist[i % channels];
        for (c, s) in coefs.iter().zip(h.iter_mut()) {
            let [b0, b1, b2, a1, a2] = *c;
            let [x1, x2, y1, y2] = *s;
            let y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            *s = [v, x1, y, y1];
            v = y;
        }
        e.sig += v * v;
        e.err += (v - g) * (v - g);
    }
    e
}

fn coefs_f32() -> [[f64; 5]; 2] {
    BIQUAD_F32.map(|s| s.map(f64::from))
}

fn coefs_q14() -> [[f64; 5]; 2] {
    BIQUAD_Q14.map(|s| s.map(|c| f64::from(c) / 16384.0))
}

/// Which implementation a biquad run counts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Impl {
    /// Our fallback (`sdrk::Fallback`).
    Ours,
    /// CMSIS-DSP 1.18.0 (`<kernel>@cmsis`).
    Cmsis,
}

#[cfg(feature = "cmsis-dsp")]
mod cmsis {
    //! The handles of `benches/refs/cmsis-biquad.c` around CMSIS-DSP's cascades, built for the
    //! M4F by `benches/refs/build.sh cortex-m4` (`libcmsisbq-cortex-m4.a`).
    use core::ffi::c_void;

    unsafe extern "C" {
        fn sdrk_cmsis_df2t_f32_new(coefs: *const f32, stages: u32) -> *mut c_void;
        fn sdrk_cmsis_df2t_f32_run(h: *mut c_void, input: *const f32, output: *mut f32, n: u32);
        fn sdrk_cmsis_stereo_df2t_f32_new(coefs: *const f32, stages: u32) -> *mut c_void;
        fn sdrk_cmsis_stereo_df2t_f32_run(
            h: *mut c_void,
            input: *const f32,
            output: *mut f32,
            n: u32,
        );
        fn sdrk_cmsis_df1_q15_new(coefs: *const i16, stages: u32) -> *mut c_void;
        fn sdrk_cmsis_df1_q15_run(h: *mut c_void, input: *const i16, output: *mut i16, n: u32);
    }

    /// `arm_biquad_cascade_df2T_f32`, one channel.
    pub struct Df2tF32(*mut c_void);
    /// `arm_biquad_cascade_stereo_df2T_f32`, two interleaved channels.
    pub struct StereoDf2tF32(*mut c_void);
    /// `arm_biquad_cascade_df1_q15` with `postShift = 1` (Q2.14 coefficients).
    pub struct Df1Q15(*mut c_void);

    impl Df2tF32 {
        pub fn new(coefs: &[[f32; 5]; 2]) -> Option<Self> {
            // SAFETY: the constructor reads `5 * stages` coefficients and copies them.
            let h = unsafe { sdrk_cmsis_df2t_f32_new(coefs.as_ptr().cast(), 2) };
            (!h.is_null()).then_some(Df2tF32(h))
        }

        pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
            let n = input.len().min(output.len()) as u32;
            // SAFETY: the handle is live; both buffers hold `n` values.
            unsafe { sdrk_cmsis_df2t_f32_run(self.0, input.as_ptr(), output.as_mut_ptr(), n) }
        }
    }

    impl StereoDf2tF32 {
        pub fn new(coefs: &[[f32; 5]; 2]) -> Option<Self> {
            // SAFETY: as in `Df2tF32::new`.
            let h = unsafe { sdrk_cmsis_stereo_df2t_f32_new(coefs.as_ptr().cast(), 2) };
            (!h.is_null()).then_some(StereoDf2tF32(h))
        }

        /// `input` and `output` hold interleaved pairs.
        pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
            let n = (input.len().min(output.len()) / 2) as u32;
            // SAFETY: the handle is live; both buffers hold `2 n` values.
            unsafe {
                sdrk_cmsis_stereo_df2t_f32_run(self.0, input.as_ptr(), output.as_mut_ptr(), n)
            }
        }
    }

    impl Df1Q15 {
        pub fn new(coefs: &[[i16; 5]; 2]) -> Option<Self> {
            // SAFETY: as in `Df2tF32::new`.
            let h = unsafe { sdrk_cmsis_df1_q15_new(coefs.as_ptr().cast(), 2) };
            (!h.is_null()).then_some(Df1Q15(h))
        }

        pub fn run(&mut self, input: &[i16], output: &mut [i16]) {
            let n = input.len().min(output.len()) as u32;
            // SAFETY: the handle is live; both buffers hold `n` values.
            unsafe { sdrk_cmsis_df1_q15_run(self.0, input.as_ptr(), output.as_mut_ptr(), n) }
        }
    }
}

/// `C32` as two interleaved `f32` channels (CMSIS-DSP's stereo layout).
fn c32_flat(x: &[C32]) -> &[f32] {
    // SAFETY: `C32` is `repr(C)` with the fields `re`, `im` (two `f32`, no padding).
    unsafe { core::slice::from_raw_parts(x.as_ptr().cast(), 2 * x.len()) }
}

/// Mutable form of [`c32_flat`].
#[cfg(feature = "cmsis-dsp")]
fn c32_flat_mut(x: &mut [C32]) -> &mut [f32] {
    // SAFETY: as in `c32_flat`.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), 2 * x.len()) }
}

/// `x` as raw `i16` (CMSIS-DSP's `q15_t`).
#[cfg(feature = "cmsis-dsp")]
fn q15_raw(x: &[Q15]) -> &[i16] {
    // SAFETY: `Q15` is `repr(transparent)` over `i16`.
    unsafe { core::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

/// Mutable form of [`q15_raw`].
#[cfg(feature = "cmsis-dsp")]
fn q15_raw_mut(x: &mut [Q15]) -> &mut [i16] {
    // SAFETY: as in `q15_raw`.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

/// The two-section state of the measurements.
type State<T> = BiquadState<T, [[<T as BiquadSample>::Coef; 5]; 2], [T; 8]>;
/// The same for the real kernel.
type RealState<T> = BiquadRealState<T, [[<T as BiquadSample>::Coef; 5]; 2], [T; 8]>;

/// One call of our `biquad` in a function of its own that is never inlined, as on the host
/// (`benches/kernels`); with the coefficients behind `black_box` when the state is made, the
/// compiler cannot fold them into the loop (CMSIS-DSP, a library, sees them at run time too).
#[inline(never)]
fn biquad_call<T: sdrk::ComplexSample + BiquadSample>(
    state: &mut State<T>,
    input: &[T],
    out: &mut [T],
) where
    Fallback: Biquad<T>,
{
    let _ = black_box(Fallback.biquad(state, black_box(input), out));
}

/// The same for `biquad_real`.
#[inline(never)]
fn biquad_real_call<T: sdrk::RealSample + BiquadSample>(
    state: &mut RealState<T>,
    input: &[T],
    out: &mut [T],
) where
    Fallback: BiquadReal<T>,
{
    let _ = black_box(Fallback.biquad_real(state, black_box(input), out));
}

/// `biquad` `C32` on noise of the amplitude 0.3; CMSIS-DSP's stereo cascade with re and im as
/// the two channels.
#[inline(never)]
fn run_biquad(which: Impl, mode: Mode) -> Option<Energies> {
    let mut rng = Pcg32::of_host_kernel(HOST_BIQUAD);
    let mut input = [C32::default(); BIQUAD_N];
    for s in &mut input {
        *s = C32::new(rng.unit() * 0.3, rng.unit() * 0.3);
    }
    let mut out = [C32::default(); BIQUAD_N];
    match which {
        Impl::Ours => {
            let mut state = BiquadState::new(black_box(BIQUAD_F32), [C32::default(); 8]).ok()?;
            for _ in 0..=mode.calls() {
                biquad_call(&mut state, &input, &mut out);
                black_box(&out);
            }
        }
        #[cfg(feature = "cmsis-dsp")]
        Impl::Cmsis => {
            let mut h = cmsis::StereoDf2tF32::new(&BIQUAD_F32)?;
            for _ in 0..=mode.calls() {
                h.run(black_box(c32_flat(&input)), c32_flat_mut(&mut out));
                black_box(&out);
            }
        }
        #[cfg(not(feature = "cmsis-dsp"))]
        Impl::Cmsis => return None,
    }
    let x = c32_flat(&input).iter().map(|&v| f64::from(v));
    let got = c32_flat(&out).iter().map(|&v| f64::from(v));
    Some(model_energies(&coefs_f32(), 2, x, got))
}

/// `biquad` `CQ15` on the same kind of noise in Q15; CMSIS-DSP's Q15 cascade twice, on planar
/// re and im (the deinterleaving is done before the calls and not counted, as on the host).
#[inline(never)]
fn run_biquad_cq15(which: Impl, mode: Mode) -> Option<Energies> {
    let mut rng = Pcg32::of_host_kernel(HOST_BIQUAD_CQ15);
    let mut input = [CQ15::default(); BIQUAD_N];
    for s in &mut input {
        s.re = Q15((rng.unit() * 0.3 * 32768.0) as i16);
        s.im = Q15((rng.unit() * 0.3 * 32768.0) as i16);
    }
    let mut out = [CQ15::default(); BIQUAD_N];
    match which {
        Impl::Ours => {
            let mut state = BiquadState::new(black_box(BIQUAD_Q14), [CQ15::default(); 8]).ok()?;
            for _ in 0..=mode.calls() {
                biquad_call(&mut state, &input, &mut out);
                black_box(&out);
            }
        }
        #[cfg(feature = "cmsis-dsp")]
        Impl::Cmsis => {
            let mut re = [0i16; BIQUAD_N];
            let mut im = [0i16; BIQUAD_N];
            for ((r, i), s) in re.iter_mut().zip(im.iter_mut()).zip(&input) {
                (*r, *i) = (s.re.0, s.im.0);
            }
            let mut out_re = [0i16; BIQUAD_N];
            let mut out_im = [0i16; BIQUAD_N];
            let mut h_re = cmsis::Df1Q15::new(&BIQUAD_Q14)?;
            let mut h_im = cmsis::Df1Q15::new(&BIQUAD_Q14)?;
            for _ in 0..=mode.calls() {
                h_re.run(black_box(&re), &mut out_re);
                h_im.run(black_box(&im), &mut out_im);
                black_box(&out_re);
                black_box(&out_im);
            }
            for ((o, r), i) in out.iter_mut().zip(&out_re).zip(&out_im) {
                *o = CQ15 {
                    re: Q15(*r),
                    im: Q15(*i),
                };
            }
        }
        #[cfg(not(feature = "cmsis-dsp"))]
        Impl::Cmsis => return None,
    }
    let x = input
        .iter()
        .flat_map(|s| [f64::from(s.re.0), f64::from(s.im.0)]);
    let got = out
        .iter()
        .flat_map(|s| [f64::from(s.re.0), f64::from(s.im.0)]);
    Some(model_energies(&coefs_q14(), 2, x, got))
}

/// `biquad_real` `f32` on noise of the amplitude 0.3; CMSIS-DSP's `df2T_f32`.
#[inline(never)]
fn run_biquad_real(which: Impl, mode: Mode) -> Option<Energies> {
    let mut rng = Pcg32::of_host_kernel(HOST_BIQUAD_REAL);
    let mut input = [0f32; BIQUAD_N];
    for s in &mut input {
        *s = rng.unit() * 0.3 + 0.0;
    }
    let mut out = [0f32; BIQUAD_N];
    match which {
        Impl::Ours => {
            let mut state = BiquadRealState::new(black_box(BIQUAD_F32), [0f32; 8]).ok()?;
            for _ in 0..=mode.calls() {
                biquad_real_call(&mut state, &input, &mut out);
                black_box(&out);
            }
        }
        #[cfg(feature = "cmsis-dsp")]
        Impl::Cmsis => {
            let mut h = cmsis::Df2tF32::new(&BIQUAD_F32)?;
            for _ in 0..=mode.calls() {
                h.run(black_box(&input), &mut out);
                black_box(&out);
            }
        }
        #[cfg(not(feature = "cmsis-dsp"))]
        Impl::Cmsis => return None,
    }
    let x = input.iter().map(|&v| f64::from(v));
    let got = out.iter().map(|&v| f64::from(v));
    Some(model_energies(&coefs_f32(), 1, x, got))
}

/// `biquad_real` `Q15` on the same kind of noise in Q15; CMSIS-DSP's `df1_q15`.
#[inline(never)]
fn run_biquad_real_q15(which: Impl, mode: Mode) -> Option<Energies> {
    let mut rng = Pcg32::of_host_kernel(HOST_BIQUAD_REAL_Q15);
    let mut input = [Q15::default(); BIQUAD_N];
    for s in &mut input {
        *s = Q15(((rng.unit() * 0.3 + 0.0) * 32768.0) as i16);
    }
    let mut out = [Q15::default(); BIQUAD_N];
    match which {
        Impl::Ours => {
            let mut state =
                BiquadRealState::new(black_box(BIQUAD_Q14), [Q15::default(); 8]).ok()?;
            for _ in 0..=mode.calls() {
                biquad_real_call(&mut state, &input, &mut out);
                black_box(&out);
            }
        }
        #[cfg(feature = "cmsis-dsp")]
        Impl::Cmsis => {
            let mut h = cmsis::Df1Q15::new(&BIQUAD_Q14)?;
            for _ in 0..=mode.calls() {
                h.run(black_box(q15_raw(&input)), q15_raw_mut(&mut out));
                black_box(&out);
            }
        }
        #[cfg(not(feature = "cmsis-dsp"))]
        Impl::Cmsis => return None,
    }
    let x = input.iter().map(|s| f64::from(s.0));
    let got = out.iter().map(|s| f64::from(s.0));
    Some(model_energies(&coefs_q14(), 1, x, got))
}

/// Runs a biquad kernel (`<kernel>` or `<kernel>@cmsis`); `None` for any other name, and
/// `Some(None)` when the run failed (or CMSIS-DSP is not linked).
fn run_biquad_kernel(name: &str, mode: Mode) -> Option<Option<Energies>> {
    let (kernel, which) = match name.strip_suffix("@cmsis") {
        Some(kernel) => (kernel, Impl::Cmsis),
        None => (name, Impl::Ours),
    };
    Some(match kernel {
        "biquad" => run_biquad(which, mode),
        "biquad_cq15" => run_biquad_cq15(which, mode),
        "biquad_real" => run_biquad_real(which, mode),
        "biquad_real_q15" => run_biquad_real_q15(which, mode),
        _ => return None,
    })
}

/// Runs kernel `name`: one warm-up call and `k` counted calls. Each kernel has a function of its
/// own that is never inlined, so that the code generated for one kernel does not depend on the
/// others in the binary (in one function, adding a kernel changed the counts of the FIR and FFT
/// by up to 10 %).
fn run(name: &str, k: usize) -> bool {
    let mut rng = Lcg(0x1234_5678);
    let rng = &mut rng;
    match name {
        "convert_cs8" => run_convert_cs8(rng, k),
        "convert_cs16" => run_convert_cs16(rng, k),
        "convert_out_cs16" => run_convert_out_cs16(rng, k),
        "rotate" => run_rotate(rng, k),
        "fir" => run_fir(rng, k),
        "fir_decimate" => run_fir_decimate(rng, k),
        "mag_sq" => run_mag_sq(rng, k),
        "quad_demod" => run_quad_demod(rng, k),
        "fm_mod" => run_fm_mod(rng, k),
        "fft_1024" => run_fft_1024(rng, k),
        "fft_cq15_1024" => run_fft_cq15_1024(rng, k),
        "agc" => run_agc(rng, k),
        "agc_cq15" => run_agc_cq15(rng, k),
        "dc_block_real" => run_dc_block_real(rng, k),
        "dc_block_real_q15" => run_dc_block_real_q15(rng, k),
        _ => false,
    }
}

#[entry]
fn main() -> ! {
    let mut buf = [0u8; 128];
    let Some((name, mode)) = arguments(&mut buf) else {
        if let Ok(mut out) = hio::hstdout() {
            let _ = writeln!(
                out,
                "sdrk-count error: no `<kernel> <k>` on the command line"
            );
        }
        exit(false)
    };
    let (ok, energies) = match run_biquad_kernel(name, mode) {
        Some(e) => (e.is_some(), e),
        None => match mode {
            Mode::Count(k) => (run(name, k), None),
            Mode::Snr => (false, None),
        },
    };
    let k = mode.calls();
    if let Ok(mut out) = hio::hstdout() {
        if let (Mode::Snr, Some(e)) = (mode, energies) {
            let _ = writeln!(out, "sdrk-snr {name} sig={:?} err={:?}", e.sig, e.err);
        }
        let _ = writeln!(
            out,
            "sdrk-count {name} k={k} ok={ok} kernels={}",
            KERNELS.len()
        );
    }
    exit(ok)
}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    exit(false)
}
