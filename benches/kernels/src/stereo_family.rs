//! The stereo entries (milestone M24): `stereo_matrix` and `stereo_matrix_cq15` with their
//! `_fallback` entries, and the `stereo_pll_fallback` and `stereo_pll_q15_fallback` entries
//! (`stereo_pll` has no SIMD branch and therefore no per-level entry).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `stereo_matrix` | facade | VOLK 3.3.0 composition (five calls, below) | |
//! | `stereo_matrix_cq15` | facade | our fallback (efficiency verdict) | CMSIS-DSP planar composition |
//! | `stereo_matrix_fallback` | our fallback | CMSIS-DSP generic C on planar data | |
//! | `stereo_matrix_cq15_fallback` | our fallback | CMSIS-DSP generic C on planar data | |
//! | `stereo_pll_fallback` | our fallback | liquid-dsp 1.8.3 `nco_crcf` (`LIQUID_VCO_INTERP`) | |
//! | `stereo_pll_q15_fallback` | our fallback | none (efficiency verdict) | CMSIS-DSP `arm_cos_q15` / `arm_sin_q15` loop |
//!
//! The VOLK composition of `stereo_matrix` is what a VOLK user writes (VOLK has no kernel for the
//! sum/difference matrix): `volk_32fc_deinterleave_32f_x2` into `s` and `d`,
//! `volk_32f_x2_add_32f` for `d2 = d + d`, `volk_32f_x2_add_32f` for `l = s + d2`,
//! `volk_32f_x2_subtract_32f` for `r = s - d2` and `volk_32f_x2_interleave_32fc`: five passes over
//! memory where our kernel makes one. The temporaries are 64-byte aligned buffers prepared once;
//! the calls go through VOLK's dispatchers as GNU Radio's blocks do.
//!
//! The CMSIS-DSP side works on planar data (as for `dot`, D57): the planar `s` and `d` are
//! prepared once and not counted, the three calls `arm_add_f32` (`d + d`), `arm_add_f32`
//! (`s + d2`) and `arm_sub_f32` (`s - d2`) (`arm_*_q15` for `CQ15`) are counted.
//!
//! The `stereo_pll` workload is the one of the amendment of D108 to D111 on 4096 samples at
//! 480 kHz with a locked pilot: L 1 kHz and R 3 kHz, the pilot 0.1 and the noise of the golden case
//! `lock_f32_16384` (real noise sigma 0.01 on the multiplex signal, 0.005 on the pilot). The
//! pilot has exactly 162 cycles per call (18 984.375 Hz) so that the repeated buffer is
//! continuous in phase, while the loop is created for 19 000 Hz (a 15.6 Hz offset, as a real
//! receiver has it), and every implementation is run over 24 calls at preparation, so that every
//! counted call runs on a locked loop whose lock decision is on.

use std::hint::black_box;

use cref_sys::LiquidNcoPll;
use cref_sys::cmsis_dsp::{self, add_f32, add_q15, sub_f32, sub_q15};
use sdrk::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use sdrk::{C32, CQ15, Fallback, Q15, Rate, Sdrk};
use sdrk_core::RealSample;
use volk_sys::lv_32fc_t;

use super::{
    Aligned, BiquadEntry, Impl, Kernel, Layout, Pcg32, c32s, cq15s, describe_cmsis,
    dot_scale_supports, noise_c32, snr_db,
};

// ---------------------------------------------------------------------------------------------
// stereo_matrix

/// The names of the VOLK kernels of the composition.
const VOLK_COMPOSITION: &[&str] = &[
    "volk_32fc_deinterleave_32f_x2",
    "volk_32f_x2_add_32f",
    "volk_32f_x2_subtract_32f",
    "volk_32f_x2_interleave_32fc",
];

/// `stereo_matrix` of `n` `C32` pairs `(s, d)`.
pub(crate) struct MatrixK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    entry: BiquadEntry,
    // The planar form for CMSIS-DSP (inputs and outputs) and for VOLK (temporaries).
    s: Aligned<f32>,
    d: Aligned<f32>,
    d2: Aligned<f32>,
    l: Aligned<f32>,
    r: Aligned<f32>,
}

impl MatrixK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let n = layout.n;
        let input = noise_c32(rng, n, 0.6, layout.skew(8));
        let skew = layout.skew(4);
        let mut k = MatrixK {
            input,
            out: Aligned::skewed(n, layout.skew(8)),
            entry,
            s: Aligned::skewed(n, skew),
            d: Aligned::skewed(n, skew),
            d2: Aligned::skewed(n, skew),
            l: Aligned::skewed(n, skew),
            r: Aligned::skewed(n, skew),
        };
        for (i, v) in k.input.iter().enumerate() {
            k.s[i] = v.re;
            k.d[i] = v.im;
        }
        k
    }

    /// The five VOLK calls.
    #[inline(never)]
    fn volk_composition(&mut self) {
        let n = self.input.len() as u32;
        // `d` is read twice by the first addition; VOLK's kernels read their inputs only.
        // SAFETY: every buffer holds `n` floats (`n` complex samples for the interleaved ones)
        // and no output aliases an input of its call.
        unsafe {
            volk_sys::volk_32fc_deinterleave_32f_x2(
                self.s.as_mut_ptr(),
                self.d.as_mut_ptr(),
                self.input.as_ptr().cast::<lv_32fc_t>(),
                n,
            );
            volk_sys::volk_32f_x2_add_32f(
                self.d2.as_mut_ptr(),
                self.d.as_ptr(),
                self.d.as_ptr(),
                n,
            );
            volk_sys::volk_32f_x2_add_32f(
                self.l.as_mut_ptr(),
                self.s.as_ptr(),
                self.d2.as_ptr(),
                n,
            );
            volk_sys::volk_32f_x2_subtract_32f(
                self.r.as_mut_ptr(),
                self.s.as_ptr(),
                self.d2.as_ptr(),
                n,
            );
            volk_sys::volk_32f_x2_interleave_32fc(
                self.out.as_mut_ptr().cast::<lv_32fc_t>(),
                self.l.as_ptr(),
                self.r.as_ptr(),
                n,
            );
        }
    }

    /// The three CMSIS-DSP calls on the planar data.
    #[inline(never)]
    fn cmsis_composition(&mut self) {
        add_f32(&self.d, &self.d, &mut self.d2);
        add_f32(&self.s, &self.d2, &mut self.l);
        sub_f32(&self.s, &self.d2, &mut self.r);
    }
}

#[inline(never)]
fn matrix<T: sdrk_core::ComplexSample, B: StereoMatrix<T>>(b: &B, input: &[T], out: &mut [T]) {
    let _ = black_box(b.stereo_matrix(black_box(input), out));
}

impl Kernel for MatrixK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => matrix(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => matrix(&Fallback, &self.input, &mut self.out),
                Impl::Volk => self.volk_composition(),
                Impl::Cmsis => self.cmsis_composition(),
                other => unreachable!("stereo_matrix has no {other:?}"),
            }
            black_box(&self.out);
            black_box(&self.l);
            black_box(&self.r);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = Vec::with_capacity(2 * self.input.len());
        let mut got = Vec::with_capacity(2 * self.input.len());
        for (i, v) in self.input.iter().enumerate() {
            let (s, d) = (f64::from(v.re), f64::from(v.im));
            want.extend([s + 2.0 * d, s - 2.0 * d]);
            if which == Impl::Cmsis {
                got.extend([f64::from(self.l[i]), f64::from(self.r[i])]);
            }
        }
        if which != Impl::Cmsis {
            got = c32s(&self.out);
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, volk_sys::FULL)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => {
                "VOLK 3.3.0 composition: volk_32fc_deinterleave_32f_x2, volk_32f_x2_add_32f (d + d), volk_32f_x2_add_32f (s + d2), volk_32f_x2_subtract_32f (s - d2), volk_32f_x2_interleave_32fc (five passes over memory)"
            }
            Impl::Cmsis => describe_cmsis(
                "arm_add_f32 (d + d), arm_add_f32 (s + d2), arm_sub_f32 (s - d2) on planar data",
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if volk_sys::FULL && self.entry == BiquadEntry::Simd {
            VOLK_COMPOSITION
        } else {
            &[]
        }
    }
}

/// `stereo_matrix` of `n` `CQ15` pairs over the whole range of `i16` (it saturates often).
pub(crate) struct MatrixCq15K {
    input: Aligned<CQ15>,
    out: Aligned<CQ15>,
    entry: BiquadEntry,
    s: Aligned<i16>,
    d: Aligned<i16>,
    d2: Aligned<i16>,
    l: Aligned<i16>,
    r: Aligned<i16>,
}

impl MatrixCq15K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let n = layout.n;
        let mut input = Aligned::<CQ15>::skewed(n, layout.skew(4));
        for v in input.iter_mut() {
            v.re = Q15((rng.next_u32() >> 16) as i16);
            v.im = Q15((rng.next_u32() >> 16) as i16);
        }
        let skew = layout.skew(2);
        let mut k = MatrixCq15K {
            input,
            out: Aligned::skewed(n, layout.skew(4)),
            entry,
            s: Aligned::skewed(n, skew),
            d: Aligned::skewed(n, skew),
            d2: Aligned::skewed(n, skew),
            l: Aligned::skewed(n, skew),
            r: Aligned::skewed(n, skew),
        };
        for (i, v) in k.input.iter().enumerate() {
            k.s[i] = v.re.0;
            k.d[i] = v.im.0;
        }
        k
    }

    /// `arm_add_q15` saturates `d + d` before the final sum, `arm_sub_q15` the difference.
    #[inline(never)]
    fn cmsis_composition(&mut self) {
        add_q15(&self.d, &self.d, &mut self.d2);
        add_q15(&self.s, &self.d2, &mut self.l);
        sub_q15(&self.s, &self.d2, &mut self.r);
    }
}

impl Kernel for MatrixCq15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => matrix(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => matrix(&Fallback, &self.input, &mut self.out),
                Impl::Cmsis => self.cmsis_composition(),
                other => unreachable!("stereo_matrix_cq15 has no {other:?}"),
            }
            black_box(&self.out);
            black_box(&self.l);
            black_box(&self.r);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let sat = |v: i64| v.clamp(-32768, 32767) as f64;
        let mut want = Vec::with_capacity(2 * self.input.len());
        let mut got = Vec::with_capacity(2 * self.input.len());
        for (i, v) in self.input.iter().enumerate() {
            let (s, d) = (i64::from(v.re.0), i64::from(v.im.0));
            want.extend([sat(s + 2 * d), sat(s - 2 * d)]);
            if which == Impl::Cmsis {
                got.extend([f64::from(self.l[i]), f64::from(self.r[i])]);
            }
        }
        if which != Impl::Cmsis {
            got = cq15s(&self.out, 1.0);
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        match (self.entry, which) {
            (_, Impl::Fallback) | (BiquadEntry::Simd, Impl::Ours) => true,
            (_, Impl::Cmsis) => cmsis_dsp::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis(
                "arm_add_q15 (d + d, saturating), arm_add_q15 (s + d2), arm_sub_q15 (s - d2) on planar data",
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// stereo_pll

/// The rate of the workload.
const PLL_RATE_HZ: u64 = 480_000;
/// The pilot frequency the loops are created for.
const PLL_PILOT_HZ: f64 = 19_000.0;
/// Natural frequency of the loop (the default of the block `fm_stereo_demod`).
const PLL_NATURAL_HZ: f64 = 20.0;
/// Pilot cycles per call of 4096 samples (18 984.375 Hz): the repeated buffer is continuous.
const PILOT_CYCLES: u64 = 162;
/// Calls over which the loops lock before the measurement (0.2 s).
const WARM_CALLS: usize = 24;
/// The start phase of the pilot.
const PILOT_START_RAD: f64 = 1.0;

/// The multiplex signal and the pilot of the workload as `f64`, and the phase of the pilot per
/// sample in radians: `mpx = 0.45 (L + R) + 0.45 (L - R) sin(2 theta) + 0.1 sin(theta) + noise`.
fn pll_signals(rng: &mut Pcg32, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fs = PLL_RATE_HZ as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut gauss = |sigma: f64| {
        // Box-Muller from the generator of the benchmarks.
        let u1 = (f64::from(rng.next_u32()) + 1.0) / 4_294_967_297.0;
        let u2 = f64::from(rng.next_u32()) / 4_294_967_296.0;
        sigma * (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos()
    };
    let (mut mpx, mut pilot, mut theta) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::new());
    for k in 0..n {
        let kf = k as f64;
        let l = (two_pi * (1000.0 * kf / fs).fract()).sin();
        let r = (two_pi * (3000.0 * kf / fs).fract()).sin();
        let th = two_pi * ((PILOT_CYCLES * k as u64 % 4096) as f64 / 4096.0) + PILOT_START_RAD;
        let tone = 0.1 * th.sin();
        mpx.push(0.45 * (l + r) + 0.45 * (l - r) * (2.0 * th).sin() + tone + gauss(0.01));
        pilot.push(tone + gauss(0.005));
        theta.push(th);
    }
    (mpx, pilot, theta)
}

/// The `Q15` constants of the loop as `StereoPllState` derives them (`kp`, `ki` in `Q34`, the
/// integrator limit in `Q64`, the lock coefficient in `Q31`, the thresholds in `Q61`).
struct Q15Loop {
    kp: i64,
    ki: i64,
    imax: i64,
    beta: i64,
    on: i64,
    off: i64,
    step0: u32,
}

impl Q15Loop {
    fn new(state: &StereoPllState<Q15>) -> Self {
        let c = state.coefficients();
        let q = |v: f64, bits: i32| (v * 2f64.powi(bits)).round() as i64;
        Q15Loop {
            kp: q(c.kp, 34),
            ki: q(c.ki, 34),
            imax: (c.imax * 18_446_744_073_709_551_616.0).round() as i64,
            beta: q(c.beta, 31),
            on: (c.on * 2_305_843_009_213_693_952.0).round() as i64,
            off: (c.off * 2_305_843_009_213_693_952.0).round() as i64,
            step0: c.step0,
        }
    }
}

/// The loop of `stereo_pll` for `Q15` with the oscillator of CMSIS-DSP (`arm_cos_q15`,
/// `arm_sin_q15`: a 512-entry table with linear interpolation) in place of the table of
/// `rotate`; the arithmetic around it is the one of the kernel.
#[inline(never)]
fn cmsis_stereo_pll_q15(
    lp: &Q15Loop,
    st: &mut (u32, i64, i64, bool),
    mpx: &[Q15],
    pilot: &[Q15],
    out: &mut [CQ15],
) {
    let (mut phase, mut integ, mut level, mut stereo) = *st;
    for ((o, x), p) in out.iter_mut().zip(mpx).zip(pilot) {
        let t = (phase >> 17) as i16;
        let t2 = (phase.wrapping_mul(2) >> 17) as i16;
        // SAFETY: plain scalar functions of CMSIS-DSP; the callers check `cmsis_dsp::AVAILABLE`.
        let (c, s, s2) = unsafe {
            (
                cmsis_dsp::ffi::arm_cos_q15(t),
                cmsis_dsp::ffi::arm_sin_q15(t),
                cmsis_dsp::ffi::arm_sin_q15(t2),
            )
        };
        let pv = i64::from(p.0);
        let e = pv * i64::from(c);
        let q = pv * i64::from(s);
        level += lp.beta * (q - ((level + (1 << 30)) >> 31));
        if level >= lp.on {
            stereo = true;
        } else if level < lp.off {
            stereo = false;
        }
        let diff = ((i32::from(x.0) * i32::from(s2) + (1 << 14)) >> 15).clamp(-32768, 32767) as i16;
        *o = CQ15 {
            re: *x,
            im: Q15(if stereo { diff } else { 0 }),
        };
        integ = (integ + lp.ki * e).clamp(-lp.imax, lp.imax);
        let f = lp.kp * e + integ;
        phase = phase
            .wrapping_add(lp.step0)
            .wrapping_add(((f + (1 << 31)) >> 32) as u32);
    }
    *st = (phase, integ, level, stereo);
}

/// `stereo_pll` of `f32`: our fallback, and liquid-dsp's `nco_crcf`.
pub(crate) struct PllF32K {
    mpx: Aligned<f32>,
    pilot: Aligned<f32>,
    out: Aligned<C32>,
    ours: StereoPllState<f32>,
    fallback: StereoPllState<f32>,
    liquid: Option<LiquidNcoPll>,
    /// `kp`, `ki` in radians per unit of the detector output, as liquid takes them.
    gains: (f32, f32),
    theta: Vec<f64>,
}

fn pll_state<T: RealSample>(level: f64) -> StereoPllState<T> {
    StereoPllState::new(
        PLL_PILOT_HZ,
        PLL_NATURAL_HZ,
        level,
        Rate::hz(PLL_RATE_HZ).expect("rate"),
    )
    .expect("stereo loop")
}

impl PllF32K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let (mpx, pilot, theta) = pll_signals(rng, n);
        let mut k = PllF32K {
            mpx: Aligned::skewed(n, layout.skew(4)),
            pilot: Aligned::skewed(n, layout.skew(4)),
            out: Aligned::skewed(n, layout.skew(8)),
            ours: pll_state(0.1),
            fallback: pll_state(0.1),
            liquid: LiquidNcoPll::new(PLL_PILOT_HZ, PLL_RATE_HZ as f64),
            gains: (0.0, 0.0),
            theta,
        };
        let co = k.ours.coefficients();
        let two_pi = 2.0 * std::f64::consts::PI;
        k.gains = ((co.kp * two_pi) as f32, (co.ki * two_pi) as f32);
        for i in 0..n {
            k.mpx[i] = mpx[i] as f32;
            k.pilot[i] = pilot[i] as f32;
        }
        for which in [Impl::Ours, Impl::Fallback, Impl::Liquid] {
            if k.supports(which) {
                k.run(which, WARM_CALLS);
            }
        }
        k
    }

    fn liquid_block(&mut self) {
        let Some(l) = self.liquid.as_mut() else {
            return;
        };
        // SAFETY: `C32` is `repr(C)` of two `f32`, so the slice has the layout of `[[f32; 2]]`.
        let out = unsafe {
            std::slice::from_raw_parts_mut(self.out.as_mut_ptr().cast::<[f32; 2]>(), self.out.len())
        };
        l.pll_block(&self.mpx, &self.pilot, out, self.gains.0, self.gains.1);
    }
}

#[inline(never)]
fn pll<T: RealSample, B: StereoPll<T>>(
    b: &B,
    st: &mut StereoPllState<T>,
    mpx: &[T],
    pilot: &[T],
    out: &mut [T::Complex],
) {
    let _ = black_box(b.stereo_pll(st, black_box(mpx), black_box(pilot), out));
}

impl Kernel for PllF32K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Fallback => pll(
                    &Fallback,
                    &mut self.fallback,
                    &self.mpx,
                    &self.pilot,
                    &mut self.out,
                ),
                Impl::Ours => pll(&Sdrk, &mut self.ours, &self.mpx, &self.pilot, &mut self.out),
                Impl::Liquid => self.liquid_block(),
                other => unreachable!("stereo_pll has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The SNR of the difference channel `x sin(2 theta)` of one call on the locked loop against
    /// the ideal oscillator (the true phase of the pilot, exact sine): it contains the phase
    /// noise of the loop, which both sides share, and the error of the oscillator table.
    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .mpx
            .iter()
            .zip(&self.theta)
            .map(|(&x, &th)| f64::from(x) * (2.0 * th).sin())
            .collect();
        let got: Vec<f64> = self.out.iter().map(|v| f64::from(v.im)).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Fallback => true,
            Impl::Liquid => self.liquid.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Liquid => Box::leak(
                format!(
                    "liquid-dsp 1.8.3 nco_crcf (LIQUID_VCO_INTERP): sincos at theta and 2 theta from the 1024-entry table, nco_crcf_adjust_frequency and nco_crcf_adjust_phase (double modulo in _constrain), nco_crcf_step; no lock detector, no integrator limit (-O3 -ffp-contract=off, {})",
                    cref_sys::BIQUAD_REF_LEVEL
                )
                .into_boxed_str(),
            ),
            _ => "sdrk fallback",
        }
    }
}

/// `stereo_pll` of `Q15`: our fallback, and the CMSIS-DSP oscillator loop as information.
pub(crate) struct PllQ15K {
    mpx: Aligned<Q15>,
    pilot: Aligned<Q15>,
    out: Aligned<CQ15>,
    ours: StereoPllState<Q15>,
    fallback: StereoPllState<Q15>,
    cmsis: (Q15Loop, (u32, i64, i64, bool)),
    theta: Vec<f64>,
}

impl PllQ15K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let (mpx, pilot, theta) = pll_signals(rng, n);
        let quant = |v: f64| Q15((v * 0.625 * 32768.0).round().clamp(-32768.0, 32767.0) as i16);
        let ours = pll_state::<Q15>(0.0625);
        let mut k = PllQ15K {
            mpx: Aligned::skewed(n, layout.skew(2)),
            pilot: Aligned::skewed(n, layout.skew(2)),
            out: Aligned::skewed(n, layout.skew(4)),
            cmsis: (Q15Loop::new(&ours), (0, 0, 0, false)),
            ours,
            fallback: pll_state(0.0625),
            theta,
        };
        for i in 0..n {
            k.mpx[i] = quant(mpx[i]);
            k.pilot[i] = quant(pilot[i]);
        }
        for which in [Impl::Ours, Impl::Fallback, Impl::Cmsis] {
            if k.supports(which) {
                k.run(which, WARM_CALLS);
            }
        }
        k
    }
}

impl Kernel for PllQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Fallback => pll(
                    &Fallback,
                    &mut self.fallback,
                    &self.mpx,
                    &self.pilot,
                    &mut self.out,
                ),
                Impl::Ours => pll(&Sdrk, &mut self.ours, &self.mpx, &self.pilot, &mut self.out),
                Impl::Cmsis => cmsis_stereo_pll_q15(
                    &self.cmsis.0,
                    &mut self.cmsis.1,
                    &self.mpx,
                    &self.pilot,
                    &mut self.out,
                ),
                other => unreachable!("stereo_pll_q15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// As [`PllF32K::accuracy`], in raw `Q15` units.
    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .mpx
            .iter()
            .zip(&self.theta)
            .map(|(x, &th)| f64::from(x.0) * (2.0 * th).sin())
            .collect();
        let got: Vec<f64> = self.out.iter().map(|v| f64::from(v.im.0)).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Fallback => true,
            Impl::Cmsis => cmsis_dsp::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis(
                "arm_cos_q15 and arm_sin_q15 (twice) per sample in the loop of the kernel (the oscillator only: CMSIS-DSP has no PLL)",
            ),
            _ => "sdrk fallback",
        }
    }
}

#[cfg(test)]
mod tests {
    //! The references must compute what we compare them with.

    use super::*;

    #[test]
    fn matrix_references_compute_the_matrix() {
        let mut rng = Pcg32::new(5);
        let mut k = MatrixK::new(&mut rng, Layout::default(), BiquadEntry::Simd);
        for which in [Impl::Ours, Impl::Fallback, Impl::Volk, Impl::Cmsis] {
            if which == Impl::Cmsis && !cmsis_dsp::AVAILABLE {
                continue;
            }
            if which == Impl::Volk && !volk_sys::FULL {
                continue;
            }
            let snr = k.accuracy(which);
            assert!(snr > 130.0, "{which:?}: {snr} dB");
        }
    }

    #[test]
    fn pll_locks_and_liquid_follows() {
        let mut rng = Pcg32::new(6);
        let mut k = PllF32K::new(&mut rng, Layout::default());
        assert!(k.fallback.is_stereo(), "our loop is locked");
        let ours = k.accuracy(Impl::Fallback);
        assert!(ours > 40.0, "ours {ours} dB");
        if k.supports(Impl::Liquid) {
            let liquid = k.accuracy(Impl::Liquid);
            // liquid's `_constrain` rounds the small corrections of this narrow loop to
            // multiples of 4.8e-7 rad (see tests/liquid_nco_refs.rs): it follows the pilot
            // with a much lower accuracy than ours, which is what the accuracy line reports.
            assert!(liquid > 3.0, "liquid {liquid} dB (ours {ours} dB)");
        }
        let mut q = PllQ15K::new(&mut Pcg32::new(7), Layout::default());
        assert!(q.fallback.is_stereo(), "the Q15 loop is locked");
        let ours = q.accuracy(Impl::Fallback);
        assert!(ours > 40.0, "Q15 ours {ours} dB");
        if q.supports(Impl::Cmsis) {
            let cmsis = q.accuracy(Impl::Cmsis);
            assert!(cmsis > 30.0, "Q15 CMSIS {cmsis} dB (ours {ours} dB)");
        }
    }
}
