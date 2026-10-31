//! The halfband entries (milestone M23): `halfband_decimate`, `halfband_decimate_real` and
//! `real_to_complex`, each for the float and the Q15 type, with their `_fallback` entries, and the
//! two comparison workloads of our own kernels.
//!
//! The workload is the one of the milestone: a halfband filter of 47 taps (SDRk's design,
//! `halfband_taps` with a Kaiser window of 70 dB: 24 non-zero taps at the odd offsets from the
//! centre tap and the centre tap, which equals liquid-dsp's `m = 12` and the 47-tap kernel of
//! libairspy), decimation by 2, 4096 input samples per call, so the counts are per input sample
//! as those of the other filters. The references get SDRk's taps, so that their accuracy against
//! the same `f64` model is comparable:
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `halfband_decimate` | facade | liquid-dsp `resamp2_crcf` (SIMD dot product of the level) | `fir_decimate` by 2 with the expanded taps |
//! | `halfband_decimate_real` | facade | liquid-dsp `resamp2_rrrf` | |
//! | `real_to_complex` | facade | liquid-dsp `firhilbf` | libairspy `iqconverter_float` (SSE2) |
//! | `halfband_decimate_cq15`, `halfband_decimate_real_q15` | facade | our fallback | the CMSIS-DSP composition |
//! | `real_to_complex_cq15` | facade | libairspy `iqconverter_int16` (information until the maintainer decides) | |
//! | `<kernel>_fallback` | our fallback | liquid-dsp with the portable dot product (`real_to_complex`: also libairspy's generic code as information); the Q15 types: the CMSIS-DSP composition (`real_to_complex_cq15`: libairspy `iqconverter_int16`) | |
//!
//! The CMSIS-DSP composition of the Q15 kernels (CMSIS-DSP has no halfband decimator): the
//! polyphase split of the input into its even and odd samples, `arm_fir_q15` with the 24 compacted
//! taps on the even samples (the filter branch), `arm_scale_q15` with the centre tap on the odd
//! samples (the delay branch, delayed by 12 samples), and `arm_add_q15`. The complex type runs it
//! on the planar `re` and `im` (prepared once and not counted, as for the FIR family).
//!
//! What the external libraries compute is mapped to SDRk's definition before the SNR is taken
//! (outside the counted region): liquid's `firhilbf` is `j / centre` times SDRk's
//! `real_to_complex`, libairspy's converters are `-conj` of it.

use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;

use cref_sys::cmsis_dsp::{CmsisFirQ15, add_q15, scale_q15};
use cref_sys::{AirspyFloat, AirspyInt16, LiquidFirhilbf, LiquidResamp2Crcf, LiquidResamp2Rrrf};
use sdrk::kernels::decim::{FirDecimState, FirDecimate};
use sdrk::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState,
};
use sdrk::{C32, CQ15, ComplexSample, Fallback, Q15, Sdrk};
use sdrk_core::{RealSample, Reference};

use super::{
    Aligned, BiquadEntry, Impl, Kernel, Layout, Pcg32, RealNoise, SquelchSample, describe_cmsis,
    snr_db,
};

/// Taps of the halfband workload (liquid's `m = 12`: `4 m - 1`).
pub(crate) const HB_TAPS: usize = 47;
/// The semi-length `m` of liquid's prototype of `4 m + 1` taps.
const HB_M: usize = 12;
/// The attenuation of the Kaiser design of the workload taps.
const HB_ATTENUATION_DB: f64 = 70.0;

/// A real sample type of the halfband entries: `f32` or Q15.
pub(crate) trait Scalar: RealNoise + Copy + Default + 'static {
    /// Whether this is the Q15 type.
    const Q15: bool;
    /// A tap from the `f64` design (Q15: rounded, saturated).
    fn design(h: f64) -> Self;
    /// The value as `f64` in the units of the `f64` model (Q15: raw integers).
    fn value(self) -> f64;
    /// The output sample of a float library (`f32` only).
    fn from_float_lib(v: f32) -> Self;
    /// The output sample of a Q15 library (Q15 only).
    fn from_q15_lib(v: i16) -> Self;
}

impl Scalar for f32 {
    const Q15: bool = false;
    fn design(h: f64) -> Self {
        h as f32
    }
    fn value(self) -> f64 {
        f64::from(self)
    }
    fn from_float_lib(v: f32) -> f32 {
        v
    }
    fn from_q15_lib(_: i16) -> f32 {
        unreachable!("no Q15 library output for f32")
    }
}

impl Scalar for Q15 {
    const Q15: bool = true;
    fn design(h: f64) -> Self {
        Q15((h * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
    }
    fn value(self) -> f64 {
        f64::from(self.0)
    }
    fn from_float_lib(_: f32) -> Q15 {
        unreachable!("no float library output for Q15")
    }
    fn from_q15_lib(v: i16) -> Q15 {
        Q15(v)
    }
}

/// The 47 taps of the workload as `f64`.
fn design_taps() -> Vec<f64> {
    let mut h = vec![0.0f64; HB_TAPS];
    sdrk_blocks::design::halfband_taps(sdrk_blocks::design::kaiser_beta(HB_ATTENUATION_DB), &mut h)
        .expect("the halfband taps of the workload");
    h
}

fn taps_of<S: Scalar>() -> Vec<S> {
    design_taps().into_iter().map(S::design).collect()
}

/// The 4 m + 1 taps of liquid's prototype: the 47 taps between two zero taps, divided by the
/// centre tap (liquid's delay branch has the gain 1; its `scale` or the mapping of the output
/// restores the centre tap).
fn liquid_prototype(taps: &[f32], divide: bool) -> Vec<f32> {
    let c = taps[HB_TAPS / 2];
    let mut h = vec![0.0f32; 4 * HB_M + 1];
    for (d, s) in h[1..=HB_TAPS].iter_mut().zip(taps) {
        *d = if divide { s / c } else { *s };
    }
    h
}

fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

/// The dot product of liquid at the level of this build: `rrrf` for the real dot product of
/// `resamp2_rrrf` and `firhilbf`, whose SSE loop needs the SSE3 horizontal add.
fn liquid_dot(rrrf: bool) -> &'static str {
    match (cref_sys::BIQUAD_REF_LEVEL, rrrf) {
        ("sse2", true) => "portable",
        ("sse2", false) | ("sse4.2", false) => "SSE",
        ("sse4.2", true) => "SSE (SSE3 horizontal add)",
        ("avx2", _) => "AVX",
        ("avx512", _) => "AVX-512F",
        ("neon", _) => "NEON",
        _ => "portable",
    }
}

fn describe_liquid(function: &str, rrrf: bool, simd: bool) -> &'static str {
    let dot = if simd { liquid_dot(rrrf) } else { "portable" };
    leak(format!(
        "liquid-dsp 1.8.3 {function} with its {dot} dot product (-O3 -ffp-contract=off, {})",
        cref_sys::BIQUAD_REF_LEVEL
    ))
}

fn describe_airspy(function: &str, note: &str) -> &'static str {
    leak(format!(
        "libairspy 1.0.10 {function}, conversion without remove_dc{note} (-O3 -ffp-contract=off, {})",
        cref_sys::BIQUAD_REF_LEVEL
    ))
}

/// An external implementation prepared on its own copy of the input: `run` does the counted
/// work, `result` maps the output to SDRk's definition (not counted).
pub(crate) struct Ext<T> {
    run: Box<dyn FnMut()>,
    result: Box<dyn Fn() -> Vec<T>>,
}

/// Wraps the state `st` of an external run in an [`Ext`].
fn ext<S: 'static, T: 'static>(
    st: S,
    run: impl Fn(&mut S) + 'static,
    result: impl Fn(&S) -> Vec<T> + 'static,
) -> Ext<T> {
    let st = Rc::new(RefCell::new(st));
    let st2 = Rc::clone(&st);
    Ext {
        run: Box::new(move || run(&mut st.borrow_mut())),
        result: Box::new(move || result(&st2.borrow())),
    }
}

// ---------------------------------------------------------------------------------------------
// The CMSIS-DSP composition (Q15)

/// One channel of the CMSIS-DSP composition of the Q15 halfband decimator.
struct CmsisHalfband {
    fir: CmsisFirQ15,
    centre: i16,
    even: Vec<i16>,
    odd: Vec<i16>,
    filtered: Vec<i16>,
    /// `HB_M` samples of the delay branch of the previous call, then this call's.
    delay: Vec<i16>,
}

impl CmsisHalfband {
    fn new(taps: &[i16], block: usize) -> Option<Self> {
        let half = block / 2;
        let compact: Vec<i16> = taps.iter().step_by(2).copied().collect();
        Some(CmsisHalfband {
            fir: CmsisFirQ15::new(&compact, half)?,
            centre: taps[HB_TAPS / 2],
            even: vec![0; half],
            odd: vec![0; half],
            filtered: vec![0; half],
            delay: vec![0; half + HB_M],
        })
    }

    /// `input.len() / 2` outputs.
    fn run(&mut self, input: &[i16], out: &mut [i16]) {
        let half = out.len();
        for ((e, o), pair) in self
            .even
            .iter_mut()
            .zip(&mut self.odd)
            .zip(input.chunks_exact(2))
        {
            *e = pair[0];
            *o = pair[1];
        }
        self.fir.run(&self.even, &mut self.filtered);
        scale_q15(&self.odd, self.centre, 0, &mut self.delay[HB_M..]);
        add_q15(&self.filtered, &self.delay[..half], out);
        self.delay.copy_within(half.., 0);
    }
}

// ---------------------------------------------------------------------------------------------
// The complex types

/// `C32` and `CQ15` for the halfband entries.
pub(crate) trait HbCx: ComplexSample + SquelchSample + Copy + Default + 'static
where
    <Self as ComplexSample>::Real: Scalar,
{
    /// liquid-dsp's `resamp2_crcf` decimator (`C32` only).
    fn liquid_decim(_taps: &[f32], _input: &[Self], _simd: bool) -> Option<Ext<Self>> {
        None
    }
    /// libairspy's converter (`sse2`: the SSE2 build).
    fn airspy(
        _taps: &[<Self as ComplexSample>::Real],
        _input: &[<Self as ComplexSample>::Real],
        _sse2: bool,
    ) -> Option<Ext<Self>> {
        None
    }
    /// liquid-dsp's `firhilbf` (`C32` only).
    fn liquid_hilbert(
        _taps: &[<Self as ComplexSample>::Real],
        _input: &[<Self as ComplexSample>::Real],
        _simd: bool,
    ) -> Option<Ext<Self>> {
        None
    }
    /// The CMSIS-DSP composition (`CQ15` only).
    fn cmsis_decim(_taps: &[<Self as ComplexSample>::Real], _input: &[Self]) -> Option<Ext<Self>> {
        None
    }
}

impl HbCx for C32 {
    fn liquid_decim(taps: &[f32], input: &[C32], simd: bool) -> Option<Ext<C32>> {
        let c = taps[HB_TAPS / 2];
        let q = LiquidResamp2Crcf::new(&liquid_prototype(taps, true), c, simd)?;
        let x: Vec<[f32; 2]> = input.iter().map(|v| [v.re, v.im]).collect();
        let y = vec![[0.0f32; 2]; input.len() / 2];
        Some(ext(
            (q, x, y),
            |(q, x, y)| q.decim(x, y),
            |(_, _, y)| y.iter().map(|v| C32::new(v[0], v[1])).collect(),
        ))
    }

    fn airspy(taps: &[f32], input: &[f32], sse2: bool) -> Option<Ext<C32>> {
        let q = AirspyFloat::new(taps, sse2)?;
        Some(ext(
            (q, input.to_vec(), vec![0.0f32; input.len()]),
            |(q, x, y)| q.convert(x, y),
            // libairspy's output is `-conj` of SDRk's.
            |(_, _, y)| y.chunks_exact(2).map(|p| C32::new(-p[0], p[1])).collect(),
        ))
    }

    fn liquid_hilbert(taps: &[f32], input: &[f32], simd: bool) -> Option<Ext<C32>> {
        let c = taps[HB_TAPS / 2];
        let q = LiquidFirhilbf::new(&liquid_prototype(taps, true), simd)?;
        let y = vec![[0.0f32; 2]; input.len() / 2];
        Some(ext(
            (q, input.to_vec(), y),
            |(q, x, y)| q.decim(x, y),
            // liquid's output is `j / centre` times SDRk's: `-j c z`.
            move |(_, _, y)| y.iter().map(|v| C32::new(c * v[1], -c * v[0])).collect(),
        ))
    }
}

impl HbCx for CQ15 {
    fn airspy(taps: &[Q15], input: &[Q15], _sse2: bool) -> Option<Ext<CQ15>> {
        let raw: Vec<i16> = taps.iter().map(|t| t.0).collect();
        let q = AirspyInt16::new(&raw)?;
        let x: Vec<i16> = input.iter().map(|v| v.0).collect();
        let n = x.len();
        Some(ext(
            (q, x, vec![0i16; n]),
            |(q, x, y)| q.convert(x, y),
            // `-conj`, saturating.
            |(_, _, y)| {
                y.chunks_exact(2)
                    .map(|p| CQ15 {
                        re: Q15(p[0].saturating_neg()),
                        im: Q15(p[1]),
                    })
                    .collect()
            },
        ))
    }

    fn cmsis_decim(taps: &[Q15], input: &[CQ15]) -> Option<Ext<CQ15>> {
        let raw: Vec<i16> = taps.iter().map(|t| t.0).collect();
        let n = input.len();
        let re: Vec<i16> = input.iter().map(|v| v.re.0).collect();
        let im: Vec<i16> = input.iter().map(|v| v.im.0).collect();
        let st = (
            CmsisHalfband::new(&raw, n)?,
            CmsisHalfband::new(&raw, n)?,
            re,
            im,
            vec![0i16; n / 2],
            vec![0i16; n / 2],
        );
        Some(ext(
            st,
            |(a, b, re, im, ore, oim)| {
                a.run(re, ore);
                b.run(im, oim);
            },
            |(_, _, _, _, ore, oim)| {
                ore.iter()
                    .zip(oim)
                    .map(|(r, i)| CQ15 {
                        re: Q15(*r),
                        im: Q15(*i),
                    })
                    .collect()
            },
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// halfband_decimate

/// Which side an entry of the family measures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    /// Our kernel through the facade.
    Simd,
    /// Our fallback.
    Fallback,
}

impl From<BiquadEntry> for Side {
    fn from(e: BiquadEntry) -> Side {
        if e == BiquadEntry::Simd {
            Side::Simd
        } else {
            Side::Fallback
        }
    }
}

type CxState<T> = HalfbandState<T, Vec<<T as ComplexSample>::Real>, Vec<T>>;

#[inline(never)]
fn halfband_decimate<T: ComplexSample, B: HalfbandDecimate<T>>(
    backend: &B,
    state: &mut CxState<T>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.halfband_decimate(state, black_box(input), out));
}

/// `halfband_decimate` for `C32` and `CQ15`.
pub(crate) struct DecimateK<T: HbCx>
where
    <T as ComplexSample>::Real: Scalar,
{
    input: Aligned<T>,
    out: Aligned<T>,
    taps: Vec<<T as ComplexSample>::Real>,
    state: CxState<T>,
    side: Side,
    n: usize,
}

impl<T: HbCx> DecimateK<T>
where
    <T as ComplexSample>::Real: Scalar,
{
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        assert!(
            layout.n % 2 == 0,
            "halfband_decimate: even block lengths only"
        );
        let size = std::mem::size_of::<T>();
        let mut input = Aligned::<T>::skewed(layout.n, layout.skew(size));
        for s in input.iter_mut() {
            *s = <T as SquelchSample>::noise(rng, 0.9);
        }
        let taps = taps_of::<<T as ComplexSample>::Real>();
        DecimateK {
            input,
            out: Aligned::skewed(layout.n / 2, layout.skew(size)),
            state: Self::state(&taps),
            taps,
            side: entry.into(),
            n: layout.n,
        }
    }

    fn state(taps: &[<T as ComplexSample>::Real]) -> CxState<T> {
        HalfbandState::new(taps.to_vec(), vec![T::default(); HB_TAPS - 1]).expect("halfband taps")
    }

    fn external(&self, which: Impl) -> Option<Ext<T>> {
        let taps = &self.taps;
        match which {
            Impl::Liquid => {
                let f: Vec<f32> = taps.iter().map(|t| t.value() as f32).collect();
                T::liquid_decim(&f, &self.input, self.side == Side::Simd)
            }
            Impl::Cmsis => T::cmsis_decim(taps, &self.input),
            _ => None,
        }
    }
}

impl<T: HbCx> Kernel for DecimateK<T>
where
    <T as ComplexSample>::Real: Scalar,
    Sdrk: HalfbandDecimate<T>,
    Fallback: HalfbandDecimate<T>,
    Reference: HalfbandDecimate<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        let mut ext = self.external(which);
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    halfband_decimate(&Sdrk, &mut self.state, &self.input, &mut self.out);
                }
                Impl::Fallback => {
                    halfband_decimate(&Fallback, &mut self.state, &self.input, &mut self.out);
                }
                _ => (ext.as_mut().expect("external implementation").run)(),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let mut want = vec![T::default(); self.n / 2];
        let mut s = Self::state(&self.taps);
        Reference
            .halfband_decimate(&mut s, &self.input, &mut want)
            .expect("reference");
        let got: Vec<T> = if let Some(mut e) = self.external(which) {
            (e.run)();
            (e.result)()
        } else {
            self.state = Self::state(&self.taps);
            self.run(which, 1);
            self.out.to_vec()
        };
        snr_db(&values(&want), &values(&got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Fallback => true,
            Impl::Ours => self.side == Side::Simd,
            Impl::Liquid | Impl::Cmsis => self.external(which).is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Liquid => describe_liquid(
                "resamp2_crcf_decim_execute (dotprod_crcf)",
                false,
                self.side == Side::Simd,
            ),
            Impl::Cmsis => describe_cmsis(
                "arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator)",
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

fn values<T: ComplexSample>(x: &[T]) -> Vec<f64>
where
    <T as ComplexSample>::Real: Scalar,
{
    x.iter()
        .flat_map(|v| [v.re().value(), v.im().value()])
        .collect()
}

// ---------------------------------------------------------------------------------------------
// halfband_decimate_real

type RealState<S> = HalfbandRealState<S, Vec<S>, Vec<S>>;

#[inline(never)]
fn halfband_decimate_real<S: sdrk_core::RealSample, B: HalfbandDecimateReal<S>>(
    backend: &B,
    state: &mut RealState<S>,
    input: &[S],
    out: &mut [S],
) {
    let _ = black_box(backend.halfband_decimate_real(state, black_box(input), out));
}

/// `halfband_decimate_real` for `f32` and Q15.
pub(crate) struct DecimateRealK<S: Scalar> {
    input: Aligned<S>,
    out: Aligned<S>,
    taps: Vec<S>,
    state: RealState<S>,
    side: Side,
    n: usize,
}

impl<S: Scalar> DecimateRealK<S> {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        assert!(
            layout.n % 2 == 0,
            "halfband_decimate_real: even block lengths only"
        );
        let size = std::mem::size_of::<S>();
        let mut input = Aligned::<S>::skewed(layout.n, layout.skew(size));
        for s in input.iter_mut() {
            *s = S::noise(rng, 0.9, 0.0);
        }
        let taps = taps_of::<S>();
        DecimateRealK {
            input,
            out: Aligned::skewed(layout.n / 2, layout.skew(size)),
            state: Self::state(&taps),
            taps,
            side: entry.into(),
            n: layout.n,
        }
    }

    fn state(taps: &[S]) -> RealState<S> {
        HalfbandRealState::new(taps.to_vec(), vec![S::default(); HB_TAPS - 1])
            .expect("halfband taps")
    }

    fn external(&self, which: Impl) -> Option<Ext<S>> {
        match which {
            Impl::Liquid if !S::Q15 => {
                let f: Vec<f32> = self.taps.iter().map(|t| t.value() as f32).collect();
                let c = f[HB_TAPS / 2];
                let q = LiquidResamp2Rrrf::new(
                    &liquid_prototype(&f, true),
                    c,
                    self.side == Side::Simd,
                )?;
                let x: Vec<f32> = self.input.iter().map(|v| v.value() as f32).collect();
                let y = vec![0.0f32; self.n / 2];
                Some(ext(
                    (q, x, y),
                    |(q, x, y)| q.decim(x, y),
                    |(_, _, y)| y.iter().map(|v| S::from_float_lib(*v)).collect(),
                ))
            }
            Impl::Cmsis if S::Q15 => {
                let raw: Vec<i16> = self.taps.iter().map(|t| t.value() as i16).collect();
                let x: Vec<i16> = self.input.iter().map(|v| v.value() as i16).collect();
                let st = (CmsisHalfband::new(&raw, self.n)?, x, vec![0i16; self.n / 2]);
                Some(ext(
                    st,
                    |(h, x, y)| h.run(x, y),
                    |(_, _, y)| y.iter().map(|v| S::from_q15_lib(*v)).collect(),
                ))
            }
            _ => None,
        }
    }
}

impl<S: Scalar> Kernel for DecimateRealK<S>
where
    Sdrk: HalfbandDecimateReal<S>,
    Fallback: HalfbandDecimateReal<S>,
    Reference: HalfbandDecimateReal<S>,
{
    fn run(&mut self, which: Impl, k: usize) {
        let mut ext = self.external(which);
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    halfband_decimate_real(&Sdrk, &mut self.state, &self.input, &mut self.out);
                }
                Impl::Fallback => {
                    halfband_decimate_real(&Fallback, &mut self.state, &self.input, &mut self.out);
                }
                _ => (ext.as_mut().expect("external implementation").run)(),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let mut want = vec![S::default(); self.n / 2];
        let mut s = Self::state(&self.taps);
        Reference
            .halfband_decimate_real(&mut s, &self.input, &mut want)
            .expect("reference");
        let got: Vec<S> = if let Some(mut e) = self.external(which) {
            (e.run)();
            (e.result)()
        } else {
            self.state = Self::state(&self.taps);
            self.run(which, 1);
            self.out.to_vec()
        };
        let v = |x: &[S]| x.iter().map(|s| s.value()).collect::<Vec<f64>>();
        snr_db(&v(&want), &v(&got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Fallback => true,
            Impl::Ours => self.side == Side::Simd,
            Impl::Liquid | Impl::Cmsis => self.external(which).is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Liquid => describe_liquid(
                "resamp2_rrrf_decim_execute (dotprod_rrrf)",
                true,
                self.side == Side::Simd,
            ),
            Impl::Cmsis => describe_cmsis(
                "arm_fir_q15 (24 taps) on the even samples, arm_scale_q15 (centre tap) on the odd samples and arm_add_q15 (a composition: CMSIS-DSP has no halfband decimator)",
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// real_to_complex

type R2cState<T> = RealToComplexState<T, Vec<<T as ComplexSample>::Real>, Vec<T>>;

#[inline(never)]
fn real_to_complex<T: ComplexSample, B: RealToComplex<T>>(
    backend: &B,
    state: &mut R2cState<T>,
    input: &[T::Real],
    out: &mut [T],
) {
    let _ = black_box(backend.real_to_complex(state, black_box(input), out));
}

/// `real_to_complex` for `C32` and `CQ15`.
pub(crate) struct RealToComplexK<T: HbCx>
where
    <T as ComplexSample>::Real: Scalar,
{
    input: Aligned<<T as ComplexSample>::Real>,
    out: Aligned<T>,
    taps: Vec<<T as ComplexSample>::Real>,
    state: R2cState<T>,
    side: Side,
    n: usize,
}

impl<T: HbCx> RealToComplexK<T>
where
    <T as ComplexSample>::Real: Scalar,
{
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        assert!(
            layout.n % 2 == 0,
            "real_to_complex: even block lengths only"
        );
        let mut input = Aligned::<<T as ComplexSample>::Real>::skewed(
            layout.n,
            layout.skew(std::mem::size_of::<<T as ComplexSample>::Real>()),
        );
        for s in input.iter_mut() {
            *s = <<T as ComplexSample>::Real>::noise(rng, 0.9, 0.0);
        }
        let taps = taps_of::<<T as ComplexSample>::Real>();
        RealToComplexK {
            input,
            out: Aligned::skewed(layout.n / 2, layout.skew(std::mem::size_of::<T>())),
            state: Self::state(&taps),
            taps,
            side: entry.into(),
            n: layout.n,
        }
    }

    fn state(taps: &[<T as ComplexSample>::Real]) -> R2cState<T> {
        RealToComplexState::new(taps.to_vec(), vec![T::default(); HB_TAPS - 1])
            .expect("halfband taps")
    }

    fn external(&self, which: Impl) -> Option<Ext<T>> {
        match which {
            // liquid-dsp's firhilbf exists for `C32` only (`T::liquid_hilbert` is `None`).
            Impl::Liquid => T::liquid_hilbert(&self.taps, &self.input, self.side == Side::Simd),
            // The SIMD entry compares with the SSE2 build, the fallback entry with the generic
            // code.
            Impl::Airspy => T::airspy(&self.taps, &self.input, self.side == Side::Simd),
            _ => None,
        }
    }
}

impl<T: HbCx> Kernel for RealToComplexK<T>
where
    <T as ComplexSample>::Real: Scalar,
    Sdrk: RealToComplex<T>,
    Fallback: RealToComplex<T>,
    Reference: RealToComplex<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        let mut ext = self.external(which);
        for _ in 0..k {
            match which {
                Impl::Ours => real_to_complex(&Sdrk, &mut self.state, &self.input, &mut self.out),
                Impl::Fallback => {
                    real_to_complex(&Fallback, &mut self.state, &self.input, &mut self.out);
                }
                _ => (ext.as_mut().expect("external implementation").run)(),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let mut want = vec![T::default(); self.n / 2];
        let mut s = Self::state(&self.taps);
        Reference
            .real_to_complex(&mut s, &self.input, &mut want)
            .expect("reference");
        let got: Vec<T> = if let Some(mut e) = self.external(which) {
            (e.run)();
            (e.result)()
        } else {
            self.state = Self::state(&self.taps);
            self.run(which, 1);
            self.out.to_vec()
        };
        snr_db(&values(&want), &values(&got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Fallback => true,
            Impl::Ours => self.side == Side::Simd,
            Impl::Liquid | Impl::Airspy => self.external(which).is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Liquid => describe_liquid(
                "firhilbf_decim_execute_block (dotprod_rrrf)",
                true,
                self.side == Side::Simd,
            ),
            Impl::Airspy if <<T as ComplexSample>::Real as Scalar>::Q15 => describe_airspy(
                "iqconverter_int16 (truncating >> 15, 32-bit accumulator, the odd samples halved)",
                "",
            ),
            Impl::Airspy => describe_airspy(
                "iqconverter_float",
                if self.side == Side::Simd {
                    " (USE_SSE2)"
                } else {
                    " (generic code)"
                },
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The comparison workloads of our own kernels (information: work saved)

/// `fir_decimate` by 2 with the 47 taps of the halfband filter (zeros included): what
/// `halfband_decimate` replaces.
pub(crate) struct FirDecimateX2K<T: HbCx>
where
    <T as ComplexSample>::Real: Scalar,
{
    input: Aligned<T>,
    out: Aligned<T>,
    taps: Vec<<T as ComplexSample>::Real>,
    state: FirDecimState<T, Vec<<T as ComplexSample>::Real>, Vec<T>>,
    n: usize,
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

impl<T: HbCx> FirDecimateX2K<T>
where
    <T as ComplexSample>::Real: Scalar,
{
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        assert!(
            layout.n % 2 == 0,
            "fir_decimate_halfband: even block lengths only"
        );
        let size = std::mem::size_of::<T>();
        let mut input = Aligned::<T>::skewed(layout.n, layout.skew(size));
        for s in input.iter_mut() {
            *s = <T as SquelchSample>::noise(rng, 0.9);
        }
        let taps = taps_of::<<T as ComplexSample>::Real>();
        FirDecimateX2K {
            input,
            out: Aligned::skewed(layout.n / 2, layout.skew(size)),
            state: Self::state(&taps),
            taps,
            n: layout.n,
        }
    }

    fn state(
        taps: &[<T as ComplexSample>::Real],
    ) -> FirDecimState<T, Vec<<T as ComplexSample>::Real>, Vec<T>> {
        FirDecimState::new(taps.to_vec(), vec![T::default(); HB_TAPS - 1], 2).expect("taps")
    }
}

impl<T: HbCx> Kernel for FirDecimateX2K<T>
where
    <T as ComplexSample>::Real: Scalar,
    Sdrk: FirDecimate<T>,
    Fallback: FirDecimate<T>,
    Reference: FirDecimate<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => fir_decimate(&Sdrk, &mut self.state, &self.input, &mut self.out),
                Impl::Fallback => {
                    fir_decimate(&Fallback, &mut self.state, &self.input, &mut self.out);
                }
                other => unreachable!("fir_decimate by 2 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.state = Self::state(&self.taps);
        self.run(which, 1);
        let mut s = Self::state(&self.taps);
        let mut want = vec![T::default(); self.n / 2];
        Reference
            .fir_decimate(&mut s, &self.input, &mut want)
            .expect("reference");
        snr_db(&values(&want), &values(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        if which == Impl::Fallback {
            "sdrk fallback"
        } else {
            "sdrk"
        }
    }
}

/// The shift by fs / 4 as a separate pass, then `halfband_decimate` on the complex signal: what
/// `real_to_complex` replaces (`u[n] = x[n] (-j)^n`, the half of the components being zero).
pub(crate) struct ShiftedK<T: HbCx>
where
    <T as ComplexSample>::Real: Scalar,
{
    input: Aligned<<T as ComplexSample>::Real>,
    shifted: Aligned<T>,
    out: Aligned<T>,
    taps: Vec<<T as ComplexSample>::Real>,
    state: CxState<T>,
    n: usize,
}

/// The negation of the shift: wrapping is wrong for Q15 (`-(-32768)`), so it saturates.
trait Negate: Sized {
    fn neg_sat(self) -> Self;
}

impl Negate for f32 {
    fn neg_sat(self) -> f32 {
        -self
    }
}

impl Negate for Q15 {
    fn neg_sat(self) -> Q15 {
        Q15(self.0.saturating_neg())
    }
}

#[inline(never)]
fn shift_quarter<T: ComplexSample>(x: &[T::Real], u: &mut [T])
where
    T::Real: Negate + Copy,
{
    let zero = T::Real::from_f64_round(0.0);
    for (i, (v, d)) in x.iter().zip(u.iter_mut()).enumerate() {
        *d = match i & 3 {
            0 => T::new(*v, zero),
            1 => T::new(zero, v.neg_sat()),
            2 => T::new(v.neg_sat(), zero),
            _ => T::new(zero, *v),
        };
    }
}

impl<T: HbCx> ShiftedK<T>
where
    <T as ComplexSample>::Real: Scalar,
{
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        assert!(
            layout.n % 2 == 0,
            "halfband_decimate_shifted: even block lengths only"
        );
        let mut input = Aligned::<<T as ComplexSample>::Real>::skewed(
            layout.n,
            layout.skew(std::mem::size_of::<<T as ComplexSample>::Real>()),
        );
        for s in input.iter_mut() {
            *s = <<T as ComplexSample>::Real>::noise(rng, 0.9, 0.0);
        }
        let taps = taps_of::<<T as ComplexSample>::Real>();
        let size = std::mem::size_of::<T>();
        ShiftedK {
            input,
            shifted: Aligned::skewed(layout.n, layout.skew(size)),
            out: Aligned::skewed(layout.n / 2, layout.skew(size)),
            state: DecimateK::<T>::state(&taps),
            taps,
            n: layout.n,
        }
    }
}

impl<T: HbCx> Kernel for ShiftedK<T>
where
    <T as ComplexSample>::Real: Scalar + Negate,
    Sdrk: HalfbandDecimate<T>,
    Fallback: HalfbandDecimate<T>,
    Reference: RealToComplex<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            shift_quarter::<T>(black_box(&self.input), &mut self.shifted);
            match which {
                Impl::Ours => {
                    halfband_decimate(&Sdrk, &mut self.state, &self.shifted, &mut self.out);
                }
                Impl::Fallback => {
                    halfband_decimate(&Fallback, &mut self.state, &self.shifted, &mut self.out);
                }
                other => unreachable!("the shifted halfband has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.state = DecimateK::<T>::state(&self.taps);
        self.run(which, 1);
        let mut s = RealToComplexK::<T>::state(&self.taps);
        let mut want = vec![T::default(); self.n / 2];
        Reference
            .real_to_complex(&mut s, &self.input, &mut want)
            .expect("reference");
        snr_db(&values(&want), &values(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        if which == Impl::Fallback {
            "sdrk fallback"
        } else {
            "sdrk"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Impl, Layout, prepare};

    /// Every implementation an entry supports runs and agrees with the `f64` model.
    #[test]
    fn the_halfband_entries_run_every_implementation_they_support() {
        for (name, floor) in [
            ("halfband_decimate", 100.0),
            ("halfband_decimate_cq15", 60.0),
            ("halfband_decimate_real", 100.0),
            ("halfband_decimate_real_q15", 60.0),
            ("real_to_complex", 100.0),
            ("real_to_complex_cq15", 60.0),
            ("halfband_decimate_fallback", 100.0),
            ("halfband_decimate_cq15_fallback", 60.0),
            ("halfband_decimate_real_fallback", 100.0),
            ("halfband_decimate_real_q15_fallback", 60.0),
            ("real_to_complex_fallback", 100.0),
            ("real_to_complex_cq15_fallback", 60.0),
            ("halfband_ref_fir_decimate", 100.0),
            ("halfband_ref_fir_decimate_cq15", 60.0),
            ("halfband_ref_shifted", 100.0),
            ("halfband_ref_shifted_cq15", 60.0),
        ] {
            for misaligned in [false, true] {
                let mut kernel = prepare(name, Layout { n: 512, misaligned }).expect(name);
                for which in Impl::ALL {
                    if kernel.supports(which) {
                        kernel.run(which, 2);
                        let snr = kernel.accuracy(which);
                        assert!(snr > floor, "{name} {}: {snr} dB", which.name());
                    }
                }
            }
        }
    }
}
