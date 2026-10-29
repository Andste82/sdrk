//! The FIR family entries that compare with CMSIS-DSP (round 2 of issue #20) and the `CQ15`
//! entries of the FIR family.
//!
//! - `fir_cq15`, `fir_decimate_cq15`, `fir_interpolate_cq15`, `resample_cq15`: our kernel
//!   through the facade against our fallback. No SIMD library has a `CQ15` FIR with real `Q15`
//!   taps on our levels (VOLK's 16-bit dot product takes complex taps and saturates in 16 bits),
//!   so they are information and the branch is judged by its efficiency (D32).
//! - `<kernel>_fallback` of `fir`, `fir_decimate`, `fir_interpolate`, `resample` (C32 and CQ15),
//!   `fir_interpolate_real` and `resample_real` (f32 and Q15): our fallback against CMSIS-DSP
//!   1.18.0 generic C (`arm_fir_*`, D16), compiled for the level.
//!
//! The work of CMSIS-DSP, per output sample:
//!
//! - complex signals: CMSIS-DSP filters real signals, so it runs twice, on the planar `re` and
//!   `im` of the same input (the same filter with real taps on I and Q, the same multiplies and
//!   adds); the deinterleaving is done once at preparation and not counted (the best case for
//!   CMSIS-DSP);
//! - `fir`: `arm_fir_*` on the `n` input samples (`n` outputs);
//! - `fir_decimate` by 10: `arm_fir_decimate_*` on the largest multiple of 10 not above `n`
//!   (409 outputs for 4096 input samples, ours produces 410: CMSIS-DSP does 0.2 % less);
//! - `fir_interpolate` by 2: `arm_fir_interpolate_*` by 2 on `n / 2` samples (`n` outputs, the
//!   64-tap branch dot product per output of ours);
//! - `resample` by 3/2: CMSIS-DSP has no rational resampler. `arm_fir_interpolate_*` by 3 on
//!   `n / 3` samples gives `n - 1` outputs of the same 192 taps, each a 64-tap branch dot
//!   product, the work of one output of ours; the selection of every second output (ours
//!   skips the others) is not part of it. The comparison is per output sample.

use std::hint::black_box;

use cref_sys::cmsis_dsp::{
    CmsisFirDecimateF32, CmsisFirDecimateQ15, CmsisFirF32, CmsisFirInterpolateF32,
    CmsisFirInterpolateQ15, CmsisFirQ15,
};
use sdrk::kernels::decim::{FirDecimState, ResampleRealState, ResampleState};
use sdrk::kernels::filter::FirState;
use sdrk::kernels::interp::{FirInterpRealState, FirInterpState};
use sdrk::{C32, CQ15, Fallback, Q15, Sdrk};
use sdrk_core::Reference;
use sdrk_core::kernels::decim::{Resample, ResampleReal};

use super::{
    Aligned, DECIM, DECIM_TAPS, FIR_TAPS, INTERP, INTERP_TAPS, Impl, Kernel, Layout, Pcg32,
    RESAMPLE_DOWN, RESAMPLE_TAPS, RESAMPLE_UP, RealNoise, SquelchSample, describe_cmsis, fir,
    fir_decimate, fir_interpolate, fir_interpolate_real, resample, resample_real, snr_db, taps,
};

/// The filter of a measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// `fir`, 64 taps.
    Fir,
    /// `fir_decimate` by 10, 127 taps.
    Decimate,
    /// `fir_interpolate` by 2, 128 taps (64 per branch).
    Interpolate,
    /// `resample` by 3/2, 192 taps (64 per branch).
    Resample,
}

impl Mode {
    fn taps(self) -> usize {
        match self {
            Mode::Fir => FIR_TAPS,
            Mode::Decimate => DECIM_TAPS,
            Mode::Interpolate => INTERP_TAPS,
            Mode::Resample => RESAMPLE_TAPS,
        }
    }

    /// `(L, M)`.
    fn ratio(self) -> (usize, usize) {
        match self {
            Mode::Fir => (1, 1),
            Mode::Decimate => (1, DECIM),
            Mode::Interpolate => (INTERP, 1),
            Mode::Resample => (RESAMPLE_UP, RESAMPLE_DOWN),
        }
    }

    /// Samples in front of the input in the buffer (as the VOLK entries keep them).
    fn lead(self) -> usize {
        let (up, _) = self.ratio();
        self.taps() / up - 1
    }

    /// Input samples and outputs of our call for `n`.
    fn ours(self, n: usize) -> (usize, usize) {
        match self {
            Mode::Fir => (n, n),
            Mode::Decimate => (n, n.div_ceil(DECIM)),
            // Enough input for `n` outputs from any phase counter.
            Mode::Interpolate | Mode::Resample => {
                let (up, down) = self.ratio();
                (n * down / up + 2, n)
            }
        }
    }

    /// Input samples and outputs of one CMSIS-DSP call for `n` (see the module documentation).
    fn cmsis(self, n: usize) -> (usize, usize) {
        match self {
            Mode::Fir => (n, n),
            Mode::Decimate => (n - n % DECIM, n / DECIM),
            Mode::Interpolate | Mode::Resample => {
                let (up, _) = self.ratio();
                (n / up, n / up * up)
            }
        }
    }

    /// `(L, M)` of what CMSIS-DSP computes: the resampler's interpolation without the
    /// decimation.
    fn cmsis_ratio(self) -> (usize, usize) {
        match self {
            Mode::Resample => (RESAMPLE_UP, 1),
            other => other.ratio(),
        }
    }

    fn cmsis_name(self, q15: bool) -> &'static str {
        match (self, q15) {
            (Mode::Fir, false) => "arm_fir_f32",
            (Mode::Fir, true) => "arm_fir_q15",
            (Mode::Decimate, false) => "arm_fir_decimate_f32 (block 4090 of 4096)",
            (Mode::Decimate, true) => "arm_fir_decimate_q15 (block 4090 of 4096)",
            (Mode::Interpolate, false) => "arm_fir_interpolate_f32 by 2",
            (Mode::Interpolate, true) => "arm_fir_interpolate_q15 by 2",
            (Mode::Resample, false) => {
                "arm_fir_interpolate_f32 by 3, the same outputs per sample (no rational resampler)"
            }
            (Mode::Resample, true) => {
                "arm_fir_interpolate_q15 by 3, the same outputs per sample (no rational resampler)"
            }
        }
    }
}

/// A CMSIS-DSP filter on one real channel.
type CmsisRun<R> = Box<dyn FnMut(&[R], &mut [R])>;

/// The sample types of CMSIS-DSP: `f32` and Q15 as `i16`.
pub(crate) trait CmsisRaw: Copy + Default + 'static {
    /// The CMSIS-DSP filter of `mode` with `taps` (SDRk's order) for blocks of `max_block`.
    fn filter(mode: Mode, taps: &[Self], max_block: usize) -> Option<CmsisRun<Self>>;
}

impl CmsisRaw for f32 {
    fn filter(mode: Mode, taps: &[f32], max_block: usize) -> Option<CmsisRun<f32>> {
        let (up, _) = mode.ratio();
        Some(match mode {
            Mode::Fir => {
                let mut f = CmsisFirF32::new(taps, max_block)?;
                Box::new(move |i: &[f32], o: &mut [f32]| f.run(i, o))
            }
            Mode::Decimate => {
                let mut f = CmsisFirDecimateF32::new(taps, DECIM, max_block)?;
                Box::new(move |i: &[f32], o: &mut [f32]| f.run(i, o))
            }
            Mode::Interpolate | Mode::Resample => {
                let mut f = CmsisFirInterpolateF32::new(taps, up, max_block)?;
                Box::new(move |i: &[f32], o: &mut [f32]| f.run(i, o))
            }
        })
    }
}

impl CmsisRaw for i16 {
    fn filter(mode: Mode, taps: &[i16], max_block: usize) -> Option<CmsisRun<i16>> {
        let (up, _) = mode.ratio();
        Some(match mode {
            Mode::Fir => {
                let mut f = CmsisFirQ15::new(taps, max_block)?;
                Box::new(move |i: &[i16], o: &mut [i16]| f.run(i, o))
            }
            Mode::Decimate => {
                let mut f = CmsisFirDecimateQ15::new(taps, DECIM, max_block)?;
                Box::new(move |i: &[i16], o: &mut [i16]| f.run(i, o))
            }
            Mode::Interpolate | Mode::Resample => {
                let mut f = CmsisFirInterpolateQ15::new(taps, up, max_block)?;
                Box::new(move |i: &[i16], o: &mut [i16]| f.run(i, o))
            }
        })
    }
}

/// Our implementations.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Backend {
    Ours,
    Fallback,
}

/// A sample type of the FIR family: `C32`, `CQ15` (complex), `f32`, `Q15` (real).
pub(crate) trait FamSample: Copy + Default + 'static {
    /// Tap type.
    type Tap: Copy;
    /// CMSIS-DSP's sample type.
    type Raw: CmsisRaw;
    /// The state of one of our filters.
    type State;
    /// Channels CMSIS-DSP runs (2 for complex samples).
    const CHANNELS: usize;
    /// Taps, then the buffer of `len` samples, drawn as the existing entries of the type do.
    fn draw(
        rng: &mut Pcg32,
        n_taps: usize,
        len: usize,
        skew: usize,
    ) -> (Vec<Self::Tap>, Aligned<Self>);
    fn state(mode: Mode, taps: &[Self::Tap], up: usize, down: usize) -> Self::State;
    fn run(state: &mut Self::State, backend: Backend, input: &[Self], out: &mut [Self]);
    /// The reference resampler by `up / down` from zero history; returns the outputs.
    fn reference(taps: &[Self::Tap], up: usize, down: usize, input: &[Self], n: usize)
    -> Vec<Self>;
    fn raw_taps(taps: &[Self::Tap]) -> Vec<Self::Raw>;
    /// Channel `c` of the sample.
    fn channel(self, c: usize) -> Self::Raw;
    fn join(re: Self::Raw, im: Self::Raw) -> Self;
    fn values(x: &[Self]) -> Vec<f64>;
}

/// Our complex states.
pub(crate) enum CxState<T: sdrk::ComplexSample> {
    Fir(FirState<T, Vec<T::Real>, Vec<T>>),
    Decimate(FirDecimState<T, Vec<T::Real>, Vec<T>>),
    Interpolate(FirInterpState<T, Vec<T::Real>, Vec<T>>),
    Resample(ResampleState<T, Vec<T::Real>, Vec<T>>),
}

/// Our real states (the real kernels have no `fir` and `fir_decimate`).
pub(crate) enum RealState<T: sdrk_core::RealSample> {
    Interpolate(FirInterpRealState<T, Vec<T>, Vec<T>>),
    Resample(ResampleRealState<T, Vec<T>, Vec<T>>),
}

macro_rules! complex_sample {
    ($t:ty, $tap:ty, $raw:ty, $draw_taps:expr, $raw_tap:expr, $part:expr, $join:expr, $value:expr) => {
        impl FamSample for $t {
            type Tap = $tap;
            type Raw = $raw;
            type State = CxState<$t>;
            const CHANNELS: usize = 2;

            fn draw(
                rng: &mut Pcg32,
                n_taps: usize,
                len: usize,
                skew: usize,
            ) -> (Vec<$tap>, Aligned<$t>) {
                let taps: Vec<$tap> = ($draw_taps)(rng, n_taps);
                let mut buf = Aligned::<$t>::skewed(len, skew);
                for s in buf.iter_mut() {
                    *s = <$t as SquelchSample>::noise(rng, 0.9);
                }
                (taps, buf)
            }

            fn state(mode: Mode, taps: &[$tap], up: usize, down: usize) -> CxState<$t> {
                let n = taps.len();
                let t = taps.to_vec();
                let z = <$t>::default();
                match mode {
                    Mode::Fir => CxState::Fir(FirState::new(t, vec![z; n - 1]).expect("taps")),
                    Mode::Decimate => CxState::Decimate(
                        FirDecimState::new(t, vec![z; n - 1], down).expect("taps"),
                    ),
                    Mode::Interpolate => CxState::Interpolate(
                        FirInterpState::new(t, vec![z; n.div_ceil(up)], up).expect("taps"),
                    ),
                    Mode::Resample => CxState::Resample(
                        ResampleState::new(t, vec![z; n.div_ceil(up)], up, down).expect("taps"),
                    ),
                }
            }

            fn run(state: &mut CxState<$t>, backend: Backend, input: &[$t], out: &mut [$t]) {
                match (state, backend) {
                    (CxState::Fir(s), Backend::Ours) => fir(&Sdrk, s, input, out),
                    (CxState::Fir(s), Backend::Fallback) => fir(&Fallback, s, input, out),
                    (CxState::Decimate(s), Backend::Ours) => fir_decimate(&Sdrk, s, input, out),
                    (CxState::Decimate(s), Backend::Fallback) => {
                        fir_decimate(&Fallback, s, input, out);
                    }
                    (CxState::Interpolate(s), Backend::Ours) => {
                        fir_interpolate(&Sdrk, s, input, out);
                    }
                    (CxState::Interpolate(s), Backend::Fallback) => {
                        fir_interpolate(&Fallback, s, input, out);
                    }
                    (CxState::Resample(s), Backend::Ours) => resample(&Sdrk, s, input, out),
                    (CxState::Resample(s), Backend::Fallback) => {
                        resample(&Fallback, s, input, out);
                    }
                }
            }

            fn reference(taps: &[$tap], up: usize, down: usize, input: &[$t], n: usize) -> Vec<$t> {
                let hist = vec![<$t>::default(); taps.len().div_ceil(up)];
                let mut s = ResampleState::new(taps.to_vec(), hist, up, down).expect("taps");
                let mut want = vec![<$t>::default(); n];
                let p = Reference
                    .resample(&mut s, input, &mut want)
                    .expect("reference");
                want.truncate(p.produced);
                want
            }

            fn raw_taps(taps: &[$tap]) -> Vec<$raw> {
                taps.iter().map($raw_tap).collect()
            }

            fn channel(self, c: usize) -> $raw {
                ($part)(self, c)
            }

            fn join(re: $raw, im: $raw) -> $t {
                ($join)(re, im)
            }

            fn values(x: &[$t]) -> Vec<f64> {
                x.iter().flat_map($value).collect()
            }
        }
    };
}

complex_sample!(
    C32,
    f32,
    f32,
    |rng: &mut Pcg32, n| taps(rng, n),
    |h: &f32| *h,
    |x: C32, c| if c == 0 { x.re } else { x.im },
    C32::new,
    |v: &C32| [f64::from(v.re), f64::from(v.im)]
);

complex_sample!(
    CQ15,
    Q15,
    i16,
    |rng: &mut Pcg32, n| {
        taps(rng, n)
            .into_iter()
            .map(|h| Q15((h * 32768.0).round() as i16))
            .collect()
    },
    |h: &Q15| h.0,
    |x: CQ15, c| if c == 0 { x.re.0 } else { x.im.0 },
    |re, im| CQ15 {
        re: Q15(re),
        im: Q15(im)
    },
    |v: &CQ15| [f64::from(v.re.0) / 32768.0, f64::from(v.im.0) / 32768.0]
);

macro_rules! real_sample {
    ($t:ty, $raw:ty, $raw_tap:expr, $part:expr, $join:expr) => {
        impl FamSample for $t {
            type Tap = $t;
            type Raw = $raw;
            type State = RealState<$t>;
            const CHANNELS: usize = 1;

            fn draw(
                rng: &mut Pcg32,
                n_taps: usize,
                len: usize,
                skew: usize,
            ) -> (Vec<$t>, Aligned<$t>) {
                // As `PolyRealK` draws them.
                let scale = 1.0 / (n_taps as f32).sqrt();
                let taps: Vec<$t> = (0..n_taps)
                    .map(|_| <$t as RealNoise>::noise(rng, scale, 0.0))
                    .collect();
                let mut buf = Aligned::<$t>::skewed(len, skew);
                for s in buf.iter_mut() {
                    *s = <$t as RealNoise>::noise(rng, 0.9, 0.0);
                }
                (taps, buf)
            }

            fn state(mode: Mode, taps: &[$t], up: usize, down: usize) -> RealState<$t> {
                let hist = vec![<$t>::default(); taps.len().div_ceil(up)];
                match mode {
                    Mode::Interpolate => RealState::Interpolate(
                        FirInterpRealState::new(taps.to_vec(), hist, up).expect("taps"),
                    ),
                    Mode::Resample => RealState::Resample(
                        ResampleRealState::new(taps.to_vec(), hist, up, down).expect("taps"),
                    ),
                    other => unreachable!("no real {other:?}"),
                }
            }

            fn run(state: &mut RealState<$t>, backend: Backend, input: &[$t], out: &mut [$t]) {
                match (state, backend) {
                    (RealState::Interpolate(s), Backend::Ours) => {
                        fir_interpolate_real(&Sdrk, s, input, out);
                    }
                    (RealState::Interpolate(s), Backend::Fallback) => {
                        fir_interpolate_real(&Fallback, s, input, out);
                    }
                    (RealState::Resample(s), Backend::Ours) => {
                        resample_real(&Sdrk, s, input, out);
                    }
                    (RealState::Resample(s), Backend::Fallback) => {
                        resample_real(&Fallback, s, input, out);
                    }
                }
            }

            fn reference(taps: &[$t], up: usize, down: usize, input: &[$t], n: usize) -> Vec<$t> {
                let hist = vec![<$t>::default(); taps.len().div_ceil(up)];
                let mut s = ResampleRealState::new(taps.to_vec(), hist, up, down).expect("taps");
                let mut want = vec![<$t>::default(); n];
                let p = Reference
                    .resample_real(&mut s, input, &mut want)
                    .expect("reference");
                want.truncate(p.produced);
                want
            }

            fn raw_taps(taps: &[$t]) -> Vec<$raw> {
                taps.iter().map($raw_tap).collect()
            }

            fn channel(self, _c: usize) -> $raw {
                ($part)(self)
            }

            fn join(re: $raw, _im: $raw) -> $t {
                ($join)(re)
            }

            fn values(x: &[$t]) -> Vec<f64> {
                x.iter().map(|v| v.as_f64()).collect()
            }
        }
    };
}

real_sample!(f32, f32, |h: &f32| *h, |x: f32| x, |x: f32| x);
real_sample!(Q15, i16, |h: &Q15| h.0, |x: Q15| x.0, Q15);

/// Which side of the family an entry measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Entry {
    /// Our kernel through the facade against our fallback (`info`).
    Simd,
    /// Our fallback against CMSIS-DSP.
    Fallback,
}

/// A FIR family measurement (see the module documentation).
pub(crate) struct FirFamK<T: FamSample> {
    /// [`Mode::lead`] samples in front of the input, then the input.
    buf: Aligned<T>,
    taps: Vec<T::Tap>,
    out: Aligned<T>,
    state: T::State,
    mode: Mode,
    entry: Entry,
    /// The planar channels of the CMSIS-DSP input, and its outputs.
    planar_in: Vec<Aligned<T::Raw>>,
    planar_out: Vec<Aligned<T::Raw>>,
}

impl<T: FamSample> FirFamK<T> {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, mode: Mode, entry: Entry) -> Self {
        let size = std::mem::size_of::<T>();
        let (input, outputs) = mode.ours(layout.n);
        let lead = mode.lead();
        let (taps, buf) = T::draw(
            rng,
            mode.taps(),
            input + lead,
            layout.skew_before(size, lead),
        );
        let (up, down) = mode.ratio();
        let (cm_in, cm_out) = mode.cmsis(layout.n);
        let raw = std::mem::size_of::<T::Raw>();
        let planar_in = (0..T::CHANNELS)
            .map(|c| {
                let mut v = Aligned::<T::Raw>::skewed(cm_in, layout.skew(raw));
                for (d, s) in v.iter_mut().zip(&buf[lead..]) {
                    *d = s.channel(c);
                }
                v
            })
            .collect();
        let planar_out = (0..T::CHANNELS)
            .map(|_| Aligned::<T::Raw>::skewed(cm_out, layout.skew(raw)))
            .collect();
        FirFamK {
            state: T::state(mode, &taps, up, down),
            buf,
            taps,
            out: Aligned::skewed(outputs, layout.skew(size)),
            mode,
            entry,
            planar_in,
            planar_out,
        }
    }

    fn cmsis_filters(&self) -> Option<Vec<CmsisRun<T::Raw>>> {
        let taps = T::raw_taps(&self.taps);
        let block = self.planar_in.first()?.len().max(1);
        (0..T::CHANNELS)
            .map(|_| T::Raw::filter(self.mode, &taps, block))
            .collect()
    }
}

impl<T: FamSample> Kernel for FirFamK<T> {
    fn run(&mut self, which: Impl, k: usize) {
        let mut cmsis = (which == Impl::Cmsis).then(|| self.cmsis_filters().expect("cmsis"));
        let input = &self.buf[self.mode.lead()..];
        for _ in 0..k {
            match which {
                Impl::Ours => T::run(&mut self.state, Backend::Ours, input, &mut self.out),
                Impl::Fallback => T::run(&mut self.state, Backend::Fallback, input, &mut self.out),
                Impl::Cmsis => {
                    let filters = cmsis.as_mut().expect("cmsis");
                    for ((f, i), o) in filters
                        .iter_mut()
                        .zip(&self.planar_in)
                        .zip(&mut self.planar_out)
                    {
                        f(i, o);
                        black_box(&*o);
                    }
                }
                other => unreachable!("the FIR family has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        let lead = self.mode.lead();
        if which == Impl::Cmsis {
            self.run(which, 1);
            let (up, down) = self.mode.cmsis_ratio();
            let got: Vec<T> = (0..self.planar_out[0].len())
                .map(|j| {
                    let re = self.planar_out[0][j];
                    let im = self.planar_out.get(1).map_or(re, |o| o[j]);
                    T::join(re, im)
                })
                .collect();
            let input = &self.buf[lead..lead + self.planar_in[0].len()];
            let want = T::reference(&self.taps, up, down, input, got.len());
            let len = want.len().min(got.len());
            return snr_db(&T::values(&want[..len]), &T::values(&got[..len]));
        }
        let (up, down) = self.mode.ratio();
        self.state = T::state(self.mode, &self.taps, up, down);
        self.run(which, 1);
        let want = T::reference(&self.taps, up, down, &self.buf[lead..], self.out.len());
        let len = want.len().min(self.out.len());
        snr_db(&T::values(&want[..len]), &T::values(&self.out[..len]))
    }

    fn supports(&self, which: Impl) -> bool {
        match (self.entry, which) {
            (_, Impl::Fallback) | (Entry::Simd, Impl::Ours) => true,
            (Entry::Fallback, Impl::Cmsis) => self.cmsis_filters().is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis(self.mode.cmsis_name(is_q15::<T>())),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

/// Whether CMSIS-DSP runs its Q15 functions for `T`.
fn is_q15<T: FamSample>() -> bool {
    std::any::TypeId::of::<T::Raw>() == std::any::TypeId::of::<i16>()
}

#[cfg(test)]
mod tests {
    use super::super::{Impl, Layout, prepare};

    #[test]
    fn the_fir_family_entries_run_every_implementation_they_support() {
        for name in [
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
        ] {
            for misaligned in [false, true] {
                let mut kernel = prepare(name, Layout { n: 512, misaligned }).expect(name);
                for which in Impl::ALL {
                    if kernel.supports(which) {
                        kernel.run(which, 2);
                        let snr = kernel.accuracy(which);
                        assert!(snr > 60.0, "{name} {}: {snr} dB", which.name());
                    }
                }
            }
        }
    }
}
