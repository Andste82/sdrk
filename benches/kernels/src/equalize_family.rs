//! The entries of milestone M43: the complex dot product `dot_complex` (`C32`) and the adaptive
//! equalizer `equalize` (`C32`).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `dot_complex` | facade | VOLK `volk_32fc_x2_dot_prod_32fc`; at `avx2`, `avx512` and `neon` its protokernel without FMA | VOLK's fused dispatch at those levels |
//! | `dot_complex_fallback` | our fallback | CMSIS-DSP `arm_cmplx_dot_prod_f32` (generic C) | — |
//! | `equalize` | facade | none (`info`) | our fallback |
//! | `equalize_fallback` | our fallback | GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` | — |
//!
//! `dot_complex` is the bit-exact complex dot product of D800 and D801 (eight partial sums per
//! component, no fused multiply-add). VOLK's `volk_32fc_x2_dot_prod_32fc` does the same work
//! (`sum a[i] b[i]`, not conjugated); at the levels where VOLK's dispatcher fuses (`a_avx_fma`,
//! `neonv8`) the gate is VOLK's protokernel without fused operations (`u_avx`, `neon`,
//! `benches/refs/volk-nofma.c`), as for `mul` under D287, because the contract of ours forbids
//! the fusion (D821). One call of `n` samples per run, as `dot`.
//!
//! `equalize` runs the SIMD `dot_complex` for its filter part and the scalar weight update of
//! `sdrk-core` (D803). The facade entry is counted against our own fallback as information.
//! `equalize_fallback` is counted against the loop of GNU Radio's `linear_equalizer` with
//! `adaptive_algorithm_nlms` (`benches/refs/gr-equalizer.c`, D838): per symbol the same work in
//! decision mode, an FIR of 11 complex taps (VOLK's generic `volk_32fc_x2_dot_prod_32fc`), the
//! QPSK decision, the power of the window (VOLK's generic conjugate dot product) and the
//! normalized update of the 11 taps; ours adds the phase tracker (one interpolated table pair,
//! two rotations, the loop filter) and the smoothed `mse`, GNU Radio the copy of every tap into
//! its aligned filter taps.
//!
//! The `equalize` workload is the block default (11 taps, no feedback, QPSK, `mu_cma` 0.05,
//! `mu_dd` 0.02, `cma_symbols` 1000, `phase_bw` 0.02) on QPSK symbols through the channel
//! `1 + 0.3 e^(j0.6) z^-1 + 0.1 e^(-j2) z^-2` with a constant carrier phase of 0.7 rad and
//! uniform noise of 0.05 per component, one symbol per sample. The state lives across the runs:
//! the warm-up call covers the 1000 symbols in CMA mode, so the counted calls run in decision
//! mode, the steady state. GNU Radio runs the same input with the same normalized step 0.02
//! from the centre-spike taps (its QPSK points are `+-sqrt 2 +-j sqrt 2`, so its taps learn the
//! gain 2 as well; the arithmetic is the same).

use std::hint::black_box;

use cref_sys::{GrLinearEqualizer, cmsis_dsp};
use sdrk::kernels::dot::DotComplex;
use sdrk::kernels::equalize::{EqParams, Equalize, EqualizeState};
use sdrk::{C32, Fallback, Sdrk};
use sdrk_core::Reference;

use super::{
    Aligned, BiquadEntry, Impl, Kernel, Layout, Pcg32, c32_pairs, c32_pairs_ref, c32s,
    describe_cmsis, dot_windows, noise_c32, snr_db,
};

// ---------------------------------------------------------------------------------------------
// dot_complex

#[inline(never)]
fn dot_complex<B: DotComplex<C32>>(backend: &B, x: &[C32], w: &[C32]) -> C32 {
    black_box(backend.dot_complex(black_box(x), black_box(w))).unwrap_or_default()
}

/// `dot_complex` of `n` `C32` samples with `n` complex weights, one call per run.
///
/// `dot_complex`: VOLK's `volk_32fc_x2_dot_prod_32fc` through its dispatcher (which checks the
/// alignment), and at `avx2`, `avx512` and `neon` its protokernel without FMA.
/// `dot_complex_fallback`: our fallback against CMSIS-DSP's `arm_cmplx_dot_prod_f32` on the
/// same interleaved buffers.
pub(crate) struct DotComplexK {
    x: Aligned<C32>,
    w: Aligned<C32>,
    result: Aligned<C32>,
    entry: BiquadEntry,
}

impl DotComplexK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let x = noise_c32(rng, layout.n, 0.9, layout.skew(8));
        // Weights of `1 / sqrt(n)` per component keep the sum near the size of one sample.
        let w = noise_c32(
            rng,
            layout.n,
            1.0 / (layout.n.max(1) as f32).sqrt(),
            layout.skew(8),
        );
        DotComplexK {
            x,
            w,
            result: Aligned::new(1),
            entry,
        }
    }

    fn call(&mut self, which: Impl, start: usize, len: usize) -> C32 {
        let (x, w) = (&self.x[start..start + len], &self.w[start..start + len]);
        match which {
            Impl::Ours => dot_complex(&Sdrk, x, w),
            Impl::Fallback => dot_complex(&Fallback, x, w),
            // SAFETY: `x` and `w` hold `len` elements, `result` one.
            Impl::Volk => unsafe {
                volk_sys::volk_32fc_x2_dot_prod_32fc(
                    self.result.as_mut_ptr().cast(),
                    x.as_ptr().cast(),
                    w.as_ptr().cast(),
                    len as u32,
                );
                self.result[0]
            },
            Impl::VolkNoFma => {
                let [re, im] =
                    cref_sys::volk_nofma::dot_prod_32fc(c32_pairs_ref(x), c32_pairs_ref(w))
                        .expect("volk-nofma");
                black_box(C32::new(re, im))
            }
            Impl::Cmsis => {
                let (mut re, mut im) = (0.0f32, 0.0f32);
                // SAFETY: `x` and `w` hold `len` interleaved complex samples; the results are
                // locals.
                unsafe {
                    cmsis_dsp::ffi::arm_cmplx_dot_prod_f32(
                        x.as_ptr().cast(),
                        w.as_ptr().cast(),
                        len as u32,
                        &raw mut re,
                        &raw mut im,
                    );
                }
                black_box(C32::new(re, im))
            }
            other => unreachable!("dot_complex has no {other:?}"),
        }
    }
}

impl Kernel for DotComplexK {
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
            for (a, b) in self.x[s..s + len].iter().zip(&self.w[s..s + len]) {
                let (ar, ai) = (f64::from(a.re), f64::from(a.im));
                let (br, bi) = (f64::from(b.re), f64::from(b.im));
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            want.extend([re, im]);
            got.extend([f64::from(y.re), f64::from(y.im)]);
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        match (self.entry, which) {
            (_, Impl::Fallback) | (BiquadEntry::Simd, Impl::Ours) => true,
            (BiquadEntry::Simd, Impl::Volk) => volk_sys::FULL,
            (BiquadEntry::Simd, Impl::VolkNoFma) => cref_sys::volk_nofma::AVAILABLE,
            (BiquadEntry::Fallback, Impl::Cmsis) => cmsis_dsp::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_32fc_x2_dot_prod_32fc",
            Impl::VolkNoFma => match cref_sys::volk_nofma::protokernel() {
                Some("u_avx") => {
                    "VOLK 3.3.0 volk_32fc_x2_dot_prod_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off)"
                }
                Some("neon") => {
                    "VOLK 3.3.0 volk_32fc_x2_dot_prod_32fc_neon without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off)"
                }
                _ => {
                    "VOLK 3.3.0 volk_32fc_x2_dot_prod_32fc without FMA (benches/refs/volk-nofma.c)"
                }
            },
            Impl::Cmsis => describe_cmsis("arm_cmplx_dot_prod_f32"),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if self.entry == BiquadEntry::Simd {
            &["volk_32fc_x2_dot_prod_32fc"]
        } else {
            &[]
        }
    }
}

// ---------------------------------------------------------------------------------------------
// equalize

/// The taps of the block default (M43 §2; the other parameters are those of
/// `EqParams::default`).
const EQ_TAPS: usize = 11;
/// The carrier phase (rad) and the uniform noise per component of the workload.
const EQ_PHASE: f64 = 0.7;
const EQ_NOISE: f64 = 0.05;
/// The channel: `(delay in symbols, magnitude, phase)`.
const EQ_CHANNEL: [(usize, f64, f64); 3] = [(0, 1.0, 0.0), (1, 0.3, 0.6), (2, 0.1, -2.0)];

type EqState = EqualizeState<[C32; EQ_TAPS], [C32; EQ_TAPS]>;

fn eq_state() -> EqState {
    EqualizeState::new(
        [C32::default(); EQ_TAPS],
        [C32::default(); EQ_TAPS],
        EqParams::default(),
    )
    .expect("equalize")
}

/// The centre-spike taps of GNU Radio's equalizer (natural order).
fn gr_taps() -> [[f32; 2]; EQ_TAPS] {
    let mut t = [[0.0f32; 2]; EQ_TAPS];
    t[EQ_TAPS / 2] = [1.0, 0.0];
    t
}

/// GNU Radio's `linear_equalizer` for `n` symbols per call with the normalized decision-mode
/// step of ours.
fn gr_equalizer(n: usize) -> Option<GrLinearEqualizer> {
    GrLinearEqualizer::new(&gr_taps(), EqParams::default().mu_dd, n)
}

/// The f64 model of GNU Radio's loop (the accuracy reference of [`Impl::GnuRadio`]): the FIR of
/// the window (oldest first) with the reversed taps, the QPSK point `+-sqrt 2 +-j sqrt 2` of the
/// signs (`> 0`), `e = d - y`, `taps[i] += (mu / sum |x|^2) conj(x[i]) e`.
fn gr_equalizer_model(input: &[C32]) -> Vec<C32> {
    let n = EQ_TAPS;
    let mu = f64::from(EqParams::default().mu_dd);
    let mut taps: Vec<(f64, f64)> = gr_taps()
        .iter()
        .rev()
        .map(|t| (f64::from(t[0]), f64::from(t[1])))
        .collect();
    let mut window = vec![(0.0f64, 0.0f64); n];
    let a = core::f64::consts::SQRT_2;
    let mut out = Vec::with_capacity(input.len());
    for x in input {
        window.rotate_left(1);
        window[n - 1] = (f64::from(x.re), f64::from(x.im));
        let (mut yr, mut yi) = (0.0, 0.0);
        for (&(xr, xi), &(tr, ti)) in window.iter().zip(&taps) {
            yr += xr * tr - xi * ti;
            yi += xr * ti + xi * tr;
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a model value rounded to f32"
        )]
        out.push(C32::new(yr as f32, yi as f32));
        let (dr, di) = (if yr > 0.0 { a } else { -a }, if yi > 0.0 { a } else { -a });
        let (er, ei) = (dr - yr, di - yi);
        let p: f64 = window.iter().map(|&(xr, xi)| xr * xr + xi * xi).sum();
        let m = mu / p;
        for (&(xr, xi), t) in window.iter().zip(taps.iter_mut()) {
            // conj(x) e
            t.0 += m * (xr * er + xi * ei);
            t.1 += m * (xr * ei - xi * er);
        }
    }
    out
}

#[inline(never)]
fn equalize<B: Equalize>(backend: &B, state: &mut EqState, input: &[C32], out: &mut [C32]) {
    let _ = black_box(backend.equalize(state, black_box(input), out));
}

/// `equalize` of `n` symbols (see the module documentation).
pub(crate) struct EqualizeK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    /// The state of the counted calls, carried across the runs.
    state: EqState,
    /// GNU Radio's equalizer of the counted calls (the fallback entry), carried across the runs.
    gr: Option<GrLinearEqualizer>,
    /// The entry: `equalize_fallback`.
    fallback_entry: bool,
}

impl EqualizeK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, fallback_entry: bool) -> Self {
        let n = layout.n;
        let skew = layout.skew(core::mem::size_of::<C32>());
        let r = core::f64::consts::FRAC_1_SQRT_2;
        let symbols: Vec<(f64, f64)> = (0..n)
            .map(|_| {
                let b = rng.next_u32();
                (
                    if b & 1 == 1 { r } else { -r },
                    if b & 2 == 2 { r } else { -r },
                )
            })
            .collect();
        let (sp, cp) = EQ_PHASE.sin_cos();
        let mut input = Aligned::<C32>::skewed(n, skew);
        for (k, s) in input.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for &(d, mag, ph) in &EQ_CHANNEL {
                if let Some(&(xr, xi)) = k.checked_sub(d).and_then(|j| symbols.get(j)) {
                    let (gs, gc) = ph.sin_cos();
                    re += mag * (xr * gc - xi * gs);
                    im += mag * (xr * gs + xi * gc);
                }
            }
            let nr = f64::from(rng.unit()) * EQ_NOISE;
            let ni = f64::from(rng.unit()) * EQ_NOISE;
            #[allow(clippy::cast_possible_truncation, reason = "the f32 input")]
            {
                *s = C32::new(
                    (re * cp - im * sp + nr) as f32,
                    (re * sp + im * cp + ni) as f32,
                );
            }
        }
        EqualizeK {
            input,
            out: Aligned::skewed(n, skew),
            state: eq_state(),
            gr: if fallback_entry {
                gr_equalizer(n)
            } else {
                None
            },
            fallback_entry,
        }
    }
}

impl Kernel for EqualizeK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => equalize(&Sdrk, &mut self.state, &self.input, &mut self.out),
                Impl::Fallback => {
                    equalize(&Fallback, &mut self.state, &self.input, &mut self.out);
                }
                Impl::GnuRadio => self
                    .gr
                    .as_mut()
                    .expect("gnuradio")
                    .work(c32_pairs_ref(&self.input), c32_pairs(&mut self.out)),
                other => unreachable!("equalize has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The SNR of one call from the start state against the float32 model of `Reference` (the
    /// kernel is bit-exact, so every implementation gives the cap of [`snr_db`]); GNU Radio's
    /// loop against its f64 model.
    fn accuracy(&mut self, which: Impl) -> f64 {
        let mut got = vec![C32::default(); self.input.len()];
        let mut state = eq_state();
        match which {
            Impl::Ours => equalize(&Sdrk, &mut state, &self.input, &mut got),
            Impl::Fallback => equalize(&Fallback, &mut state, &self.input, &mut got),
            Impl::GnuRadio => {
                if let Some(mut gr) = gr_equalizer(self.input.len()) {
                    gr.work(c32_pairs_ref(&self.input), c32_pairs(&mut got));
                }
                return snr_db(&c32s(&gr_equalizer_model(&self.input)), &c32s(&got));
            }
            other => unreachable!("equalize has no {other:?}"),
        }
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.equalize(&mut eq_state(), &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours => !self.fallback_entry,
            Impl::Fallback => true,
            Impl::GnuRadio => self.fallback_entry && self.gr.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio linear_equalizer with adaptive_algorithm_nlms (linear_equalizer_impl.cc of gr-digital, VOLK generic dot products, generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}
