//! The entries of the `gauss_noise` kernel and the `noise` block (milestone M48): `n` samples of
//! Gaussian noise per call, PCG32 with one draw per real value and the inverse CDF of the
//! 1024-entry Q17 table, scaled by `sigma` per real component.
//!
//! | entry | measured | comparison | information |
//! | --- | --- | --- | --- |
//! | `gauss_noise_c32` | facade (the SIMD path of the level) | liquid-dsp 1.8.3 `crandnf` times `sigma` (`info`, reference does different work) | our fallback |
//! | `gauss_noise_f32` | facade (the SIMD path of the level) | liquid-dsp 1.8.3 `randnf` times `sigma` (`info`, reference does different work) | our fallback |
//! | `gauss_noise_q15` | facade (the SIMD path of the level) | our fallback (`info`; no library makes `Q15` noise) | |
//! | `gauss_noise_q15_fallback` | our fallback | our fallback (`info`, ratio 1; the pair of `gauss_noise_q15` for the D321 cycle gate) | |
//! | `noise_block` | `sdrk_blocks::NoiseBlock<C32>` | none (`info`) | |
//! | `gauss_noise_c32_fastnoise` | facade | GNU Radio `fastnoise_source_c` (`info`, reference does less work) | |
//! | `gauss_noise_f32_fastnoise` | facade | GNU Radio `fastnoise_source_f` (`info`, reference does less work) | |
//! | `gauss_noise_f32_ziggurat` | facade | rand_distr `StandardNormal` as `f32` (`info`, reference does different work) | |
//!
//! liquid draws its uniform values from libc `rand()` (glibc's `random_r` under a lock on
//! x86_64 and aarch64, wasi-libc's generator on wasm32) and evaluates Box–Muller in single
//! precision (`sqrtf`, `logf`, `sinf` for `randnf`, `cexpf` for `crandnf`), without random access
//! and with a sequence that depends on the platform's libc; ours runs PCG32 and the inverse CDF
//! in integers with the same bits on every target. The rows are information, not a gap to close
//! (M48 §5).
//!
//! GNU Radio's `fastnoise_source` (`benches/refs/gr-fastnoise.c`, the work loop of GNU Radio
//! main at ec1bf15) draws one xoroshiro128+ value per sample and copies the entry of a pool of
//! 8192 Gaussian values (the default of GNU Radio Companion) that it selects: no normal mapping
//! per sample, values that repeat, a tail that ends at the largest pool value, no random access.
//! It does less work and shows the floor of what a noise sample can cost. The pool is filled at
//! construction, outside the counted calls. `fastnoise_source_c` divides its amplitude by
//! `sqrt(2)`, so it gets `sigma * sqrt(2)` (GNU Radio's power convention, M48 §3).
//!
//! The Ziggurat row is rand_distr's `StandardNormal` (sampled as `f32`: the `f64` Ziggurat
//! rounded to `f32`) over rand's `SmallRng` (xoshiro256++ on the 64-bit targets, xoshiro128++
//! on wasm32), times `sigma`: rejection sampling with a variable number of draws per value, no
//! random access and no result that is the same across platforms and versions; different work
//! (maintainer decision 2026-11-17, M48 §5).
//!
//! Accuracy: ours and the fallback against `sdrk_core::Reference` at the same position (999 when
//! bit-exact). liquid's sequence is another one, so an SNR against ours means nothing: its
//! accuracy, and that of GNU Radio and rand_distr, is a statistics marker, 999 when the mean and
//! the variance of one call (per real component) match `sigma` (|mean| below 5 sigma / sqrt(values), variance within 10 % of
//! sigma²), 0 otherwise.
//!
//! Workload: `sigma` 0.1 (the block default), stream 1, a seed from the entry's input generator;
//! the state carries on from call to call, so every call makes new samples (the warm-up call
//! of the counted runs is the same in both runs of the differential count). `noise_block` runs the
//! block with the default configuration on a clock of `n` zero `C32` samples per call (the block
//! never reads the values).

use std::hint::black_box;

use cref_sys::{GR_FASTNOISE_POOL, GrFastnoiseSourceC, GrFastnoiseSourceF, LiquidRandn};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use sdrk::kernels::noise::{GaussNoise, GaussNoiseState};
use sdrk::{C32, Fallback, Q15, Rate, Sample, Sdrk};
use sdrk_blocks::{ArenaStorage, Block, NoiseBlock, NoiseConfig};
use sdrk_core::Reference;

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32s, f32s, snr_db};

/// `sigma` of the workload (the block default).
const SIGMA: f32 = 0.1;
/// The stream of the workload (the block default).
const STREAM: u64 = 1;

#[inline(never)]
fn noise_call<T: Sample, B: GaussNoise<T>>(b: &B, state: &mut GaussNoiseState, out: &mut [T]) {
    let _ = black_box(b.gauss_noise(state, out));
}

/// 999 when the values (real components) have the mean and the variance of `sigma` (see the
/// module documentation), 0 otherwise.
fn statistics_marker(values: &[f64], sigma: f64) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let s2 = sigma * sigma;
    let ok = mean.abs() < 5.0 * sigma / n.sqrt() && (var - s2).abs() < 0.1 * s2;
    if ok { 999.0 } else { 0.0 }
}

/// The `C32` buffer as liquid's `float complex` pairs.
fn pairs_mut(x: &mut [C32]) -> &mut [[f32; 2]] {
    // SAFETY: `C32` is `repr(C)` of two `f32`, the layout of `[f32; 2]`.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

/// The implementations of the `f32` and `C32` entries: ours, the fallback, liquid (when
/// linked), GNU Radio's `fastnoise_source` (when linked) and, for `f32`, rand_distr.
fn supports_noise(which: Impl, gnuradio: bool, rand_distr: bool) -> bool {
    match which {
        Impl::Ours | Impl::Fallback => true,
        Impl::Liquid => LiquidRandn::AVAILABLE,
        Impl::GnuRadio => gnuradio,
        Impl::RandDistr => rand_distr,
        _ => false,
    }
}

fn describe_noise(which: Impl, liquid: &'static str, gnuradio: &'static str) -> &'static str {
    match which {
        Impl::Liquid => liquid,
        Impl::GnuRadio => gnuradio,
        Impl::RandDistr => {
            "rand_distr 0.5.1 StandardNormal as f32 times sigma over rand 0.9.5 SmallRng (Ziggurat in f64 with rejection: a variable number of draws per value, no random access, no bit-exact sequence across platforms; reference does different work)"
        }
        Impl::Fallback => "sdrk fallback",
        _ => "sdrk",
    }
}

/// `n` values of rand_distr's `StandardNormal` (as `f32`) times `sigma`.
#[inline(never)]
fn ziggurat_call(rng: &mut SmallRng, sigma: f32, out: &mut [f32]) {
    for v in out.iter_mut() {
        let z: f32 = StandardNormal.sample(rng);
        *v = z * sigma;
    }
}

/// The square root of 2 in `f32`: GNU Radio's complex amplitude for `sigma` per component.
const SQRT_2: f32 = core::f32::consts::SQRT_2;

// ---------------------------------------------------------------------------------------------
// C32

/// `gauss_noise` of `n` `C32` samples per call; liquid-dsp's `crandnf` as information.
pub(crate) struct GaussNoiseC32K {
    state: GaussNoiseState,
    out: Aligned<C32>,
    gnuradio: Option<GrFastnoiseSourceC>,
}

impl GaussNoiseC32K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        LiquidRandn::seed(rng.next_u32());
        let state = GaussNoiseState::new(u64::from(rng.next_u32()), STREAM, SIGMA)
            .expect("gauss_noise state");
        let gr_seed = u64::from(rng.next_u32());
        GaussNoiseC32K {
            state,
            out: Aligned::skewed(layout.n, layout.skew(8)),
            gnuradio: GrFastnoiseSourceC::new(SIGMA * SQRT_2, gr_seed, GR_FASTNOISE_POOL),
        }
    }
}

impl Kernel for GaussNoiseC32K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => noise_call(&Sdrk, &mut self.state, &mut self.out),
                Impl::Fallback => noise_call(&Fallback, &mut self.state, &mut self.out),
                Impl::Liquid => {
                    LiquidRandn::crandnf_block(black_box(SIGMA), pairs_mut(&mut self.out));
                }
                Impl::GnuRadio => {
                    let gr = self.gnuradio.as_mut().expect("fastnoise_source_c");
                    gr.work(pairs_mut(&mut self.out));
                }
                other => unreachable!("gauss_noise_c32 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        if matches!(which, Impl::Liquid | Impl::GnuRadio) {
            self.run(which, 1);
            return statistics_marker(&c32s(&self.out), f64::from(SIGMA));
        }
        let mut reference = self.state;
        let mut want = vec![C32::default(); self.out.len()];
        let _ = Reference.gauss_noise(&mut reference, &mut want);
        self.run(which, 1);
        if reference != self.state {
            return 0.0;
        }
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        supports_noise(which, self.gnuradio.is_some(), false)
    }

    fn describe(&self, which: Impl) -> &'static str {
        describe_noise(
            which,
            "liquid-dsp 1.8.3 crandnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and cexpf; no random access, no bit-exact sequence; reference does different work; -O3)",
            "GNU Radio fastnoise_source_c, amplitude sigma * sqrt(2) (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3)",
        )
    }
}

// ---------------------------------------------------------------------------------------------
// f32

/// `gauss_noise` of `n` `f32` samples per call; liquid-dsp's `randnf` as information.
pub(crate) struct GaussNoiseF32K {
    state: GaussNoiseState,
    out: Aligned<f32>,
    gnuradio: Option<GrFastnoiseSourceF>,
    ziggurat: SmallRng,
}

impl GaussNoiseF32K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        LiquidRandn::seed(rng.next_u32());
        let state = GaussNoiseState::new(u64::from(rng.next_u32()), STREAM, SIGMA)
            .expect("gauss_noise state");
        let gr_seed = u64::from(rng.next_u32());
        let zig_seed = u64::from(rng.next_u32());
        GaussNoiseF32K {
            state,
            out: Aligned::skewed(layout.n, layout.skew(4)),
            gnuradio: GrFastnoiseSourceF::new(SIGMA, gr_seed, GR_FASTNOISE_POOL),
            ziggurat: SmallRng::seed_from_u64(zig_seed),
        }
    }
}

impl Kernel for GaussNoiseF32K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => noise_call(&Sdrk, &mut self.state, &mut self.out),
                Impl::Fallback => noise_call(&Fallback, &mut self.state, &mut self.out),
                Impl::Liquid => LiquidRandn::randnf_block(black_box(SIGMA), &mut self.out),
                Impl::GnuRadio => {
                    let gr = self.gnuradio.as_mut().expect("fastnoise_source_f");
                    gr.work(&mut self.out);
                }
                Impl::RandDistr => {
                    ziggurat_call(&mut self.ziggurat, black_box(SIGMA), &mut self.out);
                }
                other => unreachable!("gauss_noise_f32 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        if matches!(which, Impl::Liquid | Impl::GnuRadio | Impl::RandDistr) {
            self.run(which, 1);
            return statistics_marker(&f32s(&self.out), f64::from(SIGMA));
        }
        let mut reference = self.state;
        let mut want = vec![0.0f32; self.out.len()];
        let _ = Reference.gauss_noise(&mut reference, &mut want);
        self.run(which, 1);
        if reference != self.state {
            return 0.0;
        }
        snr_db(&f32s(&want), &f32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        supports_noise(which, self.gnuradio.is_some(), true)
    }

    fn describe(&self, which: Impl) -> &'static str {
        describe_noise(
            which,
            "liquid-dsp 1.8.3 randnf times sigma (libc rand(), f32 Box-Muller with sqrtf, logf and sinf; no random access, no bit-exact sequence; reference does different work; -O3)",
            "GNU Radio fastnoise_source_f, amplitude sigma (benches/refs/gr-fastnoise.c, main at ec1bf15: one xoroshiro128+ draw and a copy from a pool of 8192 precomputed values per sample, no normal mapping, no random access; reference does less work; -O3)",
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Q15

/// `gauss_noise` of `n` `Q15` samples per call; no library makes `Q15` noise, so our fallback is
/// the comparison (information, as for `qam_soft_bits`).
pub(crate) struct GaussNoiseQ15K {
    state: GaussNoiseState,
    out: Aligned<Q15>,
}

impl GaussNoiseQ15K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        GaussNoiseQ15K {
            state: GaussNoiseState::new(u64::from(rng.next_u32()), STREAM, SIGMA)
                .expect("gauss_noise state"),
            out: Aligned::skewed(layout.n, layout.skew(2)),
        }
    }
}

impl Kernel for GaussNoiseQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => noise_call(&Sdrk, &mut self.state, &mut self.out),
                Impl::Fallback => noise_call(&Fallback, &mut self.state, &mut self.out),
                other => unreachable!("gauss_noise_q15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        let mut reference = self.state;
        let mut want = vec![Q15(0); self.out.len()];
        let _ = Reference.gauss_noise(&mut reference, &mut want);
        self.run(which, 1);
        if reference != self.state {
            return 0.0;
        }
        let q = |x: &[Q15]| x.iter().map(|v| f64::from(v.0)).collect::<Vec<_>>();
        snr_db(&q(&want), &q(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        describe_noise(which, "", "")
    }
}

// ---------------------------------------------------------------------------------------------
// The block

/// The block `noise` (`NoiseBlock<C32>`, default configuration) over `n` clock samples per call.
pub(crate) struct NoiseBlockK {
    block: NoiseBlock<C32, ArenaStorage<'static>>,
    clock: Vec<C32>,
    out: Vec<C32>,
}

fn noise_config() -> NoiseConfig {
    NoiseConfig::default()
}

fn noise_rate() -> Rate {
    Rate::hz(48_000).expect("rate")
}

impl NoiseBlockK {
    pub(crate) fn new(_rng: &mut Pcg32, layout: Layout) -> Self {
        NoiseBlockK {
            block: NoiseBlock::new(
                &noise_config(),
                noise_rate(),
                &mut ArenaStorage::new(&mut []),
            )
            .expect("noise block"),
            clock: vec![C32::default(); layout.n],
            out: vec![C32::default(); layout.n],
        }
    }
}

#[inline(never)]
fn noise_block_call(
    block: &mut NoiseBlock<C32, ArenaStorage<'static>>,
    clock: &[C32],
    out: &mut [C32],
) {
    let _ = black_box(block.process(black_box(clock), out));
}

impl Kernel for NoiseBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    noise_block_call(&mut self.block, &self.clock, &mut self.out);
                }
                other => unreachable!("noise_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when one call from the reset state gives the bits of the kernel from a new state of
    /// the same configuration; 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        let cfg = noise_config();
        let mut state =
            GaussNoiseState::new(cfg.seed, cfg.stream, cfg.sigma as f32).expect("state");
        let mut want = vec![C32::default(); self.out.len()];
        if Sdrk.gauss_noise(&mut state, &mut want).is_err() {
            return 0.0;
        }
        self.block.reset();
        self.run(which, 1);
        let same = want
            .iter()
            .zip(&self.out)
            .all(|(w, g)| w.re.to_bits() == g.re.to_bits() && w.im.to_bits() == g.im.to_bits());
        if same { 999.0 } else { 0.0 }
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
