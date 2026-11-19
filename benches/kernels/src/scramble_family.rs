//! The entries of the `scramble` kernel and the `scramble` block (milestones M49 §5 and M49.1
//! §5): `n` samples per call, each multiplied by `exp(j 2π p / 2^bits)` for its phase byte `p`
//! (the exact sign map for `bits = 1`, the quadrant map for `bits = 2`, one table phasor and one
//! `rotate` product for `bits = 8`).
//!
//! | entry | measured | comparison | information |
//! | --- | --- | --- | --- |
//! | `scramble_c32_b1` | facade (the SIMD path of the level) | a `±1.0` `f32` sequence formed from the index bits into a buffer + VOLK `volk_32fc_32f_multiply_32fc` through VOLK's dispatcher (no FMA in it at any level; its `generic` protokernel at `simd128`, D12): the form of srsRAN's `srsran_scrambling_c` | |
//! | `scramble_c32_b1_fallback` | our fallback | the same sequence + VOLK's `generic` protokernel (`volk-generic.c`) | |
//! | `scramble_c32_b2`, `scramble_c32_b8` | facade (the SIMD path of the level) | the phasor looked up into a buffer + VOLK `volk_32fc_x2_multiply_32fc` without FMA (`volk-nofma.c`) at `avx2`, `avx512`, `neon`; VOLK's dispatcher elsewhere (no FMA there) | VOLK's dispatcher at the FMA levels |
//! | `scramble_c32_b2_fallback`, `scramble_c32_b8_fallback` | our fallback | the same lookup + VOLK's `generic` protokernel (`volk-generic.c`) | |
//! | `scramble_cq15_b2`, `scramble_cq15_b8` | facade | the same lookup + CMSIS-DSP `arm_cmplx_mult_cmplx_q15` (3.13 output) | our fallback |
//! | `scramble_cq15_b2_fallback`, `scramble_cq15_b8_fallback` | our fallback | the same CMSIS-DSP composition | |
//! | `scramble_c32_b2_dvbs2` | facade | GNU Radio `dvbs2_physical_cc`'s per-symbol switch on `R_n` (`info`, reference does different work) | |
//! | `scramble_block` | `sdrk_blocks::ScrambleBlock<C32>`, `bits = 8`, key `prbs15` | none (`info`) | |
//!
//! The library composition (M49 §5, D32): a user without a scrambling kernel looks the phasor
//! `exp(j 2π p / 2^bits)` of each sample up in a table of `2^bits` entries (built once from the
//! same cosine table `COS_F32` / `COS_Q15` as the kernel) into a buffer of [`RUN`] samples and
//! multiplies the input by it with the library's complex product. At `bits = 2` our kernel does
//! less work by design (the quadrant map has no multiplication); the gap shows the gain of the
//! quadrant form. At `bits = 8` the kernel does the same work in one pass (M49.1: one table
//! phasor and one product), so the `C32` rows are fair. At `bits = 1` (M49.1 §5) the
//! composition is that of srsRAN's `srsran_scrambling_c` (which calls `srsran_vec_prod_cfc`
//! with a real `±1` sequence): the sign `1 − 2 (p & 1)` of each sample looked up from a table
//! of two `f32` into a buffer of [`RUN`] values, then VOLK's complex-by-real product. CMSIS-DSP's product writes Q3.13 (each partial product shifted by 17), a
//! different output scaling: its accuracy is taken on the output times 4, and the row is marked
//! "(3.13 output)".
//!
//! Accuracy: SNR in dB against the exact rotation in `f64` (`x · exp(j 2π p / 2^bits)` with the
//! phasor in `f64`), 999 when bit for bit. `scramble_block` is 999 when one call from the reset
//! state equals the kernel on the PRBS indices of the key.

use std::hint::black_box;

use cref_sys::cmsis_dsp;
use cref_sys::{gr_dvbs2, volk_generic, volk_nofma};
use sdrk::kernels::scramble::Scramble;
use sdrk::{C32, CQ15, ComplexSample, Fallback, Rate, Sdrk};
use sdrk_blocks::{ArenaStorage, Block, ScrambleBlock, ScrambleConfig};
use sdrk_core::tables::{COS_F32, COS_Q15};
use volk_sys::lv_32fc_t;

use super::{
    Aligned, BiquadEntry, Impl, Kernel, Layout, Pcg32, c32s, cq15_pairs, cq15_pairs_mut, cq15s,
    describe_cmsis, noise_c32, noise_cq15, snr_db,
};

/// Samples per run of the library compositions: the phasor buffer of one run.
pub(crate) const RUN: usize = 256;

/// The index of the sine in the cosine table (`cos(x − π/2)`), modulo its 4096 entries.
const SINE_OFFSET: usize = 3072;

/// The `2^bits` phasors `exp(j 2π k / 2^bits)` from `COS_F32` (what the composition looks up).
fn phasors_c32(bits: u32) -> Vec<[f32; 2]> {
    (0..1usize << bits)
        .map(|k| {
            let i = k << (12 - bits);
            [COS_F32[i], COS_F32[(i + SINE_OFFSET) & 4095]]
        })
        .collect()
}

/// The `2^bits` phasors from `COS_Q15`.
fn phasors_q15(bits: u32) -> Vec<[i16; 2]> {
    (0..1usize << bits)
        .map(|k| {
            let i = k << (12 - bits);
            [COS_Q15[i], COS_Q15[(i + SINE_OFFSET) & 4095]]
        })
        .collect()
}

/// The `±1.0` sequence of `bits = 1`: index 0 keeps the sample, index 1 negates it.
const SIGNS: [f32; 2] = [1.0, -1.0];

/// The exact rotation of `(re, im)` by `exp(j 2π p / 2^bits)` in `f64`.
fn ideal(re: f64, im: f64, p: u8, bits: u32) -> [f64; 2] {
    let k = u32::from(p) & ((1 << bits) - 1);
    let phi = 2.0 * std::f64::consts::PI * f64::from(k) / f64::from(1u32 << bits);
    let (s, c) = phi.sin_cos();
    [re * c - im * s, re * s + im * c]
}

fn ideal_c32(input: &[C32], phase: &[u8], bits: u32) -> Vec<f64> {
    input
        .iter()
        .zip(phase)
        .flat_map(|(x, &p)| ideal(f64::from(x.re), f64::from(x.im), p, bits))
        .collect()
}

fn ideal_cq15(input: &[CQ15], phase: &[u8], bits: u32) -> Vec<f64> {
    input
        .iter()
        .zip(phase)
        .flat_map(|(x, &p)| ideal(f64::from(x.re.0), f64::from(x.im.0), p, bits))
        .collect()
}

fn phase_bytes(rng: &mut Pcg32, n: usize) -> Vec<u8> {
    (0..n).map(|_| (rng.next_u32() >> 24) as u8).collect()
}

#[inline(never)]
fn scramble_call<T: ComplexSample, B: Scramble<T>>(
    backend: &B,
    input: &[T],
    phase: &[u8],
    bits: u32,
    out: &mut [T],
) {
    let _ = black_box(backend.scramble(black_box(input), phase, bits, false, out));
}

fn pairs(x: &[C32]) -> &[[f32; 2]] {
    // SAFETY: `C32` is `repr(C)` of two `f32`, the layout of `[f32; 2]`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

fn pairs_mut(x: &mut [C32]) -> &mut [[f32; 2]] {
    // SAFETY: as in `pairs`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

/// The VOLK flavour of the `C32` composition.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VolkKind {
    /// VOLK's dispatcher.
    Dispatch,
    /// `volk-nofma.c` (`u_avx` or `neon`).
    NoFma,
    /// `volk-generic.c`.
    Generic,
}

/// The `C32` composition: per run of [`RUN`] samples the phasors looked up into `buf`, then
/// VOLK's complex product.
#[inline(never)]
fn volk_scramble(
    kind: VolkKind,
    table: &[[f32; 2]],
    mask: u8,
    input: &[C32],
    phase: &[u8],
    buf: &mut [[f32; 2]],
    out: &mut [C32],
) {
    for ((x, p), y) in input
        .chunks(RUN)
        .zip(phase.chunks(RUN))
        .zip(out.chunks_mut(RUN))
    {
        let b = &mut buf[..x.len()];
        for (v, &k) in b.iter_mut().zip(p) {
            *v = table[usize::from(k & mask)];
        }
        match kind {
            // SAFETY: the three buffers hold `x.len()` samples and the output aliases no input.
            VolkKind::Dispatch => unsafe {
                volk_sys::volk_32fc_x2_multiply_32fc(
                    y.as_mut_ptr().cast::<lv_32fc_t>(),
                    x.as_ptr().cast::<lv_32fc_t>(),
                    b.as_ptr().cast::<lv_32fc_t>(),
                    x.len() as u32,
                );
            },
            VolkKind::NoFma => {
                volk_nofma::multiply_32fc(pairs(x), b, pairs_mut(y));
            }
            VolkKind::Generic => {
                volk_generic::multiply_32fc(pairs(x), b, pairs_mut(y));
            }
        }
    }
}

/// The `bits = 1` composition (the form of srsRAN's `srsran_scrambling_c`): per run of [`RUN`]
/// samples the `±1.0` sequence of the index bits formed into `buf`, then VOLK's
/// `volk_32fc_32f_multiply_32fc` (`generic`: its portable protokernel).
#[inline(never)]
fn volk_sign_scramble(
    generic: bool,
    input: &[C32],
    phase: &[u8],
    buf: &mut [f32],
    out: &mut [C32],
) {
    for ((x, p), y) in input
        .chunks(RUN)
        .zip(phase.chunks(RUN))
        .zip(out.chunks_mut(RUN))
    {
        let b = &mut buf[..x.len()];
        for (v, &k) in b.iter_mut().zip(p) {
            *v = SIGNS[usize::from(k & 1)];
        }
        if generic {
            volk_generic::multiply_32fc_32f(pairs(x), b, pairs_mut(y));
        } else {
            // SAFETY: the three buffers hold `x.len()` elements and the output aliases no input.
            unsafe {
                volk_sys::native::volk_32fc_32f_multiply_32fc(
                    y.as_mut_ptr().cast::<lv_32fc_t>(),
                    x.as_ptr().cast::<lv_32fc_t>(),
                    b.as_ptr(),
                    x.len() as u32,
                );
            }
        }
    }
}

/// The `CQ15` composition: per run of [`RUN`] samples the phasors looked up into `buf`, then
/// `arm_cmplx_mult_cmplx_q15` (Q3.13 output).
#[inline(never)]
fn cmsis_scramble(
    table: &[[i16; 2]],
    mask: u8,
    input: &[[i16; 2]],
    phase: &[u8],
    buf: &mut [[i16; 2]],
    out: &mut [[i16; 2]],
) {
    for ((x, p), y) in input
        .chunks(RUN)
        .zip(phase.chunks(RUN))
        .zip(out.chunks_mut(RUN))
    {
        let b = &mut buf[..x.len()];
        for (v, &k) in b.iter_mut().zip(p) {
            *v = table[usize::from(k & mask)];
        }
        cmsis_dsp::cmplx_mult_cmplx_q15(x, b, y);
    }
}

// ---------------------------------------------------------------------------------------------
// C32

/// `scramble` of `n` `C32` samples at `bits` (1, 2 or 8); see the module documentation.
pub(crate) struct ScrambleC32K {
    bits: u32,
    entry: BiquadEntry,
    input: Aligned<C32>,
    phase: Vec<u8>,
    out: Aligned<C32>,
    table: Vec<[f32; 2]>,
    buf: Vec<[f32; 2]>,
    /// The `±1.0` buffer of the `bits = 1` composition, on a 64-byte boundary so that VOLK's
    /// dispatcher can take its aligned protokernel (`a_sse`, `a_avx`).
    signs: Aligned<f32>,
    /// `R_n = p & 3` as the `int` sequence of GNU Radio (`scramble_c32_b2_dvbs2`).
    rn: Vec<core::ffi::c_int>,
}

impl ScrambleC32K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, bits: u32, entry: BiquadEntry) -> Self {
        let skew = layout.skew(8);
        let input = noise_c32(rng, layout.n, 0.9, skew);
        let phase = phase_bytes(rng, layout.n);
        let rn = phase.iter().map(|&p| core::ffi::c_int::from(p & 3)).collect();
        ScrambleC32K {
            bits,
            entry,
            input,
            phase,
            out: Aligned::skewed(layout.n, skew),
            table: phasors_c32(bits),
            buf: vec![[0.0; 2]; RUN],
            signs: Aligned::new(RUN),
            rn,
        }
    }

    fn mask(&self) -> u8 {
        ((1u32 << self.bits) - 1) as u8
    }
}

impl Kernel for ScrambleC32K {
    fn run(&mut self, which: Impl, k: usize) {
        let mask = self.mask();
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    scramble_call(&Sdrk, &self.input, &self.phase, self.bits, &mut self.out);
                }
                Impl::Fallback => {
                    scramble_call(&Fallback, &self.input, &self.phase, self.bits, &mut self.out);
                }
                Impl::Volk | Impl::VolkGeneric if self.bits == 1 => volk_sign_scramble(
                    which == Impl::VolkGeneric,
                    &self.input,
                    black_box(&self.phase),
                    &mut self.signs,
                    &mut self.out,
                ),
                Impl::Volk | Impl::VolkNoFma | Impl::VolkGeneric => {
                    let kind = match which {
                        Impl::Volk => VolkKind::Dispatch,
                        Impl::VolkNoFma => VolkKind::NoFma,
                        _ => VolkKind::Generic,
                    };
                    volk_scramble(
                        kind,
                        &self.table,
                        black_box(mask),
                        &self.input,
                        &self.phase,
                        &mut self.buf,
                        &mut self.out,
                    );
                }
                Impl::GnuRadio => {
                    gr_dvbs2::plscramble(
                        pairs(black_box(&self.input)),
                        &self.rn,
                        pairs_mut(&mut self.out),
                    );
                }
                other => unreachable!("scramble (C32) has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        snr_db(
            &ideal_c32(&self.input, &self.phase, self.bits),
            &c32s(&self.out),
        )
    }

    fn supports(&self, which: Impl) -> bool {
        match (self.entry, which) {
            (BiquadEntry::Simd, Impl::Ours) | (_, Impl::Fallback) => true,
            (BiquadEntry::Simd, Impl::Volk) => volk_sys::FULL,
            (BiquadEntry::Simd, Impl::VolkNoFma) => self.bits != 1 && volk_nofma::AVAILABLE,
            (BiquadEntry::Simd, Impl::GnuRadio) => self.bits == 2 && gr_dvbs2::AVAILABLE,
            (BiquadEntry::Fallback, Impl::VolkGeneric) => volk_generic::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk if self.bits == 1 => {
                "+-1.0 sequence from the index bits into a buffer of 256 + volk_32fc_32f_multiply_32fc (VOLK 3.3.0 dispatcher; the form of srsRAN srsran_scrambling_c)"
            }
            Impl::VolkGeneric if self.bits == 1 => {
                "+-1.0 sequence from the index bits into a buffer of 256 + VOLK 3.3.0 volk_32fc_32f_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off)"
            }
            Impl::Volk => {
                "phasor lookup into a buffer of 256 + volk_32fc_x2_multiply_32fc (VOLK 3.3.0 dispatcher)"
            }
            Impl::VolkNoFma => match volk_nofma::protokernel() {
                Some("u_avx") => {
                    "phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_u_avx without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off)"
                }
                Some("neon") => {
                    "phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_neon without FMA (benches/refs/volk-nofma.c, -O3 -ffp-contract=off)"
                }
                _ => {
                    "phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc without FMA (benches/refs/volk-nofma.c)"
                }
            },
            Impl::VolkGeneric => {
                "phasor lookup into a buffer of 256 + VOLK 3.3.0 volk_32fc_x2_multiply_32fc_generic (benches/refs/volk-generic.c, -O3 -ffp-contract=off)"
            }
            Impl::GnuRadio => {
                "GNU Radio dvbs2_physical_cc PL scrambling, the per-symbol switch on R_n (benches/refs/gr-dvbs2-plscramble.c, main at ec1bf15; reference does different work; -O3)"
            }
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if self.entry == BiquadEntry::Simd && volk_sys::FULL && self.bits == 1 {
            &["volk_32fc_32f_multiply_32fc"]
        } else if self.entry == BiquadEntry::Simd && volk_sys::FULL {
            &["volk_32fc_x2_multiply_32fc"]
        } else {
            &[]
        }
    }
}

// ---------------------------------------------------------------------------------------------
// CQ15

/// `scramble` of `n` `CQ15` samples at `bits` (2 or 8); see the module documentation.
pub(crate) struct ScrambleCq15K {
    bits: u32,
    entry: BiquadEntry,
    input: Aligned<CQ15>,
    phase: Vec<u8>,
    out: Aligned<CQ15>,
    table: Vec<[i16; 2]>,
    buf: Vec<[i16; 2]>,
}

impl ScrambleCq15K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, bits: u32, entry: BiquadEntry) -> Self {
        let skew = layout.skew(4);
        let input = noise_cq15(rng, layout.n, 23_000.0, skew);
        let phase = phase_bytes(rng, layout.n);
        ScrambleCq15K {
            bits,
            entry,
            input,
            phase,
            out: Aligned::skewed(layout.n, skew),
            table: phasors_q15(bits),
            buf: vec![[0; 2]; RUN],
        }
    }
}

impl Kernel for ScrambleCq15K {
    fn run(&mut self, which: Impl, k: usize) {
        let mask = ((1u32 << self.bits) - 1) as u8;
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    scramble_call(&Sdrk, &self.input, &self.phase, self.bits, &mut self.out);
                }
                Impl::Fallback => {
                    scramble_call(&Fallback, &self.input, &self.phase, self.bits, &mut self.out);
                }
                Impl::Cmsis => cmsis_scramble(
                    &self.table,
                    black_box(mask),
                    cq15_pairs(&self.input),
                    &self.phase,
                    &mut self.buf,
                    cq15_pairs_mut(&mut self.out),
                ),
                other => unreachable!("scramble (CQ15) has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        // CMSIS-DSP writes Q3.13: its output times 4 is in the Q15 units of the input.
        let scale = if which == Impl::Cmsis { 4.0 } else { 1.0 };
        snr_db(
            &ideal_cq15(&self.input, &self.phase, self.bits),
            &cq15s(&self.out, scale),
        )
    }

    fn supports(&self, which: Impl) -> bool {
        match (self.entry, which) {
            (BiquadEntry::Simd, Impl::Ours) | (_, Impl::Fallback) => true,
            (_, Impl::Cmsis) => cmsis_dsp::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis(
                "phasor lookup into a buffer of 256 + arm_cmplx_mult_cmplx_q15 (3.13 output)",
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The block

/// The configuration of `scramble_block`: `bits = 8` (the most PRBS steps per sample), the
/// default key `prbs15` from the register 1.
fn block_config() -> ScrambleConfig {
    ScrambleConfig {
        bits: 8,
        ..ScrambleConfig::default()
    }
}

fn block_rate() -> Rate {
    Rate::hz(48_000).expect("rate")
}

/// The block `scramble` (`ScrambleBlock<C32>`, [`block_config`]) over `n` samples per call,
/// called until the input is consumed (at most 256 samples per `process`).
pub(crate) struct ScrambleBlockK {
    block: ScrambleBlock<C32, ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<C32>,
}

impl ScrambleBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let input = noise_c32(rng, layout.n, 0.9, 0).to_vec();
        ScrambleBlockK {
            block: ScrambleBlock::new(
                &block_config(),
                block_rate(),
                &mut ArenaStorage::new(&mut []),
            )
            .expect("scramble block"),
            out: vec![C32::default(); input.len()],
            input,
        }
    }
}

#[inline(never)]
fn block_call(block: &mut ScrambleBlock<C32, ArenaStorage<'static>>, input: &[C32], out: &mut [C32]) {
    let mut done = 0;
    while done < input.len() {
        match block.process(black_box(&input[done..]), &mut out[done..]) {
            Ok(p) if p.consumed > 0 => done += p.consumed,
            _ => break,
        }
    }
}

impl Kernel for ScrambleBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    block_call(&mut self.block, &self.input, &mut self.out);
                }
                other => unreachable!("scramble_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when one pass from the reset state equals the kernel on the PRBS indices of the key
    /// (`bits` bits per sample, the first as the most significant), 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        let cfg = block_config();
        let Ok(mut prbs) = sdrk_proto::prbs::Prbs::new(cfg.key, cfg.key_seed) else {
            return 0.0;
        };
        let phase: Vec<u8> = (0..self.input.len())
            .map(|_| {
                (0..cfg.bits).fold(0u8, |acc, _| (acc << 1) | prbs.next_bit())
            })
            .collect();
        let mut want = vec![C32::default(); self.input.len()];
        if Sdrk
            .scramble(&self.input, &phase, cfg.bits, cfg.inverse, &mut want)
            .is_err()
        {
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
