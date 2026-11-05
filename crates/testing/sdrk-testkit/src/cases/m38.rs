//! Suite cases of M38: the kernel `symbol_sync` and its extreme inputs.
//!
//! The kernel is bit-exact (the float32 model) and not accelerated (the facade is the
//! fallback). The kernel cases compare `Fallback` with `Reference` (output and final state) and
//! the facade with `Fallback`, and run the facade in 20 chunk plans whose calls get input chunks
//! of 1 to 300 samples and output chunks of 1 to 64 samples (so calls also end with a full
//! output); every run must give the outputs and the final state of the run in one call. The
//! signals are BPSK and QPSK test signals of `r#gen::PskTest` plus noise; no files are read, so
//! the cases run on microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted};
use sdrk_core::{C32, Fallback, Reference, Sample};

use super::ensure;
use crate::arena::Arena;
use crate::r#gen::{Psk, PskTest};
use crate::invariance::ChunkPlan;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Length of the signals of the kernel cases.
const N: usize = 2048;
/// Room for every output of a signal of `N` samples (`sps >= 2`).
const OUT: usize = N / 2 + 2;
/// Chunk plans per configuration.
const PLANS: u64 = 20;
/// Largest input and output chunk of the plans.
const MAX_IN: usize = 300;
const MAX_OUT: usize = 64;
/// The loop parameters of the kernel cases: the block defaults (damping 1/√2) with a wider loop, so the strobes
/// move noticeably within 500 symbols.
const LOOP_BW: f64 = 0.03;
const DAMPING: f64 = core::f64::consts::FRAC_1_SQRT_2;
const MAX_DEVIATION: f64 = 0.01;
const TED_GAIN: f64 = 1.0;

/// The components of a sample, for comparisons that treat every NaN as equal (the payload is not
/// part of the contract).
trait Parts: Sample {
    fn parts(self) -> [f32; 2];
    fn from_complex(v: Complex64) -> Self;
}

impl Parts for C32 {
    fn parts(self) -> [f32; 2] {
        [self.re, self.im]
    }
    fn from_complex(v: Complex64) -> Self {
        C32::new(v.re as f32, v.im as f32)
    }
}

impl Parts for f32 {
    fn parts(self) -> [f32; 2] {
        [self, 0.0]
    }
    fn from_complex(v: Complex64) -> Self {
        v.re as f32
    }
}

/// Bit equality; NaN equals NaN.
fn same<T: Parts>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.parts()
                .iter()
                .zip(y.parts().iter())
                .all(|(p, q)| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan()))
        })
}

fn get<X>(s: &[X], from: usize, to: usize) -> Result<&[X], Failure> {
    s.get(from..to)
        .ok_or_else(|| Failure::new("m38", "slice range"))
}

fn get_mut<X>(s: &mut [X], from: usize, to: usize) -> Result<&mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new("m38", "slice range"))
}

fn state<T: Sample>(sps: f64, ted: Ted, max_deviation: f64) -> Result<SymbolSyncState<T>, Failure> {
    Ok(SymbolSyncState::new(
        sps,
        ted,
        LOOP_BW,
        DAMPING,
        max_deviation,
        TED_GAIN,
    )?)
}

/// One call with room for every output; it must consume the whole input. Returns the count.
fn whole<T: Sample, K: SymbolSync<T>>(
    kernel: &K,
    st: &mut SymbolSyncState<T>,
    x: &[T],
    out: &mut [T],
) -> Result<usize, Failure> {
    let p = kernel.symbol_sync(st, x, out)?;
    ensure(p.consumed == x.len() && p.produced < out.len(), || {
        Failure::new("m38", "a call with room for every output kept input")
    })?;
    Ok(p.produced)
}

/// The calls of chunk plan `seed` (input chunks of 1 to `MAX_IN`, output chunks of 1 to
/// `buf.len()`); the outputs are collected in `out`. Returns the count.
fn chunked<T: Sample, K: SymbolSync<T>>(
    kernel: &K,
    st: &mut SymbolSyncState<T>,
    x: &[T],
    out: &mut [T],
    buf: &mut [T],
    seed: u64,
) -> Result<usize, Failure> {
    let mut ins = ChunkPlan::new(seed, MAX_IN).sizes();
    let mut outs = ChunkPlan::new(seed.wrapping_add(1000), buf.len()).sizes();
    let (mut pos, mut n) = (0usize, 0usize);
    while pos < x.len() {
        let a = ins.next().unwrap_or(1).min(x.len() - pos);
        let m = outs.next().unwrap_or(1).clamp(1, buf.len());
        let chunk = get(x, pos, pos + a)?;
        let b = get_mut(buf, 0, m)?;
        let p = kernel.symbol_sync(st, chunk, b)?;
        ensure(p.consumed <= a && p.produced <= m, || {
            Failure::new("m38", "progress beyond the chunk")
        })?;
        ensure(p.produced == m || p.consumed == a, || {
            Failure::new("m38", "a call stalled with room in the output")
        })?;
        get_mut(out, n, n + p.produced)?.copy_from_slice(get(b, 0, p.produced)?);
        n += p.produced;
        pos += p.consumed;
    }
    Ok(n)
}

/// A test signal: `modulation` with `sps` samples per symbol, symbol 0 at `offset`, from random
/// bits, plus complex noise of `sigma` per component, generated in pieces of 64 samples (the
/// generator depends only on the absolute index). `bits` is scratch for the bits.
fn psk_signal<T: Parts>(
    x: &mut [T],
    bits: &mut [u8],
    modulation: Psk,
    sps: f64,
    offset: f64,
    sigma: f64,
    seed: u64,
) {
    let mut rng = Pcg32::new(seed, 38);
    for b in bits.iter_mut() {
        *b = (rng.next_u32() >> 31) as u8;
    }
    let mut g = PskTest::new(modulation, sps, 0.35);
    g.offset = offset;
    let mut tmp = [Complex64::new(0.0, 0.0); 64];
    for (k, piece) in x.chunks_mut(64).enumerate() {
        let t = tmp.get_mut(..piece.len()).unwrap_or(&mut []);
        g.generate(bits, (k * 64) as u64, t);
        for (v, s) in piece.iter_mut().zip(t.iter()) {
            let noise = Complex64::new(sigma * rng.normal(), sigma * rng.normal());
            *v = T::from_complex(*s + noise);
        }
    }
}

/// One configuration of a kernel case: Fallback against Reference (outputs and final state),
/// the facade against Fallback, the facade in 20 chunk plans.
fn compare_all<T: Parts>(
    case: &'static str,
    x: &[T],
    out: &mut [T],
    other: &mut [T],
    buf: &mut [T],
    sps: f64,
    ted: Ted,
    plan_base: u64,
) -> Result<usize, Failure>
where
    Fallback: SymbolSync<T>,
    Reference: SymbolSync<T>,
    Sdrk: SymbolSync<T>,
{
    let fail = |msg: &'static str| Failure::new(case, msg);
    let mut fb = state::<T>(sps, ted, MAX_DEVIATION)?;
    let n = whole(&Fallback, &mut fb, x, out)?;
    let mut rf = state::<T>(sps, ted, MAX_DEVIATION)?;
    let m = whole(&Reference, &mut rf, x, other)?;
    ensure(same(get(out, 0, n)?, get(other, 0, m)?), || {
        fail("reference differs from the fallback").detail(format_args!("sps {sps}, {ted:?}"))
    })?;
    ensure(fb == rf, || fail("final state of the reference differs"))?;
    let mut fc = state::<T>(sps, ted, MAX_DEVIATION)?;
    let m = whole(&Sdrk, &mut fc, x, other)?;
    ensure(same(get(out, 0, n)?, get(other, 0, m)?) && fc == fb, || {
        fail("facade differs from the fallback")
    })?;
    for k in 0..PLANS {
        let mut st = state::<T>(sps, ted, MAX_DEVIATION)?;
        let m = chunked(&Sdrk, &mut st, x, other, buf, plan_base + k)?;
        ensure(same(get(out, 0, n)?, get(other, 0, m)?) && st == fb, || {
            fail("output depends on the block size")
                .detail(format_args!("sps {sps}, {ted:?}, plan {k}"))
        })?;
    }
    // About one output per symbol.
    let want = x.len() as f64 / sps;
    ensure(
        (n as f64 - want).abs() <= want * MAX_DEVIATION + 3.0,
        || fail("output count far from one per symbol").detail(format_args!("{n}, sps {sps}")),
    )?;
    Ok(n)
}

/// `C32`: Gardner on BPSK at sps 4.1 and on QPSK at sps 3.3, Mueller–Müller on QPSK at sps 2.
pub(super) fn symbol_sync_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "symbol_sync_c32";
    let mut c = arena.carver();
    let x = c.take::<C32>(N)?;
    let out = c.take::<C32>(OUT)?;
    let other = c.take::<C32>(OUT)?;
    let buf = c.take::<C32>(MAX_OUT)?;
    let bits = c.take::<u8>(N)?;
    let configs = [
        (Psk::Bpsk, 4.1, 1.3, Ted::Gardner),
        (Psk::Qpsk, 3.3, 0.4, Ted::Gardner),
        (Psk::Qpsk, 2.0, 0.74, Ted::MuellerMuller),
    ];
    for (j, (modulation, sps, offset, ted)) in configs.into_iter().enumerate() {
        psk_signal(x, bits, modulation, sps, offset, 0.05, 3800 + j as u64);
        compare_all(CASE, x, out, other, buf, sps, ted, 38_000 + 100 * j as u64)?;
    }
    Ok(())
}

/// `f32`: Gardner on real BPSK at sps 8 and 5.5, Mueller–Müller at sps 2.5.
pub(super) fn symbol_sync_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "symbol_sync_f32";
    let mut c = arena.carver();
    let x = c.take::<f32>(N)?;
    let out = c.take::<f32>(OUT)?;
    let other = c.take::<f32>(OUT)?;
    let buf = c.take::<f32>(MAX_OUT)?;
    let bits = c.take::<u8>(N)?;
    let configs = [
        (8.0, 2.0, Ted::Gardner),
        (5.5, 0.0, Ted::Gardner),
        (2.5, 1.1, Ted::MuellerMuller),
    ];
    for (j, (sps, offset, ted)) in configs.into_iter().enumerate() {
        psk_signal(x, bits, Psk::Bpsk, sps, offset, 0.05, 3810 + j as u64);
        compare_all(CASE, x, out, other, buf, sps, ted, 38_500 + 100 * j as u64)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Extreme inputs
// ---------------------------------------------------------------------------------------

const SPECIALS: [f32; 7] = [
    0.0,
    -0.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE / 2.0,
    1.0,
];

fn special(i: usize) -> f32 {
    SPECIALS.get(i % SPECIALS.len()).copied().unwrap_or(0.0)
}

/// `+1` for the first `sps` samples, `-1` for the next, and so on (alternating symbols of
/// exactly `sps` samples, `sps` an integer), scaled by `amp`.
fn square(i: usize, sps: usize, amp: f32) -> f32 {
    if (i / sps).is_multiple_of(2) {
        amp
    } else {
        -amp
    }
}

/// Runs Fallback and Reference on `x` and requires equal outputs (NaN equals NaN), an `omega`
/// within its limits and finite, and equal estimates. Returns the output count.
fn agree<T: Parts>(
    case: &'static str,
    msg: &'static str,
    x: &[T],
    o1: &mut [T],
    o2: &mut [T],
    sps: f64,
    ted: Ted,
) -> Result<(usize, SymbolSyncState<T>), Failure>
where
    Fallback: SymbolSync<T>,
    Reference: SymbolSync<T>,
{
    let mut a = state::<T>(sps, ted, MAX_DEVIATION)?;
    let mut b = state::<T>(sps, ted, MAX_DEVIATION)?;
    let n = whole(&Fallback, &mut a, x, o1)?;
    let m = whole(&Reference, &mut b, x, o2)?;
    ensure(same(get(o1, 0, n)?, get(o2, 0, m)?), || {
        Failure::new(case, msg).detail(format_args!("{ted:?}: the implementations differ"))
    })?;
    let w = a.sps_estimate();
    ensure(
        w.to_bits() == b.sps_estimate().to_bits()
            && w.is_finite()
            && (w - sps).abs() <= a.gains().max_integ,
        || Failure::new(case, msg).detail(format_args!("{ted:?}: omega {w}")),
    )?;
    Ok((n, a))
}

/// Silence, full scale, values beyond full scale, NaN and infinities, `sps` at its lower bound 2
/// and `max_deviation` 0, for both detectors and both sample types.
pub(super) fn extremes_m38(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m38";
    const M: usize = 800;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let o1 = c.take::<C32>(M)?;
    let o2 = c.take::<C32>(M)?;
    let r = c.take::<f32>(M)?;
    let p1 = c.take::<f32>(M)?;
    let p2 = c.take::<f32>(M)?;
    let buf = c.take::<f32>(MAX_OUT)?;
    let bits = c.take::<u8>(M / 4 + 1)?;
    let teds = [Ted::Gardner, Ted::MuellerMuller];

    for ted in teds {
        // Silence: zeros out, about one per symbol, and the loop does not move.
        x.fill(C32::new(0.0, 0.0));
        r.fill(0.0);
        let (n, st) = agree(CASE, "silence c32", x, o1, o2, 4.0, ted)?;
        ensure(
            n.abs_diff(M / 4) <= 1
                && get(o1, 0, n)?.iter().all(|v| v.re == 0.0 && v.im == 0.0)
                && st.sps_estimate() == 4.0,
            || fail("silence c32 did not give zeros with a resting loop"),
        )?;
        let (n, st) = agree(CASE, "silence f32", r, p1, p2, 4.0, ted)?;
        ensure(
            n.abs_diff(M / 4) <= 1
                && get(p1, 0, n)?.iter().all(|v| *v == 0.0)
                && st.sps_estimate() == 4.0,
            || fail("silence f32 did not give zeros with a resting loop"),
        )?;

        // Full scale: alternating symbols +-1 of 4 samples, in every sign pattern of C32. The
        // cubic interpolation of +-1 stays within its overshoot of 1.25.
        for pattern in 0..4usize {
            for (i, v) in x.iter_mut().enumerate() {
                let s = square(i, 4, 1.0);
                *v = match pattern {
                    0 => C32::new(s, s),
                    1 => C32::new(s, -s),
                    2 => C32::new(s, 0.0),
                    _ => C32::new(0.0, -s),
                };
            }
            let (n, _) = agree(CASE, "full scale c32", x, o1, o2, 4.0, ted)?;
            ensure(
                get(o1, 0, n)?
                    .iter()
                    .all(|v| v.re.abs() <= 1.25 && v.im.abs() <= 1.25),
                || fail("full scale c32 overshoots"),
            )?;
        }
        for (i, v) in r.iter_mut().enumerate() {
            *v = square(i, 4, 1.0);
        }
        let (n, _) = agree(CASE, "full scale f32", r, p1, p2, 4.0, ted)?;
        ensure(get(p1, 0, n)?.iter().all(|v| v.abs() <= 1.25), || {
            fail("full scale f32 overshoots")
        })?;

        // Far beyond full scale: the interpolation and the detector overflow to infinities and
        // NaN; a non-finite error counts as 0 (D533), so the loop stays within its limits.
        for (i, v) in r.iter_mut().enumerate() {
            *v = square(i, 4, f32::MAX);
        }
        agree(CASE, "f32::MAX", r, p1, p2, 4.0, ted)?;
        for (i, v) in x.iter_mut().enumerate() {
            let s = square(i, 4, f32::MAX);
            *v = C32::new(s, -s);
        }
        agree(CASE, "f32::MAX c32", x, o1, o2, 4.0, ted)?;

        // NaN, infinities and a denormal in every phase of either component.
        for offset in 0..SPECIALS.len() {
            for (i, v) in x.iter_mut().enumerate() {
                *v = C32::new(special(i / 3 + offset), special(i / 5 + offset + 3));
            }
            agree(CASE, "non-finite c32", x, o1, o2, 4.0, ted)?;
            for (i, v) in r.iter_mut().enumerate() {
                *v = special(i / 2 + offset);
            }
            agree(CASE, "non-finite f32", r, p1, p2, 4.0, ted)?;
        }

        // A burst of NaN in a clean signal: the outputs of the burst may be NaN, the loop keeps
        // its state and the outputs are finite again after it.
        for (i, v) in r.iter_mut().enumerate() {
            *v = if (300..310).contains(&i) {
                f32::NAN
            } else {
                square(i, 4, 1.0)
            };
        }
        let (n, _) = agree(CASE, "NaN burst", r, p1, p2, 4.0, ted)?;
        ensure(
            get(p1, 310 / 4 + 4, n)?.iter().all(|v| v.is_finite()),
            || fail("the outputs after a NaN burst are not finite"),
        )?;

        // sps at its lower bound 2: Fallback, Reference and the chunked facade agree.
        let mut rng = Pcg32::new(3820, 2);
        for v in r.iter_mut() {
            *v = if rng.next_u32() >> 31 == 0 { 1.0 } else { -1.0 };
        }
        // Symbols of 2 samples: repeat every value once.
        for i in (0..M).rev() {
            let v = r.get(i / 2).copied().unwrap_or(0.0);
            if let Some(d) = r.get_mut(i) {
                *d = v;
            }
        }
        let (n, end) = agree(CASE, "sps 2", r, p1, p2, 2.0, ted)?;
        ensure(n.abs_diff(M / 2) <= 2, || {
            fail("sps 2 did not give one output per two samples").detail(format_args!("{n}"))
        })?;
        for k in 0..PLANS {
            let mut st = state::<f32>(2.0, ted, MAX_DEVIATION)?;
            let m = chunked(&Sdrk, &mut st, r, p2, buf, 38_900 + k)?;
            ensure(same(get(p1, 0, n)?, get(p2, 0, m)?) && st == end, || {
                fail("sps 2: output depends on the block size").detail(format_args!("plan {k}"))
            })?;
        }
        ensure(
            SymbolSyncState::<f32>::new(1.999_999, ted, LOOP_BW, DAMPING, 0.0, 1.0).is_err(),
            || fail("sps below 2 was accepted"),
        )?;

        // max_deviation 0: omega stays exactly sps on a signal with another symbol rate, and the
        // implementations still agree.
        let mut g = PskTest::new(Psk::Bpsk, 4.1, 0.35);
        g.offset = 0.6;
        for b in bits.iter_mut() {
            *b = (rng.next_u32() >> 31) as u8;
        }
        let mut tmp = [Complex64::new(0.0, 0.0); 64];
        for (k, piece) in r.chunks_mut(64).enumerate() {
            let t = tmp.get_mut(..piece.len()).unwrap_or(&mut []);
            g.generate(bits, (k * 64) as u64, t);
            for (v, s) in piece.iter_mut().zip(t.iter()) {
                *v = s.re as f32;
            }
        }
        let mut a = state::<f32>(4.0, ted, 0.0)?;
        let mut b = state::<f32>(4.0, ted, 0.0)?;
        let n = whole(&Fallback, &mut a, r, p1)?;
        let m = whole(&Reference, &mut b, r, p2)?;
        ensure(same(get(p1, 0, n)?, get(p2, 0, m)?) && a == b, || {
            fail("max_deviation 0: the implementations differ")
        })?;
        ensure(
            a.sps_estimate().to_bits() == 4.0f64.to_bits() && a.gains().max_integ == 0.0,
            || fail("max_deviation 0: omega moved").detail(format_args!("{}", a.sps_estimate())),
        )?;
    }
    Ok(())
}
