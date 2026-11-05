//! Suite cases of M36: the kernels `preemph_limit_stereo` and `stereo_encode` and their extreme
//! inputs.
//!
//! Both kernels are bit-exact and not accelerated (the facade is the fallback). The kernel cases
//! compare `Fallback` with `Reference` (output and final state) and the facade with `Fallback`,
//! and run the facade in 20 chunk plans; `preemph_limit_stereo_c32` also checks the linked gain
//! (test 3 of the milestone) and the mono kernel of M25 on equal channels, and
//! `stereo_encode_c32` the composition of the M30 blocks `nco` (harmonics 1 and 2),
//! `imag_part`, `mul` and `add` (test 4). No files are read, so the cases run on
//! microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use sdrk::{Rate, Sdrk};
use sdrk_blocks::design::preemphasis_coefs;
use sdrk_blocks::{
    AddRealBlock, ArenaStorage, Block, ImagPartBlock, MulRealBlock, NcoBlock, NcoConfig,
};
use sdrk_core::kernels::preemph_limit::{
    PreemphLimit, PreemphLimitState, PreemphLimitStereo, PreemphLimitStereoState,
};
use sdrk_core::kernels::stereo::{StereoEncode, StereoEncodeState};
use sdrk_core::{C32, Fallback, Reference};

use super::ensure;
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Chunk plans per case.
const PLANS: u64 = 20;

fn rate(hz: u64) -> Result<Rate, Failure> {
    Ok(Rate::hz(hz)?)
}

fn split<X>(s: &[X], n: usize) -> Result<(&[X], &[X]), Failure> {
    s.split_at_checked(n)
        .ok_or_else(|| Failure::new("m36", "slice split"))
}

fn split_mut<X>(s: &mut [X], n: usize) -> Result<(&mut [X], &mut [X]), Failure> {
    s.split_at_mut_checked(n)
        .ok_or_else(|| Failure::new("m36", "slice split"))
}

fn sin_turns(turns: f64) -> f64 {
    libm::sin(2.0 * core::f64::consts::PI * (turns - libm::floor(turns)))
}

/// Bit equality of two `f32` slices; NaN equals NaN (the payload is not part of the contract).
fn same_f32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

// ---------------------------------------------------------------------------------------
// preemph_limit_stereo
// ---------------------------------------------------------------------------------------

/// Sample rate of the limiter cases.
const PL_FS: u64 = 48_000;
/// Length of the limiter signal.
const PL_N: usize = 2048;
/// Output peak of the limiter cases.
const PL_LIMIT: f64 = 0.5;
/// Hold and release as the M25 suite cases: the signal attacks, holds, releases and ends idle.
const PL_HOLD_S: f64 = 0.01;
const PL_RELEASE_S: f64 = 0.005;

fn pl_state(limit: f64) -> Result<PreemphLimitStereoState, Failure> {
    let coefs = preemphasis_coefs(50e-6, PL_FS as f64)?;
    Ok(PreemphLimitStereoState::new(
        coefs,
        limit,
        PL_HOLD_S,
        PL_RELEASE_S,
        rate(PL_FS)?,
    )?)
}

fn mono_state(limit: f64) -> Result<PreemphLimitState<f32>, Failure> {
    let coefs = preemphasis_coefs(50e-6, PL_FS as f64)?;
    Ok(PreemphLimitState::new(
        coefs,
        limit,
        PL_HOLD_S,
        PL_RELEASE_S,
        rate(PL_FS)?,
    )?)
}

/// Left: tone bursts of amplitude 1.5 (1 kHz at samples 200 to 500, 8 kHz at 700 to 900) over
/// noise of sigma 0.02; right: a 3 kHz tone of 0.15 plus its own noise of sigma 0.02. Only the
/// left channel drives the limiter.
fn pl_fill(x: &mut [C32]) {
    let mut rng = Pcg32::new(4850, 1);
    for (n, v) in x.iter_mut().enumerate() {
        let t = n as f64 / PL_FS as f64;
        let burst = if (200..500).contains(&n) {
            1.5 * sin_turns(1000.0 * t)
        } else if (700..900).contains(&n) {
            1.5 * sin_turns(8000.0 * t)
        } else {
            0.0
        };
        let left = burst + 0.02 * rng.normal();
        let right = 0.15 * sin_turns(3000.0 * t) + 0.02 * rng.normal();
        *v = C32::new(left as f32, right as f32);
    }
}

fn pl_chunked(
    st: &mut PreemphLimitStereoState,
    input: &[C32],
    out: &mut [C32],
    plan: ChunkPlan,
) -> Result<(), Failure> {
    let (mut i, mut o) = (input, out);
    for size in plan.sizes() {
        if i.is_empty() {
            break;
        }
        let n = size.min(i.len());
        let (ih, it) = split(i, n)?;
        let (oh, ot) = split_mut(o, n)?;
        Sdrk.preemph_limit_stereo(st, ih, oh)?;
        (i, o) = (it, ot);
    }
    Ok(())
}

/// The pre-emphasized input of the 50 us design in `f32` (the arithmetic of the kernel, no
/// input rule: the signal is finite and small).
fn pre_emphasized(input: &[C32], out: &mut [C32]) -> Result<(), Failure> {
    let c = preemphasis_coefs(50e-6, PL_FS as f64)?;
    let (b0, b1, a1) = (c[0] as f32, c[1] as f32, c[2] as f32);
    let (mut x1, mut v1) = (C32::new(0.0, 0.0), C32::new(0.0, 0.0));
    for (x, v) in input.iter().zip(out.iter_mut()) {
        *v = C32::new(
            (b0 * x.re + b1 * x1.re) - a1 * v1.re,
            (b0 * x.im + b1 * x1.im) - a1 * v1.im,
        );
        x1 = *x;
        v1 = *v;
    }
    Ok(())
}

/// Fallback and Reference agree bit for bit (output and state), the limiter attacked and
/// released, no output exceeds the limit, the gain is linked (test 3), equal channels give the
/// mono kernel of M25, and the facade output does not depend on the block size (20 plans).
pub(super) fn preemph_limit_stereo_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "preemph_limit_stereo_c32";
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let input = c.take::<C32>(PL_N)?;
    let fb = c.take::<C32>(PL_N)?;
    let rf = c.take::<C32>(PL_N)?;
    pl_fill(input);
    let (mut a, mut b) = (pl_state(PL_LIMIT)?, pl_state(PL_LIMIT)?);
    Fallback.preemph_limit_stereo(&mut a, input, fb)?;
    Reference.preemph_limit_stereo(&mut b, input, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        fail("reference differs from the fallback")
    })?;
    ensure(a == b, || fail("final state differs"))?;
    let top = f64::from(PL_LIMIT as f32);
    ensure(
        fb.iter()
            .all(|v| f64::from(v.re.abs()) <= top && f64::from(v.im.abs()) <= top),
        || fail("an output sample exceeds the limit"),
    )?;
    ensure(a.gain() == 1.0, || fail("the limiter did not release"))?;

    // Test 3, linked gain: while limiting, out.im / v.im equals out.re / v.re (v the
    // pre-emphasized input), so the quiet right channel is lowered by the factor of the left.
    pre_emphasized(input, rf)?;
    let t = PL_LIMIT as f32;
    let mut limited = 0usize;
    for (o, v) in fb.iter().zip(rf.iter()) {
        let clamped = o.re.abs() >= t || o.im.abs() >= t;
        if clamped || v.re.abs() < 1e-3 || v.im.abs() < 1e-3 {
            continue;
        }
        let (gl, gr) = (o.re / v.re, o.im / v.im);
        ensure((gl - gr).abs() <= 1e-6 * gl.abs(), || {
            fail("the gains of the two channels differ").detail(format_args!("{gl} and {gr}"))
        })?;
        if gl < 0.99 {
            limited += 1;
        }
    }
    ensure(limited > 200, || {
        fail("the right channel was not lowered with the left")
            .detail(format_args!("{limited} samples"))
    })?;

    for seed in 0..PLANS {
        let plan = ChunkPlan::new(4860 + seed, 1 + (seed as usize * 37) % 500);
        let mut st = pl_state(PL_LIMIT)?;
        pl_chunked(&mut st, input, rf, plan)?;
        ensure(bit_exact(fb, rf).is_none() && st == a, || {
            fail("output depends on the block size")
        })?;
    }
    let mut st = pl_state(PL_LIMIT)?;
    Sdrk.preemph_limit_stereo(&mut st, input, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        fail("facade differs from the fallback")
    })?;

    // Equal channels give the mono kernel of M25 in both channels, bit for bit.
    let mono_in = c.take::<f32>(PL_N)?;
    let mono_out = c.take::<f32>(PL_N)?;
    for (m, x) in mono_in.iter_mut().zip(input.iter_mut()) {
        *m = x.re;
        *x = C32::new(x.re, x.re);
    }
    let mut ms = mono_state(PL_LIMIT)?;
    PreemphLimit::<f32>::preemph_limit(&Sdrk, &mut ms, mono_in, mono_out)?;
    let mut st = pl_state(PL_LIMIT)?;
    Sdrk.preemph_limit_stereo(&mut st, input, rf)?;
    ensure(
        rf.iter()
            .zip(mono_out.iter())
            .all(|(s, m)| s.re.to_bits() == m.to_bits() && s.im.to_bits() == m.to_bits())
            && st.gain() == ms.gain(),
        || fail("equal channels differ from the mono kernel"),
    )
}

// ---------------------------------------------------------------------------------------
// stereo_encode
// ---------------------------------------------------------------------------------------

const ENC_FS: u64 = 192_000;
const ENC_N: usize = 1024;
const ENC_MAIN: f32 = 0.9;
const ENC_PILOT: f32 = 0.09;
/// Start phase of the pilot (a quarter turn).
const ENC_PHASE0: u32 = 0x4000_0000;

fn enc_state(pilot: f32) -> Result<StereoEncodeState, Failure> {
    let mut st = StereoEncodeState::new(19_000.0, ENC_MAIN, pilot, rate(ENC_FS)?)?;
    st.set_phase(ENC_PHASE0);
    Ok(st)
}

/// Left 1 kHz 0.6, right 3 kHz 0.4, each plus noise of sigma 0.01.
fn enc_fill(x: &mut [C32]) {
    let mut rng = Pcg32::new(4870, 1);
    for (n, v) in x.iter_mut().enumerate() {
        let t = n as f64 / ENC_FS as f64;
        let left = 0.6 * sin_turns(1000.0 * t) + 0.01 * rng.normal();
        let right = 0.4 * sin_turns(3000.0 * t) + 0.01 * rng.normal();
        *v = C32::new(left as f32, right as f32);
    }
}

fn enc_chunked(
    st: &mut StereoEncodeState,
    input: &[C32],
    out: &mut [f32],
    plan: ChunkPlan,
) -> Result<(), Failure> {
    let (mut i, mut o) = (input, out);
    for size in plan.sizes() {
        if i.is_empty() {
            break;
        }
        let n = size.min(i.len());
        let (ih, it) = split(i, n)?;
        let (oh, ot) = split_mut(o, n)?;
        Sdrk.stereo_encode(st, ih, oh)?;
        (i, o) = (it, ot);
    }
    Ok(())
}

/// The imaginary part of the `nco` block with `harmonic` on the clock `clock`, into `out`
/// (`scratch` holds the complex tone).
fn nco_sine(
    harmonic: usize,
    clock: &[C32],
    scratch: &mut [C32],
    out: &mut [f32],
) -> Result<(), Failure> {
    let fs = rate(ENC_FS)?;
    let cfg = NcoConfig {
        frequency_hz: 19_000.0,
        harmonic,
        phase_turns: 0.25,
    };
    let mut storage = ArenaStorage::new(&mut []);
    let mut nco = NcoBlock::<C32, _>::new(&cfg, fs, &mut storage)?;
    let mut imag = ImagPartBlock::<C32, _>::new(fs, &mut storage)?;
    let p = nco.process(clock, scratch)?;
    let q = imag.process(scratch, out)?;
    ensure(
        p.produced == clock.len() && q.produced == clock.len(),
        || {
            Failure::new(
                "stereo_encode_c32",
                "a block did not process the whole input",
            )
        },
    )
}

/// Fallback and Reference agree bit for bit (output and phase), the facade equals the fallback
/// in 20 chunk plans, the peak stays within `main + pilot`, and the output equals the
/// composition of the M30 blocks within 1e-6 (test 4).
pub(super) fn stereo_encode_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "stereo_encode_c32";
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let input = c.take::<C32>(ENC_N)?;
    let fb = c.take::<f32>(ENC_N)?;
    let rf = c.take::<f32>(ENC_N)?;
    enc_fill(input);
    let (mut a, mut b) = (enc_state(ENC_PILOT)?, enc_state(ENC_PILOT)?);
    Fallback.stereo_encode(&mut a, input, fb)?;
    Reference.stereo_encode(&mut b, input, rf)?;
    ensure(bit_exact(fb, rf).is_none() && a == b, || {
        fail("reference differs from the fallback")
    })?;
    ensure(
        fb.iter().all(|y| y.abs() <= ENC_MAIN * 0.7 + ENC_PILOT),
        || fail("the multiplex peak is too high"),
    )?;
    for seed in 0..PLANS {
        let plan = ChunkPlan::new(4880 + seed, 1 + (seed as usize * 37) % 500);
        let mut st = enc_state(ENC_PILOT)?;
        enc_chunked(&mut st, input, rf, plan)?;
        ensure(bit_exact(fb, rf).is_none() && st == a, || {
            fail("output depends on the block size")
        })?;
    }
    let mut st = enc_state(ENC_PILOT)?;
    Sdrk.stereo_encode(&mut st, input, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        fail("facade differs from the fallback")
    })?;

    // Test 4: nco (harmonics 1 and 2) and imag_part give s1 and s2, mul and add the sum
    // main m + (main d) s2 + pilot s1. The audio matrix (main m, main d) is plain f32.
    let tone = c.take::<C32>(ENC_N)?;
    let s1 = c.take::<f32>(ENC_N)?;
    let s2 = c.take::<f32>(ENC_N)?;
    let main_m = c.take::<f32>(ENC_N)?;
    let main_d = c.take::<f32>(ENC_N)?;
    nco_sine(1, input, tone, s1)?;
    nco_sine(2, input, tone, s2)?;
    for ((x, m), d) in input.iter().zip(main_m.iter_mut()).zip(main_d.iter_mut()) {
        #[allow(
            clippy::manual_midpoint,
            reason = "the contract fixes the f32 sum, then the halving"
        )]
        let half_sum = (x.re + x.im) * 0.5;
        *m = ENC_MAIN * half_sum;
        *d = ENC_MAIN * ((x.re - x.im) * 0.5);
    }
    let fs = rate(ENC_FS)?;
    let mut storage = ArenaStorage::new(&mut []);
    let mut mul = MulRealBlock::<f32, _>::new(fs, &mut storage)?;
    let mut add = AddRealBlock::<f32, _>::new(fs, &mut storage)?;
    // diff = (main d) s2 into rf, then partial = main m + diff into main_d.
    mul.process((&*main_d, &*s2), rf)?;
    add.process((&*main_m, &*rf), main_d)?;
    // tone = pilot s1 into main_m (a constant stream times s1), then the sum into rf.
    main_m.fill(ENC_PILOT);
    mul.process((&*main_m, &*s1), s2)?;
    add.process((&*main_d, &*s2), rf)?;
    let worst = fb
        .iter()
        .zip(rf.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    ensure(worst <= 1e-6, || {
        fail("the encoder differs from the composition of the M30 blocks")
            .detail(format_args!("largest difference {worst}"))
    })
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

/// The `f32` specials cycled per channel with different offsets.
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

/// The identity pre-emphasis (`preemphasis_us` 0 in the block).
const PASS: [f64; 3] = [1.0, 0.0, 0.0];

pub(super) fn extremes_m36(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m36";
    const M: usize = 400;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let o1 = c.take::<C32>(M)?;
    let o2 = c.take::<C32>(M)?;
    let y1 = c.take::<f32>(M)?;
    let y2 = c.take::<f32>(M)?;
    let fs_pl = rate(PL_FS)?;

    // Silence: exact zeros and an idle limiter; the encoder gives zeros without the pilot and
    // the pilot tone alone with it.
    x.fill(C32::new(0.0, 0.0));
    let mut st = pl_state(PL_LIMIT)?;
    Sdrk.preemph_limit_stereo(&mut st, x, o1)?;
    ensure(
        o1.iter()
            .all(|v| v.re.to_bits() == 0 && v.im.to_bits() == 0)
            && st.gain() == 1.0,
        || fail("limiter: silence did not give zeros"),
    )?;
    let mut es = enc_state(0.0)?;
    Sdrk.stereo_encode(&mut es, x, y1)?;
    ensure(y1.iter().all(|v| *v == 0.0), || {
        fail("encoder: silence without pilot did not give zeros")
    })?;
    let mut es = enc_state(ENC_PILOT)?;
    Sdrk.stereo_encode(&mut es, x, y1)?;
    ensure(
        y1.iter().all(|v| v.abs() <= ENC_PILOT) && y1.iter().any(|v| v.abs() > 0.99 * ENC_PILOT),
        || fail("encoder: silence did not give the pilot alone"),
    )?;

    // Full scale in every sign pattern: the limiter output stays within the limit, the encoder
    // within main + pilot (0.99 at the defaults).
    for pattern in 0..4usize {
        for (i, v) in x.iter_mut().enumerate() {
            let s = if (i + pattern) % 2 == 0 { 1.0 } else { -1.0 };
            *v = match pattern {
                0 => C32::new(1.0, 1.0),
                1 => C32::new(1.0, -1.0),
                2 => C32::new(s, s),
                _ => C32::new(s, -s),
            };
        }
        let (mut a, mut b) = (pl_state(PL_LIMIT)?, pl_state(PL_LIMIT)?);
        Fallback.preemph_limit_stereo(&mut a, x, o1)?;
        Reference.preemph_limit_stereo(&mut b, x, o2)?;
        ensure(bit_exact(o1, o2).is_none() && a == b, || {
            fail("limiter: implementations differ at full scale")
        })?;
        ensure(
            o1.iter()
                .all(|v| v.re.abs() <= PL_LIMIT as f32 && v.im.abs() <= PL_LIMIT as f32),
            || fail("limiter: full scale exceeds the limit"),
        )?;
        let (mut ea, mut eb) = (enc_state(ENC_PILOT)?, enc_state(ENC_PILOT)?);
        Fallback.stereo_encode(&mut ea, x, y1)?;
        Reference.stereo_encode(&mut eb, x, y2)?;
        ensure(same_f32(y1, y2) && ea == eb, || {
            fail("encoder: implementations differ at full scale")
        })?;
        ensure(
            y1.iter().all(|v| v.abs() <= ENC_MAIN + ENC_PILOT + 1e-6),
            || fail("encoder: full scale exceeds main + pilot"),
        )?;
    }

    // NaN, infinities and a denormal in every phase of either channel: the implementations
    // agree bit for bit; the limiter output stays finite and within the limit (a non-finite
    // component counts as 0); the encoder passes a non-finite sample into its own output only.
    for offset in 0..SPECIALS.len() {
        for (i, v) in x.iter_mut().enumerate() {
            *v = C32::new(special(i / 3 + offset), special(i / 5 + offset + 3));
        }
        let (mut a, mut b) = (pl_state(PL_LIMIT)?, pl_state(PL_LIMIT)?);
        Fallback.preemph_limit_stereo(&mut a, x, o1)?;
        Reference.preemph_limit_stereo(&mut b, x, o2)?;
        ensure(bit_exact(o1, o2).is_none() && a == b, || {
            fail("limiter: implementations differ on non-finite input")
        })?;
        ensure(
            o1.iter().all(|v| {
                v.re.is_finite()
                    && v.im.is_finite()
                    && v.re.abs() <= PL_LIMIT as f32
                    && v.im.abs() <= PL_LIMIT as f32
            }),
            || fail("limiter: non-finite input gave a bad output"),
        )?;
        let (mut ea, mut eb) = (enc_state(ENC_PILOT)?, enc_state(ENC_PILOT)?);
        Fallback.stereo_encode(&mut ea, x, y1)?;
        Reference.stereo_encode(&mut eb, x, y2)?;
        ensure(same_f32(y1, y2) && ea == eb, || {
            fail("encoder: implementations differ on non-finite input")
        })?;
        ensure(
            x.iter()
                .zip(y1.iter())
                .all(|(v, y)| y.is_finite() == (v.re.is_finite() && v.im.is_finite())),
            || fail("encoder: a non-finite sample spread to another output"),
        )?;
    }

    // preemphasis_us 0 is the identity below the limit (bit for bit).
    let mut rng = Pcg32::new(4890, 3);
    for v in x.iter_mut() {
        *v = C32::new(
            ((rng.next_f64() * 2.0 - 1.0) * 0.9) as f32,
            ((rng.next_f64() * 2.0 - 1.0) * 0.9) as f32,
        );
    }
    let mut st = PreemphLimitStereoState::new(PASS, 1.0, PL_HOLD_S, PL_RELEASE_S, fs_pl)?;
    Sdrk.preemph_limit_stereo(&mut st, x, o1)?;
    ensure(bit_exact(x, o1).is_none() && st.gain() == 1.0, || {
        fail("preemphasis 0 below the limit is not the identity")
    })?;

    // A limit change applies from the next sample: the samples before the change equal the run
    // without it, the sample right after it is already within the new limit, and the stereo
    // run with equal channels equals the mono kernel with the same change.
    const AT: usize = 150;
    for v in x.iter_mut() {
        let s = ((rng.next_f64() * 2.0 - 1.0) * 0.95) as f32;
        *v = C32::new(s, s);
    }
    let mut whole = PreemphLimitStereoState::new(PASS, 1.0, PL_HOLD_S, PL_RELEASE_S, fs_pl)?;
    Sdrk.preemph_limit_stereo(&mut whole, x, o2)?;
    let mut st = PreemphLimitStereoState::new(PASS, 1.0, PL_HOLD_S, PL_RELEASE_S, fs_pl)?;
    let mut ms = PreemphLimitState::<f32>::new(PASS, 1.0, PL_HOLD_S, PL_RELEASE_S, fs_pl)?;
    let (xh, xt) = split(x, AT)?;
    let (oh, ot) = split_mut(o1, AT)?;
    Sdrk.preemph_limit_stereo(&mut st, xh, oh)?;
    st.set_limit(0.3)?;
    Sdrk.preemph_limit_stereo(&mut st, xt, ot)?;
    let (before, _) = split(o2, AT)?;
    let (head, tail) = split(o1, AT)?;
    ensure(bit_exact(before, head).is_none(), || {
        fail("the samples before the limit change differ")
    })?;
    ensure(
        tail.iter().all(|v| v.re.abs() <= 0.3 && v.im.abs() <= 0.3)
            && tail.first().is_some_and(|v| v.re.abs() > 0.0),
        || fail("the new limit does not apply from the next sample"),
    )?;
    let mono_in = c.take::<f32>(M)?;
    let mono_out = c.take::<f32>(M)?;
    for (m, v) in mono_in.iter_mut().zip(x.iter()) {
        *m = v.re;
    }
    let (mh, mt) = split(mono_in, AT)?;
    let (moh, mot) = split_mut(mono_out, AT)?;
    PreemphLimit::<f32>::preemph_limit(&Sdrk, &mut ms, mh, moh)?;
    ms.set_limit(0.3)?;
    PreemphLimit::<f32>::preemph_limit(&Sdrk, &mut ms, mt, mot)?;
    ensure(
        o1.iter()
            .zip(mono_out.iter())
            .all(|(s, m)| s.re.to_bits() == m.to_bits() && s.im.to_bits() == m.to_bits()),
        || fail("the limit change differs from the mono kernel"),
    )?;

    // Random stereo input up to 20 times full scale never exceeds the limit in either channel.
    for (k, limit) in [1.0, 0.7, 0.2].into_iter().enumerate() {
        let tau = [50e-6, 75e-6, 530e-6].get(k).copied().unwrap_or(50e-6);
        let coefs = preemphasis_coefs(tau, PL_FS as f64)?;
        let mut st = PreemphLimitStereoState::new(coefs, limit, 0.001, 0.002, fs_pl)?;
        let t = limit as f32;
        for _ in 0..3 {
            for v in x.iter_mut() {
                *v = C32::new(
                    ((rng.next_f64() * 2.0 - 1.0) * 20.0) as f32,
                    ((rng.next_f64() * 2.0 - 1.0) * 20.0) as f32,
                );
            }
            Sdrk.preemph_limit_stereo(&mut st, x, o1)?;
            ensure(
                o1.iter().all(|v| v.re.abs() <= t && v.im.abs() <= t),
                || fail("20 times full scale exceeds the limit").detail(format_args!("{limit}")),
            )?;
            ensure(o1.iter().any(|v| v.re.abs() > 0.9 * t), || {
                fail("20 times full scale never reached the limit")
            })?;
        }
    }
    Ok(())
}
