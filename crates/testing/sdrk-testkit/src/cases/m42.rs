//! Suite cases of M42: the kernels `rds_shape` and `stereo_encode_rds` and their extreme inputs.
//!
//! Both kernels are bit-exact and not accelerated (the facade is the fallback). The kernel cases
//! compare `Fallback` with `Reference` (output and final state) and the facade with `Fallback`,
//! and run the facade in 20 chunk plans (seed 580 for the shaper, 581 for the encoder; chunks of
//! 1 to 500 samples); every run must give the output and the final state of the run in one
//! call. `rds_shape_f32` also checks the shaper against the exact pulse sum in f64 (the error of
//! the table, D771) and `stereo_encode_rds_c32` the RDS term against `stereo_encode`
//! and an f64 `sin(3 phi)`. The bits come from `Pcg32`; no files are read, so the cases run on
//! microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use sdrk::{Rate, Sdrk};
use sdrk_core::kernels::rds::{RDS_TABLE_LEN, RdsShape, RdsShapeState, rds_pulse_table};
use sdrk_core::kernels::stereo::{
    StereoEncode, StereoEncodeRds, StereoEncodeRdsState, StereoEncodeState,
};
use sdrk_core::{C32, Fallback, Reference};

use super::ensure;
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Chunk plans per case.
const PLANS: u64 = 20;
/// Samples per signal.
const N: usize = 2048;

fn rate(hz: u64) -> Result<Rate, Failure> {
    Ok(Rate::hz(hz)?)
}

fn split<X>(s: &[X], n: usize) -> Result<(&[X], &[X]), Failure> {
    s.split_at_checked(n)
        .ok_or_else(|| Failure::new("m42", "slice split"))
}

fn split_mut<X>(s: &mut [X], n: usize) -> Result<(&mut [X], &mut [X]), Failure> {
    s.split_at_mut_checked(n)
        .ok_or_else(|| Failure::new("m42", "slice split"))
}

/// Bit equality of two `f32` slices; NaN equals NaN (the payload is not part of the contract).
fn same_f32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// A bit source over a stored slice: bit `k` is `bits[k]` (0 beyond the end); counts the calls.
struct Bits<'a> {
    bits: &'a [i8],
    next: usize,
}

impl Bits<'_> {
    fn take(&mut self) -> i8 {
        let b = self.bits.get(self.next).copied().unwrap_or(0);
        self.next += 1;
        b
    }
}

// ---------------------------------------------------------------------------------------
// rds_shape
// ---------------------------------------------------------------------------------------

/// The exact pulse `p(x)` of the table (before normalization), in f64, for the reference sum.
fn pulse_exact(x: f64) -> f64 {
    let h = |x: f64| {
        let d = 1.0 - 64.0 * x * x;
        if libm::fabs(d) < 1e-12 {
            2.0
        } else {
            8.0 * libm::cos(4.0 * core::f64::consts::PI * x) / (core::f64::consts::PI * d)
        }
    };
    if (-2.0..2.0).contains(&x) {
        h(x + 0.25) - h(x - 0.25)
    } else {
        0.0
    }
}

/// Fallback and Reference agree bit for bit (output, state, bits fetched) at 192 kHz and at
/// 50 kHz (a non-integer number of samples per bit), the facade equals the fallback in 20 chunk
/// plans, and the output is the exact pulse sum within 2e-4 (D771).
pub(super) fn rds_shape_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "rds_shape_f32";
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let table = c.take::<f32>(RDS_TABLE_LEN)?;
    rds_pulse_table(table)?;
    let table: &[f32] = table;
    let bits = c.take::<i8>(64)?;
    let fb = c.take::<f32>(N)?;
    let rf = c.take::<f32>(N)?;
    let mut rng = Pcg32::new(580, 1);
    for b in bits.iter_mut() {
        *b = if rng.next_u32() & 1 == 1 { 1 } else { -1 };
    }
    let bits: &[i8] = bits;
    let peak = sdrk_core::kernels::rds::rds_pulse_peak();
    for fs in [192_000u64, 50_000] {
        let (mut a, mut b) = (
            RdsShapeState::new(table, rate(fs)?)?,
            RdsShapeState::new(table, rate(fs)?)?,
        );
        let (mut sa, mut sb) = (Bits { bits, next: 0 }, Bits { bits, next: 0 });
        Fallback.rds_shape(&mut a, &mut || sa.take(), fb)?;
        Reference.rds_shape(&mut b, &mut || sb.take(), rf)?;
        ensure(
            bit_exact(fb, rf).is_none() && a == b && sa.next == sb.next,
            || fail("reference differs from the fallback").detail(format_args!("{fs} Hz")),
        )?;
        // Bits 0 .. floor(x_last) + 2 were fetched.
        let x_last = ((N - 1) as f64 * 1187.5) / fs as f64;
        ensure(sa.next == libm::floor(x_last) as usize + 3, || {
            fail("wrong number of bits fetched").detail(format_args!("{}", sa.next))
        })?;
        for seed in 0..PLANS {
            let plan = ChunkPlan::new(580 * 100 + seed, 1 + (seed as usize * 37) % 500);
            let mut st = RdsShapeState::new(table, rate(fs)?)?;
            let mut src = Bits { bits, next: 0 };
            let (mut o, mut done) = (&mut rf[..], 0usize);
            for size in plan.sizes() {
                if done == N {
                    break;
                }
                let n = size.min(N - done);
                let (head, tail) = split_mut(o, n)?;
                Sdrk.rds_shape(&mut st, &mut || src.take(), head)?;
                o = tail;
                done += n;
            }
            ensure(bit_exact(fb, rf).is_none() && st == a, || {
                fail("output depends on the block size")
            })?;
        }
        // The exact pulse sum: bits k0 .. k0 + 3, peak-normalized.
        let mut worst = 0.0f64;
        for (n, &y) in fb.iter().enumerate() {
            let x = (n as f64 * 1187.5) / fs as f64;
            let k0 = libm::floor(x) as i64 - 1;
            let mut want = 0.0;
            for k in k0..k0 + 4 {
                if let Ok(i) = usize::try_from(k) {
                    let a = f64::from(bits.get(i).copied().unwrap_or(0));
                    want += a * pulse_exact(x - k as f64) / peak;
                }
            }
            worst = worst.max(libm::fabs(f64::from(y) - want));
        }
        ensure(worst <= 2e-4, || {
            fail("the shaper is too far from the exact pulse sum")
                .detail(format_args!("{fs} Hz: {worst:e}"))
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// stereo_encode_rds
// ---------------------------------------------------------------------------------------

const ENC_FS: u64 = 192_000;
const ENC_MAIN: f32 = 0.87;
const ENC_PILOT: f32 = 0.09;
const ENC_RDS: f32 = 0.04;
/// Start phase of the pilot (a quarter turn).
const ENC_PHASE0: u32 = 0x4000_0000;

fn enc_state(rds: f32) -> Result<StereoEncodeRdsState, Failure> {
    let mut st = StereoEncodeRdsState::new(19_000.0, ENC_MAIN, ENC_PILOT, rds, rate(ENC_FS)?)?;
    st.set_phase(ENC_PHASE0);
    Ok(st)
}

/// Random audio within 0.7 and a random RDS baseband within 1.
fn enc_fill(x: &mut [C32], b: &mut [f32], rng: &mut Pcg32) {
    for (v, r) in x.iter_mut().zip(b.iter_mut()) {
        *v = C32::new(
            ((rng.next_f64() * 2.0 - 1.0) * 0.7) as f32,
            ((rng.next_f64() * 2.0 - 1.0) * 0.7) as f32,
        );
        *r = (rng.next_f64() * 2.0 - 1.0) as f32;
    }
}

/// Fallback and Reference agree bit for bit (output and phase), the facade equals the fallback
/// in 20 chunk plans, the RDS level 0 gives `stereo_encode` bit for bit, and the difference to
/// `stereo_encode` is `rds b sin(3 phi)` within 1e-6.
pub(super) fn stereo_encode_rds_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "stereo_encode_rds_c32";
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let input = c.take::<C32>(N)?;
    let rds = c.take::<f32>(N)?;
    let fb = c.take::<f32>(N)?;
    let rf = c.take::<f32>(N)?;
    let mut rng = Pcg32::new(581, 1);
    enc_fill(input, rds, &mut rng);
    let (input, rds): (&[C32], &[f32]) = (input, rds);
    let (mut a, mut b) = (enc_state(ENC_RDS)?, enc_state(ENC_RDS)?);
    Fallback.stereo_encode_rds(&mut a, input, rds, fb)?;
    Reference.stereo_encode_rds(&mut b, input, rds, rf)?;
    ensure(bit_exact(fb, rf).is_none() && a == b, || {
        fail("reference differs from the fallback")
    })?;
    ensure(
        fb.iter()
            .all(|y| y.abs() <= ENC_MAIN * 0.7 + ENC_PILOT + ENC_RDS + 1e-6),
        || fail("the multiplex peak is too high"),
    )?;
    for seed in 0..PLANS {
        let plan = ChunkPlan::new(581 * 100 + seed, 1 + (seed as usize * 37) % 500);
        let mut st = enc_state(ENC_RDS)?;
        let (mut i, mut r, mut o) = (input, rds, &mut rf[..]);
        for size in plan.sizes() {
            if i.is_empty() {
                break;
            }
            let n = size.min(i.len());
            let (ih, it) = split(i, n)?;
            let (rh, rt) = split(r, n)?;
            let (oh, ot) = split_mut(o, n)?;
            Sdrk.stereo_encode_rds(&mut st, ih, rh, oh)?;
            (i, r, o) = (it, rt, ot);
        }
        ensure(bit_exact(fb, rf).is_none() && st == a, || {
            fail("output depends on the block size")
        })?;
    }
    let mut st = enc_state(ENC_RDS)?;
    Sdrk.stereo_encode_rds(&mut st, input, rds, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        fail("facade differs from the fallback")
    })?;

    // The RDS level 0 is `stereo_encode`; the difference at 0.04 is the RDS term.
    let plain = c.take::<f32>(N)?;
    let mut enc = StereoEncodeState::new(19_000.0, ENC_MAIN, ENC_PILOT, rate(ENC_FS)?)?;
    enc.set_phase(ENC_PHASE0);
    let inc = enc.increment();
    Sdrk.stereo_encode(&mut enc, input, plain)?;
    let mut st = enc_state(0.0)?;
    Sdrk.stereo_encode_rds(&mut st, input, rds, rf)?;
    ensure(same_f32(plain, rf) && st.phase() == enc.phase(), || {
        fail("the RDS level 0 differs from stereo_encode")
    })?;
    let mut worst = 0.0f64;
    for (n, ((&y, &p), &b)) in fb.iter().zip(plain.iter()).zip(rds).enumerate() {
        let phase = ENC_PHASE0.wrapping_add(inc.wrapping_mul(n as u32));
        let phi = 2.0 * core::f64::consts::PI * (f64::from(phase) / 4_294_967_296.0);
        let want = f64::from(ENC_RDS) * f64::from(b) * libm::sin(3.0 * phi);
        worst = worst.max(libm::fabs(f64::from(y) - f64::from(p) - want));
    }
    ensure(worst <= 1e-6, || {
        fail("the RDS term is not rds b sin(3 phi)").detail(format_args!("{worst:e}"))
    })
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

/// The `f32` specials cycled with different offsets.
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

pub(super) fn extremes_m42(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m42";
    const M: usize = 400;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let table = c.take::<f32>(RDS_TABLE_LEN)?;
    rds_pulse_table(table)?;
    let table: &[f32] = table;
    let x = c.take::<C32>(M)?;
    let r = c.take::<f32>(M)?;
    let y1 = c.take::<f32>(M)?;
    let y2 = c.take::<f32>(M)?;

    // Shaper: zero bits give exact zeros; the extreme amplitudes -128 and 127 give the pulse
    // sum scaled (finite, the implementations agree); the lowest rate and a high rate work.
    for fs in [4_750u64, 192_000, 2_400_000] {
        let mut st = RdsShapeState::new(table, rate(fs)?)?;
        Sdrk.rds_shape(&mut st, &mut || 0, y1)?;
        ensure(y1.iter().all(|v| v.to_bits() == 0), || {
            fail("zero bits did not give zeros").detail(format_args!("{fs} Hz"))
        })?;
        let (mut a, mut b) = (
            RdsShapeState::new(table, rate(fs)?)?,
            RdsShapeState::new(table, rate(fs)?)?,
        );
        let (mut ka, mut kb) = (0u32, 0u32);
        let mut src_a = || {
            ka += 1;
            if ka % 2 == 0 { i8::MIN } else { i8::MAX }
        };
        let mut src_b = || {
            kb += 1;
            if kb % 2 == 0 { i8::MIN } else { i8::MAX }
        };
        Fallback.rds_shape(&mut a, &mut src_a, y1)?;
        Reference.rds_shape(&mut b, &mut src_b, y2)?;
        ensure(same_f32(y1, y2) && a == b, || {
            fail("shaper: implementations differ at the extreme amplitudes")
        })?;
        ensure(
            y1.iter().all(|v| v.is_finite() && v.abs() <= 4.0 * 128.0),
            || fail("shaper: extreme amplitudes gave a bad output"),
        )?;
    }
    ensure(RdsShapeState::new(table, rate(4_749)?).is_err(), || {
        fail("a rate below 4750 Hz was accepted")
    })?;

    // A reset restarts with bit 0: the same output as a new state.
    let mut st = RdsShapeState::new(table, rate(192_000)?)?;
    let mut k = 0u32;
    Sdrk.rds_shape(
        &mut st,
        &mut || {
            k += 1;
            if k.is_multiple_of(3) { 1 } else { -1 }
        },
        y1,
    )?;
    st.reset();
    let mut k = 0u32;
    Sdrk.rds_shape(
        &mut st,
        &mut || {
            k += 1;
            if k.is_multiple_of(3) { 1 } else { -1 }
        },
        y2,
    )?;
    ensure(same_f32(y1, y2), || {
        fail("reset does not restart the shaper")
    })?;

    // Encoder, silence: the pilot alone without RDS; with RDS the RDS term on top.
    x.fill(C32::new(0.0, 0.0));
    r.fill(0.0);
    let mut es = enc_state(ENC_RDS)?;
    Sdrk.stereo_encode_rds(&mut es, x, r, y1)?;
    ensure(
        y1.iter().all(|v| v.abs() <= ENC_PILOT) && y1.iter().any(|v| v.abs() > 0.99 * ENC_PILOT),
        || fail("encoder: silence did not give the pilot alone"),
    )?;
    let mut es = StereoEncodeRdsState::new(19_000.0, ENC_MAIN, 0.0, 0.2, rate(ENC_FS)?)?;
    Sdrk.stereo_encode_rds(&mut es, x, r, y1)?;
    ensure(y1.iter().all(|v| *v == 0.0), || {
        fail("encoder: silence without pilot did not give zeros")
    })?;

    // Full scale: audio and RDS at +-1 stay within main + pilot + rds.
    for pattern in 0..4usize {
        for (i, (v, b)) in x.iter_mut().zip(r.iter_mut()).enumerate() {
            let s = if (i + pattern) % 2 == 0 { 1.0 } else { -1.0 };
            *v = match pattern {
                0 => C32::new(1.0, 1.0),
                1 => C32::new(1.0, -1.0),
                2 => C32::new(s, s),
                _ => C32::new(s, -s),
            };
            *b = if pattern % 2 == 0 { s } else { 1.0 };
        }
        let (mut ea, mut eb) = (enc_state(ENC_RDS)?, enc_state(ENC_RDS)?);
        Fallback.stereo_encode_rds(&mut ea, x, r, y1)?;
        Reference.stereo_encode_rds(&mut eb, x, r, y2)?;
        ensure(same_f32(y1, y2) && ea == eb, || {
            fail("encoder: implementations differ at full scale")
        })?;
        ensure(
            y1.iter()
                .all(|v| v.abs() <= ENC_MAIN + ENC_PILOT + ENC_RDS + 1e-6),
            || fail("encoder: full scale exceeds main + pilot + rds"),
        )?;
    }

    // NaN, infinities and a denormal in the audio and in the RDS input: the implementations
    // agree bit for bit and a non-finite sample passes into its own output only.
    for offset in 0..SPECIALS.len() {
        for (i, (v, b)) in x.iter_mut().zip(r.iter_mut()).enumerate() {
            *v = C32::new(special(i / 3 + offset), special(i / 5 + offset + 3));
            *b = special(i / 7 + offset + 1);
        }
        let (mut ea, mut eb) = (enc_state(ENC_RDS)?, enc_state(ENC_RDS)?);
        Fallback.stereo_encode_rds(&mut ea, x, r, y1)?;
        Reference.stereo_encode_rds(&mut eb, x, r, y2)?;
        ensure(same_f32(y1, y2) && ea == eb, || {
            fail("encoder: implementations differ on non-finite input")
        })?;
        ensure(
            x.iter().zip(r.iter()).zip(y1.iter()).all(|((v, b), y)| {
                !(v.re.is_finite() && v.im.is_finite() && b.is_finite()) || y.is_finite()
            }),
            || fail("encoder: a non-finite sample spread to another output"),
        )?;
    }
    Ok(())
}
