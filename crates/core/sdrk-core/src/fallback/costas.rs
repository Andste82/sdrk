//! Fallback implementation of `costas`.
//!
//! Bit-exact with `Reference`. The loop is a recursion over the symbols (the phase of every
//! symbol depends on the detector output of the previous one), so the kernel is not
//! vectorized; per symbol it costs one interpolated table pair, a complex rotation, the
//! detector, the loop filter and the lock metric (one division; order 8 adds the octant
//! decision of the 8PSK point to the detector and the metric). The loop keeps the state in
//! locals and is specialized for the order, so the detector and the metric have no branch on
//! it. `floor(v 2^32 + 0.5) as i64 as u32` is the `phase_step` of `fm_mod`, which equals that
//! formula bit for bit without a library call to `floor`.

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::fallback::mix::interp_f32_pair;
use crate::fallback::modulate::phase_step;
use crate::kernels::check_len;
use crate::kernels::costas::{
    Costas, CostasState, LOCK_ALPHA, LOCK_OFF, LOCK_ON, derotate, detector, lock_metric,
};
use crate::num::flush_denormal;
use crate::types::C32;

const TURN: f64 = 4_294_967_296.0;

#[cold]
fn table_error() -> Error {
    Error::Internal("cosine table index")
}

/// `v.clamp(-limit, limit)` for a limit that is not negative, as two selects (a NaN cannot
/// occur: `freq` and `ki e` are finite).
#[inline]
fn clamp_f32(v: f32, limit: f32) -> f32 {
    let v = if v > limit { limit } else { v };
    if v < -limit { -limit } else { v }
}

#[allow(
    clippy::inline_always,
    reason = "each order must be compiled as a loop of its own, without a call"
)]
#[inline(always)]
fn run<const ORDER: u32>(s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()> {
    let (kp, ki, max) = (s.kp, s.ki, s.max);
    let (mut phase, mut freq, mut level, mut locked) = (s.phase, s.freq, s.level, s.locked);
    let mut ok = true;
    for (o, &x) in output.iter_mut().zip(input) {
        let Some((c, sn)) = interp_f32_pair(phase) else {
            ok = false;
            break;
        };
        let y = derotate(x, c, sn);
        *o = y;
        let e = detector(ORDER, y);
        freq = clamp_f32(flush_denormal(freq + ki * e), max);
        phase = phase.wrapping_add(phase_step(f64::from(freq + kp * e), TURN));
        if let Some(m) = lock_metric(ORDER, y) {
            level = flush_denormal(level + LOCK_ALPHA * (m - level));
            if level > LOCK_ON {
                locked = true;
            } else if level < LOCK_OFF {
                locked = false;
            }
        }
    }
    (s.phase, s.freq, s.level, s.locked) = (phase, freq, level, locked);
    if ok { Ok(()) } else { Err(table_error()) }
}

impl Costas for Fallback {
    fn costas(&self, s: &mut CostasState, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        match s.order {
            2 => run::<2>(s, input, output),
            4 => run::<4>(s, input, output),
            _ => run::<8>(s, input, output),
        }
    }
}
