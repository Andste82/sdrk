//! Fallback implementation of `rds_shape`.
//!
//! The arithmetic is that of the contract, bit for bit (see `Reference`). The window of four bits
//! and the sample index live in locals during the call, and the window moves only when the
//! position passes a bit centre (about every 162 samples at 192 kHz).

#![allow(clippy::many_single_char_names)]

use crate::Fallback;
use crate::error::{Error, Result};
use crate::kernels::rds::{RDS_BIT_RATE_HZ, RDS_TABLE_LEN, RdsShape, RdsShapeState};

#[cold]
fn table_error() -> Error {
    Error::Internal("rds pulse table index")
}

/// `t[i] + (t[i + 1] - t[i]) fr` at `u = (d + 2) 256`, `i = min(floor(u), 1023)`.
///
/// `d >= -2` (the offset of a sample from the centre of a bit in its window), so `u >= 0` and
/// the truncation `u as u32` is `floor(u)` (the saturating cast also covers `u >= 2^32`, which
/// the clamp maps to 1023 like `floor`). This avoids the call of `libm::floor` on targets
/// without a rounding instruction (x86-64 below SSE4.1), bit for bit the same values.
/// The table is a fixed-size array, so with the clamp the compiler drops the bounds checks.
#[inline]
fn interp(table: &[f32; RDS_TABLE_LEN], d: f64) -> f32 {
    let u = (d + 2.0) * 256.0;
    let idx = (u as u32).min(1023) as usize;
    let fr = (u - idx as f64) as f32;
    let (a, b) = match table.get(idx..idx + 2) {
        Some(&[a, b]) => (a, b),
        _ => (0.0, 0.0),
    };
    a + (b - a) * fr
}

/// `n as f64` through `i64` (one instruction on x86-64, where the unsigned conversion takes
/// several); the same value for every `n`.
#[inline]
fn to_f64(n: u64) -> f64 {
    match i64::try_from(n) {
        Ok(m) => m as f64,
        Err(_) => n as f64,
    }
}

impl RdsShape for Fallback {
    fn rds_shape<Tab: AsRef<[f32]>>(
        &self,
        s: &mut RdsShapeState<Tab>,
        bits: &mut dyn FnMut() -> i8,
        output: &mut [f32],
    ) -> Result<()> {
        let p = s.parts_mut();
        let fs = p.fs;
        let table: &[f32; RDS_TABLE_LEN] = p.table.try_into().map_err(|_| table_error())?;
        let (mut n, mut next) = (*p.n, *p.next);
        let [mut a0, mut a1, mut a2, mut a3] = *p.fifo;
        for o in output.iter_mut() {
            let x = (to_f64(n) * RDS_BIT_RATE_HZ) / fs;
            // `x >= 0`, so the truncation is `floor(x)` (exact below 2^63 bits).
            let k0 = x as i64 - 1;
            let f = (k0 + 1) as f64;
            while next <= k0 + 3 {
                let a = if next >= 0 { f32::from(bits()) } else { 0.0 };
                (a0, a1, a2, a3) = (a1, a2, a3, a);
                next += 1;
            }
            let k = (f - 1.0, f, f + 1.0, f + 2.0);
            let (v0, v1, v2, v3) = (
                interp(table, x - k.0),
                interp(table, x - k.1),
                interp(table, x - k.2),
                interp(table, x - k.3),
            );
            *o = (((0.0 + a0 * v0) + a1 * v1) + a2 * v2) + a3 * v3;
            n = n.wrapping_add(1);
        }
        *p.n = n;
        *p.next = next;
        *p.fifo = [a0, a1, a2, a3];
        Ok(())
    }
}
