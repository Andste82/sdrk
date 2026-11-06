//! Reference implementation of `rds_shape`.
//!
//! The kernel is fixed in f32 arithmetic on an f32 table (the float32 model of the golden case),
//! so the reference follows the contract step by step, written as small functions: the position,
//! the window of four bits, one interpolated table value per bit, the sum.

#![allow(clippy::many_single_char_names)]

use crate::Reference;
use crate::error::{Error, Result};
use crate::kernels::rds::{RDS_BIT_RATE_HZ, RdsShape, RdsShapeState};

/// Largest index whose right neighbour is still in the table.
const LAST_INTERVAL: f64 = 1023.0;

/// The position of the sample `n` in bits.
fn position(n: u64, fs: f64) -> f64 {
    let product = n as f64 * RDS_BIT_RATE_HZ;
    product / fs
}

/// The table value at the offset `d = x - k` (in bits) of a sample from the centre of bit `k`.
fn pulse_at(table: &[f32], d: f64) -> Result<f32> {
    let u = (d + 2.0) * 256.0;
    let i = libm::floor(u).min(LAST_INTERVAL);
    let fr = (u - i) as f32;
    let idx = i as usize;
    let (Some(&a), Some(&b)) = (table.get(idx), table.get(idx + 1)) else {
        return Err(Error::Internal("rds pulse table index"));
    };
    let diff = b - a;
    let step = diff * fr;
    Ok(a + step)
}

impl RdsShape for Reference {
    fn rds_shape<Tab: AsRef<[f32]>>(
        &self,
        s: &mut RdsShapeState<Tab>,
        bits: &mut dyn FnMut() -> i8,
        output: &mut [f32],
    ) -> Result<()> {
        let p = s.parts_mut();
        for o in output.iter_mut() {
            let x = position(*p.n, p.fs);
            let k0 = libm::floor(x) as i64 - 1;
            // Fetch every bit up to k0 + 3; the window then holds k0 .. k0 + 3.
            while *p.next <= k0 + 3 {
                let a = if *p.next >= 0 { f32::from(bits()) } else { 0.0 };
                p.fifo.rotate_left(1);
                p.fifo[3] = a;
                *p.next += 1;
            }
            let mut acc = 0.0f32;
            for (j, &a) in p.fifo.iter().enumerate() {
                let k = (k0 + j as i64) as f64;
                let v = pulse_at(p.table, x - k)?;
                let term = a * v;
                acc += term;
            }
            *o = acc;
            *p.n = p.n.wrapping_add(1);
        }
        Ok(())
    }
}
