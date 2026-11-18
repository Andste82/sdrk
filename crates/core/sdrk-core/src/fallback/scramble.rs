//! Fallback implementation of the `scramble` kernel.
//!
//! The arguments are checked once per call; every sample is then the per-sample formula of the
//! public helpers `scramble_one_c32`/`scramble_one_cq15`. For `bits ≤ 2` the residue is always
//! 0, so that loop is the quadrant map alone (no table, no product). For `bits ≥ 3` the at most
//! 64 residual phasors of the call are read from the cosine table once into a small table of
//! pairs (the same entries, so the same bits), and the loop selects the input where `r = 0`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::scramble::{
    RESIDUES, Scramble, check_args, phase_index, product_c32, product_cq15, residual_table_c32,
    residual_table_q15, split_index,
};
use crate::num::{c32_quadrant, cq15_quadrant};
use crate::types::{C32, CQ15};

impl Scramble<C32> for Fallback {
    fn scramble(
        &self,
        input: &[C32],
        phase: &[u8],
        bits: u32,
        inverse: bool,
        output: &mut [C32],
    ) -> Result<()> {
        check_args(bits, input.len(), phase.len(), output.len())?;
        let samples = output.iter_mut().zip(input).zip(phase);
        if bits <= 2 {
            for ((o, &x), &p) in samples {
                let (q, _) = split_index(phase_index(p, bits, inverse), bits);
                *o = c32_quadrant(x, q);
            }
        } else {
            let table = residual_table_c32(bits)?;
            let shift = bits - 2;
            let rmask = (1u32 << shift) - 1;
            for ((o, &x), &p) in samples {
                let index = phase_index(p, bits, inverse);
                let residue = index & rmask;
                let (cos, sin) = table
                    .get(residue as usize & (RESIDUES - 1))
                    .copied()
                    .unwrap_or((0.0, 0.0));
                let y = if residue == 0 {
                    x
                } else {
                    product_c32(x, cos, sin)
                };
                *o = c32_quadrant(y, index >> shift);
            }
        }
        Ok(())
    }
}

impl Scramble<CQ15> for Fallback {
    fn scramble(
        &self,
        input: &[CQ15],
        phase: &[u8],
        bits: u32,
        inverse: bool,
        output: &mut [CQ15],
    ) -> Result<()> {
        check_args(bits, input.len(), phase.len(), output.len())?;
        let samples = output.iter_mut().zip(input).zip(phase);
        if bits <= 2 {
            for ((o, &x), &p) in samples {
                let (q, _) = split_index(phase_index(p, bits, inverse), bits);
                *o = cq15_quadrant(x, q);
            }
        } else {
            let table = residual_table_q15(bits)?;
            let shift = bits - 2;
            let rmask = (1u32 << shift) - 1;
            for ((o, &x), &p) in samples {
                let index = phase_index(p, bits, inverse);
                let residue = index & rmask;
                let (cos, sin) = table
                    .get(residue as usize & (RESIDUES - 1))
                    .copied()
                    .unwrap_or((0, 0));
                let y = if residue == 0 {
                    x
                } else {
                    product_cq15(x, cos, sin)
                };
                *o = cq15_quadrant(y, index >> shift);
            }
        }
        Ok(())
    }
}
