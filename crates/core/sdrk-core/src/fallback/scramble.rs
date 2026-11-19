//! Fallback implementation of the `scramble` kernel.
//!
//! The arguments are checked once per call; every sample is then the per-sample formula of the
//! public helpers `scramble_one_c32`/`scramble_one_cq15`. For `bits ≤ 2` that is the quadrant
//! map alone (no table, no product); `C32` at `bits = 1` writes it as one sign mask per sample. For `bits ≥ 3` (M49.1) every sample reads its phasor from
//! the cosine table at `i = k << (12 − bits)` and `(i + 3072) & 4095` and runs the `rotate`
//! product; there is no quadrant map and no selection.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::scramble::{
    Scramble, check_args, phase_index, phasor_c32, phasor_q15, product_c32, product_cq15,
    split_index,
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
        if bits == 1 {
            // The quadrant map with `q = 2 k` (`inverse` keeps `k`, since `−k ≡ k` mod 2): both
            // sign bits flipped where `p & 1` is set, the bits of `c32_quadrant` as one XOR mask
            // that the compiler vectorizes (M49.1 §5, the `scramble_c32_b1_fallback` row).
            for ((o, &x), &p) in samples {
                let m = (u32::from(p) & 1) << 31;
                *o = C32::new(
                    f32::from_bits(x.re.to_bits() ^ m),
                    f32::from_bits(x.im.to_bits() ^ m),
                );
            }
        } else if bits == 2 {
            for ((o, &x), &p) in samples {
                let (q, _) = split_index(phase_index(p, bits, inverse), bits);
                *o = c32_quadrant(x, q);
            }
        } else {
            for ((o, &x), &p) in samples {
                let (c, s) = phasor_c32(phase_index(p, bits, inverse), bits);
                *o = product_c32(x, c, s);
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
            for ((o, &x), &p) in samples {
                let (c, s) = phasor_q15(phase_index(p, bits, inverse), bits);
                *o = product_cq15(x, c, s);
            }
        }
        Ok(())
    }
}
