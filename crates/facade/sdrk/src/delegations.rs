//! The single place that maps kernels to implementations.

use crate::{C32, CQ15, Fallback, Q15, Result};
use sdrk_core::kernels::basic::Scale;

crate::delegate! {
    Scale<C32>::scale(&self, input: &[C32], output: &mut [C32], gain: f32) -> Result<()> => Fallback;
    Scale<CQ15>::scale(&self, input: &[CQ15], output: &mut [CQ15], gain: Q15) -> Result<()> => Fallback;
}
