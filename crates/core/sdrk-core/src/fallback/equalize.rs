//! Fallback implementation of `equalize`.
//!
//! Bit-exact with `Reference`: [`equalize_with`] with the fallback `dot_complex` for the filter
//! part of the joint path (feedback taps). Without feedback taps it runs the linear path of
//! `kernels::equalize` (D840): runs of 64 symbols on a split copy of the state, the history in
//! place, the filter part and the weight update over the taps padded to a multiple of four
//! (both vectorized by the compiler), the window powers and the steps of the run computed
//! before the symbol loop.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::equalize::{Equalize, EqualizeState, equalize_with};
use crate::types::C32;

impl Equalize for Fallback {
    fn equalize<W, H>(
        &self,
        s: &mut EqualizeState<W, H>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<()>
    where
        W: AsRef<[C32]> + AsMut<[C32]>,
        H: AsRef<[C32]> + AsMut<[C32]>,
    {
        equalize_with(&Fallback, s, input, output)
    }
}
