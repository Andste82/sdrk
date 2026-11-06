//! Fallback implementation of `equalize`.
//!
//! Bit-exact with `Reference`: [`equalize_with`] with the fallback `dot_complex` for the filter
//! part. Per symbol the history moves by one (a copy of `N - 1` samples, at most 62), the dot
//! product and the update of the `N` weights are independent per element, so the compiler may
//! vectorize both without changing a bit.

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
