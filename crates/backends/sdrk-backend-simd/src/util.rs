//! Small helpers shared by the kernels.

use sdrk_core::{Error, Result};

/// `Ok` when both lengths are equal, else `LengthMismatch { expected, got }`.
#[inline(always)]
pub(crate) fn check_len(expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::LengthMismatch { expected, got })
    }
}

/// Runs `body` on every pair of whole blocks of `in_step` elements of `input` and `out_step`
/// elements of `out`; returns the number of pairs. The plain chunk iterators zip to one counted
/// loop (zipping `&mut ChunksExact` would test both ends in every iteration).
#[inline(always)]
pub(crate) fn blocks<T, U>(
    input: &[T],
    out: &mut [U],
    in_step: usize,
    out_step: usize,
    mut body: impl FnMut(&[T], &mut [U]),
) -> usize {
    let n = (input.len() / in_step).min(out.len() / out_step);
    let (input, _) = input.split_at(n * in_step);
    let (out, _) = out.split_at_mut(n * out_step);
    for (i, o) in input
        .chunks_exact(in_step)
        .zip(out.chunks_exact_mut(out_step))
    {
        body(i, o);
    }
    n
}

/// The error for a violated internal invariant.
#[cold]
pub(crate) fn internal(what: &'static str) -> Error {
    Error::Internal(what)
}
