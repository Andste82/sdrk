//! Kernel traits, one trait per kernel, generic over the sample type.

pub mod basic;
pub mod convert;
pub mod decim;
pub mod demod;
pub mod fft;
pub mod filter;
pub mod mix;
pub mod spectrum;

/// `Ok` when both lengths are equal, else `LengthMismatch { expected, got }`.
pub(crate) fn check_len(expected: usize, got: usize) -> crate::error::Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(crate::error::Error::LengthMismatch { expected, got })
    }
}
