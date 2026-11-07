//! Kernel traits, one trait per kernel, generic over the sample type.

pub mod agc;
pub mod auto_notch;
pub mod basic;
pub mod biquad;
pub mod channelize;
pub mod convert;
pub mod costas;
pub mod decim;
pub mod demod;
pub mod dot;
pub mod equalize;
pub mod fft;
pub mod filter;
pub mod fused;
pub mod halfband;
pub mod interp;
pub mod math;
pub mod mix;
pub mod modulate;
pub mod noise_blank;
pub mod preemph_limit;
pub mod rds;
pub mod spectrum;
pub mod squelch;
pub mod stereo;
pub mod symbol_sync;

/// `Ok` when both lengths are equal, else `LengthMismatch { expected, got }`.
pub(crate) fn check_len(expected: usize, got: usize) -> crate::error::Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(crate::error::Error::LengthMismatch { expected, got })
    }
}
