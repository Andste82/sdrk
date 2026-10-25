//! The blocks.

pub mod convert;
pub mod convert_out;
pub mod dc_block;
pub mod ddc;
pub mod duc;
pub mod fir;
pub mod fm_demod;
pub mod fm_mod;
pub mod resampler;
pub mod scale;
pub mod spectrum;
pub mod squelch;

use sdrk::{Delay, Rate, Ratio, RatioKind};

use crate::block::BlockTiming;
use crate::error::{Error, Result};

/// Largest absolute value of a frequency, rate or width parameter (a schema limit only; the
/// real limits follow from the sample rate).
pub(crate) const HZ_LIMIT: f64 = 1e12;

/// Timing of a same-rate block without delay.
pub(crate) fn unit_timing() -> BlockTiming {
    BlockTiming {
        ratio: Ratio::ONE,
        kind: RatioKind::Constant,
        delay: Delay::ZERO,
    }
}

/// `slice[..n]`, or an internal error.
pub(crate) fn head<T>(slice: &[T], n: usize) -> Result<&[T]> {
    slice
        .get(..n)
        .ok_or(Error::Core(sdrk_core::Error::Internal("block split")))
}

/// `slice[..n]` (mutable), or an internal error.
pub(crate) fn head_mut<T>(slice: &mut [T], n: usize) -> Result<&mut [T]> {
    slice
        .get_mut(..n)
        .ok_or(Error::Core(sdrk_core::Error::Internal("block split")))
}

/// `slice[from..]`, or an internal error.
pub(crate) fn tail<T>(slice: &[T], from: usize) -> Result<&[T]> {
    slice
        .get(from..)
        .ok_or(Error::Core(sdrk_core::Error::Internal("block split")))
}

/// `slice[from..]` (mutable), or an internal error.
pub(crate) fn tail_mut<T>(slice: &mut [T], from: usize) -> Result<&mut [T]> {
    slice
        .get_mut(from..)
        .ok_or(Error::Core(sdrk_core::Error::Internal("block split")))
}

/// Sample rate in Hz as a float (parameters are floats, the rate itself stays exact).
pub(crate) fn hz(rate: Rate) -> f64 {
    rate.as_f64()
}

/// `rate / divisor` as an exact rate.
pub(crate) fn divide_rate(rate: Rate, divisor: usize) -> Result<Rate> {
    Ok(rate.scale(Ratio::new(1, divisor as u64)?)?)
}

/// `rate * factor` as an exact rate.
pub(crate) fn multiply_rate(rate: Rate, factor: usize) -> Result<Rate> {
    Ok(rate.scale(Ratio::new(factor as u64, 1)?)?)
}

/// Checks `|value| <= limit`, otherwise `OutOfRange`.
pub(crate) fn check_symmetric(name: &'static str, value: f64, limit: f64) -> Result<()> {
    if value >= -limit && value <= limit {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name,
            min: -limit,
            max: limit,
            got: value,
        })
    }
}

/// Checks `min <= value <= max` for an integer count parameter.
pub(crate) fn check_count(name: &'static str, value: usize, min: usize, max: usize) -> Result<()> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name,
            min: min as f64,
            max: max as f64,
            got: value as f64,
        })
    }
}

/// Unknown parameter name.
pub(crate) fn unknown<T>() -> Result<T> {
    Err(Error::UnknownParameter)
}
