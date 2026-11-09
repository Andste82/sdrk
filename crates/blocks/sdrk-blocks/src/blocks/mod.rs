//! The blocks.

pub mod add;
pub mod adsb_decode;
pub mod afc;
pub mod agc;
pub mod am_demod;
pub mod am_mod;
pub mod auto_notch;
pub mod band_pulse_detect;
pub mod biquad;
pub mod channel_select;
pub mod channelize;
pub mod convert;
pub mod convert_out;
pub mod costas;
pub mod dc_block;
pub mod ddc;
pub mod decimate;
pub mod deemphasis;
pub mod delay;
pub mod duc;
pub mod equalize;
pub mod fir;
pub mod fm_demod;
pub mod fm_mod;
pub mod fm_stereo_demod;
pub mod fm_stereo_mod;
pub mod fm_stereo_rds_mod;
pub mod frame_decode;
pub mod frame_encode;
pub mod fsk_demod;
pub mod fsk_mod;
pub mod halfband_decimate;
pub mod hdlc_decode;
pub mod hdlc_encode;
pub mod imag_part;
pub mod interpolate;
pub mod mag;
pub mod make_complex;
pub mod morse_decode;
pub mod morse_encode;
pub mod mul;
pub mod nco;
pub mod noise_blank;
pub mod ook_demod;
pub mod ook_mod;
pub mod preemph_limit;
pub mod psk_demod;
pub mod psk_mod;
pub mod pulse_key;
pub mod qam_demod;
pub mod qam_mod;
pub mod quad_demod;
pub mod rds_decode;
pub mod rds_demod;
pub mod rds_encode;
pub mod real_part;
pub mod real_to_iq;
pub mod resample;
pub mod rotate;
pub mod scale;
pub mod sensor_decode;
pub mod sensor_encode;
pub mod slice;
pub mod spectrum;
pub mod squelch;
pub mod ssb_demod;
pub mod ssb_mod;
pub mod stereo_matrix;
pub mod stereo_pll;
pub mod symbol_sync;
pub mod to_complex;

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

/// The common length of the two inputs of a [`Two`](crate::Two) port, or `LengthMismatch`
/// (the lockstep rule).
pub(crate) fn lockstep<A, B>(a: &[A], b: &[B]) -> Result<usize> {
    if a.len() == b.len() {
        Ok(a.len())
    } else {
        Err(Error::Core(sdrk_core::Error::LengthMismatch {
            expected: a.len(),
            got: b.len(),
        }))
    }
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

/// The names of the `whitening` choice of `frame_encode` and `frame_decode`, in index order.
pub(crate) const WHITENING_OPTIONS: &[&str] = &["none", "pn9"];

/// The `whitening` choice index of a [`sdrk_proto::frame::Whitening`].
pub(crate) fn whitening_choice(w: sdrk_proto::frame::Whitening) -> u16 {
    match w {
        sdrk_proto::frame::Whitening::Pn9 => 1,
        _ => 0,
    }
}

/// The [`sdrk_proto::frame::Whitening`] of a `whitening` choice index (checked against
/// [`WHITENING_OPTIONS`] before).
pub(crate) fn whitening_from_choice(index: u16) -> sdrk_proto::frame::Whitening {
    match index {
        1 => sdrk_proto::frame::Whitening::Pn9,
        _ => sdrk_proto::frame::Whitening::None,
    }
}
