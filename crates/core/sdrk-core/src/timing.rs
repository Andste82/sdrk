//! Timing declarations of kernels and blocks.

pub use sdrk_device::{Rate, Ratio, SampleIndex};

use crate::error::{Error, Result};

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Group delay in *input* samples as an exact fraction (e.g. 31/1 or 63/2).
///
/// Equality compares numerator and denominator as stored; use [`Delay::reduced`]
/// before comparing delays built from different fractions.
///
/// ```
/// use sdrk_core::timing::Delay;
/// let d = Delay::new(126, 4)?.reduced();
/// assert_eq!(d, Delay::new(63, 2)?);
/// assert_eq!(d.as_f64(), 31.5);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Delay {
    num: u64,
    den: u64,
}

impl Delay {
    /// No delay.
    pub const ZERO: Delay = Delay { num: 0, den: 1 };

    /// Creates a delay; the denominator must be non-zero.
    pub fn new(num: u64, den: u64) -> Result<Delay> {
        if den == 0 {
            return Err(Error::InvalidParameter(
                "delay denominator must not be zero",
            ));
        }
        Ok(Delay { num, den })
    }

    /// Numerator.
    pub const fn num(&self) -> u64 {
        self.num
    }

    /// Denominator.
    pub const fn den(&self) -> u64 {
        self.den
    }

    /// The delay as f64.
    pub fn as_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// The same delay in lowest terms (`0/n` becomes `0/1`).
    #[must_use]
    pub fn reduced(self) -> Delay {
        let g = gcd(self.num, self.den).max(1);
        Delay {
            num: self.num / g,
            den: self.den / g,
        }
    }

    /// Exact sum of two delays, in lowest terms. Fails on overflow.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Delay) -> Result<Delay> {
        let overflow = Error::InvalidParameter("delay sum overflows");
        let (lhs, rhs) = (self.reduced(), other.reduced());
        let common = gcd(lhs.den, rhs.den);
        let lcm = (lhs.den / common)
            .checked_mul(rhs.den)
            .ok_or_else(|| overflow.clone())?;
        let left = lhs
            .num
            .checked_mul(lcm / lhs.den)
            .ok_or_else(|| overflow.clone())?;
        let right = rhs
            .num
            .checked_mul(lcm / rhs.den)
            .ok_or_else(|| overflow.clone())?;
        let num = left.checked_add(right).ok_or(overflow)?;
        Ok(Delay { num, den: lcm }.reduced())
    }

    /// Exact quotient of the delay by a ratio, in lowest terms: `self * ratio.den / ratio.num`.
    ///
    /// Converts a delay counted in the output samples of a stage into input samples of that
    /// stage (a ratio is output over input). Fails on overflow of the reduced result.
    ///
    /// ```
    /// use sdrk_core::timing::{Delay, Ratio};
    /// // 8 output samples of a 1/4 decimator are 32 input samples.
    /// let d = Delay::new(8, 1)?.div_ratio(Ratio::new(1, 4)?)?;
    /// assert_eq!(d, Delay::new(32, 1)?);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn div_ratio(self, ratio: Ratio) -> Result<Delay> {
        let overflow = Error::InvalidParameter("delay quotient overflows");
        if ratio.num() == 0 || ratio.den() == 0 {
            return Err(Error::InvalidParameter("delay divided by a zero ratio"));
        }
        let num = u128::from(self.num) * u128::from(ratio.den());
        let den = u128::from(self.den) * u128::from(ratio.num());
        let g = gcd128(num, den).max(1);
        let num = u64::try_from(num / g).map_err(|_| overflow.clone())?;
        let den = u64::try_from(den / g).map_err(|_| overflow)?;
        Ok(Delay { num, den })
    }
}

fn gcd128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Exact `ceil(num / den)` in integers; `None` for `den == 0`.
///
/// ```
/// use sdrk_core::timing::ceil_div;
/// assert_eq!(ceil_div(7, 2), Some(4));
/// assert_eq!(ceil_div(8, 2), Some(4));
/// assert_eq!(ceil_div(0, 5), Some(0));
/// assert_eq!(ceil_div(1, 0), None);
/// ```
pub fn ceil_div(num: u128, den: u128) -> Option<u128> {
    if den == 0 {
        return None;
    }
    Some(num / den + u128::from(!num.is_multiple_of(den)))
}

/// Whether the output/input rate ratio is fixed or can change at run time, and how a block
/// with a packet port relates its packets to samples.
///
/// ```
/// use sdrk_core::timing::{Ratio, RatioKind};
/// // A decoder that emits at most one packet per 56 input samples.
/// let kind = RatioKind::Packets { max_per_input: Ratio::new(1, 56)? };
/// assert_ne!(kind, RatioKind::Constant);
/// assert_ne!(RatioKind::Source, RatioKind::Variable);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RatioKind {
    /// Fixed ratio.
    Constant,
    /// Ratio may change at run time.
    Variable,
    /// The output is a packet port: the number of packets per input sample varies, and
    /// `max_per_input` bounds it (used to size packet buffers). The `ratio` of the timing is
    /// `max_per_input` as well.
    Packets {
        /// Upper bound of the packets per input sample.
        max_per_input: Ratio,
    },
    /// The input is a packet port and the output a stream (a pure encoder): the output rate
    /// is set from outside (the `rate_hz` of the packet input), not by a ratio. The `ratio`
    /// of the timing bounds the output samples per input packet.
    Source,
}

/// Implemented by every kernel *state* and every block.
pub trait Timing {
    /// Output rate divided by input rate.
    fn ratio(&self) -> Ratio;
    /// Whether [`Timing::ratio`] is constant.
    fn ratio_kind(&self) -> RatioKind {
        RatioKind::Constant
    }
    /// Delay between an input sample and its effect on the output, in input samples.
    fn group_delay(&self) -> Delay;
}

/// Result of a rate-changing kernel call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Progress {
    /// Input samples consumed.
    pub consumed: usize,
    /// Output samples produced.
    pub produced: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn div_ratio_is_exact_and_reduced() {
        let d = Delay::new(31, 2).unwrap();
        // x / (1/4) = 4x
        assert_eq!(
            d.div_ratio(Ratio::new(1, 4).unwrap()).unwrap(),
            Delay::new(62, 1).unwrap()
        );
        // x / (3/2) = 2x/3
        assert_eq!(
            d.div_ratio(Ratio::new(3, 2).unwrap()).unwrap(),
            Delay::new(31, 3).unwrap()
        );
        assert_eq!(
            Delay::ZERO.div_ratio(Ratio::new(5, 7).unwrap()).unwrap(),
            Delay::ZERO
        );
        assert_eq!(d.div_ratio(Ratio::ONE).unwrap(), d);
    }

    #[test]
    fn div_ratio_overflow_is_an_error() {
        let big = Delay::new(u64::MAX, 1).unwrap();
        let r = Ratio::new(1, u64::MAX - 1).unwrap();
        assert!(big.div_ratio(r).is_err());
        // Intermediates beyond u64 that reduce back are fine.
        let d = Delay::new(u64::MAX, 3).unwrap();
        let r = Ratio::new(u64::MAX, 3).unwrap();
        assert_eq!(d.div_ratio(r).unwrap(), Delay::new(1, 1).unwrap());
    }

    #[test]
    fn ceil_div_cases() {
        assert_eq!(ceil_div(u128::MAX, 1), Some(u128::MAX));
        assert_eq!(ceil_div(u128::MAX, 2), Some(u128::MAX / 2 + 1));
        assert_eq!(ceil_div(1, 0), None);
    }
}
