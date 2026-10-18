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
}

/// Whether the output/input rate ratio is fixed or can change at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatioKind {
    /// Fixed ratio.
    Constant,
    /// Ratio may change at run time.
    Variable,
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
