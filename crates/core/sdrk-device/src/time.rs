//! Exact time types: sample index, rational rates and ratios, time anchors.
//!
//! Floating point is never used for time. An `f64` has 53 bits of mantissa;
//! at several MS/s the sample counter passes 2^53 after a few days of
//! continuous operation and `f64` would silently lose individual samples.
//! All conversions here use integers (`u128`/`i128` intermediates).

use crate::{Error, Result};

const NS_PER_S: u128 = 1_000_000_000;

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Reduces `num/den` (both non-zero) and narrows to `u64`.
fn reduce(num: u128, den: u128, what: &'static str) -> Result<(u64, u64)> {
    let g = gcd(num, den).max(1);
    let n = u64::try_from(num / g).map_err(|_| Error::Overflow(what))?;
    let d = u64::try_from(den / g).map_err(|_| Error::Overflow(what))?;
    Ok((n, d))
}

/// Index of a sample in a stream, counted from the start of the stream.
///
/// ```
/// use sdrk_device::SampleIndex;
/// assert_eq!(SampleIndex(5).checked_add(3), Some(SampleIndex(8)));
/// assert_eq!(SampleIndex(5).checked_sub(SampleIndex(7)), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SampleIndex(pub u64);

impl SampleIndex {
    /// Adds `n`, `None` on overflow.
    pub const fn checked_add(self, n: u64) -> Option<SampleIndex> {
        match self.0.checked_add(n) {
            Some(v) => Some(SampleIndex(v)),
            None => None,
        }
    }

    /// Adds `n`, saturating at `u64::MAX`.
    #[must_use]
    pub const fn saturating_add(self, n: u64) -> SampleIndex {
        SampleIndex(self.0.saturating_add(n))
    }

    /// Distance `self - other`, `None` if `other` is later.
    pub const fn checked_sub(self, other: SampleIndex) -> Option<u64> {
        self.0.checked_sub(other.0)
    }
}

/// Exact rational rate in samples per second, always reduced with
/// `num > 0` and `den > 0`.
///
/// ```
/// use sdrk_device::{Rate, Ratio};
/// let r = Rate::hz(2_400_000)?.scale(Ratio::new(1, 10)?)?;
/// assert_eq!(r, Rate::hz(240_000)?);
/// # Ok::<(), sdrk_device::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rate {
    num: u64,
    den: u64,
}

impl Rate {
    /// Creates a reduced rate; zero numerator or denominator is an error.
    pub fn new(num: u64, den: u64) -> Result<Rate> {
        if num == 0 || den == 0 {
            return Err(Error::InvalidRate { num, den });
        }
        let (num, den) = reduce(u128::from(num), u128::from(den), "rate")?;
        Ok(Rate { num, den })
    }

    /// A whole-hertz rate, shorthand for `Rate::new(hz, 1)`.
    pub fn hz(hz: u64) -> Result<Rate> {
        Rate::new(hz, 1)
    }

    /// Numerator (samples).
    pub const fn num(&self) -> u64 {
        self.num
    }

    /// Denominator (seconds).
    pub const fn den(&self) -> u64 {
        self.den
    }

    /// The rate as a float (for display only, never for time arithmetic).
    pub fn as_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Multiplies by an exact ratio.
    pub fn scale(&self, ratio: Ratio) -> Result<Rate> {
        let num = u128::from(self.num) * u128::from(ratio.num);
        let den = u128::from(self.den) * u128::from(ratio.den);
        let (num, den) = reduce(num, den, "rate scale")?;
        Ok(Rate { num, den })
    }

    /// Number of whole samples in `ns` nanoseconds: `floor(ns * num / (den * 1e9))`.
    pub fn samples_in_ns(&self, ns: u64) -> Result<u64> {
        let top = u128::from(ns) * u128::from(self.num);
        let bottom = u128::from(self.den) * NS_PER_S;
        u64::try_from(top / bottom).map_err(|_| Error::Overflow("samples in ns"))
    }

    /// Nanoseconds spanned by `n` samples: `floor(n * den * 1e9 / num)`.
    pub fn ns_for_samples(&self, n: u64) -> Result<u64> {
        let top = (u128::from(n) * u128::from(self.den))
            .checked_mul(NS_PER_S)
            .ok_or(Error::Overflow("ns for samples"))?;
        u64::try_from(top / u128::from(self.num)).map_err(|_| Error::Overflow("ns for samples"))
    }
}

/// Exact ratio out/in of a rate-changing kernel or block, e.g. 1/10 for
/// decimation by 10. Always reduced, both parts non-zero.
///
/// ```
/// use sdrk_device::Ratio;
/// let r = Ratio::new(1, 10)?.mul(Ratio::new(1, 5)?)?;
/// assert_eq!((r.num(), r.den()), (1, 50));
/// # Ok::<(), sdrk_device::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ratio {
    num: u64,
    den: u64,
}

impl Ratio {
    /// The identity ratio 1/1.
    pub const ONE: Ratio = Ratio { num: 1, den: 1 };

    /// Creates a reduced ratio; zero parts are an error.
    pub fn new(num: u64, den: u64) -> Result<Ratio> {
        if num == 0 || den == 0 {
            return Err(Error::InvalidRate { num, den });
        }
        let (num, den) = reduce(u128::from(num), u128::from(den), "ratio")?;
        Ok(Ratio { num, den })
    }

    /// Numerator.
    pub const fn num(&self) -> u64 {
        self.num
    }

    /// Denominator.
    pub const fn den(&self) -> u64 {
        self.den
    }

    /// Product of two ratios.
    #[allow(clippy::should_implement_trait)] // name is normative and the operation is fallible
    pub fn mul(self, other: Ratio) -> Result<Ratio> {
        let num = u128::from(self.num) * u128::from(other.num);
        let den = u128::from(self.den) * u128::from(other.den);
        let (num, den) = reduce(num, den, "ratio mul")?;
        Ok(Ratio { num, den })
    }

    /// Maps an input index to the output index: `floor(input_index * num / den)`.
    pub fn map_index(&self, input_index: u64) -> Result<u64> {
        let v = u128::from(input_index) * u128::from(self.num) / u128::from(self.den);
        u64::try_from(v).map_err(|_| Error::Overflow("map index"))
    }

    /// The inverse ratio `den/num`.
    #[must_use]
    pub const fn inverse(self) -> Ratio {
        Ratio {
            num: self.den,
            den: self.num,
        }
    }
}

/// Where a timestamp came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TimeSource {
    /// The device clock.
    Device,
    /// The host clock.
    Host,
    /// A GPS receiver.
    Gps,
    /// Unknown origin.
    Unknown,
}

/// "Sample `index` was taken at `unix_ns` nanoseconds since 1970-01-01T00:00:00Z."
///
/// ```
/// use sdrk_device::{Rate, SampleIndex, TimeAnchor, TimeSource};
/// let a = TimeAnchor { index: SampleIndex(0), unix_ns: 1_000, source: TimeSource::Host };
/// assert_eq!(a.time_of(SampleIndex(1_000_000), Rate::hz(1_000_000)?)?, 1_000_001_000);
/// # Ok::<(), sdrk_device::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeAnchor {
    /// Index of the anchored sample.
    pub index: SampleIndex,
    /// Absolute time of that sample in unix nanoseconds.
    pub unix_ns: i64,
    /// Origin of the timestamp.
    pub source: TimeSource,
}

impl TimeAnchor {
    /// Absolute time of `index` at sample rate `rate`, rounded toward
    /// negative infinity. Works for indices before and after the anchor.
    pub fn time_of(&self, index: SampleIndex, rate: Rate) -> Result<i64> {
        let delta = i128::from(index.0) - i128::from(self.index.0);
        let top = delta
            .checked_mul(1_000_000_000)
            .and_then(|v| v.checked_mul(i128::from(rate.den)))
            .ok_or(Error::Overflow("time of"))?;
        let offset = top.div_euclid(i128::from(rate.num));
        let total = offset
            .checked_add(i128::from(self.unix_ns))
            .ok_or(Error::Overflow("time of"))?;
        i64::try_from(total).map_err(|_| Error::Overflow("time of"))
    }
}
