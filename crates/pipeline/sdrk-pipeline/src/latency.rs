//! Latency arithmetic: delays and buffering along a path, exact in rational numbers.

use sdrk::{Delay, Rate};

/// What a node on a path contributes: the rate at its input, its group delay in its input
/// samples and the smallest input count it needs to produce anything.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PathNode {
    pub rate_in: Rate,
    pub delay: Delay,
    pub min_input: usize,
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `value * unit / rate_in` as a reduced fraction of `u128`, `None` on overflow.
fn convert(num: u128, den: u128, unit: Rate, rate_in: Rate) -> Option<(u128, u128)> {
    let top = num
        .checked_mul(u128::from(unit.num()))?
        .checked_mul(u128::from(rate_in.den()))?;
    let bottom = den
        .checked_mul(u128::from(unit.den()))?
        .checked_mul(u128::from(rate_in.num()))?;
    let div = gcd(top, bottom).max(1);
    Some((top / div, bottom / div))
}

fn add(lhs: (u128, u128), rhs: (u128, u128)) -> Option<(u128, u128)> {
    let num = lhs
        .0
        .checked_mul(rhs.1)?
        .checked_add(rhs.0.checked_mul(lhs.1)?)?;
    let den = lhs.1.checked_mul(rhs.1)?;
    let div = gcd(num, den).max(1);
    Some((num / div, den / div))
}

/// Group delay and worst-case buffered samples of a path, in samples at the rate `unit`.
///
/// `None` if a figure does not fit into 64 bits.
pub(crate) fn path_latency(path: &[PathNode], unit: Rate) -> Option<(Delay, u64)> {
    let mut delay = (0u128, 1u128);
    let mut buffered = 0u128;
    for n in path {
        let d = convert(
            u128::from(n.delay.num()),
            u128::from(n.delay.den()),
            unit,
            n.rate_in,
        )?;
        delay = add(delay, d)?;
        let held = (n.min_input.saturating_sub(1)) as u128;
        let (num, den) = convert(held, 1, unit, n.rate_in)?;
        buffered = buffered.checked_add(num.div_ceil(den))?;
    }
    let delay = Delay::new(u64::try_from(delay.0).ok()?, u64::try_from(delay.1).ok()?).ok()?;
    Some((delay, u64::try_from(buffered).ok()?))
}

/// `ceil(delay)` as a sample count.
pub(crate) fn ceil_delay(delay: Delay) -> u64 {
    let d = delay.reduced();
    d.num().div_ceil(d.den().max(1))
}

/// An exact non-negative fraction `(numerator, denominator)`, reduced.
pub(crate) type Frac = (u128, u128);

/// The delay of a node in seconds: `delay / rate_in`. `None` on overflow.
pub(crate) fn delay_seconds(delay: Delay, rate_in: Rate) -> Option<Frac> {
    convert(
        u128::from(delay.num()),
        u128::from(delay.den()),
        Rate::hz(1).ok()?,
        rate_in,
    )
}

/// `lhs + rhs`, `None` on overflow.
pub(crate) fn frac_add(lhs: Frac, rhs: Frac) -> Option<Frac> {
    add(lhs, rhs)
}

/// `|lhs - rhs|`, `None` on overflow.
pub(crate) fn frac_abs_diff(lhs: Frac, rhs: Frac) -> Option<Frac> {
    let (big, small) = if frac_cmp(lhs, rhs)? == core::cmp::Ordering::Less {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    };
    let a = big.0.checked_mul(small.1)?;
    let b = small.0.checked_mul(big.1)?;
    let den = big.1.checked_mul(small.1)?;
    let num = a.checked_sub(b)?;
    let div = gcd(num, den).max(1);
    Some((num / div, den / div))
}

/// Compares two fractions, `None` on overflow.
pub(crate) fn frac_cmp(lhs: Frac, rhs: Frac) -> Option<core::cmp::Ordering> {
    Some(lhs.0.checked_mul(rhs.1)?.cmp(&rhs.0.checked_mul(lhs.1)?))
}

/// `value * rate`, the samples that a time of `value` seconds holds at `rate`, `None` on
/// overflow.
pub(crate) fn seconds_to_samples(value: Frac, rate: Rate) -> Option<Frac> {
    let num = value.0.checked_mul(u128::from(rate.num()))?;
    let den = value.1.checked_mul(u128::from(rate.den()))?;
    let div = gcd(num, den).max(1);
    Some((num / div, den / div))
}
