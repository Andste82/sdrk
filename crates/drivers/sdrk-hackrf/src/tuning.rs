//! What a HackRF One can do: tuning range, sample rates, baseband filters
//! and gain stages, and the mapping of requested values to the values the
//! hardware takes.

use sdrk_device::{Range, Rate};

/// Lowest centre frequency declared.
pub const MIN_FREQ_HZ: u64 = 1_000_000;
/// Highest centre frequency declared.
pub const MAX_FREQ_HZ: u64 = 6_000_000_000;

/// Lowest sample rate declared.
pub const MIN_RATE_HZ: u64 = 2_000_000;
/// Highest sample rate declared.
pub const MAX_RATE_HZ: u64 = 20_000_000;

/// The largest divider of `SAMPLE_RATE_SET` the backend uses (libhackrf
/// tries 1 to 31 for a fractional rate).
pub const MAX_DIVIDER: u32 = 31;

/// The baseband filters of the MAX2837 (libhackrf `max2837_ft`), in Hz.
pub const BANDWIDTHS_HZ: [u32; 16] = [
    1_750_000, 2_500_000, 3_500_000, 5_000_000, 5_500_000, 6_000_000, 7_000_000, 8_000_000,
    9_000_000, 10_000_000, 12_000_000, 14_000_000, 15_000_000, 20_000_000, 24_000_000, 28_000_000,
];

/// The RX LNA (IF) stage.
pub const LNA: &str = "lna";
/// The VGA (baseband) stage, RX and TX.
pub const VGA: &str = "vga";
/// The RF amplifier, RX and TX.
pub const AMP: &str = "amp";

/// The gain of the RF amplifier when it is on.
pub const AMP_DB: u32 = 14;

/// The tuning range of both channels (hertz, step 1).
pub const FREQ_RANGE: Range = Range {
    min: MIN_FREQ_HZ as f64,
    max: MAX_FREQ_HZ as f64,
    step: 1.0,
};

/// The rate range of both channels (continuous, step 0).
pub const RATE_RANGE: Range = Range {
    min: MIN_RATE_HZ as f64,
    max: MAX_RATE_HZ as f64,
    step: 0.0,
};

/// The gain stages of `rx0`, in the order the overall gain fills them.
pub const RX_GAINS: [(&str, Range); 3] = [
    (
        LNA,
        Range {
            min: 0.0,
            max: 40.0,
            step: 8.0,
        },
    ),
    (
        VGA,
        Range {
            min: 0.0,
            max: 62.0,
            step: 2.0,
        },
    ),
    (
        AMP,
        Range {
            min: 0.0,
            max: AMP_DB as f64,
            step: AMP_DB as f64,
        },
    ),
];

/// The gain stages of `tx0`, in the order the overall gain fills them.
pub const TX_GAINS: [(&str, Range); 2] = [
    (
        VGA,
        Range {
            min: 0.0,
            max: 47.0,
            step: 1.0,
        },
    ),
    (
        AMP,
        Range {
            min: 0.0,
            max: AMP_DB as f64,
            step: AMP_DB as f64,
        },
    ),
];

/// The gain stages of a direction ([`RX_GAINS`] or [`TX_GAINS`]).
#[must_use]
pub fn gain_stages(tx: bool) -> &'static [(&'static str, Range)] {
    if tx { &TX_GAINS } else { &RX_GAINS }
}

/// A requested gain mapped to the hardware.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GainMapping {
    /// The value the stage takes, in dB (the nearest step).
    Value(u32),
    /// The stage does not exist in this direction.
    Unknown,
    /// Outside the range of the stage (`min`, `max`).
    OutOfRange(f64, f64),
}

/// Maps `db` for `stage` to the nearest step of the stage. The amplifier is
/// on from half of its gain up.
#[must_use]
pub fn map_gain(tx: bool, stage: &str, db: f64) -> GainMapping {
    const EPS: f64 = 1e-9;
    let Some(&(_, caps)) = gain_stages(tx).iter().find(|(name, _)| *name == stage) else {
        return GainMapping::Unknown;
    };
    if !db.is_finite() || db < caps.min - EPS || db > caps.max + EPS {
        return GainMapping::OutOfRange(caps.min, caps.max);
    }
    let steps = (db / caps.step).round();
    let value = (steps * caps.step).clamp(caps.min, caps.max);
    GainMapping::Value(value as u32)
}

/// The filter nearest to `hz` (the lower one on a tie).
#[must_use]
pub fn nearest_bandwidth(hz: u64) -> u32 {
    let mut best = BANDWIDTHS_HZ[0];
    for &bw in &BANDWIDTHS_HZ {
        if u64::from(bw).abs_diff(hz) < u64::from(best).abs_diff(hz) {
            best = bw;
        }
    }
    best
}

/// The filter libhackrf chooses for a rate when none was set: the widest
/// filter not above three quarters of the rate (at least the narrowest)
/// (`hackrf_compute_baseband_filter_bw(0.75 * rate)`).
#[must_use]
pub fn auto_bandwidth(rate: Rate) -> u32 {
    let den = u128::from(rate.den().max(1));
    let target = u128::from(rate.num()) * 3 / (4 * den);
    let mut best = BANDWIDTHS_HZ[0];
    for &bw in &BANDWIDTHS_HZ {
        if u128::from(bw) <= target {
            best = bw;
        }
    }
    best
}

/// A rate mapped to `SAMPLE_RATE_SET`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HwRate {
    /// The `freq_hz` of the request.
    pub freq_hz: u32,
    /// The divider of the request.
    pub divider: u32,
}

impl HwRate {
    /// The rate the hardware runs at, `freq_hz / divider` as a fraction.
    ///
    /// # Errors
    /// `InvalidRate` for a zero `freq_hz`.
    pub fn rate(self) -> sdrk_device::Result<Rate> {
        Rate::new(u64::from(self.freq_hz), u64::from(self.divider.max(1)))
    }
}

/// Whether a rate lies in the declared range.
#[must_use]
pub fn rate_in_range(rate: Rate) -> bool {
    let (num, den) = (u128::from(rate.num()), u128::from(rate.den()));
    den > 0 && num >= u128::from(MIN_RATE_HZ) * den && num <= u128::from(MAX_RATE_HZ) * den
}

/// Maps a rate in the declared range to `freq_hz / divider`: exact if the
/// reduced fraction has a denominator up to [`MAX_DIVIDER`], otherwise the
/// nearest fraction with such a divider (the smallest divider on a tie).
/// `None` outside the range.
#[must_use]
pub fn map_rate(rate: Rate) -> Option<HwRate> {
    if !rate_in_range(rate) {
        return None;
    }
    let (num, den) = (u128::from(rate.num()), u128::from(rate.den()));
    if den <= u128::from(MAX_DIVIDER) {
        return Some(HwRate {
            freq_hz: u32::try_from(num).ok()?,
            divider: u32::try_from(den).ok()?,
        });
    }
    // (|freq * den - num * d|, d, rate): the error of freq / d is the first
    // value divided by d * den.
    let mut best: Option<(u128, u128, HwRate)> = None;
    for d in 1..=MAX_DIVIDER {
        let d128 = u128::from(d);
        // freq = round(num * d / den)
        let freq = (num * d128 * 2 + den) / (den * 2);
        let err = (freq * den).abs_diff(num * d128);
        let candidate = HwRate {
            freq_hz: u32::try_from(freq).ok()?,
            divider: d,
        };
        // err / d < best_err / best_d
        if best.is_none_or(|(e, bd, _)| err * bd < e * d128) {
            best = Some((err, d128, candidate));
        }
    }
    best.map(|(_, _, r)| r)
}
