//! What the RTL2832U and its tuners can do: tuning ranges, the gain list,
//! the exact sample rates of the resampler and the frequency correction.

use crate::driver::{DirectSampling, TunerKind};
use sdrk_device::{Range, Rate};

/// The reference crystal of the RTL2832U.
pub const XTAL_HZ: u64 = 28_800_000;

/// The name of the single gain stage.
pub const GAIN_STAGE: &str = "tuner";

/// Lowest frequency of the R820T and R828D tuners (librtlsdr `r82xx`).
pub const R82XX_MIN_HZ: u64 = 24_000_000;
/// Highest frequency of the R820T and R828D tuners.
pub const R82XX_MAX_HZ: u64 = 1_766_000_000;
/// Lowest frequency of the RTL-SDR Blog V4 and V4L, which receive HF through
/// an upconverter.
pub const UPCONVERTER_MIN_HZ: u64 = 500_000;
/// Highest frequency in direct sampling mode (the crystal frequency; above
/// it the spectrum folds).
pub const DIRECT_SAMPLING_MAX_HZ: u64 = XTAL_HZ;

/// The range of the frequency correction. The RTL2832U computes the
/// correction as `ppm * 2^24` in 32-bit arithmetic, which holds up to 127.
pub const PPM_LIMIT: i32 = 127;

/// The sample rates the resampler accepts, in whole hertz: 225001 to 300000
/// and 900001 to 3200000 S/s.
pub const RATE_RANGES_HZ: [(u32, u32); 2] = [(225_001, 300_000), (900_001, 3_200_000)];

/// The tuning range for a tuner, with or without an upconverter, in a
/// direct sampling mode (hertz, step 1).
#[must_use]
pub fn freq_range(tuner: TunerKind, upconverter: bool, ds: DirectSampling) -> Range {
    if ds != DirectSampling::Off {
        return Range {
            min: 0.0,
            max: DIRECT_SAMPLING_MAX_HZ as f64,
            step: 1.0,
        };
    }
    let min = if upconverter {
        UPCONVERTER_MIN_HZ
    } else {
        R82XX_MIN_HZ
    };
    match tuner {
        TunerKind::R820t | TunerKind::R828d | TunerKind::Unknown => Range {
            min: min as f64,
            max: R82XX_MAX_HZ as f64,
            step: 1.0,
        },
    }
}

/// The rate ranges in samples per second (continuous within each range,
/// step 0).
pub const RATE_RANGES: [Range; 2] = [
    Range {
        min: RATE_RANGES_HZ[0].0 as f64,
        max: RATE_RANGES_HZ[0].1 as f64,
        step: 0.0,
    },
    Range {
        min: RATE_RANGES_HZ[1].0 as f64,
        max: RATE_RANGES_HZ[1].1 as f64,
        step: 0.0,
    },
];

/// Whether the resampler takes `hz`.
#[must_use]
pub fn rate_supported(hz: u32) -> bool {
    RATE_RANGES_HZ
        .iter()
        .any(|&(min, max)| (min..=max).contains(&hz))
}

/// The whole-hertz rate nearest to a requested rate.
#[must_use]
pub fn nearest_hz(rate: Rate) -> Option<u32> {
    let num = u128::from(rate.num());
    let den = u128::from(rate.den());
    let hz = (num * 2 + den) / (den * 2);
    u32::try_from(hz).ok()
}

/// The rate the hardware runs at for a request of `hz`, exactly: the
/// resampler divides `XTAL_HZ * 2^22` by a 28-bit ratio whose lowest two
/// bits are zero (librtlsdr `rtlsdr_set_sample_rate`). `None` if the rate is
/// not supported.
#[must_use]
pub fn exact_rate(hz: u32) -> Option<Rate> {
    if !rate_supported(hz) {
        return None;
    }
    let scaled = u128::from(XTAL_HZ) << 22;
    let ratio = (scaled / u128::from(hz)) & 0x0fff_fffc;
    let real = ratio | ((ratio & 0x0800_0000) << 1);
    let num = u64::try_from(scaled).ok()?;
    let den = u64::try_from(real).ok()?;
    Rate::new(num, den).ok()
}

/// The range of the gain list (tenths of a dB) in dB: one continuous stage
/// [`GAIN_STAGE`] from the lowest to the highest value (step 0); a request
/// is set to the nearest value of the list ([`nearest_gain`]).
#[must_use]
pub fn gain_range(gains_tenths: &[i32]) -> Option<Range> {
    let min = gains_tenths.iter().copied().min()?;
    let max = gains_tenths.iter().copied().max()?;
    Some(Range {
        min: f64::from(min) / 10.0,
        max: f64::from(max) / 10.0,
        step: 0.0,
    })
}

/// The value of the gain list nearest to `db`, in tenths of a dB; the lower
/// one on a tie.
#[must_use]
pub fn nearest_gain(gains_tenths: &[i32], db: f64) -> Option<i32> {
    let target = db * 10.0;
    gains_tenths.iter().copied().min_by(|a, b| {
        let da = (f64::from(*a) - target).abs();
        let db = (f64::from(*b) - target).abs();
        da.total_cmp(&db).then(a.cmp(b))
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// The gain list of the R820T in librtlsdr (`r82xx_gains`).
    const R820T: [i32; 29] = [
        0, 9, 14, 27, 37, 77, 87, 125, 144, 157, 166, 197, 207, 229, 254, 280, 297, 328, 338, 364,
        372, 386, 402, 421, 434, 439, 445, 480, 496,
    ];

    #[test]
    fn req_dev_4_gain_mapping_nearest_value_of_the_list() {
        let stage = gain_range(&R820T).unwrap();
        assert_eq!(GAIN_STAGE, "tuner");
        assert_eq!(stage.min, 0.0);
        assert_eq!(stage.max, 49.6);
        assert_eq!(stage.step, 0.0);
        assert_eq!(nearest_gain(&R820T, 29.7), Some(297));
        assert_eq!(nearest_gain(&R820T, 30.0), Some(297));
        assert_eq!(nearest_gain(&R820T, 31.3), Some(328));
        assert_eq!(nearest_gain(&R820T, 0.4), Some(0));
        // 0.0 and 0.9 are 0.45 dB from 0.45: the lower one wins.
        assert_eq!(nearest_gain(&R820T, 0.45), Some(0));
        assert_eq!(nearest_gain(&R820T, 100.0), Some(496));
        assert_eq!(nearest_gain(&R820T, -5.0), Some(0));
        assert_eq!(nearest_gain(&[], 10.0), None);
        assert!(gain_range(&[]).is_none());
    }

    #[test]
    fn req_dev_4_rate_ranges_and_exact_rates() {
        assert!(!rate_supported(225_000));
        assert!(rate_supported(225_001));
        assert!(rate_supported(300_000));
        assert!(!rate_supported(300_001));
        assert!(!rate_supported(900_000));
        assert!(rate_supported(900_001));
        assert!(rate_supported(3_200_000));
        assert!(!rate_supported(3_200_001));
        // Rates that divide the crystal evenly are exact.
        assert_eq!(exact_rate(2_400_000), Some(Rate::hz(2_400_000).unwrap()));
        assert_eq!(exact_rate(2_048_000), Some(Rate::hz(2_048_000).unwrap()));
        assert_eq!(exact_rate(3_200_000), Some(Rate::hz(3_200_000).unwrap()));
        assert_eq!(exact_rate(288_000), Some(Rate::hz(288_000).unwrap()));
        assert_eq!(exact_rate(300_000), Some(Rate::hz(300_000).unwrap()));
        // Others are the exact fraction of the resampler.
        let r = exact_rate(1_000_000).unwrap();
        assert_ne!(r, Rate::hz(1_000_000).unwrap());
        assert!((r.as_f64() - 1_000_000.0).abs() < 1.0);
        assert!(r.as_f64() >= 1_000_000.0);
        assert_eq!(exact_rate(225_000), None);
        assert_eq!(exact_rate(1_000), None);
        assert_eq!(RATE_RANGES[0].min, 225_001.0);
        assert_eq!(RATE_RANGES[1].max, 3_200_000.0);
        assert_eq!(RATE_RANGES[1].step, 0.0);
        assert!(RATE_RANGES[1].contains(2_400_000.0));
    }

    #[test]
    fn req_dev_4_nearest_hz_of_fractional_rates() {
        assert_eq!(nearest_hz(Rate::hz(2_400_000).unwrap()), Some(2_400_000));
        assert_eq!(
            nearest_hz(Rate::new(4_800_001, 2).unwrap()),
            Some(2_400_001)
        );
        assert_eq!(
            nearest_hz(Rate::new(4_799_999, 2).unwrap()),
            Some(2_400_000)
        );
    }

    #[test]
    fn req_dev_4_tuning_ranges_from_the_tuner_type() {
        let r = freq_range(TunerKind::R820t, false, DirectSampling::Off);
        assert_eq!((r.min, r.max), (24e6, 1766e6));
        let r = freq_range(TunerKind::R828d, true, DirectSampling::Off);
        assert_eq!((r.min, r.max), (5e5, 1766e6));
        let r = freq_range(TunerKind::R820t, false, DirectSampling::Q);
        assert_eq!((r.min, r.max), (0.0, 28.8e6));
    }
}
