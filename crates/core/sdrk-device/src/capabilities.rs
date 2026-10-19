//! Static description of what a device can do: value ranges, directions and the
//! capability record that a driver returns by reference.

use crate::SampleFormat;

/// A closed range of values with an optional step, used for frequencies, rates, gains and
/// bandwidths.
///
/// ```
/// use sdrk_device::Range;
/// let r = Range { min: 24e6, max: 1.766e9, step: 0.0 };
/// assert!(r.contains(100e6));
/// assert!(!r.contains(10e6));
/// let g = Range { min: 0.0, max: 40.0, step: 8.0 };
/// assert!(g.contains(16.0));
/// assert!(!g.contains(17.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    /// Smallest value.
    pub min: f64,
    /// Largest value.
    pub max: f64,
    /// Distance between neighbouring values; `0.0` means continuous.
    pub step: f64,
}

impl Range {
    /// True if `value` lies within `min..=max` and, for a stepped range, on the grid
    /// `min + k * step` (within `1e-9` of a step).
    pub fn contains(&self, value: f64) -> bool {
        if !(value >= self.min && value <= self.max) {
            return false;
        }
        if self.step <= 0.0 {
            return true;
        }
        let k = (value - self.min) / self.step;
        (k - round_half_away(k)).abs() <= 1e-9
    }
}

/// Rounds half away from zero without `std` or `libm` (this crate has no dependencies).
/// Only used for grid positions, which are far below `i64::MAX`.
fn round_half_away(x: f64) -> f64 {
    let t = x as i64 as f64;
    let frac = x - t;
    if frac >= 0.5 {
        t + 1.0
    } else if frac <= -0.5 {
        t - 1.0
    } else {
        t
    }
}

/// The direction of a channel from the host's point of view.
///
/// ```
/// use sdrk_device::Direction;
/// assert_ne!(Direction::Rx, Direction::Tx);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Receive.
    Rx,
    /// Transmit.
    Tx,
}

/// What a device offers. Drivers build one from `static` data and return it by reference.
///
/// ```
/// use sdrk_device::{formats, Capabilities, SampleFormat};
/// static FORMATS: [SampleFormat; 1] = [formats::CU8];
/// let caps = Capabilities {
///     rx_channels: 1,
///     tx_channels: 0,
///     full_duplex: false,
///     formats: &FORMATS,
///     hw_timestamps: false,
///     timed_commands: false,
/// };
/// assert_eq!(caps.formats[0], formats::CU8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capabilities {
    /// Number of receive channels.
    pub rx_channels: u8,
    /// Number of transmit channels.
    pub tx_channels: u8,
    /// True if receive and transmit can run at the same time.
    pub full_duplex: bool,
    /// Raw sample formats the streams can deliver or accept.
    pub formats: &'static [SampleFormat],
    /// True if blocks carry hardware timestamps (a `TimeAnchor` from the device clock).
    pub hw_timestamps: bool,
    /// True if the driver implements `TimedCommands`.
    pub timed_commands: bool,
}
