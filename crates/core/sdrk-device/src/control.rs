//! Optional control capabilities of a device. A driver implements only the traits its
//! hardware supports; everything is `no_std` and allocation-free.
//!
//! Every setter returns the value that was actually applied (hardware rounds), and every
//! failure maps to an `ErrorKind` such as `Unsupported`, `OutOfRange`, `Device`, `Timeout`
//! or `Disconnected`.

use crate::capabilities::Capabilities;
use crate::{Direction, Range, Rate, SampleIndex, SdrkError, TimeAnchor};

/// Identification of a device.
///
/// ```
/// use sdrk_device::{formats, Capabilities, Device, Error, SampleFormat};
/// static FORMATS: [SampleFormat; 1] = [formats::CU8];
/// static CAPS: Capabilities = Capabilities {
///     rx_channels: 1, tx_channels: 0, full_duplex: false,
///     formats: &FORMATS, hw_timestamps: false, timed_commands: false,
/// };
/// struct Dongle;
/// impl Device for Dongle {
///     type Error = Error;
///     fn capabilities(&self) -> &Capabilities { &CAPS }
///     fn driver(&self) -> &str { "dongle" }
///     fn serial(&self) -> Option<&str> { Some("0001") }
/// }
/// assert_eq!(Dongle.driver(), "dongle");
/// ```
pub trait Device {
    /// Error type of the driver.
    type Error: SdrkError;
    /// What the device can do.
    fn capabilities(&self) -> &Capabilities;
    /// Name of the driver, for example `"rtlsdr"`.
    fn driver(&self) -> &str;
    /// Serial number, if the hardware has one.
    fn serial(&self) -> Option<&str>;
}

/// Centre-frequency control.
pub trait Tuner {
    /// Error type of the driver.
    type Error: SdrkError;
    /// Tunes channel `ch`; returns the actual frequency in hertz.
    fn set_frequency(&mut self, dir: Direction, ch: u8, hz: f64) -> Result<f64, Self::Error>;
    /// The current frequency in hertz.
    fn frequency(&self, dir: Direction, ch: u8) -> Result<f64, Self::Error>;
    /// The tuning ranges of the channel.
    fn frequency_ranges(&self, dir: Direction, ch: u8) -> &[Range];
}

/// Sample-rate control.
pub trait RateControl {
    /// Error type of the driver.
    type Error: SdrkError;
    /// Sets the sample rate; returns the rate that was actually applied.
    fn set_sample_rate(&mut self, dir: Direction, ch: u8, rate: Rate) -> Result<Rate, Self::Error>;
    /// The supported sample-rate ranges in samples per second.
    fn sample_rates(&self, dir: Direction, ch: u8) -> &[Range];
}

/// Gain control.
pub trait GainControl {
    /// Error type of the driver.
    type Error: SdrkError;
    /// The named gain stages with their ranges in decibels.
    fn gain_stages(&self, dir: Direction, ch: u8) -> &[(&'static str, Range)];
    /// Sets the gain of one stage, or the overall gain for `None`; returns the applied value.
    fn set_gain(
        &mut self,
        dir: Direction,
        ch: u8,
        stage: Option<&str>,
        db: f64,
    ) -> Result<f64, Self::Error>;
    /// Switches automatic gain control on or off.
    fn set_agc(&mut self, dir: Direction, ch: u8, on: bool) -> Result<(), Self::Error>;
}

/// Analogue bandwidth control.
pub trait BandwidthControl {
    /// Error type of the driver.
    type Error: SdrkError;
    /// Sets the bandwidth; returns the applied value in hertz.
    fn set_bandwidth(&mut self, dir: Direction, ch: u8, hz: f64) -> Result<f64, Self::Error>;
    /// The supported bandwidth ranges in hertz.
    fn bandwidths(&self, dir: Direction, ch: u8) -> &[Range];
}

/// Access to the device clock.
pub trait HardwareTime {
    /// Error type of the driver.
    type Error: SdrkError;
    /// The current sample index of the stream and the device time it corresponds to.
    fn hardware_time(&self) -> Result<TimeAnchor, Self::Error>;
    /// Sets the device clock to `unix_ns` nanoseconds since 1970-01-01T00:00:00Z.
    fn set_hardware_time(&mut self, unix_ns: i64) -> Result<(), Self::Error>;
}

/// A control command that can be scheduled.
///
/// ```
/// use sdrk_device::{Command, Direction};
/// let c = Command::Gain { dir: Direction::Tx, ch: 0, db: 20.0 };
/// assert_eq!(c, c);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Command {
    /// Retune a channel.
    Frequency {
        /// Direction of the channel.
        dir: Direction,
        /// Channel number.
        ch: u8,
        /// New frequency in hertz.
        hz: f64,
    },
    /// Change the overall gain of a channel.
    Gain {
        /// Direction of the channel.
        dir: Direction,
        /// Channel number.
        ch: u8,
        /// New gain in decibels.
        db: f64,
    },
}

/// When a scheduled command takes effect.
///
/// ```
/// use sdrk_device::{CommandTime, SampleIndex};
/// assert_ne!(CommandTime::Index(SampleIndex(5)), CommandTime::UnixNs(5));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandTime {
    /// At a sample index of the stream.
    Index(SampleIndex),
    /// At an absolute time in unix nanoseconds.
    UnixNs(i64),
}

/// Commands that the hardware executes at a given time.
pub trait TimedCommands {
    /// Error type of the driver.
    type Error: SdrkError;
    /// Schedules `cmd` for `at`.
    fn schedule(&mut self, at: CommandTime, cmd: Command) -> Result<(), Self::Error>;
}
