//! [`RtlSdr`]: one open dongle with the device traits of `sdrk-device`.

use crate::driver::{
    DirectSampling, Driver, DriverError, DriverErrorKind, Radio, TunerKind, UsbInfo,
};
use crate::stream::{Queue, Reader, RtlRx};
use crate::tuner::{
    GAIN_STAGE, PPM_LIMIT, RATE_RANGES, exact_rate, freq_range, gain_range, nearest_gain,
    nearest_hz,
};
use sdrk_device::{
    Capabilities, Direction, GainControl, Range, Rate, RateControl, SampleFormat, Tuner, formats,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The name of the driver (`Device::driver`).
pub const DRIVER: &str = "rtlsdr";

/// Centre frequency after opening.
pub const DEFAULT_CENTER_HZ: u32 = 100_000_000;
/// Sample rate after opening.
pub const DEFAULT_RATE_HZ: u32 = 2_400_000;
/// Tuner gain after opening (the nearest value of the gain list is used).
pub const DEFAULT_GAIN_DB: f64 = 29.7;

const FORMATS: [SampleFormat; 1] = [formats::CU8];

const CAPABILITIES: Capabilities = Capabilities {
    rx_channels: 1,
    tx_channels: 0,
    full_duplex: false,
    formats: &FORMATS,
    hw_timestamps: false,
    timed_commands: false,
};

/// How a stream reads from the dongle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamOptions {
    /// Bulk transfers kept in flight (librtlsdr default 15).
    pub buffers: usize,
    /// Bytes of one transfer (librtlsdr default 16 × 16384 = 262144).
    pub buffer_bytes: usize,
    /// Transfers the queue between the reader thread and the stream holds
    /// before it drops the newest one.
    pub queue_transfers: usize,
    /// A read waits at least this long for samples before it fails with
    /// `Timeout` (and at least four transfers).
    pub read_timeout: Duration,
}

impl Default for StreamOptions {
    fn default() -> StreamOptions {
        StreamOptions {
            buffers: 15,
            buffer_bytes: 16 * 16384,
            queue_transfers: 32,
            read_timeout: Duration::from_secs(2),
        }
    }
}

fn invalid(message: impl Into<String>) -> DriverError {
    DriverError::new(DriverErrorKind::InvalidArgument, message)
}

fn check_rx(dir: Direction, ch: u8) -> Result<(), DriverError> {
    if dir == Direction::Rx && ch == 0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "an RTL-SDR has the channel rx0 only, not {dir:?} {ch}"
        )))
    }
}

/// One open RTL-SDR.
///
/// After opening it is tuned to [`DEFAULT_CENTER_HZ`] at [`DEFAULT_RATE_HZ`]
/// with the tuner gain nearest to [`DEFAULT_GAIN_DB`], no frequency
/// correction and the bias tee off. Every setter works while a stream runs:
/// the reading pauses for it, and the next block of the stream carries
/// `GAP_BEFORE` with its index advanced by the samples of the pause (at
/// least one), so the index stays continuous.
///
/// ```
/// use sdrk_device::{Direction, RxStream, Tuner};
/// use sdrk_rtlsdr::fake::{FakeDongle, FakeDriver};
/// use sdrk_rtlsdr::{Driver, RtlSdr, StreamOptions};
///
/// let bus = FakeDriver::new(vec![FakeDongle::r820t(0, Some("00000001"), 1)]);
/// let info = bus.list()?.remove(0);
/// let mut sdr = RtlSdr::open(&bus, &info, true, StreamOptions::default())?;
/// sdr.set_frequency(Direction::Rx, 0, 433.92e6)?;
/// let mut rx = sdr.start_rx()?;
/// bus.feed(0, 1);
/// let mut buf = vec![0u8; 4096];
/// let (n, meta) = rx.read(&mut buf)?;
/// assert_eq!((n, meta.start.0), (4096, 0));
/// sdr.stop_rx();
/// # Ok::<(), sdrk_rtlsdr::DriverError>(())
/// ```
pub struct RtlSdr {
    serial: Option<String>,
    options: StreamOptions,
    tuner: TunerKind,
    upconverter: bool,
    gains: Vec<i32>,
    /// The radio while no stream runs; the reader thread holds it while one
    /// runs. `None` if it was lost (a reader thread that panicked).
    radio: Option<Box<dyn Radio>>,
    reader: Option<(Reader, Arc<Queue>)>,
    center_hz: u32,
    /// The whole-hertz rate last given to the driver.
    rate_hz: u32,
    /// The tuner gain in tenths of a dB while the AGC is off.
    gain_tenths: i32,
    agc: bool,
    ds: DirectSampling,
    freq_ranges: [Range; 1],
    gain_stages: Vec<(&'static str, Range)>,
}

impl core::fmt::Debug for RtlSdr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RtlSdr")
            .field("serial", &self.serial)
            .field("tuner", &self.tuner)
            .field("center_hz", &self.center_hz)
            .field("rate_hz", &self.rate_hz)
            .field("streaming", &self.reader.is_some())
            .finish_non_exhaustive()
    }
}

impl RtlSdr {
    /// Opens a dongle of a bus listing ([`Driver::list`]) by serial number
    /// if `by_serial` is set (use it when the serial number is unique on the
    /// bus), otherwise by its position.
    ///
    /// # Errors
    /// The error of the driver.
    pub fn open(
        driver: &dyn Driver,
        info: &UsbInfo,
        by_serial: bool,
        options: StreamOptions,
    ) -> Result<RtlSdr, DriverError> {
        let radio = driver.open(info, by_serial)?;
        RtlSdr::from_radio(radio, info.serial.clone(), options)
    }

    /// Wraps an opened radio and puts it into the default configuration.
    ///
    /// # Errors
    /// The error of the driver.
    pub fn from_radio(
        mut radio: Box<dyn Radio>,
        serial: Option<String>,
        options: StreamOptions,
    ) -> Result<RtlSdr, DriverError> {
        let tuner = radio.tuner();
        let upconverter = radio.has_upconverter();
        let mut gains = radio.tuner_gains();
        gains.sort_unstable();
        let gain_tenths = nearest_gain(&gains, DEFAULT_GAIN_DB).unwrap_or(0);
        radio.set_sample_rate(DEFAULT_RATE_HZ)?;
        radio.set_center_freq(DEFAULT_CENTER_HZ)?;
        radio.set_tuner_gain(Some(gain_tenths))?;
        radio.set_ppm(0)?;
        radio.set_bias_tee(false)?;
        let gain_stages = gain_range(&gains)
            .map(|r| vec![(GAIN_STAGE, r)])
            .unwrap_or_default();
        Ok(RtlSdr {
            serial,
            options,
            tuner,
            upconverter,
            gains,
            radio: Some(radio),
            reader: None,
            center_hz: DEFAULT_CENTER_HZ,
            rate_hz: DEFAULT_RATE_HZ,
            gain_tenths,
            agc: false,
            ds: DirectSampling::Off,
            freq_ranges: [freq_range(tuner, upconverter, DirectSampling::Off)],
            gain_stages,
        })
    }

    /// The tuner.
    pub fn tuner_kind(&self) -> TunerKind {
        self.tuner
    }

    /// Whether the dongle is an RTL-SDR Blog V4 or V4L with an upconverter.
    pub fn has_upconverter(&self) -> bool {
        self.upconverter
    }

    /// The gain list of the tuner in tenths of a dB, ascending.
    pub fn tuner_gains(&self) -> &[i32] {
        &self.gains
    }

    /// The centre frequency last set, in hertz.
    pub fn center_hz(&self) -> u32 {
        self.center_hz
    }

    /// The whole-hertz rate last given to the driver; the hardware runs at
    /// [`exact_rate`] of it.
    pub fn rate_hz(&self) -> u32 {
        self.rate_hz
    }

    /// The tuner gain in tenths of a dB (while the AGC is off).
    pub fn gain_tenths(&self) -> i32 {
        self.gain_tenths
    }

    /// Whether the AGC of the tuner is on.
    pub fn agc(&self) -> bool {
        self.agc
    }

    /// The direct sampling mode.
    pub fn direct_sampling(&self) -> DirectSampling {
        self.ds
    }

    /// The samples `d` holds at the current rate.
    fn samples_in(&self, d: Duration) -> u64 {
        let n = d.as_nanos() * u128::from(self.rate_hz) / 1_000_000_000;
        u64::try_from(n).unwrap_or(u64::MAX)
    }

    fn spawn_reader(
        &mut self,
        radio: Box<dyn Radio>,
        queue: Arc<Queue>,
    ) -> Result<(), DriverError> {
        let name = format!("rtlsdr-{}", self.serial.as_deref().unwrap_or("dongle"));
        let reader = Reader::spawn(
            name,
            radio,
            Arc::clone(&queue),
            self.options.buffers,
            self.options.buffer_bytes,
        )?;
        self.reader = Some((reader, queue));
        Ok(())
    }

    /// Runs `f` on the radio. A running stream is paused for it: the reader
    /// thread gives the radio back and starts again afterwards, feeding the
    /// same stream, whose next block carries `GAP_BEFORE` and an index
    /// advanced by the time of the pause (at least one sample).
    ///
    /// # Errors
    /// The error of `f`; `InvalidState` if the radio was lost (the reader
    /// thread panicked; open the dongle again).
    pub fn with_radio<R>(
        &mut self,
        f: impl FnOnce(&mut dyn Radio) -> Result<R, DriverError>,
    ) -> Result<R, DriverError> {
        let paused_at = Instant::now();
        let resume = match self.reader.take() {
            Some((reader, queue)) => {
                self.radio = reader.stop();
                Some(queue).filter(|q| !q.failed())
            }
            None => None,
        };
        let mut radio = self.radio.take().ok_or_else(lost)?;
        let result = f(radio.as_mut());
        match resume {
            Some(queue) => {
                queue.paused(self.samples_in(paused_at.elapsed()).max(1));
                // The result of `f` stands even if the reading cannot go on:
                // the values may be in effect. The stream then fails with the
                // error, and the radio is lost until the dongle is opened
                // again.
                if let Err(e) = self.spawn_reader(radio, Arc::clone(&queue)) {
                    queue.fail(e);
                }
            }
            None => self.radio = Some(radio),
        }
        result
    }

    /// Tunes to `hz`.
    ///
    /// # Errors
    /// `InvalidArgument` outside the tuning range; the error of the driver.
    pub fn set_center_hz(&mut self, hz: u32) -> Result<(), DriverError> {
        let [range] = self.freq_ranges;
        if !range.contains(f64::from(hz)) {
            return Err(invalid(format!(
                "{hz} Hz is outside {} to {} Hz",
                range.min, range.max
            )));
        }
        self.with_radio(|r| r.set_center_freq(hz))?;
        self.center_hz = hz;
        Ok(())
    }

    /// Sets the rate to a whole number of samples per second.
    ///
    /// # Errors
    /// `InvalidArgument` for a rate the resampler does not take; the error of
    /// the driver.
    pub fn set_rate_hz(&mut self, hz: u32) -> Result<Rate, DriverError> {
        let rate = exact_rate(hz).ok_or_else(|| {
            invalid(format!(
                "{hz} S/s is not supported (225001 to 300000 and 900001 to 3200000)"
            ))
        })?;
        self.with_radio(|r| r.set_sample_rate(hz))?;
        self.rate_hz = hz;
        Ok(rate)
    }

    /// Sets the tuner gain to the value of the gain list nearest to `db` and
    /// switches the AGC off; returns the gain set, in tenths of a dB.
    ///
    /// # Errors
    /// `Unsupported` if the tuner has no gain list; the error of the driver.
    pub fn set_gain_db(&mut self, db: f64) -> Result<i32, DriverError> {
        let tenths = nearest_gain(&self.gains, db).ok_or_else(|| {
            DriverError::new(DriverErrorKind::Unsupported, "the tuner has no gain list")
        })?;
        self.with_radio(|r| r.set_tuner_gain(Some(tenths)))?;
        self.gain_tenths = tenths;
        self.agc = false;
        Ok(tenths)
    }

    /// Switches the AGC of the tuner on, or off again to the last manual
    /// gain.
    ///
    /// # Errors
    /// The error of the driver.
    pub fn set_agc_on(&mut self, on: bool) -> Result<(), DriverError> {
        let gain = (!on).then_some(self.gain_tenths);
        self.with_radio(|r| r.set_tuner_gain(gain))?;
        self.agc = on;
        Ok(())
    }

    /// Sets the frequency correction in whole ppm (-127 to 127).
    ///
    /// # Errors
    /// `InvalidArgument` outside the range; the error of the driver.
    pub fn set_ppm(&mut self, ppm: i32) -> Result<(), DriverError> {
        if ppm.abs() > PPM_LIMIT {
            return Err(invalid(format!(
                "{ppm} ppm is outside -{PPM_LIMIT} to {PPM_LIMIT}"
            )));
        }
        self.with_radio(|r| r.set_ppm(ppm))
    }

    /// Switches the bias tee (power on the antenna input).
    ///
    /// # Errors
    /// The error of the driver.
    pub fn set_bias_tee(&mut self, on: bool) -> Result<(), DriverError> {
        self.with_radio(|r| r.set_bias_tee(on))
    }

    /// Sets the direct sampling mode, which changes the tuning range: the
    /// centre frequency is moved into the new range (clamped) and set again.
    /// Leaving direct sampling initialises the tuner again, so the rate and
    /// the gain are set again as well. Returns the centre frequency.
    ///
    /// When the mode was switched but a later step failed, the mode stays
    /// switched and the error is returned.
    ///
    /// # Errors
    /// The error of the driver.
    pub fn set_direct_sampling(&mut self, mode: DirectSampling) -> Result<u32, DriverError> {
        let range = freq_range(self.tuner, self.upconverter, mode);
        let center = (f64::from(self.center_hz).clamp(range.min, range.max)) as u32;
        let rate = self.rate_hz;
        let gain = (!self.agc && !self.gains.is_empty()).then_some(self.gain_tenths);
        let leaving = mode == DirectSampling::Off;
        let mut switched = false;
        let result = self.with_radio(|r| {
            r.set_direct_sampling(mode)?;
            switched = true;
            if leaving {
                r.set_sample_rate(rate)?;
                r.set_tuner_gain(gain)?;
            }
            r.set_center_freq(center)
        });
        if switched {
            self.ds = mode;
            self.freq_ranges = [range];
        }
        result?;
        self.center_hz = center;
        Ok(center)
    }

    /// Starts the RX stream at the current rate.
    ///
    /// # Errors
    /// `InvalidState` if a stream already runs or the radio was lost; the
    /// error of the driver.
    pub fn start_rx(&mut self) -> Result<RtlRx, DriverError> {
        if let Some((_, queue)) = &self.reader {
            if !queue.failed() {
                return Err(DriverError::new(
                    DriverErrorKind::InvalidState,
                    "rx0 already has a running stream",
                ));
            }
            // The last stream ended with an error: take the radio back.
            self.stop_rx();
        }
        let rate = exact_rate(self.rate_hz).ok_or_else(|| invalid("no rate set"))?;
        let radio = self.radio.take().ok_or_else(lost)?;
        let capacity = self
            .options
            .queue_transfers
            .max(1)
            .saturating_mul(self.options.buffer_bytes);
        let queue = Arc::new(Queue::new(capacity));
        self.spawn_reader(radio, Arc::clone(&queue))?;
        let timeout = self
            .options
            .read_timeout
            .max(self.transfer_duration().saturating_mul(4));
        Ok(RtlRx {
            queue,
            rate,
            next_index: 0,
            anchored: false,
            timeout,
        })
    }

    /// Stops the RX stream: its reads return zero bytes from now on.
    pub fn stop_rx(&mut self) {
        if let Some((reader, queue)) = self.reader.take() {
            self.radio = reader.stop();
            queue.stop();
        }
    }

    /// Whether a stream runs (and has not failed).
    pub fn is_streaming(&self) -> bool {
        self.reader.as_ref().is_some_and(|(_, q)| !q.failed())
    }

    /// Bytes the running stream dropped because it was not read fast enough.
    pub fn lost_bytes(&self) -> u64 {
        self.reader.as_ref().map_or(0, |(_, q)| q.lost_bytes())
    }

    /// Whether the radio is still there (not lost to a panicked reader
    /// thread).
    pub fn has_radio(&self) -> bool {
        self.radio.is_some() || self.reader.is_some()
    }

    /// The time one transfer takes at the current rate.
    pub fn transfer_duration(&self) -> Duration {
        let samples = (self.options.buffer_bytes / 2) as u64;
        let hz = u64::from(self.rate_hz.max(1));
        Duration::from_nanos(samples.saturating_mul(1_000_000_000) / hz)
    }
}

fn lost() -> DriverError {
    DriverError::new(
        DriverErrorKind::InvalidState,
        "the radio was lost; open the dongle again",
    )
}

impl Drop for RtlSdr {
    fn drop(&mut self) {
        self.stop_rx();
    }
}

impl sdrk_device::Device for RtlSdr {
    type Error = DriverError;

    fn capabilities(&self) -> &Capabilities {
        &CAPABILITIES
    }

    fn driver(&self) -> &str {
        DRIVER
    }

    fn serial(&self) -> Option<&str> {
        self.serial.as_deref()
    }
}

impl Tuner for RtlSdr {
    type Error = DriverError;

    fn set_frequency(&mut self, dir: Direction, ch: u8, hz: f64) -> Result<f64, DriverError> {
        check_rx(dir, ch)?;
        if !hz.is_finite() || hz < 0.0 || hz > f64::from(u32::MAX) {
            return Err(invalid(format!("{hz} Hz is not a tunable frequency")));
        }
        let whole = hz.round() as u32;
        self.set_center_hz(whole)?;
        Ok(f64::from(whole))
    }

    fn frequency(&self, dir: Direction, ch: u8) -> Result<f64, DriverError> {
        check_rx(dir, ch)?;
        Ok(f64::from(self.center_hz))
    }

    fn frequency_ranges(&self, dir: Direction, ch: u8) -> &[Range] {
        if check_rx(dir, ch).is_ok() {
            &self.freq_ranges
        } else {
            &[]
        }
    }
}

impl RateControl for RtlSdr {
    type Error = DriverError;

    fn set_sample_rate(&mut self, dir: Direction, ch: u8, rate: Rate) -> Result<Rate, DriverError> {
        check_rx(dir, ch)?;
        let hz = nearest_hz(rate).ok_or_else(|| invalid("the rate is too high"))?;
        self.set_rate_hz(hz)
    }

    fn sample_rates(&self, dir: Direction, ch: u8) -> &[Range] {
        if check_rx(dir, ch).is_ok() {
            &RATE_RANGES
        } else {
            &[]
        }
    }
}

impl GainControl for RtlSdr {
    type Error = DriverError;

    fn gain_stages(&self, dir: Direction, ch: u8) -> &[(&'static str, Range)] {
        if check_rx(dir, ch).is_ok() {
            &self.gain_stages
        } else {
            &[]
        }
    }

    fn set_gain(
        &mut self,
        dir: Direction,
        ch: u8,
        stage: Option<&str>,
        db: f64,
    ) -> Result<f64, DriverError> {
        check_rx(dir, ch)?;
        if let Some(s) = stage
            && s != GAIN_STAGE
        {
            return Err(invalid(format!(
                "no gain stage {s}; the stage is {GAIN_STAGE}"
            )));
        }
        if !db.is_finite() {
            return Err(invalid("the gain is not a number"));
        }
        let tenths = self.set_gain_db(db)?;
        Ok(f64::from(tenths) / 10.0)
    }

    fn set_agc(&mut self, dir: Direction, ch: u8, on: bool) -> Result<(), DriverError> {
        check_rx(dir, ch)?;
        self.set_agc_on(on)
    }
}
