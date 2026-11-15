//! [`HackRf`]: one open board with the device traits of `sdrk-device`.

use crate::driver::{
    BoardInfo, Driver, DriverError, DriverErrorKind, Radio, TransceiverMode, UsbInfo,
};
use crate::protocol::FIRMWARE_BUFFER;
use crate::stream::{HackRx, HackTx, Reader, RxQueue, SAMPLE_BYTES, TxQueue, Writer};
use crate::tuning::{
    AMP_DB, BANDWIDTHS_HZ, FREQ_RANGE, GainMapping, HwRate, LNA, MAX_FREQ_HZ, MIN_FREQ_HZ,
    RATE_RANGE, VGA, auto_bandwidth, gain_stages, map_gain, map_rate, nearest_bandwidth,
};
use sdrk_device::{
    BandwidthControl, Capabilities, Direction, GainControl, Range, Rate, RateControl, SampleFormat,
    Tuner, formats,
};
use std::sync::Arc;
use std::time::Duration;

/// The name of the driver (`Device::driver`).
pub const DRIVER: &str = "hackrf";

/// Centre frequency of `rx0` after opening (`tx0` has none until it is set).
pub const DEFAULT_RX_CENTER_HZ: u64 = 100_000_000;
/// Sample rate of both channels after opening.
pub const DEFAULT_RATE_HZ: u32 = 10_000_000;
/// LNA gain of `rx0` after opening.
pub const DEFAULT_LNA_DB: u32 = 16;
/// VGA gain of `rx0` after opening.
pub const DEFAULT_RX_VGA_DB: u32 = 16;

const FORMATS: [SampleFormat; 1] = [formats::CS8];

const CAPABILITIES: Capabilities = Capabilities {
    rx_channels: 1,
    tx_channels: 1,
    full_duplex: false,
    formats: &FORMATS,
    hw_timestamps: false,
    timed_commands: false,
};

const FREQ_RANGES: [Range; 1] = [FREQ_RANGE];
const MIN_BANDWIDTH_HZ: u32 = BANDWIDTHS_HZ[0];
const RATE_RANGES: [Range; 1] = [RATE_RANGE];
const fn fixed(hz: u32) -> Range {
    Range {
        min: hz as f64,
        max: hz as f64,
        step: 0.0,
    }
}

const BANDWIDTH_RANGES: [Range; 16] = {
    let b = BANDWIDTHS_HZ;
    [
        fixed(b[0]),
        fixed(b[1]),
        fixed(b[2]),
        fixed(b[3]),
        fixed(b[4]),
        fixed(b[5]),
        fixed(b[6]),
        fixed(b[7]),
        fixed(b[8]),
        fixed(b[9]),
        fixed(b[10]),
        fixed(b[11]),
        fixed(b[12]),
        fixed(b[13]),
        fixed(b[14]),
        fixed(b[15]),
    ]
};

/// How the streams move samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamOptions {
    /// RX bulk transfers kept in flight.
    pub rx_buffers: usize,
    /// Bytes per RX transfer, a multiple of 512 (libhackrf: 262144).
    pub rx_buffer_bytes: usize,
    /// RX transfers the queue between the USB thread and the stream holds
    /// before it drops new ones.
    pub queue_transfers: usize,
    /// A read fails with `Timeout` when no samples arrive for this long (at
    /// least four transfers).
    pub read_timeout: Duration,
    /// TX bulk transfers kept in flight.
    pub tx_buffers: usize,
    /// Bytes per TX transfer, a multiple of 16384 (the transfer size of the
    /// firmware).
    pub tx_buffer_bytes: usize,
    /// Bytes the TX queue between the stream and the USB thread holds.
    pub tx_queue_bytes: usize,
    /// A write fails with `Timeout` when the queue has no space for this
    /// long.
    pub write_timeout: Duration,
}

impl Default for StreamOptions {
    fn default() -> StreamOptions {
        StreamOptions {
            rx_buffers: 8,
            rx_buffer_bytes: 262_144,
            queue_transfers: 16,
            read_timeout: Duration::from_secs(2),
            tx_buffers: 4,
            tx_buffer_bytes: 65_536,
            tx_queue_bytes: 524_288,
            write_timeout: Duration::from_secs(1),
        }
    }
}

impl StreamOptions {
    /// Checks the values.
    ///
    /// # Errors
    /// A description of the first bad value.
    pub fn validate(&self) -> Result<(), String> {
        if self.rx_buffers == 0 || self.tx_buffers == 0 {
            return Err("rx_buffers and tx_buffers must be positive".to_owned());
        }
        if self.rx_buffer_bytes == 0 || !self.rx_buffer_bytes.is_multiple_of(512) {
            return Err("rx_buffer_bytes must be a positive multiple of 512".to_owned());
        }
        if self.tx_buffer_bytes == 0 || !self.tx_buffer_bytes.is_multiple_of(0x4000) {
            return Err("tx_buffer_bytes must be a positive multiple of 16384".to_owned());
        }
        Ok(())
    }
}

/// The values of one channel. Each channel keeps its own; they reach the
/// hardware when its stream starts, and at once while it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelSettings {
    /// Centre frequency; `tx0` has none until it is set.
    pub center_hz: Option<u64>,
    /// Sample rate as `freq_hz / divider`.
    pub rate: HwRate,
    /// Baseband filter; `None` follows the rate as in libhackrf
    /// ([`auto_bandwidth`]).
    pub bandwidth_hz: Option<u32>,
    /// LNA gain (RX only).
    pub lna_db: u32,
    /// VGA gain (RX baseband or TX).
    pub vga_db: u32,
    /// RF amplifier (about 14 dB); only ever switched on explicitly.
    pub amp: bool,
    /// Antenna port power.
    pub bias_tee: bool,
}

impl ChannelSettings {
    /// The values of a channel after opening.
    pub fn initial(dir: Direction) -> ChannelSettings {
        let rx = dir == Direction::Rx;
        ChannelSettings {
            center_hz: rx.then_some(DEFAULT_RX_CENTER_HZ),
            rate: HwRate {
                freq_hz: DEFAULT_RATE_HZ,
                divider: 1,
            },
            bandwidth_hz: None,
            lna_db: if rx { DEFAULT_LNA_DB } else { 0 },
            vga_db: if rx { DEFAULT_RX_VGA_DB } else { 0 },
            amp: false,
            bias_tee: false,
        }
    }

    /// The baseband filter in effect.
    pub fn effective_bandwidth_hz(&self) -> u32 {
        self.bandwidth_hz
            .unwrap_or_else(|| self.rate.rate().map_or(MIN_BANDWIDTH_HZ, auto_bandwidth))
    }
}

enum Running {
    Idle,
    Rx { reader: Reader, queue: Arc<RxQueue> },
    Tx { writer: Writer, queue: Arc<TxQueue> },
}

fn invalid(message: impl Into<String>) -> DriverError {
    DriverError::new(DriverErrorKind::InvalidArgument, message)
}

fn state_error(message: impl Into<String>) -> DriverError {
    DriverError::new(DriverErrorKind::InvalidState, message)
}

fn check_channel(ch: u8) -> Result<(), DriverError> {
    if ch == 0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "a HackRF has the channels rx0 and tx0, not channel {ch}"
        )))
    }
}

fn name(dir: Direction) -> &'static str {
    match dir {
        Direction::Rx => "rx",
        Direction::Tx => "tx",
    }
}

/// Switches the transceiver, the RF amplifier and the antenna power of a
/// board off. Every request is tried; the first error is returned.
///
/// # Errors
/// The first error of the driver.
pub fn transmitter_off(r: &dyn Radio) -> Result<(), DriverError> {
    let mode = r.set_transceiver_mode(TransceiverMode::Off);
    let amp = r.set_amp(false);
    let antenna = r.set_antenna_power(false);
    mode.and(amp).and(antenna)
}

/// One open HackRF One: half duplex, `rx0` and `tx0` share the RF path.
///
/// Opening switches the transmitter off (transceiver, amplifier, antenna
/// power), as do [`HackRf::stop`], [`HackRf::tx_off`] and dropping the
/// value. Each channel keeps its own values ([`ChannelSettings`]): they are
/// put on the hardware when the stream of the channel starts and at once
/// while it runs; the sample rate of a running channel cannot change (stop
/// it first). A stream of one direction while the other runs fails with
/// `InvalidState`. The transmitter switches to transmit with the first
/// samples written, not before.
///
/// The gain without a stage name is the sum of the stages without the
/// amplifier: LNA first in steps of 8 dB, the rest on the VGA (RX), the VGA
/// alone (TX). The amplifier is only switched by its own stage `amp`.
///
/// ```
/// use sdrk_device::{Direction, GainControl, RxStream, Tuner};
/// use sdrk_hackrf::fake::{FakeBoard, FakeDriver};
/// use sdrk_hackrf::{Driver, HackRf, StreamOptions};
///
/// let bus = FakeDriver::new(vec![FakeBoard::example()]);
/// let info = bus.list()?.remove(0);
/// let mut sdr = HackRf::open(&bus, &info, StreamOptions::default())?;
/// sdr.set_frequency(Direction::Rx, 0, 433.92e6)?;
/// assert_eq!(sdr.set_gain(Direction::Rx, 0, None, 30.0)?, 30.0);
/// let mut rx = sdr.start_rx()?;
/// bus.feed(0, 1);
/// let mut buf = vec![0u8; 4096];
/// let (n, meta) = rx.read(&mut buf)?;
/// assert_eq!((n, meta.start.0), (4096, 0));
/// sdr.stop()?;
/// # Ok::<(), sdrk_hackrf::DriverError>(())
/// ```
pub struct HackRf {
    radio: Arc<dyn Radio>,
    board: BoardInfo,
    serial: Option<String>,
    options: StreamOptions,
    rx: ChannelSettings,
    tx: ChannelSettings,
    clkout: bool,
    running: Running,
}

impl core::fmt::Debug for HackRf {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HackRf")
            .field("board", &self.board)
            .field("rx", &self.rx)
            .field("tx", &self.tx)
            .field("running", &self.running_direction())
            .finish_non_exhaustive()
    }
}

impl HackRf {
    /// Opens a board of a bus listing ([`Driver::list`]) and switches its
    /// transmitter off.
    ///
    /// # Errors
    /// The error of the driver.
    pub fn open(
        driver: &dyn Driver,
        info: &UsbInfo,
        options: StreamOptions,
    ) -> Result<HackRf, DriverError> {
        let radio = driver.open(info)?;
        HackRf::from_radio(radio, info.serial.clone(), options)
    }

    /// Wraps an opened board and switches its transmitter off.
    ///
    /// # Errors
    /// `InvalidArgument` for invalid options; the error of the driver.
    pub fn from_radio(
        radio: Arc<dyn Radio>,
        usb_serial: Option<String>,
        options: StreamOptions,
    ) -> Result<HackRf, DriverError> {
        options.validate().map_err(invalid)?;
        transmitter_off(radio.as_ref())?;
        let board = radio.board();
        let serial = board.serial.clone().or(usb_serial);
        Ok(HackRf {
            radio,
            board,
            serial,
            options,
            rx: ChannelSettings::initial(Direction::Rx),
            tx: ChannelSettings::initial(Direction::Tx),
            clkout: false,
            running: Running::Idle,
        })
    }

    /// What the firmware reported when the board was opened.
    pub fn board(&self) -> &BoardInfo {
        &self.board
    }

    /// The values of a channel.
    pub fn settings(&self, dir: Direction) -> &ChannelSettings {
        match dir {
            Direction::Rx => &self.rx,
            Direction::Tx => &self.tx,
        }
    }

    fn settings_mut(&mut self, dir: Direction) -> &mut ChannelSettings {
        match dir {
            Direction::Rx => &mut self.rx,
            Direction::Tx => &mut self.tx,
        }
    }

    /// The direction whose stream runs without an error.
    pub fn running_direction(&self) -> Option<Direction> {
        match &self.running {
            Running::Rx { queue, .. } if !queue.failed() => Some(Direction::Rx),
            Running::Tx { queue, .. } if !queue.failed() => Some(Direction::Tx),
            _ => None,
        }
    }

    /// Bytes the running RX stream dropped because it was not read fast
    /// enough.
    pub fn lost_bytes(&self) -> u64 {
        match &self.running {
            Running::Rx { queue, .. } => queue.lost_bytes(),
            _ => 0,
        }
    }

    /// TX underruns the firmware reported for the running TX stream.
    pub fn underruns(&self) -> u64 {
        match &self.running {
            Running::Tx { queue, .. } => queue.underruns(),
            _ => 0,
        }
    }

    /// Changes the values of a channel with `f`, and puts what changed on
    /// the hardware at once if the channel runs. The values are kept only
    /// if the hardware took them.
    fn update(
        &mut self,
        dir: Direction,
        f: impl FnOnce(&mut ChannelSettings),
    ) -> Result<(), DriverError> {
        let old = *self.settings(dir);
        let mut new = old;
        f(&mut new);
        if self.running_direction() == Some(dir) {
            self.apply(dir, Some(&old), &new)?;
        }
        *self.settings_mut(dir) = new;
        Ok(())
    }

    /// Puts the values of `new` that differ from `old` (all with `None`) on
    /// the hardware.
    fn apply(
        &self,
        dir: Direction,
        old: Option<&ChannelSettings>,
        new: &ChannelSettings,
    ) -> Result<(), DriverError> {
        let r = self.radio.as_ref();
        let differs = |f: &dyn Fn(&ChannelSettings) -> bool| old.is_none_or(f);
        if let Some(hz) = new.center_hz
            && differs(&|o| o.center_hz != new.center_hz)
        {
            r.set_freq(hz)?;
        }
        if differs(&|o| o.rate != new.rate) {
            r.set_sample_rate(new.rate.freq_hz, new.rate.divider)?;
        }
        let bw = new.effective_bandwidth_hz();
        if differs(&|o| o.effective_bandwidth_hz() != bw) {
            r.set_baseband_filter(bw)?;
        }
        match dir {
            Direction::Rx => {
                if differs(&|o| o.lna_db != new.lna_db) {
                    r.set_lna_gain(new.lna_db)?;
                }
                if differs(&|o| o.vga_db != new.vga_db) {
                    r.set_vga_gain(new.vga_db)?;
                }
            }
            Direction::Tx => {
                if differs(&|o| o.vga_db != new.vga_db) {
                    r.set_txvga_gain(new.vga_db)?;
                }
            }
        }
        if differs(&|o| o.amp != new.amp) {
            r.set_amp(new.amp)?;
        }
        if differs(&|o| o.bias_tee != new.bias_tee) {
            match &self.running {
                // The writer may switch the antenna power on again after the
                // mode switch; it reads the value under the same lock.
                Running::Tx { queue, .. } if dir == Direction::Tx => {
                    queue.with_antenna(|want| {
                        let result = r.set_antenna_power(new.bias_tee);
                        if result.is_ok() {
                            *want = new.bias_tee;
                        }
                        result
                    })?;
                }
                _ => r.set_antenna_power(new.bias_tee)?,
            }
        }
        Ok(())
    }

    /// Tunes a channel to `hz` (1 MHz to 6 GHz).
    ///
    /// # Errors
    /// `InvalidArgument` outside the range; the error of the driver.
    pub fn set_center_hz(&mut self, dir: Direction, hz: u64) -> Result<(), DriverError> {
        if !(MIN_FREQ_HZ..=MAX_FREQ_HZ).contains(&hz) {
            return Err(invalid(format!(
                "{hz} Hz is outside {MIN_FREQ_HZ} to {MAX_FREQ_HZ} Hz"
            )));
        }
        self.update(dir, |s| s.center_hz = Some(hz))
    }

    /// Sets the rate of a channel that does not run: the nearest
    /// `freq_hz / divider` with a divider up to 31; returns that rate.
    ///
    /// # Errors
    /// `InvalidArgument` outside 2 to 20 MS/s; `InvalidState` while the
    /// channel runs.
    pub fn set_rate(&mut self, dir: Direction, rate: Rate) -> Result<Rate, DriverError> {
        let hw = map_rate(rate)
            .ok_or_else(|| invalid(format!("{} S/s is outside 2 to 20 MS/s", rate.as_f64())))?;
        if self.running_direction() == Some(dir) && self.settings(dir).rate != hw {
            return Err(state_error(format!(
                "{}0 runs; stop it before changing its rate",
                name(dir)
            )));
        }
        let exact = hw.rate().map_err(|e| invalid(e.to_string()))?;
        self.update(dir, |s| s.rate = hw)?;
        Ok(exact)
    }

    /// Sets the baseband filter of a channel to the nearest filter of the
    /// MAX2837, or back to following the rate with `None`; returns the
    /// filter in effect.
    ///
    /// # Errors
    /// The error of the driver.
    pub fn set_bandwidth_hz(
        &mut self,
        dir: Direction,
        hz: Option<u64>,
    ) -> Result<u32, DriverError> {
        let bw = hz.map(nearest_bandwidth);
        self.update(dir, |s| s.bandwidth_hz = bw)?;
        Ok(self.settings(dir).effective_bandwidth_hz())
    }

    /// Sets one gain stage (`lna`, `vga`, `amp`) to its nearest step;
    /// returns the value set.
    ///
    /// # Errors
    /// `InvalidArgument` for an unknown stage or a value out of range; the
    /// error of the driver.
    pub fn set_stage_db(
        &mut self,
        dir: Direction,
        stage: &str,
        db: f64,
    ) -> Result<u32, DriverError> {
        let tx = dir == Direction::Tx;
        let value = match map_gain(tx, stage, db) {
            GainMapping::Value(v) => v,
            GainMapping::Unknown => {
                return Err(invalid(format!("{}0 has no gain stage {stage}", name(dir))));
            }
            GainMapping::OutOfRange(min, max) => {
                return Err(invalid(format!(
                    "{db} dB is outside {min} to {max} dB for {stage}"
                )));
            }
        };
        match stage {
            LNA => self.update(dir, |s| s.lna_db = value)?,
            VGA => self.update(dir, |s| s.vga_db = value)?,
            _ => self.update(dir, |s| s.amp = value >= AMP_DB)?,
        }
        Ok(value)
    }

    /// Distributes an overall gain over the stages without the amplifier
    /// (RX: LNA in steps of 8 dB first, the rest on the VGA in steps of
    /// 2 dB; TX: the VGA); returns the sum set.
    ///
    /// # Errors
    /// `InvalidArgument` for a value that is not a number; the error of the
    /// driver.
    pub fn set_overall_db(&mut self, dir: Direction, db: f64) -> Result<f64, DriverError> {
        if !db.is_finite() {
            return Err(invalid("the gain is not a number"));
        }
        let db = db.max(0.0);
        match dir {
            Direction::Rx => {
                let lna = ((db / 8.0).floor() * 8.0).min(40.0);
                let vga = (((db - lna) / 2.0).round() * 2.0).clamp(0.0, 62.0);
                let (lna, vga) = (lna as u32, vga as u32);
                self.update(dir, |s| {
                    s.lna_db = lna;
                    s.vga_db = vga;
                })?;
                Ok(f64::from(lna + vga))
            }
            Direction::Tx => {
                let vga = db.round().min(47.0) as u32;
                self.update(dir, |s| s.vga_db = vga)?;
                Ok(f64::from(vga))
            }
        }
    }

    /// Switches the antenna port power (bias tee) of a channel.
    ///
    /// # Errors
    /// The error of the driver.
    pub fn set_bias_tee(&mut self, dir: Direction, on: bool) -> Result<(), DriverError> {
        self.update(dir, |s| s.bias_tee = on)
    }

    /// Switches the clock output CLKOUT (firmware with USB API 1.03 and
    /// later).
    ///
    /// # Errors
    /// `Unsupported` on older firmware; the error of the driver.
    pub fn set_clkout(&mut self, on: bool) -> Result<(), DriverError> {
        self.radio.set_clkout(on)?;
        self.clkout = on;
        Ok(())
    }

    /// Whether the clock output is on.
    pub fn clkout(&self) -> bool {
        self.clkout
    }

    /// Refuses a stream of `dir` if one runs, cleans up a stream that ended
    /// with an error and puts the values of `dir` on the hardware.
    fn prepare(&mut self, dir: Direction) -> Result<(), DriverError> {
        match self.running_direction() {
            Some(d) if d == dir => {
                return Err(state_error(format!(
                    "{}0 already has a running stream",
                    name(dir)
                )));
            }
            Some(other) => {
                return Err(state_error(format!(
                    "{}0 runs; RX and TX share the RF path of the HackRF",
                    name(other)
                )));
            }
            None => {}
        }
        if !matches!(self.running, Running::Idle) {
            // The last stream ended with an error; the board is switched
            // off either way.
            drop(self.halt());
        }
        let s = *self.settings(dir);
        if s.center_hz.is_none() {
            return Err(state_error(format!(
                "{}0 has no centre frequency; set one first",
                name(dir)
            )));
        }
        self.apply(dir, None, &s)
    }

    fn thread_name(&self, dir: Direction) -> String {
        let serial = self.serial.as_deref().unwrap_or("board");
        let short = serial
            .get(serial.len().saturating_sub(8)..)
            .unwrap_or(serial);
        format!("hackrf-{}-{short}", name(dir))
    }

    fn duration_of(rate: HwRate, bytes: usize) -> Duration {
        let samples = (bytes / SAMPLE_BYTES) as u128;
        let ns = samples * 1_000_000_000 * u128::from(rate.divider.max(1))
            / u128::from(rate.freq_hz.max(1));
        Duration::from_nanos(u64::try_from(ns).unwrap_or(u64::MAX))
    }

    /// Starts the RX stream with the values of `rx0`.
    ///
    /// # Errors
    /// `InvalidState` if a stream runs; the error of the driver.
    pub fn start_rx(&mut self) -> Result<HackRx, DriverError> {
        self.prepare(Direction::Rx)?;
        let rate = self.rx.rate.rate().map_err(|e| invalid(e.to_string()))?;
        if let Err(e) = self.radio.set_transceiver_mode(TransceiverMode::Receive) {
            drop(self.radio.set_transceiver_mode(TransceiverMode::Off));
            return Err(e);
        }
        // Some firmware versions switch the antenna power off with the mode.
        if self.rx.bias_tee {
            self.radio.set_antenna_power(true)?;
        }
        let capacity = self
            .options
            .queue_transfers
            .max(1)
            .saturating_mul(self.options.rx_buffer_bytes);
        let queue = Arc::new(RxQueue::new(capacity));
        let reader = match Reader::spawn(
            self.thread_name(Direction::Rx),
            Arc::clone(&self.radio),
            Arc::clone(&queue),
            self.options.rx_buffers,
            self.options.rx_buffer_bytes,
        ) {
            Ok(r) => r,
            Err(e) => {
                drop(self.radio.set_transceiver_mode(TransceiverMode::Off));
                return Err(e);
            }
        };
        self.running = Running::Rx {
            reader,
            queue: Arc::clone(&queue),
        };
        let timeout = self
            .options
            .read_timeout
            .max(Self::duration_of(self.rx.rate, self.options.rx_buffer_bytes).saturating_mul(4));
        Ok(HackRx::new(queue, rate, timeout))
    }

    /// Starts the TX stream with the values of `tx0`; the board switches to
    /// transmit with the first samples written.
    ///
    /// # Errors
    /// `InvalidState` if a stream runs or `tx0` has no centre frequency; the
    /// error of the driver.
    pub fn start_tx(&mut self) -> Result<HackTx, DriverError> {
        self.prepare(Direction::Tx)?;
        let rate = self.tx.rate.rate().map_err(|e| invalid(e.to_string()))?;
        // After the last transfer the board plays out its own buffer; a
        // burst is done only then (and 10 ms later, for the USB round trip).
        let settle = Self::duration_of(self.tx.rate, FIRMWARE_BUFFER)
            .saturating_add(Duration::from_millis(10));
        let queue = Arc::new(TxQueue::new(
            self.options
                .tx_queue_bytes
                .max(self.options.tx_buffer_bytes),
            settle,
        ));
        let bias_tee = self.tx.bias_tee;
        queue.with_antenna(|on| *on = bias_tee);
        let writer = Writer::spawn(
            self.thread_name(Direction::Tx),
            Arc::clone(&self.radio),
            Arc::clone(&queue),
            self.options.tx_buffers,
            self.options.tx_buffer_bytes,
        )?;
        self.running = Running::Tx {
            writer,
            queue: Arc::clone(&queue),
        };
        Ok(HackTx::new(queue, rate, self.options.write_timeout))
    }

    /// Stops what runs on the RF path and switches the transceiver, the
    /// amplifier and the antenna power off. The stream is stopped even if
    /// switching off fails.
    ///
    /// # Errors
    /// The first error of the driver while switching off.
    pub fn stop(&mut self) -> Result<(), DriverError> {
        self.halt()
    }

    /// Switches the transmitter off: stops a running TX stream; a running
    /// RX stream goes on.
    ///
    /// # Errors
    /// The first error of the driver.
    pub fn tx_off(&mut self) -> Result<(), DriverError> {
        match &self.running {
            Running::Rx { queue, .. } if !queue.failed() => Ok(()),
            _ => self.halt(),
        }
    }

    fn halt(&mut self) -> Result<(), DriverError> {
        let mut panicked = false;
        match std::mem::replace(&mut self.running, Running::Idle) {
            Running::Idle => {}
            Running::Rx { reader, queue } => {
                panicked |= !reader.stop();
                queue.stop();
            }
            Running::Tx { writer, queue } => {
                panicked |= !writer.stop();
                queue.stop();
            }
        }
        let off = transmitter_off(self.radio.as_ref());
        if panicked {
            return Err(DriverError::new(
                DriverErrorKind::Other,
                "the stream thread of the HackRF panicked",
            ));
        }
        off
    }
}

impl Drop for HackRf {
    fn drop(&mut self) {
        // Nothing to report to: the board is switched off as far as it can
        // be.
        drop(self.halt());
    }
}

impl sdrk_device::Device for HackRf {
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

impl Tuner for HackRf {
    type Error = DriverError;

    fn set_frequency(&mut self, dir: Direction, ch: u8, hz: f64) -> Result<f64, DriverError> {
        check_channel(ch)?;
        if !hz.is_finite() || hz < 0.0 {
            return Err(invalid(format!("{hz} Hz is not a tunable frequency")));
        }
        let whole = hz.round() as u64;
        self.set_center_hz(dir, whole)?;
        Ok(whole as f64)
    }

    fn frequency(&self, dir: Direction, ch: u8) -> Result<f64, DriverError> {
        check_channel(ch)?;
        self.settings(dir)
            .center_hz
            .map(|hz| hz as f64)
            .ok_or_else(|| state_error(format!("{}0 has no centre frequency", name(dir))))
    }

    fn frequency_ranges(&self, _dir: Direction, ch: u8) -> &[Range] {
        if ch == 0 { &FREQ_RANGES } else { &[] }
    }
}

impl RateControl for HackRf {
    type Error = DriverError;

    fn set_sample_rate(&mut self, dir: Direction, ch: u8, rate: Rate) -> Result<Rate, DriverError> {
        check_channel(ch)?;
        self.set_rate(dir, rate)
    }

    fn sample_rates(&self, _dir: Direction, ch: u8) -> &[Range] {
        if ch == 0 { &RATE_RANGES } else { &[] }
    }
}

impl BandwidthControl for HackRf {
    type Error = DriverError;

    fn set_bandwidth(&mut self, dir: Direction, ch: u8, hz: f64) -> Result<f64, DriverError> {
        check_channel(ch)?;
        if !hz.is_finite() || hz < 0.0 {
            return Err(invalid(format!("{hz} Hz is not a bandwidth")));
        }
        let bw = self.set_bandwidth_hz(dir, Some(hz.round() as u64))?;
        Ok(f64::from(bw))
    }

    fn bandwidths(&self, _dir: Direction, ch: u8) -> &[Range] {
        if ch == 0 { &BANDWIDTH_RANGES } else { &[] }
    }
}

impl GainControl for HackRf {
    type Error = DriverError;

    fn gain_stages(&self, dir: Direction, ch: u8) -> &[(&'static str, Range)] {
        if ch == 0 {
            gain_stages(dir == Direction::Tx)
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
        check_channel(ch)?;
        match stage {
            None => self.set_overall_db(dir, db),
            Some(s) => Ok(f64::from(self.set_stage_db(dir, s, db)?)),
        }
    }

    fn set_agc(&mut self, _dir: Direction, ch: u8, on: bool) -> Result<(), DriverError> {
        check_channel(ch)?;
        if on {
            Err(DriverError::new(
                DriverErrorKind::Unsupported,
                "a HackRF has no automatic gain",
            ))
        } else {
            Ok(())
        }
    }
}
