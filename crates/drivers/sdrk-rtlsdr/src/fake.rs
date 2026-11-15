//! A stand-in for the USB driver, for tests without hardware.
//!
//! A [`FakeDriver`] holds a bus of [`FakeDongle`]s. Tests plug and unplug
//! dongles, feed transfers one by one ([`FakeDriver::feed`]) and read back
//! what the backend set on the hardware. Sample `k` of a dongle (counted
//! from its first transfer) is the byte pair `(k mod 256, k / 256 mod 256)`,
//! so a test can tell from the bytes which samples arrived.

use crate::driver::{
    Cancel, DirectSampling, Driver, DriverError, DriverErrorKind, Radio, TunerKind, UsbInfo,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// The gain list of the R820T (librtlsdr `r82xx_gains`), tenths of a dB.
pub const R820T_GAINS: [i32; 29] = [
    0, 9, 14, 27, 37, 77, 87, 125, 144, 157, 166, 197, 207, 229, 254, 280, 297, 328, 338, 364, 372,
    386, 402, 421, 434, 439, 445, 480, 496,
];

/// What the backend last set on a dongle.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FakeSettings {
    /// Centre frequency.
    pub center_hz: Option<u32>,
    /// Sample rate (with the IF filter of the tuner that follows it);
    /// `None` again after the tuner was initialised by leaving direct
    /// sampling.
    pub rate_hz: Option<u32>,
    /// Tuner gain: `Some(None)` is automatic; `None` again after the tuner
    /// was initialised by leaving direct sampling.
    pub gain_tenths: Option<Option<i32>>,
    /// Frequency correction.
    pub ppm: Option<i32>,
    /// Bias tee.
    pub bias_tee: Option<bool>,
    /// Direct sampling.
    pub direct_sampling: Option<DirectSampling>,
}

/// One simulated dongle.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct FakeDongle {
    /// What the bus scan reports.
    pub info: UsbInfo,
    /// Its tuner.
    pub tuner: TunerKind,
    /// RTL-SDR Blog V4 style upconverter.
    pub upconverter: bool,
    /// Gain list in tenths of a dB.
    pub gains: Vec<i32>,
    /// Whether it is plugged in.
    pub present: bool,
    /// The values the backend set.
    pub settings: FakeSettings,
    /// Transfers that the stream may still deliver.
    pub fed: u64,
    /// Samples delivered since the first transfer.
    pub produced: u64,
    /// How often it was opened.
    pub opens: u32,
    /// Whether a stream runs.
    pub streaming: bool,
    /// Buffer count and size of the last stream.
    pub buffers: Option<(usize, usize)>,
    /// The name of the thread that ran the last stream.
    pub stream_thread: Option<String>,
    /// How often a stream was started.
    pub streams: u32,
    /// Transfers delivered, by the centre frequency the dongle was tuned to
    /// when it took them.
    pub transfers_at: BTreeMap<u32, u64>,
    /// Whether a handle holds the dongle: a second open fails as busy, as
    /// the USB interface is claimed.
    pub claimed: bool,
    /// The running stream fails once with `Disconnected` although the dongle
    /// stays on the bus.
    pub break_stream: bool,
    /// How the running stream fails when the dongle is unplugged:
    /// `Disconnected`, or `Other` as when the host controller ends the
    /// transfers in flight with a protocol error (a fault) before it reports
    /// the device as gone.
    pub unplug_error: DriverErrorKind,
    /// Radio calls (by method name, for example `"set_center_freq"`) that
    /// fail with a USB timeout while the dongle stays on the bus.
    pub failing: Vec<&'static str>,
}

impl FakeDongle {
    /// An R820T dongle at USB port `port` of bus 1.
    pub fn r820t(index: usize, serial: Option<&str>, port: u8) -> FakeDongle {
        FakeDongle {
            info: UsbInfo {
                index,
                bus: "1".to_owned(),
                ports: vec![port],
                vendor_id: 0x0bda,
                product_id: 0x2838,
                manufacturer: Some("Realtek".to_owned()),
                product: Some("RTL2838UHIDIR".to_owned()),
                serial: serial.map(str::to_owned),
                device_version: Some(0x0100),
            },
            tuner: TunerKind::R820t,
            upconverter: false,
            gains: R820T_GAINS.to_vec(),
            present: true,
            settings: FakeSettings::default(),
            fed: 0,
            produced: 0,
            opens: 0,
            streaming: false,
            buffers: None,
            stream_thread: None,
            streams: 0,
            transfers_at: BTreeMap::new(),
            claimed: false,
            break_stream: false,
            unplug_error: DriverErrorKind::Disconnected,
            failing: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct Bus {
    dongles: Mutex<Vec<FakeDongle>>,
    cond: Condvar,
}

fn lock(bus: &Bus) -> MutexGuard<'_, Vec<FakeDongle>> {
    bus.dongles.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The simulated bus; clones share it.
#[derive(Debug, Clone)]
pub struct FakeDriver {
    bus: Arc<Bus>,
}

fn gone() -> DriverError {
    DriverError::new(DriverErrorKind::Disconnected, "the dongle was unplugged")
}

impl FakeDriver {
    /// A bus with these dongles.
    pub fn new(dongles: Vec<FakeDongle>) -> FakeDriver {
        FakeDriver {
            bus: Arc::new(Bus {
                dongles: Mutex::new(dongles),
                cond: Condvar::new(),
            }),
        }
    }

    /// A copy of dongle `n` (by its position in the list given to `new`).
    pub fn dongle(&self, n: usize) -> Option<FakeDongle> {
        lock(&self.bus).get(n).cloned()
    }

    fn with(&self, n: usize, f: impl FnOnce(&mut FakeDongle)) {
        if let Some(d) = lock(&self.bus).get_mut(n) {
            f(d);
        }
        self.bus.cond.notify_all();
    }

    /// Lets the stream of dongle `n` deliver `transfers` more transfers.
    pub fn feed(&self, n: usize, transfers: u64) {
        self.with(n, |d| d.fed += transfers);
    }

    /// The hardware of dongle `n` takes `samples` samples that nobody
    /// receives (time passes while no stream reads it): the samples it
    /// delivers next are counted from there.
    pub fn advance(&self, n: usize, samples: u64) {
        self.with(n, |d| d.produced += samples);
    }

    /// Makes the running stream of dongle `n` fail once with `Disconnected`
    /// while the dongle stays on the bus (a USB error).
    pub fn break_stream(&self, n: usize) {
        self.with(n, |d| d.break_stream = true);
    }

    /// Unplugs dongle `n`: it is no longer listed, its stream fails and
    /// every call on it fails with `Disconnected`.
    pub fn unplug(&self, n: usize) {
        self.with(n, |d| d.present = false);
    }

    /// Unplugs dongle `n` like [`FakeDriver::unplug`], but its running
    /// stream fails with an error of `kind` (calls on it still fail with
    /// `Disconnected`).
    pub fn unplug_with(&self, n: usize, kind: DriverErrorKind) {
        self.with(n, |d| {
            d.unplug_error = kind;
            d.present = false;
        });
    }

    /// Makes the radio calls `names` of dongle `n` fail with a USB timeout;
    /// an empty list lets every call work again.
    pub fn fail_calls(&self, n: usize, names: &[&'static str]) {
        self.with(n, |d| d.failing = names.to_vec());
    }

    /// Plugs dongle `n` in again.
    pub fn plug(&self, n: usize) {
        self.with(n, |d| d.present = true);
    }

    /// Waits up to `timeout` until dongle `n` has delivered all fed
    /// transfers; returns whether it has.
    pub fn wait_drained(&self, n: usize, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut guard = lock(&self.bus);
        loop {
            match guard.get(n) {
                Some(d) if d.fed > 0 => {}
                _ => return true,
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            guard = self
                .bus
                .cond
                .wait_timeout(guard, (deadline - now).min(Duration::from_millis(10)))
                .map_or_else(|e| e.into_inner().0, |(g, _)| g);
        }
    }

    /// Waits up to `timeout` until a stream of dongle `n` runs; returns
    /// whether one does. A stream counts in [`FakeDongle::streams`] from the
    /// moment it runs, so after this returns `true` the count includes every
    /// stream started so far: the backend stops a stream (and waits for it
    /// to end) before it starts the next one.
    pub fn wait_streaming(&self, n: usize, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut guard = lock(&self.bus);
        loop {
            if guard.get(n).is_some_and(|d| d.streaming) {
                return true;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            guard = self
                .bus
                .cond
                .wait_timeout(guard, (deadline - now).min(Duration::from_millis(10)))
                .map_or_else(|e| e.into_inner().0, |(g, _)| g);
        }
    }
}

impl Driver for FakeDriver {
    fn list(&self) -> Result<Vec<UsbInfo>, DriverError> {
        Ok(lock(&self.bus)
            .iter()
            .filter(|d| d.present)
            .map(|d| d.info.clone())
            .collect())
    }

    fn open(&self, info: &UsbInfo, by_serial: bool) -> Result<Box<dyn Radio>, DriverError> {
        let mut dongles = lock(&self.bus);
        let slot = dongles
            .iter()
            .position(|d| {
                d.present
                    && if by_serial {
                        d.info.serial.is_some() && d.info.serial == info.serial
                    } else {
                        d.info.index == info.index
                    }
            })
            .ok_or_else(|| DriverError::new(DriverErrorKind::NotFound, "no such dongle"))?;
        let d = dongles
            .get_mut(slot)
            .ok_or_else(|| DriverError::new(DriverErrorKind::NotFound, "no such dongle"))?;
        if d.claimed {
            return Err(DriverError::new(
                DriverErrorKind::Busy,
                "the dongle is already open",
            ));
        }
        d.claimed = true;
        d.opens += 1;
        Ok(Box::new(FakeRadio {
            bus: Arc::clone(&self.bus),
            slot,
            tuner: d.tuner,
            upconverter: d.upconverter,
            gains: d.gains.clone(),
        }))
    }
}

struct FakeRadio {
    bus: Arc<Bus>,
    slot: usize,
    tuner: TunerKind,
    upconverter: bool,
    gains: Vec<i32>,
}

impl FakeRadio {
    fn set(
        &self,
        call: &'static str,
        f: impl FnOnce(&mut FakeSettings),
    ) -> Result<(), DriverError> {
        let mut dongles = lock(&self.bus);
        let d = dongles
            .get_mut(self.slot)
            .filter(|d| d.present)
            .ok_or_else(gone)?;
        if d.failing.contains(&call) {
            return Err(DriverError::new(
                DriverErrorKind::Timeout,
                format!("{call}: the USB control transfer timed out"),
            ));
        }
        f(&mut d.settings);
        Ok(())
    }
}

impl Drop for FakeRadio {
    fn drop(&mut self) {
        if let Some(d) = lock(&self.bus).get_mut(self.slot) {
            d.claimed = false;
        }
        self.bus.cond.notify_all();
    }
}

impl Radio for FakeRadio {
    fn tuner(&self) -> TunerKind {
        self.tuner
    }

    fn has_upconverter(&self) -> bool {
        self.upconverter
    }

    fn tuner_gains(&self) -> Vec<i32> {
        self.gains.clone()
    }

    fn set_center_freq(&mut self, hz: u32) -> Result<(), DriverError> {
        self.set("set_center_freq", |s| s.center_hz = Some(hz))
    }

    fn set_sample_rate(&mut self, hz: u32) -> Result<(), DriverError> {
        self.set("set_sample_rate", |s| s.rate_hz = Some(hz))
    }

    fn set_tuner_gain(&mut self, tenths: Option<i32>) -> Result<(), DriverError> {
        self.set("set_tuner_gain", |s| s.gain_tenths = Some(tenths))
    }

    fn set_ppm(&mut self, ppm: i32) -> Result<(), DriverError> {
        self.set("set_ppm", |s| s.ppm = Some(ppm))
    }

    fn set_bias_tee(&mut self, on: bool) -> Result<(), DriverError> {
        self.set("set_bias_tee", |s| s.bias_tee = Some(on))
    }

    fn set_direct_sampling(&mut self, mode: DirectSampling) -> Result<(), DriverError> {
        self.set("set_direct_sampling", |s| {
            let leaving = mode == DirectSampling::Off
                && s.direct_sampling.is_some_and(|m| m != DirectSampling::Off);
            if leaving {
                // As `rtl-sdr-rs`: leaving direct sampling initialises the
                // tuner again, which loses its gain and the IF filter that
                // the sample rate set.
                s.gain_tenths = None;
                s.rate_hz = None;
            }
            s.direct_sampling = Some(mode);
        })
    }

    fn reset_buffer(&mut self) -> Result<(), DriverError> {
        self.set("reset_buffer", |_| {})
    }

    fn stream(
        &mut self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<(), DriverError> {
        let mut guard = lock(&self.bus);
        if let Some(d) = guard.get_mut(self.slot) {
            d.streaming = true;
            d.buffers = Some((buffers, buffer_bytes));
            d.stream_thread = std::thread::current().name().map(str::to_owned);
            d.streams += 1;
        }
        self.bus.cond.notify_all();
        let result = loop {
            let Some(d) = guard.get_mut(self.slot) else {
                break Err(gone());
            };
            if !d.present {
                break Err(match d.unplug_error {
                    DriverErrorKind::Disconnected => gone(),
                    kind => DriverError::new(kind, "the transfers ended with a protocol error"),
                });
            }
            if std::mem::take(&mut d.break_stream) {
                break Err(gone());
            }
            if cancel.is_cancelled() {
                break Ok(());
            }
            if d.fed == 0 {
                guard = self
                    .bus
                    .cond
                    .wait_timeout(guard, Duration::from_millis(5))
                    .map_or_else(|e| e.into_inner().0, |(g, _)| g);
                continue;
            }
            let first = d.produced;
            let samples = (buffer_bytes / 2) as u64;
            d.produced += samples;
            *d.transfers_at
                .entry(d.settings.center_hz.unwrap_or(0))
                .or_default() += 1;
            drop(guard);
            let mut chunk = Vec::with_capacity(buffer_bytes);
            for k in first..first + samples {
                chunk.push((k & 0xff) as u8);
                chunk.push(((k >> 8) & 0xff) as u8);
            }
            sink(&chunk);
            guard = lock(&self.bus);
            // Counted as delivered only once the backend has it.
            if let Some(d) = guard.get_mut(self.slot) {
                d.fed = d.fed.saturating_sub(1);
            }
            self.bus.cond.notify_all();
        };
        if let Some(d) = guard.get_mut(self.slot) {
            d.streaming = false;
        }
        self.bus.cond.notify_all();
        result
    }
}
