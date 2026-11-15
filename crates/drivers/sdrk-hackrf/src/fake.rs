//! A stand-in for the USB driver, for tests without hardware.
//!
//! A [`FakeDriver`] holds a bus of [`FakeBoard`]s. Tests plug and unplug
//! boards, feed RX transfers one by one ([`FakeDriver::feed`]), read back the
//! vendor requests the backend sent ([`FakeBoard::calls`]) and the bytes of
//! the TX transfers it submitted ([`FakeBoard::tx_bytes`], padding included),
//! and make the firmware report TX
//! underruns ([`FakeDriver::tx_underrun`]). RX sample `k` of a board
//! (counted from its first transfer) is the byte pair
//! `(k mod 256, k / 256 mod 256)`, so a test can tell from the bytes which
//! samples arrived.

use crate::driver::{
    BoardInfo, Cancel, Driver, DriverError, DriverErrorKind, Radio, TransceiverMode, TxSource,
    UsbInfo,
};
use crate::protocol::{API_M0_STATE, PRODUCT_IDS, VENDOR_ID};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// The serial number of the capability example.
pub const EXAMPLE_SERIAL: &str = "0000000000000000457863dc2b5a4c5f";

/// A vendor request the backend sent, with its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeCall {
    /// `SET_TRANSCEIVER_MODE`.
    Mode(TransceiverMode),
    /// `SET_FREQ`.
    Freq(u64),
    /// `SAMPLE_RATE_SET` (`freq_hz`, `divider`).
    SampleRate(u32, u32),
    /// `BASEBAND_FILTER_BANDWIDTH_SET`.
    Filter(u32),
    /// `SET_LNA_GAIN`.
    Lna(u32),
    /// `SET_VGA_GAIN`.
    Vga(u32),
    /// `SET_TXVGA_GAIN`.
    TxVga(u32),
    /// `AMP_ENABLE`.
    Amp(bool),
    /// `ANTENNA_ENABLE`.
    AntennaPower(bool),
    /// `CLKOUT_ENABLE`.
    ClkOut(bool),
}

/// One simulated board.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct FakeBoard {
    /// What the bus scan reports.
    pub info: UsbInfo,
    /// What the firmware reports when the board is opened.
    pub board: BoardInfo,
    /// Whether it is plugged in.
    pub present: bool,
    /// Whether a handle holds the board (a second open fails as busy).
    pub claimed: bool,
    /// How often it was opened.
    pub opens: u32,
    /// Every vendor request, in order.
    pub calls: Vec<FakeCall>,
    /// The transceiver mode.
    pub mode: TransceiverMode,
    /// The RF amplifier.
    pub amp: bool,
    /// The antenna power.
    pub antenna_power: bool,
    /// RX transfers the stream may still deliver.
    pub fed: u64,
    /// RX samples delivered since the first transfer.
    pub produced: u64,
    /// Whether an RX stream runs.
    pub rx_streaming: bool,
    /// How often an RX stream was started.
    pub rx_streams: u32,
    /// Buffer count and size of the last RX stream.
    pub rx_buffers: Option<(usize, usize)>,
    /// Whether a TX stream runs.
    pub tx_streaming: bool,
    /// Buffer count and size of the last TX stream.
    pub tx_buffers: Option<(usize, usize)>,
    /// Every byte of every TX transfer submitted, in order, the zeros a
    /// partial transfer was padded with included.
    pub tx_bytes: Vec<u8>,
    /// The length of every TX transfer submitted and the bytes of it that
    /// carried samples, in order.
    pub tx_lens: Vec<(usize, usize)>,
    /// TX transfers submitted.
    pub tx_transfers: u64,
    /// The most TX transfers that were in flight at once.
    pub tx_max_in_flight: usize,
    /// The size of every TX transfer submitted was a multiple of 16384.
    pub tx_whole_transfers: bool,
    /// Whether the transceiver was in transmit mode when samples were sent.
    pub tx_mode_ok: bool,
    /// The thread names of the last RX and TX streams.
    pub threads: Vec<String>,
    /// Underruns the firmware reports to the running TX stream next.
    pub underruns: u32,
    /// Radio calls (by the variant name, for example `"Freq"`) that fail
    /// with a USB time-out.
    pub failing: Vec<&'static str>,
    /// Radio calls (by the variant name) that the firmware refuses as not
    /// supported (a stalled control request).
    pub refusing: Vec<&'static str>,
}

impl FakeBoard {
    /// A HackRF One r9 with a serial number at USB port `port` of bus 1.
    pub fn one(serial: &str, port: u8) -> FakeBoard {
        FakeBoard {
            info: UsbInfo {
                bus: "1".to_owned(),
                ports: vec![port],
                vendor_id: VENDOR_ID,
                product_id: PRODUCT_IDS[0],
                manufacturer: Some("Great Scott Gadgets".to_owned()),
                product: Some("HackRF One".to_owned()),
                serial: Some(serial.to_owned()),
                device_version: Some(0x0107),
            },
            board: BoardInfo {
                board_id: 4,
                board_rev: Some(0x84),
                firmware: "2024.02.1".to_owned(),
                usb_api: 0x0107,
                serial: Some(serial.to_ascii_lowercase()),
            },
            present: true,
            claimed: false,
            opens: 0,
            calls: Vec::new(),
            mode: TransceiverMode::Off,
            amp: false,
            antenna_power: false,
            fed: 0,
            produced: 0,
            rx_streaming: false,
            rx_streams: 0,
            rx_buffers: None,
            tx_streaming: false,
            tx_buffers: None,
            tx_bytes: Vec::new(),
            tx_lens: Vec::new(),
            tx_transfers: 0,
            tx_max_in_flight: 0,
            tx_whole_transfers: true,
            tx_mode_ok: true,
            threads: Vec::new(),
            underruns: 0,
            failing: Vec::new(),
            refusing: Vec::new(),
        }
    }

    /// The board of the capability example
    /// (`schema/examples/device.get.response.json`).
    pub fn example() -> FakeBoard {
        FakeBoard::one(EXAMPLE_SERIAL, 1)
    }

    /// The vendor requests since the last call of [`FakeDriver::clear_calls`].
    pub fn calls(&self) -> &[FakeCall] {
        &self.calls
    }
}

#[derive(Debug)]
struct Bus {
    boards: Mutex<Vec<FakeBoard>>,
    cond: Condvar,
}

fn lock(bus: &Bus) -> MutexGuard<'_, Vec<FakeBoard>> {
    bus.boards.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The simulated bus; clones share it.
#[derive(Debug, Clone)]
pub struct FakeDriver {
    bus: Arc<Bus>,
}

fn gone() -> DriverError {
    DriverError::new(DriverErrorKind::Disconnected, "the HackRF was unplugged")
}

impl FakeDriver {
    /// A bus with these boards.
    pub fn new(boards: Vec<FakeBoard>) -> FakeDriver {
        FakeDriver {
            bus: Arc::new(Bus {
                boards: Mutex::new(boards),
                cond: Condvar::new(),
            }),
        }
    }

    /// A copy of board `n` (by its position in the list given to `new`).
    pub fn board(&self, n: usize) -> Option<FakeBoard> {
        lock(&self.bus).get(n).cloned()
    }

    fn with(&self, n: usize, f: impl FnOnce(&mut FakeBoard)) {
        if let Some(d) = lock(&self.bus).get_mut(n) {
            f(d);
        }
        self.bus.cond.notify_all();
    }

    /// Forgets the vendor requests of board `n` recorded so far.
    pub fn clear_calls(&self, n: usize) {
        self.with(n, |d| d.calls.clear());
    }

    /// Lets the RX stream of board `n` deliver `transfers` more transfers.
    pub fn feed(&self, n: usize, transfers: u64) {
        self.with(n, |d| d.fed += transfers);
    }

    /// The firmware of board `n` reports `count` TX underruns to the running
    /// TX stream.
    pub fn tx_underrun(&self, n: usize, count: u32) {
        self.with(n, |d| d.underruns += count);
    }

    /// Unplugs board `n`: it is no longer listed, its streams fail and every
    /// call on it fails with `Disconnected`.
    pub fn unplug(&self, n: usize) {
        self.with(n, |d| d.present = false);
    }

    /// Plugs board `n` in again.
    pub fn plug(&self, n: usize) {
        self.with(n, |d| d.present = true);
    }

    /// Makes the radio calls `names` of board `n` fail with a USB time-out;
    /// an empty list lets every call work again.
    pub fn fail_calls(&self, n: usize, names: &[&'static str]) {
        self.with(n, |d| d.failing = names.to_vec());
    }

    /// Makes the firmware of board `n` refuse the radio calls `names` as not
    /// supported; an empty list lets every call work again.
    pub fn refuse_calls(&self, n: usize, names: &[&'static str]) {
        self.with(n, |d| d.refusing = names.to_vec());
    }

    /// Waits up to `timeout` until `cond` holds for board `n`; returns
    /// whether it does.
    pub fn wait_for(
        &self,
        n: usize,
        timeout: Duration,
        mut cond: impl FnMut(&FakeBoard) -> bool,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        let mut guard = lock(&self.bus);
        loop {
            if guard.get(n).is_some_and(&mut cond) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            guard = self
                .bus
                .cond
                .wait_timeout(
                    guard,
                    deadline
                        .saturating_duration_since(now)
                        .min(Duration::from_millis(10)),
                )
                .map_or_else(|e| e.into_inner().0, |(g, _)| g);
        }
    }

    /// Waits up to `timeout` until board `n` has delivered all fed RX
    /// transfers.
    pub fn wait_drained(&self, n: usize, timeout: Duration) -> bool {
        self.wait_for(n, timeout, |d| d.fed == 0)
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

    fn open(&self, info: &UsbInfo) -> Result<Arc<dyn Radio>, DriverError> {
        let mut boards = lock(&self.bus);
        let slot = boards
            .iter()
            .position(|d| d.present && d.info.bus_path() == info.bus_path())
            .ok_or_else(|| DriverError::new(DriverErrorKind::NotFound, "no such HackRF"))?;
        let d = boards
            .get_mut(slot)
            .ok_or_else(|| DriverError::new(DriverErrorKind::NotFound, "no such HackRF"))?;
        if d.claimed {
            return Err(DriverError::new(
                DriverErrorKind::Busy,
                "the HackRF is already open",
            ));
        }
        d.claimed = true;
        d.opens += 1;
        Ok(Arc::new(FakeRadio {
            bus: Arc::clone(&self.bus),
            slot,
            board: d.board.clone(),
        }))
    }
}

struct FakeRadio {
    bus: Arc<Bus>,
    slot: usize,
    board: BoardInfo,
}

fn call_name(c: &FakeCall) -> &'static str {
    match c {
        FakeCall::Mode(_) => "Mode",
        FakeCall::Freq(_) => "Freq",
        FakeCall::SampleRate(..) => "SampleRate",
        FakeCall::Filter(_) => "Filter",
        FakeCall::Lna(_) => "Lna",
        FakeCall::Vga(_) => "Vga",
        FakeCall::TxVga(_) => "TxVga",
        FakeCall::Amp(_) => "Amp",
        FakeCall::AntennaPower(_) => "AntennaPower",
        FakeCall::ClkOut(_) => "ClkOut",
    }
}

impl FakeRadio {
    fn call(&self, c: FakeCall) -> Result<(), DriverError> {
        let mut boards = lock(&self.bus);
        let d = boards
            .get_mut(self.slot)
            .filter(|d| d.present)
            .ok_or_else(gone)?;
        let name = call_name(&c);
        if d.failing.contains(&name) {
            return Err(DriverError::new(
                DriverErrorKind::Timeout,
                format!("{name}: the USB control transfer timed out"),
            ));
        }
        if d.refusing.contains(&name) {
            return Err(DriverError::new(
                DriverErrorKind::Unsupported,
                format!("{name}: the firmware refused the request"),
            ));
        }
        d.calls.push(c);
        match c {
            FakeCall::Mode(m) => d.mode = m,
            FakeCall::Amp(on) => d.amp = on,
            FakeCall::AntennaPower(on) => d.antenna_power = on,
            _ => {}
        }
        drop(boards);
        self.bus.cond.notify_all();
        Ok(())
    }

    fn thread_name() -> String {
        std::thread::current()
            .name()
            .map(str::to_owned)
            .unwrap_or_default()
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
    fn board(&self) -> BoardInfo {
        self.board.clone()
    }

    fn set_transceiver_mode(&self, mode: TransceiverMode) -> Result<(), DriverError> {
        self.call(FakeCall::Mode(mode))
    }

    fn set_freq(&self, hz: u64) -> Result<(), DriverError> {
        self.call(FakeCall::Freq(hz))
    }

    fn set_sample_rate(&self, freq_hz: u32, divider: u32) -> Result<(), DriverError> {
        self.call(FakeCall::SampleRate(freq_hz, divider))
    }

    fn set_baseband_filter(&self, hz: u32) -> Result<(), DriverError> {
        self.call(FakeCall::Filter(hz))
    }

    fn set_lna_gain(&self, db: u32) -> Result<(), DriverError> {
        self.call(FakeCall::Lna(db))
    }

    fn set_vga_gain(&self, db: u32) -> Result<(), DriverError> {
        self.call(FakeCall::Vga(db))
    }

    fn set_txvga_gain(&self, db: u32) -> Result<(), DriverError> {
        self.call(FakeCall::TxVga(db))
    }

    fn set_amp(&self, on: bool) -> Result<(), DriverError> {
        self.call(FakeCall::Amp(on))
    }

    fn set_antenna_power(&self, on: bool) -> Result<(), DriverError> {
        self.call(FakeCall::AntennaPower(on))
    }

    fn set_clkout(&self, on: bool) -> Result<(), DriverError> {
        self.call(FakeCall::ClkOut(on))
    }

    fn rx_stream(
        &self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<(), DriverError> {
        let mut guard = lock(&self.bus);
        if let Some(d) = guard.get_mut(self.slot) {
            if d.mode != TransceiverMode::Receive {
                return Err(DriverError::new(
                    DriverErrorKind::Other,
                    "RX transfers while the transceiver is not receiving",
                ));
            }
            d.rx_streaming = true;
            d.rx_buffers = Some((buffers, buffer_bytes));
            d.rx_streams += 1;
            d.threads.push(FakeRadio::thread_name());
        }
        self.bus.cond.notify_all();
        let result = loop {
            let Some(d) = guard.get_mut(self.slot) else {
                break Err(gone());
            };
            if !d.present {
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
            drop(guard);
            let mut chunk = Vec::with_capacity(buffer_bytes);
            for k in first..first + samples {
                chunk.push((k & 0xff) as u8);
                chunk.push(((k >> 8) & 0xff) as u8);
            }
            sink(&chunk);
            guard = lock(&self.bus);
            if let Some(d) = guard.get_mut(self.slot) {
                d.fed = d.fed.saturating_sub(1);
            }
            self.bus.cond.notify_all();
        };
        if let Some(d) = guard.get_mut(self.slot) {
            d.rx_streaming = false;
        }
        self.bus.cond.notify_all();
        result
    }

    fn tx_stream(
        &self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        source: &mut dyn TxSource,
    ) -> Result<(), DriverError> {
        {
            let mut guard = lock(&self.bus);
            if let Some(d) = guard.get_mut(self.slot) {
                d.tx_streaming = true;
                d.tx_buffers = Some((buffers, buffer_bytes));
                d.threads.push(FakeRadio::thread_name());
            }
        }
        self.bus.cond.notify_all();
        let mut buf = vec![0u8; buffer_bytes];
        // Transfers submitted and not yet played out by the "hardware": up to
        // `buffers` are kept in flight; the hardware plays one out when all
        // are in flight, or when the source has nothing to submit (time
        // passes).
        let mut in_flight = 0usize;
        let result = loop {
            if cancel.is_cancelled() {
                break Ok(());
            }
            let underruns = {
                let mut guard = lock(&self.bus);
                let Some(d) = guard.get_mut(self.slot).filter(|d| d.present) else {
                    break Err(gone());
                };
                // Firmware older than USB API 1.06 has no underrun counter
                // (as `UsbRadio`): its underruns are not reported.
                let counted = d.board.usb_api >= API_M0_STATE;
                let n = std::mem::take(&mut d.underruns);
                if counted { n } else { 0 }
            };
            if underruns > 0 {
                source.underrun(underruns);
            }
            if in_flight >= buffers.max(1) {
                in_flight -= 1;
                continue;
            }
            // The source sets the bytes after the samples to zero.
            buf.fill(0xa5);
            if let Some(n) = source.fill(&mut buf, in_flight) {
                let len = crate::protocol::tx_transfer_len(n, buffer_bytes);
                in_flight += 1;
                let mut guard = lock(&self.bus);
                if let Some(d) = guard.get_mut(self.slot) {
                    d.tx_transfers += 1;
                    d.tx_max_in_flight = d.tx_max_in_flight.max(in_flight);
                    d.tx_whole_transfers &= len.is_multiple_of(crate::protocol::TX_GRANULE);
                    d.tx_mode_ok &= d.mode == TransceiverMode::Transmit;
                    d.tx_lens.push((len, n));
                    d.tx_bytes
                        .extend_from_slice(buf.get(..len).unwrap_or_default());
                }
                drop(guard);
                self.bus.cond.notify_all();
            } else {
                in_flight = in_flight.saturating_sub(1);
            }
        };
        if let Some(d) = lock(&self.bus).get_mut(self.slot) {
            d.tx_streaming = false;
        }
        self.bus.cond.notify_all();
        result
    }
}
