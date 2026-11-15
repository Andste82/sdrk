//! The interface between the backend and the USB driver.
//!
//! [`Driver`] lists and opens boards, [`Radio`] is one open board. The real
//! implementation is [`crate::usb::UsbDriver`] (the HackRF vendor requests
//! over `nusb`); [`crate::fake::FakeDriver`] stands in for the hardware in
//! tests.
//!
//! A [`Radio`] is shared between the handle and the thread that streams: the
//! control requests (tuning, gains) work while the bulk transfers run, as with
//! libhackrf, so every method takes `&self`.

use sdrk_device::{ErrorKind, SdrkError};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// One HackRF on the USB bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsbInfo {
    /// Identifier of the bus or host controller.
    pub bus: String,
    /// The hub ports from the root hub to the board.
    pub ports: Vec<u8>,
    /// USB vendor ID.
    pub vendor_id: u16,
    /// USB product ID.
    pub product_id: u16,
    /// USB manufacturer string.
    pub manufacturer: Option<String>,
    /// USB product string.
    pub product: Option<String>,
    /// USB serial number string (32 hexadecimal digits on a HackRF One).
    pub serial: Option<String>,
    /// `bcdDevice` of the device descriptor: the USB API version of the
    /// firmware (`0x0106` is 1.06).
    pub device_version: Option<u16>,
}

impl UsbInfo {
    /// The bus path `<bus>-<port>.<port>…`, like the Linux sysfs name.
    pub fn bus_path(&self) -> String {
        let ports: Vec<String> = self.ports.iter().map(u8::to_string).collect();
        if ports.is_empty() {
            format!("{}-0", self.bus)
        } else {
            format!("{}-{}", self.bus, ports.join("."))
        }
    }
}

/// What the firmware reports about the board when it is opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoardInfo {
    /// Board ID (`BOARD_ID_READ`): 2 is a HackRF One, 4 a HackRF One r9.
    pub board_id: u8,
    /// Board revision (`BOARD_REV_READ`, USB API 1.06 and later); `None` if
    /// the firmware cannot tell.
    pub board_rev: Option<u8>,
    /// Firmware version string.
    pub firmware: String,
    /// USB API version of the firmware.
    pub usb_api: u16,
    /// Serial number of the MCU (`BOARD_PARTID_SERIALNO_READ`), 32 lower-case
    /// hexadecimal digits; `None` if it is all zero.
    pub serial: Option<String>,
}

/// The transceiver mode (`SET_TRANSCEIVER_MODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransceiverMode {
    /// Neither receiving nor transmitting: the RF path is idle.
    Off,
    /// Receiving.
    Receive,
    /// Transmitting.
    Transmit,
}

impl TransceiverMode {
    /// The value of the vendor request.
    pub fn value(self) -> u16 {
        match self {
            TransceiverMode::Off => 0,
            TransceiverMode::Receive => 1,
            TransceiverMode::Transmit => 2,
        }
    }
}

/// Asks a running stream of the driver to end.
#[derive(Debug, Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A new, not cancelled handle.
    pub fn new() -> Cancel {
        Cancel::default()
    }

    /// Asks the stream to end.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether the stream was asked to end.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// What went wrong in the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverErrorKind {
    /// The board was unplugged.
    Disconnected,
    /// No such board.
    NotFound,
    /// Another program holds the board.
    Busy,
    /// The operating system refused access (udev rules, permissions).
    PermissionDenied,
    /// A USB transfer timed out.
    Timeout,
    /// The firmware does not support the request or the value.
    Unsupported,
    /// An argument the driver cannot use (a value out of range, a buffer
    /// smaller than one sample).
    InvalidArgument,
    /// The device is not in a state for the operation (a stream already
    /// runs, the other direction runs, the stream was stopped).
    InvalidState,
    /// Any other failure.
    Other,
}

/// An error of the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverError {
    /// The category.
    pub kind: DriverErrorKind,
    /// A description.
    pub message: String,
}

impl DriverError {
    /// An error of a kind.
    pub fn new(kind: DriverErrorKind, message: impl Into<String>) -> DriverError {
        DriverError {
            kind,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DriverError {}

impl SdrkError for DriverError {
    fn kind(&self) -> ErrorKind {
        match self.kind {
            DriverErrorKind::Disconnected => ErrorKind::Disconnected,
            DriverErrorKind::Timeout => ErrorKind::Timeout,
            DriverErrorKind::Unsupported => ErrorKind::Unsupported,
            DriverErrorKind::InvalidArgument => ErrorKind::InvalidArgument,
            DriverErrorKind::InvalidState => ErrorKind::InvalidState,
            DriverErrorKind::NotFound
            | DriverErrorKind::Busy
            | DriverErrorKind::PermissionDenied
            | DriverErrorKind::Other => ErrorKind::Device,
        }
    }
}

/// Lists and opens boards.
pub trait Driver: Send + Sync {
    /// The boards on the bus now, in bus-scan order.
    ///
    /// # Errors
    /// [`DriverError`] if the bus cannot be scanned.
    fn list(&self) -> Result<Vec<UsbInfo>, DriverError>;

    /// Opens a board (found by its bus path and serial number) and reads its
    /// identity.
    ///
    /// # Errors
    /// [`DriverError`] if the board cannot be opened.
    fn open(&self, info: &UsbInfo) -> Result<Arc<dyn Radio>, DriverError>;
}

/// Where the samples of a transmission come from (see [`Radio::tx_stream`]).
pub trait TxSource {
    /// Fills `buf`, the next bulk transfer, with samples. `in_flight` is the
    /// number of transfers the driver has submitted that have not completed.
    /// Returns the bytes `n` of `buf` that carry samples (the rest is set to
    /// zero, the samples of silence), or `None` if no transfer should be
    /// submitted now. The driver submits the first
    /// [`crate::protocol::tx_transfer_len`]`(n, buf.len())` bytes. May wait a
    /// few milliseconds for samples.
    fn fill(&mut self, buf: &mut [u8], in_flight: usize) -> Option<usize>;

    /// The firmware or the driver saw `count` more TX underruns: the
    /// hardware ran out of samples and sent zeros.
    fn underrun(&mut self, count: u32);
}

/// One open board. Frequencies and rates are whole hertz, gains whole dB.
pub trait Radio: Send + Sync {
    /// The identity read when the board was opened.
    fn board(&self) -> BoardInfo;

    /// Sets the transceiver mode.
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_transceiver_mode(&self, mode: TransceiverMode) -> Result<(), DriverError>;

    /// Tunes (`SET_FREQ`).
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_freq(&self, hz: u64) -> Result<(), DriverError>;

    /// Sets the sample rate `freq_hz / divider` (`SAMPLE_RATE_SET`).
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_sample_rate(&self, freq_hz: u32, divider: u32) -> Result<(), DriverError>;

    /// Sets the baseband filter of the MAX2837
    /// (`BASEBAND_FILTER_BANDWIDTH_SET`); `hz` is one of
    /// [`crate::tuning::BANDWIDTHS_HZ`].
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_baseband_filter(&self, hz: u32) -> Result<(), DriverError>;

    /// Sets the RX LNA gain (0 to 40 dB in steps of 8).
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_lna_gain(&self, db: u32) -> Result<(), DriverError>;

    /// Sets the RX VGA (baseband) gain (0 to 62 dB in steps of 2).
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_vga_gain(&self, db: u32) -> Result<(), DriverError>;

    /// Sets the TX VGA gain (0 to 47 dB in steps of 1).
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_txvga_gain(&self, db: u32) -> Result<(), DriverError>;

    /// Switches the RF amplifier (about 14 dB, RX or TX after the mode).
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_amp(&self, on: bool) -> Result<(), DriverError>;

    /// Switches the antenna port power (bias tee).
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_antenna_power(&self, on: bool) -> Result<(), DriverError>;

    /// Switches the clock output CLKOUT.
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_clkout(&self, on: bool) -> Result<(), DriverError>;

    /// Receives with `buffers` bulk transfers of `buffer_bytes` each in
    /// flight and calls `sink` with every completed transfer, on the calling
    /// thread, until `cancel` is set (`Ok`) or a transfer fails (`Err`). The
    /// transceiver must be in [`TransceiverMode::Receive`].
    ///
    /// # Errors
    /// [`DriverError`], [`DriverErrorKind::Disconnected`] when the board was
    /// unplugged.
    fn rx_stream(
        &self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<(), DriverError>;

    /// Transmits: keeps up to `buffers` bulk transfers of `buffer_bytes`
    /// filled by `source` in flight until `cancel` is set (`Ok`) or a
    /// transfer fails (`Err`), on the calling thread, and reports the TX
    /// underruns it learns of to the source. The transceiver must be in
    /// [`TransceiverMode::Transmit`].
    ///
    /// # Errors
    /// [`DriverError`], [`DriverErrorKind::Disconnected`] when the board was
    /// unplugged.
    fn tx_stream(
        &self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        source: &mut dyn TxSource,
    ) -> Result<(), DriverError>;
}
