//! The interface between [`crate::RtlSdr`] and the USB driver.
//!
//! [`Driver`] lists and opens dongles, [`Radio`] is one open dongle. The real
//! implementation is [`crate::usb::UsbDriver`] (the pure-Rust `rtl-sdr-rs`
//! driver over `nusb`); [`crate::fake::FakeDriver`] stands in for the
//! hardware in tests.

pub use rtl_sdr_rs::CancelHandle as Cancel;
use sdrk_device::{ErrorKind, SdrkError};

/// One RTL-SDR on the USB bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsbInfo {
    /// Position among the RTL-SDR dongles in the order of the bus scan; the
    /// driver opens a dongle by it when the serial number is not unique.
    pub index: usize,
    /// Identifier of the bus or host controller.
    pub bus: String,
    /// The hub ports from the root hub to the dongle.
    pub ports: Vec<u8>,
    /// USB vendor ID.
    pub vendor_id: u16,
    /// USB product ID.
    pub product_id: u16,
    /// USB manufacturer string.
    pub manufacturer: Option<String>,
    /// USB product string.
    pub product: Option<String>,
    /// USB serial number string.
    pub serial: Option<String>,
    /// `bcdDevice` of the device descriptor.
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

/// The tuner of a dongle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TunerKind {
    /// Rafael Micro R820T or R820T2 (also the R828S of the RTL-SDR Blog V4L).
    R820t,
    /// Rafael Micro R828D.
    R828d,
    /// A tuner the driver does not know.
    Unknown,
}

impl TunerKind {
    /// The name used in the capabilities and logs.
    pub fn name(self) -> &'static str {
        match self {
            TunerKind::R820t => "R820T",
            TunerKind::R828d => "R828D",
            TunerKind::Unknown => "unknown",
        }
    }
}

/// Direct sampling: the tuner is bypassed and one ADC branch samples the
/// antenna input directly (HF reception on modified dongles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectSampling {
    /// Normal operation through the tuner.
    Off,
    /// The I branch.
    I,
    /// The Q branch.
    Q,
}

impl DirectSampling {
    /// The value of the setting `direct_sampling`.
    pub fn as_str(self) -> &'static str {
        match self {
            DirectSampling::Off => "off",
            DirectSampling::I => "i",
            DirectSampling::Q => "q",
        }
    }

    /// Parses the value of the setting `direct_sampling`.
    pub fn parse(s: &str) -> Option<DirectSampling> {
        match s {
            "off" => Some(DirectSampling::Off),
            "i" => Some(DirectSampling::I),
            "q" => Some(DirectSampling::Q),
            _ => None,
        }
    }
}

/// What went wrong in the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverErrorKind {
    /// The dongle was unplugged.
    Disconnected,
    /// No such dongle.
    NotFound,
    /// Another program holds the dongle.
    Busy,
    /// The operating system refused access (udev rules, permissions).
    PermissionDenied,
    /// A USB transfer timed out.
    Timeout,
    /// The driver does not support the operation or the value.
    Unsupported,
    /// An argument the driver cannot use (a value out of range, a buffer smaller than one
    /// sample).
    InvalidArgument,
    /// The device is not in a state for the operation (a stream already runs, the radio was
    /// lost).
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

/// Lists and opens dongles.
pub trait Driver: Send + Sync {
    /// The dongles on the bus now, in bus-scan order.
    ///
    /// # Errors
    /// [`DriverError`] if the bus cannot be scanned.
    fn list(&self) -> Result<Vec<UsbInfo>, DriverError>;

    /// Opens a dongle: by serial number if `by_serial` is set, otherwise by
    /// its position [`UsbInfo::index`].
    ///
    /// # Errors
    /// [`DriverError`] if the dongle cannot be opened or initialised.
    fn open(&self, info: &UsbInfo, by_serial: bool) -> Result<Box<dyn Radio>, DriverError>;
}

/// One open dongle. Frequencies and rates are whole hertz; gains are in
/// tenths of a dB as the tuner reports them.
pub trait Radio: Send {
    /// The tuner.
    fn tuner(&self) -> TunerKind;

    /// Whether the dongle is an RTL-SDR Blog V4 or V4L, whose upconverter
    /// receives HF below 28.8 MHz without direct sampling.
    fn has_upconverter(&self) -> bool;

    /// The gains the tuner offers, in tenths of a dB, ascending.
    fn tuner_gains(&self) -> Vec<i32>;

    /// Tunes.
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_center_freq(&mut self, hz: u32) -> Result<(), DriverError>;

    /// Sets the sample rate.
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_sample_rate(&mut self, hz: u32) -> Result<(), DriverError>;

    /// Sets a manual tuner gain in tenths of a dB, or the automatic gain of
    /// the tuner with `None`.
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_tuner_gain(&mut self, tenths: Option<i32>) -> Result<(), DriverError>;

    /// Sets the frequency correction in ppm.
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_ppm(&mut self, ppm: i32) -> Result<(), DriverError>;

    /// Switches the bias tee (power on the antenna input).
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_bias_tee(&mut self, on: bool) -> Result<(), DriverError>;

    /// Sets the direct sampling mode.
    ///
    /// # Errors
    /// [`DriverError`].
    fn set_direct_sampling(&mut self, mode: DirectSampling) -> Result<(), DriverError>;

    /// Clears the sample FIFO of the dongle before streaming.
    ///
    /// # Errors
    /// [`DriverError`].
    fn reset_buffer(&mut self) -> Result<(), DriverError>;

    /// Streams with `buffers` bulk transfers of `buffer_bytes` each in
    /// flight and calls `sink` with every completed transfer, on the calling
    /// thread, until `cancel` is set (`Ok`) or a transfer fails (`Err`).
    ///
    /// # Errors
    /// [`DriverError`], [`DriverErrorKind::Disconnected`] when the dongle was
    /// unplugged.
    fn stream(
        &mut self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<(), DriverError>;
}
