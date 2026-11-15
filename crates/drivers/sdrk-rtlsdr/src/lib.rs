//! SDRk driver for RTL-SDR dongles: an RTL2832U with an R820T, R820T2 or
//! R828D tuner, receive only, one channel `rx0` in the native format `cu8`.
//!
//! Layer `crates/drivers` (std, heap allowed); depends on `sdrk-device` only.
//! The USB driver is `rtl-sdr-rs`, a pure-Rust port of librtlsdr over `nusb`:
//! no C library and no libusb are needed (D1708, D1709).
//!
//! - [`RtlSdr`] is one open dongle. It implements [`sdrk_device::Device`],
//!   [`sdrk_device::Tuner`], [`sdrk_device::RateControl`] and
//!   [`sdrk_device::GainControl`]; the bias tee, the frequency correction and
//!   direct sampling are methods of its own. [`RtlSdr::start_rx`] returns an
//!   [`RtlRx`], an [`sdrk_device::RxStream`].
//! - Tuning range from the tuner: 24 to 1766 MHz; from 500 kHz with the
//!   upconverter of the RTL-SDR Blog V4; 0 to 28.8 MHz with direct sampling.
//!   Rates 225001 to 300000 and 900001 to 3200000 S/s; the hardware runs at
//!   the exact fraction of its resampler ([`tuner::exact_rate`]). One gain
//!   stage `tuner`; a value is set to the nearest entry of the gain list.
//! - Reading: a dedicated thread per running dongle keeps the bulk transfers
//!   in flight and fills a bounded queue; transfers that do not fit are
//!   dropped and the next block carries `OVERFLOW` and `GAP_BEFORE` with the
//!   index advanced by the lost samples. An unplugged dongle makes the stream
//!   fail with `Disconnected`.
//! - [`Driver`] and [`Radio`] separate the logic from the USB transport:
//!   [`UsbDriver`] is the real bus, [`fake::FakeDriver`] a simulated one for
//!   tests without hardware.
//!
//! On Linux the user needs access to the dongle (a udev rule) and the kernel
//! DVB driver `dvb_usb_rtl28xxu` must not hold it (blacklist the module); on
//! Windows the dongle needs the `WinUSB` driver (installed with Zadig).

#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]

mod device;
pub mod driver;
pub mod fake;
mod stream;
pub mod tuner;
mod usb;

pub use device::{
    DEFAULT_CENTER_HZ, DEFAULT_GAIN_DB, DEFAULT_RATE_HZ, DRIVER, RtlSdr, StreamOptions,
};
pub use driver::{
    Cancel, DirectSampling, Driver, DriverError, DriverErrorKind, Radio, TunerKind, UsbInfo,
};
pub use stream::RtlRx;
pub use usb::UsbDriver;
