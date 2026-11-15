//! SDRk driver for the HackRF One: half duplex, one channel `rx0` and one
//! channel `tx0` in the native format `cs8`, which share the RF path.
//!
//! Layer `crates/drivers` (std, heap allowed); depends on `sdrk-device` only.
//! The driver speaks the USB protocol of the HackRF firmware (the vendor
//! requests of libhackrf) directly over `nusb`: no libhackrf and no libusb
//! are needed (D1708, D1710). Raw `cs8` samples are passed through
//! unconverted.
//!
//! - [`HackRf`] is one open board. It implements [`sdrk_device::Device`],
//!   [`sdrk_device::Tuner`], [`sdrk_device::RateControl`],
//!   [`sdrk_device::BandwidthControl`] and [`sdrk_device::GainControl`]; the
//!   antenna power (bias tee) and the clock output are methods of its own.
//!   [`HackRf::start_rx`] returns a [`HackRx`] ([`sdrk_device::RxStream`]),
//!   [`HackRf::start_tx`] a [`HackTx`] ([`sdrk_device::TxStream`]).
//! - 1 MHz to 6 GHz; 2 to 20 MS/s, set as `freq_hz / divider` with a divider
//!   up to 31 (the nearest such fraction); the 16 baseband filters of the
//!   MAX2837 (without one the filter follows the rate as in libhackrf); the
//!   gain stages `lna` (0 to 40 dB, step 8), `vga` (0 to 62 dB, step 2) and
//!   `amp` (0 or 14 dB) on `rx0`, `vga` (0 to 47 dB) and `amp` on `tx0`.
//! - Safety: opening, stopping and dropping switch the transmitter off
//!   (transceiver, amplifier, antenna power); the board switches to transmit
//!   only with the first samples written; the amplifier is switched only by
//!   its own gain stage.
//! - Streams: a dedicated thread per running direction keeps the bulk
//!   transfers in flight. RX transfers that do not fit into the queue are
//!   dropped and the next block carries `OVERFLOW` and `GAP_BEFORE`. TX
//!   transfers are sent only when full or at the end of a burst (padded with
//!   zeros), so no zeros come between the samples of a transmission; TX
//!   underruns of the firmware (USB API 1.06 and later) come out of
//!   `poll_status` with `UNDERRUN`, one per underrun.
//! - [`Driver`] and [`Radio`] separate the logic from the USB transport:
//!   [`UsbDriver`] is the real bus, [`fake::FakeDriver`] a simulated one for
//!   tests without hardware. The stream building blocks ([`Reader`],
//!   [`Writer`], [`RxQueue`], [`TxQueue`]) are public for hosts that manage
//!   the RF path themselves.
//!
//! On Linux the user needs access to the board (a udev rule); on Windows the
//! board needs the `WinUSB` driver (installed with Zadig).

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
pub mod protocol;
mod stream;
pub mod tuning;
mod usb;

pub use device::{
    ChannelSettings, DEFAULT_LNA_DB, DEFAULT_RATE_HZ, DEFAULT_RX_CENTER_HZ, DEFAULT_RX_VGA_DB,
    DRIVER, HackRf, StreamOptions, transmitter_off,
};
pub use driver::{
    BoardInfo, Cancel, Driver, DriverError, DriverErrorKind, Radio, TransceiverMode, TxSource,
    UsbInfo,
};
pub use stream::{HackRx, HackTx, Reader, RxQueue, SAMPLE_BYTES, TxQueue, Writer};
pub use usb::UsbDriver;
