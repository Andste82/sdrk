//! The real driver: the vendor requests and bulk transfers of the HackRF
//! firmware over `nusb` (pure Rust; no libhackrf, no libusb).

use crate::driver::{
    BoardInfo, Cancel, Driver, DriverError, DriverErrorKind, Radio, TransceiverMode, TxSource,
    UsbInfo,
};
use crate::protocol::{
    self, API_CLKOUT, API_M0_STATE, PRODUCT_IDS, RX_ENDPOINT, TX_ENDPOINT, VENDOR_ID, request,
};
use nusb::MaybeFuture;
use nusb::transfer::{Bulk, ControlIn, ControlOut, ControlType, In, Out, Recipient, TransferError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Time-out of a control request.
const CONTROL_TIMEOUT: Duration = Duration::from_millis(500);
/// How long a streaming loop waits for a transfer before it looks at the
/// stop request again.
const POLL: Duration = Duration::from_millis(10);
/// How often the TX loop reads the underrun counter of the firmware.
const M0_POLL: Duration = Duration::from_millis(100);

/// The USB driver.
#[derive(Debug, Default, Clone, Copy)]
pub struct UsbDriver;

fn map_usb(e: &nusb::Error) -> DriverError {
    let kind = match e.kind() {
        nusb::ErrorKind::Disconnected => DriverErrorKind::Disconnected,
        nusb::ErrorKind::Busy => DriverErrorKind::Busy,
        nusb::ErrorKind::PermissionDenied => DriverErrorKind::PermissionDenied,
        nusb::ErrorKind::NotFound => DriverErrorKind::NotFound,
        nusb::ErrorKind::Unsupported => DriverErrorKind::Unsupported,
        _ => DriverErrorKind::Other,
    };
    let mut message = e.to_string();
    match kind {
        DriverErrorKind::Busy => message.push_str(" (another program holds the HackRF)"),
        DriverErrorKind::PermissionDenied => message.push_str(
            " (install a udev rule that gives the user access to the HackRF, for example 53-hackrf.rules)",
        ),
        _ => {}
    }
    DriverError::new(kind, message)
}

fn map_transfer(what: &str, e: TransferError) -> DriverError {
    let kind = match e {
        TransferError::Disconnected => DriverErrorKind::Disconnected,
        TransferError::Cancelled => DriverErrorKind::Timeout,
        TransferError::Stall => DriverErrorKind::Unsupported,
        _ => DriverErrorKind::Other,
    };
    DriverError::new(kind, format!("{what}: {e}"))
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn bus_location(d: &nusb::DeviceInfo) -> (String, Vec<u8>) {
    (d.bus_id().to_owned(), d.port_chain().to_vec())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn bus_location(_d: &nusb::DeviceInfo) -> (String, Vec<u8>) {
    ("0".to_owned(), Vec::new())
}

fn is_hackrf(d: &nusb::DeviceInfo) -> bool {
    d.vendor_id() == VENDOR_ID && PRODUCT_IDS.contains(&d.product_id())
}

fn usb_info(d: &nusb::DeviceInfo) -> UsbInfo {
    let (bus, ports) = bus_location(d);
    let text = |s: Option<&str>| s.filter(|s| !s.is_empty()).map(str::to_owned);
    UsbInfo {
        bus,
        ports,
        vendor_id: d.vendor_id(),
        product_id: d.product_id(),
        manufacturer: text(d.manufacturer_string()),
        product: text(d.product_string()),
        serial: text(d.serial_number()),
        device_version: Some(d.device_version()),
    }
}

impl Driver for UsbDriver {
    fn list(&self) -> Result<Vec<UsbInfo>, DriverError> {
        Ok(nusb::list_devices()
            .wait()
            .map_err(|e| map_usb(&e))?
            .filter(is_hackrf)
            .map(|d| usb_info(&d))
            .collect())
    }

    #[allow(
        clippy::large_stack_frames,
        reason = "the futures of nusb are waited on in place; opening runs once per board, on a thread with the default stack"
    )]
    fn open(&self, info: &UsbInfo) -> Result<Arc<dyn Radio>, DriverError> {
        let found = nusb::list_devices()
            .wait()
            .map_err(|e| map_usb(&e))?
            .filter(is_hackrf)
            .find(|d| {
                let u = usb_info(d);
                u.bus_path() == info.bus_path() && u.serial == info.serial
            })
            .ok_or_else(|| DriverError::new(DriverErrorKind::NotFound, "the HackRF is gone"))?;
        let device = found.open().wait().map_err(|e| map_usb(&e))?;
        let active = device
            .active_configuration()
            .map_or(0, |c| c.configuration_value());
        if active != 1 {
            device
                .set_configuration(1)
                .wait()
                .map_err(|e| map_usb(&e))?;
        }
        let iface = device.claim_interface(0).wait().map_err(|e| map_usb(&e))?;
        let mut radio = UsbRadio {
            iface,
            board: BoardInfo {
                board_id: 0xff,
                board_rev: None,
                firmware: String::new(),
                usb_api: found.device_version(),
                serial: None,
            },
        };
        radio.board = radio.read_board()?;
        Ok(Arc::new(radio))
    }
}

/// An open board.
struct UsbRadio {
    iface: nusb::Interface,
    board: BoardInfo,
}

impl UsbRadio {
    fn control_in(
        &self,
        req: u8,
        value: u16,
        index: u16,
        length: u16,
    ) -> Result<Vec<u8>, DriverError> {
        self.iface
            .control_in(
                ControlIn {
                    control_type: ControlType::Vendor,
                    recipient: Recipient::Device,
                    request: req,
                    value,
                    index,
                    length,
                },
                CONTROL_TIMEOUT,
            )
            .wait()
            .map_err(|e| map_transfer(&format!("vendor request {req}"), e))
    }

    fn control_out(&self, req: u8, value: u16, index: u16, data: &[u8]) -> Result<(), DriverError> {
        self.iface
            .control_out(
                ControlOut {
                    control_type: ControlType::Vendor,
                    recipient: Recipient::Device,
                    request: req,
                    value,
                    index,
                    data,
                },
                CONTROL_TIMEOUT,
            )
            .wait()
            .map_err(|e| map_transfer(&format!("vendor request {req}"), e))
    }

    /// A gain request: the firmware answers one byte, zero if it refused
    /// the value.
    fn gain(&self, req: u8, index: u16) -> Result<(), DriverError> {
        let answer = self.control_in(req, 0, index, 1)?;
        if answer.first().copied().unwrap_or(0) == 0 {
            return Err(DriverError::new(
                DriverErrorKind::Unsupported,
                format!("the firmware refused the gain {index} (request {req})"),
            ));
        }
        Ok(())
    }

    fn read_board(&self) -> Result<BoardInfo, DriverError> {
        let usb_api = self.board.usb_api;
        let board_id = self
            .control_in(request::BOARD_ID_READ, 0, 0, 1)?
            .first()
            .copied()
            .unwrap_or(0xff);
        let firmware =
            protocol::parse_version(&self.control_in(request::VERSION_STRING_READ, 0, 0, 255)?);
        let serial = self
            .control_in(request::BOARD_PARTID_SERIALNO_READ, 0, 0, 24)
            .ok()
            .and_then(|a| protocol::parse_serial(&a));
        let board_rev = if usb_api >= API_M0_STATE {
            self.control_in(request::BOARD_REV_READ, 0, 0, 1)
                .ok()
                .and_then(|a| a.first().copied())
        } else {
            None
        };
        Ok(BoardInfo {
            board_id,
            board_rev,
            firmware,
            usb_api,
            serial,
        })
    }

    /// The underrun counter of the M0 core, if the firmware has one.
    fn shortfalls(&self) -> Option<u32> {
        if self.board.usb_api < API_M0_STATE {
            return None;
        }
        self.control_in(request::GET_M0_STATE, 0, 0, 40)
            .ok()
            .and_then(|a| protocol::parse_shortfalls(&a))
    }
}

impl Radio for UsbRadio {
    fn board(&self) -> BoardInfo {
        self.board.clone()
    }

    fn set_transceiver_mode(&self, mode: TransceiverMode) -> Result<(), DriverError> {
        self.control_out(request::SET_TRANSCEIVER_MODE, mode.value(), 0, &[])
    }

    fn set_freq(&self, hz: u64) -> Result<(), DriverError> {
        self.control_out(request::SET_FREQ, 0, 0, &protocol::freq_payload(hz))
    }

    fn set_sample_rate(&self, freq_hz: u32, divider: u32) -> Result<(), DriverError> {
        self.control_out(
            request::SAMPLE_RATE_SET,
            0,
            0,
            &protocol::sample_rate_payload(freq_hz, divider),
        )
    }

    fn set_baseband_filter(&self, hz: u32) -> Result<(), DriverError> {
        let (value, index) = protocol::baseband_filter_value_index(hz);
        self.control_out(request::BASEBAND_FILTER_BANDWIDTH_SET, value, index, &[])
    }

    fn set_lna_gain(&self, db: u32) -> Result<(), DriverError> {
        self.gain(request::SET_LNA_GAIN, protocol::lna_index(db))
    }

    fn set_vga_gain(&self, db: u32) -> Result<(), DriverError> {
        self.gain(request::SET_VGA_GAIN, protocol::vga_index(db))
    }

    fn set_txvga_gain(&self, db: u32) -> Result<(), DriverError> {
        self.gain(request::SET_TXVGA_GAIN, protocol::txvga_index(db))
    }

    fn set_amp(&self, on: bool) -> Result<(), DriverError> {
        self.control_out(request::AMP_ENABLE, u16::from(on), 0, &[])
    }

    fn set_antenna_power(&self, on: bool) -> Result<(), DriverError> {
        self.control_out(request::ANTENNA_ENABLE, u16::from(on), 0, &[])
    }

    fn set_clkout(&self, on: bool) -> Result<(), DriverError> {
        if self.board.usb_api < API_CLKOUT {
            return Err(DriverError::new(
                DriverErrorKind::Unsupported,
                "the firmware has no clock output control (USB API 1.03 or later needed)",
            ));
        }
        self.control_out(request::CLKOUT_ENABLE, u16::from(on), 0, &[])
    }

    fn rx_stream(
        &self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<(), DriverError> {
        let mut ep = self
            .iface
            .endpoint::<Bulk, In>(RX_ENDPOINT)
            .map_err(|e| map_usb(&e))?;
        for _ in 0..buffers {
            let buf = ep.allocate(buffer_bytes);
            ep.submit(buf);
        }
        let mut result = Ok(());
        while ep.pending() > 0 {
            if cancel.is_cancelled() {
                ep.cancel_all();
            }
            let Some(c) = ep.wait_next_complete(POLL) else {
                continue;
            };
            if c.actual_len > 0
                && let Some(data) = c.buffer.get(..c.actual_len)
            {
                sink(data);
            }
            match c.status {
                Ok(()) if !cancel.is_cancelled() => ep.submit(c.buffer),
                Ok(()) | Err(TransferError::Cancelled) => {}
                Err(e) => {
                    if result.is_ok() {
                        result = Err(map_transfer("RX transfer", e));
                    }
                    ep.cancel_all();
                }
            }
        }
        result
    }

    fn tx_stream(
        &self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        source: &mut dyn TxSource,
    ) -> Result<(), DriverError> {
        let mut ep = self
            .iface
            .endpoint::<Bulk, Out>(TX_ENDPOINT)
            .map_err(|e| map_usb(&e))?;
        let mut spare: Vec<nusb::transfer::Buffer> =
            (0..buffers).map(|_| ep.allocate(buffer_bytes)).collect();
        let mut scratch = vec![0u8; buffer_bytes];
        // The underrun counter of the firmware (USB API 1.06 and later);
        // older firmware reports none.
        let mut shortfalls = self.shortfalls();
        let mut polled = Instant::now();
        let mut result = Ok(());
        while !cancel.is_cancelled() {
            while let Some(mut buf) = spare.pop() {
                buf.clear();
                let filled = source.fill(&mut scratch, ep.pending());
                if let Some(n) = filled {
                    // A partial transfer (the end of a burst) is padded only
                    // to the next granule of the firmware.
                    let len = protocol::tx_transfer_len(n, buffer_bytes);
                    buf.extend_from_slice(scratch.get(..len).unwrap_or_default());
                    ep.submit(buf);
                } else {
                    spare.push(buf);
                    break;
                }
                if cancel.is_cancelled() {
                    break;
                }
            }
            if ep.pending() > 0
                && let Some(c) = ep.wait_next_complete(POLL)
            {
                match c.status {
                    Ok(()) => spare.push(c.buffer),
                    Err(TransferError::Cancelled) if cancel.is_cancelled() => {}
                    Err(e) => {
                        result = Err(map_transfer("TX transfer", e));
                        break;
                    }
                }
            }
            if let Some(before) = shortfalls
                && polled.elapsed() >= M0_POLL
            {
                polled = Instant::now();
                if let Some(now) = self.shortfalls() {
                    source.underrun(now.wrapping_sub(before));
                    shortfalls = Some(now);
                }
            }
        }
        ep.cancel_all();
        while ep.pending() > 0 {
            let _ = ep.wait_next_complete(POLL);
        }
        result
    }
}
