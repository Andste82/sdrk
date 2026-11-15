//! The real driver: `rtl-sdr-rs`, a pure-Rust port of librtlsdr over
//! `nusb` (no C library, no libusb). The bus path for IDs comes from `nusb`.

use crate::driver::{
    Cancel, DirectSampling, Driver, DriverError, DriverErrorKind, Radio, TunerKind, UsbInfo,
};
use nusb::MaybeFuture;
use rtl_sdr_rs::error::{RtlsdrError, UsbErrorKind};
use rtl_sdr_rs::{BlogModel, DirectSampleMode, RtlSdr, TunerGain, TunerId};
use std::collections::BTreeSet;

/// The USB driver.
#[derive(Debug, Default, Clone, Copy)]
pub struct UsbDriver;

fn map_error(e: &RtlsdrError) -> DriverError {
    let kind = match e {
        RtlsdrError::Usb(u) => match u.kind() {
            UsbErrorKind::Disconnected => DriverErrorKind::Disconnected,
            UsbErrorKind::NotFound => DriverErrorKind::NotFound,
            UsbErrorKind::Busy => DriverErrorKind::Busy,
            UsbErrorKind::PermissionDenied => DriverErrorKind::PermissionDenied,
            UsbErrorKind::Timeout => DriverErrorKind::Timeout,
            UsbErrorKind::Unsupported => DriverErrorKind::Unsupported,
            _ => DriverErrorKind::Other,
        },
        _ => DriverErrorKind::Other,
    };
    let mut message = e.to_string();
    match kind {
        DriverErrorKind::Busy => message.push_str(
            " (another program or the kernel DVB driver dvb_usb_rtl28xxu holds the dongle; blacklist the module)",
        ),
        DriverErrorKind::PermissionDenied => {
            message.push_str(" (install a udev rule that gives the user access to the dongle)");
        }
        _ => {}
    }
    DriverError::new(kind, message)
}

fn non_empty(s: &str) -> Option<String> {
    (!s.is_empty()).then(|| s.to_owned())
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn bus_location(d: &nusb::DeviceInfo) -> (String, Vec<u8>) {
    (d.bus_id().to_owned(), d.port_chain().to_vec())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn bus_location(_d: &nusb::DeviceInfo) -> (String, Vec<u8>) {
    ("0".to_owned(), Vec::new())
}

impl Driver for UsbDriver {
    fn list(&self) -> Result<Vec<UsbInfo>, DriverError> {
        let rtl = RtlSdr::list_devices().map_err(|e| map_error(&e))?;
        let known: BTreeSet<(u16, u16)> = rtl.iter().map(|d| (d.vendor_id, d.product_id)).collect();
        // The same bus scan order as the driver: the n-th dongle of the
        // driver list is the n-th device with a known vendor and product ID.
        let usb: Vec<nusb::DeviceInfo> = nusb::list_devices()
            .wait()
            .map_err(|e| DriverError::new(DriverErrorKind::Other, e.to_string()))?
            .filter(|d| known.contains(&(d.vendor_id(), d.product_id())))
            .collect();
        Ok(rtl
            .iter()
            .enumerate()
            .map(|(n, d)| {
                let bus = usb.get(n).filter(|u| {
                    u.vendor_id() == d.vendor_id
                        && u.product_id() == d.product_id
                        && u.serial_number().unwrap_or_default() == d.serial
                });
                let (bus_id, ports) = bus.map_or_else(
                    || ("0".to_owned(), vec![u8::try_from(n).unwrap_or(u8::MAX)]),
                    bus_location,
                );
                UsbInfo {
                    index: d.index,
                    bus: bus_id,
                    ports,
                    vendor_id: d.vendor_id,
                    product_id: d.product_id,
                    manufacturer: non_empty(&d.manufacturer),
                    product: non_empty(&d.product),
                    serial: non_empty(&d.serial),
                    device_version: bus.map(nusb::DeviceInfo::device_version),
                }
            })
            .collect())
    }

    fn open(&self, info: &UsbInfo, by_serial: bool) -> Result<Box<dyn Radio>, DriverError> {
        let serial = info.serial.clone().filter(|_| by_serial);
        let index = info.index;
        // The driver panics when it finds no tuner it supports; that must not
        // take the server down.
        let opened = std::panic::catch_unwind(move || match &serial {
            Some(s) => RtlSdr::open_with_serial(s),
            None => RtlSdr::open_with_index(index),
        });
        let sdr = match opened {
            Ok(Ok(sdr)) => sdr,
            Ok(Err(e)) => return Err(map_error(&e)),
            Err(_) => {
                return Err(DriverError::new(
                    DriverErrorKind::Unsupported,
                    "the dongle has a tuner that the driver does not support (only R820T, R820T2, R828D)",
                ));
            }
        };
        let tuner = match sdr.get_tuner_id() {
            Ok(id) if id == TunerId::R820T => TunerKind::R820t,
            Ok(id) if id == TunerId::R828D => TunerKind::R828d,
            _ => TunerKind::Unknown,
        };
        let upconverter = sdr.blog_model() != BlogModel::None;
        Ok(Box::new(UsbRadio {
            sdr,
            tuner,
            upconverter,
        }))
    }
}

/// An open dongle.
struct UsbRadio {
    sdr: RtlSdr,
    tuner: TunerKind,
    upconverter: bool,
}

impl Radio for UsbRadio {
    fn tuner(&self) -> TunerKind {
        self.tuner
    }

    fn has_upconverter(&self) -> bool {
        self.upconverter
    }

    fn tuner_gains(&self) -> Vec<i32> {
        let mut g = self.sdr.get_tuner_gains().unwrap_or_default();
        g.sort_unstable();
        g
    }

    fn set_center_freq(&mut self, hz: u32) -> Result<(), DriverError> {
        self.sdr.set_center_freq(hz).map_err(|e| map_error(&e))
    }

    fn set_sample_rate(&mut self, hz: u32) -> Result<(), DriverError> {
        self.sdr.set_sample_rate(hz).map_err(|e| map_error(&e))
    }

    fn set_tuner_gain(&mut self, tenths: Option<i32>) -> Result<(), DriverError> {
        let gain = tenths.map_or(TunerGain::Auto, TunerGain::Manual);
        self.sdr.set_tuner_gain(gain).map_err(|e| map_error(&e))
    }

    fn set_ppm(&mut self, ppm: i32) -> Result<(), DriverError> {
        self.sdr.set_freq_correction(ppm).map_err(|e| map_error(&e))
    }

    fn set_bias_tee(&mut self, on: bool) -> Result<(), DriverError> {
        self.sdr.set_bias_tee(on).map_err(|e| map_error(&e))
    }

    fn set_direct_sampling(&mut self, mode: DirectSampling) -> Result<(), DriverError> {
        let mode = match mode {
            DirectSampling::Off => DirectSampleMode::Off,
            DirectSampling::I => DirectSampleMode::On,
            DirectSampling::Q => DirectSampleMode::OnSwap,
        };
        self.sdr
            .set_direct_sampling(mode)
            .map_err(|e| map_error(&e))
    }

    fn reset_buffer(&mut self) -> Result<(), DriverError> {
        self.sdr.reset_buffer().map_err(|e| map_error(&e))
    }

    fn stream(
        &mut self,
        buffers: usize,
        buffer_bytes: usize,
        cancel: &Cancel,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<(), DriverError> {
        self.sdr
            .read_async(buffers, buffer_bytes, cancel, |data| sink(data))
            .map_err(|e| map_error(&e))
    }
}
