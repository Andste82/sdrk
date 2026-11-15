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
            " (another program holds the dongle, or the kernel DVB driver dvb_usb_rtl28xxu could not be detached; blacklist the module)",
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

/// A kernel driver that [`detach_kernel_driver`] took off a dongle: it is
/// bound again when this is dropped, after the dongle was released.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct Reattach(nusb::DeviceInfo);

impl Drop for Reattach {
    fn drop(&mut self) {
        if let Ok(device) = self.0.open().wait() {
            let _ = device.attach_kernel_driver(0);
        }
    }
}

/// Detaches the kernel DVB-T driver (`dvb_usb_rtl28xxu`, or any other kernel
/// driver except `usbfs`, which is another program) from interface 0 of the
/// dongle, as librtlsdr does, so that the dongle can be claimed without
/// blacklisting the module; it needs the same access to the device node as
/// the claim, no root. The returned guard binds the driver again when the
/// dongle is closed. Failures are left to the claim that follows, which
/// reports the busy interface. Linux only; elsewhere nothing happens.
#[cfg(target_os = "linux")]
#[allow(
    clippy::large_stack_frames,
    reason = "the futures of nusb are waited on in place; this runs once per open, on a thread with the default stack"
)]
fn detach_kernel_driver(info: &UsbInfo) -> Option<Reattach> {
    let target = nusb::list_devices()
        .wait()
        .ok()?
        .filter(|d| d.vendor_id() == info.vendor_id && d.product_id() == info.product_id)
        .find(|d| {
            let (bus, ports) = bus_location(d);
            bus == info.bus && ports == info.ports
        })?;
    let (bus, ports) = bus_location(&target);
    let chain = ports
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(".");
    // The driver bound to interface 0 (configuration 1), as sysfs names it.
    let link = format!("/sys/bus/usb/devices/{bus}-{chain}:1.0/driver");
    let driver = std::fs::read_link(&link).ok()?;
    if driver.file_name().is_none_or(|n| n == "usbfs") {
        return None;
    }
    // The detach lasts after this handle is dropped.
    target.open().wait().ok()?.detach_kernel_driver(0).ok()?;
    Some(Reattach(target))
}

#[cfg(not(target_os = "linux"))]
fn detach_kernel_driver(_info: &UsbInfo) -> Option<Reattach> {
    None
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
        let reattach = detach_kernel_driver(info);
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
            _reattach: reattach,
        }))
    }
}

/// An open dongle.
struct UsbRadio {
    // Dropped first: the dongle is released before the kernel driver is
    // bound again.
    sdr: RtlSdr,
    tuner: TunerKind,
    upconverter: bool,
    _reattach: Option<Reattach>,
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
