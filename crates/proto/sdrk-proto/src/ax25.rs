//! The address header of AX.25 frames (the link layer of APRS).
//!
//! An AX.25 frame (the payload of an HDLC frame, without the FCS) starts with the destination
//! and the source address and up to 8 digipeater addresses, 7 bytes each: six callsign
//! characters (upper-case letters, digits, padded with spaces), each shifted left by one bit,
//! then the SSID byte `CRRSSSSE` (`C`: the command bit of destination and source, or the
//! has-been-repeated bit `H` of a digipeater; `SSSS`: the SSID 0 to 15; `E`: 1 on the last
//! address). The control byte follows; I frames (bit 0 of the control byte 0) and UI frames
//! (control `0x03`, or `0x13` with the poll bit) then carry a PID byte, and the information
//! field starts after it.
//!
//! [`Header::parse`] reads these fields; [`Header`] is displayed in the usual APRS notation
//! `SRC>DST,DIGI1,DIGI2:` (an SSID of 0 is not shown, nor are the `C` and `H` bits).
//!
//! ```
//! use sdrk_proto::ax25::Header;
//!
//! // N0CALL>APRS: UI frame, PID 0xF0, information ">hi".
//! let frame = [
//!     b'A' << 1, b'P' << 1, b'R' << 1, b'S' << 1, b' ' << 1, b' ' << 1, 0x60,
//!     b'N' << 1, b'0' << 1, b'C' << 1, b'A' << 1, b'L' << 1, b'L' << 1, 0x61,
//!     0x03, 0xF0, b'>', b'h', b'i',
//! ];
//! let h = Header::parse(&frame)?;
//! assert_eq!(h.to_string(), "N0CALL>APRS:");
//! assert_eq!(h.pid(), Some(0xF0));
//! assert_eq!(&frame[h.info_offset()..], b">hi");
//! # Ok::<(), sdrk_proto::Error>(())
//! ```

use core::fmt;

use crate::error::{Error, Result};

/// The largest number of digipeater addresses of a header.
pub const MAX_DIGIPEATERS: usize = 8;

/// The bytes of one address field.
const ADDRESS_BYTES: usize = 7;

/// One address of an AX.25 header: callsign, SSID and the `C`/`H` bit.
///
/// ```
/// use sdrk_proto::ax25::Header;
///
/// let frame = [
///     b'A' << 1, b'P' << 1, b'R' << 1, b'S' << 1, b' ' << 1, b' ' << 1, 0x60,
///     b'N' << 1, b'0' << 1, b'C' << 1, b'A' << 1, b'L' << 1, b'L' << 1, 0x6F,
///     0x03, 0xF0,
/// ];
/// let src = *Header::parse(&frame)?.source();
/// assert_eq!(src.callsign(), "N0CALL");
/// assert_eq!(src.ssid(), 7);
/// assert_eq!(src.to_string(), "N0CALL-7");
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Address {
    /// The callsign characters, `len` of them.
    call: [u8; 6],
    len: u8,
    ssid: u8,
    ch: bool,
}

impl Address {
    /// Decodes an address field (7 bytes); returns the address and the extension bit.
    fn decode(field: &[u8]) -> Result<(Self, bool)> {
        let bad = Error::Parse("invalid ax25 address field");
        let (&ssid_byte, chars) = field.split_last().ok_or(bad.clone())?;
        let mut a = Self::default();
        let mut padding = false;
        for (dst, &b) in a.call.iter_mut().zip(chars) {
            let c = b >> 1;
            if b & 1 != 0 {
                return Err(bad);
            }
            match c {
                b' ' => padding = true,
                b'A'..=b'Z' | b'0'..=b'9' if !padding => {
                    *dst = c;
                    a.len += 1;
                }
                _ => return Err(bad),
            }
        }
        if a.len == 0 {
            return Err(bad);
        }
        a.ssid = (ssid_byte >> 1) & 0x0F;
        a.ch = ssid_byte & 0x80 != 0;
        Ok((a, ssid_byte & 1 != 0))
    }

    /// The callsign without padding (1 to 6 upper-case letters and digits).
    pub fn callsign(&self) -> &str {
        let call = self.call.get(..usize::from(self.len)).unwrap_or(&[]);
        core::str::from_utf8(call).unwrap_or("")
    }

    /// The SSID, 0 to 15.
    pub fn ssid(&self) -> u8 {
        self.ssid
    }

    /// The bit 7 of the SSID byte: the command bit `C` of the destination and source, the
    /// has-been-repeated bit `H` of a digipeater.
    pub fn ch_bit(&self) -> bool {
        self.ch
    }
}

impl fmt::Display for Address {
    /// `CALL` or `CALL-SSID` (an SSID of 0 is not shown).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.callsign())?;
        if self.ssid != 0 {
            write!(f, "-{}", self.ssid)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Address({self}, ch: {})", self.ch)
    }
}

/// The address header of an AX.25 frame: destination, source, up to 8 digipeaters, the
/// control byte and the PID.
///
/// ```
/// use sdrk_proto::ax25::Header;
///
/// let frame = [
///     b'A' << 1, b'P' << 1, b'R' << 1, b'S' << 1, b' ' << 1, b' ' << 1, 0x60,
///     b'N' << 1, b'0' << 1, b'C' << 1, b'A' << 1, b'L' << 1, b'L' << 1, 0x60,
///     b'W' << 1, b'I' << 1, b'D' << 1, b'E' << 1, b'1' << 1, b' ' << 1, 0x63,
///     0x03, 0xF0,
/// ];
/// let h = Header::parse(&frame)?;
/// assert_eq!(h.destination().callsign(), "APRS");
/// assert_eq!(h.digipeaters().len(), 1);
/// assert_eq!(h.control(), 0x03);
/// assert_eq!(h.to_string(), "N0CALL>APRS,WIDE1-1:");
/// assert_eq!(h.info_offset(), frame.len());
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Header {
    destination: Address,
    source: Address,
    digipeaters: [Address; MAX_DIGIPEATERS],
    n_digipeaters: u8,
    control: u8,
    pid: Option<u8>,
    info_offset: usize,
}

impl Header {
    /// Reads the header at the start of the AX.25 frame `frame` (without the FCS).
    ///
    /// # Errors
    /// [`Error::Parse`] when an address field is invalid (a character other than upper-case
    /// letters, digits and trailing spaces, a set bit 0 in a callsign byte, an empty
    /// callsign), when the frame ends before the last address, the control byte or the PID,
    /// when the source address is missing (the extension bit is set on the destination) or
    /// when there are more than [`MAX_DIGIPEATERS`] digipeaters.
    pub fn parse(frame: &[u8]) -> Result<Self> {
        let mut fields = frame.chunks_exact(ADDRESS_BYTES);
        let cut = Error::Parse("ax25 header is cut short");
        let (destination, last) = Address::decode(fields.next().ok_or(cut.clone())?)?;
        if last {
            return Err(Error::Parse("ax25 header has no source address"));
        }
        let (source, mut last) = Address::decode(fields.next().ok_or(cut.clone())?)?;
        let mut digipeaters = [Address::default(); MAX_DIGIPEATERS];
        let mut n = 0usize;
        while !last {
            let slot = digipeaters
                .get_mut(n)
                .ok_or(Error::Parse("ax25 header has more than 8 digipeaters"))?;
            let (a, l) = Address::decode(fields.next().ok_or(cut.clone())?)?;
            *slot = a;
            last = l;
            n += 1;
        }
        let control_at = (n + 2) * ADDRESS_BYTES;
        let control = *frame.get(control_at).ok_or(cut.clone())?;
        let has_pid = control & 1 == 0 || control & !0x10 == 0x03;
        let (pid, info_offset) = if has_pid {
            (Some(*frame.get(control_at + 1).ok_or(cut)?), control_at + 2)
        } else {
            (None, control_at + 1)
        };
        Ok(Self {
            destination,
            source,
            digipeaters,
            n_digipeaters: n as u8,
            control,
            pid,
            info_offset,
        })
    }

    /// The destination address (in APRS the software or device identifier, e.g. `APRS`).
    pub fn destination(&self) -> &Address {
        &self.destination
    }

    /// The source address.
    pub fn source(&self) -> &Address {
        &self.source
    }

    /// The digipeater addresses in the order of the frame (0 to [`MAX_DIGIPEATERS`]).
    pub fn digipeaters(&self) -> &[Address] {
        self.digipeaters
            .get(..usize::from(self.n_digipeaters))
            .unwrap_or(&[])
    }

    /// The control byte (`0x03` for the UI frames of APRS).
    pub fn control(&self) -> u8 {
        self.control
    }

    /// The PID byte of I and UI frames (`0xF0`: no layer 3), `None` for other frames.
    pub fn pid(&self) -> Option<u8> {
        self.pid
    }

    /// The offset of the information field in the frame (the length of the header).
    pub fn info_offset(&self) -> usize {
        self.info_offset
    }
}

impl fmt::Display for Header {
    /// `SRC>DST,DIGI1,DIGI2:`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}>{}", self.source, self.destination)?;
        for d in self.digipeaters() {
            write!(f, ",{d}")?;
        }
        f.write_str(":")
    }
}

impl fmt::Debug for Header {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Header")
            .field("destination", &self.destination)
            .field("source", &self.source)
            .field("digipeaters", &self.digipeaters())
            .field("control", &self.control)
            .field("pid", &self.pid)
            .field("info_offset", &self.info_offset)
            .finish_non_exhaustive()
    }
}
