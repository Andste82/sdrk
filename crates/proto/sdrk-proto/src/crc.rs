//! Table-driven CRCs.
//!
//! Each function computes the CRC of a whole message in one call, one table lookup per byte.
//! The tables (256 entries each) are computed at compile time by `const fn`s. Parameters as in
//! the CRC catalogue (width, polynomial, init, reflected input and output, final xor):
//!
//! | Function | Catalogue name | Width | Poly | Init | Reflected | Xor out | Check (`"123456789"`) |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | [`crc16_ccitt_false`] | CRC-16/IBM-3740 (CCITT-FALSE) | 16 | `0x1021` | `0xFFFF` | no | `0` | `0x29B1` |
//! | [`crc16_x25`] | CRC-16/IBM-SDLC (X-25) | 16 | `0x1021` | `0xFFFF` | yes | `0xFFFF` | `0x906E` |
//! | [`crc32`] | CRC-32/ISO-HDLC | 32 | `0x04C11DB7` | `0xFFFFFFFF` | yes | `0xFFFFFFFF` | `0xCBF43926` |
//! | [`crc24_adsb`] | Mode S parity (ADS-B) | 24 | `0xFFF409` | `0` | no | `0` | residue `0` over a frame |

/// The MSB-first table of a 16-bit polynomial.
const fn table_msb16(poly: u16) -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = (i as u16) << 8;
        let mut k = 0;
        while k < 8 {
            c = if c & 0x8000 != 0 {
                (c << 1) ^ poly
            } else {
                c << 1
            };
            k += 1;
        }
        // Evaluated at compile time only: an out-of-range index would fail the build.
        #[allow(clippy::indexing_slicing)]
        {
            t[i] = c;
        }
        i += 1;
    }
    t
}

/// The LSB-first (reflected) table of a 16-bit polynomial given in reflected form.
const fn table_lsb16(poly_reflected: u16) -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u16;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ poly_reflected
            } else {
                c >> 1
            };
            k += 1;
        }
        #[allow(clippy::indexing_slicing)]
        {
            t[i] = c;
        }
        i += 1;
    }
    t
}

/// The LSB-first (reflected) table of a 32-bit polynomial given in reflected form.
const fn table_lsb32(poly_reflected: u32) -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ poly_reflected
            } else {
                c >> 1
            };
            k += 1;
        }
        #[allow(clippy::indexing_slicing)]
        {
            t[i] = c;
        }
        i += 1;
    }
    t
}

/// The MSB-first table of a 24-bit polynomial (held in the low 24 bits).
const fn table_msb24(poly: u32) -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = (i as u32) << 16;
        let mut k = 0;
        while k < 8 {
            c = if c & 0x0080_0000 != 0 {
                ((c << 1) ^ poly) & 0x00FF_FFFF
            } else {
                (c << 1) & 0x00FF_FFFF
            };
            k += 1;
        }
        #[allow(clippy::indexing_slicing)]
        {
            t[i] = c;
        }
        i += 1;
    }
    t
}

static CCITT_MSB: [u16; 256] = table_msb16(0x1021);
static CCITT_LSB: [u16; 256] = table_lsb16(0x8408);
static CRC32_LSB: [u32; 256] = table_lsb32(0xEDB8_8320);
static ADSB_MSB: [u32; 256] = table_msb24(0x00FF_F409);

/// The table entry of byte `i` (a `u8` index is always in range of a 256-entry table).
#[inline]
fn entry<T: Copy + Default>(table: &[T; 256], i: u8) -> T {
    table.get(usize::from(i)).copied().unwrap_or_default()
}

/// CRC-16/CCITT-FALSE (CRC-16/IBM-3740): polynomial `0x1021`, init `0xFFFF`, MSB first, no
/// final xor. The checksum of the framing of [`crate::frame`].
///
/// ```
/// use sdrk_proto::crc::crc16_ccitt_false;
/// assert_eq!(crc16_ccitt_false(b"123456789"), 0x29B1);
/// ```
pub fn crc16_ccitt_false(d: &[u8]) -> u16 {
    d.iter().fold(0xFFFF_u16, |crc, &b| {
        (crc << 8) ^ entry(&CCITT_MSB, (crc >> 8) as u8 ^ b)
    })
}

/// CRC-16/X-25 (CRC-16/IBM-SDLC, the HDLC and AX.25 frame check sequence): polynomial
/// `0x1021` reflected, init `0xFFFF`, final xor `0xFFFF`. Transmitted low byte first.
///
/// ```
/// use sdrk_proto::crc::crc16_x25;
/// assert_eq!(crc16_x25(b"123456789"), 0x906E);
/// ```
pub fn crc16_x25(d: &[u8]) -> u16 {
    // The register is kept in a `u32` (its upper half stays zero: the shift brings in zeros and
    // the entries have 16 bits), so the loop has no 16-bit operations in its dependency chain.
    let crc = d.iter().fold(0xFFFF_u32, |crc, &b| {
        (crc >> 8) ^ u32::from(entry(&CCITT_LSB, crc as u8 ^ b))
    });
    !(crc as u16)
}

/// CRC-32/ISO-HDLC (the CRC of Ethernet, zlib and PNG): polynomial `0x04C11DB7` reflected,
/// init and final xor `0xFFFFFFFF`.
///
/// ```
/// use sdrk_proto::crc::crc32;
/// assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
/// ```
pub fn crc32(d: &[u8]) -> u32 {
    !d.iter().fold(0xFFFF_FFFF_u32, |crc, &b| {
        (crc >> 8) ^ entry(&CRC32_LSB, crc as u8 ^ b)
    })
}

/// CRC-24 of Mode S and ADS-B: polynomial `0xFFF409`, init 0, MSB first, no final xor; the
/// result is in the low 24 bits. Over a whole Mode S frame (message and parity) the result is
/// 0 when the frame has no error.
///
/// ```
/// use sdrk_proto::crc::crc24_adsb;
/// let frame = [0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98];
/// assert_eq!(crc24_adsb(&frame), 0);
/// assert_eq!(crc24_adsb(&frame[..11]), 0x57_6098);
/// ```
pub fn crc24_adsb(d: &[u8]) -> u32 {
    d.iter().fold(0_u32, |crc, &b| {
        ((crc << 8) & 0x00FF_FFFF) ^ entry(&ADSB_MSB, (crc >> 16) as u8 ^ b)
    })
}
