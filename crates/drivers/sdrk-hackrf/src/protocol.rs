//! The USB protocol of the HackRF firmware: vendor request numbers and the
//! encoding of their values, as libhackrf sends them. Pure functions, so
//! that the translation can be tested without a board.

/// USB vendor ID of Great Scott Gadgets (`OpenMoko` range).
pub const VENDOR_ID: u16 = 0x1d50;
/// USB product IDs: HackRF One, Jawbreaker, rad1o.
pub const PRODUCT_IDS: [u16; 3] = [0x6089, 0x604b, 0xcc15];

/// The bulk IN endpoint (RX samples).
pub const RX_ENDPOINT: u8 = 0x81;
/// The bulk OUT endpoint (TX samples).
pub const TX_ENDPOINT: u8 = 0x02;

/// The firmware moves TX samples in USB transfers of this size; a bulk OUT
/// transfer must be a multiple of it, or the sample buffer of the firmware
/// gets out of step.
pub const TX_GRANULE: usize = 0x4000;

/// The sample buffer of the firmware between the USB transfers and the
/// radio (two halves of [`TX_GRANULE`]). After the last TX transfer has
/// completed, the board still plays out up to this many bytes.
pub const FIRMWARE_BUFFER: usize = 2 * TX_GRANULE;

/// The length of the bulk OUT transfer that carries `samples_bytes` bytes of
/// samples in a buffer of `buffer_bytes` (a multiple of [`TX_GRANULE`]): the
/// bytes rounded up to the next granule, at least one granule, at most the
/// buffer. The bytes after the samples are zeros.
pub fn tx_transfer_len(samples_bytes: usize, buffer_bytes: usize) -> usize {
    samples_bytes
        .max(1)
        .div_ceil(TX_GRANULE)
        .saturating_mul(TX_GRANULE)
        .min(buffer_bytes)
}

/// Vendor request numbers (libhackrf `hackrf_vendor_request`).
pub mod request {
    /// Off, receive, transmit.
    pub const SET_TRANSCEIVER_MODE: u8 = 1;
    /// Sample rate as `freq_hz / divider`.
    pub const SAMPLE_RATE_SET: u8 = 6;
    /// Baseband filter of the MAX2837.
    pub const BASEBAND_FILTER_BANDWIDTH_SET: u8 = 7;
    /// Board ID, one byte.
    pub const BOARD_ID_READ: u8 = 14;
    /// Firmware version string.
    pub const VERSION_STRING_READ: u8 = 15;
    /// Centre frequency in MHz and Hz.
    pub const SET_FREQ: u8 = 16;
    /// RF amplifier.
    pub const AMP_ENABLE: u8 = 17;
    /// Part ID and serial number, 24 bytes.
    pub const BOARD_PARTID_SERIALNO_READ: u8 = 18;
    /// RX LNA gain.
    pub const SET_LNA_GAIN: u8 = 19;
    /// RX VGA gain.
    pub const SET_VGA_GAIN: u8 = 20;
    /// TX VGA gain.
    pub const SET_TXVGA_GAIN: u8 = 21;
    /// Antenna port power (bias tee).
    pub const ANTENNA_ENABLE: u8 = 23;
    /// Clock output.
    pub const CLKOUT_ENABLE: u8 = 32;
    /// State of the M0 core, 40 bytes (USB API 1.06).
    pub const GET_M0_STATE: u8 = 41;
    /// Board revision, one byte (USB API 1.06).
    pub const BOARD_REV_READ: u8 = 45;
}

/// The first USB API version with `CLKOUT_ENABLE`.
pub const API_CLKOUT: u16 = 0x0103;
/// The first USB API version with `GET_M0_STATE` and `BOARD_REV_READ`.
pub const API_M0_STATE: u16 = 0x0106;

/// The 8 data bytes of `SET_FREQ`: the frequency in whole MHz and the rest in
/// Hz, both little-endian `u32`.
pub fn freq_payload(hz: u64) -> [u8; 8] {
    let mhz = u32::try_from(hz / 1_000_000).unwrap_or(u32::MAX);
    let rest = u32::try_from(hz % 1_000_000).unwrap_or(0);
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&mhz.to_le_bytes());
    out[4..].copy_from_slice(&rest.to_le_bytes());
    out
}

/// The 8 data bytes of `SAMPLE_RATE_SET`: `freq_hz` and `divider`, both
/// little-endian `u32`.
pub fn sample_rate_payload(freq_hz: u32, divider: u32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&freq_hz.to_le_bytes());
    out[4..].copy_from_slice(&divider.to_le_bytes());
    out
}

/// `wValue` and `wIndex` of `BASEBAND_FILTER_BANDWIDTH_SET`: the low and the
/// high 16 bits of the bandwidth.
pub fn baseband_filter_value_index(hz: u32) -> (u16, u16) {
    ((hz & 0xffff) as u16, (hz >> 16) as u16)
}

/// `wIndex` of `SET_LNA_GAIN`: the firmware takes multiples of 8 up to 40.
pub fn lna_index(db: u32) -> u16 {
    (db.min(40) & !0x07) as u16
}

/// `wIndex` of `SET_VGA_GAIN`: even values up to 62.
pub fn vga_index(db: u32) -> u16 {
    (db.min(62) & !0x01) as u16
}

/// `wIndex` of `SET_TXVGA_GAIN`: 0 to 47.
pub fn txvga_index(db: u32) -> u16 {
    db.min(47) as u16
}

/// The serial number from the answer of `BOARD_PARTID_SERIALNO_READ` (two
/// part ID words, then four serial words, little-endian): 32 lower-case
/// hexadecimal digits as `hackrf_info` prints them, `None` if all zero or the
/// answer is short.
pub fn parse_serial(answer: &[u8]) -> Option<String> {
    let words: Vec<u32> = answer
        .get(8..24)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    if words.len() != 4 || words.iter().all(|w| *w == 0) {
        return None;
    }
    Some(words.iter().fold(String::with_capacity(32), |mut out, w| {
        use std::fmt::Write as _;
        // Writing to a String cannot fail.
        let _ = write!(out, "{w:08x}");
        out
    }))
}

/// `num_shortfalls` from the answer of `GET_M0_STATE`: the number of times
/// the M0 core ran out of TX samples (or of RX buffer space).
pub fn parse_shortfalls(answer: &[u8]) -> Option<u32> {
    answer
        .get(16..20)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
}

/// The version string from the answer of `VERSION_STRING_READ`.
pub fn parse_version(answer: &[u8]) -> String {
    let end = answer.iter().position(|b| *b == 0).unwrap_or(answer.len());
    String::from_utf8_lossy(answer.get(..end).unwrap_or_default())
        .trim()
        .to_owned()
}

/// The name of a board ID (libhackrf `hackrf_board_id_name`).
pub fn board_name(board_id: u8) -> &'static str {
    match board_id {
        0 => "Jellybean",
        1 => "Jawbreaker",
        2 | 4 => "HackRF One",
        3 => "rad1o",
        5 => "HackRF Pro",
        _ => "HackRF",
    }
}

/// The name of a board revision (libhackrf `hackrf_board_rev_name`), `None`
/// for "unrecognized" (0xFE) and "undetected" (0xFF). Revisions of boards
/// made by Great Scott Gadgets carry the bit 0x80 and have the same names.
pub fn board_rev_name(rev: u8) -> Option<String> {
    if rev >= 0xfe {
        return None;
    }
    match rev & 0x7f {
        0 => Some("older than r6".to_owned()),
        r @ 1..=5 => Some(format!("r{}", r + 5)),
        r => Some(format!("unknown ({r})")),
    }
}
