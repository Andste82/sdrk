//! Helpers shared by the device decoders: CRC-8 models and the loop over the rows of a train.

use super::{DecodeResult, Reading, ReadingSink};
use crate::slicers::{Row, Rows};

/// CRC-8 with polynomial 0x31, MSB first, no reflection, no final xor, over `bytes`.
pub(crate) fn crc8_msb(init: u8, bytes: impl IntoIterator<Item = u8>) -> u8 {
    let mut crc = init;
    for b in bytes {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x31
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Byte `i` of a row (bits `8 i .. 8 i + 8`), `None` past the end.
pub(crate) fn byte_at(row: &Row, i: usize) -> Option<u8> {
    row.field(i.checked_mul(8)?, 8).map(|v| v as u8)
}

/// Runs `f` on every row and collects the readings; the result is the best of the rows.
pub(crate) fn collect<F>(rows: &Rows, out: &mut ReadingSink, mut f: F) -> DecodeResult
where
    F: FnMut(&Row) -> Result<Reading, DecodeResult>,
{
    let mut res = DecodeResult::NoMatch;
    for row in rows.iter() {
        match f(row) {
            Ok(r) => {
                if out.push(r) {
                    res = res.best(DecodeResult::Decoded(1));
                }
            }
            Err(e) => res = res.best(e),
        }
    }
    res
}

/// The result for a row that ended with an error (its bits are only a prefix): a train that
/// started like a message is `Truncated`, anything else `NoMatch`.
pub(crate) fn prefix_result(row: &Row, looks_like_start: bool) -> DecodeResult {
    if looks_like_start && !row.is_empty() {
        DecodeResult::Truncated
    } else {
        DecodeResult::NoMatch
    }
}
