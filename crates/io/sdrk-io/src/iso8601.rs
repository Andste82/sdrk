//! ISO 8601 UTC timestamps (`YYYY-MM-DDTHH:MM:SS[.fraction]Z`) without a date crate.

use crate::error::{Error, Result};

const NS_PER_S: i128 = 1_000_000_000;
const FILE: &str = "core:datetime";

/// Days from 1970-01-01 to the civil date `y-m-d` (proleptic Gregorian calendar).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date `(year, month, day)` of a day count since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(y) => 29,
        _ => 28,
    }
}

fn bad(reason: &str) -> Error {
    Error::parse(FILE, reason)
}

/// Parses exactly `len` ASCII digits starting at `at`.
fn digits(b: &[u8], at: usize, len: usize) -> Result<i64> {
    let part = b
        .get(at..at + len)
        .ok_or_else(|| bad("timestamp too short"))?;
    let mut v = 0i64;
    for &c in part {
        if !c.is_ascii_digit() {
            return Err(bad("expected a digit"));
        }
        v = v * 10 + i64::from(c - b'0');
    }
    Ok(v)
}

fn expect(b: &[u8], at: usize, c: u8) -> Result<()> {
    if b.get(at) == Some(&c) {
        Ok(())
    } else {
        Err(bad("unexpected character in the timestamp"))
    }
}

/// Parses `YYYY-MM-DDTHH:MM:SS[.fraction]Z` into nanoseconds since 1970-01-01T00:00:00Z.
///
/// The fraction has one or more digits; digits beyond the ninth are truncated. The year must
/// be 0001 to 9999 and the result must fit an `i64` (years 1677 to 2262).
///
/// ```
/// use sdrk_io::parse_iso8601_ns;
/// assert_eq!(parse_iso8601_ns("1970-01-01T00:00:00Z")?, 0);
/// assert_eq!(parse_iso8601_ns("2000-02-29T12:00:00.5Z")?, 951_825_600_500_000_000);
/// assert!(parse_iso8601_ns("2001-02-29T00:00:00Z").is_err());
/// # Ok::<(), sdrk_io::Error>(())
/// ```
///
/// # Errors
/// `Parse` for any deviation from the format, an impossible date or time, or a value that
/// does not fit.
#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
pub fn parse_iso8601_ns(s: &str) -> Result<i64> {
    let b = s.as_bytes();
    let year = digits(b, 0, 4)?;
    expect(b, 4, b'-')?;
    let month = digits(b, 5, 2)?;
    expect(b, 7, b'-')?;
    let day = digits(b, 8, 2)?;
    expect(b, 10, b'T')?;
    let hour = digits(b, 11, 2)?;
    expect(b, 13, b':')?;
    let minute = digits(b, 14, 2)?;
    expect(b, 16, b':')?;
    let second = digits(b, 17, 2)?;
    let mut at = 19;
    let mut nanos = 0i64;
    if b.get(at) == Some(&b'.') {
        at += 1;
        let start = at;
        while b.get(at).is_some_and(u8::is_ascii_digit) {
            at += 1;
        }
        if at == start {
            return Err(bad("empty fraction"));
        }
        let mut scale = 100_000_000i64;
        for &c in b.get(start..at.min(start + 9)).unwrap_or(&[]) {
            nanos += i64::from(c - b'0') * scale;
            scale /= 10;
        }
    }
    expect(b, at, b'Z')?;
    if b.len() != at + 1 {
        return Err(bad("trailing characters after the timestamp"));
    }
    if year < 1 || !(1..=12).contains(&month) {
        return Err(bad("year or month out of range"));
    }
    if day < 1 || day > days_in_month(year, month) {
        return Err(bad("day out of range"));
    }
    if hour > 23 || minute > 59 || second > 59 {
        return Err(bad("time of day out of range"));
    }
    let secs = days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second;
    let total = i128::from(secs) * NS_PER_S + i128::from(nanos);
    i64::try_from(total).map_err(|_| bad("the timestamp does not fit 64-bit nanoseconds"))
}

/// Formats nanoseconds since the epoch as `YYYY-MM-DDTHH:MM:SS[.fraction]Z`; the fraction is
/// omitted when zero and otherwise written without trailing zeros.
///
/// ```
/// use sdrk_io::format_iso8601_ns;
/// assert_eq!(format_iso8601_ns(0), "1970-01-01T00:00:00Z");
/// assert_eq!(format_iso8601_ns(1_500_000_000), "1970-01-01T00:00:01.5Z");
/// ```
pub fn format_iso8601_ns(unix_ns: i64) -> String {
    let ns = i128::from(unix_ns);
    let secs = ns.div_euclid(NS_PER_S);
    let frac = ns.rem_euclid(NS_PER_S);
    let days = i64::try_from(secs.div_euclid(86_400)).unwrap_or(0);
    let tod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let mut out = format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}",
        tod / 3600,
        tod % 3600 / 60,
        tod % 60
    );
    if frac != 0 {
        let digits = format!("{frac:09}");
        out.push('.');
        out.push_str(digits.trim_end_matches('0'));
    }
    out.push('Z');
    out
}
