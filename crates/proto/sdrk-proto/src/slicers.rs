//! Slicers (M46 §1): pulse trains of `band_pulse_detect` cut into rows of bits.
//!
//! A slicer has a configuration in microseconds and a [`Tolerance`] (default 20 percent of the
//! nominal width, at least 2 samples). It reads a [`PulseTrain`] and writes [`Rows`]: a gap
//! longer than `reset_us` ends a row (the repeats of a message appear as rows), and a width
//! outside the tolerance ends the row with an error ([`RowEnd::BadWidth`], [`RowEnd::Violation`]
//! or [`RowEnd::Full`]). An error row keeps the bits decoded before the error; the rest of the
//! burst up to the next reset gap is skipped. The last gap of a train is the end of the train
//! whatever its width (`band_pulse_detect` writes 0 there).
//!
//! Every row reports its number of bits, the measured average of its unit in microseconds and
//! the worst relative timing error, so a decoder can reject bad rows.
//!
//! | Slicer | Bit | Unit | Tolerance of a width |
//! | --- | --- | --- | --- |
//! | [`Pwm`] | the pulse: near `short_us` or `long_us` | the short pulse (a long one is scaled by `short_us / long_us`) | relative to the nominal width; the gap must be near `gap_us` |
//! | [`Ppm`] | the gap after the pulse: near `short_gap_us` or `long_gap_us` | the short gap | relative to the nominal gap; the pulse width is not checked; the last pulse of a row is an end marker and carries no bit |
//! | [`Manchester`] | two half bits, see below | the half bit | a run is 1 or 2 half bits, relative to the nominal run |
//! | [`Nrz`] | a pulse is `round(w / bit_us)` ones, a gap as many zeros | the bit | absolute, a fraction of one bit, so long runs keep their count |
//!
//! Manchester: pulses are the high level and gaps the low one, the half bits of a row start at
//! a bit boundary. [`Convention::Ieee8023`] reads a high-then-low pair as 1 and a low-then-high
//! pair as 0, [`Convention::Thomas`] the other way round. A pair of equal levels is a violation.
//! At the end of a row a lone high half bit is completed by a low one (the idle level). A
//! message whose first bit is 0 in the IEEE convention starts mid-bit, because the transmitter
//! is silent in the first half bit: [`Manchester::with_virtual_lead`] adds that low half bit.
//!
//! ```
//! use sdrk_proto::pulses::{self, Header, Modulation, Pair};
//! use sdrk_proto::slicers::{Pwm, Rows, Slicer};
//! use sdrk_proto::PacketFlags;
//!
//! // Bits 1, 0, 1: a short pulse is a 1, 500 us and 1500 us pulses, 1000 us gaps, 250 kHz.
//! let header = Header {
//!     modulation: Modulation::Ook,
//!     channel: 0,
//!     freq_offset_hz: 0,
//!     level_db10: -300,
//!     noise_db10: -600,
//!     rate_hz: 250_000,
//! };
//! let pairs = [
//!     Pair { pulse: 125, gap: 250 },
//!     Pair { pulse: 375, gap: 250 },
//!     Pair { pulse: 125, gap: 0 },
//! ];
//! let packet = pulses::encode_packet(0, PacketFlags::NONE, &header, &pairs)?;
//! let train = pulses::decode(packet.data())?;
//!
//! let pwm = Pwm::new(500.0, 1500.0, 1000.0, 4000.0, true)?;
//! let mut rows = Rows::new();
//! pwm.slice(&train, &mut rows);
//! let row = rows.get(0).ok_or("no row")?;
//! assert_eq!(row.len(), 3);
//! assert_eq!(row.bytes(), &[0b1010_0000]);
//! assert!(!row.end().is_error());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![allow(clippy::many_single_char_names)]

use crate::error::{Error, Result};
use crate::pulses::PulseTrain;

/// Bytes of the bit storage of one [`Row`].
pub const ROW_BYTES: usize = 32;

/// The most bits one [`Row`] holds; a longer row ends with [`RowEnd::Full`].
pub const MAX_ROW_BITS: usize = ROW_BYTES * 8;

/// The most rows one [`Rows`] holds; later rows are dropped and counted.
pub const MAX_ROWS: usize = 8;

fn absf(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// The timing tolerance of a slicer: `rel` of the nominal width, at least `min_samples`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    rel: f32,
    min_samples: f32,
}

impl Tolerance {
    /// 20 percent, at least 2 samples.
    pub const DEFAULT: Self = Self {
        rel: 0.2,
        min_samples: 2.0,
    };

    /// A tolerance of `rel` (a fraction, above 0 and below 0.4) of the nominal width, at least
    /// `min_samples` samples.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a `rel` outside the range (or NaN).
    ///
    /// ```
    /// use sdrk_proto::slicers::Tolerance;
    /// assert!(Tolerance::new(0.1, 3).is_ok());
    /// assert!(Tolerance::new(0.5, 3).is_err());
    /// ```
    pub fn new(rel: f32, min_samples: u32) -> Result<Self> {
        if !(rel > 0.0 && rel < 0.4) {
            return Err(Error::InvalidArgument("tolerance must be in (0, 0.4)"));
        }
        Ok(Self {
            rel,
            min_samples: min_samples as f32,
        })
    }

    /// The relative part.
    pub fn rel(&self) -> f32 {
        self.rel
    }

    /// The minimum in samples.
    pub fn min_samples(&self) -> f32 {
        self.min_samples
    }

    fn limit(self, nominal: f32) -> f32 {
        (self.rel * nominal).max(self.min_samples)
    }
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// How a [`Row`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RowEnd {
    /// A gap longer than `reset_us` followed.
    Reset,
    /// The train ended.
    TrainEnd,
    /// A width outside the tolerance (error).
    BadWidth,
    /// Two equal levels in one Manchester bit (error).
    Violation,
    /// More than [`MAX_ROW_BITS`] bits (error).
    Full,
}

impl RowEnd {
    /// Whether the row ended with an error: its bits are a prefix of the message only.
    pub const fn is_error(self) -> bool {
        matches!(self, Self::BadWidth | Self::Violation | Self::Full)
    }
}

/// One row of bits: the bits MSB first, the measured unit and the worst timing error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    bytes: [u8; ROW_BYTES],
    bits: u16,
    start_pair: u32,
    unit_avg_us: f32,
    worst_err: f32,
    end: RowEnd,
}

impl Row {
    const EMPTY: Self = Self {
        bytes: [0; ROW_BYTES],
        bits: 0,
        start_pair: 0,
        unit_avg_us: 0.0,
        worst_err: 0.0,
        end: RowEnd::Reset,
    };

    /// The number of bits.
    pub fn len(&self) -> usize {
        usize::from(self.bits)
    }

    /// Whether the row has no bit.
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Bit `i` (0 the first one), `None` past the end.
    pub fn bit(&self, i: usize) -> Option<bool> {
        if i >= self.len() {
            return None;
        }
        let byte = self.bytes.get(i / 8)?;
        Some((byte >> (7 - (i % 8))) & 1 == 1)
    }

    /// The bits packed MSB first, the last byte padded with zeros.
    pub fn bytes(&self) -> &[u8] {
        self.bytes.get(..self.len().div_ceil(8)).unwrap_or(&[])
    }

    /// The bits as an iterator.
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len()).filter_map(|i| self.bit(i))
    }

    /// `n` bits (at most 32) from bit `start` on as an integer, the first bit the most
    /// significant; `None` when the row is shorter than `start + n` or `n > 32`.
    ///
    /// ```
    /// use sdrk_proto::slicers::{Nrz, Rows, Slicer};
    /// use sdrk_proto::pulses::{self, Header, Modulation, Pair};
    /// use sdrk_proto::PacketFlags;
    /// let h = Header { modulation: Modulation::Fsk, channel: 0, freq_offset_hz: 0,
    ///     level_db10: 0, noise_db10: 0, rate_hz: 100_000 };
    /// // 1 1 0 1 at 100 us per bit: pulse 2 bits, gap 1 bit, pulse 1 bit
    /// let pairs = [Pair { pulse: 20, gap: 10 }, Pair { pulse: 10, gap: 0 }];
    /// let p = pulses::encode_packet(0, PacketFlags::NONE, &h, &pairs)?;
    /// let t = pulses::decode(p.data())?;
    /// let mut rows = Rows::new();
    /// Nrz::new(100.0, 1000.0)?.slice(&t, &mut rows);
    /// let row = rows.get(0).ok_or("no row")?;
    /// assert_eq!(row.field(0, 4), Some(0b1101));
    /// assert_eq!(row.field(1, 4), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn field(&self, start: usize, n: usize) -> Option<u32> {
        if n > 32 || start.checked_add(n)? > self.len() {
            return None;
        }
        let mut v = 0u32;
        for i in start..start + n {
            v = (v << 1) | u32::from(self.bit(i)?);
        }
        Some(v)
    }

    /// The index of the pair of the train where the row starts.
    pub fn start_pair(&self) -> usize {
        self.start_pair as usize
    }

    /// The measured average of the unit of the slicer in microseconds (see the table of the
    /// [module](self)); 0 for a row without a measurement.
    pub fn unit_avg_us(&self) -> f32 {
        self.unit_avg_us
    }

    /// The worst relative timing error of the row: the largest `|w - nominal| / nominal` of a
    /// checked width (for [`Nrz`] relative to one bit).
    pub fn worst_err(&self) -> f32 {
        self.worst_err
    }

    /// How the row ended.
    pub fn end(&self) -> RowEnd {
        self.end
    }
}

/// The rows of one train in fixed storage (at most [`MAX_ROWS`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    list: [Row; MAX_ROWS],
    len: usize,
    dropped: u32,
}

impl Default for Rows {
    fn default() -> Self {
        Self::new()
    }
}

impl Rows {
    /// An empty set.
    pub const fn new() -> Self {
        Self {
            list: [Row::EMPTY; MAX_ROWS],
            len: 0,
            dropped: 0,
        }
    }

    /// Removes every row.
    pub fn clear(&mut self) {
        self.len = 0;
        self.dropped = 0;
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there is no row.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The rows that did not fit (a train has more than [`MAX_ROWS`] rows).
    pub fn dropped(&self) -> u32 {
        self.dropped
    }

    /// Row `i`.
    pub fn get(&self, i: usize) -> Option<&Row> {
        if i < self.len { self.list.get(i) } else { None }
    }

    /// The rows in order.
    pub fn iter(&self) -> impl Iterator<Item = &Row> {
        self.list.iter().take(self.len)
    }

    fn push(&mut self, row: Row) {
        match self.list.get_mut(self.len) {
            Some(slot) => {
                *slot = row;
                self.len += 1;
            }
            None => self.dropped = self.dropped.saturating_add(1),
        }
    }
}

/// A slicer: pulse train in, rows out.
pub trait Slicer {
    /// Cuts `train` into rows; `rows` is cleared first. A header with the rate 0 gives no row.
    fn slice(&self, train: &PulseTrain<'_>, rows: &mut Rows);
}

fn check_us(v: f32) -> bool {
    v.is_finite() && v > 0.0
}

fn samples(us: f32, rate_hz: u32) -> f32 {
    us * (rate_hz as f32) / 1.0e6
}

/// Picks the nearer of two nominal widths `a` and `b` (samples) that `w` is within the
/// tolerance of: `(is_a, relative error)`.
fn pick2(w: f32, a: f32, b: f32, tol: Tolerance) -> Option<(bool, f32)> {
    let ea = absf(w - a);
    let eb = absf(w - b);
    let ok_a = ea <= tol.limit(a);
    let ok_b = eb <= tol.limit(b);
    match (ok_a, ok_b) {
        (true, true) => Some(if ea <= eb {
            (true, ea / a)
        } else {
            (false, eb / b)
        }),
        (true, false) => Some((true, ea / a)),
        (false, true) => Some((false, eb / b)),
        (false, false) => None,
    }
}

/// `w` as `round(w / unit)` units with an absolute tolerance of a fraction of a unit:
/// `(count, error in units)`; a count of 0 is possible (a width below half a unit).
fn units(w: f32, unit: f32, tol: Tolerance) -> Option<(u32, f32)> {
    let n = (w / unit + 0.5) as u32;
    let e = absf(w - (n as f32) * unit);
    if e <= tol.limit(unit) {
        Some((n, e / unit))
    } else {
        None
    }
}

fn windows_apart(a_us: f32, b_us: f32, tol: Tolerance) -> bool {
    a_us * (1.0 + tol.rel) < b_us * (1.0 - tol.rel)
}

/// Collects the rows of one train.
struct Builder<'a> {
    rows: &'a mut Rows,
    row: Row,
    open: bool,
    skip: bool,
    unit_sum: f32,
    unit_n: u32,
    us_per_sample: f32,
}

impl<'a> Builder<'a> {
    fn new(rows: &'a mut Rows, rate_hz: u32) -> Self {
        rows.clear();
        Self {
            rows,
            row: Row::EMPTY,
            open: false,
            skip: false,
            unit_sum: 0.0,
            unit_n: 0,
            us_per_sample: 1.0e6 / (rate_hz as f32),
        }
    }

    /// Opens a row at pair `i` when none is open; true when it was just opened.
    fn open_at(&mut self, i: usize) -> bool {
        if self.open {
            return false;
        }
        self.row = Row::EMPTY;
        self.row.start_pair = i as u32;
        self.unit_sum = 0.0;
        self.unit_n = 0;
        self.open = true;
        true
    }

    fn bits(&self) -> usize {
        self.row.len()
    }

    fn push_bit(&mut self, bit: bool) -> bool {
        let i = self.bits();
        if i >= MAX_ROW_BITS {
            return false;
        }
        if let Some(byte) = self.row.bytes.get_mut(i / 8)
            && bit
        {
            *byte |= 0x80 >> (i % 8);
        }
        self.row.bits += 1;
        true
    }

    /// Records a checked width: the unit it implies (samples) and its relative error.
    fn measure(&mut self, unit_samples: f32, err: f32) {
        self.unit_sum += unit_samples * self.us_per_sample;
        self.unit_n += 1;
        self.measure_err(err);
    }

    /// Records the error of a width that carries no unit.
    fn measure_err(&mut self, err: f32) {
        if err > self.row.worst_err {
            self.row.worst_err = err;
        }
    }

    fn emit(&mut self, end: RowEnd) {
        self.row.end = end;
        if self.unit_n > 0 {
            self.row.unit_avg_us = self.unit_sum / (self.unit_n as f32);
        }
        let row = self.row;
        self.rows.push(row);
        self.open = false;
    }

    /// Ends the row with an error and skips to the next reset gap.
    fn fail(&mut self, end: RowEnd) {
        if self.open {
            self.emit(end);
        }
        self.skip = true;
    }

    /// The end of a row at a reset gap or the end of the train.
    fn terminal(&mut self, end: RowEnd) {
        if self.skip {
            self.skip = false;
        } else if self.open {
            if self.row.bits > 0 {
                self.emit(end);
            } else {
                self.open = false;
            }
        }
    }
}

fn end_kind(last: bool, reset: bool) -> RowEnd {
    if last && !reset {
        RowEnd::TrainEnd
    } else {
        RowEnd::Reset
    }
}

/// Pulse-width slicer: the pulse carries the bit, the gap is a fixed separator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pwm {
    short_us: f32,
    long_us: f32,
    gap_us: f32,
    reset_us: f32,
    short_is_one: bool,
    tol: Tolerance,
}

impl Pwm {
    /// A slicer for pulses near `short_us` and `long_us`, gaps near `gap_us` and rows split at
    /// gaps above `reset_us`; a short pulse is a 1 when `short_is_one`.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] unless `0 < short_us < long_us`, `gap_us > 0`,
    /// `reset_us > gap_us` and the tolerance windows of the two pulse widths are apart.
    pub fn new(
        short_us: f32,
        long_us: f32,
        gap_us: f32,
        reset_us: f32,
        short_is_one: bool,
    ) -> Result<Self> {
        Self {
            short_us,
            long_us,
            gap_us,
            reset_us,
            short_is_one,
            tol: Tolerance::DEFAULT,
        }
        .with_tolerance(Tolerance::DEFAULT)
    }

    /// The same slicer with another tolerance.
    ///
    /// # Errors
    /// As [`Pwm::new`].
    pub fn with_tolerance(mut self, tol: Tolerance) -> Result<Self> {
        let ok = check_us(self.short_us)
            && check_us(self.long_us)
            && check_us(self.gap_us)
            && check_us(self.reset_us)
            && self.reset_us > self.gap_us
            && windows_apart(self.short_us, self.long_us, tol);
        if !ok {
            return Err(Error::InvalidArgument("invalid pwm slicer configuration"));
        }
        self.tol = tol;
        Ok(self)
    }
}

impl Slicer for Pwm {
    fn slice(&self, train: &PulseTrain<'_>, rows: &mut Rows) {
        let rate = train.header().rate_hz;
        if rate == 0 {
            rows.clear();
            return;
        }
        let (s, l) = (samples(self.short_us, rate), samples(self.long_us, rate));
        let (g, r) = (samples(self.gap_us, rate), samples(self.reset_us, rate));
        let mut b = Builder::new(rows, rate);
        let n = train.len();
        for (i, p) in train.pairs().enumerate() {
            let last = i + 1 == n;
            let reset = (p.gap as f32) > r;
            let terminal = last || reset;
            if !b.skip {
                b.open_at(i);
                match pick2(p.pulse as f32, s, l, self.tol) {
                    Some((is_short, e)) => {
                        let unit = if is_short {
                            p.pulse as f32
                        } else {
                            p.pulse as f32 * s / l
                        };
                        if b.push_bit(is_short == self.short_is_one) {
                            b.measure(unit, e);
                        } else {
                            b.fail(RowEnd::Full);
                        }
                    }
                    None => b.fail(RowEnd::BadWidth),
                }
                if !b.skip && !terminal {
                    let e = absf(p.gap as f32 - g);
                    if e <= self.tol.limit(g) {
                        b.measure_err(e / g);
                    } else {
                        b.fail(RowEnd::BadWidth);
                    }
                }
            }
            if terminal {
                b.terminal(end_kind(last, reset));
            }
        }
    }
}

/// Pulse-position slicer: the gap after a pulse carries the bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ppm {
    short_gap_us: f32,
    long_gap_us: f32,
    reset_us: f32,
    tol: Tolerance,
}

impl Ppm {
    /// A slicer for gaps near `short_gap_us` (a 0) and `long_gap_us` (a 1); rows are split at
    /// gaps above `reset_us`. The pulse before a reset gap (or the end of the train) is an end
    /// marker and carries no bit.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] unless `0 < short_gap_us < long_gap_us`,
    /// `reset_us > long_gap_us` and the tolerance windows are apart.
    pub fn new(short_gap_us: f32, long_gap_us: f32, reset_us: f32) -> Result<Self> {
        Self {
            short_gap_us,
            long_gap_us,
            reset_us,
            tol: Tolerance::DEFAULT,
        }
        .with_tolerance(Tolerance::DEFAULT)
    }

    /// The same slicer with another tolerance.
    ///
    /// # Errors
    /// As [`Ppm::new`].
    pub fn with_tolerance(mut self, tol: Tolerance) -> Result<Self> {
        let ok = check_us(self.short_gap_us)
            && check_us(self.long_gap_us)
            && check_us(self.reset_us)
            && self.reset_us > self.long_gap_us
            && windows_apart(self.short_gap_us, self.long_gap_us, tol);
        if !ok {
            return Err(Error::InvalidArgument("invalid ppm slicer configuration"));
        }
        self.tol = tol;
        Ok(self)
    }
}

impl Slicer for Ppm {
    fn slice(&self, train: &PulseTrain<'_>, rows: &mut Rows) {
        let rate = train.header().rate_hz;
        if rate == 0 {
            rows.clear();
            return;
        }
        let (s, l) = (
            samples(self.short_gap_us, rate),
            samples(self.long_gap_us, rate),
        );
        let r = samples(self.reset_us, rate);
        let mut b = Builder::new(rows, rate);
        let n = train.len();
        for (i, p) in train.pairs().enumerate() {
            let last = i + 1 == n;
            let reset = (p.gap as f32) > r;
            let terminal = last || reset;
            if !b.skip {
                b.open_at(i);
                if !terminal {
                    match pick2(p.gap as f32, s, l, self.tol) {
                        Some((is_short, e)) => {
                            let unit = if is_short {
                                p.gap as f32
                            } else {
                                p.gap as f32 * s / l
                            };
                            if b.push_bit(!is_short) {
                                b.measure(unit, e);
                            } else {
                                b.fail(RowEnd::Full);
                            }
                        }
                        None => b.fail(RowEnd::BadWidth),
                    }
                }
            }
            if terminal {
                b.terminal(end_kind(last, reset));
            }
        }
    }
}

/// The half-bit convention of [`Manchester`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Convention {
    /// A high half bit followed by a low one is a 1 (falling edge in the middle), low then high
    /// a 0.
    Ieee8023,
    /// The opposite of [`Convention::Ieee8023`]: low then high is a 1.
    Thomas,
}

/// Manchester slicer: pulses and gaps are cut into half-bit periods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Manchester {
    half_bit_us: f32,
    reset_us: f32,
    convention: Convention,
    virtual_lead: bool,
    tol: Tolerance,
}

impl Manchester {
    /// A slicer for half bits of `half_bit_us`, rows split at gaps above `reset_us`.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] unless `half_bit_us > 0` and `reset_us > 2.4 half_bit_us`.
    pub fn new(half_bit_us: f32, reset_us: f32, convention: Convention) -> Result<Self> {
        Self {
            half_bit_us,
            reset_us,
            convention,
            virtual_lead: false,
            tol: Tolerance::DEFAULT,
        }
        .with_tolerance(Tolerance::DEFAULT)
    }

    /// The same slicer with another tolerance.
    ///
    /// # Errors
    /// As [`Manchester::new`].
    pub fn with_tolerance(mut self, tol: Tolerance) -> Result<Self> {
        let ok = check_us(self.half_bit_us)
            && check_us(self.reset_us)
            && self.reset_us > 2.0 * self.half_bit_us * (1.0 + tol.rel);
        if !ok {
            return Err(Error::InvalidArgument(
                "invalid manchester slicer configuration",
            ));
        }
        self.tol = tol;
        Ok(self)
    }

    /// The same slicer where every row starts with a virtual low half bit, for a message whose
    /// first bit is 0 in the IEEE convention (the transmitter is silent in that half bit).
    #[must_use]
    pub fn with_virtual_lead(mut self, on: bool) -> Self {
        self.virtual_lead = on;
        self
    }
}

/// The half-bit machine of [`Manchester`].
struct Halves {
    pending: Option<bool>,
    ieee: bool,
}

impl Halves {
    /// Adds one half bit; the error is the end of the row.
    fn push(&mut self, b: &mut Builder<'_>, level: bool) -> core::result::Result<(), RowEnd> {
        match self.pending.take() {
            None => {
                self.pending = Some(level);
                Ok(())
            }
            Some(first) => {
                if first == level {
                    return Err(RowEnd::Violation);
                }
                let bit = if self.ieee { first } else { !first };
                if b.push_bit(bit) {
                    Ok(())
                } else {
                    Err(RowEnd::Full)
                }
            }
        }
    }

    fn run(
        &mut self,
        b: &mut Builder<'_>,
        w: f32,
        level: bool,
        half: f32,
        tol: Tolerance,
    ) -> core::result::Result<(), RowEnd> {
        let (one, e) = pick2(w, half, 2.0 * half, tol).ok_or(RowEnd::BadWidth)?;
        let count = if one { 1 } else { 2 };
        b.measure(w / (count as f32), e);
        for _ in 0..count {
            self.push(b, level)?;
        }
        Ok(())
    }
}

impl Slicer for Manchester {
    fn slice(&self, train: &PulseTrain<'_>, rows: &mut Rows) {
        let rate = train.header().rate_hz;
        if rate == 0 {
            rows.clear();
            return;
        }
        let half = samples(self.half_bit_us, rate);
        let r = samples(self.reset_us, rate);
        let mut b = Builder::new(rows, rate);
        let mut h = Halves {
            pending: None,
            ieee: self.convention == Convention::Ieee8023,
        };
        let n = train.len();
        for (i, p) in train.pairs().enumerate() {
            let last = i + 1 == n;
            let reset = (p.gap as f32) > r;
            let terminal = last || reset;
            if !b.skip {
                if b.open_at(i) {
                    h.pending = if self.virtual_lead { Some(false) } else { None };
                }
                let mut res = h.run(&mut b, p.pulse as f32, true, half, self.tol);
                if res.is_ok() && !terminal {
                    res = h.run(&mut b, p.gap as f32, false, half, self.tol);
                }
                if res.is_ok() && terminal && h.pending == Some(true) {
                    // the idle level completes the last bit
                    res = h.push(&mut b, false);
                }
                if let Err(end) = res {
                    b.fail(end);
                }
            }
            if terminal {
                b.terminal(end_kind(last, reset));
            }
        }
    }
}

/// NRZ slicer for FSK: a pulse (upper frequency) is `round(w / bit)` ones, a gap zeros.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nrz {
    bit_us: f32,
    reset_us: f32,
    tol: Tolerance,
}

impl Nrz {
    /// A slicer for bits of `bit_us`, rows split at gaps above `reset_us`.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] unless `bit_us > 0` and `reset_us > bit_us`.
    pub fn new(bit_us: f32, reset_us: f32) -> Result<Self> {
        Self {
            bit_us,
            reset_us,
            tol: Tolerance::DEFAULT,
        }
        .with_tolerance(Tolerance::DEFAULT)
    }

    /// The same slicer with another tolerance (a run may differ from a whole number of bits by
    /// `rel` of a bit, at least `min_samples`).
    ///
    /// # Errors
    /// As [`Nrz::new`].
    pub fn with_tolerance(mut self, tol: Tolerance) -> Result<Self> {
        if !(check_us(self.bit_us) && check_us(self.reset_us) && self.reset_us > self.bit_us) {
            return Err(Error::InvalidArgument("invalid nrz slicer configuration"));
        }
        self.tol = tol;
        Ok(self)
    }
}

impl Slicer for Nrz {
    fn slice(&self, train: &PulseTrain<'_>, rows: &mut Rows) {
        let rate = train.header().rate_hz;
        if rate == 0 {
            rows.clear();
            return;
        }
        let bit = samples(self.bit_us, rate);
        let r = samples(self.reset_us, rate);
        let mut b = Builder::new(rows, rate);
        let n = train.len();
        for (i, p) in train.pairs().enumerate() {
            let last = i + 1 == n;
            let reset = (p.gap as f32) > r;
            let terminal = last || reset;
            if !b.skip {
                b.open_at(i);
                let first = b.bits() == 0;
                if let Err(end) = nrz_run(&mut b, p.pulse as f32, true, bit, self.tol, first) {
                    b.fail(end);
                } else if !terminal
                    && let Err(end) = nrz_run(&mut b, p.gap as f32, false, bit, self.tol, false)
                {
                    b.fail(end);
                }
            }
            if terminal {
                b.terminal(end_kind(last, reset));
            }
        }
    }
}

/// One run of an NRZ train; a count of 0 is allowed only for the first pulse of a row (a
/// frame that starts with the lower frequency).
fn nrz_run(
    b: &mut Builder<'_>,
    w: f32,
    level: bool,
    bit: f32,
    tol: Tolerance,
    first: bool,
) -> core::result::Result<(), RowEnd> {
    let (count, e) = units(w, bit, tol).ok_or(RowEnd::BadWidth)?;
    if count == 0 {
        if first {
            b.measure_err(e);
            return Ok(());
        }
        return Err(RowEnd::BadWidth);
    }
    b.measure(w / (count as f32), e);
    for _ in 0..count {
        if !b.push_bit(level) {
            return Err(RowEnd::Full);
        }
    }
    Ok(())
}
