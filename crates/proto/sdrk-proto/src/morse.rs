//! Morse code (CW): the table of ITU-R M.1677-1, an [`Encoder`] that turns text into key
//! states one unit at a time and a [`Decoder`] that turns key-down and key-up samples back
//! into text.
//!
//! # Code and timing
//!
//! The code is built from dots and dashes. Timing is counted in units: a dot is 1 unit of
//! key down, a dash 3, the gap inside a character is 1 unit of key up, the gap between
//! characters 3 and the gap between words 7. The unit lasts `1.2 / wpm` seconds, so the word
//! `PARIS` plus its word gap is 50 units (the standard that defines "words per minute").
//! With Farnsworth timing the characters are sent at `char_wpm` and only the gaps between
//! characters and between words are stretched, so that the effective speed is `wpm`.
//!
//! # Text
//!
//! Letters (either case), digits and the punctuation `. , : ? ' - / ( ) " = + @` have a code.
//! The prosigns `AR`, `BT` and `KN` have the codes of `+`, `=` and `(`, so those three
//! characters stand for them; `SK` is the byte [`SK`] (ASCII end of transmission). A space
//! (or a tab, a carriage return or a line feed) is a word gap. A byte without a code is
//! skipped and counted by the caller.
//!
//! ```
//! use sdrk_proto::morse::{self, Encoder, Push};
//!
//! // "E T": a dot, the word gap of 7 units, a dash.
//! let mut enc = Encoder::new(20.0, 20.0)?;
//! let mut units = [false; 32];
//! let mut n = 0;
//! for &c in b"E T" {
//!     assert_ne!(enc.push(c)?, Push::Skipped);
//!     while let Some(key) = enc.next_unit() {
//!         units[n] = key;
//!         n += 1;
//!     }
//! }
//! let on = |k: bool| if k { '#' } else { '.' };
//! let text: String = units[..n].iter().map(|&k| on(k)).collect();
//! assert_eq!(text, "#.......###");
//! assert_eq!(morse::code(b'k'), Some("-.-"));
//! # Ok::<(), sdrk_proto::Error>(())
//! ```

use crate::error::{Error, Result};

/// The prosign `SK` (end of contact, `...-.-`) as a byte of the text: ASCII end of
/// transmission.
pub const SK: u8 = 0x04;

/// The longest code in the table, in elements.
pub const MAX_ELEMENTS: usize = 6;

/// Most key-down runs the [`Decoder`] collects for one character; a character with more is
/// discarded and counted as an overflow.
pub const MAX_COLLECT: usize = 8;

/// The most units one character takes: six dashes (`6 * 3`) with five gaps of one unit.
pub const MAX_CHAR_UNITS: u32 = 23;

/// The largest `wpm` and `char_wpm` the encoder accepts.
pub const MAX_WPM: f32 = 100.0;

/// The smallest `wpm` the encoder accepts.
pub const MIN_WPM: f32 = 1.0;

/// The table: byte, code (`.` dot, `-` dash). One entry per code.
const TABLE: [(u8, &str); 50] = [
    (b'A', ".-"),
    (b'B', "-..."),
    (b'C', "-.-."),
    (b'D', "-.."),
    (b'E', "."),
    (b'F', "..-."),
    (b'G', "--."),
    (b'H', "...."),
    (b'I', ".."),
    (b'J', ".---"),
    (b'K', "-.-"),
    (b'L', ".-.."),
    (b'M', "--"),
    (b'N', "-."),
    (b'O', "---"),
    (b'P', ".--."),
    (b'Q', "--.-"),
    (b'R', ".-."),
    (b'S', "..."),
    (b'T', "-"),
    (b'U', "..-"),
    (b'V', "...-"),
    (b'W', ".--"),
    (b'X', "-..-"),
    (b'Y', "-.--"),
    (b'Z', "--.."),
    (b'0', "-----"),
    (b'1', ".----"),
    (b'2', "..---"),
    (b'3', "...--"),
    (b'4', "....-"),
    (b'5', "....."),
    (b'6', "-...."),
    (b'7', "--..."),
    (b'8', "---.."),
    (b'9', "----."),
    (b'.', ".-.-.-"),
    (b',', "--..--"),
    (b':', "---..."),
    (b'?', "..--.."),
    (b'\'', ".----."),
    (b'-', "-....-"),
    (b'/', "-..-."),
    (b'(', "-.--."),
    (b')', "-.--.-"),
    (b'"', ".-..-."),
    (b'=', "-...-"),
    (b'+', ".-.-."),
    (b'@', ".--.-."),
    (SK, "...-.-"),
];

/// The code of the byte `c` as dots and dashes (`".-"` for `A`); lower case letters are
/// folded to upper case. `None` for a byte without a code (a space included).
///
/// ```
/// use sdrk_proto::morse::code;
/// assert_eq!(code(b'a'), Some(".-"));
/// assert_eq!(code(b'0'), Some("-----"));
/// assert_eq!(code(b'+'), Some(".-.-."));
/// assert_eq!(code(b' '), None);
/// ```
pub fn code(c: u8) -> Option<&'static str> {
    let c = c.to_ascii_uppercase();
    TABLE.iter().find(|e| e.0 == c).map(|e| e.1)
}

/// The byte whose code is `elements` (`true` is a dash); `None` when no character has it.
///
/// ```
/// use sdrk_proto::morse::character;
/// assert_eq!(character(&[false, true]), Some(b'A'));
/// assert_eq!(character(&[true, true, true, true, true, true, true]), None);
/// ```
pub fn character(elements: &[bool]) -> Option<u8> {
    TABLE
        .iter()
        .find(|e| {
            e.1.len() == elements.len()
                && e.1
                    .bytes()
                    .zip(elements)
                    .all(|(b, &dash)| (b == b'-') == dash)
        })
        .map(|e| e.0)
}

/// The number of units the characters of `text` take, with `char_gap` units between
/// characters and `word_gap` between words, without the gap before the first character
/// (bytes without a code are skipped). Counts what [`Encoder`] sends.
///
/// ```
/// use sdrk_proto::morse::units;
/// // PARIS and its word gap, the word gap counted before the next word: 43 + 7 = 50.
/// assert_eq!(units(b"PARIS E", 3, 7), 43 + 7 + 1);
/// ```
pub fn units(text: &[u8], char_gap: u32, word_gap: u32) -> u32 {
    let mut total = 0;
    let mut sent = false;
    let mut word = false;
    for &c in text {
        if matches!(c, b' ' | b'\t' | b'\r' | b'\n') {
            word = true;
            continue;
        }
        let Some(code) = code(c) else {
            continue;
        };
        if sent {
            total += if word { word_gap } else { char_gap };
        }
        sent = true;
        word = false;
        let mut first = true;
        for e in code.bytes() {
            if !first {
                total += 1;
            }
            first = false;
            total += if e == b'-' { 3 } else { 1 };
        }
    }
    total
}

/// What [`Encoder::push`] did with a byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Push {
    /// The character was queued; [`Encoder::next_unit`] returns its units.
    Queued,
    /// The byte was a space (or tab, carriage return, line feed): the next character follows
    /// after a word gap.
    Space,
    /// The byte has no code and was skipped.
    Skipped,
}

/// Morse encoder: bytes in, one key state per unit out. Heap-free.
///
/// The caller pushes one byte at a time with [`Encoder::push`] and drains its units with
/// [`Encoder::next_unit`] (`true` is key down) until it returns `None`. The gap before a
/// character is sent in front of it, so the stream ends with the last element and the caller
/// decides what follows (a pause, or the next packet, which starts with
/// [`Encoder::word_break`]).
///
/// ```
/// use sdrk_proto::morse::{Encoder, Push};
/// let mut enc = Encoder::new(20.0, 20.0)?;
/// assert_eq!(enc.push(b'A')?, Push::Queued);
/// let on: Vec<bool> = core::iter::from_fn(|| enc.next_unit()).collect();
/// assert_eq!(on, [true, false, true, true, true]); // dot, gap, dash
/// assert!(enc.is_idle());
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Encoder {
    char_gap: u32,
    word_gap: u32,
    /// Units of key up still to send before the next element.
    lead: u32,
    /// Units of key down still to send in the element in progress.
    on_left: u32,
    /// The elements after the one in progress (`.` and `-`).
    rest: &'static [u8],
    /// Whether one unit of key up is due before the next element of `rest`.
    gap_due: bool,
    /// Whether a character has been sent since construction or reset.
    sent: bool,
    /// Whether a word gap is due before the next character.
    word_due: bool,
}

impl Encoder {
    /// An encoder for the speed `wpm` of the characters with the effective speed `wpm`
    /// (Farnsworth timing when `char_wpm` is larger: the characters go at `char_wpm`, the
    /// gaps are stretched so that the text takes as long as at `wpm`).
    ///
    /// The gaps are whole units: `max(3, round(g))` and `max(7, round(7 g / 3))` with
    /// `g = (1800 r - 1116) / 228` and `r = char_wpm / wpm` (`g` is 3 at `r = 1`).
    ///
    /// # Errors
    /// `InvalidArgument` unless `1 <= wpm <= char_wpm <= 100`.
    ///
    /// ```
    /// use sdrk_proto::morse::Encoder;
    /// assert!(Encoder::new(20.0, 20.0).is_ok());
    /// assert!(Encoder::new(20.0, 10.0).is_err()); // the characters cannot be slower
    /// assert!(Encoder::new(0.5, 0.5).is_err());
    /// ```
    pub fn new(wpm: f32, char_wpm: f32) -> Result<Self> {
        if !(MIN_WPM..=MAX_WPM).contains(&wpm) || !(wpm..=MAX_WPM).contains(&char_wpm) {
            return Err(Error::InvalidArgument("morse speed"));
        }
        let ratio = char_wpm / wpm;
        let g = (1800.0 * ratio - 1116.0) / 228.0;
        let char_gap = ((g + 0.5) as u32).max(3);
        let word_gap = ((g * 7.0 / 3.0 + 0.5) as u32).max(7);
        Ok(Self {
            char_gap,
            word_gap,
            lead: 0,
            on_left: 0,
            rest: &[],
            gap_due: false,
            sent: false,
            word_due: false,
        })
    }

    /// Units of the gap between two characters.
    pub fn char_gap(&self) -> u32 {
        self.char_gap
    }

    /// Units of the gap between two words.
    pub fn word_gap(&self) -> u32 {
        self.word_gap
    }

    /// Forgets the character in progress and everything sent so far: the next character is
    /// sent without a gap in front.
    pub fn reset(&mut self) {
        self.lead = 0;
        self.on_left = 0;
        self.rest = &[];
        self.gap_due = false;
        self.sent = false;
        self.word_due = false;
    }

    /// Whether every unit of the pushed characters has been returned.
    pub fn is_idle(&self) -> bool {
        self.lead == 0 && self.on_left == 0 && !self.gap_due
    }

    /// Marks a word gap before the next character (ignored before the first one).
    pub fn word_break(&mut self) {
        self.word_due = true;
    }

    /// Tells the encoder that the caller has sent the gap after the last character itself:
    /// the next character follows without a gap in front, as the first one does.
    ///
    /// ```
    /// use sdrk_proto::morse::Encoder;
    /// let mut enc = Encoder::new(20.0, 20.0)?;
    /// enc.push(b'E')?;
    /// while enc.next_unit().is_some() {}
    /// enc.settle();
    /// enc.push(b'E')?;
    /// assert_eq!(enc.next_unit(), Some(true)); // no gap first
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn settle(&mut self) {
        self.sent = false;
        self.word_due = false;
    }

    /// Queues the byte `c`.
    ///
    /// # Errors
    /// `InvalidArgument` when the units of the previous character have not all been drained
    /// ([`Encoder::is_idle`] is false).
    pub fn push(&mut self, c: u8) -> Result<Push> {
        if !self.is_idle() {
            return Err(Error::InvalidArgument("morse encoder is busy"));
        }
        if matches!(c, b' ' | b'\t' | b'\r' | b'\n') {
            self.word_due = true;
            return Ok(Push::Space);
        }
        let Some(code) = code(c) else {
            return Ok(Push::Skipped);
        };
        self.lead = match (self.sent, self.word_due) {
            (false, _) => 0,
            (true, true) => self.word_gap,
            (true, false) => self.char_gap,
        };
        self.sent = true;
        self.word_due = false;
        self.rest = code.as_bytes();
        self.load();
        Ok(Push::Queued)
    }

    /// Takes the next element of `rest` as the element in progress.
    fn load(&mut self) {
        if let Some((&e, rest)) = self.rest.split_first() {
            self.on_left = if e == b'-' { 3 } else { 1 };
            self.rest = rest;
        }
    }

    /// The next unit of the queued characters (`true` is key down), or `None` when they are
    /// all sent.
    pub fn next_unit(&mut self) -> Option<bool> {
        if self.lead > 0 {
            self.lead -= 1;
            return Some(false);
        }
        if self.on_left > 0 {
            self.on_left -= 1;
            if self.on_left == 0 && !self.rest.is_empty() {
                self.gap_due = true;
            }
            return Some(true);
        }
        if self.gap_due {
            self.gap_due = false;
            self.load();
            return Some(false);
        }
        None
    }
}

/// Most key-down runs the decoder keeps for the estimate of the dot and dash lengths.
const WINDOW: usize = 16;

/// Key-up run after which the decoder forgets the speed, in word gaps (D903: 20).
const IDLE_WORDS: f32 = 20.0;

/// A character or a space the [`Decoder`] has finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Emit {
    /// The character (upper case), `b' '` for a word gap or `b'?'` for a sequence of dots and
    /// dashes that is no character.
    pub byte: u8,
    /// Index of the first sample of the first key-down run of the character; for a space,
    /// that of the last character.
    pub start: u64,
}

/// Morse decoder: key-down and key-up samples in, characters out. Heap-free.
///
/// The caller gives one key state per sample with [`Decoder::push`] (`true` is key down) and
/// the index of that sample. The decoder measures the runs itself:
///
/// - **Elements.** It keeps the last 16 key-down runs. If the 20th and the 80th percentile
///   of them differ by a factor of 1.8 or more, the runs form two clusters (dots and dashes)
///   split in the middle between those two values, and the dot length `m_dot` and the dash
///   length `m_dash` are the means of the clusters. A window of equal runs is taken as dots
///   when its mean is near the previous dot length, as dashes when it is near the previous
///   dash length, and otherwise by the speed hint (dashes when it is at least twice the
///   hint unit). A run is a dash when it is at least `(m_dot + m_dash) / 2`.
/// - **Gaps.** A key-up run of `2 m_dot` ends the character, of `5 m_dot` the word, and of
///   `140 m_dot` (20 word gaps) the transmission: the estimate is forgotten, so a sender at
///   another speed is picked up.
/// - **Characters.** The elements of a character are looked up in the table; a sequence
///   without a code is returned as `?` (counted in [`Decoder::unknown`]); more than
///   [`MAX_COLLECT`] runs discard the character (counted in [`Decoder::overflow`]).
///
/// The first characters of a transmission are decoded with the estimate of what has been
/// seen so far; a character with elements of both lengths gives it at once, a start with
/// several equal elements (`E`, `T`, `5`) relies on the hint, so a preamble such as `VVV`
/// settles the estimate.
///
/// ```
/// use sdrk_proto::morse::Decoder;
/// let mut dec = Decoder::new(6.0)?; // a dot of 6 samples (the hint)
/// let mut out = Vec::new();
/// // "A" = dot, gap, dash at 6 samples per unit, then silence until the word gap.
/// let key = [(true, 6), (false, 6), (true, 18), (false, 40)];
/// let mut index = 0;
/// for (on, n) in key {
///     for _ in 0..n {
///         out.extend(dec.push(on, index).map(|e| e.byte));
///         index += 1;
///     }
/// }
/// assert_eq!(out, b"A ");
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
// The four flags are independent states of the run, the character and the estimate.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug)]
pub struct Decoder {
    /// The dot length of the speed hint, in samples.
    hint: f32,
    level: bool,
    /// Length of the current run so far, in samples.
    run: u32,
    /// Key-down runs of the character in progress.
    lens: [u32; MAX_COLLECT],
    n: usize,
    over: bool,
    /// Index of the first sample of the first run of the character in progress.
    start: u64,
    /// Whether a character has been finished since the last space.
    word_open: bool,
    win: [u32; WINDOW],
    wn: usize,
    wi: usize,
    m_dot: f32,
    m_dash: f32,
    known: bool,
    char_thr: u32,
    word_thr: u32,
    idle_thr: u32,
    /// The key-up run length at which `gap` has something to do (`u32::MAX`: nothing).
    event: u32,
    unknown: u32,
    overflow: u32,
    uncertain: u32,
}

impl Decoder {
    /// A decoder for runs counted in samples, with the dot length `hint_samples` as the
    /// hint of the speed.
    ///
    /// # Errors
    /// `InvalidArgument` unless `hint_samples` is at least 1 and finite.
    pub fn new(hint_samples: f32) -> Result<Self> {
        if !(hint_samples >= 1.0 && hint_samples.is_finite()) {
            return Err(Error::InvalidArgument("morse decoder hint"));
        }
        Ok(Self::fresh(hint_samples))
    }

    /// A decoder in its initial state (`hint` is at least 1).
    fn fresh(hint: f32) -> Self {
        let mut d = Self {
            hint,
            level: false,
            run: 0,
            lens: [0; MAX_COLLECT],
            n: 0,
            over: false,
            start: 0,
            word_open: false,
            win: [0; WINDOW],
            wn: 0,
            wi: 0,
            m_dot: hint,
            m_dash: 3.0 * hint,
            known: false,
            char_thr: 0,
            word_thr: 0,
            idle_thr: 0,
            event: u32::MAX,
            unknown: 0,
            overflow: 0,
            uncertain: 0,
        };
        d.set_thresholds();
        d
    }

    /// Back to the state after construction (the counters too).
    pub fn reset(&mut self) {
        *self = Self::fresh(self.hint);
    }

    /// The dot length now estimated, in samples (the hint before the first runs).
    pub fn dot_samples(&self) -> f32 {
        self.m_dot
    }

    /// The dash length now estimated, in samples.
    pub fn dash_samples(&self) -> f32 {
        self.m_dash
    }

    /// Characters returned as `?` since construction or reset.
    pub fn unknown(&self) -> u32 {
        self.unknown
    }

    /// Characters discarded for more than [`MAX_COLLECT`] runs.
    pub fn overflow(&self) -> u32 {
        self.overflow
    }

    /// Characters with a run within 20 % of the split between dots and dashes.
    pub fn uncertain(&self) -> u32 {
        self.uncertain
    }

    /// Whether a character is in progress (a key-down run has started or ended and the
    /// character is not finished).
    pub fn pending(&self) -> bool {
        self.n > 0 || self.over
    }

    fn set_thresholds(&mut self) {
        self.char_thr = (2.0 * self.m_dot) as u32;
        self.word_thr = (5.0 * self.m_dot) as u32;
        self.idle_thr = (7.0 * IDLE_WORDS * self.m_dot) as u32;
    }

    /// Forgets the speed: the next transmission is learned afresh.
    fn forget(&mut self) {
        self.win = [0; WINDOW];
        self.wn = 0;
        self.wi = 0;
        self.known = false;
        self.m_dot = self.hint;
        self.m_dash = 3.0 * self.hint;
        self.set_thresholds();
    }

    /// Takes the key state of one sample; returns a character or a space when this sample
    /// finishes one.
    // `always`: the fast path of a sample inside a run is a few instructions, and the call and
    // the return of the `Option<Emit>` cost more (measured, D909).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn push(&mut self, on: bool, index: u64) -> Option<Emit> {
        if on == self.level {
            self.run = self.run.saturating_add(1);
            if on || self.run != self.event {
                return None;
            }
        }
        self.push_slow(on, index)
    }

    /// The samples that do something: a change of the key state and a key-up run that reaches
    /// the end of a character, a word or the transmission.
    #[inline(never)]
    fn push_slow(&mut self, on: bool, index: u64) -> Option<Emit> {
        if on == self.level {
            return self.gap();
        }
        if self.level {
            self.finish_run();
        }
        self.level = on;
        self.run = 1;
        if on {
            if self.n == 0 && !self.over {
                self.start = index;
            }
        } else {
            self.event = self.next_event();
        }
        None
    }

    /// The key-up run length at which the next thing happens: the end of the character, the
    /// word or the transmission.
    fn next_event(&self) -> u32 {
        if self.pending() {
            self.char_thr
        } else if self.word_open {
            self.word_thr
        } else {
            self.idle_thr
        }
        .max(2)
    }

    /// Finishes the character in progress and the word, as if a long gap had come.
    pub fn flush(&mut self) -> (Option<Emit>, Option<Emit>) {
        if self.level {
            // The key-down run in progress ends here.
            self.finish_run();
            self.level = false;
            self.run = 0;
        }
        let first = if self.pending() {
            self.finish_char()
        } else {
            None
        };
        let second = if self.word_open {
            self.word_open = false;
            Some(Emit {
                byte: b' ',
                start: self.start,
            })
        } else {
            None
        };
        if !self.level {
            self.event = self.next_event();
        }
        (first, second)
    }

    /// A key-down run of `self.run` samples has ended.
    fn finish_run(&mut self) {
        let r = self.run;
        if let Some(slot) = self.lens.get_mut(self.n) {
            *slot = r;
            self.n += 1;
        } else {
            self.over = true;
        }
        if let Some(slot) = self.win.get_mut(self.wi) {
            *slot = r;
        }
        self.wi = (self.wi + 1) % WINDOW;
        self.wn = (self.wn + 1).min(WINDOW);
        self.estimate();
    }

    /// One more sample of key up: the gap thresholds.
    fn gap(&mut self) -> Option<Emit> {
        let run = self.run;
        let emit = if self.pending() {
            if run >= self.char_thr {
                self.finish_char()
            } else {
                None
            }
        } else if self.word_open && run >= self.word_thr {
            self.word_open = false;
            Some(Emit {
                byte: b' ',
                start: self.start,
            })
        } else {
            None
        };
        if run == self.idle_thr {
            self.forget();
            self.event = u32::MAX;
        } else {
            // The next thing happens later in this key-up run (or never).
            let next = self.next_event();
            self.event = if next > run { next } else { u32::MAX };
        }
        emit
    }

    /// Looks up the character in progress.
    fn finish_char(&mut self) -> Option<Emit> {
        let n = self.n;
        let over = self.over;
        let start = self.start;
        let split = f32::midpoint(self.m_dot, self.m_dash);
        let els: [bool; MAX_COLLECT] =
            core::array::from_fn(|i| self.lens.get(i).is_some_and(|&l| l as f32 >= split));
        let unsure = self
            .lens
            .iter()
            .take(n)
            .any(|&l| (l as f32 - split).abs() < 0.2 * split);
        self.n = 0;
        self.over = false;
        if over {
            self.overflow = self.overflow.saturating_add(1);
            return None;
        }
        if unsure {
            self.uncertain = self.uncertain.saturating_add(1);
        }
        self.word_open = true;
        let byte = els.get(..n).and_then(character).unwrap_or_else(|| {
            self.unknown = self.unknown.saturating_add(1);
            b'?'
        });
        Some(Emit { byte, start })
    }

    /// The dot and dash lengths from the window of runs.
    fn estimate(&mut self) {
        let mut s = self.win;
        let n = self.wn;
        let Some(sorted) = s.get_mut(..n) else {
            return;
        };
        sorted.sort_unstable();
        let (Some(&lo), Some(&hi)) = (
            sorted.get(n / 5),
            sorted.get(n.saturating_sub(1).min(4 * n / 5)),
        ) else {
            return;
        };
        let (lo, hi) = (lo as f32, hi as f32);
        if lo <= 0.0 {
            return;
        }
        if hi >= 1.8 * lo {
            let t = f32::midpoint(lo, hi);
            let (mut sd, mut nd, mut sh, mut nh) = (0.0f32, 0u32, 0.0f32, 0u32);
            for &r in sorted.iter() {
                let r = r as f32;
                if r < t {
                    sd += r;
                    nd += 1;
                } else {
                    sh += r;
                    nh += 1;
                }
            }
            if nd > 0 && nh > 0 {
                self.m_dot = sd / nd as f32;
                self.m_dash = sh / nh as f32;
            }
        } else {
            let sum: f32 = sorted.iter().map(|&r| r as f32).sum();
            let mean = sum / n as f32;
            let (dot, dash) = (self.m_dot, self.m_dash);
            if self.known && (0.6 * dot..=1.6 * dot).contains(&mean) {
                self.m_dash = dash * (mean / dot);
                self.m_dot = mean;
            } else if self.known && (0.6 * dash..=1.6 * dash).contains(&mean) {
                self.m_dot = dot * (mean / dash);
                self.m_dash = mean;
            } else if mean >= 2.0 * self.hint {
                self.m_dash = mean;
                self.m_dot = mean / 3.0;
            } else {
                self.m_dot = mean;
                self.m_dash = 3.0 * mean;
            }
        }
        self.known = true;
        self.set_thresholds();
    }
}
