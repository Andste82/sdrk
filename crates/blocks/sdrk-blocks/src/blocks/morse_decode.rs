//! `morse_decode`: the envelope of a keyed carrier or tone to text packets, with an adaptive
//! threshold and a decoder that estimates the speed.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_proto::morse::{Decoder, Emit};
use sdrk_proto::{MAX_PACKET_BYTES, Packet, PacketFlags};

use super::{hz, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// Fewest samples per unit at `wpm_max` (`OutOfRange` of `wpm_max` below; D900).
const MIN_SAMPLES_PER_UNIT: f64 = 8.0;

/// Where the threshold lies between the noise and the signal level (D903).
const THRESHOLD_POSITION: f32 = 0.5;

/// Half width of the hysteresis around the threshold, as a part of the span (D903).
const HYSTERESIS: f32 = 0.05;

/// Time constant of the smoothing in units of the hint speed (D902).
const SMOOTH_UNITS: f64 = 0.125;

/// Time constant of the slow level trackers in units of the hint speed (D903).
const TRACK_UNITS: f64 = 20.0;

/// The key is down only above the noise mean plus this many standard deviations of the
/// smoothed noise (D904).
const NOISE_SIGMAS: f32 = 4.0;

/// Time constant of the noise statistics in units of the hint speed (D904).
const NOISE_UNITS: f64 = 8.0;

/// A run shorter than this part of a unit at `wpm_max` is merged (D905).
const GLITCH_FRACTION: f64 = 1.0 / 3.0;

/// Description of the `morse_decode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "morse_decode",
    summary: "Decodes Morse code (CW) from the envelope of a keyed carrier or tone to text \
        packets.",
    details: "Reads the envelope of a keyed carrier or tone (for example mag after a narrow \
        ddc) as one f32 per sample (a value that is not finite counts as 0). A one-pole smoother with a time constant of 1/8 unit at \
        the speed wpm removes the noise; the level trackers follow the signal level (peak hold, \
        decaying towards the noise level in 20 units) and the noise level (minimum hold, \
        rising in 20 units while the key is up) and the key is down when the smoothed value is \
        above the middle between them plus 5 % of their span, up when it is below the middle \
        minus 5 %; the threshold is at least the mean plus 4 standard deviations of the \
        smoothed envelope while the key is up (learned from the key-up samples, so noise alone \
        gives no key down) and the key is up when the signal level does not exceed that. With \
        threshold above 0 the level is fixed instead (a value above it is key down). A change \
        of the key state counts only after it has lasted a third of a unit at wpm_max (at \
        least 2 samples), shorter changes are merged into the surrounding run. The runs go to the \
        Morse decoder: it keeps the last 16 key-down runs, finds the dot and dash lengths as \
        the two clusters of the 20th and 80th percentile (a window of equal runs follows the \
        previous estimate, at the start the speed wpm), calls a run a dash from half way \
        between them, ends a character after a key-up run of 2 dot lengths and a word after \
        5, forgets the speed after 20 word gaps of silence, and looks the characters up in \
        the table of ITU-R M.1677-1 (a sequence without a code is ?). One packet per word: \
        the characters up to the word gap (or 256 characters), without the space; the index \
        is the input sample index of the first key-down sample of the word, counted since \
        construction or reset. The estimated speed is wpm() and the counters are unknown(), \
        overflow() and uncertain(). Pitfall: a transmission that starts with several equal \
        elements (E, T, 5) is decoded with the speed wpm until the first character with a dot \
        and a dash settles the estimate; start with a preamble such as VVV, or set wpm near \
        the speed of the sender. Pitfall: the first block (four time constants of the smoother, about half a unit) \
        after the start or a reset only learns the noise level and gives no key down, so the \
        stream must begin with noise or silence of about a unit; a signal on from the first \
        sample is taken for noise until the key has been up for some blocks. Pitfall: the front end sets the noise bandwidth; \
        100 to 250 Hz is right for CW up to 30 wpm.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "The envelope of the keyed signal, at least 8 samples per unit at \
            wpm_max.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "The decoded words as ASCII text (upper case), one packet per word.",
    }],
    params: &[
        ParamDesc {
            name: "wpm",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 5.0,
                max: 60.0,
                default: Some(20.0),
            },
            typical: "20; near the speed of the sender",
            mutability: Mutability::Build,
            description: "Speed hint in words per minute: the starting estimate of the dot \
                length and the time constant of the smoother.",
        },
        ParamDesc {
            name: "wpm_max",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 5.0,
                max: 100.0,
                default: Some(40.0),
            },
            typical: "40",
            mutability: Mutability::Build,
            description: "Fastest speed expected, at least wpm: the shortest run that counts \
                is a third of a unit at this speed, and a unit at this speed needs 8 samples.",
        },
        ParamDesc {
            name: "threshold",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0e6,
                default: Some(0.0),
            },
            typical: "0 (adaptive); a fixed level for a known signal",
            mutability: Mutability::Build,
            description: "Fixed level of the smoothed envelope above which the key is down; 0 \
                is the adaptive threshold.",
        },
    ],
    required: &[],
};

/// Configuration of [`MorseDecodeBlock`].
///
/// ```
/// use sdrk_blocks::MorseDecodeConfig;
/// let cfg = MorseDecodeConfig::default();
/// assert_eq!((cfg.wpm, cfg.wpm_max, cfg.threshold), (20.0, 40.0, 0.0));
/// assert!(cfg.validate(sdrk::Rate::hz(8000)?).is_ok());
/// // 40 wpm at 200 Hz is 5 samples per unit: too few.
/// assert!(cfg.validate(sdrk::Rate::hz(200)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorseDecodeConfig {
    /// Speed hint in words per minute (5 to 60).
    pub wpm: f64,
    /// Fastest speed expected in words per minute (`wpm` to 100).
    pub wpm_max: f64,
    /// Fixed threshold (0: adaptive).
    pub threshold: f64,
}

impl Default for MorseDecodeConfig {
    fn default() -> Self {
        Self {
            wpm: 20.0,
            wpm_max: 40.0,
            threshold: 0.0,
        }
    }
}

fn out_of_range(name: &'static str, min: f64, max: f64, got: f64) -> Error {
    Error::OutOfRange {
        name,
        min,
        max,
        got,
    }
}

impl MorseDecodeConfig {
    /// Checks the ranges and that a unit at `wpm_max` has at least 8 samples at `rate`.
    ///
    /// # Errors
    /// `OutOfRange` of `wpm`, `wpm_max` or `threshold`.
    pub fn validate(&self, rate: Rate) -> Result<()> {
        if !(5.0..=60.0).contains(&self.wpm) {
            return Err(out_of_range("wpm", 5.0, 60.0, self.wpm));
        }
        if !(self.wpm..=100.0).contains(&self.wpm_max) {
            return Err(out_of_range("wpm_max", self.wpm, 100.0, self.wpm_max));
        }
        if !(0.0..=1.0e6).contains(&self.threshold) {
            return Err(out_of_range("threshold", 0.0, 1.0e6, self.threshold));
        }
        let fastest = hz(rate) * 1.2 / MIN_SAMPLES_PER_UNIT;
        if self.wpm_max > fastest {
            return Err(out_of_range("wpm_max", self.wpm, fastest, self.wpm_max));
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "wpm" => self.wpm = value.as_f64()?,
            "wpm_max" => self.wpm_max = value.as_f64()?,
            "threshold" => self.threshold = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "wpm" => Ok(ParamValue::F64(self.wpm)),
            "wpm_max" => Ok(ParamValue::F64(self.wpm_max)),
            "threshold" => Ok(ParamValue::F64(self.threshold)),
            _ => unknown(),
        }
    }
}

/// Morse decoder: the key envelope in (`f32`), [`Packets`] of text out, one per word. The
/// block needs no storage.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, MorseDecodeBlock, MorseDecodeConfig, Packet};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = MorseDecodeConfig::default();
/// let mut dec = MorseDecodeBlock::new(&cfg, Rate::hz(1000)?, &mut arena)?;
/// // "K" (-.-) at 20 wpm and 1000 Hz: units of 60 samples, then silence for the word gap.
/// let mut env = vec![0.0f32; 240];
/// for (on, n) in [(1, 180), (0, 60), (1, 60), (0, 60), (1, 180), (0, 600)] {
///     env.extend(std::iter::repeat_n(on as f32, n));
/// }
/// let mut out = [Packet::default(); 4];
/// let p = dec.process(&env, &mut out)?;
/// assert_eq!(p.consumed, env.len());
/// assert_eq!((p.produced, out[0].data()), (1, &b"K"[..]));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
// The flags are independent states of the detector and the noise blocks.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
pub struct MorseDecodeBlock<S: Storage> {
    config: MorseDecodeConfig,
    /// The input sample rate in Hz.
    rate_hz: f64,
    decoder: Decoder,
    /// Smoothing coefficient of the envelope.
    smooth: f32,
    /// Coefficient of the slow level trackers.
    slow: f32,
    /// Shortest key state that counts, in samples.
    glitch: u32,
    /// The smoothed envelope.
    y: f32,
    /// Whether the first sample has been seen (it starts the smoother and the trackers).
    started: bool,
    /// Signal level (peak hold).
    sig: f32,
    /// Noise level (minimum hold).
    noise: f32,
    /// Mean of the smoothed envelope in the quiet blocks (key up, away from an edge).
    mu: f64,
    /// Variance of the smoothed envelope in the quiet blocks.
    var: f64,
    /// Quiet blocks folded into `mu` and `var` so far (saturating).
    n_blk: u32,
    /// The noise floor, `mu + 4 sigma` (0 before the first block), refreshed with every block.
    floor: f32,
    /// Key-up samples since the key went up (the tail of the smoother does not count as noise).
    quiet: u32,
    /// Samples after which the tail of the smoother has died away (four time constants); also
    /// the length of a block of the noise statistics.
    settle: u32,
    /// Weight of one block in the one-pole average of the statistics.
    beta: f64,
    /// The block in progress: samples, sum, sum of squares, and whether it has stayed quiet.
    blk_n: u32,
    blk_sum: f64,
    blk_sq: f64,
    blk_quiet: bool,
    /// The last quiet block, waiting for the next block to be quiet too: samples, sum, sum of
    /// squares (0 samples: none).
    pend_n: u32,
    pend_sum: f64,
    pend_sq: f64,
    /// Whether the key is down according to the threshold.
    raw: bool,
    /// The key state the decoder has been given.
    committed: bool,
    /// Samples since `committed` and `raw` differ.
    pending: u32,
    /// Samples given to the decoder (the index of the next one).
    fed: u64,
    /// The word in progress.
    word: [u8; MAX_PACKET_BYTES],
    wlen: usize,
    wstart: u64,
    /// Input samples since construction or reset.
    count: u64,
    /// At most one packet per this many input samples.
    max_per_input: Ratio,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> MorseDecodeBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration or a rate with fewer than 8 samples per unit
    /// at `wpm_max`.
    pub fn new(config: &MorseDecodeConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let fs = hz(rate_in);
        let unit = fs * 1.2 / config.wpm;
        let settle = (4.0 * (1.0 + SMOOTH_UNITS * unit)) as u32;
        // The small addition keeps a quotient that rounding leaves just below a whole number.
        let unit_min = fs * 1.2 / config.wpm_max;
        let glitch = (libm::floor(GLITCH_FRACTION * unit_min + 1e-6) as u32).max(2);
        Ok(Self {
            config: *config,
            rate_hz: fs,
            decoder: Decoder::new(unit as f32)?,
            smooth: (1.0 / (1.0 + SMOOTH_UNITS * unit)) as f32,
            slow: (1.0 / (1.0 + TRACK_UNITS * unit)) as f32,
            glitch,
            y: 0.0,
            started: false,
            sig: 0.0,
            noise: 0.0,
            mu: 0.0,
            var: 0.0,
            n_blk: 0,
            floor: 0.0,
            quiet: u32::MAX / 2,
            settle,
            beta: f64::from(settle) / (f64::from(settle) + NOISE_UNITS * unit),
            blk_n: 0,
            blk_sum: 0.0,
            blk_sq: 0.0,
            blk_quiet: true,
            pend_n: 0,
            pend_sum: 0.0,
            pend_sq: 0.0,
            raw: false,
            committed: false,
            pending: 0,
            fed: 0,
            word: [0; MAX_PACKET_BYTES],
            wlen: 0,
            wstart: 0,
            count: 0,
            // A word needs a key-down run of `glitch` samples and a gap of five of them.
            max_per_input: Ratio::new(1, 6 * u64::from(glitch))?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &MorseDecodeConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Input samples consumed since construction or the last reset.
    pub fn position(&self) -> u64 {
        self.count
    }

    /// The speed now estimated, in words per minute (`wpm` before the first runs).
    pub fn wpm(&self) -> f64 {
        self.rate_hz * 1.2 / f64::from(self.decoder.dot_samples())
    }

    /// Characters returned as `?` (a sequence of dots and dashes without a code).
    pub fn unknown(&self) -> u32 {
        self.decoder.unknown()
    }

    /// Characters discarded for having more than 8 key-down runs.
    pub fn overflow(&self) -> u32 {
        self.decoder.overflow()
    }

    /// Characters with a run within 20 % of the split between dots and dashes.
    pub fn uncertain(&self) -> u32 {
        self.decoder.uncertain()
    }

    /// Adds the smoothed value `y` to the block of the noise statistics. A block is folded
    /// into the mean and the variance only if it and the next one were quiet (the key up, the
    /// tail of the smoother gone), so that the rising edge in front of an element, which the
    /// threshold takes some samples to see, does not count as noise.
    #[inline]
    fn noise_block(&mut self, y: f32) {
        if self.raw || self.quiet <= self.settle {
            self.blk_quiet = false;
        }
        self.blk_n += 1;
        let y = f64::from(y);
        self.blk_sum += y;
        self.blk_sq += y * y;
        if self.blk_n < self.settle {
            return;
        }
        if self.blk_quiet {
            if self.n_blk == 0 {
                // The first quiet block is folded at once: the stream starts with noise.
                self.pend_n = self.blk_n;
                self.pend_sum = self.blk_sum;
                self.pend_sq = self.blk_sq;
                self.fold_pending();
            } else {
                if self.pend_n > 0 {
                    self.fold_pending();
                }
                self.pend_n = self.blk_n;
                self.pend_sum = self.blk_sum;
                self.pend_sq = self.blk_sq;
            }
        } else {
            self.pend_n = 0;
        }
        self.blk_n = 0;
        self.blk_sum = 0.0;
        self.blk_sq = 0.0;
        self.blk_quiet = true;
    }

    /// Folds the pending quiet block into the mean and the variance.
    fn fold_pending(&mut self) {
        let count = f64::from(self.pend_n);
        let mean = self.pend_sum / count;
        let spread = (self.pend_sq / count - mean * mean).max(0.0);
        self.n_blk = self.n_blk.saturating_add(1);
        let weight = self.beta.max(1.0 / f64::from(self.n_blk));
        let shift = mean - self.mu;
        self.mu += weight * shift;
        self.var =
            (1.0 - weight) * self.var + weight * spread + weight * (1.0 - weight) * shift * shift;
        self.floor = (self.mu + f64::from(NOISE_SIGMAS) * libm::sqrt(self.var.max(0.0))) as f32;
        self.pend_n = 0;
    }

    /// The key state of the smoothed value `y`, with the threshold and the level trackers.
    #[inline]
    fn detect(&mut self, x: f32) -> bool {
        // A sample that is not finite counts as 0, so that one bad value does not stay in the
        // smoother.
        let x = if x.is_finite() { x } else { 0.0 };
        if self.started {
            self.y += self.smooth * (x - self.y);
        } else {
            // The smoother and the level trackers start at the first sample, so that the rise
            // from zero to the noise level is not taken for a signal.
            self.started = true;
            self.y = x;
            self.sig = x;
            self.noise = x;
        }
        let y = self.y;
        if self.config.threshold > 0.0 {
            let t = self.config.threshold as f32;
            let h = HYSTERESIS * t;
            return if self.raw { y >= t - h } else { y > t + h };
        }
        if self.raw {
            self.quiet = 0;
        } else {
            self.quiet = self.quiet.saturating_add(1);
        }
        self.noise_block(y);
        // The floor (0 until a quiet block has been folded in, so that a signal that starts at
        // once is not mistaken for noise) is refreshed with every block.
        let floor = self.floor;
        // Peak hold, decaying towards the noise level; minimum hold, rising while the key is up.
        if y > self.sig {
            self.sig = y;
        } else {
            self.sig -= (self.sig - self.noise) * self.slow;
        }
        if y < self.noise {
            self.noise = y;
        } else if !self.raw {
            self.noise += (y - self.noise) * self.slow;
        }
        let span = self.sig - self.noise;
        // The first block after the start or a reset only learns the noise: no key down.
        if self.n_blk == 0 || span <= 0.0 || self.sig <= floor || self.sig < 1.0e-12 {
            return false;
        }
        let t = (self.noise + THRESHOLD_POSITION * span).max(floor);
        let h = HYSTERESIS * span;
        if self.raw { y >= t - h } else { y > t + h }
    }

    /// Gives one sample of key state to the decoder and collects what it finishes.
    #[inline]
    fn feed(&mut self, on: bool, out: &mut Option<Packet>) {
        let index = self.fed;
        self.fed = self.fed.wrapping_add(1);
        if let Some(Emit { byte, start }) = self.decoder.push(on, index) {
            if byte == b' ' {
                self.finish_word(out);
            } else {
                if self.wlen == 0 {
                    self.wstart = start;
                }
                if let Some(slot) = self.word.get_mut(self.wlen) {
                    *slot = byte;
                    self.wlen += 1;
                }
                if self.wlen == MAX_PACKET_BYTES {
                    self.finish_word(out);
                }
            }
        }
    }

    /// Turns the word in progress into a packet (none for an empty word).
    fn finish_word(&mut self, out: &mut Option<Packet>) {
        if self.wlen == 0 {
            return;
        }
        let data = self.word.get(..self.wlen).unwrap_or(&[]);
        // The word has at most 256 bytes, so `new` cannot fail.
        *out = Packet::new(self.wstart, PacketFlags::NONE, data).ok();
        self.wlen = 0;
    }

    /// Takes the next input sample; returns a packet when the sample finishes a word.
    #[inline]
    fn step(&mut self, x: f32) -> Option<Packet> {
        let mut out = None;
        self.raw = self.detect(x);
        if self.raw == self.committed {
            // A change that did not last is merged into the run it interrupted.
            for _ in 0..self.pending {
                let c = self.committed;
                self.feed(c, &mut out);
            }
            self.pending = 0;
            let c = self.committed;
            self.feed(c, &mut out);
        } else {
            self.pending += 1;
            if self.pending >= self.glitch {
                self.committed = self.raw;
                for _ in 0..self.pending {
                    let c = self.committed;
                    self.feed(c, &mut out);
                }
                self.pending = 0;
            }
        }
        out
    }
}

impl<S: Storage> Block for MorseDecodeBlock<S> {
    type In = f32;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [Packet]) -> Result<Progress> {
        let mut slots = output.iter_mut();
        let Some(mut slot) = slots.next() else {
            return Ok(Progress::default());
        };
        let mut consumed = input.len();
        let mut produced = 0;
        for (i, &x) in input.iter().enumerate() {
            if let Some(packet) = self.step(x) {
                *slot = packet;
                produced += 1;
                let Some(next) = slots.next() else {
                    consumed = i + 1;
                    break;
                };
                slot = next;
            }
        }
        self.count = self.count.wrapping_add(consumed as u64);
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.max_per_input,
            kind: RatioKind::Packets {
                max_per_input: self.max_per_input,
            },
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.y = 0.0;
        self.started = false;
        self.sig = 0.0;
        self.noise = 0.0;
        self.mu = 0.0;
        self.var = 0.0;
        self.n_blk = 0;
        self.floor = 0.0;
        self.quiet = u32::MAX / 2;
        self.blk_n = 0;
        self.blk_sum = 0.0;
        self.blk_sq = 0.0;
        self.blk_quiet = true;
        self.pend_n = 0;
        self.raw = false;
        self.committed = false;
        self.pending = 0;
        self.fed = 0;
        self.wlen = 0;
        self.wstart = 0;
        self.count = 0;
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(_) => Err(Error::ReadOnlyParameter),
            None => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
