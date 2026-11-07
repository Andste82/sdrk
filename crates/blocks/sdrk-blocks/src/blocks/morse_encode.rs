//! `morse_encode`: text packets to the keying envelope of Morse code (CW), one `f32` sample
//! per output sample, with raised-cosine edges.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_proto::Packet;
use sdrk_proto::morse::{Encoder, MAX_CHAR_UNITS};

use super::{hz, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// Fewest output samples per unit (`OutOfRange` of `wpm` below; D900).
const MIN_SAMPLES_PER_UNIT: f64 = 8.0;

/// Largest `rise_ms`.
const MAX_RISE_MS: f64 = 50.0;

/// Description of the `morse_encode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "morse_encode",
    summary: "Turns text packets into the keying envelope of Morse code (CW).",
    details: "Sends the bytes of every packet as Morse code (ITU-R M.1677-1): letters (either \
        case), digits and . , : ? ' - / ( ) \" = + @; a space, tab, carriage return or line \
        feed is a word gap, a byte without a code is skipped; the codes of + = ( are the \
        prosigns AR BT KN and the byte 0x04 is the prosign SK. The unit lasts 1.2 / \
        char_wpm seconds (the standard word PARIS is 50 units at wpm = char_wpm); a dot is 1 \
        unit of key down, a dash 3, the gap inside a character 1, between characters 3 and \
        between words 7. With char_wpm above wpm the characters go at char_wpm and the gaps \
        between characters and words are stretched to whole units so that the text takes as \
        long as at wpm (Farnsworth timing). The output is the envelope of the key, 1.0 for \
        key down and 0.0 for key up, at the sample rate of the block; every edge is a \
        raised cosine of rise_ms (0 for hard keying) so that the keyed carrier makes no key \
        clicks. Every packet ends with one word gap of key up, which separates it from the next \
        packet and lets a decoder finish the last word. It is a source: the output rate is \
        the rate_hz of the packet input of the pipeline, and in push mode the block produces \
        the samples of the packets it has; in pull mode, with no packet, it outputs key up. \
        A packet is consumed when its word gap is written; a packet that does not fit the \
        output is continued in the next call, which must start with the same packet. \
        Pitfall: the envelope is not a carrier; multiply it with a carrier (nco, duc) or \
        put it through to_complex and duc for an IQ transmitter.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Packets],
        description: "The text to send; index and flags are ignored.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The key envelope, 0.0 to 1.0, one sample per output sample.",
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
            typical: "20",
            mutability: Mutability::Build,
            description: "Speed of the text in words per minute (the word PARIS and its word \
                gap, 50 units, counts as one word).",
        },
        ParamDesc {
            name: "char_wpm",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 100.0,
                default: Some(0.0),
            },
            typical: "0 (same as wpm); 18 with wpm 10 for Farnsworth timing",
            mutability: Mutability::Build,
            description: "Speed of the characters in words per minute; 0 is wpm, otherwise at \
                least wpm.",
        },
        ParamDesc {
            name: "rise_ms",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: MAX_RISE_MS,
                default: Some(5.0),
            },
            typical: "5",
            mutability: Mutability::Build,
            description: "Length of the raised-cosine edge in milliseconds, 0 to 50; at most \
                half of a unit, longer values are cut to that.",
        },
    ],
    required: &[],
};

/// Configuration of [`MorseEncodeBlock`].
///
/// ```
/// use sdrk_blocks::MorseEncodeConfig;
/// let cfg = MorseEncodeConfig::default();
/// assert_eq!((cfg.wpm, cfg.char_wpm, cfg.rise_ms), (20.0, 0.0, 5.0));
/// assert!(cfg.validate(sdrk::Rate::hz(8000)?).is_ok());
/// // 60 wpm at 300 Hz is 6 samples per unit: too few.
/// let fast = MorseEncodeConfig { wpm: 60.0, ..cfg };
/// assert!(fast.validate(sdrk::Rate::hz(300)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorseEncodeConfig {
    /// Speed of the text in words per minute (5 to 60).
    pub wpm: f64,
    /// Speed of the characters in words per minute (0: same as `wpm`; else `wpm` to 100).
    pub char_wpm: f64,
    /// Length of an edge in milliseconds (0 to 50).
    pub rise_ms: f64,
}

impl Default for MorseEncodeConfig {
    fn default() -> Self {
        Self {
            wpm: 20.0,
            char_wpm: 0.0,
            rise_ms: 5.0,
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

impl MorseEncodeConfig {
    /// The speed of the characters: `char_wpm`, or `wpm` when that is 0.
    pub fn char_speed(&self) -> f64 {
        if self.char_wpm == 0.0 {
            self.wpm
        } else {
            self.char_wpm
        }
    }

    /// Output samples per unit at the sample rate `rate`.
    pub fn samples_per_unit(&self, rate: Rate) -> f64 {
        hz(rate) * 1.2 / self.char_speed()
    }

    /// Checks the ranges and that a unit has at least 8 samples at `rate`.
    ///
    /// # Errors
    /// `OutOfRange` for `wpm`, `char_wpm` or `rise_ms`; a `wpm` or `char_wpm` that gives fewer
    /// than 8 samples per unit is `OutOfRange` of that parameter too.
    pub fn validate(&self, rate: Rate) -> Result<()> {
        if !(5.0..=60.0).contains(&self.wpm) {
            return Err(out_of_range("wpm", 5.0, 60.0, self.wpm));
        }
        if self.char_wpm != 0.0 && !(self.wpm..=100.0).contains(&self.char_wpm) {
            return Err(out_of_range("char_wpm", self.wpm, 100.0, self.char_wpm));
        }
        if !(0.0..=MAX_RISE_MS).contains(&self.rise_ms) {
            return Err(out_of_range("rise_ms", 0.0, MAX_RISE_MS, self.rise_ms));
        }
        let fastest = hz(rate) * 1.2 / MIN_SAMPLES_PER_UNIT;
        if self.char_speed() > fastest {
            let name = if self.char_wpm == 0.0 {
                "wpm"
            } else {
                "char_wpm"
            };
            return Err(out_of_range(name, 5.0, fastest, self.char_speed()));
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
            "char_wpm" => self.char_wpm = value.as_f64()?,
            "rise_ms" => self.rise_ms = value.as_f64()?,
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
            "char_wpm" => Ok(ParamValue::F64(self.char_wpm)),
            "rise_ms" => Ok(ParamValue::F64(self.rise_ms)),
            _ => unknown(),
        }
    }
}

/// Morse encoder: [`Packets`] of text in, the key envelope out (`f32`, 1.0 key down, 0.0 key
/// up). The block needs no storage.
///
/// The units of the text come from [`sdrk_proto::morse::Encoder`]; unit `k` ends at the
/// sample `round((k + 1) * samples_per_unit)`, so the speed is exact for any rate. An edge
/// between two units of different key state is a raised cosine of `rise_ms` (at most half of
/// a unit). Every packet ends with one word gap of key up (7 units at the speed of the
/// characters; it lets a decoder finish the last word); with input packets the call stops
/// after the last gap, an empty input is idle and outputs key up.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, MorseEncodeBlock, MorseEncodeConfig, Packet};
/// use sdrk_proto::PacketFlags;
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = MorseEncodeConfig { rise_ms: 0.0, ..MorseEncodeConfig::default() };
/// let mut enc = MorseEncodeBlock::new(&cfg, Rate::hz(1000)?, &mut arena)?;
/// // 20 wpm: a unit is 60 samples at 1000 Hz. "E" is one unit of key down and the tail.
/// let p = Packet::new(0, PacketFlags::NONE, b"E")?;
/// let mut out = vec![0.0f32; 1000];
/// let progress = enc.process(&[p], &mut out)?;
/// assert_eq!((progress.consumed, progress.produced), (1, 60 + 7 * 60));
/// assert!(out[..60].iter().all(|&x| x == 1.0));
/// assert!(out[60..420].iter().all(|&x| x == 0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct MorseEncodeBlock<S: Storage> {
    config: MorseEncodeConfig,
    enc: Encoder,
    /// The next byte of the first input packet.
    pos: usize,
    /// Output samples per unit.
    spu: f64,
    /// The next unit to start.
    unit_k: u64,
    /// The first sample after the current unit.
    unit_end: u64,
    /// Output samples since construction or reset.
    n: u64,
    /// The key state of the current unit.
    key: bool,
    /// Samples of the current edge so far.
    ramp: u32,
    /// Samples of one edge.
    ramp_len: u32,
    /// The level at the start of the current edge.
    from: f32,
    /// The level of the last sample.
    level: f32,
    /// Units of key up still to send after the last byte of the packet in progress (its tail).
    tail: u32,
    /// Whether the tail of the packet in progress has been queued.
    tail_done: bool,
    /// Largest number of output samples of one packet.
    max_samples: Ratio,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> MorseEncodeBlock<S> {
    /// Creates the block; it needs no storage. The key starts up.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration or a rate that gives fewer than 8 samples
    /// per unit.
    pub fn new(config: &MorseEncodeConfig, rate_out: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_out)?;
        let enc = Encoder::new(config.wpm as f32, config.char_speed() as f32)?;
        let spu = config.samples_per_unit(rate_out);
        let ramp_len =
            libm::round(config.rise_ms * 1e-3 * hz(rate_out)).min(libm::floor(spu / 2.0));
        // 256 characters, each with its gap in front (one more unit for rounding).
        let units = 256 * (u64::from(MAX_CHAR_UNITS) + u64::from(enc.word_gap()));
        let max = units * (libm::ceil(spu) as u64 + 1);
        Ok(Self {
            config: *config,
            enc,
            pos: 0,
            spu,
            unit_k: 0,
            unit_end: 0,
            n: 0,
            key: false,
            ramp: 0,
            ramp_len: ramp_len as u32,
            from: 0.0,
            level: 0.0,
            tail: 0,
            tail_done: false,
            max_samples: Ratio::new(max, 1)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &MorseEncodeConfig, rate_out: Rate) -> Result<usize> {
        config.validate(rate_out)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate: the rate the block is built with (the `rate_hz` of its packet input).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn rate_out(config: &MorseEncodeConfig, rate: Rate) -> Result<Rate> {
        config.validate(rate)?;
        Ok(rate)
    }

    /// The next key state: from the encoder, from the bytes of the next packets, or key up.
    /// `idle_ok` is true for a call without packets (pull mode): then the key stays up when
    /// the encoder has nothing. Otherwise the end of the packets ends the call (`None`). After
    /// the last byte of every packet comes its tail, one word gap of key up.
    fn next_key(&mut self, input: &[Packet], consumed: &mut usize, idle_ok: bool) -> Option<bool> {
        loop {
            if let Some(k) = self.enc.next_unit() {
                return Some(k);
            }
            let Some(packet) = input.get(*consumed) else {
                return idle_ok.then_some(false);
            };
            if let Some(&c) = packet.data().get(self.pos) {
                self.pos += 1;
                // The encoder is idle here, so `push` cannot fail.
                let _ = self.enc.push(c);
                continue;
            }
            // The packet is complete; it is consumed after its tail, so that a call that
            // ends inside the tail is continued with it.
            if !self.tail_done {
                self.tail_done = true;
                self.tail = self.enc.word_gap();
            }
            if self.tail > 0 {
                self.tail -= 1;
                return Some(false);
            }
            self.enc.settle();
            *consumed += 1;
            self.pos = 0;
            self.tail_done = false;
        }
    }

    /// Starts the unit with the key state `key` at the current sample.
    fn start_unit(&mut self, key: bool) {
        if key != self.key {
            self.key = key;
            self.from = self.level;
            self.ramp = 0;
        }
        self.unit_k += 1;
        self.unit_end = libm::floor(self.unit_k as f64 * self.spu + 0.5) as u64;
    }

    /// The next output sample (the key state is that of the current unit).
    #[inline]
    fn shape(&mut self) -> f32 {
        let target = if self.key { 1.0 } else { 0.0 };
        if self.ramp < self.ramp_len {
            self.ramp += 1;
            let x = self.ramp as f32 / self.ramp_len as f32;
            let w = 0.5 * (1.0 - libm::cosf(core::f32::consts::PI * x));
            self.level = self.from + (target - self.from) * w;
        } else {
            self.level = target;
        }
        self.level
    }
}

impl<S: Storage> Block for MorseEncodeBlock<S> {
    type In = Packets;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[Packet], output: &mut [f32]) -> Result<Progress> {
        let idle_ok = input.is_empty();
        if idle_ok {
            // A packet that was not resubmitted is dropped.
            self.pos = 0;
        }
        let mut consumed = 0;
        let mut produced = 0;
        for o in output.iter_mut() {
            if self.n >= self.unit_end {
                let Some(key) = self.next_key(input, &mut consumed, idle_ok) else {
                    break;
                };
                self.start_unit(key);
            }
            *o = self.shape();
            self.n += 1;
            produced += 1;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.max_samples,
            kind: RatioKind::Source,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.enc.reset();
        self.pos = 0;
        self.unit_k = 0;
        self.unit_end = 0;
        self.n = 0;
        self.key = false;
        self.ramp = 0;
        self.from = 0.0;
        self.level = 0.0;
        self.tail = 0;
        self.tail_done = false;
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
