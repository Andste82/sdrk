//! `sensor_decode`: the pulse trains of `band_pulse_detect` to sensor readings as JSON text
//! packets (M46 §4). The block joins the trains that came in several packets, offers every
//! train to the enabled device decoders of [`sdrk_proto::sensors`], drops copies of a reading
//! that were already seen (repeats of the sender, the same burst on two neighbouring channels)
//! and writes one packet of JSON text per reading.

use core::fmt::Write as _;

use sdrk::{Delay, Progress, Ratio, RatioKind};
use sdrk_proto::pulses::{self, CONTINUED, HEADER_BYTES, PAIR_BYTES};
use sdrk_proto::sensors::{
    DECODERS, JsonBuf, MAX_READINGS, Mic, Reading, ReadingSink, SensorDecoder,
};
use sdrk_proto::{Packet, PacketFlags};

use super::unknown;
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// Pairs of the longest train the block joins; more pairs are dropped and the train is
/// decoded as far as it goes (D1113).
pub const SENSOR_TRAIN_PAIRS: usize = 1200;

/// Readings the block remembers for the removal of copies (D1114).
pub const SENSOR_SEEN: usize = 32;

/// The time within which the same reading on another channel is the same transmission, in
/// seconds (D1114).
pub const SENSOR_CROSS_CHANNEL_S: f64 = 0.5;

/// The bit mask of all built-in decoders: bit `i` selects the decoder `i` of
/// [`sdrk_proto::sensors::DECODERS`].
pub const SENSOR_ALL_DECODERS: i64 = 0b11_1111;

const TRAIN_BYTES: usize = HEADER_BYTES + SENSOR_TRAIN_PAIRS * PAIR_BYTES;
const CROSS_US: u64 = 500_000;
/// Words of one remembered reading: state and channel, hash, time, count.
const WORDS: usize = 4;

/// Description of the `sensor_decode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "sensor_decode",
    summary: "Decodes the pulse trains of band_pulse_detect into sensor readings as JSON text \
        packets.",
    details: "Reads the pulse train packets of band_pulse_detect (a train longer than one packet \
        arrives as several packets with the continued flag and is joined first; up to 1200 pairs, \
        the rest is dropped) and offers every train to the enabled device decoders of the \
        matching modulation: wh2, nexus, oregon_v21, wh1080 and bresser_3ch at 433 MHz (on-off \
        keying; one burst of bresser_3ch is about 1100 pairs) and lacrosse_tx29 at 868 MHz \
        (FSK). A decoder checks its sync pattern, the length and the \
        checksum or CRC (a device without one needs two equal repeats in the train) and the \
        range of every value; a train that fails is dropped. Every reading becomes one packet \
        whose payload is one JSON object (fields time, model, id, channel, battery_ok, the \
        values, mic, freq_MHz, rssi_dB and repeats in a fixed order) and whose index is the \
        index of the train; time is the index in seconds since construction or reset. A \
        reading is written when it is first seen: a copy of the same reading (the same model, \
        id, channel and values) on the same radio channel within merge_s of the first copy is \
        dropped, and so is a copy on another radio channel within 0.5 s (one transmission seen \
        on two neighbouring channels); the dropped copies are counted in the reading merged. \
        The repeats of a reading are the equal frames that one train held. With emit_unknown \
        a train that no decoder accepts is written as a JSON object with model unknown and \
        its pulse and gap widths in samples (cut after about 25 pairs to fit one packet, with \
        the truncated flag). A train gives at most 4 readings; the block needs room for 4 \
        packets before it reads a train. center_hz makes freq_MHz absolute: it is the RF \
        centre of the band, 0 writes the offset from the centre. Pitfall: the decoders of a \
        device without a check byte (nexus) report only what two equal frames confirm. \
        Pitfall: a reading is not held back to count its repeats, so repeats says how many \
        equal frames one train held, not how many trains the sender needed. Cost: 50 per \
        pulse train.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Packets],
        description: "The pulse trains of band_pulse_detect (a train longer than 30 pairs is \
            several packets, the first ones with the continued flag).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "One packet per reading: the JSON text of the reading, the index of the \
            pulse train.",
    }],
    params: &[
        ParamDesc {
            name: "decoders",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: SENSOR_ALL_DECODERS,
                default: Some(SENSOR_ALL_DECODERS),
            },
            typical: "63 (all); bit 0 wh2, 1 nexus, 2 oregon_v21, 3 wh1080, 4 lacrosse_tx29, 5 bresser_3ch",
            mutability: Mutability::Build,
            description: "The enabled decoders as a bit mask in the order of \
                sdrk_proto::sensors::DECODERS: bit 0 wh2, 1 nexus, 2 oregon_v21, 3 wh1080, 4 \
                lacrosse_tx29, 5 bresser_3ch.",
        },
        ParamDesc {
            name: "merge_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 3600.0,
                default: Some(2.0),
            },
            typical: "2; the time a sender needs for its repeats",
            mutability: Mutability::Build,
            description: "A copy of a reading on the same radio channel within this time of the \
                first copy is dropped; 0 keeps every copy on the same channel.",
        },
        ParamDesc {
            name: "emit_unknown",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true to analyse a new device",
            mutability: Mutability::Build,
            description: "Write a train that no decoder accepts as a JSON object with model \
                unknown and its pulse and gap widths.",
        },
        ParamDesc {
            name: "center_hz",
            unit: Unit::Hz,
            kind: ParamKind::I64 {
                min: 0,
                max: 4_000_000_000,
                default: Some(0),
            },
            typical: "433920000; 0 writes the offset from the band centre",
            mutability: Mutability::Build,
            description: "RF centre of the band in Hz; freq_MHz is this plus the frequency \
                offset of the train.",
        },
    ],
    required: &[],
};

/// Configuration of [`SensorDecodeBlock`].
///
/// ```
/// use sdrk_blocks::{SensorDecodeConfig, ParamValue};
/// let mut cfg = SensorDecodeConfig::default();
/// assert_eq!((cfg.decoders, cfg.merge_s, cfg.emit_unknown), (63, 2.0, false));
/// cfg.set("decoders", ParamValue::I64(SensorDecodeConfig::mask_of(&["nexus", "wh2"])?))?;
/// assert_eq!(cfg.decoders, 3);
/// assert!(cfg.set("decoders", ParamValue::I64(0)).is_err());
/// assert!(SensorDecodeConfig::mask_of(&["nope"]).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorDecodeConfig {
    /// The enabled decoders as a bit mask (bit `i`: decoder `i` of
    /// [`sdrk_proto::sensors::DECODERS`]).
    pub decoders: u32,
    /// Time in seconds within which a copy of a reading on the same channel is dropped.
    pub merge_s: f64,
    /// Whether trains that no decoder accepts are written.
    pub emit_unknown: bool,
    /// RF centre of the band in Hz (0: `freq_MHz` is the offset).
    pub center_hz: u32,
}

impl Default for SensorDecodeConfig {
    fn default() -> Self {
        Self {
            decoders: 0b11_1111,
            merge_s: 2.0,
            emit_unknown: false,
            center_hz: 0,
        }
    }
}

impl SensorDecodeConfig {
    /// The mask of the decoders with these names, for the parameter `decoders`.
    ///
    /// # Errors
    /// `InvalidParameter` (`decoders`) for a name that no decoder has.
    pub fn mask_of(names: &[&str]) -> Result<i64> {
        let mut mask = 0i64;
        for name in names {
            let i =
                DECODERS
                    .iter()
                    .position(|d| d.name() == *name)
                    .ok_or(Error::InvalidParameter {
                        name: "decoders",
                        reason: "unknown decoder name",
                    })?;
            mask |= 1 << i;
        }
        Ok(mask)
    }

    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` of `decoders` (no bit, or a bit without a decoder) or `merge_s`.
    pub fn validate(&self) -> Result<()> {
        let all = (1u32 << DECODERS.len()) - 1;
        if self.decoders == 0 || self.decoders & !all != 0 {
            return Err(Error::OutOfRange {
                name: "decoders",
                min: 1.0,
                max: f64::from(all),
                got: f64::from(self.decoders),
            });
        }
        if !(0.0..=3600.0).contains(&self.merge_s) {
            return Err(Error::OutOfRange {
                name: "merge_s",
                min: 0.0,
                max: 3600.0,
                got: self.merge_s,
            });
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `OutOfRange` of `decoders` also for
    /// a bit without a decoder.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        let mut c = *self;
        match p.name {
            "decoders" => c.decoders = u32::try_from(value.as_i64()?).unwrap_or(0),
            "merge_s" => c.merge_s = value.as_f64()?,
            "emit_unknown" => c.emit_unknown = value.as_bool()?,
            "center_hz" => c.center_hz = u32::try_from(value.as_i64()?).unwrap_or(0),
            _ => return unknown(),
        }
        c.validate()?;
        *self = c;
        Ok(())
    }

    /// Current value of a parameter.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "decoders" => Ok(ParamValue::I64(i64::from(self.decoders))),
            "merge_s" => Ok(ParamValue::F64(self.merge_s)),
            "emit_unknown" => Ok(ParamValue::Bool(self.emit_unknown)),
            "center_hz" => Ok(ParamValue::I64(i64::from(self.center_hz))),
            _ => unknown(),
        }
    }
}

/// Sensor decoder: pulse train [`Packets`] in, [`Packets`] of JSON readings out.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, Packet, SensorDecodeBlock, SensorDecodeConfig};
/// use sdrk_proto::pulses::{self, Header, Modulation, Pair};
/// use sdrk_proto::PacketFlags;
/// let cfg = SensorDecodeConfig::default();
/// let mut mem = vec![0u8; SensorDecodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut dec = SensorDecodeBlock::new(&cfg, &mut arena)?;
/// // A train that no decoder knows gives no reading.
/// let header = Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: 0,
///     level_db10: -200, noise_db10: -600, rate_hz: 250_000 };
/// let pairs = [Pair { pulse: 100, gap: 100 }; 8];
/// let train = pulses::encode_packet(0, PacketFlags::NONE, &header, &pairs)?;
/// let mut out = [Packet::default(); 4];
/// let p = dec.process(&[train], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (1, 0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SensorDecodeBlock<S: Storage> {
    config: SensorDecodeConfig,
    /// The train in progress as the payload of one packet of any length: the header, then the
    /// pairs of all packets so far.
    train: S::Buf<u8>,
    /// Remembered readings, `WORDS` words each.
    seen: S::Buf<u64>,
    /// Bytes of `train` in use.
    len: usize,
    /// Index and header bytes of the train in progress.
    open: Option<(u64, [u8; HEADER_BYTES])>,
    /// Whether pairs of the train in progress were dropped.
    cut: bool,
    /// The slot of `seen` that the next new reading takes.
    next: usize,
    merge_us: u64,
    merged: u64,
    unknown: u64,
    invalid: u64,
    lost: u64,
}

impl<S: Storage> core::fmt::Debug for SensorDecodeBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SensorDecodeBlock")
            .field("config", &self.config)
            .field("merged", &self.merged)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> SensorDecodeBlock<S> {
    /// Creates the block.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration, `Storage` if `storage` is too small.
    pub fn new(config: &SensorDecodeConfig, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let train = storage.alloc::<u8>(TRAIN_BYTES)?;
        let seen = storage.alloc::<u64>(SENSOR_SEEN * WORDS)?;
        Ok(Self {
            config: *config,
            train,
            seen,
            len: 0,
            open: None,
            cut: false,
            next: 0,
            merge_us: (config.merge_s * 1.0e6) as u64,
            merged: 0,
            unknown: 0,
            invalid: 0,
            lost: 0,
        })
    }

    /// Exact bytes of storage the block needs.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &SensorDecodeConfig) -> Result<usize> {
        config.validate()?;
        let mut plan = Plan::default();
        plan.add::<u8>(TRAIN_BYTES)?;
        plan.add::<u64>(SENSOR_SEEN * WORDS)?;
        Ok(plan.bytes())
    }

    /// Copies of readings dropped so far (repeats and the same burst on two channels).
    pub fn merged(&self) -> u64 {
        self.merged
    }

    /// Trains written as unknown so far.
    pub fn unknown(&self) -> u64 {
        self.unknown
    }

    /// Packets that were not a valid pulse train.
    pub fn invalid(&self) -> u64 {
        self.invalid
    }

    /// Readings that did not fit the output or the JSON buffer.
    pub fn lost(&self) -> u64 {
        self.lost
    }

    /// Adds `packet` to the train in progress; true when it was the last one.
    fn assemble(&mut self, packet: &Packet) -> bool {
        let data = packet.data();
        if pulses::decode(data).is_err() {
            self.open = None;
            self.len = 0;
            self.invalid += 1;
            return false;
        }
        let Some(head) = data
            .get(..HEADER_BYTES)
            .and_then(|h| <[u8; HEADER_BYTES]>::try_from(h).ok())
        else {
            return false;
        };
        let train = self.train.as_mut();
        if self.open != Some((packet.index, head)) {
            self.open = Some((packet.index, head));
            self.cut = false;
            if let Some(dst) = train.get_mut(..HEADER_BYTES) {
                dst.copy_from_slice(&head);
            }
            self.len = HEADER_BYTES;
        }
        let body = data.get(HEADER_BYTES..).unwrap_or(&[]);
        let room = TRAIN_BYTES.saturating_sub(self.len) / PAIR_BYTES * PAIR_BYTES;
        let take = body.len().min(room);
        if take < body.len() {
            self.cut = true;
        }
        if let (Some(dst), Some(src)) = (train.get_mut(self.len..self.len + take), body.get(..take))
        {
            dst.copy_from_slice(src);
            self.len += take;
        }
        !packet.flags.contains(CONTINUED)
    }

    /// Decodes the joined train and writes the packets; returns how many.
    fn finish(&mut self, index: u64, output: &mut [Packet]) -> usize {
        let cut = self.cut;
        self.open = None;
        let len = self.len;
        self.len = 0;
        let Some(Ok(train)) = self.train.as_ref().get(..len).map(pulses::decode) else {
            self.invalid += 1;
            return 0;
        };
        let header = train.header();
        let t_us = u64::try_from(u128::from(index) * 1_000_000 / u128::from(header.rate_hz.max(1)))
            .unwrap_or(u64::MAX);
        let mut produced = 0usize;
        let mut accepted = false;
        for (i, d) in DECODERS.iter().enumerate() {
            if self.config.decoders >> i & 1 == 0 || d.modulation() != header.modulation {
                continue;
            }
            let mut run = Run {
                seen: self.seen.as_mut(),
                next: &mut self.next,
                merged: &mut self.merged,
                lost: &mut self.lost,
                merge_us: self.merge_us,
                center_hz: self.config.center_hz,
                index,
                t_us,
            };
            accepted |= run.decoder(*d, &train, output, &mut produced);
        }
        if !accepted && self.config.emit_unknown {
            let packet = unknown_packet(&train, index, t_us, self.config.center_hz, cut);
            match (packet, output.get_mut(produced)) {
                (Some(p), Some(slot)) => {
                    *slot = p;
                    produced += 1;
                    self.unknown += 1;
                }
                _ => self.lost += 1,
            }
        }
        produced
    }
}

/// The state one train needs while the decoders are offered it: the remembered readings and
/// counters of the block, and the index and time of the train.
struct Run<'a> {
    seen: &'a mut [u64],
    next: &'a mut usize,
    merged: &'a mut u64,
    lost: &'a mut u64,
    merge_us: u64,
    center_hz: u32,
    index: u64,
    t_us: u64,
}

impl Run<'_> {
    /// Offers `train` to one decoder and writes the packets of its new readings after the
    /// first `produced` of `output`; true if the decoder accepted the train.
    fn decoder(
        &mut self,
        d: &dyn SensorDecoder,
        train: &pulses::PulseTrain<'_>,
        output: &mut [Packet],
        produced: &mut usize,
    ) -> bool {
        let mut sink = ReadingSink::new();
        if d.decode(train, &mut sink).count() == 0 {
            return false;
        }
        for r in sink.iter() {
            let mut r = *r;
            r.time_s = Some(self.t_us as f64 / 1.0e6);
            r.center_hz = self.center_hz;
            let dup = seen_before(
                self.seen,
                r.content_hash(),
                self.t_us,
                train.header().channel,
                self.merge_us,
            );
            if dup {
                *self.merged += 1;
                continue;
            }
            match (reading_packet(&r, self.index), output.get_mut(*produced)) {
                (Some(p), Some(slot)) => {
                    *slot = p;
                    *produced += 1;
                    // Remembered only once delivered: a lost reading must not block its copies.
                    remember(
                        self.seen,
                        self.next,
                        r.content_hash(),
                        self.t_us,
                        train.header().channel,
                    );
                }
                _ => *self.lost += 1,
            }
        }
        true
    }
}

/// The packet of a reading: its JSON text, flagged `CRC_OK` when a checksum or CRC matched.
fn reading_packet(r: &Reading, index: u64) -> Option<Packet> {
    let mut json = JsonBuf::new();
    let flags = if r.mic == Mic::Repeat {
        PacketFlags::NONE
    } else {
        PacketFlags::CRC_OK
    };
    r.to_json(&mut json).ok()?;
    Packet::new(index, flags, json.as_bytes()).ok()
}

/// The packet of an unknown train; `cut` when pairs of the train were dropped.
fn unknown_packet(
    train: &pulses::PulseTrain<'_>,
    index: u64,
    t_us: u64,
    center_hz: u32,
    cut: bool,
) -> Option<Packet> {
    let mut json = JsonBuf::new();
    let truncated = write_unknown(&mut json, train, t_us, center_hz);
    let flags = if truncated || cut {
        PacketFlags::TRUNCATED
    } else {
        PacketFlags::NONE
    };
    Packet::new(index, flags, json.as_bytes()).ok()
}

/// Looks `hash` up among the remembered readings; true if it is a copy (the reading is then
/// counted).
fn seen_before(seen: &mut [u64], hash: u64, t_us: u64, channel: i16, merge_us: u64) -> bool {
    for e in seen.chunks_exact_mut(WORDS) {
        let Ok::<&mut [u64; WORDS], _>([state, h, t, count]) = e.try_into() else {
            continue;
        };
        if *state & 1 == 0 || *h != hash {
            continue;
        }
        let ch = (*state >> 8) as u16 as i16;
        let limit = if ch == channel { merge_us } else { CROSS_US };
        if t.abs_diff(t_us) <= limit {
            *count += 1;
            return true;
        }
    }
    false
}

/// Remembers a delivered reading in the next slot of the ring.
fn remember(seen: &mut [u64], next: &mut usize, hash: u64, t_us: u64, channel: i16) {
    let slot = *next % SENSOR_SEEN;
    *next = (slot + 1) % SENSOR_SEEN;
    if let Some(Ok::<&mut [u64; WORDS], _>(e)) = seen
        .chunks_exact_mut(WORDS)
        .nth(slot)
        .map(TryInto::try_into)
    {
        *e = [1 | (u64::from(channel as u16) << 8), hash, t_us, 1];
    }
}

/// Writes the JSON object of an unknown train; true when pairs were left out.
fn write_unknown(
    json: &mut JsonBuf,
    train: &pulses::PulseTrain<'_>,
    t_us: u64,
    center_hz: u32,
) -> bool {
    let h = train.header();
    let f = (f64::from(center_hz) + f64::from(h.freq_offset_hz)) / 1.0e6;
    let rssi = f32::from(h.level_db10) / 10.0 + 0.0;
    let _ = write!(
        json,
        "{{\"time\":{:.3},\"model\":\"unknown\",\"mod\":\"{}\",\"freq_MHz\":{f:.4},\
         \"rssi_dB\":{rssi:.1},\"rate_hz\":{},\"pairs\":{},\"pulses\":[",
        t_us as f64 / 1.0e6,
        h.modulation.as_str(),
        h.rate_hz,
        train.len()
    );
    let mut truncated = false;
    for (k, p) in train.pairs().enumerate() {
        let mut piece = JsonBuf::new();
        let _ = write!(
            piece,
            "{}[{},{}]",
            if k > 0 { "," } else { "" },
            p.pulse,
            p.gap
        );
        // Leave room for the closing "]}".
        if json.len() + piece.len() + 2 > JsonBuf::CAPACITY {
            truncated = true;
            break;
        }
        if let Ok(s) = core::str::from_utf8(piece.as_bytes()) {
            let _ = json.write_str(s);
        }
    }
    let _ = json.write_str("]}");
    truncated
}

impl<S: Storage> Block for SensorDecodeBlock<S> {
    type In = Packets;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[Packet], output: &mut [Packet]) -> Result<Progress> {
        let mut consumed = 0;
        let mut produced = 0;
        for packet in input {
            let last = !packet.flags.contains(CONTINUED);
            // A finished train gives up to MAX_READINGS packets; the block waits for room.
            if last && output.len().saturating_sub(produced) < MAX_READINGS {
                break;
            }
            consumed += 1;
            if self.assemble(packet) {
                let room = output.get_mut(produced..).unwrap_or(&mut []);
                produced += self.finish(packet.index, room);
            }
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let r = Ratio::new(MAX_READINGS as u64, 1).unwrap_or(Ratio::ONE);
        BlockTiming {
            ratio: r,
            kind: RatioKind::Packets { max_per_input: r },
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.len = 0;
        self.open = None;
        self.cut = false;
        self.next = 0;
        self.seen.as_mut().fill(0);
        self.merged = 0;
        self.unknown = 0;
        self.invalid = 0;
        self.lost = 0;
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

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        match index {
            0 => Some(("merged", self.merged as f64)),
            1 => Some(("unknown", self.unknown as f64)),
            2 => Some(("invalid", self.invalid as f64)),
            3 => Some(("lost", self.lost as f64)),
            _ => None,
        }
    }
}
