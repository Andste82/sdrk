//! `pulse_key`: pulse train packets to the on-off keying envelope at the rate of a clock
//! stream (M46 extension, D1124): the keying of a transmitter. The clock (port a) only paces
//! the block (its rate and its sample count matter, as with `rds_encode`), the trains of
//! `sensor_encode` arrive at port b, the output is the envelope for `ook_mod`.

use sdrk::{Progress, Rate};
use sdrk_proto::Packet;
use sdrk_proto::pulses::{self, CONTINUED, Modulation, Pair};

use super::{head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::{Clocked, Packets};
use crate::storage::{Plan, Storage};

/// Trains the block holds at once, the one being sent included (D1129).
pub const PULSE_KEY_QUEUE: usize = 4;

/// Pairs of the longest train (the capacity of `sensor_decode` and `sensor_encode`); a longer
/// train is dropped whole and counted in `dropped_trains`.
pub const PULSE_KEY_PAIRS: usize = 1200;

/// The envelope while the carrier is on.
pub const PULSE_KEY_ON: f32 = 1.0;
/// The envelope while the carrier is off (bipolar: `ook_mod` keys at a value above 0, and a
/// resampler behind it rings at a step from 0, D1129).
pub const PULSE_KEY_OFF: f32 = -1.0;

/// Description of the `pulse_key` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "pulse_key",
    summary: "Keys a carrier: pulse train packets to the on-off envelope (+1 on, -1 off) at the \
        rate of a clock stream.",
    details: "Port a is the clock: any f32 stream at the rate of the output (for a transmitter \
        the audio input of the flow); its values are ignored and one output sample is produced \
        per clock sample. Port b takes the pulse train packets of sensor_encode (a train longer \
        than 30 pairs is several packets with the continued flag; only on-off keying trains \
        count, any other packet is ignored and counted in ignored_packets). The output is +1.0 \
        while the carrier is on (a pulse), -1.0 while it is off (a gap, between trains and \
        before the first one) with delay 0: the output is bipolar because ook_mod keys at a \
        value above 0 and the resampler in front of a transmitter rings at a step from 0. The \
        widths are converted from the sample rate of the train header to the clock rate with \
        the cumulative edges rounded half up (edge = round(sum of the widths * clock rate / \
        header rate)), so the length of a train is exact and does not drift; a width shorter \
        than half a clock sample can collapse to 0 samples (at 48 kHz: a pulse under 10 us). \
        After each train the carrier stays off for guard_us so that two trains are never one \
        pulse. A train that arrives while another is sent waits in a queue of 4 trains (the one \
        being sent included); a further train is dropped and counted in dropped_trains, so is \
        a train whose packets are cut by a packet of another train and a train of more than \
        1200 pairs (it is never sent shortened). Packets are taken at the \
        start of every call and a complete train starts with the first sample of that call, so \
        the start of a train depends on which clock call first sees it and the envelope has \
        the same shape for every chunking. Cost: 2 per clock sample.",
    inputs: &[
        PortDesc {
            name: "clock",
            types: &[PortType::F32],
            description: "Any f32 stream at the output rate; only its sample count is used.",
        },
        PortDesc {
            name: "trains",
            types: &[PortType::Packets],
            description: "The pulse train packets of sensor_encode.",
        },
    ],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The envelope: +1.0 carrier on, -1.0 off, at the clock rate, for ook_mod.",
    }],
    params: &[ParamDesc {
        name: "guard_us",
        unit: Unit::Microseconds,
        kind: ParamKind::I64 {
            min: 0,
            max: 1_000_000,
            default: Some(20_000),
        },
        typical: "20000; longer than the repeat gap of the devices (8000)",
        mutability: Mutability::Build,
        description: "The carrier stays off this long after every train.",
    }],
    required: &[],
};

/// Configuration of [`PulseKeyBlock`].
///
/// ```
/// use sdrk_blocks::{PulseKeyConfig, ParamValue};
/// let mut cfg = PulseKeyConfig::default();
/// assert_eq!(cfg.guard_us, 20_000);
/// cfg.set("guard_us", ParamValue::I64(0))?;
/// assert!(cfg.set("guard_us", ParamValue::I64(-1)).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PulseKeyConfig {
    /// The carrier stays off this long after every train, in microseconds.
    pub guard_us: u32,
}

impl Default for PulseKeyConfig {
    fn default() -> Self {
        Self { guard_us: 20_000 }
    }
}

impl PulseKeyConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` of `guard_us`.
    pub fn validate(&self) -> Result<()> {
        for p in DESCRIPTION.params {
            p.check(&self.get(p.name)?)?;
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
            "guard_us" => {
                self.guard_us =
                    u32::try_from(value.as_i64()?).map_err(|_| Error::WrongParamType)?;
            }
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
            "guard_us" => Ok(ParamValue::I64(i64::from(self.guard_us))),
            _ => unknown(),
        }
    }
}

/// One train of the queue.
#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    /// Pairs stored.
    len: usize,
    /// Sample rate of the widths.
    rate_hz: u32,
    /// Index of the packets of the train.
    index: u64,
}

/// The train being sent: the position and the cumulative edges.
#[derive(Debug, Clone, Copy)]
struct Play {
    /// The pair being sent.
    pair: usize,
    /// Whether its gap (not its pulse) is being sent.
    in_gap: bool,
    /// End of the segment in samples of the header rate since the train start.
    t_end: u64,
    /// Clock samples sent since the train start.
    emitted: u64,
}

/// Pulse key: a clock stream (port a, values ignored) with pulse train packets (port b) in,
/// the envelope (`f32`, +1 on and -1 off, one sample per clock sample) out.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, PulseKeyBlock, PulseKeyConfig};
/// use sdrk_proto::PacketFlags;
/// use sdrk_proto::pulses::{self, Header, Modulation, Pair};
/// let cfg = PulseKeyConfig { guard_us: 0 };
/// let rate = Rate::hz(48_000)?;
/// let mut mem = vec![0u8; PulseKeyBlock::<ArenaStorage<'_>>::storage_bytes(&cfg)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut key = PulseKeyBlock::new(&cfg, rate, &mut arena)?;
/// // 250 us on, 500 us off at 1 MHz: 12 samples on, 24 off at 48 kHz.
/// let header = Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: 0,
///     level_db10: 0, noise_db10: 0, rate_hz: 1_000_000 };
/// let train = pulses::encode_packet(0, PacketFlags::NONE, &header,
///     &[Pair { pulse: 250, gap: 500 }])?;
/// let clock = [0.0f32; 48];
/// let mut out = [0.0f32; 48];
/// let p = key.process((&clock, &[train]), &mut out)?;
/// assert_eq!((p.consumed, p.produced, key.consumed_packets()), (48, 48, 1));
/// assert_eq!(out.iter().filter(|&&y| y > 0.0).count(), 12);
/// assert!(out[..12].iter().all(|&y| y == 1.0) && out[12..].iter().all(|&y| y == -1.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PulseKeyBlock<S: Storage> {
    config: PulseKeyConfig,
    /// Clock rate as a ratio of samples per second.
    num: u64,
    den: u64,
    /// The pairs of the queue, [`PULSE_KEY_PAIRS`] per slot.
    pairs: S::Buf<Pair>,
    slots: [Slot; PULSE_KEY_QUEUE],
    /// The oldest slot of the queue and the slots in use (the last one is still filling when
    /// `filling`).
    head: usize,
    count: usize,
    filling: bool,
    /// The packets of a dropped train are skipped up to its last one.
    dropping: bool,
    play: Option<Play>,
    /// Clock samples of guard left after a train.
    guard_left: u64,
    guard_samples: u64,
    taken: usize,
    ignored: u64,
    dropped: u64,
    sent: u64,
}

impl<S: Storage> core::fmt::Debug for PulseKeyBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PulseKeyBlock")
            .field("config", &self.config)
            .field("queued", &self.count)
            .field("sent", &self.sent)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> PulseKeyBlock<S> {
    /// Creates the block for a clock at `rate_in`.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration, `Storage` if `storage` is too small.
    pub fn new(config: &PulseKeyConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let pairs = storage.alloc::<Pair>(PULSE_KEY_QUEUE * PULSE_KEY_PAIRS)?;
        let (num, den) = (rate_in.num(), rate_in.den());
        let guard = u128::from(config.guard_us) * u128::from(num);
        let unit = u128::from(den) * 1_000_000;
        Ok(Self {
            config: *config,
            num,
            den,
            pairs,
            slots: [Slot::default(); PULSE_KEY_QUEUE],
            head: 0,
            count: 0,
            filling: false,
            dropping: false,
            play: None,
            guard_left: 0,
            guard_samples: u64::try_from((2 * guard + unit) / (2 * unit)).unwrap_or(u64::MAX),
            taken: 0,
            ignored: 0,
            dropped: 0,
            sent: 0,
        })
    }

    /// Exact bytes of storage the block needs.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &PulseKeyConfig) -> Result<usize> {
        config.validate()?;
        let mut plan = Plan::default();
        plan.add::<Pair>(PULSE_KEY_QUEUE * PULSE_KEY_PAIRS)?;
        Ok(plan.bytes())
    }

    /// Output rate for the configuration (equal to the clock rate).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn rate_out(config: &PulseKeyConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in)
    }

    /// Packets that were not a pulse train of on-off keying.
    pub fn ignored_packets(&self) -> u64 {
        self.ignored
    }

    /// Trains dropped (queue full, or cut by the packets of another train).
    pub fn dropped_trains(&self) -> u64 {
        self.dropped
    }

    /// Trains sent completely.
    pub fn trains_sent(&self) -> u64 {
        self.sent
    }

    /// Trains in the queue, the one being sent included.
    pub fn queued_trains(&self) -> usize {
        self.count - usize::from(self.filling)
    }

    /// Whether a train is being sent.
    pub fn active(&self) -> bool {
        self.play.is_some()
    }

    fn slot_of(&self, k: usize) -> usize {
        (self.head + k) % PULSE_KEY_QUEUE
    }

    /// Takes every packet: joins the trains in the queue.
    fn take(&mut self, packets: &[Packet]) {
        for p in packets {
            let train = match pulses::decode(p.data()) {
                Ok(t) if t.header().modulation == Modulation::Ook => t,
                _ => {
                    self.ignored += 1;
                    continue;
                }
            };
            let header = train.header();
            let last = !p.flags.contains(CONTINUED);
            if self.dropping {
                self.dropping = !last;
                continue;
            }
            let tail = self.slot_of(self.count.saturating_sub(1));
            let same = self.filling
                && self
                    .slots
                    .get(tail)
                    .is_some_and(|s| s.index == p.index && s.rate_hz == header.rate_hz);
            if self.filling && !same {
                // The packets of another train cut this one: it is lost.
                self.count -= 1;
                self.filling = false;
                self.dropped += 1;
            }
            if !self.filling {
                if self.count >= PULSE_KEY_QUEUE {
                    self.dropped += 1;
                    self.dropping = !last;
                    continue;
                }
                let k = self.slot_of(self.count);
                if let Some(s) = self.slots.get_mut(k) {
                    *s = Slot {
                        len: 0,
                        rate_hz: header.rate_hz,
                        index: p.index,
                    };
                }
                self.count += 1;
                self.filling = true;
            }
            let k = self.slot_of(self.count - 1);
            let Some(slot) = self.slots.get_mut(k) else {
                continue;
            };
            let base = k * PULSE_KEY_PAIRS;
            let mut overflow = false;
            for pair in train.pairs() {
                if slot.len >= PULSE_KEY_PAIRS {
                    overflow = true;
                    break;
                }
                if let Some(dst) = self.pairs.as_mut().get_mut(base + slot.len) {
                    *dst = pair;
                    slot.len += 1;
                }
            }
            if overflow {
                // A train longer than the slot is never sent shortened: it is dropped whole
                // (the slot is the last one of the queue) and the rest of it is skipped.
                self.count -= 1;
                self.filling = false;
                self.dropped += 1;
                self.dropping = !last;
                continue;
            }
            if last {
                self.filling = false;
            }
        }
    }

    /// The clock sample of the cumulative width `t` of the header rate (half up).
    fn edge(&self, t: u64, rate_hz: u32) -> u64 {
        let unit = u128::from(self.den) * u128::from(rate_hz.max(1));
        let v = (2 * u128::from(t) * u128::from(self.num) + unit) / (2 * unit);
        u64::try_from(v).unwrap_or(u64::MAX)
    }

    /// The pair `i` of the head train.
    fn pair(&self, i: usize) -> Pair {
        let base = self.head * PULSE_KEY_PAIRS;
        self.pairs
            .as_ref()
            .get(base + i)
            .copied()
            .unwrap_or_default()
    }

    /// Writes the envelope into `out`.
    fn render(&mut self, out: &mut [f32]) {
        let mut n = 0;
        while n < out.len() {
            let room = (out.len() - n) as u64;
            if self.guard_left > 0 {
                let k = self.guard_left.min(room);
                fill(out, &mut n, k, PULSE_KEY_OFF);
                self.guard_left -= k;
                continue;
            }
            let Some(mut play) = self.play else {
                // The head train starts when it is complete.
                if self.queued_trains() > 0 {
                    let first = self.pair(0).pulse;
                    self.play = Some(Play {
                        pair: 0,
                        in_gap: false,
                        t_end: u64::from(first),
                        emitted: 0,
                    });
                    continue;
                }
                let k = room;
                fill(out, &mut n, k, PULSE_KEY_OFF);
                continue;
            };
            let Some(slot) = self.slots.get(self.head).copied() else {
                return;
            };
            let end = self.edge(play.t_end, slot.rate_hz);
            let k = end.saturating_sub(play.emitted).min(room);
            let level = if play.in_gap {
                PULSE_KEY_OFF
            } else {
                PULSE_KEY_ON
            };
            fill(out, &mut n, k, level);
            play.emitted += k;
            self.play = Some(play);
            if play.emitted < end {
                continue;
            }
            // The segment is finished: the next one.
            if !play.in_gap {
                play.in_gap = true;
                play.t_end += u64::from(self.pair(play.pair).gap);
            } else if play.pair + 1 >= slot.len {
                self.head = (self.head + 1) % PULSE_KEY_QUEUE;
                self.count -= 1;
                self.play = None;
                self.sent += 1;
                self.guard_left = self.guard_samples;
                continue;
            } else {
                play.pair += 1;
                play.in_gap = false;
                play.t_end += u64::from(self.pair(play.pair).pulse);
            }
            self.play = Some(play);
        }
    }
}

/// Writes `k` samples of `level` at `out[*n..]` and advances `*n`.
fn fill(out: &mut [f32], n: &mut usize, k: u64, level: f32) {
    let k = usize::try_from(k).unwrap_or(usize::MAX);
    if let Some(dst) = out.get_mut(*n..) {
        let k = k.min(dst.len());
        if let Some(d) = dst.get_mut(..k) {
            d.fill(level);
        }
        *n += k;
    }
}

impl<S: Storage> Block for PulseKeyBlock<S> {
    type In = Clocked<f32, Packets>;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: (&[f32], &[Packet]), output: &mut [f32]) -> Result<Progress> {
        let (clock, packets) = input;
        self.taken = packets.len();
        self.take(packets);
        let n = clock.len().min(output.len());
        self.render(head_mut(output, n)?);
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    /// Drops every train and the guard; the counters are 0.
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.filling = false;
        self.dropping = false;
        self.play = None;
        self.guard_left = 0;
        self.taken = 0;
        self.ignored = 0;
        self.dropped = 0;
        self.sent = 0;
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

    fn consumed_packets(&self) -> usize {
        self.taken
    }

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        match index {
            0 => Some(("ignored_packets", self.ignored as f64)),
            1 => Some(("dropped_trains", self.dropped as f64)),
            2 => Some(("trains_sent", self.sent as f64)),
            3 => Some(("queued_trains", self.queued_trains() as f64)),
            4 => Some(("active", f64::from(u8::from(self.play.is_some())))),
            _ => None,
        }
    }
}
