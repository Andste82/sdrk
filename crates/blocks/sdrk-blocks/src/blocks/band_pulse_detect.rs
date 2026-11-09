//! `band_pulse_detect`: the pulse trains of every short-range transmission in a band, from the
//! lanes of a channelizer (M45.1 §3). Per lane: the magnitude (kernel `mag`, one call over all
//! lanes of a piece), a first-order low-pass of the envelope, and a pulse detector after the
//! method of `rtl_433`'s `pulse_detect_package` (implemented from its description, clean room):
//! noise and signal level tracking, a threshold midway with hysteresis, a lead-in, glitch
//! removal and the end of a burst by a long gap; for FSK the instantaneous frequency of the
//! channel sliced between two tracked levels. Bursts of neighbouring channels that are one
//! transmission are merged.

// Signal-processing code with the usual one-letter names (x, e, p, k, ...).
#![allow(clippy::many_single_char_names)]

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::num::flush_denormal;
use sdrk_proto::pulses::{CONTINUED, Header, MAX_PAIRS, Modulation, Pair};
use sdrk_proto::{Packet, PacketFlags};

use super::{HZ_LIMIT, check_count, hz, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, LaneElem, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::{Lanes, LanesRef, Packets};
use crate::storage::{Plan, Storage};

/// The smallest step rate of the input in Hz (`rate_unsupported` below; spec §3).
pub const BAND_PULSE_MIN_RATE_HZ: f64 = 100_000.0;

/// Time steps per call of the magnitude kernel at most (the size of the envelope buffer is
/// this many steps of all lanes).
pub const BAND_PULSE_CHUNK: usize = 64;

/// Steps of the lead-in of a burst: after a lane triggers, the envelope of this many steps
/// sets the signal level before the first pulse is measured (D957). With `min_pulse_us`
/// shorter than 5 steps the lead-in is `4 min_steps - 1` steps, so that its replay finishes at
/// most one FSK frame and one OOK train (D971).
pub const BAND_PULSE_LEAD: usize = 16;

/// Steps after construction or reset in which every lane only learns its noise level (a
/// running mean) and no burst starts (D957).
pub const BAND_PULSE_WARMUP: u64 = 1024;

/// The most lanes of the input.
pub const BAND_PULSE_MAX_LANES: usize = 1024;

/// Largest `max_pulses`.
const MAX_PULSES: usize = 65_536;

/// Cutoff of the envelope low-pass as a fraction of the step rate (D956).
const ENV_CUTOFF: f64 = 1.0 / 8.0;

/// Step of the noise level per idle sample (`rtl_433`'s rate, spec §3).
const NOISE_RATIO: f32 = 1.0 / 1024.0;

/// Step of the signal level per pulse sample (spec §3).
const HIGH_RATIO: f32 = 1.0 / 64.0;

/// Half width of the hysteresis as a fraction of the threshold: a pulse starts above
/// `t (1 + 1/8)` and ends below `t (1 - 1/8)`, `t` midway between the noise and the signal
/// level (D956).
const HYSTERESIS: f32 = 1.0 / 8.0;

/// The noise level never falls below this envelope (-140 dBFS).
const NOISE_FLOOR: f32 = 1e-7;

/// Largest envelope value (larger values, also infinity, are clamped; NaN counts as 0).
const LEVEL_LIMIT: f32 = 1e18;

/// The smallest envelope that is not flushed to 0 (R6.3).
const FLUSH: f32 = 1e-30;

/// A pulse longer than this (seconds) is a carrier, not a pulse (D958).
const LONG_PULSE_S: f64 = 0.1;

/// The first frequency transition of an FSK frame: the instantaneous frequency leaves the
/// mean of the first run by more than this many cycles per step (D959).
const FSK_DELTA_CYCLES: f32 = 0.03;

/// The two tracked FSK levels of an FSK frame are at least this many cycles per step apart
/// (D959).
const FSK_SPLIT_CYCLES: f32 = 0.06;

/// Step of the two FSK levels per sample (D959).
const FSK_RATIO: f32 = 1.0 / 16.0;

/// Half width of the hysteresis of the FSK slicer as a fraction of the distance of the two
/// levels: the same 1/16 of the swing as the envelope threshold, so both edges of a frequency
/// run are delayed by the low-pass as much as the edges of the envelope (D959).
const FSK_HYSTERESIS: f32 = 1.0 / 16.0;

/// Fewest frequency runs of an FSK frame in `auto` (D959).
const FSK_MIN_RUNS: u32 = 4;

/// The envelope edge (midpoint) replaces the presence edge of an FSK frame when it is at most
/// this many steps inside the frame (D959).
const EDGE_WINDOW: f64 = 4.0;

/// The most samples a frequency run needs to count (and the length of the candidate buffer
/// of the first transition).
const FSK_CAND: usize = 8;

/// Raw magnitudes kept per busy lane for the second stage of the envelope edges (a power of 2,
/// at least `BAND_PULSE_LEAD + HIST`; D1023).
const RING: usize = 32;

/// Steps of raw magnitudes before the current one that the block keeps for every lane (the
/// rows before a piece in the magnitude buffer), so a lane that triggers knows the samples
/// before its lead-in (D1023).
const HIST: usize = 8;

/// The level of the second stage of an edge is the magnitude of half the signal amplitude
/// over the noise: `sqrt(high^2 / 4 + (3/4) (4/pi) noise^2)`, with `high^2` the power of the
/// plateau (signal and noise) and `(4/pi) noise^2` the noise power (`noise` is the mean
/// magnitude of the noise, Rayleigh); so the midpoint of a pulse edge is not biased towards
/// the plateau at low levels (D1023).
const NOISE_POWER: f32 = 0.75 * 4.0 / core::f32::consts::PI;

/// Samples before the envelope crossing of an edge in which the raw edge is searched (D1023).
const REFINE_BACK: u64 = 6;

/// The length of a block of the wideband rule in seconds (D1029): every block of this many
/// steps (at least `WIDE_MIN_BLOCK`) is classified as wideband or not when it ends.
const WIDE_BLOCK_S: f64 = 100e-6;
const WIDE_MIN_BLOCK: u32 = 8;

/// A lane is active in a block when its envelope is above the trigger level (`min_snr_db`
/// above its noise) in at least this share of the steps of the block (D1029).
const WIDE_ACTIVE_SHARE: u32 = 4;

/// The fewest lanes for wideband events (D1029): with fewer lanes the rule is off.
const WIDE_MIN_WIDTH: usize = 4;

/// The names of the `modulation` choice, in index order.
const MODULATIONS: &[&str] = &["ook", "fsk", "auto"];

/// Lane modes.
const IDLE: u32 = 0;
const LEAD: u32 = 1;
const BURST: u32 = 2;

/// States of an output record.
const OUT_NONE: u32 = 0;
const OUT_DECIDE: u32 = 1;
const OUT_EMIT: u32 = 2;

/// Description of the `band_pulse_detect` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "band_pulse_detect",
    summary: "Detects short-range transmissions in every lane of a channelizer and emits their pulse trains as packets.",
    details: "Reads the lane stream of a channelize node (M lanes at the step rate 2 fs / M, \
        at least 100 kHz) and runs a pulse detector after the method of rtl_433 in every lane. \
        Per lane value: the magnitude (mag) and a one-pole low-pass of the envelope with the \
        cutoff 1/8 of the step rate; while a lane is idle its noise level follows the envelope \
        with the step 1/1024 per sample (the first 1024 steps after construction or reset \
        only learn it, as a running mean), and a burst starts when the envelope is min_snr_db \
        above the noise. A lead-in of 16 steps sets the signal level (their largest \
        envelope); then the threshold t lies midway between the noise and the signal level, a \
        pulse starts above t (1 + 1/8) and ends below t (1 - 1/8), and the signal level \
        follows the pulses with the step 1/64. Every edge is then refined on the raw magnitudes \
        around it: the change point of a step at half the signal amplitude over the noise, \
        interpolated between two samples; widths are rounded to whole steps. Pulses \
        and gaps shorter than min_pulse_us are glitches and are merged into their neighbours. \
        A burst ends after a gap longer than max_gap_ms or gap_ratio times its longest pulse; \
        a pulse longer than 100 ms is a carrier (the lane takes its level as the noise). For \
        FSK (modulation fsk or auto) a second run machine marks the presence of a carrier \
        (the envelope min_snr_db above the noise), and during a presence the instantaneous \
        frequency (the angle of the low-passed product of a sample and the conjugate of the \
        one before, the product scaled by a power of two near the inverse square of the \
        envelope, at most 12 dB \
        below the signal level) is sliced between two tracked levels (step 1/16, hysteresis \
        1/16 of their distance) into upper and lower runs (runs shorter than half of \
        min_pulse_us, 2 to 8 steps, are merged); each presence is one FSK frame, and with \
        auto it is an FSK frame when it has at least 4 runs and the two levels are at least \
        0.06 of the step rate apart (its OOK pulses are then dropped). The transitions of a \
        frame with at least 5 of them are fitted with the grid of its symbol clock (least \
        squares), its start and end moved to whole symbols before the first and after the \
        last transition, and its widths rounded as times after its start. Every burst is \
        one train of \
        sdrk_proto::pulses (widths in steps of the input, a train of more than 30 pairs is \
        split over packets flagged CONTINUED; the last gap is 0); the packet index is the \
        step of the first pulse since construction or reset. A transmission seen in the \
        neighbouring channels c and c + 1 (signed channel numbers; from 4 lanes on the lane \
        -M/2, centred on the edge of the input band, also borders M/2 - 1) is emitted once: \
        two bursts that overlap by more than half of the shorter one and whose frequencies \
        are less than a channel spacing apart (modulo the input rate) are one transmission, \
        and only the one with the higher level is emitted; an FSK frame that spans at least \
        half of an OOK train of the same transmission is kept instead of the train. Wideband \
        events: every 100 us block (at least 8 steps) in which all lanes that can hear it but \
        at most max(1, L / 8) had their envelope above the trigger level in a quarter of the \
        steps (L of at least 4 lanes and half of them; a lane that learned a carrier as its \
        noise cannot hear) is wideband, and wideband blocks at most one block apart are one \
        event; of the bursts that lie by more than half in an event (widened by a block) only \
        the first decided one is emitted. The frequency is c times the spacing (half the step rate) \
        plus the mean phase rotation of the pulses (FSK: the middle of the two levels). \
        Pitfall: the input must begin with about 4 ms of band noise (1024 steps), and a \
        carrier present from the start is learned as noise.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Lanes {
            elem: LaneElem::C32,
            width_param: "",
        }],
        description: "The lanes of a channelize node (any width), at a step rate of at least \
            100 kHz.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "One pulse train per transmission (sdrk_proto::pulses).",
    }],
    params: &[
        ParamDesc {
            name: "modulation",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: MODULATIONS,
                default: 2,
            },
            typical: "auto",
            mutability: Mutability::Build,
            description: "ook: envelope pulses only; fsk: every pulse is an FSK frame; auto: a \
                pulse with frequency runs is an FSK frame, the rest on-off keying.",
        },
        ParamDesc {
            name: "min_snr_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 60.0,
                default: Some(9.0),
            },
            typical: "9",
            mutability: Mutability::Runtime,
            description: "A burst starts when the envelope is this far above the noise level \
                (an amplitude ratio of 10^(min_snr_db / 20)); a change applies to the next \
                burst.",
        },
        ParamDesc {
            name: "min_pulse_us",
            unit: Unit::Microseconds,
            kind: ParamKind::F64 {
                min: 0.1,
                max: 100_000.0,
                default: Some(40.0),
            },
            typical: "40",
            mutability: Mutability::Build,
            description: "Pulses and gaps shorter than this are glitches (rounded to steps, at \
                least 1).",
        },
        ParamDesc {
            name: "max_gap_ms",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.01,
                max: 10_000.0,
                default: Some(10.0),
            },
            typical: "10",
            mutability: Mutability::Build,
            description: "A gap longer than this many milliseconds ends a burst.",
        },
        ParamDesc {
            name: "gap_ratio",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 1.0,
                max: 1000.0,
                default: Some(10.0),
            },
            typical: "10",
            mutability: Mutability::Build,
            description: "A gap longer than this many times the longest pulse of the burst \
                ends it.",
        },
        ParamDesc {
            name: "max_pulses",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_PULSES as i64,
                default: Some(1200),
            },
            typical: "1200",
            mutability: Mutability::Build,
            description: "Most pairs of a train; a longer burst is emitted flagged TRUNCATED.",
        },
        ParamDesc {
            name: "center_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: -HZ_LIMIT,
                max: HZ_LIMIT,
                default: Some(0.0),
            },
            typical: "433920000",
            mutability: Mutability::Build,
            description: "The RF centre of the band, metadata of the flow: it does not \
                change the packets, which carry the offset from it; a tool reads it from the \
                flow for the absolute frequency of the JSON form (pulses::write_json).",
        },
    ],
    required: &[],
};

/// The `modulation` parameter of [`BandPulseDetectConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PulseModulation {
    /// Envelope pulses only (on-off keying).
    Ook,
    /// Every pulse is an FSK frame.
    Fsk,
    /// A pulse with frequency runs is an FSK frame, the others on-off keying.
    Auto,
}

impl PulseModulation {
    fn choice(self) -> u16 {
        match self {
            Self::Ook => 0,
            Self::Fsk => 1,
            Self::Auto => 2,
        }
    }

    fn from_choice(c: u16) -> Self {
        match c {
            0 => Self::Ook,
            1 => Self::Fsk,
            _ => Self::Auto,
        }
    }
}

/// Configuration of [`BandPulseDetectBlock`].
///
/// ```
/// use sdrk_blocks::{BandPulseDetectConfig, PulseModulation};
/// let cfg = BandPulseDetectConfig::default();
/// assert_eq!(cfg.modulation, PulseModulation::Auto);
/// assert!(cfg.validate(sdrk::Rate::hz(250_000)?).is_ok());
/// assert!(cfg.validate(sdrk::Rate::hz(50_000)?).is_err()); // below 100 kHz
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandPulseDetectConfig {
    /// OOK, FSK or both.
    pub modulation: PulseModulation,
    /// Start of a burst above the noise in dB (runtime).
    pub min_snr_db: f64,
    /// Shortest pulse or gap in µs.
    pub min_pulse_us: f64,
    /// Longest gap inside a burst in ms.
    pub max_gap_ms: f64,
    /// Longest gap inside a burst in units of its longest pulse.
    pub gap_ratio: f64,
    /// Most pairs of a train.
    pub max_pulses: usize,
    /// RF centre of the band in Hz: metadata of the flow, it does not change the packets (a tool
    /// reads it from the flow and passes it to `pulses::write_json`, D976).
    pub center_hz: f64,
}

impl Default for BandPulseDetectConfig {
    /// `auto`, 9 dB, 40 µs, 10 ms, 10, 1200 pairs, centre 0.
    fn default() -> Self {
        Self {
            modulation: PulseModulation::Auto,
            min_snr_db: 9.0,
            min_pulse_us: 40.0,
            max_gap_ms: 10.0,
            gap_ratio: 10.0,
            max_pulses: 1200,
            center_hz: 0.0,
        }
    }
}

fn range(name: &'static str, v: f64) -> Result<()> {
    let p = DESCRIPTION
        .param(name)
        .ok_or(Error::Core(sdrk_core::Error::Internal("band_pulse_detect")))?;
    if let ParamKind::F64 { min, max, .. } = p.kind
        && !(v >= min && v <= max)
    {
        return Err(Error::OutOfRange {
            name,
            min,
            max,
            got: v,
        });
    }
    Ok(())
}

/// `round(v)` steps as a `u32`, at least 1 (saturating).
fn steps(v: f64) -> u32 {
    let r = libm::round(v);
    if r >= f64::from(u32::MAX) {
        u32::MAX
    } else if r >= 1.0 {
        r as u32
    } else {
        1
    }
}

impl BandPulseDetectConfig {
    /// Checks the configuration against the step rate of the input.
    ///
    /// # Errors
    /// `OutOfRange` for a parameter outside its range; `Core(Unsupported)` for a step rate
    /// below 100 kHz.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        range("min_snr_db", self.min_snr_db)?;
        range("min_pulse_us", self.min_pulse_us)?;
        range("max_gap_ms", self.max_gap_ms)?;
        range("gap_ratio", self.gap_ratio)?;
        range("center_hz", self.center_hz)?;
        check_count("max_pulses", self.max_pulses, 1, MAX_PULSES)?;
        if hz(rate_in) < BAND_PULSE_MIN_RATE_HZ {
            return Err(Error::Core(sdrk_core::Error::Unsupported(
                "band_pulse_detect needs a step rate of at least 100 kHz (the channel rate 2 fs / M of channelize)",
            )));
        }
        Ok(())
    }

    /// The trigger factor `10^(min_snr_db / 20)` on the envelope.
    pub fn snr_factor(&self) -> f32 {
        libm::pow(10.0, self.min_snr_db / 20.0) as f32
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "modulation" => self.modulation = PulseModulation::from_choice(value.as_choice()?),
            "min_snr_db" => self.min_snr_db = value.as_f64()?,
            "min_pulse_us" => self.min_pulse_us = value.as_f64()?,
            "max_gap_ms" => self.max_gap_ms = value.as_f64()?,
            "gap_ratio" => self.gap_ratio = value.as_f64()?,
            "max_pulses" => self.max_pulses = value.as_usize()?,
            "center_hz" => self.center_hz = value.as_f64()?,
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
            "modulation" => Ok(ParamValue::Choice(self.modulation.choice())),
            "min_snr_db" => Ok(ParamValue::F64(self.min_snr_db)),
            "min_pulse_us" => Ok(ParamValue::F64(self.min_pulse_us)),
            "max_gap_ms" => Ok(ParamValue::F64(self.max_gap_ms)),
            "gap_ratio" => Ok(ParamValue::F64(self.gap_ratio)),
            "max_pulses" => Ok(ParamValue::I64(self.max_pulses as i64)),
            "center_hz" => Ok(ParamValue::F64(self.center_hz)),
            _ => unknown(),
        }
    }
}

/// The derived constants of a block (all in steps of the input).
#[derive(Debug, Clone, Copy)]
struct Params {
    modulation: PulseModulation,
    /// Envelope (and FSK product) low-pass step.
    alpha: f32,
    /// Shortest pulse or gap.
    min_steps: u32,
    /// Longest gap.
    gap_steps: u32,
    gap_ratio: f32,
    /// Longest gap before the first pulse of a burst (the lead-in found no pulse).
    pre_timeout: u32,
    /// Longest pulse.
    long_steps: u32,
    /// Shortest frequency run.
    fsk_min: u32,
    max_pulses: u32,
    /// `2 pi FSK_DELTA_CYCLES` and `2 pi FSK_SPLIT_CYCLES` in radians per step.
    fsk_delta: f32,
    fsk_split: f32,
    /// Step rate in Hz.
    rate_hz: f64,
    /// The trigger factor `10^(min_snr_db / 20)` (runtime; a lane copies it when it
    /// triggers).
    factor: f32,
    /// Steps of the lead-in: `BAND_PULSE_LEAD`, at most `4 min_steps - 1` (D971).
    lead_len: u32,
    /// Steps of a block of the wideband rule, and the active steps that make a lane active
    /// in it (D1029).
    wide_block: u32,
    wide_active: u32,
}

impl Params {
    fn new(config: &BandPulseDetectConfig, rate_in: Rate) -> Result<Self> {
        config.validate(rate_in)?;
        let fs = hz(rate_in);
        let min_steps = steps(config.min_pulse_us * 1e-6 * fs);
        let gap_steps = steps(config.max_gap_ms * 1e-3 * fs);
        let pre = steps(config.gap_ratio * f64::from(min_steps)).min(gap_steps);
        let fsk_min = (min_steps / 2).clamp(2, FSK_CAND as u32);
        let tau = core::f64::consts::TAU;
        Ok(Self {
            modulation: config.modulation,
            alpha: (1.0 - libm::exp(-tau * ENV_CUTOFF)) as f32,
            min_steps,
            gap_steps,
            gap_ratio: config.gap_ratio as f32,
            pre_timeout: pre,
            long_steps: steps(LONG_PULSE_S * fs),
            fsk_min,
            max_pulses: config.max_pulses as u32,
            fsk_delta: (tau * f64::from(FSK_DELTA_CYCLES)) as f32,
            fsk_split: (tau * f64::from(FSK_SPLIT_CYCLES)) as f32,
            rate_hz: fs,
            factor: config.snr_factor(),
            lead_len: lead_len(min_steps),
            wide_block: steps(WIDE_BLOCK_S * fs).max(WIDE_MIN_BLOCK),
            wide_active: (steps(WIDE_BLOCK_S * fs).max(WIDE_MIN_BLOCK) / WIDE_ACTIVE_SHARE).max(1),
        })
    }
}

/// The steps of the lead-in for the shortest run `min_steps` (D971): a finished FSK frame
/// needs a presence and an absence of `min_steps` each, so two of them (or two OOK trains cut by
/// them) need `4 min_steps` steps and a lead-in of `4 min_steps - 1` steps finishes at most one
/// of each kind in its replay; its decision follows the replay.
fn lead_len(min_steps: u32) -> u32 {
    (BAND_PULSE_LEAD as u32).min(min_steps.saturating_mul(4).saturating_sub(1))
}

/// A burst that has ended and waits for its decision or its packets.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Out {
    /// Packet index (step of the first pulse).
    index: u64,
    /// One past the last pulse.
    end: u64,
    /// `OUT_NONE`, `OUT_DECIDE` or `OUT_EMIT`.
    state: u32,
    flags: u32,
    /// Pairs of the train.
    pairs: u32,
    /// Pairs already sent.
    sent: u32,
    /// Signal level (envelope).
    level: f32,
    /// Offset of the frequency from the channel centre in Hz.
    freq: f32,
    /// 0 OOK, 1 FSK.
    fsk: u32,
    _pad: u32,
}

/// A burst that a lane emitted: its span, its frequency (Hz), whether it was an FSK frame.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Emitted {
    start: u64,
    end: u64,
    freq: f32,
    fsk: u32,
    valid: u32,
    _pad: u32,
}

/// The state of one lane.
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Lane {
    /// The two output records (0 the OOK train, 1 the FSK frame).
    out: [Out; 2],
    /// The last FSK frame this lane emitted: an OOK train of a neighbour that shows one tone
    /// of it is decided when its burst ends, which may be after the next transmission
    /// (D1026).
    frame: Emitted,
    /// Step of the first lead-in sample.
    lead_at: u64,
    /// Step of the envelope crossing of the edge that waits for its refinement.
    ref_idx: u64,
    /// Step of the first pulse of the OOK train.
    start: u64,
    /// One past the last committed pulse (rounded edge).
    pulse_end: u64,
    /// The last burst decided in this lane (emitted or not): span.
    sum_start: u64,
    sum_end: u64,
    /// Times (in steps, interpolated between samples) of the start of the current envelope
    /// run, of the previous run and of the current pulse.
    t_run: f64,
    t_prev: f64,
    /// The envelope crossings (first stage) of the current and of the previous run: the
    /// positions of the bursts (the packet index, the spans of the merging) and the FSK frames
    /// keep these, delayed by the low-pass as much as the frequency runs; only the widths come
    /// from the refined edges (D1023).
    c_run: f64,
    c_prev: f64,
    /// Times of the start of the current and of the previous presence run.
    p_t_run: f64,
    p_t_prev: f64,
    /// Times of the start of the current and of the previous frequency run.
    ft_run: f64,
    ft_prev: f64,
    /// The start of the first committed frequency run of the frame (the origin of the run
    /// ends kept in the FSK pairs until the frame ends) and the end of the last one (D1025).
    f_origin: f64,
    f_last: f64,
    /// The time of the last restart of the first level of a frame (D1025).
    t_early: f64,
    /// The last burst decided in this lane: level, frequency (Hz).
    sum_level: f32,
    sum_freq: f32,
    sum_valid: u32,
    sum_emitted: u32,
    /// `IDLE`, `LEAD` or `BURST`.
    mode: u32,
    lead_n: u32,
    /// The signal level and the envelope of the sample before.
    high: f32,
    e_prev: f32,
    /// The trigger factor of the presence machine, copied when the lane triggered (a runtime
    /// change of `min_snr_db` applies to the next burst, D972).
    p_factor: f32,
    /// Whether the edge of the current envelope run waits for its refinement.
    ref_pending: u32,
    /// Envelope run: level (0 off, 1 on), length in samples, confirmed (at least
    /// `min_steps`), length of the previous run.
    level: u32,
    run_len: u32,
    confirmed: u32,
    prev_len: u32,
    /// Presence run: level, length, confirmed, length of the previous run.
    p_level: u32,
    p_run: u32,
    p_confirmed: u32,
    p_prev: u32,
    /// Whether the burst has a confirmed pulse.
    started: u32,
    /// Pairs of the OOK train and whether its last pulse waits for its gap.
    ook_n: u32,
    open: u32,
    longest: u32,
    truncated: u32,
    /// FSK frames of this burst.
    frames: u32,
    /// Sum of the phase rotation of the train's pulses and of the current pulse.
    acc: [f32; 2],
    pac: [f32; 2],
    /// FSK: the low-passed product, the mean of the first run, the two levels, the frequency
    /// of the sample before and of the sample before the first candidate.
    fm: [f32; 2],
    f_ref: f32,
    f_hi: f32,
    f_lo: f32,
    f_prev: f32,
    cand_before: f32,
    /// The mean frequency before the last restart of the first level, and whether there was
    /// one: a first run too short to set the first level (D1025).
    f_early: f32,
    early: u32,
    /// 0 before the first transition, 1 after.
    fsk_phase: u32,
    fsk_level: u32,
    fsk_run: u32,
    fsk_confirmed: u32,
    fsk_prev: u32,
    fsk_has_prev: u32,
    fsk_open: u32,
    fsk_n: u32,
    fsk_runs: u32,
    fsk_cand: u32,
    fsk_cnt: u32,
    fsk_full: u32,
    /// Whether the last decided burst was an FSK frame.
    sum_fsk: u32,
    /// Frequencies of the candidate samples of the first transition.
    cand_f: [f32; FSK_CAND],
    /// The raw magnitudes of the last `RING` samples of a burst and of the `HIST - 1` before
    /// its lead-in (`ring[idx % RING]`), for the refinement of the envelope edges (D1023).
    ring: [f32; RING],
    /// The envelope of the lead-in and its samples (`lead_x[0]` the sample before).
    lead_e: [f32; BAND_PULSE_LEAD],
    lead_x: [C32; BAND_PULSE_LEAD + 1],
}

/// What a lane step asks the block to do with the per-lane arrays of the idle lanes.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Next {
    /// Nothing.
    Stay,
    /// The lane is idle again, keeping its noise level.
    Idle,
    /// The lane is idle again with this noise level (a carrier).
    IdleAt(f32),
}

/// The pairs of the OOK train and of the FSK frame of one lane.
struct LanePairs<'a> {
    ook: &'a mut [u32],
    fsk: &'a mut [u32],
}

fn put(pairs: &mut [u32], i: usize, v: u32) {
    if let Some(s) = pairs.get_mut(i) {
        *s = v;
    }
}

fn cmul_conj(x: C32, y: C32) -> [f32; 2] {
    [x.re * y.re + x.im * y.im, x.im * y.re - x.re * y.im]
}

/// Where between the sample before (`prev`) and the current one (`cur`) the line through them
/// crosses `thr`, as a fraction 0 to 1 of the step (1 when they are equal).
fn cross(prev: f32, cur: f32, thr: f32) -> f64 {
    let d = cur - prev;
    if d.abs() > 0.0 {
        f64::from((thr - prev) / d).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// `atan2(y, x)` to about 1e-5 rad (a minimax polynomial of `atan` on `[0, 1]` after the
/// octant reduction), for the frequency of every sample of a frame; 0 for `(0, 0)`, NaN for a
/// NaN input. The frequency resolution the slicer needs is about 1e-2 rad (D959).
fn atan2_fast(y: f32, x: f32) -> f32 {
    let (ax, ay) = (x.abs(), y.abs());
    let (mn, mx) = if ax < ay { (ax, ay) } else { (ay, ax) };
    if mx == 0.0 {
        return 0.0;
    }
    let a = mn / mx;
    let s = a * a;
    let mut r = a
        * (0.999_977_26
            + s * (-0.332_623_47
                + s * (0.193_543_46 + s * (-0.116_432_87 + s * (0.052_653_32 - s * 0.011_721_2)))));
    if ay > ax {
        r = core::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = core::f32::consts::PI - r;
    }
    if y < 0.0 { -r } else { r }
}

/// The time of the crossing before sample `idx`.
fn edge(idx: u64, frac: f64) -> f64 {
    idx as f64 - 1.0 + frac
}

/// The step of a time (rounded, at least 0).
fn step_of(t: f64) -> u64 {
    let r = libm::round(t);
    if r > 0.0 { r as u64 } else { 0 }
}

/// A width in whole steps (rounded, at least 0, saturating).
fn whole(r: f64) -> u32 {
    let r = libm::round(r);
    if r <= 0.0 {
        0
    } else if r >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        r as u32
    }
}

/// The width of an envelope run from `a` to `b`: `b - a` rounded, so every width is within
/// half a step of the measured one (D956).
fn width(a: f64, b: f64) -> u32 {
    whole(b - a)
}

/// The width of a frequency run from `a` to `b`: the difference of the two rounded times, so
/// the sum of consecutive widths is the distance of the rounded first and last time and the
/// transitions do not drift (D959).
fn fsk_width(a: f64, b: f64) -> u32 {
    whole(libm::round(b) - libm::round(a))
}

/// The AGC of the frequency slicer equalises envelopes down to this fraction of the signal
/// level (12 dB): a tone that the channel filter attenuates up to that much weighs as much as
/// the other, while the edges of an OOK pulse, far below the signal level, do not outweigh
/// its plateau (D1025).
const AGC_FLOOR: f32 = 0.25;

/// The gain of the AGC of the frequency slicer for the envelope `e`: `1 / e^2` rounded up to
/// a power of two (`e^2 g` lies in `[1, 2)`; the exponent bits only, no division in the
/// per-sample path), 0 for an envelope of 0 (D1025).
fn a_gain(e: f32) -> f32 {
    let q = e * e;
    if q > FLUSH {
        let exp = (q.to_bits() >> 23) & 0xff;
        f32::from_bits(254u32.saturating_sub(exp) << 23)
    } else {
        0.0
    }
}

/// A short first run of a frame (D1025): its mean frequency lies within this fraction of the
/// distance of the two levels from the other level than the run after it, and it lasts at
/// least `EARLY_MIN` steps (the frequency of the first samples of a frame, while the channel
/// filter rises, lies anywhere).
const EARLY_TOL: f32 = 0.3;
const EARLY_MIN: f64 = 2.0;

/// The resolution of the run ends kept during an FSK frame: `1 / RUN_FIX` steps (D1025).
const RUN_FIX: f64 = 64.0;

/// A time after the origin of a frame in `1 / RUN_FIX` steps (saturating, at least 0).
fn run_fix(t: f64) -> u32 {
    let v = libm::round(t * RUN_FIX);
    if v <= 0.0 {
        0
    } else if v >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        v as u32
    }
}

/// The largest distance of a transition from the grid of the symbol clock of a frame, and of
/// the length of a run from its whole number of symbols, as fractions of the symbol (D1025;
/// the slicer delays the two kinds of transition in opposite directions, so a run is off by
/// up to twice as much as a transition).
const GRID_TOL: f64 = 0.25;
const RUN_TOL: f64 = 0.4;

/// The fewest transitions of a frame for its symbol grid (D1025): with fewer, a grid fits
/// almost any runs.
const GRID_MIN: u32 = 5;

/// The runs of one symbol whose lengths are tried as the symbol of the grid, besides their
/// mean (D1025).
const GRID_CAND: usize = 8;

/// The transitions of a frame whose slots hold the ends of its runs (`fix` steps after the
/// origin, `fsk_commit`): the ends of every run with a length except the last, in order.
fn transitions(slots: &[u32], mut f: impl FnMut(f64)) {
    let last = slots
        .iter()
        .enumerate()
        .rev()
        .find(|&(i, &v)| {
            v > i
                .checked_sub(1)
                .and_then(|j| slots.get(j))
                .copied()
                .unwrap_or(0)
        })
        .map_or(0, |(i, _)| i);
    let mut before = 0u32;
    for &v in slots.iter().take(last) {
        if v > before {
            f(f64::from(v) / RUN_FIX);
        }
        before = v;
    }
}

/// The symbols of a run of `d` steps for the symbol `t` steps long (at least 1).
fn symbols(d: f64, t: f64) -> f64 {
    libm::round(d / t).max(1.0)
}

/// The least-squares grid `a + k t` of the transitions of a frame for the symbol length
/// `guess` (every run between two transitions a whole number of symbols, rounded with
/// `guess`, then once more with the fitted symbol): `(a, t, cost)` when every transition lies
/// within `GRID_TOL` symbols of it and every run within `RUN_TOL` symbols of its whole number,
/// `cost` the sum of the squared distances of the transitions in symbols.
fn grid_fit(slots: &[u32], guess: f64) -> Option<(f64, f64, f64)> {
    let fit = |t_sym: f64| {
        let (mut n, mut sk, mut skk, mut st, mut skt) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut k, mut prev) = (0.0f64, None::<f64>);
        transitions(slots, |t| {
            if let Some(tp) = prev {
                k += symbols(t - tp, t_sym);
            }
            prev = Some(t);
            n += 1.0;
            sk += k;
            skk += k * k;
            st += t;
            skt += k * t;
        });
        let det = n * skk - sk * sk;
        if det <= 0.0 {
            return None;
        }
        let t = (n * skt - sk * st) / det;
        let a = (st - t * sk) / n;
        (t > 0.0).then_some((a, t))
    };
    let (_, t1) = fit(guess)?;
    let (a, t) = fit(t1)?;
    // Every transition near its grid point and every run near its whole number of symbols
    // (rounded with `t`).
    let (mut k, mut prev, mut ok, mut cost) = (0.0f64, None::<f64>, true, 0.0f64);
    transitions(slots, |x| {
        if let Some(tp) = prev {
            let n = symbols(x - tp, t);
            ok &= (x - tp - n * t).abs() <= RUN_TOL * t;
            k += n;
        }
        prev = Some(x);
        let r = (x - (a + k * t)) / t;
        ok &= r.abs() <= GRID_TOL;
        cost += r * r;
    });
    ok.then_some((a, t, cost))
}

/// Turns the run ends of a finished frame (`slots`, times after the start of its first run,
/// `fsk_commit`) into widths in whole steps (D1025). The bit clock of a transmitter is steady,
/// so the transitions are fitted with the grid `a + k t` of their symbol clock (least squares;
/// the guesses of the symbol `t` are the runs between two transitions shorter than 1.5 times
/// the shortest, the runs of one symbol, and their mean; the grid of the smallest distances
/// wins); the start and the end of the frame, whose envelope edges are less precise, are
/// moved to the nearest whole number of symbols (at least one) before the first and after the
/// last transition; every boundary is
/// rounded as a time after the start, so every transition is within half a step of its time
/// on the grid after the start, and every width is the difference of two rounded boundaries.
/// A frame with fewer than `GRID_MIN` transitions, or whose transitions do not lie on one
/// grid, keeps its measured times (rounded after the start).
fn fsk_widths(slots: &mut [u32]) {
    let (mut count, mut shortest, mut prev) = (0u32, f64::INFINITY, None::<f64>);
    transitions(slots, |t| {
        if let Some(tp) = prev {
            shortest = shortest.min(t - tp);
        }
        prev = Some(t);
        count += 1;
    });
    // The first guess of the symbol: the mean of the runs shorter than 1.5 times the shortest
    // (the runs of one symbol, whose upper and lower runs the slicer biases in opposite
    // directions).
    let (mut sum, mut n, mut prev) = (0.0f64, 0.0f64, None::<f64>);
    transitions(slots, |t| {
        if let Some(tp) = prev
            && t - tp < 1.5 * shortest
        {
            sum += t - tp;
            n += 1.0;
        }
        prev = Some(t);
    });
    // The grid of the smallest cost from the guesses: the mean of the runs of one symbol and
    // each of the first `GRID_CAND` of them.
    let mut grid: Option<(f64, f64, f64)> = None;
    if count >= GRID_MIN && n > 0.0 {
        let mut better = |g: Option<(f64, f64, f64)>| {
            if let Some(g) = g
                && grid.is_none_or(|b| g.2 < b.2)
            {
                grid = Some(g);
            }
        };
        better(grid_fit(slots, sum / n));
        let (mut tried, mut prev) = (0usize, None::<f64>);
        transitions(slots, |t| {
            if let Some(tp) = prev
                && t - tp < 1.5 * shortest
                && tried < GRID_CAND
            {
                tried += 1;
                better(grid_fit(slots, t - tp));
            }
            prev = Some(t);
        });
    }
    let Some((a, t, _)) = grid else {
        // The measured times: widths of the rounded times after the start.
        let mut before = 0.0;
        for v in slots.iter_mut() {
            let r = libm::round(f64::from(*v) / RUN_FIX);
            *v = whole(r - before);
            before = r;
        }
        return;
    };
    // The symbols from the first transition to the last one.
    let (mut k_last, mut prev) = (0.0f64, None::<f64>);
    transitions(slots, |x| {
        if let Some(tp) = prev {
            k_last += symbols(x - tp, t);
        }
        prev = Some(x);
    });
    let end = slots.last().map_or(0.0, |&v| f64::from(v) / RUN_FIX);
    // The symbols before the first transition and after the last one.
    let lead = symbols(a, t);
    let stop = lead + k_last + symbols(end - (a + k_last * t), t);
    // The boundaries in order after the start: the transitions on the grid, the end.
    let n_tr = count as usize;
    let mut before_raw = 0u32;
    let mut before = 0.0;
    let (mut seen, mut k, mut prev) = (0usize, 0.0f64, None::<f64>);
    for v in slots.iter_mut() {
        let raw = *v;
        // An empty slot (a frame that starts at the lower level, the last gap 0) has no width.
        let r = if raw <= before_raw {
            before
        } else if seen < n_tr {
            let x = f64::from(raw) / RUN_FIX;
            if let Some(tp) = prev {
                k += symbols(x - tp, t);
            }
            prev = Some(x);
            seen += 1;
            libm::round((lead + k) * t)
        } else {
            libm::round(stop * t)
        };
        *v = whole(r - before);
        before = r;
        before_raw = raw;
    }
}

impl Lane {
    /// Starts the lead-in at step `idx` with the sample `x` (`xp` the one before).
    fn begin_lead(&mut self, idx: u64, factor: f32, e: f32, x: C32, xp: C32) {
        self.mode = LEAD;
        self.p_factor = factor;
        self.lead_at = idx;
        self.lead_n = 1;
        self.lead_x = [C32::default(); BAND_PULSE_LEAD + 1];
        self.lead_e = [0.0; BAND_PULSE_LEAD];
        if let Some(v) = self.lead_x.get_mut(0) {
            *v = xp;
        }
        if let Some(v) = self.lead_x.get_mut(1) {
            *v = x;
        }
        if let Some(v) = self.lead_e.get_mut(0) {
            *v = e;
        }
    }

    /// The state of a burst without a pulse yet, at the time `t`.
    fn begin_burst(&mut self, t: f64, e: f32) {
        self.mode = BURST;
        self.level = 0;
        self.run_len = 0;
        self.confirmed = 1;
        self.prev_len = 0;
        self.t_run = t;
        self.t_prev = t;
        self.c_run = t;
        self.c_prev = t;
        self.e_prev = e;
        self.p_level = 0;
        self.p_run = 0;
        self.p_confirmed = 1;
        self.p_prev = 0;
        self.p_t_run = t;
        self.p_t_prev = t;
        self.started = 0;
        self.ook_n = 0;
        self.open = 0;
        self.longest = 0;
        self.truncated = 0;
        self.frames = 0;
        self.acc = [0.0; 2];
        self.pac = [0.0; 2];
    }

    /// Resets the FSK frame at the start of a pulse at the time `t`.
    fn begin_frame(&mut self, t: f64) {
        self.fm = [0.0; 2];
        self.f_ref = 0.0;
        self.f_hi = 0.0;
        self.f_lo = 0.0;
        self.f_prev = 0.0;
        self.cand_before = 0.0;
        self.f_early = 0.0;
        self.early = 0;
        self.t_early = t;
        self.ft_run = t;
        self.ft_prev = t;
        self.fsk_phase = 0;
        self.fsk_level = 0;
        self.fsk_run = 0;
        self.fsk_confirmed = 1;
        self.fsk_prev = 0;
        self.fsk_has_prev = 0;
        self.fsk_open = 0;
        self.fsk_n = 0;
        self.fsk_runs = 0;
        self.fsk_cand = 0;
        self.fsk_cnt = 0;
        self.fsk_full = 0;
    }

    /// One lead-in sample; at the end of the lead-in the samples are replayed through the
    /// burst detector.
    fn lead(&mut self, p: &Params, noise: f32, pairs: &mut LanePairs<'_>, e: f32, x: C32) -> Next {
        let n = self.lead_n as usize;
        if let Some(v) = self.lead_e.get_mut(n) {
            *v = e;
        }
        if let Some(v) = self.lead_x.get_mut(n + 1) {
            *v = x;
        }
        self.lead_n += 1;
        if self.lead_n < p.lead_len {
            return Next::Stay;
        }
        let len = p.lead_len as usize;
        // The signal level: the largest envelope of the lead-in.
        self.high = self
            .lead_e
            .iter()
            .take(len)
            .fold(0.0f32, |m, &v| if v > m { v } else { m });
        let first = self.lead_e.first().copied().unwrap_or(0.0);
        self.begin_burst(self.lead_at as f64, first);
        for i in 0..len {
            let (Some(&e), Some(&xp), Some(&x)) = (
                self.lead_e.get(i),
                self.lead_x.get(i),
                self.lead_x.get(i + 1),
            ) else {
                continue;
            };
            let next = self.sample(p, noise, pairs, e, x, xp, self.lead_at + i as u64);
            if next != Next::Stay {
                return next;
            }
        }
        Next::Stay
    }

    /// Pushes a confirmed pulse of `len` steps to the OOK train; `false` when the train is
    /// full (the burst then ends truncated without it).
    fn push_pulse(&mut self, p: &Params, pairs: &mut LanePairs<'_>, len: u32) -> bool {
        if self.ook_n >= p.max_pulses {
            self.truncated = 1;
            return false;
        }
        put(pairs.ook, 2 * self.ook_n as usize, len);
        self.open = 1;
        self.acc[0] += self.pac[0];
        self.acc[1] += self.pac[1];
        true
    }

    /// Ends the OOK train (the last gap 0) into the output record 0, to be decided.
    fn end_train(&mut self, pairs: &mut LanePairs<'_>) {
        if self.open == 1 {
            put(pairs.ook, 2 * self.ook_n as usize + 1, 0);
            self.ook_n += 1;
            self.open = 0;
        }
        if self.ook_n > 0 {
            let freq = libm::atan2f(self.acc[1], self.acc[0]);
            self.out[0] = Out {
                index: self.start,
                end: self.pulse_end,
                state: OUT_DECIDE,
                flags: if self.truncated == 1 {
                    PacketFlags::TRUNCATED.bits()
                } else {
                    0
                },
                pairs: self.ook_n,
                sent: 0,
                level: self.high,
                freq,
                fsk: 0,
                _pad: 0,
            };
        }
        self.ook_n = 0;
        self.open = 0;
        self.longest = 0;
        self.truncated = 0;
        self.started = 0;
        self.acc = [0.0; 2];
    }

    /// Commits a frequency run from the time `a` to `b`. Until the frame ends, its slot of the
    /// FSK pairs holds the end of the run (`b`) after the origin of the frame in `1 / RUN_FIX`
    /// steps (an empty slot the end of the run before); `end_frame` turns them into widths
    /// (D1025).
    fn fsk_commit(&mut self, p: &Params, pairs: &mut LanePairs<'_>, upper: bool, a: f64, b: f64) {
        if self.fsk_full == 1 || fsk_width(a, b) == 0 {
            return;
        }
        if self.fsk_runs == 0 {
            self.f_origin = a;
        }
        self.fsk_runs += 1;
        let at = |t: f64| run_fix(t - self.f_origin);
        if upper {
            if self.fsk_n >= p.max_pulses {
                self.fsk_full = 1;
                return;
            }
            put(pairs.fsk, 2 * self.fsk_n as usize, at(b));
            self.fsk_open = 1;
        } else {
            if self.fsk_open == 0 {
                if self.fsk_n >= p.max_pulses {
                    self.fsk_full = 1;
                    return;
                }
                put(pairs.fsk, 2 * self.fsk_n as usize, at(a));
            }
            put(pairs.fsk, 2 * self.fsk_n as usize + 1, at(b));
            self.fsk_n += 1;
            self.fsk_open = 0;
        }
        self.f_last = b;
    }

    /// One sample of the frequency slicer at step `idx` (`d` the product of the sample and the
    /// conjugate of the one before).
    fn fsk_sample(&mut self, p: &Params, pairs: &mut LanePairs<'_>, d: [f32; 2], e: f32, idx: u64) {
        // The product scaled by the envelope squared (an AGC before the low-pass): a tone that
        // the channel filter attenuates weighs as much as the other one, so the low-passed
        // frequency leaves the stronger tone as fast as the weaker (D1025).
        let g = a_gain(e.max(AGC_FLOOR * self.high));
        let a = p.alpha;
        self.fm[0] = flush_denormal(self.fm[0] + a * (g * d[0] - self.fm[0]));
        self.fm[1] = flush_denormal(self.fm[1] + a * (g * d[1] - self.fm[1]));
        let f = atan2_fast(self.fm[1], self.fm[0]);
        let fp = self.f_prev;
        self.f_prev = f;
        if self.fsk_phase == 0 {
            self.fsk_run = self.fsk_run.saturating_add(1);
            if self.fsk_cnt == 0 && self.fsk_cand == 0 {
                self.f_ref = f;
            }
            let dev = f - self.f_ref;
            if dev.abs() > p.fsk_delta {
                let side = dev > 0.0;
                let last_side = self.cand_f.first().is_some_and(|&c| c > self.f_ref);
                if self.fsk_cand > 0 && side != last_side {
                    self.fsk_cand = 0;
                }
                if self.fsk_cand == 0 {
                    self.cand_before = fp;
                }
                if let Some(c) = self.cand_f.get_mut(self.fsk_cand as usize) {
                    *c = f;
                }
                self.fsk_cand += 1;
                if self.fsk_cand >= p.fsk_min {
                    if self.fsk_cnt < p.fsk_min {
                        // Too few samples before the candidates to know the first level (the
                        // start of the pulse): the mean starts over from here. The samples
                        // before may be a short first run: their mean and the time of the
                        // change are kept for the first transition (D1025).
                        if self.fsk_cnt > 0 {
                            let first = (idx + 1).saturating_sub(u64::from(self.fsk_cand));
                            let at = self.cand_f.first().copied().unwrap_or(f);
                            let thr = self.f_ref.midpoint(f);
                            self.f_early = self.f_ref;
                            self.t_early = edge(first, cross(self.cand_before, at, thr));
                            self.early = 1;
                        }
                        self.f_ref = f;
                        self.fsk_cnt = 1;
                        self.fsk_cand = 0;
                    } else {
                        self.fsk_resolve(p, pairs, f, idx);
                    }
                }
            } else {
                self.fsk_cand = 0;
                let n = (self.fsk_cnt + 1).min(16);
                self.f_ref += (f - self.f_ref) / n as f32;
                self.fsk_cnt += 1;
            }
            return;
        }
        let mid = self.f_hi.midpoint(self.f_lo);
        let h = (self.f_hi - self.f_lo) * FSK_HYSTERESIS;
        let thr = if self.fsk_level == 1 {
            mid - h
        } else {
            mid + h
        };
        let upper = if self.fsk_level == 1 {
            f >= thr
        } else {
            f > thr
        };
        if upper {
            self.f_hi += (f - self.f_hi) * FSK_RATIO;
        } else {
            self.f_lo += (f - self.f_lo) * FSK_RATIO;
        }
        let tc = edge(idx, cross(fp, f, thr));
        self.fsk_run_step(p, pairs, upper, tc);
    }

    /// The first transition of a frame at step `idx` (the last candidate `f`): the levels and
    /// the end of the first run.
    fn fsk_resolve(&mut self, p: &Params, pairs: &mut LanePairs<'_>, f: f32, idx: u64) {
        let first_upper = self.f_ref > f;
        let (hi, lo) = if first_upper {
            (self.f_ref, f)
        } else {
            (f, self.f_ref)
        };
        self.f_hi = hi;
        self.f_lo = lo;
        let mid = hi.midpoint(lo);
        let h = (hi - lo) * FSK_HYSTERESIS;
        let thr = if first_upper { mid - h } else { mid + h };
        let cand = self.fsk_cand as usize;
        let j = self
            .cand_f
            .iter()
            .take(cand)
            .position(|&c| if first_upper { c < thr } else { c > thr })
            .unwrap_or(cand.saturating_sub(1));
        let before = match j.checked_sub(1) {
            Some(i) => self.cand_f.get(i).copied().unwrap_or(self.cand_before),
            None => self.cand_before,
        };
        let at = self.cand_f.get(j).copied().unwrap_or(f);
        let sample = (idx + 1).saturating_sub((cand - j) as u64);
        let tc = edge(sample, cross(before, at, thr));
        let second = (cand - j) as u32;
        self.fsk_phase = 1;
        self.fsk_level = u32::from(!first_upper);
        // A short first run before the last restart at the other level is the first run of
        // the frame (D1025).
        let rel = (self.f_early - lo) / (hi - lo);
        let other = if first_upper { rel } else { 1.0 - rel };
        if self.early == 1 && other.abs() <= EARLY_TOL && self.t_early - self.ft_run >= EARLY_MIN {
            self.fsk_commit(p, pairs, !first_upper, self.ft_run, self.t_early);
            self.ft_run = self.t_early;
        }
        self.fsk_prev = self.fsk_run.saturating_sub(second);
        self.fsk_has_prev = 1;
        self.ft_prev = self.ft_run;
        self.ft_run = tc;
        self.fsk_run = second;
        self.fsk_confirmed = 0;
        if second >= p.fsk_min {
            self.fsk_confirmed = 1;
            self.fsk_commit(p, pairs, first_upper, self.ft_prev, self.ft_run);
            self.fsk_has_prev = 0;
        }
    }

    /// The run machine of the frequency levels; `tc` the time of a change of the level.
    fn fsk_run_step(&mut self, p: &Params, pairs: &mut LanePairs<'_>, upper: bool, tc: f64) {
        let level = u32::from(upper);
        if level == self.fsk_level {
            self.fsk_run = self.fsk_run.saturating_add(1);
        } else if self.fsk_confirmed == 0 && self.fsk_has_prev == 1 {
            // A glitch: back to the previous run, which is confirmed.
            self.fsk_level = level;
            self.fsk_run = self.fsk_prev.saturating_add(self.fsk_run).saturating_add(1);
            self.ft_run = self.ft_prev;
            self.fsk_confirmed = 1;
            self.fsk_has_prev = 0;
            return;
        } else {
            self.fsk_prev = self.fsk_run;
            self.fsk_has_prev = 1;
            self.ft_prev = self.ft_run;
            self.ft_run = tc;
            self.fsk_level = level;
            self.fsk_run = 1;
            self.fsk_confirmed = 0;
        }
        if self.fsk_confirmed == 0 && self.fsk_run >= p.fsk_min {
            self.fsk_confirmed = 1;
            if self.fsk_has_prev == 1 {
                self.fsk_commit(p, pairs, !upper, self.ft_prev, self.ft_run);
                self.fsk_has_prev = 0;
            }
        }
    }

    /// Ends the frame of the presence from `start` to `end` (times); `true` when it is an FSK
    /// frame for `p.modulation` (its record 1 then waits for its decision).
    fn end_frame(&mut self, p: &Params, pairs: &mut LanePairs<'_>, start: f64, end: f64) -> bool {
        if p.modulation == PulseModulation::Ook {
            return false;
        }
        let split = self.fsk_phase == 1 && self.f_hi - self.f_lo >= p.fsk_split;
        let is_fsk = match p.modulation {
            PulseModulation::Fsk => true,
            _ => split && self.fsk_runs + 1 + self.fsk_has_prev >= FSK_MIN_RUNS,
        };
        if !is_fsk {
            return false;
        }
        if self.fsk_phase == 0 {
            self.fsk_commit(p, pairs, true, self.ft_run, end);
        } else {
            let upper = self.fsk_level == 1;
            if self.fsk_has_prev == 1 {
                // The last run is too short: it belongs to the run before.
                self.fsk_commit(p, pairs, !upper, self.ft_prev, end);
            } else {
                self.fsk_commit(p, pairs, upper, self.ft_run, end);
            }
        }
        if self.fsk_open == 1 {
            put(
                pairs.fsk,
                2 * self.fsk_n as usize + 1,
                run_fix(self.f_last - self.f_origin),
            );
            self.fsk_n += 1;
            self.fsk_open = 0;
        }
        let slots = pairs
            .fsk
            .get_mut(..2 * self.fsk_n as usize)
            .unwrap_or_default();
        fsk_widths(slots);
        let freq = if self.fsk_phase == 1 {
            self.f_hi.midpoint(self.f_lo)
        } else {
            libm::atan2f(self.pac[1], self.pac[0])
        };
        self.out[1] = Out {
            index: step_of(start),
            end: step_of(end),
            state: OUT_DECIDE,
            flags: if self.fsk_full == 1 {
                PacketFlags::TRUNCATED.bits()
            } else {
                0
            },
            pairs: self.fsk_n,
            sent: 0,
            level: self.high,
            freq,
            fsk: 1,
            _pad: 0,
        };
        true
    }

    /// The current pulse is final: `len` steps, ending at the time `end` (where the gap after
    /// it started). With `modulation` `fsk` the envelope pulses are not kept (the frames come
    /// from the presence of the carrier).
    fn pulse_done(&mut self, p: &Params, pairs: &mut LanePairs<'_>, len: u32, end: f64) -> Next {
        self.pulse_end = step_of(end);
        if p.modulation == PulseModulation::Fsk {
            self.started = 0;
            return Next::Stay;
        }
        if len > self.longest {
            self.longest = len;
        }
        if !self.push_pulse(p, pairs, len) {
            self.end_train(pairs);
            return Next::Idle;
        }
        Next::Stay
    }

    /// An FSK frame from `start` to `end` (times) is decided (record 1 waits): the pulses of
    /// the OOK train from its start on belong to it and are dropped; the pulses before it end
    /// as a train of their own.
    fn cut_train(&mut self, pairs: &mut LanePairs<'_>, start: f64) {
        let total = self.ook_n + self.open;
        let mut t = self.start as f64;
        let mut keep = 0u32;
        let mut end = self.start;
        for k in 0..total {
            let i = 2 * k as usize;
            let pulse = pairs.ook.get(i).copied().unwrap_or(0);
            let gap = if k < self.ook_n {
                pairs.ook.get(i + 1).copied().unwrap_or(0)
            } else {
                0
            };
            if t >= start - 2.0 {
                break;
            }
            keep = k + 1;
            end = step_of(t + f64::from(pulse));
            t += f64::from(pulse) + f64::from(gap);
        }
        if keep == 0 {
            self.ook_n = 0;
            self.open = 0;
            self.longest = 0;
            self.truncated = 0;
            self.started = 0;
            self.acc = [0.0; 2];
            return;
        }
        // The last kept pulse ends the train (its gap is 0).
        self.ook_n = keep - 1;
        self.open = 1;
        self.pulse_end = end;
        self.end_train(pairs);
    }

    /// The presence machine (`modulation` `fsk` or `auto`): the carrier is present while the
    /// envelope is `min_snr_db` above the noise (hysteresis 1/8); each presence of at least
    /// `min_steps` is one FSK frame, whose frequency runs the slicer measures. Returns the
    /// request of a carrier that lasts too long.
    #[allow(clippy::too_many_arguments)]
    fn presence(
        &mut self,
        p: &Params,
        noise: f32,
        pairs: &mut LanePairs<'_>,
        e: f32,
        ep: f32,
        d: [f32; 2],
        idx: u64,
    ) -> Option<Next> {
        let trig = noise * self.p_factor;
        let thr = if self.p_level == 1 {
            trig * (1.0 - HYSTERESIS)
        } else {
            trig * (1.0 + HYSTERESIS)
        };
        let on = if self.p_level == 1 { e >= thr } else { e > thr };
        let level = u32::from(on);
        if level == self.p_level {
            self.p_run = self.p_run.saturating_add(1);
        } else if self.p_confirmed == 0 {
            // A glitch: the run joins the previous one.
            let n = self.p_run;
            self.p_level = level;
            self.p_run = self.p_prev.saturating_add(n).saturating_add(1);
            self.p_t_run = self.p_t_prev;
            self.p_confirmed = 1;
            if on {
                // A short absence inside a frame: its samples extend the frequency run.
                self.fsk_run = self.fsk_run.saturating_add(n);
            }
        } else {
            let tc = edge(idx, cross(ep, e, thr));
            self.p_prev = self.p_run;
            self.p_t_prev = self.p_t_run;
            self.p_t_run = tc;
            self.p_level = level;
            self.p_run = 1;
            self.p_confirmed = 0;
            if on {
                self.begin_frame(tc);
            }
        }
        if on {
            self.fsk_sample(p, pairs, d, e, idx);
        }
        if self.p_confirmed == 0 && self.p_run >= p.min_steps {
            self.p_confirmed = 1;
            if !on && self.p_prev >= p.min_steps {
                let (start, mut end) = (self.p_t_prev, self.p_t_run);
                // The midpoint edge of the envelope just before is the better end.
                if self.level == 0 && (0.0..=EDGE_WINDOW).contains(&(end - self.c_run)) {
                    end = self.c_run;
                }
                self.frame_done(p, pairs, start, end);
            }
        }
        if on && self.p_run > p.long_steps {
            // A carrier: the frame so far is kept (truncated), the lane learns the carrier as
            // its noise.
            self.fsk_full = 1;
            self.frame_done(p, pairs, self.p_t_run, idx as f64 + 1.0);
            self.end_train(pairs);
            return Some(Next::IdleAt(e));
        }
        None
    }

    /// The presence from `start` to `end` is over: an FSK frame if the slicer saw one.
    fn frame_done(&mut self, p: &Params, pairs: &mut LanePairs<'_>, start: f64, end: f64) {
        if self.end_frame(p, pairs, start, end) {
            self.frames = self.frames.saturating_add(1);
            if p.modulation == PulseModulation::Auto {
                self.cut_train(pairs, start);
            }
        }
    }

    /// One sample of a burst at step `idx`: envelope `e`, sample `x`, the one before `xp`
    /// (its raw magnitude is in the ring already).
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &mut self,
        p: &Params,
        noise: f32,
        pairs: &mut LanePairs<'_>,
        e: f32,
        x: C32,
        xp: C32,
        idx: u64,
    ) -> Next {
        let t = noise.midpoint(self.high);
        let h = t * HYSTERESIS;
        let thr = if self.level == 1 { t - h } else { t + h };
        let on = if self.level == 1 { e >= thr } else { e > thr };
        let ep = self.e_prev;
        self.e_prev = e;
        let level = u32::from(on);
        if level == self.level {
            self.run_len = self.run_len.saturating_add(1);
        } else if self.confirmed == 0 {
            // A glitch: the run joins the previous one (which is confirmed).
            let n = self.run_len;
            self.level = level;
            self.run_len = self.prev_len.saturating_add(n).saturating_add(1);
            self.t_run = self.t_prev;
            self.c_run = self.c_prev;
            self.confirmed = 1;
            self.ref_pending = 0;
            if !on {
                // A short pulse: nothing of it stays.
                self.pac = [0.0; 2];
            }
        } else {
            self.begin_run(edge(idx, cross(ep, e, thr)), idx, on);
        }
        // Only the edge of a run that is no glitch is refined, when the run is confirmed.
        if self.ref_pending == 1 && self.run_len >= p.min_steps {
            self.ref_pending = 0;
            self.t_run = self.refine(p, noise, on, idx);
        }
        let d = cmul_conj(x, xp);
        if on {
            // The signal level follows the samples above the upper threshold (not the edges).
            if e > t + h {
                self.high = flush_denormal(self.high + HIGH_RATIO * (e - self.high));
            }
            self.pac[0] += d[0];
            self.pac[1] += d[1];
        }
        if self.confirmed == 0 && self.run_len >= p.min_steps {
            self.confirmed = 1;
            if on {
                if self.started == 1 {
                    // The gap before this pulse is final.
                    if self.open == 1 {
                        let gap = width(self.t_prev, self.t_run);
                        put(pairs.ook, 2 * self.ook_n as usize + 1, gap);
                        self.ook_n += 1;
                        self.open = 0;
                    }
                } else {
                    self.started = 1;
                    self.start = step_of(self.c_run);
                    self.acc = [0.0; 2];
                }
            } else if self.started == 1 {
                let len = width(self.t_prev, self.t_run);
                let next = self.pulse_done(p, pairs, len, self.c_run);
                if next != Next::Stay {
                    return next;
                }
            }
        }
        if p.modulation != PulseModulation::Ook
            && let Some(next) = self.presence(p, noise, pairs, e, ep, d, idx)
        {
            return next;
        }
        if on && self.run_len > p.long_steps {
            // A carrier: the train before it ends, the lane learns the carrier as its noise.
            self.end_train(pairs);
            return Next::IdleAt(e);
        }
        // The end of the burst: the envelope and the carrier are both gone long enough.
        if !on && self.confirmed == 1 && self.p_level == 0 && self.p_confirmed == 1 {
            if self.started == 1 && self.ook_n + self.open > 0 {
                if self.run_len > p.gap_steps || self.gap_ratio_exceeded(p) {
                    self.end_train(pairs);
                    return Next::Idle;
                }
            } else {
                // No pulse yet: a short wait, or after an FSK frame the wait of a train (for
                // the repeats).
                let limit = if self.frames > 0 {
                    p.gap_steps
                } else {
                    p.pre_timeout
                };
                if self.run_len > limit {
                    self.started = 0;
                    return Next::Idle;
                }
            }
        }
        Next::Stay
    }

    /// A new envelope run (`on` a pulse) after a confirmed one, its edge (first stage) at `tc`
    /// detected at sample `idx`; the edge waits for its refinement (D1023).
    #[inline]
    fn begin_run(&mut self, tc: f64, idx: u64, on: bool) {
        self.prev_len = self.run_len;
        self.t_prev = self.t_run;
        self.t_run = tc;
        self.c_prev = self.c_run;
        self.c_run = tc;
        self.ref_idx = idx;
        self.ref_pending = 1;
        self.level = u32::from(on);
        self.run_len = 1;
        self.confirmed = 0;
        if on {
            self.pac = [0.0; 2];
            // The midpoint edge of the envelope is the better start of a frame that the
            // presence (a lower threshold) began just before.
            let lead = tc - self.p_t_run;
            if self.p_level == 1
                && self.fsk_phase == 0
                && self.fsk_runs == 0
                && (0.0..=EDGE_WINDOW).contains(&lead)
            {
                self.p_t_run = tc;
                self.ft_run = tc;
            }
        }
    }

    /// The second stage of an envelope edge (D1023): the edge whose envelope crossing (at
    /// `t_run`, detected at sample `ref_idx`) is delayed by the low-pass is found again in the
    /// raw magnitudes of the ring, from `REFINE_BACK` samples before that sample to the current
    /// one `idx`, at the level `t` of half the signal amplitude over the noise (`NOISE_POWER`):
    /// the split of the window that maximises the sum of `m - t` after it (a rise, `on`) or of
    /// `t - m` (a fall), the change point of a step at the level `t`, and there the crossing of
    /// `t` interpolated between the two samples. The window holds samples of this burst only
    /// (from `HIST - 1` before its lead-in on) and no earlier edge (at most `min_steps - 1`
    /// samples before the crossing).
    fn refine(&self, p: &Params, noise: f32, on: bool, idx: u64) -> f64 {
        let t = libm::sqrtf(0.25 * self.high * self.high + NOISE_POWER * noise * noise);
        let back = REFINE_BACK.min(u64::from(p.min_steps.saturating_sub(1)));
        let lo = self
            .ref_idx
            .saturating_sub(back)
            .max(self.lead_at.saturating_sub(HIST as u64 - 1))
            .max(idx.saturating_sub(RING as u64 - 1));
        if idx <= lo {
            return self.t_run;
        }
        let at = |j: u64| {
            self.ring
                .get(j as usize & (RING - 1))
                .copied()
                .unwrap_or(0.0)
        };
        // Splits s in lo + 1 ..= idx: the samples from s on are after the edge.
        let mut sum = 0.0f32;
        let mut best = f32::NEG_INFINITY;
        let mut split = idx;
        let mut j = idx;
        while j > lo {
            let v = at(j) - t;
            sum += if on { v } else { -v };
            if sum > best {
                best = sum;
                split = j;
            }
            j -= 1;
        }
        edge(split, cross(at(split - 1), at(split), t))
    }

    fn gap_ratio_exceeded(&self, p: &Params) -> bool {
        self.longest > 0 && self.run_len as f32 > p.gap_ratio * self.longest as f32
    }
}

/// Band pulse detector: [`Lanes<C32>`](Lanes) (the lanes of a channelizer) in, [`Packets`]
/// (pulse trains of `sdrk_proto::pulses`) out.
///
/// Per piece of at most [`BAND_PULSE_CHUNK`] steps the block calls `mag` once over all lanes,
/// then per step a loop over the lanes (the envelope low-pass, the noise level of the idle
/// lanes and the trigger, plain Rust over the lanes of a step) and the detector of every lane
/// that is not idle (see `docs/blocks.md` and the block description). The packets of a burst are written
/// at the step where it is decided; when the output is full the block stops after that step
/// and writes the rest first in the next call, so the packets do not depend on the chunking.
/// Timing: `RatioKind::Packets` with at most `M / 16` packets per step (a bound for the
/// buffers), no delay.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, BandPulseDetectBlock, BandPulseDetectConfig, Block, LanesRef, Packet};
/// let cfg = BandPulseDetectConfig::default();
/// let rate = Rate::hz(250_000)?; // the step rate of a 16-channel channelizer at 2 MHz
/// let bytes = BandPulseDetectBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, 1)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut det = BandPulseDetectBlock::new(&cfg, rate, 1, &mut arena)?;
/// // One lane: 8 ms of weak noise, then ten pulses of 400 µs (100 steps) with gaps of 200 µs.
/// let mut x = Vec::new();
/// let mut s = 1u32;
/// for n in 0..6000usize {
///     s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
///     let noise = ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.002;
///     let on = n >= 2000 && n < 2000 + 10 * 150 && (n - 2000) % 150 < 100;
///     x.push(C32::new(if on { 0.5 } else { 0.0 } + noise, noise));
/// }
/// let mut out = vec![Packet::default(); 8];
/// let p = det.process(LanesRef::new(&x, 1)?, &mut out)?;
/// assert_eq!(p.produced, 1);
/// let train = sdrk_proto::pulses::decode(out[0].data())?;
/// assert_eq!(train.len(), 10);
/// assert!(train.pairs().take(9).all(|q| q.pulse.abs_diff(100) <= 1 && q.gap.abs_diff(50) <= 1));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct BandPulseDetectBlock<S: Storage> {
    config: BandPulseDetectConfig,
    p: Params,
    width: usize,
    /// The trigger factor of the idle lanes.
    factor: f32,
    /// Steps since construction or reset.
    pos: u64,
    /// Lanes that are not idle.
    active: usize,
    /// Per lane: the low-passed envelope, the noise level, the noise step (0 while not idle,
    /// so an idle lane is one with a step above 0), the steps of the current wideband block
    /// with the envelope above the trigger level (D1029).
    lp: S::Buf<f32>,
    noise: S::Buf<f32>,
    gain: S::Buf<f32>,
    act: S::Buf<u32>,
    /// The last step of the previous call.
    last: S::Buf<C32>,
    /// Magnitudes of a piece (from row `HIST` on) and of the `HIST` steps before it.
    env: S::Buf<f32>,
    /// Rows of `env` that hold steps (the last `HIST` of them are the history of the next
    /// piece).
    env_rows: usize,
    lanes: S::Buf<Lane>,
    /// One bit per lane that is not idle (`ceil(M / 64)` words).
    busy: S::Buf<u64>,
    /// Pairs of the OOK trains and the FSK frames, `2 max_pulses` values per lane each.
    ook: S::Buf<u32>,
    fsk: S::Buf<u32>,
    /// Whether an output record waits for its packets.
    pending: bool,
    /// Steps of the current block of the wideband rule (D1029).
    blk_n: u32,
    /// The span of the last wideband event (start, end; steps of its wideband blocks) and
    /// whether a burst of it was emitted (D1029).
    wide: (u64, u64),
    wide_emitted: bool,
}

impl<S: Storage> core::fmt::Debug for BandPulseDetectBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BandPulseDetectBlock")
            .field("config", &self.config)
            .field("width", &self.width)
            .field("pos", &self.pos)
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

fn internal() -> Error {
    Error::Core(sdrk_core::Error::Internal("band_pulse_detect"))
}

fn check_width(width: usize) -> Result<()> {
    check_count("width", width, 1, BAND_PULSE_MAX_LANES)
}

fn fsk_len(config: &BandPulseDetectConfig, width: usize) -> usize {
    if config.modulation == PulseModulation::Ook {
        0
    } else {
        width * 2 * config.max_pulses
    }
}

impl<S: Storage> BandPulseDetectBlock<S> {
    /// Creates the block for an input of `width` lanes at the step rate `rate_in`.
    ///
    /// # Errors
    /// As [`BandPulseDetectConfig::validate`]; `OutOfRange` (`width`) for 0 or more than
    /// [`BAND_PULSE_MAX_LANES`] lanes; `Storage` if the storage is too small.
    pub fn new(
        config: &BandPulseDetectConfig,
        rate_in: Rate,
        width: usize,
        storage: &mut S,
    ) -> Result<Self> {
        let p = Params::new(config, rate_in)?;
        check_width(width)?;
        let factor = config.snr_factor();
        let lp = storage.alloc::<f32>(width)?;
        let noise = storage.alloc::<f32>(width)?;
        let mut gain = storage.alloc::<f32>(width)?;
        gain.as_mut().fill(NOISE_RATIO);
        let act = storage.alloc::<u32>(width)?;
        let last = storage.alloc::<C32>(width)?;
        let env = storage.alloc::<f32>(width * (HIST + BAND_PULSE_CHUNK))?;
        let lanes = storage.alloc::<Lane>(width)?;
        let busy = storage.alloc::<u64>(width.div_ceil(64))?;
        let ook = storage.alloc::<u32>(width * 2 * config.max_pulses)?;
        let fsk = storage.alloc::<u32>(fsk_len(config, width))?;
        Ok(Self {
            config: *config,
            p,
            width,
            factor,
            pos: 0,
            active: 0,
            lp,
            noise,
            gain,
            act,
            last,
            env,
            lanes,
            busy,
            ook,
            fsk,
            pending: false,
            blk_n: 0,
            wide: (0, 0),
            wide_emitted: false,
            env_rows: HIST,
        })
    }

    /// Exact bytes of storage the block needs for `width` lanes, for an arena that starts at a
    /// 16-byte aligned address.
    ///
    /// # Errors
    /// `OutOfRange` (`width`, `max_pulses`).
    pub fn storage_bytes(config: &BandPulseDetectConfig, width: usize) -> Result<usize> {
        check_width(width)?;
        check_count("max_pulses", config.max_pulses, 1, MAX_PULSES)?;
        let mut plan = Plan::default();
        for _ in 0..3 {
            plan.add::<f32>(width)?;
        }
        plan.add::<u32>(width)?;
        plan.add::<C32>(width)?;
        plan.add::<f32>(width * (HIST + BAND_PULSE_CHUNK))?;
        plan.add::<Lane>(width)?;
        plan.add::<u64>(width.div_ceil(64))?;
        plan.add::<u32>(width * 2 * config.max_pulses)?;
        plan.add::<u32>(fsk_len(config, width))?;
        Ok(plan.bytes())
    }

    /// The nominal rate of the packet stream: the step rate times `width / 16`.
    ///
    /// # Errors
    /// As [`BandPulseDetectConfig::validate`]; `OutOfRange` (`width`).
    pub fn rate_out(config: &BandPulseDetectConfig, rate_in: Rate, width: usize) -> Result<Rate> {
        config.validate(rate_in)?;
        check_width(width)?;
        Ok(rate_in.scale(Self::max_per_step(width)?)?)
    }

    fn max_per_step(width: usize) -> Result<Ratio> {
        Ok(Ratio::new(width as u64, 16)?)
    }

    /// The width of the input (lanes per step).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Steps since construction or reset.
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// The noise level (envelope, dBFS of the lane) of every lane, in lane order.
    pub fn noise_levels(&self) -> &[f32] {
        self.noise.as_ref()
    }

    /// The signed channel number of lane `k`: `k` for the lower half, `k - M` above (the
    /// convention of `channel_select`).
    fn channel(&self, k: usize) -> i64 {
        let w = self.width as i64;
        let k = k as i64;
        if k < w - w / 2 { k } else { k - w }
    }

    /// The lane of the signed channel `c`, if it exists.
    fn lane_of(&self, c: i64) -> Option<usize> {
        let w = self.width as i64;
        if c < -(w / 2) || c > w - w / 2 - 1 {
            return None;
        }
        Some(c.rem_euclid(w) as usize)
    }

    /// Channel spacing in Hz (half the step rate).
    fn spacing(&self) -> f64 {
        0.5 * self.p.rate_hz
    }

    /// The offset of a burst of lane `k` from the band centre in Hz, from its rotation
    /// `rad` per step. The lane of channel `-M/2` (from 2 lanes on) is centred on the edge of
    /// the input band, minus and plus half the input rate: a negative rotation there lies
    /// below `-fs/2`, which is the upper end of the band, so its offset is taken there (`M`
    /// spacings up, D964).
    fn offset_hz(&self, k: usize, rad: f32) -> f64 {
        let c = self.channel(k);
        let w = self.width as i64;
        let rot = f64::from(rad) * self.p.rate_hz / core::f64::consts::TAU;
        let f = c as f64 * self.spacing() + rot;
        if w >= 2 && c == -(w / 2) && rot < 0.0 {
            f + w as f64 * self.spacing()
        } else {
            f
        }
    }

    /// The two neighbours of lane `k` (signed channels `c - 1` and `c + 1`). The lane of
    /// channel `-M/2` is centred on the edge of the input band (plus and minus half the input
    /// rate) and also borders channel `M/2 - 1`, so from 4 lanes on the two wrap (D964).
    fn neighbours(&self, k: usize) -> [Option<usize>; 2] {
        let c = self.channel(k);
        let w = self.width as i64;
        let wrap = |n: i64| {
            if w < 4 {
                n
            } else if n < -(w / 2) {
                n + w
            } else if n > w - w / 2 - 1 {
                n - w
            } else {
                n
            }
        };
        [self.lane_of(wrap(c - 1)), self.lane_of(wrap(c + 1))]
    }

    /// Whether two bursts overlap by more than half of the shorter one.
    fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
        let over = i128::from(a.1.min(b.1)) - i128::from(a.0.max(b.0));
        let shorter = i128::from((a.1.saturating_sub(a.0)).min(b.1.saturating_sub(b.0)));
        2 * over > shorter
    }

    /// Whether two bursts are one transmission: they overlap by more than half of the shorter
    /// one and their frequencies are less than a spacing apart (modulo the input rate, which
    /// is `M` spacings: the band wraps at its edge, D964).
    fn same(&self, a: (u64, u64, f64), b: (u64, u64, f64)) -> bool {
        let band = self.width as f64 * self.spacing();
        let mut d = (a.2 - b.2) % band;
        if d > 0.5 * band {
            d -= band;
        } else if d < -0.5 * band {
            d += band;
        }
        Self::overlap((a.0, a.1), (b.0, b.1)) && d.abs() < self.spacing()
    }

    /// The spans so far of the burst in progress of lane `n` (the OOK train and the FSK frame),
    /// without their frequencies (see `in_progress`).
    fn spans(&self, n: &Lane) -> [Option<(u64, u64)>; 2] {
        let mut out = [None, None];
        if n.mode != BURST {
            return out;
        }
        if n.started == 1 {
            let end = if n.level == 1 {
                self.pos + 1
            } else {
                step_of(n.c_run)
            };
            out[0] = Some((n.start, end.max(n.start)));
        }
        if n.p_level == 1 && n.p_confirmed == 1 {
            out[1] = Some((step_of(n.p_t_run), self.pos + 1));
        }
        out
    }

    /// The burst in progress of lane `j` (its span so far, the frequency of lane `j` from
    /// `rot`): the OOK train (index 0) and the FSK frame (index 1), with their level.
    fn in_progress(&self, j: usize, n: &Lane) -> [Option<(u64, u64, f64, f32)>; 2] {
        let [ook, frame] = self.spans(n);
        [
            ook.map(|(a, b)| {
                let rot = libm::atan2f(n.acc[1] + n.pac[1], n.acc[0] + n.pac[0]);
                (a, b, self.offset_hz(j, rot), n.high)
            }),
            frame.map(|(a, b)| {
                // At the middle of its two levels once it has them.
                let rot = if n.fsk_phase == 1 {
                    n.f_hi.midpoint(n.f_lo)
                } else {
                    libm::atan2f(n.pac[1], n.pac[0])
                };
                (a, b, self.offset_hz(j, rot), n.high)
            }),
        ]
    }

    /// The end of a block of the wideband rule (D1029): the block is wideband when all lanes
    /// that can hear it but at most `max(1, L / 8)` were active in it (above the trigger level
    /// in at least a quarter of its steps; `L` the lanes that can hear, at least 4 and half of
    /// the lanes). A lane cannot hear when it was not active and its noise level is more than
    /// the trigger factor above the mean noise level of the active lanes (it learned a carrier
    /// as its noise). A narrowband transmission lifts one or two lanes (three for a wide FSK
    /// deviation), so 2 or 3 of them (spec §3, test 4) stay below that from 8 lanes on, while
    /// noise over the band lifts every lane at once (spec test 7). Wideband blocks at most one
    /// block apart form one event.
    #[inline(never)]
    fn close_block(&mut self) {
        let n = u64::from(self.blk_n);
        self.blk_n = 0;
        let need = self.p.wide_active;
        let w = self.width;
        let (mut active, mut sum) = (0usize, 0f32);
        for (a, nz) in self.act.as_ref().iter().zip(self.noise.as_ref()) {
            if *a >= need {
                active += 1;
                sum += nz;
            }
        }
        if w < WIDE_MIN_WIDTH || active < WIDE_MIN_WIDTH || 2 * active < w {
            self.act.as_mut().fill(0);
            return;
        }
        let deaf_above = self.factor * sum / active as f32;
        let deaf = self
            .act
            .as_ref()
            .iter()
            .zip(self.noise.as_ref())
            .filter(|(a, nz)| **a < need && **nz > deaf_above)
            .count();
        self.act.as_mut().fill(0);
        let hear = w - deaf;
        if hear < WIDE_MIN_WIDTH || 2 * hear < w || active + (hear / 8).max(1) < hear {
            return;
        }
        let block = (self.pos - n, self.pos);
        if self.wide.1 > self.wide.0 && self.wide.1 + n >= block.0 {
            self.wide.1 = block.1;
        } else {
            self.wide = block;
            self.wide_emitted = false;
        }
    }

    /// The wideband rule (D1029) for the burst `mine`: `Some(true)` when it is part of the
    /// last wideband event and a burst of that event was emitted already (it is dropped),
    /// `Some(false)` when it is the first burst of the event (it is emitted for the event),
    /// `None` when it is no part of an event. A burst is part of the event when more than half
    /// of its span lies in the span of the event widened by one block on either side (the
    /// first and last blocks of a burst of noise are only partly lifted).
    fn wideband(&self, mine: (u64, u64)) -> Option<bool> {
        if self.wide.1 <= self.wide.0 {
            return None;
        }
        let b = u64::from(self.p.wide_block);
        let (a, z) = (self.wide.0.saturating_sub(b), self.wide.1 + b);
        let inside = mine.1.min(z).saturating_sub(mine.0.max(a));
        let len = mine.1.saturating_sub(mine.0).max(1);
        (2 * inside > len).then_some(self.wide_emitted)
    }

    /// Decides the record `r` of lane `k`: emitted unless a neighbour already emitted the
    /// same transmission or has it at a higher level, or another lane emits the wideband event
    /// it belongs to.
    #[inline(never)]
    fn decide(&mut self, k: usize, r: usize) -> Result<()> {
        let lanes = self.lanes.as_ref();
        let o = *lanes
            .get(k)
            .and_then(|me| me.out.get(r))
            .ok_or_else(internal)?;
        let mine = (o.index, o.end, self.offset_hz(k, o.freq));
        let fsk = o.fsk == 1;
        let wide = self.wideband((mine.0, mine.1));
        let mut emit = wide != Some(true);
        // The end of the last frame this lane emitted on the same frequency: the part of a
        // neighbour's train before it belongs to that frame (an earlier copy, D1026).
        let own = lanes.get(k).map(|l| l.frame).unwrap_or_default();
        let before = if own.valid == 1 && (f64::from(own.freq) - mine.2).abs() < self.spacing() {
            own.end
        } else {
            0
        };
        for j in self.neighbours(k).into_iter().flatten() {
            let n = lanes.get(j).ok_or_else(internal)?;
            // An FSK frame that spans the transmission describes it completely, an OOK train
            // in a neighbour only one of its tones: such a frame is not dropped for a train,
            // and a train is dropped for it (D966). A frame that spans less than half of the
            // train (a frequency jump inside a pulse) counts by its level as before.
            let span = |a: u64, b: u64| b.saturating_sub(a);
            let covers = |frame: u64, train: u64| 2 * frame >= train;
            let me = span(mine.0, mine.1);
            if n.sum_valid == 1 && n.sum_emitted == 1 {
                let theirs = (n.sum_start, n.sum_end, f64::from(n.sum_freq));
                let other = span(theirs.0, theirs.1);
                let kept = fsk && n.sum_fsk == 0 && covers(me, other);
                if !kept && self.same(mine, theirs) {
                    emit = false;
                }
            }
            // The last frame the neighbour emitted (D1026).
            let e = n.frame;
            if !fsk && e.valid == 1 && self.same(mine, (e.start, e.end, f64::from(e.freq))) {
                emit = false;
            }
            for (r, b) in self.in_progress(j, n).into_iter().enumerate() {
                let Some(b) = b else { continue };
                let frame = r == 1;
                let other = if fsk && !frame {
                    span(b.0.max(before), b.1)
                } else {
                    span(b.0, b.1)
                };
                let louder = b.3 > o.level || (b.3 == o.level && j < k);
                let wins = match (fsk, frame) {
                    (true, false) if covers(me, other) => false,
                    (false, true) if covers(other, me) => true,
                    _ => louder,
                };
                if wins && self.same(mine, (b.0, b.1, b.2)) {
                    emit = false;
                }
            }
        }
        if emit && wide == Some(false) {
            self.wide_emitted = true;
        }
        let lane = self.lanes.as_mut().get_mut(k).ok_or_else(internal)?;
        lane.sum_valid = 1;
        lane.sum_emitted = u32::from(emit);
        lane.sum_fsk = o.fsk;
        lane.sum_start = mine.0;
        lane.sum_end = mine.1;
        lane.sum_level = o.level;
        lane.sum_freq = mine.2 as f32;
        if emit && fsk {
            lane.frame = Emitted {
                start: mine.0,
                end: mine.1,
                freq: mine.2 as f32,
                fsk: 1,
                valid: 1,
                _pad: 0,
            };
        }
        let rec = lane.out.get_mut(r).ok_or_else(internal)?;
        rec.state = if emit { OUT_EMIT } else { OUT_NONE };
        if emit {
            self.pending = true;
        }
        Ok(())
    }

    /// Writes the packets of the records to emit, lane by lane (OOK before FSK), into `out`;
    /// returns the number written. `self.pending` stays set when `out` was too small.
    fn flush(&mut self, out: &mut [Packet]) -> Result<usize> {
        if !self.pending {
            return Ok(0);
        }
        let mut produced = 0usize;
        for k in 0..self.width {
            for r in 0..2 {
                let (n, done) = self.emit(k, r, out.get_mut(produced..).ok_or_else(internal)?)?;
                produced += n;
                if !done {
                    return Ok(produced);
                }
            }
        }
        self.pending = false;
        Ok(produced)
    }

    /// Writes the packets of record `r` of lane `k` (if it is to be emitted) that fit into
    /// `out`: the number written and whether the record is done.
    #[inline(never)]
    fn emit(&mut self, k: usize, r: usize, out: &mut [Packet]) -> Result<(usize, bool)> {
        let channel = self.channel(k);
        let cap2 = 2 * self.config.max_pulses;
        let rate = self.p.rate_hz;
        let noise = self.noise.as_ref().get(k).copied().unwrap_or(NOISE_FLOOR);
        let rot = self
            .lanes
            .as_ref()
            .get(k)
            .and_then(|l| l.out.get(r))
            .map(|o| o.freq)
            .ok_or_else(internal)?;
        let freq = self.offset_hz(k, rot);
        let lane = self.lanes.as_mut().get_mut(k).ok_or_else(internal)?;
        let rec = lane.out.get_mut(r).ok_or_else(internal)?;
        if rec.state != OUT_EMIT {
            return Ok((0, true));
        }
        let buf = if r == 0 {
            self.ook.as_ref()
        } else {
            self.fsk.as_ref()
        };
        let pairs = buf.get(k * cap2..(k + 1) * cap2).ok_or_else(internal)?;
        let header = Header {
            modulation: if rec.fsk == 1 {
                Modulation::Fsk
            } else {
                Modulation::Ook
            },
            channel: channel.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16,
            freq_offset_hz: libm::round(freq).clamp(f64::from(i32::MIN), f64::from(i32::MAX))
                as i32,
            level_db10: db10(rec.level),
            noise_db10: db10(noise),
            rate_hz: libm::round(rate).min(f64::from(u32::MAX)) as u32,
        };
        let mut produced = 0usize;
        for slot in out.iter_mut() {
            let from = rec.sent as usize;
            let n = (rec.pairs as usize - from).min(MAX_PAIRS);
            let last = from + n == rec.pairs as usize;
            let src = pairs.get(2 * from..2 * (from + n)).ok_or_else(internal)?;
            let mut flags = PacketFlags(rec.flags);
            if !last {
                flags |= CONTINUED;
            }
            *slot = train_packet(rec.index, flags, &header, src)?;
            produced += 1;
            rec.sent += n as u32;
            if last {
                rec.state = OUT_NONE;
                return Ok((produced, true));
            }
        }
        Ok((produced, false))
    }

    /// The magnitudes of the `n` steps of `data` into the rows from `HIST` on of `env`, after
    /// the last `HIST` steps before them moved to the front (the history of a lane that
    /// triggers, D1023; a forward copy, so overlapping rows are fine).
    #[inline(never)]
    fn magnitudes(&mut self, data: &[C32], n: usize) -> Result<()>
    where
        Sdrk: Mag<C32>,
    {
        let w = self.width;
        let from = self.env_rows.saturating_sub(HIST) * w;
        let env = self.env.as_mut();
        for j in 0..HIST * w {
            let v = env.get(from + j).copied().unwrap_or(0.0);
            if let Some(d) = env.get_mut(j) {
                *d = v;
            }
        }
        let out = env.get_mut(HIST * w..(HIST + n) * w).ok_or_else(internal)?;
        <Sdrk as Mag<C32>>::mag(&Sdrk, data, out)?;
        self.env_rows = HIST + n;
        Ok(())
    }

    /// The detectors of the lanes that are not idle or trigger at step `i` of `data` (one
    /// piece of whole steps; `self.last` is the step before its first).
    fn detect(&mut self, data: &[C32], i: usize, hit: bool) -> Result<()> {
        let w = self.width;
        let step = data.get(i * w..(i + 1) * w).ok_or_else(internal)?;
        let before = if i == 0 {
            None
        } else {
            Some(data.get((i - 1) * w..i * w).ok_or_else(internal)?)
        };
        // The busy lanes of the step before (a lane that becomes idle in this step is not
        // triggered again in it), then the idle lanes that trigger (only when the vector part
        // saw one).
        let mut snapshot = [0u64; BAND_PULSE_MAX_LANES / 64];
        for (s, b) in snapshot.iter_mut().zip(self.busy.as_ref()) {
            *s = *b;
        }
        for (word, &bits) in snapshot.iter().enumerate() {
            let mut rest = bits;
            while rest != 0 {
                let k = word * 64 + rest.trailing_zeros() as usize;
                rest &= rest - 1;
                let mode = self.lanes.as_ref().get(k).map_or(IDLE, |l| l.mode);
                self.lane_at(k, mode, i, step, before)?;
            }
        }
        if hit {
            for k in 0..w {
                let busy = snapshot.get(k / 64).is_some_and(|b| b >> (k % 64) & 1 == 1);
                if busy {
                    continue;
                }
                let e = self.lp.as_ref().get(k).copied().unwrap_or(0.0);
                let noise = self.noise.as_ref().get(k).copied().unwrap_or(NOISE_FLOOR);
                if e > noise * self.factor {
                    self.lane_at(k, IDLE, i, step, before)?;
                }
            }
        }
        Ok(())
    }

    /// Runs lane `k` in mode `mode` on its value of `step` (`before` the step before, or
    /// `self.last`).
    fn lane_at(
        &mut self,
        k: usize,
        mode: u32,
        i: usize,
        step: &[C32],
        before: Option<&[C32]>,
    ) -> Result<()> {
        let e = self.lp.as_ref().get(k).copied().unwrap_or(0.0);
        let noise = self.noise.as_ref().get(k).copied().unwrap_or(NOISE_FLOOR);
        let x = step.get(k).copied().unwrap_or_default();
        let xp = match before {
            Some(b) => b.get(k).copied(),
            None => self.last.as_ref().get(k).copied(),
        }
        .unwrap_or_default();
        self.lane_step(k, mode, i, e, noise, x, xp)
    }

    /// The detector of lane `k` for one step.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn lane_step(
        &mut self,
        k: usize,
        mode: u32,
        i: usize,
        e: f32,
        noise: f32,
        x: C32,
        xp: C32,
    ) -> Result<()> {
        let cap2 = 2 * self.config.max_pulses;
        let fsk_cap = if self.fsk.as_ref().is_empty() {
            0
        } else {
            cap2
        };
        let p = self.p;
        let idx = self.pos;
        let w = self.width;
        // Row `HIST + i` of the magnitudes is this step, the `HIST` rows before it the steps
        // before (D1023).
        let env = self.env.as_ref();
        let m = env.get((HIST + i) * w + k).copied().unwrap_or(0.0);
        let mut pairs = LanePairs {
            ook: self
                .ook
                .as_mut()
                .get_mut(k * cap2..(k + 1) * cap2)
                .ok_or_else(internal)?,
            fsk: self
                .fsk
                .as_mut()
                .get_mut(k * fsk_cap..(k + 1) * fsk_cap)
                .ok_or_else(internal)?,
        };
        let lane = self.lanes.as_mut().get_mut(k).ok_or_else(internal)?;
        if let Some(r) = lane.ring.get_mut(idx as usize & (RING - 1)) {
            *r = m;
        }
        let next = match mode {
            IDLE => {
                lane.begin_lead(idx, p.factor, e, x, xp);
                history(lane, env, (HIST + i) * w + k, w, idx);
                self.active += 1;
                if let Some(b) = self.busy.as_mut().get_mut(k / 64) {
                    *b |= 1 << (k % 64);
                }
                if let Some(g) = self.gain.as_mut().get_mut(k) {
                    *g = 0.0;
                }
                Next::Stay
            }
            LEAD => lane.lead(&p, noise, &mut pairs, e, x),
            _ => lane.sample(&p, noise, &mut pairs, e, x, xp, idx),
        };
        let decide = [
            lane.out[0].state == OUT_DECIDE,
            lane.out[1].state == OUT_DECIDE,
        ];
        if next != Next::Stay {
            lane.mode = IDLE;
            self.active = self.active.saturating_sub(1);
            if let Some(b) = self.busy.as_mut().get_mut(k / 64) {
                *b &= !(1 << (k % 64));
            }
            if let Some(g) = self.gain.as_mut().get_mut(k) {
                *g = NOISE_RATIO;
            }
            if let Next::IdleAt(level) = next
                && let Some(n) = self.noise.as_mut().get_mut(k)
            {
                *n = level.max(NOISE_FLOOR);
            }
        }
        for (r, d) in decide.iter().enumerate() {
            if *d {
                self.decide(k, r)?;
            }
        }
        Ok(())
    }
}

/// Copies the raw magnitudes of the `HIST - 1` steps before `idx` of a lane that triggers
/// into its ring (`at` the index of step `idx` in `env`, `w` the stride of a step; D1023).
#[inline(never)]
fn history(lane: &mut Lane, env: &[f32], at: usize, w: usize, idx: u64) {
    for back in 1..HIST {
        let v = env.get(at - back * w).copied().unwrap_or(0.0);
        if let Some(r) = lane
            .ring
            .get_mut(idx.wrapping_sub(back as u64) as usize & (RING - 1))
        {
            *r = v;
        }
    }
}

/// One packet of a train: `src` the pairs as `u32` values (pulse, gap, ...), at most
/// [`MAX_PAIRS`].
#[inline(never)]
fn train_packet(index: u64, flags: PacketFlags, header: &Header, src: &[u32]) -> Result<Packet> {
    let mut part = [Pair::default(); MAX_PAIRS];
    let mut n = 0usize;
    for (d, s) in part.iter_mut().zip(src.chunks_exact(2)) {
        if let (Some(&a), Some(&b)) = (s.first(), s.get(1)) {
            *d = Pair { pulse: a, gap: b };
            n += 1;
        }
    }
    let part = part.get(..n).ok_or_else(internal)?;
    Ok(sdrk_proto::pulses::encode_packet(
        index, flags, header, part,
    )?)
}

/// The vector part of one step (plain Rust over the lanes of the step): the envelope
/// low-pass, the noise level of the idle lanes and whether one of them triggers. In the
/// warm-up (`warm` is `Some(1 / (n + 1))`) every lane takes the running mean and none
/// triggers.
#[allow(clippy::too_many_arguments)]
fn envelope(
    lp: &mut [f32],
    noise: &mut [f32],
    gain: &[f32],
    act: &mut [u32],
    mag: &[f32],
    alpha: f32,
    factor: f32,
    warm: Option<f32>,
) -> bool {
    if let Some(g) = warm {
        for ((y, nz), &x) in lp.iter_mut().zip(noise.iter_mut()).zip(mag) {
            let e = *y + alpha * (x - *y);
            let e = if e > FLUSH { e.min(LEVEL_LIMIT) } else { 0.0 };
            *y = e;
            *nz = (*nz + g * (e - *nz)).max(NOISE_FLOOR);
        }
        return false;
    }
    let mut hit = 0u32;
    for ((((y, nz), &g), a), &x) in lp
        .iter_mut()
        .zip(noise.iter_mut())
        .zip(gain)
        .zip(act.iter_mut())
        .zip(mag)
    {
        let e = *y + alpha * (x - *y);
        let e = if e > FLUSH { e.min(LEVEL_LIMIT) } else { 0.0 };
        *y = e;
        let n = (*nz + g * (e - *nz)).max(NOISE_FLOOR);
        *nz = n;
        // Above the trigger level: an idle lane triggers, and every lane counts the step as
        // active in the wideband block (D1029).
        let above = e > n * factor;
        *a += u32::from(above);
        hit |= u32::from(above) & u32::from(g > 0.0);
    }
    hit != 0
}

/// `round(200 log10(v))` (dB × 10 of an amplitude) as an `i16` (saturating; 0 and NaN give
/// the smallest value).
fn db10(v: f32) -> i16 {
    if v.is_nan() || v <= 0.0 {
        return i16::MIN;
    }
    let d = libm::roundf(200.0 * libm::log10f(v));
    d.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

impl<S: Storage> Block for BandPulseDetectBlock<S>
where
    Sdrk: Mag<C32>,
{
    type In = Lanes<C32>;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: LanesRef<'_, C32>, output: &mut [Packet]) -> Result<Progress> {
        if input.width() != self.width {
            return Err(Error::Core(sdrk_core::Error::LengthMismatch {
                expected: self.width,
                got: input.width(),
            }));
        }
        let mut produced = self.flush(output)?;
        if self.pending {
            return Ok(Progress {
                consumed: 0,
                produced,
            });
        }
        let w = self.width;
        let total = input.steps();
        let mut consumed = 0usize;
        while consumed < total {
            let n = (total - consumed).min(BAND_PULSE_CHUNK);
            let data = input
                .data()
                .get(consumed * w..(consumed + n) * w)
                .ok_or_else(internal)?;
            self.magnitudes(data, n)?;
            for i in 0..n {
                let warm = (self.pos < BAND_PULSE_WARMUP).then(|| 1.0 / (self.pos + 1) as f32);
                let hit = envelope(
                    self.lp.as_mut(),
                    self.noise.as_mut(),
                    self.gain.as_ref(),
                    self.act.as_mut(),
                    self.env
                        .as_ref()
                        .get((HIST + i) * w..)
                        .ok_or_else(internal)?,
                    self.p.alpha,
                    self.factor,
                    warm,
                );
                if hit || self.active > 0 {
                    self.detect(data, i, hit)?;
                }
                self.pos += 1;
                self.blk_n += 1;
                if self.blk_n >= self.p.wide_block {
                    self.close_block();
                }
                if self.pending {
                    produced += self.flush(output.get_mut(produced..).ok_or_else(internal)?)?;
                    if self.pending {
                        self.env_rows = HIST + i + 1;
                        let step = data.get(i * w..(i + 1) * w).ok_or_else(internal)?;
                        self.last.as_mut().copy_from_slice(step);
                        return Ok(Progress {
                            consumed: consumed + i + 1,
                            produced,
                        });
                    }
                }
            }
            let step = data.get((n - 1) * w..n * w).ok_or_else(internal)?;
            self.last.as_mut().copy_from_slice(step);
            consumed += n;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let r = Self::max_per_step(self.width).unwrap_or(Ratio::ONE);
        BlockTiming {
            ratio: r,
            kind: RatioKind::Packets { max_per_input: r },
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.pos = 0;
        self.active = 0;
        self.pending = false;
        self.blk_n = 0;
        self.wide = (0, 0);
        self.wide_emitted = false;
        self.act.as_mut().fill(0);
        self.env.as_mut().fill(0.0);
        self.env_rows = HIST;
        self.lp.as_mut().fill(0.0);
        self.noise.as_mut().fill(0.0);
        self.gain.as_mut().fill(NOISE_RATIO);
        self.last.as_mut().fill(C32::default());
        self.lanes.as_mut().fill(Lane::default());
        self.busy.as_mut().fill(0);
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(p) if p.name == "min_snr_db" => {
                let mut c = self.config;
                c.set(name, value)?;
                self.config = c;
                self.factor = c.snr_factor();
                self.p.factor = self.factor;
                Ok(())
            }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
mod tests {
    extern crate std;

    use super::*;
    use std::vec::Vec;

    #[test]
    fn the_fast_atan2_is_within_2e_5_rad() {
        let mut worst = 0.0f32;
        for i in 0..3600 {
            let a = f64::from(i) * core::f64::consts::TAU / 3600.0 - core::f64::consts::PI;
            for r in [1e-6f64, 0.3, 1.0, 1e6] {
                let (y, x) = ((r * libm::sin(a)) as f32, (r * libm::cos(a)) as f32);
                let d = (atan2_fast(y, x) - libm::atan2f(y, x)).abs();
                // The same angle on both sides of the cut at pi.
                let d = d.min((d - core::f32::consts::TAU).abs());
                worst = worst.max(d);
            }
        }
        assert!(worst < 2e-5, "{worst}");
        assert_eq!(atan2_fast(0.0, 0.0), 0.0);
        assert!(atan2_fast(f32::NAN, 1.0).is_nan());
    }

    /// The slots of a frame (`fsk_commit`): the end of every run after the start, from the
    /// widths in steps (pulse, gap, ...).
    fn slots_of(widths: &[f64]) -> Vec<u32> {
        let mut t = 0.0;
        widths
            .iter()
            .map(|w| {
                t += w;
                run_fix(t)
            })
            .collect()
    }

    #[test]
    fn the_transitions_of_a_frame_are_fitted_with_its_symbol_grid() {
        // Symbols of 17.3 steps: runs of 2, 1, 3, 1, 1, 2, 1, 4 and 2 symbols (the frame starts
        // with the upper level, the last gap 0), the transitions measured with errors of up to
        // 2 steps, the start 4 steps late and the end 3 steps early.
        let t = 17.3;
        let runs = [2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 4.0, 2.0];
        let jitter = [1.5, -2.0, 0.7, 1.9, -1.2, 0.4, -1.8, 1.1];
        let mut ends: Vec<f64> = runs
            .iter()
            .scan(0.0, |a, n| {
                *a += n * t;
                Some(*a)
            })
            .collect();
        for (e, j) in ends.iter_mut().zip(jitter) {
            *e += j - 4.0;
        }
        let last = ends.len() - 1;
        ends[last] -= 3.0 + 4.0;
        let mut widths: Vec<f64> = Vec::new();
        let mut before = 0.0;
        for e in &ends {
            widths.push(e - before);
            before = *e;
        }
        widths.push(0.0);
        let mut slots = slots_of(&widths);
        fsk_widths(&mut slots);
        // Every boundary is the rounded time of its symbol after the start.
        let mut k = 0.0;
        let mut before = 0.0;
        for (v, n) in slots.iter().zip(runs) {
            k += n;
            let r = libm::round(k * t);
            assert_eq!(f64::from(*v), r - before, "{slots:?}");
            before = r;
        }
        assert_eq!(slots[runs.len()], 0);
        // A frame whose transitions lie on no grid keeps its measured widths.
        let mut odd = slots_of(&[20.0, 31.0, 26.0, 23.0, 40.0, 29.0, 35.0, 37.0, 0.0]);
        fsk_widths(&mut odd);
        assert_eq!(odd, [20, 31, 26, 23, 40, 29, 35, 37, 0]);
        // Fewer than `GRID_MIN` transitions: the measured times, rounded after the start.
        let mut short = slots_of(&[0.0, 18.4, 35.2, 0.0]);
        fsk_widths(&mut short);
        assert_eq!(short, [0, 18, 36, 0]);
    }
}
