//! `resample`: rational polyphase resampler and ppm-tunable polyphase resampler.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::kernels::decim::{Resample, ResampleReal, ResampleRealState, ResampleState};
use sdrk_core::timing::Timing;

use super::fir::MAX_TAPS_LIMIT;
use super::{HZ_LIMIT, check_symmetric, hz, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{DesignSpec, DesignWindow, Lowpass, kaiserord};
use crate::error::{Error, Result};
use crate::sample::{BlockReal, BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// Number of phases of the tuned resampler's filter bank.
pub const TUNED_PHASES: usize = 128;

/// Extra samples of the tuned resampler's work buffer beyond the filter length.
const WORK_EXTRA: usize = 64;

/// Description of the `resample` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "resample",
    summary: "Changes the sample rate by a rational factor, optionally tunable in ppm.",
    details: "Resamples to rate_out_hz. The default mode is an exact rational polyphase \
        resampler L/M with the ratio reduced from the two rates (numerator and denominator at \
        most 1024); it is bit-exact reproducible and has a constant ratio. With tunable = true \
        the block instead uses a 128-phase filter bank with linear interpolation between \
        phases and a 32.32 fixed-point read position, so the ratio can be corrected at run \
        time with ppm (plus or minus 1000 ppm), for example to follow the clock of a sound \
        card; the ratio kind is then variable. The anti-alias filter has its cutoff at 0.45 \
        times the lower of the two rates unless given. Works on complex (C32, CQ15) and real \
        (f32, Q15) signals. Pitfall: the rational mode may need many taps for ratios like \
        44100/48000 (raise max_taps).",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples at the input rate.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples of the same type at rate_out_hz (corrected by ppm).",
    }],
    params: &[
        ParamDesc {
            name: "rate_out_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1.0,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "48000 for audio",
            mutability: Mutability::Build,
            description: "Output sample rate in Hz, a whole number. Required.",
        },
        ParamDesc {
            name: "cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "default 0.45 times the lower of the input and output rates",
            mutability: Mutability::Build,
            description: "Cutoff (-6 dB point) of the anti-alias filter in Hz. The default is \
                0.45 * min(rate_in, rate_out).",
        },
        ParamDesc {
            name: "transition_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "default 0.1 times the lower of the input and output rates",
            mutability: Mutability::Build,
            description: "Width of the transition band in Hz. The default is \
                0.1 * min(rate_in, rate_out).",
        },
        ParamDesc {
            name: "attenuation_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: 8.0,
                max: 200.0,
                default: Some(60.0),
            },
            typical: "60",
            mutability: Mutability::Build,
            description: "Stop-band attenuation in dB (at least 8).",
        },
        ParamDesc {
            name: "max_taps",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_TAPS_LIMIT,
                default: Some(1023),
            },
            typical: "1023; 8192 for 48000 to 44100 in the rational mode",
            mutability: Mutability::Build,
            description: "Upper limit of the number of taps (of the whole filter in the rational \
                mode, per phase in the tunable mode); a design that needs more fails.",
        },
        ParamDesc {
            name: "tunable",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true to follow a drifting clock",
            mutability: Mutability::Build,
            description: "Use the 128-phase tunable resampler; required for the ppm parameter.",
        },
        ParamDesc {
            name: "ppm",
            unit: Unit::Ppm,
            kind: ParamKind::F64 {
                min: -1000.0,
                max: 1000.0,
                default: Some(0.0),
            },
            typical: "0; +100 makes the output rate 100 ppm higher",
            mutability: Mutability::Runtime,
            description: "Correction of the output rate in parts per million; changes only with \
                tunable = true.",
        },
    ],
    required: &["rate_out_hz"],
};

/// Configuration of [`ResampleBlock`] and [`ResampleRealBlock`].
///
/// ```
/// use sdrk_blocks::ResampleConfig;
/// let cfg = ResampleConfig { rate_out_hz: 8_000.0, ..ResampleConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResampleConfig {
    /// Output sample rate in Hz (a whole number).
    pub rate_out_hz: f64,
    /// Cutoff in Hz; `None` means `0.45 * min(rate_in, rate_out)`.
    pub cutoff_hz: Option<f64>,
    /// Transition width in Hz; `None` means `0.1 * min(rate_in, rate_out)`.
    pub transition_hz: Option<f64>,
    /// Stop-band attenuation in dB.
    pub attenuation_db: f64,
    /// Upper limit of the number of taps.
    pub max_taps: usize,
    /// Use the tunable 128-phase resampler.
    pub tunable: bool,
    /// Correction of the output rate in ppm (needs `tunable`).
    pub ppm: f64,
}

impl Default for ResampleConfig {
    fn default() -> Self {
        Self {
            rate_out_hz: 48_000.0,
            cutoff_hz: None,
            transition_hz: None,
            attenuation_db: 60.0,
            max_taps: 1023,
            tunable: false,
            ppm: 0.0,
        }
    }
}

/// What a configuration turns into for a given input rate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResPlan {
    pub(crate) rate_out: Rate,
    pub(crate) ratio: Ratio,
    pub(crate) spec: DesignSpec,
    pub(crate) mode: PlanMode,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PlanMode {
    /// Rational `l / m` with `taps` taps at `l` times the input rate.
    Rational { l: usize, m: usize, taps: usize },
    /// Tuned bank with `k` taps per phase and Kaiser `beta`.
    Tuned { k: usize, beta: f64 },
}

impl ResampleConfig {
    pub(crate) fn plan(&self, rate_in: Rate) -> Result<ResPlan> {
        let out = self.rate_out_hz;
        if !(1.0..=HZ_LIMIT).contains(&out) || libm::floor(out) != out {
            return Err(Error::InvalidParameter {
                name: "rate_out_hz",
                reason: "the output rate must be a whole number of Hz of at least 1",
            });
        }
        let rate_out = Rate::hz(out as u64)?;
        check_symmetric("ppm", self.ppm, 1000.0)?;
        if !self.tunable && self.ppm != 0.0 {
            return Err(Error::InvalidParameter {
                name: "ppm",
                reason: "a ppm correction needs tunable = true",
            });
        }
        let ratio = Ratio::new(
            rate_out
                .num()
                .checked_mul(rate_in.den())
                .ok_or(Error::InvalidParameter {
                    name: "rate_out_hz",
                    reason: "the rate ratio does not fit into 64 bits",
                })?,
            rate_in
                .num()
                .checked_mul(rate_out.den())
                .ok_or(Error::InvalidParameter {
                    name: "rate_out_hz",
                    reason: "the rate ratio does not fit into 64 bits",
                })?,
        )?;
        let fin = hz(rate_in);
        let low = fin.min(hz(rate_out));
        let spec = DesignSpec {
            cutoff_hz: self.cutoff_hz.unwrap_or(0.45 * low),
            transition_hz: self.transition_hz.unwrap_or(0.1 * low),
            attenuation_db: self.attenuation_db,
        };
        let max = low / 2.0 - spec.transition_hz / 2.0;
        if !(spec.transition_hz > 0.0
            && spec.cutoff_hz > 0.0
            && spec.cutoff_hz <= max * (1.0 + 1e-9))
        {
            return Err(Error::OutOfRange {
                name: "cutoff_hz",
                min: 0.0,
                max,
                got: spec.cutoff_hz,
            });
        }
        let mode = if self.tunable {
            let phases = TUNED_PHASES as f64;
            let (n_hi, beta) = kaiserord(spec.attenuation_db, spec.transition_hz, phases * fin)?;
            let k = n_hi.div_ceil(TUNED_PHASES);
            if k > self.max_taps {
                return Err(Error::OutOfRange {
                    name: "max_taps",
                    min: k as f64,
                    max: MAX_TAPS_LIMIT as f64,
                    got: self.max_taps as f64,
                });
            }
            PlanMode::Tuned { k, beta }
        } else {
            let (l, m) = (ratio.num(), ratio.den());
            if l > 1024 || m > 1024 {
                return Err(Error::InvalidParameter {
                    name: "rate_out_hz",
                    reason: "the ratio to the input rate needs a numerator and a denominator of \
                        at most 1024 in the rational mode",
                });
            }
            let (l, m) = (l as usize, m as usize);
            let taps = spec.num_taps(l as f64 * fin, self.max_taps)?;
            PlanMode::Rational { l, m, taps }
        };
        Ok(ResPlan {
            rate_out,
            ratio,
            spec,
            mode,
        })
    }

    /// Checks the configuration against the input sample rate.
    ///
    /// ```
    /// use sdrk_blocks::ResampleConfig;
    /// let rate = sdrk::Rate::hz(48_000)?;
    /// // 44100 / 48000 = 147 / 160 needs more than the default 1023 taps in the rational mode.
    /// let cfg = ResampleConfig { rate_out_hz: 44_100.0, ..ResampleConfig::default() };
    /// assert!(cfg.validate(rate).is_err());
    /// assert!(ResampleConfig { max_taps: 8192, ..cfg }.validate(rate).is_ok());
    /// assert!(ResampleConfig { tunable: true, ..cfg }.validate(rate).is_ok());
    /// # Ok::<(), sdrk::Error>(())
    /// ```
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.plan(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// ```
    /// use sdrk_blocks::{ParamValue, ResampleConfig};
    /// let mut cfg = ResampleConfig::default();
    /// cfg.set("rate_out_hz", ParamValue::F64(44_100.0))?;
    /// cfg.set("tunable", ParamValue::Bool(true))?;
    /// assert_eq!(cfg.get("tunable"), Ok(ParamValue::Bool(true)));
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "rate_out_hz" => self.rate_out_hz = value.as_f64()?,
            "cutoff_hz" => self.cutoff_hz = Some(value.as_f64()?),
            "transition_hz" => self.transition_hz = Some(value.as_f64()?),
            "attenuation_db" => self.attenuation_db = value.as_f64()?,
            "max_taps" => self.max_taps = value.as_usize()?,
            "tunable" => self.tunable = value.as_bool()?,
            "ppm" => self.ppm = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter. `cutoff_hz` and `transition_hz` are an error while they
    /// are not set (their default depends on the sample rates; the block reports the value in
    /// use).
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        let unset = |n: &'static str| Error::InvalidParameter {
            name: n,
            reason: "not set; the default depends on the sample rates",
        };
        match name {
            "rate_out_hz" => Ok(ParamValue::F64(self.rate_out_hz)),
            "cutoff_hz" => self
                .cutoff_hz
                .map(ParamValue::F64)
                .ok_or_else(|| unset("cutoff_hz")),
            "transition_hz" => self
                .transition_hz
                .map(ParamValue::F64)
                .ok_or_else(|| unset("transition_hz")),
            "attenuation_db" => Ok(ParamValue::F64(self.attenuation_db)),
            "max_taps" => Ok(ParamValue::I64(self.max_taps as i64)),
            "tunable" => Ok(ParamValue::Bool(self.tunable)),
            "ppm" => Ok(ParamValue::F64(self.ppm)),
            _ => unknown(),
        }
    }

    /// Bytes of storage for a sample type with elements of size `E` and taps of size `Tap`.
    fn storage<E: TunedElem>(&self, rate_in: Rate) -> Result<usize> {
        let plan = self.plan(rate_in)?;
        let mut p = Plan::default();
        match plan.mode {
            PlanMode::Rational { l, taps, .. } => {
                p.add::<E::Tap>(taps)?;
                p.add::<E>(taps.div_ceil(l))?;
            }
            PlanMode::Tuned { k, .. } => {
                p.add::<E::Tap>(TUNED_PHASES * k)?;
                p.add::<E>(k + WORK_EXTRA)?;
            }
        }
        Ok(p.bytes())
    }
}

/// The position step of the tuned resampler in 32.32 fixed point:
/// `round(2^32 rate_in / rate_out / (1 + ppm 1e-6))`.
fn tuned_step(rate_in: f64, rate_out: f64, ppm: f64) -> u64 {
    libm::floor(4_294_967_296.0 * rate_in / rate_out / (1.0 + ppm * 1e-6) + 0.5) as u64
}

/// The polyphase filter bank with a fractional read position.
struct TunedCore<E: TunedElem, S: Storage> {
    bank: S::Buf<E::Tap>,
    work: S::Buf<E>,
    k: usize,
    filled: usize,
    skip: usize,
    pos: u64,
    step: u64,
    rate_in: f64,
    rate_out: f64,
}

impl<E: TunedElem, S: Storage> TunedCore<E, S> {
    fn new(
        plan: &ResPlan,
        rate_in: Rate,
        ppm: f64,
        k: usize,
        beta: f64,
        storage: &mut S,
    ) -> Result<Self> {
        let phases = TUNED_PHASES as f64;
        let fin = hz(rate_in);
        let proto = Lowpass::new(
            plan.spec.cutoff_hz,
            phases * fin,
            DesignWindow::Kaiser { beta },
            TUNED_PHASES * k,
        )?;
        let mut bank = storage.alloc::<E::Tap>(TUNED_PHASES * k)?;
        for (idx, t) in bank.as_mut().iter_mut().enumerate() {
            // Phase p holds h[p + i P] reversed, so that a dot product with the oldest sample
            // first applies the newest sample to h[p].
            let (p, j) = (idx / k, idx % k);
            *t = E::tap(proto.tap(p + (k - 1 - j) * TUNED_PHASES) * phases)?;
        }
        let work = storage.alloc::<E>(k + WORK_EXTRA)?;
        let rate_out = hz(plan.rate_out);
        Ok(Self {
            bank,
            work,
            k,
            filled: k - 1,
            skip: 0,
            pos: ((k - 1) as u64) << 32,
            step: tuned_step(fin, rate_out, ppm),
            rate_in: fin,
            rate_out,
        })
    }

    fn reset(&mut self) {
        self.work.as_mut().fill(E::ZERO);
        self.filled = self.k - 1;
        self.skip = 0;
        self.pos = ((self.k - 1) as u64) << 32;
    }

    fn set_ppm(&mut self, ppm: f64) {
        self.step = tuned_step(self.rate_in, self.rate_out, ppm);
    }

    /// Skips input between two windows and copies input into the work buffer; returns the
    /// number of input samples taken (skipped ones included).
    fn fill(&mut self, input: &[E]) -> Result<usize> {
        let internal = || Error::Core(sdrk_core::Error::Internal("tuned resampler fill"));
        let skipped = self.skip.min(input.len());
        self.skip -= skipped;
        let rest = input.get(skipped..).ok_or_else(internal)?;
        let room = self.work.as_ref().len() - self.filled;
        let take = room.min(rest.len());
        let dst = self
            .work
            .as_mut()
            .get_mut(self.filled..self.filled + take)
            .ok_or_else(internal)?;
        dst.copy_from_slice(rest.get(..take).ok_or_else(internal)?);
        self.filled += take;
        Ok(skipped + take)
    }

    /// Computes the output at the read position, or `None` if its window is not complete.
    fn next_output(&self) -> Result<Option<E>> {
        let internal = || Error::Core(sdrk_core::Error::Internal("tuned resampler window"));
        let k = self.k;
        let index = (self.pos >> 32) as usize;
        let frac = self.pos & 0xFFFF_FFFF;
        let phase = (frac >> 25) as usize;
        let mu = ((frac >> 9) & 0xFFFF) as u32;
        let wrap = phase + 1 == TUNED_PHASES;
        if index + 1 + usize::from(wrap) > self.filled {
            return Ok(None);
        }
        let work = self.work.as_ref();
        let bank = self.bank.as_ref();
        let window = work.get(index + 1 - k..=index).ok_or_else(internal)?;
        let first = bank.get(phase * k..(phase + 1) * k).ok_or_else(internal)?;
        let a = E::dot(window, first)?;
        let b = if wrap {
            let later = work.get(index + 2 - k..=index + 1).ok_or_else(internal)?;
            E::dot(later, bank.get(..k).ok_or_else(internal)?)?
        } else {
            let second = bank
                .get((phase + 1) * k..(phase + 2) * k)
                .ok_or_else(internal)?;
            E::dot(window, second)?
        };
        Ok(Some(E::lerp(a, b, mu)))
    }

    /// Releases the samples no later window needs; returns how many were dropped.
    fn release(&mut self) -> usize {
        let drop_total = ((self.pos >> 32) as usize).saturating_sub(self.k - 1);
        let from_buffer = drop_total.min(self.filled);
        if from_buffer > 0 {
            self.work.as_mut().copy_within(from_buffer..self.filled, 0);
            self.filled -= from_buffer;
        }
        self.skip += drop_total - from_buffer;
        self.pos -= (drop_total as u64) << 32;
        drop_total
    }

    fn process(&mut self, input: &[E], output: &mut [E]) -> Result<Progress> {
        let (mut consumed, mut produced) = (0usize, 0usize);
        loop {
            let taken = self.fill(input.get(consumed..).unwrap_or_default())?;
            consumed += taken;
            let mut made = 0usize;
            while let Some(slot) = output.get_mut(produced) {
                let Some(value) = self.next_output()? else {
                    break;
                };
                *slot = value;
                self.pos = self.pos.saturating_add(self.step);
                produced += 1;
                made += 1;
            }
            let dropped = self.release();
            if consumed == input.len() || (taken == 0 && made == 0 && dropped == 0) {
                break;
            }
        }
        Ok(Progress { consumed, produced })
    }

    fn min_input(&self) -> usize {
        (self.step >> 32) as usize + 3
    }

    /// Number of outputs `j` whose position `j * step` lies before `rel_in * 2^32`.
    fn map_index(&self, rel_in: u64) -> u64 {
        let v = (u128::from(rel_in) << 32).div_ceil(u128::from(self.step.max(1)));
        u64::try_from(v).unwrap_or(u64::MAX)
    }
}

/// `ceil(rel_in * L / M)` for the exact ratio of a rational engine.
fn map_nominal(ratio: Ratio, rel_in: u64) -> u64 {
    let v = (u128::from(rel_in) * u128::from(ratio.num())).div_ceil(u128::from(ratio.den()));
    u64::try_from(v).unwrap_or(u64::MAX)
}

fn tuned_delay(k: usize) -> Delay {
    Delay::new((TUNED_PHASES * k - 1) as u64, 2 * TUNED_PHASES as u64).unwrap_or(Delay::ZERO)
}

/// Resampler for complex signals (`C32`, `CQ15`).
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ResampleBlock, ResampleConfig};
/// use sdrk::{Rate, C32};
/// let cfg = ResampleConfig { rate_out_hz: 2_000.0, ..ResampleConfig::default() };
/// let rate = Rate::hz(1_000)?;
/// let bytes = ResampleBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = ResampleBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 64];
/// let p = block.process(&[C32::new(1.0, 0.0); 16], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (16, 32)); // 1 kHz to 2 kHz doubles the count
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ResampleBlock<T: BlockSample, S: Storage> {
    config: ResampleConfig,
    ratio: Ratio,
    engine: Engine<T, S>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for ResampleBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResampleBlock")
            .field("config", &self.config)
            .field("ratio", &self.ratio)
            .finish_non_exhaustive()
    }
}

enum Engine<T: BlockSample, S: Storage> {
    Rational(ResampleState<T, S::Buf<T::Real>, S::Buf<T>>),
    Tuned(TunedCore<T, S>),
}

fn resolve(config: &ResampleConfig, plan: &ResPlan) -> ResampleConfig {
    let mut resolved = *config;
    resolved.cutoff_hz = Some(plan.spec.cutoff_hz);
    resolved.transition_hz = Some(plan.spec.transition_hz);
    resolved
}

impl<T: BlockSample, S: Storage> ResampleBlock<T, S> {
    /// Designs the filter and creates the block; filter, history and work buffers come from
    /// `storage`.
    pub fn new(config: &ResampleConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let plan = config.plan(rate_in)?;
        let engine = match plan.mode {
            PlanMode::Rational { l, m, taps } => {
                let lp = plan.spec.lowpass(hz(rate_in) * l as f64, config.max_taps)?;
                let mut t = storage.alloc::<T::Real>(taps)?;
                for (i, v) in t.as_mut().iter_mut().enumerate() {
                    *v = <T as TunedElem>::tap(lp.tap(i) * l as f64)?;
                }
                let hist = storage.alloc::<T>(taps.div_ceil(l))?;
                Engine::Rational(ResampleState::new(t, hist, l, m)?)
            }
            PlanMode::Tuned { k, beta } => Engine::Tuned(TunedCore::new(
                &plan, rate_in, config.ppm, k, beta, storage,
            )?),
        };
        Ok(Self {
            config: resolve(config, &plan),
            ratio: plan.ratio,
            engine,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &ResampleConfig, rate_in: Rate) -> Result<usize> {
        config.storage::<T>(rate_in)
    }

    /// Nominal output rate (`rate_out_hz`; `ppm` is not included).
    pub fn rate_out(config: &ResampleConfig, rate_in: Rate) -> Result<Rate> {
        Ok(config.plan(rate_in)?.rate_out)
    }
}

impl<T: BlockSample, S: Storage> Block for ResampleBlock<T, S>
where
    Sdrk: Resample<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        match &mut self.engine {
            Engine::Rational(state) => Ok(<Sdrk as Resample<T>>::resample(
                &Sdrk, state, input, output,
            )?),
            Engine::Tuned(core) => core.process(input, output),
        }
    }

    fn timing(&self) -> BlockTiming {
        match &self.engine {
            Engine::Rational(state) => BlockTiming {
                ratio: state.ratio(),
                kind: RatioKind::Constant,
                delay: state.group_delay(),
            },
            Engine::Tuned(core) => BlockTiming {
                ratio: self.ratio,
                kind: RatioKind::Variable,
                delay: tuned_delay(core.k),
            },
        }
    }

    fn reset(&mut self) {
        match &mut self.engine {
            Engine::Rational(state) => state.reset(),
            Engine::Tuned(core) => core.reset(),
        }
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "ppm" => match &mut self.engine {
                Engine::Tuned(core) => {
                    let ppm = value.as_f64()?;
                    core.set_ppm(ppm);
                    self.config.ppm = ppm;
                    Ok(())
                }
                Engine::Rational(_) => Err(Error::ReadOnlyParameter),
            },
            _ => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        match &self.engine {
            Engine::Rational(state) => state.down(),
            Engine::Tuned(core) => core.min_input(),
        }
    }

    fn map_index(&self, rel_in: u64) -> u64 {
        match &self.engine {
            Engine::Rational(_) => map_nominal(self.ratio, rel_in),
            Engine::Tuned(core) => core.map_index(rel_in),
        }
    }
}

/// Resampler for real signals (`f32`, `Q15`), for example audio.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ResampleConfig, ResampleRealBlock};
/// use sdrk::Rate;
/// let cfg = ResampleConfig { rate_out_hz: 1_000.0, ..ResampleConfig::default() };
/// let rate = Rate::hz(2_000)?;
/// let bytes = ResampleRealBlock::<f32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = ResampleRealBlock::<f32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [0.0f32; 64];
/// let p = block.process(&[1.0; 32], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (32, 16));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ResampleRealBlock<R: BlockReal, S: Storage> {
    config: ResampleConfig,
    ratio: Ratio,
    engine: RealEngine<R, S>,
}

impl<R: BlockReal, S: Storage> core::fmt::Debug for ResampleRealBlock<R, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResampleRealBlock")
            .field("config", &self.config)
            .field("ratio", &self.ratio)
            .finish_non_exhaustive()
    }
}

enum RealEngine<R: BlockReal, S: Storage> {
    Rational(ResampleRealState<R, S::Buf<R>, S::Buf<R>>),
    Tuned(TunedCore<R, S>),
}

impl<R: BlockReal, S: Storage> ResampleRealBlock<R, S> {
    /// Designs the filter and creates the block; filter, history and work buffers come from
    /// `storage`.
    pub fn new(config: &ResampleConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let plan = config.plan(rate_in)?;
        let engine = match plan.mode {
            PlanMode::Rational { l, m, taps } => {
                let lp = plan.spec.lowpass(hz(rate_in) * l as f64, config.max_taps)?;
                let mut t = storage.alloc::<R>(taps)?;
                for (i, v) in t.as_mut().iter_mut().enumerate() {
                    *v = <R as TunedElem>::tap(lp.tap(i) * l as f64)?;
                }
                let hist = storage.alloc::<R>(taps.div_ceil(l))?;
                RealEngine::Rational(ResampleRealState::new(t, hist, l, m)?)
            }
            PlanMode::Tuned { k, beta } => RealEngine::Tuned(TunedCore::new(
                &plan, rate_in, config.ppm, k, beta, storage,
            )?),
        };
        Ok(Self {
            config: resolve(config, &plan),
            ratio: plan.ratio,
            engine,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &ResampleConfig, rate_in: Rate) -> Result<usize> {
        config.storage::<R>(rate_in)
    }

    /// Nominal output rate (`rate_out_hz`; `ppm` is not included).
    pub fn rate_out(config: &ResampleConfig, rate_in: Rate) -> Result<Rate> {
        Ok(config.plan(rate_in)?.rate_out)
    }
}

impl<R: BlockReal, S: Storage> Block for ResampleRealBlock<R, S>
where
    Sdrk: ResampleReal<R>,
{
    type In = R;
    type Out = R;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[R], output: &mut [R]) -> Result<Progress> {
        match &mut self.engine {
            RealEngine::Rational(state) => Ok(<Sdrk as ResampleReal<R>>::resample_real(
                &Sdrk, state, input, output,
            )?),
            RealEngine::Tuned(core) => core.process(input, output),
        }
    }

    fn timing(&self) -> BlockTiming {
        match &self.engine {
            RealEngine::Rational(state) => BlockTiming {
                ratio: state.ratio(),
                kind: RatioKind::Constant,
                delay: state.group_delay(),
            },
            RealEngine::Tuned(core) => BlockTiming {
                ratio: self.ratio,
                kind: RatioKind::Variable,
                delay: tuned_delay(core.k),
            },
        }
    }

    fn reset(&mut self) {
        match &mut self.engine {
            RealEngine::Rational(state) => state.reset(),
            RealEngine::Tuned(core) => core.reset(),
        }
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "ppm" => match &mut self.engine {
                RealEngine::Tuned(core) => {
                    let ppm = value.as_f64()?;
                    core.set_ppm(ppm);
                    self.config.ppm = ppm;
                    Ok(())
                }
                RealEngine::Rational(_) => Err(Error::ReadOnlyParameter),
            },
            _ => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        match &self.engine {
            RealEngine::Rational(state) => state.down(),
            RealEngine::Tuned(core) => core.min_input(),
        }
    }

    fn map_index(&self, rel_in: u64) -> u64 {
        match &self.engine {
            RealEngine::Rational(_) => map_nominal(self.ratio, rel_in),
            RealEngine::Tuned(core) => core.map_index(rel_in),
        }
    }
}
