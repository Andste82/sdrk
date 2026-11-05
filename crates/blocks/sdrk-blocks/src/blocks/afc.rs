//! `afc`: automatic frequency control, a closed loop of the kernels `rotate` and `quad_demod`
//! that measures the carrier offset and removes it.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::kernels::mix::{Nco, Rotate};

use super::{HZ_LIMIT, head, head_mut, hz, tail, tail_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Samples per piece: the estimate is updated once per piece (`P` of the specification).
pub const AFC_PIECE: usize = 256;

/// Smallest `time_constant_s`.
const MIN_TIME_CONSTANT_S: f64 = 1e-6;
/// Largest `time_constant_s`.
const MAX_TIME_CONSTANT_S: f64 = 1e6;

/// Description of the `afc` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "afc",
    summary: "Automatic frequency control: measures the carrier offset of a signal and shifts \
        it to 0 Hz.",
    details: "A closed loop over pieces of 256 samples: the piece is rotated by minus the \
        current estimate f_est (rotate, phase continuous), quad_demod of the rotated piece \
        gives the remaining instantaneous frequency in Hz, and its mean r (summed in f64) \
        moves the estimate: f_est = clamp(f_est + k r, -max_offset_hz, +max_offset_hz) with \
        k = 1 - exp(-256 / (fs time_constant_s)), so the loop settles with time_constant_s. \
        The output is the rotated signal (ratio 1, no delay). While hold is set the estimate \
        stays where it is (for example during a packet, so the data cannot pull it). The \
        block reports offset_hz (f_est) and offset_ppm (f_est / carrier_hz 1e6; NaN while \
        carrier_hz is 0) as readings in the pipeline statistics. It works for signals whose \
        mean instantaneous frequency is the carrier: FSK with balanced data, FM, AM. It does \
        not work for SSB (the mean frequency is the voice, not the carrier) or for data with \
        long runs of one symbol. Place it after the ddc and before fsk_demod. A piece whose \
        mean is NaN or infinite leaves the estimate unchanged. Pitfall: max_offset_hz must \
        cover the largest offset, and the signal must stay inside the ddc filter at that \
        offset.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex baseband with a carrier offset.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The input shifted by minus the estimated offset, at the same rate.",
    }],
    params: &[
        ParamDesc {
            name: "time_constant_s",
            unit: Unit::Seconds,
            kind: ParamKind::F64 {
                min: MIN_TIME_CONSTANT_S,
                max: MAX_TIME_CONSTANT_S,
                default: Some(0.5),
            },
            typical: "0.5; 0.05 for fast acquisition",
            mutability: Mutability::Build,
            description: "Time constant of the loop in seconds.",
        },
        ParamDesc {
            name: "max_offset_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "20000 for 20 ppm at 868 MHz",
            mutability: Mutability::Build,
            description: "Largest offset the estimate may reach in Hz, below half the sample \
                rate. Required.",
        },
        ParamDesc {
            name: "hold",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true while a packet is received",
            mutability: Mutability::Runtime,
            description: "Freezes the estimate (the signal is still shifted by it).",
        },
        ParamDesc {
            name: "carrier_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 0.0,
                max: HZ_LIMIT,
                default: Some(0.0),
            },
            typical: "868e6; 0 when the RF frequency is not known",
            mutability: Mutability::Build,
            description: "RF frequency of the carrier in Hz, only for the reading offset_ppm.",
        },
    ],
    required: &["max_offset_hz"],
};

/// Configuration of [`AfcBlock`].
///
/// ```
/// use sdrk_blocks::AfcConfig;
/// let cfg = AfcConfig { max_offset_hz: 20_000.0, ..AfcConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(200_000)?).is_ok());
/// // the limit must stay below half the sample rate
/// assert!(AfcConfig { max_offset_hz: 100_000.0, ..cfg }.validate(sdrk::Rate::hz(200_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AfcConfig {
    /// Time constant of the loop in seconds.
    pub time_constant_s: f64,
    /// Largest offset of the estimate in Hz (required by the pipeline; below `fs / 2`).
    pub max_offset_hz: f64,
    /// Freezes the estimate (changeable at run time).
    pub hold: bool,
    /// RF frequency of the carrier in Hz for the ppm reading (0: unknown).
    pub carrier_hz: f64,
}

impl Default for AfcConfig {
    /// `time_constant_s` 0.5, `max_offset_hz` 1000 (a placeholder: the pipeline requires
    /// it), `hold` false, `carrier_hz` 0.
    fn default() -> Self {
        Self {
            time_constant_s: 0.5,
            max_offset_hz: 1_000.0,
            hold: false,
            carrier_hz: 0.0,
        }
    }
}

impl AfcConfig {
    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `OutOfRange` for `time_constant_s` outside 1e-6 to 1e6 s, `max_offset_hz` not between
    /// 0 and half the sample rate (both excluded) or a negative or non-finite `carrier_hz`.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        let tc = self.time_constant_s;
        if !(MIN_TIME_CONSTANT_S..=MAX_TIME_CONSTANT_S).contains(&tc) {
            return Err(Error::OutOfRange {
                name: "time_constant_s",
                min: MIN_TIME_CONSTANT_S,
                max: MAX_TIME_CONSTANT_S,
                got: tc,
            });
        }
        let half = hz(rate_in) / 2.0;
        let m = self.max_offset_hz;
        if !(m > 0.0 && m < half) {
            return Err(Error::OutOfRange {
                name: "max_offset_hz",
                min: 0.0,
                max: half,
                got: m,
            });
        }
        let c = self.carrier_hz;
        if !(0.0..=HZ_LIMIT).contains(&c) {
            return Err(Error::OutOfRange {
                name: "carrier_hz",
                min: 0.0,
                max: HZ_LIMIT,
                got: c,
            });
        }
        Ok(())
    }

    /// The loop gain per piece, `k = 1 - exp(-256 / (fs time_constant_s))`.
    ///
    /// ```
    /// use sdrk_blocks::AfcConfig;
    /// let k = AfcConfig::default().gain(sdrk::Rate::hz(512)?); // one piece per 0.5 s
    /// assert!((k - (1.0 - (-1.0f64).exp())).abs() < 1e-15);
    /// # Ok::<(), sdrk::Error>(())
    /// ```
    pub fn gain(&self, rate_in: Rate) -> f64 {
        1.0 - libm::exp(-(AFC_PIECE as f64) / (hz(rate_in) * self.time_constant_s))
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "time_constant_s" => self.time_constant_s = value.as_f64()?,
            "max_offset_hz" => self.max_offset_hz = value.as_f64()?,
            "hold" => self.hold = value.as_bool()?,
            "carrier_hz" => self.carrier_hz = value.as_f64()?,
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
            "time_constant_s" => Ok(ParamValue::F64(self.time_constant_s)),
            "max_offset_hz" => Ok(ParamValue::F64(self.max_offset_hz)),
            "hold" => Ok(ParamValue::Bool(self.hold)),
            "carrier_hz" => Ok(ParamValue::F64(self.carrier_hz)),
            _ => unknown(),
        }
    }
}

/// Automatic frequency control for `C32` (no Q15 variant, D529).
///
/// Per piece of [`AFC_PIECE`] samples (counted from construction or `reset`, so the result
/// does not depend on how the input is split) the block calls `rotate` and `quad_demod`
/// (gain `fs / 2 pi`, so the output is in Hz); the mean of the piece in `f64` updates the
/// estimate. A call that ends inside a piece processes the samples it has and keeps the
/// partial sum. Timing: ratio 1, delay 0. The work buffer holds one piece of `f32`.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{AfcBlock, AfcConfig, ArenaStorage, Block, ParamValue};
/// let cfg = AfcConfig { max_offset_hz: 5_000.0, time_constant_s: 0.01, carrier_hz: 100e6, ..AfcConfig::default() };
/// let rate = Rate::hz(48_000)?;
/// let bytes = AfcBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut afc = AfcBlock::new(&cfg, rate, &mut arena)?;
/// // A carrier 1 kHz above 0 Hz, for 0.2 s.
/// let w = 2.0 * core::f64::consts::PI * 1_000.0 / 48_000.0;
/// let input: Vec<C32> = (0..9600).map(|n| C32::new((w * n as f64).cos() as f32, (w * n as f64).sin() as f32)).collect();
/// let mut out = vec![C32::default(); 9600];
/// afc.process(&input, &mut out)?;
/// assert!((afc.offset_hz() - 1_000.0).abs() < 1.0);
/// assert!((afc.offset_ppm() - 10.0).abs() < 0.01); // 1 kHz at 100 MHz
/// afc.set_param("hold", ParamValue::Bool(true))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AfcBlock<S: Storage> {
    config: AfcConfig,
    rate: Rate,
    k: f64,
    gain: f32,
    nco: Nco,
    quad: QuadDemodState<C32>,
    /// The estimate in Hz.
    f_est: f64,
    /// Sum of the frequencies of the current piece.
    sum: f64,
    /// Samples of the current piece so far.
    count: usize,
    work: S::Buf<f32>,
}

impl<S: Storage> core::fmt::Debug for AfcBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AfcBlock")
            .field("config", &self.config)
            .field("f_est", &self.f_est)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> AfcBlock<S> {
    /// Creates the block with the estimate 0 Hz; the work buffer comes from `storage`.
    ///
    /// # Errors
    /// As [`AfcConfig::validate`]; `Storage` if the storage is too small.
    pub fn new(config: &AfcConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let work = storage.alloc::<f32>(AFC_PIECE)?;
        Ok(Self {
            config: *config,
            rate: rate_in,
            k: config.gain(rate_in),
            gain: C32::demod_gain(1.0, hz(rate_in)),
            nco: Nco::new(0.0, rate_in)?,
            quad: QuadDemodState::new(),
            f_est: 0.0,
            sum: 0.0,
            count: 0,
            work,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`AfcConfig::validate`].
    pub fn storage_bytes(config: &AfcConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<f32>(AFC_PIECE)?;
        Ok(plan.bytes())
    }

    /// Output rate (equal to the input rate).
    ///
    /// # Errors
    /// As [`AfcConfig::validate`].
    pub fn rate_out(config: &AfcConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The estimated carrier offset in Hz (the signal is shifted by minus this value).
    pub fn offset_hz(&self) -> f64 {
        self.f_est
    }

    /// The estimated offset relative to `carrier_hz` in ppm; NaN while `carrier_hz` is 0.
    pub fn offset_ppm(&self) -> f64 {
        if self.config.carrier_hz > 0.0 {
            self.f_est / self.config.carrier_hz * 1e6
        } else {
            f64::NAN
        }
    }

    /// Ends a piece: updates the estimate (unless `hold` is set or the mean is not finite)
    /// and retunes the oscillator.
    fn update(&mut self) -> Result<()> {
        let r = self.sum / AFC_PIECE as f64;
        self.sum = 0.0;
        self.count = 0;
        if self.config.hold || !r.is_finite() {
            return Ok(());
        }
        let lim = self.config.max_offset_hz;
        self.f_est = (self.f_est + self.k * r).max(-lim).min(lim);
        self.nco.set_frequency(-self.f_est, self.rate)?;
        Ok(())
    }
}

impl<S: Storage> Block for AfcBlock<S>
where
    Sdrk: Rotate<C32> + QuadDemod<C32>,
{
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        let mut done = 0usize;
        while done < n {
            let m = (n - done).min(AFC_PIECE - self.count);
            let x = head(tail(input, done)?, m)?;
            let y = head_mut(tail_mut(output, done)?, m)?;
            <Sdrk as Rotate<C32>>::rotate(&Sdrk, &mut self.nco, x, y)?;
            let freq = head_mut(self.work.as_mut(), m)?;
            <Sdrk as QuadDemod<C32>>::quad_demod(&Sdrk, &mut self.quad, y, freq, self.gain)?;
            for &f in freq.iter() {
                self.sum += f64::from(f);
            }
            self.count += m;
            if self.count == AFC_PIECE {
                self.update()?;
            }
            done += m;
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {
        self.quad.reset();
        self.nco = Nco::default();
        self.f_est = 0.0;
        self.sum = 0.0;
        self.count = 0;
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "hold" => {
                self.config.hold = value.as_bool()?;
                Ok(())
            }
            _ => unknown(),
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
            0 => Some(("offset_hz", self.offset_hz())),
            1 => Some(("offset_ppm", self.offset_ppm())),
            _ => None,
        }
    }
}
