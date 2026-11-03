//! `nco`: a numerically controlled oscillator whose clock is an input stream.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::mix::{Nco, Oscillate};

use super::{HZ_LIMIT, check_count, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Description of the `nco` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "nco",
    summary: "Generates a complex tone at frequency_hz, one sample per input sample.",
    details: "A numerically controlled oscillator: for every input sample it outputs one complex \
        sample exp(j 2 pi k phase), where the phase advances by \
        frequency_hz / fs turns per sample and k is the harmonic. The input values are ignored; \
        the input only provides the clock, so the oscillator runs at the rate of whatever stream \
        is connected and a pipeline needs no source nodes. The input may be any stream of the \
        precision (C32 or f32 gives C32, CQ15 or Q15 gives CQ15). frequency_hz can change while \
        the block runs and the phase stays continuous. Two blocks on the same clock with \
        harmonic 1 and 2 give phase-locked tones at f and 2 f (a 19 kHz pilot and its 38 kHz \
        subcarrier). phase_turns sets the start phase; reset returns to it. Pitfall: the \
        frequency must be below half the sample rate, and the k-th harmonic aliases above \
        fs / (2 k).",
    inputs: &[PortDesc {
        name: "clock",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Any stream; only its length matters.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex tone at the input rate (C32 for C32 or f32 input, CQ15 for CQ15 \
            or Q15 input).",
    }],
    params: &[
        ParamDesc {
            name: "frequency_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: -HZ_LIMIT,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "19e3 for the stereo pilot at 192 kHz",
            mutability: Mutability::Runtime,
            description: "Frequency of the tone in Hz, strictly between minus and plus half the \
                sample rate. Required.",
        },
        ParamDesc {
            name: "harmonic",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 8,
                default: Some(1),
            },
            typical: "1; 2 for the 38 kHz subcarrier of a 19 kHz pilot",
            mutability: Mutability::Build,
            description: "The output is the oscillator at this multiple of the frequency, phase \
                locked to the harmonic 1 of the same clock, 1 to 8.",
        },
        ParamDesc {
            name: "phase_turns",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0,
                default: Some(0.0),
            },
            typical: "0; 0.25 starts at +j",
            mutability: Mutability::Build,
            description: "Start phase of the oscillator as a fraction of a turn, 0 to 1.",
        },
    ],
    required: &["frequency_hz"],
};

/// Configuration of [`NcoBlock`] and [`NcoRealBlock`].
///
/// ```
/// use sdrk_blocks::NcoConfig;
/// let cfg = NcoConfig { frequency_hz: 19_000.0, ..NcoConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(192_000)?).is_ok());
/// // The frequency must be below half the sample rate.
/// assert!(cfg.validate(sdrk::Rate::hz(38_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NcoConfig {
    /// Frequency in Hz, strictly within plus or minus half the sample rate.
    pub frequency_hz: f64,
    /// Harmonic, 1 to 8.
    pub harmonic: usize,
    /// Start phase as a fraction of a turn, 0 to 1.
    pub phase_turns: f64,
}

impl Default for NcoConfig {
    fn default() -> Self {
        Self {
            frequency_hz: 0.0,
            harmonic: 1,
            phase_turns: 0.0,
        }
    }
}

/// `Ok` when `|f| < fs / 2`.
fn check_frequency(frequency_hz: f64, rate_in: Rate) -> Result<()> {
    let half = hz(rate_in) / 2.0;
    if frequency_hz.abs() < half {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name: "frequency_hz",
            min: -half,
            max: half,
            got: frequency_hz,
        })
    }
}

/// The start phase in units of 2 pi / 2^32 (`1.0` turn wraps to 0).
fn start_phase(turns: f64) -> u32 {
    (libm::floor(turns * 4_294_967_296.0 + 0.5) as u64) as u32
}

impl NcoConfig {
    /// Checks the configuration against the sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        check_count("harmonic", self.harmonic, 1, 8)?;
        DESCRIPTION
            .param("phase_turns")
            .ok_or(Error::UnknownParameter)?
            .check(&ParamValue::F64(self.phase_turns))?;
        check_frequency(self.frequency_hz, rate_in)
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "frequency_hz" => self.frequency_hz = value.as_f64()?,
            "harmonic" => self.harmonic = value.as_usize()?,
            "phase_turns" => self.phase_turns = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "frequency_hz" => Ok(ParamValue::F64(self.frequency_hz)),
            "harmonic" => Ok(ParamValue::I64(self.harmonic as i64)),
            "phase_turns" => Ok(ParamValue::F64(self.phase_turns)),
            _ => unknown(),
        }
    }
}

/// State shared by the two oscillator blocks.
#[derive(Debug)]
struct NcoCore {
    config: NcoConfig,
    rate_in: Rate,
    nco: Nco,
    start: u32,
}

impl NcoCore {
    fn new(config: &NcoConfig, rate_in: Rate) -> Result<Self> {
        config.validate(rate_in)?;
        let start = start_phase(config.phase_turns);
        let mut nco = Nco::new(config.frequency_hz, rate_in)?;
        nco.set_phase(start);
        Ok(Self {
            config: *config,
            rate_in,
            nco,
            start,
        })
    }

    fn reset(&mut self) {
        self.nco.set_phase(self.start);
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "frequency_hz" => {
                let f = value.as_f64()?;
                check_frequency(f, self.rate_in)?;
                self.nco.set_frequency(f, self.rate_in)?;
                self.config.frequency_hz = f;
                Ok(())
            }
            _ => unknown(),
        }
    }

    fn generate<T: BlockSample>(&mut self, n: usize, output: &mut [T]) -> Result<Progress>
    where
        Sdrk: Oscillate<T>,
    {
        let n = n.min(output.len());
        <Sdrk as Oscillate<T>>::oscillate(
            &Sdrk,
            &mut self.nco,
            self.config.harmonic as u32,
            head_mut(output, n)?,
        )?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
}

/// Oscillator clocked by a complex stream: `C32` to `C32`, `CQ15` to `CQ15`. The values of the
/// input are ignored.
///
/// A frequency change keeps the phase. For a real clock (`f32` or `Q15`) use [`NcoRealBlock`].
/// The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, NcoBlock, NcoConfig, ParamValue};
/// use sdrk::{Rate, C32};
/// let rate = Rate::hz(48_000)?;
/// let cfg = NcoConfig { frequency_hz: 12_000.0, ..NcoConfig::default() }; // a quarter turn per sample
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = NcoBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 3];
/// let p = block.process(&[C32::default(); 3], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (3, 3));
/// assert!((out[0].re - 1.0).abs() < 1e-6 && (out[1].im - 1.0).abs() < 1e-6);
/// block.set_param("frequency_hz", ParamValue::F64(6_000.0))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct NcoBlock<T: BlockSample, S: Storage> {
    core: NcoCore,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> NcoBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &NcoConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Ok(Self {
            core: NcoCore::new(config, rate_in)?,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &NcoConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate (equal to the input rate).
    pub fn rate_out(config: &NcoConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for NcoBlock<T, S>
where
    Sdrk: Oscillate<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        self.core.generate(input.len(), output)
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {
        self.core.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        self.core.set_param(name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.core.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}

/// Oscillator clocked by a real stream: `f32` to `C32`, `Q15` to `CQ15` (`T` is the complex
/// output type). The values of the input are ignored.
///
/// Otherwise like [`NcoBlock`]. The block needs no storage.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, NcoConfig, NcoRealBlock};
/// use sdrk::{Rate, C32};
/// let rate = Rate::hz(48_000)?;
/// let cfg = NcoConfig { frequency_hz: 12_000.0, ..NcoConfig::default() };
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut block = NcoRealBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 2];
/// // The clock is audio; its values do not matter.
/// let p = block.process(&[0.3f32, -0.7], &mut out)?;
/// assert_eq!(p.produced, 2);
/// assert!((out[1].im - 1.0).abs() < 1e-6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct NcoRealBlock<T: BlockSample, S: Storage> {
    core: NcoCore,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> NcoRealBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &NcoConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Ok(Self {
            core: NcoCore::new(config, rate_in)?,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    pub fn storage_bytes(config: &NcoConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate (equal to the input rate).
    pub fn rate_out(config: &NcoConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for NcoRealBlock<T, S>
where
    Sdrk: Oscillate<T>,
{
    type In = T::Real;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T::Real], output: &mut [T]) -> Result<Progress> {
        self.core.generate(input.len(), output)
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {
        self.core.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        self.core.set_param(name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.core.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
