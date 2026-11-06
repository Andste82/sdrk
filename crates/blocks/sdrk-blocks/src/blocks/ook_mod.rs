//! `ook_mod`: on-off keying modulator, built from the kernels `fir_interpolate_real` and
//! `interleave`.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::interp::{FirInterpRealState, FirInterpolateReal};
use sdrk_core::kernels::math::Interleave;

use super::{check_count, head, head_mut, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Output samples per piece the block processes at a time (a piece holds
/// `max(1, OOK_MOD_CHUNK / samples_per_symbol)` symbols).
pub const OOK_MOD_CHUNK: usize = 256;

/// Largest `samples_per_symbol` (D638).
const MAX_SPS: usize = 4096;

/// Description of the `ook_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "ook_mod",
    summary: "Modulates soft bits to on-off keying: the carrier on for a 1, off for a 0.",
    details: "Takes soft bits (one f32 per bit, above 0 is a 1, as frame_encode outputs them) \
        and outputs samples_per_symbol complex samples per bit: the real value level for a 1 \
        and 0 for a 0, with rectangular steps (fir_interpolate_real with a boxcar of \
        samples_per_symbol taps, then interleave as the real part; the imaginary part is 0). \
        The output rate is the bit rate times samples_per_symbol. Put a duc or a rotate after \
        it to move the carrier off 0 Hz. Pitfall: the steps are hard, so the spectrum is wide \
        (sinc shaped); filter before a real transmitter.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft bits, one sample per bit; above 0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "Complex baseband, samples_per_symbol samples per bit: level or 0.",
    }],
    params: &[
        ParamDesc {
            name: "samples_per_symbol",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_SPS as i64,
                default: None,
            },
            typical: "500 for 2000 bit/s at 1 MHz",
            mutability: Mutability::Build,
            description: "Output samples per bit, 1 to 4096. Required.",
        },
        ParamDesc {
            name: "level",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0,
                default: Some(0.9),
            },
            typical: "0.9",
            mutability: Mutability::Build,
            description: "Amplitude of a 1 bit, 0 to 1.",
        },
    ],
    required: &["samples_per_symbol"],
};

/// Configuration of [`OokModBlock`].
///
/// ```
/// use sdrk_blocks::OokModConfig;
/// let cfg = OokModConfig { samples_per_symbol: 500, ..OokModConfig::default() };
/// assert_eq!(cfg.level, 0.9);
/// assert_eq!(cfg.rate_out(sdrk::Rate::hz(2_000)?)?, sdrk::Rate::hz(1_000_000)?);
/// assert!(OokModConfig { level: 1.5, ..cfg }.validate().is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OokModConfig {
    /// Output samples per bit, 1 to 4096 (required by the pipeline).
    pub samples_per_symbol: usize,
    /// Amplitude of a 1 bit, 0 to 1.
    pub level: f64,
}

impl Default for OokModConfig {
    /// `samples_per_symbol` 8 (a placeholder: the pipeline requires it), `level` 0.9.
    fn default() -> Self {
        Self {
            samples_per_symbol: 8,
            level: 0.9,
        }
    }
}

impl OokModConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for `samples_per_symbol` outside 1 to 4096 or `level` outside 0 to 1.
    pub fn validate(&self) -> Result<()> {
        check_count("samples_per_symbol", self.samples_per_symbol, 1, MAX_SPS)?;
        if (0.0..=1.0).contains(&self.level) {
            Ok(())
        } else {
            Err(Error::OutOfRange {
                name: "level",
                min: 0.0,
                max: 1.0,
                got: self.level,
            })
        }
    }

    /// The output rate: the input (bit) rate times `samples_per_symbol`.
    ///
    /// # Errors
    /// As [`OokModConfig::validate`].
    pub fn rate_out(&self, rate_in: Rate) -> Result<Rate> {
        self.validate()?;
        Ok(rate_in.scale(Ratio::new(self.samples_per_symbol as u64, 1)?)?)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "samples_per_symbol" => self.samples_per_symbol = value.as_usize()?,
            "level" => self.level = value.as_f64()?,
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
            "samples_per_symbol" => Ok(ParamValue::I64(self.samples_per_symbol as i64)),
            "level" => Ok(ParamValue::F64(self.level)),
            _ => unknown(),
        }
    }

    /// Symbols per piece: `OOK_MOD_CHUNK / samples_per_symbol`, at least 1.
    fn piece_symbols(&self) -> usize {
        (OOK_MOD_CHUNK / self.samples_per_symbol.max(1)).max(1)
    }
}

/// OOK modulator: `f32` soft bits in, `C32` out.
///
/// Per piece of at most `max(OOK_MOD_CHUNK, samples_per_symbol)` output samples the block maps
/// the bits to `level` or 0 (in Rust), calls `fir_interpolate_real` once (a boxcar of
/// `samples_per_symbol` taps 1.0, factor `samples_per_symbol`) and `interleave` once (real
/// part only). Every output sample is exactly `level as f32` or 0. A call takes whole
/// symbols: `min(input.len(), output.len() / samples_per_symbol)` of them. Timing: constant
/// ratio `samples_per_symbol`, delay `(samples_per_symbol - 1) / 2` output samples, given in
/// input bits.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, OokModBlock, OokModConfig};
/// let cfg = OokModConfig { samples_per_symbol: 4, level: 0.5 };
/// let rate = Rate::hz(1_000)?;
/// let bytes = OokModBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = OokModBlock::new(&cfg, rate, &mut arena)?;
/// let mut out = vec![C32::default(); 12];
/// let p = block.process(&[1.0, -1.0, 1.0], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (3, 12));
/// assert_eq!(out[..4], [C32::new(0.5, 0.0); 4]);
/// assert_eq!(out[4..8], [C32::new(0.0, 0.0); 4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct OokModBlock<S: Storage> {
    config: OokModConfig,
    level: f32,
    fir: FirInterpRealState<f32, S::Buf<f32>, S::Buf<f32>>,
    levels: S::Buf<f32>,
    shaped: S::Buf<f32>,
}

impl<S: Storage> core::fmt::Debug for OokModBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OokModBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> OokModBlock<S> {
    /// Creates the block; taps, history and the two work buffers come from `storage`.
    ///
    /// # Errors
    /// As [`OokModConfig::validate`]; `Storage` if the storage is too small.
    pub fn new(config: &OokModConfig, _rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let sps = config.samples_per_symbol;
        let mut taps = storage.alloc::<f32>(sps)?;
        taps.as_mut().fill(1.0);
        let hist = storage.alloc::<f32>(1)?;
        let syms = config.piece_symbols();
        let levels = storage.alloc::<f32>(syms)?;
        let shaped = storage.alloc::<f32>(syms * sps)?;
        Ok(Self {
            config: *config,
            level: config.level as f32,
            fir: FirInterpRealState::new(taps, hist, sps)?,
            levels,
            shaped,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`OokModConfig::validate`].
    pub fn storage_bytes(config: &OokModConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        let sps = config.samples_per_symbol;
        let syms = config.piece_symbols();
        let mut plan = Plan::default();
        plan.add::<f32>(sps)?;
        plan.add::<f32>(1)?;
        plan.add::<f32>(syms)?;
        plan.add::<f32>(syms * sps)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate times `samples_per_symbol`.
    ///
    /// # Errors
    /// As [`OokModConfig::validate`].
    pub fn rate_out(config: &OokModConfig, rate_in: Rate) -> Result<Rate> {
        config.rate_out(rate_in)
    }
}

impl<S: Storage> Block for OokModBlock<S>
where
    Sdrk: FirInterpolateReal<f32> + Interleave<C32>,
{
    type In = f32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> Result<Progress> {
        let sps = self.config.samples_per_symbol;
        let total = input.len().min(output.len() / sps);
        let piece = self.config.piece_symbols();
        let level = self.level;
        let mut done = 0usize;
        while done < total {
            let m = (total - done).min(piece);
            let bits = head(tail(input, done)?, m)?;
            let lv = head_mut(self.levels.as_mut(), m)?;
            for (l, &b) in lv.iter_mut().zip(bits) {
                *l = if b > 0.0 { level } else { 0.0 };
            }
            let shaped = head_mut(self.shaped.as_mut(), m * sps)?;
            let p = <Sdrk as FirInterpolateReal<f32>>::fir_interpolate_real(
                &Sdrk,
                &mut self.fir,
                head(self.levels.as_ref(), m)?,
                shaped,
            )?;
            if p.consumed != m || p.produced != m * sps {
                return Err(Error::Core(sdrk_core::Error::Internal("ook_mod piece")));
            }
            let out = head_mut(tail_mut(output, done * sps)?, m * sps)?;
            <Sdrk as Interleave<C32>>::interleave(
                &Sdrk,
                head(self.shaped.as_ref(), m * sps)?,
                None,
                out,
            )?;
            done += m;
        }
        Ok(Progress {
            consumed: total,
            produced: total * sps,
        })
    }

    fn timing(&self) -> BlockTiming {
        let sps = self.config.samples_per_symbol as u64;
        BlockTiming {
            ratio: Ratio::new(sps, 1).unwrap_or(Ratio::ONE),
            kind: RatioKind::Constant,
            // (sps - 1) / 2 output samples = (sps - 1) / (2 sps) input bits.
            delay: Delay::new(sps.saturating_sub(1), 2 * sps).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
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
