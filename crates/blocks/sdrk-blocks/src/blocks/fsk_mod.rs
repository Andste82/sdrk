//! `fsk_mod`: 2-FSK and 4-FSK modulator with an optional Gaussian pulse (GFSK, GMSK), built from
//! the kernels `fir_interpolate_real` and `fm_mod`.

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::interp::{FirInterpRealState, FirInterpolateReal};
use sdrk_core::kernels::modulate::{FmMod, FmModState};

use super::{HZ_LIMIT, check_count, head, head_mut, hz, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::Gaussian;
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Output samples per piece the block processes at a time (the size of its shaping buffer;
/// a piece holds `FSK_MOD_CHUNK / samples_per_symbol` symbols).
pub const FSK_MOD_CHUNK: usize = 256;

/// Largest `samples_per_symbol`.
const MAX_SPS: usize = 64;

/// The level of the inner 4-FSK symbols, 1/3 (rounded once to `f32`).
const INNER: f32 = 1.0 / 3.0;

/// Description of the `fsk_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fsk_mod",
    summary: "Modulates soft bits to 2-FSK or 4-FSK, with an optional Gaussian pulse for GFSK \
        and GMSK.",
    details: "Takes soft bits (one f32 per bit, above 0 is a 1, as frame_encode outputs them) \
        and maps them to levels: 2 levels give +1 for a 1 and -1 for a 0 (one bit per symbol); \
        4 levels take two bits b0 b1 per symbol (b0 first) and give 01 -> +1, 00 -> +1/3, 10 \
        -> -1/3, 11 -> -1 (the M17 and DMR convention). The levels are shaped by \
        fir_interpolate_real (factor samples_per_symbol) with a boxcar of samples_per_symbol \
        taps (bt 0) or that boxcar convolved with a Gaussian of bandwidth-time product bt over \
        span symbols (GFSK; bt 0.5 with deviation_hz = symbol rate / 4 is GMSK), scaled so \
        that a constant level keeps its value, and fm_mod turns them into a unit-magnitude \
        C32 signal: level +1 is +deviation_hz (for 4-FSK the outer symbols), +1/3 a third of \
        it. The output rate is the bit rate times samples_per_symbol / bits per symbol. The \
        deviation can change while the block runs; the phase stays continuous. Put a duc \
        after it to reach the transmit rate. Pitfalls: the deviation must stay below half the \
        output rate; the output is delayed by the half length of the pulse filter.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft bits, one sample per bit; above 0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "Complex baseband, samples_per_symbol samples per symbol.",
    }],
    params: &[
        ParamDesc {
            name: "levels",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 2,
                max: 4,
                default: Some(2),
            },
            typical: "2; 4 for M17 and DMR",
            mutability: Mutability::Build,
            description: "Number of frequencies: 2 (one bit per symbol) or 4 (two bits).",
        },
        ParamDesc {
            name: "samples_per_symbol",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 2,
                max: MAX_SPS as i64,
                default: None,
            },
            typical: "8; 4 before a duc",
            mutability: Mutability::Build,
            description: "Output samples per symbol, 2 to 64. Required.",
        },
        ParamDesc {
            name: "deviation_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 0.0,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "symbol rate / 2 for 2-FSK with h = 1, symbol rate / 4 for GMSK",
            mutability: Mutability::Runtime,
            description: "Frequency of the level +1 in Hz; at most half the output rate. \
                Required.",
        },
        ParamDesc {
            name: "bt",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 1.0,
                default: Some(0.0),
            },
            typical: "0 for FSK, 0.5 for GFSK and GMSK, 0.3 for GSM",
            mutability: Mutability::Build,
            description: "Bandwidth-time product of the Gaussian pulse: 0 (no Gaussian, \
                rectangular frequency steps) or 0.2 to 1.",
        },
        ParamDesc {
            name: "span",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 2,
                max: 8,
                default: Some(4),
            },
            typical: "4",
            mutability: Mutability::Build,
            description: "Length of the Gaussian in symbols, 2 to 8 (not used with bt 0).",
        },
    ],
    required: &["samples_per_symbol", "deviation_hz"],
};

/// Configuration of [`FskModBlock`].
///
/// ```
/// use sdrk_blocks::FskModConfig;
/// let cfg = FskModConfig { samples_per_symbol: 4, deviation_hz: 25_000.0, bt: 0.5, ..FskModConfig::default() };
/// let rate = sdrk::Rate::hz(50_000)?; // the bit rate
/// assert!(cfg.validate(rate).is_ok());
/// assert_eq!(cfg.num_taps()?, 20); // (span + 1) samples_per_symbol
/// assert_eq!(cfg.rate_out(rate)?, sdrk::Rate::hz(200_000)?);
/// // 3 levels and a bt between 0 and 0.2 are rejected
/// assert!(FskModConfig { levels: 3, ..cfg }.validate(rate).is_err());
/// assert!(FskModConfig { bt: 0.1, ..cfg }.validate(rate).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FskModConfig {
    /// Number of frequencies, 2 or 4.
    pub levels: usize,
    /// Output samples per symbol, 2 to 64 (required by the pipeline).
    pub samples_per_symbol: usize,
    /// Frequency of the level +1 in Hz (required by the pipeline; changeable at run time).
    pub deviation_hz: f64,
    /// Bandwidth-time product of the Gaussian pulse: 0 (none) or 0.2 to 1.
    pub bt: f64,
    /// Length of the Gaussian in symbols, 2 to 8.
    pub span: usize,
}

impl Default for FskModConfig {
    /// `levels` 2, `samples_per_symbol` 8 and `deviation_hz` 600 (placeholders: the pipeline
    /// requires both), `bt` 0, `span` 4.
    fn default() -> Self {
        Self {
            levels: 2,
            samples_per_symbol: 8,
            deviation_hz: 600.0,
            bt: 0.0,
            span: 4,
        }
    }
}

/// Checks `levels` (2 or 4) and returns the bits per symbol.
pub(crate) fn bits_per_symbol(levels: usize) -> Result<usize> {
    match levels {
        2 => Ok(1),
        4 => Ok(2),
        _ => Err(Error::InvalidParameter {
            name: "levels",
            reason: "must be 2 or 4",
        }),
    }
}

impl FskModConfig {
    /// Bits per symbol: 1 for 2 levels, 2 for 4 levels.
    ///
    /// # Errors
    /// `InvalidParameter` for `levels` other than 2 and 4.
    pub fn bits_per_symbol(&self) -> Result<usize> {
        bits_per_symbol(self.levels)
    }

    /// Checks everything that does not depend on the rate and returns the bits per symbol.
    fn check(&self) -> Result<usize> {
        let bps = self.bits_per_symbol()?;
        check_count("samples_per_symbol", self.samples_per_symbol, 2, MAX_SPS)?;
        check_count("span", self.span, 2, 8)?;
        if !(self.bt == 0.0 || (0.2..=1.0).contains(&self.bt)) {
            return Err(Error::OutOfRange {
                name: "bt",
                min: 0.2,
                max: 1.0,
                got: self.bt,
            });
        }
        Ok(bps)
    }

    /// Number of taps of the pulse filter: `samples_per_symbol` for `bt` 0, otherwise the
    /// length of the boxcar convolved with the Gaussian, `(span + 1) samples_per_symbol`.
    ///
    /// # Errors
    /// As [`FskModConfig::validate`] without the rate checks.
    pub fn num_taps(&self) -> Result<usize> {
        self.check()?;
        Ok(if self.bt == 0.0 {
            self.samples_per_symbol
        } else {
            (self.span + 1) * self.samples_per_symbol
        })
    }

    /// The output rate: the input (bit) rate times `samples_per_symbol / bits_per_symbol`.
    ///
    /// # Errors
    /// As [`FskModConfig::validate`].
    pub fn rate_out(&self, rate_in: Rate) -> Result<Rate> {
        let bps = self.check()?;
        let ratio = Ratio::new(self.samples_per_symbol as u64, bps as u64)?;
        let out = rate_in.scale(ratio)?;
        let half = hz(out) / 2.0;
        if !(self.deviation_hz >= 0.0 && self.deviation_hz <= half) {
            return Err(Error::OutOfRange {
                name: "deviation_hz",
                min: 0.0,
                max: half,
                got: self.deviation_hz,
            });
        }
        Ok(out)
    }

    /// Checks the configuration against the input (bit) rate.
    ///
    /// # Errors
    /// `InvalidParameter` for `levels` other than 2 and 4; `OutOfRange` for
    /// `samples_per_symbol`, `span`, `bt` (neither 0 nor 0.2 to 1) or a `deviation_hz` above
    /// half the output rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.rate_out(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "levels" => self.levels = value.as_usize()?,
            "samples_per_symbol" => self.samples_per_symbol = value.as_usize()?,
            "deviation_hz" => self.deviation_hz = value.as_f64()?,
            "bt" => self.bt = value.as_f64()?,
            "span" => self.span = value.as_usize()?,
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
            "levels" => Ok(ParamValue::I64(self.levels as i64)),
            "samples_per_symbol" => Ok(ParamValue::I64(self.samples_per_symbol as i64)),
            "deviation_hz" => Ok(ParamValue::F64(self.deviation_hz)),
            "bt" => Ok(ParamValue::F64(self.bt)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            _ => unknown(),
        }
    }

    /// Symbols per piece: `FSK_MOD_CHUNK / samples_per_symbol` (at least 4).
    fn piece_symbols(&self) -> usize {
        (FSK_MOD_CHUNK / self.samples_per_symbol.max(1)).max(1)
    }
}

/// Writes the pulse taps (boxcar, or boxcar convolved with the Gaussian) scaled to sum
/// `sps`, in `f64` and rounded once to `f32`. `taps.len()` is `cfg.num_taps()`.
fn pulse_taps(cfg: &FskModConfig, taps: &mut [f32]) -> Result<()> {
    let sps = cfg.samples_per_symbol;
    if cfg.bt == 0.0 {
        for t in taps.iter_mut() {
            *t = 1.0;
        }
        return Ok(());
    }
    let g = Gaussian::new(cfg.bt, sps, cfg.span)?;
    // c[k] = sum of g[j] for j = k - sps + 1 ..= k (inside 0 .. len), in that order.
    let conv = |k: usize| -> f64 {
        let first = k.saturating_sub(sps - 1);
        let last = k.min(g.len() - 1);
        let mut c = 0.0f64;
        for j in first..=last {
            c += g.tap(j);
        }
        c
    };
    let mut sum = 0.0f64;
    for k in 0..taps.len() {
        sum += conv(k);
    }
    let scale = sps as f64 / sum;
    for (k, t) in taps.iter_mut().enumerate() {
        *t = (conv(k) * scale) as f32;
    }
    Ok(())
}

/// FSK modulator: `f32` soft bits in, `C32` out (no Q15 variant, D529).
///
/// Per piece of at most [`FSK_MOD_CHUNK`] output samples the block maps the bits to levels
/// (in Rust), calls `fir_interpolate_real` once (the pulse filter, factor
/// `samples_per_symbol`) and `fm_mod` once. A call takes whole symbols: `min(input.len() /
/// bits_per_symbol, output.len() / samples_per_symbol)` of them. Timing: constant ratio
/// `samples_per_symbol / bits_per_symbol`, delay `(taps - 1) / 2` output samples, given in
/// input bits.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, FskModBlock, FskModConfig};
/// let cfg = FskModConfig { samples_per_symbol: 8, deviation_hz: 1_000.0, ..FskModConfig::default() };
/// let rate = Rate::hz(2_000)?; // 2000 bit/s, 16 kHz out
/// let bytes = FskModBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = FskModBlock::new(&cfg, rate, &mut arena)?;
/// let mut out = vec![C32::default(); 32];
/// let p = block.process(&[1.0, 1.0, -1.0, -1.0], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (4, 32));
/// // A 1 at +1 kHz turns by 1/16 of a cycle per sample.
/// let z = out[2] * out[1].conj();
/// let step = z.im.atan2(z.re);
/// assert!((step - core::f32::consts::PI / 8.0).abs() < 1e-4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FskModBlock<S: Storage> {
    config: FskModConfig,
    bps: usize,
    rate_out: Rate,
    fir: FirInterpRealState<f32, S::Buf<f32>, S::Buf<f32>>,
    fm: FmModState,
    levels: S::Buf<f32>,
    shaped: S::Buf<f32>,
}

impl<S: Storage> core::fmt::Debug for FskModBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FskModBlock")
            .field("config", &self.config)
            .field("rate_out", &self.rate_out)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> FskModBlock<S> {
    /// Designs the pulse and creates the block; taps, history and the two work buffers come
    /// from `storage`. The oscillator starts at phase 0.
    ///
    /// # Errors
    /// As [`FskModConfig::validate`]; `Storage` if the storage is too small.
    pub fn new(config: &FskModConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let rate_out = config.rate_out(rate_in)?;
        let bps = config.bits_per_symbol()?;
        let n = config.num_taps()?;
        let sps = config.samples_per_symbol;
        let mut taps = storage.alloc::<f32>(n)?;
        pulse_taps(config, taps.as_mut())?;
        let hist = storage.alloc::<f32>(n.div_ceil(sps))?;
        let syms = config.piece_symbols();
        let levels = storage.alloc::<f32>(syms)?;
        let shaped = storage.alloc::<f32>(syms * sps)?;
        Ok(Self {
            config: *config,
            bps,
            rate_out,
            fir: FirInterpRealState::new(taps, hist, sps)?,
            fm: FmModState::new(config.deviation_hz, rate_out)?,
            levels,
            shaped,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`FskModConfig::validate`].
    pub fn storage_bytes(config: &FskModConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        let n = config.num_taps()?;
        let sps = config.samples_per_symbol;
        let syms = config.piece_symbols();
        let mut plan = Plan::default();
        plan.add::<f32>(n)?;
        plan.add::<f32>(n.div_ceil(sps))?;
        plan.add::<f32>(syms)?;
        plan.add::<f32>(syms * sps)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate times `samples_per_symbol / bits_per_symbol`.
    ///
    /// # Errors
    /// As [`FskModConfig::validate`].
    pub fn rate_out(config: &FskModConfig, rate_in: Rate) -> Result<Rate> {
        config.rate_out(rate_in)
    }

    /// The pulse taps (`f32`, sum `samples_per_symbol` up to rounding).
    pub fn taps(&self) -> &[f32] {
        self.fir.taps()
    }

    /// Maps the bits of one piece to levels.
    fn map_levels(&mut self, bits: &[f32], syms: usize) -> Result<()> {
        let lv = head_mut(self.levels.as_mut(), syms)?;
        if self.bps == 1 {
            for (l, &b) in lv.iter_mut().zip(bits) {
                *l = if b > 0.0 { 1.0 } else { -1.0 };
            }
        } else {
            for (l, pair) in lv.iter_mut().zip(bits.chunks_exact(2)) {
                if let [b0, b1] = pair {
                    let mag = if *b1 > 0.0 { 1.0 } else { INNER };
                    *l = if *b0 > 0.0 { -mag } else { mag };
                }
            }
        }
        Ok(())
    }
}

impl<S: Storage> Block for FskModBlock<S>
where
    Sdrk: FirInterpolateReal<f32> + FmMod<C32>,
{
    type In = f32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> Result<Progress> {
        let sps = self.config.samples_per_symbol;
        let bps = self.bps;
        let total = (input.len() / bps).min(output.len() / sps);
        let piece = self.config.piece_symbols();
        let mut done = 0usize;
        while done < total {
            let m = (total - done).min(piece);
            let bits = head(tail(input, done * bps)?, m * bps)?;
            self.map_levels(bits, m)?;
            let levels = head(self.levels.as_ref(), m)?;
            let shaped = head_mut(self.shaped.as_mut(), m * sps)?;
            let p = <Sdrk as FirInterpolateReal<f32>>::fir_interpolate_real(
                &Sdrk,
                &mut self.fir,
                levels,
                shaped,
            )?;
            if p.consumed != m || p.produced != m * sps {
                return Err(Error::Core(sdrk_core::Error::Internal("fsk_mod piece")));
            }
            let out = head_mut(tail_mut(output, done * sps)?, m * sps)?;
            <Sdrk as FmMod<C32>>::fm_mod(&Sdrk, &mut self.fm, shaped, out)?;
            done += m;
        }
        Ok(Progress {
            consumed: total * bps,
            produced: total * sps,
        })
    }

    fn timing(&self) -> BlockTiming {
        let sps = self.config.samples_per_symbol as u64;
        let taps = self.fir.taps().len() as u64;
        let bps = self.bps as u64;
        BlockTiming {
            ratio: Ratio::new(sps, bps).unwrap_or(Ratio::ONE),
            kind: RatioKind::Constant,
            // (taps - 1) / 2 output samples = (taps - 1) bps / (2 sps) input bits.
            delay: Delay::new(taps.saturating_sub(1) * bps, 2 * sps).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
        self.fm.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "deviation_hz" => {
                let d = value.as_f64()?;
                let half = hz(self.rate_out) / 2.0;
                if !(d >= 0.0 && d <= half) {
                    return Err(Error::OutOfRange {
                        name: "deviation_hz",
                        min: 0.0,
                        max: half,
                        got: d,
                    });
                }
                let phase = self.fm.phase();
                self.fm = FmModState::new(d, self.rate_out)?;
                self.fm.set_phase(phase);
                self.config.deviation_hz = d;
                Ok(())
            }
            _ => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        self.bps
    }
}
