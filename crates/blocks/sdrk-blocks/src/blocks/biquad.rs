//! `biquad`: Butterworth and RBJ cookbook IIR filters as cascaded second-order sections.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::biquad::{
    Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState, MAX_SECTIONS,
};
use sdrk_core::{Q15, Sample};

use super::{HZ_LIMIT, head, head_mut, hz, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{BiquadResponse, biquad_coefs, butterworth_sos, to_f32_sos, to_q14};
use crate::error::{Error, Result};
use crate::sample::{BlockReal, BlockSample};
use crate::storage::{Plan, Storage};

/// Default quality factor of the specification (4 digits, not exactly 1/sqrt(2)).
#[allow(clippy::approx_constant)]
const DEFAULT_Q: f64 = 0.7071;

/// Most sections of the block (order 8).
const BLOCK_SECTIONS: usize = 4;

/// The Q15 low-pass and high-pass cutoff must be at least `fs / Q15_CUTOFF_DIVISOR`.
const Q15_CUTOFF_DIVISOR: f64 = 500.0;

/// Names of the responses, in the order of [`BiquadResponse`] as the parameter lists them.
const RESPONSES: [&str; 4] = ["lowpass", "highpass", "bandpass", "notch"];

/// Description of the `biquad` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "biquad",
    summary: "Filters with a Butterworth low-pass or high-pass, a band-pass or a notch (IIR).",
    details: "Cascaded second-order sections (biquads). Lowpass and highpass are Butterworth \
        filters of even order 2 to 8 (order / 2 sections, -3 dB at cutoff_hz); bandpass (0 dB \
        peak) and notch are single sections of the RBJ Audio EQ Cookbook with the centre \
        frequency cutoff_hz and the quality q. Typical uses are audio shaping after a \
        demodulator: a 300 Hz high-pass removes the CTCSS sub-tone of two-way radio, a notch \
        removes a whistle, a low-pass limits hiss. cutoff_hz and q can change while the block \
        runs; the filter state is kept, so the output stays continuous. Works on complex \
        (C32, CQ15: real and imaginary part are filtered independently) and real (f32, Q15) \
        signals; there is no delay in the sense of exact timing (an IIR filter delays by a \
        frequency-dependent amount). Pitfall: in Q15 the coefficients have 14 fractional bits \
        and the rounding error of the feedback grows as 1 / |1 + a1 + a2| at DC, so a low-pass \
        or high-pass with a cutoff below sample rate / 500 is rejected; use f32 or a higher \
        cutoff. A Q15 filter may also keep a small limit cycle at silence.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples of any of the four types.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "The filtered samples, same type and rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "response",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: &RESPONSES,
                default: 0,
            },
            typical: "lowpass for hiss, highpass to remove a sub-tone, notch for a whistle",
            mutability: Mutability::Build,
            description: "Shape of the filter.",
        },
        ParamDesc {
            name: "cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "4000 for an AM audio low-pass, 300 to remove CTCSS",
            mutability: Mutability::Runtime,
            description: "-3 dB corner of a low-pass or high-pass, centre frequency of a \
                band-pass or notch, in Hz; above 0 and below half the sample rate (for Q15 \
                low-pass and high-pass at least sample rate / 500). Required.",
        },
        ParamDesc {
            name: "order",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 2,
                max: 8,
                default: Some(2),
            },
            typical: "4",
            mutability: Mutability::Build,
            description: "Order of a Butterworth low-pass or high-pass: even, 2 to 8. Must be 2 \
                for band-pass and notch.",
        },
        ParamDesc {
            name: "q",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 1e-3,
                max: 1e3,
                default: Some(DEFAULT_Q),
            },
            typical: "2 for a band-pass, 10 for a narrow notch",
            mutability: Mutability::Runtime,
            description: "Quality factor of a band-pass or notch; ignored by low-pass and \
                high-pass.",
        },
    ],
    required: &["cutoff_hz"],
};

/// Configuration of [`BiquadBlock`] and [`BiquadRealBlock`].
///
/// ```
/// use sdrk_blocks::{BiquadConfig, BiquadResponse};
/// let cfg = BiquadConfig { response: BiquadResponse::Highpass, order: 4, cutoff_hz: 300.0, ..BiquadConfig::default() };
/// assert_eq!(cfg.sections(), 2);
/// assert!(cfg.validate(sdrk::Rate::hz(8_000)?, false).is_ok());
/// // Q15 rejects a cutoff below fs / 500 (16 Hz at 8 kHz).
/// let low = BiquadConfig { cutoff_hz: 10.0, ..cfg };
/// assert!(low.validate(sdrk::Rate::hz(8_000)?, true).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadConfig {
    /// Shape of the filter.
    pub response: BiquadResponse,
    /// Corner (low-pass, high-pass) or centre frequency (band-pass, notch) in Hz.
    pub cutoff_hz: f64,
    /// Butterworth order 2, 4, 6 or 8 (must be 2 for band-pass and notch).
    pub order: usize,
    /// Quality factor of band-pass and notch (ignored otherwise).
    pub q: f64,
}

impl Default for BiquadConfig {
    fn default() -> Self {
        Self {
            response: BiquadResponse::Lowpass,
            cutoff_hz: 1000.0,
            order: 2,
            q: DEFAULT_Q,
        }
    }
}

impl BiquadConfig {
    /// Number of second-order sections: `order / 2` for low-pass and high-pass, 1 otherwise.
    pub fn sections(&self) -> usize {
        match self.response {
            BiquadResponse::Lowpass | BiquadResponse::Highpass => self.order / 2,
            _ => 1,
        }
    }

    /// Checks the configuration against the input sample rate; `q15` adds the rule of the Q15
    /// types (cutoff of a low-pass or high-pass at least `fs / 500`).
    pub fn validate(&self, rate_in: Rate, q15: bool) -> Result<()> {
        let fs = hz(rate_in);
        let half = fs / 2.0;
        let low_high = matches!(
            self.response,
            BiquadResponse::Lowpass | BiquadResponse::Highpass
        );
        if !(self.cutoff_hz > 0.0 && self.cutoff_hz < half) {
            return Err(Error::OutOfRange {
                name: "cutoff_hz",
                min: 0.0,
                max: half,
                got: self.cutoff_hz,
            });
        }
        if low_high {
            if !(2..=8).contains(&self.order) || !self.order.is_multiple_of(2) {
                return Err(Error::InvalidParameter {
                    name: "order",
                    reason: "the order must be even, from 2 to 8",
                });
            }
        } else if self.order != 2 {
            return Err(Error::InvalidParameter {
                name: "order",
                reason: "the order must be 2 for a band-pass and a notch",
            });
        }
        if !(self.q > 0.0 && self.q.is_finite()) {
            return Err(Error::OutOfRange {
                name: "q",
                min: 0.0,
                max: f64::INFINITY,
                got: self.q,
            });
        }
        let limit = fs / Q15_CUTOFF_DIVISOR;
        if q15 && low_high && self.cutoff_hz < limit {
            return Err(Error::OutOfRange {
                name: "cutoff_hz",
                min: limit,
                max: half,
                got: self.cutoff_hz,
            });
        }
        Ok(())
    }

    /// The `f64` coefficients of every section (the rest of the array is zero) and their count.
    fn design(&self, rate_in: Rate) -> Result<([[f64; 5]; BLOCK_SECTIONS], usize)> {
        let fs = hz(rate_in);
        let mut sos = [[0.0f64; 5]; BLOCK_SECTIONS];
        let n = self.sections();
        match self.response {
            BiquadResponse::Lowpass | BiquadResponse::Highpass => {
                let dst = sos.get_mut(..n).ok_or(Error::InvalidParameter {
                    name: "order",
                    reason: "the order must be even, from 2 to 8",
                })?;
                butterworth_sos(self.response, self.order, self.cutoff_hz, fs, dst)?;
            }
            _ => {
                if let Some(first) = sos.first_mut() {
                    *first = biquad_coefs(self.response, self.cutoff_hz, self.q, fs)?;
                }
            }
        }
        Ok((sos, n))
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "response" => {
                self.response = match value.as_choice()? {
                    0 => BiquadResponse::Lowpass,
                    1 => BiquadResponse::Highpass,
                    2 => BiquadResponse::Bandpass,
                    _ => BiquadResponse::Notch,
                }
            }
            "cutoff_hz" => self.cutoff_hz = value.as_f64()?,
            "order" => self.order = value.as_usize()?,
            "q" => self.q = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "response" => Ok(ParamValue::Choice(match self.response {
                BiquadResponse::Lowpass => 0,
                BiquadResponse::Highpass => 1,
                BiquadResponse::Bandpass => 2,
                _ => 3,
            })),
            "cutoff_hz" => Ok(ParamValue::F64(self.cutoff_hz)),
            "order" => Ok(ParamValue::I64(self.order as i64)),
            "q" => Ok(ParamValue::F64(self.q)),
            _ => unknown(),
        }
    }
}

/// A sample type with a `biquad` kernel: the coefficient conversion and the Q15 rule.
pub trait BiquadElem: Sample + BiquadSample<Coef: bytemuck::Pod + Default> {
    /// Whether the coefficients are Q2.14 (the Q15 cutoff rule applies).
    const Q15: bool;

    /// Converts designed coefficients to the coefficient type.
    fn convert(src: &[[f64; 5]], dst: &mut [[Self::Coef; 5]]) -> Result<()>;
}

macro_rules! biquad_elem {
    ($t:ty, $q15:expr, $conv:ident) => {
        impl BiquadElem for $t {
            const Q15: bool = $q15;
            fn convert(src: &[[f64; 5]], dst: &mut [[Self::Coef; 5]]) -> Result<()> {
                $conv(src, dst)
            }
        }
    };
}

biquad_elem!(sdrk::C32, false, to_f32_sos);
biquad_elem!(f32, false, to_f32_sos);
biquad_elem!(sdrk::CQ15, true, to_q14);
biquad_elem!(Q15, true, to_q14);

/// Converts the designed sections of `config` to the coefficients of `E`.
fn coefs_for<E: BiquadElem>(
    config: &BiquadConfig,
    rate_in: Rate,
) -> Result<([[E::Coef; 5]; BLOCK_SECTIONS], usize)> {
    config.validate(rate_in, E::Q15)?;
    let (sos, n) = config.design(rate_in)?;
    let mut out = [[<E::Coef as Default>::default(); 5]; BLOCK_SECTIONS];
    let src = sos
        .get(..n)
        .ok_or(Error::Core(sdrk_core::Error::Internal("biquad sections")))?;
    let dst = out
        .get_mut(..n)
        .ok_or(Error::Core(sdrk_core::Error::Internal("biquad sections")))?;
    E::convert(src, dst)?;
    Ok((out, n))
}

/// Exact storage of a cascade of `sections` sections.
fn plan_bytes<E: BiquadElem>(sections: usize) -> Result<usize> {
    let mut plan = Plan::default();
    plan.add::<[E::Coef; 5]>(sections)?;
    plan.add::<E>(4 * sections)?;
    Ok(plan.bytes())
}

macro_rules! biquad_block {
    (
        $(#[$meta:meta])*
        $name:ident, $bound:ident, $state:ident, $kernel:ident, $method:ident
    ) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name<T: $bound + BiquadElem, S: Storage> {
            config: BiquadConfig,
            rate: Rate,
            state: $state<T, S::Buf<[T::Coef; 5]>, S::Buf<T>>,
        }

        impl<T: $bound + BiquadElem, S: Storage> $name<T, S> {
            /// Creates the block: designs the cascade and takes its buffers from `storage`.
            pub fn new(config: &BiquadConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
                let (coefs, n) = coefs_for::<T>(config, rate_in)?;
                let mut buf = storage.alloc::<[T::Coef; 5]>(n)?;
                let src = coefs.get(..n).ok_or(Error::Core(sdrk_core::Error::Internal(
                    "biquad sections",
                )))?;
                buf.as_mut().copy_from_slice(src);
                let hist = storage.alloc::<T>(4 * n)?;
                Ok(Self {
                    config: *config,
                    rate: rate_in,
                    state: $state::new(buf, hist)?,
                })
            }

            /// Exact bytes of storage the block needs.
            pub fn storage_bytes(config: &BiquadConfig, rate_in: Rate) -> Result<usize> {
                config.validate(rate_in, T::Q15)?;
                plan_bytes::<T>(config.sections())
            }

            /// Output rate for the configuration (equal to the input rate).
            pub fn rate_out(config: &BiquadConfig, rate_in: Rate) -> Result<Rate> {
                config.validate(rate_in, T::Q15)?;
                Ok(rate_in)
            }

            /// Number of second-order sections.
            pub fn sections(&self) -> usize {
                self.state.sections()
            }

            /// Designs `config` again and writes the coefficients into the running filter.
            fn retune(&mut self, config: BiquadConfig) -> Result<()> {
                let (coefs, n) = coefs_for::<T>(&config, self.rate)?;
                let src = coefs.get(..n).ok_or(Error::Core(sdrk_core::Error::Internal(
                    "biquad sections",
                )))?;
                self.state.coefs_mut().copy_from_slice(src);
                self.config = config;
                Ok(())
            }
        }

        impl<T: $bound + BiquadElem, S: Storage> Block for $name<T, S>
        where
            Sdrk: $kernel<T>,
        {
            type In = T;
            type Out = T;

            fn description(&self) -> &'static BlockDescription {
                &DESCRIPTION
            }

            fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
                let n = input.len().min(output.len());
                <Sdrk as $kernel<T>>::$method(
                    &Sdrk,
                    &mut self.state,
                    head(input, n)?,
                    head_mut(output, n)?,
                )?;
                Ok(Progress {
                    consumed: n,
                    produced: n,
                })
            }

            fn timing(&self) -> BlockTiming {
                unit_timing()
            }

            fn reset(&mut self) {
                self.state.reset();
            }

            fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
                let mut next = self.config;
                match runtime_param(&DESCRIPTION, name, &value)? {
                    "cutoff_hz" => next.cutoff_hz = value.as_f64()?,
                    "q" => next.q = value.as_f64()?,
                    _ => return unknown(),
                }
                self.retune(next)
            }

            fn param(&self, name: &str) -> Result<ParamValue> {
                self.config.get(name)
            }

            fn min_input(&self) -> usize {
                1
            }
        }
    };
}

biquad_block! {
    /// Biquad filter for `C32` and `CQ15`: the real and the imaginary part are filtered
    /// independently with the same coefficients.
    ///
    /// ```
    /// use sdrk_blocks::{ArenaStorage, Block, BiquadBlock, BiquadConfig, ParamValue};
    /// use sdrk::{Rate, C32};
    /// let cfg = BiquadConfig { cutoff_hz: 1_000.0, order: 4, ..BiquadConfig::default() };
    /// let rate = Rate::hz(48_000)?;
    /// let mut mem = vec![0u8; BiquadBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + 16];
    /// let mut arena = ArenaStorage::new(&mut mem);
    /// let mut block = BiquadBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
    /// let input = [C32::new(1.0, -1.0); 2000];
    /// let mut out = [C32::default(); 2000];
    /// block.process(&input, &mut out)?;
    /// assert!((out[1999].re - 1.0).abs() < 1e-3); // unity gain at DC
    /// block.set_param("cutoff_hz", ParamValue::F64(2_000.0))?; // retune while running
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    BiquadBlock, BlockSample, BiquadState, Biquad, biquad
}

biquad_block! {
    /// Biquad filter for `f32` and `Q15`.
    ///
    /// ```
    /// use sdrk_blocks::{ArenaStorage, Block, BiquadRealBlock, BiquadConfig, BiquadResponse};
    /// use sdrk::Rate;
    /// let cfg = BiquadConfig { response: BiquadResponse::Notch, cutoff_hz: 1_000.0, q: 10.0, ..BiquadConfig::default() };
    /// let rate = Rate::hz(48_000)?;
    /// let mut mem = vec![0u8; BiquadRealBlock::<f32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + 16];
    /// let mut arena = ArenaStorage::new(&mut mem);
    /// let mut block = BiquadRealBlock::<f32, _>::new(&cfg, rate, &mut arena)?;
    /// let tone: Vec<f32> = (0..4800).map(|n| (n as f32 * 2.0 * core::f32::consts::PI / 48.0).sin()).collect();
    /// let mut out = vec![0.0f32; 4800];
    /// block.process(&tone, &mut out)?;
    /// assert!(out[4799].abs() < 0.01); // 1 kHz is notched out
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    BiquadRealBlock, BlockReal, BiquadRealState, BiquadReal, biquad_real
}

// The cascade never has more sections than the kernel allows.
const _: () = assert!(BLOCK_SECTIONS <= MAX_SECTIONS);
