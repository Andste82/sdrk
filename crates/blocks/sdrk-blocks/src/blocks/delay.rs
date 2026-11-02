//! `delay`: a fixed delay of whole samples.

use sdrk::{Delay, Progress, Rate};

use super::{head, head_mut, tail, tail_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::StreamElem;
use crate::sample::{BlockReal, BlockSample};
use crate::storage::{Plan, Storage};

/// Largest delay in samples.
const MAX_SAMPLES: usize = 65_536;

/// Description of the `delay` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "delay",
    summary: "Delays a signal by a whole number of samples.",
    details: "Output n is input n - samples; the first samples outputs are zero. It aligns two \
        paths before a block with two inputs, for example the direct path and a filtered path \
        whose filter has a group delay; Fork and the pipeline insert it automatically where \
        the delays of two paths differ. The block copies samples and does not change them, \
        so it works for all four sample types. The ring buffer needs samples elements of \
        storage. Pitfall: the delay is in input samples at the block's own rate, so a delay \
        placed after a decimator counts output samples.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "Samples of any of the four types.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
        description: "The delayed samples, same type and rate as the input.",
    }],
    params: &[ParamDesc {
        name: "samples",
        unit: Unit::Samples,
        kind: ParamKind::I64 {
            min: 0,
            max: MAX_SAMPLES as i64,
            default: None,
        },
        typical: "63 to match the group delay of a 127-tap linear-phase filter",
        mutability: Mutability::Build,
        description: "Delay in samples, 0 to 65536. Required.",
    }],
    required: &["samples"],
};

/// Configuration of [`DelayBlock`] and [`DelayRealBlock`].
///
/// ```
/// use sdrk_blocks::DelayConfig;
/// let cfg = DelayConfig { samples: 63 };
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// assert!(DelayConfig { samples: 65_537 }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DelayConfig {
    /// Delay in samples, 0 to 65536.
    pub samples: usize,
}

impl DelayConfig {
    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        super::check_count("samples", self.samples, 0, MAX_SAMPLES)
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "samples" => self.samples = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "samples" => Ok(ParamValue::I64(self.samples as i64)),
            _ => unknown(),
        }
    }
}

/// The ring buffer shared by both variants.
struct Ring<E: StreamElem, S: Storage> {
    samples: usize,
    pos: usize,
    buf: S::Buf<E>,
}

impl<E: StreamElem, S: Storage> Ring<E, S> {
    fn new(config: DelayConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        Ok(Self {
            samples: config.samples,
            pos: 0,
            buf: storage.alloc::<E>(config.samples)?,
        })
    }

    fn storage_bytes(config: DelayConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<E>(config.samples)?;
        Ok(plan.bytes())
    }

    fn process(&mut self, input: &[E], output: &mut [E]) -> Result<Progress> {
        let n = input.len().min(output.len());
        let (input, output) = (head(input, n)?, head_mut(output, n)?);
        if self.samples == 0 {
            output.copy_from_slice(input);
        } else {
            let mut done = 0;
            while done < n {
                let k = (n - done).min(self.samples - self.pos);
                let src = head(tail(input, done)?, k)?;
                let dst = head_mut(tail_mut(output, done)?, k)?;
                let ring = head_mut(tail_mut(self.buf.as_mut(), self.pos)?, k)?;
                dst.copy_from_slice(ring);
                ring.copy_from_slice(src);
                done += k;
                self.pos += k;
                if self.pos == self.samples {
                    self.pos = 0;
                }
            }
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            delay: Delay::new(self.samples as u64, 1).unwrap_or(Delay::ZERO),
            ..unit_timing()
        }
    }

    fn reset(&mut self) {
        self.buf.as_mut().fill(E::default());
        self.pos = 0;
    }
}

/// Reports every parameter as read-only.
fn read_only<T>(name: &str) -> Result<T> {
    if DESCRIPTION.param(name).is_some() {
        Err(Error::ReadOnlyParameter)
    } else {
        unknown()
    }
}

macro_rules! delay_block {
    ($(#[$doc:meta])* $name:ident, $bound:ident) => {
        $(#[$doc])*
        pub struct $name<T: $bound, S: Storage> {
            config: DelayConfig,
            ring: Ring<T, S>,
        }

        impl<T: $bound, S: Storage> core::fmt::Debug for $name<T, S> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("config", &self.config)
                    .finish_non_exhaustive()
            }
        }

        impl<T: $bound, S: Storage> $name<T, S> {
            /// Creates the block; the ring buffer of `samples` elements comes from `storage`.
            pub fn new(config: &DelayConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
                Ok(Self {
                    config: *config,
                    ring: Ring::new(*config, rate_in, storage)?,
                })
            }

            /// Exact bytes of storage the block needs for an arena that starts at a 16-byte
            /// aligned address (an unaligned slice may need up to 15 bytes more).
            pub fn storage_bytes(config: &DelayConfig, rate_in: Rate) -> Result<usize> {
                Ring::<T, S>::storage_bytes(*config, rate_in)
            }

            /// Output rate for the configuration (equal to the input rate).
            pub fn rate_out(config: &DelayConfig, rate_in: Rate) -> Result<Rate> {
                config.validate(rate_in)?;
                Ok(rate_in)
            }
        }

        impl<T: $bound, S: Storage> Block for $name<T, S> {
            type In = T;
            type Out = T;

            fn description(&self) -> &'static BlockDescription {
                &DESCRIPTION
            }

            fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
                self.ring.process(input, output)
            }

            fn timing(&self) -> BlockTiming {
                self.ring.timing()
            }

            fn reset(&mut self) {
                self.ring.reset();
            }

            fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
                read_only(name)
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

delay_block!(
    /// Delay for `C32` and `CQ15`.
    ///
    /// ```
    /// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, DelayBlock, DelayConfig};
    /// use sdrk::{Rate, C32};
    /// let cfg = DelayConfig { samples: 2 };
    /// let rate = Rate::hz(48_000)?;
    /// let bytes = DelayBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
    /// let mut mem = AlignedBytes([0u8; 64]);
    /// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
    /// let mut block = DelayBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
    /// let input = [1.0, 2.0, 3.0, 4.0].map(|re| C32::new(re, 0.0));
    /// let mut out = [C32::default(); 4];
    /// block.process(&input, &mut out)?;
    /// assert_eq!(out.map(|c| c.re), [0.0, 0.0, 1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    DelayBlock,
    BlockSample
);

delay_block!(
    /// Delay for `f32` and `Q15`.
    ///
    /// ```
    /// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, DelayConfig, DelayRealBlock};
    /// use sdrk::Rate;
    /// let cfg = DelayConfig { samples: 1 };
    /// let rate = Rate::hz(48_000)?;
    /// let bytes = DelayRealBlock::<f32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
    /// let mut mem = AlignedBytes([0u8; 16]);
    /// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
    /// let mut block = DelayRealBlock::<f32, _>::new(&cfg, rate, &mut arena)?;
    /// let mut out = [0.0f32; 3];
    /// block.process(&[1.0, 2.0, 3.0], &mut out)?;
    /// assert_eq!(out, [0.0, 1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    DelayRealBlock,
    BlockReal
);
