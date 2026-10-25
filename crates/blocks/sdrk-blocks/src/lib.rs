//! Blocks of SDRk: filter design, storage abstraction and self-describing signal-processing
//! blocks for the receive and transmit chains and for spectra.
//!
//! Layer `crates/blocks`, heap-free. A block is built from a config, the input sample rate and
//! a [`Storage`]; it uses the facade [`sdrk::Sdrk`] for every kernel call, so backends speed
//! up the blocks automatically. Every block kind has a machine-readable
//! [`BlockDescription`] (see [`all_descriptions`]).
//!
//! ```
//! use sdrk::{Rate, C32};
//! use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, DdcBlock, DdcConfig};
//!
//! // DDC from 2.4 MHz to 240 kHz around a carrier at +600 kHz.
//! let cfg = DdcConfig { frequency_hz: 600e3, decimation: 10, ..DdcConfig::default() };
//! let rate = Rate::hz(2_400_000)?;
//! let bytes = DdcBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
//! let mut mem = vec![0u8; bytes + 16];
//! let mut arena = ArenaStorage::new(&mut mem);
//! let mut ddc = DdcBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
//! let mut out = vec![C32::default(); 100];
//! let p = ddc.process(&[C32::new(0.5, 0.0); 1000], &mut out)?;
//! assert_eq!((p.consumed, p.produced), (1000, 100));
//! # let _ = AlignedBytes([0u8; 1]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![no_std]
#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]

mod block;
mod blocks;
pub mod describe;
pub mod design;
mod error;
mod sample;
mod storage;

pub use block::{Block, BlockTiming, FrameBlock, FrameMeta, ParamValue};
pub use blocks::convert::{ConvertBlock, ConvertConfig};
pub use blocks::convert_out::{ConvertOutBlock, ConvertOutConfig};
pub use blocks::dc_block::{DcBlockBlock, DcBlockConfig};
pub use blocks::ddc::{DdcBlock, DdcConfig};
pub use blocks::duc::{DucBlock, DucConfig};
pub use blocks::fir::{FirBlock, FirConfig};
pub use blocks::fm_demod::{FM_CHUNK, FmDemodBlock, FmDemodConfig};
pub use blocks::fm_mod::{FmModBlock, FmModConfig};
pub use blocks::resampler::{ResamplerBlock, ResamplerConfig, ResamplerRealBlock, TUNED_PHASES};
pub use blocks::scale::{ScaleBlock, ScaleConfig};
pub use blocks::spectrum::{SpectrumBlock, SpectrumConfig, SpectrumSample, SpectrumWindow};
pub use blocks::squelch::{SquelchBlock, SquelchConfig};
pub use describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit, all_descriptions,
};
pub use design::{DesignSpec, DesignWindow};
pub use error::{Error, Result};
pub use sample::{BlockReal, BlockSample, TunedElem};
pub use storage::{ARENA_ALIGN, AlignedBytes, ArenaStorage, Storage};
