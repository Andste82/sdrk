//! Testkit of SDRk: signal generators, virtual devices, loopback and the
//! on-target test suite.
//!
//! Layer `crates/testing`; `no_std` by default, the `std` feature adds file
//! I/O and host helpers (the `StdReporter` today).
//!
//! The modules are:
//!
//! - `storage` (feature `std`): `VecStorage`, the `Vec`-backed block storage for host tests,
//! - `golden` (feature `std`): loader and comparison for the golden vectors,
//! - [`e2e`]: the streaming FM receive and transmit chains of the end-to-end tests,
//! - [`fm`]: the analytic FM baseband generator,
//! - [`meters`]: tone fit, SINAD, frequency estimate and spectral mask,
//! - [`design_helpers`]: the windowed-sinc low-pass used by tests until M09,
//! - [`rng`]: deterministic PCG32 random numbers,
//! - `r#gen` (`gen` is a reserved word in edition 2024): signal generators that depend only on the absolute sample index,
//! - [`metrics`]: SNR, error and bit-exact comparison,
//! - [`invariance`]: the block-size-invariance harness,
//! - [`arena`]: a bump allocator over a caller buffer (no heap),
//! - [`encode`]: the independent raw-format encoder and decoder (oracle),
//! - [`virtual_rx`]: a minimal virtual receiver,
//! - [`suite`]: the runner that executes registered cases on any target.
//!
//! ```
//! use sdrk_testkit::rng::Pcg32;
//! let mut rng = Pcg32::new(42, 54);
//! assert_eq!(rng.next_u32(), 0xa15c_02b7);
//! ```
#![no_std]
#![forbid(unsafe_code)]
// `Failure` carries a 160-byte detail string by specification, so every
// `Result<_, Failure>` is large.
#![allow(clippy::result_large_err)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]

#[cfg(feature = "std")]
extern crate std;

pub mod arena;
pub mod design_helpers;
pub mod e2e;
pub mod encode;
pub mod fm;
pub mod r#gen;
#[cfg(feature = "std")]
pub mod golden;
pub mod invariance;
pub mod meters;
pub mod metrics;
pub mod rng;
#[cfg(feature = "std")]
pub mod storage;
pub mod suite;
pub mod virtual_rx;

mod cases;

pub use arena::{Arena, Carver};
pub use invariance::{ChunkPlan, InvarianceReport, check_block_invariance};
pub use metrics::{Comparison, bit_exact, compare_complex, compare_real};
pub use rng::Pcg32;
#[cfg(feature = "std")]
pub use storage::VecStorage;
#[cfg(feature = "std")]
pub use suite::StdReporter;
pub use suite::{Failure, Reporter, Summary, TestCase, Tier, WriteReporter, all_cases, run_suite};
pub use virtual_rx::{NoiseSource, SignalSource, SumSource, ToneSource, VirtualRx};
