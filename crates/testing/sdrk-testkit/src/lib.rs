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
//! - [`e2e_nfm`]: the end-to-end narrowband FM receive with squelch (M19),
//! - [`e2e_ssb`]: the end-to-end SSB receive with squelch and AGC (M21),
//! - [`e2e_stereo`]: the end-to-end FM stereo receive with channel separation, switching and mono checks (M24),
//! - [`fm`]: the analytic FM baseband generator,
//! - [`meters`]: tone fit, SINAD, frequency estimate and spectral mask,
//! - [`design_helpers`]: the windowed-sinc low-pass used by tests until M09,
//! - [`drift`]: `TwoClockSim`, the two-clock buffer simulation of the clock drift tests (M28),
//! - [`rds`]: the RDS reference signal of M41, re-synthesized after `tests/golden/sdrk_golden/cases/rds.py`,
//! - [`rng`]: deterministic PCG32 random numbers,
//! - [`ber`]: `BerMeter`, the PRBS bit error rate meter (M38),
//! - `r#gen` (`gen` is a reserved word in edition 2024): signal generators that depend only on the absolute sample index (including the BPSK/QPSK test signal `PskTest` of M38),
//! - [`metrics`]: SNR, error and bit-exact comparison,
//! - [`invariance`]: the block-size-invariance harness,
//! - [`arena`]: a bump allocator over a caller buffer (no heap),
//! - [`encode`]: the independent raw-format encoder and decoder (oracle),
//! - [`virtual_rx`]: a minimal virtual receiver,
//! - [`virtual_tx`]: a virtual transmitter that records writes and injects statuses,
//! - [`conformance`]: `check_rx` and `check_tx`, the conformance checks for stream implementations,
//! - `channel` (feature `std`): the `Loopback` channel simulator (delay, attenuation, frequency offset, noise) and the impairments `Awgn`, `FrequencyOffset`, `ClockDrift` and `ChannelChain` (M38),
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
pub mod ber;
#[cfg(feature = "std")]
pub mod channel;
pub mod conformance;
pub mod design_helpers;
pub mod drift;
pub mod e2e;
pub mod e2e_am;
pub mod e2e_nfm;
pub mod e2e_real;
pub mod e2e_ssb;
pub mod e2e_stereo;
pub mod encode;
pub mod fm;
pub mod r#gen;
#[cfg(feature = "std")]
pub mod golden;
pub mod invariance;
pub mod meters;
pub mod metrics;
pub mod rds;
pub mod rng;
#[cfg(feature = "std")]
pub mod storage;
pub mod suite;
pub mod virtual_rx;
pub mod virtual_tx;

mod cases;

pub use arena::{Arena, Carver};
pub use cases::{
    CHAIN_FM_RECEIVER_AUDIO, CHAIN_FM_RECEIVER_DIGEST, CHAIN_FM_RECEIVER_SAMPLES, DIGEST_START,
    chain_fm_receiver_digest, digest_q15,
};
pub use drift::TwoClockSim;
pub use invariance::{ChunkPlan, InvarianceReport, check_block_invariance, check_block_tolerance};
pub use metrics::{Comparison, bit_exact, compare_complex, compare_real};
pub use rng::Pcg32;
#[cfg(feature = "std")]
pub use storage::VecStorage;
#[cfg(feature = "std")]
pub use suite::StdReporter;
pub use suite::{Failure, Reporter, Summary, TestCase, Tier, WriteReporter, all_cases, run_suite};
pub use virtual_rx::{NoiseSource, SignalSource, SumSource, ToneSource, VirtualRx};
pub use virtual_tx::{TxRecord, VirtualTx};
