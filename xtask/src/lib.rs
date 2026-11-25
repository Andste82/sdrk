//! Repository automation for SDRk: architecture checks, bare-metal build and
//! the combined test run.
//!
//! The binary (`cargo xtask ...`) is a thin wrapper around this library so that
//! the checks can be tested against small fake workspaces.

pub mod bench;
pub mod blockcount;
pub mod branches;
pub mod commands;
pub mod count;
pub mod count_embedded;
pub mod cycle_gate;
pub mod cycle_results;
pub mod emu;
pub mod gap_report;
pub mod layers;
pub mod mca;
pub mod measure;
pub mod names;
pub mod own_code;
pub mod release;
pub mod scan;
pub mod wasm;
