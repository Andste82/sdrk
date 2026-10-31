//! Repository automation for SDRk: architecture checks, bare-metal build and
//! the combined test run.
//!
//! The binary (`cargo xtask ...`) is a thin wrapper around this library so that
//! the checks can be tested against small fake workspaces.

pub mod blockcount;
pub mod commands;
pub mod count;
pub mod count_embedded;
pub mod emu;
pub mod layers;
pub mod mca;
pub mod measure;
pub mod scan;
pub mod wasm;
