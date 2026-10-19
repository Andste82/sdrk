//! Repository automation for SDRk: architecture checks, bare-metal build and
//! the combined test run.
//!
//! The binary (`cargo xtask ...`) is a thin wrapper around this library so that
//! the checks can be tested against small fake workspaces.

pub mod commands;
pub mod count;
pub mod layers;
pub mod scan;
pub mod wasm;
