//! Generates the cosine tables used by the NCO mixer (`rotate` kernel).
//!
//! `COS_F32[k] = cos(2 pi k / 4096)` and `COS_Q15[k]` is the same value rounded
//! to Q15 (round half up, saturating), both computed in f64 with the pure-Rust
//! `libm`, so every host and target gets bit-identical tables.

// A build script talks to cargo through stdout and may use `std`.
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::fmt::Write as _;
use std::path::PathBuf;

const SIZE: usize = 4096;

fn q15_from_f64(x: f64) -> i16 {
    let v = libm::floor(x * 32768.0 + 0.5);
    if v >= 32767.0 {
        32767
    } else if v <= -32768.0 {
        -32768
    } else {
        v as i16
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut src = String::new();
    src.push_str("/// `cos(2 pi k / 4096)` as f32 for `k` in `0..=4096` (the last entry closes the period).\n");
    writeln!(src, "pub(crate) static COS_F32: [f32; {}] = [", SIZE + 1).unwrap();
    for k in 0..=SIZE {
        let c = libm::cos(2.0 * core::f64::consts::PI * k as f64 / SIZE as f64);
        writeln!(src, "    {:?}_f32,", c as f32).unwrap();
    }
    src.push_str("];\n\n");
    src.push_str("/// The same table rounded to Q15 raw values.\n");
    writeln!(src, "pub(crate) static COS_Q15: [i16; {}] = [", SIZE + 1).unwrap();
    for k in 0..=SIZE {
        let c = libm::cos(2.0 * core::f64::consts::PI * k as f64 / SIZE as f64);
        writeln!(src, "    {},", q15_from_f64(c)).unwrap();
    }
    src.push_str("];\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("tables.rs");
    std::fs::write(out, src).unwrap();
}
