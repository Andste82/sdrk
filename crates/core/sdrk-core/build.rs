//! Generates the cosine tables used by the NCO mixer (`rotate` kernel) and, with the feature
//! `static-twiddles`, the 4096-point FFT twiddle tables.
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

/// `(re, im)` of `e^(-j 2 pi k / n)`; the same expression as `kernels::fft::twiddle_f64`.
fn twiddle(k: usize, n: usize) -> (f64, f64) {
    let theta = 2.0 * core::f64::consts::PI * (k as f64 / n as f64);
    (libm::cos(theta), -libm::sin(theta))
}

/// The static FFT twiddle tables of feature `static-twiddles`.
fn emit_twiddles(src: &mut String) {
    const N: usize = 4096;
    src.push_str("\n/// `e^(-j 2 pi k / 4096)` for `k < 2048` as `C32` (16 KiB).\n");
    writeln!(
        src,
        "pub static TWIDDLES_C32_4096: [crate::C32; {}] = [",
        N / 2
    )
    .unwrap();
    for k in 0..N / 2 {
        let (re, im) = twiddle(k, N);
        writeln!(
            src,
            "    crate::C32 {{ re: {:?}_f32, im: {:?}_f32 }},",
            re as f32, im as f32
        )
        .unwrap();
    }
    src.push_str("];\n\n/// The same table rounded to Q15 (8 KiB).\n");
    writeln!(
        src,
        "pub static TWIDDLES_CQ15_4096: [crate::CQ15; {}] = [",
        N / 2
    )
    .unwrap();
    for k in 0..N / 2 {
        let (re, im) = twiddle(k, N);
        writeln!(
            src,
            "    crate::CQ15 {{ re: crate::Q15({}), im: crate::Q15({}) }},",
            q15_from_f64(re),
            q15_from_f64(im)
        )
        .unwrap();
    }
    src.push_str("];\n");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut src = String::new();
    src.push_str("/// `cos(2 pi k / 4096)` as f32 for `k` in `0..=4096` (the last entry closes the period).\n");
    writeln!(src, "pub static COS_F32: [f32; {}] = [", SIZE + 1).unwrap();
    for k in 0..=SIZE {
        let c = libm::cos(2.0 * core::f64::consts::PI * k as f64 / SIZE as f64);
        writeln!(src, "    {:?}_f32,", c as f32).unwrap();
    }
    src.push_str("];\n\n");
    src.push_str("/// The same table rounded to Q15 raw values.\n");
    writeln!(src, "pub static COS_Q15: [i16; {}] = [", SIZE + 1).unwrap();
    for k in 0..=SIZE {
        let c = libm::cos(2.0 * core::f64::consts::PI * k as f64 / SIZE as f64);
        writeln!(src, "    {},", q15_from_f64(c)).unwrap();
    }
    src.push_str("];\n");
    if std::env::var_os("CARGO_FEATURE_STATIC_TWIDDLES").is_some() {
        emit_twiddles(&mut src);
    }
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("tables.rs");
    std::fs::write(out, src).unwrap();
}
