//! Generates the static twiddle tables of the `C32` FFT (feature `fft-tables`).
//!
//! The radix-4 pass of span `S` (it combines four sub-transforms of size `S / 4` into one of size
//! `S`) multiplies the lane `k < S / 4` by `W_S^k`, `W_S^2k` and `W_S^3k`, where
//! `W_S = e^(-j 2 pi / S)`. These values do not depend on the transform size, so one table per
//! span serves every size. Per span `S = 8, 16, ..., 4096` the script emits
//!
//! - `A_S`: `W_S^k` for `k < S / 2` (for `S = 4096` only `k < S / 4`): the first twiddle of span
//!   `S` and the second twiddle of span `2 S` (`W_2S^2k = W_S^k`), and the twiddle of the radix-2
//!   pass of span `S`;
//! - `B_S`: `W_S^3k` for `k < S / 4`, the third twiddle of span `S`;
//!
//! each as `n` real parts followed by `n` imaginary parts, so that the lanes `k .. k + L` of any
//! vector width `L` are two contiguous loads. Plus the twiddles of the in-register transforms of
//! 16 and 64 points (`W_N^(c k)`, `c = 1 .. L - 1`, `k < L`, `L = sqrt N`), for 64 points also in
//! the lane order of the AVX2 shuffles.
//!
//! The values are computed in f64 with the pure-Rust `libm` and the same expression as
//! `sdrk_core::kernels::fft::twiddle_f64`, then rounded to f32. For power-of-two sizes
//! `twiddle_f64(k, S)` is the same f64 as `twiddle_f64(k * N / S, N)`, so the tables are
//! bit-identical with the twiddles of `FftPlan::new` (and exact for `W^3k`, which the plan does
//! not hold).

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "host-side generator: writes to a String cannot fail, cargo sets OUT_DIR, and the indices are bounded by the loop ranges; a panic only fails the build"
)]

use std::fmt::Write as _;
use std::path::PathBuf;

/// The largest span with a table; larger transforms take the twiddles of their largest passes
/// from the plan.
const MAX_SPAN: usize = 4096;

/// `(re, im)` of `e^(-j 2 pi k / n)`; the same expression as `kernels::fft::twiddle_f64`.
fn twiddle(k: usize, n: usize) -> (f64, f64) {
    let theta = 2.0 * core::f64::consts::PI * (k as f64 / n as f64);
    (libm::cos(theta), -libm::sin(theta))
}

/// Emits `static NAME: [f32; 2 * len]` with the real parts of `w(k)`, `k < len`, followed by the
/// imaginary parts.
fn emit(src: &mut String, doc: &str, name: &str, len: usize, w: impl Fn(usize) -> (f64, f64)) {
    writeln!(src, "/// {doc}").unwrap();
    writeln!(src, "static {name}: [f32; {}] = [", 2 * len).unwrap();
    for part in 0..2 {
        for k in 0..len {
            let (re, im) = w(k);
            let v = if part == 0 { re } else { im };
            writeln!(src, "    {:?}_f32,", v as f32).unwrap();
        }
    }
    src.push_str("];\n\n");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut src = String::new();
    if std::env::var_os("CARGO_FEATURE_FFT_TABLES").is_some() {
        let mut arms = String::new();
        let mut span = 8;
        while span <= MAX_SPAN {
            let a_len = if span == MAX_SPAN { span / 4 } else { span / 2 };
            emit(
                &mut src,
                &format!("`W_{span}^k` for `k < {a_len}`."),
                &format!("A_{span}"),
                a_len,
                |k| twiddle(k, span),
            );
            emit(
                &mut src,
                &format!("`W_{span}^3k` for `k < {}`.", span / 4),
                &format!("B_{span}"),
                span / 4,
                |k| twiddle(3 * k, span),
            );
            writeln!(
                arms,
                "        {} => Some((&A_{span}, &B_{span})),",
                span.trailing_zeros()
            )
            .unwrap();
            span *= 2;
        }
        // The lane order of the 8-lane codelet on AVX2 (`IN_LANE` of `fft::c32`).
        let in_lane = [0, 1, 4, 5, 2, 3, 6, 7];
        let natural: [usize; 8] = core::array::from_fn(|k| k);
        for (name, n, order) in [
            ("CODELET_16", 16usize, &natural[..4]),
            ("CODELET_64", 64, &natural[..]),
            ("CODELET_64_IN_LANE", 64, &in_lane[..]),
        ] {
            let l = order.len();
            writeln!(
                src,
                "/// `W_{n}^(c k)` for `c = 1 .. {l}` and the columns `k = {order:?}`: per `c`, the \
                 {l} real parts, then the {l} imaginary parts."
            )
            .unwrap();
            writeln!(
                src,
                "pub(super) static {name}: [f32; {}] = [",
                2 * l * (l - 1)
            )
            .unwrap();
            for c in 1..l {
                for part in 0..2 {
                    for &k in order {
                        let (re, im) = twiddle(c * k, n);
                        let v = if part == 0 { re } else { im };
                        writeln!(src, "    {:?}_f32,", v as f32).unwrap();
                    }
                }
            }
            src.push_str("];\n\n");
        }
        src.push_str("/// The tables `(A, B)` of the span `2^log2` (8 to 4096).\n");
        src.push_str(
            "pub(super) fn span(log2: u32) -> Option<(&'static [f32], &'static [f32])> {\n",
        );
        src.push_str("    match log2 {\n");
        src.push_str(&arms);
        src.push_str("        _ => None,\n    }\n}\n");
    }
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("fft_tables.rs");
    std::fs::write(out, src).unwrap();
}
