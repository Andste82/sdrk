//! The measurement binary of `cargo xtask count`.
//!
//! ```text
//! count list
//! count run <kernel> <ours|library|fallback> <k> [n]
//! count accuracy <kernel> <ours|library|fallback> [n]
//! count info <kernel>
//! ```
//!
//! `run` performs one warm-up call and then `k` calls on the same buffers; the counters measure
//! the whole process, and the differential method of `xtask` removes everything but the `k`
//! calls. `accuracy` prints the SNR in dB against the f64 reference as a line of JSON.

use kernels::{Impl, KERNELS, N, prepare};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = |i: usize| args.get(i).map(String::as_str);
    match a(0) {
        Some("list") => {
            for k in KERNELS {
                println!("{k}");
            }
        }
        Some("run") => {
            let (Some(name), Some(which), Some(k)) = (
                a(1),
                a(2).and_then(Impl::parse),
                a(3).and_then(|s| s.parse::<usize>().ok()),
            ) else {
                usage()
            };
            let n = a(4).and_then(|s| s.parse().ok()).unwrap_or(N);
            let Some(mut kernel) = prepare(name, n) else {
                usage()
            };
            kernel.run(which, 1);
            kernel.run(which, k);
        }
        Some("accuracy") => {
            let (Some(name), Some(which)) = (a(1), a(2).and_then(Impl::parse)) else {
                usage()
            };
            let n = a(3).and_then(|s| s.parse().ok()).unwrap_or(N);
            let Some(mut kernel) = prepare(name, n) else {
                usage()
            };
            println!(
                "{{\"kernel\":\"{name}\",\"impl\":\"{}\",\"snr_db\":{:.3}}}",
                a(2).unwrap_or(""),
                kernel.accuracy(which)
            );
        }
        Some("info") => {
            let Some(name) = a(1) else { usage() };
            let Some(kernel) = prepare(name, N) else {
                usage()
            };
            println!(
                "{{\"kernel\":\"{name}\",\"has_library\":{},\"library\":\"{}\",\"level\":\"{}\"}}",
                kernel.has_library(),
                kernel.library_name(),
                sdrk_backend_simd::level_name()
            );
        }
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: count list | run <kernel> <ours|library|fallback> <k> [n] | accuracy <kernel> <impl> [n] | info <kernel>"
    );
    std::process::exit(2);
}
