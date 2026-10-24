//! The measurement binary of `cargo xtask count`.
//!
//! ```text
//! count list
//! count run <kernel> <impl> <k> [--n N] [--misaligned]
//! count accuracy <kernel> <impl> [--n N] [--misaligned]
//! count info <kernel>
//! count volk-info
//! count volk-impls <kernel>
//! ```
//!
//! `<impl>` is `ours`, `fallback`, `volk`, `rustfft`, `fftw` (out of place), `fftw-in-place`,
//! `kissfft`, `ne10` or `restore`.
//!
//! `run` performs one warm-up call and then `k` calls on the same buffers between the SSC
//! markers of Intel SDE; the counters measure the whole process (SDE only the region between
//! the markers), and the differential method of `xtask` removes everything but the `k` calls
//! (with the few set-up instructions of the kernel's loop, under 0.001 per sample).
//! `accuracy` prints the SNR in dB against the f64 reference as a line of JSON. `info` lists
//! the implementations this build supports for the kernel, `volk-info` the linked VOLK, and
//! `volk-impls` the implementations of the kernel's VOLK functions on the selected machine.

use kernels::{Impl, KERNELS, Layout, prepare, region_start, region_stop};
use serde_json::{Value, json};

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
            let mut kernel = kernel(name, which, &args[4..]);
            kernel.run(which, 1);
            region_start();
            // With `k = 0` the region holds only the markers, whatever the kernel: SDE counts
            // it once per binary.
            if k > 0 {
                kernel.run(which, k);
            }
            region_stop();
        }
        Some("accuracy") => {
            let (Some(name), Some(which)) = (a(1), a(2).and_then(Impl::parse)) else {
                usage()
            };
            let mut kernel = kernel(name, which, &args[3..]);
            print(&json!({
                "kernel": name,
                "impl": which.name(),
                "snr_db": (kernel.accuracy(which) * 1000.0).round() / 1000.0,
            }));
        }
        Some("info") => {
            let Some(name) = a(1) else { usage() };
            let Some(kernel) = prepare(name, Layout::default()) else {
                usage()
            };
            let impls: serde_json::Map<String, Value> = Impl::ALL
                .into_iter()
                .filter(|i| kernel.supports(*i))
                .map(|i| (i.name().to_string(), Value::from(kernel.describe(i))))
                .collect();
            print(&json!({
                "kernel": name,
                "level": sdrk_backend_simd::level_name(),
                "impls": impls,
                "volk_functions": kernel.volk_functions(),
            }));
        }
        Some("volk-info") => match volk_sys::info() {
            Some(info) => print(&json!({
                "version": info.version,
                "machine": info.machine,
                "compiler_flags": info.compiler_flags,
                "alignment": volk_sys::alignment(),
            })),
            None => print(&json!({})),
        },
        Some("volk-impls") => {
            let Some(name) = a(1) else { usage() };
            let Some(kernel) = prepare(name, Layout::default()) else {
                usage()
            };
            let functions: Vec<Value> = kernel
                .volk_functions()
                .iter()
                .map(|f| {
                    let list = volk_sys::implementations(f).unwrap_or_default();
                    let default = volk_sys::default_implementation(&list).map(|i| i.name.clone());
                    let names: Vec<&str> = list.iter().map(|i| i.name.as_str()).collect();
                    json!({ "name": f, "impls": names, "default": default })
                })
                .collect();
            print(&json!({ "kernel": name, "functions": functions }));
        }
        _ => usage(),
    }
}

/// The prepared kernel `name` with the layout options in `options`; exits if `which` is not
/// supported by this build.
fn kernel(name: &str, which: Impl, options: &[String]) -> Box<dyn kernels::Kernel> {
    let mut layout = Layout::default();
    let mut it = options.iter();
    while let Some(option) = it.next() {
        match option.as_str() {
            "--n" => match it.next().and_then(|s| s.parse().ok()) {
                Some(n) => layout.n = n,
                None => usage(),
            },
            "--misaligned" => layout.misaligned = true,
            _ => usage(),
        }
    }
    let Some(kernel) = prepare(name, layout) else {
        usage()
    };
    if !kernel.supports(which) {
        eprintln!("{name}: `{}` is not available in this build", which.name());
        std::process::exit(3);
    }
    kernel
}

fn print(value: &Value) {
    println!("{value}");
}

fn usage() -> ! {
    eprintln!(
        "usage: count list | run <kernel> <impl> <k> [--n N] [--misaligned] | accuracy <kernel> <impl> [--n N] [--misaligned] | info <kernel> | volk-info | volk-impls <kernel>"
    );
    std::process::exit(2);
}
