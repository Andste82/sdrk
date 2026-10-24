//! Finds VOLK with pkg-config; `cargo xtask count` points `PKG_CONFIG_PATH` at the build of
//! `benches/refs/build.sh` (shared on x86-64 and AArch64, the static library of the generic
//! protokernels on wasm32). The library is optional: without it the crate builds with the cfg
//! `no_volk` and every declaration is replaced by a failing stub, so the benchmarks that do not
//! compare against VOLK still work.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(no_volk)");
    let wasm = std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|a| a == "wasm32");
    match pkg_config::Config::new()
        .atleast_version("2.0")
        .statik(wasm)
        .probe("volk")
    {
        Ok(_) => {}
        Err(e) => {
            println!(
                "cargo:warning=VOLK not found by pkg-config ({e}); the VOLK comparisons are disabled"
            );
            println!("cargo:rustc-cfg=no_volk");
        }
    }
}
