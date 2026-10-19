//! Finds the system VOLK with pkg-config. The library is optional: without it the crate
//! builds with the cfg `no_volk` and every declaration is replaced by a failing stub, so the
//! benchmarks that do not compare against VOLK still work.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(no_volk)");
    match pkg_config::Config::new()
        .atleast_version("2.0")
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
