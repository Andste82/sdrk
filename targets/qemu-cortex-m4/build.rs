//! Makes `memory.x` visible to the linker script of `cortex-m-rt`, and with the feature
//! `cmsis-dsp` links `libcmsisbq-cortex-m4.a` from `$SDRK_REFS_DIR/lib` (built by
//! `benches/refs/build.sh cortex-m4`; only the `count` workload calls it).
use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    fs::copy("memory.x", out.join("memory.x")).expect("copy memory.x");
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-env-changed=SDRK_REFS_DIR");
    if env::var_os("CARGO_FEATURE_CMSIS_DSP").is_some() {
        let refs =
            PathBuf::from(env::var_os("SDRK_REFS_DIR").expect(
                "the feature cmsis-dsp needs SDRK_REFS_DIR (benches/refs/build.sh cortex-m4)",
            ));
        let lib = refs.join("lib");
        let archive = lib.join("libcmsisbq-cortex-m4.a");
        assert!(archive.exists(), "{} is missing", archive.display());
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-lib=static=cmsisbq-cortex-m4");
        println!("cargo:rerun-if-changed={}", archive.display());
    }
}
