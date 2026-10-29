//! Links the static C reference libraries that `benches/refs/build.sh` built into the prefix
//! named by `SDRK_REFS_DIR` (xtask sets it). Each library is optional: a missing one leaves its
//! cfg (`has_fftw`, `has_kissfft`, `has_ne10`, `has_gragc`, `has_grsquelch`, `has_grrotator`,
//! `has_grfmmod`, `has_grquaddemod`, `has_grdeemph`, `has_liquiddc`, `has_kfr`, `has_cmsis`,
//! `has_cmsisdsp`) unset, and its wrapper then reports that the library is not available.
//!
//! The biquad references (KFR, CMSIS-DSP) and the other CMSIS-DSP references exist once per SIMD
//! level (`libkfrbq-<level>.a`, `libcmsisbq-<level>.a`, `libcmsisdsp-<level>.a`): the level is the one the target features of this build select, the
//! same rule as `sdrk-backend-simd` uses for our kernels, so that both sides of a comparison are
//! compiled for the same instruction set. The level is exported as `SDRK_BIQUAD_REF_LEVEL`.

use std::path::PathBuf;

/// The libraries: cfg name, archive name below `<prefix>/lib`, link name.
const LIBRARIES: [(&str, &str, &str); 10] = [
    ("has_fftw", "libfftw3f.a", "fftw3f"),
    ("has_kissfft", "libkissfft16.a", "kissfft16"),
    ("has_ne10", "libNE10.a", "NE10"),
    ("has_gragc", "libgragc.a", "gragc"),
    ("has_grsquelch", "libgrsquelch.a", "grsquelch"),
    ("has_grrotator", "libgrrotator.a", "grrotator"),
    ("has_grfmmod", "libgrfmmod.a", "grfmmod"),
    ("has_grquaddemod", "libgrquaddemod.a", "grquaddemod"),
    ("has_grdeemph", "libgrdeemph.a", "grdeemph"),
    ("has_liquiddc", "libliquiddc.a", "liquiddc"),
];

/// The per-level libraries: cfg name, link name without the level.
const LEVEL_LIBRARIES: [(&str, &str); 3] = [
    ("has_kfr", "kfrbq"),
    ("has_cmsis", "cmsisbq"),
    ("has_cmsisdsp", "cmsisdsp"),
];

/// The SIMD level of the target, from the target features cargo passes to build scripts.
fn level() -> &'static str {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |f: &str| features.split(',').any(|x| x == f);
    match arch.as_str() {
        "x86_64" if has("avx512f") => "avx512",
        "x86_64" if has("avx2") => "avx2",
        "x86_64" if has("sse4.2") => "sse4.2",
        "x86_64" => "sse2",
        "aarch64" => "neon",
        "wasm32" => "simd128",
        _ => "none",
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SDRK_REFS_DIR");
    for (cfg, _, _) in LIBRARIES {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }
    for (cfg, _) in LEVEL_LIBRARIES {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }
    let level = level();
    println!("cargo:rustc-env=SDRK_BIQUAD_REF_LEVEL={level}");
    let Some(prefix) = std::env::var_os("SDRK_REFS_DIR").map(PathBuf::from) else {
        println!(
            "cargo:warning=SDRK_REFS_DIR is not set; the FFTW, KissFFT, Ne10, GNU Radio, liquid-dsp, KFR and CMSIS-DSP comparisons are disabled"
        );
        return;
    };
    let lib = prefix.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    // `benches/refs/build.sh` rewrites its stamp whenever it builds, so the stamp covers a
    // library that appears or disappears. A missing file must not be named here: cargo would
    // run this script, and relink everything that depends on it, on every build.
    let stamp = prefix.join("refs.stamp");
    if stamp.exists() {
        println!("cargo:rerun-if-changed={}", stamp.display());
    }
    let level_libraries = LEVEL_LIBRARIES.map(|(cfg, name)| {
        let name = format!("{name}-{level}");
        (cfg, format!("lib{name}.a"), name)
    });
    let all = LIBRARIES
        .iter()
        .map(|(cfg, archive, name)| (*cfg, (*archive).to_string(), (*name).to_string()))
        .chain(level_libraries);
    for (cfg, archive, name) in all {
        let path = lib.join(archive);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
            println!("cargo:rustc-link-lib=static={name}");
            println!("cargo:rustc-cfg={cfg}");
        }
    }
}
