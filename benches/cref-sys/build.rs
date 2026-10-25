//! Links the static C reference libraries that `benches/refs/build.sh` built into the prefix
//! named by `SDRK_REFS_DIR` (xtask sets it). Each library is optional: a missing one leaves its
//! cfg (`has_fftw`, `has_kissfft`, `has_ne10`) unset, and its wrapper then reports that the
//! library is not available.

use std::path::PathBuf;

/// The libraries: cfg name, archive name below `<prefix>/lib`, link name.
const LIBRARIES: [(&str, &str, &str); 4] = [
    ("has_fftw", "libfftw3f.a", "fftw3f"),
    ("has_kissfft", "libkissfft16.a", "kissfft16"),
    ("has_ne10", "libNE10.a", "NE10"),
    ("has_gragc", "libgragc.a", "gragc"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SDRK_REFS_DIR");
    for (cfg, _, _) in LIBRARIES {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }
    let Some(prefix) = std::env::var_os("SDRK_REFS_DIR").map(PathBuf::from) else {
        println!(
            "cargo:warning=SDRK_REFS_DIR is not set; the FFTW, KissFFT and Ne10 comparisons are disabled"
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
    for (cfg, archive, name) in LIBRARIES {
        let path = lib.join(archive);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
            println!("cargo:rustc-link-lib=static={name}");
            println!("cargo:rustc-cfg={cfg}");
        }
    }
}
