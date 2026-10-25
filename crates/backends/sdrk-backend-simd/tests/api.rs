//! The public surface of the backend: `Send`/`Sync` (R4.6), the level name, the layer rules.

use sdrk_backend_simd::{Simd, level_name};

fn assert_send_sync<T: Send + Sync + Copy + Default + core::fmt::Debug>() {}

#[test]
fn the_backend_type_is_a_plain_copyable_unit_struct() {
    assert_send_sync::<Simd>();
    assert_eq!(core::mem::size_of::<Simd>(), 0);
}

#[test]
fn the_level_name_is_one_of_the_documented_names() {
    let name = level_name();
    assert!(
        [
            "sse2", "sse4.2", "avx2", "avx512", "neon", "simd128", "scalar"
        ]
        .contains(&name),
        "{name}"
    );
    // The level fixed at compile time (the target features of `-C target-cpu=...`).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "sse4.2"),
        not(feature = "std")
    ))]
    assert_eq!(name, "sse2");
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse4.2",
        target_feature = "popcnt",
        not(target_feature = "avx2"),
        not(feature = "std")
    ))]
    assert_eq!(name, "sse4.2");
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(feature = "std")))]
    assert!(["avx2", "avx512"].contains(&name), "{name}");
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi",
        not(feature = "std")
    ))]
    assert_eq!(name, "avx512");
    #[cfg(all(target_arch = "aarch64", not(feature = "std")))]
    assert_eq!(name, "neon");
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    assert_eq!(name, "simd128");
}

/// The level a CI step means to test, also with run-time detection (under SDE with the CPUID of
/// a level, or on a host of known level): `SDRK_EXPECT_LEVEL` names it. Without the variable the
/// test checks nothing, so that a run on any host passes.
#[test]
fn the_level_is_the_one_the_run_expects() {
    if let Ok(want) = std::env::var("SDRK_EXPECT_LEVEL") {
        assert_eq!(level_name(), want);
    }
}

#[test]
fn the_name_matches_the_cpu_when_detection_is_on() {
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        let expected = if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
        {
            // The AVX-512 level of the vector library needs the Ice Lake feature set.
            None
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            Some("avx2")
        } else {
            None
        };
        if let Some(expected) = expected {
            assert_eq!(level_name(), expected);
        }
    }
}
