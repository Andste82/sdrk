//! The pure parts of `cargo xtask wasm-pkg` and `wasm-test`.

use xtask::wasm::{
    RELAXED, SCALAR, SIMD, TestEnv, cli_version, find_program, pinned_bindgen_version,
    variant_named, variants, workspace_version,
};

#[test]
fn the_pinned_version_is_read_from_the_manifest() {
    let manifest = r#"
[dependencies]
wasm-bindgen = "=0.2.129"
js-sys = "0.3.106"

[dev-dependencies]
wasm-bindgen-test = "0.3.79"
"#;
    assert_eq!(
        pinned_bindgen_version(manifest),
        Some("0.2.129".to_string())
    );
}

#[test]
fn an_unpinned_or_missing_version_is_not_accepted() {
    assert_eq!(pinned_bindgen_version("wasm-bindgen = \"0.2.129\"\n"), None);
    assert_eq!(
        pinned_bindgen_version("wasm-bindgen-test = \"=0.3.79\"\n"),
        None
    );
    assert_eq!(pinned_bindgen_version(""), None);
}

#[test]
fn the_cli_version_is_read_from_its_output() {
    assert_eq!(
        cli_version("wasm-bindgen 0.2.129\n"),
        Some("0.2.129".to_string())
    );
    assert_eq!(
        cli_version("wasm-bindgen 0.2.129 (abc)"),
        Some("0.2.129".to_string())
    );
    assert_eq!(cli_version("something else 1.0"), None);
    assert_eq!(cli_version(""), None);
}

#[test]
fn the_workspace_version_is_read_from_its_section() {
    let manifest = "[workspace]\nmembers = []\n\n[workspace.package]\nedition = \"2024\"\nversion = \"0.0.1\"\n\n[workspace.dependencies]\nversion = \"9\"\n";
    assert_eq!(workspace_version(manifest), Some("0.0.1".to_string()));
    assert_eq!(workspace_version("[package]\nversion = \"1.0.0\"\n"), None);
}

#[test]
fn the_variants_differ_in_flags_and_features() {
    assert_eq!(variants(false), vec![SIMD, SCALAR]);
    assert_eq!(variants(true), vec![SIMD, SCALAR, RELAXED]);
    assert!(SIMD.rustflags.contains("+simd128"));
    assert!(!SIMD.rustflags.contains("relaxed"));
    assert_eq!(SCALAR.rustflags, "--cfg sdrk_fallback_only");
    assert!(RELAXED.rustflags.contains("+relaxed-simd"));
    assert_eq!(RELAXED.features, &["relaxed-simd"]);
    assert_eq!(variant_named("scalar"), Some(SCALAR));
    assert_eq!(variant_named("nope"), None);
}

#[test]
fn a_browser_run_selects_exactly_one_driver() {
    assert_eq!(TestEnv::parse("node"), Some(TestEnv::Node));
    assert_eq!(TestEnv::parse("safari"), None);
    let (set, unset) = TestEnv::Chrome.runner_env();
    assert_eq!(set, vec![("WASM_BINDGEN_USE_BROWSER", "1")]);
    assert!(unset.contains(&"GECKODRIVER") && !unset.contains(&"CHROMEDRIVER"));
    let (_, unset) = TestEnv::Firefox.runner_env();
    assert!(unset.contains(&"CHROMEDRIVER") && !unset.contains(&"GECKODRIVER"));
    let (set, unset) = TestEnv::Node.runner_env();
    assert_eq!(set, Vec::new());
    assert!(unset.contains(&"WASM_BINDGEN_USE_BROWSER"));
}

#[test]
fn a_browser_run_names_its_driver() {
    assert_eq!(TestEnv::Node.driver(), None);
    assert_eq!(
        TestEnv::Chrome.driver(),
        Some(("CHROMEDRIVER", "chromedriver"))
    );
    assert_eq!(
        TestEnv::Firefox.driver(),
        Some(("GECKODRIVER", "geckodriver"))
    );
}

#[test]
fn programs_are_found_in_a_path_list() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let name = if cfg!(windows) {
        "geckodriver.exe"
    } else {
        "geckodriver"
    };
    std::fs::write(second.path().join(name), b"").unwrap();
    let path = std::env::join_paths([first.path(), second.path()]).unwrap();
    assert_eq!(
        find_program("geckodriver", &path),
        Some(second.path().join(name))
    );
    assert_eq!(find_program("chromedriver", &path), None);
    assert_eq!(find_program("geckodriver", std::ffi::OsStr::new("")), None);
}
