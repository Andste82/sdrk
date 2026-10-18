//! `SDRK_GOLDEN_DIR` override. Kept in its own test binary (own process) because it
//! changes the process environment.
#![cfg(feature = "std")]
#![allow(unsafe_code, clippy::unwrap_used)]

use sdrk_testkit::golden;
use std::fs;

#[test]
fn env_var_overrides_the_directory() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden_env");
    let kernel = root.join("k");
    fs::create_dir_all(&kernel).unwrap();
    fs::write(
        kernel.join("c.json"),
        r#"{"format_version":1,"kernel":"k","case":"c","seed":7,"params":{"n":3},
        "arrays":{"x":{"file":"c.x.bin","dtype":"u8","len":2}},
        "tolerance":{"class":"bit_exact"},"notes":"","provenance":{}}"#,
    )
    .unwrap();
    fs::write(kernel.join("c.x.bin"), [1u8, 2]).unwrap();

    // SAFETY: this is the only test of this binary, so no other thread reads the environment.
    unsafe { std::env::set_var("SDRK_GOLDEN_DIR", &root) };
    assert_eq!(golden::golden_dir().unwrap(), root);
    let case = golden::load("k", "c").unwrap();
    assert_eq!(case.seed, 7);
    assert_eq!(case.param_u64("n").unwrap(), 3);
    assert_eq!(golden::list("k").unwrap(), ["c"]);

    // SAFETY: as above.
    unsafe { std::env::remove_var("SDRK_GOLDEN_DIR") };
    let default = golden::golden_dir().unwrap();
    assert!(
        default.ends_with("tests/golden/vectors") || default.ends_with("tests\\golden\\vectors")
    );
    assert!(default.is_dir());
}
