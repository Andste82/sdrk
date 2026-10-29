//! Unit tests of the golden loader and `assert_matches` with hand-written fixtures.
#![cfg(feature = "std")]
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]

use sdrk_core::{C32, CQ15, Q15};
use sdrk_device::{ErrorKind, SdrkError};
use sdrk_testkit::golden::{self, Array, GoldenError, RefUnit, Tolerance};
use std::fs;
use std::path::PathBuf;

/// A fresh fixture directory below the target directory of this test run.
fn fixture(test: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("golden_loader")
        .join(test);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("k")).unwrap();
    root
}

fn write_case(
    root: &std::path::Path,
    name: &str,
    dtype: &str,
    len: usize,
    bin: &[u8],
    tolerance: &str,
) {
    let json = format!(
        r#"{{"format_version":1,"kernel":"k","case":"{name}","seed":1,"params":{{"gain":0.5,"n":4,"mode":"x"}},
        "arrays":{{"expected":{{"file":"{name}.expected.bin","dtype":"{dtype}","len":{len}}}}},
        "tolerance":{tolerance},"notes":"","provenance":{{}}}}"#
    );
    fs::write(root.join("k").join(format!("{name}.json")), json).unwrap();
    fs::write(root.join("k").join(format!("{name}.expected.bin")), bin).unwrap();
}

fn i16_bytes(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn loads_and_decodes_every_dtype() {
    let root = fixture("dtypes");
    let cases: [(&str, usize, Vec<u8>); 9] = [
        ("u8", 2, vec![1, 2]),
        ("i8", 2, vec![0xff, 2]),
        ("i16", 2, i16_bytes(&[-2, 5])),
        ("i32", 1, (-7i32).to_le_bytes().to_vec()),
        ("f32", 1, 1.5f32.to_le_bytes().to_vec()),
        ("f64", 1, 2.5f64.to_le_bytes().to_vec()),
        ("ci16", 1, i16_bytes(&[3, -4])),
        (
            "cf32",
            1,
            [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
        ),
        ("cf64", 1, f64_bytes(&[1.0, -2.0])),
    ];
    for (dtype, len, bin) in cases {
        write_case(&root, dtype, dtype, len, &bin, r#"{"class":"bit_exact"}"#);
        let case = golden::load_from(&root, "k", dtype).unwrap();
        let array = case.array("expected").unwrap();
        assert_eq!(array.dtype(), dtype);
        assert_eq!(array.len(), len);
    }
    let ci16 = golden::load_from(&root, "k", "ci16").unwrap();
    assert_eq!(
        ci16.array("expected").unwrap(),
        &Array::CI16(vec![CQ15 {
            re: Q15(3),
            im: Q15(-4)
        }])
    );
    assert_eq!(
        golden::load_from(&root, "k", "cf32")
            .unwrap()
            .array("expected")
            .unwrap(),
        &Array::CF32(vec![C32::new(1.0, 2.0)])
    );
    assert_eq!(golden::list_from(&root, "k").unwrap().len(), 9);
    assert_eq!(golden::list_from(&root, "k").unwrap()[0], "cf32");
}

#[test]
fn parameters_are_typed() {
    let root = fixture("params");
    write_case(&root, "c", "u8", 1, &[0], r#"{"class":"bit_exact"}"#);
    let case = golden::load_from(&root, "k", "c").unwrap();
    assert_eq!(case.param_f64("gain").unwrap(), 0.5);
    assert_eq!(case.param_u64("n").unwrap(), 4);
    assert_eq!(case.param_str("mode").unwrap(), "x");
    assert_eq!(
        case.param_u64("gain").unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        case.param_f64("nope").unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        case.array("nope").unwrap_err().kind(),
        ErrorKind::Validation
    );
}

#[test]
fn corrupt_length_is_a_validation_error() {
    let root = fixture("length");
    write_case(
        &root,
        "c",
        "i16",
        4,
        &i16_bytes(&[1, 2, 3]),
        r#"{"class":"bit_exact"}"#,
    );
    let err = golden::load_from(&root, "k", "c").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(matches!(err, GoldenError::Validation(_)));
}

#[test]
fn unknown_dtype_is_a_parse_error() {
    let root = fixture("dtype");
    write_case(
        &root,
        "c",
        "q31",
        1,
        &[0, 0, 0, 0],
        r#"{"class":"bit_exact"}"#,
    );
    let err = golden::load_from(&root, "k", "c").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Parse);
}

#[test]
fn bad_json_and_tolerance_are_parse_errors() {
    let root = fixture("json");
    fs::write(root.join("k").join("broken.json"), "{not json").unwrap();
    assert_eq!(
        golden::load_from(&root, "k", "broken").unwrap_err().kind(),
        ErrorKind::Parse
    );
    write_case(&root, "c", "u8", 1, &[0], r#"{"class":"fuzzy"}"#);
    assert_eq!(
        golden::load_from(&root, "k", "c").unwrap_err().kind(),
        ErrorKind::Parse
    );
    write_case(
        &root,
        "d",
        "u8",
        1,
        &[0],
        r#"{"class":"tolerance","skip":0}"#,
    );
    assert_eq!(
        golden::load_from(&root, "k", "d").unwrap_err().kind(),
        ErrorKind::Parse
    );
    write_case(
        &root,
        "e",
        "u8",
        1,
        &[0],
        r#"{"class":"max_abs_err","max_abs_err":1.0,"min_ref_db":-3.0}"#,
    );
    assert_eq!(
        golden::load_from(&root, "k", "e").unwrap_err().kind(),
        ErrorKind::Parse
    );
}

#[test]
fn wrong_format_version_is_a_validation_error() {
    let root = fixture("version");
    write_case(&root, "c", "u8", 1, &[0], r#"{"class":"bit_exact"}"#);
    let path = root.join("k").join("c.json");
    let text = fs::read_to_string(&path)
        .unwrap()
        .replace("\"format_version\":1", "\"format_version\":2");
    fs::write(&path, text).unwrap();
    assert_eq!(
        golden::load_from(&root, "k", "c").unwrap_err().kind(),
        ErrorKind::Validation
    );
}

#[test]
fn missing_case_is_an_io_error() {
    let root = fixture("missing");
    assert_eq!(
        golden::load_from(&root, "k", "nope").unwrap_err().kind(),
        ErrorKind::Io
    );
    assert_eq!(
        golden::list_from(&root, "nokernel").unwrap_err().kind(),
        ErrorKind::Io
    );
}

#[test]
fn path_traversal_is_rejected() {
    let root = fixture("traversal");
    assert_eq!(
        golden::load_from(&root, "k", "../c").unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        golden::load_from(&root, "..", "c").unwrap_err().kind(),
        ErrorKind::Validation
    );
}

#[test]
fn wrong_expected_value_reports_the_first_mismatch_index() {
    let root = fixture("mismatch");
    write_case(
        &root,
        "c",
        "i16",
        4,
        &i16_bytes(&[10, 20, 30, 40]),
        r#"{"class":"bit_exact"}"#,
    );
    let case = golden::load_from(&root, "k", "c").unwrap();
    let expected = case.array("expected").unwrap();

    let good = [Q15(10), Q15(20), Q15(30), Q15(40)];
    let c = golden::assert_matches(&case, expected, &good).unwrap();
    assert_eq!(c.first_mismatch, None);

    let bad = [Q15(10), Q15(20), Q15(31), Q15(41)];
    match golden::assert_matches(&case, expected, &bad).unwrap_err() {
        GoldenError::Mismatch {
            case,
            first_mismatch,
            snr_db,
            ..
        } => {
            assert_eq!(case, "k/c");
            assert_eq!(first_mismatch, Some(2));
            assert!(snr_db.is_finite() && snr_db > 0.0);
        }
        other => panic!("unexpected error {other}"),
    }
    // Length differences and a wrong sample type are errors too.
    assert!(golden::assert_matches(&case, expected, &good[..3]).is_err());
    assert!(golden::assert_matches(&case, expected, &[0.0f32; 4]).is_err());
}

#[test]
fn bit_exact_float_compares_bits_not_values() {
    let root = fixture("bits");
    write_case(
        &root,
        "c",
        "f32",
        1,
        &0.0f32.to_le_bytes(),
        r#"{"class":"bit_exact"}"#,
    );
    let case = golden::load_from(&root, "k", "c").unwrap();
    let expected = case.array("expected").unwrap();
    golden::assert_matches(&case, expected, &[0.0f32]).unwrap();
    assert!(golden::assert_matches(&case, expected, &[-0.0f32]).is_err());
    assert!(golden::assert_matches(&case, expected, &[f32::NAN]).is_err());
}

#[test]
fn snr_tolerance_applies_skip_and_limit() {
    let root = fixture("snr");
    let expected = f64_bytes(&[1.0, 1.0, 1.0, 1.0]);
    write_case(
        &root,
        "a",
        "f64",
        4,
        &expected,
        r#"{"class":"tolerance","snr_db_min":40.0,"skip":1}"#,
    );
    let case = golden::load_from(&root, "k", "a").unwrap();
    assert_eq!(
        case.tolerance,
        Tolerance::Snr {
            min_db: 40.0,
            skip: 1,
            max_abs_err: None,
            guard_db: None
        }
    );
    let exp = case.array("expected").unwrap();
    // Sample 0 is skipped; the rest has an error of 0.001 (60 dB).
    golden::assert_matches(&case, exp, &[5.0f32, 1.001, 1.001, 1.001]).unwrap();
    let err = golden::assert_matches(&case, exp, &[1.0f32, 1.1, 1.0, 1.0]).unwrap_err();
    assert!(matches!(
        err,
        GoldenError::Mismatch {
            first_mismatch: Some(1),
            ..
        }
    ));

    write_case(
        &root,
        "b",
        "f64",
        4,
        &expected,
        r#"{"class":"tolerance","snr_db_min":10.0,"skip":0,"max_abs_err":0.01}"#,
    );
    let case = golden::load_from(&root, "k", "b").unwrap();
    let exp = case.array("expected").unwrap();
    golden::assert_matches(&case, exp, &[1.0f32, 1.0, 1.0, 1.0]).unwrap();
    // SNR is fine (about 26 dB) but the largest error is above the limit.
    assert!(golden::assert_matches(&case, exp, &[1.0f32, 1.0, 1.0, 1.05]).is_err());
}

#[test]
fn max_abs_err_with_reference_level() {
    let root = fixture("maxerr");
    let expected = f64_bytes(&[-80.0, -10.0, -5.0, -90.0]);
    write_case(
        &root,
        "db",
        "f64",
        4,
        &expected,
        r#"{"class":"max_abs_err","max_abs_err":0.5,"skip":0,"min_ref_db":-20.0,"ref_unit":"db"}"#,
    );
    let case = golden::load_from(&root, "k", "db").unwrap();
    assert_eq!(
        case.tolerance,
        Tolerance::MaxAbsErr {
            max: 0.5,
            skip: 0,
            min_ref: Some((-20.0, RefUnit::Db))
        }
    );
    let exp = case.array("expected").unwrap();
    // Samples below -20 are not checked, however wrong they are.
    let ok = [0.0f32, -10.2, -5.3, 7.0];
    golden::assert_matches(&case, exp, &ok).unwrap();
    let bad = [0.0f32, -10.0, -4.0, 7.0];
    match golden::assert_matches(&case, exp, &bad).unwrap_err() {
        GoldenError::Mismatch { first_mismatch, .. } => assert_eq!(first_mismatch, Some(2)),
        other => panic!("unexpected error {other}"),
    }

    let expected = f64_bytes(&[0.5, 0.001, 0.25, 0.0001]);
    write_case(
        &root,
        "lin",
        "f64",
        4,
        &expected,
        r#"{"class":"max_abs_err","max_abs_err":0.01,"skip":0,"min_ref_db":-30.0,"ref_unit":"linear"}"#,
    );
    let case = golden::load_from(&root, "k", "lin").unwrap();
    let exp = case.array("expected").unwrap();
    // 0.001 is -60 dB and 0.0001 is -80 dB: both are ignored.
    let out = [0.505f32, 5.0, 0.255, 5.0];
    golden::assert_matches(&case, exp, &out).unwrap();
    assert!(golden::assert_matches(&case, exp, &[0.6f32, 0.0, 0.25, 0.0]).is_err());
}

#[test]
fn error_display_and_kinds() {
    let e = GoldenError::Mismatch {
        case: "k/c".into(),
        snr_db: 12.5,
        first_mismatch: Some(3),
        detail: "raw bits differ".into(),
    };
    let text = e.to_string();
    assert!(text.contains("k/c") && text.contains("12.50") && text.contains("index 3"));
    assert_eq!(e.kind(), ErrorKind::Validation);
}
