//! Golden tests of M41: the Rust model of the RDS reference signal (`sdrk_testkit::rds`) and
//! `sdrk_proto::rds` against the case `rds/reference_short` of the `NumPy` modulator
//! `tests/golden/sdrk_golden/cases/rds.py`.
//!
//! The integer arrays (groups, checkwords, bits) must be equal; the float arrays (RDS baseband
//! and multiplex at the start and at `late_start`) within the case's `max_abs_err`; the `cs16`
//! RF samples within `rf_max_lsb`. The long end-to-end input of M41 is produced by the model,
//! so this cross-check is what ties it to the Python reference (decision D663). The test
//! `python_reference_file_matches_the_model` (ignored) compares a file written by
//! `generate.py --emit rds_reference` with the model when `SDRK_RDS_REFERENCE` names it.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::print_stderr
)]

use num_complex::Complex64;
use sdrk_proto::rds::{self as proto_rds, Group, Offset, Version};
use sdrk_testkit::golden::{self, Array, GoldenCase, Tolerance};
use sdrk_testkit::rds::{
    CAROUSEL_PERIOD, OFFSET_WORDS, RdsReference, TABLE_LEN, carousel_group, checkword, data_bit,
    to_cs16,
};

fn case() -> GoldenCase {
    golden::load("rds", "reference_short").unwrap()
}

fn model() -> RdsReference<Vec<f64>> {
    RdsReference::new(vec![0.0; TABLE_LEN], 0.04).unwrap()
}

fn i32s(c: &GoldenCase, name: &str) -> Vec<i32> {
    match c.array(name).unwrap() {
        Array::I32(v) => v.clone(),
        other => panic!("{name}: i32 expected, got {}", other.dtype()),
    }
}

fn u8s(c: &GoldenCase, name: &str) -> Vec<u8> {
    match c.array(name).unwrap() {
        Array::U8(v) => v.clone(),
        other => panic!("{name}: u8 expected, got {}", other.dtype()),
    }
}

fn f32s(c: &GoldenCase, name: &str) -> Vec<f32> {
    match c.array(name).unwrap() {
        Array::F32(v) => v.clone(),
        other => panic!("{name}: f32 expected, got {}", other.dtype()),
    }
}

fn max_abs_err(c: &GoldenCase) -> f64 {
    match c.tolerance {
        Tolerance::MaxAbsErr { max, skip: 0, .. } => max,
        ref t => panic!("unexpected tolerance {t:?}"),
    }
}

#[test]
fn the_case_of_m41_is_present() {
    assert_eq!(golden::list("rds").unwrap(), ["reference_short"]);
    let c = case();
    assert_eq!(c.seed, 570);
    assert_eq!(c.param_u64("groups").unwrap(), CAROUSEL_PERIOD);
    assert_eq!(c.param_u64("pi").unwrap(), 0xD123);
    assert_eq!(c.param_str("ps").unwrap(), "SDRK TST");
    assert_eq!(c.param_str("rt").unwrap(), "Hello from sdrk!");
    assert_eq!(c.param_u64("grid_num").unwrap(), 19);
    assert_eq!(c.param_u64("grid_den").unwrap(), 3840);
}

#[test]
fn groups_checkwords_and_bits_are_those_of_the_model() {
    let c = case();
    let blocks = i32s(&c, "blocks");
    let checks = i32s(&c, "checkwords");
    assert_eq!(blocks.len(), 4 * CAROUSEL_PERIOD as usize);
    for g in 0..CAROUSEL_PERIOD {
        let words = carousel_group(g);
        for i in 0..4 {
            let at = 4 * g as usize + i;
            assert_eq!(blocks[at], i32::from(words[i]), "group {g} block {i}");
            let offset = OFFSET_WORDS[[0, 1, 2, 4][i]];
            assert_eq!(checks[at], i32::from(checkword(words[i], offset)));
        }
    }
    let data = u8s(&c, "data_bits");
    let tx = u8s(&c, "tx_bits");
    let m = model();
    for (k, (&d, &t)) in data.iter().zip(&tx).enumerate() {
        assert_eq!(data_bit(k as u64), d, "data bit {k}");
        assert_eq!(m.tx_bit(k as u64), t, "transmitted bit {k}");
    }
}

#[test]
fn sdrk_proto_agrees_with_the_reference_groups() {
    let c = case();
    let blocks = i32s(&c, "blocks");
    let checks = i32s(&c, "checkwords");
    let offsets = [Offset::A, Offset::B, Offset::C, Offset::D];
    let mut state = proto_rds::RdsState::new();
    for (g, (words, cws)) in blocks
        .chunks_exact(4)
        .zip(checks.chunks_exact(4))
        .enumerate()
    {
        let words: [u16; 4] = core::array::from_fn(|i| words[i] as u16);
        for ((&w, &cw), o) in words.iter().zip(cws).zip(offsets) {
            assert_eq!(i32::from(proto_rds::checkword(w, o)), cw, "group {g}");
            let block = proto_rds::block26(w, o);
            assert_eq!(Offset::from_syndrome(proto_rds::syndrome(block)), Some(o));
        }
        let group = Group::new(words);
        assert_eq!(group.pi(), 0xD123);
        assert_eq!(group.pty(), 10);
        assert!(!group.tp());
        assert_eq!(
            group.group_type(),
            (if g % 2 == 0 { 0 } else { 2 }, Version::A)
        );
        state.update(&group);
    }
    assert_eq!(state.ps(), "SDRK TST");
    assert_eq!(state.rt(), "Hello from sdrk!");
}

/// The largest error of the model `f` against the stored `f32` values.
fn worst(want: &[f32], start: u64, f: impl Fn(u64) -> f64) -> f64 {
    want.iter()
        .enumerate()
        .map(|(i, &w)| (f(start + i as u64) - f64::from(w)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn baseband_and_multiplex_match_the_reference() {
    let c = case();
    let tol = max_abs_err(&c);
    let m = model();
    let peak = c.param_f64("pulse_peak").unwrap();
    assert!(
        (m.pulse_peak() - peak).abs() <= 1e-12 * peak,
        "pulse peak {} vs {peak}",
        m.pulse_peak()
    );
    let late = c.param_u64("late_start").unwrap();
    for (name, start, rds) in [
        ("baseband", 0, true),
        ("mpx", 0, false),
        ("baseband_late", late, true),
        ("mpx_late", late, false),
    ] {
        let want = f32s(&c, name);
        let err = if rds {
            worst(&want, start, |k| m.baseband(k))
        } else {
            worst(&want, start, |k| m.mpx(k))
        };
        eprintln!("{name}: largest error {err:.3e} (limit {tol:.0e})");
        assert!(err <= tol, "{name}: {err} > {tol}");
    }
}

#[test]
fn rf_matches_the_reference_and_is_chunk_invariant() {
    let c = case();
    let Array::CI16(want) = c.array("rf").unwrap() else {
        panic!("rf must be ci16");
    };
    let lsb = c.param_u64("rf_max_lsb").unwrap() as i32;
    let mut m = model();
    let mut whole = vec![Complex64::new(0.0, 0.0); want.len()];
    m.fill(0, &mut whole).unwrap();
    let mut worst_lsb = 0;
    for (i, (x, w)) in whole.iter().zip(want).enumerate() {
        let q = to_cs16(*x);
        let d = (i32::from(q.re.0) - i32::from(w.re.0))
            .abs()
            .max((i32::from(q.im.0) - i32::from(w.im.0)).abs());
        assert!(d <= lsb, "sample {i}: {q:?} vs {w:?}");
        worst_lsb = worst_lsb.max(d);
    }
    eprintln!("rf: largest difference {worst_lsb} LSB (limit {lsb})");
    // Consecutive pieces give the samples of one call.
    m.reset();
    let mut pieces = vec![Complex64::new(0.0, 0.0); want.len()];
    let mut at = 0;
    for (k, len) in [1usize, 2, 3, 517, 1000, 4096].iter().cycle().enumerate() {
        if at >= pieces.len() {
            break;
        }
        let end = (at + len + k % 3).min(pieces.len());
        m.fill(at as u64, &mut pieces[at..end]).unwrap();
        at = end;
    }
    assert_eq!(whole, pieces);
    assert!(m.fill(0, &mut pieces[..1]).is_err());
}

#[test]
fn the_model_without_rds_differs_only_by_the_rds_term() {
    let with = model();
    let without = RdsReference::new(vec![0.0; TABLE_LEN], 0.0).unwrap();
    for k in (0..240_000).step_by(997) {
        let d = with.mpx(k) - without.mpx(k);
        assert!(d.abs() <= 0.04 * 1.06, "sample {k}: {d}");
    }
    assert!(RdsReference::new(vec![0.0; 3], 0.04).is_err());
    assert!(RdsReference::new(vec![0.0; TABLE_LEN], f64::NAN).is_err());
}

/// Compares a file of `generate.py --emit rds_reference --emit-out PATH` with the model: the
/// difference must be the noise of the stated SNR (within 0.1 dB, measured over the whole
/// file).
#[test]
#[ignore = "needs SDRK_RDS_REFERENCE=<path of generate.py --emit rds_reference>"]
fn python_reference_file_matches_the_model() {
    let path = std::env::var("SDRK_RDS_REFERENCE").expect("SDRK_RDS_REFERENCE is not set");
    let raw = std::fs::read(&path).unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{path}.json")).unwrap()).unwrap();
    let snr_db = meta["snr_db"].as_f64().unwrap();
    assert_eq!(meta["rate_hz"].as_u64().unwrap(), 480_000);
    let n = raw.len() / 4;
    assert_eq!(meta["samples"].as_u64().unwrap() as usize, n);
    let mut m = model();
    let mut buf = vec![Complex64::new(0.0, 0.0); 48_000];
    let (mut sig, mut err) = (0.0, 0.0);
    let mut at = 0;
    while at < n {
        let len = buf.len().min(n - at);
        m.fill(at as u64, &mut buf[..len]).unwrap();
        for (i, x) in buf[..len].iter().enumerate() {
            let b = &raw[4 * (at + i)..4 * (at + i) + 4];
            let re = f64::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0;
            let im = f64::from(i16::from_le_bytes([b[2], b[3]])) / 32768.0;
            sig += x.norm_sqr();
            err += (Complex64::new(re, im) - x).norm_sqr();
        }
        at += len;
    }
    let measured = 10.0 * (sig / err).log10();
    eprintln!("{path}: SNR of the file against the model {measured:.3} dB (stated {snr_db} dB)");
    assert!((measured - snr_db).abs() <= 0.1, "{measured} vs {snr_db}");
}
