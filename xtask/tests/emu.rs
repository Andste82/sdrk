//! The pure parts of `cargo xtask emu`: the protocol judge, the argument parser and the target
//! selection.
#![allow(clippy::unwrap_used)]

use std::time::Duration;

use xtask::emu::{
    FULL_TIMEOUT_S, Options, Outcome, QUICK_TIMEOUT_S, Row, TARGETS, build_features, judge,
    parse_args, select, strip_ansi, summary_table, target_named,
};

const GOOD: &str = "\
boot: ESP-IDF 2nd stage bootloader
SDRK-SUITE START 3
SDRK-CASE PASS one
SDRK-CASE PASS two
SDRK-CASE PASS three
SDRK-SUITE END passed=3 failed=0
";

#[test]
fn a_log_with_only_pass_lines_and_a_clean_end_passes() {
    let v = judge(GOOD);
    assert!(v.ok, "{:?}", v.reasons);
    assert_eq!((v.passed, v.failed, v.pass_lines), (3, 0, 3));
    assert!(v.ended);
}

#[test]
fn a_fail_line_is_a_failure() {
    let log = "SDRK-SUITE START 2\nSDRK-CASE PASS a\nSDRK-CASE FAIL b mismatch (index 3)\nSDRK-SUITE END passed=1 failed=0\n";
    let v = judge(log);
    assert!(!v.ok);
    assert_eq!(v.fail_lines, vec!["b mismatch (index 3)".to_string()]);
}

#[test]
fn a_missing_end_is_a_failure() {
    let log = "SDRK-SUITE START 2\nSDRK-CASE PASS a\nSDRK-CASE PASS b\n";
    let v = judge(log);
    assert!(!v.ok);
    assert!(!v.ended);
    assert!(v.reasons.iter().any(|r| r.contains("no `SDRK-SUITE END`")));
}

#[test]
fn a_panic_is_a_failure_even_with_a_clean_end() {
    let log = format!("SDRK-PANIC index out of bounds\n{GOOD}");
    let v = judge(&log);
    assert!(!v.ok);
    assert_eq!(v.panics, vec!["index out of bounds".to_string()]);
}

#[test]
fn a_panic_without_end_is_a_failure() {
    let v = judge("SDRK-SUITE START 1\n\nSDRK-PANIC overflow\n");
    assert!(!v.ok);
    assert!(!v.ended);
}

#[test]
fn failed_one_in_the_end_line_is_a_failure() {
    let log = "SDRK-SUITE START 2\nSDRK-CASE PASS a\nSDRK-SUITE END passed=1 failed=1\n";
    let v = judge(log);
    assert!(!v.ok);
    assert_eq!(v.failed, 1);
    assert!(v.reasons.iter().any(|r| r.contains("failed=1")));
}

#[test]
fn an_empty_log_fails() {
    assert!(!judge("").ok);
    assert!(!judge("hello\nworld\n").ok);
}

#[test]
fn a_boot_loop_with_two_starts_fails() {
    let log = format!("{GOOD}{GOOD}");
    let v = judge(&log);
    assert!(!v.ok);
}

#[test]
fn counts_that_do_not_add_up_fail() {
    // START announces 3 but END counts 2; PASS lines are missing.
    let log = "SDRK-SUITE START 3\nSDRK-CASE PASS a\nSDRK-CASE PASS b\nSDRK-SUITE END passed=2 failed=0\n";
    assert!(!judge(log).ok);
    // END claims 3 passes but only two PASS lines were seen (a lost line).
    let log = "SDRK-SUITE START 3\nSDRK-CASE PASS a\nSDRK-CASE PASS b\nSDRK-SUITE END passed=3 failed=0\n";
    assert!(!judge(log).ok);
}

#[test]
fn a_malformed_end_line_fails() {
    let log = "SDRK-SUITE START 1\nSDRK-CASE PASS a\nSDRK-SUITE END passed=1\n";
    assert!(!judge(log).ok);
}

#[test]
fn colour_codes_and_prefixes_do_not_hide_the_protocol() {
    let log = "\u{1b}[0;32mI (36) app: SDRK-SUITE START 1\u{1b}[0m\r\n\
               \u{1b}[0mSDRK-CASE PASS a\u{1b}[0m\r\n\
               SDRK-SUITE END passed=1 failed=0\r\n";
    assert!(judge(log).ok);
    assert_eq!(strip_ansi("\u{1b}[1;31mred\u{1b}[0m"), "red");
}

#[test]
fn the_exit_code_of_the_emulator_is_not_an_input() {
    // `judge` only takes the text; a log that passes passes whatever the emulator returned.
    assert!(judge(GOOD).ok);
}

#[test]
fn arguments_are_parsed() {
    let o = parse_args(&[
        "--target",
        "qemu-cortex-m4",
        "--full",
        "--features",
        "a,b",
        "--timeout",
        "10",
        "--skip-missing",
    ])
    .unwrap();
    assert_eq!(o.targets, vec!["qemu-cortex-m4".to_string()]);
    assert!(o.full && o.skip_missing);
    assert_eq!(o.features, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(o.timeout_s, Some(10));
    assert!(parse_args(&["--bogus"]).is_err());
    assert!(parse_args(&["--tier", "3"]).is_err());
    assert!(parse_args(&["--target"]).is_err());
}

#[test]
fn the_full_profile_adds_the_feature_full() {
    let o = parse_args(&["--full", "--features", "inject-failure"]).unwrap();
    assert_eq!(
        build_features(&o),
        vec!["inject-failure".to_string(), "full".to_string()]
    );
    assert_eq!(build_features(&Options::default()), Vec::<String>::new());
}

fn names(o: &Options) -> Vec<&'static str> {
    select(o).unwrap().iter().map(|t| t.name).collect()
}

#[test]
fn the_selection_follows_tier_target_and_all() {
    assert_eq!(names(&Options::default()), vec!["qemu-cortex-m4"]);
    let tier2 = parse_args(&["--tier", "2"]).unwrap();
    assert_eq!(
        names(&tier2),
        vec!["qemu-cortex-m55", "qemu-riscv32", "esp32p4", "esp32s3"]
    );
    let all = parse_args(&["--all"]).unwrap();
    assert_eq!(names(&all).len(), 5);
    let one = parse_args(&["--target", "esp32p4"]).unwrap();
    assert_eq!(names(&one), vec!["esp32p4"]);
    let two = parse_args(&["--target", "esp32p4", "--target", "qemu-cortex-m4"]).unwrap();
    assert_eq!(names(&two), vec!["qemu-cortex-m4", "esp32p4"]);
    assert!(select(&parse_args(&["--target", "nope"]).unwrap()).is_err());
}

#[test]
fn the_table_lists_the_five_targets_with_their_tiers() {
    assert_eq!(TARGETS.len(), 5);
    assert_eq!(target_named("qemu-cortex-m4").unwrap().tier, 1);
    assert_eq!(TARGETS.iter().filter(|t| t.tier == 1).count(), 1);
    for t in &TARGETS {
        assert!(t.quick_s >= QUICK_TIMEOUT_S);
        assert!(t.full_s >= FULL_TIMEOUT_S);
        assert_eq!(t.timeout(false), Duration::from_secs(t.quick_s));
    }
    // The target crates exist and are named after the directory.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    for t in &TARGETS {
        let manifest =
            std::fs::read_to_string(root.join("targets").join(t.name).join("Cargo.toml")).unwrap();
        assert!(
            manifest.contains(&format!("name = \"{}\"", t.binary())),
            "{}",
            t.name
        );
    }
}

#[test]
fn the_summary_table_shows_every_outcome() {
    let row = |name, outcome, counts| Row {
        name,
        outcome,
        counts,
        elapsed: Duration::from_secs(7),
        log: "x.log".into(),
    };
    let table = summary_table(&[
        row("a", Outcome::Pass, Some((75, 0))),
        row("b", Outcome::Fail("PANIC x".into()), None),
        row("c", Outcome::Skipped("missing: esp-emu".into()), None),
    ]);
    assert!(table.contains("PASS") && table.contains("FAIL") && table.contains("SKIP"));
    assert!(table.contains("75") && table.contains("PANIC x"));
}

#[test]
fn the_pinned_checksums_are_sha256_hex_strings() {
    use xtask::emu::pins;
    for sum in [
        pins::ESP_EMU_X86_64_LINUX,
        pins::ESP_EMU_AARCH64_LINUX,
        pins::ESP_QEMU_X86_64_LINUX,
        pins::ESP_QEMU_AARCH64_LINUX,
        pins::ESP_QEMU_X86_64_WINDOWS,
    ] {
        assert_eq!(sum.len(), 64, "{sum}");
        assert!(
            sum.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "{sum}"
        );
    }
    assert!(pins::ESP_QEMU_TAG.contains("esp-develop"));
    assert!(pins::ESP_QEMU_STAMP.starts_with("esp_develop"));
}

#[test]
fn every_target_needs_its_tools() {
    use xtask::emu::{Tool, tools_of};
    let tools = |name: &str| tools_of(target_named(name).unwrap());
    assert_eq!(tools("qemu-cortex-m4"), vec![Tool::QemuArm]);
    assert_eq!(tools("qemu-cortex-m55"), vec![Tool::QemuArm]);
    assert_eq!(tools("qemu-riscv32"), vec![Tool::QemuRiscv32]);
    assert_eq!(tools("esp32p4"), vec![Tool::EspEmu, Tool::Espflash]);
    assert_eq!(
        tools("esp32s3"),
        vec![Tool::EspQemu, Tool::Espflash, Tool::XtensaGcc]
    );
    // Every tool has an install hint.
    for t in &TARGETS {
        for tool in tools_of(t) {
            assert_ne!(tool.hint(), "");
        }
    }
}

#[test]
fn the_instruction_count_is_the_difference_per_sample() {
    use xtask::count_embedded::per_sample;
    let v = per_sample(1_000, 1_000 + 16 * 1024 * 10, 16, 1024).unwrap();
    assert!((v - 10.0).abs() < 1e-12);
    assert!(per_sample(2_000, 1_000, 16, 1024).is_err());
}

#[cfg(unix)]
mod process {
    use std::process::Command;
    use std::time::{Duration, Instant};

    use xtask::emu::{judge, run_emulator};

    fn shell(script: &str) -> Command {
        let mut c = Command::new("sh");
        c.args(["-c", script]);
        c
    }

    #[test]
    fn the_emulator_is_stopped_once_the_suite_has_ended() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("t.log");
        let started = Instant::now();
        // A process that prints the protocol and then keeps running, like the ESP emulators.
        let run = run_emulator(
            shell("echo SDRK-SUITE START 1; echo SDRK-CASE PASS a; echo SDRK-SUITE END passed=1 failed=0; sleep 60"),
            Duration::from_secs(50),
            &log,
        )
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(20));
        assert!(!run.timed_out);
        assert!(judge(&run.text).ok);
        assert!(
            std::fs::read_to_string(&log)
                .unwrap()
                .contains("SDRK-SUITE END")
        );
    }

    #[test]
    fn a_silent_emulator_times_out_and_fails() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("t.log");
        let run = run_emulator(
            shell("echo SDRK-SUITE START 1; sleep 60"),
            Duration::from_secs(1),
            &log,
        )
        .unwrap();
        assert!(run.timed_out);
        assert!(!judge(&run.text).ok);
    }

    #[test]
    fn a_zero_exit_code_does_not_make_a_failing_log_pass() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("t.log");
        let run = run_emulator(
            shell("echo SDRK-SUITE START 1; echo SDRK-CASE FAIL a boom; echo SDRK-SUITE END passed=0 failed=1; exit 0"),
            Duration::from_secs(20),
            &log,
        )
        .unwrap();
        assert!(!judge(&run.text).ok);
    }

    #[test]
    fn standard_error_is_part_of_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("t.log");
        let run = run_emulator(
            shell("echo SDRK-PANIC oops 1>&2"),
            Duration::from_secs(20),
            &log,
        )
        .unwrap();
        assert!(run.text.contains("SDRK-PANIC oops"));
    }
}

#[test]
fn a_marker_in_quotes_is_a_mention_not_a_protocol_line() {
    // `esp-emu` logs this line when the exit string matched.
    let log = "SDRK-SUITE START 1\nSDRK-CASE PASS a\n[INFO  esp_emu] Exit-on string 'SDRK-SUITE END' found in output\nSDRK-SUITE END passed=1 failed=0\n";
    let v = judge(log);
    assert!(v.ok, "{:?}", v.reasons);
    // Without the real line the mention does not count as an end.
    let log = "SDRK-SUITE START 1\nSDRK-CASE PASS a\n[INFO  esp_emu] Exit-on string 'SDRK-SUITE END' found in output\n";
    assert!(!judge(log).ok);
}
