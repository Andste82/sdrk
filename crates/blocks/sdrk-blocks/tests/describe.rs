//! The machine-readable descriptions: rules, consistency with the configs and the blocks, and
//! the generated table of `docs/blocks.md`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::assert_is_empty,
    clippy::range_plus_one,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::multiple_bound_locations,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation
)]

mod common;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use common::rate;
use sdrk::{C32, CQ15};
use sdrk_blocks::{
    AgcBlock, AgcConfig, AmDemodBlock, AmDemodConfig, BiquadBlock, BiquadConfig, BiquadRealBlock,
    Block, BlockDescription, ConvertBlock, ConvertConfig, ConvertOutBlock, ConvertOutConfig,
    DcBlockBlock, DcBlockConfig, DdcBlock, DdcConfig, DucBlock, DucConfig, Error, FirBlock,
    FirConfig, FmDemodBlock, FmDemodConfig, FmModBlock, FmModConfig, FmStereoDemodBlock,
    FmStereoDemodConfig, FrameBlock, Mutability, ParamKind, ParamValue, PortType, PreemphLimitBlock, PreemphLimitConfig, ResampleBlock,
    ResampleConfig, ResampleRealBlock, ScaleBlock, ScaleConfig, SpectrumBlock, SpectrumConfig,
    SquelchBlock, SquelchConfig, SsbDemodBlock, SsbDemodConfig, all_descriptions,
};
use sdrk_testkit::VecStorage;

fn is_snake_case(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !s.starts_with('_')
        && !s.ends_with('_')
        && !s.starts_with(|c: char| c.is_ascii_digit())
}

fn text_ok(what: &str, s: &str) {
    assert!(!s.is_empty(), "{what} is empty");
    assert_eq!(s, s.trim(), "{what} has leading or trailing whitespace");
    assert!(!s.contains('\n'), "{what} contains a line break");
}

#[test]
fn there_are_twenty_unique_kinds() {
    let kinds: BTreeSet<_> = all_descriptions().iter().map(|d| d.kind).collect();
    assert_eq!(kinds.len(), 20);
    assert_eq!(all_descriptions().len(), 20);
    for kind in [
        "convert",
        "convert_out",
        "scale",
        "dc_block",
        "fir",
        "ddc",
        "duc",
        "fm_demod",
        "preemph_limit",
        "fm_mod",
        "fm_stereo_demod",
        "resample",
        "spectrum",
        "squelch",
        "agc",
        "am_demod",
        "ssb_demod",
        "biquad",
    ] {
        assert!(kinds.contains(kind), "missing kind {kind}");
    }
}

#[test]
fn texts_are_well_formed() {
    for d in all_descriptions() {
        text_ok(&format!("{} kind", d.kind), d.kind);
        assert!(is_snake_case(d.kind), "{} is not snake_case", d.kind);
        text_ok(&format!("{} summary", d.kind), d.summary);
        assert!(
            d.summary.chars().count() <= 120,
            "{} summary too long",
            d.kind
        );
        assert!(
            d.summary.ends_with('.'),
            "{} summary needs a period",
            d.kind
        );
        text_ok(&format!("{} details", d.kind), d.details);
        assert!(
            d.details.len() > d.summary.len(),
            "{} details too short",
            d.kind
        );
        for ports in [d.inputs, d.outputs] {
            assert!(!ports.is_empty(), "{} has no ports", d.kind);
            let mut names = BTreeSet::new();
            for p in ports {
                assert!(is_snake_case(p.name));
                assert!(names.insert(p.name), "duplicate port {}", p.name);
                text_ok(&format!("{}.{} description", d.kind, p.name), p.description);
                assert!(!p.types.is_empty());
                if let PortType::Frame { len_param } = p.types[0] {
                    assert!(
                        d.param(len_param).is_some(),
                        "{} frame length parameter {len_param} is missing",
                        d.kind
                    );
                }
            }
        }
    }
}

#[test]
fn parameters_are_well_formed() {
    for d in all_descriptions() {
        let mut names = BTreeSet::new();
        for p in d.params {
            let what = format!("{}.{}", d.kind, p.name);
            assert!(is_snake_case(p.name), "{what} is not snake_case");
            assert!(names.insert(p.name), "duplicate parameter {what}");
            text_ok(&format!("{what} typical"), p.typical);
            text_ok(&format!("{what} description"), p.description);
            // The unit and the name suffix agree.
            use sdrk_blocks::Unit;
            for (suffix, unit) in [
                ("_hz", Unit::Hz),
                ("_db", Unit::Decibel),
                ("_us", Unit::Microseconds),
            ] {
                assert_eq!(
                    p.name.ends_with(suffix),
                    p.unit == unit,
                    "{what}: unit {:?} does not match the name",
                    p.unit
                );
            }
            match p.kind {
                ParamKind::F64 { min, max, default } => {
                    assert!(min < max, "{what}: empty range");
                    if let Some(v) = default {
                        assert!(
                            (min..=max).contains(&v),
                            "{what}: default outside the range"
                        );
                    }
                    assert_ne!(p.unit, Unit::Samples, "{what}: samples are integers");
                }
                ParamKind::I64 { min, max, default } => {
                    assert!(min < max, "{what}: empty range");
                    if let Some(v) = default {
                        assert!(
                            (min..=max).contains(&v),
                            "{what}: default outside the range"
                        );
                    }
                }
                ParamKind::Choice { options, default } => {
                    assert!(!options.is_empty());
                    assert!(usize::from(default) < options.len());
                    let unique: BTreeSet<_> = options.iter().collect();
                    assert_eq!(unique.len(), options.len());
                    assert!(options.iter().all(|o| is_snake_case(o)));
                }
                ParamKind::Bool { .. } | ParamKind::Format => {}
                _ => panic!("{what}: new kind without a test"),
            }
        }
        for r in d.required {
            let p = d
                .param(r)
                .unwrap_or_else(|| panic!("{} requires unknown {r}", d.kind));
            assert_eq!(
                p.default_value(),
                None,
                "{}.{r} is required but has a default",
                d.kind
            );
        }
    }
}

#[test]
fn parameters_without_default_are_required_or_rate_dependent() {
    // A parameter without a default must be listed as required, or its default depends on the
    // sample rate (the description says so).
    for d in all_descriptions() {
        for p in d.params {
            if p.default_value().is_none() && !d.required.contains(&p.name) {
                let text = format!("{} {}", p.typical, p.description).to_lowercase();
                assert!(
                    text.contains("default"),
                    "{}.{} has no default, is not required and does not explain its default",
                    d.kind,
                    p.name
                );
            }
        }
    }
}

fn find(kind: &str) -> &'static BlockDescription {
    all_descriptions()
        .iter()
        .copied()
        .find(|d| d.kind == kind)
        .unwrap()
}

/// Every description default equals what the default config reports.
fn defaults_match(kind: &str, get: impl Fn(&str) -> Result<ParamValue, Error>, skip: &[&str]) {
    let d = find(kind);
    for p in d.params {
        if skip.contains(&p.name) {
            continue;
        }
        if let Some(v) = p.default_value() {
            assert_eq!(get(p.name), Ok(v), "{kind}.{}: default differs", p.name);
        }
    }
}

#[test]
fn config_defaults_match_the_descriptions() {
    defaults_match("scale", |n| ScaleConfig::default().get(n), &[]);
    defaults_match("dc_block", |n| DcBlockConfig::default().get(n), &[]);
    defaults_match("fir", |n| FirConfig::default().get(n), &[]);
    defaults_match("ddc", |n| DdcConfig::default().get(n), &[]);
    defaults_match("duc", |n| DucConfig::default().get(n), &[]);
    defaults_match("fm_demod", |n| FmDemodConfig::default().get(n), &[]);
    defaults_match("fm_mod", |n| FmModConfig::default().get(n), &[]);
    defaults_match(
    defaults_match(
        "preemph_limit",
        |n| PreemphLimitConfig::default().get(n),
        &[],
    );
        "fm_stereo_demod",
        |n| FmStereoDemodConfig::default().get(n),
        &[],
    );
    defaults_match("resample", |n| ResampleConfig::default().get(n), &[]);
    defaults_match(
        "spectrum",
        |n| SpectrumConfig::default().get(n),
        &["interval_samples"],
    );
    defaults_match("convert", |n| ConvertConfig::default().get(n), &[]);
    defaults_match("squelch", |n| SquelchConfig::default().get(n), &[]);
    defaults_match("agc", |n| AgcConfig::default().get(n), &[]);
    defaults_match("am_demod", |n| AmDemodConfig::default().get(n), &[]);
    defaults_match("ssb_demod", |n| SsbDemodConfig::default().get(n), &[]);
    defaults_match("biquad", |n| BiquadConfig::default().get(n), &[]);
    defaults_match("convert_out", |n| ConvertOutConfig::default().get(n), &[]);
}

/// Every parameter of the description is known to the config: `get` finds it and `set` accepts
/// its default or a value inside the range.
fn config_round_trip(
    kind: &str,
    get: impl Fn(&str) -> Result<ParamValue, Error>,
    set: impl Fn(&str, ParamValue) -> Result<(), Error>,
) {
    let d = find(kind);
    for p in d.params {
        let sample = match p.kind {
            ParamKind::F64 { min, max, .. } => ParamValue::F64(min + (max - min) / 3.0),
            ParamKind::I64 { min, max, .. } => ParamValue::I64(min + (max - min) / 3),
            ParamKind::Bool { default } => ParamValue::Bool(!default),
            ParamKind::Choice { options, .. } => ParamValue::Choice(options.len() as u16 - 1),
            ParamKind::Format => ParamValue::Format(sdrk_device::formats::CS16),
            _ => unreachable!(),
        };
        set(p.name, sample).unwrap_or_else(|e| panic!("{kind}.{}: {e}", p.name));
        assert_eq!(get(p.name), Ok(sample), "{kind}.{}", p.name);
        // The wrong type and a value outside the range are rejected.
        assert_eq!(
            set(p.name, ParamValue::Choice(60_000))
                .err()
                .map(|e| matches!(e, Error::WrongParamType | Error::OutOfRange { .. })),
            Some(true)
        );
        match p.kind {
            ParamKind::F64 { max, .. } => {
                assert!(matches!(
                    set(p.name, ParamValue::F64(max * 2.0 + 1.0)),
                    Err(Error::OutOfRange { .. })
                ));
                assert!(matches!(
                    set(p.name, ParamValue::F64(f64::NAN)),
                    Err(Error::OutOfRange { .. })
                ));
            }
            ParamKind::I64 { max, .. } => {
                assert!(matches!(
                    set(p.name, ParamValue::I64(max + 1)),
                    Err(Error::OutOfRange { .. })
                ));
            }
            _ => {}
        }
    }
    assert_eq!(
        set("no_such_parameter", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(get("no_such_parameter"), Err(Error::UnknownParameter));
}

macro_rules! round_trip {
    ($kind:literal, $cfg:ty) => {{
        let cell = std::cell::RefCell::new(<$cfg>::default());
        config_round_trip(
            $kind,
            |n| cell.borrow().get(n),
            |n, v| cell.borrow_mut().set(n, v),
        );
    }};
}

#[test]
fn configs_accept_every_described_parameter() {
    round_trip!("scale", ScaleConfig);
    round_trip!("dc_block", DcBlockConfig);
    round_trip!("fir", FirConfig);
    round_trip!("ddc", DdcConfig);
    round_trip!("duc", DucConfig);
    round_trip!("fm_demod", FmDemodConfig);
    round_trip!("fm_mod", FmModConfig);
    round_trip!("preemph_limit", PreemphLimitConfig);
    round_trip!("fm_stereo_demod", FmStereoDemodConfig);
    round_trip!("resample", ResampleConfig);
    round_trip!("spectrum", SpectrumConfig);
    round_trip!("convert", ConvertConfig);
    round_trip!("convert_out", ConvertOutConfig);
    round_trip!("squelch", SquelchConfig);
    round_trip!("agc", AgcConfig);
    round_trip!("am_demod", AmDemodConfig);
    round_trip!("ssb_demod", SsbDemodConfig);
    round_trip!("biquad", BiquadConfig);
}

/// Every runtime parameter is accepted by `set_param` and readable by `param`; build parameters
/// are readable and read-only.
fn check_block_params<B: Block>(kind: &str, block: &mut B) {
    let d = find(kind);
    assert_eq!(block.description().kind, kind);
    for p in d.params {
        let value = block
            .param(p.name)
            .unwrap_or_else(|e| panic!("{kind}.{}: {e}", p.name));
        p.check(&value)
            .unwrap_or_else(|e| panic!("{kind}.{}: {value:?} is invalid: {e}", p.name));
        match p.mutability {
            Mutability::Runtime => block
                .set_param(p.name, value)
                .unwrap_or_else(|e| panic!("{kind}.{}: {e}", p.name)),
            Mutability::Build => assert_eq!(
                block.set_param(p.name, value),
                Err(Error::ReadOnlyParameter),
                "{kind}.{}",
                p.name
            ),
            _ => panic!("new mutability"),
        }
        assert_eq!(block.param(p.name), Ok(value), "{kind}.{}", p.name);
    }
    assert_eq!(block.param("nope"), Err(Error::UnknownParameter));
    assert_eq!(
        block.set_param("nope", ParamValue::Bool(true)),
        Err(Error::UnknownParameter)
    );
}

#[test]
fn blocks_accept_and_report_every_described_parameter() {
    let mut s = VecStorage;
    let r = rate(240_000);
    check_block_params(
        "convert",
        &mut ConvertBlock::<C32, _>::new(&ConvertConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "convert_out",
        &mut ConvertOutBlock::<CQ15, _>::new(&ConvertOutConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "scale",
        &mut ScaleBlock::<C32, _>::new(&ScaleConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "preemph_limit",
        &mut PreemphLimitBlock::<sdrk::Q15, _>::new(&PreemphLimitConfig::default(), r, &mut s)
            .unwrap(),
    );
    check_block_params(
        "squelch",
        &mut SquelchBlock::<CQ15, _>::new(&SquelchConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "agc",
        &mut AgcBlock::<C32, _>::new(&AgcConfig::default(), r, &mut s).unwrap(),
    );
    let am = AmDemodConfig {
        audio_decimation: 5,
        ..AmDemodConfig::default()
    };
    check_block_params(
        "am_demod",
        &mut AmDemodBlock::<CQ15, _>::new(&am, r, &mut s).unwrap(),
    );
    let ssb = SsbDemodConfig {
        audio_decimation: 5,
        transition_hz: 2_000.0,
        ..SsbDemodConfig::default()
    };
    check_block_params(
        "ssb_demod",
        &mut SsbDemodBlock::<CQ15, _>::new(&ssb, r, &mut s).unwrap(),
    );
    let biquad = BiquadConfig {
        cutoff_hz: 2_000.0,
        order: 4,
        ..BiquadConfig::default()
    };
    check_block_params(
        "biquad",
        &mut BiquadBlock::<CQ15, _>::new(&biquad, r, &mut s).unwrap(),
    );
    check_block_params(
        "biquad",
        &mut BiquadRealBlock::<f32, _>::new(&biquad, r, &mut s).unwrap(),
    );
    check_block_params(
        "dc_block",
        &mut DcBlockBlock::<CQ15, _>::new(&DcBlockConfig::default(), r, &mut s).unwrap(),
    );
    let fir = FirConfig {
        cutoff_hz: 10e3,
        transition_hz: 5e3,
        ..FirConfig::default()
    };
    check_block_params(
        "fir",
        &mut FirBlock::<C32, _>::new(&fir, r, &mut s).unwrap(),
    );
    check_block_params(
        "ddc",
        &mut DdcBlock::<CQ15, _>::new(&DdcConfig::default(), r, &mut s).unwrap(),
    );
    let duc = DucConfig {
        interpolation: 4,
        ..DucConfig::default()
    };
    check_block_params(
        "duc",
        &mut DucBlock::<C32, _>::new(&duc, rate(48_000), &mut s).unwrap(),
    );
    let demod = FmDemodConfig {
        audio_decimation: 5,
        ..FmDemodConfig::default()
    };
    check_block_params(
        "fm_demod",
        &mut FmDemodBlock::<CQ15, _>::new(&demod, r, &mut s).unwrap(),
    );
    check_block_params(
        "fm_stereo_demod",
        &mut FmStereoDemodBlock::<C32, _>::new(
            &FmStereoDemodConfig::default(),
            rate(480_000),
            &mut s,
        )
        .unwrap(),
    );
    check_block_params(
        "fm_mod",
        &mut FmModBlock::<C32, _>::new(&FmModConfig::default(), r, &mut s).unwrap(),
    );
    let tuned = ResampleConfig {
        tunable: true,
        ..ResampleConfig::default()
    };
    check_block_params(
        "resample",
        &mut ResampleBlock::<C32, _>::new(&tuned, rate(48_000), &mut s).unwrap(),
    );
    check_block_params(
        "resample",
        &mut ResampleRealBlock::<f32, _>::new(&tuned, rate(48_000), &mut s).unwrap(),
    );
    let sp = SpectrumConfig {
        fft_size: 64,
        ..SpectrumConfig::default()
    };
    let mut block = SpectrumBlock::<C32, _>::new(&sp, r, &mut s).unwrap();
    let d = find("spectrum");
    for p in d.params {
        let value = FrameBlock::param(&block, p.name).unwrap();
        p.check(&value).unwrap();
        match p.mutability {
            Mutability::Runtime => FrameBlock::set_param(&mut block, p.name, value).unwrap(),
            Mutability::Build => assert_eq!(
                FrameBlock::set_param(&mut block, p.name, value),
                Err(Error::ReadOnlyParameter)
            ),
            _ => panic!(),
        }
    }
    assert_eq!(FrameBlock::description(&block).kind, "spectrum");
}

#[test]
fn a_non_tunable_resampler_rejects_ppm() {
    let cfg = ResampleConfig::default();
    let mut block = ResampleBlock::<C32, _>::new(&cfg, rate(48_000), &mut VecStorage).unwrap();
    assert_eq!(
        block.set_param("ppm", ParamValue::F64(10.0)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(block.param("ppm"), Ok(ParamValue::F64(0.0)));
    let bad = ResampleConfig { ppm: 5.0, ..cfg };
    assert!(matches!(
        bad.validate(rate(48_000)),
        Err(Error::InvalidParameter { name: "ppm", .. })
    ));
}

// ---------------------------------------------------------------------------------------------
// docs/blocks.md

const BEGIN: &str = "<!-- blocks-table:begin -->";
const END: &str = "<!-- blocks-table:end -->";

fn port_types(ports: &[sdrk_blocks::PortDesc]) -> String {
    let mut out = Vec::new();
    for p in ports {
        let types: Vec<String> = p
            .types
            .iter()
            .map(|t| match t {
                PortType::C32 => "c32".to_string(),
                PortType::CQ15 => "cq15".to_string(),
                PortType::F32 => "f32".to_string(),
                PortType::Q15 => "q15".to_string(),
                PortType::Raw => "raw".to_string(),
                PortType::Frame { len_param } => format!("frame({len_param})"),
                _ => "?".to_string(),
            })
            .collect();
        out.push(format!("`{}`: {}", p.name, types.join(", ")));
    }
    out.join("; ")
}

fn kind_text(k: &ParamKind) -> String {
    match k {
        ParamKind::F64 { min, max, default } => format!(
            "f64, {min} to {max}, {}",
            default.map_or("no default".to_string(), |d| format!("default {d}"))
        ),
        ParamKind::I64 { min, max, default } => format!(
            "integer, {min} to {max}, {}",
            default.map_or("no default".to_string(), |d| format!("default {d}"))
        ),
        ParamKind::Bool { default } => format!("flag, default {default}"),
        ParamKind::Choice { options, default } => format!(
            "choice {}, default {}",
            options.join(" / "),
            options[usize::from(*default)]
        ),
        ParamKind::Format => "sample format name, no default".to_string(),
        _ => "?".to_string(),
    }
}

fn table() -> String {
    let mut s = String::new();
    s.push_str(
        "| Kind | Summary | Inputs | Outputs | Required |\n| --- | --- | --- | --- | --- |\n",
    );
    for d in all_descriptions() {
        let required = if d.required.is_empty() {
            "-".to_string()
        } else {
            d.required
                .iter()
                .map(|r| format!("`{r}`"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            s,
            "| `{}` | {} | {} | {} | {} |",
            d.kind,
            d.summary,
            port_types(d.inputs),
            port_types(d.outputs),
            required
        )
        .unwrap();
    }
    s.push('\n');
    s.push_str("| Block | Parameter | Unit | When | Type, range, default | Typical |\n| --- | --- | --- | --- | --- | --- |\n");
    for d in all_descriptions() {
        for p in d.params {
            let when = match p.mutability {
                Mutability::Build => "build",
                Mutability::Runtime => "runtime",
                _ => "?",
            };
            let unit = if p.unit.symbol().is_empty() {
                "-"
            } else {
                p.unit.symbol()
            };
            writeln!(
                s,
                "| `{}` | `{}` | {} | {} | {} | {} |",
                d.kind,
                p.name,
                unit,
                when,
                kind_text(&p.kind),
                p.typical
            )
            .unwrap();
        }
    }
    s
}

fn blocks_md() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("docs")
        .join("blocks.md")
}

#[test]
fn docs_blocks_md_contains_the_generated_table() {
    let text = std::fs::read_to_string(blocks_md())
        .unwrap()
        .replace("\r\n", "\n");
    let begin = text.find(BEGIN).expect("begin marker in docs/blocks.md") + BEGIN.len();
    let end = text.find(END).expect("end marker in docs/blocks.md");
    let committed = text[begin..end].trim_matches('\n');
    let generated = table();
    assert_eq!(
        committed,
        generated.trim_matches('\n'),
        "docs/blocks.md is out of date; regenerate with \
         `cargo test -p sdrk-blocks --test describe -- --ignored print --nocapture`"
    );
}

/// Prints the Markdown table of `docs/blocks.md` to stdout.
#[test]
#[ignore = "prints the table of docs/blocks.md; run with --ignored print --nocapture"]
#[allow(clippy::print_stdout)]
fn print() {
    println!("{BEGIN}\n\n{}\n{END}", table());
}
