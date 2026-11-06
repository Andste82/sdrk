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
    AddBlock, AddRealBlock, AfcBlock, AfcConfig, AgcBlock, AgcConfig, AmDemodBlock, AmDemodConfig,
    AmModBlock, AmModConfig, AutoNotchConfig, AutoNotchRealBlock, BiquadBlock, BiquadConfig,
    BiquadRealBlock, Block, BlockDescription, ConvertBlock, ConvertConfig, ConvertOutBlock,
    ConvertOutConfig, CostasBlock, CostasConfig, DcBlockBlock, DcBlockConfig, DcBlockRealBlock,
    DdcBlock, DdcConfig, DecimateConfig, DeemphasisConfig, DelayConfig, DucBlock, DucConfig, Error,
    FirBlock, FirConfig, FmDemodBlock, FmDemodConfig, FmModBlock, FmModConfig, FmStereoDemodBlock,
    FmStereoDemodConfig, FmStereoModBlock, FmStereoModConfig, FrameDecodeBlock, FrameDecodeConfig,
    FrameEncodeBlock, FrameEncodeConfig, FramePolarity, FskDemodBlock, FskDemodConfig, FskModBlock,
    FskModConfig, ImagPartBlock, InterpolateConfig, MakeComplexBlock, MulBlock, MulRealBlock,
    Mutability, NcoBlock, NcoConfig, NcoRealBlock, NoiseBlankBlock, NoiseBlankConfig, ParamKind,
    ParamValue, PortType, PreemphLimitBlock, PreemphLimitConfig, PskDemodBlock, PskDemodConfig,
    PskModBlock, PskModConfig, QuadDemodConfig, RealPartBlock, ResampleBlock, ResampleConfig,
    ResampleRealBlock, RotateConfig, ScaleBlock, ScaleConfig, SliceBlock, SliceConfig,
    SliceConstellation, SpectrumBlock, SpectrumConfig, SquelchBlock, SquelchConfig, SsbDemodBlock,
    SsbDemodConfig, SsbModBlock, SsbModConfig, StereoMatrixBlock, StereoPllBlock, StereoPllConfig,
    SymbolSyncBlock, SymbolSyncConfig, Ted, ToComplexBlock, all_descriptions,
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
fn there_are_fifty_one_unique_kinds() {
    let kinds: BTreeSet<_> = all_descriptions().iter().map(|d| d.kind).collect();
    assert_eq!(kinds.len(), 51);
    assert_eq!(all_descriptions().len(), 51);
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
        "halfband_decimate",
        "real_to_iq",
        "quad_demod",
        "deemphasis",
        "rotate",
        "decimate",
        "interpolate",
        "mag",
        "delay",
        "add",
        "mul",
        "make_complex",
        "to_complex",
        "real_part",
        "imag_part",
        "nco",
        "stereo_pll",
        "stereo_matrix",
        "noise_blank",
        "auto_notch",
        "am_mod",
        "ssb_mod",
        "fm_stereo_mod",
        "frame_encode",
        "frame_decode",
        "symbol_sync",
        "slice",
        "fsk_mod",
        "fsk_demod",
        "afc",
        "costas",
        "psk_mod",
        "psk_demod",
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
                    // Choice options are values, not identifiers: a leading digit is allowed
                    // (the constellation `8psk` of M40.1, D643).
                    assert!(options.iter().all(|o| is_snake_case(o)
                        || (o.starts_with(|c: char| c.is_ascii_digit())
                            && is_snake_case(&format!("x{o}")))));
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
        "preemph_limit",
        |n| PreemphLimitConfig::default().get(n),
        &[],
    );
    defaults_match(
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
    defaults_match("quad_demod", |n| QuadDemodConfig::default().get(n), &[]);
    defaults_match("deemphasis", |n| DeemphasisConfig::default().get(n), &[]);
    defaults_match("rotate", |n| RotateConfig::default().get(n), &[]);
    defaults_match("decimate", |n| DecimateConfig::default().get(n), &[]);
    defaults_match("interpolate", |n| InterpolateConfig::default().get(n), &[]);
    defaults_match("delay", |n| DelayConfig::default().get(n), &[]);
    defaults_match("nco", |n| NcoConfig::default().get(n), &[]);
    defaults_match("stereo_pll", |n| StereoPllConfig::default().get(n), &[]);
    defaults_match("noise_blank", |n| NoiseBlankConfig::default().get(n), &[]);
    defaults_match("auto_notch", |n| AutoNotchConfig::default().get(n), &[]);
    defaults_match("am_mod", |n| AmModConfig::default().get(n), &[]);
    defaults_match("ssb_mod", |n| SsbModConfig::default().get(n), &[]);
    defaults_match(
        "fm_stereo_mod",
        |n| FmStereoModConfig::default().get(n),
        &[],
    );
    defaults_match("frame_encode", |n| FrameEncodeConfig::default().get(n), &[]);
    defaults_match("frame_decode", |n| FrameDecodeConfig::default().get(n), &[]);
    defaults_match("symbol_sync", |n| SymbolSyncConfig::default().get(n), &[]);
    defaults_match("slice", |n| SliceConfig::default().get(n), &[]);
    defaults_match("fsk_mod", |n| FskModConfig::default().get(n), &[]);
    defaults_match("fsk_demod", |n| FskDemodConfig::default().get(n), &[]);
    defaults_match("afc", |n| AfcConfig::default().get(n), &[]);
    defaults_match("costas", |n| CostasConfig::default().get(n), &[]);
    defaults_match("psk_mod", |n| PskModConfig::default().get(n), &[]);
    defaults_match("psk_demod", |n| PskDemodConfig::default().get(n), &[]);
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
    round_trip!("nco", NcoConfig);
    round_trip!("stereo_pll", StereoPllConfig);
    round_trip!("noise_blank", NoiseBlankConfig);
    round_trip!("auto_notch", AutoNotchConfig);
    round_trip!("am_mod", AmModConfig);
    round_trip!("ssb_mod", SsbModConfig);
    round_trip!("fm_stereo_mod", FmStereoModConfig);
    round_trip!("frame_encode", FrameEncodeConfig);
    round_trip!("frame_decode", FrameDecodeConfig);
    round_trip!("symbol_sync", SymbolSyncConfig);
    round_trip!("slice", SliceConfig);
    round_trip!("fsk_mod", FskModConfig);
    round_trip!("fsk_demod", FskDemodConfig);
    round_trip!("afc", AfcConfig);
    round_trip!("costas", CostasConfig);
    round_trip!("psk_mod", PskModConfig);
    round_trip!("psk_demod", PskDemodConfig);
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
        "preemph_limit",
        &mut PreemphLimitBlock::<C32, _>::new(&PreemphLimitConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "fm_stereo_mod",
        &mut FmStereoModBlock::new(&FmStereoModConfig::default(), rate(192_000), &mut s).unwrap(),
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
    let nco = NcoConfig {
        frequency_hz: 1_000.0,
        ..NcoConfig::default()
    };
    check_block_params(
        "nco",
        &mut NcoBlock::<C32, _>::new(&nco, r, &mut s).unwrap(),
    );
    check_block_params(
        "nco",
        &mut NcoRealBlock::<CQ15, _>::new(&nco, r, &mut s).unwrap(),
    );
    check_block_params(
        "stereo_pll",
        &mut StereoPllBlock::<C32, _>::new(&StereoPllConfig::default(), rate(192_000), &mut s)
            .unwrap(),
    );
    check_block_params("add", &mut AddBlock::<C32, _>::new(r, &mut s).unwrap());
    check_block_params("add", &mut AddRealBlock::<f32, _>::new(r, &mut s).unwrap());
    check_block_params("mul", &mut MulBlock::<CQ15, _>::new(r, &mut s).unwrap());
    check_block_params(
        "mul",
        &mut MulRealBlock::<sdrk::Q15, _>::new(r, &mut s).unwrap(),
    );
    check_block_params(
        "make_complex",
        &mut MakeComplexBlock::<C32, _>::new(r, &mut s).unwrap(),
    );
    check_block_params(
        "to_complex",
        &mut ToComplexBlock::<CQ15, _>::new(r, &mut s).unwrap(),
    );
    check_block_params(
        "real_part",
        &mut RealPartBlock::<C32, _>::new(r, &mut s).unwrap(),
    );
    check_block_params(
        "imag_part",
        &mut ImagPartBlock::<CQ15, _>::new(r, &mut s).unwrap(),
    );
    check_block_params(
        "stereo_matrix",
        &mut StereoMatrixBlock::<C32, _>::new(r, &mut s).unwrap(),
    );
    check_block_params(
        "noise_blank",
        &mut NoiseBlankBlock::<C32, _>::new(&NoiseBlankConfig::default(), rate(192_000), &mut s)
            .unwrap(),
    );
    check_block_params(
        "noise_blank",
        &mut NoiseBlankBlock::<CQ15, _>::new(&NoiseBlankConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "auto_notch",
        &mut AutoNotchRealBlock::new(&AutoNotchConfig::default(), rate(8_000), &mut s).unwrap(),
    );
    check_block_params(
        "dc_block",
        &mut DcBlockBlock::<CQ15, _>::new(&DcBlockConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "dc_block",
        &mut DcBlockRealBlock::<f32, _>::new(&DcBlockConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "dc_block",
        &mut DcBlockRealBlock::<sdrk::Q15, _>::new(&DcBlockConfig::default(), r, &mut s).unwrap(),
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
    check_block_params(
        "am_mod",
        &mut AmModBlock::<C32, _>::new(&AmModConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "am_mod",
        &mut AmModBlock::<CQ15, _>::new(&AmModConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "ssb_mod",
        &mut SsbModBlock::<C32, _>::new(&SsbModConfig::default(), rate(48_000), &mut s).unwrap(),
    );
    check_block_params(
        "ssb_mod",
        &mut SsbModBlock::<CQ15, _>::new(&SsbModConfig::default(), rate(48_000), &mut s).unwrap(),
    );
    check_block_params(
        "frame_encode",
        &mut FrameEncodeBlock::new(&FrameEncodeConfig::default(), r, &mut s).unwrap(),
    );
    check_block_params(
        "frame_decode",
        &mut FrameDecodeBlock::new(&FrameDecodeConfig::default(), r, &mut s).unwrap(),
    );
    let sync = SymbolSyncConfig {
        symbol_rate_hz: 2375.0,
        ..SymbolSyncConfig::default()
    };
    check_block_params(
        "symbol_sync",
        &mut SymbolSyncBlock::<C32, _>::new(&sync, rate(19_000), &mut s).unwrap(),
    );
    check_block_params(
        "symbol_sync",
        &mut SymbolSyncBlock::<f32, _>::new(&sync, rate(19_000), &mut s).unwrap(),
    );
    check_block_params(
        "slice",
        &mut SliceBlock::<C32, _>::new(&SliceConfig::default(), r, &mut s).unwrap(),
    );
    let real = SliceConfig {
        constellation: SliceConstellation::Real,
        ..SliceConfig::default()
    };
    check_block_params(
        "slice",
        &mut SliceBlock::<f32, _>::new(&real, r, &mut s).unwrap(),
    );
    let fsk = FskModConfig {
        samples_per_symbol: 4,
        deviation_hz: 25_000.0,
        bt: 0.5,
        ..FskModConfig::default()
    };
    check_block_params(
        "fsk_mod",
        &mut FskModBlock::new(&fsk, rate(50_000), &mut s).unwrap(),
    );
    let fsk = FskDemodConfig {
        levels: 4,
        symbol_rate_hz: 50_000.0,
        deviation_hz: 25_000.0,
        ..FskDemodConfig::default()
    };
    check_block_params(
        "fsk_demod",
        &mut FskDemodBlock::new(&fsk, rate(200_000), &mut s).unwrap(),
    );
    let afc = AfcConfig {
        max_offset_hz: 20_000.0,
        ..AfcConfig::default()
    };
    check_block_params(
        "afc",
        &mut AfcBlock::new(&afc, rate(200_000), &mut s).unwrap(),
    );
    check_block_params(
        "costas",
        &mut CostasBlock::new(&CostasConfig::default(), r, &mut s).unwrap(),
    );
    let pm = PskModConfig {
        order: 4,
        ..PskModConfig::default()
    };
    check_block_params("psk_mod", &mut PskModBlock::new(&pm, r, &mut s).unwrap());
    let pd = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 25_000.0,
        ..PskDemodConfig::default()
    };
    check_block_params(
        "psk_demod",
        &mut PskDemodBlock::new(&pd, rate(200_000), &mut s).unwrap(),
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
        let value = Block::param(&block, p.name).unwrap();
        p.check(&value).unwrap();
        match p.mutability {
            Mutability::Runtime => Block::set_param(&mut block, p.name, value).unwrap(),
            Mutability::Build => assert_eq!(
                Block::set_param(&mut block, p.name, value),
                Err(Error::ReadOnlyParameter)
            ),
            _ => panic!(),
        }
    }
    assert_eq!(Block::description(&block).kind, "spectrum");
}

/// The kind `dc_block` takes complex and real samples (M32): the complex block and the real
/// block share one description, whose ports list all four types in the order of `biquad`.
#[test]
fn dc_block_describes_the_complex_and_the_real_variant() {
    let d = find("dc_block");
    let four = [PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15];
    assert_eq!(d.inputs.len(), 1);
    assert_eq!(d.outputs.len(), 1);
    assert_eq!(d.inputs[0].types, four);
    assert_eq!(d.outputs[0].types, four);
    assert_eq!(d.inputs[0].types, find("biquad").inputs[0].types);
    let r = rate(48_000);
    let cfg = DcBlockConfig::default();
    let complex = DcBlockBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap();
    let real = DcBlockRealBlock::<f32, _>::new(&cfg, r, &mut VecStorage).unwrap();
    assert!(std::ptr::eq(complex.description(), d));
    assert!(std::ptr::eq(real.description(), d));
    assert!(d.details.contains("dc_block_real"));
}

/// The kinds of M33: `noise_blank` takes the complex types, `auto_notch` only `f32` (a `Q15`
/// variant is backlog); the runtime parameters are exactly `threshold_db` and `mu`.
#[test]
fn noise_blank_and_auto_notch_describe_their_ports_and_parameters() {
    let nb = find("noise_blank");
    assert_eq!(nb.inputs[0].types, [PortType::C32, PortType::CQ15]);
    assert_eq!(nb.outputs[0].types, [PortType::C32, PortType::CQ15]);
    let an = find("auto_notch");
    assert_eq!(an.inputs[0].types, [PortType::F32]);
    assert_eq!(an.outputs[0].types, [PortType::F32]);
    let runtime = |d: &BlockDescription| -> Vec<&str> {
        d.params
            .iter()
            .filter(|p| p.mutability == Mutability::Runtime)
            .map(|p| p.name)
            .collect()
    };
    assert_eq!(runtime(nb), ["threshold_db"]);
    assert_eq!(runtime(an), ["mu"]);
    let names = |d: &BlockDescription| -> Vec<&str> { d.params.iter().map(|p| p.name).collect() };
    assert_eq!(
        names(nb),
        ["threshold_db", "tau_s", "hold_s", "lookahead_s"]
    );
    assert_eq!(names(an), ["taps", "delay", "mu"]);
    assert!(nb.required.is_empty() && an.required.is_empty());
}

/// The kinds of M35: real audio in, complex baseband out; the runtime parameters are exactly
/// `depth` (`am_mod`) and `sideband` (`ssb_mod`), whose options are those of `ssb_demod`.
#[test]
fn am_mod_and_ssb_mod_describe_their_ports_and_parameters() {
    let am = find("am_mod");
    let ssb = find("ssb_mod");
    for d in [am, ssb] {
        assert_eq!(d.inputs.len(), 1);
        assert_eq!(d.outputs.len(), 1);
        assert_eq!(d.inputs[0].types, [PortType::F32, PortType::Q15]);
        assert_eq!(d.outputs[0].types, [PortType::C32, PortType::CQ15]);
        assert!(d.required.is_empty());
    }
    let runtime = |d: &BlockDescription| -> Vec<&str> {
        d.params
            .iter()
            .filter(|p| p.mutability == Mutability::Runtime)
            .map(|p| p.name)
            .collect()
    };
    assert_eq!(runtime(am), ["depth"]);
    assert_eq!(runtime(ssb), ["sideband"]);
    let names = |d: &BlockDescription| -> Vec<&str> { d.params.iter().map(|p| p.name).collect() };
    assert_eq!(names(am), ["carrier", "depth"]);
    assert_eq!(
        names(ssb),
        [
            "sideband",
            "low_hz",
            "high_hz",
            "transition_hz",
            "attenuation_db",
            "max_taps"
        ]
    );
    let demod = find("ssb_demod");
    assert_eq!(ssb.params[0].kind, demod.params[0].kind);
}

/// The kinds of M36: `preemph_limit` takes `c32` (stereo) besides the real types and never
/// `cq15`; `fm_stereo_mod` turns `c32` into `f32`, its only runtime parameter is `pilot`; both
/// blocks of `preemph_limit` share the description.
/// The kinds of M38: `symbol_sync` keeps the type (`c32` or `f32`) and requires the symbol
/// rate, all its parameters are build parameters; `slice` takes `c32` or `f32` and outputs
/// `f32` soft bits, its only runtime parameter is `scale`; the details state the soft-bit
/// convention and the chain position.
#[test]
fn the_symbol_kinds_describe_their_ports_and_parameters() {
    let sync = find("symbol_sync");
    assert_eq!(sync.inputs[0].types, [PortType::C32, PortType::F32]);
    assert_eq!(sync.outputs[0].types, [PortType::C32, PortType::F32]);
    assert_eq!(sync.required, ["symbol_rate_hz"]);
    let names: Vec<_> = sync.params.iter().map(|p| (p.name, p.mutability)).collect();
    assert_eq!(
        names,
        [
            ("symbol_rate_hz", Mutability::Build),
            ("ted", Mutability::Build),
            ("loop_bw", Mutability::Build),
            ("damping", Mutability::Build),
            ("max_deviation", Mutability::Build),
            ("ted_gain", Mutability::Build),
        ]
    );
    assert_eq!(
        sync.param("ted").unwrap().kind,
        ParamKind::Choice {
            options: &["gardner", "mm"],
            default: 0
        }
    );
    for word in ["matched filter", "agc", "variable", "ted_gain", "1/1000 Hz"] {
        assert!(sync.details.contains(word), "symbol_sync details: {word}");
    }
    let slice = find("slice");
    assert_eq!(slice.inputs[0].types, [PortType::C32, PortType::F32]);
    assert_eq!(slice.outputs[0].types, [PortType::F32]);
    assert!(slice.required.is_empty());
    let names: Vec<_> = slice
        .params
        .iter()
        .map(|p| (p.name, p.mutability))
        .collect();
    assert_eq!(
        names,
        [
            ("constellation", Mutability::Build),
            ("scale", Mutability::Runtime),
        ]
    );
    assert_eq!(
        slice.param("constellation").unwrap().kind,
        ParamKind::Choice {
            options: &["real", "bpsk", "qpsk", "fsk4", "8psk"],
            default: 1
        }
    );
    for word in [
        "above 0 means bit 1",
        "frame_decode",
        "2 / sigma^2",
        "symbol_sync",
    ] {
        assert!(slice.details.contains(word), "slice details: {word}");
    }
    let cfg = SymbolSyncConfig {
        symbol_rate_hz: 1187.5,
        ted: Ted::MuellerMuller,
        ..SymbolSyncConfig::default()
    };
    let b = SymbolSyncBlock::<f32, _>::new(&cfg, rate(19_000), &mut VecStorage).unwrap();
    assert!(std::ptr::eq(b.description(), sync));
    assert_eq!(b.param("ted"), Ok(ParamValue::Choice(1)));
    let b =
        SliceBlock::<C32, _>::new(&SliceConfig::default(), rate(1000), &mut VecStorage).unwrap();
    assert!(std::ptr::eq(b.description(), slice));
}

/// The kinds of M39: `fsk_mod` turns `f32` soft bits into `c32` and changes only
/// `deviation_hz` at run time; `fsk_demod` turns `c32` into `f32` soft bits, all its parameters
/// are build parameters; `afc` keeps `c32`, its only runtime parameter is `hold`; the details
/// name the level map, the chain and the SSB limitation.
#[test]
fn the_fsk_kinds_describe_their_ports_and_parameters() {
    let names = |d: &BlockDescription| -> Vec<(&str, Mutability)> {
        d.params.iter().map(|p| (p.name, p.mutability)).collect()
    };
    let m = find("fsk_mod");
    assert_eq!(m.inputs[0].types, [PortType::F32]);
    assert_eq!(m.outputs[0].types, [PortType::C32]);
    assert_eq!(m.required, ["samples_per_symbol", "deviation_hz"]);
    assert_eq!(
        names(m),
        [
            ("levels", Mutability::Build),
            ("samples_per_symbol", Mutability::Build),
            ("deviation_hz", Mutability::Runtime),
            ("bt", Mutability::Build),
            ("span", Mutability::Build),
        ]
    );
    for word in [
        "01 -> +1",
        "11 -> -1",
        "fir_interpolate_real",
        "fm_mod",
        "GMSK",
    ] {
        assert!(m.details.contains(word), "fsk_mod details: {word}");
    }
    let d = find("fsk_demod");
    assert_eq!(d.inputs[0].types, [PortType::C32]);
    assert_eq!(d.outputs[0].types, [PortType::F32]);
    assert_eq!(d.required, ["symbol_rate_hz", "deviation_hz"]);
    assert_eq!(
        names(d),
        [
            ("levels", Mutability::Build),
            ("symbol_rate_hz", Mutability::Build),
            ("deviation_hz", Mutability::Build),
            ("loop_bw", Mutability::Build),
        ]
    );
    for word in [
        "quad_demod",
        "symbol_sync",
        "fsk4",
        "frame_decode",
        "4 times",
    ] {
        assert!(d.details.contains(word), "fsk_demod details: {word}");
    }
    let a = find("afc");
    assert_eq!(a.inputs[0].types, [PortType::C32]);
    assert_eq!(a.outputs[0].types, [PortType::C32]);
    assert_eq!(a.required, ["max_offset_hz"]);
    assert_eq!(
        names(a),
        [
            ("time_constant_s", Mutability::Build),
            ("max_offset_hz", Mutability::Build),
            ("hold", Mutability::Runtime),
            ("carrier_hz", Mutability::Build),
        ]
    );
    for word in ["SSB", "offset_hz", "offset_ppm", "256", "hold"] {
        assert!(a.details.contains(word), "afc details: {word}");
    }
}

/// The kinds of M40: `costas` (`c32` to `c32`, four build parameters), `psk_mod` (`f32` soft
/// bits to `c32`, `samples_per_symbol` required), `psk_demod` (`c32` to `f32` soft bits,
/// `symbol_rate_hz` required) and the parameter `polarity` of `frame_decode` (last, build,
/// choice `normal`/`both`).
#[test]
fn the_psk_kinds_describe_their_ports_and_parameters() {
    let costas = find("costas");
    assert_eq!(costas.inputs[0].types, [PortType::C32]);
    assert_eq!(costas.outputs[0].types, [PortType::C32]);
    assert!(costas.required.is_empty());
    let names: Vec<_> = costas
        .params
        .iter()
        .map(|p| (p.name, p.mutability))
        .collect();
    assert_eq!(
        names,
        [
            ("order", Mutability::Build),
            ("loop_bw", Mutability::Build),
            ("damping", Mutability::Build),
            ("max_offset", Mutability::Build),
        ]
    );
    let m = find("psk_mod");
    assert_eq!(m.inputs[0].types, [PortType::F32]);
    assert_eq!(m.outputs[0].types, [PortType::C32]);
    assert_eq!(m.required, ["samples_per_symbol"]);
    let names: Vec<_> = m.params.iter().map(|p| p.name).collect();
    assert_eq!(
        names,
        [
            "order",
            "samples_per_symbol",
            "rolloff",
            "span",
            "differential",
            "variant"
        ]
    );
    assert_eq!(
        m.param("samples_per_symbol").unwrap().kind,
        ParamKind::I64 {
            min: 2,
            max: 64,
            default: None
        }
    );
    let d = find("psk_demod");
    assert_eq!(d.inputs[0].types, [PortType::C32]);
    assert_eq!(d.outputs[0].types, [PortType::F32]);
    assert_eq!(d.required, ["symbol_rate_hz"]);
    let names: Vec<_> = d.params.iter().map(|p| p.name).collect();
    assert_eq!(
        names,
        [
            "order",
            "symbol_rate_hz",
            "rolloff",
            "span",
            "differential",
            "sync_bw",
            "carrier_bw",
            "max_offset",
            "variant"
        ]
    );
    for p in m.params.iter().chain(d.params).chain(costas.params) {
        assert_eq!(p.mutability, Mutability::Build, "{}", p.name);
    }
    // M40.1: the orders 2, 4 and 8, and the variant choice (last, build, default plain)
    for desc in [m, d, costas] {
        assert_eq!(
            desc.param("order").unwrap().kind,
            ParamKind::I64 {
                min: 2,
                max: 8,
                default: Some(2)
            },
            "{}",
            desc.kind
        );
    }
    for desc in [m, d] {
        assert_eq!(
            desc.params.last().unwrap().kind,
            ParamKind::Choice {
                options: &["plain", "pi4", "offset"],
                default: 0
            },
            "{}",
            desc.kind
        );
    }
    for word in [
        "agc",
        "resample",
        "polarity both",
        "differential",
        "symbol_sync",
    ] {
        assert!(d.details.contains(word), "psk_demod details: {word}");
    }
    let fd = find("frame_decode");
    let last = fd.params.last().unwrap();
    assert_eq!(last.name, "polarity");
    assert_eq!(last.mutability, Mutability::Build);
    assert_eq!(
        last.kind,
        ParamKind::Choice {
            options: &["normal", "both"],
            default: 0
        }
    );
    let mut cfg = FrameDecodeConfig::default();
    cfg.set("polarity", ParamValue::Choice(1)).unwrap();
    assert_eq!(cfg.polarity, FramePolarity::Both);
    assert_eq!(cfg.get("polarity"), Ok(ParamValue::Choice(1)));
    // the orders 3, 5, 6 and 7 lie inside the range of the description but are no orders
    for (kind, r) in [
        (
            "costas",
            CostasConfig::default().set("order", ParamValue::I64(5)),
        ),
        (
            "psk_mod",
            PskModConfig::default().set("order", ParamValue::I64(6)),
        ),
        (
            "psk_demod",
            PskDemodConfig::default().set("order", ParamValue::I64(7)),
        ),
        (
            "costas",
            CostasConfig::default().set("order", ParamValue::I64(3)),
        ),
        (
            "psk_mod",
            PskModConfig::default().set("order", ParamValue::I64(3)),
        ),
        (
            "psk_demod",
            PskDemodConfig::default().set("order", ParamValue::I64(3)),
        ),
    ] {
        assert!(
            matches!(r, Err(Error::InvalidParameter { name: "order", .. })),
            "{kind}: {r:?}"
        );
    }
    let b = CostasBlock::new(&CostasConfig::default(), rate(1000), &mut VecStorage).unwrap();
    assert!(std::ptr::eq(b.description(), costas));
}

#[test]
fn the_stereo_transmit_kinds_describe_their_ports_and_parameters() {
    let pl = find("preemph_limit");
    let three = [PortType::F32, PortType::Q15, PortType::C32];
    assert_eq!(pl.inputs[0].types, three);
    assert_eq!(pl.outputs[0].types, three);
    assert!(pl.details.contains("fm_stereo_mod"));
    let fsm = find("fm_stereo_mod");
    assert_eq!(fsm.inputs[0].types, [PortType::C32]);
    assert_eq!(fsm.outputs[0].types, [PortType::F32]);
    let names: Vec<_> = fsm.params.iter().map(|p| (p.name, p.mutability)).collect();
    assert_eq!(
        names,
        [
            ("pilot_hz", Mutability::Build),
            ("main", Mutability::Build),
            ("pilot", Mutability::Runtime),
        ]
    );
    assert!(fsm.required.is_empty());
    for word in ["120 kHz", "fm_mod", "75000", "preemph_limit", "resample"] {
        assert!(fsm.details.contains(word), "fm_stereo_mod details: {word}");
    }
    let r = rate(192_000);
    let cfg = PreemphLimitConfig::default();
    let stereo = PreemphLimitBlock::<C32, _>::new(&cfg, r, &mut VecStorage).unwrap();
    let mono = PreemphLimitBlock::<f32, _>::new(&cfg, r, &mut VecStorage).unwrap();
    assert!(std::ptr::eq(stereo.description(), pl));
    assert!(std::ptr::eq(mono.description(), pl));
    let enc = FmStereoModBlock::new(&FmStereoModConfig::default(), r, &mut VecStorage).unwrap();
    assert!(std::ptr::eq(enc.description(), fsm));
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
                PortType::Packets => "packets".to_string(),
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
