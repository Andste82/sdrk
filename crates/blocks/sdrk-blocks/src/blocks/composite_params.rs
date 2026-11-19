//! Test 3 of M44.1.1 (the naming rule, §1.6): every composite block that lists a `ParamMap`
//! answers each public parameter through its building blocks.
//!
//! For `qam_mod`, `psk_mod`, `qam_demod`, `qam_frame_encode` and `qam_frame_decode`, each built
//! with two or three configurations that differ in every parameter: every public parameter of
//! the description appears once and maps to one or more paths; no path is reached by two
//! public parameters (a building block parameter that the composite exposes is reached by
//! exactly one); every path reaches a parameter of a building block of the same type, unit and
//! mutability whose range holds the range of the composite (a choice: the same names at the
//! same indices; `equalizer_taps` adds the value 0, which leaves the optional equalizer out;
//! the `iq_offset` of `psk_mod` is there for the variant `offset` only; the `bit_scramble` and
//! `bit_descramble` of the QAM frame composites, left out for `scrambler` `none`, answer from the
//! configuration);
//! `param` answers the value of the first path, all paths hold the same value, and the value is
//! the one the composite was built with; a `set` reaches all paths or none (all parameters are
//! build parameters: `ReadOnlyParameter`, nothing changes); the old name `order` is unknown.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::field_reassign_with_default
)]

extern crate std;

use std::format;
use std::string::String;
use std::vec;
use std::vec::Vec;

use sdrk::Rate;

use super::bit_scramble::{BitPolynomial, SCRAMBLER_PARAM, Scrambler};
use super::frame_decode::FrameRotations;
use super::psk_mod::{PskModBlock, PskModConfig, PskVariant};
use super::qam_demod::{QamDemodBlock, QamDemodConfig};
use super::qam_frame_decode::{FRAME_TIMEOUT_PARAM, QamFrameDecodeBlock, QamFrameDecodeConfig};
use super::qam_frame_encode::{QamFrameEncodeBlock, QamFrameEncodeConfig};
use super::qam_mod::{QAM_MOD_PARAMS, QamModBlock, QamModConfig};
use super::symbol_map::SymbolMapConstellation;
use super::{
    bit_descramble, bit_scramble, costas, equalize, frame_decode, frame_encode, iq_offset,
    pulse_match, pulse_shape, qam_demod, qam_mod, symbol_demap, symbol_map, symbol_sync,
};
use crate::block::{Block, ParamValue};
use crate::describe::{ParamDesc, ParamKind};
use crate::error::Error;
use crate::param_map::{ParamMap, Stages};
use crate::storage::ArenaStorage;

/// The description of the building-block parameter a path reaches.
type Resolve = fn(&str) -> Option<&'static ParamDesc>;

fn split(path: &str) -> (&str, &str) {
    path.split_once('.').unwrap()
}

fn mod_stage(path: &str) -> Option<&'static ParamDesc> {
    match split(path) {
        ("0", n) => symbol_map::DESCRIPTION.param(n),
        ("1", n) => pulse_shape::DESCRIPTION.param(n),
        ("2", n) => iq_offset::DESCRIPTION.param(n),
        _ => None,
    }
}

fn demod_stage(path: &str) -> Option<&'static ParamDesc> {
    match split(path) {
        ("pulse", n) => pulse_match::DESCRIPTION.param(n),
        ("sync", n) => symbol_sync::DESCRIPTION.param(n),
        ("eq", n) => equalize::DESCRIPTION.param(n),
        ("costas", n) => costas::DESCRIPTION.param(n),
        ("demap", n) => symbol_demap::DESCRIPTION.param(n),
        _ => None,
    }
}

fn encode_stage(path: &str) -> Option<&'static ParamDesc> {
    match split(path) {
        ("0", n) => frame_encode::DESCRIPTION.param(n),
        ("1", n) => qam_mod::DESCRIPTION.param(n),
        ("2", n) => bit_scramble::DESCRIPTION.param(n),
        ("3", "scrambler") => Some(&SCRAMBLER_PARAM),
        _ => None,
    }
}

fn decode_stage(path: &str) -> Option<&'static ParamDesc> {
    match split(path) {
        ("0", n) => qam_demod::DESCRIPTION.param(n),
        ("1", n) => frame_decode::DESCRIPTION.param(n),
        ("2", "frame_timeout") => Some(&FRAME_TIMEOUT_PARAM),
        ("3", n) => bit_descramble::DESCRIPTION.param(n),
        ("4", "scrambler") => Some(&SCRAMBLER_PARAM),
        _ => None,
    }
}

/// The range of the composite's parameter `c` lies in that of the building block's `s`;
/// `off` allows the value 0 below the range (an optional stage left out).
fn range_holds(c: &ParamDesc, s: &ParamDesc, off: bool) -> bool {
    match (c.kind, s.kind) {
        (ParamKind::F64 { min: a, max: b, .. }, ParamKind::F64 { min: x, max: y, .. }) => {
            x <= a && b <= y
        }
        (ParamKind::I64 { min: a, max: b, .. }, ParamKind::I64 { min: x, max: y, .. }) => {
            (x <= a || (off && a == 0)) && b <= y
        }
        (ParamKind::Bool { .. }, ParamKind::Bool { .. }) => true,
        (ParamKind::Choice { options: a, .. }, ParamKind::Choice { options: x, .. }) => {
            a.len() <= x.len() && a.iter().zip(x.iter()).all(|(p, q)| p == q)
        }
        _ => false,
    }
}

/// The checks of the module documentation on one composite; `built` are the values it was
/// built with (every public parameter).
/// `map` is the composite's map and `stage` reads the building-block parameter at a path of it.
fn check<T: Block>(
    label: &str,
    block: &mut T,
    map: ParamMap,
    stage: fn(&T, &str) -> crate::error::Result<ParamValue>,
    resolve: Resolve,
    built: &[(&str, ParamValue)],
) {
    let desc = block.description();
    let names: Vec<&str> = desc.params.iter().map(|p| p.name).collect();
    let mut flat: Vec<&str> = Vec::new();
    for (name, _) in map.entries() {
        if !flat.contains(name) {
            flat.push(name);
        }
    }
    assert_eq!(
        flat, names,
        "{label}: the flat names are the parameters, once each"
    );
    let mut paths: Vec<&str> = map.entries().iter().map(|(_, p)| *p).collect();
    paths.sort_unstable();
    let n = paths.len();
    paths.dedup();
    assert_eq!(
        paths.len(),
        n,
        "{label}: a path reached by two public parameters"
    );
    assert_eq!(
        built.len(),
        names.len(),
        "{label}: a value for every parameter"
    );
    for p in desc.params {
        let value = block.param(p.name).unwrap();
        let want = built.iter().find(|(n, _)| *n == p.name).unwrap().1;
        assert_eq!(
            value, want,
            "{label}.{}: the value it was built with",
            p.name
        );
        let mut reached = 0;
        for path in map.paths(p.name) {
            let s = resolve(path).unwrap_or_else(|| panic!("{label}: {path} reaches nothing"));
            assert_eq!(
                core::mem::discriminant(&p.kind),
                core::mem::discriminant(&s.kind),
                "{label}.{}: type of {path}",
                p.name
            );
            assert_eq!(p.unit, s.unit, "{label}.{}: unit of {path}", p.name);
            assert_eq!(
                p.mutability, s.mutability,
                "{label}.{}: mutability of {path}",
                p.name
            );
            if (label, p.name) == ("psk_mod", "variant") {
                // `offset` is the `iq_offset` stage after the pulse filter, not a value of the
                // `variant` of `symbol_map` (structure, D1823): the other names at their
                // indices, and `offset` the one more.
                let (ParamKind::Choice { options: a, .. }, ParamKind::Choice { options: x, .. }) =
                    (p.kind, s.kind)
                else {
                    panic!("{label}.variant is a choice");
                };
                assert_eq!((&a[..x.len()], &a[x.len()..]), (x, &["offset"][..]));
            } else {
                assert!(
                    range_holds(p, s, path.starts_with("eq.")),
                    "{label}.{}: range of {path}",
                    p.name
                );
            }
            match stage(block, path) {
                Ok(v) => {
                    assert_eq!(v, value, "{label}.{}: {path} holds the same value", p.name);
                    reached += 1;
                }
                // Only an optional stage that is left out has nothing to read: the equalizer of
                // `qam_demod` and the `iq_offset` of `psk_mod`.
                Err(e) => assert!(
                    (path.starts_with("eq.") || (label == "psk_mod" && path.starts_with("2.")))
                        && e == Error::UnknownParameter,
                    "{label}.{}: {path}: {e:?}",
                    p.name
                ),
            }
        }
        assert!(reached > 0, "{label}.{}: no building block answers", p.name);
        assert_eq!(
            map.path(p.name).map(|f| stage(block, f).unwrap()),
            Some(value),
            "{label}.{}: param reads the first path",
            p.name
        );
        // A set reaches all or none: build parameters, so none, and nothing changes.
        assert_eq!(
            block.check_param(p.name, &value),
            Err(Error::ReadOnlyParameter),
            "{label}.{}",
            p.name
        );
        assert_eq!(
            block.set_param(p.name, value),
            Err(Error::ReadOnlyParameter),
            "{label}"
        );
        for path in map.paths(p.name) {
            if let Ok(v) = stage(block, path) {
                assert_eq!(v, value, "{label}.{}: {path} after a refused set", p.name);
            }
        }
    }
    for name in ["order", "no_such_parameter"] {
        assert_eq!(block.param(name), Err(Error::UnknownParameter), "{label}");
        assert_eq!(
            block.set_param(name, ParamValue::I64(4)),
            Err(Error::UnknownParameter),
            "{label}"
        );
    }
    // A path is not a public name, and a composite answers no path outside its map.
    for path in map.entries().iter().map(|(_, p)| *p) {
        assert_eq!(
            block.param(path),
            Err(Error::UnknownParameter),
            "{label}: {path}"
        );
    }
    assert!(stage(block, "0.no_such_parameter").is_err(), "{label}");
}

fn rate(hz: u64) -> Rate {
    Rate::hz(hz).unwrap()
}

fn choice(c: SymbolMapConstellation) -> ParamValue {
    ParamValue::Choice(c.index())
}

fn mod_values(c: &QamModConfig) -> Vec<(&'static str, ParamValue)> {
    vec![
        ("constellation", choice(c.constellation)),
        (
            "samples_per_symbol",
            ParamValue::I64(c.samples_per_symbol as i64),
        ),
        ("rolloff", ParamValue::F64(c.rolloff)),
        ("span", ParamValue::I64(c.span as i64)),
    ]
}

fn demod_values(c: &QamDemodConfig) -> Vec<(&'static str, ParamValue)> {
    vec![
        ("constellation", choice(c.constellation)),
        ("symbol_rate_hz", ParamValue::F64(c.symbol_rate_hz)),
        ("rolloff", ParamValue::F64(c.rolloff)),
        ("span", ParamValue::I64(c.span as i64)),
        ("equalizer_taps", ParamValue::I64(c.equalizer_taps as i64)),
        ("sync_bw", ParamValue::F64(c.sync_bw())),
        ("carrier_bw", ParamValue::F64(c.carrier_bw)),
    ]
}

fn arena() -> Vec<u8> {
    vec![0u8; 1 << 20]
}

#[test]
fn the_parameters_of_every_composite_follow_the_naming_rule() {
    // qam_mod
    for cfg in [
        QamModConfig::default(),
        QamModConfig {
            constellation: SymbolMapConstellation::Qam64,
            samples_per_symbol: 5,
            rolloff: 0.25,
            span: 8,
        },
    ] {
        let mut mem = arena();
        let mut st = ArenaStorage::new(&mut mem);
        let mut b = QamModBlock::new(&cfg, rate(48_000), &mut st).unwrap();
        check(
            "qam_mod",
            &mut b,
            QAM_MOD_PARAMS,
            |b, p| b.stages.param(p),
            mod_stage,
            &mod_values(&cfg),
        );
    }
    // psk_mod: plain, pi/4-DQPSK, offset
    for cfg in [
        PskModConfig::default(),
        PskModConfig {
            constellation: SymbolMapConstellation::Qpsk,
            samples_per_symbol: 6,
            rolloff: 0.5,
            span: 12,
            differential: true,
            variant: PskVariant::Pi4,
        },
        PskModConfig {
            constellation: SymbolMapConstellation::Qpsk,
            samples_per_symbol: 4,
            variant: PskVariant::Offset,
            ..PskModConfig::default()
        },
        PskModConfig {
            constellation: SymbolMapConstellation::Psk8,
            samples_per_symbol: 3,
            ..PskModConfig::default()
        },
    ] {
        let mut mem = arena();
        let mut st = ArenaStorage::new(&mut mem);
        let mut b = PskModBlock::new(&cfg, rate(9600), &mut st).unwrap();
        let mut values = vec![
            ("constellation", choice(cfg.constellation)),
            (
                "samples_per_symbol",
                ParamValue::I64(cfg.samples_per_symbol as i64),
            ),
            ("rolloff", ParamValue::F64(cfg.rolloff)),
            ("span", ParamValue::I64(cfg.span as i64)),
            ("differential", ParamValue::Bool(cfg.differential)),
        ];
        values.push(("variant", ParamValue::Choice(cfg.variant.index())));
        // `variant` `offset` is the `iq_offset` stage, not a value of `symbol_map` (D1823):
        // checked here, outside the generic resolution.
        if cfg.variant == PskVariant::Offset {
            assert_eq!(b.param("variant"), Ok(ParamValue::Choice(2)));
            assert_eq!(b.config(), cfg);
        }
        check(
            "psk_mod",
            &mut b,
            PskModBlock::<ArenaStorage<'_>>::PARAMS,
            Stages::stage_param,
            mod_stage,
            &values,
        );
    }
    // qam_demod: with and without the equalizer
    for cfg in [
        QamDemodConfig {
            symbol_rate_hz: 25_000.0,
            ..QamDemodConfig::default()
        },
        QamDemodConfig {
            constellation: SymbolMapConstellation::Qam64,
            symbol_rate_hz: 20_000.0,
            rolloff: 0.25,
            span: 8,
            equalizer_taps: 0,
            sync_bw: Some(0.01),
            carrier_bw: 0.004,
        },
    ] {
        let mut mem = arena();
        let mut st = ArenaStorage::new(&mut mem);
        let mut b = QamDemodBlock::new(&cfg, rate(200_000), &mut st).unwrap();
        check(
            "qam_demod",
            &mut b,
            QamDemodBlock::<ArenaStorage<'_>>::PARAMS,
            Stages::stage_param,
            demod_stage,
            &demod_values(&cfg),
        );
    }
    // qam_frame_encode: with the scrambler (the default) and without it
    let mut enc = QamFrameEncodeConfig::default();
    enc.qam = QamModConfig {
        constellation: SymbolMapConstellation::Qam64,
        samples_per_symbol: 4,
        rolloff: 0.3,
        span: 6,
    };
    enc.frame.preamble_bytes = 6;
    let without = QamFrameEncodeConfig {
        scrambler: Scrambler::Off,
        polynomial: BitPolynomial::X58X39,
        ..enc
    };
    for enc in [enc, without] {
        let mut mem = arena();
        let mut st = ArenaStorage::new(&mut mem);
        let mut b = QamFrameEncodeBlock::new(&enc, rate(4800), &mut st).unwrap();
        assert_eq!(
            (b.config().scrambler, b.config().scramble()),
            (enc.scrambler, enc.scramble())
        );
        assert_eq!(b.config(), enc);
        assert_eq!(
            b.scrambler().is_some(),
            enc.scrambler == Scrambler::Multiplicative
        );
        let mut values = vec![
            ("preamble_bytes", ParamValue::I64(6)),
            ("idle", ParamValue::Choice(0)),
            ("whitening", ParamValue::Choice(0)),
        ];
        values.extend(mod_values(&enc.qam));
        values.extend([
            ("scrambler", ParamValue::Choice(enc.scrambler.index())),
            ("polynomial", ParamValue::Choice(enc.polynomial.index())),
        ]);
        check(
            "qam_frame_encode",
            &mut b,
            QamFrameEncodeBlock::<ArenaStorage<'_>>::PARAMS,
            Stages::stage_param,
            encode_stage,
            &values,
        );
    }
    // qam_frame_decode
    let mut dec = QamFrameDecodeConfig {
        frame_timeout: Some(8192),
        ..QamFrameDecodeConfig::default()
    };
    dec.demod.symbol_rate_hz = 25_000.0;
    dec.demod.equalizer_taps = 15;
    dec.frame.max_sync_errors = 2;
    dec.frame.emit_bad = true;
    let without = QamFrameDecodeConfig {
        scrambler: Scrambler::Off,
        polynomial: BitPolynomial::X17X12,
        ..dec
    };
    for dec in [dec, without] {
        let mut mem = arena();
        let mut st = ArenaStorage::new(&mut mem);
        let mut b = QamFrameDecodeBlock::new(&dec, rate(200_000), &mut st).unwrap();
        assert_eq!(b.config().decoder().rotations, FrameRotations::Four);
        // The default `sync_bw` of the constellation stays unset, the polynomial is the one field
        // of the stage (review of M44.2, D1536).
        assert_eq!(b.config().demod, dec.demod);
        assert_eq!(b.config().polynomial, dec.polynomial);
        assert_eq!(
            b.descrambler().is_some(),
            dec.scrambler == Scrambler::Multiplicative
        );
        let mut values = demod_values(&dec.demod);
        values.extend([
            ("max_sync_errors", ParamValue::I64(2)),
            ("emit_bad", ParamValue::Bool(true)),
            ("whitening", ParamValue::Choice(0)),
            ("frame_timeout", ParamValue::I64(8192)),
            ("scrambler", ParamValue::Choice(dec.scrambler.index())),
            ("polynomial", ParamValue::Choice(dec.polynomial.index())),
        ]);
        check(
            "qam_frame_decode",
            &mut b,
            QamFrameDecodeBlock::<ArenaStorage<'_>>::PARAMS,
            Stages::stage_param,
            decode_stage,
            &values,
        );
    }
}

/// The resolution of the paths above names every stage of every map (a guard against a map
/// entry that the test would not see).
#[test]
fn every_path_of_every_map_is_resolved() {
    let maps: [(&str, crate::param_map::ParamMap, Resolve); 5] = [
        ("qam_mod", QAM_MOD_PARAMS, mod_stage),
        (
            "psk_mod",
            PskModBlock::<ArenaStorage<'static>>::PARAMS,
            mod_stage,
        ),
        (
            "qam_demod",
            QamDemodBlock::<ArenaStorage<'static>>::PARAMS,
            demod_stage,
        ),
        (
            "qam_frame_encode",
            QamFrameEncodeBlock::<ArenaStorage<'static>>::PARAMS,
            encode_stage,
        ),
        (
            "qam_frame_decode",
            QamFrameDecodeBlock::<ArenaStorage<'static>>::PARAMS,
            decode_stage,
        ),
    ];
    for (label, map, resolve) in maps {
        for (name, path) in map.entries() {
            let what: String = format!("{label}.{name} -> {path}");
            assert!(resolve(path).is_some(), "{what}");
        }
    }
}
