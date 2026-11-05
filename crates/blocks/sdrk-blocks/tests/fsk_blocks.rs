//! The blocks of M39: `FskModBlock` (the level maps of 2-FSK and 4-FSK, a constant level gives
//! exactly the deviation, the Gaussian pulse, the runtime `deviation_hz`, timing and exact
//! storage), `FskDemodBlock` (the loop back from `fsk_mod`, `rate_unsupported` below 4 samples
//! per symbol, the variable timing, exact storage, small outputs), the `fsk4` map of `slice`
//! at corner values, and `AfcBlock` (convergence, the readings, `hold`, the limit, `reset`,
//! NaN input); all three blocks are chunk invariant with the 20 plans of seed 537.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

mod common;

use common::{check_exact_storage, rate, run_block, same_bits, twenty_plans};
use sdrk::{C32, Delay, Ratio, RatioKind};
use sdrk_blocks::{
    AFC_PIECE, AfcBlock, AfcConfig, ArenaStorage, Block, Error, FmModBlock, FmModConfig,
    FskDemodBlock, FskDemodConfig, FskModBlock, FskModConfig, ParamValue, SliceBlock, SliceConfig,
    SliceConstellation, TED_GAIN_2FSK, TED_GAIN_4FSK,
};
use sdrk_testkit::{Pcg32, VecStorage};

/// Seed of the chunk plans of M39 (test 6; seed table of docs/milestones/README.md).
const SEED: u64 = 537;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn the_new_types_are_send_and_sync() {
    assert_send::<FskModBlock<VecStorage>>();
    assert_send::<FskDemodBlock<VecStorage>>();
    assert_send::<AfcBlock<VecStorage>>();
    assert_send::<FskModConfig>();
    assert_send::<FskDemodConfig>();
    assert_send::<AfcConfig>();
    assert_sync::<FskModBlock<VecStorage>>();
    assert_sync::<FskDemodBlock<VecStorage>>();
    assert_sync::<AfcBlock<VecStorage>>();
}

/// Random soft bits `+1` / `-1`.
fn soft_bits(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

fn fsk_mod(cfg: &FskModConfig, bit_rate: u64) -> FskModBlock<VecStorage> {
    FskModBlock::new(cfg, rate(bit_rate), &mut VecStorage).unwrap()
}

/// `fm_mod` at the output rate of `fsk_mod`, fed with `levels`.
fn fm_reference(deviation_hz: f64, rate_out: u64, levels: &[f32]) -> Vec<C32> {
    let mut fm = FmModBlock::<C32, _>::new(
        &FmModConfig { deviation_hz },
        rate(rate_out),
        &mut VecStorage,
    )
    .unwrap();
    let mut out = vec![C32::default(); levels.len()];
    fm.process(levels, &mut out).unwrap();
    out
}

/// Instantaneous frequency in Hz between consecutive samples.
fn freq_hz(x: &[C32], fs: f64) -> Vec<f64> {
    x.windows(2)
        .map(|w| {
            let a = num_complex::Complex64::new(f64::from(w[0].re), f64::from(w[0].im));
            let b = num_complex::Complex64::new(f64::from(w[1].re), f64::from(w[1].im));
            let z = b * a.conj();
            z.im.atan2(z.re) * fs / std::f64::consts::TAU
        })
        .collect()
}

#[test]
fn a_constant_level_gives_exactly_the_deviation() {
    // bt 0: the boxcar keeps the level exactly, so the output is fm_mod of +1 or -1.
    for (bit, level) in [(1.0f32, 1.0f32), (-1.0, -1.0)] {
        let cfg = FskModConfig {
            samples_per_symbol: 8,
            deviation_hz: 1_200.0,
            ..FskModConfig::default()
        };
        let mut block = fsk_mod(&cfg, 2_400);
        let mut out = vec![C32::default(); 800];
        let p = block.process(&[bit; 100], &mut out).unwrap();
        assert_eq!((p.consumed, p.produced), (100, 800));
        same_bits(&fm_reference(1_200.0, 19_200, &[level; 800]), &out, "bt 0");
    }
    // 4-FSK: the outer symbols 01 and 11 give exactly +deviation and -deviation.
    for (pair, level) in [([-1.0f32, 1.0], 1.0f32), ([1.0, 1.0], -1.0)] {
        let cfg = FskModConfig {
            levels: 4,
            samples_per_symbol: 4,
            deviation_hz: 3_000.0,
            ..FskModConfig::default()
        };
        let mut block = fsk_mod(&cfg, 4_800);
        let bits: Vec<f32> = pair.iter().copied().cycle().take(100).collect();
        let mut out = vec![C32::default(); 200];
        let p = block.process(&bits, &mut out).unwrap();
        assert_eq!((p.consumed, p.produced), (100, 200));
        same_bits(&fm_reference(3_000.0, 9_600, &[level; 200]), &out, "4-FSK");
    }
    // bt 0.5: after the pulse has settled the frequency is the deviation within the f32
    // rounding of the taps.
    let cfg = FskModConfig {
        samples_per_symbol: 8,
        deviation_hz: 600.0,
        bt: 0.5,
        ..FskModConfig::default()
    };
    let mut block = fsk_mod(&cfg, 2_400);
    let taps = block.taps().to_vec();
    assert_eq!(taps.len(), 40);
    let sum: f64 = taps.iter().map(|&t| f64::from(t)).sum();
    assert!((sum - 8.0).abs() < 1e-5, "tap sum {sum}");
    for phase in 0..8 {
        let branch: f64 = taps
            .iter()
            .skip(phase)
            .step_by(8)
            .map(|&t| f64::from(t))
            .sum();
        assert!((branch - 1.0).abs() < 1e-6, "branch {phase}: {branch}");
    }
    let mut out = vec![C32::default(); 800];
    block.process(&[1.0; 100], &mut out).unwrap();
    for f in &freq_hz(&out[100..], 19_200.0) {
        assert!((f - 600.0).abs() < 1e-2, "{f}");
    }
}

#[test]
fn four_fsk_maps_two_bits_per_symbol() {
    // (b0, b1): 01 -> +1, 00 -> +1/3, 10 -> -1/3, 11 -> -1; b0 first.
    let pairs = [
        ([-1.0f32, 1.0f32], 1.0f32),
        ([-1.0, -1.0], 1.0 / 3.0),
        ([1.0, -1.0], -1.0 / 3.0),
        ([1.0, 1.0], -1.0),
    ];
    let cfg = FskModConfig {
        levels: 4,
        samples_per_symbol: 4,
        deviation_hz: 2_400.0,
        ..FskModConfig::default()
    };
    let mut block = fsk_mod(&cfg, 9_600);
    let mut bits = Vec::new();
    let mut levels = Vec::new();
    for i in 0..64 {
        let (pair, level) = pairs[(i * 7 + i / 4) % 4];
        bits.extend_from_slice(&pair);
        levels.extend_from_slice(&[level; 4]);
    }
    // Soft bits of any size: only the sign counts; 0 and NaN are bit 0.
    let mut soft = bits.clone();
    soft[0] *= 0.25;
    soft[1] *= 7.0;
    let mut out = vec![C32::default(); 256];
    let p = block.process(&soft, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (128, 256));
    same_bits(&fm_reference(2_400.0, 19_200, &levels), &out, "4-FSK map");
    let mut block = fsk_mod(&cfg, 9_600);
    let mut out0 = vec![C32::default(); 4];
    block.process(&[0.0, f32::NAN], &mut out0).unwrap();
    same_bits(
        &fm_reference(2_400.0, 19_200, &[1.0 / 3.0; 4]),
        &out0,
        "0 and NaN",
    );
    // An odd bit stays unconsumed; an output without room for a whole symbol takes nothing.
    let mut block = fsk_mod(&cfg, 9_600);
    let p = block.process(&bits[..3], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 4));
    let p = block.process(&bits[..4], &mut out[..3]).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    assert_eq!(block.min_input(), 2);
}

#[test]
fn fsk_mod_deviation_is_a_runtime_parameter_with_a_continuous_phase() {
    let cfg = FskModConfig {
        samples_per_symbol: 8,
        deviation_hz: 1_000.0,
        ..FskModConfig::default()
    };
    let bits = soft_bits(200, 1);
    let levels: Vec<f32> = bits.iter().flat_map(|&b| [b; 8]).collect();
    let mut block = fsk_mod(&cfg, 2_000);
    let mut out = vec![C32::default(); 1600];
    block.process(&bits[..100], &mut out[..800]).unwrap();
    block
        .set_param("deviation_hz", ParamValue::F64(2_500.0))
        .unwrap();
    assert_eq!(block.param("deviation_hz"), Ok(ParamValue::F64(2_500.0)));
    block.process(&bits[100..], &mut out[800..]).unwrap();
    let mut fm = FmModBlock::<C32, _>::new(
        &FmModConfig {
            deviation_hz: 1_000.0,
        },
        rate(16_000),
        &mut VecStorage,
    )
    .unwrap();
    let mut want = vec![C32::default(); 1600];
    fm.process(&levels[..800], &mut want[..800]).unwrap();
    fm.set_param("deviation_hz", ParamValue::F64(2_500.0))
        .unwrap();
    fm.process(&levels[800..], &mut want[800..]).unwrap();
    same_bits(&want, &out, "deviation change");
    // At most half the output rate; build parameters are read-only.
    assert!(matches!(
        block.set_param("deviation_hz", ParamValue::F64(8_001.0)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(
        block.set_param("bt", ParamValue::F64(0.5)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(block.param("deviation_hz"), Ok(ParamValue::F64(2_500.0)));
}

#[test]
fn fsk_mod_timing_rates_storage_and_checks() {
    for (levels, sps, bt, taps, ratio, delay) in [
        (2usize, 8usize, 0.0, 8usize, (8u64, 1u64), (7u64, 16u64)),
        (2, 4, 0.5, 20, (4, 1), (19, 8)),
        (4, 4, 0.0, 4, (2, 1), (6, 8)),
        (4, 5, 0.3, 25, (5, 2), (48, 10)),
    ] {
        let cfg = FskModConfig {
            levels,
            samples_per_symbol: sps,
            deviation_hz: 1_000.0,
            bt,
            ..FskModConfig::default()
        };
        let r = rate(10_000);
        let block = fsk_mod(&cfg, 10_000);
        assert_eq!(block.taps().len(), taps);
        assert_eq!(cfg.num_taps().unwrap(), taps);
        let t = block.timing();
        assert_eq!(t.ratio, Ratio::new(ratio.0, ratio.1).unwrap());
        assert_eq!(t.kind, RatioKind::Constant);
        assert_eq!(
            t.delay.reduced(),
            Delay::new(delay.0, delay.1).unwrap().reduced()
        );
        assert_eq!(
            FskModBlock::<VecStorage>::rate_out(&cfg, r).unwrap(),
            r.scale(t.ratio).unwrap()
        );
        let bytes = FskModBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| FskModBlock::new(&cfg, r, a).map(|_| ()));
    }
    let ok = FskModConfig {
        samples_per_symbol: 4,
        deviation_hz: 1_000.0,
        ..FskModConfig::default()
    };
    let r = rate(1_000);
    for bad in [
        FskModConfig { levels: 3, ..ok },
        FskModConfig {
            samples_per_symbol: 1,
            ..ok
        },
        FskModConfig {
            samples_per_symbol: 65,
            ..ok
        },
        FskModConfig { bt: 0.1, ..ok },
        FskModConfig { bt: 1.5, ..ok },
        FskModConfig { bt: f64::NAN, ..ok },
        FskModConfig {
            span: 9,
            bt: 0.5,
            ..ok
        },
        FskModConfig {
            deviation_hz: 2_001.0,
            ..ok
        },
        FskModConfig {
            deviation_hz: -1.0,
            ..ok
        },
    ] {
        assert!(bad.validate(r).is_err(), "{bad:?}");
        assert!(
            FskModBlock::new(&bad, r, &mut VecStorage).is_err(),
            "{bad:?}"
        );
    }
    assert!(
        FskModConfig {
            deviation_hz: 2_000.0,
            ..ok
        }
        .validate(r)
        .is_ok()
    );
}

/// Feeds `input` with an output of `room` soft bits per call (resubmitting what is not
/// consumed) until everything is consumed and nothing more comes out.
fn run_small<B: Block<In = C32, Out = f32>>(block: &mut B, input: &[C32], room: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; room];
    let mut pos = 0;
    loop {
        let p = block.process(&input[pos..], &mut buf).unwrap();
        out.extend_from_slice(&buf[..p.produced]);
        pos += p.consumed;
        if p.consumed == 0 && p.produced == 0 {
            break;
        }
    }
    assert_eq!(pos, input.len());
    out
}

/// `fsk_mod` (bt 0, h = 1) of random bits, as the input of `fsk_demod`.
fn fsk_signal(levels: usize, sps: usize, bits: &[f32]) -> Vec<C32> {
    let symbol_rate = 1_000.0;
    let bps = if levels == 4 { 2 } else { 1 };
    let cfg = FskModConfig {
        levels,
        samples_per_symbol: sps,
        deviation_hz: symbol_rate / 2.0,
        ..FskModConfig::default()
    };
    let mut block = fsk_mod(&cfg, 1_000 * bps as u64);
    let mut out = vec![C32::default(); bits.len() / bps * sps];
    let p = block.process(bits, &mut out).unwrap();
    assert_eq!(p.produced, out.len());
    out
}

/// The offset at which `got` (signs) matches `want` best, and the number of mismatches
/// there, over the bits `from..` of `got`.
fn best_alignment(want: &[f32], got: &[f32], from: usize) -> (usize, usize) {
    (0..40)
        .map(|lag| {
            let errors = got[from..]
                .iter()
                .zip(&want[from - lag.min(from)..])
                .filter(|(g, w)| (**g > 0.0) != (**w > 0.0))
                .count();
            (lag, errors)
        })
        .min_by_key(|&(_, e)| e)
        .unwrap()
}

#[test]
fn fsk_demod_recovers_the_bits_of_fsk_mod() {
    for (levels, sps) in [(2usize, 8usize), (2, 4), (4, 8), (4, 5)] {
        let bps = if levels == 4 { 2 } else { 1 };
        let bits = soft_bits(4000 * bps, 2 + levels as u64);
        let signal = fsk_signal(levels, sps, &bits);
        let cfg = FskDemodConfig {
            levels,
            symbol_rate_hz: 1_000.0,
            deviation_hz: 500.0,
            ..FskDemodConfig::default()
        };
        let fs = rate(1_000 * sps as u64);
        let mut block = FskDemodBlock::new(&cfg, fs, &mut VecStorage).unwrap();
        let got = run_block(&mut block, &signal, None);
        assert!(
            got.len().abs_diff(bits.len()) <= 2 * bps,
            "{levels} {sps}: {}",
            got.len()
        );
        // Skip the acquisition (1000 symbols), then every bit is right.
        let (lag, errors) = best_alignment(&bits, &got, 1000 * bps);
        assert_eq!(errors, 0, "levels {levels}, sps {sps}, lag {lag}");
        // Small outputs (pending symbols) give the same soft bits.
        let rooms: [usize; 5] = if bps == 1 {
            [1, 2, 3, 7, 64]
        } else {
            [2, 3, 4, 8, 64]
        };
        for room in rooms {
            let mut b = FskDemodBlock::new(&cfg, fs, &mut VecStorage).unwrap();
            same_bits(&got, &run_small(&mut b, &signal, room), "small output");
        }
    }
}

#[test]
fn fsk_demod_timing_rates_storage_and_checks() {
    for (levels, fs, sym, taps, ratio, delay) in [
        (
            2usize,
            200_000u64,
            50_000.0,
            4usize,
            (1u64, 4u64),
            (6u64, 2u64),
        ),
        (4, 48_000, 4_800.0, 10, (1, 5), (12, 2)),
        (2, 10_100, 1_000.0, 10, (10, 101), (12, 2)),
    ] {
        let cfg = FskDemodConfig {
            levels,
            symbol_rate_hz: sym,
            deviation_hz: sym / 2.0,
            ..FskDemodConfig::default()
        };
        let r = rate(fs);
        let block = FskDemodBlock::new(&cfg, r, &mut VecStorage).unwrap();
        assert_eq!(block.num_taps(), taps);
        let t = block.timing();
        assert_eq!(t.ratio, Ratio::new(ratio.0, ratio.1).unwrap());
        assert_eq!(t.kind, RatioKind::Variable);
        assert_eq!(
            t.delay.reduced(),
            Delay::new(delay.0, delay.1).unwrap().reduced()
        );
        assert_eq!(
            FskDemodBlock::<VecStorage>::rate_out(&cfg, r).unwrap(),
            r.scale(t.ratio).unwrap()
        );
        assert_eq!(
            cfg.symbol_sync().unwrap().ted_gain,
            if levels == 2 {
                TED_GAIN_2FSK
            } else {
                TED_GAIN_4FSK
            }
        );
        let bytes = FskDemodBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| FskDemodBlock::new(&cfg, r, a).map(|_| ()));
    }
    let ok = FskDemodConfig {
        symbol_rate_hz: 1_000.0,
        deviation_hz: 500.0,
        ..FskDemodConfig::default()
    };
    // Exactly 4 samples per symbol is fine, fewer is OutOfRange of symbol_rate_hz.
    assert!(ok.validate(rate(4_000)).is_ok());
    assert!(matches!(
        ok.validate(rate(3_999)),
        Err(Error::OutOfRange {
            name: "symbol_rate_hz",
            ..
        })
    ));
    assert!(matches!(
        ok.validate(rate(1_024_001)),
        Err(Error::OutOfRange {
            name: "symbol_rate_hz",
            ..
        })
    ));
    for bad in [
        FskDemodConfig { levels: 3, ..ok },
        FskDemodConfig {
            deviation_hz: 0.0,
            ..ok
        },
        FskDemodConfig {
            deviation_hz: 4_001.0,
            ..ok
        },
        FskDemodConfig { loop_bw: 0.2, ..ok },
    ] {
        assert!(bad.validate(rate(8_000)).is_err(), "{bad:?}");
    }
    let mut block = FskDemodBlock::new(&ok, rate(8_000), &mut VecStorage).unwrap();
    assert_eq!(
        block.set_param("loop_bw", ParamValue::F64(0.02)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(block.param("loop_bw"), Ok(ParamValue::F64(0.01)));
    assert!(block.min_input() >= 10);
}

#[test]
fn slice_fsk4_follows_the_table_at_corner_values() {
    let cfg = SliceConfig {
        constellation: SliceConstellation::Fsk4,
        scale: 2.0,
    };
    let mut slice = SliceBlock::<f32, _>::new(&cfg, rate(1200), &mut VecStorage).unwrap();
    let third = 1.0f32 / 3.0;
    let two_thirds = 2.0f32 / 3.0;
    let input = [1.0f32, third, -third, -1.0, 0.0, two_thirds, 3.0];
    let mut out = [0.0f32; 14];
    let p = slice.process(&input, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (7, 14));
    let want: Vec<f32> = input
        .iter()
        .flat_map(|&v| [2.0 * -v, 2.0 * (v.abs() - two_thirds)])
        .collect();
    same_bits(&want, &out, "fsk4");
    // The signs give the bits of fsk_mod: +1 = 01, +1/3 = 00, -1/3 = 10, -1 = 11.
    let signs: Vec<bool> = out[..8].iter().map(|&b| b > 0.0).collect();
    assert_eq!(signs, [false, true, false, false, true, false, true, true]);
    assert_eq!(out[11], 0.0); // |v| = 2/3 is the threshold
    // Two soft bits per symbol, f32 input only.
    assert_eq!(slice.timing().ratio, Ratio::new(2, 1).unwrap());
    let p = slice.process(&input, &mut out[..5]).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 4));
    assert!(SliceBlock::<C32, _>::new(&cfg, rate(1200), &mut VecStorage).is_err());
    assert_eq!(SliceConstellation::Fsk4.bits_per_symbol(), 2);
    let mut cfg = SliceConfig::default();
    cfg.set("constellation", ParamValue::Choice(3)).unwrap();
    assert_eq!(cfg.constellation, SliceConstellation::Fsk4);
    assert_eq!(cfg.get("constellation"), Ok(ParamValue::Choice(3)));
}

/// A complex tone at `f(n)` Hz (continuous phase) at `fs`.
fn tone(n: usize, fs: f64, f: impl Fn(usize) -> f64) -> Vec<C32> {
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let s = C32::new(phase.cos() as f32, phase.sin() as f32);
            phase += std::f64::consts::TAU * f(i) / fs;
            s
        })
        .collect()
}

fn afc_block(cfg: &AfcConfig, fs: u64) -> AfcBlock<VecStorage> {
    AfcBlock::new(cfg, rate(fs), &mut VecStorage).unwrap()
}

#[test]
fn afc_converges_and_reports_hz_and_ppm() {
    let fs = 100_000u64;
    let cfg = AfcConfig {
        max_offset_hz: 20_000.0,
        time_constant_s: 0.05,
        carrier_hz: 868e6,
        ..AfcConfig::default()
    };
    let mut afc = afc_block(&cfg, fs);
    assert_eq!(afc.offset_hz(), 0.0);
    let input = tone(50_000, fs as f64, |_| 5_000.0);
    let mut out = vec![C32::default(); input.len()];
    afc.process(&input, &mut out).unwrap();
    assert!(
        (afc.offset_hz() - 5_000.0).abs() < 0.5,
        "{}",
        afc.offset_hz()
    );
    let ppm = 5_000.0 / 868e6 * 1e6;
    assert!(
        (afc.offset_ppm() - ppm).abs() < 1e-3,
        "{}",
        afc.offset_ppm()
    );
    // The output is at 0 Hz.
    for f in &freq_hz(&out[45_000..], fs as f64) {
        assert!(f.abs() < 2.0, "{f}");
    }
    assert_eq!(afc.reading(0), Some(("offset_hz", afc.offset_hz())));
    assert_eq!(afc.reading(1), Some(("offset_ppm", afc.offset_ppm())));
    assert_eq!(afc.reading(2), None);
    // The estimate follows k = 1 - exp(-P / (fs tc)) per piece: after one piece of a
    // constant offset it is k times the offset (up to the first discriminator output).
    let mut afc = afc_block(&cfg, fs);
    afc.process(&input[..AFC_PIECE], &mut out[..AFC_PIECE])
        .unwrap();
    let k = cfg.gain(rate(fs));
    let want = k * 5_000.0 * (AFC_PIECE - 1) as f64 / AFC_PIECE as f64;
    assert!(
        (afc.offset_hz() - want).abs() < 0.05,
        "{} vs {want}",
        afc.offset_hz()
    );
    // Without carrier_hz the ppm reading is NaN; the estimate is clamped to the limit.
    let cfg = AfcConfig {
        max_offset_hz: 2_000.0,
        time_constant_s: 0.01,
        ..AfcConfig::default()
    };
    let mut afc = afc_block(&cfg, fs);
    afc.process(&input, &mut out).unwrap();
    assert_eq!(afc.offset_hz(), 2_000.0);
    assert!(afc.offset_ppm().is_nan());
    afc.process(&tone(20_000, fs as f64, |_| -9_000.0), &mut out[..20_000])
        .unwrap();
    assert_eq!(afc.offset_hz(), -2_000.0);
    // reset clears the estimate.
    afc.reset();
    assert_eq!(afc.offset_hz(), 0.0);
}

#[test]
fn afc_hold_freezes_the_estimate() {
    let fs = 100_000u64;
    let cfg = AfcConfig {
        max_offset_hz: 20_000.0,
        time_constant_s: 0.02,
        ..AfcConfig::default()
    };
    let mut afc = afc_block(&cfg, fs);
    let mut out = vec![C32::default(); 30_000];
    afc.process(&tone(30_000, fs as f64, |_| 3_000.0), &mut out)
        .unwrap();
    let held = afc.offset_hz();
    assert!((held - 3_000.0).abs() < 0.5, "{held}");
    afc.set_param("hold", ParamValue::Bool(true)).unwrap();
    assert_eq!(afc.param("hold"), Ok(ParamValue::Bool(true)));
    // The carrier moves; the estimate stays, and the output keeps the shift by it.
    afc.process(&tone(30_000, fs as f64, |_| 4_000.0), &mut out)
        .unwrap();
    assert_eq!(afc.offset_hz(), held);
    for f in &freq_hz(&out[1000..], fs as f64) {
        assert!((f - (4_000.0 - held)).abs() < 2.0, "{f}");
    }
    afc.set_param("hold", ParamValue::Bool(false)).unwrap();
    afc.process(&tone(30_000, fs as f64, |_| 4_000.0), &mut out)
        .unwrap();
    assert!(
        (afc.offset_hz() - 4_000.0).abs() < 0.5,
        "{}",
        afc.offset_hz()
    );
    // Build parameters are read-only; the limit must stay below half the sample rate.
    assert_eq!(
        afc.set_param("max_offset_hz", ParamValue::F64(1.0)),
        Err(Error::ReadOnlyParameter)
    );
    for bad in [
        AfcConfig {
            max_offset_hz: 50_000.0,
            ..cfg
        },
        AfcConfig {
            max_offset_hz: 0.0,
            ..cfg
        },
        AfcConfig {
            time_constant_s: 0.0,
            ..cfg
        },
        AfcConfig {
            carrier_hz: -1.0,
            ..cfg
        },
        AfcConfig {
            carrier_hz: f64::NAN,
            ..cfg
        },
    ] {
        assert!(bad.validate(rate(fs)).is_err(), "{bad:?}");
    }
    let r = rate(fs);
    let bytes = AfcBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, r).unwrap();
    assert_eq!(bytes, AFC_PIECE * 4);
    check_exact_storage(bytes, |a| AfcBlock::new(&cfg, r, a).map(|_| ()));
    let t = afc.timing();
    assert_eq!((t.ratio, t.kind), (Ratio::ONE, RatioKind::Constant));
    assert_eq!(t.delay.reduced(), Delay::ZERO.reduced());
}

#[test]
fn afc_survives_nan_and_recovers() {
    let fs = 100_000u64;
    let cfg = AfcConfig {
        max_offset_hz: 20_000.0,
        time_constant_s: 0.02,
        ..AfcConfig::default()
    };
    let mut afc = afc_block(&cfg, fs);
    let mut input = tone(40_000, fs as f64, |_| 2_000.0);
    for s in &mut input[10_000..10_100] {
        *s = C32::new(f32::NAN, f32::NAN);
    }
    input[10_200] = C32::new(f32::INFINITY, 0.0);
    let mut out = vec![C32::default(); input.len()];
    afc.process(&input, &mut out).unwrap();
    assert!(afc.offset_hz().is_finite());
    assert!(
        (afc.offset_hz() - 2_000.0).abs() < 0.5,
        "{}",
        afc.offset_hz()
    );
    assert!(
        out[30_000..]
            .iter()
            .all(|s| s.re.is_finite() && s.im.is_finite())
    );
}

#[test]
fn the_three_blocks_are_chunk_invariant() {
    let plans = twenty_plans(SEED);
    // fsk_mod: 4-FSK GFSK.
    let cfg = FskModConfig {
        levels: 4,
        samples_per_symbol: 5,
        deviation_hz: 1_000.0,
        bt: 0.5,
        ..FskModConfig::default()
    };
    let bits = soft_bits(6000, 10);
    let whole = run_block(&mut fsk_mod(&cfg, 4_000), &bits, None);
    assert_eq!(whole.len(), 15_000);
    for plan in &plans {
        let got = run_block(&mut fsk_mod(&cfg, 4_000), &bits, Some(*plan));
        same_bits(&whole, &got, "fsk_mod chunks");
    }
    // fsk_demod: 2-FSK and 4-FSK at a fractional sps.
    for levels in [2usize, 4] {
        let bps = if levels == 4 { 2 } else { 1 };
        let signal = fsk_signal(levels, 8, &soft_bits(1500 * bps, 11));
        let cfg = FskDemodConfig {
            levels,
            symbol_rate_hz: 1_010.0,
            deviation_hz: 500.0,
            ..FskDemodConfig::default()
        };
        let mk = || FskDemodBlock::new(&cfg, rate(8_000), &mut VecStorage).unwrap();
        let whole = run_block(&mut mk(), &signal, None);
        for plan in &plans {
            same_bits(
                &whole,
                &run_block(&mut mk(), &signal, Some(*plan)),
                "fsk_demod chunks",
            );
        }
    }
    // afc.
    let cfg = AfcConfig {
        max_offset_hz: 20_000.0,
        time_constant_s: 0.01,
        ..AfcConfig::default()
    };
    let input = tone(20_000, 100_000.0, |i| 1_500.0 + 0.05 * i as f64);
    let whole = run_block(&mut afc_block(&cfg, 100_000), &input, None);
    for plan in &plans {
        let got = run_block(&mut afc_block(&cfg, 100_000), &input, Some(*plan));
        same_bits(&whole, &got, "afc chunks");
    }
}
