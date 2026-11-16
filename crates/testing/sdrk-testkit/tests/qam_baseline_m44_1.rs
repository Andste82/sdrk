//! The baselines of M44.1 (spec §4 and §5, tests 2 and 5): the outputs of `qam_mod`, `psk_mod`
//! and `qam_demod` as they are before the blocks become composites, recorded from the unchanged
//! code in `tests/baseline/m44_1/` (`sdrk_testkit::baseline`: one 64-bit FNV-1a digest of the
//! output bits per case and SIMD level, and the full output of a few short cases). The same
//! checks then hold the composites to the same bits. The signals of `qam_ber_m44`,
//! `qam_restart_m44` and the flow loop `qam_loop` are checked in those tests themselves.
//!
//! - **Modulators** (test 2, seed 841): `qam_mod` (orders 16 and 64) and `psk_mod` (every valid
//!   combination of `order`, `differential` and `variant`: BPSK, QPSK and 8PSK coherent and
//!   differential, pi/4-DQPSK, OQPSK) at 2, 4, 5 and 8 samples per symbol (OQPSK: the even
//!   ones) with the pulses of roll-off 0.35 and span 10 (the default), 0.2 and 6, and 0.5 and
//!   32 (the longest), on 700 symbols of random soft bits with the extreme values of the sign
//!   rule mixed in (0, -0, NaN, the infinities, the smallest normal numbers); every
//!   combination at 4 samples per symbol also in the 20 cut plans of seed 841, half of them
//!   with small output room.
//! - **`qam_demod`** (test 5, seed 843): the signals of `qam_blocks` (the tests of M44 and its
//!   rounds I44-3 to I44-17: the comparison with the kernels, the soft bits that do not fit,
//!   the error-free links with carrier offsets up to the edge of the range, the acquisition
//!   after its window, after silence and after noise, the phases of the acquisition and of
//!   the verify windows on a weak signal and after a carrier, the search again after a
//!   carrier and after noise, and the link through `frame_decode` turned by 0 to 270 degrees),
//!   whole, and three of them (a restart by the verify window, the restart after a carrier
//!   with the noise between, and a large offset) in the 20 cut plans of seed 843.
//!
//! The signals are those of `sdrk-blocks/tests/qam_blocks.rs` (seed 607 and its streams), with
//! the sines and cosines of `libm` instead of the platform's, so that the inputs are the same
//! bits on every host.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::large_stack_frames,
    clippy::useless_vec
)]

use sdrk::Rate;
use sdrk_blocks::{
    Block, PskModBlock, PskModConfig, PskVariant, QAM_ACQUISITION_CHECK, QAM_ACQUISITION_SKIP,
    QAM_ACQUISITION_SYMBOLS, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig, StreamElem,
};
use sdrk_core::C32;
use sdrk_proto::frame::{Whitening, encode, frame_bits};
use sdrk_proto::{Packet, PacketFlags};
use sdrk_testkit::baseline::{self, Baseline, Digest};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

/// The baseline set of M44.1.
const SET: &str = "m44_1";
/// The seed of the modulator inputs and plans (test 2) and of the `qam_demod` plans (test 5).
const SEED_MOD: u64 = 841;
const SEED_DEMOD: u64 = 843;
/// The seed of the signals of `qam_blocks`.
const SEED_BLOCKS: u64 = 607;
/// Symbols per modulator case.
const MOD_SYMBOLS: usize = 700;
/// Symbols of the short modulator cases whose full output is stored.
const VECTOR_SYMBOLS: usize = 128;

fn level() -> &'static str {
    baseline::level(sdrk_backend_simd::level_name())
}

fn table(name: &str) -> Baseline {
    Baseline::load(SET, name, level()).unwrap()
}

fn rate(hz: u64) -> Rate {
    Rate::hz(hz).unwrap()
}

/// The 20 cut plans of `seed`: input pieces of 1 to 4000 samples (the sizes of the plans of
/// `sdrk-blocks/tests/common`).
fn twenty_plans(seed: u64) -> Vec<ChunkPlan> {
    [
        1usize, 2, 3, 5, 7, 11, 16, 33, 64, 100, 128, 255, 256, 257, 300, 512, 700, 1000, 1500,
        4000,
    ]
    .iter()
    .enumerate()
    .map(|(i, &m)| ChunkPlan::new(seed * 100 + i as u64, m))
    .collect()
}

/// Runs `input` through `block`: whole (`plan` `None`, output room 4096) or in the pieces of
/// `plan`; with `small_room` the output room of each call also follows the plan (at least
/// `min_room`), otherwise it is 4096.
fn run<I: StreamElem, O: StreamElem + Default, B: Block<In = I, Out = O>>(
    block: &mut B,
    input: &[I],
    plan: Option<ChunkPlan>,
    small_room: bool,
    min_room: usize,
) -> Vec<O> {
    let mut out = Vec::new();
    let mut room = vec![O::default(); 4096];
    let mut pending: Vec<I> = Vec::new();
    let mut sizes = plan.map(|p| p.sizes());
    let mut rooms = plan.map(|p| ChunkPlan::new(p.seed() + 7, p.max()).sizes());
    let mut pos = 0;
    while pos < input.len() {
        let want = sizes
            .as_mut()
            .map_or(input.len(), |s| s.next().unwrap())
            .min(input.len() - pos);
        pending.extend_from_slice(&input[pos..pos + want]);
        pos += want;
        loop {
            let r = match rooms.as_mut() {
                Some(r) if small_room => r.next().unwrap().clamp(min_room, 4096),
                _ => 4096,
            };
            let p = block.process(&pending, &mut room[..r]).unwrap();
            out.extend_from_slice(&room[..p.produced]);
            pending.drain(..p.consumed);
            if (p.consumed == 0 && p.produced == 0) || (pending.is_empty() && p.produced < r) {
                break;
            }
        }
    }
    out
}

// ---- modulators (test 2) ----------------------------------------------------------------

/// Random soft bits `+-1` of seed 841 and `stream`, with an extreme value of the sign rule at
/// every 37th position.
fn mod_bits(n: usize, stream: u64) -> Vec<f32> {
    const EXTREMES: [f32; 10] = [
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-30,
        0.5,
        -2.0,
    ];
    let mut rng = Pcg32::new(SEED_MOD, stream);
    (0..n)
        .map(|i| {
            let r = rng.next_u32();
            if i % 37 == 36 {
                EXTREMES[(r % 10) as usize]
            } else if r & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// The pulses of the cases: (roll-off, span).
const SHAPES: [(f64, usize); 3] = [(0.35, 10), (0.2, 6), (0.5, 32)];
/// The samples per symbol of the cases.
const SPS: [usize; 4] = [2, 4, 5, 8];
/// The valid combinations of `order`, `differential` and `variant` of `psk_mod`.
const PSK: [(u32, bool, PskVariant); 8] = [
    (2, false, PskVariant::Plain),
    (2, true, PskVariant::Plain),
    (4, false, PskVariant::Plain),
    (4, true, PskVariant::Plain),
    (8, false, PskVariant::Plain),
    (8, true, PskVariant::Plain),
    (4, true, PskVariant::Pi4),
    (4, false, PskVariant::Offset),
];

fn psk_name(order: u32, differential: bool, variant: PskVariant) -> String {
    let v = match variant {
        PskVariant::Plain => "plain",
        PskVariant::Pi4 => "pi4",
        _ => "offset",
    };
    let d = if differential { "diff" } else { "coh" };
    format!("psk{order}_{d}_{v}")
}

fn psk_block(cfg: &PskModConfig) -> PskModBlock<VecStorage> {
    PskModBlock::new(cfg, rate(48_000), &mut VecStorage).unwrap()
}

fn qam_block(cfg: &QamModConfig) -> QamModBlock<VecStorage> {
    QamModBlock::new(cfg, rate(48_000), &mut VecStorage).unwrap()
}

/// Every case of a modulator whole; the cases at 4 samples per symbol and the default pulse in
/// the 20 plans (alternately with output room 4096 and small output room).
fn check_modulator<B, F>(t: &Baseline, case: &str, bits: &[f32], sps: usize, plans: bool, make: F)
where
    B: Block<In = f32, Out = C32>,
    F: Fn() -> B,
{
    let whole = run(&mut make(), bits, None, false, sps);
    assert_eq!(whole.len(), MOD_SYMBOLS * sps, "{case}");
    t.check(case, &Digest::of_c32(&whole)).unwrap();
    if plans {
        for (i, plan) in twenty_plans(SEED_MOD).into_iter().enumerate() {
            let got = run(&mut make(), bits, Some(plan), i % 2 == 1, sps);
            t.check(case, &Digest::of_c32(&got))
                .unwrap_or_else(|e| panic!("{plan:?}: {e}"));
        }
    }
}

#[test]
fn qam_mod_keeps_its_baseline() {
    let t = table("qam_mod");
    let mut stream = 0u64;
    for order in [16u32, 64] {
        for (rolloff, span) in SHAPES {
            for sps in SPS {
                let cfg = QamModConfig {
                    order,
                    samples_per_symbol: sps,
                    rolloff,
                    span,
                };
                let bits = mod_bits(MOD_SYMBOLS * cfg.bits_per_symbol(), stream);
                stream += 1;
                let case = format!("qam{order}_sps{sps}_r{rolloff}_s{span}");
                let plans = sps == 4 && span == 10;
                check_modulator(&t, &case, &bits, sps, plans, || qam_block(&cfg));
            }
        }
    }
    // the full output of a short case of each order
    for order in [16u32, 64] {
        let cfg = QamModConfig {
            order,
            samples_per_symbol: 4,
            ..QamModConfig::default()
        };
        let bits = mod_bits(
            VECTOR_SYMBOLS * cfg.bits_per_symbol(),
            100 + u64::from(order),
        );
        let out = run(&mut qam_block(&cfg), &bits, None, false, 4);
        t.check_vector(
            &format!("qam{order}_sps4_short"),
            &baseline::c32_bytes(&out),
        )
        .unwrap();
    }
}

#[test]
fn psk_mod_keeps_its_baseline() {
    let t = table("psk_mod");
    let mut stream = 1000u64;
    for (order, differential, variant) in PSK {
        for (rolloff, span) in SHAPES {
            for sps in SPS {
                if variant == PskVariant::Offset && sps % 2 == 1 {
                    continue;
                }
                let cfg = PskModConfig {
                    order,
                    samples_per_symbol: sps,
                    rolloff,
                    span,
                    differential,
                    variant,
                };
                let bits = mod_bits(MOD_SYMBOLS * cfg.bits_per_symbol(), stream);
                stream += 1;
                let case = format!(
                    "{}_sps{sps}_r{rolloff}_s{span}",
                    psk_name(order, differential, variant)
                );
                let plans = sps == 4 && span == 10;
                check_modulator(&t, &case, &bits, sps, plans, || psk_block(&cfg));
            }
        }
    }
    // the full output of a short case of the three variants that act beyond the mapping
    for (order, differential, variant) in [
        (8u32, true, PskVariant::Plain),
        (4, true, PskVariant::Pi4),
        (4, false, PskVariant::Offset),
    ] {
        let cfg = PskModConfig {
            order,
            samples_per_symbol: 4,
            differential,
            variant,
            ..PskModConfig::default()
        };
        let bits = mod_bits(
            VECTOR_SYMBOLS * cfg.bits_per_symbol(),
            2000 + u64::from(order),
        );
        let out = run(&mut psk_block(&cfg), &bits, None, false, 4);
        let case = format!("{}_sps4_short", psk_name(order, differential, variant));
        t.check_vector(&case, &baseline::c32_bytes(&out)).unwrap();
    }
}

// ---- qam_demod on the signals of qam_blocks (test 5) ------------------------------------

/// Random soft bits `+-1` (stream `stream` of seed 607), as in `qam_blocks`.
fn random_bits(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED_BLOCKS, stream);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

fn modulate(order: u32, sps: usize, bits: &[f32]) -> Vec<C32> {
    let cfg = QamModConfig {
        order,
        samples_per_symbol: sps,
        ..QamModConfig::default()
    };
    run(&mut qam_block(&cfg), bits, None, false, sps)
}

fn bits_of(order: u32) -> usize {
    if order == 64 { 6 } else { 4 }
}

/// `x` turned by `phase0 + 2 pi offset k / sps` at sample `k`, with Gaussian noise of `noise`
/// per component from `rng`.
fn turn(x: &[C32], phase0: f64, offset: f64, sps: usize, noise: f64, rng: &mut Pcg32) -> Vec<C32> {
    x.iter()
        .enumerate()
        .map(|(i, x)| {
            let ph = phase0 + core::f64::consts::TAU * offset * i as f64 / sps as f64;
            let (s, c) = (libm::sin(ph), libm::cos(ph));
            let (re, im) = (f64::from(x.re), f64::from(x.im));
            C32::new(
                (re * c - im * s + noise * rng.normal()) as f32,
                (re * s + im * c + noise * rng.normal()) as f32,
            )
        })
        .collect()
}

/// The `received` of `qam_blocks`: the phase `0.6 + 2 pi 0.001 k / sps` and noise of `noise`
/// per component (stream `100 + stream`).
fn received(order: u32, sps: usize, n_sym: usize, noise: f64, stream: u64) -> Vec<C32> {
    let tx = modulate(order, sps, &random_bits(n_sym * bits_of(order), stream));
    let mut rng = Pcg32::new(SEED_BLOCKS, 100 + stream);
    turn(&tx, 0.6, 0.001, sps, noise, &mut rng)
}

/// The noise per component of [`received`] for `Es/N0` in dB, as in `qam_blocks`.
fn sigma_for(order: u32, sps: usize, n_sym: usize, stream: u64, esn0_db: f64) -> f64 {
    let clean = received(order, sps, n_sym, 0.0, stream);
    let body = &clean[100 * sps..];
    let power = body
        .iter()
        .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
        .sum::<f64>()
        / body.len() as f64;
    libm::sqrt(power * sps as f64 / (2.0 * libm::pow(10.0, esn0_db / 10.0)))
}

/// A carrier at 0.06 cycles per symbol of amplitude 0.7 for `n` samples at 4 samples per symbol.
fn carrier(n: usize) -> Vec<C32> {
    (0..n)
        .map(|j| {
            let ph = core::f64::consts::TAU * 0.06 * j as f64 / 4.0;
            C32::new((0.7 * libm::cos(ph)) as f32, (0.7 * libm::sin(ph)) as f32)
        })
        .collect()
}

/// Gaussian noise of 0.7 per component for `n` samples (stream `stream` of seed 607).
fn noise(n: usize, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED_BLOCKS, stream);
    (0..n)
        .map(|_| C32::new(rng.normal() as f32, rng.normal() as f32) * 0.7)
        .collect()
}

/// A `qam_demod` case: its name, the configuration, the sample rate and the input.
struct DemodCase {
    name: String,
    cfg: QamDemodConfig,
    fs: Rate,
    input: Vec<C32>,
}

fn config(order: u32, symbol_rate: f64, taps: usize) -> QamDemodConfig {
    QamDemodConfig {
        order,
        symbol_rate_hz: symbol_rate,
        equalizer_taps: taps,
        ..QamDemodConfig::default()
    }
}

/// The frames of `qam_link_with_frame_decode_four_resolves_every_rotation`: 4000 random symbols,
/// then 12 frames with 50 random bits between them, then 200 random symbols.
fn frame_bits_of(order: u32) -> Vec<f32> {
    let k = bits_of(order);
    let mut rng = Pcg32::new(SEED_BLOCKS, 200 + u64::from(order));
    let packets: Vec<Packet> = (0..12)
        .map(|_| {
            let len = 1 + rng.below(40) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect();
    let random = |rng: &mut Pcg32, n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect()
    };
    let mut bits = random(&mut rng, 4000 * k);
    for p in &packets {
        let mut frame = vec![0u8; frame_bits(p.data().len(), 4)];
        encode(p, 4, Whitening::None, &mut frame).unwrap();
        bits.extend(frame.iter().map(|&b| if b == 1 { 1.0f32 } else { -1.0 }));
        bits.extend(random(&mut rng, 50));
    }
    bits.extend(random(&mut rng, 200 * k));
    bits.truncate(bits.len() / k * k);
    bits
}

/// Every signal of the `qam_demod` tests of `qam_blocks`.
fn demod_cases() -> Vec<DemodCase> {
    let mut cases = Vec::new();
    let mut add = |name: String, cfg: QamDemodConfig, fs: Rate, input: Vec<C32>| {
        cases.push(DemodCase {
            name,
            cfg,
            fs,
            input,
        });
    };
    // qam_demod_equals_the_kernels_whole_and_in_cut_plans
    for (i, (order, sps, taps)) in [
        (16u32, 4usize, 11usize),
        (64, 5, 15),
        (16, 2, 0),
        (64, 8, 0),
    ]
    .into_iter()
    .enumerate()
    {
        add(
            format!("kernels{i}_qam{order}_sps{sps}_taps{taps}"),
            config(order, 9600.0, taps),
            rate(9600 * sps as u64),
            received(order, sps, 3000, 0.02, 20 + i as u64),
        );
    }
    // qam_demod_keeps_soft_bits_that_do_not_fit
    add(
        "room_qam64_sps4".to_string(),
        config(64, 9600.0, 11),
        rate(38_400),
        received(64, 4, 2000, 0.01, 30),
    );
    // qam_mod_to_qam_demod_with_little_noise_is_error_free
    for (i, (order, sps, offset)) in [
        (16u32, 4usize, 0.0005),
        (64, 4, 0.0005),
        (16, 8, 0.0005),
        (64, 4, 0.04),
        (16, 4, -0.1),
        (64, 8, -0.12),
    ]
    .into_iter()
    .enumerate()
    {
        let sig = modulate(
            order,
            sps,
            &random_bits(6000 * bits_of(order), 40 + i as u64),
        );
        let mut rng = Pcg32::new(SEED_BLOCKS, 140 + i as u64);
        add(
            format!("link{i}_qam{order}_sps{sps}_offset{offset}"),
            config(order, 1000.0, 11),
            rate(1000 * sps as u64),
            turn(&sig, 1.1, offset, sps, 0.005, &mut rng),
        );
    }
    for taps in [11usize, 0] {
        // qam_demod_acquires_once_after_its_window_and_again_after_reset
        let input = received(16, 4, 2000, 0.01, 50);
        let mut late = noise(4 * 800, 60);
        late.extend_from_slice(&input);
        let cfg = config(16, 9600.0, taps);
        add(format!("acquire_taps{taps}"), cfg, rate(38_400), input);
        add(
            format!("acquire_silence_taps{taps}"),
            cfg,
            rate(38_400),
            vec![C32::default(); 4000],
        );
        add(format!("acquire_late_taps{taps}"), cfg, rate(38_400), late);
        // qam_demod_acquisition_phases_are_independent_of_the_cut_plan
        let n_sig =
            QAM_ACQUISITION_SKIP + QAM_ACQUISITION_SYMBOLS + 3 * QAM_ACQUISITION_CHECK + 200;
        let sigma = sigma_for(16, 4, n_sig, 90, 12.0);
        let sig = received(16, 4, n_sig, sigma, 90);
        let mut lead = carrier(4 * 1000);
        lead.extend_from_slice(&sig);
        add(format!("phases_weak_taps{taps}"), cfg, rate(38_400), sig);
        add(
            format!("phases_carrier_weak_taps{taps}"),
            cfg,
            rate(38_400),
            lead,
        );
        // qam_demod_lock_and_verify_are_independent_of_the_cut_plan
        let gap = noise(4 * 9000, 92);
        let strong = received(16, 4, 6000, 0.01, 91);
        let mut strong_noise = strong.clone();
        strong_noise.extend_from_slice(&gap);
        let mut carrier_noise = carrier(4 * 1000);
        carrier_noise.extend_from_slice(&gap);
        carrier_noise.extend_from_slice(&strong);
        add(
            format!("verify_strong_noise_taps{taps}"),
            cfg,
            rate(38_400),
            strong_noise,
        );
        add(
            format!("verify_carrier_noise_strong_taps{taps}"),
            cfg,
            rate(38_400),
            carrier_noise,
        );
        // qam_demod_searches_again_when_the_loop_does_not_lock
        let sig = received(16, 4, 12_000, 0.01, 71);
        let mut tone = carrier(4 * 6000);
        tone.extend_from_slice(&sig);
        let mut lead = noise(4 * 40 * QAM_ACQUISITION_SYMBOLS, 70);
        lead.extend_from_slice(&sig);
        add(
            format!("search_carrier_taps{taps}"),
            cfg,
            rate(38_400),
            tone,
        );
        add(format!("search_noise_taps{taps}"), cfg, rate(38_400), lead);
    }
    // qam_link_with_frame_decode_four_resolves_every_rotation
    for order in [16u32, 64] {
        let sig = modulate(order, 4, &frame_bits_of(order));
        for quarter in 0..4u32 {
            let turned: Vec<C32> = sig
                .iter()
                .map(|x| match quarter {
                    0 => *x,
                    1 => C32::new(-x.im, x.re),
                    2 => C32::new(-x.re, -x.im),
                    _ => C32::new(x.im, -x.re),
                })
                .collect();
            add(
                format!("frames_qam{order}_turn{quarter}"),
                config(order, 1000.0, 11),
                rate(4000),
                turned,
            );
        }
    }
    cases
}

/// The receiver of a case with the recovery of M44.1 §6 off (`set_recovery(false)`): test 5 asks
/// for the bits of step 1 (D1570; with the recovery on, the weight watch takes the receiver of
/// `verify_carrier_noise_strong_taps11` out of its lock on the pure carrier in the noise, D1574).
fn demod_block(case: &DemodCase) -> QamDemodBlock<VecStorage> {
    let mut block = QamDemodBlock::new(&case.cfg, case.fs, &mut VecStorage).unwrap();
    block.set_recovery(false);
    block
}

#[test]
fn qam_demod_keeps_its_baseline_on_the_signals_of_qam_blocks() {
    let t = table("qam_demod");
    let cases = demod_cases();
    let results: Vec<(String, Digest)> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|case| {
                s.spawn(move || {
                    let soft = run(&mut demod_block(case), &case.input, None, false, 1);
                    (case.name.clone(), Digest::of_f32(&soft))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (name, digest) in &results {
        assert!(!digest.is_empty(), "{name}: no output");
        t.check(name, digest).unwrap();
    }
}

#[test]
fn qam_demod_keeps_its_baseline_in_twenty_cut_plans() {
    let t = table("qam_demod");
    let cases = demod_cases();
    let chosen: Vec<&DemodCase> = [
        "phases_carrier_weak_taps11",
        "verify_carrier_noise_strong_taps0",
        "link3_qam64_sps4_offset0.04",
    ]
    .iter()
    .map(|n| cases.iter().find(|c| c.name == *n).unwrap())
    .collect();
    std::thread::scope(|s| {
        for case in chosen {
            let t = &t;
            s.spawn(move || {
                for (i, plan) in twenty_plans(SEED_DEMOD).into_iter().enumerate() {
                    let soft = run(
                        &mut demod_block(case),
                        &case.input,
                        Some(plan),
                        i % 2 == 1,
                        1,
                    );
                    t.check(&case.name, &Digest::of_f32(&soft))
                        .unwrap_or_else(|e| panic!("{plan:?}: {e}"));
                }
            });
        }
    });
}

#[test]
fn qam_demod_keeps_its_full_output_on_a_short_signal() {
    // 1500 symbols of the first signal of `qam_demod_equals_the_kernels_whole_and_in_cut_plans`:
    // the acquisition window, the blind symbols of the equalizer and the hand-over to the
    // decision mode
    let t = table("qam_demod");
    let case = DemodCase {
        name: "short_qam16_sps4_taps11".to_string(),
        cfg: config(16, 9600.0, 11),
        fs: rate(38_400),
        input: received(16, 4, 1500, 0.02, 20),
    };
    let soft = run(&mut demod_block(&case), &case.input, None, false, 1);
    t.check_vector(&case.name, &baseline::f32_bytes(&soft))
        .unwrap();
}

/// The frequency acquisition of `qam_demod` (test 3 of M44.1): after every call of 4096 input
/// samples the acquired frequency (its `f64` bits, all ones before the first estimate), the
/// restarts and the lock of the carrier loop, on every signal of `qam_blocks`. Recorded from the
/// block before it became a composite, so that the estimates of `qam_acquire` inside the composite
/// are checked window by window against those of the old block.
#[test]
fn qam_demod_keeps_its_acquired_frequencies() {
    let t = table("qam_acquire");
    let cases = demod_cases();
    let results: Vec<(String, Digest)> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|case| {
                s.spawn(move || {
                    let mut block = demod_block(case);
                    let mut room = vec![0.0f32; 16_384];
                    let mut d = Digest::new();
                    for piece in case.input.chunks(4096) {
                        let mut at = 0;
                        while at < piece.len() {
                            let p = block.process(&piece[at..], &mut room).unwrap();
                            at += p.consumed;
                        }
                        let f = block.acquired_frequency().map_or(u64::MAX, f64::to_bits);
                        d.bytes(&f.to_le_bytes());
                        d.bytes(&block.restarts().to_le_bytes());
                        d.bytes(&[u8::from(block.locked())]);
                    }
                    (case.name.clone(), d)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (name, digest) in &results {
        t.check(name, digest).unwrap();
    }
}
