//! The blocks of M47: `MorseEncodeBlock` (the key envelope equals the units of the encoder
//! of `sdrk-proto` at any rate, the raised-cosine edges, Farnsworth, the tail of every packet,
//! the idle output, 20 cut plans) and `MorseDecodeBlock` (a round trip at 5 to 40 words per
//! minute, the threshold, glitches, the packet fields, the output room, the extremes, the
//! speed estimate, 20 cut plans), their parameters, timing and rates.

#![allow(
    clippy::assert_is_empty,
    clippy::needless_range_loop,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

mod common;

use common::{check_exact_storage, rate, same_bits, twenty_plans};
use sdrk::{Delay, RatioKind};
use sdrk_blocks::{
    ArenaStorage, Block, Error, MorseDecodeBlock, MorseDecodeConfig, MorseEncodeBlock,
    MorseEncodeConfig, Packet, PacketFlags, Packets, ParamValue, StreamElem,
};
use sdrk_proto::morse::Encoder;
use sdrk_testkit::{ChunkPlan, Pcg32};

const FS: u64 = 8000;

/// Runs the encoder over `packets` with packet and output pieces from the plans (`None`: all
/// at once, output room 8192) and returns every sample.
fn run_encoder(
    enc: &mut MorseEncodeBlock<ArenaStorage<'static>>,
    packets: &[Packet],
    plan_in: Option<ChunkPlan>,
    plan_out: Option<ChunkPlan>,
) -> Vec<f32> {
    let mut sizes_in = plan_in.map(|p| p.sizes());
    let mut sizes_out = plan_out.map(|p| p.sizes());
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < packets.len() {
        let want_in = sizes_in
            .as_mut()
            .map_or(packets.len(), |s| s.next().unwrap());
        let want_out = sizes_out.as_mut().map_or(8192, |s| s.next().unwrap());
        let end = (pos + want_in).min(packets.len());
        let mut room = vec![f32::NAN; want_out];
        let p = enc.process(&packets[pos..end], &mut room).unwrap();
        if p.consumed < end - pos {
            assert_eq!(p.produced, want_out, "the room is used up");
        }
        out.extend_from_slice(&room[..p.produced]);
        pos += p.consumed;
    }
    out
}

/// Runs a packet decoder over `input` with input and output pieces from the plans (`None`:
/// all at once, room 64) and returns every packet.
fn run_decoder<I: StreamElem, B: Block<In = I, Out = Packets>>(
    dec: &mut B,
    input: &[I],
    plan_in: Option<ChunkPlan>,
    plan_out: Option<ChunkPlan>,
) -> Vec<Packet> {
    let mut sizes_in = plan_in.map(|p| p.sizes());
    let mut sizes_out = plan_out.map(|p| p.sizes());
    let mut out = Vec::new();
    let mut pos = 0;
    let mut stalls = 0;
    loop {
        let want_in = sizes_in.as_mut().map_or(input.len(), |s| s.next().unwrap());
        let want_out = sizes_out.as_mut().map_or(64, |s| s.next().unwrap());
        let end = (pos + want_in).min(input.len());
        let mut room = vec![Packet::default(); want_out];
        let p = dec.process(&input[pos..end], &mut room).unwrap();
        assert!(p.consumed <= end - pos && p.produced <= want_out);
        out.extend_from_slice(&room[..p.produced]);
        pos += p.consumed;
        if pos == input.len() && p.produced < want_out {
            break;
        }
        stalls = if p.consumed == 0 && p.produced == 0 {
            stalls + 1
        } else {
            0
        };
        assert!(stalls < 3, "the decoder stalls at {pos}");
    }
    out
}

fn packet(text: &[u8]) -> Packet {
    Packet::new(0, PacketFlags::NONE, text).unwrap()
}

fn encoder(cfg: &MorseEncodeConfig, fs: u64) -> MorseEncodeBlock<ArenaStorage<'static>> {
    MorseEncodeBlock::new(cfg, rate(fs), &mut ArenaStorage::new(&mut [])).unwrap()
}

fn decoder(cfg: &MorseDecodeConfig, fs: u64) -> MorseDecodeBlock<ArenaStorage<'static>> {
    MorseDecodeBlock::new(cfg, rate(fs), &mut ArenaStorage::new(&mut [])).unwrap()
}

/// The envelope of `texts` (one packet each) at the rate `fs`, through the block.
fn encode(texts: &[&[u8]], cfg: &MorseEncodeConfig, fs: u64) -> Vec<f32> {
    let packets: Vec<Packet> = texts.iter().map(|t| packet(t)).collect();
    run_encoder(&mut encoder(cfg, fs), &packets, None, None)
}

/// The units the encoder of `sdrk-proto` sends for `texts`, each packet followed by its
/// word gap of `gap` units.
fn units(texts: &[&[u8]], wpm: f32, char_wpm: f32) -> Vec<bool> {
    let mut enc = Encoder::new(wpm, char_wpm).unwrap();
    let mut out = Vec::new();
    for t in texts {
        for &c in *t {
            enc.push(c).unwrap();
            while let Some(k) = enc.next_unit() {
                out.push(k);
            }
        }
        out.extend(std::iter::repeat_n(false, enc.word_gap() as usize));
        enc.settle();
    }
    out
}

/// Unit `k` of the stream ends at the sample `round((k + 1) * spu)`.
fn expand(units: &[bool], spu: f64) -> Vec<f32> {
    let mut out = Vec::new();
    for (k, &on) in units.iter().enumerate() {
        let end = ((k + 1) as f64 * spu + 0.5).floor() as usize;
        while out.len() < end {
            out.push(if on { 1.0 } else { 0.0 });
        }
    }
    out
}

fn hard() -> MorseEncodeConfig {
    MorseEncodeConfig {
        rise_ms: 0.0,
        ..MorseEncodeConfig::default()
    }
}

/// `env` after 3000 samples of silence: the decoder starts its smoother and its level trackers
/// at the first sample, so a stream begins with noise or silence.
fn lead(env: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 3000];
    out.extend_from_slice(env);
    out
}

/// The words the decoder returns for `env`, with their indices.
fn words(env: &[f32], cfg: &MorseDecodeConfig, fs: u64, plan: Option<ChunkPlan>) -> Vec<Packet> {
    run_decoder(&mut decoder(cfg, fs), &lead(env), plan, None)
}

fn texts_of(packets: &[Packet]) -> Vec<String> {
    packets
        .iter()
        .map(|p| String::from_utf8(p.data().to_vec()).unwrap())
        .collect()
}

#[test]
fn the_envelope_is_the_units_of_the_encoder() {
    let texts: [&[u8]; 3] = [b"CQ DE W1AW", b"73", b"e"];
    for (fs, wpm) in [
        (8000u64, 20.0),
        (8000, 5.0),
        (8000, 40.0),
        (44100, 17.0),
        (1000, 12.0),
    ] {
        let cfg = MorseEncodeConfig { wpm, ..hard() };
        let got = encode(&texts, &cfg, fs);
        let want = expand(
            &units(&texts, wpm as f32, wpm as f32),
            fs as f64 * 1.2 / wpm,
        );
        same_bits(&got, &want, &format!("{wpm} wpm at {fs} Hz"));
    }
}

#[test]
fn paris_lasts_fifty_units_at_every_speed() {
    // "PARIS" and its word gap are 50 units: the packet ends with the gap of 7.
    for (fs, wpm) in [(8000u64, 20.0), (8000, 10.0), (48000, 25.0)] {
        let cfg = MorseEncodeConfig { wpm, ..hard() };
        let got = encode(&[b"PARIS"], &cfg, fs);
        let want = 50.0 * fs as f64 * 1.2 / wpm;
        assert!(
            (got.len() as f64 - want).abs() <= 1.0,
            "{} vs {want}",
            got.len()
        );
    }
}

#[test]
fn farnsworth_stretches_the_gaps() {
    let cfg = MorseEncodeConfig {
        wpm: 10.0,
        char_wpm: 18.0,
        ..hard()
    };
    let got = encode(&[b"EE"], &cfg, FS);
    let want = expand(&units(&[b"EE"], 10.0, 18.0), FS as f64 * 1.2 / 18.0);
    same_bits(&got, &want, "farnsworth");
    // E, 9 units of gap, E, 22 units of word gap: 33 units of 1.2 / 18 s.
    assert_eq!(got.len(), (33.0 * FS as f64 * 1.2 / 18.0) as usize);
}

#[test]
fn the_edges_are_raised_cosines() {
    // 5 ms at 8 kHz are 40 samples; the unit at 20 wpm is 480 samples.
    let cfg = MorseEncodeConfig::default();
    let out = encode(&[b"EE"], &cfg, FS);
    for i in 0..40 {
        let x = (i + 1) as f32 / 40.0;
        let want = 0.5 * (1.0 - (core::f32::consts::PI * x).cos());
        assert!((out[i] - want).abs() < 1e-6, "rise {i}");
    }
    // Flat at 1.0 until the end of the unit, then the fall of 40 samples.
    assert!(out[40..480].iter().all(|&x| x == 1.0));
    for i in 0..40 {
        let x = (i + 1) as f32 / 40.0;
        let want = 1.0 - 0.5 * (1.0 - (core::f32::consts::PI * x).cos());
        assert!((out[480 + i] - want).abs() < 1e-6, "fall {i}");
    }
    assert!(out[520..1440].iter().all(|&x| x == 0.0));
    // The edge is cut to half a unit: 50 ms at 40 wpm (240 samples a unit) is 120 samples.
    let cfg = MorseEncodeConfig {
        wpm: 40.0,
        rise_ms: 50.0,
        ..MorseEncodeConfig::default()
    };
    let out = encode(&[b"E"], &cfg, FS);
    assert!(out[..120].windows(2).all(|w| w[1] > w[0]));
    assert!((out[119] - 1.0).abs() < 1e-6 && out[118] < 1.0);
    assert!(out[120..240].iter().all(|&x| x == 1.0));
    // Every sample is between 0 and 1.
    let out = encode(&[b"THE QUICK BROWN FOX"], &MorseEncodeConfig::default(), FS);
    assert!(out.iter().all(|&x| (0.0..=1.0).contains(&x)));
}

#[test]
fn a_call_without_packets_is_idle() {
    let mut enc = encoder(&hard(), FS);
    let mut out = vec![1.0f32; 3000];
    let p = enc.process(&[], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 3000));
    assert!(out.iter().all(|&x| x == 0.0));
    // A packet after idle starts on the next unit boundary (3360 = 7 units of 480) and ends in
    // its word gap: 360 samples, the unit of E and the 7 units of the gap.
    let mut out2 = vec![0.0f32; 5000];
    let p = enc.process(&[packet(b"E")], &mut out2).unwrap();
    assert_eq!((p.consumed, p.produced), (1, 360 + 8 * 480));
    assert!(out2[..360].iter().all(|&x| x == 0.0));
    assert!(out2[360..840].iter().all(|&x| x == 1.0));
    assert!(out2[840..360 + 8 * 480].iter().all(|&x| x == 0.0));
}
#[test]
fn packets_are_consumed_after_their_gap_and_continued() {
    // A small output room: the call that ends inside a packet must be continued with it.
    let texts: [&[u8]; 2] = [b"THE QUICK BROWN FOX", b"JUMPS"];
    let packets: Vec<Packet> = texts.iter().map(|t| packet(t)).collect();
    let want = encode(&texts, &MorseEncodeConfig::default(), FS);
    for room_len in [1usize, 7, 64, 479, 480, 5000] {
        let mut enc = encoder(&MorseEncodeConfig::default(), FS);
        let mut room = vec![0.0f32; room_len];
        let mut got = Vec::new();
        let mut pending: &[Packet] = &packets;
        while !pending.is_empty() {
            let p = enc.process(pending, &mut room).unwrap();
            got.extend_from_slice(&room[..p.produced]);
            pending = &pending[p.consumed..];
        }
        same_bits(&got, &want, &format!("room {room_len}"));
    }
}

#[test]
fn the_output_does_not_depend_on_how_the_packets_are_cut() {
    let mut rng = Pcg32::new(650, 1);
    let packets: Vec<Packet> = (0..30)
        .map(|_| {
            let len = 1 + rng.below(40) as usize;
            let text: Vec<u8> = (0..len)
                .map(|_| b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?/"[rng.below(41) as usize])
                .collect();
            packet(&text)
        })
        .collect();
    let want = run_encoder(
        &mut encoder(&MorseEncodeConfig::default(), FS),
        &packets,
        None,
        None,
    );
    for (plan, plan_out) in twenty_plans(650).into_iter().zip(twenty_plans(652)) {
        let got = run_encoder(
            &mut encoder(&MorseEncodeConfig::default(), FS),
            &packets,
            Some(plan),
            Some(plan_out),
        );
        same_bits(&got, &want, "plan");
    }
}

#[test]
fn skipped_bytes_and_empty_packets() {
    // A byte without a code takes no time; an empty packet is its word gap.
    let a = encode(&[b"E#E"], &hard(), FS);
    let b = encode(&[b"EE"], &hard(), FS);
    same_bits(&a, &b, "skipped byte");
    let c = encode(&[b""], &hard(), FS);
    assert_eq!(c.len(), 7 * 480);
    assert!(c.iter().all(|&x| x == 0.0));
}

#[test]
fn the_encoder_parameters_and_timing() {
    let enc = encoder(&MorseEncodeConfig::default(), FS);
    assert_eq!(enc.description().kind, "morse_encode");
    assert_eq!(enc.timing().kind, RatioKind::Source);
    assert_eq!(enc.timing().delay, Delay::ZERO);
    assert_eq!(enc.param("wpm").unwrap(), ParamValue::F64(20.0));
    assert_eq!(enc.param("char_wpm").unwrap(), ParamValue::F64(0.0));
    assert_eq!(enc.param("rise_ms").unwrap(), ParamValue::F64(5.0));
    let mut enc = enc;
    assert_eq!(
        enc.set_param("wpm", ParamValue::F64(25.0)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        enc.set_param("nope", ParamValue::F64(1.0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(enc.param("nope"), Err(Error::UnknownParameter));
    assert_eq!(
        MorseEncodeBlock::<ArenaStorage<'static>>::rate_out(
            &MorseEncodeConfig::default(),
            rate(FS)
        )
        .unwrap(),
        rate(FS)
    );
    check_exact_storage(
        MorseEncodeBlock::<ArenaStorage<'static>>::storage_bytes(
            &MorseEncodeConfig::default(),
            rate(FS),
        )
        .unwrap(),
        |arena| MorseEncodeBlock::new(&MorseEncodeConfig::default(), rate(FS), arena).map(|_| ()),
    );
    for bad in [
        MorseEncodeConfig {
            wpm: 4.9,
            ..MorseEncodeConfig::default()
        },
        MorseEncodeConfig {
            wpm: 60.1,
            ..MorseEncodeConfig::default()
        },
        MorseEncodeConfig {
            char_wpm: 19.0,
            ..MorseEncodeConfig::default()
        },
        MorseEncodeConfig {
            char_wpm: 100.1,
            ..MorseEncodeConfig::default()
        },
        MorseEncodeConfig {
            rise_ms: 50.1,
            ..MorseEncodeConfig::default()
        },
        MorseEncodeConfig {
            rise_ms: -1.0,
            ..MorseEncodeConfig::default()
        },
        MorseEncodeConfig {
            wpm: f64::NAN,
            ..MorseEncodeConfig::default()
        },
    ] {
        assert!(
            matches!(bad.validate(rate(FS)), Err(Error::OutOfRange { .. })),
            "{bad:?}"
        );
    }
    // Fewer than 8 samples per unit.
    assert!(MorseEncodeConfig::default().validate(rate(100)).is_err());
    assert!(MorseEncodeConfig::default().validate(rate(160)).is_ok());
}

#[test]
fn the_decoder_returns_the_words() {
    let text: &[u8] = b"VVV CQ CQ DE DL1ABC 599 K";
    for (fs, wpm) in [
        (8000u64, 20.0),
        (8000, 5.0),
        (8000, 40.0),
        (16000, 12.0),
        (2000, 25.0),
    ] {
        let cfg = MorseEncodeConfig {
            wpm,
            ..MorseEncodeConfig::default()
        };
        let env = encode(&[b"VVV CQ CQ DE DL1ABC 599 K"], &cfg, fs);
        let dcfg = MorseDecodeConfig {
            wpm: wpm.max(5.0),
            wpm_max: (wpm * 1.5).max(wpm),
            ..MorseDecodeConfig::default()
        };
        let got = words(&env, &dcfg, fs, None);
        assert_eq!(
            texts_of(&got),
            ["VVV", "CQ", "CQ", "DE", "DL1ABC", "599", "K"],
            "{wpm} wpm at {fs} Hz"
        );
        assert!(got.iter().all(|p| p.flags == PacketFlags::NONE));
        let _ = text;
    }
}

#[test]
fn the_packet_index_is_the_first_key_down_sample_of_the_word() {
    // Hard keying at 8 kHz, 20 wpm (480 samples a unit): the first word starts at sample 0,
    // the second after "EE" (1 + 3 + 1 units) and the word gap of 7 units.
    let env = encode(&[b"EE", b"E"], &hard(), FS);
    let got = words(&env, &MorseDecodeConfig::default(), FS, None);
    assert_eq!(texts_of(&got), ["EE", "E"]);
    // The decoder sees a key-down run at the smoothed value passing the middle: a short, fixed
    // delay of the smoother after the true edge (0.69 of 60 samples).
    let first = got[0].index;
    let second = got[1].index;
    // `words` puts 3000 samples of silence in front.
    assert!((3000..=3080).contains(&first), "{first}");
    let true_second = 3000 + (5 + 7) * 480;
    assert!((second as i64 - true_second as i64).abs() <= 80, "{second}");
    assert!(second > first);
}

#[test]
fn chunking_does_not_change_the_words() {
    let mut rng = Pcg32::new(651, 1);
    let texts: Vec<Vec<u8>> = (0..12)
        .map(|_| {
            let len = 3 + rng.below(20) as usize;
            (0..len)
                .map(|_| b"ETAOINSHRDLUCMWFGYPBVKJXQZ 0123456789"[rng.below(37) as usize])
                .collect()
        })
        .collect();
    let refs: Vec<&[u8]> = texts.iter().map(Vec::as_slice).collect();
    let mut prefixed = vec![b"VVV".as_slice()];
    prefixed.extend(refs);
    let env = encode(&prefixed, &MorseEncodeConfig::default(), FS);
    let want = words(&env, &MorseDecodeConfig::default(), FS, None);
    assert!(want.len() > 12);
    for plan in twenty_plans(651) {
        let got = words(&env, &MorseDecodeConfig::default(), FS, Some(plan));
        same_bits(&got, &want, "plan");
    }
}

#[test]
fn a_fixed_threshold_decodes_a_clean_envelope() {
    let env = encode(&[b"VVV TEST"], &hard(), FS);
    let cfg = MorseDecodeConfig {
        threshold: 0.5,
        ..MorseDecodeConfig::default()
    };
    assert_eq!(texts_of(&words(&env, &cfg, FS, None)), ["VVV", "TEST"]);
    // The level of the signal does not matter to the adaptive threshold.
    for gain in [1.0e-4f32, 0.02, 1.0, 300.0] {
        let scaled: Vec<f32> = env.iter().map(|&x| 0.05 * gain + x * gain).collect();
        let got = words(&scaled, &MorseDecodeConfig::default(), FS, None);
        assert_eq!(texts_of(&got), ["VVV", "TEST"], "gain {gain}");
    }
}

#[test]
fn glitches_are_merged() {
    // 20 wpm at 8 kHz: a unit is 480 samples; the glitch limit is 25 samples (40 wpm max).
    let mut env = encode(&[b"VVV TEST"], &hard(), FS);
    let want = texts_of(&words(&env, &MorseDecodeConfig::default(), FS, None));
    // A dropout of 8 samples inside the first dash and a spike of 8 in the first gap.
    let first_on = env.iter().position(|&x| x > 0.5).unwrap();
    for x in &mut env[first_on + 480 * 11 + 100..first_on + 480 * 11 + 108] {
        *x = 0.0;
    }
    for x in &mut env[first_on + 480 + 100..first_on + 480 + 108] {
        *x = 1.0;
    }
    assert_eq!(
        texts_of(&words(&env, &MorseDecodeConfig::default(), FS, None)),
        want
    );
    assert_eq!(want, ["VVV", "TEST"]);
}

#[test]
fn a_long_word_is_split_at_256_characters() {
    // 300 characters without a space: a packet holds at most 256 bytes, so the word goes through
    // the encoder of `sdrk-proto` in one piece and the decoder cuts it.
    let mut enc = Encoder::new(40.0, 40.0).unwrap();
    let mut units = Vec::new();
    for _ in 0..300 {
        enc.push(b'E').unwrap();
        while let Some(k) = enc.next_unit() {
            units.push(k);
        }
    }
    units.extend(std::iter::repeat_n(false, 12));
    let env = expand(&units, FS as f64 * 1.2 / 40.0);
    let dcfg = MorseDecodeConfig {
        wpm: 40.0,
        wpm_max: 60.0,
        ..MorseDecodeConfig::default()
    };
    let got = words(&env, &dcfg, FS, None);
    assert_eq!(got.len(), 2);
    assert_eq!((got[0].len(), got[1].len()), (256, 44));
    assert!(got.iter().all(|p| p.data().iter().all(|&c| c == b'E')));
    // The second packet starts at the 257th character.
    assert!(got[1].index > got[0].index);
}

#[test]
fn silence_and_the_extremes() {
    let cfg = MorseDecodeConfig::default();
    assert!(words(&vec![0.0; 100_000], &cfg, FS, None).is_empty());
    assert!(words(&vec![0.7; 100_000], &cfg, FS, None).is_empty());
    // NaN and infinity count as 0 and do not stay in the smoother.
    let mut env = encode(&[b"VVV CQ"], &hard(), FS);
    let want = texts_of(&words(&env, &cfg, FS, None));
    env[3] = f32::NAN;
    env[10] = f32::INFINITY;
    env[11] = f32::NEG_INFINITY;
    assert_eq!(texts_of(&words(&env, &cfg, FS, None)), want);
    // Full scale values.
    let big: Vec<f32> = env.iter().map(|&x| x * 3.0e38).collect();
    assert_eq!(texts_of(&words(&big, &cfg, FS, None)), want);
}

#[test]
fn the_output_room_is_respected() {
    let env = lead(&encode(&[b"E E E E"], &hard(), FS));
    let mut dec = decoder(&MorseDecodeConfig::default(), FS);
    // No room: nothing is consumed.
    let p = dec.process(&env, &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    // One slot: the call stops after the sample that finished a word and the rest is resubmitted.
    let mut got = Vec::new();
    let mut pos = 0;
    let mut slot = [Packet::default(); 1];
    while pos < env.len() {
        let p = dec.process(&env[pos..], &mut slot).unwrap();
        pos += p.consumed;
        if p.produced == 1 {
            got.push(slot[0]);
        }
        assert!(p.consumed > 0);
    }
    assert_eq!(texts_of(&got), ["E", "E", "E", "E"]);
    assert_eq!(dec.position(), env.len() as u64);
}

#[test]
fn the_decoder_reports_its_speed_and_counters() {
    let env = lead(&encode(
        &[b"VVV CQ CQ DE W1AW"],
        &MorseEncodeConfig {
            wpm: 12.0,
            ..hard()
        },
        FS,
    ));
    // The hint is far from the speed; the estimate settles on it.
    let mut dec = decoder(&MorseDecodeConfig::default(), FS);
    let mut room = vec![Packet::default(); 64];
    dec.process(&env, &mut room).unwrap();
    assert!((dec.wpm() - 12.0).abs() < 0.8, "{}", dec.wpm());
    assert_eq!(dec.overflow(), 0);
    assert_eq!(dec.position(), env.len() as u64);
    // A reset gives the state of a new block.
    dec.reset();
    assert_eq!(dec.position(), 0);
    assert_eq!(dec.wpm(), 20.0);
    let fresh = run_decoder(
        &mut decoder(&MorseDecodeConfig::default(), FS),
        &env,
        None,
        None,
    );
    let again = run_decoder(&mut dec, &env, None, None);
    same_bits(&again, &fresh, "after reset");
}

#[test]
fn the_decoder_parameters_and_timing() {
    let dec = decoder(&MorseDecodeConfig::default(), FS);
    assert_eq!(dec.description().kind, "morse_decode");
    assert!(matches!(dec.timing().kind, RatioKind::Packets { .. }));
    assert_eq!(dec.timing().delay, Delay::ZERO);
    assert_eq!(dec.param("wpm").unwrap(), ParamValue::F64(20.0));
    assert_eq!(dec.param("wpm_max").unwrap(), ParamValue::F64(40.0));
    assert_eq!(dec.param("threshold").unwrap(), ParamValue::F64(0.0));
    let mut dec = dec;
    assert_eq!(
        dec.set_param("wpm", ParamValue::F64(25.0)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(dec.param("nope"), Err(Error::UnknownParameter));
    check_exact_storage(
        MorseDecodeBlock::<ArenaStorage<'static>>::storage_bytes(
            &MorseDecodeConfig::default(),
            rate(FS),
        )
        .unwrap(),
        |arena| MorseDecodeBlock::new(&MorseDecodeConfig::default(), rate(FS), arena).map(|_| ()),
    );
    for bad in [
        MorseDecodeConfig {
            wpm: 4.0,
            ..MorseDecodeConfig::default()
        },
        MorseDecodeConfig {
            wpm: 61.0,
            ..MorseDecodeConfig::default()
        },
        MorseDecodeConfig {
            wpm_max: 19.0,
            ..MorseDecodeConfig::default()
        },
        MorseDecodeConfig {
            wpm_max: 101.0,
            ..MorseDecodeConfig::default()
        },
        MorseDecodeConfig {
            threshold: -0.1,
            ..MorseDecodeConfig::default()
        },
        MorseDecodeConfig {
            wpm: f64::NAN,
            ..MorseDecodeConfig::default()
        },
    ] {
        assert!(
            matches!(bad.validate(rate(FS)), Err(Error::OutOfRange { .. })),
            "{bad:?}"
        );
    }
    // 40 wpm needs 8 samples per unit: 267 Hz.
    assert!(MorseDecodeConfig::default().validate(rate(266)).is_err());
    assert!(MorseDecodeConfig::default().validate(rate(267)).is_ok());
}

/// Key states per sample for `units`, unit `k` lasting `unit(k)` samples (the fractions are
/// carried so that the total stays exact).
fn expand_with(units: &[bool], mut unit: impl FnMut(usize) -> f64) -> Vec<f32> {
    let mut out = Vec::new();
    let mut t = 0.0f64;
    for (k, &on) in units.iter().enumerate() {
        t += unit(k);
        let end = (t + 0.5) as usize;
        while out.len() < end {
            out.push(if on { 1.0 } else { 0.0 });
        }
    }
    out
}

/// The units of `text` followed by 14 units of key up.
fn text_units(text: &[u8]) -> Vec<bool> {
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    let mut out = Vec::new();
    for &c in text {
        enc.push(c).unwrap();
        while let Some(k) = enc.next_unit() {
            out.push(k);
        }
    }
    out.extend(std::iter::repeat_n(false, 14));
    out
}

fn add_noise(env: &mut [f32], sigma: f64, seed: u64) {
    let mut rng = Pcg32::new(seed, 3);
    for x in env {
        *x += (sigma * rng.normal()) as f32;
    }
}

#[test]
fn a_speed_drift_of_twenty_percent_in_noise_is_followed() {
    // M47 test 4: the unit swings by -20 % to +20 % once over the text; noise of 0.1 of the
    // key down level (20 dB below it).
    let text: &[u8] = b"VVV THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890 K";
    let units = text_units(text);
    let n = units.len() as f64;
    let mut env = expand_with(&units, |k| {
        480.0 * (1.0 + 0.2 * (std::f64::consts::TAU * k as f64 / n).sin())
    });
    add_noise(&mut env, 0.1, 653);
    let got = texts_of(&words(&env, &MorseDecodeConfig::default(), FS, None));
    let want: Vec<String> = String::from_utf8(text.to_vec())
        .unwrap()
        .split(' ')
        .map(String::from)
        .collect();
    assert_eq!(got, want);
}

#[test]
fn a_step_of_thirty_percent_at_an_idle_gap_is_learned() {
    // M47 test 4: a second sender 30 % slower after more than 20 word gaps of silence.
    let first = text_units(b"VVV CQ DE W1AW K");
    let second = text_units(b"VVV QRZ DE DL1ABC K");
    let mut units = first.clone();
    units.extend(std::iter::repeat_n(false, 200));
    let mid = units.len();
    units.extend(second);
    let mut env = expand_with(&units, |k| if k < mid { 480.0 } else { 480.0 * 1.3 });
    add_noise(&mut env, 0.05, 654);
    let got = texts_of(&words(&env, &MorseDecodeConfig::default(), FS, None));
    assert_eq!(
        got,
        [
            "VVV", "CQ", "DE", "W1AW", "K", "VVV", "QRZ", "DE", "DL1ABC", "K"
        ]
    );
}

#[test]
fn sloppy_keying_in_noise_is_decoded() {
    // M47 test 6, the hand sender: every key-down unit +-15 %, every key-up unit +-25 %
    // (seed 655), noise of 0.1 of the key down level; the character error rate is at most 2 %.
    let text: &[u8] = b"VVV CQ CQ DE DL1ABC PSE K THE QUICK BROWN FOX 73 DE DL1ABC K";
    let units = text_units(text);
    let mut rng = Pcg32::new(655, 1);
    let mut env = expand_with(&units, |k| {
        let a = if units[k] { 0.15 } else { 0.25 };
        480.0 * (1.0 + a * (2.0 * rng.next_f64() - 1.0))
    });
    add_noise(&mut env, 0.1, 656);
    let got = texts_of(&words(&env, &MorseDecodeConfig::default(), FS, None)).join(" ");
    let want = String::from_utf8(text.to_vec()).unwrap();
    let errors = edit_distance(want.as_bytes(), got.as_bytes());
    assert!(
        errors as f64 / want.len() as f64 <= 0.02,
        "{errors} errors: {got:?}"
    );
}

/// The edit distance (insertions, deletions and substitutions) of two byte strings.
fn edit_distance(a: &[u8], b: &[u8]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &x) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, &y) in b.iter().enumerate() {
            cur.push(
                (prev[j] + usize::from(x != y))
                    .min(prev[j + 1] + 1)
                    .min(cur[j] + 1),
            );
        }
        prev = cur;
    }
    prev[b.len()]
}
