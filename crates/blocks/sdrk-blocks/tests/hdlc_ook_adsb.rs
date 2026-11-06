//! The blocks of M39.1: `HdlcEncodeBlock` and `HdlcDecodeBlock` (the round trip of frames with
//! stuffing corner cases, NRZI and its polarity, idle flags, `emit_bad`, the packet index,
//! resumed frames, full outputs), `OokModBlock` (an exact burst, timing, storage),
//! `OokDemodBlock` (the loop back from `ook_mod` with a carrier offset, the level trackers,
//! `rate_unsupported` below 4 samples per symbol, the mean trackers and the swing
//! normalization) and `AdsbDecodeBlock` (a DF 17 frame built in the test, a damaged one
//! dropped, address/parity frames with flag bit 16 only for a cached address, the accepted
//! formats, no packets from noise, a frame sampled off phase, frames back to back, a full
//! output, the 2 MHz rule); `hdlc_decode`, `ook_demod` and `adsb_decode` are chunk invariant
//! with the 20 plans of seed 546 (test 6).
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
    clippy::many_single_char_names,
    clippy::trivially_copy_pass_by_ref,
    clippy::large_stack_frames,
    clippy::assert_is_empty
)]

mod common;

use common::{check_exact_storage, rate, run_block, same_bits, twenty_plans};
use sdrk::{C32, Delay, Ratio, RatioKind};
use sdrk_blocks::{
    ADSB_ADDRESS_PARITY, AdsbDecodeBlock, AdsbDecodeConfig, ArenaStorage, Block, Error,
    HdlcDecodeBlock, HdlcDecodeConfig, HdlcEncodeBlock, HdlcEncodeConfig, OOK_LOW_RATIO,
    OOK_THRESHOLD, OokDemodBlock, OokDemodConfig, OokModBlock, OokModConfig, Packet, PacketFlags,
    Packets, ParamValue, StreamElem, TED_GAIN_OOK,
};
use sdrk_proto::crc::crc24_adsb;
use sdrk_proto::hdlc;
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

/// Seed of M39.1 test 6 (seed table of docs/milestones/README.md).
const SEED: u64 = 546;

type Enc = HdlcEncodeBlock<VecStorage>;
type Dec = HdlcDecodeBlock<VecStorage>;
type Adsb = AdsbDecodeBlock<VecStorage>;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn the_new_types_are_send_and_sync() {
    assert_send::<Enc>();
    assert_send::<Dec>();
    assert_send::<OokModBlock<VecStorage>>();
    assert_send::<OokDemodBlock<VecStorage>>();
    assert_send::<Adsb>();
    assert_send::<HdlcEncodeConfig>();
    assert_send::<HdlcDecodeConfig>();
    assert_send::<OokModConfig>();
    assert_send::<OokDemodConfig>();
    assert_send::<AdsbDecodeConfig>();
    assert_sync::<Enc>();
    assert_sync::<Dec>();
    assert_sync::<OokModBlock<VecStorage>>();
    assert_sync::<OokDemodBlock<VecStorage>>();
    assert_sync::<Adsb>();
}

// ------------------------------------------------------------------------------ HDLC

fn encoder(cfg: &HdlcEncodeConfig) -> Enc {
    HdlcEncodeBlock::new(cfg, rate(1200), &mut VecStorage).unwrap()
}

fn decoder(cfg: &HdlcDecodeConfig) -> Dec {
    HdlcDecodeBlock::new(cfg, rate(1200), &mut VecStorage).unwrap()
}

/// Packets with the stuffing corner cases and random payloads of 2 to 256 bytes.
fn test_packets(rng: &mut Pcg32, random: usize) -> Vec<Packet> {
    let mut out = vec![
        Packet::new(0, PacketFlags::NONE, &[0xFF; 40]).unwrap(),
        Packet::new(0, PacketFlags::NONE, &[0x7E; 17]).unwrap(),
        Packet::new(0, PacketFlags::NONE, &[0x7C, 0x7C, 0x3E, 0x1F, 0xF8]).unwrap(),
        Packet::new(0, PacketFlags::NONE, &[0x00, 0x00]).unwrap(),
        Packet::new(0, PacketFlags::NONE, &[0xA5; 256]).unwrap(),
    ];
    for _ in 0..random {
        let len = 2 + rng.below(255) as usize;
        let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
        out.push(Packet::new(0, PacketFlags::NONE, &data).unwrap());
    }
    out
}

/// Runs the encoder over `packets` with packet and output pieces from the plans (`None`: all
/// at once, output room 8192) and returns every sample.
fn run_encoder(
    enc: &mut Enc,
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

/// Number of bits of the frame of `p` (`preamble_flags`, `tail_flags`).
fn frame_bits(p: &Packet, pre: usize, tail: usize) -> usize {
    let mut bits = vec![0u8; hdlc::max_bits(p.len(), pre, tail)];
    hdlc::encode(p, pre, tail, &mut bits).unwrap()
}

#[test]
fn hdlc_round_trip_with_stuffing_corner_cases_and_indices() {
    let mut rng = Pcg32::new(SEED, 1);
    let packets = test_packets(&mut rng, 20);
    for nrzi in [true, false] {
        let enc_cfg = HdlcEncodeConfig {
            nrzi,
            ..HdlcEncodeConfig::default()
        };
        let soft = run_encoder(&mut encoder(&enc_cfg), &packets, None, None);
        assert!(soft.iter().all(|&x| x == 1.0 || x == -1.0));
        let lens: Vec<usize> = packets.iter().map(|p| frame_bits(p, 32, 4)).collect();
        assert_eq!(soft.len(), lens.iter().sum::<usize>());
        let dec_cfg = HdlcDecodeConfig {
            nrzi,
            ..HdlcDecodeConfig::default()
        };
        let got = run_decoder(&mut decoder(&dec_cfg), &soft, None, None);
        assert_eq!(got.len(), packets.len(), "nrzi {nrzi}");
        let mut start = 0usize;
        for ((g, p), len) in got.iter().zip(&packets).zip(&lens) {
            assert_eq!(g.data(), p.data());
            assert_eq!(g.flags, PacketFlags::CRC_OK);
            // The first bit after the last of the 32 opening flags.
            assert_eq!(g.index, (start + 32 * 8) as u64);
            start += len;
        }
        // With NRZI the polarity of the line does not matter.
        let inverted: Vec<f32> = soft.iter().map(|x| -x).collect();
        let got = run_decoder(&mut decoder(&dec_cfg), &inverted, None, None);
        if nrzi {
            assert_eq!(got.len(), packets.len());
        } else {
            assert!(got.is_empty(), "inverted plain bits hold no frame");
        }
    }
}

#[test]
fn hdlc_encoder_writes_the_bits_of_sdrk_proto_with_nrzi() {
    let p = Packet::new(0, PacketFlags::NONE, &[0xFF, 0x7E, 0x00]).unwrap();
    let cfg = HdlcEncodeConfig {
        nrzi: true,
        preamble_flags: 2,
        tail_flags: 1,
    };
    let soft = run_encoder(&mut encoder(&cfg), &[p], None, None);
    let n = frame_bits(&p, 2, 1);
    let mut bits = vec![0u8; n];
    hdlc::encode(&p, 2, 1, &mut bits).unwrap();
    // NRZI from the level -1.0: a 0 toggles, a 1 keeps.
    let mut level = false;
    let want: Vec<f32> = bits
        .iter()
        .map(|&b| {
            level = sdrk_proto::nrzi::encode(level, b == 1);
            if level { 1.0 } else { -1.0 }
        })
        .collect();
    assert_eq!(soft, want);
}

#[test]
fn hdlc_encoder_resumes_frames_and_idles_with_flags() {
    let mut rng = Pcg32::new(SEED, 2);
    let packets = test_packets(&mut rng, 6);
    let cfg = HdlcEncodeConfig::default();
    let whole = run_encoder(&mut encoder(&cfg), &packets, None, None);
    for (k, max) in [1usize, 2, 7, 8, 9, 100, 333, 2000].iter().enumerate() {
        let got = run_encoder(
            &mut encoder(&cfg),
            &packets,
            Some(ChunkPlan::new(SEED, 1 + k % 3)),
            Some(ChunkPlan::new(SEED + 1, *max)),
        );
        same_bits(&whole, &got, "hdlc_encode pieces");
    }
    // Idle calls between frames send flags; the decoder still finds every frame and every
    // idle stretch is whole flags (NRZI decoded: 01111110 repeated).
    let mut enc = encoder(&HdlcEncodeConfig { nrzi: false, ..cfg });
    let mut stream = Vec::new();
    let mut room = vec![0.0f32; 4096];
    let mut idle = vec![0.0f32; 24];
    enc.process(&[], &mut idle).unwrap();
    let flag: Vec<f32> = (0..8)
        .map(|k| if (0x7E >> k) & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    for k in 0..24 {
        assert_eq!(idle[k], flag[k % 8], "idle bit {k}");
    }
    stream.extend_from_slice(&idle);
    for p in &packets[..4] {
        let r = enc.process(std::slice::from_ref(p), &mut room).unwrap();
        assert_eq!(r.consumed, 1);
        stream.extend_from_slice(&room[..r.produced]);
        let r = enc.process(&[], &mut idle[..13]).unwrap();
        assert_eq!(r.produced, 13);
        stream.extend_from_slice(&idle[..13]);
    }
    let got = run_decoder(
        &mut decoder(&HdlcDecodeConfig {
            nrzi: false,
            ..HdlcDecodeConfig::default()
        }),
        &stream,
        None,
        None,
    );
    assert_eq!(got.len(), 4);
    for (g, p) in got.iter().zip(&packets) {
        assert_eq!(g.data(), p.data());
    }
}

#[test]
fn hdlc_encoder_rejects_short_packets() {
    let short = Packet::new(0, PacketFlags::NONE, &[1]).unwrap();
    let good = Packet::new(0, PacketFlags::NONE, &[1, 2]).unwrap();
    let mut enc = encoder(&HdlcEncodeConfig::default());
    let mut out = vec![0.0f32; 4096];
    assert!(matches!(
        enc.process(&[short], &mut out),
        Err(Error::Proto(_))
    ));
    // After a good packet the call stops before the short one.
    let p = enc.process(&[good, short], &mut out).unwrap();
    assert_eq!(p.consumed, 1);
    assert_eq!(p.produced, frame_bits(&good, 32, 4));
    // A packet of 256 bytes is fine.
    let long = Packet::new(0, PacketFlags::NONE, &[0xFF; 256]).unwrap();
    let p = enc.process(&[long], &mut out).unwrap();
    assert_eq!(p.consumed, 1);
}

#[test]
fn hdlc_decoder_emit_bad_and_a_full_output() {
    // Plain bits, payload of zeros: a flipped payload bit cannot form a flag.
    let p = Packet::new(0, PacketFlags::NONE, &[0u8; 8]).unwrap();
    let cfg = HdlcEncodeConfig {
        nrzi: false,
        preamble_flags: 4,
        tail_flags: 2,
    };
    let mut soft = run_encoder(&mut encoder(&cfg), &[p], None, None);
    soft[4 * 8 + 20] = 1.0;
    let plain = HdlcDecodeConfig {
        nrzi: false,
        emit_bad: false,
    };
    assert!(run_decoder(&mut decoder(&plain), &soft, None, None).is_empty());
    let bad = run_decoder(
        &mut decoder(&HdlcDecodeConfig {
            emit_bad: true,
            ..plain
        }),
        &soft,
        None,
        None,
    );
    assert_eq!(bad.len(), 1);
    assert_eq!(bad[0].flags, PacketFlags::CRC_FAIL);
    assert_eq!(bad[0].index, 32);
    // A NaN soft bit reads as a 0.
    let mut clean = run_encoder(&mut encoder(&cfg), &[p, p], None, None);
    let mut dec = decoder(&plain);
    let mut one = [Packet::default(); 1];
    let r = dec.process(&clean, &mut one).unwrap();
    assert_eq!(r.produced, 1);
    // The call stops right after the bit that completed the first packet.
    assert_eq!(r.consumed, frame_bits(&p, 4, 2) - 8);
    assert_eq!(dec.position(), r.consumed as u64);
    let r2 = dec.process(&clean[r.consumed..], &mut one).unwrap();
    assert_eq!((r2.consumed, r2.produced), (frame_bits(&p, 4, 2), 1));
    assert_eq!(one[0].index, (frame_bits(&p, 4, 2) + 32) as u64);
    // No room: nothing is consumed.
    let r3 = dec.process(&clean, &mut []).unwrap();
    assert_eq!((r3.consumed, r3.produced), (0, 0));
    clean[0] = f32::NAN; // the first flag bit is a 0 anyway
    assert_eq!(
        run_decoder(&mut decoder(&plain), &clean, None, None).len(),
        2
    );
    // Reset starts the index at 0 again.
    dec.reset();
    assert_eq!(dec.position(), 0);
}

#[test]
fn hdlc_timing_parameters_and_storage() {
    let enc = encoder(&HdlcEncodeConfig::default());
    let t = enc.timing();
    assert_eq!(t.kind, RatioKind::Source);
    assert_eq!(
        t.ratio,
        Ratio::new(hdlc::max_bits(256, 32, 4) as u64, 1).unwrap()
    );
    assert_eq!(t.delay, Delay::ZERO);
    let dec = decoder(&HdlcDecodeConfig::default());
    let t = dec.timing();
    let max = Ratio::new(1, 40).unwrap();
    assert_eq!(t.kind, RatioKind::Packets { max_per_input: max });
    assert_eq!(t.ratio, max);
    assert_eq!(
        HdlcEncodeBlock::<ArenaStorage<'_>>::storage_bytes(
            &HdlcEncodeConfig::default(),
            rate(1200)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        HdlcDecodeBlock::<ArenaStorage<'_>>::storage_bytes(
            &HdlcDecodeConfig::default(),
            rate(1200)
        )
        .unwrap(),
        0
    );
    for (pre, tail, ok) in [
        (0, 4, false),
        (1, 1, true),
        (1024, 64, true),
        (1025, 4, false),
        (32, 65, false),
    ] {
        let cfg = HdlcEncodeConfig {
            preamble_flags: pre,
            tail_flags: tail,
            ..HdlcEncodeConfig::default()
        };
        assert_eq!(cfg.validate().is_ok(), ok, "{pre} {tail}");
    }
    let mut dec = decoder(&HdlcDecodeConfig::default());
    assert_eq!(dec.param("nrzi").unwrap(), ParamValue::Bool(true));
    assert!(matches!(
        dec.set_param("emit_bad", ParamValue::Bool(true)),
        Err(Error::ReadOnlyParameter)
    ));
    let mut enc = encoder(&HdlcEncodeConfig::default());
    assert_eq!(enc.param("preamble_flags").unwrap(), ParamValue::I64(32));
    assert!(matches!(
        enc.set_param("tail_flags", ParamValue::I64(3)),
        Err(Error::ReadOnlyParameter)
    ));
}

// ------------------------------------------------------------------------------- OOK

fn ook_mod(sps: usize, level: f64, bit_rate: u64) -> OokModBlock<VecStorage> {
    let cfg = OokModConfig {
        samples_per_symbol: sps,
        level,
    };
    OokModBlock::new(&cfg, rate(bit_rate), &mut VecStorage).unwrap()
}

#[test]
fn ook_mod_writes_an_exact_burst() {
    let bits = [1.0f32, 1.0, -1.0, 0.0, 1.0, f32::NAN, 0.3];
    for sps in [1usize, 3, 8, 300] {
        let out = run_block(&mut ook_mod(sps, 0.7, 1000), &bits, None);
        assert_eq!(out.len(), bits.len() * sps);
        for (k, s) in out.iter().enumerate() {
            let want = if bits[k / sps] > 0.0 { 0.7f32 } else { 0.0 };
            assert_eq!(*s, C32::new(want, 0.0), "sps {sps} sample {k}");
        }
    }
    // A call takes whole symbols only.
    let mut m = ook_mod(4, 0.9, 1000);
    let mut out = vec![C32::default(); 10];
    let p = m.process(&[1.0, 1.0, 1.0], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 8));
}

#[test]
fn ook_mod_timing_rates_and_storage() {
    let m = ook_mod(500, 0.9, 2000);
    let t = m.timing();
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.ratio, Ratio::new(500, 1).unwrap());
    assert_eq!(t.delay, Delay::new(499, 1000).unwrap());
    assert_eq!(
        OokModBlock::<VecStorage>::rate_out(
            &OokModConfig {
                samples_per_symbol: 500,
                level: 0.9
            },
            rate(2000)
        )
        .unwrap(),
        rate(1_000_000)
    );
    for sps in [1usize, 8, 255, 256, 257, 500, 4096] {
        let cfg = OokModConfig {
            samples_per_symbol: sps,
            level: 0.9,
        };
        let bytes = OokModBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate(1000)).unwrap();
        check_exact_storage(bytes, |a| OokModBlock::new(&cfg, rate(1000), a).map(|_| ()));
    }
    for (sps, level, ok) in [
        (0usize, 0.9, false),
        (4097, 0.9, false),
        (8, -0.1, false),
        (8, 1.0, true),
        (8, 0.0, true),
    ] {
        let cfg = OokModConfig {
            samples_per_symbol: sps,
            level,
        };
        assert_eq!(cfg.validate().is_ok(), ok, "{sps} {level}");
    }
}

/// `ook_mod` at `sps` samples per bit, turned by `cycles` per sample, plus complex noise of
/// standard deviation `sigma` per component.
fn ook_signal(bits: &[f32], sps: usize, cycles: f64, sigma: f64, stream: u64) -> Vec<C32> {
    let clean = run_block(&mut ook_mod(sps, 0.8, 1000), bits, None);
    let mut rng = Pcg32::new(SEED, stream);
    clean
        .iter()
        .enumerate()
        .map(|(n, s)| {
            let ph = 2.0 * std::f64::consts::PI * cycles * n as f64;
            let (si, co) = ph.sin_cos();
            let re = f64::from(s.re) * co + sigma * rng.normal();
            let im = f64::from(s.re) * si + sigma * rng.normal();
            C32::new(re as f32, im as f32)
        })
        .collect()
}

/// Random soft bits after 64 alternating training bits.
fn ook_bits(n: usize, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|k| {
            let one = if k < 64 {
                k % 2 == 0
            } else {
                rng.next_u32() & 1 == 1
            };
            if one { 1.0 } else { -1.0 }
        })
        .collect()
}

fn ook_demod(symbol_rate_hz: f64, fs: u64) -> OokDemodBlock<VecStorage> {
    let cfg = OokDemodConfig {
        symbol_rate_hz,
        ..OokDemodConfig::default()
    };
    OokDemodBlock::new(&cfg, rate(fs), &mut VecStorage).unwrap()
}

/// The shift of `got` against `want` (from bit `from` on) with the fewest differing signs, and
/// that number.
fn best_alignment(want: &[f32], got: &[f32], from: usize) -> (usize, usize) {
    (0..8)
        .map(|shift| {
            let errors = want[from..]
                .iter()
                .zip(got.iter().skip(from + shift))
                .filter(|(w, g)| (**w > 0.0) != (**g > 0.0))
                .count();
            (shift, errors)
        })
        .min_by_key(|&(_, e)| e)
        .unwrap()
}

#[test]
fn ook_demod_recovers_the_bits_of_ook_mod_with_a_carrier_offset() {
    for (sps, symbol_rate) in [(8usize, 1000.0), (10, 1010.0), (25, 400.0)] {
        let bits = ook_bits(3000, 3);
        let fs = (symbol_rate * sps as f64).round() as u64;
        // The signal is made at exactly sps samples per bit; the demodulator expects
        // fs / symbol_rate (1 % off for 1010 Bd at 10.1 kHz is a clock offset).
        let x = ook_signal(&bits, sps, 0.013, 0.03, 4);
        let mut dem = ook_demod(symbol_rate, fs);
        let soft = run_block(&mut dem, &x, None);
        assert!(soft.len() + 10 >= bits.len() && soft.len() <= bits.len() + 10);
        let (_, errors) = best_alignment(&bits, &soft, 200);
        assert_eq!(errors, 0, "sps {sps}");
        // The trackers follow the on level 0.8 and the off level 0.
        let (hi, lo) = dem.levels();
        assert!((hi - 0.8).abs() < 0.1 && lo.abs() < 0.1, "hi {hi} lo {lo}");
        assert!((dem.sps_estimate() - fs as f64 / symbol_rate).abs() < 0.2);
    }
}

#[test]
fn ook_demod_soft_bits_follow_the_threshold_rule() {
    // Without noise and with the loop locked, alternating symbols 0.8 and 0 give the mean
    // levels hi = 0.8 and lo = 0, the threshold lo + OOK_THRESHOLD (hi - lo) = 0.36 and soft
    // bits of 0.44 and -0.36 times the gain 1 / 0.8: 0.55 and -0.45 (D720, D727).
    let bits: Vec<f32> = (0..2000)
        .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let x = ook_signal(&bits, 8, 0.0, 0.0, 5);
    let mut dem = ook_demod(1000.0, 8000);
    let soft = run_block(&mut dem, &x, None);
    for v in &soft[1500..1990] {
        let want = if *v > 0.0 {
            1.0 - OOK_THRESHOLD
        } else {
            -OOK_THRESHOLD
        };
        assert!((v - want).abs() < 0.02, "{v}");
    }
    assert_eq!(OOK_THRESHOLD, 0.45);
    let (hi, lo) = dem.levels();
    assert!((hi - 0.8).abs() < 0.01 && lo.abs() < 0.01, "{hi} {lo}");
    assert!((dem.swing() - 0.8).abs() < 0.02, "{}", dem.swing());
    let cfg = OokDemodConfig::default();
    assert!((cfg.low_step() - (1.0 - (-1.0 / (16.0 * 16.0f64)).exp())).abs() < 1e-15);
    assert_eq!(OOK_LOW_RATIO, 16.0);
    // Silence: everything decays to 0 without denormals.
    let silent = vec![C32::default(); 40_000];
    let mut dem = ook_demod(1000.0, 8000);
    run_block(&mut dem, &x[..4000], None);
    let tail = run_block(&mut dem, &silent, None);
    assert!(tail.iter().all(|v| v.is_finite()));
    let (hi, lo) = dem.levels();
    // Both levels decay with the slow step of the off level (the silence is a long run of
    // zeros), flushed below 1e-30.
    let tiny = |v: f32| v.abs() < 1e-6 && (v == 0.0 || v.abs() > 1e-30);
    assert!(tiny(hi) && tiny(lo), "{hi} {lo}");
    // NaN input gives NaN soft bits but leaves the trackers usable.
    let mut nan = x[..800].to_vec();
    nan[400] = C32::new(f32::NAN, 0.0);
    let mut dem = ook_demod(1000.0, 8000);
    run_block(&mut dem, &nan, None);
    let (hi, lo) = dem.levels();
    assert!(hi.is_finite() && lo.is_finite());
}

#[test]
fn ook_demod_trackers_follow_the_means_and_normalize_the_swing() {
    // In noise the trackers sit at the mean levels of the symbols, not at their extremes:
    // the on level 0.8 (the noise adds about sigma^2 / (2 0.8) to the magnitude) and the off
    // level, the mean magnitude of the noise, sigma sqrt(pi / 2) after the boxcar.
    let bits = ook_bits(4000, 30);
    let sigma = 0.1;
    let x = ook_signal(&bits, 10, 0.01, sigma, 31);
    let mut dem = ook_demod(1000.0, 10_000);
    let soft = run_block(&mut dem, &x, None);
    let (_, errors) = best_alignment(&bits, &soft, 400);
    assert!(errors <= 2, "{errors}");
    let (hi, lo) = dem.levels();
    let off = sigma * (std::f64::consts::PI / 2.0).sqrt();
    assert!((f64::from(hi) - 0.8).abs() < 0.03, "hi {hi}");
    assert!((f64::from(lo) - off).abs() < 0.03, "lo {lo} against {off}");
    // A tenth of the input level gives the same decisions: the boxcar output is divided by the
    // tracked swing before the timing loop, so the loop runs at the same gain.
    let small: Vec<C32> = x.iter().map(|v| *v * 0.1).collect();
    let mut dem_small = ook_demod(1000.0, 10_000);
    let soft_small = run_block(&mut dem_small, &small, None);
    let differ = soft
        .iter()
        .zip(&soft_small)
        .skip(400)
        .filter(|(a, b)| (**a > 0.0) != (**b > 0.0))
        .count();
    assert!(differ <= 2, "{differ}");
    let ratio = dem_small.swing() / dem.swing();
    assert!((ratio - 0.1).abs() < 0.002, "{ratio}");
}

#[test]
fn ook_demod_trackers_recover_when_the_off_level_rises_above_the_threshold() {
    // 1000 alternating symbols 0.8 / 0, then the off level steps to 0.5 (above the
    // threshold 0.36): every symbol is a 1 at first, and after more than track_symbols ones in
    // a row the off level follows them slowly, until the threshold lies between 0.5 and 0.8
    // again.
    let sps = 8;
    let x: Vec<C32> = (0..6000 * sps)
        .map(|n| {
            let k = n / sps;
            let a = match (k % 2 == 0, k < 1000) {
                (true, _) => 0.8,
                (false, true) => 0.0,
                (false, false) => 0.5,
            };
            C32::new(a, 0.0)
        })
        .collect();
    let mut dem = ook_demod(1000.0, 8000);
    let soft = run_block(&mut dem, &x, None);
    let tail = &soft[5000..5900];
    assert!(tail.windows(2).all(|w| w[0] * w[1] < 0.0));
    let (hi, lo) = dem.levels();
    assert!(
        (hi - 0.8).abs() < 0.05 && (lo - 0.5).abs() < 0.05,
        "{hi} {lo}"
    );
}

/// The timing loop locks within the 32 alternating preamble bits of the test framing of
/// `frame_encode` at the rates of `ook_receiver` (2000 bit/s, 125 samples per symbol): a burst
/// after silence at 16 start phases over one symbol, every bit from the sync word on (bit 32
/// of the burst) is right (D727: with the loop bandwidth 0.01 the last wrong bit is bit 17 at
/// the worst phase; 0.003 leaves 21 wrong bits after the sync word at phase 0).
#[test]
#[allow(clippy::print_stderr)]
fn ook_demod_locks_within_the_preamble_of_the_test_framing() {
    let sps = 125;
    let mut rng = Pcg32::new(SEED, 40);
    let burst: Vec<f32> = (0..232)
        .map(|k| {
            let one = if k < 32 {
                k % 2 == 0
            } else {
                rng.next_u32() & 1 == 1
            };
            if one { 1.0 } else { -1.0 }
        })
        .collect();
    let clean = run_block(&mut ook_mod(sps, 0.8, 2000), &burst, None);
    let mut worst = 0;
    for phase in 0..16 {
        let lead = 4000 + phase * sps / 16;
        let mut x = vec![C32::default(); lead];
        x.extend(clean.iter().map(|s| C32::new(s.re * 0.6, s.re * 0.8)));
        x.extend(vec![C32::default(); 4000]);
        let soft = run_block(&mut ook_demod(2000.0, 250_000), &x, None);
        // The alignment with the fewest errors from the sync word on, then the first bit from
        // which every bit is right.
        let errors_at = |o: usize| {
            burst[32..]
                .iter()
                .zip(&soft[o + 32..])
                .filter(|(w, g)| (**w > 0.0) != (**g > 0.0))
                .count()
        };
        let base = lead / sps;
        let o = (base - 3..base + 4).min_by_key(|&o| errors_at(o)).unwrap();
        let lock = (0..burst.len())
            .rev()
            .find(|&k| (burst[k] > 0.0) != (soft[o + k] > 0.0))
            .map_or(0, |k| k + 1);
        worst = worst.max(lock);
        assert_eq!(errors_at(o), 0, "phase {phase}/16: lock at bit {lock}");
    }
    eprintln!(
        "ook_demod lock: every bit right from bit {worst} of the burst on (worst of 16 phases)"
    );
    assert!(worst <= 32, "{worst}");
}

#[test]
fn ook_demod_timing_rates_storage_and_checks() {
    let cfg = OokDemodConfig {
        symbol_rate_hz: 2000.0,
        ..OokDemodConfig::default()
    };
    let fs = rate(250_000);
    let d = OokDemodBlock::new(&cfg, fs, &mut VecStorage).unwrap();
    assert_eq!(d.num_taps(), 125);
    let t = d.timing();
    assert_eq!(t.kind, RatioKind::Variable);
    assert_eq!(t.ratio, Ratio::new(1, 125).unwrap());
    assert_eq!(t.delay, Delay::new(126, 2).unwrap());
    assert_eq!(cfg.rate_out(fs).unwrap(), rate(2000));
    assert_eq!(cfg.symbol_sync().ted_gain, TED_GAIN_OOK);
    for (symbol_rate, fs) in [
        (2000.0, 250_000u64),
        (1000.0, 4000),
        (1.0, 1024),
        (1010.0, 10_100),
    ] {
        let cfg = OokDemodConfig {
            symbol_rate_hz: symbol_rate,
            ..OokDemodConfig::default()
        };
        let bytes = OokDemodBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate(fs)).unwrap();
        check_exact_storage(bytes, |a| OokDemodBlock::new(&cfg, rate(fs), a).map(|_| ()));
    }
    // Fewer than 4 or more than 1024 samples per symbol.
    for (symbol_rate, fs) in [(1000.0, 3999u64), (1.0, 1025)] {
        let cfg = OokDemodConfig {
            symbol_rate_hz: symbol_rate,
            ..OokDemodConfig::default()
        };
        assert!(matches!(
            cfg.validate(rate(fs)),
            Err(Error::OutOfRange {
                name: "symbol_rate_hz",
                ..
            })
        ));
    }
    let bad = OokDemodConfig {
        track_symbols: 0,
        ..cfg
    };
    assert!(matches!(
        bad.validate(fs),
        Err(Error::OutOfRange {
            name: "track_symbols",
            ..
        })
    ));
    // Small outputs: the soft bits that do not fit wait in the block.
    let bits = ook_bits(400, 6);
    let x = ook_signal(&bits, 8, 0.0, 0.0, 7);
    let whole = run_block(&mut ook_demod(1000.0, 8000), &x, None);
    let mut dem = ook_demod(1000.0, 8000);
    let mut got = Vec::new();
    let mut pos = 0;
    let mut one = [0.0f32; 1];
    loop {
        let p = dem.process(&x[pos..], &mut one).unwrap();
        got.extend_from_slice(&one[..p.produced]);
        pos += p.consumed;
        if p.produced == 0 && pos == x.len() {
            break;
        }
    }
    same_bits(&whole, &got, "ook_demod one at a time");
}

// ------------------------------------------------------------------------------ ADS-B

/// The DF 17 frame of the Mode S literature (CRC residue 0).
const DF17: [u8; 14] = [
    0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98,
];

fn adsb() -> Adsb {
    AdsbDecodeBlock::new(
        &AdsbDecodeConfig::default(),
        rate(2_000_000),
        &mut VecStorage,
    )
    .unwrap()
}

/// Adds the pulses of `frame` (preamble and one pulse per bit, amplitude `a`, phase `phase`)
/// at sample `p` of `x`.
fn put_frame(x: &mut [C32], p: usize, frame: &[u8], a: f32, phase: f32) {
    let pulse = C32::new(a * phase.cos(), a * phase.sin());
    for k in [0, 2, 7, 9] {
        x[p + k] += pulse;
    }
    for k in 0..frame.len() * 8 {
        let bit = (frame[k / 8] >> (7 - k % 8)) & 1;
        x[p + 16 + 2 * k + usize::from(bit == 0)] += pulse;
    }
}

/// Complex noise of standard deviation `sigma` per component.
fn noise(n: usize, sigma: f64, stream: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(SEED, stream);
    (0..n)
        .map(|_| C32::new((sigma * rng.normal()) as f32, (sigma * rng.normal()) as f32))
        .collect()
}

/// A valid DF 17 frame with a random address and message.
fn random_df17(rng: &mut Pcg32) -> [u8; 14] {
    let mut f = [0u8; 14];
    f[0] = 0x8D;
    for b in &mut f[1..11] {
        *b = rng.next_u32() as u8;
    }
    let crc = crc24_adsb(&f[..11]);
    f[11] = (crc >> 16) as u8;
    f[12] = (crc >> 8) as u8;
    f[13] = crc as u8;
    assert_eq!(crc24_adsb(&f), 0);
    f
}

#[test]
fn adsb_decodes_a_df17_frame_in_noise() {
    let mut x = noise(4000, 0.01, 10);
    put_frame(&mut x, 1234, &DF17, 0.4, 0.7);
    let got = run_decoder(&mut adsb(), &x, None, None);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].data(), &DF17);
    assert_eq!(got[0].index, 1234);
    assert_eq!(got[0].flags, PacketFlags::CRC_OK);
    assert_eq!(sdrk_proto::adsb::icao(got[0].data()), 0x48_40D6);
}

/// `body` with its 24 parity bits: the CRC-24 of the body xor `overlay` (the address of an
/// address/parity frame, 0 for a checked frame).
fn with_parity(body: &[u8], overlay: u32) -> Vec<u8> {
    let p = crc24_adsb(body) ^ overlay;
    let mut f = body.to_vec();
    f.extend_from_slice(&p.to_be_bytes()[1..]);
    f
}

/// The frames `frames` 600 samples apart from sample 300 without noise (no other position
/// passes the preamble test), through a new decoder.
fn decode_frames(frames: &[&[u8]]) -> Vec<Packet> {
    let mut x = vec![C32::default(); 600 * frames.len() + 600];
    for (k, f) in frames.iter().enumerate() {
        put_frame(&mut x, 300 + 600 * k, f, 0.5, k as f32);
    }
    run_decoder(&mut adsb(), &x, None, None)
}

#[test]
fn adsb_drops_a_damaged_frame_and_checks_addresses_in_the_cache() {
    let mut bad = DF17;
    bad[6] ^= 0x04;
    assert!(decode_frames(&[&bad]).is_empty());
    // Address/parity frames of an address not seen before are dropped (D722).
    let addr = 0x48_40D6;
    let df4 = with_parity(&[0x20, 0x00, 0x17, 0x30], addr);
    let df11 = with_parity(&[0x5D, 0x48, 0x40, 0xD6], addr);
    let df20 = with_parity(&[0xA0, 0, 0x1F, 0x22, 1, 2, 3, 4, 5, 6, 7], addr);
    assert!(decode_frames(&[&df4, &df11, &df20]).is_empty());
    // After a DF 17 frame of that address they come out with flag bit 16, short and long.
    let got = decode_frames(&[&DF17, &df4, &df11, &df20]);
    let flags: Vec<(u64, PacketFlags, usize)> = got
        .iter()
        .map(|p| (p.index, p.flags, p.data().len()))
        .collect();
    assert_eq!(
        flags,
        [
            (300, PacketFlags::CRC_OK, 14),
            (900, ADSB_ADDRESS_PARITY, 7),
            (1500, ADSB_ADDRESS_PARITY, 7),
            (2100, ADSB_ADDRESS_PARITY, 14),
        ]
    );
    assert_eq!(got[1].data(), &df4[..]);
    assert_eq!(got[2].data(), &df11[..]);
    assert_eq!(got[3].data(), &df20[..]);
    assert_eq!(ADSB_ADDRESS_PARITY.bits(), 1 << 16);
    // A DF 11 with residue 0 (interrogator code 0) is checked: CRC_OK, and its address is
    // cached for the DF 4 behind it; another address stays unknown.
    let df11_ok = with_parity(&[0x5D, 0xAB, 0xCD, 0xEF], 0);
    let df4_known = with_parity(&[0x20, 0x00, 0x17, 0x30], 0xAB_CDEF);
    let df4_other = with_parity(&[0x20, 0x00, 0x17, 0x30], 0xAB_CDEE);
    let got = decode_frames(&[&df11_ok, &df4_known, &df4_other]);
    let flags: Vec<(u64, PacketFlags)> = got.iter().map(|p| (p.index, p.flags)).collect();
    assert_eq!(
        flags,
        [(300, PacketFlags::CRC_OK), (900, ADSB_ADDRESS_PARITY)]
    );
    // Formats outside DF 0, 4, 5, 11, 16, 17, 18, 20, 21, 24 are dropped, even with a cached
    // address: DF 19, 22 and 25 (long), DF 1 and 12 (short).
    let mut frames: Vec<Vec<u8>> = vec![DF17.to_vec()];
    for first in [0x98u8, 0xB0, 0xC8] {
        frames.push(with_parity(
            &[first, 0, 0x1F, 0x22, 1, 2, 3, 4, 5, 6, 7],
            addr,
        ));
    }
    for first in [0x08u8, 0x60] {
        frames.push(with_parity(&[first, 0x00, 0x17, 0x30], addr));
    }
    let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
    let got = decode_frames(&refs);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].data(), &DF17);
}

#[test]
fn adsb_noise_gives_no_packets() {
    // 400000 samples (0.2 s) of complex noise: preambles found in the noise give neither a
    // CRC-24 match nor a cached address.
    let x = noise(400_000, 0.05, 15);
    assert!(run_decoder(&mut adsb(), &x, None, None).is_empty());
}

#[test]
fn adsb_decodes_a_frame_sampled_a_quarter_chip_late() {
    // Every pulse split 0.7 / 0.3 over two neighbouring samples (sampling a quarter chip
    // late): the preamble test of dump1090 does not test the samples next to the pulses, so
    // the frame is found and decoded.
    let mut x = noise(3000, 0.002, 16);
    let pulse = |x: &mut [C32], k: usize| {
        x[k] += C32::new(0.35, 0.0);
        x[k + 1] += C32::new(0.15, 0.0);
    };
    let p = 1000;
    for k in [0, 2, 7, 9] {
        pulse(&mut x, p + k);
    }
    for k in 0..112 {
        let bit = (DF17[k / 8] >> (7 - k % 8)) & 1;
        pulse(&mut x, p + 16 + 2 * k + usize::from(bit == 0));
    }
    let got = run_decoder(&mut adsb(), &x, None, None);
    assert_eq!(got.len(), 1, "{got:?}");
    assert_eq!((got[0].index, got[0].data()), (1000, &DF17[..]));
}

#[test]
fn adsb_frames_back_to_back_a_full_output_and_the_noise_level() {
    let mut rng = Pcg32::new(SEED, 11);
    let frames: Vec<[u8; 14]> = (0..3).map(|_| random_df17(&mut rng)).collect();
    let mut x = noise(3000, 0.01, 12);
    // The second frame starts right behind the first (240 samples).
    for (k, f) in frames.iter().enumerate() {
        put_frame(&mut x, 1000 + 240 * k, f, 0.3, k as f32);
    }
    let whole = run_decoder(&mut adsb(), &x, None, None);
    assert_eq!(whole.len(), 3);
    for (k, (g, f)) in whole.iter().zip(&frames).enumerate() {
        assert_eq!(g.data(), f);
        assert_eq!(g.index, (1000 + 240 * k) as u64);
    }
    // One slot per call: the first call stops after the first packet; the next calls find the
    // rest without new input.
    let mut dec = adsb();
    let mut one = [Packet::default(); 1];
    let p = dec.process(&x, &mut one).unwrap();
    assert_eq!(p.produced, 1);
    assert!(p.consumed < x.len());
    let rest = &x[p.consumed..];
    assert_eq!(one[0].index, 1000);
    assert_eq!(dec.position(), 1001);
    for k in 1..3 {
        // The candidates already in the buffer come first: no new input is taken.
        let p = dec.process(rest, &mut one).unwrap();
        assert_eq!((p.consumed, p.produced), (0, 1));
        assert_eq!(one[0].index, (1000 + 240 * k) as u64);
    }
    let p = dec.process(rest, &mut one).unwrap();
    assert_eq!((p.consumed, p.produced), (rest.len(), 0));
    let p = dec.process(&[], &mut one).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    // The noise level is the median magnitude of the 256 samples before the next candidate:
    // for Gaussian noise sigma sqrt(2 ln 2), known to 1/32 octave.
    let level = f64::from(dec.noise_level());
    let want = 0.01 * (2.0 * 2f64.ln()).sqrt();
    assert!(
        level <= want * 1.15 && level >= want / 1.15 / 2f64.powf(1.0 / 32.0),
        "{level}"
    );
    // No room: nothing is consumed.
    let p = dec.process(&x, &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

#[test]
fn adsb_min_snr_db_rejects_weak_preambles() {
    // Pulses 0.05 over noise of median magnitude about 0.012: 12.5 dB.
    let mut x = noise(3000, 0.01, 13);
    put_frame(&mut x, 1000, &DF17, 0.05, 0.0);
    let mk = |db: f64| {
        AdsbDecodeBlock::new(
            &AdsbDecodeConfig { min_snr_db: db },
            rate(2_000_000),
            &mut VecStorage,
        )
        .unwrap()
    };
    let found = |db: f64| {
        run_decoder(&mut mk(db), &x, None, None)
            .iter()
            .any(|p| p.flags == PacketFlags::CRC_OK)
    };
    assert!(found(6.0));
    assert!(!found(20.0));
}

#[test]
fn adsb_min_snr_db_compares_with_the_exact_median() {
    // The window before p = 1000: 127 values 0.5 m, one value m, 128 values 1.5 m, so the
    // lower median is m exactly; m = 1.34 * 2^-7 lies 2 % (0.09 dB) above the lower edge of
    // its 1/32 octave bin. The frame pulses are at 0.5 % below or above 10^(6/20) m, every
    // other sample of the frame is 0; the steps of the window cannot pass the preamble test.
    let m = 0.010_468_75f32;
    let factor = 10f64.powf(6.0 / 20.0) as f32;
    let signal = |scale: f32| {
        let mut x = vec![C32::new(1.5 * m, 0.0); 1600];
        for v in &mut x[..1000 - 129] {
            *v = C32::new(0.5 * m, 0.0);
        }
        x[1000 - 129] = C32::new(m, 0.0);
        for v in &mut x[1000..] {
            *v = C32::default();
        }
        put_frame(&mut x, 1000, &DF17, factor * m * scale, 0.0);
        x
    };
    let below = run_decoder(&mut adsb(), &signal(0.995), None, None);
    assert!(below.is_empty(), "{below:?}");
    let above = run_decoder(&mut adsb(), &signal(1.005), None, None);
    assert_eq!(above.len(), 1);
    assert_eq!(above[0].data(), &DF17);
    assert_eq!(
        (above[0].index, above[0].flags),
        (1000, PacketFlags::CRC_OK)
    );
}

#[test]
fn adsb_accepts_only_2_mhz_and_its_storage_is_exact() {
    let cfg = AdsbDecodeConfig::default();
    for fs in [1_999_999u64, 2_400_000, 1_000_000] {
        assert!(matches!(
            AdsbDecodeBlock::<VecStorage>::new(&cfg, rate(fs), &mut VecStorage),
            Err(Error::Core(sdrk_core::Error::Unsupported(_)))
        ));
    }
    assert!(
        AdsbDecodeConfig { min_snr_db: 61.0 }
            .validate(rate(2_000_000))
            .is_err()
    );
    let bytes = AdsbDecodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate(2_000_000)).unwrap();
    check_exact_storage(bytes, |a| {
        AdsbDecodeBlock::new(&cfg, rate(2_000_000), a).map(|_| ())
    });
    let d = adsb();
    let t = d.timing();
    let max = Ratio::new(1, 128).unwrap();
    assert_eq!(t.kind, RatioKind::Packets { max_per_input: max });
    assert_eq!(d.param("min_snr_db").unwrap(), ParamValue::F64(6.0));
}

#[test]
fn adsb_survives_extreme_input() {
    let mut x = noise(3000, 0.01, 14);
    x[100] = C32::new(f32::NAN, 0.0);
    x[200] = C32::new(f32::INFINITY, 0.0);
    x[300] = C32::new(1e30, -1e30);
    put_frame(&mut x, 1000, &DF17, 0.4, 0.0);
    let got = run_decoder(&mut adsb(), &x, None, None);
    assert!(got.iter().any(|p| p.data() == DF17 && p.index == 1000));
    let full = vec![C32::new(1.0, 1.0); 3000];
    assert!(run_decoder(&mut adsb(), &full, None, None).is_empty());
}

// ------------------------------------------------------------------------- test 6

#[test]
fn hdlc_ook_adsb_decoders_are_chunk_invariant_with_20_plans() {
    let plans = twenty_plans(SEED);
    // hdlc_decode: frames with idle flags and random bits between them.
    let mut rng = Pcg32::new(SEED, 20);
    let packets = test_packets(&mut rng, 10);
    let mut enc = encoder(&HdlcEncodeConfig {
        preamble_flags: 8,
        ..HdlcEncodeConfig::default()
    });
    let mut stream = Vec::new();
    let mut room = vec![0.0f32; 4096];
    for p in &packets {
        let r = enc.process(std::slice::from_ref(p), &mut room).unwrap();
        stream.extend_from_slice(&room[..r.produced]);
        for _ in 0..rng.below(50) {
            stream.push(if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 });
        }
        let r = enc.process(&[], &mut room[..17]).unwrap();
        stream.extend_from_slice(&room[..r.produced]);
    }
    let mk = || {
        decoder(&HdlcDecodeConfig {
            emit_bad: true,
            ..HdlcDecodeConfig::default()
        })
    };
    let whole = run_decoder(&mut mk(), &stream, None, None);
    assert!(
        whole
            .iter()
            .filter(|p| p.flags == PacketFlags::CRC_OK)
            .count()
            >= packets.len() - 1
    );
    for (k, plan) in plans.iter().enumerate() {
        let out_plan = ChunkPlan::new(SEED * 100 + 50 + k as u64, 1 + k % 3);
        let got = run_decoder(&mut mk(), &stream, Some(*plan), Some(out_plan));
        assert_eq!(whole, got, "hdlc_decode plan {k}");
    }
    // ook_demod at a fractional rate with noise.
    let bits = ook_bits(1500, 21);
    let x = ook_signal(&bits, 10, 0.02, 0.05, 22);
    let whole = run_block(&mut ook_demod(1010.0, 10_000), &x, None);
    for plan in &plans {
        let got = run_block(&mut ook_demod(1010.0, 10_000), &x, Some(*plan));
        same_bits(&whole, &got, "ook_demod chunks");
    }
    // adsb_decode: 20 frames at random positions in noise, some short.
    // The short frames carry the address of an earlier DF 17 frame, so they pass the cache.
    let mut x = noise(40_000, 0.01, 23);
    let mut p = 100usize;
    let mut sent = 0;
    let mut last = None;
    while p + 300 < x.len() {
        if let Some(addr) = last.filter(|_| rng.below(4) == 0) {
            put_frame(&mut x, p, &with_parity(&[0x20, 1, 2, 3], addr), 0.3, 0.5);
        } else {
            let f = random_df17(&mut rng);
            last = Some(sdrk_proto::adsb::icao(&f));
            put_frame(&mut x, p, &f, 0.1 + 0.3 * rng.next_f64() as f32, 1.0);
        }
        sent += 1;
        p += 241 + rng.below(1500) as usize;
    }
    let whole = run_decoder(&mut adsb(), &x, None, None);
    assert!(whole.len() >= sent, "{} of {sent}", whole.len());
    for (k, plan) in plans.iter().enumerate() {
        let out_plan = ChunkPlan::new(SEED * 100 + 50 + k as u64, 1 + k % 3);
        let got = run_decoder(&mut adsb(), &x, Some(*plan), Some(out_plan));
        assert_eq!(whole, got, "adsb_decode plan {k}");
    }
}
