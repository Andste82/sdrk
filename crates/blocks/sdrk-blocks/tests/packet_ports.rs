//! Packet ports of M37: `Packets`, `Clocked<S, Packets>`, the timing kinds `Packets` and
//! `Source`, and the blocks `frame_encode` and `frame_decode` (descriptions, exact storage,
//! chunk invariance with 20 chunk plans, resumption, idle signals, errors).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::large_stack_arrays,
    clippy::many_single_char_names,
    clippy::assert_is_empty
)]

mod common;

use common::{check_exact_storage, rate};
use sdrk::{Delay, Progress, Ratio, RatioKind};
use sdrk_blocks::{
    ArenaStorage, Block, BlockDescription, BlockTiming, Clocked, Error, FrameDecodeBlock,
    FrameDecodeConfig, FrameEncodeBlock, FrameEncodeConfig, FrameIdle, Packet, PacketFlags,
    Packets, ParamValue, PortType, Ports, all_descriptions,
};
use sdrk_proto::frame::{self, MAX_PAYLOAD};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

const SEED: u64 = 479;

/// The 20 largest chunk sizes of the invariance tests (all with seed 479).
const CHUNK_MAX: [usize; 20] = [
    1, 2, 3, 5, 7, 8, 11, 16, 33, 55, 56, 57, 64, 100, 255, 256, 700, 1000, 2112, 5000,
];

type Enc = FrameEncodeBlock<VecStorage>;
type Dec = FrameDecodeBlock<VecStorage>;

fn find(kind: &str) -> &'static BlockDescription {
    all_descriptions()
        .iter()
        .copied()
        .find(|d| d.kind == kind)
        .unwrap()
}

fn encoder(cfg: &FrameEncodeConfig) -> Enc {
    FrameEncodeBlock::new(cfg, rate(48_000), &mut VecStorage).unwrap()
}

fn decoder(cfg: &FrameDecodeConfig) -> Dec {
    FrameDecodeBlock::new(cfg, rate(48_000), &mut VecStorage).unwrap()
}

/// `n` random packets with lengths 0 to 253.
fn random_packets(rng: &mut Pcg32, n: usize) -> Vec<Packet> {
    (0..n)
        .map(|_| {
            let len = rng.below(MAX_PAYLOAD as u32 + 1) as usize;
            let data: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect()
}

/// The frame of `p` as `±1.0` samples, from `sdrk_proto::frame::encode`.
fn frame_samples(p: &Packet, preamble_bytes: usize) -> Vec<f32> {
    let mut bits = vec![0u8; frame::frame_bits(p.len(), preamble_bytes)];
    frame::encode(p, preamble_bytes, &mut bits).unwrap();
    bits.iter()
        .map(|&b| if b == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// A soft-bit stream with `packets` framed at random positions between random noise, and the
/// expected packets (index = first sync word bit, flag `CRC_OK`).
fn stream_with_frames(rng: &mut Pcg32, packets: &[Packet]) -> (Vec<f32>, Vec<Packet>) {
    let mut soft = Vec::new();
    let mut want = Vec::new();
    for p in packets {
        // Random soft values between the frames (never a sync word: the decoder would only
        // take one if all 32 bits matched by chance, which the expectation would catch).
        for _ in 0..rng.below(300) {
            soft.push(rng.next_f64() as f32 * 2.0 - 1.0);
        }
        let preamble = rng.below(5) as usize;
        let start = soft.len() + 8 * preamble;
        soft.extend(frame_samples(p, preamble));
        let mut got = Packet::new(start as u64, PacketFlags::CRC_OK, p.data()).unwrap();
        got.index = start as u64;
        want.push(got);
    }
    for _ in 0..100 {
        soft.push(-0.5);
    }
    (soft, want)
}

/// Runs the decoder over `input` with input and output pieces from the plans (`None`: all at
/// once) and returns every packet.
fn run_decoder(
    dec: &mut Dec,
    input: &[f32],
    plan_in: Option<ChunkPlan>,
    plan_out: Option<ChunkPlan>,
) -> Vec<Packet> {
    let mut sizes_in = plan_in.map(|p| p.sizes());
    let mut sizes_out = plan_out.map(|p| p.sizes());
    let mut out = Vec::new();
    let mut pos = 0;
    let mut stalls = 0;
    while pos < input.len() {
        let want_in = sizes_in.as_mut().map_or(input.len(), |s| s.next().unwrap());
        let want_out = sizes_out.as_mut().map_or(64, |s| s.next().unwrap());
        let end = (pos + want_in).min(input.len());
        let mut room = vec![Packet::default(); want_out];
        let p = dec.process(&input[pos..end], &mut room).unwrap();
        assert!(p.consumed <= end - pos && p.produced <= want_out);
        out.extend_from_slice(&room[..p.produced]);
        pos += p.consumed;
        stalls = if p.consumed == 0 { stalls + 1 } else { 0 };
        assert!(stalls < 3, "the decoder stalls at {pos}");
    }
    out
}

/// Runs the encoder over `packets` with packet and output pieces from the plans (`None`: all
/// at once, output room 4096) and returns every sample; the encoder never sees an empty
/// input, so no idle samples appear.
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
        let want_out = sizes_out.as_mut().map_or(4096, |s| s.next().unwrap());
        let end = (pos + want_in).min(packets.len());
        let mut room = vec![f32::NAN; want_out];
        let p = enc.process(&packets[pos..end], &mut room).unwrap();
        assert!(p.consumed <= end - pos);
        // The whole room is used unless every offered packet was consumed.
        if p.consumed < end - pos {
            assert_eq!(p.produced, want_out);
        }
        out.extend_from_slice(&room[..p.produced]);
        pos += p.consumed;
    }
    out
}

#[test]
fn descriptions_name_packet_ports() {
    let enc = find("frame_encode");
    assert_eq!(enc.inputs.len(), 1);
    assert_eq!(enc.inputs[0].types, [PortType::Packets]);
    assert_eq!(enc.outputs[0].types, [PortType::F32]);
    let dec = find("frame_decode");
    assert_eq!(dec.inputs[0].types, [PortType::F32]);
    assert_eq!(dec.outputs[0].types, [PortType::Packets]);
    // Parameters as specified: kind, range, default, build time.
    let p = enc.param("preamble_bytes").unwrap();
    assert_eq!(p.default_value(), Some(ParamValue::I64(4)));
    assert!(p.check(&ParamValue::I64(32)).is_ok());
    assert!(p.check(&ParamValue::I64(33)).is_err());
    assert!(p.check(&ParamValue::I64(-1)).is_err());
    let idle = enc.param("idle").unwrap();
    assert_eq!(idle.default_value(), Some(ParamValue::Choice(0)));
    assert!(matches!(
        idle.kind,
        sdrk_blocks::ParamKind::Choice {
            options: ["zero", "preamble"],
            ..
        }
    ));
    let m = dec.param("max_sync_errors").unwrap();
    assert_eq!(m.default_value(), Some(ParamValue::I64(0)));
    assert!(m.check(&ParamValue::I64(4)).is_ok());
    assert!(m.check(&ParamValue::I64(5)).is_err());
    let b = dec.param("emit_bad").unwrap();
    assert_eq!(b.default_value(), Some(ParamValue::Bool(false)));
    for p in enc.params.iter().chain(dec.params) {
        assert_eq!(p.mutability, sdrk_blocks::Mutability::Build, "{}", p.name);
    }
    assert!(enc.required.is_empty() && dec.required.is_empty());
}

#[test]
fn blocks_report_their_parameters_and_reject_changes() {
    let cfg = FrameEncodeConfig {
        preamble_bytes: 7,
        idle: FrameIdle::Preamble,
    };
    let mut enc = encoder(&cfg);
    assert_eq!(enc.description().kind, "frame_encode");
    assert_eq!(enc.param("preamble_bytes"), Ok(ParamValue::I64(7)));
    assert_eq!(enc.param("idle"), Ok(ParamValue::Choice(1)));
    assert_eq!(
        enc.set_param("preamble_bytes", ParamValue::I64(1)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        enc.set_param("nope", ParamValue::I64(1)),
        Err(Error::UnknownParameter)
    );
    let cfg = FrameDecodeConfig {
        max_sync_errors: 3,
        emit_bad: true,
        ..FrameDecodeConfig::default()
    };
    let mut dec = decoder(&cfg);
    assert_eq!(dec.description().kind, "frame_decode");
    assert_eq!(dec.param("max_sync_errors"), Ok(ParamValue::I64(3)));
    assert_eq!(dec.param("emit_bad"), Ok(ParamValue::Bool(true)));
    assert_eq!(
        dec.set_param("emit_bad", ParamValue::Bool(false)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(dec.param("nope"), Err(Error::UnknownParameter));
    // Invalid configurations are rejected by `new` and `storage_bytes`.
    let bad = FrameEncodeConfig {
        preamble_bytes: 33,
        ..FrameEncodeConfig::default()
    };
    assert!(matches!(
        FrameEncodeBlock::<VecStorage>::new(&bad, rate(48_000), &mut VecStorage),
        Err(Error::OutOfRange {
            name: "preamble_bytes",
            ..
        })
    ));
    assert!(FrameEncodeBlock::<VecStorage>::storage_bytes(&bad, rate(48_000)).is_err());
    let bad = FrameDecodeConfig {
        max_sync_errors: 5,
        ..FrameDecodeConfig::default()
    };
    assert!(matches!(
        FrameDecodeBlock::<VecStorage>::new(&bad, rate(48_000), &mut VecStorage),
        Err(Error::OutOfRange {
            name: "max_sync_errors",
            ..
        })
    ));
    assert!(FrameDecodeBlock::<VecStorage>::storage_bytes(&bad, rate(48_000)).is_err());
}

#[test]
fn storage_is_exact() {
    let r = rate(9600);
    let cfg = FrameEncodeConfig::default();
    let bytes = FrameEncodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, r).unwrap();
    assert_eq!(bytes, 0);
    check_exact_storage(bytes, |arena| {
        FrameEncodeBlock::new(&cfg, r, arena).map(|_| ())
    });
    let cfg = FrameDecodeConfig::default();
    let bytes = FrameDecodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, r).unwrap();
    assert_eq!(bytes, 0);
    check_exact_storage(bytes, |arena| {
        FrameDecodeBlock::new(&cfg, r, arena).map(|_| ())
    });
}

#[test]
fn timing_kinds_of_packet_blocks() {
    let enc = encoder(&FrameEncodeConfig::default());
    let t = enc.timing();
    assert_eq!(t.kind, RatioKind::Source);
    // At most one frame of 253 payload bytes and 4 preamble bytes per packet.
    assert_eq!(t.ratio, Ratio::new(8 * (4 + 4 + 1 + 253 + 2), 1).unwrap());
    assert_eq!(t.delay, Delay::ZERO);
    assert_eq!(enc.min_input(), 1);
    let enc0 = encoder(&FrameEncodeConfig {
        preamble_bytes: 0,
        ..FrameEncodeConfig::default()
    });
    assert_eq!(enc0.timing().ratio, Ratio::new(2080, 1).unwrap());
    assert_eq!(enc0.frame_samples(253), 2080);

    let dec = decoder(&FrameDecodeConfig::default());
    let t = dec.timing();
    let max = Ratio::new(1, 56).unwrap();
    assert_eq!(t.kind, RatioKind::Packets { max_per_input: max });
    assert_eq!(t.ratio, max);
    assert_eq!(t.delay, Delay::ZERO);
    assert_eq!(dec.min_input(), 1);
    // Neither block has a clocked input.
    assert_eq!(enc.consumed_packets(), 0);
    assert_eq!(dec.consumed_packets(), 0);
}

#[test]
fn encoder_writes_the_frames_of_sdrk_proto() {
    let mut rng = Pcg32::new(SEED, 1);
    let packets = random_packets(&mut rng, 40);
    for preamble in [0, 1, 4, 32] {
        let cfg = FrameEncodeConfig {
            preamble_bytes: preamble,
            ..FrameEncodeConfig::default()
        };
        let want: Vec<f32> = packets
            .iter()
            .flat_map(|p| frame_samples(p, preamble))
            .collect();
        let got = run_encoder(&mut encoder(&cfg), &packets, None, None);
        assert_eq!(got, want, "preamble {preamble}");
    }
    // Every length 0 to 253.
    let all: Vec<Packet> = (0..=MAX_PAYLOAD)
        .map(|len| {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + len) as u8).collect();
            Packet::new(0, PacketFlags::NONE, &data).unwrap()
        })
        .collect();
    let want: Vec<f32> = all.iter().flat_map(|p| frame_samples(p, 4)).collect();
    let got = run_encoder(
        &mut encoder(&FrameEncodeConfig::default()),
        &all,
        None,
        None,
    );
    assert_eq!(got, want);
}

#[test]
fn encoder_resumes_a_frame_across_full_outputs() {
    let mut rng = Pcg32::new(SEED, 2);
    let packets = random_packets(&mut rng, 30);
    let cfg = FrameEncodeConfig::default();
    let want = run_encoder(&mut encoder(&cfg), &packets, None, None);
    for (k, max) in CHUNK_MAX.iter().enumerate() {
        let plan_in = ChunkPlan::new(SEED, 1 + k % 5);
        let plan_out = ChunkPlan::new(SEED, *max);
        let got = run_encoder(&mut encoder(&cfg), &packets, Some(plan_in), Some(plan_out));
        assert!(got == want, "plan {k} (output pieces up to {max})");
    }
    // A packet stays unconsumed until its last sample is written.
    let mut enc = encoder(&cfg);
    let p = packets[0];
    let total = enc.frame_samples(p.len());
    let mut out = vec![0.0f32; total - 1];
    let r = enc.process(&[p], &mut out).unwrap();
    assert_eq!((r.consumed, r.produced), (0, total - 1));
    assert!(enc.in_frame());
    let mut last = [0.0f32; 5];
    let r = enc.process(&[p], &mut last).unwrap();
    assert_eq!((r.consumed, r.produced), (1, 1));
    assert!(!enc.in_frame());
    // No output room: nothing happens.
    let r = enc.process(&[p], &mut []).unwrap();
    assert_eq!(r, Progress::default());
}

#[test]
fn encoder_idle_zero_and_preamble() {
    let p = Packet::new(0, PacketFlags::NONE, b"idle").unwrap();
    // Zero idle.
    let mut enc = encoder(&FrameEncodeConfig::default());
    let mut out = [f32::NAN; 37];
    let r = enc.process(&[], &mut out).unwrap();
    assert_eq!((r.consumed, r.produced), (0, 37));
    assert!(out.iter().all(|&x| x.to_bits() == 0));
    // Preamble idle: the bits of 0x55 (0, 1, 0, 1, ...) as -1, +1, continued across calls.
    let mut enc = encoder(&FrameEncodeConfig {
        idle: FrameIdle::Preamble,
        ..FrameEncodeConfig::default()
    });
    let mut idle = Vec::new();
    for n in [3, 1, 8, 5] {
        let mut out = vec![f32::NAN; n];
        let r = enc.process(&[], &mut out).unwrap();
        assert_eq!((r.consumed, r.produced), (0, n));
        idle.extend(out);
    }
    let want: Vec<f32> = (0..17)
        .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    assert_eq!(idle, want);
    // A frame after idle is unchanged; an empty call in the middle of a frame drops it.
    let frame_len = enc.frame_samples(p.len());
    let mut out = vec![0.0f32; 10];
    assert_eq!(enc.process(&[p], &mut out).unwrap().consumed, 0);
    assert_eq!(out, frame_samples(&p, 4)[..10]);
    assert!(enc.in_frame());
    let mut out = vec![0.0f32; 4];
    enc.process(&[], &mut out).unwrap();
    assert!(!enc.in_frame());
    let mut out = vec![0.0f32; frame_len];
    let r = enc.process(&[p], &mut out).unwrap();
    assert_eq!((r.consumed, r.produced), (1, frame_len));
    assert_eq!(out, frame_samples(&p, 4));
    // `reset` drops the frame in progress and restarts the idle pattern.
    let mut out = vec![0.0f32; 3];
    enc.process(&[p], &mut out).unwrap();
    enc.reset();
    assert!(!enc.in_frame());
    enc.process(&[], &mut out).unwrap();
    assert_eq!(out, [-1.0, 1.0, -1.0]);
}

#[test]
fn encoder_rejects_a_payload_above_253_bytes() {
    let ok = Packet::new(0, PacketFlags::NONE, &[1, 2, 3]).unwrap();
    let mut enc = encoder(&FrameEncodeConfig::default());
    for len in [254, 255, 256] {
        let long = Packet::new(0, PacketFlags::NONE, &vec![0xA5; len]).unwrap();
        let mut out = vec![0.0f32; 10_000];
        // First in the call: an error, nothing consumed.
        assert!(matches!(
            enc.process(&[long, ok], &mut out),
            Err(Error::Proto(sdrk_proto::Error::TooLong { .. }))
        ));
        assert!(!enc.in_frame());
        // After a good packet: the call stops before it.
        let r = enc.process(&[ok, long], &mut out).unwrap();
        assert_eq!((r.consumed, r.produced), (1, enc.frame_samples(3)));
    }
    let e = enc
        .process(
            &[Packet::new(0, PacketFlags::NONE, &[0; 254]).unwrap()],
            &mut [0.0; 4],
        )
        .unwrap_err();
    assert_eq!(
        sdrk_device::SdrkError::kind(&e),
        sdrk_device::ErrorKind::OutOfRange
    );
}

#[test]
fn decoder_finds_the_frames_with_their_indices() {
    let mut rng = Pcg32::new(SEED, 3);
    let packets = random_packets(&mut rng, 60);
    let (soft, want) = stream_with_frames(&mut rng, &packets);
    let got = run_decoder(
        &mut decoder(&FrameDecodeConfig::default()),
        &soft,
        None,
        None,
    );
    assert_eq!(got, want);
    // NaN and 0.0 are 0 bits: a frame of +1/NaN decodes like +1/-1.
    let p = Packet::new(0, PacketFlags::NONE, b"nan").unwrap();
    let nan: Vec<f32> = frame_samples(&p, 2)
        .iter()
        .enumerate()
        .map(|(i, &x)| match (x > 0.0, i % 2) {
            (true, _) => x,
            (false, 0) => f32::NAN,
            (false, _) => 0.0,
        })
        .collect();
    let got = run_decoder(
        &mut decoder(&FrameDecodeConfig::default()),
        &nan,
        None,
        None,
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].data(), b"nan");
    assert_eq!(got[0].index, 16);
}

#[test]
fn decoder_is_chunk_invariant_with_20_plans() {
    let mut rng = Pcg32::new(SEED, 4);
    let packets = random_packets(&mut rng, 25);
    let (soft, want) = stream_with_frames(&mut rng, &packets);
    let cfg = FrameDecodeConfig {
        max_sync_errors: 2,
        emit_bad: true,
        ..FrameDecodeConfig::default()
    };
    let whole = run_decoder(&mut decoder(&cfg), &soft, None, None);
    // With 2 sync errors and emit_bad, noise may give extra CRC_FAIL packets; the sent
    // packets are all there with CRC_OK.
    let ok: Vec<Packet> = whole
        .iter()
        .copied()
        .filter(|p| p.flags == PacketFlags::CRC_OK)
        .collect();
    assert_eq!(ok, want);
    for (k, max) in CHUNK_MAX.iter().enumerate() {
        let plan_in = ChunkPlan::new(SEED, *max);
        // Output room of 1 to 3 packets: the decoder must stop after a full output.
        let plan_out = ChunkPlan::new(SEED, 1 + k % 3);
        let mut dec = decoder(&cfg);
        let got = run_decoder(&mut dec, &soft, Some(plan_in), Some(plan_out));
        assert!(got == whole, "plan {k} (input pieces up to {max})");
        assert_eq!(dec.position(), soft.len() as u64);
    }
}

#[test]
fn decoder_stops_after_the_packet_that_fills_the_output() {
    let a = Packet::new(0, PacketFlags::NONE, b"a").unwrap();
    let b = Packet::new(0, PacketFlags::NONE, b"bb").unwrap();
    let mut soft = frame_samples(&a, 0);
    let first_end = soft.len();
    soft.extend(frame_samples(&b, 0));
    let mut dec = decoder(&FrameDecodeConfig::default());
    // No room: nothing consumed.
    assert_eq!(dec.process(&soft, &mut []).unwrap(), Progress::default());
    let mut one = [Packet::default(); 1];
    let r = dec.process(&soft, &mut one).unwrap();
    assert_eq!((r.consumed, r.produced), (first_end, 1));
    assert_eq!((one[0].data(), one[0].index), (&b"a"[..], 0));
    let r = dec.process(&soft[first_end..], &mut one).unwrap();
    assert_eq!((r.consumed, r.produced), (soft.len() - first_end, 1));
    assert_eq!(
        (one[0].data(), one[0].index),
        (&b"bb"[..], first_end as u64)
    );
    // `reset` restarts the index count.
    dec.reset();
    assert_eq!(dec.position(), 0);
    let r = dec.process(&soft, &mut one).unwrap();
    assert_eq!((r.produced, one[0].index), (1, 0));
}

#[test]
fn decoder_options_sync_errors_and_bad_frames() {
    let p = Packet::new(0, PacketFlags::NONE, b"sync").unwrap();
    let clean = frame_samples(&p, 1);
    // Two flipped sync word bits (sync starts at 8).
    let mut damaged = clean.clone();
    damaged[8 + 3] = -damaged[8 + 3];
    damaged[8 + 20] = -damaged[8 + 20];
    let strict = run_decoder(
        &mut decoder(&FrameDecodeConfig::default()),
        &damaged,
        None,
        None,
    );
    assert!(strict.is_empty());
    let lax = run_decoder(
        &mut decoder(&FrameDecodeConfig {
            max_sync_errors: 2,
            emit_bad: false,
            ..FrameDecodeConfig::default()
        }),
        &damaged,
        None,
        None,
    );
    assert_eq!(lax.len(), 1);
    assert_eq!((lax[0].index, lax[0].flags), (8, PacketFlags::CRC_OK));
    // A flipped payload bit: dropped by default, CRC_FAIL with emit_bad.
    let mut bad = clean.clone();
    let payload = 8 + 32 + 8;
    bad[payload + 5] = -bad[payload + 5];
    assert!(
        run_decoder(
            &mut decoder(&FrameDecodeConfig::default()),
            &bad,
            None,
            None
        )
        .is_empty()
    );
    let got = run_decoder(
        &mut decoder(&FrameDecodeConfig {
            max_sync_errors: 0,
            emit_bad: true,
            ..FrameDecodeConfig::default()
        }),
        &bad,
        None,
        None,
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].flags, PacketFlags::CRC_FAIL);
    assert_eq!(got[0].index, 8);
    assert_ne!(got[0].data(), p.data());
}

#[test]
fn encoder_to_decoder_loop() {
    let mut rng = Pcg32::new(SEED, 5);
    let packets = random_packets(&mut rng, 200);
    for idle in [FrameIdle::Zero, FrameIdle::Preamble] {
        let cfg = FrameEncodeConfig {
            preamble_bytes: 2,
            idle,
        };
        let mut enc = encoder(&cfg);
        let mut soft = Vec::new();
        let mut want = Vec::new();
        let mut room = vec![0.0f32; 700];
        let mut pos = 0;
        while pos < packets.len() {
            // Idle gap, then a group of packets.
            let gap = rng.below(100) as usize;
            let r = enc.process(&[], &mut room[..gap]).unwrap();
            soft.extend_from_slice(&room[..r.produced]);
            let group = (1 + rng.below(4) as usize).min(packets.len() - pos);
            let mut start = soft.len();
            let mut done = 0;
            while done < group {
                let r = enc
                    .process(&packets[pos + done..pos + group], &mut room)
                    .unwrap();
                soft.extend_from_slice(&room[..r.produced]);
                done += r.consumed;
            }
            for p in &packets[pos..pos + group] {
                let mut w = *p;
                w.index = (start + 16) as u64;
                w.flags = PacketFlags::CRC_OK;
                want.push(w);
                start += enc.frame_samples(p.len());
            }
            pos += group;
        }
        let got = run_decoder(
            &mut decoder(&FrameDecodeConfig::default()),
            &soft,
            Some(ChunkPlan::new(SEED, 333)),
            None,
        );
        assert_eq!(got, want, "{idle:?}");
    }
}

/// A tiny clocked encoder for the test of `Clocked`: per clock sample it outputs the next
/// payload byte of the queued packets as `f32` (0.0 when no packet is queued).
struct ByteClock {
    /// Next byte of the first packet.
    pos: usize,
    taken: usize,
}

impl Block for ByteClock {
    type In = Clocked<f32, Packets>;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &sdrk_blocks::CHAIN_DESCRIPTION
    }

    fn process(
        &mut self,
        (clock, packets): (&[f32], &[Packet]),
        output: &mut [f32],
    ) -> Result<Progress, Error> {
        self.taken = 0;
        let mut n = 0;
        for (_, o) in clock.iter().zip(output.iter_mut()) {
            *o = match packets.get(self.taken) {
                Some(p) => {
                    let byte = p.data()[self.pos];
                    self.pos += 1;
                    if self.pos == p.len() {
                        self.pos = 0;
                        self.taken += 1;
                    }
                    f32::from(byte)
                }
                None => 0.0,
            };
            n += 1;
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::ONE,
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.pos = 0;
    }

    fn set_param(&mut self, _: &str, _: ParamValue) -> Result<(), Error> {
        Err(Error::UnknownParameter)
    }

    fn param(&self, _: &str) -> Result<ParamValue, Error> {
        Err(Error::UnknownParameter)
    }

    fn min_input(&self) -> usize {
        1
    }

    fn consumed_packets(&self) -> usize {
        self.taken
    }
}

#[test]
fn clocked_counts_clock_samples_and_packets_separately() {
    type C = Clocked<f32, Packets>;
    assert_eq!(<C as Ports>::COUNT, 2);
    let clock = [0.5f32; 10];
    let queue = [
        Packet::new(0, PacketFlags::NONE, &[1, 2, 3]).unwrap(),
        Packet::new(0, PacketFlags::NONE, &[4, 5]).unwrap(),
    ];
    let view: <C as Ports>::Ref<'_> = (&clock, &queue);
    assert_eq!(<C as Ports>::ref_len(&view), 10);
    let rest = <C as Ports>::ref_skip(view, 4);
    assert_eq!((rest.0.len(), rest.1.len()), (6, 2));
    let (mut a, mut b) = ([0.0f32; 3], [Packet::default(); 1]);
    let mut m: <C as Ports>::Mut<'_> = (&mut a, &mut b);
    assert_eq!(<C as Ports>::mut_len(&m), 3);
    let r = <C as Ports>::mut_reborrow(&mut m);
    assert_eq!(<C as Ports>::mut_len(&<C as Ports>::mut_skip(r, 2)), 1);

    // The block: the queue goes out byte by byte, then zeros; the caller advances the clock
    // by `consumed` and the queue by `consumed_packets`.
    let mut block = ByteClock { pos: 0, taken: 0 };
    let mut out = Vec::new();
    let mut packets: &[Packet] = &queue;
    for n in [2, 2, 4] {
        let mut room = vec![0.0f32; n];
        let p = block.process((&clock[..n], packets), &mut room).unwrap();
        assert_eq!(p.consumed, n);
        packets = &packets[block.consumed_packets()..];
        out.extend(room);
    }
    assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0]);
    assert!(packets.is_empty());
    // Without packets the clock still runs.
    let mut room = [9.0f32; 3];
    let p = block.process((&clock[..3], &[]), &mut room).unwrap();
    assert_eq!((p.consumed, block.consumed_packets()), (3, 0));
    assert_eq!(room, [0.0; 3]);
}

#[test]
fn packets_ports_are_slices_of_packets() {
    let mut buf = [Packet::default(); 4];
    let m: <Packets as Ports>::Mut<'_> = &mut buf;
    assert_eq!(<Packets as Ports>::mut_len(&m), 4);
    let m = <Packets as Ports>::mut_skip(m, 3);
    assert_eq!(m.len(), 1);
    let m = <Packets as Ports>::mut_skip(m, 9);
    assert!(m.is_empty());
    let r: <Packets as Ports>::Ref<'_> = &buf;
    assert!(<Packets as Ports>::ref_skip(r, 5).is_empty());
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FrameEncodeBlock<VecStorage>>();
    assert_send_sync::<FrameDecodeBlock<VecStorage>>();
    assert_send_sync::<FrameEncodeConfig>();
    assert_send_sync::<FrameDecodeConfig>();
    assert_send_sync::<FrameIdle>();
    assert_send_sync::<Packets>();
    assert_send_sync::<Clocked<f32, Packets>>();
    assert_send_sync::<Packet>();
    assert_send_sync::<RatioKind>();
}
