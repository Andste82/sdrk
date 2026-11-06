//! The blocks of M42: `RdsEncodeBlock` (the control packets, the ignored count, the raw-group
//! back-pressure, the groups and bits against an independent model of the carousel and the
//! kernel `rds_shape`, test 4: the cross-check with the reference modulator of M41 through the
//! golden case `rds_encode/cross_check_585`, test 5: chunk invariance with packets at fixed
//! clock positions in 20 cut plans of seed 586, reset, the clock types and extremes) and
//! `FmStereoRdsModBlock` (equal to the kernel `stereo_encode_rds` whole and in 20 cut plans with
//! runtime changes of `pilot` and `rds`, the lockstep rule, extremes, test 2: the spectrum of
//! the RDS component of the multiplex with seed 582), plus storage, timing and `Send`/`Sync`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::large_stack_frames,
    clippy::many_single_char_names,
    clippy::print_stderr,
    clippy::type_complexity,
    clippy::trivially_copy_pass_by_ref
)]

mod common;

use std::collections::VecDeque;

use common::{Aligned, check_exact_storage, rate, same_bits, twenty_plans};
use num_complex::Complex64;
use sdrk::{C32, CQ15, Delay, Q15, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, Error, FmStereoRdsModBlock, FmStereoRdsModConfig, Packet, PacketFlags,
    ParamValue, RdsEncodeBlock, RdsEncodeConfig, StreamElem,
};
use sdrk_core::kernels::rds::{RDS_TABLE_LEN, RdsShape, RdsShapeState, rds_pulse_table};
use sdrk_core::kernels::stereo::{StereoEncodeRds, StereoEncodeRdsState};
use sdrk_proto::rds::RAW_QUEUE;
use sdrk_testkit::golden::{self, Array, Tolerance};
use sdrk_testkit::rds::{OFFSET_WORDS, checkword, data_bit};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

/// Seed of the spectrum test (test 2).
const SPECTRUM_SEED: u64 = 582;
/// Seed of the chunk-invariance test (test 5) and of its random groups.
const CHUNK_SEED: u64 = 586;
/// The multiplex rate of the transmitter flow.
const FS: u64 = 192_000;
/// The bit rate of RDS.
const BIT_RATE: f64 = 1187.5;

fn encoder_config() -> RdsEncodeConfig {
    RdsEncodeConfig {
        pi: 0xD123,
        pty: 10,
        tp: false,
    }
}

fn packet(data: &[u8]) -> Packet {
    Packet::new(0, PacketFlags::NONE, data).unwrap()
}

fn ps_packet(name: &[u8; 8]) -> Packet {
    let mut d = vec![0x01];
    d.extend_from_slice(name);
    packet(&d)
}

fn rt_packet(text: &[u8]) -> Packet {
    let mut d = vec![0x02];
    d.extend_from_slice(text);
    packet(&d)
}

fn group_packet(words: [u16; 4]) -> Packet {
    let mut d = vec![0x03];
    for w in words {
        d.extend_from_slice(&w.to_be_bytes());
    }
    packet(&d)
}

/// The data bits of a group: each word with its checkword (offsets A, B, C or C' for version
/// B, D), MSB first. Independent of `sdrk_proto` (the checkword of the testkit).
fn group_bits(words: [u16; 4]) -> Vec<u8> {
    let third = if words[1] & 0x0800 != 0 { 3 } else { 2 };
    let offsets = [
        OFFSET_WORDS[0],
        OFFSET_WORDS[1],
        OFFSET_WORDS[third],
        OFFSET_WORDS[4],
    ];
    let mut out = Vec::with_capacity(104);
    for (w, o) in words.iter().zip(offsets) {
        let block = (u32::from(*w) << 10) | u32::from(checkword(*w, o));
        out.extend((0..26).map(|i| ((block >> (25 - i)) & 1) as u8));
    }
    out
}

/// The baseband of the data bits `d` (differential coding from 0, amplitudes +-1) through the
/// kernel `rds_shape` at `fs`, `n` samples.
fn shaped(d: &[u8], fs: u64, n: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; RDS_TABLE_LEN];
    rds_pulse_table(&mut table).unwrap();
    let mut st = RdsShapeState::new(table, rate(fs)).unwrap();
    let mut t = 0u8;
    let mut k = 0usize;
    let mut bits = || {
        t ^= d[k];
        k += 1;
        if t == 1 { 1 } else { -1 }
    };
    let mut out = vec![0.0f32; n];
    Sdrk.rds_shape(&mut st, &mut bits, &mut out).unwrap();
    out
}

/// The bits fetched for the samples `0 .. n` at `fs` (bits `0 ..= floor(x) + 2` of the last).
fn bits_fetched(n: u64, fs: u64) -> u64 {
    ((n - 1) as f64 * BIT_RATE / fs as f64).floor() as u64 + 3
}

fn new_encoder<A: StreamElem>(fs: u64) -> RdsEncodeBlock<A, VecStorage> {
    RdsEncodeBlock::new(&encoder_config(), rate(fs), &mut VecStorage).unwrap()
}

/// Runs the encoder for `n` clock samples with the packets of `schedule` delivered at their
/// clock positions (a call never crosses a position), in pieces of `plan` (`None`: one piece
/// between the positions). Unconsumed packets are resubmitted with the next call.
fn run_encoder<A: StreamElem>(
    block: &mut RdsEncodeBlock<A, VecStorage>,
    n: usize,
    schedule: &[(usize, Vec<Packet>)],
    plan: Option<ChunkPlan>,
) -> Vec<f32> {
    let clock = vec![A::default(); n];
    let mut out = vec![0.0f32; n];
    let mut pending: VecDeque<Packet> = VecDeque::new();
    let mut sizes = plan.map(|p| p.sizes());
    let mut pos = 0;
    let mut next_event = 0;
    while pos < n || (next_event < schedule.len() && schedule[next_event].0 <= pos) {
        while next_event < schedule.len() && schedule[next_event].0 <= pos {
            pending.extend(schedule[next_event].1.iter().copied());
            next_event += 1;
        }
        let limit = schedule.get(next_event).map_or(n, |e| e.0.min(n));
        let piece = sizes
            .as_mut()
            .map_or(limit - pos, |s| s.next().unwrap().min(limit - pos));
        let queue: Vec<Packet> = pending.iter().copied().collect();
        let p = block
            .process(
                (&clock[pos..pos + piece], &queue),
                &mut out[pos..pos + piece],
            )
            .unwrap();
        assert_eq!((p.consumed, p.produced), (piece, piece));
        pending.drain(..block.consumed_packets());
        pos += piece;
        if pos >= n {
            break;
        }
    }
    out
}

/// 20 random groups of seed `seed` (half of them version B).
fn random_groups(seed: u64, count: usize) -> Vec<[u16; 4]> {
    let mut rng = Pcg32::new(seed, 42);
    (0..count)
        .map(|_| [0u16; 4].map(|_| rng.next_u32() as u16))
        .collect()
}

// ---- construction, timing, parameters ----------------------------------------------------

/// `rds_encode` takes exactly its pulse table from the storage (1025 `f32`), at every rate from
/// 4750 Hz; `fm_stereo_rds_mod` takes nothing and needs 120 kHz.
#[test]
fn storage_and_rates() {
    type A<'a> = ArenaStorage<'a>;
    for fs in [4750, 48_000, 192_000, 1_000_000] {
        let bytes =
            RdsEncodeBlock::<f32, A<'_>>::storage_bytes(&encoder_config(), rate(fs)).unwrap();
        assert_eq!(bytes, 4 * RDS_TABLE_LEN);
        check_exact_storage(bytes, |a| {
            RdsEncodeBlock::<C32, _>::new(&encoder_config(), rate(fs), a).map(|_| ())
        });
    }
    assert!(matches!(
        RdsEncodeBlock::<f32, A<'_>>::storage_bytes(&encoder_config(), rate(4749)),
        Err(Error::OutOfRange {
            name: "rate_in",
            ..
        })
    ));
    let bad = RdsEncodeConfig {
        pty: 32,
        ..encoder_config()
    };
    assert!(matches!(
        RdsEncodeBlock::<f32, A<'_>>::storage_bytes(&bad, rate(FS)),
        Err(Error::OutOfRange { name: "pty", .. })
    ));
    for fs in [120_000, 192_000] {
        let cfg = FmStereoRdsModConfig::default();
        let bytes = FmStereoRdsModBlock::<A<'_>>::storage_bytes(&cfg, rate(fs)).unwrap();
        assert_eq!(bytes, 0);
        check_exact_storage(bytes, |a| {
            FmStereoRdsModBlock::new(&cfg, rate(fs), a).map(|_| ())
        });
    }
    assert!(matches!(
        FmStereoRdsModBlock::<A<'_>>::storage_bytes(&FmStereoRdsModConfig::default(), rate(96_000)),
        Err(Error::OutOfRange {
            name: "rate_in",
            ..
        })
    ));
    assert!(
        FmStereoRdsModConfig {
            rds: 0.21,
            ..FmStereoRdsModConfig::default()
        }
        .validate(rate(FS))
        .is_err()
    );
}

#[test]
fn timing_is_one_to_one_without_delay() {
    let unit = sdrk_blocks::BlockTiming {
        ratio: Ratio::ONE,
        kind: RatioKind::Constant,
        delay: Delay::ZERO,
    };
    assert_eq!(new_encoder::<f32>(FS).timing(), unit);
    let m = FmStereoRdsModBlock::new(&FmStereoRdsModConfig::default(), rate(FS), &mut VecStorage)
        .unwrap();
    assert_eq!(m.timing(), unit);
    assert_eq!(new_encoder::<f32>(FS).map_index(1000), 1000);
}

#[test]
fn encoder_parameters_are_build_parameters() {
    let mut b = new_encoder::<f32>(FS);
    assert_eq!(b.param("pi"), Ok(ParamValue::I64(0xD123)));
    assert_eq!(b.param("pty"), Ok(ParamValue::I64(10)));
    assert_eq!(b.param("tp"), Ok(ParamValue::Bool(false)));
    assert_eq!(
        b.set_param("pi", ParamValue::I64(1)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        b.set_param("nope", ParamValue::I64(1)),
        Err(Error::UnknownParameter)
    );
    let mut m =
        FmStereoRdsModBlock::new(&FmStereoRdsModConfig::default(), rate(FS), &mut VecStorage)
            .unwrap();
    m.set_param("rds", ParamValue::F64(0.1)).unwrap();
    assert_eq!(m.param("rds"), Ok(ParamValue::F64(0.1)));
    assert!(matches!(
        m.set_param("rds", ParamValue::F64(0.3)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(m.param("rds"), Ok(ParamValue::F64(0.1)));
    assert_eq!(
        m.set_param("main", ParamValue::F64(0.5)),
        Err(Error::ReadOnlyParameter)
    );
}

#[test]
fn the_blocks_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<RdsEncodeBlock<C32, ArenaStorage<'static>>>();
    assert_sync::<RdsEncodeBlock<C32, ArenaStorage<'static>>>();
    assert_send::<RdsEncodeBlock<Q15, VecStorage>>();
    assert_send::<RdsEncodeConfig>();
    assert_sync::<RdsEncodeConfig>();
    assert_send::<FmStereoRdsModBlock<ArenaStorage<'static>>>();
    assert_sync::<FmStereoRdsModBlock<ArenaStorage<'static>>>();
    assert_send::<FmStereoRdsModConfig>();
    assert_sync::<FmStereoRdsModConfig>();
}

// ---- rds_encode: packets -------------------------------------------------------------------

/// The three control packets act on the carousel; every other packet is consumed, ignored and
/// counted (the reading `ignored_packets`).
#[test]
fn control_packets_and_the_ignored_count() {
    let mut b = new_encoder::<f32>(FS);
    assert_eq!(b.reading(0), Some(("ignored_packets", 0.0)));
    assert_eq!(b.reading(1), None);
    let packets = [
        ps_packet(b"SDRK TST"),
        rt_packet(b"Hello"),
        group_packet([0xD123, 0x4000, 1, 2]),
        // Ignored: wrong lengths, unknown or no first byte.
        packet(b"\x01SHORT"),
        packet(&[0x02; 66]),
        packet(b"\x03\x00\x01\x02\x03\x04\x05\x06\x07\x08"),
        packet(b"\x04abc"),
        packet(&[]),
    ];
    let mut out = [0.0f32; 4];
    let p = b.process((&[0.0; 4], &packets), &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (4, 4));
    assert_eq!(b.consumed_packets(), packets.len());
    assert_eq!(b.ignored_packets(), 5);
    assert_eq!(b.reading(0), Some(("ignored_packets", 5.0)));
    assert_eq!(b.carousel().ps(), b"SDRK TST");
    assert_eq!(&b.carousel().rt()[..7], b"Hello\r ");
    // The raw group was fetched as the first group (bits 0 to 2 before sample 0).
    assert_eq!(b.carousel().raw_len(), 0);
    // An empty radiotext packet stops the groups 2A.
    b.process((&[], &[rt_packet(b"")]), &mut []).unwrap();
    assert!(!b.carousel().rt_active());
    // A full 64-byte text has no carriage return.
    b.process((&[], &[rt_packet(&[b'x'; 64])]), &mut [])
        .unwrap();
    assert_eq!(b.carousel().rt(), &[b'x'; 64]);
    assert_eq!(b.ignored_packets(), 5);
    b.reset();
    assert_eq!(b.ignored_packets(), 0);
    // The texts stay after a reset.
    assert_eq!(b.carousel().ps(), b"SDRK TST");
}

/// A raw group that finds the queue full waits in the packet queue: the call consumes the
/// packets before it, and a later call takes it once a group has left the queue.
#[test]
fn raw_groups_wait_when_the_queue_is_full() {
    let mut b = new_encoder::<f32>(FS);
    let groups: Vec<Packet> = random_groups(CHUNK_SEED, 20)
        .into_iter()
        .map(group_packet)
        .collect();
    // No clock: only the packets.
    b.process((&[], &groups), &mut []).unwrap();
    assert_eq!(b.consumed_packets(), RAW_QUEUE);
    assert!(b.carousel().is_raw_full());
    // One clock sample: the queue is still full at the start, then group 0 is fetched.
    let mut out = [0.0f32; 1];
    b.process((&[0.0], &groups[RAW_QUEUE..]), &mut out).unwrap();
    assert_eq!(b.consumed_packets(), 0);
    assert_eq!(b.carousel().raw_len(), RAW_QUEUE - 1);
    b.process((&[], &groups[RAW_QUEUE..]), &mut []).unwrap();
    assert_eq!(b.consumed_packets(), 1);
    // A packet after a waiting raw group waits too (the order is kept).
    let rest = [groups[RAW_QUEUE + 1], ps_packet(b"LATER   ")];
    b.process((&[], &rest), &mut []).unwrap();
    assert_eq!(b.consumed_packets(), 0);
    assert_eq!(b.carousel().ps(), b"        ");
}

// ---- rds_encode: bits, test 4, test 5 --------------------------------------------------------

/// The encoder sends the carousel of the texts it received and the raw groups in between, bit
/// for bit as the independent model: the groups of the M41 reference carousel
/// (`sdrk_testkit::rds::data_bit`, the same PI, PTY and texts), 20 raw groups (seed 586) after
/// the groups fetched before sample 30000 (four of them wait in the packet queue), then the
/// carousel where it stopped; the differential coding and the kernel `rds_shape`. Whole and in
/// 20 cut plans (test 5 without text changes), for the four clock types.
#[test]
fn the_encoder_sends_the_carousel_and_the_raw_groups() {
    const N: usize = 450_000;
    const AT: usize = 30_000;
    let raw = random_groups(CHUNK_SEED, 20);
    let mut schedule = vec![(
        0,
        vec![ps_packet(b"SDRK TST"), rt_packet(b"Hello from sdrk!")],
    )];
    let mut late: Vec<Packet> = raw.iter().copied().map(group_packet).collect();
    late.push(packet(b"\x09"));
    schedule.push((AT, late));

    // The model: the groups fetched before sample AT, the raw groups, the carousel again.
    let before = bits_fetched(AT as u64, FS).div_ceil(104);
    let total_groups = bits_fetched(N as u64, FS).div_ceil(104) + 1;
    let mut d: Vec<u8> = (0..before * 104).map(data_bit).collect();
    for g in &raw {
        d.extend(group_bits(*g));
    }
    d.extend((before * 104..(total_groups - 20) * 104).map(data_bit));
    let want = shaped(&d, FS, N);

    let mut block = new_encoder::<f32>(FS);
    let whole = run_encoder(&mut block, N, &schedule, None);
    same_bits(&want, &whole, "rds_encode whole");
    assert_eq!(block.ignored_packets(), 1);
    assert_eq!(block.next_bit(), bits_fetched(N as u64, FS));
    for plan in twenty_plans(CHUNK_SEED) {
        let got = run_encoder(&mut new_encoder::<f32>(FS), N, &schedule, Some(plan));
        same_bits(&want, &got, &format!("rds_encode {plan:?}"));
    }
    // The clock type and its values do not matter.
    let plan = Some(ChunkPlan::new(CHUNK_SEED, 3000));
    same_bits(
        &want,
        &run_encoder(&mut new_encoder::<C32>(FS), N, &schedule, plan),
        "c32 clock",
    );
    same_bits(
        &want,
        &run_encoder(&mut new_encoder::<CQ15>(FS), N, &schedule, plan),
        "cq15 clock",
    );
    same_bits(
        &want,
        &run_encoder(&mut new_encoder::<Q15>(FS), N, &schedule, plan),
        "q15 clock",
    );
    let mut nan = new_encoder::<f32>(FS);
    let clock = vec![f32::NAN; 1000];
    let mut out = vec![0.0f32; 1000];
    nan.process(
        (
            &clock,
            &[ps_packet(b"SDRK TST"), rt_packet(b"Hello from sdrk!")],
        ),
        &mut out,
    )
    .unwrap();
    same_bits(&want[..1000], &out, "nan clock");
}

/// Test 5: chunk invariance with 20 cut plans (seed 586) and packets at fixed clock positions:
/// the texts at 0, 20 raw groups and an unknown packet at 30000, a new name and radiotext
/// (the A/B flag toggles) at 120000, an empty radiotext at 250000. Every plan gives the bits of
/// the run cut only at the packet positions.
#[test]
fn test5_chunk_invariance_with_packets_at_fixed_positions() {
    const N: usize = 300_000;
    let mut late: Vec<Packet> = random_groups(CHUNK_SEED, 20)
        .into_iter()
        .map(group_packet)
        .collect();
    late.push(packet(b"\x09"));
    let schedule = vec![
        (
            0,
            vec![ps_packet(b"SDRK TST"), rt_packet(b"Hello from sdrk!")],
        ),
        (30_000, late),
        (
            120_000,
            vec![ps_packet(b"NEW NAME"), rt_packet(b"Another radiotext")],
        ),
        (250_000, vec![rt_packet(b"")]),
    ];
    let start = std::time::Instant::now();
    let mut block = new_encoder::<f32>(FS);
    let whole = run_encoder(&mut block, N, &schedule, None);
    assert!(block.carousel().ab_flag());
    assert_eq!(block.carousel().ps(), b"NEW NAME");
    for plan in twenty_plans(CHUNK_SEED) {
        let got = run_encoder(&mut new_encoder::<f32>(FS), N, &schedule, Some(plan));
        same_bits(&whole, &got, &format!("test 5 {plan:?}"));
    }
    eprintln!(
        "test 5: 20 plans of {N} samples in {:.2} s",
        start.elapsed().as_secs_f64()
    );
}

/// After `reset` the encoder sends what a new encoder with the same texts and queued groups
/// sends.
#[test]
fn reset_restarts_the_bit_stream_and_keeps_the_texts() {
    let texts = vec![(
        0,
        vec![ps_packet(b"SDRK TST"), rt_packet(b"Hello from sdrk!")],
    )];
    let mut a = new_encoder::<f32>(FS);
    run_encoder(&mut a, 50_123, &texts, None);
    a.reset();
    assert_eq!(a.next_bit(), 0);
    let after = run_encoder(&mut a, 40_000, &[], None);
    let mut b = new_encoder::<f32>(FS);
    let fresh = run_encoder(&mut b, 40_000, &texts, None);
    same_bits(&fresh, &after, "after reset");
}

/// A short output takes as many clock samples as it has room for.
#[test]
fn a_short_output_limits_the_clock_samples() {
    let mut b = new_encoder::<f32>(FS);
    let mut out = [0.0f32; 10];
    let p = b.process((&[0.0; 100], &[]), &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (10, 10));
    let p = b.process((&[0.0; 5], &[]), &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (5, 5));
}

/// Test 4: the encoder gives the baseband of the Python reference modulator of M41 for the
/// same groups (golden case `rds_encode/cross_check_585`, seed 585, 192 kHz, samples 0..4095
/// and 100000..104095) within the tolerance of the case, 1e-4 (D771; the 256-point table
/// reaches about 7.9e-5, stored as `model_max_abs_err`). The measured error equals that of the
/// float32 model of the generator.
#[test]
fn test4_cross_check_with_the_reference_modulator() {
    let case = golden::load("rds_encode", "cross_check_585").unwrap();
    let fs = case.param_u64("rate_hz").unwrap();
    let late = case.param_u64("late_start").unwrap() as usize;
    let Array::I32(words) = case.array("words").unwrap() else {
        panic!("words")
    };
    let Array::U8(d) = case.array("data_bits").unwrap() else {
        panic!("data_bits")
    };
    let (Array::F32(early_want), Array::F32(late_want)) = (
        case.array("baseband").unwrap(),
        case.array("baseband_late").unwrap(),
    ) else {
        panic!("baseband")
    };
    let Tolerance::MaxAbsErr { max, .. } = case.tolerance else {
        panic!("tolerance")
    };
    let groups: Vec<[u16; 4]> = words
        .chunks_exact(4)
        .map(|w| [w[0] as u16, w[1] as u16, w[2] as u16, w[3] as u16])
        .collect();
    // The bits of the groups (offset C' for version B) are those of the generator.
    let bits: Vec<u8> = groups.iter().flat_map(|g| group_bits(*g)).collect();
    assert_eq!(&bits, d);
    let packets: Vec<Packet> = groups.iter().copied().map(group_packet).collect();
    let n = late + late_want.len();
    let mut block = RdsEncodeBlock::<C32, _>::new(
        &RdsEncodeConfig {
            pi: 0x1234,
            pty: 0,
            tp: false,
        },
        rate(fs),
        &mut VecStorage,
    )
    .unwrap();
    let out = run_encoder(&mut block, n, &[(0, packets)], None);
    let err = |got: &[f32], want: &[f32]| {
        got.iter()
            .zip(want)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0, f64::max)
    };
    let e = err(&out[..early_want.len()], early_want).max(err(&out[late..], late_want));
    eprintln!(
        "test 4: largest error {e:.3e} (tolerance {max:.3e}, specification {:.0e})",
        case.param_f64("spec_max_abs_err").unwrap()
    );
    assert!(e <= max, "{e} > {max}");
    assert_eq!(e, case.param_f64("model_max_abs_err").unwrap());
    // The encoder is the kernel on the bits of the groups.
    same_bits(&shaped(&bits, fs, n), &out, "kernel on the bits");
}

// ---- fm_stereo_rds_mod ---------------------------------------------------------------------

/// Stereo audio of a 1 kHz and a 2 kHz tone plus noise, within full scale.
fn audio(n: usize, seed: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, 3);
    (0..n)
        .map(|i| {
            let t = i as f64 / FS as f64;
            let l = 0.6 * (std::f64::consts::TAU * 1000.0 * t).sin() + 0.1 * rng.normal();
            let r = 0.6 * (std::f64::consts::TAU * 2000.0 * t).sin() + 0.1 * rng.normal();
            C32::new(l.clamp(-1.0, 1.0) as f32, r.clamp(-1.0, 1.0) as f32)
        })
        .collect()
}

/// The RDS baseband of the reference carousel at `FS`.
fn baseband(n: usize) -> Vec<f32> {
    let d: Vec<u8> = (0..bits_fetched(n as u64, FS)).map(data_bit).collect();
    shaped(&d, FS, n)
}

/// Feeds the pair in pieces of `plan`; `change` is applied before the piece that starts at
/// `at` (the pieces are cut there).
fn run_mod(
    block: &mut FmStereoRdsModBlock<VecStorage>,
    a: &[C32],
    b: &[f32],
    plan: Option<ChunkPlan>,
    change: Option<(usize, &dyn Fn(&mut FmStereoRdsModBlock<VecStorage>))>,
) -> Vec<f32> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let mut sizes = plan.map(|p| p.sizes());
    let mut pos = 0;
    while pos < n {
        if let Some((at, f)) = change
            && at == pos
        {
            f(block);
        }
        let limit = change.map_or(n, |(at, _)| if at > pos { at } else { n });
        let piece = sizes
            .as_mut()
            .map_or(limit - pos, |s| s.next().unwrap().min(limit - pos));
        let p = block
            .process(
                (&a[pos..pos + piece], &b[pos..pos + piece]),
                &mut out[pos..pos + piece],
            )
            .unwrap();
        assert_eq!((p.consumed, p.produced), (piece, piece));
        pos += piece;
    }
    out
}

/// `FmStereoRdsModBlock` equals `stereo_encode_rds` with
/// `StereoEncodeRdsState::new(pilot_hz, main, pilot, rds)` (`f32` of the config), whole and in
/// 20 cut plans; runtime changes of `rds` and `pilot` at sample 3000 equal `set_rds` and
/// `set_pilot` there.
#[test]
fn fm_stereo_rds_mod_equals_the_kernel() {
    const N: usize = 8000;
    const AT: usize = 3000;
    let a = audio(N, CHUNK_SEED);
    let b = baseband(N);
    for (cfg, fs) in [
        (FmStereoRdsModConfig::default(), FS),
        (
            FmStereoRdsModConfig {
                pilot_hz: 18_000.0,
                main: 1.0,
                pilot: 0.2,
                rds: 0.2,
            },
            120_000,
        ),
        (
            FmStereoRdsModConfig {
                pilot_hz: 20_000.0,
                main: 0.0,
                pilot: 0.0,
                rds: 0.0,
            },
            1_000_000,
        ),
    ] {
        let state = || {
            StereoEncodeRdsState::new(
                cfg.pilot_hz,
                cfg.main as f32,
                cfg.pilot as f32,
                cfg.rds as f32,
                rate(fs),
            )
            .unwrap()
        };
        let make = || FmStereoRdsModBlock::new(&cfg, rate(fs), &mut VecStorage).unwrap();
        let mut st = state();
        let mut want = vec![0.0f32; N];
        Sdrk.stereo_encode_rds(&mut st, &a, &b, &mut want).unwrap();
        let mut block = make();
        same_bits(&want, &run_mod(&mut block, &a, &b, None, None), "whole");
        assert_eq!(block.phase(), st.phase());
        block.reset();
        assert_eq!(block.phase(), 0);
        for plan in twenty_plans(CHUNK_SEED) {
            let got = run_mod(&mut make(), &a, &b, Some(plan), None);
            same_bits(&want, &got, &format!("{cfg:?} {plan:?}"));
        }
        // Runtime changes.
        let mut st = state();
        let mut want = vec![0.0f32; N];
        let (w0, w1) = want.split_at_mut(AT);
        Sdrk.stereo_encode_rds(&mut st, &a[..AT], &b[..AT], w0)
            .unwrap();
        st.set_rds(0.1).unwrap();
        st.set_pilot(0.05).unwrap();
        Sdrk.stereo_encode_rds(&mut st, &a[AT..], &b[AT..], w1)
            .unwrap();
        let change: &dyn Fn(&mut FmStereoRdsModBlock<VecStorage>) = &|blk| {
            blk.set_param("rds", ParamValue::F64(0.1)).unwrap();
            blk.set_param("pilot", ParamValue::F64(0.05)).unwrap();
        };
        for plan in [None, Some(ChunkPlan::new(CHUNK_SEED, 700))] {
            let got = run_mod(&mut make(), &a, &b, plan, Some((AT, change)));
            same_bits(&want, &got, &format!("{cfg:?} changed {plan:?}"));
        }
    }
}

/// Unequal lengths of the pair are `LengthMismatch` and change nothing; a short output limits
/// the pair.
#[test]
fn fm_stereo_rds_mod_keeps_the_lockstep() {
    let mut m =
        FmStereoRdsModBlock::new(&FmStereoRdsModConfig::default(), rate(FS), &mut VecStorage)
            .unwrap();
    let mut out = [0.0f32; 8];
    assert!(
        m.process((&[C32::default(); 8], &[0.0; 7]), &mut out)
            .is_err()
    );
    assert_eq!(m.phase(), 0);
    let p = m
        .process((&[C32::default(); 8], &[0.0; 8]), &mut out[..3])
        .unwrap();
    assert_eq!((p.consumed, p.produced), (3, 3));
}

/// Extremes: silence and no RDS give 0; full-scale audio and baseband stay within
/// `main + pilot + rds` (1.0 at the defaults); NaN or infinity in either input poisons only its
/// own output sample.
#[test]
fn fm_stereo_rds_mod_extremes() {
    let n = 4000;
    let make = || {
        FmStereoRdsModBlock::new(&FmStereoRdsModConfig::default(), rate(FS), &mut VecStorage)
            .unwrap()
    };
    let silent = vec![C32::default(); n];
    let zeros = vec![0.0f32; n];
    let mut m = make();
    m.set_param("pilot", ParamValue::F64(0.0)).unwrap();
    let out = run_mod(&mut m, &silent, &zeros, None, None);
    assert!(out.iter().all(|&y| y == 0.0));
    // Full scale: the peak is at most 0.87 + 0.09 + 0.04.
    let mut rng = Pcg32::new(CHUNK_SEED, 11);
    let full: Vec<C32> = (0..n)
        .map(|_| {
            let s = |r: &mut Pcg32| if r.below(2) == 0 { 1.0 } else { -1.0 };
            C32::new(s(&mut rng), s(&mut rng))
        })
        .collect();
    let ones: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let out = run_mod(&mut make(), &full, &ones, None, None);
    let peak = out.iter().fold(0.0f32, |p, y| p.max(y.abs()));
    assert!(peak <= 1.0 + 1e-6, "{peak}");
    // NaN and infinity in either input.
    let a = audio(n, CHUNK_SEED);
    let b = baseband(n);
    let clean = run_mod(&mut make(), &a, &b, None, None);
    let mut a2 = a.clone();
    let mut b2 = b.clone();
    a2[100] = C32::new(f32::NAN, 0.0);
    a2[200] = C32::new(0.0, f32::INFINITY);
    b2[300] = f32::NAN;
    b2[400] = f32::NEG_INFINITY;
    let got = run_mod(&mut make(), &a2, &b2, None, None);
    for (i, (x, y)) in got.iter().zip(&clean).enumerate() {
        if [100, 200, 300, 400].contains(&i) {
            assert!(!x.is_finite(), "{i}: {x}");
        } else {
            assert_eq!(x.to_bits(), y.to_bits(), "{i}");
        }
    }
}

/// The peaks of D770: the pulse table has the peak 1, but overlapping pulses take the baseband
/// of `rds_encode` above 1, at most to `max_j sum_k |table[j + 256 k]|` (about 1.058; the
/// interpolated samples lie between grid points, so their sum of magnitudes is within the
/// bound too). With the defaults and full-scale audio the multiplex stays within
/// `0.87 + 0.09 + 0.04 bound` (about 1.0023).
#[test]
fn baseband_and_multiplex_peaks() {
    let mut table = vec![0.0f32; RDS_TABLE_LEN];
    rds_pulse_table(&mut table).unwrap();
    let bound = (0..256)
        .map(|j| {
            (j..RDS_TABLE_LEN)
                .step_by(256)
                .map(|i| f64::from(table[i].abs()))
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    assert!((1.057..1.059).contains(&bound), "{bound}");
    let n = FS as usize;
    let mut enc = new_encoder::<f32>(FS);
    let groups: Vec<Packet> = random_groups(CHUNK_SEED, RAW_QUEUE)
        .into_iter()
        .map(group_packet)
        .collect();
    let b = run_encoder(&mut enc, n, &[(0, groups)], None);
    let peak_b = b.iter().fold(0.0f64, |p, &y| p.max(f64::from(y.abs())));
    assert!(peak_b > 1.0 && peak_b <= bound + 1e-5, "{peak_b}");
    let mut rng = Pcg32::new(CHUNK_SEED, 12);
    let full: Vec<C32> = (0..n)
        .map(|_| {
            let s = |r: &mut Pcg32| if r.below(2) == 0 { 1.0 } else { -1.0 };
            C32::new(s(&mut rng), s(&mut rng))
        })
        .collect();
    let mut m =
        FmStereoRdsModBlock::new(&FmStereoRdsModConfig::default(), rate(FS), &mut VecStorage)
            .unwrap();
    let out = run_mod(&mut m, &full, &b, None, None);
    let peak = out.iter().fold(0.0f64, |p, &y| p.max(f64::from(y.abs())));
    let limit = 0.87 + 0.09 + 0.04 * bound;
    assert!(peak <= limit + 1e-5, "{peak} > {limit}");
}

// ---- test 2: the spectrum ------------------------------------------------------------------

/// In-place radix-2 FFT (decimation in time).
fn fft(x: &mut [Complex64]) {
    let n = x.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -std::f64::consts::TAU / len as f64;
        let w = Complex64::new(ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let mut wk = Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = x[start + k];
                let v = x[start + k + len / 2] * wk;
                x[start + k] = u + v;
                x[start + k + len / 2] = u - v;
                wk *= w;
            }
        }
        len <<= 1;
    }
}

/// Welch power spectrum of a real signal (Hann window, half overlap), bins `0 ..= seg / 2`.
fn welch(x: &[f64], seg: usize) -> Vec<f64> {
    let win: Vec<f64> = (0..seg)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / seg as f64).cos())
        .collect();
    let mut psd = vec![0.0; seg / 2 + 1];
    let mut buf = vec![Complex64::new(0.0, 0.0); seg];
    let mut start = 0;
    let mut count = 0;
    while start + seg <= x.len() {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex64::new(x[start + i] * win[i], 0.0);
        }
        fft(&mut buf);
        for (p, b) in psd.iter_mut().zip(&buf) {
            *p += b.norm_sqr();
        }
        count += 1;
        start += seg / 2;
    }
    psd.iter().map(|p| p / f64::from(count)).collect()
}

/// Test 2: the RDS component of the multiplex (the multiplex with `rds` 0.04 minus the one with
/// `rds` 0, both of the same stereo audio with the pilot) lies within 57 kHz +- 2.4 kHz: every
/// bin outside is at least 40 dB below the peak, and the bin at exactly 57 kHz is at least 30
/// dB below it (biphase coding has no carrier). 3 s at 192 kHz of random groups (seed 582),
/// Welch with 16384-point Hann segments (11.7 Hz bins).
#[test]
fn test2_the_rds_spectrum_lies_around_57_khz() {
    const N: usize = 3 * FS as usize;
    const SEG: usize = 16_384;
    let groups: Vec<Packet> = random_groups(SPECTRUM_SEED, 300)
        .into_iter()
        .map(group_packet)
        .collect();
    // The groups arrive 16 at a time as the queue drains; the rest waits in the packet queue.
    let mut enc = new_encoder::<C32>(FS);
    let rds = run_encoder(
        &mut enc,
        N,
        &[(0, groups)],
        Some(ChunkPlan::new(SPECTRUM_SEED, 4096)),
    );
    let a = audio(N, SPECTRUM_SEED);
    let with = run_mod(
        &mut FmStereoRdsModBlock::new(&FmStereoRdsModConfig::default(), rate(FS), &mut VecStorage)
            .unwrap(),
        &a,
        &rds,
        None,
        None,
    );
    let without = run_mod(
        &mut FmStereoRdsModBlock::new(
            &FmStereoRdsModConfig {
                rds: 0.0,
                ..FmStereoRdsModConfig::default()
            },
            rate(FS),
            &mut VecStorage,
        )
        .unwrap(),
        &a,
        &rds,
        None,
        None,
    );
    let component: Vec<f64> = with
        .iter()
        .zip(&without)
        .map(|(x, y)| f64::from(*x) - f64::from(*y))
        .collect();
    let psd = welch(&component, SEG);
    let bin_hz = FS as f64 / SEG as f64;
    let freq = |k: usize| k as f64 * bin_hz;
    let (k_peak, peak) =
        psd.iter().copied().enumerate().fold(
            (0, 0.0),
            |(kb, pb), (k, p)| if p > pb { (k, p) } else { (kb, pb) },
        );
    let db = |p: f64| 10.0 * (p / peak).log10();
    assert!(
        (freq(k_peak) - 57_000.0).abs() <= 2_400.0,
        "peak at {} Hz",
        freq(k_peak)
    );
    let outside = psd
        .iter()
        .enumerate()
        .filter(|(k, _)| (freq(*k) - 57_000.0).abs() > 2_400.0)
        .map(|(k, p)| (db(*p), freq(k)))
        .fold(
            (f64::NEG_INFINITY, 0.0),
            |a, b| if b.0 > a.0 { b } else { a },
        );
    let k57 = (57_000.0 / bin_hz).round() as usize;
    assert_eq!(freq(k57), 57_000.0);
    let at57 = db(psd[k57]);
    eprintln!(
        "test 2: peak at {:.1} Hz; outside 57 kHz +- 2.4 kHz at most {:.1} dB (at {:.1} Hz); at 57 kHz {:.1} dB",
        freq(k_peak),
        outside.0,
        outside.1,
        at57
    );
    assert!(outside.0 <= -40.0, "{outside:?}");
    assert!(at57 <= -30.0, "{at57}");
}

/// The arena of the doc example of `RdsEncodeBlock` is large enough (4112 bytes, 16 aligned).
#[test]
fn the_documented_arena_fits() {
    let mut mem = Aligned::new(4112);
    let mut arena = ArenaStorage::new(mem.bytes(4112));
    RdsEncodeBlock::<f32, _>::new(&encoder_config(), rate(FS), &mut arena).unwrap();
}
