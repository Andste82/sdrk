//! The blocks of M41: `RdsDemodBlock` (the composition `interleave` -> `rotate` ->
//! `fir_decimate` -> `symbol_sync` -> `costas` -> biphase and differential decoding, equal to
//! the kernel chain whole and in 20 cut plans of seed 575, test 5) and `RdsDecodeBlock` (block
//! synchronization, the packets and their flags, the loss of synchronization, 20 cut plans of
//! seed 575, and test 4: injected bit errors and a dropout with a bit slip, seed 574), the
//! timing, rates and parameters, the output room and the extremes (NaN, infinity, silence,
//! full scale). The multiplex is the reference signal of `sdrk_testkit::rds` (the Rust model
//! of `rds.py`).
#![allow(
    clippy::approx_constant,
    clippy::assert_is_empty,
    clippy::many_single_char_names,
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

use common::{check_block, check_exact_storage, rate, run_block, same_bits, twenty_plans};
use sdrk::{C32, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, DesignSpec, Error, Packet, PacketFlags, ParamValue, RdsDecodeBlock,
    RdsDecodeConfig, RdsDemodBlock, RdsDemodConfig, SymbolSyncConfig,
};
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::math::Interleave;
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted};
use sdrk_device::Rate;
use sdrk_proto::rds::{self, Group, Offset, RdsState, VERSION_B};
use sdrk_testkit::rds::{RdsReference, TABLE_LEN, carousel_group, data_bit};
use sdrk_testkit::{ChunkPlan, Pcg32, VecStorage};

/// Seed of the chunk-invariance tests of M41 (test 5) and of their noise.
const CHUNK_SEED: u64 = 575;
/// Seed of the bit-error and dropout test of M41 (test 4).
const ERROR_SEED: u64 = 574;
/// The bit rate of RDS.
const BIT_RATE: f64 = 1187.5;

/// `n` samples of the reference multiplex at 240 kHz (RDS level 4 %), plus Gaussian noise of
/// the standard deviation `sigma` (stream `stream` of the chunk seed).
fn reference_mpx(n: usize, sigma: f64, stream: u64) -> Vec<f32> {
    let r = RdsReference::new(vec![0.0; TABLE_LEN], 0.04).unwrap();
    let mut rng = Pcg32::new(CHUNK_SEED, stream);
    (0..n as u64)
        .map(|m| (r.mpx(m) + sigma * rng.normal()) as f32)
        .collect()
}

/// The decimation `floor(fs / 19000)`.
fn decimation(fs: Rate) -> usize {
    (fs.as_f64() / 19_000.0).floor() as usize
}

/// The kernels of §2, each over the whole input, and the biphase and differential decoding
/// written out independently.
fn demod_kernels(cfg: &RdsDemodConfig, fs: Rate, input: &[f32]) -> Vec<f32> {
    let d = decimation(fs);
    let spec = DesignSpec {
        cutoff_hz: 2400.0,
        transition_hz: 1500.0,
        attenuation_db: 60.0,
    };
    let n = spec.num_taps(fs.as_f64(), 65_536).unwrap();
    let mut h = vec![0.0f64; n];
    spec.design(fs.as_f64(), 65_536, &mut h).unwrap();
    let taps: Vec<f32> = h.iter().map(|t| (t * 50.0) as f32).collect();
    let mut x = vec![C32::default(); input.len()];
    <Sdrk as Interleave<C32>>::interleave(&Sdrk, input, None, &mut x).unwrap();
    let mut nco = Nco::new(-57_000.0, fs).unwrap();
    let mut mixed = vec![C32::default(); input.len()];
    <Sdrk as Rotate<C32>>::rotate(&Sdrk, &mut nco, &x, &mut mixed).unwrap();
    let mut fir = FirDecimState::<C32, _, _>::new(taps, vec![C32::default(); n - 1], d).unwrap();
    let mut dec = vec![C32::default(); input.len() / d + 2];
    let p = <Sdrk as FirDecimate<C32>>::fir_decimate(&Sdrk, &mut fir, &mixed, &mut dec).unwrap();
    assert_eq!(p.consumed, input.len());
    dec.truncate(p.produced);
    let rate_dec = fs.scale(Ratio::new(1, d as u64).unwrap()).unwrap();
    let sps = SymbolSyncConfig {
        symbol_rate_hz: 2375.0,
        ..SymbolSyncConfig::default()
    }
    .sps(rate_dec)
    .unwrap();
    let mut sync =
        SymbolSyncState::<C32>::new(sps, Ted::Gardner, cfg.sync_bw, 0.7071, 0.005, 1.0).unwrap();
    let mut chips = vec![C32::default(); dec.len()];
    let q = <Sdrk as SymbolSync<C32>>::symbol_sync(&Sdrk, &mut sync, &dec, &mut chips).unwrap();
    assert_eq!(q.consumed, dec.len());
    chips.truncate(q.produced);
    let mut st = CostasState::new(2, cfg.carrier_bw, 0.7071, 0.05).unwrap();
    let mut y = vec![C32::default(); chips.len()];
    <Sdrk as Costas>::costas(&Sdrk, &mut st, &chips, &mut y).unwrap();
    // biphase: chip j >= 1 completes a pair of pairing (j - 1) mod 2
    let mut level = [0.0f32; 2];
    let mut prev_u = [0.0f32; 2];
    let mut selected = 0usize;
    let mut soft = Vec::new();
    for j in 1..y.len() {
        let k = (j - 1) % 2;
        let u = (y[j - 1].re - y[j].re) / 2.0;
        level[k] += (u.abs() - level[k]) * (1.0 / 32.0);
        let s = -u * prev_u[k];
        prev_u[k] = u;
        if k == selected {
            soft.push(s);
        }
        if level[1 - selected] > level[selected] * 1.2 {
            selected = 1 - selected;
        }
    }
    soft
}

fn demod(cfg: &RdsDemodConfig, fs: Rate) -> RdsDemodBlock<VecStorage> {
    RdsDemodBlock::<VecStorage>::new(cfg, fs, &mut VecStorage).unwrap()
}

fn decoder(cfg: &RdsDecodeConfig) -> RdsDecodeBlock<VecStorage> {
    RdsDecodeBlock::<VecStorage>::new(cfg, rate(1187), &mut VecStorage).unwrap()
}

/// The lag (-16 to 16 bits) at which the hard bits of `soft` equal the data bits of the
/// carousel from soft bit `from` on (`soft[i]` is data bit `i - lag`), and the number of
/// errors there.
fn best_lag(soft: &[f32], from: usize) -> (i64, usize) {
    (-16i64..16)
        .map(|lag| {
            let errors = (from..soft.len())
                .filter(|&i| u8::from(soft[i] > 0.0) != data_bit((i as i64 - lag) as u64))
                .count();
            (lag, errors)
        })
        .min_by_key(|&(_, e)| e)
        .unwrap()
}

// ---- rds_demod ---------------------------------------------------------------------------

/// Test 5 for `rds_demod`: the block equals the kernel chain over the whole input, and every
/// one of 20 cut plans (seed 575) gives the same bits; at 240 kHz (the reference with noise)
/// and at 128 kHz (d = 6, a fractional 64000 / 3 Hz after the decimation). Both rates use the
/// same 20 plans; their inputs differ (the reference, and stream 2 of seed 575).
#[test]
fn rds_demod_equals_the_kernels_whole_and_in_twenty_plans() {
    let cfg = RdsDemodConfig::default();
    let fs = rate(240_000);
    let input = reference_mpx(120_000, 0.05, 1);
    let want = demod_kernels(&cfg, fs, &input);
    assert!((580..=600).contains(&want.len()), "{}", want.len());
    check_block(
        &want,
        &input,
        CHUNK_SEED,
        || demod(&cfg, fs),
        "rds_demod 240 kHz",
    );

    let fs = rate(128_000);
    let other = RdsDemodConfig {
        sync_bw: 0.02,
        carrier_bw: 0.03,
    };
    let mut rng = Pcg32::new(CHUNK_SEED, 2);
    let input: Vec<f32> = (0..64_000)
        .map(|n| {
            let chip = if (n * 2375 / 128_000) % 3 == 0 {
                1.0
            } else {
                -1.0
            };
            let phase = 2.0 * core::f64::consts::PI * 57_000.0 * n as f64 / 128_000.0;
            (0.04 * chip * phase.sin() + 0.4 * (0.05 * n as f64).sin() + 0.02 * rng.normal()) as f32
        })
        .collect();
    let want = demod_kernels(&other, fs, &input);
    assert!(want.len() > 550, "{}", want.len());
    check_block(
        &want,
        &input,
        CHUNK_SEED,
        || demod(&other, fs),
        "rds_demod 128 kHz",
    );
}

/// The clean reference multiplex is demodulated without a bit error once the loops settled,
/// at the bit position of the carousel (lag 0: the first soft bit is that of data bit 0); the
/// carrier loop reports lock.
#[test]
fn rds_demod_recovers_the_bits_of_the_reference() {
    let fs = rate(240_000);
    let input = reference_mpx(480_000, 0.0, 3);
    let mut block = demod(&RdsDemodConfig::default(), fs);
    let soft = run_block(&mut block, &input, None);
    assert!((2370..=2380).contains(&soft.len()), "{}", soft.len());
    assert_eq!(best_lag(&soft, 200), (0, 0));
    assert!(block.locked());
    // the confidence of a clean bit is about 1 (the chips of the nominal level, D670)
    let mean = soft[200..].iter().map(|s| s.abs()).sum::<f32>() / (soft.len() - 200) as f32;
    assert!((0.9..1.1).contains(&mean), "{mean}");
    assert!((block.sps_estimate() - 240_000.0 / 12.0 / 2375.0).abs() < 0.01);
}

/// Soft bits that do not fit into the output wait for the next call; with an output room of
/// 1, 2 and 7 bits the stream is the same as with a large room; without room nothing is
/// consumed.
#[test]
fn rds_demod_keeps_soft_bits_that_do_not_fit() {
    let fs = rate(240_000);
    let cfg = RdsDemodConfig::default();
    let input = reference_mpx(60_000, 0.02, 4);
    let want = demod_kernels(&cfg, fs, &input);
    let mut block = demod(&cfg, fs);
    for room_len in [1usize, 2, 7] {
        block.reset();
        let mut got = Vec::new();
        let mut rest: &[f32] = &input;
        let mut room = vec![0.0f32; room_len];
        loop {
            let p = block.process(rest, &mut room).unwrap();
            assert!(p.produced <= room_len);
            got.extend_from_slice(&room[..p.produced]);
            rest = &rest[p.consumed..];
            if p.consumed == 0 && p.produced == 0 {
                break;
            }
        }
        assert!(rest.is_empty());
        same_bits(&want, &got, &format!("room {room_len}"));
    }
    block.reset();
    let p = block.process(&input, &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

/// Decimation, taps, ratio, delay, rates, the rate limit of 128 kHz, the ranges and the
/// read-only parameters; the storage plan is exact.
#[test]
fn rds_demod_timing_rates_and_parameters() {
    let cfg = RdsDemodConfig::default();
    let fs = rate(240_000);
    let block = demod(&cfg, fs);
    assert_eq!((block.decimation(), block.num_taps()), (12, 583));
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(19, 3840).unwrap()); // 1187.5 / 240000
    assert_eq!(t.kind, RatioKind::Variable);
    assert_eq!(t.delay, Delay::new(582 + 24, 2).unwrap()); // 291 + d
    let chip_min = (240_000.0f64 / 12.0 / 2375.0 * 1.005 * 1.25).ceil() as usize + 3;
    assert_eq!(block.min_input(), 2 * chip_min * 12);
    assert_eq!(
        RdsDemodBlock::<VecStorage>::rate_out(&cfg, fs).unwrap(),
        Rate::new(2375, 2).unwrap()
    );
    assert!(!block.locked());
    assert_eq!(block.pairing(), 0);
    for (hz, d) in [
        (128_000u64, 6usize),
        (250_000, 13),
        (480_000, 25),
        (1_000_000, 52),
    ] {
        let b = demod(&cfg, rate(hz));
        assert_eq!(b.decimation(), d, "{hz}");
        assert_eq!(
            b.timing().ratio,
            Ratio::new(2375, 2 * hz).unwrap(),
            "{hz}: the ratio is 1187.5 / fs"
        );
        assert_eq!(
            RdsDemodBlock::<VecStorage>::rate_out(&cfg, rate(hz)).unwrap(),
            Rate::new(2375, 2).unwrap()
        );
    }
    // a fractional rate: d from the exact value
    let frac = Rate::new(384_000_001, 3000).unwrap(); // 128000.0003 Hz
    assert_eq!(cfg.decimation(frac).unwrap(), 6);
    for hz in [127_999u64, 96_000, 48_000] {
        match RdsDemodBlock::<VecStorage>::new(&cfg, rate(hz), &mut VecStorage) {
            Err(Error::OutOfRange { name, min, .. }) => {
                assert_eq!((name, min), ("sample_rate", 128_000.0));
            }
            other => panic!("{hz}: {other:?}"),
        }
    }
    for bad in [
        RdsDemodConfig {
            sync_bw: 0.2,
            ..cfg
        },
        RdsDemodConfig {
            sync_bw: f64::NAN,
            ..cfg
        },
        RdsDemodConfig {
            carrier_bw: 0.3,
            ..cfg
        },
        RdsDemodConfig {
            carrier_bw: 0.0,
            ..cfg
        },
    ] {
        assert!(
            matches!(bad.validate(fs), Err(Error::OutOfRange { .. })),
            "{bad:?}"
        );
    }
    let mut block = demod(
        &RdsDemodConfig {
            sync_bw: 0.02,
            carrier_bw: 0.005,
        },
        fs,
    );
    assert_eq!(block.param("sync_bw"), Ok(ParamValue::F64(0.02)));
    assert_eq!(block.param("carrier_bw"), Ok(ParamValue::F64(0.005)));
    assert_eq!(
        block.set_param("sync_bw", ParamValue::F64(0.01)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(block.param("nope"), Err(Error::UnknownParameter));
    let mut c = RdsDemodConfig::default();
    assert_eq!(
        c.set("sync_bw", ParamValue::Bool(true)),
        Err(Error::WrongParamType)
    );
    c.set("sync_bw", ParamValue::F64(0.05)).unwrap();
    assert_eq!(c.sync_bw, 0.05);
    for hz in [128_000u64, 240_000, 250_000] {
        let bytes = RdsDemodBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate(hz)).unwrap();
        check_exact_storage(bytes, |arena| {
            RdsDemodBlock::new(&cfg, rate(hz), arena).map(|_| ())
        });
    }
}

/// Silence gives zero soft bits; NaN, infinity and full-scale values in the middle of the
/// reference do not panic, the bits well after the burst are right again, and `reset` gives
/// the output of a new block.
#[test]
fn rds_demod_extremes_nan_infinity_silence_and_full_scale() {
    let fs = rate(240_000);
    let cfg = RdsDemodConfig::default();
    let mut block = demod(&cfg, fs);
    let silence = run_block(&mut block, &vec![0.0f32; 48_000], None);
    assert!(silence.len() > 200);
    assert!(silence.iter().all(|s| *s == 0.0));
    assert!(!block.locked());

    let mut input = reference_mpx(480_000, 0.0, 5);
    let mid = 120_000;
    for (k, x) in input[mid..mid + 400].iter_mut().enumerate() {
        *x = match k % 4 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            _ => 1.0,
        };
    }
    for x in &mut input[mid + 1000..mid + 3000] {
        *x = if *x > 0.0 { 1.0 } else { -1.0 }; // a full-scale square burst
    }
    block.reset();
    let soft = run_block(&mut block, &input, None);
    assert!(soft.len() > 2300);
    let after = ((mid + 3000) as f64 / 240_000.0 * BIT_RATE) as usize + 400;
    let (lag, errors) = best_lag(&soft, after);
    assert_eq!(errors, 0, "lag {lag}");
    assert!(soft[after..].iter().all(|s| s.is_finite()));
    // reset recovers completely
    block.reset();
    let clean = reference_mpx(120_000, 0.0, 6);
    let again = run_block(&mut block, &clean, None);
    let fresh = run_block(&mut demod(&cfg, fs), &clean, None);
    same_bits(&fresh, &again, "after reset");
}

// ---- rds_decode --------------------------------------------------------------------------

/// The soft bits (`+-1`) of the information words with the offsets of their positions.
fn group_bits(words: [u16; 4], third: Offset) -> Vec<f32> {
    let mut out = Vec::new();
    for (w, o) in words.iter().zip([Offset::A, Offset::B, third, Offset::D]) {
        let block = rds::block26(*w, o);
        out.extend(
            (0..26)
                .rev()
                .map(|i| if (block >> i) & 1 == 1 { 1.0f32 } else { -1.0 }),
        );
    }
    out
}

/// The data bits `start..start + n` of the carousel as soft bits.
fn carousel_bits(start: u64, n: usize) -> Vec<f32> {
    (start..start + n as u64)
        .map(|k| if data_bit(k) == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// Feeds `input` in pieces of the input plan and with output rooms of the output plan (`None`:
/// all at once, room 64) and returns every packet.
fn run_decoder(
    dec: &mut RdsDecodeBlock<VecStorage>,
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

/// A clean carousel from bit 0: the first two blocks synchronize, every group is output with
/// `CRC_OK`, all four validity bits, no `VERSION_B`, the index of its block A and the words of
/// the carousel; `RdsState` assembles PS and RT from the packets.
#[test]
fn rds_decode_outputs_the_groups_of_a_clean_carousel() {
    let bits = carousel_bits(0, 104 * 40);
    let mut dec = decoder(&RdsDecodeConfig::default());
    assert!(!dec.synchronized());
    let packets = run_decoder(&mut dec, &bits, None, None);
    assert!(dec.synchronized());
    assert_eq!(dec.position(), bits.len() as u64);
    assert_eq!(packets.len(), 40);
    let mut state = RdsState::new();
    for (g, p) in packets.iter().enumerate() {
        assert_eq!(p.index, 104 * g as u64);
        assert_eq!(p.flags, PacketFlags(0xF << 16) | PacketFlags::CRC_OK);
        let group = Group::from_packet(p).unwrap();
        assert_eq!(group, Group::new(carousel_group(g as u64)));
        state.update(&group);
    }
    assert_eq!(state.ps(), "SDRK TST");
    assert_eq!(state.rt(), "Hello from sdrk!");
    // reset starts again: the same packets
    dec.reset();
    assert_eq!(dec.position(), 0);
    assert_eq!(run_decoder(&mut dec, &bits, None, None), packets);
}

/// A stream that starts inside a group: the partial group after the synchronization is not
/// output, the first packet is the next group with its block A; a stream that starts with
/// block D synchronizes on D and A and outputs that group.
#[test]
fn rds_decode_drops_the_partial_group_after_synchronization() {
    // from block B of group 3: synchronized on B and C, group 3 is dropped
    let bits = carousel_bits(3 * 104 + 26, 104 * 5);
    let packets = run_decoder(&mut decoder(&RdsDecodeConfig::default()), &bits, None, None);
    assert_eq!(packets.len(), 4);
    assert_eq!(packets[0].index, 104 - 26);
    assert_eq!(
        Group::from_packet(&packets[0]).unwrap(),
        Group::new(carousel_group(4))
    );
    // from block D of group 3: synchronized on D and A of group 4
    let bits = carousel_bits(3 * 104 + 78, 104 * 2);
    let packets = run_decoder(&mut decoder(&RdsDecodeConfig::default()), &bits, None, None);
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].index, 26);
    // shifted by an arbitrary number of bits: found at every phase
    for skip in [1u64, 7, 13, 25] {
        let bits = carousel_bits(skip, 104 * 4);
        let packets = run_decoder(&mut decoder(&RdsDecodeConfig::default()), &bits, None, None);
        assert_eq!(packets.len(), 3, "skip {skip}");
        assert_eq!(packets[0].index, 104 - skip, "skip {skip}");
    }
}

/// Version B groups (offset C') carry `VERSION_B`; an invalid block gives a `CRC_FAIL` packet
/// with its validity bit cleared only with `emit_bad`; the flag of C' needs the C' syndrome.
#[test]
fn rds_decode_flags_version_b_and_bad_blocks() {
    let a = [0xD123, 0x0148, 0xE0E0, u16::from_be_bytes(*b"SD")];
    let b = [0xD123, 0x0948, 0xD123, u16::from_be_bytes(*b"RK")]; // 0B, segment 0
    let mut bits = group_bits(a, Offset::C);
    bits.extend(group_bits(b, Offset::CPrime));
    let mut bad = group_bits(a, Offset::C);
    bad[26 + 5] = -bad[26 + 5]; // block B of the third group is invalid
    bits.extend(&bad);
    bits.extend(group_bits(b, Offset::CPrime));
    let good = run_decoder(&mut decoder(&RdsDecodeConfig::default()), &bits, None, None);
    assert_eq!(good.len(), 3);
    assert_eq!(good[1].flags.bits(), (0xF << 16) | VERSION_B | 1);
    assert_eq!(Group::from_packet(&good[1]).unwrap(), Group::new(b));
    assert_eq!(good[2].index, 3 * 104);
    let all = run_decoder(
        &mut decoder(&RdsDecodeConfig {
            emit_bad: true,
            max_bad: 10,
        }),
        &bits,
        None,
        None,
    );
    assert_eq!(all.len(), 4);
    assert_eq!(all[2].index, 2 * 104);
    assert_eq!(
        all[2].flags.bits(),
        (0b1101 << 16) | PacketFlags::CRC_FAIL.bits()
    );
    let g = Group::from_packet(&all[2]).unwrap();
    assert_eq!(g.valid, [true, false, true, true]);
    assert_eq!(g.blocks[0], 0xD123);
}

/// More than `max_bad` invalid blocks among the last 50 end the synchronization: with
/// `max_bad` 10 the 11th invalid block, with 0 the first one; with 50 never.
#[test]
fn rds_decode_loses_synchronization_after_max_bad_invalid_blocks() {
    let mut rng = Pcg32::new(ERROR_SEED, 9);
    let noise: Vec<f32> = (0..26 * 60)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect();
    // the random blocks are invalid at their positions (checked for this seed)
    for (max_bad, lost_after) in [(10usize, Some(11usize)), (0, Some(1)), (50, None)] {
        let mut dec = decoder(&RdsDecodeConfig {
            emit_bad: false,
            max_bad,
        });
        run_decoder(&mut dec, &carousel_bits(0, 104 * 3), None, None);
        assert!(dec.synchronized());
        for k in 1..=60 {
            run_decoder(&mut dec, &noise[(k - 1) * 26..k * 26], None, None);
            let expect = lost_after.is_none_or(|n| k < n);
            if k <= lost_after.unwrap_or(60) {
                assert_eq!(dec.synchronized(), expect, "max_bad {max_bad}, block {k}");
            }
        }
    }
}

/// When a call fills its last output slot it stops after the bit that completed the packet;
/// without output room nothing is consumed.
#[test]
fn rds_decode_stops_when_the_output_is_full() {
    let bits = carousel_bits(0, 104 * 4);
    let mut dec = decoder(&RdsDecodeConfig::default());
    let mut one = [Packet::default(); 1];
    let p = dec.process(&bits, &mut one).unwrap();
    assert_eq!((p.consumed, p.produced), (104, 1));
    let p = dec.process(&bits[104..], &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    let p = dec.process(&bits[104..], &mut one).unwrap();
    assert_eq!((p.consumed, p.produced), (104, 1));
    assert_eq!(one[0].index, 104);
    assert_eq!(dec.position(), 208);
}

/// The parameters: ranges, defaults, read-only, the timing of a packet decoder, no storage.
#[test]
fn rds_decode_parameters_and_timing() {
    let cfg = RdsDecodeConfig {
        emit_bad: true,
        max_bad: 7,
    };
    let mut dec = decoder(&cfg);
    assert_eq!(dec.param("emit_bad"), Ok(ParamValue::Bool(true)));
    assert_eq!(dec.param("max_bad"), Ok(ParamValue::I64(7)));
    assert_eq!(
        dec.set_param("max_bad", ParamValue::I64(3)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(dec.param("nope"), Err(Error::UnknownParameter));
    let t = dec.timing();
    assert_eq!(t.ratio, Ratio::new(1, 104).unwrap());
    assert_eq!(
        t.kind,
        RatioKind::Packets {
            max_per_input: Ratio::new(1, 104).unwrap()
        }
    );
    assert_eq!(t.delay, Delay::ZERO);
    assert_eq!(dec.min_input(), 1);
    let mut c = RdsDecodeConfig::default();
    assert!(matches!(
        c.set("max_bad", ParamValue::I64(51)),
        Err(Error::OutOfRange { .. })
    ));
    c.set("max_bad", ParamValue::I64(50)).unwrap();
    c.set("emit_bad", ParamValue::Bool(true)).unwrap();
    assert_eq!(
        c,
        RdsDecodeConfig {
            emit_bad: true,
            max_bad: 50
        }
    );
    assert!(
        RdsDecodeBlock::<VecStorage>::new(
            &RdsDecodeConfig { max_bad: 51, ..cfg },
            rate(1187),
            &mut VecStorage
        )
        .is_err()
    );
    assert_eq!(
        RdsDecodeBlock::<VecStorage>::storage_bytes(&cfg, rate(1187)).unwrap(),
        0
    );
    // NaN and infinities are bits (NaN and -inf a 0)
    let mut odd = carousel_bits(0, 208);
    for x in &mut odd {
        *x = if *x > 0.0 { f32::INFINITY } else { f32::NAN };
    }
    assert_eq!(run_decoder(&mut decoder(&cfg), &odd, None, None).len(), 2);
}

/// Test 5 for `rds_decode`: soft bits with errors (every 200th bit inverted on average, plus noise) give
/// the same packets, `CRC_FAIL` ones included, whole and in 20 plans of input pieces and output
/// rooms (seed 575: the input plans are `twenty_plans(575)`, plan `i` gets the output plan of
/// seed `575 * 1000 + i`, the same derivation from the base seed as `seed * 100 + i`).
#[test]
fn rds_decode_is_chunk_invariant_with_twenty_plans() {
    let mut rng = Pcg32::new(CHUNK_SEED, 10);
    let bits: Vec<f32> = carousel_bits(5, 104 * 60)
        .into_iter()
        .map(|b| {
            let flip = rng.next_u32().is_multiple_of(200);
            let noise = 0.3 * rng.normal() as f32;
            if flip { -b + noise } else { b + noise }
        })
        .collect();
    let cfg = RdsDecodeConfig {
        emit_bad: true,
        max_bad: 20,
    };
    let whole = run_decoder(&mut decoder(&cfg), &bits, None, None);
    let ok = whole
        .iter()
        .filter(|p| p.flags.contains(PacketFlags::CRC_OK))
        .count();
    assert!(
        whole.len() > 50 && ok > 10 && ok < whole.len(),
        "{} {ok}",
        whole.len()
    );
    for (i, plan) in twenty_plans(CHUNK_SEED).into_iter().enumerate() {
        let out_plan = ChunkPlan::new(CHUNK_SEED * 1000 + i as u64, 1 + i % 4);
        let got = run_decoder(&mut decoder(&cfg), &bits, Some(plan), Some(out_plan));
        assert_eq!(got, whole, "plan {plan:?}");
    }
}

/// Test 4 (seed 574): 20 s of carousel bits with a bit error rate of 1e-3 and, after 10 s, a
/// dropout of 1 s of random bits that also slips the stream by one bit. No `CRC_OK` packet
/// holds a wrong block (every one is the carousel group of its position); the decoder
/// synchronizes again and outputs a `CRC_OK` group within 0.2 s after the dropout (237 bits,
/// counted to the last bit of that group); at least 85 % of the groups outside the dropout are
/// `CRC_OK` (a group of 104 bits is free of errors with the probability `0.999^104 = 0.90`).
#[test]
#[allow(clippy::print_stdout)]
fn rds_decode_survives_bit_errors_and_a_dropout_with_a_bit_slip() {
    let mut rng = Pcg32::new(ERROR_SEED, 1);
    let total = (20.0 * BIT_RATE) as usize;
    let drop_at = (10.0 * BIT_RATE) as usize;
    let drop_len = BIT_RATE as usize; // 1187 bits
    let resume = drop_at + drop_len;
    // the data bit index of stream position i: one bit is lost at the end of the dropout
    let source = |i: usize| -> Option<u64> {
        if i < drop_at {
            Some(i as u64)
        } else if i < resume {
            None
        } else {
            Some(i as u64 + 1)
        }
    };
    let mut errors = 0usize;
    let soft: Vec<f32> = (0..total)
        .map(|i| match source(i) {
            Some(k) => {
                let b = data_bit(k) == 1;
                let flip = rng.next_f64() < 1e-3;
                errors += usize::from(flip);
                if b == flip { -1.0 } else { 1.0 }
            }
            None => {
                if rng.next_u32() & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            }
        })
        .collect();
    let ber = errors as f64 / (total - drop_len) as f64;
    assert!((0.5e-3..2e-3).contains(&ber), "{ber}");
    let packets = run_decoder(&mut decoder(&RdsDecodeConfig::default()), &soft, None, None);
    let mut first_after = None;
    let mut ok_outside = 0usize;
    for p in &packets {
        if !p.flags.contains(PacketFlags::CRC_OK) {
            continue;
        }
        let i = p.index as usize;
        let k = source(i).filter(|_| source(i + 103).is_some());
        let Some(k) = k else {
            panic!("a CRC_OK packet at {i} overlaps the dropout");
        };
        assert_eq!(k % 104, 0, "a CRC_OK packet at {i} is not at a group start");
        let g = Group::from_packet(p).unwrap();
        assert_eq!(g.blocks, carousel_group(k / 104), "wrong block at {i}");
        ok_outside += 1;
        if i >= resume && first_after.is_none() {
            first_after = Some(i + 104 - resume);
        }
    }
    let resync = first_after.unwrap();
    assert!(resync <= 237, "resynchronized after {resync} bits");
    let groups = (total - drop_len) / 104;
    assert!(
        ok_outside as f64 >= 0.85 * groups as f64,
        "{ok_outside} of {groups}"
    );
    println!(
        "test 4: ber {ber:.2e}, {ok_outside} of {groups} groups CRC_OK, first CRC_OK group \
         complete {resync} bits after the dropout"
    );
}
