//! M44.1 test 9 (`qam_frame_decode`, seed 850) and its part of test 11 (seed 851):
//! `qam_frame_encode` (`idle` `pn9`) -> carrier offset 0.001 cycles per symbol -> noise ->
//! `qam_frame_decode` with the defaults of `qam_demod` (`equalizer_taps` 11) and `frame_timeout`.
//!
//! - **A gap in the middle of a transmission** (seed 850): the packets flow without a pause; over
//!   [`GAP`] symbols in the middle the signal is replaced by loud noise (at the signal's power, an
//!   AGC that turned the noise up). The frame watch starts the demodulator again
//!   `frame_timeout` + 256 symbols after the last frame with `CRC_OK`, and from
//!   `frame_timeout` + [`QAM_RECOVERY_SYMBOLS`] symbols after the end of the gap on at least
//!   [`REQUIRED`] of the packets arrive with `CRC_OK` (16-QAM at Eb/N0 16 dB, the level of the
//!   flow test of M44, and 64-QAM at Es/N0 30 dB, the gap case of I44-17).
//! - **`frame_timeout` 0** gives the packets of the pair `qam_demod` -> `frame_decode`, with the
//!   index moved to the input samples, on the signal of the gap (where the watch acts).
//! - **Chunk invariance** (seed 851, test 11): the packets and the restarts of the watch in 20
//!   random cut plans of input and packet room.
//!
//! The measured values are printed (run with `--nocapture`).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::large_stack_frames
)]

use num_complex::Complex64;
use sdrk::Rate;
use sdrk_blocks::SymbolMapConstellation;
use sdrk_blocks::{
    Block, FrameDecodeBlock, FrameIdle, Packet, QAM_FRAME_TIMEOUT, QAM_FRAME_WATCH_LAG,
    QamDemodBlock, QamFrameDecodeBlock, QamFrameDecodeConfig, QamFrameEncodeBlock,
    QamFrameEncodeConfig,
};
use sdrk_core::C32;
use sdrk_proto::PacketFlags;
use sdrk_testkit::channel::{Awgn, Impairment};
use sdrk_testkit::{Pcg32, VecStorage};

/// Samples per symbol and the symbol rate (the sample rate is `SPS` times it).
const SPS: usize = 4;
const SYMBOL_RATE: u64 = 10_000;
/// The carrier offset in cycles per symbol.
const OFFSET: f64 = 0.001;
/// Idle symbols before the first packet (the acquisition and the blind equalizer).
const LEAD: usize = 20_000;
/// Symbols of packets before the gap, of the gap, and after it.
const BEFORE: usize = 40_000;
const GAP: usize = 50_000;
const AFTER: usize = 60_000;
/// The bound of the recovery of `qam_demod` (test 7, D1571).
const QAM_RECOVERY_SYMBOLS: usize = 24_576;
/// Bytes of payload per packet (a sequence number in the first four).
const PAYLOAD: usize = 20;
/// The share of the packets sent after the recovery that must arrive with `CRC_OK`.
const REQUIRED: f64 = 0.95;

/// One case: the order and the level.
#[derive(Clone, Copy, Debug)]
struct Case {
    order: u32,
    esn0_db: f64,
}

fn bps(order: u32) -> usize {
    if order == 64 { 6 } else { 4 }
}

/// The transmitted baseband and, per packet, its sequence number and the input sample where its
/// frame starts (to within a symbol).
struct Sent {
    signal: Vec<C32>,
    starts: Vec<usize>,
}

/// Idle symbols, then packets without a pause up to `LEAD + BEFORE + GAP + AFTER` symbols.
fn transmit(order: u32) -> Sent {
    let cfg = QamFrameEncodeConfig {
        frame: sdrk_blocks::FrameEncodeConfig {
            idle: FrameIdle::Pn9,
            ..sdrk_blocks::FrameEncodeConfig::default()
        },
        qam: sdrk_blocks::QamModConfig {
            constellation: c_of(order),
            samples_per_symbol: SPS,
            ..sdrk_blocks::QamModConfig::default()
        },
        // The signal of M44.1: no scrambler (M44.2 added it, D1531).
        scrambler: sdrk_blocks::Scrambler::Off,
        ..QamFrameEncodeConfig::default()
    };
    let rate = Rate::hz(SYMBOL_RATE * bps(order) as u64).unwrap();
    let mut tx = QamFrameEncodeBlock::new(&cfg, rate, &mut VecStorage).unwrap();
    let total = (LEAD + BEFORE + GAP + AFTER) * SPS;
    let mut signal = vec![C32::default(); total + 4096];
    let mut at = tx.process(&[], &mut signal[..LEAD * SPS]).unwrap().produced;
    let mut starts = Vec::new();
    let mut seq = 0u32;
    while at < total {
        let mut data = [0u8; PAYLOAD];
        data[..4].copy_from_slice(&seq.to_le_bytes());
        for (k, b) in data[4..].iter_mut().enumerate() {
            *b = (seq as usize * 31 + k * 7) as u8;
        }
        let p = Packet::new(0, PacketFlags::NONE, &data).unwrap();
        starts.push(at);
        let r = tx.process(&[p], &mut signal[at..]).unwrap();
        assert_eq!(r.consumed, 1);
        at += r.produced;
        seq += 1;
    }
    signal.truncate(total);
    Sent { signal, starts }
}

/// The channel: the carrier offset, Gaussian noise at `esn0_db` (the power of the signal) and,
/// over the gap, noise at the signal's power instead of the signal.
fn channel(sent: &[C32], order: u32, esn0_db: f64, seed: u64) -> Vec<C32> {
    let turned: Vec<Complex64> = sent
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let ph = core::f64::consts::TAU * OFFSET * i as f64 / SPS as f64;
            Complex64::new(f64::from(c.re), f64::from(c.im))
                * Complex64::new(libm::cos(ph), libm::sin(ph))
        })
        .collect();
    let body = &turned[LEAD * SPS..];
    let power = body.iter().map(Complex64::norm_sqr).sum::<f64>() / body.len() as f64;
    let k = bps(order);
    let ebn0 = esn0_db - 10.0 * libm::log10(k as f64);
    let mut awgn = Awgn::from_ebn0(ebn0, k as u32, SPS as f64, power, seed + 100).unwrap();
    let mut noisy = Vec::with_capacity(turned.len());
    awgn.process(&turned, &mut noisy);
    let mut input: Vec<C32> = noisy
        .iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect();
    let sigma = libm::sqrt(power / 2.0);
    let mut rng = Pcg32::new(seed, 2);
    let gap = (LEAD + BEFORE) * SPS..(LEAD + BEFORE + GAP) * SPS;
    for c in &mut input[gap] {
        *c = C32::new((sigma * rng.normal()) as f32, (sigma * rng.normal()) as f32);
    }
    input
}

fn config(order: u32, frame_timeout: usize) -> QamFrameDecodeConfig {
    let mut cfg = QamFrameDecodeConfig {
        frame_timeout,
        scrambler: sdrk_blocks::Scrambler::Off,
        ..QamFrameDecodeConfig::default()
    };
    cfg.demod.constellation = c_of(order);
    cfg.demod.symbol_rate_hz = SYMBOL_RATE as f64;
    cfg
}

fn receiver(order: u32, frame_timeout: usize) -> QamFrameDecodeBlock<VecStorage> {
    let cfg = config(order, frame_timeout);
    QamFrameDecodeBlock::new(
        &cfg,
        Rate::hz(SYMBOL_RATE * SPS as u64).unwrap(),
        &mut VecStorage,
    )
    .unwrap()
}

/// Runs `input` through `rx` in calls of `piece` samples with room for `room` packets.
fn receive(
    rx: &mut QamFrameDecodeBlock<VecStorage>,
    input: &[C32],
    piece: usize,
    room: usize,
) -> Vec<Packet> {
    let mut got = Vec::new();
    let mut out = vec![Packet::default(); room];
    for p in input.chunks(piece) {
        let mut rest = p;
        loop {
            let r = rx.process(rest, &mut out).unwrap();
            got.extend_from_slice(&out[..r.produced]);
            rest = &rest[r.consumed..];
            if rest.is_empty() && r.produced < room {
                break;
            }
        }
    }
    got
}

/// The sequence numbers of the packets with `CRC_OK` and the input sample of each.
fn received(packets: &[Packet]) -> Vec<(u32, u64)> {
    packets
        .iter()
        .filter(|p| p.flags.contains(PacketFlags::CRC_OK) && p.len() == PAYLOAD)
        .map(|p| {
            (
                u32::from_le_bytes(p.data()[..4].try_into().unwrap()),
                p.index,
            )
        })
        .collect()
}

#[test]
fn a_gap_in_the_middle_of_a_transmission_recovers() {
    let cases = [
        Case {
            order: 16,
            esn0_db: 16.0 + 10.0 * libm::log10(4.0),
        },
        Case {
            order: 64,
            esn0_db: 30.0,
        },
    ];
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .iter()
            .map(|&c| {
                s.spawn(move || {
                    let sent = transmit(c.order);
                    let input = channel(&sent.signal, c.order, c.esn0_db, 850);
                    let mut rx = receiver(c.order, QAM_FRAME_TIMEOUT);
                    let packets = receive(&mut rx, &input, 4096, 64);
                    (
                        c,
                        sent,
                        packets,
                        rx.frame_watch().restarts(),
                        rx.demodulator().restarts(),
                    )
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (c, sent, packets, watch_restarts, demod_restarts) in results {
        let ok = received(&packets);
        let gap_from = (LEAD + BEFORE) * SPS;
        let gap_end = (LEAD + BEFORE + GAP) * SPS;
        let last_before = ok
            .iter()
            .filter(|&&(_, i)| (i as usize) < gap_from)
            .map(|&(_, i)| i)
            .max();
        let first_after = ok
            .iter()
            .filter(|&&(_, i)| (i as usize) >= gap_end)
            .map(|&(_, i)| i)
            .min();
        let from = gap_end + (QAM_FRAME_TIMEOUT + QAM_RECOVERY_SYMBOLS) * SPS;
        let due: Vec<u32> = (0..sent.starts.len() as u32)
            .filter(|&k| sent.starts[k as usize] >= from)
            .filter(|&k| sent.starts.get(k as usize + 1).is_some())
            .collect();
        let got = due
            .iter()
            .filter(|k| ok.iter().any(|(s, _)| s == *k))
            .count();
        let share = got as f64 / due.len() as f64;
        let before = (0..sent.starts.len())
            .filter(|&k| {
                sent.starts[k] >= LEAD * SPS + 10_000 * SPS && sent.starts[k] + 400 < gap_from
            })
            .filter(|&k| ok.iter().any(|(s, _)| *s as usize == k))
            .count();
        eprintln!(
            "{}-QAM, Es/N0 {:.2} dB: {} packets with CRC_OK of {} sent ({before} in the 30000 \
             symbols before the gap); restarts of the frame watch {watch_restarts}, of the \
             demodulator {demod_restarts}; the last before the gap at symbol {:?}, the first \
             after it {:?} symbols after its end; from {} symbols after the end of the gap {got} \
             of {} ({share:.4})",
            c.order,
            c.esn0_db,
            ok.len(),
            sent.starts.len(),
            last_before.map(|i| i as usize / SPS),
            first_after.map(|i| (i as usize - gap_end) / SPS),
            QAM_FRAME_TIMEOUT + QAM_RECOVERY_SYMBOLS,
            due.len()
        );
        assert!(
            watch_restarts >= 1,
            "{c:?}: the frame watch acts in the gap"
        );
        assert!(due.len() > 100, "{c:?}");
        assert!(
            share >= REQUIRED,
            "{c:?}: {got} of {} after the recovery",
            due.len()
        );
    }
}

/// The pair `qam_demod` -> `frame_decode` by hand, as the composite runs it (pieces of 256
/// samples, every soft bit decoded before the next piece), with the indices of the decoder.
fn pair(order: u32, input: &[C32]) -> Vec<Packet> {
    let cfg = config(order, 0);
    let rate = Rate::hz(SYMBOL_RATE * SPS as u64).unwrap();
    let mut demod = QamDemodBlock::new(&cfg.demod, rate, &mut VecStorage).unwrap();
    let mut dec = FrameDecodeBlock::new(&cfg.decoder(), rate, &mut VecStorage).unwrap();
    let mut soft = vec![0.0f32; 1 << 16];
    let mut out = vec![Packet::default(); 64];
    let mut got = Vec::new();
    for p in input.chunks(4096) {
        let r = demod.process(p, &mut soft).unwrap();
        assert_eq!(r.consumed, p.len());
        let mut bits = &soft[..r.produced];
        while !bits.is_empty() {
            let q = dec.process(bits, &mut out).unwrap();
            got.extend_from_slice(&out[..q.produced]);
            bits = &bits[q.consumed..];
        }
    }
    got
}

#[test]
fn frame_timeout_zero_gives_the_pair() {
    let order = 16;
    let sent = transmit(order);
    let input = channel(&sent.signal, order, 16.0 + 10.0 * libm::log10(4.0), 850);
    let mut rx = receiver(order, 0);
    let got = receive(&mut rx, &input, 4096, 64);
    let want = pair(order, &input);
    assert_eq!(rx.frame_watch().restarts(), 0);
    assert_eq!(got.len(), want.len());
    let delay = {
        let d = rx.demodulator().timing().delay;
        d.num() as f64 / d.den() as f64
    };
    for (g, w) in got.iter().zip(&want) {
        assert_eq!((g.data(), g.flags), (w.data(), w.flags));
        let index = ((w.index / 4) as f64 * SPS as f64 + delay) as u64;
        assert_eq!(g.index, index);
    }
    eprintln!("frame_timeout 0: {} packets as the pair", got.len());
}

#[test]
fn qam_frame_decode_is_chunk_invariant() {
    let order = 16;
    let sent = transmit(order);
    let input = channel(&sent.signal, order, 16.0 + 10.0 * libm::log10(4.0), 850);
    let mut whole = receiver(order, QAM_FRAME_TIMEOUT);
    let want = receive(&mut whole, &input, input.len(), 1 << 14);
    assert!(whole.frame_watch().restarts() >= 1);
    assert_eq!(QAM_FRAME_WATCH_LAG, 256);
    let mut plans = Pcg32::new(851, 9);
    for _ in 0..20 {
        let mut rx = receiver(order, QAM_FRAME_TIMEOUT);
        let mut got = Vec::new();
        let mut out = vec![Packet::default(); 4];
        let mut at = 0;
        loop {
            let len = 1 + plans.next_u32() as usize % 9000;
            let room = 1 + plans.next_u32() as usize % out.len();
            let end = (at + len).min(input.len());
            let r = rx.process(&input[at..end], &mut out[..room]).unwrap();
            got.extend_from_slice(&out[..r.produced]);
            at += r.consumed;
            if at == input.len() && r.produced == 0 {
                break;
            }
        }
        assert_eq!(got, want, "the packets of a cut plan");
        assert_eq!(rx.frame_watch().restarts(), whole.frame_watch().restarts());
        assert_eq!(rx.demodulator().restarts(), whole.demodulator().restarts());
    }
}

/// The constellation of a numeric order of the tables of this file (2, 4, 8, 16 or 64).
fn c_of(order: u32) -> SymbolMapConstellation {
    match order {
        2 => SymbolMapConstellation::Bpsk,
        4 => SymbolMapConstellation::Qpsk,
        8 => SymbolMapConstellation::Psk8,
        16 => SymbolMapConstellation::Qam16,
        64 => SymbolMapConstellation::Qam64,
        other => unreachable!("no constellation of the order {other}"),
    }
}
