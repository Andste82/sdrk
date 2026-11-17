//! M44.1 §7: the composites `qam_frame_encode` and `qam_frame_decode` and the frame watch.
//!
//! - `qam_frame_encode` equals the pair `frame_encode` -> `qam_mod` bit for bit, whole and in 20
//!   cut plans of packets, idle calls and output room (seed 851), for 16-QAM and 64-QAM, every
//!   `idle` and both whitenings.
//! - The flat parameters of both composites through their `ParamMap`, read-only at run time.
//! - `FrameWatch` (test 6 for §7): scripted frames, the deadline, the restart, the frames that
//!   end before a restart or after the timeout ran out.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

use sdrk::{C32, Rate};
use sdrk_blocks::SymbolMapConstellation;
use sdrk_blocks::{
    Block, Error, FrameEncodeBlock, FrameEncodeConfig, FrameIdle, FrameWatch, Packet, ParamValue,
    QAM_FRAME_DECODE_PARAMS, QAM_FRAME_ENCODE_PARAMS, QamFrameDecodeBlock, QamFrameDecodeConfig,
    QamFrameEncodeBlock, QamFrameEncodeConfig, QamModBlock, QamModConfig,
};
use sdrk_proto::PacketFlags;
use sdrk_proto::frame::Whitening;
use sdrk_testkit::{Pcg32, VecStorage};

/// One step of a transmission: packets, or an idle call for `n` output samples.
#[derive(Clone, Debug)]
enum Step {
    Packets(Vec<Packet>),
    Idle(usize),
}

fn script(seed: u64) -> Vec<Step> {
    let mut rng = Pcg32::new(seed, 5);
    let mut steps = vec![Step::Idle(37)];
    for _ in 0..24 {
        let n = 1 + rng.below(4) as usize;
        let packets = (0..n)
            .map(|_| {
                let len = rng.below(40) as usize;
                let data: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
                Packet::new(0, PacketFlags::NONE, &data).unwrap()
            })
            .collect();
        steps.push(Step::Packets(packets));
        if rng.below(2) == 0 {
            steps.push(Step::Idle(1 + rng.below(300) as usize));
        }
    }
    steps.push(Step::Idle(200));
    steps
}

/// The pair by hand: the bits of `frame_encode` into a queue, `qam_mod` takes whole symbols. An
/// idle step of `n` samples gives the idle bits of the whole symbols of `n` samples that the
/// waiting bits do not fill.
fn pair(cfg: &QamFrameEncodeConfig, steps: &[Step]) -> Vec<C32> {
    let rate = Rate::hz(4800).unwrap();
    let mut enc = FrameEncodeBlock::new(&cfg.frame, rate, &mut VecStorage).unwrap();
    let mut qam = QamModBlock::new(&cfg.qam, rate, &mut VecStorage).unwrap();
    let (bps, sps) = (cfg.qam.bits_per_symbol(), cfg.qam.samples_per_symbol);
    let mut bits: Vec<f32> = Vec::new();
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; 1 << 16];
    let mut samples = vec![C32::default(); 1 << 18];
    let mut modulate = |bits: &mut Vec<f32>, out: &mut Vec<C32>| {
        let p = qam.process(bits, &mut samples).unwrap();
        out.extend_from_slice(&samples[..p.produced]);
        bits.drain(..p.consumed);
    };
    for step in steps {
        match step {
            Step::Packets(packets) => {
                let p = enc.process(packets, &mut buf).unwrap();
                assert_eq!(p.consumed, packets.len());
                bits.extend_from_slice(&buf[..p.produced]);
                modulate(&mut bits, &mut out);
            }
            Step::Idle(n) => {
                let want = (n / sps * bps).saturating_sub(bits.len());
                let p = enc.process(&[], &mut buf[..want]).unwrap();
                bits.extend_from_slice(&buf[..p.produced]);
                modulate(&mut bits, &mut out);
            }
        }
    }
    out
}

fn composite(cfg: &QamFrameEncodeConfig, steps: &[Step], plans: Option<&mut Pcg32>) -> Vec<C32> {
    let rate = Rate::hz(4800).unwrap();
    let mut tx = QamFrameEncodeBlock::new(cfg, rate, &mut VecStorage).unwrap();
    let sps = cfg.qam.samples_per_symbol;
    let mut out = Vec::new();
    let mut buf = vec![C32::default(); 1 << 18];
    let mut plans = plans;
    for step in steps {
        match step {
            Step::Packets(packets) => {
                let mut rest = &packets[..];
                while !rest.is_empty() {
                    let room = match plans.as_deref_mut() {
                        Some(r) => 1 + r.below(3000) as usize,
                        None => buf.len(),
                    };
                    let p = tx.process(rest, &mut buf[..room]).unwrap();
                    out.extend_from_slice(&buf[..p.produced]);
                    rest = &rest[p.consumed..];
                }
                // the whole symbols that wait (the room was short)
                let n = drain_symbols(&mut tx, &mut buf, cfg.qam.bits_per_symbol());
                out.extend_from_slice(&buf[..n]);
            }
            Step::Idle(n) => {
                let p = tx.process(&[], &mut buf[..*n]).unwrap();
                assert_eq!(p.produced, n / sps * sps);
                out.extend_from_slice(&buf[..p.produced]);
            }
        }
    }
    out
}

/// Writes the whole symbols that wait in the composite without taking idle bits: an idle call
/// with room for exactly those symbols.
fn drain_symbols(tx: &mut QamFrameEncodeBlock<VecStorage>, buf: &mut [C32], bps: usize) -> usize {
    let sps = tx.config().qam.samples_per_symbol;
    let n = tx.waiting() / bps * sps;
    tx.process(&[], &mut buf[..n]).unwrap().produced
}

#[test]
fn qam_frame_encode_equals_the_pair() {
    let mut plans = Pcg32::new(851, 11);
    for (order, sps, idle, whitening) in [
        (16, 8, FrameIdle::Pn9, Whitening::None),
        (64, 4, FrameIdle::Pn9, Whitening::Pn9),
        (16, 3, FrameIdle::Zero, Whitening::None),
        (64, 5, FrameIdle::Preamble, Whitening::None),
    ] {
        let cfg = QamFrameEncodeConfig {
            frame: FrameEncodeConfig {
                preamble_bytes: 3,
                idle,
                whitening,
            },
            qam: QamModConfig {
                constellation: c_of(order),
                samples_per_symbol: sps,
                ..QamModConfig::default()
            },
        };
        let steps = script(u64::from(order) + sps as u64);
        let want = pair(&cfg, &steps);
        assert!(want.len() > 1000);
        let got = composite(&cfg, &steps, None);
        assert!(
            got.len() == want.len()
                && got
                    .iter()
                    .zip(&want)
                    .all(|(a, b)| a.re.to_bits() == b.re.to_bits()
                        && a.im.to_bits() == b.im.to_bits()),
            "{order} {sps}: whole"
        );
        for _ in 0..20 {
            let got = composite(&cfg, &steps, Some(&mut plans));
            assert_eq!(got.len(), want.len(), "{order} {sps}: a cut plan");
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.re.to_bits() == b.re.to_bits()
                        && a.im.to_bits() == b.im.to_bits()),
                "{order} {sps}: a cut plan"
            );
        }
    }
}

#[test]
fn the_flat_parameters_go_through_the_param_maps() {
    let rate = Rate::hz(4800).unwrap();
    let mut cfg = QamFrameEncodeConfig::default();
    for (name, value) in [
        ("preamble_bytes", ParamValue::I64(6)),
        ("idle", ParamValue::Choice(2)),
        ("whitening", ParamValue::Choice(1)),
        ("constellation", ParamValue::Choice(4)),
        ("samples_per_symbol", ParamValue::I64(5)),
        ("rolloff", ParamValue::F64(0.25)),
        ("span", ParamValue::I64(8)),
    ] {
        assert!(QAM_FRAME_ENCODE_PARAMS.path(name).is_some());
        cfg.set(name, value).unwrap();
        assert_eq!(cfg.get(name).unwrap(), value, "{name}");
    }
    assert_eq!(
        cfg.set("rotations", ParamValue::Choice(0)),
        Err(Error::UnknownParameter)
    );
    let mut tx = QamFrameEncodeBlock::new(&cfg, rate, &mut VecStorage).unwrap();
    assert_eq!(tx.config(), cfg);
    for &(name, _) in QAM_FRAME_ENCODE_PARAMS.entries() {
        assert_eq!(tx.param(name).unwrap(), cfg.get(name).unwrap(), "{name}");
        assert_eq!(
            tx.set_param(name, cfg.get(name).unwrap()),
            Err(Error::ReadOnlyParameter)
        );
    }
    assert_eq!(tx.param("frame_timeout"), Err(Error::UnknownParameter));

    let mut cfg = QamFrameDecodeConfig::default();
    for (name, value) in [
        ("constellation", ParamValue::Choice(4)),
        ("symbol_rate_hz", ParamValue::F64(1200.0)),
        ("rolloff", ParamValue::F64(0.25)),
        ("span", ParamValue::I64(8)),
        ("equalizer_taps", ParamValue::I64(15)),
        ("sync_bw", ParamValue::F64(0.002)),
        ("carrier_bw", ParamValue::F64(0.003)),
        ("max_sync_errors", ParamValue::I64(2)),
        ("emit_bad", ParamValue::Bool(true)),
        ("whitening", ParamValue::Choice(1)),
        ("frame_timeout", ParamValue::I64(8192)),
    ] {
        cfg.set(name, value).unwrap();
        assert_eq!(cfg.get(name).unwrap(), value, "{name}");
    }
    // since M44.1.1 the frame watch is stage 2 of the map (D1805)
    assert_eq!(
        QAM_FRAME_DECODE_PARAMS.path("frame_timeout"),
        Some("2.frame_timeout")
    );
    assert_eq!(
        cfg.set("rotations", ParamValue::Choice(0)),
        Err(Error::UnknownParameter)
    );
    assert_eq!(
        cfg.set("bits_per_symbol", ParamValue::I64(4)),
        Err(Error::UnknownParameter)
    );
    assert!(matches!(
        cfg.set("frame_timeout", ParamValue::I64(-1)),
        Err(Error::OutOfRange { .. })
    ));
    assert_eq!(cfg.decoder().bits_per_symbol, 6);
    let rx = QamFrameDecodeBlock::new(&cfg, rate, &mut VecStorage).unwrap();
    // `rotations` and `bits_per_symbol` of the decoder are set by the composite
    let got = rx.config();
    assert_eq!(
        (got.demod, got.decoder(), got.frame_timeout),
        (cfg.demod, cfg.decoder(), cfg.frame_timeout)
    );
    for name in [
        "constellation",
        "equalizer_taps",
        "max_sync_errors",
        "whitening",
        "frame_timeout",
    ] {
        assert_eq!(rx.param(name).unwrap(), cfg.get(name).unwrap(), "{name}");
    }
}

#[test]
fn the_frame_watch_follows_the_frames() {
    let mut w = FrameWatch::new(1000);
    assert_eq!(
        (w.timeout(), w.expected(), w.deadline()),
        (1000, false, None)
    );
    // nothing is expected before the first frame with CRC_OK
    w.frame_ok(500);
    assert_eq!((w.expected(), w.deadline()), (true, Some(1500)));
    w.frame_ok(900);
    w.frame_ok(700); // decoded late: the deadline stays
    assert_eq!(w.deadline(), Some(1900));
    // a frame that ends after the timeout ran out does not move it: the restart follows
    w.frame_ok(1950);
    assert_eq!(w.deadline(), Some(1900));
    w.restarted(2156);
    assert_eq!((w.expected(), w.deadline(), w.restarts()), (false, None, 1));
    // frames that end before the restart do not count
    w.frame_ok(2100);
    assert!(!w.expected());
    w.frame_ok(3000);
    assert_eq!(w.deadline(), Some(4000));
    w.reset();
    assert_eq!((w.expected(), w.restarts()), (false, 0));
    // timeout 0: the watch is off
    let mut off = FrameWatch::new(0);
    off.frame_ok(10);
    assert_eq!(off.deadline(), None);
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
