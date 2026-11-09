//! `PulseKeyBlock` (M46 extension): the envelope of the pulse trains at the clock rate. The widths
//! are exact in clock samples at 48 kHz for the Bresser timings, the length of a train is the
//! rounded sum (no drift, also at 44.1 kHz), the levels are bipolar, the carrier is off before,
//! between and after trains, a train that arrives while another is sent is queued and what does
//! not fit is counted, and the envelope does not depend on the chunking (20 plans, seed 774).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::float_cmp
)]

mod common;

use sdrk::Rate;
use sdrk_blocks::{
    Block, Error, PULSE_KEY_PAIRS, PULSE_KEY_QUEUE, Packet, ParamValue, PulseKeyBlock,
    PulseKeyConfig,
};
use sdrk_proto::PacketFlags;
use sdrk_proto::pulses::{self, CONTINUED, Header, Modulation, Pair};
use sdrk_testkit::VecStorage;

const SEED_CHUNKS: u64 = 774;

fn header(rate_hz: u32, m: Modulation) -> Header {
    Header {
        modulation: m,
        channel: 0,
        freq_offset_hz: 0,
        level_db10: 0,
        noise_db10: 0,
        rate_hz,
    }
}

fn key(guard_us: u32, rate: u64) -> PulseKeyBlock<VecStorage> {
    PulseKeyBlock::new(
        &PulseKeyConfig { guard_us },
        Rate::hz(rate).unwrap(),
        &mut VecStorage,
    )
    .unwrap()
}

fn train(index: u64, rate_hz: u32, pairs: &[Pair]) -> Vec<Packet> {
    pulses::packets(
        index,
        PacketFlags::NONE,
        &header(rate_hz, Modulation::Ook),
        pairs,
    )
    .map(|p| p.unwrap())
    .collect()
}

/// The pairs of a Bresser frame: the preamble of four pairs and the 40 bits (the nominal
/// timings 735/500/250 of the decoder: a 1 is a long pulse, a 0 a short one).
fn bresser_like() -> Vec<Pair> {
    let mut v = vec![
        Pair {
            pulse: 250,
            gap: 500,
        };
        4
    ];
    for k in 0..40 {
        v.push(Pair {
            pulse: if k % 3 == 0 { 750 } else { 250 },
            gap: 500,
        });
    }
    v.push(Pair { pulse: 735, gap: 0 });
    v
}

/// Runs the block over `clock` samples with the packets given at the first call.
fn run(b: &mut PulseKeyBlock<VecStorage>, packets: &[Packet], n: usize, size: usize) -> Vec<f32> {
    let clock = vec![0.0f32; size.max(1)];
    let mut out = Vec::new();
    let mut first = true;
    while out.len() < n {
        let k = size.min(n - out.len());
        let mut buf = vec![9.0f32; k];
        let p = b
            .process((&clock[..k], if first { packets } else { &[] }), &mut buf)
            .unwrap();
        assert_eq!((p.consumed, p.produced), (k, k));
        first = false;
        out.extend_from_slice(&buf);
    }
    out
}

/// The run lengths of an envelope: (level, count).
fn runs(x: &[f32]) -> Vec<(f32, usize)> {
    let mut v: Vec<(f32, usize)> = Vec::new();
    for &y in x {
        match v.last_mut() {
            Some((l, n)) if *l == y => *n += 1,
            _ => v.push((y, 1)),
        }
    }
    v
}

/// The nearest integer of `t * num / den`, half up, in integers (an independent form).
fn edge(t: u64, num: u64, den: u64) -> u64 {
    (2 * t * num + den) / (2 * den)
}

#[test]
fn the_bresser_widths_are_exact_clock_samples_at_48_khz() {
    // 250, 500 and 750 us are 12, 24 and 36 samples; 735 us is 35.28 samples: 35.
    let mut b = key(0, 48_000);
    let pairs = bresser_like();
    let out = run(&mut b, &train(0, 1_000_000, &pairs), 3000, 3000);
    let r = runs(&out);
    // Starts with the pulse of the first pair (the train starts with the first sample).
    assert_eq!(r[0], (1.0, 12));
    assert_eq!(r[1], (-1.0, 24));
    assert_eq!(r[2], (1.0, 12));
    // The first pair after the preamble has a 750 us pulse: 36 samples.
    assert_eq!(r[8], (1.0, 36));
    // The last pulse (735 us) and then silence to the end.
    let n = r.len();
    assert_eq!(r[n - 2], (1.0, 35));
    assert_eq!(r[n - 1].0, -1.0);
    assert!(out.iter().all(|&y| y == 1.0 || y == -1.0));
    // The train's length is the exact sum: no drift.
    let total: u64 = pairs.iter().map(|p| u64::from(p.pulse + p.gap)).sum();
    let on_end = out.iter().rposition(|&y| y > 0.0).unwrap() + 1;
    let last_gap_start = total;
    assert_eq!(on_end as u64, edge(last_gap_start, 48, 1000));
    assert_eq!(b.trains_sent(), 1);
}

#[test]
fn the_length_of_a_train_is_the_rounded_sum_at_any_rate() {
    // 10 pairs of 333 us pulses and 777 us gaps: the sum is 11100 us.
    let pairs = vec![
        Pair {
            pulse: 333,
            gap: 777,
        };
        10
    ];
    for rate in [48_000u64, 44_100, 8_000, 200_000, 1_000_000] {
        let mut b = key(0, rate);
        let n = (rate / 10) as usize;
        let out = run(&mut b, &train(0, 1_000_000, &pairs), n, 1000);
        // The last gap is 777 us: the last pulse ends at 11100 - 777 us.
        let last_on = out.iter().rposition(|&y| y > 0.0).unwrap() + 1;
        assert_eq!(
            last_on as u64,
            edge(11_100 - 777, rate, 1_000_000),
            "rate {rate}"
        );
        // Every edge is the rounded cumulative time.
        let mut t = 0u64;
        let mut at = 0usize;
        for (r, want) in runs(&out)
            .iter()
            .zip(pairs.iter().flat_map(|p| [p.pulse, p.gap]))
        {
            t += u64::from(want);
            let next = edge(t, rate, 1_000_000) as usize;
            assert_eq!(at + r.1, next, "rate {rate}");
            at = next;
            if at >= last_on {
                break;
            }
        }
    }
}

#[test]
fn the_levels_are_bipolar_and_the_carrier_is_off_when_idle() {
    let mut b = key(0, 48_000);
    // No train: off.
    let out = run(&mut b, &[], 100, 100);
    assert!(out.iter().all(|&y| y == -1.0));
    assert!(!b.active());
    // A train and the idle after it.
    let p = [Pair {
        pulse: 1000,
        gap: 1000,
    }];
    let out = run(&mut b, &train(0, 1_000_000, &p), 500, 500);
    assert!(out[..48].iter().all(|&y| y == 1.0));
    assert!(out[48..].iter().all(|&y| y == -1.0));
    assert!(out.iter().all(|&y| y.abs() == 1.0));
    assert_eq!((b.queued_trains(), b.trains_sent()), (0, 1));
}

#[test]
fn the_guard_keeps_two_trains_apart() {
    // The last gap of a train is 0, so only the guard separates the trains.
    let p = [Pair { pulse: 500, gap: 0 }];
    let mut two = train(0, 1_000_000, &p);
    two.extend(train(1, 1_000_000, &p));
    let mut b = key(1000, 48_000); // 48 samples of guard
    let out = run(&mut b, &two, 400, 400);
    let r = runs(&out);
    assert_eq!(
        &r[..4],
        &[(1.0, 24), (-1.0, 48), (1.0, 24), (-1.0, 400 - 24 - 48 - 24)]
    );
    // Without a guard the two pulses are one.
    let mut b = key(0, 48_000);
    let out = run(&mut b, &two, 400, 400);
    assert_eq!(runs(&out)[0], (1.0, 48));
}

#[test]
fn trains_wait_in_a_queue_and_the_rest_is_dropped_and_counted() {
    let p = [Pair {
        pulse: 1000,
        gap: 1000,
    }];
    let mut packets = Vec::new();
    for i in 0..6 {
        packets.extend(train(i, 1_000_000, &p));
    }
    let mut b = key(0, 48_000);
    let out = run(&mut b, &packets, 1000, 1000);
    assert_eq!(PULSE_KEY_QUEUE, 4);
    assert_eq!(b.trains_sent(), 4);
    assert_eq!(b.dropped_trains(), 2);
    assert_eq!(runs(&out).iter().filter(|r| r.0 > 0.0).count(), 4);
    // The queue is free again.
    assert_eq!(b.queued_trains(), 0);
    // A train that arrives while another is sent is queued and follows it.
    let mut b = key(0, 48_000);
    let mut buf = vec![0.0f32; 20];
    let clock = [0.0f32; 20];
    b.process((&clock, &train(0, 1_000_000, &p)), &mut buf)
        .unwrap();
    assert!(b.active());
    b.process((&clock, &train(1, 1_000_000, &p)), &mut buf)
        .unwrap();
    assert_eq!(b.queued_trains(), 2);
    let out = run(&mut b, &[], 200, 200);
    assert_eq!(b.trains_sent(), 2);
    assert_eq!(out[0], 1.0); // the first pulse is still on
    assert_eq!(b.dropped_trains(), 0);
}

#[test]
fn a_train_beyond_the_pair_capacity_is_dropped_whole_and_counted() {
    let p = Pair {
        pulse: 100,
        gap: 100,
    };
    let long = vec![p; PULSE_KEY_PAIRS + 5];
    let ok = [p; 3];
    let mut packets = train(0, 1_000_000, &long);
    assert!(packets.len() > 1);
    packets.extend(train(1, 1_000_000, &ok));
    let mut b = key(0, 48_000);
    let out = run(&mut b, &packets, 1000, 1000);
    assert_eq!(b.dropped_trains(), 1);
    assert_eq!(b.trains_sent(), 1);
    assert_eq!(runs(&out).iter().filter(|r| r.0 > 0.0).count(), 3);
    // Exactly the capacity is still sent.
    let mut b = key(0, 48_000);
    let full = train(0, 1_000_000, &long[..PULSE_KEY_PAIRS]);
    let out = run(&mut b, &full, 13_000, 13_000);
    assert_eq!((b.dropped_trains(), b.trains_sent()), (0, 1));
    assert_eq!(
        runs(&out).iter().filter(|r| r.0 > 0.0).count(),
        PULSE_KEY_PAIRS
    );
}

#[test]
fn a_train_of_several_packets_starts_when_it_is_complete() {
    let pairs = vec![
        Pair {
            pulse: 100,
            gap: 100,
        };
        100
    ];
    let packets = train(3, 1_000_000, &pairs);
    assert!(packets.len() > 3 && packets[0].flags.contains(CONTINUED));
    let mut b = key(0, 48_000);
    let mut buf = vec![0.0f32; 50];
    let clock = [0.0f32; 50];
    // All but the last packet: nothing is sent yet.
    let p = b
        .process((&clock, &packets[..packets.len() - 1]), &mut buf)
        .unwrap();
    assert_eq!(p.produced, 50);
    assert!(buf.iter().all(|&y| y == -1.0));
    assert_eq!((b.queued_trains(), b.active()), (0, false));
    assert_eq!(b.consumed_packets(), packets.len() - 1);
    // The last packet completes it; it starts with that call's first sample.
    b.process((&clock, &packets[packets.len() - 1..]), &mut buf)
        .unwrap();
    assert!(b.active());
    assert_eq!(buf[0], 1.0);
    // The result equals the same packets given at once.
    let mut c = key(0, 48_000);
    let whole = run(&mut c, &packets, 1200, 1200);
    let mut d = key(0, 48_000);
    let _ = run(&mut d, &packets[..packets.len() - 1], 50, 50);
    let late = run(&mut d, &packets[packets.len() - 1..], 1200, 1200);
    assert_eq!(whole, late);
}

#[test]
fn other_packets_are_ignored_and_a_cut_train_is_dropped() {
    let mut b = key(0, 48_000);
    let p = [Pair {
        pulse: 100,
        gap: 100,
    }];
    let fsk = pulses::packets(
        0,
        PacketFlags::NONE,
        &header(1_000_000, Modulation::Fsk),
        &p,
    )
    .map(|x| x.unwrap())
    .collect::<Vec<_>>();
    let garbage = Packet::new(0, PacketFlags::NONE, b"hello").unwrap();
    let mut packets = vec![garbage];
    packets.extend(fsk);
    // A train of two packets whose second packet never comes, and then another train.
    let long = vec![Pair { pulse: 10, gap: 10 }; 40];
    let cut = train(7, 1_000_000, &long);
    packets.push(cut[0]);
    packets.extend(train(8, 1_000_000, &p));
    let out = run(&mut b, &packets, 200, 200);
    assert_eq!(b.ignored_packets(), 2);
    assert_eq!(b.dropped_trains(), 1);
    assert_eq!(b.trains_sent(), 1);
    assert_eq!(runs(&out).iter().filter(|r| r.0 > 0.0).count(), 1);
    b.reset();
    assert_eq!(
        (b.ignored_packets(), b.dropped_trains(), b.trains_sent()),
        (0, 0, 0)
    );
}

#[test]
fn the_envelope_is_chunk_invariant_over_20_plans() {
    let mut packets = train(0, 1_000_000, &bresser_like());
    packets.extend(train(
        1,
        500_000,
        &[Pair {
            pulse: 125,
            gap: 250,
        }; 9],
    ));
    let n = 6000;
    let want = run(&mut key(20_000, 48_000), &packets, n, n);
    assert!(want.iter().any(|&y| y > 0.0));
    for (k, plan) in common::twenty_plans(SEED_CHUNKS).into_iter().enumerate() {
        let mut b = key(20_000, 48_000);
        let clock = vec![0.0f32; 4096];
        let mut sizes = plan.sizes();
        let mut got = Vec::new();
        let mut first = true;
        while got.len() < n {
            let k = sizes.next().unwrap().min(n - got.len()).min(clock.len());
            let mut buf = vec![0.0f32; k];
            b.process((&clock[..k], if first { &packets } else { &[] }), &mut buf)
                .unwrap();
            first = false;
            got.extend_from_slice(&buf);
        }
        assert!(got == want, "plan {k}");
    }
}

#[test]
fn the_parameter_the_arena_and_the_struct() {
    let mut cfg = PulseKeyConfig::default();
    assert_eq!(cfg.guard_us, 20_000);
    cfg.set("guard_us", ParamValue::I64(5)).unwrap();
    assert_eq!(cfg.get("guard_us").unwrap(), ParamValue::I64(5));
    assert!(matches!(
        cfg.set("guard_us", ParamValue::I64(1_000_001)),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(cfg.get("nope"), Err(Error::UnknownParameter)));
    let mut b = key(5, 48_000);
    assert!(matches!(
        b.set_param("guard_us", ParamValue::I64(1)),
        Err(Error::ReadOnlyParameter)
    ));
    assert_eq!(
        b.reading(2),
        Some(("trains_sent", 0.0)),
        "readings: ignored, dropped, sent, queued, active"
    );
    assert_eq!(b.reading(5), None);
    let bytes = PulseKeyBlock::<sdrk_blocks::ArenaStorage<'static>>::storage_bytes(&cfg).unwrap();
    assert_eq!(bytes, 4 * 1200 * 8);
    let rate = Rate::hz(48_000).unwrap();
    common::check_exact_storage(bytes, |arena| {
        PulseKeyBlock::new(&cfg, rate, arena).map(|_| ())
    });
    for size in [
        std::mem::size_of::<PulseKeyBlock<VecStorage>>(),
        std::mem::size_of::<PulseKeyBlock<sdrk_blocks::ArenaStorage<'static>>>(),
    ] {
        assert!(size <= 512, "{size}");
    }
    assert_eq!(
        PulseKeyBlock::<VecStorage>::rate_out(&cfg, rate).unwrap(),
        rate
    );
    fn send_sync<T: Send + Sync>() {}
    send_sync::<PulseKeyBlock<VecStorage>>();
}
