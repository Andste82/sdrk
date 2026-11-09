//! The flows `ism433_sensors` and `ism868_sensors` of M46 §5 on generated scenes: a light
//! end-to-end test per flow (the heavy test 3 of the milestone, 200 transmissions per band, is not
//! part of it).
//!
//! The scenes are rendered by `sdrk_testkit::ism` (encoded as `cu8` like an RTL-SDR recording)
//! from the pulse trains of the Rust encoders of `sdrk_proto::sensors`: the train of a reading is
//! turned back into the bits and the line code of its device, which the scene generator sends
//! (a few transmissions per device at different frequencies, with copies, a timing error up to
//! +-2 % and noise, several at the same time on different frequencies; seed 779). The flow must
//! report every transmission once (the copies merged) with the encoded values at its frequency
//! and nothing else. 433 MHz: `nexus`, `wh2`, `wh1080` and both Oregon models (the OOK devices
//! whose line code the scene generator has; `bresser_3ch` has a fixed bit period with a preamble
//! and is covered by the golden and block tests). 868 MHz: `lacrosse_tx29` and `lacrosse_tx35`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::large_stack_frames
)]

mod common;

use num_complex::{Complex32, Complex64};
use sdrk_device::formats;
use sdrk_pipeline::{InputData, Pipeline};
use sdrk_proto::Packet;
use sdrk_proto::pulses::{Header, Pair};
use sdrk_proto::sensors::{self, MAX_TRAIN_PAIRS, Reading, TrainSink};
use sdrk_testkit::Pcg32;
use sdrk_testkit::encode::encode;
use sdrk_testkit::ism::{Coding, Placed, Scene, Transmission};

use common::{build, flow};

/// The seed of the scenes (decision D1143, the reserve of the block 775 to 779).
const SEED: u64 = 779;
const FS: f64 = 2e6;
const NOISE_DB: f64 = -35.0;

/// How a device is sent: the line code and the bits of one copy.
struct Sent {
    coding: Coding,
    bits: Vec<bool>,
    repeats: u32,
    repeat_gap_us: f64,
}

/// The train of a reading from its encoder, `(pairs, repeats, repeat gap)`.
fn encoded(r: &Reading) -> (Vec<Pair>, u32, f64) {
    let enc = sensors::encoder_by_model(r.model).unwrap();
    let mut storage = vec![Pair::default(); MAX_TRAIN_PAIRS];
    let mut sink = TrainSink::new(&mut storage);
    let repeats = enc.default_repeats();
    enc.encode(r, repeats, &mut sink)
        .unwrap_or_else(|e| panic!("{r:?}: {e}"));
    (
        sink.pairs().to_vec(),
        repeats,
        f64::from(enc.repeat_gap_us()),
    )
}

/// The pairs of the first copy (a gap of 3 ms or more is the pause between two copies).
fn first_copy(pairs: &[Pair]) -> &[Pair] {
    let n = pairs
        .iter()
        .position(|p| p.gap >= 3000)
        .map_or(pairs.len(), |i| i + 1);
    &pairs[..n]
}

/// The bits and the line code of the first copy of `r`.
fn sent(r: &Reading) -> Sent {
    let (pairs, repeats, repeat_gap_us) = encoded(r);
    let copy = first_copy(&pairs);
    let (coding, bits) = match r.model {
        "wh2" | "wh1080" => {
            let (short_us, long_us) = if r.model == "wh2" {
                (500.0, 1500.0)
            } else {
                (544.0, 1524.0)
            };
            let mid = f64::midpoint(short_us, long_us) as u32;
            (
                Coding::Pwm {
                    short_us,
                    long_us,
                    gap_us: 1000.0,
                },
                copy.iter().map(|p| p.pulse < mid).collect(),
            )
        }
        "nexus" => (
            Coding::Ppm {
                pulse_us: 500.0,
                short_gap_us: 1000.0,
                long_gap_us: 2000.0,
            },
            // n bits are n + 1 pulses: the last one closes the last gap
            copy[..copy.len() - 1]
                .iter()
                .map(|p| p.gap > 1500)
                .collect(),
        ),
        "oregon_thn132n" | "oregon_thgr122nx" => {
            // the half bits as levels (the first pulse of the train is not at the start of a
            // bit: one unit of silence in front), a bit is the level of its first half
            let mut levels = vec![false];
            for (i, p) in copy.iter().enumerate() {
                levels.extend(std::iter::repeat_n(true, (p.pulse as usize + 244) / 488));
                if i + 1 < copy.len() {
                    levels.extend(std::iter::repeat_n(false, (p.gap as usize + 244) / 488));
                }
            }
            if levels.len() % 2 == 1 {
                levels.push(false);
            }
            (
                Coding::Manchester { half_bit_us: 488.0 },
                levels.chunks(2).map(|c| c[0]).collect(),
            )
        }
        "lacrosse_tx29" | "lacrosse_tx35" => {
            let bit_us = if r.model == "lacrosse_tx29" {
                58.0
            } else {
                105.0
            };
            let mut bits = Vec::new();
            for (i, p) in copy.iter().enumerate() {
                bits.extend(std::iter::repeat_n(
                    true,
                    (f64::from(p.pulse) / bit_us).round() as usize,
                ));
                if i + 1 < copy.len() {
                    bits.extend(std::iter::repeat_n(
                        false,
                        (f64::from(p.gap) / bit_us).round() as usize,
                    ));
                }
            }
            let deviation_hz = if r.model == "lacrosse_tx29" {
                35e3
            } else {
                25e3
            };
            (
                Coding::Fsk {
                    bit_us,
                    deviation_hz,
                },
                bits,
            )
        }
        other => panic!("{other}"),
    };
    Sent {
        coding,
        bits,
        repeats,
        repeat_gap_us,
    }
}

fn header_of(model: &str) -> Header {
    sensors::encoder_by_model(model).unwrap().header()
}

/// A reading as the decoder returns it: through the JSON text (the printed resolution).
fn reading(model: &'static str, id: u32, f: impl FnOnce(&mut Reading)) -> Reading {
    let enc = sensors::encoder_by_model(model).unwrap();
    let mut r = Reading::new(model, id, enc.mic(), &header_of(model));
    r.temp_decimals = enc.temp_decimals();
    f(&mut r);
    let mut text = String::new();
    r.to_json(&mut text).unwrap();
    Reading::from_json(&text).unwrap()
}

/// CU8 bytes of `seconds` of `scene`.
fn scene_cu8(scene: &Scene<'_>, seconds: f64) -> Vec<u8> {
    let n = (seconds * scene.fs) as usize;
    let mut r = scene.renderer().unwrap();
    let mut x = vec![Complex32::new(0.0, 0.0); 4096];
    let mut wide = vec![Complex64::new(0.0, 0.0); 4096];
    let mut bytes = vec![0u8; 2 * n];
    for piece in bytes.chunks_mut(2 * 4096) {
        let k = piece.len() / 2;
        x[..k].fill(Complex32::new(0.0, 0.0));
        r.render(&mut x[..k]);
        for (w, v) in wide.iter_mut().zip(&x[..k]) {
            *w = Complex64::new(f64::from(v.re), f64::from(v.im));
        }
        encode(&formats::CU8, &wide[..k], piece).unwrap();
    }
    bytes
}

/// Pushes `bytes` in pieces of 4096 samples and reads the readings after every push.
fn run(p: &mut Pipeline, bytes: &[u8]) -> Vec<Packet> {
    let mut got = Vec::new();
    let mut buf = vec![Packet::default(); 4096];
    for piece in bytes.chunks(2 * 4096) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        loop {
            let k = p.read_packets("readings", &mut buf).unwrap();
            if k == 0 {
                break;
            }
            got.extend_from_slice(&buf[..k]);
        }
    }
    got
}

/// One planned transmission: the reading, its frequency (Hz from the centre), the start (s), the
/// level (dBFS) and the timing error.
struct Plan {
    reading: Reading,
    freq_hz: f64,
    start_s: f64,
    level_db: f64,
    timing_error: f64,
}

/// Renders the plans as a scene, runs the flow and checks that every reading is reported once at
/// its frequency and that nothing else is reported.
fn check_flow(name: &str, center_hz: f64, plans: &[Plan]) {
    let sends: Vec<Sent> = plans.iter().map(|p| sent(&p.reading)).collect();
    let placed: Vec<Placed<'_>> = plans
        .iter()
        .zip(&sends)
        .map(|(p, s)| Placed {
            start_s: p.start_s,
            transmission: Transmission {
                coding: s.coding,
                bits: &s.bits,
                repeats: s.repeats,
                repeat_gap_us: s.repeat_gap_us,
                freq_offset_hz: p.freq_hz,
                level_db: p.level_db,
                timing_error: p.timing_error,
            },
        })
        .collect();
    let scene = Scene {
        fs: FS,
        center_hz,
        clock_error_ppm: 0.0,
        noise_db: NOISE_DB,
        transmissions: &placed,
        interferers: &[],
        seed: SEED,
    };
    let end = placed
        .iter()
        .map(|p| p.start_s + p.transmission.duration_us() * 1e-6)
        .fold(0.0, f64::max);
    let bytes = scene_cu8(&scene, end + 0.05);
    let packets = run(&mut build(&flow(name)), &bytes);
    let got: Vec<(Reading, f64)> = packets
        .iter()
        .map(|p| {
            let text = std::str::from_utf8(p.data()).unwrap();
            let v: serde_json::Value = serde_json::from_str(text).unwrap();
            (
                Reading::from_json(text).unwrap_or_else(|e| panic!("{text}: {e}")),
                v["freq_MHz"].as_f64().unwrap(),
            )
        })
        .collect();
    for p in plans {
        let found: Vec<&(Reading, f64)> = got
            .iter()
            .filter(|(r, _)| r.same_content(&p.reading))
            .collect();
        assert_eq!(
            found.len(),
            1,
            "{name}: {:?} at {} Hz reported {} times; got {got:?}",
            p.reading,
            p.freq_hz,
            found.len()
        );
        let want_mhz = (center_hz + p.freq_hz) * 1e-6;
        assert!(
            (found[0].1 - want_mhz).abs() < 0.03,
            "{name}: {} MHz, sent at {want_mhz} MHz",
            found[0].1
        );
    }
    assert_eq!(
        got.len(),
        plans.len(),
        "{name}: a reading that was not sent: {got:?}"
    );
}

/// Plans for a list of `(reading, frequency)`: rounds of five or four transmissions at the same
/// time (a few ms apart); `lengths` (the duration of the longest of a round) tell when the next
/// round starts.
fn rounds(items: &[(Reading, f64)], per_round: usize, stream: u64, level_db: f64) -> Vec<Plan> {
    let mut rng = Pcg32::new(SEED, stream);
    let mut plans = Vec::new();
    let mut round_start = 0.02;
    for round in items.chunks(per_round) {
        let mut longest = 0.0f64;
        for (k, (r, freq)) in round.iter().enumerate() {
            let s = sent(r);
            let t = Transmission {
                coding: s.coding,
                bits: &s.bits,
                repeats: s.repeats,
                repeat_gap_us: s.repeat_gap_us,
                freq_offset_hz: *freq,
                level_db: 0.0,
                timing_error: 0.0,
            };
            longest = longest.max(t.duration_us() * 1e-6);
            plans.push(Plan {
                reading: *r,
                freq_hz: *freq,
                start_s: round_start + 0.003 * k as f64,
                level_db: level_db + 5.0 * rng.next_f64(),
                timing_error: (rng.next_f64() - 0.5) * 0.04,
            });
        }
        round_start += longest + 0.04;
    }
    plans
}

#[test]
fn the_433_flow_reports_the_devices_of_the_band_once_each() {
    let items = vec![
        (
            reading("nexus", 0x5C, |r| {
                r.channel = Some(2);
                r.battery_ok = Some(true);
                r.temperature_c = Some(27.1);
                r.humidity = Some(56.0);
            }),
            -500e3,
        ),
        (
            reading("wh2", 0x4A, |r| {
                r.temperature_c = Some(21.3);
                r.humidity = Some(55.0);
            }),
            -250e3,
        ),
        (
            reading("wh1080", 0xA5, |r| {
                r.temperature_c = Some(18.7);
                r.humidity = Some(62.0);
                r.battery_ok = Some(true);
                r.wind_avg_km_h = Some(10.0 * 1.224);
                r.wind_max_km_h = Some(17.0 * 1.224);
                r.wind_dir_deg = Some(135.0);
                r.rain_mm = Some(20.0 * 0.3);
            }),
            0.0,
        ),
        (
            reading("oregon_thn132n", 0xAB, |r| {
                r.channel = Some(1);
                r.battery_ok = Some(true);
                r.temperature_c = Some(21.6);
            }),
            250e3,
        ),
        (
            reading("oregon_thgr122nx", 0x58, |r| {
                r.channel = Some(2);
                r.battery_ok = Some(false);
                r.temperature_c = Some(-3.5);
                r.humidity = Some(48.0);
            }),
            500e3,
        ),
        (
            reading("nexus", 0x21, |r| {
                r.channel = Some(1);
                r.battery_ok = Some(false);
                r.temperature_c = Some(-3.4);
                r.humidity = Some(80.0);
            }),
            -400e3,
        ),
        (
            reading("wh2", 0x07, |r| {
                r.temperature_c = Some(-4.5);
                r.humidity = Some(91.0);
            }),
            -150e3,
        ),
        (
            reading("wh1080", 0x19, |r| {
                r.temperature_c = Some(30.2);
                r.humidity = Some(40.0);
                r.battery_ok = Some(false);
                r.wind_avg_km_h = Some(3.0 * 1.224);
                r.wind_max_km_h = Some(8.0 * 1.224);
                r.wind_dir_deg = Some(270.0);
                r.rain_mm = Some(113.0 * 0.3);
            }),
            100e3,
        ),
        (
            reading("oregon_thn132n", 0x3C, |r| {
                r.channel = Some(2);
                r.battery_ok = Some(true);
                r.temperature_c = Some(-12.0);
            }),
            350e3,
        ),
        (
            reading("oregon_thgr122nx", 0xE1, |r| {
                r.channel = Some(1);
                r.battery_ok = Some(true);
                r.temperature_c = Some(24.9);
                r.humidity = Some(33.0);
            }),
            -650e3,
        ),
    ];
    check_flow("ism433_sensors", 433.92e6, &rounds(&items, 5, 0, -22.0));
}

#[test]
fn the_868_flow_reports_the_lacrosse_sensors_once_each() {
    let tx29 = |id: u32, t: f32, h: Option<f32>, ok: bool| {
        reading("lacrosse_tx29", id, |r| {
            r.temperature_c = Some(t);
            r.humidity = h;
            r.battery_ok = Some(ok);
        })
    };
    let tx35 = |id: u32, t: f32, h: Option<f32>, ok: bool| {
        reading("lacrosse_tx35", id, |r| {
            r.temperature_c = Some(t);
            r.humidity = h;
            r.battery_ok = Some(ok);
        })
    };
    let items = vec![
        (tx29(8, -0.6, Some(89.0), true), -700e3),
        (tx35(10, 22.4, Some(40.0), true), -200e3),
        (tx29(33, 18.2, None, false), 300e3),
        (tx35(61, -15.7, Some(77.0), true), 800e3),
        (tx35(5, 31.5, None, true), -450e3),
        (tx29(42, 4.4, Some(64.0), false), 50e3),
        (tx35(17, 9.9, Some(51.0), false), 550e3),
        (tx29(2, 25.0, Some(30.0), true), -700e3),
    ];
    check_flow("ism868_sensors", 868.5e6, &rounds(&items, 3, 1, -22.0));
}
