//! Golden protocol cases of the sensor decoders (M46 tests 2, 4 and 5).
//!
//! Test 2: for every decoder (`wh2`, `nexus`, `oregon_v21`, `wh1080`, `lacrosse_tx29`, `bresser_3ch`)
//! the golden case holds 20 trains written by the independent Python encoder (seeds 637, 638, 639,
//! 644, 645, 649,
//! recorded in the case) with timing error; the Rust decoder must return exactly the encoded
//! readings (one per copy, one for the device without a check and for `bresser_3ch`, whose frames
//! are merged and counted in `repeats`) and no other decoder may return a
//! reading for the same train. The worked examples of the public descriptions are rendered here
//! and decoded as well.
//!
//! Test 4 (seed 646): a flipped bit in the payload or the check of a device with a CRC or checksum
//! gives no reading (`BadIntegrity`); Nexus, a device without a check, gives a reading only when
//! three equal frames are left (D1138). Test 5 (seed 647): out-of-range values and truncated rows give no
//! reading. The trains of tests 4 and 5 are rendered by the helpers below from the coding rules.
//! The corrections of the fact document (D1136 to D1142) have their own tests at the end of the
//! file (seed 776: the choice of the ids; 777: the frames that pass the Rubicson check).
//! The case `bresser_3ch_recorded` holds the four messages recorded from the sensor (22.11 C 64 %, 15.5 C 60 %,
//! 22.5 C 71 % and 14.39 C 81 %, D1119, two decimals D1123).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::type_complexity
)]

use sdrk_proto::pulses::{self, Header, Modulation, Pair};
use sdrk_proto::sensors::{self, DecodeResult, Mic, Reading, ReadingSink};
use sdrk_proto::slicers::{Convention, Manchester, Nrz, Ppm, Pwm, Rows, Slicer};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::rng::Pcg32;

const RATE: u32 = 250_000;
const CASES: [&str; 7] = [
    "bresser_3ch",
    "lacrosse_tx29",
    "lacrosse_tx35",
    "nexus",
    "oregon_v21",
    "wh1080",
    "wh2",
];
/// The case of the three recorded messages of `bresser_3ch` (not a generated case).
const RECORDED: &str = "bresser_3ch_recorded";
/// The seeds of tests 4 and 5.
const SEED_INTEGRITY: u64 = 646;
const SEED_RANGE: u64 = 647;
/// The seeds of the tests of the corrected protocols (D1143).
const SEED_CORRECTED: u64 = 776;
const SEED_RUBICSON: u64 = 777;

#[test]
fn the_cases_of_m46_are_present() {
    let mut names = golden::list("sensors").unwrap();
    names.sort();
    let mut want: Vec<&str> = CASES.to_vec();
    want.push(RECORDED);
    want.sort_unstable();
    assert_eq!(names, want);
    for (name, seed) in [
        ("bresser_3ch", 649),
        (RECORDED, 649),
        ("wh2", 637),
        ("nexus", 638),
        ("oregon_v21", 639),
        ("wh1080", 644),
        ("lacrosse_tx29", 645),
        ("lacrosse_tx35", 775),
    ] {
        assert_eq!(golden::load("sensors", name).unwrap().seed, seed, "{name}");
    }
}

fn i32s(c: &GoldenCase, name: &str) -> Vec<i32> {
    match c.array(name).unwrap() {
        Array::I32(v) => v.clone(),
        other => panic!("{name} is {}", other.dtype()),
    }
}

fn f64s(c: &GoldenCase, name: &str) -> Vec<f64> {
    match c.array(name).unwrap() {
        Array::F64(v) => v.clone(),
        other => panic!("{name} is {}", other.dtype()),
    }
}

fn modulation_of(decoder: &str) -> Modulation {
    sensors::by_name(decoder).unwrap().modulation()
}

fn header(m: Modulation) -> Header {
    Header {
        modulation: m,
        channel: 0,
        freq_offset_hz: 0,
        level_db10: -250,
        noise_db10: -600,
        rate_hz: RATE,
    }
}

/// The trains of a case: the pairs of train `i`.
fn trains(c: &GoldenCase) -> Vec<Vec<Pair>> {
    let counts = i32s(c, "pair_counts");
    let flat = i32s(c, "pairs");
    let mut at = 0;
    let mut out = Vec::new();
    for n in counts {
        let n = n as usize;
        let pairs = flat[at..at + 2 * n]
            .chunks_exact(2)
            .map(|w| Pair {
                pulse: w[0] as u32,
                gap: w[1] as u32,
            })
            .collect();
        at += 2 * n;
        out.push(pairs);
    }
    assert_eq!(at, flat.len());
    out
}

fn decode_with(decoder: &str, m: Modulation, pairs: &[Pair]) -> (DecodeResult, Vec<Reading>) {
    let h = header(m);
    let mut buf = vec![0u8; 16 + 8 * pairs.len()];
    let n = pulses::encode_joined(&h, pairs, &mut buf).unwrap();
    let train = pulses::decode(&buf[..n]).unwrap();
    let mut sink = ReadingSink::new();
    let res = sensors::by_name(decoder).unwrap().decode(&train, &mut sink);
    (res, sink.iter().copied().collect())
}

fn mic_of(decoder: &str) -> Mic {
    match decoder {
        "nexus" => Mic::Repeat,
        "oregon_v21" | "bresser_3ch" => Mic::Checksum,
        _ => Mic::Crc,
    }
}

fn opt(v: f64) -> Option<f32> {
    if v.is_nan() { None } else { Some(v as f32) }
}

fn assert_close(got: Option<f32>, want: Option<f32>, what: &str) {
    match (got, want) {
        (None, None) => {}
        (Some(g), Some(w)) => assert!((g - w).abs() < 1e-3, "{what}: {g} != {w}"),
        _ => panic!("{what}: {got:?} != {want:?}"),
    }
}

#[test]
fn every_decoder_returns_the_encoded_readings() {
    for name in CASES {
        let c = golden::load("sensors", name).unwrap();
        let decoder = c.param_str("decoder").unwrap().to_string();
        assert_eq!(decoder, name);
        let per_train = c.param_u64("readings_per_train").unwrap() as usize;
        let repeats = c.param_u64("repeats").unwrap() as u32;
        let models: Vec<String> = c.params["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let (ids, chans, bats, mids) = (
            i32s(&c, "ids"),
            i32s(&c, "channels"),
            i32s(&c, "battery_ok"),
            i32s(&c, "model_ids"),
        );
        let temp = f64s(&c, "temperature_c");
        let hum = f64s(&c, "humidity");
        let wavg = f64s(&c, "wind_avg_km_h");
        let wmax = f64s(&c, "wind_max_km_h");
        let wdir = f64s(&c, "wind_dir_deg");
        let rain = f64s(&c, "rain_mm");
        let list = trains(&c);
        assert_eq!(list.len(), 20);
        let frames = if decoder == "bresser_3ch" {
            i32s(&c, "repeat_counts")
        } else {
            Vec::new()
        };
        let m = modulation_of(&decoder);
        for (i, pairs) in list.iter().enumerate() {
            let (res, readings) = decode_with(&decoder, m, pairs);
            assert_eq!(res, DecodeResult::Decoded(per_train), "{name} train {i}");
            assert_eq!(readings.len(), per_train, "{name} train {i}");
            for r in readings {
                let ctx = format!("{name} train {i}");
                assert_eq!(r.model, models[mids[i] as usize], "{ctx}");
                assert_eq!(r.id, ids[i] as u32, "{ctx} id");
                assert_eq!(
                    r.channel,
                    u8::try_from(chans[i]).ok().filter(|&x| x > 0),
                    "{ctx}"
                );
                let bat = match bats[i] {
                    -1 => None,
                    b => Some(b == 1),
                };
                assert_eq!(r.battery_ok, bat, "{ctx} battery");
                assert_close(r.temperature_c, opt(temp[i]), &format!("{ctx} temperature"));
                assert_close(r.humidity, opt(hum[i]), &format!("{ctx} humidity"));
                assert_close(r.wind_avg_km_h, opt(wavg[i]), &format!("{ctx} wind avg"));
                assert_close(r.wind_max_km_h, opt(wmax[i]), &format!("{ctx} wind max"));
                assert_close(r.wind_dir_deg, opt(wdir[i]), &format!("{ctx} wind dir"));
                assert_close(r.rain_mm, opt(rain[i]), &format!("{ctx} rain"));
                assert_eq!(r.pressure_hpa, None, "{ctx}");
                assert_eq!(r.mic, mic_of(&decoder), "{ctx}");
                let want_repeats = match decoder.as_str() {
                    "nexus" => repeats,
                    "bresser_3ch" => frames[i] as u32,
                    _ => 1,
                };
                assert_eq!(r.repeats, want_repeats, "{ctx}");
                assert_eq!(r.rssi_db, -25.0, "{ctx}");
            }
            // no other decoder reads the train of this device
            for other in sensors::DECODERS {
                if other.name() == decoder || other.modulation() != m {
                    continue;
                }
                let (res, readings) = decode_with(other.name(), m, pairs);
                assert!(
                    readings.is_empty(),
                    "{} read a train of {name}: {res:?}",
                    other.name()
                );
            }
        }
    }
}

// ---------------------------------------------------------------- rendering of trains

fn samples(us: f64) -> u32 {
    (us * f64::from(RATE) / 1.0e6).round() as u32
}

fn to_pairs(runs: &[(f64, f64)]) -> Vec<Pair> {
    runs.iter()
        .map(|&(p, g)| Pair {
            pulse: samples(p),
            gap: samples(g),
        })
        .collect()
}

fn pwm_runs(bits: &[bool]) -> Vec<(f64, f64)> {
    bits.iter()
        .map(|&b| (if b { 500.0 } else { 1500.0 }, 1000.0))
        .collect()
}

/// The pulse widths of the WH1080: 544 and 1524 us (D1140).
fn pwm_runs_wh1080(bits: &[bool]) -> Vec<(f64, f64)> {
    bits.iter()
        .map(|&b| (if b { 544.0 } else { 1524.0 }, 1000.0))
        .collect()
}

fn ppm_runs(bits: &[bool]) -> Vec<(f64, f64)> {
    let mut v: Vec<_> = bits
        .iter()
        .map(|&b| (500.0, if b { 2000.0 } else { 1000.0 }))
        .collect();
    v.push((500.0, 4000.0));
    v
}

/// Runs of `unit` us from a level per unit; leading low and trailing low units are dropped.
fn level_runs(levels: &[bool], unit: f64) -> Vec<(f64, f64)> {
    let first = levels.iter().position(|&l| l).unwrap();
    let last = levels.iter().rposition(|&l| l).unwrap();
    let mut runs: Vec<(bool, f64)> = Vec::new();
    for &l in &levels[first..=last] {
        match runs.last_mut() {
            Some(r) if r.0 == l => r.1 += unit,
            _ => runs.push((l, unit)),
        }
    }
    runs.chunks(2)
        .map(|c| (c[0].1, c.get(1).map_or(0.0, |g| g.1)))
        .collect()
}

fn manchester_runs(bits: &[bool]) -> Vec<(f64, f64)> {
    let levels: Vec<bool> = bits.iter().flat_map(|&b| [b, !b]).collect();
    level_runs(&levels, 1.0e6 / 2048.0)
}

fn nrz_runs(bits: &[bool]) -> Vec<(f64, f64)> {
    level_runs(bits, 58.0)
}

/// Copies joined by `gap` us; the gap of the last pair is 0.
fn join(copies: &[Vec<(f64, f64)>], gap: f64) -> Vec<Pair> {
    let mut runs: Vec<(f64, f64)> = Vec::new();
    for c in copies {
        if let Some(last) = runs.last_mut() {
            last.1 = gap;
        }
        runs.extend_from_slice(c);
    }
    runs.last_mut().unwrap().1 = 0.0;
    to_pairs(&runs)
}

fn bits_of(v: u64, n: usize) -> Vec<bool> {
    (0..n).map(|i| (v >> (n - 1 - i)) & 1 == 1).collect()
}

fn bytes_bits(b: &[u8]) -> Vec<bool> {
    b.iter().flat_map(|&x| bits_of(u64::from(x), 8)).collect()
}

fn crc_msb(data: &[u8]) -> u8 {
    crc_msb_init(0, data)
}

fn crc_msb_init(init: u8, data: &[u8]) -> u8 {
    let mut crc = init;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x31
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// The reflected CRC of the earlier assumption (D1106), which the fact document replaced.
fn crc_dallas(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0x8C
            } else {
                crc >> 1
            };
        }
    }
    crc
}

/// The WH2 message with the type nibble `ty` and the CRC `crc` of the payload.
fn wh2_bits_with(ty: u8, id: u32, raw_temp: u32, hum: u8, crc: fn(&[u8]) -> u8) -> Vec<bool> {
    let p = [
        (ty << 4) | (id >> 4) as u8,
        (((id & 0xF) << 4) | (raw_temp >> 8)) as u8,
        raw_temp as u8,
        hum,
    ];
    let mut b = vec![0xFF];
    b.extend_from_slice(&p);
    b.push(crc(&p));
    bytes_bits(&b)
}

fn wh2_bits(id: u32, raw_temp: u32, hum: u8) -> Vec<bool> {
    wh2_bits_with(4, id, raw_temp, hum, crc_msb)
}

fn nexus_bits(id: u8, bat: bool, ch: u8, temp12: u32, hum: u8) -> Vec<bool> {
    nexus_bits_test(id, bat, false, ch, temp12, hum)
}

/// A Nexus frame with the test-button bit.
fn nexus_bits_test(id: u8, bat: bool, test: bool, ch: u8, temp12: u32, hum: u8) -> Vec<bool> {
    let mut v = bits_of(u64::from(id), 8);
    v.extend([bat, test]);
    v.extend(bits_of(u64::from(ch), 2));
    v.extend(bits_of(u64::from(temp12), 12));
    v.extend([true; 4]);
    v.extend(bits_of(u64::from(hum), 8));
    v
}

fn oregon_bits(nibbles: &[u8]) -> Vec<bool> {
    oregon_bits_post(nibbles, &[0, 0])
}

/// The Oregon data bits with a chosen post-amble (none: the message ends with its checksum).
fn oregon_bits_post(nibbles: &[u8], post: &[u8]) -> Vec<bool> {
    let mut data = vec![true; 16];
    for n in std::iter::once(&0xA).chain(nibbles).chain(post) {
        data.extend((0..4).map(|j| (n >> j) & 1 == 1));
    }
    data.iter().flat_map(|&d| [!d, d]).collect()
}

/// The nibbles of a hex string as written in the descriptions.
fn nibbles(s: &str) -> Vec<u8> {
    s.chars().map(|c| c.to_digit(16).unwrap() as u8).collect()
}

/// Appends the checksum (sum of the nibbles, low nibble first) to the payload nibbles.
fn with_sum(mut n: Vec<u8>) -> Vec<u8> {
    let s: u32 = n.iter().map(|&x| u32::from(x)).sum();
    n.push((s & 0xF) as u8);
    n.push(((s >> 4) & 0xF) as u8);
    n
}

fn tx29_bits(frame: [u8; 5]) -> Vec<bool> {
    tx29_bits_pre(&[0xAA; 4], frame)
}

/// A TX29 frame with the preamble bytes `pre`.
fn tx29_bits_pre(pre: &[u8], frame: [u8; 5]) -> Vec<bool> {
    let mut b = pre.to_vec();
    b.extend([0x2D, 0xD4]);
    b.extend_from_slice(&frame);
    bytes_bits(&b)
}

fn tx29_frame(b: [u8; 4]) -> [u8; 5] {
    [b[0], b[1], b[2], b[3], crc_msb(&b)]
}

/// One device: the slicer that reads its rows, the renderer of a row and the gap between copies.
struct Dev {
    name: &'static str,
    slicer: Box<dyn Slicer>,
    render: fn(&[bool]) -> Vec<(f64, f64)>,
    gap: f64,
}

fn dev(name: &str) -> Dev {
    match name {
        "wh2" => Dev {
            name: "wh2",
            slicer: Box::new(Pwm::new(500.0, 1500.0, 1000.0, 2500.0, true).unwrap()),
            render: pwm_runs,
            gap: 8000.0,
        },
        "wh1080" => Dev {
            name: "wh1080",
            slicer: Box::new(Pwm::new(544.0, 1524.0, 1000.0, 2500.0, true).unwrap()),
            render: pwm_runs_wh1080,
            gap: 8000.0,
        },
        "nexus" => Dev {
            name: "nexus",
            slicer: Box::new(Ppm::new(1000.0, 2000.0, 3000.0).unwrap()),
            render: ppm_runs,
            gap: 4000.0,
        },
        "oregon_v21" => Dev {
            name: "oregon_v21",
            slicer: Box::new(
                Manchester::new(1.0e6 / 2048.0, 3000.0, Convention::Ieee8023)
                    .unwrap()
                    .with_virtual_lead(true),
            ),
            render: manchester_runs,
            gap: 8000.0,
        },
        "lacrosse_tx29" => Dev {
            name: "lacrosse_tx29",
            slicer: Box::new(Nrz::new(58.0, 2000.0).unwrap()),
            render: nrz_runs,
            gap: 4000.0,
        },
        _ => panic!("{name}"),
    }
}

impl Dev {
    fn m(&self) -> Modulation {
        modulation_of(self.name)
    }

    /// The rows of a train as bit vectors.
    fn rows(&self, pairs: &[Pair]) -> Vec<Vec<bool>> {
        let h = header(self.m());
        let mut buf = vec![0u8; 16 + 8 * pairs.len()];
        let n = pulses::encode_joined(&h, pairs, &mut buf).unwrap();
        let train = pulses::decode(&buf[..n]).unwrap();
        let mut rows = Rows::new();
        self.slicer.slice(&train, &mut rows);
        rows.iter().map(|r| r.iter().collect()).collect()
    }

    fn train(&self, rows: &[Vec<bool>]) -> Vec<Pair> {
        let copies: Vec<_> = rows.iter().map(|r| (self.render)(r)).collect();
        join(&copies, self.gap)
    }

    fn decode(&self, pairs: &[Pair]) -> (DecodeResult, Vec<Reading>) {
        decode_with(self.name, self.m(), pairs)
    }
}

// ---------------------------------------------------------------- worked examples

fn assert_reading(r: &Reading, model: &str, id: u32, ch: Option<u8>, t: f32, h: Option<f32>) {
    assert_eq!(r.model, model);
    assert_eq!(r.id, id);
    assert_eq!(r.channel, ch);
    assert_close(r.temperature_c, Some(t), "temperature");
    assert_close(r.humidity, h, "humidity");
}

#[test]
fn the_worked_examples_of_the_descriptions_decode() {
    // Nexus: 5c910ff38 is id 0x5c, battery good, channel 2, 27.1 C, 56 %; three equal frames.
    let d = dev("nexus");
    let bits = bits_of(0x5_c910_ff38, 36);
    let frames = vec![(d.render)(&bits); 3];
    let (res, r) = d.decode(&join(&frames, d.gap));
    assert_eq!(res, DecodeResult::Decoded(1));
    assert_reading(&r[0], "nexus", 0x5c, Some(2), 27.1, Some(56.0));
    assert_eq!((r[0].battery_ok, r[0].repeats), (Some(true), 3));

    // Oregon Scientific v2.1, two examples of the document (the sensor 1D20).
    let d = dev("oregon_v21");
    for (hex, id, ch, t, h, bat) in [
        ("1D20485C480882835", 0x58, 3, -8.4, 28.0, false),
        ("1D2016B1091073A14", 0xB6, 1, 19.0, 37.0, true),
    ] {
        let n = nibbles(hex);
        let body = &n[..n.len() - 2];
        assert_eq!(
            with_sum(body.to_vec()),
            n,
            "{hex}: the checksum of the document"
        );
        let (res, r) = d.decode(&join(&[manchester_runs(&oregon_bits(&n))], d.gap));
        assert_eq!(res, DecodeResult::Decoded(1), "{hex}");
        assert_reading(&r[0], "oregon_thgr122nx", id, Some(ch), t, Some(h));
        assert_eq!(r[0].battery_ok, Some(bat));
    }

    // LaCrosse IT+: the two examples of the article (the ids are 8 and 10 with the 6 bit id of
    // the fact document, D1141; they were 32 and 40 with the whole byte of D1110).
    let d = dev("lacrosse_tx29");
    for (frame, id, t, h) in [
        ([0x92, 0x03, 0x94, 0x59, 0x3C], 8, -0.6, 89.0),
        ([0x92, 0x86, 0x24, 0x28, 0xDE], 10, 22.4, 40.0),
    ] {
        let (res, r) = d.decode(&join(&[nrz_runs(&tx29_bits(frame))], d.gap));
        assert_eq!(res, DecodeResult::Decoded(1), "{frame:02X?}");
        assert_reading(&r[0], "lacrosse_tx29", id, None, t, Some(h));
        assert_eq!(r[0].battery_ok, Some(true));
    }
}

// ---------------------------------------------------------------- test 4: integrity

/// The bit positions flipped per device: in the payload and in the check, as row bits (for
/// Oregon as data bits: payload nibble 8, the channel nibble and nibble 12).
fn flips(name: &str) -> &'static [usize] {
    match name {
        "wh2" => &[21, 35, 44],
        "wh1080" => &[20, 50, 85],
        "lacrosse_tx29" => &[60, 68, 76],
        "oregon_v21" => &[53, 40, 70],
        _ => panic!("{name}"),
    }
}

#[test]
fn a_flipped_bit_gives_no_reading() {
    let mut rng = Pcg32::new(SEED_INTEGRITY, 1);
    for name in ["wh2", "wh1080", "lacrosse_tx29", "oregon_v21"] {
        let c = golden::load("sensors", name).unwrap();
        let d = dev(name);
        for (i, pairs) in trains(&c).iter().enumerate() {
            let mut rows = d.rows(pairs);
            assert_ne!(rows.len(), 0);
            // the unmodified train decodes (the rendering is the one of the encoder)
            let (res, r) = d.decode(&d.train(&rows));
            assert!(
                matches!(res, DecodeResult::Decoded(n) if n == r.len()),
                "{name} {i}"
            );
            let at = flips(name)[rng.below(3) as usize];
            for row in &mut rows {
                if name == "oregon_v21" {
                    row[2 * at] = !row[2 * at];
                    row[2 * at + 1] = !row[2 * at + 1];
                } else {
                    row[at] = !row[at];
                }
            }
            let (res, r) = d.decode(&d.train(&rows));
            assert!(r.is_empty(), "{name} train {i} flip {at}: {r:?}");
            assert_eq!(
                res,
                DecodeResult::BadIntegrity,
                "{name} train {i} flip {at}"
            );
        }
    }
}

#[test]
fn nexus_needs_three_equal_frames() {
    let c = golden::load("sensors", "nexus").unwrap();
    let d = dev("nexus");
    let mut rng = Pcg32::new(SEED_INTEGRITY, 2);
    for (i, pairs) in trains(&c).iter().enumerate() {
        let rows = d.rows(pairs);
        assert_eq!(rows.len(), 3, "train {i}");
        assert!(rows.iter().all(|r| r == &rows[0]));
        let (a, b) = (12 + rng.below(8) as usize, 20 + rng.below(4) as usize);
        let flip = |row: &Vec<bool>, p: usize| {
            let mut r = row.clone();
            r[p] = !r[p];
            r
        };
        // three equal frames: one reading
        let (res, r) = d.decode(&d.train(&rows));
        assert_eq!(
            (res, r.len(), r[0].repeats),
            (DecodeResult::Decoded(1), 1, 3)
        );
        // four frames, one flipped: the other three still agree
        let t = d.train(&[
            flip(&rows[0], a),
            rows[1].clone(),
            rows[2].clone(),
            rows[0].clone(),
        ]);
        let (res, r) = d.decode(&t);
        assert_eq!(
            (res, r.len(), r[0].repeats),
            (DecodeResult::Decoded(1), 1, 3)
        );
        // two equal frames: no reading, the device needs three
        let (res, r) = d.decode(&d.train(&rows[..2]));
        assert!(r.is_empty(), "train {i}: {r:?}");
        assert_eq!(res, DecodeResult::BadIntegrity);
        // three frames with one flipped: only two agree
        let t = d.train(&[flip(&rows[0], a), rows[1].clone(), rows[2].clone()]);
        let (res, r) = d.decode(&t);
        assert!(r.is_empty(), "train {i}: {r:?}");
        assert_eq!(res, DecodeResult::BadIntegrity);
        // two frames flipped in different places: only one frame is left
        let t = d.train(&[flip(&rows[0], a), flip(&rows[1], b), rows[2].clone()]);
        let (res, r) = d.decode(&t);
        assert!(r.is_empty(), "train {i}: {r:?}");
        assert_eq!(res, DecodeResult::BadIntegrity);
        // a single frame
        let (res, r) = d.decode(&d.train(&rows[..1]));
        assert_eq!(r.len(), 0);
        assert_eq!(res, DecodeResult::BadIntegrity);
    }
}

// ---------------------------------------------------------------- test 5: range and truncation

/// A train of two copies of `bits` of device `name`.
fn two_copies(name: &str, bits: &[bool]) -> Vec<Pair> {
    let d = dev(name);
    join(&[(d.render)(bits), (d.render)(bits)], d.gap)
}

fn three_copies(bits: &[bool]) -> Vec<Pair> {
    let d = dev("nexus");
    join(&vec![(d.render)(bits); 3], d.gap)
}

#[test]
#[allow(clippy::too_many_lines)]
fn out_of_range_values_give_no_reading() {
    // (device, train): every train has a valid check and a value the device cannot send
    let oregon = |dev_id: [u8; 4], ch: u8, temp: [u8; 4], hum: Option<[u8; 2]>| {
        let mut n = dev_id.to_vec();
        n.extend([ch, 0x3, 0x4, 0x0]);
        n.extend(temp);
        if let Some(h) = hum {
            n.extend(h);
            n.push(0);
        }
        oregon_bits(&with_sum(n))
    };
    let wh1080 = |temp_raw: u32, hum: u8| {
        let b = [
            0xFFu8,
            0xA1,
            0x30 | (temp_raw >> 8) as u8,
            temp_raw as u8,
            hum,
            3,
            4,
            0,
            5,
            0x1A,
        ];
        let crc = crc_msb(&b[1..10]);
        let mut v = b.to_vec();
        v.push(crc);
        bytes_bits(&v)
    };
    let cases: Vec<(&str, Vec<Pair>)> = vec![
        ("wh2", two_copies("wh2", &wh2_bits(0x12, 900, 50))),
        ("wh2", two_copies("wh2", &wh2_bits(0x12, 0x800 | 0xFA, 101))),
        ("nexus", three_copies(&nexus_bits(0x5C, true, 1, 900, 50))),
        ("nexus", three_copies(&nexus_bits(0x5C, true, 1, 250, 101))),
        ("nexus", three_copies(&nexus_bits(0x5C, true, 1, 0xE0C, 50))),
        (
            "oregon_v21",
            two_copies(
                "oregon_v21",
                &oregon([0xE, 0xC, 0x4, 0x0], 1, [0xA, 5, 2, 0], None),
            ),
        ),
        (
            "oregon_v21",
            two_copies(
                "oregon_v21",
                &oregon([0xE, 0xC, 0x4, 0x0], 3, [4, 5, 2, 0], None),
            ),
        ),
        (
            "oregon_v21",
            two_copies(
                "oregon_v21",
                &oregon([0x1, 0xD, 0x2, 0x0], 1, [4, 5, 2, 0], Some([0xB, 3])),
            ),
        ),
        (
            "oregon_v21",
            two_copies(
                "oregon_v21",
                &oregon([0x1, 0xD, 0x2, 0x0], 1, [4, 5, 2, 0], Some([1, 10])),
            ),
        ),
        ("wh1080", two_copies("wh1080", &wh1080(800, 120))),
        (
            "lacrosse_tx29",
            two_copies(
                "lacrosse_tx29",
                &tx29_bits(tx29_frame([0x92, 0x03, 0xAA, 0x59])),
            ),
        ),
        (
            "lacrosse_tx29",
            two_copies(
                "lacrosse_tx29",
                &tx29_bits(tx29_frame([0x92, 0x03, 0x94, 107])),
            ),
        ),
        (
            // the humidity 125 is the probe of the TX25U
            "lacrosse_tx29",
            two_copies(
                "lacrosse_tx29",
                &tx29_bits(tx29_frame([0x92, 0x03, 0x94, 125])),
            ),
        ),
        // the sign nibble of the Oregon sensors is 8 or 0: other bits are out of range
        (
            "oregon_v21",
            two_copies(
                "oregon_v21",
                &oregon([0xE, 0xC, 0x4, 0x0], 1, [4, 5, 2, 0x9], None),
            ),
        ),
        (
            // THN132N below -50.0 C
            "oregon_v21",
            two_copies(
                "oregon_v21",
                &oregon([0xE, 0xC, 0x4, 0x0], 1, [1, 0, 5, 0x8], None),
            ),
        ),
    ];
    for (name, pairs) in &cases {
        let (res, r) = decode_with(name, modulation_of(name), pairs);
        assert!(r.is_empty(), "{name}: {r:?}");
        assert_eq!(res, DecodeResult::OutOfRange, "{name}");
    }
}

#[test]
fn truncated_rows_give_no_reading() {
    let mut rng = Pcg32::new(SEED_RANGE, 1);
    for name in CASES {
        let c = golden::load("sensors", name).unwrap();
        let copies = c.param_u64("repeats").unwrap() as usize;
        let frames = if name == "bresser_3ch" {
            i32s(&c, "repeat_counts")
        } else {
            Vec::new()
        };
        for (i, pairs) in trains(&c).iter().enumerate() {
            // keep a part of the first copy only: 55 to 90 percent of it
            let copies = frames.get(i).map_or(copies, |&n| n as usize);
            let copy = pairs.len() / copies;
            let keep = copy * (55 + rng.below(36) as usize) / 100;
            let mut cut = pairs[..keep].to_vec();
            cut.last_mut().unwrap().gap = 0;
            let (res, r) = decode_with(name, modulation_of(name), &cut);
            assert!(r.is_empty(), "{name} train {i}: {r:?}");
            assert!(
                matches!(res, DecodeResult::Truncated | DecodeResult::NoMatch),
                "{name} train {i}: {res:?}"
            );
        }
    }
}

#[test]
fn a_whole_row_that_is_not_a_message_gives_no_match() {
    // pulses of 500 us every 1500 us: a row of ones without a preamble match for any device
    let pairs = vec![
        Pair {
            pulse: 125,
            gap: 250
        };
        60
    ];
    let mut pairs = pairs;
    pairs.last_mut().unwrap().gap = 0;
    for dcd in sensors::DECODERS {
        let (res, r) = decode_with(dcd.name(), dcd.modulation(), &pairs);
        assert!(r.is_empty(), "{}", dcd.name());
        assert!(!matches!(res, DecodeResult::Decoded(_)), "{}", dcd.name());
    }
}

// ---------------------------------------------------------------- bresser_3ch (D1119)

/// The frame bytes with their sum, from the id, the byte 1 (flags and high temperature bits),
/// the low temperature byte and the humidity.
fn bresser_frame(id: u8, b1: u8, b2: u8, hum: u8) -> [u8; 5] {
    let sum = id.wrapping_add(b1).wrapping_add(b2).wrapping_add(hum);
    [id, b1, b2, hum, sum]
}

/// The frame bytes of a raw temperature value with the flags `flags` in the high nibble of byte
/// 1 (0x6: battery ok, button pressed, channel 2, as in all recordings).
fn bresser_raw(id: u8, flags: u8, raw: u32, hum: u8) -> [u8; 5] {
    bresser_frame(id, (flags << 4) | (raw >> 8) as u8, raw as u8, hum)
}

/// `preamble` pairs (735 us) then the 40 bits of `frame` (1: 500/250 us, 0: 250/500 us).
fn bresser_frame_runs(frame: [u8; 5], preamble: usize) -> Vec<(f64, f64)> {
    let mut runs = vec![(735.0, 735.0); preamble];
    runs.extend(
        bytes_bits(&frame)
            .iter()
            .map(|&b| if b { (500.0, 250.0) } else { (250.0, 500.0) }),
    );
    runs
}

/// A burst of the frames in `frames` back to back; the last gap is 0 (the pause).
fn bresser_train(frames: &[[u8; 5]], preamble: usize) -> Vec<Pair> {
    let mut runs: Vec<(f64, f64)> = frames
        .iter()
        .flat_map(|&f| bresser_frame_runs(f, preamble))
        .collect();
    runs.last_mut().unwrap().1 = 0.0;
    to_pairs(&runs)
}

fn bresser_decode(pairs: &[Pair]) -> (DecodeResult, Vec<Reading>) {
    decode_with("bresser_3ch", Modulation::Ook, pairs)
}

#[test]
fn the_three_recorded_messages_decode_to_the_display_values() {
    let c = golden::load("sensors", RECORDED).unwrap();
    assert_eq!(c.param_str("decoder").unwrap(), "bresser_3ch");
    assert_eq!(c.params["recorded"], true);
    let list = trains(&c);
    assert_eq!(list.len(), 4);
    let frames = i32s(&c, "repeat_counts");
    // recording 1 (display not known), 2 (22.1 C, 64 %), 3 (22.5 C, 71 %) and 4 (15.5 C, 60 %)
    let want = [
        (14.39f32, 81.0f32),
        (22.11, 64.0),
        (22.5, 71.0),
        (15.5, 60.0),
    ];
    for (i, pairs) in list.iter().enumerate() {
        let (res, r) = bresser_decode(pairs);
        assert_eq!(res, DecodeResult::Decoded(1), "recording {}", i + 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].model, "Bresser-3CH");
        assert_eq!(r[0].id, 0xD8);
        assert_eq!((r[0].channel, r[0].battery_ok), (Some(2), Some(true)));
        assert_eq!(r[0].mic, Mic::Checksum);
        assert_eq!(r[0].repeats, frames[i] as u32);
        assert_eq!(r[0].temperature_c, Some(want[i].0), "recording {}", i + 1);
        assert_eq!(r[0].humidity, Some(want[i].1), "recording {}", i + 1);
        // no other decoder reads it
        for other in sensors::DECODERS {
            if other.name() != "bresser_3ch" {
                let (_, r) = decode_with(other.name(), other.modulation(), pairs);
                assert!(r.is_empty(), "{}", other.name());
            }
        }
    }
    // the same bytes rendered here give the same readings
    let frames = [
        [0xD8, 0x65, 0xC7, 0x51, 0x55],
        [0xD8, 0x66, 0x52, 0x40, 0xD0],
        [0xD8, 0x66, 0x59, 0x47, 0xDE],
        [0xD8, 0x65, 0xDB, 0x3C, 0x54],
    ];
    for (f, w) in frames.iter().zip(want) {
        assert_eq!(*f, bresser_frame(f[0], f[1], f[2], f[3]));
        let (_, r) = bresser_decode(&bresser_train(&[*f; 23], 4));
        assert_eq!((r[0].temperature_c, r[0].humidity), (Some(w.0), Some(w.1)));
    }
}

#[test]
fn bresser_flags_are_battery_and_channel() {
    // high nibble: bit 3 battery low, bit 2 button, bits 1 and 0 the channel
    for (flags, ch, bat) in [
        (0x6u8, 2u8, true),
        (0x1, 1, true),
        (0x3, 3, true),
        (0xA, 2, false),
        (0xD, 1, false),
        (0xF, 3, false),
    ] {
        let (res, r) = bresser_decode(&bresser_train(&[bresser_raw(7, flags, 1618, 64); 3], 4));
        assert_eq!(res, DecodeResult::Decoded(1), "{flags:X}");
        assert_eq!((r[0].channel, r[0].battery_ok), (Some(ch), Some(bat)));
    }
    // channel 0 is not a frame of the device
    for flags in [0x0u8, 0x4, 0x8, 0xC] {
        let (res, r) = bresser_decode(&bresser_train(&[bresser_raw(7, flags, 1618, 64); 3], 4));
        assert_eq!(r.len(), 0);
        assert_eq!(res, DecodeResult::NoMatch, "{flags:X}");
    }
}

#[test]
fn bresser_one_frame_with_a_valid_sum_is_a_reading() {
    for preamble in 2..=6 {
        let (res, r) = bresser_decode(&bresser_train(&[bresser_raw(0x11, 6, 1618, 64)], preamble));
        assert_eq!(res, DecodeResult::Decoded(1), "preamble {preamble}");
        assert_eq!(
            (r[0].id, r[0].temperature_c, r[0].repeats),
            (0x11, Some(22.11), 1)
        );
    }
    // one preamble pair is not enough, and neither is none
    for preamble in [0, 1] {
        let (res, r) = bresser_decode(&bresser_train(&[bresser_raw(0x11, 6, 1618, 64)], preamble));
        assert!(r.is_empty(), "preamble {preamble}");
        assert_eq!(res, DecodeResult::NoMatch);
    }
}

#[test]
fn bresser_a_flipped_bit_gives_no_reading() {
    let mut rng = Pcg32::new(SEED_INTEGRITY, 3);
    let c = golden::load("sensors", "bresser_3ch").unwrap();
    let frames = i32s(&c, "repeat_counts");
    let preambles = i32s(&c, "preamble_pairs");
    let ids = i32s(&c, "ids");
    for i in 0..20 {
        let raw = 700 + rng.below(1801);
        let hum = rng.below(101) as u8;
        let good = bresser_raw(ids[i] as u8, 6, raw, hum);
        let n = frames[i] as usize;
        let preamble = preambles[i] as usize;
        let (res, r) = bresser_decode(&bresser_train(&vec![good; n], preamble));
        assert_eq!((res, r.len()), (DecodeResult::Decoded(1), 1), "train {i}");
        // one bit of the first four bytes or of the sum flipped in every frame
        let bit = rng.below(40) as usize;
        let mut bad = good;
        bad[bit / 8] ^= 0x80 >> (bit % 8);
        let (res, r) = bresser_decode(&bresser_train(&vec![bad; n], preamble));
        assert!(r.is_empty(), "train {i} bit {bit}: {r:?}");
        assert_eq!(res, DecodeResult::BadIntegrity, "train {i} bit {bit}");
        // one damaged frame among good ones: the good ones are still read, counted without it
        let mut frames_v = vec![good; n];
        frames_v[n / 2] = bad;
        let (res, r) = bresser_decode(&bresser_train(&frames_v, preamble));
        assert_eq!((res, r.len()), (DecodeResult::Decoded(1), 1), "train {i}");
        assert_eq!(r[0].repeats as usize, n - 1);
    }
}

#[test]
fn bresser_out_of_range_values_give_no_reading() {
    // (raw temperature, humidity): a valid sum and a value the device cannot send
    for (raw, hum) in [
        (1618, 101),
        (1618, 255),
        (699, 50),  // -20.1 F
        (2501, 50), // 160.1 F
        (0xFFF, 50),
        (0, 50),
    ] {
        let (res, r) = bresser_decode(&bresser_train(&[bresser_raw(0x11, 6, raw, hum); 22], 4));
        assert!(r.is_empty(), "{raw} {hum}: {r:?}");
        assert_eq!(res, DecodeResult::OutOfRange, "{raw} {hum}");
    }
    // the limits themselves are readings
    for (raw, hum, t) in [
        (700, 0, -28.89f32),
        (2500, 100, 71.11),
        (1220, 50, 0.0),
        (1229, 50, 0.5),
        (1230, 50, 0.56),
        (1626, 50, 22.56),
    ] {
        let (_, r) = bresser_decode(&bresser_train(&[bresser_raw(0x11, 6, raw, hum); 22], 4));
        assert_eq!(r[0].temperature_c, Some(t), "{raw}");
    }
}

#[test]
fn bresser_trains_that_start_or_end_mid_frame_and_glitches() {
    let good = bresser_raw(0xD8, 6, 1618, 64);
    let full = bresser_frame_runs(good, 4);
    let n = full.len();
    // the start of the train cut at every pair of the first frame: later frames still read
    for cut in 0..n {
        let mut runs: Vec<(f64, f64)> = full[cut..].to_vec();
        for _ in 0..21 {
            runs.extend(full.iter().copied());
        }
        runs.last_mut().unwrap().1 = 0.0;
        let (res, r) = bresser_decode(&to_pairs(&runs));
        assert_eq!(res, DecodeResult::Decoded(1), "cut {cut}");
        // the first frame survives while two of its four preamble pairs are left
        let want = if cut <= 2 { 22 } else { 21 };
        assert_eq!(r[0].repeats, want, "cut {cut}");
    }
    // the end of the train cut at every pair of the last frame: one frame less, never a reading
    // of the cut one
    for keep in 1..n {
        let mut runs: Vec<(f64, f64)> = Vec::new();
        for _ in 0..22 {
            runs.extend(full.iter().copied());
        }
        runs.extend(full[..keep].iter().copied());
        runs.last_mut().unwrap().1 = 0.0;
        let (res, r) = bresser_decode(&to_pairs(&runs));
        assert_eq!(res, DecodeResult::Decoded(1), "keep {keep}");
        assert_eq!(r[0].repeats, 22, "keep {keep}");
    }
    // a glitch pair inside one frame drops that frame only; a pair missing from one drops it too
    let mut runs: Vec<(f64, f64)> = Vec::new();
    for k in 0..24 {
        let mut f = full.clone();
        match k {
            5 => f[20] = (100.0, 100.0),
            9 => {
                f.remove(30);
            }
            14 => f.insert(30, (250.0, 500.0)),
            _ => {}
        }
        runs.extend(f);
    }
    runs.last_mut().unwrap().1 = 0.0;
    let (res, r) = bresser_decode(&to_pairs(&runs));
    assert_eq!(res, DecodeResult::Decoded(1));
    assert_eq!(r[0].repeats, 21);
}

#[test]
fn bresser_two_different_messages_in_one_train_are_two_readings() {
    let a = bresser_raw(0xD8, 6, 1618, 64);
    let b = bresser_raw(0xD8, 6, 1625, 71);
    let mut frames = vec![a; 10];
    frames.extend(vec![b; 12]);
    let (res, r) = bresser_decode(&bresser_train(&frames, 4));
    assert_eq!(res, DecodeResult::Decoded(2));
    assert_eq!(
        (
            r[0].temperature_c,
            r[0].repeats,
            r[1].temperature_c,
            r[1].repeats
        ),
        (Some(22.11), 10, Some(22.5), 12)
    );
}

#[test]
fn bresser_tolerates_the_timing_of_the_real_chain() {
    // widths within +-3 samples of 4 us (12 us) and a clock error of 10 % on top
    let mut rng = Pcg32::new(SEED_INTEGRITY, 4);
    let good = bresser_raw(0xD8, 6, 1618, 64);
    for scale in [0.9, 1.0, 1.1] {
        let runs: Vec<(f64, f64)> = (0..22)
            .flat_map(|_| bresser_frame_runs(good, 4))
            .map(|(p, g)| (p * scale, g * scale))
            .collect();
        let mut pairs = to_pairs(&runs);
        for p in &mut pairs {
            p.pulse = (i64::from(p.pulse) + i64::from(rng.below(7)) - 3).max(1) as u32;
            p.gap = (i64::from(p.gap) + i64::from(rng.below(7)) - 3).max(1) as u32;
        }
        pairs.last_mut().unwrap().gap = 0;
        let (res, r) = bresser_decode(&pairs);
        assert_eq!(res, DecodeResult::Decoded(1), "scale {scale}");
        assert_eq!(r[0].temperature_c, Some(22.11));
    }
}

// ---------------------------------------------------------------- the encoders (D1123 to D1127)

/// Seed of the choice of the repeats in the test of the encoders.
const SEED_ENCODERS: u64 = 772;

/// The model string of the golden case as a `'static` string of the registry.
fn registry_model(model: &str) -> &'static str {
    sensors::encoder_by_model(model)
        .unwrap()
        .models()
        .iter()
        .copied()
        .find(|m| *m == model)
        .unwrap()
}

#[test]
fn the_rust_encoders_reproduce_the_readings_of_the_independent_python_encoders() {
    use sdrk_proto::sensors::{EncodeError, MAX_TRAIN_PAIRS, TrainSink};
    let mut rng = Pcg32::new(SEED_ENCODERS, 0);
    let mut checked = 0;
    for name in CASES {
        let c = golden::load("sensors", name).unwrap();
        let models: Vec<String> = c.params["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let (ids, chans, bats, mids) = (
            i32s(&c, "ids"),
            i32s(&c, "channels"),
            i32s(&c, "battery_ok"),
            i32s(&c, "model_ids"),
        );
        let temp = f64s(&c, "temperature_c");
        let hum = f64s(&c, "humidity");
        let wavg = f64s(&c, "wind_avg_km_h");
        let wmax = f64s(&c, "wind_max_km_h");
        let wdir = f64s(&c, "wind_dir_deg");
        let rain = f64s(&c, "rain_mm");
        for (i, py_pairs) in trains(&c).iter().enumerate() {
            // the reading the Python encoder encoded (its expected values), as JSON text and back
            let model = models[mids[i] as usize].as_str();
            let enc = sensors::encoder_by_model(model).unwrap();
            let mut x = Reading::new(
                registry_model(model),
                ids[i] as u32,
                enc.mic(),
                &header(enc.modulation()),
            );
            x.temp_decimals = enc.temp_decimals();
            x.channel = u8::try_from(chans[i]).ok().filter(|&v| v > 0);
            x.battery_ok = match bats[i] {
                -1 => None,
                b => Some(b == 1),
            };
            x.temperature_c = opt(temp[i]);
            x.humidity = opt(hum[i]);
            x.wind_avg_km_h = opt(wavg[i]);
            x.wind_max_km_h = opt(wmax[i]);
            x.wind_dir_deg = opt(wdir[i]);
            x.rain_mm = opt(rain[i]);
            let mut text = String::new();
            x.to_json(&mut text).unwrap();
            let y = Reading::from_json(&text).unwrap();
            // a train of the Rust encoder with 3 or 4 repeats (Nexus needs 3, the sink holds 4)
            let repeats = 3 + rng.below(2);
            let mut storage = vec![Pair::default(); MAX_TRAIN_PAIRS];
            let mut sink = TrainSink::new(&mut storage);
            match enc.encode(&y, repeats, &mut sink) {
                Ok(()) => {}
                Err(EncodeError::Unrepresentable) => {
                    assert!(
                        matches!(name, "lacrosse_tx29" | "lacrosse_tx35" | "nexus"),
                        "{name}"
                    );
                    continue;
                }
                Err(e) => panic!("{name} train {i}: {text}: {e}"),
            }
            let mut buf = vec![0u8; 16 + 8 * sink.len()];
            let n = pulses::encode_joined(&enc.header(), sink.pairs(), &mut buf).unwrap();
            let train = pulses::decode(&buf[..n]).unwrap();
            let mut got = ReadingSink::new();
            let res = sensors::by_name(name).unwrap().decode(&train, &mut got);
            assert!(
                matches!(res, DecodeResult::Decoded(_)),
                "{name} train {i}: {res:?}"
            );
            // the decoder gives the same readings for the Python train and the Rust train
            let (_, from_python) = decode_with(name, enc.modulation(), py_pairs);
            for r in got.iter() {
                assert!(
                    r.same_content(&from_python[0]),
                    "{name} train {i}: {text}\n{r:?}\n{:?}",
                    from_python[0]
                );
            }
            checked += 1;
        }
    }
    assert!(checked >= 7 * 18, "{checked}");
}

// ---------------------------------------------------------------- the fact document (D1136 to D1142)

/// `n` copies of the row `bits` of device `name`.
fn copies_of(name: &str, bits: &[bool], n: usize) -> Vec<Pair> {
    let d = dev(name);
    join(&vec![(d.render)(bits); n], d.gap)
}

fn bytes_of_bits(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|c| c.iter().fold(0u8, |a, &b| (a << 1) | u8::from(b)))
        .collect()
}

#[test]
fn wh2_uses_the_msb_first_crc_the_type_nibble_4_and_an_8_bit_id() {
    let mut rng = Pcg32::new(SEED_CORRECTED, 1);
    for _ in 0..40 {
        let id = rng.below(256);
        let raw = rng.below(0x321) | if rng.below(2) == 0 { 0x800 } else { 0 };
        let hum = rng.below(101) as u8;
        let good = wh2_bits(id, raw, hum);
        let (res, r) = decode_with("wh2", Modulation::Ook, &copies_of("wh2", &good, 2));
        assert_eq!(res, DecodeResult::Decoded(2), "id {id}");
        assert_eq!((r[0].id, r[0].humidity), (id, Some(f32::from(hum))));
        // the reflected CRC of the earlier assumption is a bad CRC now
        let old = wh2_bits_with(4, id, raw, hum, crc_dallas);
        if old != good {
            let (res, r) = decode_with("wh2", Modulation::Ook, &copies_of("wh2", &old, 2));
            assert_eq!(r.len(), 0);
            assert_eq!(res, DecodeResult::BadIntegrity);
        }
        // any other type nibble is another device
        let other = wh2_bits_with(rng.below(4) as u8, id, raw, hum, crc_msb);
        let (res, r) = decode_with("wh2", Modulation::Ook, &copies_of("wh2", &other, 2));
        assert_eq!(r.len(), 0);
        assert_eq!(res, DecodeResult::NoMatch);
        // the humidity byte 0xFF means no humidity sensor
        let none = wh2_bits(id, raw, 0xFF);
        let (res, r) = decode_with("wh2", Modulation::Ook, &copies_of("wh2", &none, 2));
        assert_eq!(res, DecodeResult::Decoded(2));
        assert_eq!((r[0].id, r[0].humidity), (id, None));
    }
}

#[test]
fn nexus_accepts_the_test_button_rejects_channel_3_and_reads_humidity_0_as_none() {
    // the test-button bit (bit 6 of the second byte) is no reason to reject a frame
    let mut found = 0;
    for id in 0..=255u8 {
        let b = nexus_bits_test(id, true, true, 1, 0x0F5, 56);
        let by = bytes_of_bits(&b);
        if crc_msb_init(0x6C, &[by[0], by[1], by[2], 0xF0, 56]) == 0 {
            continue; // a Rubicson frame, tested below
        }
        let (res, r) = decode_with("nexus", Modulation::Ook, &copies_of("nexus", &b, 3));
        assert_eq!(res, DecodeResult::Decoded(1), "id {id}");
        assert_eq!(
            (r[0].id, r[0].channel, r[0].battery_ok, r[0].humidity),
            (u32::from(id), Some(2), Some(true), Some(56.0))
        );
        found += 1;
    }
    assert!(found > 250);
    // the channel value 3 is invalid
    let b = nexus_bits(0x5C, true, 3, 0x0F5, 56);
    let (res, r) = decode_with("nexus", Modulation::Ook, &copies_of("nexus", &b, 3));
    assert_eq!(r.len(), 0);
    assert_eq!(res, DecodeResult::NoMatch);
    // humidity 0: no humidity sensor
    let b = nexus_bits(0x5C, true, 1, 0x0F5, 0);
    let (res, r) = decode_with("nexus", Modulation::Ook, &copies_of("nexus", &b, 3));
    assert_eq!(res, DecodeResult::Decoded(1));
    assert_eq!((r[0].humidity, r[0].temperature_c), (None, Some(24.5)));
}

#[test]
fn nexus_frames_that_pass_the_rubicson_check_are_not_nexus_frames() {
    let mut rng = Pcg32::new(SEED_RUBICSON, 1);
    let mut hits = 0;
    for _ in 0..200_000 {
        let (id, ch, t, hum) = (
            rng.below(256) as u8,
            rng.below(3) as u8,
            rng.below(0x321),
            1 + rng.below(100) as u8,
        );
        let b = nexus_bits_test(id, rng.below(2) == 1, rng.below(2) == 1, ch, t, hum);
        let by = bytes_of_bits(&b);
        let is_rubicson = crc_msb_init(0x6C, &[by[0], by[1], by[2], 0xF0, hum]) == 0;
        let (res, r) = decode_with("nexus", Modulation::Ook, &copies_of("nexus", &b, 3));
        if is_rubicson {
            hits += 1;
            assert!(r.is_empty(), "{b:?}");
            assert_eq!(res, DecodeResult::NoMatch);
        } else {
            assert_eq!(res, DecodeResult::Decoded(1));
        }
        if hits >= 12 {
            break;
        }
    }
    assert!(hits >= 12, "{hits}");
}

#[test]
fn oregon_has_the_length_of_the_device_type_and_the_sign_bit_3() {
    let d = dev("oregon_v21");
    let thn = |sign: u8| {
        let mut n = nibbles("EC4015A2");
        n.extend([0x4, 0x5, 0x2, sign]);
        with_sum(n)
    };
    let train = |bits: &[bool]| join(&[manchester_runs(bits)], d.gap);
    // THN132N: 14 nibbles + checksum with the post-amble is 64 bits, without it 56 bits
    for post in [&[0u8, 0][..], &[]] {
        let (res, r) = d.decode(&train(&oregon_bits_post(&thn(0), post)));
        assert_eq!(res, DecodeResult::Decoded(1), "post {post:?}");
        assert_eq!(
            (r[0].model, r[0].temperature_c),
            ("oregon_thn132n", Some(25.4))
        );
    }
    // the same id with 76 bits is another device (THR228N): not decoded
    let (res, r) = d.decode(&train(&oregon_bits_post(&thn(0), &[0, 0, 0, 0, 0])));
    assert_eq!(r.len(), 0);
    assert_eq!(res, DecodeResult::NoMatch);
    // THGR: 76 bits with the post-amble, 68 bits (THGR228N) without it
    let body = nibbles("1D20485C4808828");
    for post in [&[0u8, 0][..], &[]] {
        let n = with_sum(body.clone());
        let (res, r) = d.decode(&train(&oregon_bits_post(&n, post)));
        assert_eq!(res, DecodeResult::Decoded(1), "post {post:?}");
        assert_eq!(r[0].model, "oregon_thgr122nx");
    }
    // the sign is bit 3 of the nibble 11; the bits 0 to 2 are 0 in practice
    for (sign, ok) in [(8u8, true), (0, true), (9, false), (0xC, false), (4, false)] {
        let (res, r) = d.decode(&train(&oregon_bits(&thn(sign))));
        assert_eq!(!r.is_empty(), ok, "sign {sign:#x}");
        if ok {
            let want = if sign == 8 { -25.4 } else { 25.4 };
            assert_close(r[0].temperature_c, Some(want), "temperature");
        } else {
            assert_eq!(res, DecodeResult::OutOfRange);
        }
    }
    // the THN132N is plausible from -50.0 C; the THGR keeps -70.0
    for (hex, tenths, ok) in [
        ("EC4015A2", 500u32, true),
        ("EC4015A2", 501, false),
        ("1D20485C", 700, true),
    ] {
        let mut n = nibbles(hex);
        n.extend([
            (tenths % 10) as u8,
            (tenths / 10 % 10) as u8,
            (tenths / 100) as u8,
            8,
        ]);
        n.splice(4..8, [4, 5, 2, 0x4]);
        if hex.starts_with("1D") {
            n.extend([1, 2, 0]);
        }
        let (res, r) = d.decode(&train(&oregon_bits(&with_sum(n))));
        assert_eq!(!r.is_empty(), ok, "{hex} {tenths}: {res:?}");
        if !ok {
            assert_eq!(res, DecodeResult::OutOfRange);
        }
    }
}

/// The WH1080 bits from the ten bytes B0 to B9 and `extra` stray bits after the CRC; the CRC is
/// the CRC-8 with init 0xFF over all eleven bytes.
fn wh1080_row(b: [u8; 10], extra: usize) -> Vec<bool> {
    let crc = crc_msb_init(0xFF, &b);
    assert_eq!(
        crc,
        crc_msb(&b[1..]),
        "init 0xFF over B0..B9 is init 0 over B1..B9"
    );
    let mut v = b.to_vec();
    v.push(crc);
    let mut bits = bytes_bits(&v);
    bits.extend(std::iter::repeat_n(true, extra));
    bits
}

#[test]
fn wh1080_follows_the_fact_document() {
    let d = dev("wh1080");
    let decode = |bits: &[bool]| d.decode(&join(&[(d.render)(bits), (d.render)(bits)], d.gap));
    // type A, id 0x2B, raw temperature 0x2F5 = 757 (35.7 C), humidity 55, wind 10 and 20 counts,
    // rain 0x123 counts, status 1 (battery low), direction nibble 9 (203 degrees)
    let b = [0xFF, 0xA2, 0xB2, 0xF5, 55, 10, 20, 0x01, 0x23, 0x19];
    let (res, r) = decode(&wh1080_row(b, 0));
    assert_eq!(res, DecodeResult::Decoded(2));
    let r = &r[0];
    assert_eq!(
        (r.id, r.battery_ok, r.humidity),
        (0x2B, Some(false), Some(55.0))
    );
    assert_close(r.temperature_c, Some(35.7), "temperature");
    assert_close(r.wind_avg_km_h, Some(10.0 * 0.34 * 3.6), "wind");
    assert_close(r.wind_max_km_h, Some(20.0 * 0.34 * 3.6), "gust");
    assert_close(r.rain_mm, Some(0x123 as f32 * 0.3), "rain");
    assert_close(r.wind_dir_deg, Some(203.0), "direction");
    // a status nibble other than 1 is a good battery
    let mut g = b;
    g[9] = 0x09;
    let (_, r) = decode(&wh1080_row(g, 0));
    assert_eq!(r[0].battery_ok, Some(true));
    // the whole table of directions
    for (k, deg) in [
        0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338,
    ]
    .into_iter()
    .enumerate()
    {
        let mut g = b;
        g[9] = k as u8;
        let (_, r) = decode(&wh1080_row(g, 0));
        assert_close(r[0].wind_dir_deg, Some(deg as f32), "direction");
    }
    // the temperature has 10 bits: the bits 3 and 2 of B2 are not part of it
    let mut g = b;
    g[2] = 0xB2 | 0x0C;
    let (_, r) = decode(&wh1080_row(g, 0));
    assert_close(r[0].temperature_c, Some(35.7), "temperature");
    // rows of 88 to 99 bits are messages, 100 and more are not, 87 is a cut one
    for extra in [0, 5, 11] {
        let (res, _) = decode(&wh1080_row(b, extra));
        assert_eq!(res, DecodeResult::Decoded(2), "{extra} extra bits");
    }
    let (res, r) = decode(&wh1080_row(b, 12));
    assert_eq!(r.len(), 0);
    assert_eq!(res, DecodeResult::NoMatch);
    let cut = wh1080_row(b, 0)[..87].to_vec();
    let (res, r) = decode(&cut);
    assert_eq!(r.len(), 0);
    assert_eq!(res, DecodeResult::Truncated);
    // the time (0xB) and the UV (0x7) messages are not weather messages
    for ty in [0xB0, 0x70, 0x00] {
        let mut g = b;
        g[1] = ty | 0x02;
        let (res, r) = decode(&wh1080_row(g, 0));
        assert_eq!(r.len(), 0);
        assert_eq!(res, DecodeResult::NoMatch, "type {ty:#x}");
    }
}

#[test]
fn lacrosse_tx29_has_a_6_bit_id_and_the_24_bit_pattern() {
    let d = dev("lacrosse_tx29");
    let decode = |bits: &[bool]| d.decode(&join(&[nrz_runs(bits)], d.gap));
    // id 63 (the low nibble of B0 = F and the top two bits of B1 = 3), the new-battery flag
    // (bit 5 of B1) and the unused bit 4 set: neither is part of the temperature
    let frame = tx29_frame([0x9F, 0xF3, 0x94, 0x59]);
    let (res, r) = decode(&tx29_bits(frame));
    assert_eq!(res, DecodeResult::Decoded(1));
    assert_eq!(r[0].id, 63);
    assert_close(r[0].temperature_c, Some(-0.6), "temperature");
    // the same frame without the flags has the same temperature and id
    let (_, r2) = decode(&tx29_bits(tx29_frame([0x9F, 0xC3, 0x94, 0x59])));
    assert_eq!((r2[0].id, r2[0].temperature_c), (63, r[0].temperature_c));
    // 0x6A: no humidity sensor
    let (_, r) = decode(&tx29_bits(tx29_frame([0x92, 0x03, 0x94, 0x6A])));
    assert_eq!(r[0].humidity, None);
    // the pattern needs the last nibble of the preamble (0xA) before the sync word
    let (res, r) = decode(&tx29_bits_pre(&[0xA5, 0xA5, 0xA5, 0xA5], frame));
    assert_eq!(r.len(), 0);
    assert_eq!(res, DecodeResult::NoMatch);
    // ... and the model nibble 9 after it
    let (res, r) = decode(&tx29_bits(tx29_frame([0x82, 0x03, 0x94, 0x59])));
    assert_eq!(r.len(), 0);
    assert_eq!(res, DecodeResult::NoMatch);
    // the document names 55 us, the decoder reads both 55 and 58 us per bit
    let runs = level_runs(&tx29_bits(frame), 55.0);
    let (res, r) = d.decode(&join(&[runs], d.gap));
    assert_eq!(res, DecodeResult::Decoded(1));
    assert_eq!(r[0].id, 63);
}

#[test]
fn lacrosse_tx35_is_the_tx29_frame_at_105_us_per_bit() {
    let frame = tx29_frame([0x92, 0x03, 0x94, 0x59]);
    let at = |name: &str, unit: f64, frame: [u8; 5]| {
        let runs = level_runs(&tx29_bits(frame), unit);
        decode_with(name, Modulation::Fsk, &join(&[runs.clone(), runs], 4000.0))
    };
    // the worked example of the TX29 at 105 us per bit is a reading of the TX35 (id 8, -0.6 C)
    let (res, r) = at("lacrosse_tx35", 105.0, frame);
    assert_eq!(res, DecodeResult::Decoded(2));
    assert_reading(&r[0], "lacrosse_tx35", 8, None, -0.6, Some(89.0));
    assert_eq!(r[0].battery_ok, Some(true));
    // 9.58 kbit/s is 104.4 us per bit
    let (res, r) = at("lacrosse_tx35", 104.4, frame);
    assert_eq!(res, DecodeResult::Decoded(2));
    assert_eq!(r[0].id, 8);
    // the models are told apart by the bit rate alone
    let (res, r) = at("lacrosse_tx29", 105.0, frame);
    assert!(r.is_empty(), "{res:?}");
    let (res, r) = at("lacrosse_tx35", 58.0, frame);
    assert!(r.is_empty(), "{res:?}");
    let (res, r) = at("lacrosse_tx29", 58.0, frame);
    assert_eq!(res, DecodeResult::Decoded(2));
    assert_eq!(r[0].model, "lacrosse_tx29");
    // a flipped bit of the payload is a CRC error; humidity 0x6A is none, weak battery bit 7
    let mut bad = frame;
    bad[2] ^= 0x04;
    let (res, r) = at("lacrosse_tx35", 105.0, bad);
    assert_eq!(r.len(), 0);
    assert_eq!(res, DecodeResult::BadIntegrity);
    let (_, r) = at("lacrosse_tx35", 105.0, tx29_frame([0x92, 0x03, 0x94, 0xEA]));
    assert_eq!((r[0].humidity, r[0].battery_ok), (None, Some(false)));
}
