//! Golden protocol cases of the sensor decoders (M46 tests 2, 4 and 5).
//!
//! Test 2: for every decoder (`wh2`, `nexus`, `oregon_v21`, `wh1080`, `lacrosse_tx29`) the golden
//! case holds 20 trains written by the independent Python encoder (seeds 637, 638, 639, 644, 645,
//! recorded in the case) with timing error; the Rust decoder must return exactly the encoded
//! readings (one per copy, one for the device without a check) and no other decoder may return a
//! reading for the same train. The worked examples of the public descriptions are rendered here
//! and decoded as well.
//!
//! Test 4 (seed 646): a flipped bit in the payload or the check of a device with a CRC or checksum
//! gives no reading (`BadIntegrity`); Nexus, a device without a check, gives a reading only when
//! two equal frames are left. Test 5 (seed 647): out-of-range values and truncated rows give no
//! reading. The trains of tests 4 and 5 are rendered by the helpers below from the coding rules.
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
const CASES: [&str; 5] = ["lacrosse_tx29", "nexus", "oregon_v21", "wh1080", "wh2"];
/// The seeds of tests 4 and 5.
const SEED_INTEGRITY: u64 = 646;
const SEED_RANGE: u64 = 647;

#[test]
fn the_cases_of_m46_are_present() {
    let mut names = golden::list("sensors").unwrap();
    names.sort();
    assert_eq!(names, CASES);
    for (name, seed) in [
        ("wh2", 637),
        ("nexus", 638),
        ("oregon_v21", 639),
        ("wh1080", 644),
        ("lacrosse_tx29", 645),
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
        "oregon_v21" => Mic::Checksum,
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
                assert_eq!(
                    r.repeats,
                    if decoder == "nexus" { repeats } else { 1 },
                    "{ctx}"
                );
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
    let mut crc = 0u8;
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

fn wh2_bits(id: u32, raw_temp: u32, hum: u8) -> Vec<bool> {
    let p = [
        (id >> 4) as u8,
        (((id & 0xF) << 4) | (raw_temp >> 8)) as u8,
        raw_temp as u8,
        hum,
    ];
    let mut b = vec![0xFF];
    b.extend_from_slice(&p);
    b.push(crc_dallas(&p));
    bytes_bits(&b)
}

fn nexus_bits(id: u8, bat: bool, ch: u8, temp12: u32, hum: u8) -> Vec<bool> {
    let mut v = bits_of(u64::from(id), 8);
    v.extend([bat, false]);
    v.extend(bits_of(u64::from(ch), 2));
    v.extend(bits_of(u64::from(temp12), 12));
    v.extend([true; 4]);
    v.extend(bits_of(u64::from(hum), 8));
    v
}

fn oregon_bits(nibbles: &[u8]) -> Vec<bool> {
    let mut data = vec![true; 16];
    for n in std::iter::once(&0xA).chain(nibbles).chain(&[0, 0]) {
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
    let mut b = vec![0xAA; 4];
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
        "wh2" | "wh1080" => Dev {
            name: if name == "wh2" { "wh2" } else { "wh1080" },
            slicer: Box::new(Pwm::new(500.0, 1500.0, 1000.0, 2500.0, true).unwrap()),
            render: pwm_runs,
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
    // Nexus: 5c910ff38 is id 0x5c, battery good, channel 2, 27.1 C, 56 %; two equal frames.
    let d = dev("nexus");
    let bits = bits_of(0x5_c910_ff38, 36);
    let frames = vec![(d.render)(&bits); 2];
    let (res, r) = d.decode(&join(&frames, d.gap));
    assert_eq!(res, DecodeResult::Decoded(1));
    assert_reading(&r[0], "nexus", 0x5c, Some(2), 27.1, Some(56.0));
    assert_eq!((r[0].battery_ok, r[0].repeats), (Some(true), 2));

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

    // LaCrosse IT+: the two examples of the article.
    let d = dev("lacrosse_tx29");
    for (frame, id, t, h) in [
        ([0x92, 0x03, 0x94, 0x59, 0x3C], 32, -0.6, 89.0),
        ([0x92, 0x86, 0x24, 0x28, 0xDE], 40, 22.4, 40.0),
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
fn nexus_needs_two_equal_frames() {
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
        // two equal frames: one reading
        let (res, r) = d.decode(&d.train(&rows[..2]));
        assert_eq!(
            (res, r.len(), r[0].repeats),
            (DecodeResult::Decoded(1), 1, 2)
        );
        // one frame flipped: the other two still agree
        let t = d.train(&[flip(&rows[0], a), rows[1].clone(), rows[2].clone()]);
        let (res, r) = d.decode(&t);
        assert_eq!(
            (res, r.len(), r[0].repeats),
            (DecodeResult::Decoded(1), 1, 2)
        );
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
            0x12,
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
        ("wh2", two_copies("wh2", &wh2_bits(0x123, 900, 50))),
        (
            "wh2",
            two_copies("wh2", &wh2_bits(0x123, 0x800 | 0xFA, 101)),
        ),
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
        ("wh1080", two_copies("wh1080", &wh1080(1500, 50))),
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
        for (i, pairs) in trains(&c).iter().enumerate() {
            // keep a part of the first copy only: 55 to 90 percent of it
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
