//! The sensor encoders (M46 extension, D1123 to D1127): `decode(encode(x)) == x` for every
//! device over its whole range, through the JSON form and through packets, the recorded
//! messages of the Bresser sensor, rejected readings, small buffers and `Reading::from_json`.
//! Seeds 770 (random readings), 771 (the raw values of the Bresser sensor), 773 (`from_json`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines
)]

use sdrk_proto::PacketFlags;
use sdrk_proto::pulses::{self, Assembler, Header, Modulation, Pair};
use sdrk_proto::sensors::{
    self, DecodeResult, ENCODERS, EncodeError, JsonBuf, JsonError, MAX_TRAIN_PAIRS, Mic, Reading,
    ReadingSink, TrainSink,
};

/// A xorshift64 generator (the proto crate has no random number generator).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A value in `lo..=hi`.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
}

fn header() -> Header {
    Header {
        modulation: Modulation::Ook,
        channel: 0,
        freq_offset_hz: 0,
        level_db10: 0,
        noise_db10: 0,
        rate_hz: 250_000,
    }
}

fn reading(model: &'static str, id: u32) -> Reading {
    let mic = sensors::encoder_by_model(model).unwrap().mic();
    let mut r = Reading::new(model, id, mic, &header());
    r.temp_decimals = sensors::encoder_by_model(model).unwrap().temp_decimals();
    r
}

fn encode(r: &Reading, repeats: u32) -> Result<Vec<Pair>, EncodeError> {
    let enc = sensors::encoder_by_model(r.model).ok_or(EncodeError::UnknownModel)?;
    let mut storage = vec![Pair::default(); MAX_TRAIN_PAIRS];
    let mut sink = TrainSink::new(&mut storage);
    enc.encode(r, repeats, &mut sink)?;
    Ok(sink.pairs().to_vec())
}

/// Decodes `pairs` through packets and the assembler, as `sensor_decode` would receive them.
fn decode(r: &Reading, pairs: &[Pair]) -> (DecodeResult, Vec<Reading>) {
    let enc = sensors::encoder_by_model(r.model).unwrap();
    let dec = sensors::by_name(enc.name()).unwrap();
    let mut asm = Assembler::new(vec![Pair::default(); MAX_TRAIN_PAIRS]);
    let mut joined = None;
    for p in pulses::packets(7, PacketFlags::NONE, &enc.header(), pairs) {
        if let Some(t) = asm.push(&p.unwrap()).unwrap() {
            assert_eq!((t.index, t.header), (7, enc.header()));
            joined = Some(t.pairs.to_vec());
        }
    }
    let joined = joined.expect("the last packet completes the train");
    assert_eq!(joined, pairs);
    let mut buf = vec![0u8; 16 + 8 * joined.len()];
    let n = pulses::encode_joined(&enc.header(), &joined, &mut buf).unwrap();
    let train = pulses::decode(&buf[..n]).unwrap();
    let mut sink = ReadingSink::new();
    let res = dec.decode(&train, &mut sink);
    (res, sink.iter().copied().collect())
}

/// The JSON of a reading up to its radio fields (the printed resolution: a value of the decoder
/// is not the float of its printed digits, for example 1.224 times a count).
fn radio_free(r: &Reading) -> String {
    let mut r = *r;
    r.time_s = None;
    let s = json(&r);
    s[..s.find("\"mic\"").unwrap()].to_string()
}

fn json(r: &Reading) -> String {
    let mut s = String::new();
    r.to_json(&mut s).unwrap();
    s
}

/// A random reading of the model over the whole range of the device, with the values the
/// decoder produces (the same expressions).
fn random_reading(rng: &mut Rng, model: &'static str) -> Reading {
    let mut r = reading(model, 0);
    let t = |tenths: i64| Some(tenths as f32 / 10.0);
    let hum = |rng: &mut Rng| Some(rng.range(0, 100) as f32);
    let hum99 = |rng: &mut Rng| Some(rng.range(0, 99) as f32);
    match model {
        "wh2" => {
            r.id = rng.range(0, 0x7FF) as u32;
            r.temperature_c = t(rng.range(-800, 800));
            r.humidity = hum(rng);
        }
        "nexus" => {
            r.id = rng.range(0, 255) as u32;
            r.channel = Some(rng.range(1, 4) as u8);
            r.battery_ok = Some(rng.below(2) == 1);
            r.temperature_c = t(rng.range(-400, 800));
            r.humidity = hum(rng);
        }
        "oregon_thn132n" | "oregon_thgr122nx" => {
            r.id = rng.range(0, 255) as u32;
            r.channel = Some(rng.range(1, 3) as u8);
            r.battery_ok = Some(rng.below(2) == 1);
            r.temperature_c = t(rng.range(-700, 700));
            if model == "oregon_thgr122nx" {
                r.humidity = hum99(rng);
            }
        }
        "wh1080" => {
            r.id = rng.range(0, 0xFFF) as u32;
            r.temperature_c = t(rng.range(-400, 600));
            r.humidity = hum(rng);
            r.wind_avg_km_h = Some(rng.range(0, 255) as f32 * 1.224);
            r.wind_max_km_h = Some(rng.range(0, 255) as f32 * 1.224);
            r.wind_dir_deg = Some(rng.range(0, 15) as f32 * 22.5);
            r.rain_mm = Some(rng.range(0, 4095) as f32 * 0.3);
        }
        "lacrosse_tx29" => {
            r.id = rng.range(0, 255) as u32;
            // the expression of the decoder: it is not the nearest float of the tenths
            r.temperature_c = Some((rng.range(0, 999) as f32) / 10.0 - 40.0);
            if rng.below(5) != 0 {
                r.humidity = hum(rng);
            }
            r.battery_ok = Some(rng.below(2) == 1);
        }
        "Bresser-3CH" => {
            r.id = rng.range(0, 255) as u32;
            r.channel = Some(rng.range(1, 3) as u8);
            r.battery_ok = Some(rng.below(2) == 1);
            let raw = rng.range(700, 2500) as i32;
            r.temperature_c = Some(hundredths(raw) as f32 / 100.0);
            r.humidity = hum(rng);
        }
        _ => unreachable!(),
    }
    r
}

/// The temperature of a raw Bresser value in hundredths, with `f64` (an independent form of
/// the integer rule of the decoder).
fn hundredths(raw: i32) -> i32 {
    (f64::from(raw - 1220) * 100.0 / 18.0).round() as i32
}

const MODELS: [&str; 7] = [
    "wh2",
    "nexus",
    "oregon_thn132n",
    "oregon_thgr122nx",
    "wh1080",
    "lacrosse_tx29",
    "Bresser-3CH",
];

#[test]
fn the_registry_mirrors_the_decoders() {
    assert_eq!(ENCODERS.len(), sensors::DECODERS.len());
    for (e, d) in ENCODERS.iter().zip(sensors::DECODERS) {
        assert_eq!(e.name(), d.name());
        assert_eq!(e.modulation(), d.modulation(), "{}", e.name());
        assert!(e.default_repeats() >= 1);
        assert_eq!(e.header().rate_hz, 1_000_000);
        assert_eq!(e.header().modulation, e.modulation());
        assert!(sensors::encoder_by_name(e.name()).is_some());
    }
    assert!(sensors::encoder_by_name("nobody").is_none());
    for m in MODELS {
        assert!(sensors::encoder_by_model(m).is_some(), "{m}");
    }
    assert_eq!(
        sensors::encoder_by_model("oregon_thn132n").unwrap().name(),
        "oregon_v21"
    );
    assert_eq!(
        sensors::encoder_by_model("Bresser-3CH").unwrap().name(),
        "bresser_3ch"
    );
    assert!(sensors::encoder_by_model("bresser_3ch").is_none());
}

#[test]
fn random_readings_round_trip_through_json_packets_and_the_decoder() {
    let mut rng = Rng::new(770);
    let mut unrepresentable = 0;
    for model in MODELS {
        let enc = sensors::encoder_by_model(model).unwrap();
        for k in 0..400 {
            let x = random_reading(&mut rng, model);
            // through the JSON form
            let text = json(&x);
            let y = Reading::from_json(&text).unwrap();
            assert_eq!(radio_free(&y), radio_free(&x), "{text}");
            assert_eq!((y.mic, y.temp_decimals), (x.mic, x.temp_decimals));
            // the default repeats and some others
            let repeats = if k % 3 == 0 {
                enc.default_repeats()
            } else {
                rng.range(2, 4) as u32
            };
            let pairs = match encode(&y, repeats) {
                Ok(p) => p,
                Err(EncodeError::Unrepresentable) => {
                    assert_eq!(model, "lacrosse_tx29");
                    unrepresentable += 1;
                    continue;
                }
                Err(e) => panic!("{text}: {e}"),
            };
            assert_eq!(pairs.last().unwrap().gap, 0, "{model}");
            let (res, got) = decode(&y, &pairs);
            assert!(matches!(res, DecodeResult::Decoded(_)), "{text}: {res:?}");
            assert!(!got.is_empty(), "{text}");
            for g in &got {
                assert!(g.same_content(&x), "{text}\n{g:?}");
                assert_eq!(g.mic, x.mic);
            }
            // one reading per copy (a device with a check), or one with the number of copies
            let copies: u32 = got.iter().map(|g| g.repeats).sum();
            assert_eq!(copies, repeats, "{model}: {text}");
        }
    }
    // a frame whose CRC byte is 0x00 cannot be shown by a burst: about 1 in 256
    assert!(unrepresentable < 20, "{unrepresentable}");
}

#[test]
fn trains_longer_than_a_packet_are_cut_and_joined() {
    // the Bresser default: 22 frames of 44 pairs are 968 pairs, 33 packets
    let mut r = reading("Bresser-3CH", 0x42);
    r.channel = Some(1);
    r.battery_ok = Some(true);
    r.temperature_c = Some(35.0);
    r.humidity = Some(66.0);
    let enc = sensors::encoder_by_model("Bresser-3CH").unwrap();
    let pairs = encode(&r, enc.default_repeats()).unwrap();
    assert_eq!(pairs.len(), 22 * 44);
    assert_eq!(
        pulses::packets(0, PacketFlags::NONE, &enc.header(), &pairs).count(),
        33
    );
    let (res, got) = decode(&r, &pairs);
    assert_eq!(res, DecodeResult::Decoded(1));
    assert!(got[0].same_content(&r));
    assert_eq!(got[0].repeats, 22);
    // the duration: 22 frames of four preamble pairs (1470 us) and 40 bits (750 us)
    let total: u64 = pairs.iter().map(|p| u64::from(p.pulse + p.gap)).sum();
    // (the gap of the last pair is 0 instead of 250 or 500 us)
    let full = 22 * (4 * 1470 + 40 * 750);
    assert!(total == full - 250 || total == full - 500, "{total}");
    // the most frames that fit
    assert_eq!(encode(&r, 27).unwrap().len(), 27 * 44);
    assert_eq!(encode(&r, 28), Err(EncodeError::BufferTooSmall));
}

#[test]
fn every_raw_value_of_the_bresser_sensor_round_trips_exactly() {
    let mut rng = Rng::new(771);
    for raw in 700..=2500 {
        let h = hundredths(raw);
        let mut r = reading("Bresser-3CH", rng.range(0, 255) as u32);
        r.channel = Some(rng.range(1, 3) as u8);
        r.temperature_c = Some(h as f32 / 100.0);
        r.humidity = Some(rng.range(0, 100) as f32);
        let pairs = encode(&r, 2).unwrap();
        let f = bresser_frame(&pairs);
        let got = (i32::from(f[1] & 0x0F) << 8) | i32::from(f[2]);
        assert_eq!(got, raw, "raw {raw} = {h} hundredths");
        // the JSON form has two decimals and reads back to the same raw value
        let text = json(&r);
        assert!(
            text.contains(&format!("\"temperature_C\":{:.2}", h as f32 / 100.0)),
            "{text}"
        );
        let y = Reading::from_json(&text).unwrap();
        let again = bresser_frame(&encode(&y, 2).unwrap());
        assert_eq!(again, f, "{text}");
        let (res, back) = decode(&y, &pairs);
        assert_eq!(res, DecodeResult::Decoded(1));
        assert_eq!(back[0].temperature_c, r.temperature_c, "raw {raw}");
        assert_eq!(back[0].temp_decimals, 2);
    }
}

/// The five bytes of the first frame of a Bresser train (four preamble pairs, 40 bits), and a
/// check that the other frames are equal and the shapes are the nominal ones.
fn bresser_frame(pairs: &[Pair]) -> [u8; 5] {
    let mut first = [0u8; 5];
    for (k, frame) in pairs.chunks(44).enumerate() {
        assert_eq!(frame.len(), 44);
        for p in &frame[..4] {
            assert_eq!((p.pulse, p.gap), (735, 735));
        }
        let mut bytes = [0u8; 5];
        for (i, p) in frame[4..].iter().enumerate() {
            let last = k * 44 + 43 == pairs.len() - 1 && i == 39;
            match (p.pulse, p.gap) {
                (500, 250) => bytes[i / 8] |= 0x80 >> (i % 8),
                (250, 500) => {}
                (250 | 500, 0) if last => {
                    if p.pulse == 500 {
                        bytes[i / 8] |= 0x80 >> (i % 8);
                    }
                }
                other => panic!("pair {other:?}"),
            }
        }
        if k == 0 {
            first = bytes;
        } else {
            assert_eq!(bytes, first);
        }
    }
    let sum = first[..4].iter().fold(0u8, |a, &b| a.wrapping_add(b));
    assert_eq!(first[4], sum);
    first
}

#[test]
fn the_temperature_rule_of_the_bresser_sensor() {
    let enc = |t: f32| {
        let mut r = reading("Bresser-3CH", 1);
        r.temperature_c = Some(t);
        r.humidity = Some(50.0);
        let f = bresser_frame(&encode(&r, 1).unwrap());
        (i32::from(f[1] & 0x0F) << 8) | i32::from(f[2])
    };
    assert_eq!(enc(22.56), 1626);
    assert_eq!(enc(22.5), 1625);
    assert_eq!(enc(22.50), 1625);
    assert_eq!(enc(22.11), 1618);
    assert_eq!(enc(0.0), 1220);
    assert_eq!(enc(-28.89), 700);
    assert_eq!(enc(71.11), 2500);
    // the decoder side: raw 1626 is 22.556 C and reads 22.56
    let mut r = reading("Bresser-3CH", 1);
    r.temperature_c = Some(22.56);
    r.humidity = Some(50.0);
    let (_, got) = decode(&r, &encode(&r, 3).unwrap());
    assert_eq!(
        json(&got[0]).split("\"temperature_C\":").nth(1).unwrap()[..5].to_string(),
        "22.56"
    );
}

#[test]
fn the_recorded_bresser_messages_are_reproduced_except_for_the_button_flag() {
    // (the recorded frame, the reading)
    let recorded: [([u8; 5], f32, f32); 4] = [
        ([0xD8, 0x65, 0xC7, 0x51, 0x55], 14.39, 81.0),
        ([0xD8, 0x66, 0x52, 0x40, 0xD0], 22.11, 64.0),
        ([0xD8, 0x66, 0x59, 0x47, 0xDE], 22.5, 71.0),
        ([0xD8, 0x65, 0xDB, 0x3C, 0x54], 15.5, 60.0),
    ];
    for (frame, t, h) in recorded {
        let mut r = reading("Bresser-3CH", 0xD8);
        r.channel = Some(2);
        r.battery_ok = Some(true);
        r.temperature_c = Some(t);
        r.humidity = Some(h);
        let pairs = encode(&r, 22).unwrap();
        let mut f = bresser_frame(&pairs);
        // the button flag (bit 6 of byte 1) is not part of a reading: sent as 0, the recording has 1
        assert_eq!(f[1] & 0x40, 0);
        f[1] |= 0x40;
        f[4] = f[..4].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        assert_eq!(f, frame, "{t} C");
        let (res, got) = decode(&r, &pairs);
        assert_eq!(res, DecodeResult::Decoded(1));
        assert_eq!((got[0].temperature_c, got[0].humidity), (Some(t), Some(h)));
        assert_eq!(
            (got[0].id, got[0].channel, got[0].battery_ok),
            (0xD8, Some(2), Some(true))
        );
        assert_eq!(got[0].mic, Mic::Checksum);
    }
}

fn base(model: &'static str) -> Reading {
    let mut r = reading(model, 5);
    r.temperature_c = Some(20.0);
    match model {
        "wh2" | "wh1080" | "nexus" | "oregon_thgr122nx" | "Bresser-3CH" => {
            r.humidity = Some(50.0);
        }
        _ => {}
    }
    r
}

#[test]
fn readings_out_of_range_are_rejected_and_nothing_is_written() {
    let mut storage = vec![Pair::default(); MAX_TRAIN_PAIRS];
    let mut cases: Vec<(&str, Reading, EncodeError)> = Vec::new();
    let oor = |n: &'static str| EncodeError::OutOfRange(n);
    for model in MODELS {
        let mut r = base(model);
        r.humidity = r.humidity.map(|_| -1.0);
        if r.humidity.is_some() {
            cases.push((model, r, oor("humidity")));
            let mut r = base(model);
            r.humidity = Some(101.0);
            cases.push((model, r, oor("humidity")));
        }
        let mut r = base(model);
        r.temperature_c = Some(f32::NAN);
        cases.push((model, r, oor("temperature_C")));
        let mut r = base(model);
        r.temperature_c = None;
        cases.push((model, r, EncodeError::MissingField("temperature_C")));
        let mut r = base(model);
        r.id = 100_000;
        cases.push((model, r, oor("id")));
        let mut r = base(model);
        r.pressure_hpa = Some(1000.0);
        cases.push((model, r, EncodeError::UnsupportedField("pressure_hPa")));
        let mut r = base(model);
        r.model = "nobody";
        cases.push((model, r, EncodeError::UnknownModel));
    }
    // the limits of every device, one step beyond
    for (model, lo, hi) in [
        ("wh2", -80.1, 80.1),
        ("nexus", -40.1, 80.1),
        ("oregon_thn132n", -70.1, 70.1),
        ("oregon_thgr122nx", -70.1, 70.1),
        ("wh1080", -40.1, 60.1),
        ("lacrosse_tx29", -40.1, 60.0),
        ("Bresser-3CH", -28.95, 71.17),
    ] {
        for t in [lo, hi] {
            let mut r = base(model);
            r.temperature_c = Some(t);
            cases.push((model, r, oor("temperature_C")));
        }
    }
    let mut r = base("nexus");
    r.channel = Some(0);
    cases.push(("nexus", r, oor("channel")));
    let mut r = base("nexus");
    r.channel = Some(5);
    cases.push(("nexus", r, oor("channel")));
    let mut r = base("Bresser-3CH");
    r.channel = Some(4);
    cases.push(("Bresser-3CH", r, oor("channel")));
    let mut r = base("wh2");
    r.channel = Some(1);
    cases.push(("wh2", r, EncodeError::UnsupportedField("channel")));
    let mut r = base("wh2");
    r.rain_mm = Some(1.0);
    cases.push(("wh2", r, EncodeError::UnsupportedField("rain_mm")));
    let mut r = base("oregon_thn132n");
    r.humidity = Some(50.0);
    cases.push((
        "oregon_thn132n",
        r,
        EncodeError::UnsupportedField("humidity"),
    ));
    let mut r = base("oregon_thgr122nx");
    r.humidity = None;
    cases.push(("oregon_thgr122nx", r, EncodeError::MissingField("humidity")));
    let mut r = base("wh1080");
    r.rain_mm = Some(2000.0);
    cases.push(("wh1080", r, oor("rain_mm")));
    let mut r = base("wh1080");
    r.wind_avg_km_h = Some(-1.0);
    cases.push(("wh1080", r, oor("wind_avg_km_h")));
    let mut r = base("wh1080");
    r.wind_dir_deg = Some(400.0);
    cases.push(("wh1080", r, oor("wind_dir_deg")));
    for (model, r, want) in cases {
        let enc = sensors::encoder_by_model(model).unwrap();
        let mut sink = TrainSink::new(&mut storage);
        sink.push(1, 2).unwrap();
        assert_eq!(enc.encode(&r, 2, &mut sink), Err(want), "{model} {r:?}");
        assert_eq!(sink.len(), 1, "{model}: nothing is written");
    }
    // repeats of 0
    let enc = sensors::encoder_by_model("nexus").unwrap();
    let mut sink = TrainSink::new(&mut storage);
    assert_eq!(
        enc.encode(&base("nexus"), 0, &mut sink),
        Err(EncodeError::InvalidRepeats)
    );
    assert!(sink.is_empty());
    // the limits themselves are readings
    for (model, t) in [
        ("wh2", -80.0),
        ("wh2", 80.0),
        ("nexus", -40.0),
        ("nexus", 80.0),
        ("oregon_thn132n", -70.0),
        ("oregon_thgr122nx", 70.0),
        ("wh1080", -40.0),
        ("wh1080", 60.0),
        ("lacrosse_tx29", -40.0),
        ("lacrosse_tx29", 59.9),
        ("Bresser-3CH", -28.89),
        ("Bresser-3CH", 71.11),
    ] {
        let mut r = base(model);
        r.temperature_c = Some(t);
        let pairs = encode(&r, 3).unwrap_or_else(|e| panic!("{model} {t}: {e}"));
        let (_, got) = decode(&r, &pairs);
        assert_eq!(got[0].temperature_c, Some(t), "{model}");
    }
}

#[test]
fn a_train_that_does_not_fit_is_buffer_too_small_and_leaves_the_sink_alone() {
    for model in MODELS {
        let enc = sensors::encoder_by_model(model).unwrap();
        let r = base(model);
        let mut big = vec![Pair::default(); MAX_TRAIN_PAIRS];
        let mut sink = TrainSink::new(&mut big);
        let res = enc.encode(&r, 2, &mut sink);
        let n = match res {
            Ok(()) => sink.len(),
            Err(EncodeError::Unrepresentable) => continue,
            Err(e) => panic!("{model}: {e}"),
        };
        // exactly n pairs fit, one less does not, and every size below is an error
        for cap in [n, 0, 1, n / 2, n - 1] {
            let mut small = vec![Pair::default(); cap];
            let mut sink = TrainSink::new(&mut small);
            let res = enc.encode(&r, 2, &mut sink);
            if cap == n {
                assert_eq!(res, Ok(()), "{model}");
                assert_eq!(sink.len(), n);
            } else {
                assert_eq!(res, Err(EncodeError::BufferTooSmall), "{model} cap {cap}");
                assert_eq!(sink.len(), 0, "{model}");
            }
        }
        // after a failure the sink is reusable
        let mut small = vec![Pair::default(); n];
        let mut sink = TrainSink::new(&mut small);
        sink.push(9, 9).unwrap();
        assert_eq!(
            enc.encode(&r, 2, &mut sink),
            Err(EncodeError::BufferTooSmall)
        );
        assert_eq!(sink.len(), 1);
    }
}

#[test]
fn the_pair_counts_and_timings_follow_the_decoders() {
    let one = |model: &'static str| encode(&base(model), 1).unwrap();
    // wh2: 48 pulses of 500 or 1500 us, gaps of 1000 us, the last one 0
    let p = one("wh2");
    assert_eq!(p.len(), 48);
    assert!(p.iter().all(|p| p.pulse == 500 || p.pulse == 1500));
    assert!(p[..47].iter().all(|p| p.gap == 1000));
    assert!(
        p[..8].iter().all(|p| p.pulse == 500),
        "the preamble is eight 1 bits"
    );
    // the repeat gap is 8000 us
    let p = encode(&base("wh2"), 3).unwrap();
    assert_eq!(p.len(), 144);
    assert_eq!((p[47].gap, p[95].gap, p[143].gap), (8000, 8000, 0));
    // nexus: 36 bits and the end pulse, gaps of 1000 and 2000 us, 4000 us between frames
    let p = encode(&base("nexus"), 2).unwrap();
    assert_eq!(p.len(), 74);
    assert!(p.iter().all(|p| p.pulse == 500));
    assert_eq!((p[36].gap, p[73].gap), (4000, 0));
    assert!(p[..36].iter().all(|p| p.gap == 1000 || p.gap == 2000));
    // wh1080: 88 bits
    assert_eq!(one("wh1080").len(), 88);
    // oregon: runs of one or two half bits of 488 us
    for m in ["oregon_thn132n", "oregon_thgr122nx"] {
        for p in one(m) {
            assert!(matches!(p.pulse, 488 | 976), "{m}");
            assert!(matches!(p.gap, 0 | 488 | 976), "{m}");
        }
    }
    // lacrosse: NRZ runs of 58 us
    for p in one("lacrosse_tx29") {
        assert_eq!(p.pulse % 58, 0);
        assert_eq!(p.gap % 58, 0);
    }
    assert_eq!(
        sensors::encoder_by_model("lacrosse_tx29")
            .unwrap()
            .modulation(),
        Modulation::Fsk
    );
    // bresser: 44 pairs per frame
    assert_eq!(one("Bresser-3CH").len(), 44);
}

#[test]
fn a_frame_with_a_crc_of_zero_is_unrepresentable_for_the_fsk_device() {
    // search the ids for a frame whose CRC byte is 0x00 (more than seven trailing zeros)
    let mut found = 0;
    for id in 0..256 {
        for t in [-40.0f32, 0.0, 21.3] {
            let mut r = reading("lacrosse_tx29", id);
            r.temperature_c = Some(t);
            r.humidity = Some(50.0);
            r.battery_ok = Some(true);
            match encode(&r, 2) {
                Ok(pairs) => {
                    let (res, got) = decode(&r, &pairs);
                    assert_eq!(res.count(), 2);
                    assert!(got[0].same_content(&r));
                }
                Err(EncodeError::Unrepresentable) => found += 1,
                Err(e) => panic!("{e}"),
            }
        }
    }
    assert!(found > 0, "no frame of this kind among 768");
}

// ---------------------------------------------------------------- from_json

#[test]
fn from_json_reads_what_to_json_writes_for_every_model() {
    let mut rng = Rng::new(773);
    for model in MODELS {
        for _ in 0..50 {
            let mut x = random_reading(&mut rng, model);
            x.time_s = Some(12.345);
            x.center_hz = 433_920_000;
            x.freq_offset_hz = -2000;
            x.rssi_db = -12.5;
            x.repeats = 7;
            let text = json(&x);
            let y = Reading::from_json(&text).unwrap();
            assert_eq!(radio_free(&y), radio_free(&x), "{text}");
            assert_eq!(y.time_s, Some(12.345));
            assert_eq!(y.model, model);
            // the radio fields are ignored
            assert_eq!((y.freq_offset_hz, y.rssi_db, y.repeats), (0, 0.0, 1));
            // reading it again and writing it gives the same fields up to the radio ones
            let mut z = y;
            z.time_s = x.time_s;
            let mut buf = JsonBuf::new();
            z.to_json(&mut buf).unwrap();
            assert!(
                std::str::from_utf8(buf.as_bytes())
                    .unwrap()
                    .starts_with(&text[..text.find("\"mic\"").unwrap()])
            );
        }
    }
}

#[test]
fn from_json_ignores_unknown_fields_and_accepts_whitespace_and_escapes() {
    let r = Reading::from_json(
        " { \"extra\" : { \"a\" : [ 1 , 2 , { \"b\" : null } ] , \"s\" : \"x\\\"y\" } ,\n\"model\" : \"nexus\" , \"id\" : 12 ,\n \"channel\":3,\"battery_ok\":false, \"temperature_C\" : -3.5 , \"humidity\":null, \"mic\":\"repeat\",\"freq_MHz\":433.9200,\"rssi_dB\":-9.0,\"repeats\":4,\"time\":1.5 } ",
    )
    .unwrap();
    assert_eq!(
        (r.model, r.id, r.channel, r.battery_ok),
        ("nexus", 12, Some(3), Some(false))
    );
    assert_eq!(
        (r.temperature_c, r.humidity, r.time_s),
        (Some(-3.5), None, Some(1.5))
    );
    assert_eq!(r.mic, Mic::Repeat);
    // an escaped model string
    let r = Reading::from_json(r#"{"model":"Bresser-3CH","id":1}"#).unwrap();
    assert_eq!(r.model, "Bresser-3CH");
    let r = Reading::from_json(r#"{"id":1,"model":"wh2"}"#).unwrap();
    assert_eq!(r.model, "wh2");
    // an empty object has no model
    assert_eq!(Reading::from_json("{}"), Err(JsonError::Missing("model")));
    assert_eq!(
        Reading::from_json(r#"{"model":"wh2"}"#),
        Err(JsonError::Missing("id"))
    );
}

#[test]
fn from_json_accepts_surrogate_escapes_in_skipped_fields_and_keys() {
    // `@` stands for a backslash-u so that the escapes survive any editor.
    let j = |s: &str| Reading::from_json(&s.replace('@', "\\u"));
    // A pair, a lone high and a lone low surrogate (U+FFFD), also in a key.
    let r = j(
        r#"{"note":"@d83d@de00x","@d83d@de00":1,"lo":"@dc00","hi":"@d83d","model":"nexus","id":5}"#,
    )
    .unwrap();
    assert_eq!((r.model, r.id), ("nexus", 5));
    // A high surrogate followed by another escape that is no low surrogate.
    assert!(j(r#"{"a":"@d83d@0041","model":"nexus","id":5}"#).is_ok());
    // Still rejected: a truncated escape.
    assert!(j(r#"{"a":"@d83d@d8","model":"nexus","id":5}"#).is_err());
    assert!(j(r#"{"a":"@d83"}"#).is_err());
}

#[test]
fn from_json_rejects_everything_else_with_an_error() {
    for (text, want) in [
        ("", JsonError::Syntax),
        ("[]", JsonError::Syntax),
        ("{", JsonError::Syntax),
        ("}", JsonError::Syntax),
        (r#"{"model":"wh2","id":1"#, JsonError::Syntax),
        (r#"{"model":"wh2","id":1} x"#, JsonError::Syntax),
        (r#"{"model":"wh2" "id":1}"#, JsonError::Syntax),
        (r#"{"model":"wh2","id":1,}"#, JsonError::Syntax),
        (r#"{"model":"wh2","id":1,"a":"\q"}"#, JsonError::Syntax),
        (r#"{"model":"wh2","id":1,"a":"\u12"}"#, JsonError::Syntax),
        (
            r#"{"model":"wh2","id":1,"a":[[[[[[[[[[1]]]]]]]]]]}"#,
            JsonError::Syntax,
        ),
        (r#"{"model":"nobody","id":1}"#, JsonError::UnknownModel),
        (r#"{"model":"bresser_3ch","id":1}"#, JsonError::UnknownModel),
        (
            r#"{"model":"wh2wh2wh2wh2wh2wh2wh2wh2wh2wh2wh2wh2","id":1}"#,
            JsonError::UnknownModel,
        ),
        (r#"{"model":"wh2","id":-1}"#, JsonError::BadValue("id")),
        (r#"{"model":"wh2","id":1.5}"#, JsonError::BadValue("id")),
        (r#"{"model":"wh2","id":"1"}"#, JsonError::BadValue("id")),
        (
            r#"{"model":"wh2","id":1,"channel":300}"#,
            JsonError::BadValue("channel"),
        ),
        (
            r#"{"model":"wh2","id":1,"temperature_C":"hot"}"#,
            JsonError::BadValue("temperature_C"),
        ),
        (
            r#"{"model":"wh2","id":1,"temperature_C":1e999}"#,
            JsonError::BadValue("temperature_C"),
        ),
        (
            r#"{"model":"wh2","id":1,"battery_ok":1}"#,
            JsonError::BadValue("battery_ok"),
        ),
    ] {
        assert_eq!(Reading::from_json(text), Err(want), "{text}");
    }
}

#[test]
fn from_json_survives_mutated_text() {
    let mut rng = Rng::new(773 + 1000);
    let mut seen_ok = 0;
    for model in MODELS {
        let text = json(&random_reading(&mut rng, model));
        let bytes = text.as_bytes();
        for _ in 0..400 {
            let mut b = bytes.to_vec();
            for _ in 0..rng.range(1, 3) {
                let i = rng.below(b.len() as u64) as usize;
                match rng.below(3) {
                    0 => b[i] = rng.below(128) as u8,
                    1 => {
                        b.remove(i);
                    }
                    _ => b.insert(i, b"{}[]\":,\\ -.e0"[rng.below(13) as usize]),
                }
            }
            if let Ok(s) = std::str::from_utf8(&b)
                && Reading::from_json(s).is_ok()
            {
                seen_ok += 1;
            }
            // a prefix of the text
            let n = rng.below(bytes.len() as u64) as usize;
            assert!(Reading::from_json(&text[..n]).is_err());
        }
    }
    assert!(seen_ok > 0);
}
