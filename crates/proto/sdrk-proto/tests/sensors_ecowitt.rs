//! The Ecowitt / Fine Offset devices (M46, decisions D1150 to D1160): for every device
//! `decode(encode(x)) == x` over its whole range (also through the JSON form), a flipped bit gives
//! no reading, a truncated train gives none, and the encoder rejects what the device cannot
//! carry. Seed 780. The reference for the frames is `rtl_433` (decision D1149, see
//! `tools/crosscheck_rtl433.py`); these tests need no `rtl_433`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines
)]

use sdrk_proto::pulses::{self, Modulation, Pair};
use sdrk_proto::sensors::{
    self, DecodeResult, EncodeError, Extra, MAX_TRAIN_PAIRS, Reading, ReadingSink, TrainSink,
};

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
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
}

/// A device under test: the model, the bit time of an FSK device (`None` for pulse-width
/// coding), a generator of random readings over the whole range, and the first bit of the
/// message after which a flipped bit must be seen (FSK: after the sync word).
struct Dev {
    model: &'static str,
    bit_us: Option<u32>,
    make: fn(&mut Rng, &mut Reading),
    /// The bits of the train before the message (preamble and sync word).
    head_bits: usize,
}

/// `num / den` as the `f32` of its three-decimal text.
fn milli(num: i32, den: i32) -> f32 {
    let k = (f64::from(num) * 1000.0 / f64::from(den)).round();
    (k as i32) as f32 / 1000.0
}

fn tenths(v: i64) -> f32 {
    v as f32 / 10.0
}

fn temp_raw(raw: i64) -> f32 {
    // the expression of the decoders: (raw - 400) / 10
    (raw - 400) as f32 / 10.0
}

const DEVICES: &[Dev] = &[
    Dev {
        model: "Ecowitt-WH53",
        bit_us: None,
        make: |rng, r| {
            r.id = rng.range(0, 255) as u32;
            r.channel = Some(rng.range(1, 3) as u8);
            r.temperature_c = Some(temp_raw(rng.range(0, 1023)));
        },
        head_bits: 7,
    },
    Dev {
        model: "Fineoffset-WH55",
        bit_us: Some(60),
        make: |rng, r| {
            r.id = rng.range(0, 65535) as u32;
            r.channel = Some(rng.range(1, 4) as u8);
            r.set_extra(Extra::BatteryOk, rng.range(0, 5) as f32 / 5.0);
            r.set_extra(Extra::RawValue, rng.range(0, 65535) as f32);
            r.set_extra(Extra::Sensitivity, rng.range(0, 1) as f32);
            r.set_extra(Extra::Alarm, rng.range(0, 1) as f32);
        },
        head_bits: 3 * 8 + 16,
    },
    Dev {
        model: "Fineoffset-WH51",
        bit_us: Some(58),
        make: |rng, r| {
            r.id = rng.range(0, 0xFF_FFFF) as u32;
            let mv = rng.range(0, 31) as i32 * 100;
            r.set_extra(Extra::BatteryMv, mv as f32);
            r.set_extra(Extra::BatteryOk, milli(mv - 700, 900));
            r.set_extra(Extra::Moisture, rng.range(0, 100) as f32);
            r.set_extra(Extra::Boost, rng.range(0, 7) as f32);
            r.set_extra(Extra::AdRaw, rng.range(0, 511) as f32);
        },
        head_bits: 4 * 8 + 16,
    },
    Dev {
        model: "Fineoffset-WN34",
        bit_us: Some(58),
        make: |rng, r| wn34_values(rng, r, 0, 1000, 400),
        head_bits: 4 * 8 + 16,
    },
    Dev {
        model: "Fineoffset-WN34D",
        bit_us: Some(58),
        make: |rng, r| wn34_values(rng, r, -550, 1250, 0),
        head_bits: 4 * 8 + 16,
    },
    Dev {
        model: "AmbientWeather-WH31E",
        bit_us: Some(56),
        make: wh31,
        head_bits: 6 * 8 + 16,
    },
    Dev {
        model: "AmbientWeather-WH31B",
        bit_us: Some(56),
        make: wh31,
        head_bits: 6 * 8 + 16,
    },
    Dev {
        model: "EcoWitt-WH40",
        bit_us: Some(56),
        make: |rng, r| {
            r.id = rng.range(0, 0xFFFF) as u32;
            // 0: no battery value
            let v = rng.range(0, 31) as i32;
            if v > 0 {
                r.set_extra(Extra::BatteryV, v as f32 / 10.0);
                r.set_extra(Extra::BatteryOk, milli((v - 9).clamp(0, 6), 6));
            }
            r.rain_mm = Some(tenths(rng.range(0, 65535)));
        },
        head_bits: 6 * 8 + 16,
    },
    Dev {
        model: "Fineoffset-WH45",
        bit_us: Some(58),
        make: |rng, r| {
            r.id = rng.range(0, 0xFF_FFFF) as u32;
            r.temperature_c = Some(temp_raw(rng.range(0, 2047)));
            r.humidity = Some(rng.range(0, 100) as f32);
            let count = rng.range(0, 6);
            r.set_extra(Extra::BatteryOk, count.min(5) as f32 / 5.0);
            r.set_extra(Extra::ExtPower, f32::from(u8::from(count == 6)));
            r.set_extra(Extra::Pm25, tenths(rng.range(0, 16383)));
            r.set_extra(Extra::Pm10, tenths(rng.range(0, 16383)));
            r.set_extra(Extra::Co2, rng.range(0, 65535) as f32);
        },
        head_bits: 4 * 8 + 16,
    },
    Dev {
        model: "Fineoffset-WS80",
        bit_us: Some(58),
        make: |rng, r| {
            r.id = rng.range(0, 0xFF_FFFF) as u32;
            let mv = rng.range(0, 255) as i32 * 20;
            r.set_extra(Extra::BatteryMv, mv as f32);
            r.set_extra(Extra::BatteryOk, milli((mv - 1400).max(0), 1600));
            // every value may be "not available"
            if rng.below(6) != 0 {
                r.temperature_c = Some(temp_raw(rng.range(0, 1022)));
            }
            if rng.below(6) != 0 {
                r.humidity = Some(rng.range(0, 100) as f32);
            }
            if rng.below(6) != 0 {
                r.wind_dir_deg = Some(rng.range(0, 359) as f32);
            }
            if rng.below(6) != 0 {
                r.set_extra(Extra::WindAvgMs, tenths(rng.range(0, 510)));
            }
            if rng.below(6) != 0 {
                r.set_extra(Extra::WindMaxMs, tenths(rng.range(0, 510)));
            }
            if rng.below(6) != 0 {
                r.set_extra(Extra::Uvi, tenths(rng.range(0, 254)));
            }
            if rng.below(6) != 0 {
                r.set_extra(Extra::LightLux, (rng.range(0, 65534) * 10) as f32);
            }
        },
        head_bits: 6 * 8 + 16,
    },
];

fn wn34_values(rng: &mut Rng, r: &mut Reading, lo: i64, hi: i64, off: i64) {
    r.id = rng.range(0, 0xFF_FFFF) as u32;
    r.temperature_c = Some(tenths(rng.range(lo, hi) - off));
    let mv = rng.range(0, 127) as i32 * 20;
    let bars = [1200, 1300, 1380, 1440].iter().filter(|&&t| mv > t).count();
    r.set_extra(Extra::BatteryMv, mv as f32);
    r.set_extra(Extra::BatteryOk, bars as f32 / 4.0);
}

fn wh31(rng: &mut Rng, r: &mut Reading) {
    r.id = rng.range(0, 255) as u32;
    r.channel = Some(rng.range(1, 8) as u8);
    r.battery_ok = Some(rng.below(2) == 1);
    r.temperature_c = Some(temp_raw(rng.range(0, 1023)));
    r.humidity = Some(rng.range(0, 100) as f32);
}

fn reading(model: &'static str) -> Reading {
    let enc = sensors::encoder_by_model(model).unwrap();
    let mut r = Reading::new(model, 0, enc.mic(), &enc.header());
    r.temp_decimals = enc.temp_decimals();
    r
}

fn encode(r: &Reading, repeats: u32) -> Result<Vec<Pair>, EncodeError> {
    let enc = sensors::encoder_by_model(r.model).ok_or(EncodeError::UnknownModel)?;
    let mut storage = vec![Pair::default(); MAX_TRAIN_PAIRS];
    let mut sink = TrainSink::new(&mut storage);
    enc.encode(r, repeats, &mut sink)?;
    Ok(sink.pairs().to_vec())
}

fn decode(model: &str, pairs: &[Pair]) -> (DecodeResult, Vec<Reading>) {
    let enc = sensors::encoder_by_model(model).unwrap();
    let dec = sensors::by_name(enc.name()).unwrap();
    let mut buf = vec![0u8; 16 + 8 * pairs.len()];
    let n = pulses::encode_joined(&enc.header(), pairs, &mut buf).unwrap();
    let train = pulses::decode(&buf[..n]).unwrap();
    let mut sink = ReadingSink::new();
    let res = dec.decode(&train, &mut sink);
    (res, sink.iter().copied().collect())
}

fn full_json(r: &Reading) -> String {
    let mut r = *r;
    r.time_s = None;
    let mut s = String::new();
    r.to_json(&mut s).unwrap();
    s
}

/// The JSON of a reading up to its radio fields.
fn json(r: &Reading) -> String {
    let s = full_json(r);
    s[..s.find("\"mic\"").unwrap()].to_string()
}

/// The bits of one copy of an NRZ train (runs of `bit_us`, a pulse is a 1); the zeros after
/// the last pulse are not in the train.
fn nrz_bits(pairs: &[Pair], bit_us: u32) -> Vec<bool> {
    let mut v = Vec::new();
    for p in pairs {
        v.extend(std::iter::repeat_n(true, (p.pulse / bit_us) as usize));
        v.extend(std::iter::repeat_n(false, (p.gap / bit_us) as usize));
    }
    v
}

fn nrz_pairs(bits: &[bool], bit_us: u32) -> Vec<Pair> {
    let mut out: Vec<Pair> = Vec::new();
    let mut k = 0;
    while k < bits.len() {
        let mut pulse = 0;
        while k < bits.len() && bits[k] {
            pulse += 1;
            k += 1;
        }
        let mut gap = 0;
        while k < bits.len() && !bits[k] {
            gap += 1;
            k += 1;
        }
        out.push(Pair {
            pulse: pulse * bit_us,
            gap: gap * bit_us,
        });
    }
    if let Some(l) = out.last_mut() {
        l.gap = 0;
    }
    out
}

/// The train with bit `k` of the message flipped (a pulse width 500 and 1480 swap for the
/// pulse-width coding, a run is rebuilt for NRZ).
fn flipped(dev: &Dev, pairs: &[Pair], k: usize) -> Vec<Pair> {
    match dev.bit_us {
        None => {
            let mut v = pairs.to_vec();
            let p = &mut v[k];
            p.pulse = if p.pulse < 1000 { 1480 } else { 500 };
            v
        }
        Some(us) => {
            let mut bits = nrz_bits(pairs, us);
            bits[k] = !bits[k];
            nrz_pairs(&bits, us)
        }
    }
}

#[test]
fn random_readings_round_trip_over_the_whole_range() {
    let mut rng = Rng::new(780);
    for dev in DEVICES {
        let mut unrepresentable = 0;
        for k in 0..400 {
            let mut x = reading(dev.model);
            (dev.make)(&mut rng, &mut x);
            // through the JSON form
            let text = full_json(&x);
            let y = Reading::from_json(&text).unwrap();
            assert_eq!(json(&y), json(&x));
            let repeats = 1 + (k % 3) as u32;
            let pairs = match encode(&y, repeats) {
                Ok(p) => p,
                Err(EncodeError::Unrepresentable) => {
                    unrepresentable += 1;
                    continue;
                }
                Err(e) => panic!("{}: {text}: {e}", dev.model),
            };
            assert_eq!(pairs.last().unwrap().gap, 0);
            let (res, got) = decode(dev.model, &pairs);
            assert!(
                matches!(res, DecodeResult::Decoded(_)),
                "{}: {text}: {res:?}",
                dev.model
            );
            assert_eq!(got.len() as u32, repeats, "{}: {text}", dev.model);
            for g in &got {
                assert!(g.same_content(&x), "{}: {text}\n{g:?}", dev.model);
                assert_eq!(g.mic, x.mic);
            }
        }
        assert!(unrepresentable < 20, "{}: {unrepresentable}", dev.model);
    }
}

#[test]
fn a_flipped_bit_gives_no_reading_and_a_truncated_train_none() {
    let mut rng = Rng::new(781);
    for dev in DEVICES {
        let mut x = reading(dev.model);
        (dev.make)(&mut rng, &mut x);
        let pairs = encode(&x, 1).unwrap();
        let nbits = match dev.bit_us {
            None => pairs.len(),
            Some(us) => nrz_bits(&pairs, us).len(),
        };
        let mut seen = 0;
        // every bit of the message (not the preamble): the check bytes catch every single flip
        for k in dev.head_bits..nbits {
            let bad = flipped(dev, &pairs, k);
            let (res, got) = decode(dev.model, &bad);
            assert!(
                !matches!(res, DecodeResult::Decoded(_)) && got.is_empty(),
                "{}: bit {k}: {res:?}",
                dev.model
            );
            seen += 1;
        }
        assert!(seen > 30, "{}", dev.model);
        for cut in [1, pairs.len() / 2, pairs.len() - 3] {
            let (res, got) = decode(dev.model, &pairs[..cut]);
            assert!(
                !matches!(res, DecodeResult::Decoded(_)) && got.is_empty(),
                "{}: cut {cut}: {res:?}",
                dev.model
            );
        }
    }
}

#[test]
fn the_encoder_rejects_what_the_device_cannot_carry() {
    for dev in DEVICES {
        let mut rng = Rng::new(782);
        let mut good = reading(dev.model);
        (dev.make)(&mut rng, &mut good);
        assert!(encode(&good, 1).is_ok(), "{}", dev.model);
        let mut r = good;
        r.temperature_c = r.temperature_c.map(|_| f32::NAN);
        if r.temperature_c.is_some() {
            assert_eq!(
                encode(&r, 1),
                Err(EncodeError::OutOfRange("temperature_C")),
                "{}",
                dev.model
            );
        }
        let mut r = good;
        r.id = 1 << 30;
        assert_eq!(
            encode(&r, 1),
            Err(EncodeError::OutOfRange("id")),
            "{}",
            dev.model
        );
        let mut r = good;
        r.pressure_hpa = Some(1000.0);
        assert_eq!(
            encode(&r, 1),
            Err(EncodeError::UnsupportedField("pressure_hPa")),
            "{}",
            dev.model
        );
        assert_eq!(encode(&good, 0), Err(EncodeError::InvalidRepeats));
        let enc = sensors::encoder_by_model(dev.model).unwrap();
        assert_eq!(
            enc.modulation(),
            if dev.bit_us.is_some() {
                Modulation::Fsk
            } else {
                Modulation::Ook
            }
        );
    }
}

/// The limits of the temperature range of the WH53 are readings, one step beyond is an error.
#[test]
fn the_temperature_range_of_the_wh53() {
    for (t, ok) in [(-40.0, true), (62.3, true), (-40.1, false), (62.4, false)] {
        let mut r = reading("Ecowitt-WH53");
        r.id = 1;
        r.temperature_c = Some(t);
        assert_eq!(encode(&r, 1).is_ok(), ok, "{t}");
        if ok {
            let (_, got) = decode("Ecowitt-WH53", &encode(&r, 1).unwrap());
            assert_eq!(got[0].temperature_c, Some(t));
        }
    }
    let mut r = reading("Ecowitt-WH53");
    r.temperature_c = Some(1.0);
    r.channel = Some(4);
    assert_eq!(encode(&r, 1), Err(EncodeError::OutOfRange("channel")));
    r.channel = Some(1);
    r.humidity = Some(5.0);
    assert_eq!(encode(&r, 1), Err(EncodeError::UnsupportedField("humidity")));
}

/// The receiver chain works at 250 kHz: the widths are rounded to quarter microseconds of the
/// 1 MHz train, as the pulse detector measures them, and every device still decodes.
#[test]
fn trains_at_the_sample_rate_of_the_receiver_decode() {
    let mut rng = Rng::new(783);
    for dev in DEVICES {
        for _ in 0..40 {
            let mut x = reading(dev.model);
            (dev.make)(&mut rng, &mut x);
            let Ok(pairs) = encode(&x, 2) else { continue };
            // the edges rounded to samples of 250 kHz (a bit of 58 us is 14.5 samples: the
            // widths alternate between 14 and 15)
            let phase = rng.range(0, 3) as u64;
            let mut edge = phase;
            let mut last = (edge + 2) / 4;
            let mut coarse: Vec<Pair> = Vec::new();
            for p in &pairs {
                let mut w = [0u32; 2];
                for (k, width) in [p.pulse, p.gap].into_iter().enumerate() {
                    edge += u64::from(width);
                    let at = (edge + 2) / 4;
                    w[k] = (at - last) as u32;
                    last = at;
                }
                coarse.push(Pair {
                    pulse: w[0],
                    gap: w[1],
                });
            }
            let enc = sensors::encoder_by_model(dev.model).unwrap();
            let mut header = enc.header();
            header.rate_hz = 250_000;
            let mut buf = vec![0u8; 16 + 8 * coarse.len()];
            let n = pulses::encode_joined(&header, &coarse, &mut buf).unwrap();
            let train = pulses::decode(&buf[..n]).unwrap();
            let mut sink = ReadingSink::new();
            let res = sensors::by_name(enc.name()).unwrap().decode(&train, &mut sink);
            assert!(
                matches!(res, DecodeResult::Decoded(_)),
                "{}: {}: {res:?}",
                dev.model,
                json(&x)
            );
            assert!(sink.iter().all(|g| g.same_content(&x)), "{}", dev.model);
        }
    }
}

/// A WS80 with every value (the longest JSON of the devices) fits the 256 bytes of a packet
/// only in the short form that `sensor_decode` falls back to.
#[test]
fn the_longest_reading_fits_a_packet_in_the_short_form() {
    let mut r = reading("Fineoffset-WS80");
    r.id = 0xFF_FFFF;
    r.time_s = Some(1234.567);
    r.center_hz = 433_920_000;
    r.temperature_c = Some(-40.0);
    r.humidity = Some(100.0);
    r.wind_dir_deg = Some(359.0);
    r.set_extra(Extra::BatteryOk, -0.5);
    r.set_extra(Extra::BatteryMv, 5100.0);
    r.set_extra(Extra::WindAvgMs, 51.0);
    r.set_extra(Extra::WindMaxMs, 51.0);
    r.set_extra(Extra::Uvi, 25.4);
    r.set_extra(Extra::LightLux, 655_340.0);
    let mut full = String::new();
    r.to_json(&mut full).unwrap();
    let mut short = String::new();
    r.to_json_short(&mut short).unwrap();
    assert!(full.len() > 256 && short.len() <= 256, "{} {}", full.len(), short.len());
    assert!(Reading::from_json(&short).is_ok());
}
