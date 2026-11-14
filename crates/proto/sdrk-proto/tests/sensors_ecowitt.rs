//! The Ecowitt / Fine Offset devices (M46, decisions D1150 to D1163): for every device
//! `decode(encode(x)) == x` over its whole range (also through the JSON form), a flipped bit gives
//! no reading, a truncated train gives none, and the encoder rejects what the device cannot
//! carry. Seeds 780 to 789. The reference for the frames is `rtl_433` (decision D1149, see
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
    Dev {
        model: "Fineoffset-WS90",
        bit_us: Some(58),
        make: |rng, r| {
            r.id = rng.range(0, 0xFF_FFFF) as u32;
            let mv = rng.range(0, 255) as i32 * 20;
            r.set_extra(Extra::BatteryMv, mv as f32);
            let level = if mv < 1400 {
                0
            } else {
                ((mv - 1400) / 16).min(100)
            };
            r.set_extra(Extra::BatteryOk, milli(level, 100));
            // every value of the WS80 part may be "not available"
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
            r.rain_mm = Some(tenths(rng.range(0, 65535)));
            r.set_extra(Extra::RainStart, rng.range(0, 1) as f32);
            r.set_extra(Extra::SupercapV, tenths(rng.range(0, 63)));
            r.set_extra(Extra::Firmware, rng.range(0, 255) as f32);
        },
        head_bits: 6 * 8 + 16,
    },
    Dev {
        model: "Fineoffset-WS85",
        bit_us: Some(58),
        make: |rng, r| {
            r.id = rng.range(0, 0xFF_FFFF) as u32;
            let mv = rng.range(0, 255) as i32 * 20;
            r.set_extra(Extra::BatteryMv, mv as f32);
            let level = if mv < 1400 {
                0
            } else {
                ((mv - 1400) / 16).min(100)
            };
            r.set_extra(Extra::BatteryOk, f32::from(u8::from(mv > 2400)));
            r.set_extra(Extra::BatteryPct, level as f32);
            if rng.below(6) != 0 {
                r.wind_dir_deg = Some(rng.range(0, 359) as f32);
            }
            if rng.below(6) != 0 {
                r.set_extra(Extra::WindAvgMs, tenths(rng.range(0, 510)));
            }
            if rng.below(6) != 0 {
                r.set_extra(Extra::WindMaxMs, tenths(rng.range(0, 510)));
            }
            r.rain_mm = Some(tenths(rng.range(0, 65535)));
            r.set_extra(Extra::RainStart, rng.range(0, 1) as f32);
            r.set_extra(Extra::SupercapV, tenths(rng.range(0, 63)));
            r.set_extra(Extra::Firmware, rng.range(0, 255) as f32);
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
        // the bytes after the sum byte are not checked (WS85: 4 bytes, WS90: 1 byte)
        let tail = match dev.model {
            "Fineoffset-WS85" => 32,
            "Fineoffset-WS90" => 8,
            _ => 0,
        };
        for k in dev.head_bits..nbits - tail {
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
    assert_eq!(
        encode(&r, 1),
        Err(EncodeError::UnsupportedField("humidity"))
    );
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
            let res = sensors::by_name(enc.name())
                .unwrap()
                .decode(&train, &mut sink);
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
    assert!(
        full.len() > 256 && short.len() <= 256,
        "{} {}",
        full.len(),
        short.len()
    );
    assert!(Reading::from_json(&short).is_ok());
}

/// A WS90 and a WS85 with every optional value missing: the keys are absent in the JSON and the
/// reading comes back equal (seed 784).
#[test]
fn ws90_and_ws85_without_optional_values() {
    let mut rng = Rng::new(784);
    for model in ["Fineoffset-WS90", "Fineoffset-WS85"] {
        let dev = DEVICES.iter().find(|d| d.model == model).unwrap();
        for _ in 0..40 {
            let mut x = reading(model);
            (dev.make)(&mut rng, &mut x);
            x.temperature_c = None;
            x.humidity = None;
            x.wind_dir_deg = None;
            for e in [
                Extra::WindAvgMs,
                Extra::WindMaxMs,
                Extra::Uvi,
                Extra::LightLux,
            ] {
                x.extras[e as usize] = None;
            }
            let (res, got) = decode(model, &encode(&x, 1).unwrap());
            assert_eq!(res, DecodeResult::Decoded(1), "{}", json(&x));
            assert!(got[0].same_content(&x), "{}", json(&x));
            let text = json(&got[0]);
            for key in [
                "temperature_C",
                "humidity",
                "wind_dir_deg",
                "wind_avg_m_s",
                "uvi",
            ] {
                assert!(!text.contains(key), "{text}");
            }
            assert!(
                text.contains("rain_mm") && text.contains("supercap_V"),
                "{text}"
            );
        }
    }
}

/// What the WS90 and WS85 encoders reject (seed 785).
#[test]
fn ws90_and_ws85_encoders_reject_bad_readings() {
    let mut rng = Rng::new(785);
    for model in ["Fineoffset-WS90", "Fineoffset-WS85"] {
        let dev = DEVICES.iter().find(|d| d.model == model).unwrap();
        let mut good = reading(model);
        (dev.make)(&mut rng, &mut good);
        assert!(encode(&good, 1).is_ok());
        let with = |f: &dyn Fn(&mut Reading)| {
            let mut r = good;
            f(&mut r);
            encode(&r, 1)
        };
        assert_eq!(
            with(&|r| r.rain_mm = None),
            Err(EncodeError::MissingField("rain_mm"))
        );
        assert_eq!(
            with(&|r| r.rain_mm = Some(6553.6)),
            Err(EncodeError::OutOfRange("rain_mm"))
        );
        assert_eq!(
            with(&|r| r.set_extra(Extra::SupercapV, 6.4)),
            Err(EncodeError::OutOfRange("supercap_V"))
        );
        assert_eq!(
            with(&|r| r.set_extra(Extra::RainStart, 2.0)),
            Err(EncodeError::OutOfRange("rain_start"))
        );
        assert_eq!(
            with(&|r| r.set_extra(Extra::Firmware, 256.0)),
            Err(EncodeError::OutOfRange("firmware"))
        );
        assert_eq!(
            with(&|r| r.wind_dir_deg = Some(360.0)),
            Err(EncodeError::OutOfRange("wind_dir_deg"))
        );
        assert_eq!(
            with(&|r| r.set_extra(Extra::BatteryMv, 5120.0)),
            Err(EncodeError::OutOfRange("battery_mV"))
        );
        assert_eq!(
            with(&|r| r.wind_avg_km_h = Some(1.0)),
            Err(EncodeError::UnsupportedField("wind_avg_km_h"))
        );
        assert_eq!(
            with(&|r| r.set_extra(Extra::Pm25, 1.0)),
            Err(EncodeError::UnsupportedField("pm2_5_ug_m3"))
        );
    }
    // the battery level must follow the voltage
    let mut r = reading("Fineoffset-WS90");
    r.set_extra(Extra::BatteryMv, 2000.0);
    r.set_extra(Extra::BatteryOk, 0.99);
    r.rain_mm = Some(0.0);
    for (e, v) in [
        (Extra::RainStart, 0.0),
        (Extra::SupercapV, 0.0),
        (Extra::Firmware, 0.0),
    ] {
        r.set_extra(e, v);
    }
    assert_eq!(encode(&r, 1), Err(EncodeError::OutOfRange("battery_ok")));
}

fn crc8(bytes: &[u8]) -> u8 {
    let mut c = 0u8;
    for &b in bytes {
        c ^= b;
        for _ in 0..8 {
            c = if c & 0x80 != 0 { c << 1 ^ 0x31 } else { c << 1 };
        }
    }
    c
}

fn bytes_of(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|c| c.iter().fold(0u8, |a, &b| a << 1 | u8::from(b)))
        .collect()
}

/// The frame as `rtl_433` needs it: the row of 168 to 500 bits holds the sync `AA AA 2D D4`
/// with at least 256 bits after it, the family byte, the CRC and the sum at their places
/// (seed 786).
#[test]
fn ws90_and_ws85_frames_have_the_layout_rtl_433_needs() {
    let mut rng = Rng::new(786);
    for (model, family, crc_at) in [
        ("Fineoffset-WS90", 0x90u8, 30usize),
        ("Fineoffset-WS85", 0x85, 26),
    ] {
        let dev = DEVICES.iter().find(|d| d.model == model).unwrap();
        for _ in 0..20 {
            let mut x = reading(model);
            (dev.make)(&mut rng, &mut x);
            let pairs = encode(&x, 1).unwrap();
            let bits = nrz_bits(&pairs, 58);
            assert!((168..=500).contains(&bits.len()), "{}", bits.len());
            let b = bytes_of(&bits);
            assert_eq!(&b[..2], &[0xAA, 0xAA]);
            let s = b.iter().position(|&v| v == 0x2D).unwrap();
            assert_eq!(b[s + 1], 0xD4);
            assert!(s >= 2);
            let m = &b[s + 2..];
            assert!(m.len() * 8 >= 256, "{}", m.len());
            assert_eq!(m[0], family);
            assert_eq!(m[crc_at], crc8(&m[..crc_at]));
            let sum = m[..=crc_at].iter().fold(0u8, |a, &v| a.wrapping_add(v));
            assert_eq!(m[crc_at + 1], sum);
        }
    }
}

/// A WS85 frame of 28 bytes (`rtl_433` wants 32 after the sync) gives no reading, and the
/// decoders of the family do not take each other's frames (seed 787).
#[test]
fn ws90_and_ws85_do_not_take_other_frames() {
    let mut rng = Rng::new(787);
    let models = ["Fineoffset-WS90", "Fineoffset-WS85", "Fineoffset-WS80"];
    let mut frames = Vec::new();
    for model in models {
        let dev = DEVICES.iter().find(|d| d.model == model).unwrap();
        let mut x = reading(model);
        (dev.make)(&mut rng, &mut x);
        frames.push(encode(&x, 1).unwrap());
    }
    for (i, pairs) in frames.iter().enumerate() {
        for (k, other) in models.iter().enumerate() {
            let enc = sensors::encoder_by_model(other).unwrap();
            let dec = sensors::by_name(enc.name()).unwrap();
            let mut buf = vec![0u8; 16 + 8 * pairs.len()];
            let n = pulses::encode_joined(&enc.header(), pairs, &mut buf).unwrap();
            let train = pulses::decode(&buf[..n]).unwrap();
            let mut sink = ReadingSink::new();
            let res = dec.decode(&train, &mut sink);
            assert_eq!(
                matches!(res, DecodeResult::Decoded(_)),
                i == k,
                "{} {other}",
                models[i]
            );
        }
    }
    // the WS85 without its four padding bytes
    let bits = nrz_bits(&frames[1], 58);
    let cut = bits.len() - 32;
    let (res, got) = decode("Fineoffset-WS85", &nrz_pairs(&bits[..cut], 58));
    assert!(
        !matches!(res, DecodeResult::Decoded(_)) && got.is_empty(),
        "{res:?}"
    );
}

/// The battery rules at their steps: the level of the WS90 and WS85 and the integer
/// `battery_ok` of the WS85 above 2400 mV (seed 788).
#[test]
fn ws90_and_ws85_battery_steps() {
    let mut rng = Rng::new(788);
    for (mv, level) in [
        (0, 0),
        (1380, 0),
        (1400, 0),
        (1420, 1),
        (2220, 51),
        (3000, 100),
        (3020, 100),
        (5100, 100),
    ] {
        for model in ["Fineoffset-WS90", "Fineoffset-WS85"] {
            let dev = DEVICES.iter().find(|d| d.model == model).unwrap();
            let mut x = reading(model);
            (dev.make)(&mut rng, &mut x);
            x.set_extra(Extra::BatteryMv, mv as f32);
            if model.ends_with("90") {
                x.set_extra(Extra::BatteryOk, milli(level, 100));
            } else {
                x.set_extra(Extra::BatteryPct, level as f32);
                x.set_extra(Extra::BatteryOk, f32::from(u8::from(mv > 2400)));
            }
            let (res, got) = decode(model, &encode(&x, 1).unwrap());
            assert_eq!(res, DecodeResult::Decoded(1), "{model} {mv}");
            assert!(got[0].same_content(&x), "{model} {mv}");
        }
    }
    for (mv, ok) in [(2400, 0.0), (2420, 1.0)] {
        let mut x = reading("Fineoffset-WS85");
        x.set_extra(Extra::BatteryMv, mv as f32);
        x.set_extra(Extra::BatteryOk, ok);
        x.set_extra(Extra::BatteryPct, ((mv - 1400) / 16) as f32);
        x.rain_mm = Some(1.0);
        for (e, v) in [
            (Extra::RainStart, 1.0),
            (Extra::SupercapV, 1.0),
            (Extra::Firmware, 130.0),
        ] {
            x.set_extra(e, v);
        }
        let (_, got) = decode("Fineoffset-WS85", &encode(&x, 1).unwrap());
        assert_eq!(got[0].extra(Extra::BatteryOk), Some(ok));
    }
}

/// A WS90 with every value of a typical size does not fit a packet in the short or the compact
/// form but in the minimal one (without `freq_MHz`, `mic`, `firmware` and `supercap_V`), and a
/// WS85 fits in the compact one (seed 789).
#[test]
fn the_ws90_with_every_value_in_json() {
    let mut rng = Rng::new(789);
    let dev = DEVICES
        .iter()
        .find(|d| d.model == "Fineoffset-WS90")
        .unwrap();
    let mut x = reading("Fineoffset-WS90");
    (dev.make)(&mut rng, &mut x);
    x.id = 0xA1_B2C3;
    x.time_s = Some(1234.567);
    x.center_hz = 433_920_000;
    x.temperature_c = Some(-4.5);
    x.humidity = Some(81.0);
    x.wind_dir_deg = Some(245.0);
    x.rain_mm = Some(123.4);
    for (e, v) in [
        (Extra::WindAvgMs, 3.2),
        (Extra::WindMaxMs, 7.9),
        (Extra::Uvi, 2.5),
        (Extra::LightLux, 52_000.0),
        (Extra::SupercapV, 5.1),
        (Extra::Firmware, 130.0),
        (Extra::BatteryMv, 2800.0),
        (Extra::BatteryOk, 0.87),
        (Extra::RainStart, 1.0),
    ] {
        x.set_extra(e, v);
    }
    let mut short = String::new();
    x.to_json_short(&mut short).unwrap();
    assert!(short.len() > 256, "{short}");
    let mut compact = String::new();
    x.to_json_compact(&mut compact).unwrap();
    assert!(compact.len() > 256, "{compact}");
    let mut min = String::new();
    x.to_json_min(&mut min).unwrap();
    assert!(min.len() <= 256 && !min.contains("freq_MHz"), "{min}");
    let y = Reading::from_json(&min).unwrap();
    assert!(y.extra(Extra::WindMaxMs) == x.extra(Extra::WindMaxMs) && y.id == x.id);
    let mut widest = x;
    widest.id = 0xFF_FFFF;
    widest.temperature_c = Some(-40.0);
    widest.humidity = Some(100.0);
    widest.rain_mm = Some(6553.5);
    widest.set_extra(Extra::LightLux, 655_340.0);
    let mut text = String::new();
    widest.to_json_min(&mut text).unwrap();
    assert!(text.len() <= 256, "{text}");
}

#[test]
fn the_ws85_with_every_value_fits_the_compact_form() {
    let mut rng = Rng::new(789);
    let dev = DEVICES
        .iter()
        .find(|d| d.model == "Fineoffset-WS85")
        .unwrap();
    let mut x = reading("Fineoffset-WS85");
    (dev.make)(&mut rng, &mut x);
    x.id = 0xFF_FFFF;
    x.time_s = Some(1234.567);
    x.center_hz = 433_920_000;
    x.wind_dir_deg = Some(359.0);
    x.rain_mm = Some(6553.5);
    for (e, v) in [
        (Extra::WindAvgMs, 51.0),
        (Extra::WindMaxMs, 51.0),
        (Extra::SupercapV, 6.3),
        (Extra::Firmware, 255.0),
        (Extra::BatteryMv, 5100.0),
        (Extra::BatteryPct, 100.0),
    ] {
        x.set_extra(e, v);
    }
    let mut compact = String::new();
    x.to_json_compact(&mut compact).unwrap();
    assert!(compact.len() <= 256, "{}", compact.len());
    assert!(compact.contains("firmware") && !compact.contains("freq_MHz"));
}
