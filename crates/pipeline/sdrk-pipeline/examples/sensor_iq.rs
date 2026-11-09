//! Cross-check generator for the sensor protocols (see `tools/README-crosscheck.md`).
//!
//! `generate DIR [--count N] [--seed S] [--noise LSB] [--only DEVICE]` writes, for every encoder of
//! `sdrk_proto::sensors::ENCODERS`, one `cu8` IQ file at 2 MS/s (`DEVICE.cu8`) and a JSON
//! sidecar (`DEVICE.json`) that lists the readings transmitted. The readings are random over
//! the range of the device; every one is sent as a burst through the transmit chain
//! (`sensor_encode`, `pulse_key`, a modulator, `duc`) followed by 0.4 s of silence. OOK devices
//! use the chain of the hardware flow `bresser_3ch_tx.json`; FSK devices are synthesised
//! directly from the pulse trains of their encoder (a continuous-phase two-tone signal, the
//! pulse is the upper frequency).
//!
//! `decode FILE.cu8 --band 433|868` runs the receiver chain (`channelize`,
//! `band_pulse_detect`, `sensor_decode`) on a `cu8` file and prints one JSON line per reading
//! (the field `time` is seconds from the start of the file; `packet_index` is the packet index).

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::large_stack_frames
)]

use std::error::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use sdrk_device::{BlockMeta, SampleIndex};
use sdrk_pipeline::{BuildOptions, InputData, Packet, PacketFlags, Pipeline, Registry};
use sdrk_proto::pulses::{Modulation, Pair};
use sdrk_proto::sensors::{ENCODERS, MAX_TRAIN_PAIRS, Reading, SensorEncoder, TrainSink};
use serde_json::{Value, json};

type Res<T> = Result<T, Box<dyn Error>>;

/// Output rate of the IQ files.
const RF_RATE: u32 = 2_000_000;
/// Silence between two bursts, in seconds.
const SILENCE_S: f64 = 0.4;
/// Silence before the first burst, in seconds.
const LEAD_S: f64 = 0.2;
/// Centre frequency of the 433 MHz files.
const CENTER_433: u64 = 433_920_000;
/// Centre frequency of the 868 MHz files (the receiver flow `ism868_pulses`).
const CENTER_868: u64 = 868_500_000;
/// Carrier frequency of the 868 MHz sensors.
const CARRIER_868: u64 = 868_300_000;
/// Deviation of the FSK transmitter.
const FSK_DEVIATION_HZ: f64 = 35_000.0;

/// A xorshift64 generator.
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

fn reading(enc: &dyn SensorEncoder, model: &'static str, id: u32) -> Reading {
    let mut r = Reading::new(model, id, enc.mic(), &enc.header());
    r.temp_decimals = enc.temp_decimals();
    r
}

/// Tries to encode `r` with one repeat; true when the encoder takes it.
fn encodes(enc: &dyn SensorEncoder, r: &Reading) -> bool {
    let mut storage = vec![Pair::default(); MAX_TRAIN_PAIRS];
    let mut sink = TrainSink::new(&mut storage);
    enc.encode(r, 1, &mut sink).is_ok()
}

/// A random reading of the model over the whole range of the device, with the values at the
/// resolution of the device. A model this program does not know gets the first of a few generic
/// field sets that its encoder accepts.
fn random_reading(rng: &mut Rng, enc: &dyn SensorEncoder, model: &'static str) -> Option<Reading> {
    let tenths = |t: i64| Some(t as f32 / 10.0);
    let mut r = reading(enc, model, 0);
    match model {
        "wh2" => {
            r.id = rng.range(0, 0x7FF) as u32;
            r.temperature_c = tenths(rng.range(-800, 800));
            r.humidity = Some(rng.range(0, 100) as f32);
        }
        "nexus" => {
            r.id = rng.range(0, 255) as u32;
            r.channel = Some(rng.range(1, 4) as u8);
            r.battery_ok = Some(rng.below(2) == 1);
            r.temperature_c = tenths(rng.range(-400, 800));
            r.humidity = Some(rng.range(0, 100) as f32);
        }
        "oregon_thn132n" | "oregon_thgr122nx" => {
            r.id = rng.range(0, 255) as u32;
            r.channel = Some(rng.range(1, 3) as u8);
            r.battery_ok = Some(rng.below(2) == 1);
            r.temperature_c = tenths(rng.range(-700, 700));
            if model == "oregon_thgr122nx" {
                r.humidity = Some(rng.range(0, 99) as f32);
            }
        }
        "wh1080" => {
            r.id = rng.range(0, 0xFFF) as u32;
            r.temperature_c = tenths(rng.range(-400, 600));
            r.humidity = Some(rng.range(0, 100) as f32);
            r.wind_avg_km_h = Some(rng.range(0, 255) as f32 * 1.224);
            r.wind_max_km_h = Some(rng.range(0, 255) as f32 * 1.224);
            r.wind_dir_deg = Some(rng.range(0, 15) as f32 * 22.5);
            r.rain_mm = Some(rng.range(0, 4095) as f32 * 0.3);
        }
        "lacrosse_tx29" => {
            r.id = rng.range(0, 255) as u32;
            // the expression of the decoder
            r.temperature_c = Some((rng.range(0, 999) as f32) / 10.0 - 40.0);
            if rng.below(5) != 0 {
                r.humidity = Some(rng.range(0, 100) as f32);
            }
            r.battery_ok = Some(rng.below(2) == 1);
        }
        "Bresser-3CH" => {
            r.id = rng.range(0, 255) as u32;
            r.channel = Some(rng.range(1, 3) as u8);
            r.battery_ok = Some(rng.below(2) == 1);
            let raw = rng.range(700, 2500);
            let hundredths = ((raw - 1220) as f64 * 100.0 / 18.0).round();
            r.temperature_c = Some(hundredths as f32 / 100.0);
            r.humidity = Some(rng.range(0, 100) as f32);
        }
        _ => {
            // an unknown device: the first generic field set the encoder accepts
            r.id = rng.range(0, 255) as u32;
            r.temperature_c = tenths(rng.range(-100, 400));
            let mut with_humidity = r;
            with_humidity.humidity = Some(rng.range(0, 100) as f32);
            let mut full = with_humidity;
            full.channel = Some(1);
            full.battery_ok = Some(true);
            let mut chan = r;
            chan.channel = Some(1);
            chan.battery_ok = Some(true);
            return [full, with_humidity, chan, r]
                .into_iter()
                .find(|c| encodes(enc, c));
        }
    }
    Some(r)
}

/// The reading as the JSON object of the sidecar (the fields the device carries).
fn reading_json(r: &Reading) -> Res<Value> {
    let mut s = String::new();
    let mut r = *r;
    r.time_s = None;
    r.to_json(&mut s)?;
    let mut v: Value = serde_json::from_str(&s)?;
    if let Some(o) = v.as_object_mut() {
        for k in ["mic", "freq_MHz", "rssi_dB", "repeats", "time"] {
            o.remove(k);
        }
    }
    Ok(v)
}

/// The length of the train of `r` in microseconds (all repeats).
fn train_us(enc: &dyn SensorEncoder, r: &Reading) -> Res<u64> {
    let mut storage = vec![Pair::default(); MAX_TRAIN_PAIRS];
    let mut sink = TrainSink::new(&mut storage);
    if let Err(e) = enc.encode(r, enc.default_repeats(), &mut sink) {
        return Err(format!("{}: {e}", r.model).into());
    }
    let mut total = 0u64;
    for p in sink.pairs() {
        total += u64::from(p.pulse) + u64::from(p.gap);
    }
    Ok(total)
}

/// The transmit flow of an OOK device: the chain of the hardware flow `bresser_3ch_tx.json`.
fn tx_flow() -> (Value, u32) {
    let flow = json!({
        "sdrk": 1, "name": "sensor-tx", "settings": { "precision": "f32", "max_chunk": 512 },
        "inputs": {
            "clock": { "type": "f32", "rate_hz": 48_000 },
            "reading": { "type": "packets" }
        },
        "nodes": {
            "enc": { "kind": "sensor_encode", "params": {} },
            "key": { "kind": "pulse_key", "params": {} },
            "mod": { "kind": "ook_mod", "params": { "samples_per_symbol": 1, "level": 0.7 } },
            "up": { "kind": "resample", "params": { "rate_out_hz": 200_000 } },
            "duc": { "kind": "duc", "params": { "frequency_hz": 80_000, "interpolation": 10 } }
        },
        "edges": [
            { "from": "clock", "to": "key" },
            { "from": "reading", "to": "enc" },
            { "from": "enc", "to": "key", "port": "b" },
            { "from": "key", "to": "mod" },
            { "from": "mod", "to": "up" },
            { "from": "up", "to": "duc" },
            { "from": "duc", "to": "rf" }
        ],
        "outputs": { "rf": { "type": "stream", "format": "cs8", "underrun": "zeros" } }
    });
    (flow, 48_000)
}

/// Pushes `seconds` of clock silence and collects the cs8 bytes that come out.
fn run(
    p: &mut Pipeline,
    clock_hz: u32,
    seconds: f64,
    cs8: &mut Vec<u8>,
    sent_samples: &mut u64,
) -> Res<()> {
    let clock = [0.0f32; 480];
    let mut buf = vec![0u8; 16384];
    let want = (seconds * f64::from(clock_hz)).ceil() as u64;
    let mut done = 0u64;
    while done < want {
        let k = (want - done).min(clock.len() as u64) as usize;
        p.push("clock", InputData::F32(&clock[..k]))?;
        done += k as u64;
        loop {
            let n = p.read_raw("rf", &mut buf)?;
            if n == 0 {
                break;
            }
            cs8.extend_from_slice(&buf[..n]);
        }
    }
    *sent_samples += done;
    Ok(())
}

/// Runs `readings` through the transmit chain; returns the `cu8` bytes and the start time (s)
/// of every slot.
fn transmit(enc: &dyn SensorEncoder, readings: &[Reading]) -> Res<(Vec<u8>, Vec<f64>)> {
    let (flow, clock_hz) = tx_flow();
    let mut p = Registry::standard().build_json(&flow.to_string(), &BuildOptions::default())?;
    let mut cs8 = Vec::new();
    let mut starts = Vec::new();
    let mut sent = 0u64;
    run(&mut p, clock_hz, LEAD_S, &mut cs8, &mut sent)?;
    for (i, r) in readings.iter().enumerate() {
        starts.push(sent as f64 / f64::from(clock_hz));
        let mut text = String::new();
        let mut x = *r;
        x.time_s = None;
        x.to_json(&mut text)?;
        let pkt = Packet::new(i as u64, PacketFlags::NONE, text.as_bytes())?;
        p.push_packets("reading", &[pkt])?;
        let us = train_us(enc, r)?;
        let seconds = us as f64 * 1e-6 + 0.02 + SILENCE_S;
        run(&mut p, clock_hz, seconds, &mut cs8, &mut sent)?;
    }
    // let the filters drain
    run(&mut p, clock_hz, 0.05, &mut cs8, &mut sent)?;
    let cu8 = cs8.iter().map(|b| b ^ 0x80).collect();
    Ok((cu8, starts))
}

/// An FSK burst synthesised directly at 2 MS/s: a continuous-phase two-tone signal, the pulses
/// at the upper frequency and the gaps at the lower one, the carrier at `CARRIER_868` seen from
/// the centre `CENTER_868`. The modulator is not what is under test, the bits are; the chain
/// `pulse_key` -> `fsk_mod` -> `duc` keeps the carrier on between bursts (an idle FSK carrier
/// at the lower frequency), which no sensor does, so the burst is written here and the
/// samples between bursts are zero.
fn synth_fsk(pairs: &[Pair], phase: &mut f64, cu8: &mut Vec<u8>) {
    let carrier = CARRIER_868 as f64 - CENTER_868 as f64;
    let rate = f64::from(RF_RATE);
    let mut edge_us = 0u64;
    let mut done = 0u64;
    for p in pairs {
        for (width, freq) in [
            (p.pulse, carrier + FSK_DEVIATION_HZ),
            (p.gap, carrier - FSK_DEVIATION_HZ),
        ] {
            edge_us += u64::from(width);
            let until = edge_us * u64::from(RF_RATE) / 1_000_000;
            while done < until {
                *phase += 2.0 * std::f64::consts::PI * freq / rate;
                if *phase > std::f64::consts::PI {
                    *phase -= 2.0 * std::f64::consts::PI;
                }
                cu8.push((128.0 + 89.0 * phase.cos()).round() as u8);
                cu8.push((128.0 + 89.0 * phase.sin()).round() as u8);
                done += 1;
            }
        }
    }
}

/// An OOK burst synthesised directly: the carrier at +80 kHz from the centre during the pulses.
fn synth_ook(pairs: &[Pair], phase: &mut f64, cu8: &mut Vec<u8>) {
    let rate = f64::from(RF_RATE);
    let mut edge_us = 0u64;
    let mut done = 0u64;
    for p in pairs {
        for (width, on) in [(p.pulse, true), (p.gap, false)] {
            edge_us += u64::from(width);
            let until = edge_us * u64::from(RF_RATE) / 1_000_000;
            while done < until {
                *phase += 2.0 * std::f64::consts::PI * 80_000.0 / rate;
                if *phase > std::f64::consts::PI {
                    *phase -= 2.0 * std::f64::consts::PI;
                }
                let a = if on { 89.0 } else { 0.0 };
                cu8.push((128.0 + a * phase.cos()).round() as u8);
                cu8.push((128.0 + a * phase.sin()).round() as u8);
                done += 1;
            }
        }
    }
}

/// The mode "play": trains given as JSON (hand-made frames for experiments) to a cu8 file.
/// The JSON has a modulation (ook or fsk) and a list of trains of pulse and gap pairs in us.
fn play(trains: &Path, out: &Path) -> Res<()> {
    let v: Value = serde_json::from_slice(&std::fs::read(trains)?)?;
    let fsk = v["modulation"] == "fsk";
    let mut cu8 = Vec::new();
    let mut phase = 0.0f64;
    cu8.resize(2 * (LEAD_S * f64::from(RF_RATE)) as usize, 128);
    for t in v["trains"].as_array().ok_or("no trains")? {
        let mut pairs = Vec::new();
        for pg in t.as_array().ok_or("a train is a list")? {
            let pulse = pg[0].as_u64().ok_or("pulse")? as u32;
            let gap = pg[1].as_u64().ok_or("gap")? as u32;
            pairs.push(Pair { pulse, gap });
        }
        if fsk {
            synth_fsk(&pairs, &mut phase, &mut cu8);
        } else {
            synth_ook(&pairs, &mut phase, &mut cu8);
        }
        cu8.resize(
            cu8.len() + 2 * (SILENCE_S * f64::from(RF_RATE)) as usize,
            128,
        );
    }
    std::fs::write(out, &cu8)?;
    Ok(())
}

/// The FSK variant of `transmit`: silence, then every burst synthesised, then silence.
fn transmit_fsk(enc: &dyn SensorEncoder, readings: &[Reading]) -> Res<(Vec<u8>, Vec<f64>)> {
    let silence = |seconds: f64, cu8: &mut Vec<u8>| {
        cu8.resize(cu8.len() + 2 * (seconds * f64::from(RF_RATE)) as usize, 128);
    };
    let mut cu8 = Vec::new();
    let mut starts = Vec::new();
    let mut phase = 0.0f64;
    silence(LEAD_S, &mut cu8);
    for r in readings {
        starts.push(cu8.len() as f64 / 2.0 / f64::from(RF_RATE));
        let mut storage = vec![Pair::default(); MAX_TRAIN_PAIRS];
        let mut sink = TrainSink::new(&mut storage);
        enc.encode(r, enc.default_repeats(), &mut sink)
            .map_err(|e| format!("{}: {e}", r.model))?;
        synth_fsk(sink.pairs(), &mut phase, &mut cu8);
        silence(SILENCE_S, &mut cu8);
    }
    silence(0.05, &mut cu8);
    Ok((cu8, starts))
}

/// Adds Gaussian noise of `sigma` LSB to every byte of the cu8 samples (the sum of twelve
/// uniform values has the variance 1).
fn add_noise(cu8: &mut [u8], sigma: f64, seed: u64) {
    if sigma <= 0.0 {
        return;
    }
    let mut rng = Rng::new(seed ^ 0x5EED);
    for b in cu8.iter_mut() {
        let mut g = -6.0;
        for _ in 0..12 {
            g += (rng.below(1 << 24) as f64) / f64::from(1u32 << 24);
        }
        *b = (f64::from(*b) + sigma * g).round().clamp(0.0, 255.0) as u8;
    }
}

fn generate(dir: &Path, count: usize, seed: u64, noise: f64, only: Option<&str>) -> Res<()> {
    std::fs::create_dir_all(dir)?;
    for enc in ENCODERS {
        if only.is_some_and(|o| o != enc.name()) {
            continue;
        }
        let mut rng = Rng::new(seed);
        let models = enc.models();
        let mut readings = Vec::new();
        let mut i = 0;
        while readings.len() < count && i < count * 20 {
            let model = models[i % models.len()];
            i += 1;
            let Some(r) = random_reading(&mut rng, *enc, model) else {
                continue;
            };
            // the registry may refuse a value (for example a frame no burst shows)
            if encodes(*enc, &r) {
                readings.push(r);
            }
        }
        if readings.is_empty() {
            eprintln!("{}: no reading could be encoded, skipped", enc.name());
            continue;
        }
        let fsk = enc.modulation() == Modulation::Fsk;
        let (mut cu8, starts) = if fsk {
            transmit_fsk(*enc, &readings)?
        } else {
            transmit(*enc, &readings)?
        };
        add_noise(&mut cu8, noise, seed);
        let file = format!("{}.cu8", enc.name());
        std::fs::write(dir.join(&file), &cu8)?;
        let mut items = Vec::new();
        for (r, t) in readings.iter().zip(&starts) {
            items.push(json!({ "t_s": t, "reading": reading_json(r)? }));
        }
        let side = json!({
            "device": enc.name(),
            "modulation": if fsk { "fsk" } else { "ook" },
            "file": file,
            "rate_hz": RF_RATE,
            "center_hz": if fsk { CENTER_868 } else { CENTER_433 },
            "carrier_hz": if fsk { CARRIER_868 } else { CENTER_433 + 80_000 },
            "seed": seed,
            "noise_lsb": noise,
            "repeats": enc.default_repeats(),
            "silence_s": SILENCE_S,
            "readings": items,
        });
        std::fs::write(
            dir.join(format!("{}.json", enc.name())),
            serde_json::to_string_pretty(&side)? + "\n",
        )?;
        println!(
            "{}: {} readings, {} samples -> {}",
            enc.name(),
            readings.len(),
            cu8.len() / 2,
            dir.join(&file).display()
        );
    }
    Ok(())
}

/// The receiver flow of a band: the flow `ism433_sensors` or `ism868_pulses` plus
/// `sensor_decode`; with `pulses` the output is the pulse trains of the detector instead.
fn rx_flow(band: u32, pulses: bool) -> Res<Value> {
    let (channels, center) = match band {
        433 => (16, CENTER_433),
        868 => (8, CENTER_868),
        _ => return Err("the band is 433 or 868".into()),
    };
    if pulses {
        return Ok(json!({
            "sdrk": 1, "name": "pulses-rx", "settings": { "precision": "f32", "max_chunk": 4096 },
            "inputs": { "rf": { "format": "cu8", "rate_hz": RF_RATE } },
            "nodes": {
                "chan": { "kind": "channelize", "params": { "channels": channels, "taps_per_branch": 8 } },
                "det": { "kind": "band_pulse_detect", "params": { "center_hz": center } }
            },
            "edges": [
                { "from": "rf", "to": "chan" },
                { "from": "chan", "to": "det" },
                { "from": "det", "to": "readings" }
            ],
            "outputs": { "readings": { "type": "packets", "queue": 4096 } }
        }));
    }
    Ok(json!({
        "sdrk": 1, "name": "sensors-rx", "settings": { "precision": "f32", "max_chunk": 4096 },
        "inputs": { "rf": { "format": "cu8", "rate_hz": RF_RATE } },
        "nodes": {
            "chan": { "kind": "channelize", "params": { "channels": channels, "taps_per_branch": 8 } },
            "det": { "kind": "band_pulse_detect", "params": { "center_hz": center } },
            "dec": { "kind": "sensor_decode", "params": { "center_hz": center } }
        },
        "edges": [
            { "from": "rf", "to": "chan" },
            { "from": "chan", "to": "det" },
            { "from": "det", "to": "dec" },
            { "from": "dec", "to": "readings" }
        ],
        "outputs": { "readings": { "type": "packets", "queue": 4096 } }
    }))
}

/// Applies "--set node.param=value" options (the value is JSON, or a plain string) to a flow.
fn apply_sets(flow: &mut Value, args: &[String]) {
    for (i, a) in args.iter().enumerate() {
        if a != "--set" {
            continue;
        }
        let Some((key, val)) = args.get(i + 1).and_then(|s| s.split_once('=')) else {
            continue;
        };
        let Some((node, param)) = key.split_once('.') else {
            continue;
        };
        let v = serde_json::from_str(val).unwrap_or_else(|_| json!(val));
        if let Some(params) = flow["nodes"][node]["params"].as_object_mut() {
            params.insert(param.to_string(), v);
        }
    }
}

fn decode(path: &Path, band: u32, pulses: bool, args: &[String]) -> Res<()> {
    let mut bytes = std::fs::read(path)?;
    // two seconds of silence so that the last burst ends
    bytes.resize(bytes.len() + 2 * RF_RATE as usize * 2, 128);
    let mut flow = rx_flow(band, pulses)?;
    apply_sets(&mut flow, args);
    let mut p = Registry::standard().build_json(&flow.to_string(), &BuildOptions::default())?;
    let mut packets = vec![Packet::default(); 64];
    let mut at = 0u64;
    for piece in bytes.chunks(8192) {
        let meta = BlockMeta {
            start: SampleIndex(at),
            ..BlockMeta::default()
        };
        p.push_with_meta("rf", InputData::Raw(piece), &meta)?;
        at += (piece.len() / 2) as u64;
        loop {
            let k = p.read_packets("readings", &mut packets)?;
            if k == 0 {
                break;
            }
            for q in &packets[..k] {
                if pulses {
                    // one pulse packet per line (long trains span several packets)
                    let train = sdrk_proto::pulses::decode(q.data())?;
                    let mut line = String::new();
                    sdrk_proto::pulses::write_json(&mut line, &train.header(), train.pairs(), 0)?;
                    println!("{} {line}", q.index);
                    continue;
                }
                let mut v: Value = serde_json::from_slice(q.data())?;
                if let Some(o) = v.as_object_mut() {
                    o.insert("packet_index".into(), json!(q.index));
                }
                println!("{v}");
            }
        }
    }
    Ok(())
}

fn usage() -> String {
    let mut s = String::from(
        "usage:\n  sensor_iq generate DIR [--count N] [--seed S] [--noise LSB] [--only DEVICE]\n  sensor_iq decode FILE.cu8 --band 433|868 [--pulses] [--set node.param=value ...]\n  (nodes of the receiver: chan, det, dec)\n  sensor_iq play TRAINS.json OUT.cu8   (hand-made pulse trains, for experiments)\ndevices:",
    );
    for e in ENCODERS {
        let _ = write!(s, " {}", e.name());
    }
    s
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    match args.first().map(String::as_str) {
        Some("generate") => {
            let dir = PathBuf::from(args.get(1).ok_or_else(usage)?);
            let count = opt("--count").map_or(Ok(5), |s| s.parse())?;
            let seed = opt("--seed").map_or(Ok(1), |s| s.parse())?;
            let noise = opt("--noise").map_or(Ok(0.0), |s| s.parse())?;
            generate(&dir, count, seed, noise, opt("--only").as_deref())
        }
        Some("play") => {
            let trains = PathBuf::from(args.get(1).ok_or_else(usage)?);
            let out = PathBuf::from(args.get(2).ok_or_else(usage)?);
            play(&trains, &out)
        }
        Some("decode") => {
            let file = PathBuf::from(args.get(1).ok_or_else(usage)?);
            let band = opt("--band").map_or(Ok(433), |s| s.parse())?;
            decode(&file, band, args.iter().any(|a| a == "--pulses"), &args)
        }
        _ => Err(usage().into()),
    }
}
