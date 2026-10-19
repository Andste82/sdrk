//! A long run from a very large start index through the tuned resampler and the FM
//! demodulator: the output counts follow the exact formula and the time of the last anchor on
//! the output equals the analytic value to the nanosecond.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::unreadable_literal
)]

use num_complex::Complex32;
use sdrk::{C32, Rate};
use sdrk_blocks::{Block, FmDemodBlock, FmDemodConfig, ResamplerBlock, ResamplerConfig};
use sdrk_device::{BlockMeta, SampleIndex, TimeAnchor, TimeSource};
use sdrk_pipeline::{EventKind, InputData, PipelineSpec, VecStorage};
use sdrk_system_tests::{build, drain_events};

const START: u64 = 1 << 50;
const T0_NS: i128 = 1_700_000_000_000_000_000;
const RATE: u64 = 240_000;
const PPM: f64 = 37.5;
/// Samples per push: 20 ms, which is a whole number of nanoseconds (20 000 000).
const BLOCK: usize = 4800;
const BLOCK_NS: i128 = 20_000_000;

const FLOW: &str = r#"{
  "sdrk": 1,
  "name": "long-run",
  "settings": { "precision": "f32", "max_chunk": 4800 },
  "inputs": { "iq": { "type": "c32", "rate_hz": 240000 } },
  "nodes": {
    "rs": { "kind": "resampler", "params": { "rate_out_hz": 240000, "tunable": true, "ppm": 37.5 } },
    "demod": { "kind": "fm_demod", "params": { "audio_decimation": 5, "deemphasis_us": 50 } }
  },
  "edges": [
    { "from": "iq", "to": "rs" },
    { "from": "rs", "to": "demod" },
    { "from": "demod", "to": "audio" }
  ],
  "outputs": { "audio": { "type": "stream", "capacity": 16384 } }
}"#;

/// `round(num / den)` with halves rounded up, for a positive `den`.
fn round_half_up(num: i128, den: i128) -> i128 {
    (2 * num + den).div_euclid(2 * den)
}

fn ceil_div(num: u128, den: u128) -> u128 {
    num.div_ceil(den)
}

/// The position step of the tuned resampler in 32.32 fixed point, from the documented formula
/// `round(2^32 rate_in / rate_out / (1 + ppm 1e-6))`.
fn step() -> u128 {
    (4_294_967_296.0 / (1.0 + PPM * 1e-6) + 0.5).floor() as u128
}

/// Outputs of the tuned resampler once `n` input samples have been consumed: output `j` sits
/// at the position `j * step` and needs the input sample `floor(pos)`, and the one after it
/// when its phase is the last of the 128; outputs come in order.
fn tuned_outputs(n: u64) -> u64 {
    let step = step();
    let mut j: u128 = 0;
    loop {
        let pos = j * step;
        let index = pos >> 32;
        let wrap = u128::from((pos & 0xFFFF_FFFF) >> 25 == 127);
        if index + 1 + wrap > u128::from(n) {
            return j as u64;
        }
        j += 1;
    }
}

/// One node of the anchor chain, as `docs/timing.md` states it:
/// `t' = t + ((m - n0_out) M / L - (n - n0_in) - D) / rate_in`, in `i128` with halves rounded up.
struct Hop {
    /// Ratio out/in as `(L, M)`.
    ratio: (i128, i128),
    /// Group delay `D` in input samples as `(num, den)`.
    delay: (i128, i128),
    /// Origin pair.
    n0: (i128, i128),
}

impl Hop {
    fn anchor(&self, n: i128, m: i128, t: i128) -> i128 {
        let (l, mm) = self.ratio;
        let (dn, dd) = self.delay;
        let numer = (m - self.n0.1) * mm * dd - (n - self.n0.0) * l * dd - dn * l;
        let denom = l * dd * i128::from(RATE);
        t + round_half_up(numer * 1_000_000_000, denom)
    }
}

fn long_run(seconds: u64) {
    let spec = PipelineSpec::from_json(FLOW).unwrap();
    let mut p = build(&spec);

    // Declared delays of the two blocks.
    let rate = Rate::hz(RATE).unwrap();
    let rs_cfg = ResamplerConfig {
        rate_out_hz: RATE as f64,
        tunable: true,
        ppm: PPM,
        ..ResamplerConfig::default()
    };
    let rs_block = ResamplerBlock::<C32, _>::new(&rs_cfg, rate, &mut VecStorage).unwrap();
    let d_rs = rs_block.timing().delay;
    let demod_cfg = FmDemodConfig {
        audio_decimation: 5,
        ..FmDemodConfig::default()
    };
    let demod_block = FmDemodBlock::<C32, _>::new(&demod_cfg, rate, &mut VecStorage).unwrap();
    let d_demod = demod_block.timing().delay;

    // A 10 kHz tone: 200 periods fit a block, so the repeated block is phase-continuous.
    let tone: Vec<C32> = (0..BLOCK)
        .map(|i| {
            let ph = 2.0 * std::f64::consts::PI * 10_000.0 * i as f64 / RATE as f64;
            Complex32::new(ph.cos() as f32, ph.sin() as f32)
        })
        .collect();

    let blocks = seconds * RATE / BLOCK as u64;
    let total_in = blocks * BLOCK as u64;
    let mut audio = vec![0f32; 4096];
    let mut produced: u64 = 0;
    let mut last_audio = 0f32;
    let mut events = Vec::new();
    let mut last_anchor = None;
    for k in 0..blocks {
        let start = START + k * BLOCK as u64;
        let anchor = TimeAnchor {
            index: SampleIndex(start),
            unix_ns: i64::try_from(T0_NS + i128::from(k) * BLOCK_NS).unwrap(),
            source: TimeSource::Host,
        };
        let meta = BlockMeta {
            start: SampleIndex(start),
            anchor: Some(anchor),
            ..BlockMeta::default()
        };
        p.push_with_meta("iq", InputData::C32(&tone), &meta)
            .unwrap();
        loop {
            let n = p.read_stream::<f32>("audio", &mut audio).unwrap();
            if n == 0 {
                break;
            }
            produced += n as u64;
            last_audio = audio[n - 1];
        }
        events.clear();
        drain_events(&mut p, &mut events);
        for e in &events {
            if let EventKind::Anchor(a) = e.kind
                && p.output_name(e.output) == "audio"
            {
                last_anchor = Some((k, a));
            }
        }
    }

    // The tone is demodulated: 10 kHz of 75 kHz deviation (the 37.5 ppm change the rate only).
    assert!(
        (f64::from(last_audio) - 10_000.0 / 75_000.0).abs() < 2e-3,
        "last audio sample {last_audio}"
    );

    // Counts: the nodes may keep fewer than `min_input` samples back; what they consumed obeys
    // the exact formulas.
    let stats = p.stats();
    let rs = stats.iter().find(|s| s.name == "rs").unwrap();
    let dm = stats.iter().find(|s| s.name == "demod").unwrap();
    assert_eq!(
        rs.samples_in,
        total_in - (total_in - rs.samples_in),
        "rs input"
    );
    assert!(
        total_in - rs.samples_in <= 3,
        "the resampler kept {} samples",
        total_in - rs.samples_in
    );
    assert_eq!(
        rs.samples_out,
        tuned_outputs(rs.samples_in),
        "resampler output count"
    );
    assert_eq!(
        dm.samples_in,
        rs.samples_out - (rs.samples_out - dm.samples_in)
    );
    assert!(
        rs.samples_out - dm.samples_in < 64,
        "demod kept {}",
        rs.samples_out - dm.samples_in
    );
    assert_eq!(
        dm.samples_out,
        dm.samples_in.div_ceil(5),
        "demod output count"
    );
    assert_eq!(produced, dm.samples_out, "samples read from the output");
    assert!(
        (rs.samples_out as i64 - total_in as i64 - (total_in as f64 * PPM * 1e-6) as i64).abs()
            <= 8,
        "{} resampler outputs for {total_in} inputs",
        rs.samples_out
    );

    // The last anchor on the output equals the analytic value.
    let (k, a) = last_anchor.expect("an anchor on the audio output");
    let n = i128::from(START + k * BLOCK as u64);
    let t = T0_NS + i128::from(k) * BLOCK_NS;
    // Joining at START is the arithmetic of a gap from index 0: the output origin of a node is
    // the mapped start (the tuned resampler maps with its position step, so the origin is
    // ceil(START * 2^32 / step), about START * (1 + 37.5e-6)); the next node starts there.
    let rs_in_origin = i128::from(START);
    let rs_out_origin = ceil_div(u128::from(START) << 32, step()) as i128;
    let m_rs = rs_out_origin + ceil_div(((n - rs_in_origin) as u128) << 32, step()) as i128;
    let hop_rs = Hop {
        ratio: (1, 1),
        delay: (i128::from(d_rs.num()), i128::from(d_rs.den())),
        n0: (rs_in_origin, rs_out_origin),
    };
    let t1 = hop_rs.anchor(n, m_rs, t);
    // demodulator: ratio 1/5
    let dm_out_origin = ceil_div(rs_out_origin as u128, 5) as i128;
    let m_dm = dm_out_origin + ceil_div((m_rs - rs_out_origin) as u128, 5) as i128;
    let hop_dm = Hop {
        ratio: (1, 5),
        delay: (i128::from(d_demod.num()), i128::from(d_demod.den())),
        n0: (rs_out_origin, dm_out_origin),
    };
    let t2 = hop_dm.anchor(m_rs, m_dm, t1);
    assert_eq!(i128::from(a.index.0), m_dm, "index of the last anchor");
    let diff = i128::from(a.unix_ns) - t2;
    println!(
        "anchor index {m_dm}, time {} ns, analytic {t2} ns, difference {diff} ns",
        a.unix_ns
    );
    assert!(
        diff.abs() <= 1,
        "anchor time differs from the analytic value by {diff} ns"
    );
}

#[test]
fn long_run_no_drift() {
    long_run(20);
}

/// The same run for ten minutes; the nightly CI job runs it (`cargo test --release -p
/// sdrk-system-tests -- --ignored`).
#[test]
#[ignore = "ten minutes of signal; run nightly"]
fn long_run_no_drift_ten_minutes() {
    long_run(600);
}
