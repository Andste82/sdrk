//! `sdrk_proto::slicers` (M46 spec test 1, seed 636): every slicer on generated trains with
//! timing errors of up to 15 percent and glitches gives exact bits; errors of 25 percent give
//! error rows and never wrong bits.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use sdrk_proto::pulses::{self, Header, Modulation, Pair};
use sdrk_proto::slicers::{
    Convention, MAX_ROW_BITS, MAX_ROWS, Manchester, Nrz, Ppm, Pwm, RowEnd, Rows, Slicer, Tolerance,
};
use sdrk_testkit::rng::Pcg32;

const RATE: u32 = 250_000;

fn smp(us: f64) -> u32 {
    (us * f64::from(RATE) / 1.0e6).round() as u32
}

fn header(modulation: Modulation) -> Header {
    Header {
        modulation,
        channel: 3,
        freq_offset_hz: -12_000,
        level_db10: -250,
        noise_db10: -600,
        rate_hz: RATE,
    }
}

fn train_bytes(modulation: Modulation, pairs: &[Pair]) -> Vec<u8> {
    // `pulses::encode` stops at one packet (30 pairs); `decode` takes any length
    let mut buf = vec![0u8; pulses::HEADER_BYTES];
    pulses::encode(&header(modulation), &[], &mut buf).unwrap();
    for p in pairs {
        buf.extend_from_slice(&p.pulse.to_le_bytes());
        buf.extend_from_slice(&p.gap.to_le_bytes());
    }
    buf
}

fn uniform(rng: &mut Pcg32) -> f64 {
    f64::from(rng.below(2001)) / 1000.0 - 1.0
}

/// How the widths are disturbed.
#[derive(Clone, Copy)]
enum Mode {
    /// Every width by a uniform error within the fraction.
    Within(f64),
    /// Every width by exactly plus or minus the fraction.
    Edge(f64),
}

/// One generated message: the slicer, the true bits and one clean repetition as pairs (the
/// last gap is a placeholder).
struct Msg {
    slicer: Box<dyn Slicer>,
    bits: Vec<bool>,
    ideal: Vec<Pair>,
    /// The unit in samples (short pulse, short gap, half bit, bit).
    unit: f64,
    /// Whether the tolerance is absolute in units (NRZ) or relative to the width.
    absolute: bool,
    /// The gap between repetitions in samples (above every reset of the slicers).
    sep: u32,
    /// The nominal unit in microseconds.
    unit_us: f64,
}

fn random_bits(rng: &mut Pcg32, lo: u32, hi: u32) -> Vec<bool> {
    let n = lo + rng.below(hi - lo + 1);
    (0..n).map(|_| rng.below(2) == 1).collect()
}

fn pwm_msg(rng: &mut Pcg32) -> Msg {
    let bits = random_bits(rng, 8, 64);
    let mut ideal = Vec::new();
    for &b in &bits {
        ideal.push(Pair {
            pulse: smp(if b { 500.0 } else { 1500.0 }),
            gap: smp(1000.0),
        });
    }
    Msg {
        slicer: Box::new(Pwm::new(500.0, 1500.0, 1000.0, 4000.0, true).unwrap()),
        bits,
        ideal,
        unit: f64::from(smp(500.0)),
        absolute: false,
        sep: smp(9000.0),
        unit_us: 500.0,
    }
}

fn ppm_msg(rng: &mut Pcg32) -> Msg {
    let bits = random_bits(rng, 8, 64);
    let mut ideal = Vec::new();
    for &b in &bits {
        ideal.push(Pair {
            pulse: smp(500.0),
            gap: smp(if b { 2000.0 } else { 1000.0 }),
        });
    }
    // the end marker
    ideal.push(Pair {
        pulse: smp(500.0),
        gap: 0,
    });
    Msg {
        slicer: Box::new(Ppm::new(1000.0, 2000.0, 3000.0).unwrap()),
        bits,
        ideal,
        unit: f64::from(smp(1000.0)),
        absolute: false,
        sep: smp(4000.0),
        unit_us: 1000.0,
    }
}

fn manchester_msg(rng: &mut Pcg32) -> Msg {
    let bits = random_bits(rng, 8, 64);
    // IEEE: a 1 is high then low
    let mut halves: Vec<bool> = bits.iter().flat_map(|&b| [b, !b]).collect();
    let lead = !halves[0];
    if lead {
        halves.remove(0);
    }
    if !halves[halves.len() - 1] {
        halves.pop();
    }
    let half = smp(500.0);
    let mut ideal = Vec::new();
    let mut i = 0;
    while i < halves.len() {
        let mut pair = Pair { pulse: 0, gap: 0 };
        // a high run, then a low run
        let mut n = 0;
        while i < halves.len() && halves[i] {
            n += 1;
            i += 1;
        }
        pair.pulse = n * half;
        n = 0;
        while i < halves.len() && !halves[i] {
            n += 1;
            i += 1;
        }
        pair.gap = n * half;
        ideal.push(pair);
    }
    Msg {
        slicer: Box::new(
            Manchester::new(500.0, 4000.0, Convention::Ieee8023)
                .unwrap()
                .with_virtual_lead(lead),
        ),
        bits,
        ideal,
        unit: f64::from(half),
        absolute: false,
        sep: smp(9000.0),
        unit_us: 500.0,
    }
}

fn nrz_msg(rng: &mut Pcg32) -> Msg {
    let mut bits = random_bits(rng, 8, 60);
    let n = bits.len();
    bits[n - 1] = true;
    let bit = smp(100.0);
    let mut ideal = Vec::new();
    let mut i = 0;
    while i < n {
        let mut pair = Pair { pulse: 0, gap: 0 };
        let mut k = 0;
        while i < n && bits[i] {
            k += 1;
            i += 1;
        }
        pair.pulse = k * bit;
        k = 0;
        while i < n && !bits[i] {
            k += 1;
            i += 1;
        }
        pair.gap = k * bit;
        ideal.push(pair);
    }
    Msg {
        slicer: Box::new(Nrz::new(100.0, 20_000.0).unwrap()),
        bits,
        ideal,
        unit: f64::from(bit),
        absolute: true,
        sep: smp(30_000.0),
        unit_us: 100.0,
    }
}

fn disturb(rng: &mut Pcg32, m: &Msg, w: u32, mode: Mode) -> u32 {
    let e = match mode {
        Mode::Within(f) => f * uniform(rng),
        Mode::Edge(f) => {
            if rng.below(2) == 1 {
                f
            } else {
                -f
            }
        }
    };
    let v = if m.absolute {
        f64::from(w) + (e * m.unit).round()
    } else {
        (f64::from(w) * (1.0 + e)).round()
    };
    v.max(0.0) as u32
}

fn glitch(rng: &mut Pcg32, m: &Msg) -> Pair {
    let g = |rng: &mut Pcg32| {
        (m.unit * (0.3 + 0.4 * f64::from(rng.below(1001)) / 1000.0)).round() as u32
    };
    Pair {
        pulse: g(rng),
        gap: g(rng),
    }
}

struct Plan {
    reps: usize,
    mode: Mode,
    junk: bool,
    /// Insert a glitch pair into the first repetition before this pair.
    glitch_at: Option<usize>,
}

fn assemble(rng: &mut Pcg32, m: &Msg, plan: &Plan) -> Vec<Pair> {
    let mut out = Vec::new();
    if plan.junk {
        out.push(glitch(rng, m));
        out.push(Pair {
            pulse: glitch(rng, m).pulse,
            gap: m.sep,
        });
    }
    for rep in 0..plan.reps {
        for (k, p) in m.ideal.iter().enumerate() {
            if rep == 0 && plan.glitch_at == Some(k) {
                out.push(glitch(rng, m));
            }
            let pulse = disturb(rng, m, p.pulse, plan.mode);
            let gap = if k + 1 == m.ideal.len() {
                m.sep
            } else {
                disturb(rng, m, p.gap, plan.mode)
            };
            out.push(Pair { pulse, gap });
        }
    }
    if plan.junk {
        // the gap of the last real pair is the separator already
        out.push(glitch(rng, m));
        out.push(Pair {
            pulse: glitch(rng, m).pulse,
            gap: 0,
        });
    } else if let Some(last) = out.last_mut() {
        last.gap = 0;
    }
    out
}

type Gen = fn(&mut Pcg32) -> Msg;

const GENS: [(&str, Gen); 4] = [
    ("pwm", pwm_msg),
    ("ppm", ppm_msg),
    ("manchester", manchester_msg),
    ("nrz", nrz_msg),
];

fn slice(m: &Msg, pairs: &[Pair]) -> Rows {
    let bytes = train_bytes(Modulation::Ook, pairs);
    let train = pulses::decode(&bytes).unwrap();
    let mut rows = Rows::new();
    m.slicer.slice(&train, &mut rows);
    rows
}

#[test]
fn timing_errors_of_15_percent_and_glitches_give_exact_bits() {
    let mut rng = Pcg32::new(636, 1);
    for (name, make) in GENS {
        for case in 0..20 {
            let m = make(&mut rng);
            let plan = Plan {
                reps: 1 + rng.below(3) as usize,
                mode: if case % 2 == 0 {
                    Mode::Within(0.15)
                } else {
                    Mode::Edge(0.15)
                },
                junk: case % 4 < 2,
                glitch_at: None,
            };
            let pairs = assemble(&mut rng, &m, &plan);
            let rows = slice(&m, &pairs);
            let good: Vec<_> = rows.iter().filter(|r| !r.end().is_error()).collect();
            assert_eq!(good.len(), plan.reps, "{name} case {case}: good rows");
            for r in &good {
                let got: Vec<bool> = r.iter().collect();
                assert_eq!(got, m.bits, "{name} case {case}: bits");
                assert!(
                    r.worst_err() <= 0.2,
                    "{name} case {case}: {}",
                    r.worst_err()
                );
                let ratio = f64::from(r.unit_avg_us()) / m.unit_us;
                assert!(
                    (0.8..=1.2).contains(&ratio),
                    "{name} case {case}: unit {}",
                    r.unit_avg_us()
                );
            }
            // everything else is junk: error rows without a bit
            let junk_rows = rows.len() - good.len();
            assert_eq!(
                junk_rows,
                if plan.junk { 2 } else { 0 },
                "{name} case {case}"
            );
            for r in rows.iter().filter(|r| r.end().is_error()) {
                assert!(r.is_empty(), "{name} case {case}: junk row has bits");
            }
        }
    }
}

#[test]
fn a_glitch_inside_a_row_ends_it_with_an_error_and_the_next_repeat_is_exact() {
    let mut rng = Pcg32::new(636, 2);
    for (name, make) in GENS {
        for case in 0..20 {
            let m = make(&mut rng);
            let at = 1 + rng.below((m.ideal.len() - 1) as u32) as usize;
            let plan = Plan {
                reps: 2,
                mode: Mode::Within(0.15),
                junk: false,
                glitch_at: Some(at),
            };
            let pairs = assemble(&mut rng, &m, &plan);
            let rows = slice(&m, &pairs);
            assert_eq!(rows.len(), 2, "{name} case {case}");
            let first = rows.get(0).unwrap();
            assert!(first.end().is_error(), "{name} case {case}: first row");
            // never wrong bits: what was decoded is a prefix of the message
            let got: Vec<bool> = first.iter().collect();
            assert_eq!(got[..], m.bits[..got.len()], "{name} case {case}: prefix");
            let second = rows.get(1).unwrap();
            assert!(!second.end().is_error());
            assert_eq!(
                second.iter().collect::<Vec<_>>(),
                m.bits,
                "{name} case {case}"
            );
        }
    }
}

#[test]
fn errors_of_25_percent_give_error_rows_and_never_wrong_bits() {
    let mut rng = Pcg32::new(636, 3);
    for (name, make) in GENS {
        for case in 0..20 {
            let m = make(&mut rng);
            // every width off by exactly 25 percent: every row is an error row
            let plan = Plan {
                reps: 2,
                mode: Mode::Edge(0.25),
                junk: false,
                glitch_at: None,
            };
            let pairs = assemble(&mut rng, &m, &plan);
            let rows = slice(&m, &pairs);
            assert!(!rows.is_empty(), "{name} case {case}");
            for r in rows.iter() {
                assert!(r.end().is_error(), "{name} case {case}: {:?}", r.end());
                let got: Vec<bool> = r.iter().collect();
                assert_eq!(got[..], m.bits[..got.len().min(m.bits.len())]);
            }
            // errors up to 25 percent: a row without error has exact bits, an error row a prefix
            let plan = Plan {
                reps: 3,
                mode: Mode::Within(0.25),
                junk: false,
                glitch_at: None,
            };
            let pairs = assemble(&mut rng, &m, &plan);
            let rows = slice(&m, &pairs);
            for r in rows.iter() {
                let got: Vec<bool> = r.iter().collect();
                if r.end().is_error() {
                    assert!(got.len() <= m.bits.len());
                    assert_eq!(got[..], m.bits[..got.len()], "{name} case {case}: prefix");
                } else {
                    assert_eq!(got, m.bits, "{name} case {case}: wrong bits");
                }
            }
        }
    }
}

#[test]
fn manchester_both_conventions_and_the_virtual_lead() {
    // bits 1 0 1 1 0 in the IEEE convention: H L | L H | H L | H L | L H, the first
    // low half of a leading 0 would be silent; here the row starts with a 1
    let half = 125;
    let pairs = [
        Pair {
            pulse: half,
            gap: 2 * half,
        },
        Pair {
            pulse: 2 * half,
            gap: half,
        },
        Pair {
            pulse: half,
            gap: 2 * half,
        },
        Pair {
            pulse: half,
            gap: 0,
        },
    ];
    let bytes = train_bytes(Modulation::Ook, &pairs);
    let train = pulses::decode(&bytes).unwrap();
    let mut rows = Rows::new();
    Manchester::new(500.0, 4000.0, Convention::Ieee8023)
        .unwrap()
        .slice(&train, &mut rows);
    let r = rows.get(0).unwrap();
    assert_eq!(
        r.iter().collect::<Vec<_>>(),
        [true, false, true, true, false]
    );
    assert_eq!(r.end(), RowEnd::TrainEnd);
    Manchester::new(500.0, 4000.0, Convention::Thomas)
        .unwrap()
        .slice(&train, &mut rows);
    let r = rows.get(0).unwrap();
    assert_eq!(
        r.iter().collect::<Vec<_>>(),
        [false, true, false, false, true]
    );
    // with a virtual low lead the same train is the bits 0 (L H), then H L = 1, ...
    Manchester::new(500.0, 4000.0, Convention::Ieee8023)
        .unwrap()
        .with_virtual_lead(true)
        .slice(&train, &mut rows);
    let r = rows.get(0).unwrap();
    // L H | L L is a violation after the first bit
    assert_eq!(r.iter().collect::<Vec<_>>(), [false]);
    assert_eq!(r.end(), RowEnd::Violation);
}

#[test]
fn rows_split_at_the_reset_gap_and_overflows_are_reported() {
    let unit = smp(500.0);
    let reset = smp(9000.0);
    // 10 rows of one bit each: only MAX_ROWS are kept
    let mut pairs = Vec::new();
    for _ in 0..10 {
        pairs.push(Pair {
            pulse: unit,
            gap: reset,
        });
    }
    let bytes = train_bytes(Modulation::Ook, &pairs);
    let train = pulses::decode(&bytes).unwrap();
    let mut rows = Rows::new();
    Pwm::new(500.0, 1500.0, 1000.0, 4000.0, true)
        .unwrap()
        .slice(&train, &mut rows);
    assert_eq!(rows.len(), MAX_ROWS);
    assert_eq!(rows.dropped(), 2);
    assert_eq!(rows.get(0).unwrap().start_pair(), 0);
    assert_eq!(rows.get(5).unwrap().start_pair(), 5);

    // a row longer than the capacity ends with Full
    let mut pairs = Vec::new();
    for _ in 0..MAX_ROW_BITS + 5 {
        pairs.push(Pair {
            pulse: unit,
            gap: smp(1000.0),
        });
    }
    let bytes = train_bytes(Modulation::Ook, &pairs);
    let train = pulses::decode(&bytes).unwrap();
    Pwm::new(500.0, 1500.0, 1000.0, 4000.0, true)
        .unwrap()
        .slice(&train, &mut rows);
    let r = rows.get(0).unwrap();
    assert_eq!(r.len(), MAX_ROW_BITS);
    assert_eq!(r.end(), RowEnd::Full);
    assert_eq!(rows.len(), 1);
}

#[test]
fn nrz_starts_with_a_zero_when_the_first_pulse_is_empty_and_degenerate_input_is_safe() {
    let bit = smp(100.0);
    // 0 0 1 1 1 0 1: pulse 0, gap 2, pulse 3, gap 1, pulse 1
    let pairs = [
        Pair {
            pulse: 0,
            gap: 2 * bit,
        },
        Pair {
            pulse: 3 * bit,
            gap: bit,
        },
        Pair { pulse: bit, gap: 0 },
    ];
    let bytes = train_bytes(Modulation::Fsk, &pairs);
    let train = pulses::decode(&bytes).unwrap();
    let mut rows = Rows::new();
    Nrz::new(100.0, 5000.0).unwrap().slice(&train, &mut rows);
    let r = rows.get(0).unwrap();
    assert_eq!(r.field(0, 7), Some(0b001_1101));
    assert_eq!(r.bytes(), &[0b0011_1010]);

    // an empty train, a zero rate and huge widths give no panic and no bad bits
    let bytes = train_bytes(Modulation::Fsk, &[]);
    let train = pulses::decode(&bytes).unwrap();
    Nrz::new(100.0, 5000.0).unwrap().slice(&train, &mut rows);
    assert!(rows.is_empty());
    let bytes = train_bytes(
        Modulation::Fsk,
        &[
            Pair {
                pulse: u32::MAX,
                gap: u32::MAX,
            },
            Pair { pulse: 5, gap: 5 },
        ],
    );
    let train = pulses::decode(&bytes).unwrap();
    Nrz::new(100.0, 5000.0).unwrap().slice(&train, &mut rows);
    assert!(
        rows.iter()
            .all(|r| r.end().is_error() || r.is_empty() || r.len() <= MAX_ROW_BITS)
    );
    let mut zero = bytes.clone();
    zero[12..16].copy_from_slice(&0u32.to_le_bytes());
    let train = pulses::decode(&zero).unwrap();
    Nrz::new(100.0, 5000.0).unwrap().slice(&train, &mut rows);
    assert!(rows.is_empty());
}

#[test]
fn configurations_are_validated() {
    assert!(Pwm::new(1500.0, 500.0, 1000.0, 4000.0, true).is_err());
    assert!(Pwm::new(500.0, 1500.0, 1000.0, 900.0, true).is_err());
    assert!(Pwm::new(f32::NAN, 1500.0, 1000.0, 4000.0, true).is_err());
    // windows of 20 percent overlap
    assert!(Pwm::new(500.0, 650.0, 1000.0, 4000.0, true).is_err());
    assert!(Ppm::new(1000.0, 2000.0, 1500.0).is_err());
    assert!(Manchester::new(500.0, 1000.0, Convention::Thomas).is_err());
    assert!(Nrz::new(0.0, 1000.0).is_err());
    assert!(Nrz::new(100.0, 50.0).is_err());
    assert!(Tolerance::new(0.0, 2).is_err());
    assert!(Tolerance::new(0.1, 2).is_ok());
    let tol = Tolerance::default();
    assert_eq!(tol.rel(), 0.2);
    assert_eq!(tol.min_samples(), 2.0);
    // a tolerance of 10 percent turns a 15 percent error into an error row
    let tight = Pwm::new(500.0, 1500.0, 1000.0, 4000.0, true)
        .unwrap()
        .with_tolerance(Tolerance::new(0.1, 2).unwrap())
        .unwrap();
    let bytes = train_bytes(
        Modulation::Ook,
        &[
            Pair {
                pulse: smp(575.0),
                gap: smp(1000.0),
            },
            Pair { pulse: 125, gap: 0 },
        ],
    );
    let train = pulses::decode(&bytes).unwrap();
    let mut rows = Rows::new();
    tight.slice(&train, &mut rows);
    assert_eq!(rows.get(0).unwrap().end(), RowEnd::BadWidth);
}

fn assert_send<T: Send>() {}

#[test]
fn public_types_are_send() {
    assert_send::<Rows>();
    assert_send::<Pwm>();
    assert_send::<Ppm>();
    assert_send::<Manchester>();
    assert_send::<Nrz>();
}
