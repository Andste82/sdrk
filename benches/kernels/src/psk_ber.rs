//! The PSK receiver against GNU Radio's receiver chain on the same signals (issue #169, §D):
//! bit error rate and alignment changes of `psk_demod` and of `fir_filter_ccf` +
//! `symbol_sync_cc` + `costas_loop_cc` + the decision of `constellation_decoder_cb` (and
//! `diff_decoder_bb` for the differential receiver).
//!
//! The signal is the output of `psk_mod` for random bits (8 samples per symbol, roll-off 0.35,
//! span 10), turned by a phase and a carrier offset, delayed by whole samples, with white
//! Gaussian noise for a given Eb/N0 (the power of the samples measured, the noise per
//! component `sigma^2 = P sps / (2 k Eb/N0)`). Both receivers run the same loop constants (the
//! block defaults: `sync_bw` 0.01, `carrier_bw` 0.02, damping 0.707), in calls of 4096 samples.
//!
//! The bits of a receiver are compared with the sent ones per window of [`WINDOW`] symbols at
//! the best alignment (the phase ambiguity of the coherent carrier loop and the lag in symbols,
//! searched around the alignment of the window before): the BER is that of the best alignment
//! and an alignment change is a window whose best alignment differs from the one before it. The
//! first [`SKIP`] symbols (the acquisition) are not counted.

use cref_sys::GrCostasLoopCc;
use sdrk::C32;
use sdrk_blocks::{
    ArenaStorage, Block, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig,
    SymbolMapConstellation,
};

use super::Pcg32;
use super::psk_family::GrFront;
use super::qam_family::{BLOCK_SPS, block_rate, leaked_arena};

/// Symbols per window of the alignment.
pub const WINDOW: usize = 2000;
/// Symbols at the start that are not counted (the acquisition of the loops).
pub const SKIP: usize = 6000;
/// Input samples per call.
const CALL: usize = 4096;
const SYMBOL_RATE_HZ: f64 = 25_000.0;
const SAMPLE_RATE_HZ: u64 = 200_000;
const SYNC_BW: f32 = 0.01;
const CARRIER_BW: f64 = 0.02;
const DAMPING: f64 = 0.707;
/// The carrier offset in cycles per symbol and the start phase in radians of every signal.
const CARRIER_OFFSET: f64 = 0.003;
const START_PHASE: f64 = 0.3;
/// Whole samples the signal is delayed by (the timing loop finds the symbol phase).
const DELAY_SAMPLES: usize = 3;

/// A constellation of the comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Psk {
    Bpsk,
    Qpsk,
    Psk8,
}

impl Psk {
    pub fn name(self) -> &'static str {
        match self {
            Psk::Bpsk => "BPSK",
            Psk::Qpsk => "QPSK",
            Psk::Psk8 => "8PSK",
        }
    }

    /// Bits per symbol, also the order of the carrier loop's detector.
    pub fn bits(self) -> usize {
        match self {
            Psk::Bpsk => 1,
            Psk::Qpsk => 2,
            Psk::Psk8 => 3,
        }
    }

    fn order(self) -> usize {
        1 << self.bits()
    }

    fn constellation(self) -> SymbolMapConstellation {
        match self {
            Psk::Bpsk => SymbolMapConstellation::Bpsk,
            Psk::Qpsk => SymbolMapConstellation::Qpsk,
            Psk::Psk8 => SymbolMapConstellation::Psk8,
        }
    }

    /// The index (in units of 22.5 degrees) of the point of `bits` (the Gray mapping of M40 and
    /// M40.1: 1 is positive).
    fn index(self, b: &[bool]) -> u8 {
        match self {
            Psk::Bpsk => {
                if b[0] {
                    0
                } else {
                    8
                }
            }
            Psk::Qpsk => match (b[0], b[1]) {
                (true, true) => 2,
                (false, true) => 6,
                (false, false) => 10,
                (true, false) => 14,
            },
            Psk::Psk8 => {
                let g = b.iter().fold(0usize, |a, &v| a * 2 + usize::from(v));
                let k = match g {
                    0b000 => 0,
                    0b001 => 1,
                    0b011 => 2,
                    0b010 => 3,
                    0b110 => 4,
                    0b111 => 5,
                    0b101 => 6,
                    _ => 7,
                };
                2 * k + 1
            }
        }
    }

    /// The bits of the point nearest to the index `k` (units of 22.5 degrees).
    fn bits_of(self, k: u8) -> u8 {
        let n = self.bits();
        let mut best = (u8::MAX, 0u8);
        for v in 0..(1u8 << n) {
            let b: Vec<bool> = (0..n).map(|j| (v >> (n - 1 - j)) & 1 == 1).collect();
            let p = self.index(&b);
            let d = ((i16::from(p) - i16::from(k)).rem_euclid(16)) as u8;
            let d = d.min(16 - d);
            if d < best.0 {
                best = (d, v);
            }
        }
        best.1
    }
}

/// One case: the constellation, coherent or differential (only BPSK has a differential
/// signal that GNU Radio's chain decodes: the other differential modes of `psk_mod` step the
/// phase by odd multiples of 45 degrees), the Eb/N0.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub psk: Psk,
    pub differential: bool,
    pub ebn0_db: f64,
}

/// The result of one receiver on one case.
#[derive(Clone, Copy, Debug, Default)]
pub struct Result {
    pub ber: f64,
    pub errors: usize,
    pub bits: usize,
    /// Windows whose alignment differs from the one before.
    pub changes: usize,
    pub windows: usize,
}

fn gauss(rng: &mut Pcg32) -> f64 {
    loop {
        let u = f64::from(rng.next_u32()) / 4_294_967_296.0;
        let v = f64::from(rng.next_u32()) / 4_294_967_296.0;
        if u > 1e-12 {
            return (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos();
        }
    }
}

/// The signal of `case` for `n_sym` symbols: the sent bits (one per element) and the samples.
fn signal(case: Case, n_sym: usize, seed: u64) -> (Vec<bool>, Vec<C32>) {
    let k = case.psk.bits();
    let mut rng = Pcg32::new(seed);
    let bits: Vec<bool> = (0..k * n_sym).map(|_| rng.next_u32() & 1 == 1).collect();
    let soft: Vec<f32> = bits.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
    let cfg = PskModConfig {
        constellation: case.psk.constellation(),
        samples_per_symbol: BLOCK_SPS,
        differential: case.differential,
        ..PskModConfig::default()
    };
    let rate = block_rate(SAMPLE_RATE_HZ / 4);
    let bytes = PskModBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    let mut tx = PskModBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("psk_mod");
    let mut sig = vec![C32::default(); n_sym * BLOCK_SPS];
    let _ = tx.process(&soft, &mut sig);
    let power = sig
        .iter()
        .map(|x| f64::from(x.re).powi(2) + f64::from(x.im).powi(2))
        .sum::<f64>()
        / sig.len() as f64;
    #[allow(clippy::cast_precision_loss, reason = "small integers")]
    let sigma =
        (power * BLOCK_SPS as f64 / (2.0 * k as f64 * 10f64.powf(case.ebn0_db / 10.0))).sqrt();
    let w = 2.0 * std::f64::consts::PI * CARRIER_OFFSET / BLOCK_SPS as f64;
    let mut out = vec![C32::default(); DELAY_SAMPLES];
    out.extend(sig.iter().enumerate().map(|(n, x)| {
        let a = START_PHASE + w * n as f64;
        let (s, c) = a.sin_cos();
        let (re, im) = (f64::from(x.re), f64::from(x.im));
        C32::new(
            (re * c - im * s + sigma * gauss(&mut rng)) as f32,
            (re * s + im * c + sigma * gauss(&mut rng)) as f32,
        )
    }));
    (bits, out)
}

/// A loop bandwidth of `psk_demod` from the environment (a knob of the experiments of D2551;
/// the report runs the defaults of the block).
fn knob(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn demod_config(case: Case) -> PskDemodConfig {
    PskDemodConfig {
        constellation: case.psk.constellation(),
        symbol_rate_hz: SYMBOL_RATE_HZ,
        differential: case.differential,
        sync_bw: knob("PSK_BER_SYNC_BW", f64::from(SYNC_BW)),
        carrier_bw: knob("PSK_BER_CARRIER_BW", CARRIER_BW),
        ..PskDemodConfig::default()
    }
}

/// The data symbols `psk_demod` returns, as indices (units of 22.5 degrees) of their points.
fn run_ours(case: Case, input: &[C32]) -> Vec<u8> {
    let cfg = demod_config(case);
    let rate = block_rate(SAMPLE_RATE_HZ);
    let bytes = PskDemodBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    let mut rx = PskDemodBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("psk_demod");
    let k = case.psk.bits();
    let mut soft = vec![0.0f32; k * (CALL / BLOCK_SPS) + 64];
    let mut idx = Vec::new();
    for chunk in input.chunks(CALL) {
        let n = rx.process(chunk, &mut soft).map_or(0, |p| p.produced);
        for s in soft[..n].chunks_exact(k) {
            let b: Vec<bool> = s.iter().map(|&v| v > 0.0).collect();
            idx.push(case.psk.index(&b));
        }
    }
    idx
}

/// GNU Radio's chain; the indices are the decisions of the points after the carrier loop, the
/// differential decoder (`diff_decoder_bb`: the difference of two decisions) when asked.
fn run_gr(case: Case, input: &[C32]) -> Option<Vec<u8>> {
    let mut front = GrFront::new(CALL, SYNC_BW)?;
    let bw = (CARRIER_BW / (DAMPING + 1.0 / (4.0 * DAMPING))) as f32;
    let mut costas = GrCostasLoopCc::new(bw, u32::try_from(case.psk.order()).ok()?)?;
    let mut locked = vec![[0.0f32; 2]; CALL / BLOCK_SPS + 64];
    let mut idx: Vec<u8> = Vec::new();
    let mut prev = 0u8;
    for chunk in input.chunks(CALL) {
        let pairs: Vec<[f32; 2]> = chunk.iter().map(|c| [c.re, c.im]).collect();
        let n = front.process(&pairs);
        costas.work(&front.symbols[..n], &mut locked[..n]);
        for p in &locked[..n] {
            let a = f64::from(p[1]).atan2(f64::from(p[0])) / (std::f64::consts::PI / 8.0);
            let k = nearest(case.psk, a);
            if case.differential {
                // diff_decoder: the difference of the two decisions of the modulus.
                let m = case.psk.order() as i16;
                let step = 16 / m;
                let d = ((i16::from(k) - i16::from(prev)).rem_euclid(16)) / step % m;
                prev = k;
                idx.push((d * step) as u8);
            } else {
                idx.push(k);
            }
        }
    }
    Some(idx)
}

/// The index of the point of the constellation nearest to the angle `a` (units of 22.5
/// degrees).
fn nearest(psk: Psk, a: f64) -> u8 {
    let (first, step) = match psk {
        Psk::Bpsk => (0.0, 8.0),
        Psk::Qpsk => (2.0, 4.0),
        Psk::Psk8 => (1.0, 2.0),
    };
    let k = ((a - first) / step).round() * step + first;
    (k.rem_euclid(16.0)) as u8 % 16
}

/// The BER and the alignment changes of the received indices `got` against the sent bits.
fn evaluate(case: Case, bits: &[bool], got: &[u8]) -> Result {
    let k = case.psk.bits();
    let n_sym = bits.len() / k;
    // The rotations the carrier loop may lock at: multiples of the point spacing (none for the
    // differential receivers).
    let rotations: Vec<u8> = if case.differential {
        vec![0]
    } else {
        (0..case.psk.order())
            .map(|r| (r * 16 / case.psk.order()) as u8)
            .collect()
    };
    let errors_of = |from: usize, lag: isize, rot: u8| -> usize {
        let mut e = 0;
        for j in from..from + WINDOW {
            let jj = j as isize + lag;
            if jj < 0 || jj as usize >= got.len() {
                e += k;
                continue;
            }
            let g = (got[jj as usize] + 16 - rot) % 16;
            let b = case.psk.bits_of(g);
            for t in 0..k {
                let rx = (b >> (k - 1 - t)) & 1 == 1;
                if rx != bits[j * k + t] {
                    e += 1;
                }
            }
        }
        e
    };
    let mut res = Result::default();
    let mut prev: Option<(isize, u8)> = None;
    let mut from = SKIP;
    while from + WINDOW <= n_sym {
        let range: Vec<isize> = match prev {
            Some((l, _)) => (l - 3..=l + 3).collect(),
            None => (0..64).collect(),
        };
        let mut best: Option<(usize, isize, u8)> = None;
        for &lag in &range {
            for &rot in &rotations {
                let e = errors_of(from, lag, rot);
                let tie_prev = prev == Some((lag, rot));
                if best.is_none_or(|(be, bl, br)| {
                    e < be || (e == be && tie_prev && prev != Some((bl, br)))
                }) {
                    best = Some((e, lag, rot));
                }
            }
        }
        let (e, lag, rot) = best.expect("a candidate");
        res.errors += e;
        res.bits += WINDOW * k;
        res.windows += 1;
        if prev.is_some_and(|p| p != (lag, rot)) {
            res.changes += 1;
        }
        prev = Some((lag, rot));
        from += WINDOW;
    }
    #[allow(clippy::cast_precision_loss, reason = "counts")]
    {
        res.ber = res.errors as f64 / res.bits.max(1) as f64;
    }
    res
}

/// Both receivers on `case` for `n_sym` symbols of the seed.
pub fn compare(case: Case, n_sym: usize, seed: u64) -> Pair {
    let (bits, input) = signal(case, n_sym, seed);
    let ours = evaluate(case, &bits, &run_ours(case, &input));
    let gr = run_gr(case, &input).map(|g| evaluate(case, &bits, &g));
    (ours, gr)
}

/// The theoretical BER of coherent Gray PSK at `ebn0_db` (BPSK and QPSK `Q(sqrt(2 Eb/N0))`,
/// 8PSK the approximation `2 Q(sqrt(6 Eb/N0) sin(pi/8)) / 3`), DBPSK `exp(-Eb/N0) / 2`.
pub fn theory(case: Case) -> f64 {
    let e = 10f64.powf(case.ebn0_db / 10.0);
    let q = |x: f64| 0.5 * erfc(x / 2f64.sqrt());
    match (case.psk, case.differential) {
        (Psk::Bpsk, true) => 0.5 * (-e).exp(),
        (Psk::Bpsk | Psk::Qpsk, _) => q((2.0 * e).sqrt()),
        (Psk::Psk8, _) => 2.0 / 3.0 * q((6.0 * e).sqrt() * (std::f64::consts::PI / 8.0).sin()),
    }
}

/// `erfc` by the continued fraction of Abramowitz and Stegun 7.1.26 (1.5e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let y = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))))
        * (-z * z).exp();
    if x >= 0.0 { y } else { 2.0 - y }
}

/// Both receivers on one case and seed.
pub type Pair = (Result, Option<Result>);

/// Sums of both receivers over several seeds.
#[derive(Clone, Copy, Debug, Default)]
struct Sum {
    errors: usize,
    bits: usize,
    changes: usize,
    windows: usize,
}

impl Sum {
    fn add(&mut self, r: Result) {
        self.errors += r.errors;
        self.bits += r.bits;
        self.changes += r.changes;
        self.windows += r.windows;
    }

    #[allow(clippy::cast_precision_loss, reason = "counts")]
    fn ber(&self) -> f64 {
        self.errors as f64 / self.bits.max(1) as f64
    }

    /// Alignment changes per 1e5 symbols.
    #[allow(clippy::cast_precision_loss, reason = "counts")]
    fn changes_per_1e5(&self) -> f64 {
        self.changes as f64 / (self.windows.max(1) * WINDOW) as f64 * 1e5
    }
}

/// The table of the report: every case with both receivers, the sums over `seeds` (run in
/// parallel threads), `n_sym` symbols each.
pub fn report(n_sym: usize, seeds: &[u64]) -> String {
    let mut s = String::from(
        "| constellation | mode | Eb/N0 dB | theory | sdrk BER | GNU Radio BER | sdrk / GNU Radio | sdrk changes per 1e5 | GNU Radio changes per 1e5 | bits |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    let cases: Vec<Case> = [
        (Psk::Bpsk, false, [2.0, 4.0, 6.0, 8.0]),
        (Psk::Qpsk, false, [2.0, 4.0, 6.0, 8.0]),
        (Psk::Psk8, false, [6.0, 8.0, 10.0, 12.0]),
        (Psk::Bpsk, true, [2.0, 4.0, 6.0, 8.0]),
    ]
    .iter()
    .flat_map(|&(psk, differential, ebs)| {
        ebs.map(|ebn0_db| Case {
            psk,
            differential,
            ebn0_db,
        })
    })
    .collect();
    let jobs: Vec<(usize, u64)> = (0..cases.len())
        .flat_map(|c| seeds.iter().map(move |&sd| (c, sd)))
        .collect();
    let mut results: Vec<Option<Pair>> = vec![None; jobs.len()];
    let workers = std::thread::available_parallelism().map_or(4, usize::from);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<&mut Option<Pair>>> =
        results.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| {
                loop {
                    let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(c, sd)) = jobs.get(j) else { break };
                    let r = compare(cases[c], n_sym, sd);
                    **slots[j].lock().expect("slot") = Some(r);
                }
            });
        }
    });
    for (ci, c) in cases.iter().enumerate() {
        let (mut o, mut g) = (Sum::default(), Sum::default());
        let mut gr_ok = true;
        for (j, &(cj, _)) in jobs.iter().enumerate() {
            if cj != ci {
                continue;
            }
            let (ro, rg) = results[j].expect("result");
            o.add(ro);
            match rg {
                Some(rg) => g.add(rg),
                None => gr_ok = false,
            }
        }
        s.push_str(&format!(
            "| {} | {} | {} | {:.3e} | {:.3e} | {} | {} | {:.2} | {} | {} |\n",
            c.psk.name(),
            if c.differential {
                "differential"
            } else {
                "coherent"
            },
            c.ebn0_db,
            theory(*c),
            o.ber(),
            if gr_ok {
                format!("{:.3e}", g.ber())
            } else {
                "n/a".into()
            },
            if gr_ok {
                format!("{:.2}", o.ber() / g.ber().max(1e-12))
            } else {
                "n/a".into()
            },
            o.changes_per_1e5(),
            if gr_ok {
                format!("{:.2}", g.changes_per_1e5())
            } else {
                "n/a".into()
            },
            o.bits
        ));
    }
    s
}
