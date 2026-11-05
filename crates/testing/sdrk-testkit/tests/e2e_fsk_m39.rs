//! M39 tests 2, 3 and 4: loops through the blocks `fsk_mod` -> channel -> `fsk_demod`, and the
//! automatic frequency control `afc`. The measured values are printed (run with `--nocapture`
//! to see them).
//!
//! - **Test 2 (`fsk_loop_ber`):** 2-FSK with the modulation index 1 (9600 Bd, deviation
//!   4800 Hz, 8 samples per symbol), PRBS-15 data, white Gaussian noise at Eb/N0 10 and 12 dB,
//!   then the channel filter of the receiver (`ddc` without decimation, cutoff = the symbol
//!   rate, D580) and `fsk_demod`; 10^6 bits counted per point by the `BerMeter`. Plain FSK
//!   (`bt` 0, seed 532) must be within a factor 2 of the non-coherent theory
//!   `0.5 exp(-Eb / (2 N0))`; GFSK with `bt` 0.5 (seed 533) at most 1.5 dB worse than that
//!   theory (D581).
//! - **Test 3:** 4-FSK (4800 Bd, outer deviation 2400 Hz: the M17 levels +-2400 and +-800 Hz,
//!   8 samples per symbol), random symbols, Eb/N0 14 dB, channel filter cutoff 1.3 times the
//!   symbol rate (D580, D583); symbol error rate over 10^5 symbols below 1e-3 (seed
//!   534).
//! - **Test 4 (AFC):** GFSK (`bt` 0.5, 50 kBd, deviation 25 kHz, 8 samples per symbol) with a
//!   carrier offset of 5 kHz that drifts by 20 ppm of 868 MHz (17.36 kHz) over 10 s (a ramp,
//!   D579). The receiver: a wide `ddc` (cutoff 80 kHz, it passes the uncorrected signal), `afc`
//!   (time constant 0.01 s, `max_offset_hz` 30 kHz, `carrier_hz` 868 MHz), the channel `ddc`
//!   (cutoff 45 kHz), `fsk_demod`, and the PN9-whitened frame decoder. On the idle preamble of
//!   the framing (the alternating bits of 0x55) the estimate is within 50 Hz of the true offset
//!   from 1 s on and the reading `offset_ppm` within 0.1 ppm; `hold` freezes the estimate and
//!   after the release it converges again within 1 s. Packets at Eb/N0 12 dB decode with
//!   PER < 1 % with the AFC and mostly fail without it (seed 535; D582).
//!
//! The file needs the feature `std` (for the channel models and `VecStorage`).

#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::large_stack_frames
)]

use num_complex::Complex64;
use sdrk::{C32, Rate};
use sdrk_blocks::{
    AfcBlock, AfcConfig, Block, DdcBlock, DdcConfig, FskDemodBlock, FskDemodConfig, FskModBlock,
    FskModConfig, ParamValue,
};
use sdrk_proto::frame::{self, Decoder, Whitening};
use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_proto::{Packet, PacketFlags};
use sdrk_testkit::VecStorage;
use sdrk_testkit::ber::BerMeter;
use sdrk_testkit::channel::{Awgn, Impairment};
use sdrk_testkit::rng::Pcg32;

// ---- helpers -----------------------------------------------------------------------------

fn rate(hz: u64) -> Rate {
    Rate::hz(hz).unwrap()
}

/// `ddc` without decimation: the channel filter of a receiver (cutoff = its -6 dB point).
fn channel_filter(fs: u64, cutoff_hz: f64) -> DdcBlock<C32, VecStorage> {
    let cfg = DdcConfig {
        decimation: 1,
        cutoff_hz: Some(cutoff_hz),
        ..DdcConfig::default()
    };
    DdcBlock::new(&cfg, rate(fs), &mut VecStorage).unwrap()
}

/// Runs a `C32 -> C32` block over all of `x` (`f` is its `process`, returning the consumed and
/// produced counts).
fn run_all(x: &[C32], mut f: impl FnMut(&[C32], &mut [C32]) -> (usize, usize)) -> Vec<C32> {
    let mut y = vec![C32::default(); x.len() + 64];
    let (mut done, mut made) = (0, 0);
    while done < x.len() {
        let (c, p) = f(&x[done..], &mut y[made..]);
        assert!(c > 0 || p > 0, "the block made no progress");
        done += c;
        made += p;
    }
    y.truncate(made);
    y
}

/// Runs `fsk_demod` over all of `x` and appends the soft bits to `soft`.
fn demod_all(d: &mut FskDemodBlock<VecStorage>, x: &[C32], buf: &mut [f32], soft: &mut Vec<f32>) {
    let mut rest = x;
    loop {
        let p = d.process(rest, buf).unwrap();
        rest = &rest[p.consumed..];
        soft.extend_from_slice(&buf[..p.produced]);
        if rest.is_empty() && p.produced < buf.len() {
            break;
        }
    }
}

fn to_c64(x: &[C32]) -> Vec<Complex64> {
    x.iter()
        .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
        .collect()
}

fn to_c32(x: &[Complex64]) -> Vec<C32> {
    x.iter()
        .map(|v| C32::new(v.re as f32, v.im as f32))
        .collect()
}

fn soft_of(bit: u8) -> f32 {
    if bit == 1 { 1.0 } else { -1.0 }
}

/// The non-coherent 2-FSK theory `0.5 exp(-Eb / (2 N0))`.
fn ncfsk_theory(ebn0_db: f64) -> f64 {
    0.5 * (-(10f64.powf(ebn0_db / 10.0)) / 2.0).exp()
}

/// The Eb/N0 in dB at which the theory gives `ber` (the inverse of `ncfsk_theory`).
fn theory_ebn0_db(ber: f64) -> f64 {
    10.0 * (2.0 * (0.5 / ber).ln()).log10()
}

// ---- test 2: fsk_loop_ber -----------------------------------------------------------------

/// Symbol rate of test 2 (the sample rate is 8 times this).
const BER_SYMBOL_RATE: u64 = 9_600;
/// Samples per symbol of tests 2 and 3.
const SPS: usize = 8;
/// Bits counted per Eb/N0 point.
const BER_BITS: u64 = 1_000_000;
/// Soft bits skipped before the meter (lock of the timing loop).
const SETTLE_BITS: usize = 1_000;
/// Bits modulated per step.
const BER_CHUNK: usize = 4_096;
/// Initial register of the PRBS and the meters (all ones).
const PRBS_SEED: u32 = 0x7fff;

/// One receiver of test 2: noise, channel filter, `fsk_demod`, `BerMeter`.
struct BerReceiver {
    ebn0_db: f64,
    awgn: Awgn,
    filter: DdcBlock<C32, VecStorage>,
    demod: FskDemodBlock<VecStorage>,
    meter: BerMeter,
    skipped: usize,
    buf: Vec<f32>,
    soft: Vec<f32>,
}

impl BerReceiver {
    fn new(ebn0_db: f64, seed: u64) -> Self {
        let fs = BER_SYMBOL_RATE * SPS as u64;
        let cfg = FskDemodConfig {
            symbol_rate_hz: BER_SYMBOL_RATE as f64,
            deviation_hz: BER_SYMBOL_RATE as f64 / 2.0,
            ..FskDemodConfig::default()
        };
        Self {
            ebn0_db,
            // The modulator output has the magnitude 1, so the mean power is 1 per sample.
            awgn: Awgn::from_ebn0(ebn0_db, 1, SPS as f64, 1.0, seed).unwrap(),
            filter: channel_filter(fs, BER_SYMBOL_RATE as f64),
            demod: FskDemodBlock::new(&cfg, rate(fs), &mut VecStorage).unwrap(),
            meter: BerMeter::new(Poly::Prbs15, PRBS_SEED).unwrap(),
            skipped: 0,
            buf: vec![0.0; 4_096],
            soft: Vec::new(),
        }
    }

    fn done(&self) -> bool {
        self.meter.bits() >= BER_BITS
    }

    fn process(&mut self, clean: &[Complex64]) {
        if self.done() {
            return;
        }
        let mut noisy = Vec::with_capacity(clean.len());
        self.awgn.process(clean, &mut noisy);
        let filter = &mut self.filter;
        let filtered = run_all(&to_c32(&noisy), |x, y| {
            let p = filter.process(x, y).unwrap();
            (p.consumed, p.produced)
        });
        self.soft.clear();
        demod_all(&mut self.demod, &filtered, &mut self.buf, &mut self.soft);
        for &b in &self.soft {
            if self.skipped < SETTLE_BITS {
                self.skipped += 1;
            } else if !self.done() {
                self.meter.push(u8::from(b > 0.0));
            }
        }
    }
}

/// `(Eb/N0 in dB, BER, theory, 95 % interval, slips)` of one point.
type BerPoint = (f64, f64, f64, (f64, f64), u64);

/// PRBS-15 2-FSK (h = 1, `bt`) through every Eb/N0 point of `points`.
fn ber_run(bt: f64, seed: u64, points: &[f64]) -> Vec<BerPoint> {
    let cfg = FskModConfig {
        samples_per_symbol: SPS,
        deviation_hz: BER_SYMBOL_RATE as f64 / 2.0,
        bt,
        ..FskModConfig::default()
    };
    let mut modulator = FskModBlock::new(&cfg, rate(BER_SYMBOL_RATE), &mut VecStorage).unwrap();
    let mut rxs: Vec<BerReceiver> = points.iter().map(|&e| BerReceiver::new(e, seed)).collect();
    let mut prbs = Prbs::new(Poly::Prbs15, PRBS_SEED).unwrap();
    let mut bits = vec![0f32; BER_CHUNK];
    let mut tx = vec![C32::default(); BER_CHUNK * SPS];
    // Enough bits for the settling, the delays and 10^6 counted bits.
    let limit = 2 * BER_BITS;
    let mut sent = 0u64;
    while sent < limit && !rxs.iter().all(BerReceiver::done) {
        for b in &mut bits {
            *b = soft_of(prbs.next_bit());
        }
        let p = modulator.process(&bits, &mut tx).unwrap();
        assert_eq!((p.consumed, p.produced), (BER_CHUNK, BER_CHUNK * SPS));
        let clean = to_c64(&tx);
        for rx in &mut rxs {
            rx.process(&clean);
        }
        sent += BER_CHUNK as u64;
    }
    rxs.iter()
        .map(|rx| {
            assert!(
                rx.done(),
                "Eb/N0 {}: only {} bits",
                rx.ebn0_db,
                rx.meter.bits()
            );
            (
                rx.ebn0_db,
                rx.meter.ber(),
                ncfsk_theory(rx.ebn0_db),
                rx.meter.confidence_95(),
                rx.meter.slips(),
            )
        })
        .collect()
}

fn print_point(name: &str, &(ebn0, ber, theory, (lo, hi), slips): &BerPoint) {
    eprintln!(
        "{name} Eb/N0 {ebn0} dB: BER {ber:.4e} (95 % interval {lo:.4e} .. {hi:.4e}), \
         non-coherent theory {theory:.4e}, ratio {:.3}, the theory reaches this BER at \
         {:.2} dB (penalty {:+.2} dB), slips {slips}",
        ber / theory,
        theory_ebn0_db(ber),
        ebn0 - theory_ebn0_db(ber)
    );
}

#[test]
fn fsk_loop_ber_within_2_of_the_non_coherent_theory() {
    let results = ber_run(0.0, 532, &[10.0, 12.0]);
    let mut failures = Vec::new();
    for point in &results {
        print_point("2-FSK h 1", point);
        let &(ebn0, ber, theory, _, slips) = point;
        if !(ber <= 2.0 * theory && ber >= theory / 2.0) || slips != 0 {
            failures.push(format!(
                "Eb/N0 {ebn0}: BER {ber:.4e} vs {theory:.4e}, slips {slips}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn gfsk_loop_ber_at_most_1_5_db_worse() {
    let results = ber_run(0.5, 533, &[10.0, 12.0]);
    let mut failures = Vec::new();
    for point in &results {
        print_point("GFSK bt 0.5 h 1", point);
        let &(ebn0, ber, _, _, slips) = point;
        // At most 1.5 dB worse: the BER is at most the theory 1.5 dB lower.
        let bound = ncfsk_theory(ebn0 - 1.5);
        if ber > bound || slips != 0 {
            failures.push(format!(
                "Eb/N0 {ebn0}: BER {ber:.4e} above {bound:.4e}, slips {slips}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

// ---- test 3: 4-FSK -----------------------------------------------------------------------

/// Symbol rate of test 3 (M17).
const SER_SYMBOL_RATE: u64 = 4_800;
/// Outer deviation of test 3 (M17: +-2400 Hz and +-800 Hz).
const SER_DEVIATION_HZ: f64 = 2_400.0;
/// Symbols counted.
const SER_SYMBOLS: usize = 100_000;
/// Symbols skipped before counting.
const SER_SETTLE: usize = 1_000;

#[test]
fn four_fsk_symbol_error_rate_below_1e_3_at_14_db() {
    let ebn0_db = 14.0;
    let fs = SER_SYMBOL_RATE * SPS as u64;
    let nsym = SER_SETTLE + SER_SYMBOLS + 1_000;
    let mut rng = Pcg32::new(534, 1);
    let bits: Vec<u8> = (0..2 * nsym).map(|_| (rng.next_u32() & 1) as u8).collect();
    let soft: Vec<f32> = bits.iter().map(|&b| soft_of(b)).collect();
    let mcfg = FskModConfig {
        levels: 4,
        samples_per_symbol: SPS,
        deviation_hz: SER_DEVIATION_HZ,
        ..FskModConfig::default()
    };
    let mut modulator =
        FskModBlock::new(&mcfg, rate(2 * SER_SYMBOL_RATE), &mut VecStorage).unwrap();
    let dcfg = FskDemodConfig {
        levels: 4,
        symbol_rate_hz: SER_SYMBOL_RATE as f64,
        deviation_hz: SER_DEVIATION_HZ,
        ..FskDemodConfig::default()
    };
    let mut demod = FskDemodBlock::new(&dcfg, rate(fs), &mut VecStorage).unwrap();
    let mut filter = channel_filter(fs, 1.3 * SER_SYMBOL_RATE as f64);
    // Two bits per symbol; the mean power is 1 per sample.
    let mut awgn = Awgn::from_ebn0(ebn0_db, 2, SPS as f64, 1.0, 534).unwrap();
    let mut tx = vec![C32::default(); 2_048 * SPS];
    let mut buf = vec![0f32; 4_096];
    let mut rx: Vec<f32> = Vec::new();
    for piece in soft.chunks(4_096) {
        let p = modulator.process(piece, &mut tx).unwrap();
        assert_eq!(p.consumed, piece.len());
        let mut noisy = Vec::new();
        awgn.process(&to_c64(&tx[..p.produced]), &mut noisy);
        let filtered = run_all(&to_c32(&noisy), |x, y| {
            let q = filter.process(x, y).unwrap();
            (q.consumed, q.produced)
        });
        demod_all(&mut demod, &filtered, &mut buf, &mut rx);
    }
    let rx_bits: Vec<u8> = rx.iter().map(|&s| u8::from(s > 0.0)).collect();
    // The delay in symbols: the received symbol `k` is the sent symbol `k - delay` (best match
    // over 2000 symbols after the settling).
    let errors_at = |delay: usize, from: usize, count: usize| -> usize {
        (from..from + count)
            .filter(|&k| {
                let r = &rx_bits[2 * k..2 * k + 2];
                let s = &bits[2 * (k - delay)..2 * (k - delay) + 2];
                r != s
            })
            .count()
    };
    let delay = (0..64)
        .min_by_key(|&d| errors_at(d, SER_SETTLE, 2_000))
        .unwrap();
    assert!(
        rx_bits.len() / 2 >= SER_SETTLE + SER_SYMBOLS,
        "only {} symbols",
        rx_bits.len() / 2
    );
    let errors = errors_at(delay, SER_SETTLE, SER_SYMBOLS);
    let ser = errors as f64 / SER_SYMBOLS as f64;
    let bit_errors = (SER_SETTLE..SER_SETTLE + SER_SYMBOLS)
        .map(|k| {
            (0..2)
                .filter(|&b| rx_bits[2 * k + b] != bits[2 * (k - delay) + b])
                .count()
        })
        .sum::<usize>();
    eprintln!(
        "4-FSK (M17 levels, {SER_SYMBOL_RATE} Bd, sps {SPS}) Eb/N0 {ebn0_db} dB: SER {ser:.4e} \
         ({errors} of {SER_SYMBOLS} symbols), BER {:.4e}, delay {delay} symbols, sps estimate \
         {:.4}",
        bit_errors as f64 / (2 * SER_SYMBOLS) as f64,
        demod.sps_estimate()
    );
    assert!(ser < 1e-3, "SER {ser:.4e}");
}

// ---- test 4: AFC -------------------------------------------------------------------------

/// Symbol rate of test 4.
const AFC_SYMBOL_RATE: u64 = 50_000;
/// Deviation of test 4 (modulation index 1).
const AFC_DEVIATION_HZ: f64 = 25_000.0;
/// Sample rate of test 4 (8 samples per symbol).
const AFC_FS: u64 = AFC_SYMBOL_RATE * SPS as u64;
/// Duration of test 4 in seconds.
const AFC_SECONDS: f64 = 10.0;
/// Carrier frequency of the ppm reading.
const CARRIER_HZ: f64 = 868e6;
/// Offset at the start.
const START_OFFSET_HZ: f64 = 5_000.0;
/// The drift of 20 ppm of the carrier over 10 s, in Hz per second (1736 Hz/s).
const DRIFT_HZ_PER_S: f64 = 20e-6 * CARRIER_HZ / AFC_SECONDS;
/// Time constant of the AFC: the lag behind the ramp is `DRIFT_HZ_PER_S * 0.01` = 17.4 Hz.
const AFC_TIME_CONSTANT_S: f64 = 0.01;
/// Limit of the estimate: above offset plus drift (22.36 kHz).
const AFC_MAX_OFFSET_HZ: f64 = 30_000.0;
/// Symbols modulated per step (4096 samples, about 10 ms).
const AFC_CHUNK: usize = 512;
/// Payload bytes of the packets of test 4.
const AFC_PAYLOAD: usize = 8;

/// The true offset at the time `t` in seconds.
fn true_offset_hz(t: f64) -> f64 {
    START_OFFSET_HZ + DRIFT_HZ_PER_S * t
}

/// The bits of test 4: the idle preamble (alternating, starting with 0) or packets of
/// `AFC_PAYLOAD` random bytes framed with 4 preamble bytes and PN9 whitening, separated by 2 to
/// 9 idle preamble bytes; the payloads are returned with the bits.
fn afc_bits(packets: bool) -> (Vec<u8>, Vec<Vec<u8>>) {
    let nsym = (AFC_SECONDS * AFC_FS as f64) as usize / SPS;
    let mut bits = Vec::with_capacity(nsym);
    let mut sent = Vec::new();
    if packets {
        let mut rng = Pcg32::new(535, 1);
        let mut frame_bits = vec![0u8; 512];
        loop {
            for _ in 0..2 + rng.below(8) {
                bits.extend_from_slice(&[0, 1, 0, 1, 0, 1, 0, 1]);
            }
            let data: Vec<u8> = (0..AFC_PAYLOAD).map(|_| rng.next_u32() as u8).collect();
            let p = Packet::new(0, PacketFlags::NONE, &data).unwrap();
            let n = frame::encode(&p, 4, Whitening::Pn9, &mut frame_bits).unwrap();
            if bits.len() + n > nsym {
                break;
            }
            bits.extend_from_slice(&frame_bits[..n]);
            sent.push(data);
        }
    }
    while bits.len() < nsym {
        bits.push((bits.len() & 1) as u8);
    }
    (bits, sent)
}

/// What one run of test 4 measured.
#[derive(Debug, Default)]
struct AfcRun {
    /// `(time in s, estimate in Hz, reading in ppm)` after every step (with the AFC).
    track: Vec<(f64, f64, f64)>,
    /// Payloads decoded with `CRC_OK`.
    decoded: Vec<Vec<u8>>,
}

/// Runs 10 s of test 4: the GFSK signal of `bits` with the drifting offset, noise of `ebn0_db`
/// (none if `None`), the receiver with or without the AFC. `hold` is a time interval in
/// seconds in which `hold` is set; `demodulate` false stops after the AFC.
fn afc_run(
    bits: &[u8],
    ebn0_db: Option<f64>,
    with_afc: bool,
    hold: Option<(f64, f64)>,
    demodulate: bool,
) -> AfcRun {
    let mcfg = FskModConfig {
        samples_per_symbol: SPS,
        deviation_hz: AFC_DEVIATION_HZ,
        bt: 0.5,
        ..FskModConfig::default()
    };
    let mut modulator = FskModBlock::new(&mcfg, rate(AFC_SYMBOL_RATE), &mut VecStorage).unwrap();
    let acfg = AfcConfig {
        time_constant_s: AFC_TIME_CONSTANT_S,
        max_offset_hz: AFC_MAX_OFFSET_HZ,
        hold: false,
        carrier_hz: CARRIER_HZ,
    };
    let mut afc = AfcBlock::new(&acfg, rate(AFC_FS), &mut VecStorage).unwrap();
    // The wide filter passes the signal at up to 22.4 kHz offset (+-25 kHz deviation, 50 kBd);
    // the channel filter behind the AFC is the one of GFSK at 0.9 times the symbol rate.
    let mut wide = channel_filter(AFC_FS, 80_000.0);
    let mut narrow = channel_filter(AFC_FS, 0.9 * AFC_SYMBOL_RATE as f64);
    let dcfg = FskDemodConfig {
        symbol_rate_hz: AFC_SYMBOL_RATE as f64,
        deviation_hz: AFC_DEVIATION_HZ,
        ..FskDemodConfig::default()
    };
    let mut demod = FskDemodBlock::new(&dcfg, rate(AFC_FS), &mut VecStorage).unwrap();
    let mut awgn = ebn0_db.map(|e| Awgn::from_ebn0(e, 1, SPS as f64, 1.0, 535).unwrap());
    let mut decoder = Decoder::new(0, false, Whitening::Pn9).unwrap();
    let mut run = AfcRun::default();
    let soft: Vec<f32> = bits.iter().map(|&b| soft_of(b)).collect();
    let mut tx = vec![C32::default(); AFC_CHUNK * SPS];
    let mut buf = vec![0f32; 4_096];
    let mut rx_soft = Vec::new();
    let mut n0 = 0usize;
    let mut bit_index = 0u64;
    let fs = AFC_FS as f64;
    for piece in soft.chunks(AFC_CHUNK) {
        let p = modulator.process(piece, &mut tx).unwrap();
        assert_eq!(p.consumed, piece.len());
        // The offset: phase 2 pi (f0 t + r t^2 / 2), computed in f64 from the sample index.
        let mut x: Vec<Complex64> = tx[..p.produced]
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let t = (n0 + i) as f64 / fs;
                let cycles = START_OFFSET_HZ * t + 0.5 * DRIFT_HZ_PER_S * t * t;
                let phase = std::f64::consts::TAU * (cycles - cycles.floor());
                Complex64::new(f64::from(c.re), f64::from(c.im))
                    * Complex64::new(phase.cos(), phase.sin())
            })
            .collect();
        if let Some(a) = awgn.as_mut() {
            let mut y = Vec::with_capacity(x.len());
            a.process(&x, &mut y);
            x = y;
        }
        let t_start = n0 as f64 / fs;
        n0 += p.produced;
        let t = n0 as f64 / fs;
        if let Some((from, to)) = hold {
            let on = t_start >= from && t_start < to;
            afc.set_param("hold", ParamValue::Bool(on)).unwrap();
        }
        let w = run_all(&to_c32(&x), |x, y| {
            let q = wide.process(x, y).unwrap();
            (q.consumed, q.produced)
        });
        let a = if with_afc {
            let a = run_all(&w, |x, y| {
                let q = afc.process(x, y).unwrap();
                (q.consumed, q.produced)
            });
            run.track.push((t, afc.offset_hz(), afc.offset_ppm()));
            a
        } else {
            w
        };
        if !demodulate {
            continue;
        }
        let n = run_all(&a, |x, y| {
            let q = narrow.process(x, y).unwrap();
            (q.consumed, q.produced)
        });
        rx_soft.clear();
        demod_all(&mut demod, &n, &mut buf, &mut rx_soft);
        for &s in &rx_soft {
            if let Some(pk) = decoder.push_bit(s > 0.0, bit_index) {
                assert_eq!(pk.flags, PacketFlags::CRC_OK);
                run.decoded.push(pk.data().to_vec());
            }
            bit_index += 1;
        }
    }
    run
}

/// Payloads of `sent` found in `decoded` in order.
fn matched(sent: &[Vec<u8>], decoded: &[Vec<u8>]) -> usize {
    let mut next = 0;
    let mut ok = 0;
    for d in decoded {
        if let Some(k) = sent[next..].iter().position(|s| s == d) {
            ok += 1;
            next += k + 1;
        }
    }
    ok
}

#[test]
fn afc_converges_within_50_hz_in_1_s_and_reports_the_ppm() {
    let (bits, _) = afc_bits(false);
    // `hold` from 5 s to 6 s; afterwards the estimate must converge again within 1 s.
    let (hold_from, hold_to) = (5.0, 6.0);
    let run = afc_run(&bits, None, true, Some((hold_from, hold_to)), false);
    let checked = |t: f64| t >= 1.0 && !(t >= hold_from && t < hold_to + 1.0);
    let mut last_outside = 0.0f64;
    let mut max_err = 0.0f64;
    let mut max_ppm_err = 0.0f64;
    for &(t, est, ppm) in &run.track {
        let truth = true_offset_hz(t);
        let err = est - truth;
        if t < hold_from && err.abs() >= 50.0 {
            last_outside = t;
        }
        if checked(t) {
            max_err = max_err.max(err.abs());
            max_ppm_err = max_ppm_err.max((ppm - truth / CARRIER_HZ * 1e6).abs());
        }
    }
    let at_end = run.track.last().unwrap();
    eprintln!(
        "AFC on the idle preamble (offset {START_OFFSET_HZ} Hz + {DRIFT_HZ_PER_S:.0} Hz/s, \
         time constant {AFC_TIME_CONSTANT_S} s): within 50 Hz after {last_outside:.3} s, largest \
         error from 1 s on {max_err:.1} Hz (the lag of a first-order loop behind the ramp is \
         {:.1} Hz), largest ppm error {max_ppm_err:.4} ppm; at {:.1} s: {:.1} Hz = {:.4} ppm \
         (true {:.4} ppm)",
        DRIFT_HZ_PER_S * AFC_TIME_CONSTANT_S,
        at_end.0,
        at_end.1,
        at_end.2,
        true_offset_hz(at_end.0) / CARRIER_HZ * 1e6
    );
    assert!(
        last_outside < 1.0,
        "within 50 Hz only after {last_outside} s"
    );
    assert!(max_err < 50.0, "error {max_err} Hz");
    assert!(max_ppm_err < 0.1, "ppm error {max_ppm_err}");

    // `hold` freezes the estimate: every reading in the hold interval after its first step is
    // the value of that step, bit for bit.
    let held: Vec<f64> = run
        .track
        .windows(2)
        .filter(|w| w[0].0 > hold_from && w[1].0 <= hold_to)
        .map(|w| w[1].1)
        .collect();
    assert!(held.len() > 50, "{} steps in the hold interval", held.len());
    assert!(
        held.iter().all(|&v| v == held[0]),
        "the estimate moved during hold"
    );
    // After the release it converges again within 1 s.
    let after: Vec<f64> = run
        .track
        .iter()
        .filter(|&&(t, _, _)| t >= hold_to + 1.0 && t < hold_to + 2.0)
        .map(|&(t, est, _)| (est - true_offset_hz(t)).abs())
        .collect();
    let worst_after = after.iter().fold(0.0f64, |m, &e| m.max(e));
    eprintln!(
        "hold from {hold_from} s to {hold_to} s: the estimate stayed at {:.1} Hz (the offset \
         moved by {:.0} Hz); 1 s after the release the error is at most {worst_after:.1} Hz",
        held[0],
        DRIFT_HZ_PER_S * (hold_to - hold_from)
    );
    assert!(worst_after < 50.0, "after the release: {worst_after} Hz");
}

#[test]
fn packets_decode_at_12_db_with_the_afc_and_fail_without() {
    let (bits, sent) = afc_bits(true);
    let with = afc_run(&bits, Some(12.0), true, None, true);
    let without = afc_run(&bits, Some(12.0), false, None, true);
    let ok_with = matched(&sent, &with.decoded);
    let ok_without = matched(&sent, &without.decoded);
    let per_with = 1.0 - ok_with as f64 / sent.len() as f64;
    let per_without = 1.0 - ok_without as f64 / sent.len() as f64;
    // How far the data and the noise move the estimate (for the report).
    let wander = with
        .track
        .iter()
        .filter(|&&(t, _, _)| t >= 1.0)
        .map(|&(t, est, _)| (est - true_offset_hz(t)).abs())
        .fold(0.0f64, f64::max);
    eprintln!(
        "AFC packets ({AFC_PAYLOAD}-byte payloads, PN9, Eb/N0 12 dB): {} packets sent; with AFC \
         {ok_with} decoded (PER {per_with:.4}), largest estimate error from 1 s on \
         {wander:.0} Hz; without AFC {ok_without} decoded (PER {per_without:.4})",
        sent.len()
    );
    assert!(sent.len() >= 1_000, "{} packets", sent.len());
    assert!(per_with < 0.01, "PER with AFC {per_with}");
    assert!(per_without > 0.5, "PER without AFC {per_without}");
}
