//! The blocks of M40 (test 6 and the block rules): `CostasBlock` (the kernel `costas` through
//! the facade), `PskModBlock` (the Gray points, the differential phase, the pulse through
//! `fir_interpolate`) and `PskDemodBlock` (the composition `fir` -> `symbol_sync` -> `costas`
//! or the differential product -> soft bits), each equal to its kernels whole and in 20 cut
//! plans of seed 559; the timing, rates and parameter checks; the extremes (NaN, infinity,
//! silence, full scale); the loop `psk_mod` -> `psk_demod` without noise is error-free for
//! BPSK and QPSK, coherent and differential; `frame_decode` with `polarity` both. M40.1: the
//! `costas` order 8, the 8PSK and pi/4-DQPSK symbol tables, the `variant` combinations, the
//! timing of 8PSK and OQPSK, the new options equal to their kernel chains whole and in 20 cut
//! plans of seed 564, the noiseless loop and the extremes for 8PSK, pi/4-DQPSK and OQPSK.
#![allow(
    clippy::approx_constant,
    clippy::assert_is_empty,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

mod common;

use common::{check_block, rate, run_block, same_bits};
use sdrk::{C32, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::design::{rrc, rrc_len};
use sdrk_blocks::{
    Block, CostasBlock, CostasConfig, Error, FrameDecodeBlock, FrameDecodeConfig, FramePolarity,
    PSK8_GRAY, Packet, PacketFlags, ParamValue, PskDemodBlock, PskDemodConfig, PskModBlock,
    PskModConfig, PskVariant, SliceBlock, SliceConfig, SliceConstellation,
};
use sdrk_core::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk_core::kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize, EqualizeState};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted};
use sdrk_device::Rate;
use sdrk_testkit::{Pcg32, VecStorage};

/// Seed of the chunk-invariance tests of M40 (test 6) and of the random signals of this file.
const SEED: u64 = 559;

/// Seed of the chunk-invariance tests of M40.1 (test 5) and of the random signals of its new
/// options.
const M40_1_SEED: u64 = 564;

const H: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// Random soft bits `+-1` (stream `stream` of the seed).
fn soft_bits(n: usize, stream: u64) -> Vec<f32> {
    soft_bits_seed(n, SEED, stream)
}

/// Random soft bits `+-1` (stream `stream` of `seed`).
fn soft_bits_seed(n: usize, seed: u64, stream: u64) -> Vec<f32> {
    let mut rng = Pcg32::new(seed, stream);
    (0..n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The point at `k pi / 4`.
fn point(k: u8) -> C32 {
    [
        C32::new(1.0, 0.0),
        C32::new(H, H),
        C32::new(0.0, 1.0),
        C32::new(-H, H),
        C32::new(-1.0, 0.0),
        C32::new(-H, -H),
        C32::new(0.0, -1.0),
        C32::new(H, -H),
    ][usize::from(k & 7)]
}

/// The symbols of §3, written out independently: Gray mapping, the differential phase in
/// units of 45 degrees.
fn symbols(bits: &[f32], order: u32, differential: bool) -> Vec<C32> {
    symbols_v(bits, order, differential, PskVariant::Plain)
}

/// The point at `k pi / 8` (M40.1): the even ones are `point`, the odd ones the `f64` values
/// rounded once.
fn point16(k: u8) -> C32 {
    if k.is_multiple_of(2) {
        point(k / 2)
    } else {
        let a = f64::from(k & 15) * std::f64::consts::PI / 8.0;
        C32::new(a.cos() as f32, a.sin() as f32)
    }
}

/// The symbol index `k` of the 8PSK Gray code `g = b0 4 + b1 2 + b2`, the table of M40.1 §1.
fn psk8_index(g: usize) -> u8 {
    match g {
        0b000 => 0,
        0b001 => 1,
        0b011 => 2,
        0b010 => 3,
        0b110 => 4,
        0b111 => 5,
        0b101 => 6,
        _ => 7,
    }
}

/// The symbols of M40 §3 and M40.1 §1 and §2 in units of 22.5 degrees: BPSK and QPSK as M40,
/// 8PSK `(2 k + 1) pi / 8`, pi/4-DQPSK the increments `00 -> +pi/4`, `01 -> +3 pi/4`,
/// `11 -> -3 pi/4`, `10 -> -pi/4`.
fn symbols_v(bits: &[f32], order: u32, differential: bool, variant: PskVariant) -> Vec<C32> {
    let bps = match order {
        8 => 3,
        4 => 2,
        _ => 1,
    };
    let mut phase = 0u8;
    bits.chunks_exact(bps)
        .map(|b| {
            let k = match bps {
                1 => {
                    if b[0] > 0.0 {
                        0
                    } else {
                        8
                    }
                }
                2 if variant == PskVariant::Pi4 => match (b[0] > 0.0, b[1] > 0.0) {
                    (false, false) => 2,
                    (false, true) => 6,
                    (true, true) => 10,
                    (true, false) => 14,
                },
                2 => match (b[0] > 0.0, b[1] > 0.0) {
                    (true, true) => 2,
                    (false, true) => 6,
                    (false, false) => 10,
                    (true, false) => 14,
                },
                _ => {
                    let g = b.iter().fold(0, |acc, &v| acc * 2 + usize::from(v > 0.0));
                    2 * psk8_index(g) + 1
                }
            };
            if differential {
                phase = (phase + k) & 15;
                point16(phase)
            } else {
                point16(k)
            }
        })
        .collect()
}

fn rrc_taps(rolloff: f64, sps: usize, span: usize, gain: f64) -> Vec<f32> {
    let mut h = vec![0.0f64; rrc_len(sps, span).unwrap()];
    rrc(rolloff, sps, span, &mut h).unwrap();
    h.iter().map(|v| (v * gain) as f32).collect()
}

/// `psk_mod` composed from the kernels in one call.
fn mod_kernels(cfg: &PskModConfig, bits: &[f32]) -> Vec<C32> {
    let sps = cfg.samples_per_symbol;
    let taps = rrc_taps(cfg.rolloff, sps, cfg.span, (sps as f64).sqrt());
    let n = taps.len();
    let mut st =
        FirInterpState::<C32, _, _>::new(taps, vec![C32::default(); n.div_ceil(sps)], sps).unwrap();
    let syms = symbols_v(bits, cfg.order, cfg.differential, cfg.variant);
    let mut out = vec![C32::default(); syms.len() * sps];
    let p =
        <Sdrk as FirInterpolate<C32>>::fir_interpolate(&Sdrk, &mut st, &syms, &mut out).unwrap();
    assert_eq!(p.consumed, syms.len());
    if cfg.variant == PskVariant::Offset {
        // OQPSK: the imaginary part `sps / 2` samples later, zeros first
        let im: Vec<f32> = out.iter().map(|x| x.im).collect();
        for (i, x) in out.iter_mut().enumerate() {
            x.im = if i >= sps / 2 { im[i - sps / 2] } else { 0.0 };
        }
    }
    out
}

/// `psk_demod` composed from the kernels in one call each.
fn demod_kernels(cfg: &PskDemodConfig, fs: Rate, input: &[C32]) -> Vec<f32> {
    let n = (fs.as_f64() / cfg.symbol_rate_hz).round() as usize;
    let taps = rrc_taps(cfg.rolloff, n, cfg.span, 1.0 / (n as f64).sqrt());
    let len = taps.len();
    let mut fir = FirState::<C32, _, _>::new(taps, vec![C32::default(); len - 1]).unwrap();
    let mut mf = vec![C32::default(); input.len()];
    <Sdrk as Fir<C32>>::fir(&Sdrk, &mut fir, input, &mut mf).unwrap();
    let sps = sdrk_blocks::SymbolSyncConfig {
        symbol_rate_hz: cfg.symbol_rate_hz,
        ..sdrk_blocks::SymbolSyncConfig::default()
    }
    .sps(fs)
    .unwrap();
    let mut sync =
        SymbolSyncState::<C32>::new(sps, Ted::Gardner, cfg.sync_bw, 0.7071, 0.005, 1.0).unwrap();
    if cfg.variant == PskVariant::Offset {
        let y = offset_kernels(cfg, n, &mut sync, &mf);
        return soft_of(cfg, eq_kernel(cfg, y));
    }
    let mut syms = vec![C32::default(); input.len()];
    let p = <Sdrk as SymbolSync<C32>>::symbol_sync(&Sdrk, &mut sync, &mf, &mut syms).unwrap();
    assert_eq!(p.consumed, input.len());
    syms.truncate(p.produced);
    let mut y = vec![C32::default(); syms.len()];
    if cfg.differential {
        let syms = eq_kernel(cfg, syms.clone());
        let mut prev = C32::new(1.0, 0.0);
        for (z, &s) in y.iter_mut().zip(&syms) {
            *z = C32::new(
                s.re * prev.re + s.im * prev.im,
                s.im * prev.re - s.re * prev.im,
            );
            prev = s;
        }
    } else {
        // The equalizer before the carrier loop (D825).
        let syms = eq_kernel(cfg, syms);
        let mut st =
            CostasState::new(psk(cfg.order), cfg.carrier_bw, 0.7071, cfg.max_offset).unwrap();
        <Sdrk as Costas>::costas(&Sdrk, &mut st, &syms, &mut y).unwrap();
    }
    soft_of(cfg, y)
}

/// The equalizer of `psk_demod` (M43 §3) as one kernel call: nothing for `equalizer_taps` 0,
/// otherwise `equalize` with `equalizer_feedback_taps`, the constellation of the order,
/// `mu_cma` 0.05, `mu_dd` 0.02, 1000 CMA symbols and the phase tracker at `carrier_bw` (CMA
/// only, `u32::MAX`, and no tracker with `differential`).
fn eq_kernel(cfg: &PskDemodConfig, x: Vec<C32>) -> Vec<C32> {
    let t = cfg.equalizer_taps;
    if t == 0 {
        return x;
    }
    let c = match cfg.order {
        2 => EqConstellation::Bpsk,
        4 => EqConstellation::Qpsk,
        _ => EqConstellation::Psk8,
    };
    let f = cfg.equalizer_feedback_taps;
    let cma = if cfg.differential { u32::MAX } else { 1000 };
    let phase_bw = if cfg.differential {
        0.0
    } else {
        cfg.carrier_bw
    };
    let mut st = EqualizeState::new(
        vec![C32::default(); t + f],
        vec![C32::default(); t + f],
        EqParams {
            constellation: c,
            feedback_taps: f,
            mu_cma: 0.05,
            mu_dd: 0.02,
            cma_symbols: cma,
            phase_bw,
        },
    )
    .unwrap();
    let mut y = vec![C32::default(); x.len()];
    <Sdrk as Equalize>::equalize(&Sdrk, &mut st, &x, &mut y).unwrap();
    y
}

/// The soft bits of the symbols `y` for the order and variant of `cfg`.
fn soft_of(cfg: &PskDemodConfig, y: Vec<C32>) -> Vec<f32> {
    let mut bits = Vec::new();
    for s in y {
        match (cfg.order, cfg.variant) {
            (8, _) => bits.extend_from_slice(&psk8_reference(1.0, s)),
            (4, PskVariant::Pi4) => bits.extend_from_slice(&[-s.im, -s.re]),
            (4, _) => bits.extend_from_slice(&[s.re, s.im]),
            _ => bits.push(s.re),
        }
    }
    bits
}

/// The OQPSK receiver of `psk_demod` (D686, D687) one sample at a time: `rotate` of the
/// sample with the oscillator `(phase, -freq / sps)`, the ring of `n / 2` samples, the
/// realigned sample `(re r_(n-h), im r_n)` into `symbol_sync`, and after each strobe the phase
/// error `sign(y.re) z.im - sign(y.im) z.re` with `z` the cubic interpolation of the last four
/// `(re r_n, im r_(n-h))` at the strobe's `mu`, the loop filter with the gains of `costas`
/// order 4 and the jump `-kp e` of the phase. Returns the realigned symbols.
fn offset_kernels(
    cfg: &PskDemodConfig,
    n: usize,
    sync: &mut SymbolSyncState<C32>,
    mf: &[C32],
) -> Vec<C32> {
    let g = CostasState::new(
        CostasConstellation::Qpsk,
        cfg.carrier_bw,
        0.7071,
        cfg.max_offset,
    )
    .unwrap()
    .gains();
    let step = |v: f64| (v * 4_294_967_296.0 + 0.5).floor() as i64 as u32;
    let sign = |x: f32| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    let interp = |w: [f32; 4], mu: f32| {
        let c1 = ((-(w[0] * (1.0 / 3.0)) - w[1] * 0.5) + w[2]) - w[3] * (1.0 / 6.0);
        let c2 = (w[0] * 0.5 - w[1]) + w[2] * 0.5;
        let c3 = ((-(w[0] * (1.0 / 6.0)) + w[1] * 0.5) - w[2] * 0.5) + w[3] * (1.0 / 6.0);
        ((c3 * mu + c2) * mu + c1) * mu + w[1]
    };
    let mut ring = vec![C32::default(); n / 2];
    let mut window = [C32::default(); 4];
    let (mut phase, mut freq) = (0u32, 0.0f32);
    let (mut mu, mut remaining) = (0.0f32, 0usize);
    let mut syms = Vec::new();
    for (k, &x) in mf.iter().enumerate() {
        if remaining == 0 {
            let t = sync.next_strobe();
            mu = (t - t.floor()) as f32;
            remaining = (t.floor() as i64 + 3).max(1) as usize;
        }
        let mut nco = Nco::from_increment(step(-f64::from(freq) / sync.sps()));
        nco.set_phase(phase);
        let mut r = [C32::default()];
        <Sdrk as Rotate<C32>>::rotate(&Sdrk, &mut nco, &[x], &mut r).unwrap();
        phase = nco.phase();
        let r = r[0];
        let old = std::mem::replace(&mut ring[k % (n / 2)], r);
        window = [window[1], window[2], window[3], C32::new(r.re, old.im)];
        let mut y = [C32::default(); 2];
        let p =
            <Sdrk as SymbolSync<C32>>::symbol_sync(&Sdrk, sync, &[C32::new(old.re, r.im)], &mut y)
                .unwrap();
        assert_eq!(p.consumed, 1);
        remaining -= 1;
        assert_eq!(p.produced == 1, remaining == 0, "sample {k}");
        if p.produced == 1 {
            let y = y[0];
            let z = C32::new(
                interp(window.map(|w| w.re), mu),
                interp(window.map(|w| w.im), mu),
            );
            let e = sign(y.re) * z.im - sign(y.im) * z.re;
            let e = if e.is_finite() { e } else { 0.0 };
            let f = freq + g.ki * e;
            let f = if f.abs() < 1e-30 { 0.0 } else { f };
            freq = f.clamp(-g.max_offset, g.max_offset);
            phase = phase.wrapping_sub(step(f64::from(g.kp * e)));
            syms.push(y);
        }
    }
    syms
}

/// A noisy, turned and frequency-shifted QPSK or BPSK signal at `sps` samples per symbol from
/// `psk_mod` (the input of the demodulator tests).
fn received(order: u32, differential: bool, sps: usize, n_sym: usize, stream: u64) -> Vec<C32> {
    received_v(
        order,
        differential,
        PskVariant::Plain,
        sps,
        n_sym,
        SEED,
        stream,
    )
}

/// As `received` with a variant and the seed of the bits and the noise.
fn received_v(
    order: u32,
    differential: bool,
    variant: PskVariant,
    sps: usize,
    n_sym: usize,
    seed: u64,
    stream: u64,
) -> Vec<C32> {
    let cfg = PskModConfig {
        order,
        samples_per_symbol: sps,
        differential,
        variant,
        ..PskModConfig::default()
    };
    let bps = cfg.bits_per_symbol();
    let tx = mod_kernels(&cfg, &soft_bits_seed(n_sym * bps, seed, stream));
    let mut rng = Pcg32::new(seed, 100 + stream);
    tx.iter()
        .enumerate()
        .map(|(i, x)| {
            let ph = 0.6 + 2.0 * std::f64::consts::PI * 0.001 * i as f64 / sps as f64;
            let (s, c) = ph.sin_cos();
            let (re, im) = (f64::from(x.re), f64::from(x.im));
            C32::new(
                (re * c - im * s + 0.05 * rng.normal()) as f32,
                (re * s + im * c + 0.05 * rng.normal()) as f32,
            )
        })
        .collect()
}

// ---- costas ------------------------------------------------------------------------------

#[test]
fn costas_block_equals_the_kernel_whole_and_in_cut_plans() {
    for order in [2u32, 4] {
        let cfg = CostasConfig {
            constellation: psk(order),
            loop_bw: 0.03,
            ..CostasConfig::default()
        };
        // one sample per symbol: the symbols of `psk_mod`, turned and noisy
        let bits = soft_bits(6000, u64::from(order));
        let mut rng = Pcg32::new(SEED, 10 + u64::from(order));
        let input: Vec<C32> = symbols(&bits, order, false)
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let (sn, cs) = (1.0 + 0.004 * k as f32).sin_cos();
                C32::new(
                    s.re * cs - s.im * sn + 0.1 * rng.normal() as f32,
                    s.re * sn + s.im * cs + 0.1 * rng.normal() as f32,
                )
            })
            .collect();
        let mut st = CostasState::new(psk(order), 0.03, 0.7071, 0.05).unwrap();
        let mut want = vec![C32::default(); input.len()];
        <Sdrk as Costas>::costas(&Sdrk, &mut st, &input, &mut want).unwrap();
        check_block(
            &want,
            &input,
            SEED,
            || CostasBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap(),
            &format!("costas order {order}"),
        );
        let mut block = CostasBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap();
        let got = run_block(&mut block, &input, None);
        assert_eq!(got.len(), input.len());
        assert_eq!(block.state(), &st);
        assert!(block.locked(), "order {order}");
        assert!(
            (block.frequency() - 0.004 / (2.0 * std::f64::consts::PI)).abs() < 2e-4,
            "order {order}: {}",
            block.frequency()
        );
        let t = block.timing();
        assert_eq!(
            (t.ratio, t.kind, t.delay),
            (Ratio::ONE, RatioKind::Constant, Delay::ZERO)
        );
        assert_eq!(block.min_input(), 1);
        block.reset();
        assert_eq!(
            block.state(),
            &CostasState::new(psk(order), 0.03, 0.7071, 0.05).unwrap()
        );
        // a short output takes as many symbols as fit
        let p = block
            .process(&input[..10], &mut [C32::default(); 4])
            .unwrap();
        assert_eq!((p.consumed, p.produced), (4, 4));
    }
}

#[test]
fn costas_block_parameters() {
    let ok = CostasConfig::default();
    assert!(ok.validate().is_ok());
    let bad = [
        CostasConfig {
            loop_bw: 0.00009,
            ..ok
        },
        CostasConfig {
            loop_bw: 0.21,
            ..ok
        },
        CostasConfig {
            damping: 0.05,
            ..ok
        },
        CostasConfig {
            max_offset: -0.01,
            ..ok
        },
        CostasConfig {
            max_offset: 0.26,
            ..ok
        },
        CostasConfig {
            loop_bw: f64::NAN,
            ..ok
        },
    ];
    for cfg in bad {
        assert!(
            CostasBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).is_err(),
            "{cfg:?}"
        );
    }
    let mut b = CostasBlock::<VecStorage>::new(&ok, rate(1000), &mut VecStorage).unwrap();
    assert_eq!(
        b.set_param("loop_bw", ParamValue::F64(0.01)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(b.param("max_offset"), Ok(ParamValue::F64(0.05)));
}

// ---- psk_mod -----------------------------------------------------------------------------

#[test]
fn psk_mod_maps_gray_symbols_and_the_differential_phase() {
    // BPSK and QPSK, coherent and differential, at 4 samples per symbol and the default span:
    // the block equals the kernel chain (`mod_kernels`), whole and in cut plans; the symbol
    // mapping itself is checked against the explicit tables below.
    for (order, differential) in [(2, false), (2, true), (4, false), (4, true)] {
        let cfg = PskModConfig {
            order,
            samples_per_symbol: 4,
            differential,
            ..PskModConfig::default()
        };
        let bits = soft_bits(4000, 20 + u64::from(order) + u64::from(differential));
        let want = mod_kernels(&cfg, &bits);
        check_block(
            &want,
            &bits,
            SEED,
            || PskModBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap(),
            &format!("psk_mod order {order} differential {differential}"),
        );
    }
    // The explicit table: BPSK 1 -> +1, 0 -> -1; QPSK (b0, b1) -> ((2 b0 - 1), (2 b1 - 1)) / sqrt 2.
    assert_eq!(
        symbols(&[1.0, -1.0, f32::NAN, 0.0], 2, false),
        [
            C32::new(1.0, 0.0),
            C32::new(-1.0, 0.0),
            C32::new(-1.0, 0.0),
            C32::new(-1.0, 0.0)
        ]
    );
    assert_eq!(
        symbols(&[1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0], 4, false),
        [
            C32::new(H, H),
            C32::new(-H, H),
            C32::new(-H, -H),
            C32::new(H, -H)
        ]
    );
    // Differential BPSK: a 1 keeps the phase, a 0 flips it.
    assert_eq!(
        symbols(&[1.0, -1.0, -1.0, 1.0], 2, true),
        [
            C32::new(1.0, 0.0),
            C32::new(-1.0, 0.0),
            C32::new(1.0, 0.0),
            C32::new(1.0, 0.0)
        ]
    );
    // Differential QPSK: (1, 1) adds 45 degrees each time; eight of them close the circle.
    let q = symbols(&[1.0; 16], 4, true);
    assert_eq!(q[1], C32::new(0.0, 1.0));
    assert_eq!(q[7], C32::new(1.0, 0.0));
}

#[test]
fn psk_mod_symbol_centres_and_timing() {
    // An isolated symbol: the output is the pulse times the symbol, its peak sqrt(sps) h[c].
    let cfg = PskModConfig {
        order: 4,
        samples_per_symbol: 8,
        span: 10,
        ..PskModConfig::default()
    };
    let r = rate(50_000);
    let mut block = PskModBlock::<VecStorage>::new(&cfg, r, &mut VecStorage).unwrap();
    assert_eq!(block.num_taps(), 81);
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(4, 1).unwrap());
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, Delay::new(20, 2).unwrap()); // span / 2 symbols of 2 bits
    assert_eq!(block.min_input(), 2);
    assert_eq!(
        PskModBlock::<VecStorage>::rate_out(&cfg, r).unwrap(),
        Rate::hz(200_000).unwrap()
    );
    let bits = [1.0f32, 1.0];
    let mut out = vec![C32::default(); 8];
    let p = block.process(&bits, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 8));
    let taps = rrc_taps(0.35, 8, 10, 8f64.sqrt());
    for (k, o) in out.iter().enumerate() {
        assert_eq!(*o, C32::new(H * taps[k], H * taps[k]), "sample {k}");
    }
    // Whole symbols only: an odd bit, or less output room than a symbol, stays unconsumed.
    let p = block.process(&[1.0], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    let p = block.process(&[1.0, 1.0], &mut out[..7]).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
    // reset: the history and the differential phase start again
    block.reset();
    let mut again = vec![C32::default(); 8];
    block.process(&bits, &mut again).unwrap();
    assert_eq!(again, out);
    // parameters
    for bad in [
        PskModConfig {
            samples_per_symbol: 1,
            ..cfg
        },
        PskModConfig {
            samples_per_symbol: 65,
            ..cfg
        },
        PskModConfig { span: 3, ..cfg },
        PskModConfig { span: 33, ..cfg },
        PskModConfig {
            rolloff: 0.04,
            ..cfg
        },
        PskModConfig {
            rolloff: 1.01,
            ..cfg
        },
        PskModConfig { order: 9, ..cfg },
    ] {
        assert!(
            PskModBlock::<VecStorage>::new(&bad, r, &mut VecStorage).is_err(),
            "{bad:?}"
        );
    }
    assert_eq!(
        block.set_param("span", ParamValue::I64(12)),
        Err(Error::ReadOnlyParameter)
    );
}

// ---- psk_demod ---------------------------------------------------------------------------

#[test]
fn psk_demod_equals_the_kernels_whole_and_in_cut_plans() {
    for (i, (order, differential, sps)) in [
        (2u32, false, 4usize),
        (4, false, 8),
        (2, true, 2),
        (4, true, 5),
    ]
    .into_iter()
    .enumerate()
    {
        let fs = rate(9600 * sps as u64);
        let cfg = PskDemodConfig {
            order,
            symbol_rate_hz: 9600.0,
            differential,
            ..PskDemodConfig::default()
        };
        let input = received(order, differential, sps, 3000, 40 + i as u64);
        let want = demod_kernels(&cfg, fs, &input);
        assert!(want.len() > 2900 * cfg.bits_per_symbol(), "case {i}");
        check_block(
            &want,
            &input,
            SEED,
            || PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap(),
            &format!("psk_demod case {i}"),
        );
    }
}

#[test]
fn psk_demod_keeps_soft_bits_that_do_not_fit() {
    let fs = rate(38_400);
    let cfg = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 9600.0,
        ..PskDemodConfig::default()
    };
    let input = received(4, false, 4, 2000, 60);
    let want = demod_kernels(&cfg, fs, &input);
    let mut block = PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    for room_len in [1usize, 3, 5] {
        block.reset();
        let mut got = Vec::new();
        let mut rest: &[C32] = &input;
        let mut room = vec![0.0f32; room_len];
        loop {
            let p = block.process(rest, &mut room).unwrap();
            assert!(p.produced <= room_len);
            got.extend_from_slice(&room[..p.produced]);
            rest = &rest[p.consumed..];
            if p.consumed == 0 && p.produced == 0 {
                break;
            }
        }
        assert!(rest.is_empty());
        same_bits(&want, &got, &format!("room {room_len}"));
    }
    // no output room: nothing is consumed
    block.reset();
    let p = block.process(&input, &mut []).unwrap();
    assert_eq!((p.consumed, p.produced), (0, 0));
}

#[test]
fn psk_demod_timing_rates_and_parameters() {
    let cfg = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 25_000.0,
        ..PskDemodConfig::default()
    };
    let fs = rate(200_000);
    let block = PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    assert_eq!(block.samples_per_symbol(), 8);
    assert_eq!(block.num_taps(), 81);
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::new(1, 4).unwrap());
    assert_eq!(t.kind, RatioKind::Variable);
    assert_eq!(t.delay, Delay::new(82, 2).unwrap()); // 40 of the filter plus 1
    assert_eq!(
        block.min_input(),
        (8.0f64 * 1.005 * 1.25).ceil() as usize + 3
    );
    assert_eq!(
        PskDemodBlock::<VecStorage>::rate_out(&cfg, fs).unwrap(),
        Rate::hz(50_000).unwrap()
    );
    assert!(!block.locked());
    // within 1 % of an integer: 8.07 samples per symbol is accepted (the filter has 8, the
    // timing loop the exact value)
    let near = PskDemodConfig {
        symbol_rate_hz: 200_000.0 / 8.07,
        ..cfg
    };
    let b = PskDemodBlock::<VecStorage>::new(&near, fs, &mut VecStorage).unwrap();
    assert_eq!(b.samples_per_symbol(), 8);
    assert!((b.sps_estimate() - 8.07).abs() < 1e-6);
    // 2 and 2.01 samples per symbol are the lowest rates accepted (D625)
    for sps in [2.0, 2.01] {
        let low = PskDemodConfig {
            symbol_rate_hz: 200_000.0 / sps,
            ..cfg
        };
        assert_eq!(low.samples_per_symbol(fs).unwrap(), 2, "{sps}");
    }
    for sym in [
        200_000.0 / 8.1,
        200_000.0 / 7.5,
        200_000.0 / 1.5,
        // within 1 % of 2 but below 2 samples per symbol: the timing loop needs at least 2,
        // reported by the same rate error (D625)
        200_000.0 / 1.99,
        200_000.0 / 1.981,
        150_000.0,
        1.0,
    ] {
        let c = PskDemodConfig {
            symbol_rate_hz: sym,
            ..cfg
        };
        match PskDemodBlock::<VecStorage>::new(&c, fs, &mut VecStorage) {
            Err(Error::InvalidParameter { name, reason }) => {
                assert_eq!(name, "symbol_rate_hz");
                assert!(reason.contains("resample"), "{reason}");
            }
            other => panic!("{sym}: {other:?}"),
        }
    }
    for bad in [
        PskDemodConfig { order: 3, ..cfg },
        PskDemodConfig { span: 33, ..cfg },
        PskDemodConfig {
            rolloff: 0.01,
            ..cfg
        },
        PskDemodConfig {
            sync_bw: 0.2,
            ..cfg
        },
        PskDemodConfig {
            carrier_bw: 0.3,
            ..cfg
        },
        PskDemodConfig {
            max_offset: 0.3,
            ..cfg
        },
    ] {
        assert!(
            PskDemodBlock::<VecStorage>::new(&bad, fs, &mut VecStorage).is_err(),
            "{bad:?}"
        );
    }
    let bpsk = PskDemodConfig { order: 2, ..cfg };
    assert_eq!(
        PskDemodBlock::<VecStorage>::new(&bpsk, fs, &mut VecStorage)
            .unwrap()
            .timing()
            .ratio,
        Ratio::new(1, 8).unwrap()
    );
}

// ---- the loop ----------------------------------------------------------------------------

/// The lag (0 to 64 bits) at which the demodulated hard bits equal the sent bits from bit 400
/// on, and the number of errors there.
fn best_lag(sent: &[f32], soft: &[f32]) -> (usize, usize) {
    (0..64)
        .map(|lag| {
            let errors = soft
                .iter()
                .skip(400 + lag)
                .zip(sent.iter().skip(400))
                .filter(|(s, b)| (**s > 0.0) != (**b > 0.0))
                .count();
            (lag, errors)
        })
        .min_by_key(|&(_, e)| e)
        .unwrap()
}

#[test]
fn psk_mod_to_psk_demod_without_noise_is_error_free() {
    // M40, then the orders and variants of M40.1 (8PSK coherent and differential,
    // pi/4-DQPSK, OQPSK)
    for (order, differential, variant) in [
        (2u32, false, PskVariant::Plain),
        (2, true, PskVariant::Plain),
        (4, false, PskVariant::Plain),
        (4, true, PskVariant::Plain),
        (8, false, PskVariant::Plain),
        (8, true, PskVariant::Plain),
        (4, true, PskVariant::Pi4),
        (4, false, PskVariant::Offset),
    ] {
        let sps = 4;
        let m = PskModConfig {
            order,
            samples_per_symbol: sps,
            differential,
            variant,
            ..PskModConfig::default()
        };
        let d = PskDemodConfig {
            order,
            symbol_rate_hz: 1000.0,
            differential,
            variant,
            ..PskDemodConfig::default()
        };
        let bits = if variant == PskVariant::Plain && order != 8 {
            soft_bits(6000, 70 + u64::from(order) + u64::from(differential))
        } else {
            soft_bits_seed(
                6000,
                M40_1_SEED,
                70 + u64::from(order) + u64::from(differential),
            )
        };
        let bit_rate = rate(1000 * m.bits_per_symbol() as u64);
        let mut tx = PskModBlock::<VecStorage>::new(&m, bit_rate, &mut VecStorage).unwrap();
        let mut rx = PskDemodBlock::<VecStorage>::new(&d, rate(4000), &mut VecStorage).unwrap();
        let sig = run_block(&mut tx, &bits, None);
        assert_eq!(sig.len(), bits.len() / m.bits_per_symbol() * sps);
        let soft = run_block(&mut rx, &sig, None);
        let (lag, errors) = best_lag(&bits, &soft);
        assert_eq!(
            errors, 0,
            "order {order} differential {differential} {variant:?}: lag {lag}"
        );
        // the symbol centres have amplitude 1 (coherent; differential |y|^2 = 1) up to the
        // intersymbol interference of the truncated pulse and the timing jitter (8PSK soft
        // bits are distance differences, not amplitudes)
        // OQPSK: 0.2, for the phase jitter of its detector, whose terms at the other rail's
        // transition carry data (D687; measured 0.57 to 0.77 for H = 0.707 over 10^4 symbols)
        let tail = &soft[soft.len() - 200..];
        let want = if order == 4 { H } else { 1.0 };
        let tol = if variant == PskVariant::Offset {
            0.2
        } else {
            0.1
        };
        for s in tail.iter().filter(|_| order != 8) {
            assert!(
                (s.abs() - want).abs() < tol * want,
                "{order} {differential} {variant:?}: {s}"
            );
        }
        if !differential {
            assert!(rx.locked());
        }
    }
}

#[test]
fn extremes_nan_infinity_silence_and_full_scale() {
    // psk_mod: NaN and infinite soft bits are bits (NaN and -inf a 0, +inf a 1); the output is
    // finite.
    let cfg = PskModConfig {
        order: 4,
        samples_per_symbol: 4,
        ..PskModConfig::default()
    };
    let mut m = PskModBlock::<VecStorage>::new(&cfg, rate(2000), &mut VecStorage).unwrap();
    let odd = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MAX,
    ];
    let out = run_block(&mut m, &odd, None);
    assert_eq!(out.len(), 12);
    assert!(out.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
    let mut m2 = PskModBlock::<VecStorage>::new(&cfg, rate(2000), &mut VecStorage).unwrap();
    let same = run_block(&mut m2, &[-1.0, 1.0, -1.0, -1.0, -1.0, 1.0], None);
    assert_eq!(out, same);

    // psk_demod: silence gives zero soft bits; a burst of NaN and infinity in the middle of a
    // clean signal poisons only its neighbourhood, and the bits after it are right again.
    for (order, differential, variant) in [
        (2u32, false, PskVariant::Plain),
        (4, false, PskVariant::Plain),
        (4, true, PskVariant::Plain),
        (8, false, PskVariant::Plain),
        (8, true, PskVariant::Plain),
        (4, true, PskVariant::Pi4),
        (4, false, PskVariant::Offset),
    ] {
        let d = PskDemodConfig {
            order,
            symbol_rate_hz: 1000.0,
            differential,
            variant,
            ..PskDemodConfig::default()
        };
        let mut rx = PskDemodBlock::<VecStorage>::new(&d, rate(4000), &mut VecStorage).unwrap();
        let silence = run_block(&mut rx, &vec![C32::default(); 4000], None);
        assert!(silence.len() > 990 * d.bits_per_symbol());
        assert!(
            silence.iter().all(|s| *s == 0.0),
            "{order} {differential} {variant:?}"
        );
        assert!(!rx.locked());
        let m = PskModConfig {
            order,
            samples_per_symbol: 4,
            differential,
            variant,
            ..PskModConfig::default()
        };
        let bits = if variant == PskVariant::Plain && order != 8 {
            soft_bits(8000, 80 + u64::from(order))
        } else {
            soft_bits_seed(8000, M40_1_SEED, 80 + u64::from(order))
        };
        let mut tx = PskModBlock::<VecStorage>::new(&m, rate(1000), &mut VecStorage).unwrap();
        let mut sig = run_block(&mut tx, &bits, None);
        let mid = sig.len() / 2;
        for (k, x) in sig[mid..mid + 40].iter_mut().enumerate() {
            *x = match k % 4 {
                0 => C32::new(f32::NAN, 0.0),
                1 => C32::new(f32::INFINITY, -f32::INFINITY),
                2 => C32::new(f32::MAX, f32::MAX),
                _ => C32::new(0.0, f32::NAN),
            };
        }
        // full scale (amplitude 1 at the symbol centres) is the normal level of the loop
        rx.reset();
        let soft = run_block(&mut rx, &sig, None);
        assert!(soft.len() > (bits.len() - 64) * 9 / 10);
        // bits well after the burst are right again (coherent loops keep their phase: the
        // detector and lock rules skip non-finite symbols)
        let after = mid / 4 * m.bits_per_symbol() + 400;
        let (lag, _) = best_lag(&bits[..after / 2], &soft[..after / 2]);
        let errors = soft
            .iter()
            .skip(after + lag)
            .zip(bits.iter().skip(after))
            .filter(|(s, b)| (**s > 0.0) != (**b > 0.0))
            .count();
        assert_eq!(
            errors, 0,
            "order {order} differential {differential} {variant:?}"
        );
        assert!(soft[after + lag..].iter().all(|s| s.is_finite()));
    }
}

// ---- frame_decode polarity ---------------------------------------------------------------

#[test]
fn frame_decode_with_both_polarities_reads_inverted_frames() {
    let sent = Packet::new(0, PacketFlags::NONE, b"polarity").unwrap();
    let mut bits = vec![0u8; sdrk_proto::frame::frame_bits(8, 4)];
    sdrk_proto::frame::encode(&sent, 4, sdrk_proto::frame::Whitening::None, &mut bits).unwrap();
    let normal: Vec<f32> = bits
        .iter()
        .map(|&b| if b == 1 { 1.0 } else { -1.0 })
        .collect();
    let inverted: Vec<f32> = normal.iter().map(|x| -x).collect();
    let decode = |polarity, soft: &[f32]| {
        let cfg = FrameDecodeConfig {
            polarity,
            ..FrameDecodeConfig::default()
        };
        let mut dec =
            FrameDecodeBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap();
        let mut out = vec![Packet::default(); 4];
        let p = dec.process(soft, &mut out).unwrap();
        out.truncate(p.produced);
        out
    };
    // normal: only the sent polarity
    assert_eq!(decode(FramePolarity::Normal, &normal).len(), 1);
    assert!(decode(FramePolarity::Normal, &inverted).is_empty());
    // both: either polarity, same packet and index
    for soft in [&normal, &inverted] {
        let got = decode(FramePolarity::Both, soft);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].data(), b"polarity");
        assert_eq!(got[0].flags, PacketFlags::CRC_OK);
        assert_eq!(got[0].index, 32);
    }
    // a normal frame after an inverted one: the polarity is found per frame
    let mut two = inverted.clone();
    two.extend_from_slice(&normal);
    let got = decode(FramePolarity::Both, &two);
    assert_eq!(got.len(), 2);
    assert_eq!(got[1].index, (normal.len() + 32) as u64);
    assert!(got.iter().all(|p| p.data() == b"polarity"));
}

/// The max-log soft bits of M40.1 §1 written out independently: the points from `f64` rounded
/// once, the Gray code of each point from `PSK8_GRAY`, the minima in index order with `d < m`.
fn psk8_reference(scale: f32, y: C32) -> [f32; 3] {
    let mut code = [0u8; 8];
    for (g, &k) in PSK8_GRAY.iter().enumerate() {
        code[k as usize] = g as u8;
    }
    let dist: Vec<f32> = (0..8u32)
        .map(|k| {
            let a = f64::from(2 * k + 1) * core::f64::consts::PI / 8.0;
            let (pr, pi) = (a.cos() as f32, a.sin() as f32);
            let (dr, di) = (y.re - pr, y.im - pi);
            dr * dr + di * di
        })
        .collect();
    let mut out = [0.0f32; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let mut m = [None::<f32>; 2];
        for k in 0..8 {
            let b = usize::from((code[k] >> (2 - i)) & 1);
            if m[b].is_none_or(|v| dist[k] < v) {
                m[b] = Some(dist[k]);
            }
        }
        *o = scale * ((m[0].unwrap() - m[1].unwrap()) * 0.25);
    }
    out
}

#[test]
fn slice_8psk_gives_max_log_soft_bits() {
    let cfg = SliceConfig {
        constellation: SliceConstellation::Psk8,
        scale: 2.0,
    };
    let mut slice = SliceBlock::<C32, _>::new(&cfg, rate(1200), &mut VecStorage).unwrap();
    // The 8 points, the 8 boundaries k pi / 4, both at 0.5, random symbols, zero and specials.
    let mut input: Vec<C32> = Vec::new();
    for amp in [1.0f64, 0.5] {
        for k in 0..16u32 {
            let a = f64::from(k) * core::f64::consts::PI / 8.0;
            input.push(C32::new((amp * a.cos()) as f32, (amp * a.sin()) as f32));
        }
    }
    let mut rng = Pcg32::new(564, 1);
    for _ in 0..200 {
        input.push(C32::new(rng.normal() as f32, rng.normal() as f32));
    }
    input.push(C32::new(0.0, 0.0));
    let mut out = vec![0.0f32; 3 * input.len()];
    let p = slice.process(&input, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (input.len(), 3 * input.len()));
    let want: Vec<f32> = input.iter().flat_map(|&y| psk8_reference(2.0, y)).collect();
    same_bits(&want, &out, "8psk");
    // Each point gives its Gray code: point k (at (2k + 1) pi / 8) is input 2k + 1.
    for (g, &k) in PSK8_GRAY.iter().enumerate() {
        let b = &out[3 * (2 * k as usize + 1)..][..3];
        let bits = b
            .iter()
            .fold(0usize, |acc, &v| acc * 2 + usize::from(v > 0.0));
        assert_eq!(bits, g, "point {k}");
    }
    // NaN and infinity give NaN soft bits.
    let mut nan = [0.0f32; 6];
    slice
        .process(
            &[C32::new(f32::NAN, 0.0), C32::new(f32::INFINITY, 0.0)],
            &mut nan,
        )
        .unwrap();
    assert!(nan.iter().all(|b| b.is_nan()));
    // Three soft bits per symbol, whole symbols only, C32 input only.
    assert_eq!(slice.timing().ratio, Ratio::new(3, 1).unwrap());
    let p = slice.process(&input, &mut out[..8]).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 6));
    assert!(SliceBlock::<f32, _>::new(&cfg, rate(1200), &mut VecStorage).is_err());
    assert_eq!(SliceConstellation::Psk8.bits_per_symbol(), 3);
    assert_eq!(
        SliceBlock::<C32, VecStorage>::rate_out(&cfg, rate(1200)).unwrap(),
        rate(3600)
    );
    let mut c = SliceConfig::default();
    c.set("constellation", ParamValue::Choice(4)).unwrap();
    assert_eq!(c.constellation, SliceConstellation::Psk8);
    assert_eq!(c.get("constellation"), Ok(ParamValue::Choice(4)));
}

// ---- M40.1: 8PSK, pi/4-DQPSK and OQPSK ---------------------------------------------------

#[test]
fn costas_block_takes_order_8() {
    let cfg = CostasConfig {
        constellation: CostasConstellation::Psk8,
        loop_bw: 0.03,
        ..CostasConfig::default()
    };
    assert!(cfg.validate().is_ok());
    // one sample per symbol: 8PSK symbols turned by 0.2 rad plus 0.002 rad per symbol, noisy
    let bits = soft_bits_seed(9000, M40_1_SEED, 1);
    let mut rng = Pcg32::new(M40_1_SEED, 2);
    let input: Vec<C32> = symbols_v(&bits, 8, false, PskVariant::Plain)
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let (sn, cs) = (0.2 + 0.002 * k as f32).sin_cos();
            C32::new(
                s.re * cs - s.im * sn + 0.03 * rng.normal() as f32,
                s.re * sn + s.im * cs + 0.03 * rng.normal() as f32,
            )
        })
        .collect();
    let mut st = CostasState::new(CostasConstellation::Psk8, 0.03, 0.7071, 0.05).unwrap();
    let mut want = vec![C32::default(); input.len()];
    <Sdrk as Costas>::costas(&Sdrk, &mut st, &input, &mut want).unwrap();
    check_block(
        &want,
        &input,
        M40_1_SEED,
        || CostasBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap(),
        "costas order 8",
    );
    let mut block = CostasBlock::<VecStorage>::new(&cfg, rate(1000), &mut VecStorage).unwrap();
    let got = run_block(&mut block, &input, None);
    assert!(block.locked());
    assert!(
        (block.frequency() - 0.002 / (2.0 * std::f64::consts::PI)).abs() < 2e-4,
        "{}",
        block.frequency()
    );
    // the tail sits on the 8PSK points (no rotation: the loop started at the right octant)
    for (y, s) in got[2500..]
        .iter()
        .zip(&symbols_v(&bits, 8, false, PskVariant::Plain)[2500..])
    {
        assert!((y.re - s.re).hypot(y.im - s.im) < 0.15, "{y:?} {s:?}");
    }
    // the constellation is a choice: every name maps to its constellation and back, and an
    // index outside the choice is rejected
    let names = ["bpsk", "qpsk", "8psk", "qam16", "qam64"];
    let all = [
        CostasConstellation::Bpsk,
        CostasConstellation::Qpsk,
        CostasConstellation::Psk8,
        CostasConstellation::Qam16,
        CostasConstellation::Qam64,
    ];
    for (i, (name, want)) in names.iter().zip(all).enumerate() {
        let mut c = CostasConfig::default();
        c.set("constellation", ParamValue::Choice(i as u16))
            .unwrap();
        assert_eq!(c.constellation, want, "{name}");
        assert_eq!(c.get("constellation"), Ok(ParamValue::Choice(i as u16)));
        assert!(c.validate().is_ok());
    }
    let mut c = CostasConfig::default();
    assert!(c.set("constellation", ParamValue::Choice(5)).is_err());
    assert_eq!(
        c.set("order", ParamValue::I64(8)),
        Err(Error::UnknownParameter)
    );
}

/// The combinations of `order`, `differential`, `variant` and the samples per symbol (M40.1 §4,
/// D648): `true` where the configuration is valid; otherwise it must be `InvalidParameter` of
/// `variant` (the orders are all valid ones).
const COMBINATIONS: &[(u32, bool, PskVariant, usize, bool)] = &[
    (2, false, PskVariant::Plain, 4, true),
    (8, false, PskVariant::Plain, 4, true),
    (8, true, PskVariant::Plain, 5, true),
    (4, true, PskVariant::Pi4, 4, true),
    (4, true, PskVariant::Pi4, 5, true),
    (4, false, PskVariant::Pi4, 4, false),
    (2, true, PskVariant::Pi4, 4, false),
    (8, true, PskVariant::Pi4, 4, false),
    (4, false, PskVariant::Offset, 4, true),
    (4, false, PskVariant::Offset, 2, true),
    (4, false, PskVariant::Offset, 5, false),
    (4, true, PskVariant::Offset, 4, false),
    (2, false, PskVariant::Offset, 4, false),
    (8, false, PskVariant::Offset, 4, false),
];

#[test]
fn psk_variant_combinations_are_checked() {
    for &(order, differential, variant, sps, ok) in COMBINATIONS {
        let m = PskModConfig {
            order,
            samples_per_symbol: sps,
            differential,
            variant,
            ..PskModConfig::default()
        };
        let d = PskDemodConfig {
            order,
            symbol_rate_hz: 1000.0,
            differential,
            variant,
            ..PskDemodConfig::default()
        };
        let fs = rate(1000 * sps as u64);
        let what = format!("{order} {differential} {variant:?} {sps}");
        let rm = PskModBlock::<VecStorage>::new(&m, rate(1000), &mut VecStorage).map(|_| ());
        let rd = PskDemodBlock::<VecStorage>::new(&d, fs, &mut VecStorage).map(|_| ());
        for r in [m.validate(), rm, d.validate(fs), rd] {
            if ok {
                assert_eq!(r, Ok(()), "{what}");
            } else {
                assert!(
                    matches!(
                        r,
                        Err(Error::InvalidParameter {
                            name: "variant",
                            ..
                        })
                    ),
                    "{what}: {r:?}"
                );
            }
        }
    }
    // The demodulator checks the rounded samples per symbol: 6.05 is 6 (even), 4.96 is 5.
    let d = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 1000.0,
        variant: PskVariant::Offset,
        ..PskDemodConfig::default()
    };
    assert!(d.validate(rate(6050)).is_ok());
    assert!(matches!(
        d.validate(rate(4960)),
        Err(Error::InvalidParameter {
            name: "variant",
            ..
        })
    ));
    // `set` takes every variant on its own; the combination is checked when building.
    let mut m = PskModConfig::default();
    let mut dd = PskDemodConfig::default();
    for (i, v) in [PskVariant::Plain, PskVariant::Pi4, PskVariant::Offset]
        .into_iter()
        .enumerate()
    {
        m.set("variant", ParamValue::Choice(i as u16)).unwrap();
        dd.set("variant", ParamValue::Choice(i as u16)).unwrap();
        assert_eq!((m.variant, dd.variant), (v, v));
        assert_eq!(m.get("variant"), Ok(ParamValue::Choice(i as u16)));
        assert_eq!(dd.get("variant"), Ok(ParamValue::Choice(i as u16)));
    }
    assert_eq!(PskModConfig::default().variant, PskVariant::Plain);
    assert_eq!(PskDemodConfig::default().variant, PskVariant::Plain);
    assert!(m.set("variant", ParamValue::Choice(3)).is_err());
    // the orders 5, 6 and 7 are no PSK order; 8 is
    for order in [5u32, 6, 7] {
        assert!(matches!(
            PskModConfig {
                order,
                ..PskModConfig::default()
            }
            .validate(),
            Err(Error::InvalidParameter { name: "order", .. })
        ));
        assert!(matches!(
            PskDemodConfig {
                order,
                ..PskDemodConfig::default()
            }
            .validate(rate(8000)),
            Err(Error::InvalidParameter { name: "order", .. })
        ));
    }
}

#[test]
fn psk_mod_maps_8psk_and_pi4_symbols() {
    // One symbol into a fresh block: the output is the pulse times the point.
    let taps = rrc_taps(0.35, 8, 10, 8f64.sqrt());
    let one = |cfg: &PskModConfig, bits: &[f32]| {
        let mut b = PskModBlock::<VecStorage>::new(cfg, rate(9600), &mut VecStorage).unwrap();
        let mut out = vec![C32::default(); 8];
        let p = b.process(bits, &mut out).unwrap();
        assert_eq!((p.consumed, p.produced), (bits.len(), 8));
        out
    };
    let psk8 = PskModConfig {
        order: 8,
        samples_per_symbol: 8,
        ..PskModConfig::default()
    };
    // 8PSK: g -> k by the table of §1, the point exp(j (2 pi k / 8 + pi / 8))
    for (g, k) in [
        (0b000, 0u32),
        (0b001, 1),
        (0b011, 2),
        (0b010, 3),
        (0b110, 4),
        (0b111, 5),
        (0b101, 6),
        (0b100, 7),
    ] {
        let bits: Vec<f32> = (0..3)
            .map(|i| if (g >> (2 - i)) & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        let a = f64::from(2 * k + 1) * std::f64::consts::PI / 8.0;
        let p = C32::new(a.cos() as f32, a.sin() as f32);
        let out = one(&psk8, &bits);
        for (i, o) in out.iter().enumerate() {
            assert_eq!(
                *o,
                C32::new(p.re * taps[i], p.im * taps[i]),
                "g {g:03b}, sample {i}"
            );
        }
    }
    // pi/4-DQPSK from the start phase 0: 00 -> +pi/4, 01 -> +3 pi/4, 11 -> -3 pi/4,
    // 10 -> -pi/4
    let pi4 = PskModConfig {
        order: 4,
        samples_per_symbol: 8,
        differential: true,
        variant: PskVariant::Pi4,
        ..PskModConfig::default()
    };
    for (bits, p) in [
        ([-1.0f32, -1.0], C32::new(H, H)),
        ([-1.0, 1.0], C32::new(-H, H)),
        ([1.0, 1.0], C32::new(-H, -H)),
        ([1.0, -1.0], C32::new(H, -H)),
    ] {
        let out = one(&pi4, &bits);
        for (i, o) in out.iter().enumerate() {
            assert_eq!(
                *o,
                C32::new(p.re * taps[i], p.im * taps[i]),
                "{bits:?}, sample {i}"
            );
        }
    }
    // pi/4-DQPSK alternates between the axes and the diagonals: eight 00 pairs close the
    // circle; differential 8PSK runs in steps of pi/8
    let q = symbols_v(&[-1.0; 16], 4, true, PskVariant::Pi4);
    assert_eq!(q[0], C32::new(H, H));
    assert_eq!(q[1], C32::new(0.0, 1.0));
    assert_eq!(q[7], C32::new(1.0, 0.0));
    let e = symbols_v(&[-1.0; 6], 8, true, PskVariant::Plain);
    assert_eq!(e[1], point16(2));
}

#[test]
fn psk_mod_variants_equal_the_reference_whole_and_in_twenty_plans() {
    for (i, (order, differential, variant, sps)) in [
        (8u32, false, PskVariant::Plain, 4usize),
        (8, true, PskVariant::Plain, 5),
        (4, true, PskVariant::Pi4, 4),
        (4, false, PskVariant::Offset, 4),
        (4, false, PskVariant::Offset, 8),
    ]
    .into_iter()
    .enumerate()
    {
        let cfg = PskModConfig {
            order,
            samples_per_symbol: sps,
            differential,
            variant,
            ..PskModConfig::default()
        };
        let bits = soft_bits_seed(6000, M40_1_SEED, 10 + i as u64);
        let want = mod_kernels(&cfg, &bits);
        check_block(
            &want,
            &bits,
            M40_1_SEED,
            || PskModBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap(),
            &format!("psk_mod {order} {differential} {variant:?} {sps}"),
        );
        // reset clears the phase and the quadrature delay
        let mut b = PskModBlock::<VecStorage>::new(&cfg, rate(9600), &mut VecStorage).unwrap();
        run_block(&mut b, &bits, None);
        b.reset();
        same_bits(&want, &run_block(&mut b, &bits, None), "after reset");
    }
}

#[test]
fn psk_variant_timing_and_rates() {
    // psk_mod: 8PSK at 9600 bit/s and 8 samples per symbol is 25.6 kHz (ratio 8 / 3);
    // OQPSK adds half a symbol (one bit) to the delay.
    let psk8 = PskModConfig {
        order: 8,
        samples_per_symbol: 8,
        ..PskModConfig::default()
    };
    let b = PskModBlock::<VecStorage>::new(&psk8, rate(9600), &mut VecStorage).unwrap();
    assert_eq!(b.timing().ratio, Ratio::new(8, 3).unwrap());
    assert_eq!(b.timing().delay, Delay::new(30, 2).unwrap());
    assert_eq!(b.min_input(), 3);
    assert_eq!(
        PskModBlock::<VecStorage>::rate_out(&psk8, rate(9600)).unwrap(),
        rate(25_600)
    );
    let oq = PskModConfig {
        order: 4,
        samples_per_symbol: 8,
        span: 10,
        variant: PskVariant::Offset,
        ..PskModConfig::default()
    };
    let b = PskModBlock::<VecStorage>::new(&oq, rate(9600), &mut VecStorage).unwrap();
    assert_eq!(b.timing().ratio, Ratio::new(4, 1).unwrap());
    assert_eq!(b.timing().delay, Delay::new(22, 2).unwrap());
    let pi4 = PskModConfig {
        order: 4,
        differential: true,
        variant: PskVariant::Pi4,
        ..oq
    };
    let b = PskModBlock::<VecStorage>::new(&pi4, rate(9600), &mut VecStorage).unwrap();
    assert_eq!(b.timing().delay, Delay::new(20, 2).unwrap());
    // psk_demod: 8PSK gives 3 bits per symbol; OQPSK has the delay of plain QPSK (D658).
    let fs = rate(200_000);
    let d8 = PskDemodConfig {
        order: 8,
        symbol_rate_hz: 25_000.0,
        ..PskDemodConfig::default()
    };
    let b = PskDemodBlock::<VecStorage>::new(&d8, fs, &mut VecStorage).unwrap();
    assert_eq!(b.timing().ratio, Ratio::new(3, 8).unwrap());
    assert_eq!(b.timing().delay, Delay::new(82, 2).unwrap());
    assert_eq!(
        PskDemodBlock::<VecStorage>::rate_out(&d8, fs).unwrap(),
        rate(75_000)
    );
    let doq = PskDemodConfig {
        order: 4,
        symbol_rate_hz: 25_000.0,
        variant: PskVariant::Offset,
        ..PskDemodConfig::default()
    };
    let b = PskDemodBlock::<VecStorage>::new(&doq, fs, &mut VecStorage).unwrap();
    assert_eq!(b.timing().ratio, Ratio::new(1, 4).unwrap());
    assert_eq!(b.timing().delay, Delay::new(82, 2).unwrap()); // as plain QPSK (D658)
}

#[test]
fn psk_demod_variants_equal_the_kernels_whole_and_in_twenty_plans() {
    for (i, (order, differential, variant, sps)) in [
        (8u32, false, PskVariant::Plain, 4usize),
        (8, true, PskVariant::Plain, 5),
        (4, true, PskVariant::Pi4, 4),
        (4, false, PskVariant::Offset, 4),
        (4, false, PskVariant::Offset, 6),
    ]
    .into_iter()
    .enumerate()
    {
        let fs = rate(9600 * sps as u64);
        let cfg = PskDemodConfig {
            order,
            symbol_rate_hz: 9600.0,
            differential,
            variant,
            ..PskDemodConfig::default()
        };
        let input = received_v(
            order,
            differential,
            variant,
            sps,
            3000,
            M40_1_SEED,
            20 + i as u64,
        );
        let want = demod_kernels(&cfg, fs, &input);
        assert!(want.len() > 2900 * cfg.bits_per_symbol(), "case {i}");
        check_block(
            &want,
            &input,
            M40_1_SEED,
            || PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap(),
            &format!("psk_demod {order} {differential} {variant:?} {sps}"),
        );
        let mut b = PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
        run_block(&mut b, &input, None);
        b.reset();
        same_bits(&want, &run_block(&mut b, &input, None), "after reset");
    }
}

#[test]
fn psk_demod_8psk_keeps_soft_bits_that_do_not_fit() {
    let fs = rate(38_400);
    let cfg = PskDemodConfig {
        order: 8,
        symbol_rate_hz: 9600.0,
        ..PskDemodConfig::default()
    };
    let input = received_v(8, false, PskVariant::Plain, 4, 2000, M40_1_SEED, 30);
    let want = demod_kernels(&cfg, fs, &input);
    let mut block = PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
    for room_len in [1usize, 2, 4, 7] {
        block.reset();
        let mut got = Vec::new();
        let mut rest: &[C32] = &input;
        let mut room = vec![0.0f32; room_len];
        loop {
            let p = block.process(rest, &mut room).unwrap();
            got.extend_from_slice(&room[..p.produced]);
            rest = &rest[p.consumed..];
            if p.consumed == 0 && p.produced == 0 {
                break;
            }
        }
        assert!(rest.is_empty());
        same_bits(&want, &got, &format!("room {room_len}"));
    }
}

#[test]
fn oqpsk_chain_delay_counts_the_half_symbol_once() {
    // The delays that psk_mod and psk_demod report, summed as the pipeline sums a path, against
    // the lag of the soft bits behind the sent bits over the chain (D658). The quadrature part
    // is delayed in the transmitter and the in-phase part in the receiver, so OQPSK reports
    // exactly half a symbol (one bit) more than plain QPSK, and both sums stay within one
    // symbol of the measured lag (counting the half symbol in both blocks would not).
    for sps in [4usize, 8] {
        let mut sums = Vec::new();
        for variant in [PskVariant::Plain, PskVariant::Offset] {
            let m = PskModConfig {
                order: 4,
                samples_per_symbol: sps,
                variant,
                ..PskModConfig::default()
            };
            let d = PskDemodConfig {
                order: 4,
                symbol_rate_hz: 1000.0,
                variant,
                ..PskDemodConfig::default()
            };
            let bits = soft_bits_seed(6000, M40_1_SEED, 74);
            let mut tx = PskModBlock::<VecStorage>::new(&m, rate(2000), &mut VecStorage).unwrap();
            let mut rx =
                PskDemodBlock::<VecStorage>::new(&d, rate(1000 * sps as u64), &mut VecStorage)
                    .unwrap();
            let sig = run_block(&mut tx, &bits, None);
            let soft = run_block(&mut rx, &sig, None);
            let (lag, errors) = best_lag(&bits, &soft);
            assert_eq!(errors, 0, "sps {sps} {variant:?}: lag {lag}");
            let (t, r) = (tx.timing().delay, rx.timing().delay);
            // psk_mod counts input bits, psk_demod input samples (sps per 2 bits)
            let sum = t.as_f64() + r.as_f64() * 2.0 / sps as f64;
            assert!(
                (sum - lag as f64).abs() < 2.0,
                "sps {sps} {variant:?}: reported {sum} bits, measured {lag}"
            );
            sums.push(sum);
        }
        assert_eq!(sums[1] - sums[0], 1.0, "sps {sps}");
    }
}

// ---- M43: the option `equalizer_taps` of `psk_demod` -------------------------------------

/// Seed of the chunk plans and the random signals of the M43 tests (test 6).
const M43_SEED: u64 = 595;

/// M43 §3: `equalizer_taps` 0 (the default) leaves `psk_demod` bit-identical to the chain of
/// M40 and M40.1 (`demod_kernels` without the equalizer), with the same timing and storage.
#[test]
fn psk_demod_with_equalizer_taps_0_is_unchanged() {
    for (i, (order, differential, variant, sps)) in [
        (2u32, false, PskVariant::Plain, 4usize),
        (4, false, PskVariant::Plain, 8),
        (4, true, PskVariant::Plain, 5),
        (8, false, PskVariant::Plain, 4),
        (4, true, PskVariant::Pi4, 4),
        (4, false, PskVariant::Offset, 4),
    ]
    .into_iter()
    .enumerate()
    {
        let fs = rate(9600 * sps as u64);
        let base = PskDemodConfig {
            order,
            symbol_rate_hz: 9600.0,
            differential,
            variant,
            ..PskDemodConfig::default()
        };
        assert_eq!(base.equalizer_taps, 0);
        assert!(base.equalizer().is_none());
        let input = received_v(order, differential, variant, sps, 2000, M43_SEED, i as u64);
        let want = demod_kernels(&base, fs, &input);
        let mut b = PskDemodBlock::<VecStorage>::new(&base, fs, &mut VecStorage).unwrap();
        same_bits(
            &want,
            &run_block(&mut b, &input, None),
            &format!("case {i}"),
        );
        assert!(b.equalizer().is_none());
        // the timing of M40 and M40.1 (`offset` as plain QPSK, D658)
        let delay = (base.span * sps + 2) as u64;
        assert_eq!(b.timing().delay, Delay::new(delay, 2).unwrap(), "case {i}");
    }
}

/// M43 §3: with `equalizer_taps` the block equals the kernel chain with `equalize` before
/// `costas` (D825; for `offset` after its loop, for `differential` before the product in CMA
/// only), whole and in 20 cut plans of seed 595, and again after `reset`.
#[test]
fn psk_demod_with_equalizer_equals_the_kernels_whole_and_in_twenty_plans() {
    for (i, (order, differential, variant, sps, (taps, feedback))) in [
        (4u32, false, PskVariant::Plain, 4usize, (11usize, 0usize)),
        (2, false, PskVariant::Plain, 5, (3, 2)),
        (8, false, PskVariant::Plain, 4, (15, 3)),
        (2, true, PskVariant::Plain, 4, (7, 0)),
        (4, true, PskVariant::Pi4, 4, (11, 0)),
        (4, false, PskVariant::Offset, 4, (11, 2)),
    ]
    .into_iter()
    .enumerate()
    {
        let fs = rate(9600 * sps as u64);
        let cfg = PskDemodConfig {
            order,
            symbol_rate_hz: 9600.0,
            differential,
            variant,
            equalizer_taps: taps,
            equalizer_feedback_taps: feedback,
            ..PskDemodConfig::default()
        };
        let input = received_v(
            order,
            differential,
            variant,
            sps,
            3000,
            M43_SEED,
            10 + i as u64,
        );
        let want = demod_kernels(&cfg, fs, &input);
        assert!(want.len() > 2900 * cfg.bits_per_symbol(), "case {i}");
        let what = format!("psk_demod {order} {differential} {variant:?} taps {taps}");
        check_block(
            &want,
            &input,
            M43_SEED,
            || PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap(),
            &what,
        );
        let mut b = PskDemodBlock::<VecStorage>::new(&cfg, fs, &mut VecStorage).unwrap();
        run_block(&mut b, &input, None);
        let eq = b.equalizer().unwrap();
        assert_eq!((eq.taps(), eq.feedback_taps()), (taps, feedback));
        // coherent: CMA for 1000 symbols, then decisions; differential: CMA only
        let mode = if differential {
            EqMode::Cma
        } else {
            EqMode::Decision
        };
        assert_eq!(eq.mode(), mode, "{what}");
        assert_eq!(eq.cma_symbols(), if differential { u32::MAX } else { 1000 });
        assert_eq!((eq.mu_cma(), eq.mu_dd()), (0.05, 0.02));
        let bw = if differential { 0.0 } else { cfg.carrier_bw };
        assert_eq!(eq.params().phase_bw, bw);
        b.reset();
        assert_eq!(b.equalizer().unwrap().mode(), EqMode::Cma);
        same_bits(&want, &run_block(&mut b, &input, None), "after reset");
        // the equalizer adds (taps - 1) / 2 symbols to the delay
        assert_eq!(
            b.timing().delay,
            Delay::new((cfg.span * sps + 2 + (taps - 1) * sps) as u64, 2).unwrap(),
            "{what}"
        );
    }
}

/// M43 §3: with the equalizer on a clean signal `psk_demod` still decodes without errors, and
/// the reported delay of `psk_mod` -> `psk_demod` stays within one symbol of the measured lag.
#[test]
fn psk_demod_with_equalizer_decodes_a_clean_signal() {
    for (order, differential, variant, taps) in [
        (2u32, false, PskVariant::Plain, 11usize),
        (2, true, PskVariant::Plain, 11),
        (4, false, PskVariant::Plain, 11),
        (4, true, PskVariant::Plain, 15),
        (8, false, PskVariant::Plain, 11),
        (8, true, PskVariant::Plain, 3),
        (4, true, PskVariant::Pi4, 11),
        (4, false, PskVariant::Offset, 11),
    ] {
        let sps = 4;
        let m = PskModConfig {
            order,
            samples_per_symbol: sps,
            differential,
            variant,
            ..PskModConfig::default()
        };
        let d = PskDemodConfig {
            order,
            symbol_rate_hz: 1000.0,
            differential,
            variant,
            equalizer_taps: taps,
            ..PskDemodConfig::default()
        };
        let bits = soft_bits_seed(
            9000,
            M43_SEED,
            40 + u64::from(order) + u64::from(differential),
        );
        let bit_rate = rate(1000 * m.bits_per_symbol() as u64);
        let mut tx = PskModBlock::<VecStorage>::new(&m, bit_rate, &mut VecStorage).unwrap();
        let mut rx = PskDemodBlock::<VecStorage>::new(&d, rate(4000), &mut VecStorage).unwrap();
        let sig = run_block(&mut tx, &bits, None);
        let soft = run_block(&mut rx, &sig, None);
        let what = format!("order {order} differential {differential} {variant:?} taps {taps}");
        let (lag, errors) = best_lag(&bits, &soft);
        assert_eq!(errors, 0, "{what}: lag {lag}");
        let (t, r) = (tx.timing().delay, rx.timing().delay);
        let bps = m.bits_per_symbol() as f64;
        let sum = t.as_f64() + r.as_f64() * bps / sps as f64;
        assert!(
            (sum - lag as f64).abs() <= bps + 1.0,
            "{what}: reported {sum} bits, measured {lag}"
        );
        if !differential {
            assert!(rx.locked(), "{what}");
            assert!(rx.equalizer().unwrap().mse() < 0.05, "{what}");
        }
    }
}

/// The carrier-loop constellation of a PSK order (2, 4 or 8).
fn psk(order: u32) -> CostasConstellation {
    match order {
        2 => CostasConstellation::Bpsk,
        4 => CostasConstellation::Qpsk,
        8 => CostasConstellation::Psk8,
        _ => panic!("no PSK order {order}"),
    }
}
