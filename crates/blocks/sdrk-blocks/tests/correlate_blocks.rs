//! The block of M50: `CorrelateBlock` of the kind `correlate` (M50 §6, test 7; seed 706):
//! equality with the kernel `xcorr_fft` (through `Sdrk`) with the conjugated, reversed PRBS
//! taps (chip `i` the `i`-th `Prbs::next_bit`, chip value `1 - 2b`, shared conventions §B.1)
//! bit for bit on whole buffers, for every code and several samples per chip and FFT sizes;
//! `with_reference` with any complex reference; bit-exact chunk invariance against the block's
//! own whole run in 20 plans of input chunks and output room (0 to 3 blocks per call); a PRBS7
//! code at 2 samples per chip in noise (testkit `Awgn`) peaks at its start plus `L - 1`; the
//! automatic FFT size and its cap; validation at the parameter; exact storage; timing; the
//! parameters; `Send` and `Sync`. (The description rules of M09 and the size of the struct are
//! checked for every block in `describe.rs` and `construction.rs`.)
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod common;

use common::{check_exact_storage, rate};
use num_complex::Complex64;
use sdrk::{C32, Delay, Ratio, RatioKind, Sdrk};
use sdrk_blocks::{Block, CorrelateBlock, CorrelateConfig, Error, ParamValue, correlate_fft_size};
use sdrk_core::kernels::fft::FftPlan;
use sdrk_core::kernels::xcorr::{XcorrFft, XcorrFftState};
use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_testkit::channel::{Awgn, Impairment};
use sdrk_testkit::{Pcg32, VecStorage};

/// Seed of the signals, the chunk plans and the noise (M50 range 700 to 709; test 7 of the
/// block takes 706).
const SEED: u64 = 706;

/// Every code polynomial with the period of its code.
const CODES: [(Poly, usize); 6] = [
    (Poly::Prbs5, 31),
    (Poly::Prbs7, 127),
    (Poly::Prbs9, 511),
    (Poly::Prbs10, 1023),
    (Poly::Prbs11, 2047),
    (Poly::Prbs15, 32767),
];

fn noise(n: usize, seed: u64) -> Vec<C32> {
    let mut rng = Pcg32::new(seed, 3);
    (0..n)
        .map(|_| C32::new(rng.normal() as f32 * 0.5, rng.normal() as f32 * 0.5))
        .collect()
}

/// The reference of a configuration, written from the specification: chip `i` is the `i`-th
/// `next_bit` after `Prbs::new(code, seed)`, value `1 - 2b`, imaginary part 0, every chip
/// `samples_per_chip` times.
fn prbs_reference(cfg: &CorrelateConfig) -> Vec<C32> {
    let mut prbs = Prbs::new(cfg.code, cfg.seed).unwrap();
    let chips = (1usize << cfg.code.degree()) - 1;
    let mut r = Vec::new();
    for _ in 0..chips {
        let b = prbs.next_bit();
        let v = if b == 0 { 1.0 } else { -1.0 };
        for _ in 0..cfg.samples_per_chip {
            r.push(C32::new(v, 0.0));
        }
    }
    r
}

/// The kernel through `Sdrk` with the taps `conj(r[L - 1 - k])` (built with `new`), over the
/// whole input in one call.
fn kernel_output(reference: &[C32], n: usize, x: &[C32]) -> Vec<C32> {
    let l = reference.len();
    let taps: Vec<C32> = reference
        .iter()
        .rev()
        .map(num_complex::Complex::conj)
        .collect();
    let plan = FftPlan::new(n, vec![C32::default(); n / 2]).unwrap();
    let mut s = XcorrFftState::new(
        &taps,
        vec![C32::default(); n],
        vec![C32::default(); l - 1],
        vec![C32::default(); n],
        plan,
    )
    .unwrap();
    let mut out = vec![C32::default(); x.len()];
    let p = Sdrk.xcorr_fft(&mut s, x, &mut out).unwrap();
    out.truncate(p.produced);
    out
}

/// The block over the whole input in one call.
fn whole(block: &mut CorrelateBlock<VecStorage>, x: &[C32]) -> Vec<C32> {
    let mut out = vec![C32::default(); x.len()];
    let p = block.process(x, &mut out).unwrap();
    assert_eq!(p.consumed, p.produced);
    assert_eq!(
        p.produced,
        x.len() / block.block_size() * block.block_size()
    );
    out.truncate(p.produced);
    out
}

fn bits(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// The configurations of the equality tests: every code, 1 to 3 samples per chip, automatic
/// and manual FFT sizes (the smallest `N >= 2 L` among them).
fn configs() -> Vec<CorrelateConfig> {
    let c = |code, seed, samples_per_chip, fft_size| CorrelateConfig {
        code,
        seed,
        samples_per_chip,
        fft_size,
    };
    vec![
        c(Poly::Prbs5, 1, 1, 0),
        c(Poly::Prbs5, 17, 3, 256),
        c(Poly::Prbs7, 1, 1, 0),
        c(Poly::Prbs7, 100, 2, 512),
        c(Poly::Prbs9, 511, 1, 1024),
        c(Poly::Prbs10, 1, 2, 0),
        c(Poly::Prbs11, 1234, 1, 4096),
        c(Poly::Prbs15, 1, 1, 0),
    ]
}

#[test]
fn the_block_equals_the_kernel_with_the_conjugated_reversed_code() {
    let r = rate(48_000);
    for (i, cfg) in configs().into_iter().enumerate() {
        let mut block = CorrelateBlock::new(&cfg, r, &mut VecStorage).unwrap();
        let reference = prbs_reference(&cfg);
        assert_eq!(block.reference_len(), reference.len(), "{cfg:?}");
        assert_eq!(cfg.reference_len().unwrap(), reference.len(), "{cfg:?}");
        let n = block.fft_size();
        assert_eq!(n, cfg.resolved_fft_size().unwrap(), "{cfg:?}");
        let b = block.block_size();
        assert_eq!(b, n - reference.len() + 1);
        // Three whole blocks and a remainder that stays unconsumed.
        let x = noise(3 * b + b / 2, SEED * 10 + i as u64);
        let want = kernel_output(&reference, n, &x);
        let got = whole(&mut block, &x);
        assert_eq!(bits(&got), bits(&want), "{cfg:?}");
    }
}

#[test]
fn a_caller_reference_equals_the_kernel() {
    let mut rng = Pcg32::new(SEED, 5);
    for (l, fft_size) in [(1, 0), (2, 0), (5, 16), (64, 0), (300, 1024), (1000, 2048)] {
        let reference: Vec<C32> = (0..l)
            .map(|_| C32::new(rng.normal() as f32, rng.normal() as f32))
            .collect();
        let mut block =
            CorrelateBlock::with_reference(&reference, fft_size, &mut VecStorage).unwrap();
        let n = correlate_fft_size(l, fft_size).unwrap();
        assert_eq!((block.reference_len(), block.fft_size()), (l, n));
        let x = noise(4 * block.block_size(), SEED * 20 + l as u64);
        let want = kernel_output(&reference, n, &x);
        assert_eq!(bits(&whole(&mut block, &x)), bits(&want), "L {l}");
        // `fft_size` reads as given; the code parameters do not exist for such a block.
        assert_eq!(
            block.param("fft_size").unwrap(),
            ParamValue::I64(fft_size as i64)
        );
        for name in ["code", "seed", "samples_per_chip"] {
            assert!(
                matches!(block.param(name), Err(Error::InvalidParameter { .. })),
                "{name}"
            );
        }
    }
    assert!(matches!(
        CorrelateBlock::<VecStorage>::with_reference(&[], 0, &mut VecStorage),
        Err(Error::InvalidParameter {
            name: "reference",
            ..
        })
    ));
}

/// Runs the block over `x` in the input chunks of `chunks`; each call gets an output room of
/// `k B + r` samples (`k` 0 to 3 blocks, `r` below `B`) from `rng`; the unconsumed rest is
/// resubmitted with the next chunk. Returns the output and the history after every call.
fn chunked(
    block: &mut CorrelateBlock<VecStorage>,
    x: &[C32],
    chunks: &mut dyn Iterator<Item = usize>,
    rng: &mut Pcg32,
) -> (Vec<C32>, Vec<Vec<C32>>) {
    let b = block.block_size();
    let mut out = Vec::new();
    let mut histories = Vec::new();
    let mut room = vec![C32::default(); 4 * b];
    let mut pending: Vec<C32> = Vec::new();
    let mut at = 0;
    while at < x.len() {
        let end = (at + chunks.next().unwrap()).min(x.len());
        pending.extend_from_slice(&x[at..end]);
        at = end;
        while pending.len() >= b {
            let k = rng.next_u32() as usize % 4;
            let r = rng.next_u32() as usize % b;
            let p = block.process(&pending, &mut room[..k * b + r]).unwrap();
            assert_eq!(p.consumed, p.produced);
            assert_eq!(p.produced, k.min(pending.len() / b) * b);
            out.extend_from_slice(&room[..p.produced]);
            pending.drain(..p.consumed);
            if p.produced > 0 {
                histories.push(block.state().history().to_vec());
            }
        }
    }
    (out, histories)
}

#[test]
fn the_block_is_chunk_invariant_with_20_plans() {
    let r = rate(48_000);
    for cfg in [
        CorrelateConfig::default(),
        CorrelateConfig {
            code: Poly::Prbs5,
            samples_per_chip: 4,
            fft_size: 256,
            ..CorrelateConfig::default()
        },
    ] {
        let mut block = CorrelateBlock::new(&cfg, r, &mut VecStorage).unwrap();
        let b = block.block_size();
        let x = noise(9 * b + 17, SEED + 1);
        let mut one = Pcg32::new(SEED, 9);
        let (want, want_hist) = chunked(&mut block, &x, &mut std::iter::repeat(x.len()), &mut one);
        assert_eq!(want.len(), 9 * b);
        for k in 0..20u64 {
            let plan = sdrk_testkit::ChunkPlan::new(SEED * 100 + k, 1 + 97 * k as usize);
            let mut rng = Pcg32::new(SEED * 1000 + k, 9);
            block.reset();
            let (got, hist) = chunked(&mut block, &x, &mut plan.sizes(), &mut rng);
            assert_eq!(bits(&got), bits(&want), "{cfg:?}, plan {k}");
            // Every block of the stream ends in the same history, however it was cut.
            let last = |h: &[Vec<C32>]| bits(h.last().unwrap());
            assert_eq!(last(&hist), last(&want_hist), "{cfg:?}, plan {k}");
        }
        // After `reset` the block equals a new one.
        block.reset();
        let mut fresh = CorrelateBlock::new(&cfg, r, &mut VecStorage).unwrap();
        assert_eq!(bits(&whole(&mut block, &x)), bits(&whole(&mut fresh, &x)));
    }
}

#[test]
fn a_prbs7_code_at_two_samples_per_chip_peaks_at_its_start_plus_l_minus_1() {
    let cfg = CorrelateConfig {
        samples_per_chip: 2,
        ..CorrelateConfig::default()
    };
    let reference = prbs_reference(&cfg);
    let l = reference.len();
    assert_eq!(l, 254);
    let r = rate(48_000);
    for trial in 0..10u64 {
        let mut block = CorrelateBlock::new(&cfg, r, &mut VecStorage).unwrap();
        let b = block.block_size();
        let len = 3 * b;
        let m = 50 + 131 * trial as usize;
        let mut clean = vec![Complex64::new(0.0, 0.0); len];
        for (i, v) in reference.iter().enumerate() {
            clean[m + i] = Complex64::new(f64::from(v.re), 0.0);
        }
        // Chip SNR 0 dB: unit chips, noise power 1 per sample.
        let mut awgn = Awgn::new(core::f64::consts::FRAC_1_SQRT_2, SEED * 100 + trial).unwrap();
        let mut noisy = Vec::new();
        awgn.process(&clean, &mut noisy);
        let x: Vec<C32> = noisy
            .iter()
            .map(|v| C32::new(v.re as f32, v.im as f32))
            .collect();
        let y = whole(&mut block, &x);
        let (best, _) = y
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert_eq!(best, m + l - 1, "trial {trial}");
    }
}

#[test]
fn the_automatic_fft_size_is_4_l_capped_at_65536() {
    let r = rate(48_000);
    for ((code, l), n) in CODES.into_iter().zip([128, 512, 2048, 4096, 8192, 65_536]) {
        let cfg = CorrelateConfig {
            code,
            ..CorrelateConfig::default()
        };
        assert_eq!(cfg.reference_len().unwrap(), l, "{code:?}");
        assert_eq!(cfg.resolved_fft_size().unwrap(), n, "{code:?}");
    }
    // The four lengths of the spec, through the block (L 32767 is capped: 4 L is 131068).
    for (code, n) in [
        (Poly::Prbs5, 128),
        (Poly::Prbs7, 512),
        (Poly::Prbs10, 4096),
        (Poly::Prbs15, 65_536),
    ] {
        let cfg = CorrelateConfig {
            code,
            ..CorrelateConfig::default()
        };
        let block = CorrelateBlock::new(&cfg, r, &mut VecStorage).unwrap();
        assert_eq!(block.fft_size(), n, "{code:?}");
    }
    // The rule at its edges: at least 16, the cap from L 16385 on, an error above L 32768.
    assert_eq!(correlate_fft_size(1, 0).unwrap(), 16);
    assert_eq!(correlate_fft_size(4, 0).unwrap(), 16);
    assert_eq!(correlate_fft_size(5, 0).unwrap(), 32);
    assert_eq!(correlate_fft_size(16_384, 0).unwrap(), 65_536);
    assert_eq!(correlate_fft_size(16_385, 0).unwrap(), 65_536);
    assert_eq!(correlate_fft_size(32_768, 0).unwrap(), 65_536);
    assert_eq!(
        correlate_fft_size(32_769, 0),
        Err(Error::OutOfRange {
            name: "fft_size",
            min: 131_072.0,
            max: 65_536.0,
            got: 0.0
        })
    );
}

#[test]
fn invalid_parameters_are_errors_at_their_parameter() {
    let r = rate(48_000);
    let base = CorrelateConfig::default();
    let name = |cfg: CorrelateConfig| match cfg.validate(r) {
        Err(Error::OutOfRange { name, .. } | Error::InvalidParameter { name, .. }) => name,
        other => panic!("{cfg:?}: {other:?}"),
    };
    for (cfg, want) in [
        (CorrelateConfig { seed: 0, ..base }, "seed"),
        (
            CorrelateConfig {
                seed: 32_768,
                ..base
            },
            "seed",
        ),
        // In range, but the low 5 bits of 32 are zero: no state for prbs5.
        (
            CorrelateConfig {
                code: Poly::Prbs5,
                seed: 32,
                ..base
            },
            "seed",
        ),
        (
            CorrelateConfig {
                samples_per_chip: 0,
                ..base
            },
            "samples_per_chip",
        ),
        (
            CorrelateConfig {
                samples_per_chip: 17,
                ..base
            },
            "samples_per_chip",
        ),
        (
            CorrelateConfig {
                fft_size: 8,
                ..base
            },
            "fft_size",
        ),
        (
            CorrelateConfig {
                fft_size: 300,
                ..base
            },
            "fft_size",
        ),
        (
            CorrelateConfig {
                fft_size: 128,
                ..base
            },
            "fft_size",
        ),
        (
            CorrelateConfig {
                fft_size: 131_072,
                ..base
            },
            "fft_size",
        ),
    ] {
        assert_eq!(name(cfg), want, "{cfg:?}");
        assert!(CorrelateBlock::new(&cfg, r, &mut VecStorage).is_err());
        assert!(CorrelateBlock::<VecStorage>::storage_bytes(&cfg, r).is_err());
    }
    // A size below 2 L names the required size.
    assert_eq!(
        CorrelateConfig {
            fft_size: 128,
            ..base
        }
        .validate(r),
        Err(Error::OutOfRange {
            name: "fft_size",
            min: 256.0,
            max: 65_536.0,
            got: 128.0
        })
    );
    // prbs15 at 2 samples per chip (L 65534) needs 131072 points, more than the cap.
    assert_eq!(
        CorrelateConfig {
            code: Poly::Prbs15,
            samples_per_chip: 2,
            ..base
        }
        .validate(r),
        Err(Error::OutOfRange {
            name: "fft_size",
            min: 131_072.0,
            max: 65_536.0,
            got: 0.0
        })
    );
    // The limits are valid: 2 L exactly, the largest prbs11 code, the largest seed.
    for cfg in [
        CorrelateConfig {
            fft_size: 256,
            ..base
        },
        CorrelateConfig {
            code: Poly::Prbs11,
            samples_per_chip: 16,
            seed: 2047,
            ..base
        },
        CorrelateConfig {
            code: Poly::Prbs15,
            seed: 32_767,
            fft_size: 65_536,
            ..base
        },
    ] {
        assert!(cfg.validate(r).is_ok(), "{cfg:?}");
    }
}

#[test]
fn the_storage_is_exact() {
    let r = rate(48_000);
    for cfg in configs().into_iter().take(7) {
        let bytes = CorrelateBlock::<VecStorage>::storage_bytes(&cfg, r).unwrap();
        let (l, n) = (
            cfg.reference_len().unwrap(),
            cfg.resolved_fft_size().unwrap(),
        );
        assert_eq!(bytes, (n + (l - 1) + n + n / 2) * 8, "{cfg:?}");
        check_exact_storage(bytes, |arena| {
            CorrelateBlock::new(&cfg, r, arena).map(|_| ())
        });
    }
    let reference = noise(40, SEED);
    for fft_size in [0, 128] {
        let bytes = CorrelateBlock::<VecStorage>::reference_storage_bytes(40, fft_size).unwrap();
        check_exact_storage(bytes, |arena| {
            CorrelateBlock::with_reference(&reference, fft_size, arena).map(|_| ())
        });
    }
    // The memory figures of the rustdoc: a 1023-chip code at 2 samples per chip.
    let code = CorrelateConfig {
        code: Poly::Prbs10,
        samples_per_chip: 2,
        ..CorrelateConfig::default()
    };
    let kib = |cfg: &CorrelateConfig| {
        CorrelateBlock::<VecStorage>::storage_bytes(cfg, r).unwrap() as f64 / 1024.0
    };
    assert_eq!(kib(&code).round(), 176.0);
    assert_eq!(
        kib(&CorrelateConfig {
            fft_size: 4096,
            ..code
        })
        .round(),
        96.0
    );
}

#[test]
fn timing_and_parameters() {
    let r = rate(48_000);
    let cfg = CorrelateConfig {
        code: Poly::Prbs9,
        seed: 77,
        samples_per_chip: 3,
        fft_size: 4096,
    };
    let mut block = CorrelateBlock::new(&cfg, r, &mut VecStorage).unwrap();
    let t = block.timing();
    assert_eq!(t.ratio, Ratio::ONE);
    assert_eq!(t.kind, RatioKind::Constant);
    assert_eq!(t.delay, Delay::new(1532, 1).unwrap());
    assert_eq!(block.min_input(), 4096 - 1533 + 1);
    assert_eq!(CorrelateBlock::<VecStorage>::rate_out(&cfg, r).unwrap(), r);
    assert_eq!(block.param("code").unwrap(), ParamValue::Choice(2));
    assert_eq!(block.param("seed").unwrap(), ParamValue::I64(77));
    assert_eq!(block.param("samples_per_chip").unwrap(), ParamValue::I64(3));
    assert_eq!(block.param("fft_size").unwrap(), ParamValue::I64(4096));
    assert_eq!(
        block.set_param("seed", ParamValue::I64(2)),
        Err(Error::ReadOnlyParameter)
    );
    assert_eq!(
        block.set_param("nope", ParamValue::I64(2)),
        Err(Error::UnknownParameter)
    );
    // An input or an output room shorter than B gives nothing.
    let b = block.block_size();
    let x = noise(b, SEED);
    let mut y = vec![C32::default(); b];
    assert_eq!(block.process(&x[..b - 1], &mut y).unwrap().consumed, 0);
    assert_eq!(block.process(&x, &mut y[..b - 1]).unwrap().produced, 0);
    assert_eq!(block.process(&x, &mut y).unwrap().produced, b);
}

#[test]
fn the_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CorrelateBlock<VecStorage>>();
    assert_send_sync::<CorrelateBlock<sdrk_blocks::ArenaStorage<'static>>>();
    assert_send_sync::<CorrelateConfig>();
}
