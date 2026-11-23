//! Tests of the M50 kernel `xcorr_fft` of `sdrk-core` (tests 2, 4 and 5 of M50): the construction
//! rules of `new`, `from_spectrum` and `xcorr_spectrum`, timing, `block_size`, the consumption rule
//! (whole blocks only, limited by input and output), an impulse and a single tap (the kernel is a
//! plain convolution, no shift), the fallback against the f64 direct convolution written here and
//! against `Reference` at several `L` and `N`, `reset`, `from_spectrum` equal to `new` bit for bit,
//! block-size invariance in 20 chunk plans (seed 705, an output room of 0 to 3 blocks per call; the
//! output and the history after every call equal to the whole run bit for bit), the extremes
//! (silence, a full-scale impulse, NaN and infinities in one sample: at most the two blocks whose
//! windows hold it are spoiled, every later block equals the clean run bit for bit; the largest
//! `N` 65536 with `L` 32768) and the thread-safety of the public types. The comparison with
//! `Reference` runs only with the feature `reference`. Signals and taps: seed 709.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]

use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::kernels::xcorr::{
    MAX_FFT_SIZE, MIN_FFT_SIZE, XcorrFft, XcorrFftState, spectral_product, spectral_product_one,
    xcorr_spectrum,
};
use sdrk_core::{C32, Delay, Error, Fallback, Progress, Ratio, Timing};
use sdrk_device::{ErrorKind, SdrkError};

/// The seed of the chunk plans (M50 test 4); plan `k` uses `SEED_PLANS * 100 + k`.
const SEED_PLANS: u64 = 705;
/// The seed of the signals and taps of these tests.
const SEED: u64 = 709;
/// The contract floor in dB.
const FLOOR_DB: f64 = 100.0;

type State = XcorrFftState<C32, Vec<C32>, Vec<C32>, Vec<C32>, Vec<C32>>;
/// A state over a borrowed spectrum (`from_spectrum`).
type SpectrumState<'a> = XcorrFftState<C32, &'a [C32], Vec<C32>, Vec<C32>, Vec<C32>>;

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }

    fn below(&mut self, n: usize) -> usize {
        ((f32::midpoint(self.next(), 1.0) * n as f32) as usize).min(n - 1)
    }
}

fn noise(rng: &mut Rng, n: usize, amp: f32) -> Vec<C32> {
    (0..n)
        .map(|_| C32::new(amp * rng.next(), amp * rng.next()))
        .collect()
}

fn plan(n: usize) -> FftPlan<C32, Vec<C32>> {
    FftPlan::new(n, vec![C32::default(); n / 2]).unwrap()
}

fn z(n: usize) -> Vec<C32> {
    vec![C32::default(); n]
}

fn state(taps: &[C32], n: usize) -> State {
    XcorrFftState::new(taps, z(n), z(taps.len() - 1), z(n), plan(n)).unwrap()
}

/// The whole input (a multiple of `B`) in one call.
fn run_whole<K: XcorrFft<C32>>(k: &K, st: &mut State, input: &[C32]) -> Vec<C32> {
    let mut out = z(input.len());
    let p = k.xcorr_fft(st, input, &mut out).unwrap();
    assert_eq!(
        p,
        Progress {
            consumed: input.len(),
            produced: input.len()
        }
    );
    out
}

/// `y[n] = sum_k h[k] x[n - k]` in f64 with `x[n] = 0` for `n < 0`.
fn direct(h: &[C32], x: &[C32]) -> Vec<(f64, f64)> {
    (0..x.len())
        .map(|n| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (k, t) in h.iter().enumerate().take(n + 1) {
                let v = x[n - k];
                let (a, b) = (f64::from(t.re), f64::from(t.im));
                let (c, d) = (f64::from(v.re), f64::from(v.im));
                re += a * c - b * d;
                im += a * d + b * c;
            }
            (re, im)
        })
        .collect()
}

fn snr_db(want: &[(f64, f64)], got: &[C32]) -> f64 {
    assert_eq!(want.len(), got.len());
    let (mut p, mut e) = (0.0f64, 0.0f64);
    for (&(re, im), g) in want.iter().zip(got) {
        p += re * re + im * im;
        let (dr, di) = (re - f64::from(g.re), im - f64::from(g.im));
        e += dr * dr + di * di;
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (p / e).log10()
    }
}

#[cfg(feature = "reference")]
fn as_f64(x: &[C32]) -> Vec<(f64, f64)> {
    x.iter()
        .map(|v| (f64::from(v.re), f64::from(v.im)))
        .collect()
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// The `(L, N)` pairs of the direct-convolution tests.
const CONFIGS: [(usize, usize); 9] = [
    (1, 16),
    (2, 16),
    (8, 16),
    (5, 64),
    (65, 256),
    (127, 512),
    (300, 1024),
    (128, 1024),
    (1023, 4096),
];

// --------------------------------------------------------------------------------------------
// Construction
// --------------------------------------------------------------------------------------------

fn kind(r: Result<State, Error>) -> ErrorKind {
    r.err().unwrap().kind()
}

fn kind_s(r: Result<SpectrumState<'_>, Error>) -> ErrorKind {
    r.err().unwrap().kind()
}

#[test]
fn new_and_from_spectrum_reject_invalid_sizes_and_storage() {
    let taps = |l: usize| z(l);
    let table = z(64);
    // L = 0.
    assert!(matches!(
        XcorrFftState::new(&taps(0), z(64), z(0), z(64), plan(64)),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(
        kind_s(XcorrFftState::from_spectrum(
            &table[..],
            0,
            z(0),
            z(64),
            plan(64)
        )),
        ErrorKind::InvalidArgument
    );
    // N < 2 L.
    assert_eq!(
        kind(XcorrFftState::new(&taps(33), z(64), z(32), z(64), plan(64))),
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        kind_s(XcorrFftState::from_spectrum(
            &table[..],
            33,
            z(32),
            z(64),
            plan(64)
        )),
        ErrorKind::InvalidArgument
    );
    // N = 2 L is the smallest valid size.
    assert!(XcorrFftState::new(&taps(32), z(64), z(31), z(64), plan(64)).is_ok());
    assert!(XcorrFftState::from_spectrum(&table[..], 32, z(31), z(64), plan(64)).is_ok());
    // N not a power of two or outside 16 ..= 65536: no plan, so no state.
    for n in [24usize, 100, 8, 4, 131_072] {
        assert!(matches!(
            FftPlan::<C32, _>::new(n, z(n / 2)),
            Err(Error::InvalidParameter(_))
        ));
    }
    assert_eq!((MIN_FFT_SIZE, MAX_FFT_SIZE), (16, 65_536));
    // Storage of the wrong length: spectrum, history, scratch.
    for (h, hist, scr) in [
        (63, 9, 64),
        (65, 9, 64),
        (64, 8, 64),
        (64, 10, 64),
        (64, 9, 63),
        (64, 9, 128),
    ] {
        assert!(
            matches!(
                XcorrFftState::new(&taps(10), z(h), z(hist), z(scr), plan(64)),
                Err(Error::LengthMismatch { .. })
            ),
            "{h} {hist} {scr}"
        );
        let t = z(h);
        assert!(
            matches!(
                XcorrFftState::from_spectrum(&t[..], 10, z(hist), z(scr), plan(64)),
                Err(Error::LengthMismatch { .. })
            ),
            "{h} {hist} {scr}"
        );
    }
    // xcorr_spectrum: the same rules.
    let mut h = z(64);
    assert!(xcorr_spectrum(&taps(0), &mut h, &plan(64)).is_err());
    assert!(xcorr_spectrum(&taps(33), &mut h, &plan(64)).is_err());
    assert!(matches!(
        xcorr_spectrum(&taps(10), &mut z(32), &plan(64)),
        Err(Error::LengthMismatch {
            expected: 64,
            got: 32
        })
    ));
}

#[test]
fn accessors_timing_and_block_size() {
    let mut rng = Rng::new(SEED);
    for &(l, n) in &CONFIGS {
        let taps = noise(&mut rng, l, 1.0);
        let mut st = state(&taps, n);
        assert_eq!(st.taps_len(), l);
        assert_eq!(st.fft_size(), n);
        assert_eq!(st.block_size(), n - l + 1);
        assert_eq!(st.plan().len(), n);
        assert_eq!(st.spectrum().len(), n);
        assert_eq!(st.history().len(), l - 1);
        assert!(st.history().iter().all(|v| *v == C32::default()));
        assert_eq!(st.ratio(), Ratio::ONE);
        assert_eq!(st.group_delay(), Delay::new(l as u64 - 1, 1).unwrap());
        // The history after a call is the last L - 1 input samples.
        let b = st.block_size();
        let x = noise(&mut rng, 2 * b, 1.0);
        run_whole(&Fallback, &mut st, &x);
        assert_eq!(bits(st.history()), bits(&x[2 * b - (l - 1)..]));
        st.reset();
        assert!(
            st.history()
                .iter()
                .all(|v| v.re.to_bits() == 0 && v.im.to_bits() == 0)
        );
    }
}

#[test]
fn the_spectrum_is_the_fallback_fft_and_from_spectrum_equals_new() {
    let mut rng = Rng::new(SEED + 1);
    for &(l, n) in &CONFIGS {
        let taps = noise(&mut rng, l, 1.0);
        let mut st = state(&taps, n);
        let mut h = z(n);
        xcorr_spectrum(&taps, &mut h, &plan(n)).unwrap();
        assert_eq!(bits(&h), bits(st.spectrum()), "L {l}, N {n}");
        // The same spectrum by hand: the taps padded, the fallback forward FFT.
        let mut manual = z(n);
        manual[..l].copy_from_slice(&taps);
        Fallback
            .fft(&plan(n), &mut manual, Direction::Forward, Scaling::Default)
            .unwrap();
        assert_eq!(bits(&manual), bits(&h));
        let mut fs = XcorrFftState::from_spectrum(&h[..], l, z(l - 1), z(n), plan(n)).unwrap();
        let b = st.block_size();
        let x = noise(&mut rng, 3 * b, 1.0);
        let a = run_whole(&Fallback, &mut st, &x);
        let mut out = z(x.len());
        Fallback.xcorr_fft(&mut fs, &x, &mut out).unwrap();
        assert_eq!(bits(&a), bits(&out), "L {l}, N {n}");
        assert_eq!(bits(st.history()), bits(fs.history()));
    }
}

#[test]
fn the_spectral_product_is_the_fixed_formula() {
    let mut rng = Rng::new(SEED + 2);
    let w = noise(&mut rng, 67, 3.0);
    let h = noise(&mut rng, 67, 3.0);
    let mut y = w.clone();
    spectral_product(&mut y, &h).unwrap();
    for ((a, b), c) in w.iter().zip(&h).zip(&y) {
        let re = a.re * b.re - a.im * b.im;
        let im = a.re * b.im + a.im * b.re;
        assert_eq!(
            (c.re.to_bits(), c.im.to_bits()),
            (re.to_bits(), im.to_bits())
        );
        assert_eq!(bits(&[spectral_product_one(*a, *b)]), bits(&[*c]));
    }
    assert!(matches!(
        spectral_product(&mut y, &h[..66]),
        Err(Error::LengthMismatch { .. })
    ));
}

// --------------------------------------------------------------------------------------------
// The convolution
// --------------------------------------------------------------------------------------------

#[test]
fn an_impulse_reproduces_the_taps_without_a_shift() {
    let mut rng = Rng::new(SEED + 3);
    for &(l, n) in &CONFIGS {
        let taps = noise(&mut rng, l, 1.0);
        let mut st = state(&taps, n);
        let b = st.block_size();
        let len = 3 * b;
        for m in [0, 1, b - 1, b, b + l / 2, 2 * b - 1] {
            st.reset();
            let mut x = z(len);
            x[m] = C32::new(1.0, 0.0);
            let out = run_whole(&Fallback, &mut st, &x);
            let mut want = vec![(0.0f64, 0.0f64); len];
            for (k, t) in taps.iter().enumerate() {
                if m + k < len {
                    want[m + k] = (f64::from(t.re), f64::from(t.im));
                }
            }
            let snr = snr_db(&want, &out);
            assert!(snr >= FLOOR_DB, "L {l}, N {n}, m {m}: {snr:.1} dB");
        }
    }
}

#[test]
fn a_single_tap_is_a_scaled_delay() {
    let mut rng = Rng::new(SEED + 4);
    for &(l, n) in &CONFIGS {
        for d in [0, l / 2, l - 1] {
            let c = C32::new(0.75, -0.5);
            let mut taps = z(l);
            taps[d] = c;
            let mut st = state(&taps, n);
            let len = 2 * st.block_size();
            let x = noise(&mut rng, len, 1.0);
            let out = run_whole(&Fallback, &mut st, &x);
            let want: Vec<(f64, f64)> = (0..len)
                .map(|i| {
                    if i < d {
                        (0.0, 0.0)
                    } else {
                        let v = x[i - d];
                        let (a, b) = (f64::from(c.re), f64::from(c.im));
                        let (p, q) = (f64::from(v.re), f64::from(v.im));
                        (a * p - b * q, a * q + b * p)
                    }
                })
                .collect();
            let snr = snr_db(&want, &out);
            assert!(snr >= FLOOR_DB, "L {l}, N {n}, d {d}: {snr:.1} dB");
        }
    }
}

#[test]
fn the_fallback_is_the_direct_convolution() {
    let mut rng = Rng::new(SEED + 5);
    for &(l, n) in &CONFIGS {
        let taps = noise(&mut rng, l, 1.0);
        let mut st = state(&taps, n);
        let x = noise(&mut rng, 3 * st.block_size(), 1.0);
        let out = run_whole(&Fallback, &mut st, &x);
        let snr = snr_db(&direct(&taps, &x), &out);
        assert!(snr >= FLOOR_DB, "L {l}, N {n}: {snr:.1} dB");
    }
}

#[cfg(feature = "reference")]
#[test]
fn the_fallback_agrees_with_the_reference() {
    use sdrk_core::Reference;
    let mut rng = Rng::new(SEED + 6);
    for &(l, n) in &CONFIGS {
        let taps = noise(&mut rng, l, 1.0);
        let x = noise(&mut rng, 3 * (n - l + 1), 1.0);
        let mut a = state(&taps, n);
        let mut r = state(&taps, n);
        let f = run_whole(&Fallback, &mut a, &x);
        let refr = run_whole(&Reference, &mut r, &x);
        let snr = snr_db(&as_f64(&refr), &f);
        assert!(snr >= FLOOR_DB, "L {l}, N {n}: {snr:.1} dB");
        // The reference itself is the direct convolution of the taps up to the f32 error of H.
        let snr = snr_db(&direct(&taps, &x), &refr);
        assert!(snr >= 120.0, "reference L {l}, N {n}: {snr:.1} dB");
        // The same history and the same consumption.
        assert_eq!(bits(a.history()), bits(r.history()));
        // A reference state from a spectrum gives the same values as one from `new`.
        let mut h = z(n);
        xcorr_spectrum(&taps, &mut h, &plan(n)).unwrap();
        let mut fs = XcorrFftState::from_spectrum(&h[..], l, z(l - 1), z(n), plan(n)).unwrap();
        let mut out = z(x.len());
        Reference.xcorr_fft(&mut fs, &x, &mut out).unwrap();
        assert_eq!(bits(&out), bits(&refr));
    }
}

#[test]
fn reset_gives_the_output_of_a_fresh_state() {
    let mut rng = Rng::new(SEED + 7);
    let taps = noise(&mut rng, 127, 1.0);
    let mut st = state(&taps, 512);
    let b = st.block_size();
    let x = noise(&mut rng, 3 * b, 1.0);
    let first = run_whole(&Fallback, &mut st, &x);
    let y = noise(&mut rng, 2 * b, 1.0);
    run_whole(&Fallback, &mut st, &y);
    st.reset();
    let again = run_whole(&Fallback, &mut st, &x);
    assert_eq!(bits(&first), bits(&again));
}

// --------------------------------------------------------------------------------------------
// Consumption
// --------------------------------------------------------------------------------------------

#[test]
fn only_whole_blocks_are_consumed() {
    let mut rng = Rng::new(SEED + 8);
    let taps = noise(&mut rng, 65, 1.0);
    let mut st = state(&taps, 256);
    let b = st.block_size();
    assert_eq!(b, 192);
    let x = noise(&mut rng, 5 * b + 7, 1.0);
    // An input or an output shorter than B: nothing.
    let mut out = z(5 * b);
    for (i, o) in [
        (b - 1, 5 * b),
        (5 * b, b - 1),
        (0, 5 * b),
        (5 * b, 0),
        (b - 1, b - 1),
    ] {
        let before = bits(st.history());
        let p = Fallback.xcorr_fft(&mut st, &x[..i], &mut out[..o]).unwrap();
        assert_eq!(p, Progress::default(), "input {i}, output {o}");
        assert_eq!(bits(st.history()), before);
    }
    // Room for k blocks consumes exactly k B, limited by the smaller of input and output.
    for (i, o, k) in [
        (b, b, 1),
        (2 * b - 1, 5 * b, 1),
        (5 * b + 7, 3 * b + 5, 3),
        (4 * b + 3, 5 * b, 4),
        (5 * b + 7, 5 * b, 5),
    ] {
        st.reset();
        let p = Fallback.xcorr_fft(&mut st, &x[..i], &mut out[..o]).unwrap();
        assert_eq!(
            p,
            Progress {
                consumed: k * b,
                produced: k * b
            },
            "input {i}, output {o}"
        );
    }
}

// --------------------------------------------------------------------------------------------
// Block-size invariance
// --------------------------------------------------------------------------------------------

/// Runs `x` (a multiple of `B`) in chunks of 1 to `max` samples (plan `seed`) with an output room
/// of 0 to 3 blocks per call; the samples a call does not consume are resubmitted. After every
/// call the output so far and the history must equal those of the whole run.
fn chunked<K: XcorrFft<C32>>(k: &K, st: &mut State, x: &[C32], whole: &[C32], seed: u64) {
    let b = st.block_size();
    let l = st.taps_len();
    let mut rng = Rng::new(seed);
    let max = 3 * b;
    let mut out = z(x.len());
    let (mut pos, mut received, mut stalls) = (0usize, 0usize, 0);
    while pos < x.len() {
        received = (received + 1 + rng.below(max)).min(x.len());
        let room = rng.below(4) * b;
        let end = (pos + room).min(x.len());
        let p = k
            .xcorr_fft(st, &x[pos..received], &mut out[pos..end])
            .unwrap();
        let want = (received - pos).min(end - pos) / b * b;
        assert_eq!(p.consumed, want);
        assert_eq!(p.produced, want);
        pos += p.consumed;
        assert_eq!(bits(&out[..pos]), bits(&whole[..pos]), "seed {seed}");
        let hist: Vec<C32> = if pos == 0 {
            z(l - 1)
        } else {
            x[pos - (l - 1)..pos].to_vec()
        };
        assert_eq!(bits(st.history()), bits(&hist), "seed {seed}");
        stalls = if p.consumed == 0 { stalls + 1 } else { 0 };
        assert!(stalls < 1000, "no progress");
    }
}

#[test]
fn twenty_chunk_plans_give_the_bits_of_the_whole_run() {
    let mut rng = Rng::new(SEED + 9);
    for &(l, n) in &[
        (1usize, 16usize),
        (8, 16),
        (65, 256),
        (127, 512),
        (300, 1024),
    ] {
        let taps = noise(&mut rng, l, 1.0);
        let mut st = state(&taps, n);
        let x = noise(&mut rng, 6 * st.block_size(), 1.0);
        let whole = run_whole(&Fallback, &mut st, &x);
        for plan in 0..20 {
            st.reset();
            chunked(&Fallback, &mut st, &x, &whole, SEED_PLANS * 100 + plan);
        }
    }
}

// --------------------------------------------------------------------------------------------
// Extremes
// --------------------------------------------------------------------------------------------

#[test]
fn silence_gives_exact_zeros_and_a_full_scale_impulse_the_taps() {
    let mut rng = Rng::new(SEED + 10);
    for &(l, n) in &CONFIGS {
        let taps = noise(&mut rng, l, 1.0);
        let mut st = state(&taps, n);
        let len = 3 * st.block_size();
        let out = run_whole(&Fallback, &mut st, &z(len));
        assert!(
            out.iter().all(|v| v.re == 0.0 && v.im == 0.0),
            "L {l}, N {n}"
        );
        // A full-scale impulse (both components at 1 - 2^-15 of the largest Q15 value).
        let a = 1.0 - 1.0 / 32768.0;
        let s = C32::new(a, -a);
        let mut x = z(len);
        let m = len / 2;
        x[m] = s;
        st.reset();
        let out = run_whole(&Fallback, &mut st, &x);
        let snr = snr_db(&direct(&taps, &x), &out);
        assert!(snr >= FLOOR_DB, "L {l}, N {n}: {snr:.1} dB");
    }
}

#[test]
fn a_special_value_spoils_at_most_the_two_blocks_whose_windows_hold_it() {
    let mut rng = Rng::new(SEED + 11);
    for &(l, n) in &[(65usize, 256usize), (127, 512), (8, 16), (1, 16)] {
        let taps = noise(&mut rng, l, 1.0);
        let mut st = state(&taps, n);
        let b = st.block_size();
        let len = 5 * b;
        let x = noise(&mut rng, len, 1.0);
        let clean = run_whole(&Fallback, &mut st, &x);
        // In the middle of block 2, and among the last L - 1 samples of block 2.
        let positions = [2 * b + b / 2, 3 * b - 1, 3 * b - (l - 1).max(1)];
        for at in positions {
            for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for imag in [false, true] {
                    let mut y = x.clone();
                    y[at] = if imag {
                        C32::new(0.5, v)
                    } else {
                        C32::new(v, 0.5)
                    };
                    st.reset();
                    let out = run_whole(&Fallback, &mut st, &y);
                    let first = at / b;
                    // The next block holds the sample in its history when it is among the last
                    // L - 1 samples of its block.
                    let last = if at % b >= b - (l - 1) {
                        first + 1
                    } else {
                        first
                    };
                    for (blk, (o, c)) in out.chunks_exact(b).zip(clean.chunks_exact(b)).enumerate()
                    {
                        if (first..=last).contains(&blk) {
                            assert!(
                                o.iter().all(|s| !s.re.is_finite() || !s.im.is_finite()),
                                "L {l}, at {at}, {v}, block {blk}: a finite output"
                            );
                        } else {
                            assert_eq!(bits(o), bits(c), "L {l}, at {at}, {v}, block {blk}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn the_largest_fft_with_the_longest_taps() {
    let n = MAX_FFT_SIZE;
    let l = n / 2;
    let mut rng = Rng::new(SEED + 12);
    // Sparse taps over the whole length, so the direct convolution stays cheap.
    let mut taps = z(l);
    let nz = [
        (0usize, C32::new(1.0, 0.0)),
        (12_345, C32::new(-0.25, 0.5)),
        (l - 1, C32::new(0.0, 0.5)),
    ];
    for &(k, c) in &nz {
        taps[k] = c;
    }
    let mut st = state(&taps, n);
    let b = st.block_size();
    assert_eq!(b, l + 1);
    let x = noise(&mut rng, 2 * b, 1.0);
    let out = run_whole(&Fallback, &mut st, &x);
    let want: Vec<(f64, f64)> = (0..x.len())
        .map(|i| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for &(k, c) in &nz {
                if i >= k {
                    let v = x[i - k];
                    let (a, bb) = (f64::from(c.re), f64::from(c.im));
                    let (p, q) = (f64::from(v.re), f64::from(v.im));
                    re += a * p - bb * q;
                    im += a * q + bb * p;
                }
            }
            (re, im)
        })
        .collect();
    let snr = snr_db(&want, &out);
    assert!(snr >= FLOOR_DB, "N 65536, L 32768: {snr:.1} dB");
    assert_eq!(bits(st.history()), bits(&x[2 * b - (l - 1)..]));
    st.reset();
    let silent = run_whole(&Fallback, &mut st, &z(b));
    assert!(silent.iter().all(|v| v.re == 0.0 && v.im == 0.0));
}

#[test]
fn the_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<State>();
    assert_send_sync::<XcorrFftState<C32, &'static [C32], [C32; 7], [C32; 16], &'static [C32]>>();
}
