//! `snr_estimate`: a blind estimate of the signal-to-noise ratio Es/N0 of square QAM symbols (a
//! quality meter, M44.4 §1): decisions on the grid with a fitted gain per window, the error of
//! the decisions, and the inverse of what that estimate reads over white Gaussian noise.

use core::marker::PhantomData;

use sdrk::{Progress, Rate};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::Qam;

use super::qam_mod::{QAM_CONSTELLATION_PARAM, qam_constellation_of_value, qam_of};
use super::symbol_map::SymbolMapConstellation;
use super::{head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Smallest `window` in symbols.
pub const SNR_ESTIMATE_MIN_WINDOW: usize = 64;
/// Largest `window` in symbols.
pub const SNR_ESTIMATE_MAX_WINDOW: usize = 1 << 20;
/// The default `window` in symbols.
const DEFAULT_WINDOW: usize = 4096;

/// The lowest estimate in dB: the first node of `DD_READING`; a window that reads at or below
/// what the decisions give there reports this value.
pub const SNR_ESTIMATE_FLOOR_DB: f64 = -10.0;
/// The step of the nodes of [`DD_READING`] in dB of Es/N0.
const NODE_STEP_DB: f64 = 0.5;
/// The nodes of [`DD_READING`] per constellation: Es/N0 from -10 to 50 dB in steps of 0.5 dB.
const NODES: usize = 121;
/// Lanes of the partial sums: symbol `j` of a window goes to lane `j % LANES`, so that a group of
/// `LANES` symbols is one step of independent sums the compiler can vectorise, and the sums do
/// not depend on how the input is split.
const LANES: usize = 8;
/// Symbols of a window after which the `f32` lanes are added to the `f64` sums of the window
/// (and at its end): every lane adds at most 128 values in `f32` (the decisions exactly, at most
/// 128 times 1922 for 1024-QAM, below `2^24`), and the fold costs 0.16 cycles per symbol, against
/// 2.5 at 64 symbols (D2504).
const FOLD: usize = 1024;
/// The share of the signal in the mean power at the floor, `rho / (1 + rho)` at -10 dB: the
/// least signal power the moments give the amplitude.
const FLOOR_SIGNAL_SHARE: f64 = 1.0 / 11.0;

/// The reading of the decision estimate over white Gaussian noise, in dB, per constellation
/// (`qpsk`, `qam16`, `qam64`, `qam256`, `qam1024`) at the Es/N0 of the nodes
/// `SNR_ESTIMATE_FLOOR_DB + NODE_STEP_DB i`: with the samples `x = s + n` of the unit-power grid
/// `s` (all points equally likely) and complex Gaussian noise `n` of power `1 / rho`, the
/// decisions `d` of `x` on the grid, the gain `g = E[Re(x d*)] / E|d|^2` and the reading
/// `g^2 E|d|^2 / E|x - g d|^2`, computed exactly per axis (the probability of every decision
/// region and the mean of `x` in it, from the Gaussian distribution function) in `f64` and
/// rounded once to `f32`. The decisions fold the noise back into the grid cells, so the reading
/// lies above Es/N0 where the decisions are not reliable (for 1024-QAM 26.6 dB at an Es/N0 of
/// 15 dB) and equals it where they are (within 0.001 dB from 15, 21, 26, 31 and 36 dB on). Every
/// row is strictly increasing, so the block inverts it by interpolation. The unit test
/// `the_reading_table_is_the_model` computes every entry again (D2500).
#[allow(
    clippy::unreadable_literal,
    clippy::excessive_precision,
    reason = "the shortest decimals that read back as the f32 of the model, as printed"
)]
static DD_READING: [[f32; NODES]; 5] = [
    // 4-QAM (`qpsk`)
    [
        2.45295, 2.4572532, 2.4625592, 2.4690866, 2.4770973, 2.4869018, 2.4988673, 2.5134242,
        2.531074, 2.552396, 2.5780544, 2.608803, 2.6454895, 2.6890569, 2.7405427, 2.8010743,
        2.8718615, 2.954184, 3.049376, 3.1588054, 3.28385, 3.42587, 3.5861785, 3.7660098, 3.966485,
        4.188579, 4.433088, 4.700593, 4.991431, 5.3056655, 5.6430554, 6.003036, 6.384703, 6.786809,
        7.207773, 7.645719, 8.098528, 8.563925, 9.039587, 9.523261, 10.012883, 10.506677,
        11.003211, 11.50142, 12.000571, 12.500207, 13.000067, 13.500019, 14.000005, 14.500001,
        15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0,
        22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5,
        30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0,
        37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5,
        45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5, 50.0,
    ],
    // 16-QAM
    [
        4.738391, 4.862573, 4.99205, 5.1268325, 5.266899, 5.4121933, 5.5626216, 5.7180495,
        5.8783045, 6.0431714, 6.2123966, 6.3856874, 6.5627146, 6.743114, 6.926491, 7.1124215,
        7.300459, 7.4901342, 7.6809635, 7.87245, 8.064091, 8.2553835, 8.445826, 8.634925, 8.822201,
        9.007197, 9.189501, 9.368767, 9.544761, 9.717421, 9.886919, 10.053743, 10.218758,
        10.383246, 10.548925, 10.717909, 10.892638, 11.075772, 11.270057, 11.478195, 11.702717,
        11.945881, 12.209583, 12.495305, 12.804055, 13.136344, 13.492147, 13.870901, 14.271494,
        14.692303, 15.13125, 15.585909, 16.053638, 16.531755, 17.017706, 17.509232, 18.004461,
        18.501982, 19.000801, 19.50029, 20.000093, 20.500027, 21.000006, 21.500002, 22.0, 22.5,
        23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0,
        30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5,
        38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0,
        45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5, 50.0,
    ],
    // 64-QAM
    [
        5.444096, 5.6148324, 5.7939944, 5.9817796, 6.1783557, 6.383856, 6.5983744, 6.8219614,
        7.054618, 7.296294, 7.5468836, 7.8062224, 8.074085, 8.350185, 8.634171, 8.925623, 9.22406,
        9.52893, 9.839613, 10.155422, 10.475599, 10.799313, 11.125662, 11.453673, 11.7823,
        12.110431, 12.436894, 12.760474, 13.079924, 13.3939905, 13.701438, 14.001074, 14.291779,
        14.572531, 14.84243, 15.100719, 15.346801, 15.580246, 15.800808, 16.008425, 16.203243,
        16.385637, 16.556267, 16.716145, 16.866713, 17.009922, 17.14827, 17.284765, 17.422853,
        17.566252, 17.718775, 17.884136, 18.065777, 18.266743, 18.489592, 18.736341, 19.00844,
        19.306755, 19.631552, 19.982492, 20.358633, 20.75844, 21.179834, 21.620289, 22.076948,
        22.546806, 23.026903, 23.514507, 24.00728, 24.503372, 25.001427, 25.500546, 26.000187,
        26.500055, 27.000015, 27.500004, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0,
        32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5,
        40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0,
        47.5, 48.0, 48.5, 49.0, 49.5, 50.0,
    ],
    // 256-QAM
    [
        5.7294116, 5.9191737, 6.1187143, 6.3283334, 6.5483084, 6.7788916, 7.0203047, 7.272735,
        7.5363345, 7.811214, 8.097443, 8.395046, 8.704007, 9.02426, 9.355696, 9.698155, 10.051436,
        10.415283, 10.789388, 11.173386, 11.56685, 11.969282, 12.38011, 12.798669, 13.224201,
        13.65584, 14.092612, 14.533428, 14.977084, 15.422268, 15.867564, 16.31147, 16.752405,
        17.188734, 17.618788, 18.040886, 18.453373, 18.854643, 19.243166, 19.617537, 19.976494,
        20.318949, 20.644016, 20.95103, 21.239553, 21.509382, 21.760538, 21.993265, 22.208,
        22.405352, 22.58608, 22.751072, 22.901329, 23.037985, 23.162336, 23.27593, 23.380655,
        23.478848, 23.57334, 23.667463, 23.764963, 23.86983, 23.986101, 24.117662, 24.268063,
        24.440386, 24.63718, 24.860409, 25.111443, 25.391052, 25.699413, 26.036085, 26.400024,
        26.78959, 27.202585, 27.636335, 28.087822, 28.553844, 29.031223, 29.517004, 30.00863,
        30.504047, 31.001736, 31.500675, 32.000233, 32.500072, 33.00002, 33.500004, 34.0, 34.5,
        35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0,
        42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5,
        50.0,
    ],
    // 1024-QAM
    [
        5.8588533, 6.057051, 6.265605, 6.4848557, 6.715125, 6.9567137, 7.209896, 7.474918,
        7.7519937, 8.041304, 8.342997, 8.657186, 8.983949, 9.323334, 9.675353, 10.039992,
        10.417205, 10.806911, 11.209004, 11.62334, 12.049733, 12.487959, 12.937734, 13.398716,
        13.870492, 14.3525715, 14.844378, 15.345247, 15.85442, 16.371044, 16.894176, 17.422783,
        17.955734, 18.491821, 19.029747, 19.568132, 20.105524, 20.640394, 21.171146, 21.696127,
        22.213636, 22.72194, 23.219292, 23.703955, 24.174229, 24.628489, 25.065207, 25.483006,
        25.880676, 26.257229, 26.611906, 26.944214, 27.253923, 27.54108, 27.805986, 28.049183,
        28.271423, 28.473642, 28.656912, 28.822405, 28.97136, 29.105047, 29.224747, 29.331718,
        29.427216, 29.512499, 29.588905, 29.657936, 29.721392, 29.78148, 29.84091, 29.90288,
        29.971008, 30.049164, 30.141262, 30.251057, 30.381952, 30.536886, 30.71825, 30.927856,
        31.166931, 31.436111, 31.735445, 32.06439, 32.42179, 32.80592, 33.21449, 33.644726,
        34.09351, 34.55753, 35.033493, 35.518322, 36.009346, 36.504406, 37.001904, 37.500744,
        38.00026, 38.50008, 39.000023, 39.500004, 40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5,
        44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5, 50.0,
    ],
];

/// Per constellation, the Es/N0 in dB from which the gain of the decision fit is the gain of
/// the grid within 1e-4 (the model of [`DD_READING`]): from there on the amplitude of the next
/// window is the fitted gain, below it the signal power of the moments M2 and M4 (D2500).
const FIT_GAIN_FROM_DB: [f64; 5] = [10.5, 16.5, 21.5, 26.5, 30.5];

/// The index of a square QAM in [`DD_READING`] and [`FIT_GAIN_FROM_DB`].
const fn row(q: Qam) -> usize {
    q.bits_per_axis() - 1
}

/// Description of the `snr_estimate` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "snr_estimate",
    summary: "Estimates Es/N0 of square QAM symbols (4-QAM to 1024-QAM) from the symbols alone; they pass \
        unchanged.",
    details: "The input is one complex sample per symbol after the equalizer and the carrier \
        loop (the input of the decision, the points at any amplitude and turned by at most a \
        fraction of the grid spacing). The estimate is made per window of window symbols, \
        counted from the start: each symbol is decided on the grid of the constellation at the \
        amplitude a of the previous window, and the window gives the error power of the \
        decisions after the least-squares gain g of the window (N = sum |e|^2 - (sum Re(e d*))^2 \
        / sum |d|^2 with x / a = d + e) and the signal power (g^2 sum |d|^2). Where decisions \
        fail the noise folds into the grid cells and this ratio reads high, so the estimate is \
        its inverse through the exact reading of the same estimate over white Gaussian noise (a \
        table per constellation from -10 to 50 dB, interpolated): it is unbiased over Gaussian \
        noise from -10 dB on, and equals the signal-to-noise ratio after the gain where the \
        decisions are reliable. The amplitude of the next window is the fitted gain where it is \
        exact (from 10.5, 16.5, 21.5, 26.5 and 30.5 dB for qpsk to qam1024), below that the \
        signal power of the moments of the window (M2M4 with the kurtosis ka of the \
        constellation: S^2 = (2 M2^2 - M4) / (2 - ka), held between M2 / 11 and M2), so a lost \
        signal reads low at once. The first window only measures the amplitude, so the first \
        estimate comes after two windows. Residual intersymbol interference and phase jitter \
        count as noise: after an equalizer the estimate is the signal-to-noise ratio at its \
        output. A window with a NaN or infinite symbol (or a part above about 4e9, whose fourth \
        power overflows) or of zero power keeps the estimate and the amplitude. The block \
        reports snr_db, NaN before the first estimate and -10 at or below the floor, as a \
        reading in the pipeline statistics. No delay, ratio 1.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Symbols, one sample per symbol (after equalize and costas).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The input, unchanged.",
    }],
    params: &[
        QAM_CONSTELLATION_PARAM,
        ParamDesc {
            name: "window",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: SNR_ESTIMATE_MIN_WINDOW as i64,
                max: SNR_ESTIMATE_MAX_WINDOW as i64,
                default: Some(DEFAULT_WINDOW as i64),
            },
            typical: "4096; more for a steadier estimate, fewer to follow a fading signal",
            mutability: Mutability::Build,
            description: "Symbols per estimate (the spread falls with the square root of the \
                window).",
        },
    ],
    required: &[],
};

/// Configuration of [`SnrEstimateBlock`].
///
/// ```
/// use sdrk_blocks::{ParamValue, SnrEstimateConfig, SymbolMapConstellation};
/// let mut cfg = SnrEstimateConfig::default();
/// assert_eq!(cfg.constellation, SymbolMapConstellation::Qam16);
/// assert_eq!(cfg.window, 4096);
/// cfg.set("window", ParamValue::I64(1024))?;
/// assert_eq!(cfg.get("window")?, ParamValue::I64(1024));
/// // only the square QAM: bpsk and 8psk are refused
/// assert!(cfg.set("constellation", ParamValue::Choice(0)).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnrEstimateConfig {
    /// The square constellation of the symbols (`qpsk` to `qam1024`; default `qam16`).
    pub constellation: SymbolMapConstellation,
    /// Symbols per estimate (64 to 1048576, default 4096).
    pub window: usize,
}

impl Default for SnrEstimateConfig {
    /// `qam16`, a window of 4096 symbols.
    fn default() -> Self {
        Self {
            constellation: SymbolMapConstellation::Qam16,
            window: DEFAULT_WINDOW,
        }
    }
}

impl SnrEstimateConfig {
    /// Checks the configuration.
    ///
    /// # Errors
    /// `InvalidParameter` of `constellation` for `bpsk` and `8psk`; `OutOfRange` of `window`
    /// outside 64 to 1048576.
    pub fn validate(&self) -> Result<()> {
        qam_of(self.constellation)?;
        if !(SNR_ESTIMATE_MIN_WINDOW..=SNR_ESTIMATE_MAX_WINDOW).contains(&self.window) {
            return Err(Error::OutOfRange {
                name: "window",
                min: SNR_ESTIMATE_MIN_WINDOW as f64,
                max: SNR_ESTIMATE_MAX_WINDOW as f64,
                got: self.window as f64,
            });
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for a
    /// constellation that is not a square QAM (`bpsk`, `8psk`).
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "constellation" => self.constellation = qam_constellation_of_value(&value)?,
            "window" => self.window = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "constellation" => Ok(ParamValue::Choice(self.constellation.index())),
            "window" => Ok(ParamValue::I64(self.window as i64)),
            _ => unknown(),
        }
    }
}

/// The sums of one window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Sums {
    /// `sum |e|^2` of the decision errors in grid units.
    err: f64,
    /// `sum Re(e d*)`.
    cross: f64,
    /// `sum |d|^2` of the decisions in grid units.
    dec: f64,
    /// `sum |x|^2` of the input.
    power: f64,
    /// `sum |x|^4` of the input.
    fourth: f64,
}

/// The partial sums of the current group of [`FOLD`] symbols, one per lane, in the order of
/// [`Sums`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Lanes {
    err: [f32; LANES],
    cross: [f32; LANES],
    dec: [f32; LANES],
    power: [f32; LANES],
    fourth: [f32; LANES],
}

impl Lanes {
    /// Adds symbol `v` to lane `j`: per axis the decision and the error of [`decide_axis`] in
    /// grid units, and the second and fourth power of the symbol. The one form of the sums, for
    /// the whole groups of [`groups`] and the single symbols of the accumulation.
    #[inline]
    fn add(&mut self, j: usize, v: C32, k: f32, half: f32, top: f32) {
        let (dr, er) = decide_axis(v.re, k, half, top);
        let (di, ei) = decide_axis(v.im, k, half, top);
        let p = v.re * v.re + v.im * v.im;
        if let (Some(e), Some(c), Some(d), Some(w), Some(f)) = (
            self.err.get_mut(j),
            self.cross.get_mut(j),
            self.dec.get_mut(j),
            self.power.get_mut(j),
            self.fourth.get_mut(j),
        ) {
            *e += er * er + ei * ei;
            *c += er * dr + ei * di;
            *d += dr * dr + di * di;
            *w += p;
            *f += p * p;
        }
    }

    /// Adds the lanes to `sums` in lane order and clears them.
    fn fold_into(&mut self, sums: &mut Sums) {
        let total = |v: &[f32; LANES]| v.iter().fold(0.0f64, |a, &x| a + f64::from(x));
        sums.err += total(&self.err);
        sums.cross += total(&self.cross);
        sums.dec += total(&self.dec);
        sums.power += total(&self.power);
        sums.fourth += total(&self.fourth);
        *self = Lanes::default();
    }
}

/// `2^23`: adding and subtracting it rounds an `f32` in `0 ..= 2^23` to an integer (to nearest,
/// ties to even).
const ROUND: f32 = 8_388_608.0;

/// `t` clamped to `0 ..= top`, NaN to 0. On AArch64 by `f32::max` and `f32::min` (`fmaxnm` and
/// `fminnm`, which return the other operand for NaN); elsewhere by two selects, which give the
/// same values (NaN and `-0.0` to `+0.0` in effect: the floor below takes the level `+0` for both
/// zeros) and vectorise on every target, where `f32::max` and `f32::min` have no lane form on
/// `simd128` and were extracted lane by lane (D2518).
#[inline]
fn clamp_level(t: f32, top: f32) -> f32 {
    if cfg!(target_arch = "aarch64") {
        t.max(0.0).min(top)
    } else {
        let low = if t > 0.0 { t } else { 0.0 };
        if low < top { low } else { top }
    }
}

/// The decision of one axis value `v`: with `t = v k + half` the level `i = floor(t)` clamped to
/// `0 .. m - 1` (`top = m - 1`, [`clamp_level`]), the decision `2 i - top` and the error
/// `2 (t - i) - 1` in grid units. NaN takes level 0 (its error stays NaN). The floor of the
/// clamped value `c` is computed in `f32` without a conversion to an integer (which `as` makes
/// saturating, element by element): `r = (c + 2^23) - 2^23` is `c` rounded to an integer, and
/// the floor is `r - 1` where `r > c`, else `r`; exact for `0 <= c < 2^23`. So the decisions of
/// a group of symbols vectorise (D2504).
#[inline]
fn decide_axis(value: f32, gain: f32, half: f32, top: f32) -> (f32, f32) {
    let t = value * gain + half;
    let clamped = clamp_level(t, top);
    let rounded = (clamped + ROUND) - ROUND;
    let level = if rounded > clamped {
        rounded - 1.0
    } else {
        rounded
    };
    (2.0 * level - top, 2.0 * (t - level) - 1.0)
}

/// Whole groups of [`LANES`] symbols into the lanes, lane `j` taking symbol `j` of every group:
/// the lanes are held in local arrays over all groups (registers), each step a loop over the
/// lanes of independent sums, which the compiler vectorises.
#[inline]
fn groups(l: &mut Lanes, x: &[C32], k: f32, half: f32, top: f32) {
    let mut lanes = *l;
    for g in x.chunks_exact(LANES) {
        for (j, &v) in g.iter().enumerate().take(LANES) {
            lanes.add(j, v, k, half, top);
        }
    }
    *l = lanes;
}

/// A blind estimate of Es/N0 of square QAM symbols (M44.4 §1): `C32` in, the same `C32` out.
///
/// Per window of `window` symbols (counted from construction or `reset`, so the estimate does
/// not depend on how the input is split): the decisions on the grid at the amplitude of the
/// previous window, the least-squares gain of the window, the ratio of the fitted signal power
/// to the error power, and the inverse of the reading of that ratio over white Gaussian noise
/// (the table `DD_READING` of the constellation). The amplitude of the next window is the
/// fitted gain where the model makes it exact, otherwise the signal power of the moments M2 and
/// M4; the first window only measures it. No storage, ratio 1, delay 0. The model and the
/// measurements are in D2500 to D2503.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, SnrEstimateBlock, SnrEstimateConfig};
/// use sdrk_core::kernels::constellation::Qam;
/// let cfg = SnrEstimateConfig { window: 1024, ..SnrEstimateConfig::default() };
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut est = SnrEstimateBlock::new(&cfg, Rate::hz(10_000)?, &mut arena)?;
/// // 16-QAM points with a small deterministic error: about 40 dB
/// let mut state = 1u32;
/// let input: Vec<C32> = (0..4096)
///     .map(|_| {
///         state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
///         let p = Qam::Qam16.point(state >> 28);
///         let e = 0.01 * ((state >> 8 & 0xff) as f32 / 128.0 - 1.0);
///         C32::new(0.5 * (p.re + e), 0.5 * (p.im - e))
///     })
///     .collect();
/// let mut out = vec![C32::default(); input.len()];
/// est.process(&input, &mut out)?;
/// assert_eq!(out, input);
/// assert_eq!(est.estimates(), 3);
/// assert!(est.snr_db() > 35.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SnrEstimateBlock<S: Storage> {
    config: SnrEstimateConfig,
    qam: Qam,
    /// The amplitude of the unit-power grid in input units; 0 while the first window measures it.
    amplitude: f64,
    /// `sqrt(E) / (2 amplitude)`: an input value times this plus `m / 2` is the level index
    /// coordinate `t` (the decision is `floor(t)` clamped to `0 .. m - 1`).
    half_gain: f32,
    /// Symbols of the current window so far.
    seen: usize,
    sums: Sums,
    lanes: Lanes,
    /// The estimate in dB, NaN before the first.
    snr_db: f64,
    estimates: u64,
    _marker: PhantomData<fn() -> S>,
}

impl<S: Storage> core::fmt::Debug for SnrEstimateBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SnrEstimateBlock")
            .field("config", &self.config)
            .field("snr_db", &self.snr_db)
            .field("estimates", &self.estimates)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> SnrEstimateBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// As [`SnrEstimateConfig::validate`].
    pub fn new(config: &SnrEstimateConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: *config,
            qam: qam_of(config.constellation)?,
            amplitude: 0.0,
            half_gain: 0.0,
            seen: 0,
            sums: Sums::default(),
            lanes: Lanes::default(),
            snr_db: f64::NAN,
            estimates: 0,
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// As [`SnrEstimateConfig::validate`].
    pub fn storage_bytes(config: &SnrEstimateConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        Ok(Plan::default().bytes())
    }

    /// Output rate (equal to the input rate).
    ///
    /// # Errors
    /// As [`SnrEstimateConfig::validate`].
    pub fn rate_out(config: &SnrEstimateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in)
    }

    /// The configuration.
    pub fn config(&self) -> SnrEstimateConfig {
        self.config
    }

    /// The estimate of Es/N0 in dB of the last window that gave one: NaN before the first (the
    /// first two windows), [`SNR_ESTIMATE_FLOOR_DB`] at or below the floor, `+inf` for symbols
    /// exactly on the grid.
    pub fn snr_db(&self) -> f64 {
        self.snr_db
    }

    /// The estimates made since construction or `reset`.
    pub fn estimates(&self) -> u64 {
        self.estimates
    }

    /// The symbols until the current window completes (1 to `window`).
    pub fn remaining(&self) -> usize {
        self.config.window - self.seen
    }

    /// Takes the symbols into the estimate without writing anything (the call of a composite
    /// that holds the block, on symbols it passes on itself); returns whether a window
    /// completed.
    pub fn observe(&mut self, symbols: &[C32]) -> bool {
        let mut done = false;
        let mut rest = symbols;
        while !rest.is_empty() {
            let n = rest.len().min(self.remaining());
            let (now, later) = rest.split_at(n);
            self.accumulate(now);
            self.seen += n;
            if self.seen == self.config.window {
                self.window_end();
                done = true;
            }
            rest = later;
        }
        done
    }

    /// The sums of the symbols, which lie in the current window from its symbol `self.seen` on:
    /// per symbol and axis the decision and the error of [`decide_axis`] in grid units (with the
    /// level coordinate `t = v sqrt(E) / (2 a) + m / 2`), and the second and fourth power of the
    /// symbol, into the lane of its position; the lanes go to the sums every [`FOLD`] symbols.
    /// Before the first amplitude (`a` = 0, so `t = m / 2`) the decision sums are not used.
    fn accumulate(&mut self, x: &[C32]) {
        let k = self.half_gain;
        let m = self.qam.levels_per_axis() as f32;
        let (half, top) = (0.5 * m, m - 1.0);
        let mut pos = self.seen;
        let mut rest = x;
        while let Some((&v, tail)) = rest.split_first() {
            let lane = pos % LANES;
            let to_fold = FOLD - pos % FOLD;
            if lane == 0 && rest.len() >= LANES && to_fold >= LANES {
                // Whole groups of LANES symbols up to the next fold.
                let n = LANES * (rest.len() / LANES).min(to_fold / LANES);
                let (now, later) = rest.split_at(n);
                groups(&mut self.lanes, now, k, half, top);
                pos += n;
                rest = later;
            } else {
                self.lanes.add(lane, v, k, half, top);
                pos += 1;
                rest = tail;
            }
            if pos.is_multiple_of(FOLD) {
                self.lanes.fold_into(&mut self.sums);
            }
        }
    }

    /// Ends a window: the estimate (once there is an amplitude) and the amplitude of the next
    /// window.
    #[cold]
    #[inline(never)]
    fn window_end(&mut self) {
        self.lanes.fold_into(&mut self.sums);
        let s = core::mem::take(&mut self.sums);
        self.seen = 0;
        let n = self.config.window as f64;
        let m2 = s.power / n;
        let m4 = s.fourth / n;
        if !(m2.is_finite() && m2 > 0.0 && m4.is_finite()) {
            return;
        }
        // M2M4: S^2 = (2 M2^2 - M4) / (2 - ka) with the kurtosis ka = E|s|^4 / (E|s|^2)^2 of
        // the constellation (its CMA radius at unit power), held between the signal power of
        // the floor, M2 / 11 (-10 dB), and M2.
        let s2 = (2.0 * m2 * m2 - m4) / (2.0 - self.qam.cma_r2());
        let low = m2 * FLOOR_SIGNAL_SHARE;
        let moments = if s2 > 0.0 {
            libm::sqrt(s2).max(low).min(m2)
        } else {
            low
        };
        let mut amplitude = libm::sqrt(moments);
        if self.amplitude > 0.0 {
            if !(s.err.is_finite() && s.cross.is_finite() && s.dec > 0.0) {
                return;
            }
            let gain = 1.0 + s.cross / s.dec;
            let noise = (s.err - s.cross * s.cross / s.dec).max(0.0);
            let reading = 10.0 * libm::log10(gain * gain * s.dec / noise);
            let snr_db = self.invert(reading);
            if snr_db.is_nan() {
                return;
            }
            self.snr_db = snr_db;
            self.estimates += 1;
            if FIT_GAIN_FROM_DB
                .get(row(self.qam))
                .is_some_and(|&from| snr_db >= from)
            {
                amplitude = self.amplitude * gain;
            }
        }
        self.set_amplitude(amplitude);
    }

    /// Sets the amplitude of the grid; one that is not finite and above 0 measures it again in
    /// the next window.
    fn set_amplitude(&mut self, amplitude: f64) {
        let half_gain = (f64::from(self.qam.sqrt_energy()) / (2.0 * amplitude)) as f32;
        if amplitude.is_finite() && amplitude > 0.0 && half_gain.is_finite() && half_gain > 0.0 {
            self.amplitude = amplitude;
            self.half_gain = half_gain;
        } else {
            self.amplitude = 0.0;
            self.half_gain = 0.0;
        }
    }

    /// The Es/N0 in dB whose reading over white Gaussian noise is `reading` dB: linear between
    /// the nodes of [`DD_READING`], the floor at or below the first node and the reading less
    /// the offset of the last node above it; NaN for a NaN reading.
    fn invert(&self, reading: f64) -> f64 {
        let Some(table) = DD_READING.get(row(self.qam)) else {
            return f64::NAN;
        };
        if reading.is_nan() {
            return f64::NAN;
        }
        let first = table.first().map_or(0.0, |&v| f64::from(v));
        let last = table.last().map_or(0.0, |&v| f64::from(v));
        if reading <= first {
            return SNR_ESTIMATE_FLOOR_DB;
        }
        let top = SNR_ESTIMATE_FLOOR_DB + NODE_STEP_DB * (NODES - 1) as f64;
        if reading >= last {
            return top + (reading - last);
        }
        // The first node above the reading: 1 ..= NODES - 1.
        let i = table.partition_point(|&v| f64::from(v) <= reading);
        let (Some(&lo), Some(&hi)) = (table.get(i - 1), table.get(i)) else {
            return f64::NAN;
        };
        let (lo, hi) = (f64::from(lo), f64::from(hi));
        SNR_ESTIMATE_FLOOR_DB + NODE_STEP_DB * ((i - 1) as f64 + (reading - lo) / (hi - lo))
    }
}

impl<S: Storage> Block for SnrEstimateBlock<S> {
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        let x = head(input, n)?;
        head_mut(output, n)?.copy_from_slice(x);
        self.observe(x);
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {
        self.amplitude = 0.0;
        self.half_gain = 0.0;
        self.seen = 0;
        self.sums = Sums::default();
        self.lanes = Lanes::default();
        self.snr_db = f64::NAN;
        self.estimates = 0;
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        runtime_param(&DESCRIPTION, name, &value)?;
        unknown()
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        match index {
            0 => Some(("snr_db", self.snr_db)),
            _ => None,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]
mod tests {
    use super::*;

    /// The standard normal distribution function, from the complementary error function so
    /// that both tails keep their precision.
    fn normal_cdf(x: f64) -> f64 {
        0.5 * libm::erfc(-x / core::f64::consts::SQRT_2)
    }

    fn normal_pdf(x: f64) -> f64 {
        libm::exp(-0.5 * x * x) / libm::sqrt(2.0 * core::f64::consts::PI)
    }

    /// The model of [`DD_READING`] for `b` bits per axis at an Es/N0 of `snr_db`: the reading in
    /// dB and the gain of the fit, per axis on the integer grid of odd levels `v` (power `E / 2`
    /// per axis), the noise variance `E / (2 rho)` per axis, the decision regions between the
    /// even integers.
    fn model(b: usize, snr_db: f64) -> (f64, f64) {
        let m = 1usize << b;
        let energy = (2 * (m * m - 1)) as f64 / 3.0;
        let var = energy / (2.0 * libm::pow(10.0, snr_db / 10.0));
        let sd = libm::sqrt(var);
        let level = |j: usize| (2 * j) as f64 - (m - 1) as f64;
        let (mut exd, mut edd, mut exx) = (0.0, 0.0, 0.0);
        for i in 0..m {
            let v = level(i);
            exx += v * v + var;
            for j in 0..m {
                let d = level(j);
                // the region of level j: (d - 1, d + 1], unbounded at the ends
                let lo = if j == 0 {
                    f64::NEG_INFINITY
                } else {
                    (d - 1.0 - v) / sd
                };
                let hi = if j == m - 1 {
                    f64::INFINITY
                } else {
                    (d + 1.0 - v) / sd
                };
                let p = if lo > 0.0 {
                    normal_cdf(-lo) - normal_cdf(-hi)
                } else {
                    normal_cdf(hi) - normal_cdf(lo)
                };
                let pdf = |z: f64| if z.is_finite() { normal_pdf(z) } else { 0.0 };
                let ex = v * p + sd * (pdf(lo) - pdf(hi));
                exd += d * ex;
                edd += d * d * p;
            }
        }
        let n = m as f64;
        let (exd, edd, exx) = (exd / n, edd / n, exx / n);
        let gain = exd / edd;
        let err = exx - 2.0 * exd + edd;
        let cross = exd - edd;
        let noise = err - cross * cross / edd;
        (10.0 * libm::log10(gain * gain * edd / noise), gain)
    }

    #[test]
    fn the_reading_table_is_the_model() {
        for (r, row) in DD_READING.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                let snr = SNR_ESTIMATE_FLOOR_DB + NODE_STEP_DB * i as f64;
                let (reading, _) = model(r + 1, snr);
                let ulp = f64::from(f32::EPSILON) * reading.abs().max(1.0);
                assert!(
                    (f64::from(v) - reading).abs() <= ulp,
                    "row {r} node {i}: table {v}, model {reading}"
                );
                if i > 0 {
                    assert!(v > row[i - 1], "row {r} is not increasing at node {i}");
                }
            }
            // At the top node the reading is Es/N0.
            assert!((f64::from(row[NODES - 1]) - 50.0).abs() < 1e-5);
        }
    }

    #[test]
    fn the_fit_gain_holds_from_its_threshold_on() {
        for (r, &from) in FIT_GAIN_FROM_DB.iter().enumerate() {
            let below = model(r + 1, from - NODE_STEP_DB).1;
            assert!((below - 1.0).abs() >= 1e-4, "row {r}: {below}");
            let mut snr = from;
            while snr <= 50.0 {
                let g = model(r + 1, snr).1;
                assert!((g - 1.0).abs() < 1e-4, "row {r} at {snr}: {g}");
                snr += NODE_STEP_DB;
            }
        }
    }

    #[test]
    fn the_inverse_of_a_node_is_its_es_n0() {
        for q in Qam::ALL {
            let cfg = SnrEstimateConfig {
                constellation: match q {
                    Qam::Qpsk => SymbolMapConstellation::Qpsk,
                    Qam::Qam16 => SymbolMapConstellation::Qam16,
                    Qam::Qam64 => SymbolMapConstellation::Qam64,
                    Qam::Qam256 => SymbolMapConstellation::Qam256,
                    _ => SymbolMapConstellation::Qam1024,
                },
                window: 64,
            };
            let est = SnrEstimateBlock::<crate::ArenaStorage<'_>>::new(
                &cfg,
                Rate::hz(1).unwrap(),
                &mut crate::ArenaStorage::new(&mut []),
            )
            .unwrap();
            let row = &DD_READING[row(q)];
            for (i, &v) in row.iter().enumerate() {
                let want = SNR_ESTIMATE_FLOOR_DB + NODE_STEP_DB * i as f64;
                let got = est.invert(f64::from(v));
                assert!((got - want).abs() < 1e-9, "{q:?} node {i}: {got}");
            }
            assert_eq!(est.invert(f64::from(row[0]) - 1.0), SNR_ESTIMATE_FLOOR_DB);
            assert!((est.invert(f64::from(row[NODES - 1]) + 3.0) - 53.0).abs() < 1e-5);
            assert_eq!(est.invert(f64::INFINITY), f64::INFINITY);
            assert!(est.invert(f64::NAN).is_nan());
        }
    }
}
