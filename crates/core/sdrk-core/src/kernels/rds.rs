//! RDS transmit (M42): the shaped biphase baseband of the Radio Data System (`rds_shape`).
//!
//! The data bits of RDS (1187.5 bit/s, after the differential coding `t_k = t_{k-1} xor d_k`,
//! amplitude `a_k = 2 t_k - 1`) are sent as biphase symbols through the data-shaping filter
//! `H(f) = cos(pi f Tb / 4)` for `|f| <= 2 / Tb` (IEC 62106). One bit is the pulse
//! `p(t) = h(t + Tb/4) - h(t - Tb/4)` (first chip `+a_k`, the sign of the M41 receiver and of
//! the reference modulator `cases/rds.py`), where `h` is the impulse response of `H` in closed
//! form, with `x = t / Tb` and without the factor `1 / Tb`:
//! `h(x) = 8 cos(4 pi x) / (pi (1 - 64 x^2))`, `h(+-1/8) = 2` (decision D664 of M41).
//!
//! [`rds_pulse_table`] samples `p` at 256 points per bit over `-2 Tb <= t <= 2 Tb`
//! ([`RDS_TABLE_LEN`] = 1025 values, the last one `p(2 Tb)` for the interpolation of the last
//! interval), divided by the largest magnitude on that grid. [`RdsShape`] sums the pulses of the
//! (at most four) bits that cover each output sample, with linear interpolation in the table.
//! Bit `k` is centred at `t = k Tb`, the first bit at the first output sample.
//!
//! ```
//! use sdrk_core::kernels::rds::{rds_pulse_table, RdsShape, RdsShapeState, RDS_TABLE_LEN};
//! use sdrk_core::{Fallback, Rate};
//! let mut table = [0.0f32; RDS_TABLE_LEN];
//! rds_pulse_table(&mut table)?;
//! let mut st = RdsShapeState::new(table, Rate::hz(192_000)?)?;
//! // A steady stream of ones: the bits alternate in sign after the differential coding.
//! let mut t = 0i8;
//! let mut bits = || { t ^= 1; 2 * t - 1 };
//! let mut out = [0.0f32; 1024];
//! Fallback.rds_shape(&mut st, &mut bits, &mut out)?;
//! assert!(out.iter().all(|y| y.abs() <= 2.0));
//! assert_eq!(st.position(), 1024);
//! # Ok::<(), sdrk_core::Error>(())
//! ```

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::timing::{Delay, Rate, Timing};

/// The bit rate of RDS in bit/s.
pub const RDS_BIT_RATE_HZ: f64 = 1187.5;
/// Points of the pulse table per bit.
pub const RDS_TABLE_PER_BIT: usize = 256;
/// Entries of the pulse table: 4 bits of [`RDS_TABLE_PER_BIT`] points plus the end point.
pub const RDS_TABLE_LEN: usize = 4 * RDS_TABLE_PER_BIT + 1;
/// The lowest sample rate of `rds_shape`: twice the bandwidth `2 / Tb` of the pulse.
const MIN_RATE_HZ: f64 = 4.0 * RDS_BIT_RATE_HZ;

/// Grid points per bit as `i64` (the denominator of `x = n / 256`).
const DEN: i64 = RDS_TABLE_PER_BIT as i64;

/// `h(x)` at `x = n / 256`: `8 cos(4 pi x) / (pi (1 - 64 x^2))`, with the cosine at the exactly
/// reduced argument `2 pi r / 256`, `r = 2 n mod 256`, and `1 - 64 x^2` as the exact integer
/// `256^2 - 64 n^2` over `256^2`; the limit at `x = +-1/8` is 2 (the `h` of `cases/rds.py`).
fn h(n: i64) -> f64 {
    let d = DEN * DEN - 64 * n * n;
    if d == 0 {
        return 2.0;
    }
    let r = (2 * n).rem_euclid(DEN);
    let c = libm::cos(2.0 * core::f64::consts::PI * r as f64 / DEN as f64);
    8.0 * c * (DEN * DEN) as f64 / (core::f64::consts::PI * d as f64)
}

/// The pulse before normalization at the grid point `i` (`x = -2 + i / 256`).
fn raw_pulse(i: usize) -> f64 {
    let n = i as i64 - 2 * DEN;
    let q = DEN / 4;
    h(n + q) - h(n - q)
}

/// The largest magnitude of the pulse on the grid (about 2.71888).
///
/// ```
/// let peak = sdrk_core::kernels::rds::rds_pulse_peak();
/// assert!((peak - 2.71888).abs() < 1e-5);
/// ```
pub fn rds_pulse_peak() -> f64 {
    (0..RDS_TABLE_LEN)
        .map(|i| libm::fabs(raw_pulse(i)))
        .fold(0.0, f64::max)
}

/// The normalized pulse in f64: `out[i] = p(-2 Tb + i Tb / 256) / peak` for
/// `i = 0 .. RDS_TABLE_LEN` (see the module documentation).
///
/// # Errors
/// `LengthMismatch` when `out` does not hold exactly [`RDS_TABLE_LEN`] values.
///
/// ```
/// use sdrk_core::kernels::rds::{rds_pulse_table_f64, RDS_TABLE_LEN};
/// let mut t = [0.0f64; RDS_TABLE_LEN];
/// rds_pulse_table_f64(&mut t)?;
/// // Odd around the centre, peak 1, first chip positive.
/// assert_eq!(t[512], 0.0);
/// assert!(t.iter().all(|v| v.abs() <= 1.0) && t.iter().any(|&v| v == 1.0));
/// assert!(t[448] > 0.9 && t[576] < -0.9);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn rds_pulse_table_f64(out: &mut [f64]) -> Result<()> {
    crate::kernels::check_len(RDS_TABLE_LEN, out.len())?;
    let peak = rds_pulse_peak();
    for (i, o) in out.iter_mut().enumerate() {
        *o = raw_pulse(i) / peak;
    }
    Ok(())
}

/// The table of [`RdsShape`]: [`rds_pulse_table_f64`] rounded once to `f32`.
///
/// # Errors
/// `LengthMismatch` when `out` does not hold exactly [`RDS_TABLE_LEN`] values.
///
/// ```
/// use sdrk_core::kernels::rds::{rds_pulse_table, RDS_TABLE_LEN};
/// let mut t = [0.0f32; RDS_TABLE_LEN];
/// rds_pulse_table(&mut t)?;
/// assert_eq!(t[512], 0.0);
/// assert!(rds_pulse_table(&mut [0.0f32; 1024]).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn rds_pulse_table(out: &mut [f32]) -> Result<()> {
    crate::kernels::check_len(RDS_TABLE_LEN, out.len())?;
    let peak = rds_pulse_peak();
    for (i, o) in out.iter_mut().enumerate() {
        *o = (raw_pulse(i) / peak) as f32;
    }
    Ok(())
}

/// Mutable view of the state for the implementations of this crate.
pub(crate) struct RdsShapeParts<'a> {
    pub(crate) table: &'a [f32],
    pub(crate) fs: f64,
    pub(crate) n: &'a mut u64,
    pub(crate) fifo: &'a mut [f32; 4],
    pub(crate) next: &'a mut i64,
}

/// State of the RDS pulse shaper: the pulse table (caller storage), the sample rate, the index
/// of the next output sample and the amplitudes of the four bits of the current window.
///
/// ```
/// use sdrk_core::kernels::rds::{rds_pulse_table, RdsShapeState, RDS_TABLE_LEN};
/// use sdrk_core::Rate;
/// let mut table = [0.0f32; RDS_TABLE_LEN];
/// rds_pulse_table(&mut table)?;
/// let st = RdsShapeState::new(&table[..], Rate::hz(192_000)?)?;
/// assert_eq!(st.position(), 0);
/// // A table of another length and a rate below 4750 Hz are rejected.
/// assert!(RdsShapeState::new(&table[..1024], Rate::hz(192_000)?).is_err());
/// assert!(RdsShapeState::new(&table[..], Rate::hz(4_000)?).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RdsShapeState<Tab: AsRef<[f32]>> {
    table: Tab,
    fs: f64,
    n: u64,
    /// Amplitudes of the bits `next - 4 .. next` (0 for the bits before the first one).
    fifo: [f32; 4],
    /// Index of the next bit to fetch.
    next: i64,
}

impl<Tab: AsRef<[f32]>> RdsShapeState<Tab> {
    /// Creates a shaper at sample 0 with no bit fetched.
    ///
    /// `table` must hold [`RDS_TABLE_LEN`] values (normally [`rds_pulse_table`]); the sample
    /// rate must be at least 4750 Hz (twice the bandwidth of the pulse). Otherwise
    /// `InvalidParameter`.
    pub fn new(table: Tab, rate: Rate) -> Result<Self> {
        if table.as_ref().len() != RDS_TABLE_LEN {
            return Err(Error::InvalidParameter(
                "rds pulse table must hold 1025 values",
            ));
        }
        let fs = rate.as_f64();
        if fs < MIN_RATE_HZ || fs.is_nan() {
            return Err(Error::InvalidParameter(
                "rds shaper needs a sample rate of at least 4750 hz",
            ));
        }
        Ok(Self {
            table,
            fs,
            n: 0,
            fifo: [0.0; 4],
            next: -1,
        })
    }

    /// Back to sample 0 with no bit fetched (the next call starts again with bit 0).
    ///
    /// ```
    /// use sdrk_core::kernels::rds::{rds_pulse_table, RdsShape, RdsShapeState, RDS_TABLE_LEN};
    /// use sdrk_core::{Fallback, Rate};
    /// let mut table = [0.0f32; RDS_TABLE_LEN];
    /// rds_pulse_table(&mut table)?;
    /// let mut st = RdsShapeState::new(table, Rate::hz(48_000)?)?;
    /// let mut out = [0.0f32; 100];
    /// Fallback.rds_shape(&mut st, &mut || 1, &mut out)?;
    /// st.reset();
    /// assert_eq!((st.position(), st.next_bit()), (0, 0));
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn reset(&mut self) {
        self.n = 0;
        self.fifo = [0.0; 4];
        self.next = -1;
    }

    /// The index of the next output sample (the samples produced since construction or
    /// [`RdsShapeState::reset`]).
    pub fn position(&self) -> u64 {
        self.n
    }

    /// The index of the next bit the shaper will fetch (the bits fetched so far).
    pub fn next_bit(&self) -> u64 {
        u64::try_from(self.next).unwrap_or(0)
    }

    /// The pulse table.
    pub fn table(&self) -> &[f32] {
        self.table.as_ref()
    }

    /// The sample rate in hertz.
    pub fn rate_hz(&self) -> f64 {
        self.fs
    }

    pub(crate) fn parts_mut(&mut self) -> RdsShapeParts<'_> {
        RdsShapeParts {
            table: self.table.as_ref(),
            fs: self.fs,
            n: &mut self.n,
            fifo: &mut self.fifo,
            next: &mut self.next,
        }
    }
}

impl<Tab: AsRef<[f32]>> Timing for RdsShapeState<Tab> {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    /// Zero: the output sample `n` belongs to the time `n / fs` of the bit clock.
    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// The RDS pulse shaper: bit amplitudes to the shaped biphase baseband.
///
/// ```
/// use sdrk_core::kernels::rds::{rds_pulse_table, RdsShape, RdsShapeState, RDS_TABLE_LEN};
/// use sdrk_core::{Fallback, Rate};
/// let mut table = [0.0f32; RDS_TABLE_LEN];
/// rds_pulse_table(&mut table)?;
/// let mut st = RdsShapeState::new(table, Rate::hz(192_000)?)?;
/// // Bit 0 alone (+1), all others 0: the output is the pulse from its centre on.
/// let mut k = 0;
/// let mut bits = || { k += 1; if k == 1 { 1 } else { 0 } };
/// let mut out = [0.0f32; 400];
/// Fallback.rds_shape(&mut st, &mut bits, &mut out)?;
/// assert_eq!(out[0], 0.0);
/// // Samples 1..40 lie in the second chip of the bit, which is negative.
/// assert!(out[1..40].iter().all(|&y| y < 0.0));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait RdsShape {
    /// Fills `output` with the next samples of the baseband. `bits` gives the amplitude of the
    /// next bit (normally `+-1`) and is called whenever a new bit enters the window: bit `k` as
    /// soon as a sample needs it (at most one call per bit, in order, bit 0 to 2 before the
    /// first sample).
    ///
    /// Per sample `n` (no FMA): the position `x = (n 1187.5) / fs` in f64,
    /// `k0 = floor(x) - 1`; the bits `k0 .. k0 + 3` (bits `k < 0` are 0); for `j = 0 .. 3`:
    /// `u = ((x - (k0 + j)) + 2) 256` in f64, `i = min(floor(u), 1023)`,
    /// `fr = (u - i) as f32`, `v = t[i] + (t[i + 1] - t[i]) fr` in f32; the output is
    /// `acc = 0`, `acc = acc + a_j v` with `j` ascending, in f32. Bit-exact between `Fallback`
    /// and `Reference`.
    ///
    /// # Errors
    /// `Internal` if a table index is out of range (the constructor makes this impossible).
    fn rds_shape<Tab: AsRef<[f32]>>(
        &self,
        s: &mut RdsShapeState<Tab>,
        bits: &mut dyn FnMut() -> i8,
        output: &mut [f32],
    ) -> Result<()>;
}
