//! The FFT kernel: power-of-two sizes 16 to 65 536, in-place radix-2, for `C32` and `CQ15`,
//! plus a real-input FFT and its inverse for `f32`.
//!
//! One rule defines the scale of every result: the unnormalised sum
//! `sum x[n] e^(-+j 2 pi k n / N)` equals `output * 2^exponent`, where `exponent` is the
//! [`FftScale`] returned by the call. See `docs/kernels.md` for the conventions table.

use core::marker::PhantomData;

use num_complex::Complex64;

use crate::error::{Error, Result};
use crate::types::{C32, ComplexSample};

#[cfg(feature = "static-twiddles")]
use crate::types::CQ15;

/// Smallest supported FFT size.
const MIN_N: usize = 16;
/// Largest supported FFT size.
const MAX_N: usize = 65_536;

/// Transform direction.
///
/// ```
/// use sdrk_core::kernels::fft::Direction;
/// assert_ne!(Direction::Forward, Direction::Inverse);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// `e^(-j 2 pi k n / N)`.
    Forward,
    /// `e^(+j 2 pi k n / N)`.
    Inverse,
}

/// How a fixed-point FFT keeps its intermediate values in range.
///
/// ```
/// use sdrk_core::kernels::fft::Scaling;
/// assert_ne!(Scaling::Default, Scaling::BlockFloat);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Scaling {
    /// The default of the sample type: `C32` is unnormalised forward and `1/N` inverse, `CQ15`
    /// halves after every stage in both directions (fixed `1/N`).
    Default,
    /// Block floating point (`CQ15` only): before every stage the shift (0, 1 or 2 bits) is
    /// chosen from the largest component, and the sum of the shifts is the exponent.
    BlockFloat,
}

/// The scale of an FFT result: the unnormalised sum equals `output * 2^exponent`.
///
/// ```
/// use sdrk_core::kernels::fft::FftScale;
/// assert_eq!(FftScale::default().exponent, 0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FftScale {
    /// Power of two that restores the unnormalised transform from the output.
    pub exponent: i32,
}

/// `log2 n` for a supported FFT size, else `InvalidParameter`.
pub(crate) fn log2_of(n: usize) -> Result<u32> {
    if n.is_power_of_two() && (MIN_N..=MAX_N).contains(&n) {
        Ok(n.trailing_zeros())
    } else {
        Err(Error::InvalidParameter(
            "fft size must be a power of two from 16 to 65536",
        ))
    }
}

/// `e^(-j 2 pi k / n)` as a complex f64. Used by plan construction and by the build script
/// that emits the static tables, so both agree bit for bit.
pub(crate) fn twiddle_f64(k: usize, n: usize) -> Complex64 {
    let theta = 2.0 * core::f64::consts::PI * (k as f64 / n as f64);
    Complex64::new(libm::cos(theta), -libm::sin(theta))
}

/// A prepared FFT of one size: the size and the twiddle table `tw[k] = e^(-j 2 pi k / n)`.
///
/// The twiddles live in caller-provided storage (`[T; N/2]`, `&mut [T]`, `Vec<T>` or a
/// `static`). A plan is read-only after construction, so one plan can serve many calls.
///
/// ```
/// use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
/// use sdrk_core::{Fallback, C32};
/// let plan = FftPlan::<C32, _>::new(16, [C32::default(); 8])?;
/// let mut data = [C32::default(); 16];
/// data[0] = C32::new(1.0, 0.0);
/// Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::Default)?;
/// assert!(data.iter().all(|x| (x.re - 1.0).abs() < 1e-6 && x.im.abs() < 1e-6));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct FftPlan<T: ComplexSample, Tw: AsRef<[T]>> {
    pub(crate) n: usize,
    pub(crate) log2n: u32,
    pub(crate) twiddles: Tw,
    /// 1 for `new`/`from_twiddles`, `4096 / n` for `from_static_4096`.
    pub(crate) tw_stride: usize,
    _sample: PhantomData<fn() -> T>,
}

impl<T: ComplexSample, Tw: AsRef<[T]> + AsMut<[T]>> FftPlan<T, Tw> {
    /// Creates a plan of size `n` and fills `twiddles` with `e^(-j 2 pi k / n)`, computed in
    /// f64 with `libm::cos`/`libm::sin` and rounded to the sample type (a cast for `C32`, the
    /// Q15 rule for `CQ15`).
    ///
    /// `n` must be a power of two with `16 <= n <= 65536` (`InvalidParameter`) and
    /// `twiddles.len()` must be `n / 2` (`LengthMismatch`).
    ///
    /// ```
    /// use sdrk_core::kernels::fft::FftPlan;
    /// use sdrk_core::{C32, CQ15};
    /// let plan = FftPlan::<CQ15, _>::new(32, [CQ15::default(); 16])?;
    /// assert_eq!(plan.len(), 32);
    /// assert!(FftPlan::<C32, _>::new(24, [C32::default(); 12]).is_err());
    /// assert!(FftPlan::<C32, _>::new(32, [C32::default(); 8]).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn new(n: usize, mut twiddles: Tw) -> Result<Self> {
        let log2n = log2_of(n)?;
        check_twiddle_len(twiddles.as_ref().len(), n / 2)?;
        for (k, w) in twiddles.as_mut().iter_mut().enumerate() {
            *w = T::from_c64_round(twiddle_f64(k, n));
        }
        Ok(Self {
            n,
            log2n,
            twiddles,
            tw_stride: 1,
            _sample: PhantomData,
        })
    }
}

impl<T: ComplexSample, Tw: AsRef<[T]>> FftPlan<T, Tw> {
    /// Creates a plan from precomputed twiddles (for example a `static` table). Only the
    /// length is validated, the values are trusted.
    ///
    /// ```
    /// use sdrk_core::kernels::fft::FftPlan;
    /// use sdrk_core::C32;
    /// let table = [C32::new(1.0, 0.0); 8];
    /// let plan = FftPlan::<C32, _>::from_twiddles(16, &table[..])?;
    /// assert_eq!(plan.len(), 16);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn from_twiddles(n: usize, twiddles: Tw) -> Result<Self> {
        let log2n = log2_of(n)?;
        check_twiddle_len(twiddles.as_ref().len(), n / 2)?;
        Ok(Self {
            n,
            log2n,
            twiddles,
            tw_stride: 1,
            _sample: PhantomData,
        })
    }

    /// The transform size `N`.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false: a plan has at least 16 points.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// `log2` of the transform size.
    pub fn log2_len(&self) -> u32 {
        self.log2n
    }

    /// The twiddle storage: `e^(-j 2 pi k / N)` is at index `k * twiddle_stride()`.
    pub fn twiddles(&self) -> &[T] {
        self.twiddles.as_ref()
    }

    /// Distance between consecutive twiddles of this size in [`twiddles`](Self::twiddles)
    /// (1, or `4096 / N` for a plan over the static 4096-point table).
    pub fn twiddle_stride(&self) -> usize {
        self.tw_stride
    }
}

fn check_real_size(n: usize) -> Result<()> {
    if n.is_power_of_two() {
        Ok(())
    } else {
        Err(Error::InvalidParameter(
            "real fft size must be a power of two from 32 to 131072",
        ))
    }
}

fn check_twiddle_len(got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(Error::LengthMismatch { expected, got })
    }
}

/// The 4096-point twiddle table of `C32`, `e^(-j 2 pi k / 4096)` for `k < 2048`
/// (feature `static-twiddles`; 16 KiB of flash).
#[cfg(feature = "static-twiddles")]
pub use crate::tables::TWIDDLES_C32_4096;
/// The 4096-point twiddle table of `CQ15` (feature `static-twiddles`; 8 KiB of flash).
#[cfg(feature = "static-twiddles")]
pub use crate::tables::TWIDDLES_CQ15_4096;

#[cfg(feature = "static-twiddles")]
fn static_plan<T: ComplexSample>(
    n: usize,
    table: &'static [T],
) -> Result<FftPlan<T, &'static [T]>> {
    let log2n = log2_of(n)?;
    if n > 4096 {
        return Err(Error::InvalidParameter(
            "the static twiddle table serves sizes up to 4096",
        ));
    }
    Ok(FftPlan {
        n,
        log2n,
        twiddles: table,
        tw_stride: 4096 / n,
        _sample: PhantomData,
    })
}

#[cfg(feature = "static-twiddles")]
impl FftPlan<C32, &'static [C32]> {
    /// A plan of size `n <= 4096` over the static table [`TWIDDLES_C32_4096`]; it uses every
    /// `4096 / n`-th entry (`tw_stride`). Flash cost of the table: 16 KiB.
    ///
    /// ```
    /// use sdrk_core::kernels::fft::FftPlan;
    /// use sdrk_core::C32;
    /// let plan = FftPlan::<C32, _>::from_static_4096(256)?;
    /// assert_eq!(plan.len(), 256);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn from_static_4096(n: usize) -> Result<Self> {
        static_plan(n, &TWIDDLES_C32_4096)
    }
}

#[cfg(feature = "static-twiddles")]
impl FftPlan<CQ15, &'static [CQ15]> {
    /// A plan of size `n <= 4096` over the static table [`TWIDDLES_CQ15_4096`]; it uses every
    /// `4096 / n`-th entry (`tw_stride`). Flash cost of the table: 8 KiB.
    ///
    /// ```
    /// use sdrk_core::kernels::fft::FftPlan;
    /// use sdrk_core::CQ15;
    /// let plan = FftPlan::<CQ15, _>::from_static_4096(1024)?;
    /// assert_eq!(plan.len(), 1024);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn from_static_4096(n: usize) -> Result<Self> {
        static_plan(n, &TWIDDLES_CQ15_4096)
    }
}

/// A complex FFT, in place.
///
/// | Type | Direction | Output | `exponent` |
/// | --- | --- | --- | --- |
/// | `C32` | Forward | `sum x e^(-j..)` (like `numpy.fft.fft`) | 0 |
/// | `C32` | Inverse | `(1/N) sum X e^(+j..)` (like `numpy.fft.ifft`) | `log2 N` |
/// | `CQ15` | both, `Default` | the same, divided by `N` (halved after every stage) | `log2 N` |
/// | `CQ15` | both, `BlockFloat` | transform divided by `2^e` | `e` (sum of the stage shifts) |
///
/// ```
/// use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
/// use sdrk_core::{Fallback, CQ15, Q15};
/// let plan = FftPlan::<CQ15, _>::new(16, [CQ15::default(); 8])?;
/// let mut data = [CQ15::default(); 16];
/// data[1] = CQ15 { re: Q15(16384), im: Q15(0) };
/// let scale = Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::BlockFloat)?;
/// // The impulse (value 0.5 at n = 1) has a flat spectrum of magnitude 0.5; BlockFloat did
/// // not have to shrink it much.
/// assert!(scale.exponent <= 4);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Fft<T: ComplexSample> {
    /// Transforms `data` in place. `data.len()` must equal `plan.len()` (`LengthMismatch`).
    /// `Scaling::BlockFloat` with `C32` is `Unsupported`. Returns the scale of the result.
    fn fft<Tw: AsRef<[T]>>(
        &self,
        plan: &FftPlan<T, Tw>,
        data: &mut [T],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale>;
}

/// A prepared real-input FFT of size `N`: a complex plan of size `N / 2` and the post-processing
/// twiddles `post[k] = e^(-j 2 pi k / N)` for `k < N / 2`. One plan serves `rfft` and `irfft`.
///
/// ```
/// use sdrk_core::kernels::fft::RealFftPlan;
/// use sdrk_core::C32;
/// let plan = RealFftPlan::new(64, [C32::default(); 16], [C32::default(); 32])?;
/// assert_eq!(plan.len(), 64);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct RealFftPlan<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>> {
    pub(crate) inner: FftPlan<C32, Tw>,
    pub(crate) post: Tw2,
}

impl<Tw: AsRef<[C32]> + AsMut<[C32]>, Tw2: AsRef<[C32]> + AsMut<[C32]>> RealFftPlan<Tw, Tw2> {
    /// Creates a plan for `n` real samples and fills both tables. `n` is a power of two with
    /// `32 <= n <= 131072` (the inner complex FFT has `n / 2` points); `inner_twiddles.len()`
    /// must be `n / 4` and `post_twiddles.len()` must be `n / 2`.
    ///
    /// ```
    /// use sdrk_core::kernels::fft::RealFftPlan;
    /// use sdrk_core::C32;
    /// assert!(RealFftPlan::new(16, [C32::default(); 4], [C32::default(); 8]).is_err());
    /// assert!(RealFftPlan::new(32, [C32::default(); 8], [C32::default(); 16]).is_ok());
    /// ```
    pub fn new(n: usize, inner_twiddles: Tw, mut post_twiddles: Tw2) -> Result<Self> {
        check_real_size(n)?;
        let inner = FftPlan::new(n / 2, inner_twiddles)?;
        check_twiddle_len(post_twiddles.as_ref().len(), n / 2)?;
        for (k, w) in post_twiddles.as_mut().iter_mut().enumerate() {
            *w = C32::from_c64_round(twiddle_f64(k, n));
        }
        Ok(Self {
            inner,
            post: post_twiddles,
        })
    }
}

impl<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>> RealFftPlan<Tw, Tw2> {
    /// Creates a plan from precomputed tables; only the sizes are validated.
    ///
    /// ```
    /// use sdrk_core::kernels::fft::RealFftPlan;
    /// use sdrk_core::C32;
    /// let plan = RealFftPlan::from_twiddles(32, [C32::default(); 8], [C32::default(); 16])?;
    /// assert_eq!(plan.len(), 32);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn from_twiddles(n: usize, inner_twiddles: Tw, post_twiddles: Tw2) -> Result<Self> {
        check_real_size(n)?;
        let inner = FftPlan::from_twiddles(n / 2, inner_twiddles)?;
        check_twiddle_len(post_twiddles.as_ref().len(), n / 2)?;
        Ok(Self {
            inner,
            post: post_twiddles,
        })
    }

    /// The number of real samples `N`.
    pub fn len(&self) -> usize {
        self.inner.n * 2
    }

    /// Always false: a plan has at least 32 points.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The plan of the inner complex FFT of `N / 2` points.
    pub fn inner(&self) -> &FftPlan<C32, Tw> {
        &self.inner
    }

    /// The twiddles of the post-processing: `e^(-j 2 pi k / N)` at index `k < N / 2`.
    pub fn post_twiddles(&self) -> &[C32] {
        self.post.as_ref()
    }
}

/// The real-input FFT of `f32` data and its inverse.
///
/// ```
/// use sdrk_core::kernels::fft::{RealFft, RealFftPlan};
/// use sdrk_core::{Fallback, C32};
/// let plan = RealFftPlan::new(32, [C32::default(); 8], [C32::default(); 16])?;
/// let x: Vec<f32> = (0..32).map(|n| (n % 2) as f32).collect(); // alternating 0, 1
/// let mut bins = [C32::default(); 17];
/// let mut scratch = [C32::default(); 16];
/// Fallback.rfft(&plan, &x, &mut bins, &mut scratch)?;
/// assert!((bins[0].re - 16.0).abs() < 1e-4); // DC = sum
/// assert!((bins[16].re + 16.0).abs() < 1e-4); // Nyquist = -sum of the odd samples
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait RealFft {
    /// Forward. `input`: `N` real samples; `output`: `N / 2 + 1` complex bins
    /// (`numpy.fft.rfft`); `scratch`: `N / 2` complex. Exponent 0. Wrong lengths give
    /// `LengthMismatch`.
    fn rfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[f32],
        output: &mut [C32],
        scratch: &mut [C32],
    ) -> Result<FftScale>;

    /// Inverse. `input`: `N / 2 + 1` complex bins; `output`: `N` real samples
    /// (`numpy.fft.irfft(X, n=N)`); `scratch`: `N / 2` complex. The imaginary parts of bin 0 and
    /// bin `N / 2` are ignored, as in `NumPy`. Exponent `log2 N`. Wrong lengths give
    /// `LengthMismatch`.
    fn irfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
        &self,
        plan: &RealFftPlan<Tw, Tw2>,
        input: &[C32],
        output: &mut [f32],
        scratch: &mut [C32],
    ) -> Result<FftScale>;
}

/// Checks the buffer lengths of `rfft` and `irfft` (real samples, bins, scratch).
pub(crate) fn check_real_lengths(
    n: usize,
    real_len: usize,
    bins_len: usize,
    scratch_len: usize,
) -> Result<()> {
    crate::kernels::check_len(n, real_len)?;
    crate::kernels::check_len(n / 2 + 1, bins_len)?;
    crate::kernels::check_len(n / 2, scratch_len)
}

/// The error of a violated length invariant of [`rfft_with`] and [`irfft_with`] (the lengths are
/// checked by `check_real_lengths` first).
fn bins_error() -> Error {
    Error::Internal("rfft bins")
}

/// `x[i]`, or the error of [`bins_error`].
#[inline]
fn at(x: &[C32], i: usize) -> Result<C32> {
    x.get(i).copied().ok_or_else(bins_error)
}

/// Complex product `a * b` in f32 without FMA (the order of the specification).
#[inline]
fn cmul(a: C32, b: C32) -> C32 {
    C32::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// [`RealFft::rfft`] with the inner `N / 2`-point transform of `fft`: the fallback passes itself,
/// and a backend can pass its own `C32` FFT (and compile this code at its level; the SIMD backend
/// has a post-processing of its own for `rfft` and uses [`irfft_with`]). The packing and the
/// post-processing are those of the fallback, so the result equals the fallback's when `fft`
/// does.
///
/// # Errors
/// As [`RealFft::rfft`], and the errors of `fft`.
#[allow(
    clippy::inline_always,
    reason = "a backend may call it at its dispatched level, whose target features it must inherit"
)]
#[allow(
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    reason = "`(a + b) * 0.5` is the specified, exact halving; the names are those of the formulas"
)]
#[inline(always)]
pub fn rfft_with<F: Fft<C32>, Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
    fft: &F,
    plan: &RealFftPlan<Tw, Tw2>,
    input: &[f32],
    output: &mut [C32],
    scratch: &mut [C32],
) -> Result<FftScale> {
    let n = plan.len();
    check_real_lengths(n, input.len(), output.len(), scratch.len())?;
    let h = n / 2;
    for (z, pair) in scratch.iter_mut().zip(input.chunks_exact(2)) {
        if let [re, im] = *pair {
            *z = C32::new(re, im);
        }
    }
    Fft::<C32>::fft(
        fft,
        &plan.inner,
        scratch,
        Direction::Forward,
        Scaling::Default,
    )?;
    let post = plan.post.as_ref();
    let z: &[C32] = scratch;
    let z0 = at(z, 0)?;
    // X[k] from Z[k], Z[N/2 - k] and the twiddle `w` (bins 0 and N/2 use Z[0] twice).
    let bin = |zk: C32, zm: C32, w: C32| {
        // E = (Z[k] + conj(Z[N/2-k])) / 2, O = (Z[k] - conj(Z[N/2-k])) / 2.
        let e = C32::new((zk.re + zm.re) * 0.5, (zk.im - zm.im) * 0.5);
        let o = C32::new((zk.re - zm.re) * 0.5, (zk.im + zm.im) * 0.5);
        let p = cmul(w, o);
        // X[k] = E - j * p.
        C32::new(e.re + p.im, e.im - p.re)
    };
    let (first, rest) = output.split_first_mut().ok_or_else(bins_error)?;
    let (last, middle) = rest.split_last_mut().ok_or_else(bins_error)?;
    *first = bin(z0, z0, at(post, 0)?);
    *last = bin(z0, z0, C32::new(-1.0, 0.0));
    // Bins 1 .. N/2 - 1 in one pass without index checks: Z[k] forwards, Z[N/2 - k] backwards.
    let zs = z.get(1..h).ok_or_else(bins_error)?;
    let w = post.get(1..h).ok_or_else(bins_error)?;
    for (((out, &zk), &zm), &w) in middle.iter_mut().zip(zs).zip(zs.iter().rev()).zip(w) {
        *out = bin(zk, zm, w);
    }
    Ok(FftScale { exponent: 0 })
}

/// [`RealFft::irfft`] with the inner `N / 2`-point transform of `fft` (see [`rfft_with`]).
///
/// # Errors
/// As [`RealFft::irfft`], and the errors of `fft`.
#[allow(
    clippy::inline_always,
    reason = "a backend may call it at its dispatched level, whose target features it must inherit"
)]
#[allow(
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    reason = "`(a + b) * 0.5` is the specified, exact halving; the names are those of the formulas"
)]
#[inline(always)]
pub fn irfft_with<F: Fft<C32>, Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(
    fft: &F,
    plan: &RealFftPlan<Tw, Tw2>,
    input: &[C32],
    output: &mut [f32],
    scratch: &mut [C32],
) -> Result<FftScale> {
    let n = plan.len();
    check_real_lengths(n, output.len(), input.len(), scratch.len())?;
    let h = n / 2;
    let post = plan.post.as_ref();
    // Z[k] from X[k], X[N/2 - k] and the twiddle `w`.
    let packed = |xk: C32, xm: C32, w: C32| {
        let e = C32::new((xk.re + xm.re) * 0.5, (xk.im - xm.im) * 0.5);
        let d = C32::new((xk.re - xm.re) * 0.5, (xk.im + xm.im) * 0.5);
        let d = cmul(d, C32::new(w.re, -w.im));
        // Z[k] = E + j * D.
        C32::new(e.re - d.im, e.im + d.re)
    };
    // Bin 0 and bin N/2 are real by definition.
    let (x0, xh) = (at(input, 0)?, at(input, h)?);
    let (first, rest) = scratch.split_first_mut().ok_or_else(bins_error)?;
    *first = packed(C32::new(x0.re, 0.0), C32::new(xh.re, 0.0), at(post, 0)?);
    // Bins 1 .. N/2 - 1 in one pass without index checks: X[k] forwards, X[N/2 - k] backwards.
    let xs = input.get(1..h).ok_or_else(bins_error)?;
    let w = post.get(1..h).ok_or_else(bins_error)?;
    for (((z, &xk), &xm), &w) in rest.iter_mut().zip(xs).zip(xs.iter().rev()).zip(w) {
        *z = packed(xk, xm, w);
    }
    Fft::<C32>::fft(
        fft,
        &plan.inner,
        scratch,
        Direction::Inverse,
        Scaling::Default,
    )?;
    for (pair, z) in output.chunks_exact_mut(2).zip(scratch.iter()) {
        if let [even, odd] = pair {
            *even = z.re;
            *odd = z.im;
        }
    }
    Ok(FftScale {
        exponent: n.trailing_zeros() as i32,
    })
}
