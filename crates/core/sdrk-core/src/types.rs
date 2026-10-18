//! The four sample types and the traits that tie them together.

use num_complex::Complex64;

/// Complex f32 sample; layout = interleaved little-endian cf32.
pub type C32 = num_complex::Complex32;

/// Q1.15 fixed-point real sample. value = raw / 32768.
///
/// ```
/// use sdrk_core::Q15;
/// assert_eq!(Q15::from_f32(0.5), Q15(16384));
/// assert_eq!(Q15(16384).to_f32(), 0.5);
/// ```
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
#[repr(transparent)]
pub struct Q15(pub i16);

/// Complex Q15 sample; byte-identical to interleaved little-endian cs16 (I first).
///
/// ```
/// use sdrk_core::{CQ15, Q15};
/// let x = CQ15 { re: Q15(1), im: Q15(2) };
/// assert_eq!(bytemuck::bytes_of(&x), &[1, 0, 2, 0]);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CQ15 {
    /// In-phase component.
    pub re: Q15,
    /// Quadrature component.
    pub im: Q15,
}

/// Runtime tag of the four sample types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleType {
    /// Complex f32.
    C32,
    /// Complex Q15.
    CQ15,
    /// Real f32.
    F32,
    /// Real Q15.
    Q15,
}

impl SampleType {
    /// The tag of the sample type `T`.
    ///
    /// ```
    /// use sdrk_core::{SampleType, CQ15};
    /// assert_eq!(SampleType::of::<CQ15>(), SampleType::CQ15);
    /// ```
    pub const fn of<T: Sample>() -> Self {
        T::TYPE
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::C32 {}
    impl Sealed for super::CQ15 {}
    impl Sealed for f32 {}
    impl Sealed for super::Q15 {}
}

/// A sample type of SDRk. Sealed: implemented only for `C32`, `CQ15`, `f32` and `Q15`.
pub trait Sample:
    Copy
    + Default
    + PartialEq
    + Send
    + Sync
    + bytemuck::Pod
    + core::fmt::Debug
    + 'static
    + sealed::Sealed
{
    /// Runtime tag of this type.
    const TYPE: SampleType;
    /// The zero sample.
    const ZERO: Self;
}

/// A real sample type (`f32` or `Q15`).
pub trait RealSample: Sample {
    /// Converts to f64 (exact).
    fn to_f64(self) -> f64;
    /// Converts from f64, rounding according to the numeric conventions.
    fn from_f64_round(x: f64) -> Self;
}

/// A complex sample type (`C32` or `CQ15`).
pub trait ComplexSample: Sample {
    /// The real component type.
    type Real: RealSample;
    /// In-phase component.
    fn re(self) -> Self::Real;
    /// Quadrature component.
    fn im(self) -> Self::Real;
    /// Builds a sample from its components.
    fn new(re: Self::Real, im: Self::Real) -> Self;
    /// Converts to a complex f64 (exact).
    fn to_c64(self) -> Complex64;
    /// Converts from a complex f64, rounding each component.
    fn from_c64_round(x: Complex64) -> Self;
}

impl Sample for f32 {
    const TYPE: SampleType = SampleType::F32;
    const ZERO: Self = 0.0;
}
impl Sample for Q15 {
    const TYPE: SampleType = SampleType::Q15;
    const ZERO: Self = Q15(0);
}
impl Sample for C32 {
    const TYPE: SampleType = SampleType::C32;
    const ZERO: Self = C32::new(0.0, 0.0);
}
impl Sample for CQ15 {
    const TYPE: SampleType = SampleType::CQ15;
    const ZERO: Self = CQ15 {
        re: Q15(0),
        im: Q15(0),
    };
}

impl RealSample for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_f64_round(x: f64) -> Self {
        x as f32
    }
}

impl RealSample for Q15 {
    fn to_f64(self) -> f64 {
        f64::from(self.0) / 32768.0
    }
    fn from_f64_round(x: f64) -> Self {
        crate::num::q15_from_f64(x)
    }
}

impl ComplexSample for C32 {
    type Real = f32;
    fn re(self) -> f32 {
        self.re
    }
    fn im(self) -> f32 {
        self.im
    }
    fn new(re: f32, im: f32) -> Self {
        C32::new(re, im)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
    fn from_c64_round(x: Complex64) -> Self {
        C32::new(x.re as f32, x.im as f32)
    }
}

impl ComplexSample for CQ15 {
    type Real = Q15;
    fn re(self) -> Q15 {
        self.re
    }
    fn im(self) -> Q15 {
        self.im
    }
    fn new(re: Q15, im: Q15) -> Self {
        CQ15 { re, im }
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self.re.to_f64(), self.im.to_f64())
    }
    fn from_c64_round(x: Complex64) -> Self {
        CQ15 {
            re: Q15::from_f64_round(x.re),
            im: Q15::from_f64_round(x.im),
        }
    }
}

impl Q15 {
    /// Largest value, `32767 / 32768`.
    pub const MAX: Q15 = Q15(32767);
    /// Smallest value, `-1.0`.
    pub const MIN: Q15 = Q15(-32768);

    /// Converts from f32 (computed in f64; NaN gives 0, saturating).
    pub fn from_f32(x: f32) -> Q15 {
        crate::num::q15_from_f32(x)
    }

    /// Converts to f32 (exact).
    pub fn to_f32(self) -> f32 {
        f32::from(self.0) / 32768.0
    }
}

impl CQ15 {
    /// Converts from a complex f32 sample.
    pub fn from_c32(x: C32) -> CQ15 {
        CQ15 {
            re: Q15::from_f32(x.re),
            im: Q15::from_f32(x.im),
        }
    }

    /// Converts to a complex f32 sample (exact).
    pub fn to_c32(self) -> C32 {
        C32::new(self.re.to_f32(), self.im.to_f32())
    }
}
