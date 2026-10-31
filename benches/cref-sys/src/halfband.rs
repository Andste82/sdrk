//! The references of the halfband kernels (`halfband_decimate`, `halfband_decimate_real`,
//! `real_to_complex`):
//!
//! - liquid-dsp 1.8.3 (MIT, cfg `has_liquidhb`, `benches/refs/liquid-halfband.c`): the halfband
//!   decimators [`LiquidResamp2Crcf`] (`resamp2_crcf_decim_execute`) and [`LiquidResamp2Rrrf`]
//!   (`resamp2_rrrf_decim_execute`) and the Hilbert transform decimator [`LiquidFirhilbf`]
//!   (`firhilbf_decim_execute_block`), with liquid's own dot product: the SIMD loop of the level
//!   (SSE, AVX, AVX-512F, NEON; the portable loop at `simd128`) or, on request, the portable
//!   loop;
//! - libairspy 1.0.10 (MIT, cfg `has_airspy`, `benches/refs/airspy-iqconv.c`): the real-to-complex
//!   converters [`AirspyFloat`] (`iqconverter_float`, generic or with SSE2 on x86-64) and
//!   [`AirspyInt16`] (`iqconverter_int16`), the conversion without `remove_dc`.
//!
//! Both are built with the taps of the caller (SDRk's, so that the accuracy of both sides is
//! comparable); neither library designs a filter here.

use core::ffi::{c_int, c_uint, c_void};

#[cfg(has_liquidhb)]
mod ffi {
    use core::ffi::{c_int, c_uint, c_void};

    unsafe extern "C" {
        pub fn sdrk_liquid_resamp2_crcf_create(
            h: *const f32,
            m: c_uint,
            scale: f32,
            method: c_int,
        ) -> *mut c_void;
        pub fn sdrk_liquid_resamp2_crcf_reset(q: *mut c_void);
        pub fn sdrk_liquid_resamp2_crcf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_resamp2_crcf_decim_block(
            q: *mut c_void,
            x: *mut [f32; 2],
            n: c_uint,
            y: *mut [f32; 2],
        );
        pub fn sdrk_liquid_resamp2_rrrf_create(
            h: *const f32,
            m: c_uint,
            scale: f32,
            method: c_int,
        ) -> *mut c_void;
        pub fn sdrk_liquid_resamp2_rrrf_reset(q: *mut c_void);
        pub fn sdrk_liquid_resamp2_rrrf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_resamp2_rrrf_decim_block(
            q: *mut c_void,
            x: *mut f32,
            n: c_uint,
            y: *mut f32,
        );
        pub fn sdrk_liquid_firhilbf_create(h: *const f32, m: c_uint, method: c_int) -> *mut c_void;
        pub fn sdrk_liquid_firhilbf_reset(q: *mut c_void);
        pub fn sdrk_liquid_firhilbf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_firhilbf_decim_block(
            q: *mut c_void,
            x: *mut f32,
            n: c_uint,
            y: *mut [f32; 2],
        );
    }
}

#[cfg(has_airspy)]
mod airspy_ffi {
    use core::ffi::{c_int, c_void};

    unsafe extern "C" {
        pub fn sdrk_airspy_generic_float_new(taps: *const f32, len: c_int) -> *mut c_void;
        pub fn sdrk_airspy_generic_float_delete(q: *mut c_void);
        pub fn sdrk_airspy_generic_float_restart(q: *mut c_void);
        pub fn sdrk_airspy_generic_float_convert(
            q: *mut c_void,
            input: *const f32,
            out: *mut f32,
            len: c_int,
        );
        pub fn sdrk_airspy_int16_new(taps: *const i16, len: c_int) -> *mut c_void;
        pub fn sdrk_airspy_int16_delete(q: *mut c_void);
        pub fn sdrk_airspy_int16_restart(q: *mut c_void);
        pub fn sdrk_airspy_int16_convert(
            q: *mut c_void,
            input: *const i16,
            out: *mut i16,
            len: c_int,
        );
    }

    #[cfg(target_arch = "x86_64")]
    unsafe extern "C" {
        pub fn sdrk_airspy_sse2_float_new(taps: *const f32, len: c_int) -> *mut c_void;
        pub fn sdrk_airspy_sse2_float_delete(q: *mut c_void);
        pub fn sdrk_airspy_sse2_float_restart(q: *mut c_void);
        pub fn sdrk_airspy_sse2_float_convert(
            q: *mut c_void,
            input: *const f32,
            out: *mut f32,
            len: c_int,
        );
    }
}

/// The block length as liquid's `unsigned int`.
fn count(n: usize) -> c_uint {
    c_uint::try_from(n).expect("block too long")
}

/// The prototype length as `4 m + 1`: `Some(m)` for a valid length (at least 5).
fn semi_length(prototype: &[f32]) -> Option<c_uint> {
    let n = prototype.len();
    (n >= 5 && n % 4 == 1).then(|| count((n - 1) / 4))
}

macro_rules! liquid_resamp2 {
    ($(#[$m:meta])* $name:ident, $t:ty, $create:ident, $reset:ident, $destroy:ident, $block:ident) => {
        $(#[$m])*
        pub struct $name {
            #[cfg_attr(not(has_liquidhb), allow(dead_code))]
            q: *mut c_void,
        }

        impl $name {
            /// The decimator of the prototype `h` of `4 m + 1` taps (zero outer taps; the
            /// odd-offset taps are liquid's `h1`), with liquid's output scale `scale`; `simd`
            /// selects the SIMD dot product of the level, otherwise the portable one. `None`
            /// without the library or for an invalid length.
            #[must_use]
            pub fn new(h: &[f32], scale: f32, simd: bool) -> Option<Self> {
                let m = semi_length(h)?;
                #[cfg(has_liquidhb)]
                {
                    // SAFETY: `h` holds `4 m + 1` values, which the call copies.
                    let q = unsafe { ffi::$create(h.as_ptr(), m, scale, c_int::from(simd)) };
                    // Not `then_some`: it would build (and drop, so destroy) a null object.
                    if q.is_null() { None } else { Some(Self { q }) }
                }
                #[cfg(not(has_liquidhb))]
                {
                    let _ = (m, scale, simd);
                    None
                }
            }

            /// Clears the delay lines.
            pub fn reset(&mut self) {
                #[cfg(has_liquidhb)]
                // SAFETY: the object is valid until `drop`.
                unsafe {
                    ffi::$reset(self.q);
                }
            }
        }

        #[cfg(has_liquidhb)]
        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: the object came from the constructor and is destroyed once.
                unsafe { ffi::$destroy(self.q) }
            }
        }
    };
}

liquid_resamp2! {
    /// liquid's `resamp2_crcf` as a decimator (`resamp2_crcf_decim_execute`): complex samples
    /// in and out, real taps.
    LiquidResamp2Crcf, [f32; 2], sdrk_liquid_resamp2_crcf_create, sdrk_liquid_resamp2_crcf_reset,
    sdrk_liquid_resamp2_crcf_destroy, sdrk_liquid_resamp2_crcf_decim_block
}

liquid_resamp2! {
    /// liquid's `resamp2_rrrf` as a decimator (`resamp2_rrrf_decim_execute`): real samples in
    /// and out, real taps.
    LiquidResamp2Rrrf, f32, sdrk_liquid_resamp2_rrrf_create, sdrk_liquid_resamp2_rrrf_reset,
    sdrk_liquid_resamp2_rrrf_destroy, sdrk_liquid_resamp2_rrrf_decim_block
}

impl LiquidResamp2Crcf {
    /// `x.len() / 2` outputs from `x`.
    ///
    /// # Panics
    /// If `x` has an odd length or `y` is not half as long.
    pub fn decim(&mut self, x: &mut [[f32; 2]], y: &mut [[f32; 2]]) {
        assert!(
            x.len() % 2 == 0 && y.len() == x.len() / 2,
            "liquid resamp2: lengths"
        );
        let n = count(y.len());
        #[cfg(has_liquidhb)]
        // SAFETY: `x` holds `2 n` samples (liquid only reads them) and `y` holds `n`.
        unsafe {
            ffi::sdrk_liquid_resamp2_crcf_decim_block(self.q, x.as_mut_ptr(), n, y.as_mut_ptr());
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

impl LiquidResamp2Rrrf {
    /// `x.len() / 2` outputs from `x`.
    ///
    /// # Panics
    /// If `x` has an odd length or `y` is not half as long.
    pub fn decim(&mut self, x: &mut [f32], y: &mut [f32]) {
        assert!(
            x.len() % 2 == 0 && y.len() == x.len() / 2,
            "liquid resamp2: lengths"
        );
        let n = count(y.len());
        #[cfg(has_liquidhb)]
        // SAFETY: `x` holds `2 n` samples (liquid only reads them) and `y` holds `n`.
        unsafe {
            ffi::sdrk_liquid_resamp2_rrrf_decim_block(self.q, x.as_mut_ptr(), n, y.as_mut_ptr());
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

/// liquid's `firhilbf` as a decimator (`firhilbf_decim_execute_block`): real samples in,
/// complex samples at half the rate out. The output is `j` times the `real_to_complex` of SDRk
/// divided by the centre tap (the delay branch has the gain 1 where SDRk's has the centre tap,
/// and liquid's real part is the delay branch where SDRk's is the filter).
pub struct LiquidFirhilbf {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
}

impl LiquidFirhilbf {
    /// The transform of the prototype `h` of `4 m + 1` taps; `simd` as for
    /// [`LiquidResamp2Crcf::new`].
    #[must_use]
    pub fn new(h: &[f32], simd: bool) -> Option<Self> {
        let m = semi_length(h)?;
        #[cfg(has_liquidhb)]
        {
            // SAFETY: `h` holds `4 m + 1` values, which the call copies.
            let q = unsafe { ffi::sdrk_liquid_firhilbf_create(h.as_ptr(), m, c_int::from(simd)) };
            if q.is_null() { None } else { Some(Self { q }) }
        }
        #[cfg(not(has_liquidhb))]
        {
            let _ = (m, simd);
            None
        }
    }

    /// Clears the delay lines and the toggle.
    pub fn reset(&mut self) {
        #[cfg(has_liquidhb)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_firhilbf_reset(self.q);
        }
    }

    /// `x.len() / 2` outputs from `x`.
    ///
    /// # Panics
    /// If `x` has an odd length or `y` is not half as long.
    pub fn decim(&mut self, x: &mut [f32], y: &mut [[f32; 2]]) {
        assert!(
            x.len() % 2 == 0 && y.len() == x.len() / 2,
            "liquid firhilbf: lengths"
        );
        let n = count(y.len());
        #[cfg(has_liquidhb)]
        // SAFETY: `x` holds `2 n` samples (liquid only reads them) and `y` holds `n`.
        unsafe {
            ffi::sdrk_liquid_firhilbf_decim_block(self.q, x.as_mut_ptr(), n, y.as_mut_ptr());
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidFirhilbf {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_firhilbf_destroy(self.q) }
    }
}

/// The length of a libairspy kernel, `i32` as the C functions take it.
fn kernel_len(taps: usize) -> Option<c_int> {
    // The converters need `len / 2 + 1` taps to be the 4, 8, 12 or 24 of the unrolled loops
    // or any other count; they read `len` taps.
    c_int::try_from(taps).ok().filter(|n| *n >= 7 && n % 4 == 3)
}

/// libairspy's `iqconverter_float` (conversion without `remove_dc`): `len` real samples in,
/// `len / 2` interleaved complex samples out. The result is `-conj` of SDRk's `real_to_complex`
/// (libairspy's shift is the other sign of fs / 4), at the same sample positions.
pub struct AirspyFloat {
    #[cfg_attr(not(has_airspy), allow(dead_code))]
    q: *mut c_void,
    /// Whether the SSE2 build is used (x86-64 only).
    #[cfg_attr(not(all(has_airspy, target_arch = "x86_64")), allow(dead_code))]
    sse2: bool,
}

impl AirspyFloat {
    /// The converter with the 47 (or any `4 k + 3`) taps `taps`; `sse2` selects the build with
    /// the SSE2 intrinsics (x86-64 only, otherwise `None`). `None` without the library.
    #[must_use]
    pub fn new(taps: &[f32], sse2: bool) -> Option<Self> {
        let len = kernel_len(taps.len())?;
        #[cfg(has_airspy)]
        {
            // SAFETY: `taps` holds `len` values, which the call copies.
            let q = unsafe {
                #[cfg(target_arch = "x86_64")]
                if sse2 {
                    airspy_ffi::sdrk_airspy_sse2_float_new(taps.as_ptr(), len)
                } else {
                    airspy_ffi::sdrk_airspy_generic_float_new(taps.as_ptr(), len)
                }
                #[cfg(not(target_arch = "x86_64"))]
                if sse2 {
                    core::ptr::null_mut()
                } else {
                    airspy_ffi::sdrk_airspy_generic_float_new(taps.as_ptr(), len)
                }
            };
            if q.is_null() {
                None
            } else {
                Some(Self { q, sse2 })
            }
        }
        #[cfg(not(has_airspy))]
        {
            let _ = (len, sse2);
            None
        }
    }

    /// Clears the delay lines.
    pub fn reset(&mut self) {
        #[cfg(has_airspy)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            if self.sse2 {
                airspy_ffi::sdrk_airspy_sse2_float_restart(self.q);
                return;
            }
            airspy_ffi::sdrk_airspy_generic_float_restart(self.q);
        }
    }

    /// Converts `input` (an even count of real samples) into `out` (as many floats, interleaved
    /// I/Q).
    ///
    /// # Panics
    /// If the lengths differ or are odd.
    pub fn convert(&mut self, input: &[f32], out: &mut [f32]) {
        assert!(
            input.len() == out.len() && input.len() % 2 == 0,
            "libairspy: lengths"
        );
        let len = c_int::try_from(input.len()).expect("block too long");
        #[cfg(has_airspy)]
        // SAFETY: both buffers hold `len` floats and do not overlap.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            if self.sse2 {
                airspy_ffi::sdrk_airspy_sse2_float_convert(
                    self.q,
                    input.as_ptr(),
                    out.as_mut_ptr(),
                    len,
                );
                return;
            }
            airspy_ffi::sdrk_airspy_generic_float_convert(
                self.q,
                input.as_ptr(),
                out.as_mut_ptr(),
                len,
            );
        }
        #[cfg(not(has_airspy))]
        let _ = len;
    }
}

#[cfg(has_airspy)]
impl Drop for AirspyFloat {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is freed once.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            if self.sse2 {
                airspy_ffi::sdrk_airspy_sse2_float_delete(self.q);
                return;
            }
            airspy_ffi::sdrk_airspy_generic_float_delete(self.q);
        }
    }
}

/// libairspy's `iqconverter_int16` (conversion without `remove_dc`): `len` real Q15 samples in,
/// `len / 2` interleaved complex Q15 samples out. It truncates (`>> 15`) in a 32-bit
/// accumulator and halves the delay branch before the centre tap, a lower accuracy class than
/// SDRk's exact 64-bit sum. The result is `-conj` of SDRk's `real_to_complex`.
pub struct AirspyInt16 {
    #[cfg_attr(not(has_airspy), allow(dead_code))]
    q: *mut c_void,
}

impl AirspyInt16 {
    /// The converter with the Q15 taps `taps` (47 or any `4 k + 3`; the centre tap is not used,
    /// libairspy hard-wires 0.5).
    #[must_use]
    pub fn new(taps: &[i16]) -> Option<Self> {
        let len = kernel_len(taps.len())?;
        #[cfg(has_airspy)]
        {
            // SAFETY: `taps` holds `len` values, which the call copies.
            let q = unsafe { airspy_ffi::sdrk_airspy_int16_new(taps.as_ptr(), len) };
            if q.is_null() { None } else { Some(Self { q }) }
        }
        #[cfg(not(has_airspy))]
        {
            let _ = len;
            None
        }
    }

    /// Clears the delay lines.
    pub fn reset(&mut self) {
        #[cfg(has_airspy)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            airspy_ffi::sdrk_airspy_int16_restart(self.q);
        }
    }

    /// As [`AirspyFloat::convert`].
    ///
    /// # Panics
    /// If the lengths differ or are odd.
    pub fn convert(&mut self, input: &[i16], out: &mut [i16]) {
        assert!(
            input.len() == out.len() && input.len() % 2 == 0,
            "libairspy: lengths"
        );
        let len = c_int::try_from(input.len()).expect("block too long");
        #[cfg(has_airspy)]
        // SAFETY: both buffers hold `len` samples and do not overlap.
        unsafe {
            airspy_ffi::sdrk_airspy_int16_convert(self.q, input.as_ptr(), out.as_mut_ptr(), len);
        }
        #[cfg(not(has_airspy))]
        let _ = len;
    }
}

#[cfg(has_airspy)]
impl Drop for AirspyInt16 {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is freed once.
        unsafe { airspy_ffi::sdrk_airspy_int16_delete(self.q) }
    }
}
