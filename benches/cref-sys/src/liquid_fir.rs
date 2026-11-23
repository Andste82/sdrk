//! The references of the pulse filters of the QAM and PSK blocks (issue #169): liquid-dsp 1.8.3
//! (MIT, cfg `has_liquidhb`, `benches/refs/liquid-fir.c`), the FIR filter [`LiquidFirfiltCrcf`]
//! (`firfilt_crcf`) and the polyphase interpolator [`LiquidFirinterpCrcf`] (`firinterp_crcf`):
//! complex samples in and out, real taps, with liquid's own dot product: the SIMD loop of the
//! level (SSE, AVX, AVX-512F, NEON; the portable loop at `simd128`) or, on request, the
//! portable loop. Both are built with the taps of the caller (SDRk's, so that the accuracy of
//! both sides is comparable).

use core::ffi::{c_int, c_uint, c_void};

#[cfg(has_liquidhb)]
mod ffi {
    use core::ffi::{c_int, c_uint, c_void};

    unsafe extern "C" {
        pub fn sdrk_liquid_firinterp_crcf_create(
            h: *const f32,
            h_len: c_uint,
            m: c_uint,
            method: c_int,
        ) -> *mut c_void;
        pub fn sdrk_liquid_firinterp_crcf_reset(q: *mut c_void);
        pub fn sdrk_liquid_firinterp_crcf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_firinterp_crcf_execute_block(
            q: *mut c_void,
            x: *mut [f32; 2],
            n: c_uint,
            y: *mut [f32; 2],
        );
        pub fn sdrk_liquid_firfilt_crcf_create(
            h: *const f32,
            h_len: c_uint,
            method: c_int,
        ) -> *mut c_void;
        pub fn sdrk_liquid_firfilt_crcf_reset(q: *mut c_void);
        pub fn sdrk_liquid_firfilt_crcf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_firfilt_crcf_execute_block(
            q: *mut c_void,
            x: *mut [f32; 2],
            n: c_uint,
            y: *mut [f32; 2],
        );
    }
}

/// The block length as liquid's `unsigned int`.
fn count(n: usize) -> c_uint {
    c_uint::try_from(n).expect("block too long")
}

/// liquid's `firfilt_crcf`: `y[n] = sum h[k] x[n - k]`, zero history at the start.
pub struct LiquidFirfiltCrcf {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
}

// SAFETY: the C object owns its buffers and is used by one thread at a time.
unsafe impl Send for LiquidFirfiltCrcf {}

impl LiquidFirfiltCrcf {
    /// The filter with the taps `h`; `simd` selects the SIMD dot product of the level,
    /// otherwise the portable one. `None` without the library or without taps.
    #[must_use]
    pub fn new(h: &[f32], simd: bool) -> Option<Self> {
        if h.is_empty() {
            return None;
        }
        #[cfg(has_liquidhb)]
        {
            // SAFETY: `h` holds `h.len()` values, which the call copies.
            let q = unsafe {
                ffi::sdrk_liquid_firfilt_crcf_create(h.as_ptr(), count(h.len()), c_int::from(simd))
            };
            // Not `then_some`: it would build (and drop, so destroy) a null object.
            if q.is_null() { None } else { Some(Self { q }) }
        }
        #[cfg(not(has_liquidhb))]
        {
            let _ = simd;
            None
        }
    }

    /// Clears the delay line.
    pub fn reset(&mut self) {
        #[cfg(has_liquidhb)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_firfilt_crcf_reset(self.q);
        }
    }

    /// `x.len()` outputs from `x`.
    ///
    /// # Panics
    /// If `y` is not as long as `x`.
    pub fn execute(&mut self, x: &mut [[f32; 2]], y: &mut [[f32; 2]]) {
        assert_eq!(x.len(), y.len(), "liquid firfilt: lengths");
        let n = count(x.len());
        #[cfg(has_liquidhb)]
        // SAFETY: both buffers hold `n` samples (liquid only reads `x`).
        unsafe {
            ffi::sdrk_liquid_firfilt_crcf_execute_block(self.q, x.as_mut_ptr(), n, y.as_mut_ptr());
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidFirfiltCrcf {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_firfilt_crcf_destroy(self.q) }
    }
}

/// liquid's `firinterp_crcf`: `y[k m + i] = sum_j h[i + j m] x[k - j]` (the taps padded with
/// zeros to a multiple of `m`), zero history at the start.
pub struct LiquidFirinterpCrcf {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
    m: usize,
}

// SAFETY: the C object owns its buffers and is used by one thread at a time.
unsafe impl Send for LiquidFirinterpCrcf {}

impl LiquidFirinterpCrcf {
    /// The interpolator by `m` (at least 2) with the prototype `h`; `simd` as for
    /// [`LiquidFirfiltCrcf::new`]. `None` without the library, without taps or for a factor
    /// below 2.
    #[must_use]
    pub fn new(h: &[f32], m: usize, simd: bool) -> Option<Self> {
        if h.len() < m || m < 2 {
            return None;
        }
        #[cfg(has_liquidhb)]
        {
            // SAFETY: `h` holds `h.len()` values, which the call copies.
            let q = unsafe {
                ffi::sdrk_liquid_firinterp_crcf_create(
                    h.as_ptr(),
                    count(h.len()),
                    count(m),
                    c_int::from(simd),
                )
            };
            if q.is_null() {
                None
            } else {
                Some(Self { q, m })
            }
        }
        #[cfg(not(has_liquidhb))]
        {
            let _ = simd;
            None
        }
    }

    /// Clears the delay line.
    pub fn reset(&mut self) {
        #[cfg(has_liquidhb)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_firinterp_crcf_reset(self.q);
        }
    }

    /// `x.len() * m` outputs from `x`.
    ///
    /// # Panics
    /// If `y` is not `x.len() * m` long.
    pub fn execute(&mut self, x: &mut [[f32; 2]], y: &mut [[f32; 2]]) {
        assert_eq!(y.len(), x.len() * self.m, "liquid firinterp: lengths");
        let n = count(x.len());
        #[cfg(has_liquidhb)]
        // SAFETY: `x` holds `n` samples (liquid only reads them) and `y` holds `n m`.
        unsafe {
            ffi::sdrk_liquid_firinterp_crcf_execute_block(
                self.q,
                x.as_mut_ptr(),
                n,
                y.as_mut_ptr(),
            );
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidFirinterpCrcf {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_firinterp_crcf_destroy(self.q) }
    }
}
