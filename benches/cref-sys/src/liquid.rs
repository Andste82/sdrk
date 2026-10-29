//! The DC blocker of liquid-dsp 1.8.3 (MIT): `iirfilt_rrrf_create_dc_blocker` and
//! `iirfilt_crcf_create_dc_blocker` with `_execute_block` and liquid's portable dot products
//! (`benches/refs/liquid-dcblock.c` and `liquid-dotprod.c`, cfg `has_liquiddc`), the scalar
//! reference of `dc_block_real` and `dc_block`.
//!
//! liquid's filter is `H(z) = g (1 - z^-1) / (1 - (1 - alpha) z^-1)` in direct form II with the
//! gain `g = sqrt(1 - alpha)`: the pole and zero of SDRk's `y = x - x1 + r y1` with
//! `alpha = 1 - r`, a different structure (one state, both sums through dot-product calls) and
//! a gain. [`LiquidDcBlocker::set_scale`] (liquid's public `iirfilt_*_set_scale`) with 1.0 gives
//! SDRk's gain.

use core::ffi::c_void;

#[cfg(has_liquiddc)]
mod ffi {
    use core::ffi::{c_int, c_uint, c_void};

    unsafe extern "C" {
        pub fn sdrk_liquid_iirfilt_rrrf_create_dc_blocker(alpha: f32) -> *mut c_void;
        pub fn sdrk_liquid_iirfilt_rrrf_set_scale(q: *mut c_void, scale: f32) -> c_int;
        pub fn sdrk_liquid_iirfilt_rrrf_execute_block(
            q: *mut c_void,
            x: *const f32,
            n: c_uint,
            y: *mut f32,
        ) -> c_int;
        pub fn sdrk_liquid_iirfilt_rrrf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_iirfilt_crcf_create_dc_blocker(alpha: f32) -> *mut c_void;
        pub fn sdrk_liquid_iirfilt_crcf_set_scale(q: *mut c_void, scale: f32) -> c_int;
        pub fn sdrk_liquid_iirfilt_crcf_execute_block(
            q: *mut c_void,
            x: *const [f32; 2],
            n: c_uint,
            y: *mut [f32; 2],
        ) -> c_int;
        pub fn sdrk_liquid_iirfilt_crcf_destroy(q: *mut c_void);
    }
}

/// liquid's real DC blocker (`iirfilt_rrrf`).
pub struct LiquidDcBlocker {
    #[cfg_attr(not(has_liquiddc), allow(dead_code))]
    q: *mut c_void,
}

/// liquid's complex DC blocker (`iirfilt_crcf`: complex samples, real coefficients).
pub struct LiquidDcBlockerCrcf {
    #[cfg_attr(not(has_liquiddc), allow(dead_code))]
    q: *mut c_void,
}

/// The block length as liquid's `unsigned int`.
fn block(n: usize) -> core::ffi::c_uint {
    core::ffi::c_uint::try_from(n).expect("liquid iirfilt: block too long")
}

impl LiquidDcBlocker {
    /// `iirfilt_rrrf_create_dc_blocker(alpha)` (`alpha = 1 - r`, greater than 0); `None`
    /// without the library or for an invalid `alpha`.
    #[must_use]
    pub fn new(alpha: f32) -> Option<Self> {
        #[cfg(has_liquiddc)]
        {
            // SAFETY: creates a filter object or returns null.
            let q = unsafe { ffi::sdrk_liquid_iirfilt_rrrf_create_dc_blocker(alpha) };
            // Not `then_some`: it would build (and drop, so destroy) a null object.
            if q.is_null() { None } else { Some(Self { q }) }
        }
        #[cfg(not(has_liquiddc))]
        {
            let _ = alpha;
            None
        }
    }

    /// `iirfilt_rrrf_set_scale`: the output gain (`sqrt(1 - alpha)` after `new`).
    pub fn set_scale(&mut self, scale: f32) {
        #[cfg(has_liquiddc)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_iirfilt_rrrf_set_scale(self.q, scale);
        }
        #[cfg(not(has_liquiddc))]
        let _ = scale;
    }

    /// `iirfilt_rrrf_execute_block` from `input` into `output`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn execute_block(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "liquid iirfilt: length mismatch");
        let n = block(input.len());
        #[cfg(has_liquiddc)]
        // SAFETY: both buffers hold `n` values and do not overlap; liquid only reads `x`.
        unsafe {
            ffi::sdrk_liquid_iirfilt_rrrf_execute_block(
                self.q,
                input.as_ptr(),
                n,
                output.as_mut_ptr(),
            );
        }
        #[cfg(not(has_liquiddc))]
        let _ = n;
    }
}

impl LiquidDcBlockerCrcf {
    /// `iirfilt_crcf_create_dc_blocker(alpha)`; `None` without the library or for an invalid
    /// `alpha`.
    #[must_use]
    pub fn new(alpha: f32) -> Option<Self> {
        #[cfg(has_liquiddc)]
        {
            // SAFETY: creates a filter object or returns null.
            let q = unsafe { ffi::sdrk_liquid_iirfilt_crcf_create_dc_blocker(alpha) };
            // Not `then_some`: it would build (and drop, so destroy) a null object.
            if q.is_null() { None } else { Some(Self { q }) }
        }
        #[cfg(not(has_liquiddc))]
        {
            let _ = alpha;
            None
        }
    }

    /// `iirfilt_crcf_set_scale`: the output gain (`sqrt(1 - alpha)` after `new`).
    pub fn set_scale(&mut self, scale: f32) {
        #[cfg(has_liquiddc)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_iirfilt_crcf_set_scale(self.q, scale);
        }
        #[cfg(not(has_liquiddc))]
        let _ = scale;
    }

    /// `iirfilt_crcf_execute_block` from `input` into `output`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn execute_block(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) {
        assert_eq!(input.len(), output.len(), "liquid iirfilt: length mismatch");
        let n = block(input.len());
        #[cfg(has_liquiddc)]
        // SAFETY: both buffers hold `n` pairs and do not overlap; liquid only reads `x`.
        unsafe {
            ffi::sdrk_liquid_iirfilt_crcf_execute_block(
                self.q,
                input.as_ptr(),
                n,
                output.as_mut_ptr(),
            );
        }
        #[cfg(not(has_liquiddc))]
        let _ = n;
    }
}

#[cfg(has_liquiddc)]
impl Drop for LiquidDcBlocker {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_iirfilt_rrrf_destroy(self.q) }
    }
}

#[cfg(has_liquiddc)]
impl Drop for LiquidDcBlockerCrcf {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_iirfilt_crcf_destroy(self.q) }
    }
}
