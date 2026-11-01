//! The interpolated-table NCO of liquid-dsp 1.8.3 (MIT): `nco_crcf` of type `LIQUID_VCO_INTERP`
//! with the pilot loop of `stereo_pll` around it (`benches/refs/liquid-nco.c`, cfg
//! `has_liquidnco`), the scalar reference of the `f32` `stereo_pll` fallback.
//!
//! liquid's table has 1024 entries (a value and a slope each), its phase is a 32-bit word, and
//! `nco_crcf_adjust_frequency` / `nco_crcf_adjust_phase` reduce a phase in radians modulo two
//! pi with two `while` loops before they convert it to that word.

use core::ffi::c_void;

#[cfg(has_liquidnco)]
mod ffi {
    use core::ffi::{c_uint, c_void};

    unsafe extern "C" {
        pub fn sdrk_liquid_nco_crcf_create_vco_interp() -> *mut c_void;
        pub fn sdrk_liquid_nco_crcf_set_frequency(q: *mut c_void, dtheta: f32);
        pub fn sdrk_liquid_nco_crcf_set_phase(q: *mut c_void, phi: f32);
        pub fn sdrk_liquid_nco_crcf_get_phase_word(q: *const c_void) -> u32;
        pub fn sdrk_liquid_nco_crcf_get_frequency_word(q: *const c_void) -> u32;
        pub fn sdrk_liquid_nco_crcf_table(q: *const c_void) -> *const f32;
        pub fn sdrk_liquid_nco_crcf_sincos(q: *mut c_void, s: *mut f32, c: *mut f32);
        pub fn sdrk_liquid_nco_crcf_destroy(q: *mut c_void);
        pub fn sdrk_liquid_nco_pll_block(
            q: *mut c_void,
            mpx: *const f32,
            pilot: *const f32,
            out: *mut [f32; 2],
            n: c_uint,
            kp: f32,
            ki: f32,
        );
    }
}

/// Number of entries of liquid's sine table (`NCO_STATIC_LUT_SIZE`).
pub const LIQUID_NCO_TABLE_LEN: usize = 1024;

/// liquid's `nco_crcf` (`LIQUID_VCO_INTERP`) with the `stereo_pll` loop.
pub struct LiquidNcoPll {
    #[cfg_attr(not(has_liquidnco), allow(dead_code))]
    q: *mut c_void,
}

impl LiquidNcoPll {
    /// `nco_crcf_create(LIQUID_VCO_INTERP)` followed by `set_frequency(2 pi pilot_hz / rate_hz)`;
    /// `None` without the library.
    #[must_use]
    pub fn new(pilot_hz: f64, rate_hz: f64) -> Option<Self> {
        #[cfg(has_liquidnco)]
        {
            // SAFETY: creates an oscillator object or returns null.
            let q = unsafe { ffi::sdrk_liquid_nco_crcf_create_vco_interp() };
            if q.is_null() {
                return None;
            }
            // liquid's examples pass the frequency as a float in radians per sample.
            let w = (2.0 * core::f64::consts::PI * pilot_hz / rate_hz) as f32;
            // SAFETY: the object is valid.
            unsafe { ffi::sdrk_liquid_nco_crcf_set_frequency(q, w) };
            Some(Self { q })
        }
        #[cfg(not(has_liquidnco))]
        {
            let _ = (pilot_hz, rate_hz);
            None
        }
    }

    /// `nco_crcf_set_phase` (radians).
    pub fn set_phase(&mut self, phi: f32) {
        #[cfg(has_liquidnco)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_nco_crcf_set_phase(self.q, phi);
        }
        #[cfg(not(has_liquidnco))]
        let _ = phi;
    }

    /// The 32-bit phase word `theta`.
    #[must_use]
    pub fn phase_word(&self) -> u32 {
        #[cfg(has_liquidnco)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_nco_crcf_get_phase_word(self.q)
        }
        #[cfg(not(has_liquidnco))]
        0
    }

    /// The 32-bit frequency word `d_theta`.
    #[must_use]
    pub fn frequency_word(&self) -> u32 {
        #[cfg(has_liquidnco)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            ffi::sdrk_liquid_nco_crcf_get_frequency_word(self.q)
        }
        #[cfg(not(has_liquidnco))]
        0
    }

    /// liquid's table as (value, slope) pairs.
    #[must_use]
    pub fn table(&self) -> Vec<[f32; 2]> {
        #[cfg(has_liquidnco)]
        {
            // SAFETY: the table holds LIQUID_NCO_TABLE_LEN pairs of floats until `drop`.
            let t = unsafe {
                core::slice::from_raw_parts(
                    ffi::sdrk_liquid_nco_crcf_table(self.q),
                    2 * LIQUID_NCO_TABLE_LEN,
                )
            };
            t.as_chunks::<2>().0.to_vec()
        }
        #[cfg(not(has_liquidnco))]
        Vec::new()
    }

    /// `nco_crcf_sincos` of the current phase: `(sin, cos)`.
    #[must_use]
    pub fn sincos(&mut self) -> (f32, f32) {
        #[cfg(has_liquidnco)]
        {
            let (mut s, mut c) = (0.0, 0.0);
            // SAFETY: the object is valid and both outputs are live floats.
            unsafe { ffi::sdrk_liquid_nco_crcf_sincos(self.q, &raw mut s, &raw mut c) };
            (s, c)
        }
        #[cfg(not(has_liquidnco))]
        (0.0, 1.0)
    }

    /// The loop over `mpx.len()` samples: per sample `(mpx, mpx * sin(2 theta))` into `out`,
    /// the detector `pilot * cos(theta)` through `adjust_frequency(ki e)` and `adjust_phase(kp e)`
    /// (`kp` and `ki` in radians per unit of the detector output), then `step`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn pll_block(
        &mut self,
        mpx: &[f32],
        pilot: &[f32],
        out: &mut [[f32; 2]],
        kp: f32,
        ki: f32,
    ) {
        assert_eq!(mpx.len(), pilot.len(), "liquid nco: length mismatch");
        assert_eq!(mpx.len(), out.len(), "liquid nco: length mismatch");
        let n = core::ffi::c_uint::try_from(mpx.len()).expect("liquid nco: block too long");
        #[cfg(has_liquidnco)]
        // SAFETY: the three buffers hold `n` elements and do not overlap.
        unsafe {
            ffi::sdrk_liquid_nco_pll_block(
                self.q,
                mpx.as_ptr(),
                pilot.as_ptr(),
                out.as_mut_ptr(),
                n,
                kp,
                ki,
            );
        }
        #[cfg(not(has_liquidnco))]
        let _ = (n, kp, ki);
    }
}

impl Drop for LiquidNcoPll {
    fn drop(&mut self) {
        #[cfg(has_liquidnco)]
        // SAFETY: the object was created by `new` and is destroyed once.
        unsafe {
            ffi::sdrk_liquid_nco_crcf_destroy(self.q);
        }
    }
}
