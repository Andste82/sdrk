//! The amplitude modulator of liquid-dsp 1.8.3 (MIT): `ampmodem` of type `LIQUID_AMPMODEM_DSB`
//! with the carrier, `ampmodem_modulate_block` (`benches/refs/liquid-ampmodem.c`, cfg
//! `has_liquidam`), the scalar reference of the `C32` `am_mod` fallback.
//!
//! liquid computes `y = x * mod_index + 1` per sample as a complex value (the imaginary part
//! `0 * mod_index`): the carrier amplitude is 1, and the input is neither clamped nor scaled by
//! a carrier, which SDRk's `am_mod` (`carrier * (1 + depth * clamp(x, -1, 1))`) does.

use core::ffi::c_void;

#[cfg(has_liquidam)]
mod ffi {
    use core::ffi::{c_int, c_uint, c_void};

    unsafe extern "C" {
        pub fn sdrk_liquid_ampmodem_create_dsb(
            mod_index: f32,
            suppressed_carrier: c_int,
        ) -> *mut c_void;
        pub fn sdrk_liquid_ampmodem_modulate_block(
            q: *mut c_void,
            m: *const f32,
            n: c_uint,
            s: *mut [f32; 2],
        ) -> c_int;
        pub fn sdrk_liquid_ampmodem_destroy(q: *mut c_void);
    }
}

/// liquid's `ampmodem` (`LIQUID_AMPMODEM_DSB`, carrier not suppressed).
pub struct LiquidAmpmodem {
    #[cfg_attr(not(has_liquidam), allow(dead_code))]
    q: *mut c_void,
}

impl LiquidAmpmodem {
    /// `ampmodem_create(mod_index, LIQUID_AMPMODEM_DSB, 0)`; `None` without the library.
    #[must_use]
    pub fn new(mod_index: f32) -> Option<Self> {
        #[cfg(has_liquidam)]
        {
            // SAFETY: creates a modulator object or returns null.
            let q = unsafe { ffi::sdrk_liquid_ampmodem_create_dsb(mod_index, 0) };
            // Not `then_some`: it would build (and drop, so destroy) a null object.
            if q.is_null() { None } else { Some(Self { q }) }
        }
        #[cfg(not(has_liquidam))]
        {
            let _ = mod_index;
            None
        }
    }

    /// `ampmodem_modulate_block` from the message `input` into the complex `output`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn modulate_block(&mut self, input: &[f32], output: &mut [[f32; 2]]) {
        assert_eq!(
            input.len(),
            output.len(),
            "liquid ampmodem: length mismatch"
        );
        let n = core::ffi::c_uint::try_from(input.len()).expect("liquid ampmodem: block too long");
        #[cfg(has_liquidam)]
        // SAFETY: both buffers hold `n` values (pairs for the output) and do not overlap;
        // liquid only reads `m`.
        unsafe {
            ffi::sdrk_liquid_ampmodem_modulate_block(
                self.q,
                input.as_ptr(),
                n,
                output.as_mut_ptr(),
            );
        }
        #[cfg(not(has_liquidam))]
        let _ = n;
    }
}

#[cfg(has_liquidam)]
impl Drop for LiquidAmpmodem {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe { ffi::sdrk_liquid_ampmodem_destroy(self.q) }
    }
}
