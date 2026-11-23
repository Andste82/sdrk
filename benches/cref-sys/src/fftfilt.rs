//! The reference of the fast convolution `xcorr_fft` (M50 §5): liquid-dsp 1.8.3's FFT-based FIR
//! filter `fftfilt_cccf` (MIT, cfg `has_liquidhb`, `benches/refs/liquid-fftfilt.c`), overlap-add
//! with a transform of `2 n` points per block of `n` samples, complex taps, in two builds of the
//! halfband library of the level:
//!
//! - [`FftfiltBuild::Fftw`]: FFTW 3.3.10 for both transforms (liquid's default build where FFTW
//!   is found); not on wasm32, which has no FFTW build;
//! - [`FftfiltBuild::Portable`]: liquid's own FFT (portable C).
//!
//! Both take the taps of the caller (SDRk's, so that the accuracy of both sides is comparable).
//! The wrapper of `liquid-fftfilt.c` takes any number of samples per call and carries a partial
//! block to the next call.

use core::ffi::{c_uint, c_void};

#[cfg(has_liquidhb)]
mod ffi {
    use core::ffi::{c_uint, c_void};

    macro_rules! fftfilt_ffi {
        ($create:ident, $reset:ident, $destroy:ident, $execute:ident) => {
            unsafe extern "C" {
                pub fn $create(h: *const [f32; 2], h_len: c_uint, n: c_uint) -> *mut c_void;
                pub fn $reset(q: *mut c_void);
                pub fn $destroy(q: *mut c_void);
                pub fn $execute(
                    q: *mut c_void,
                    x: *mut [f32; 2],
                    len: c_uint,
                    y: *mut [f32; 2],
                ) -> c_uint;
            }
        };
    }

    fftfilt_ffi!(
        sdrk_liquid_fftfilt_port_create,
        sdrk_liquid_fftfilt_port_reset,
        sdrk_liquid_fftfilt_port_destroy,
        sdrk_liquid_fftfilt_port_execute
    );
    #[cfg(not(target_arch = "wasm32"))]
    fftfilt_ffi!(
        sdrk_liquid_fftfilt_fftw_create,
        sdrk_liquid_fftfilt_fftw_reset,
        sdrk_liquid_fftfilt_fftw_destroy,
        sdrk_liquid_fftfilt_fftw_execute
    );
}

/// The build of [`LiquidFftfilt`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftfiltBuild {
    /// FFTW (not on wasm32).
    Fftw,
    /// liquid's own FFT.
    Portable,
}

/// liquid's `fftfilt_cccf`: blocks of `n` complex samples, the convolution with the taps (liquid
/// scales the inverse transform by `1 / 2n`, so the output is the plain convolution).
pub struct LiquidFftfilt {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    build: FftfiltBuild,
    block: usize,
}

impl LiquidFftfilt {
    /// The filter with the complex taps `h` and liquid's block length `block` (at least
    /// `h.len() - 1`). `None` without the library, for the FFTW build on wasm32, or when liquid
    /// refuses the configuration.
    #[must_use]
    pub fn new(h: &[[f32; 2]], block: usize, build: FftfiltBuild) -> Option<Self> {
        if h.is_empty() || block + 1 < h.len() {
            return None;
        }
        let h_len = c_uint::try_from(h.len()).ok()?;
        let n = c_uint::try_from(block).ok()?;
        #[cfg(has_liquidhb)]
        {
            // SAFETY: `h` holds `h_len` complex values, which the call copies.
            let q = unsafe {
                match build {
                    FftfiltBuild::Portable => {
                        ffi::sdrk_liquid_fftfilt_port_create(h.as_ptr(), h_len, n)
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    FftfiltBuild::Fftw => {
                        ffi::sdrk_liquid_fftfilt_fftw_create(h.as_ptr(), h_len, n)
                    }
                    #[cfg(target_arch = "wasm32")]
                    FftfiltBuild::Fftw => core::ptr::null_mut(),
                }
            };
            if q.is_null() {
                None
            } else {
                Some(Self { q, build, block })
            }
        }
        #[cfg(not(has_liquidhb))]
        {
            let _ = (build, h_len, n);
            None
        }
    }

    /// liquid's block length `n`.
    #[must_use]
    pub fn block(&self) -> usize {
        self.block
    }

    /// Clears the overlap buffer and the partial block.
    pub fn reset(&mut self) {
        #[cfg(has_liquidhb)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            match self.build {
                FftfiltBuild::Portable => ffi::sdrk_liquid_fftfilt_port_reset(self.q),
                #[cfg(not(target_arch = "wasm32"))]
                FftfiltBuild::Fftw => ffi::sdrk_liquid_fftfilt_fftw_reset(self.q),
                #[cfg(target_arch = "wasm32")]
                FftfiltBuild::Fftw => {}
            }
        }
    }

    /// `x` in; the outputs of every block that `x` completes to `y` in order. Returns their
    /// number (a multiple of `n`).
    ///
    /// # Panics
    /// If `y` is shorter than `x.len() + n - 1`.
    pub fn execute(&mut self, x: &mut [[f32; 2]], y: &mut [[f32; 2]]) -> usize {
        assert!(
            y.len() + 1 >= x.len() + self.block,
            "liquid fftfilt: output too short"
        );
        let len = c_uint::try_from(x.len()).expect("call too long");
        #[cfg(has_liquidhb)]
        // SAFETY: `x` holds `len` samples (liquid only reads them) and `y` the outputs of every
        // block they complete (at most `len + n - 1`).
        let produced = unsafe {
            match self.build {
                FftfiltBuild::Portable => ffi::sdrk_liquid_fftfilt_port_execute(
                    self.q,
                    x.as_mut_ptr(),
                    len,
                    y.as_mut_ptr(),
                ),
                #[cfg(not(target_arch = "wasm32"))]
                FftfiltBuild::Fftw => ffi::sdrk_liquid_fftfilt_fftw_execute(
                    self.q,
                    x.as_mut_ptr(),
                    len,
                    y.as_mut_ptr(),
                ),
                #[cfg(target_arch = "wasm32")]
                FftfiltBuild::Fftw => 0,
            }
        };
        #[cfg(not(has_liquidhb))]
        let produced: c_uint = {
            let _ = len;
            0
        };
        produced as usize
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidFftfilt {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe {
            match self.build {
                FftfiltBuild::Portable => ffi::sdrk_liquid_fftfilt_port_destroy(self.q),
                #[cfg(not(target_arch = "wasm32"))]
                FftfiltBuild::Fftw => ffi::sdrk_liquid_fftfilt_fftw_destroy(self.q),
                #[cfg(target_arch = "wasm32")]
                FftfiltBuild::Fftw => {}
            }
        }
    }
}
