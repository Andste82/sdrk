//! The reference of the channelizer `channelize` (M45): liquid-dsp 1.8.3's analysis filter bank
//! `firpfbch2_crcf` (MIT, cfg `has_liquidhb`, `benches/refs/liquid-channelizer.c`), the
//! polyphase channelizer with the output rate `2 fs / M`, in two builds of the halfband library
//! of the level:
//!
//! - [`Firpfbch2Build::Fftw`]: the SIMD dot products of the level and FFTW 3.3.10 for the
//!   `M`-point transforms (liquid's default build where FFTW is found); not on wasm32, which has
//!   no FFTW build;
//! - [`Firpfbch2Build::Portable`]: liquid's portable dot products and its own FFT (portable C).
//!
//! Both take the prototype of the caller (`2 M m` taps, SDRk's, so that the accuracy of both
//! sides is comparable).

#[cfg(has_liquidhb)]
use core::ffi::c_int;
use core::ffi::{c_uint, c_void};

#[cfg(has_liquidhb)]
mod ffi {
    use core::ffi::{c_int, c_uint, c_void};

    macro_rules! firpfbch2_ffi {
        ($create:ident, $reset:ident, $destroy:ident, $block:ident) => {
            unsafe extern "C" {
                pub fn $create(
                    h: *const f32,
                    m_channels: c_uint,
                    m: c_uint,
                    method: c_int,
                ) -> *mut c_void;
                pub fn $reset(q: *mut c_void);
                pub fn $destroy(q: *mut c_void);
                pub fn $block(q: *mut c_void, x: *mut [f32; 2], n: c_uint, y: *mut [f32; 2]);
            }
        };
    }

    firpfbch2_ffi!(
        sdrk_liquid_firpfbch2_port_create,
        sdrk_liquid_firpfbch2_port_reset,
        sdrk_liquid_firpfbch2_port_destroy,
        sdrk_liquid_firpfbch2_port_execute_block
    );
    #[cfg(not(target_arch = "wasm32"))]
    firpfbch2_ffi!(
        sdrk_liquid_firpfbch2_fftw_create,
        sdrk_liquid_firpfbch2_fftw_reset,
        sdrk_liquid_firpfbch2_fftw_destroy,
        sdrk_liquid_firpfbch2_fftw_execute_block
    );
}

/// The build of [`LiquidFirpfbch2`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Firpfbch2Build {
    /// The SIMD dot products of the level and FFTW (not on wasm32).
    Fftw,
    /// The portable dot products and liquid's own FFT.
    Portable,
}

/// liquid's `firpfbch2_crcf` analyzer (`firpfbch2_crcf_execute` once per hop): `M / 2` complex
/// samples in, `M` channel samples out, real taps. The output is liquid's: the inverse
/// transform of the branch sums scaled by `1 / M`, so channel `k` of it is SDRk's channel
/// `M - k` (mod `M`) up to the scale and the phase convention of the branches.
pub struct LiquidFirpfbch2 {
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    q: *mut c_void,
    #[cfg_attr(not(has_liquidhb), allow(dead_code))]
    build: Firpfbch2Build,
    channels: usize,
}

impl LiquidFirpfbch2 {
    /// The analyzer of `channels` channels with the prototype `h` of `2 channels m` taps.
    /// `None` without the library, for the FFTW build on wasm32, or for a prototype that is not
    /// a whole number of `2 channels` taps.
    #[must_use]
    pub fn new(h: &[f32], channels: usize, build: Firpfbch2Build) -> Option<Self> {
        if channels < 2
            || !channels.is_multiple_of(2)
            || h.is_empty()
            || !h.len().is_multiple_of(2 * channels)
        {
            return None;
        }
        let m = c_uint::try_from(h.len() / (2 * channels)).ok()?;
        let mc = c_uint::try_from(channels).ok()?;
        #[cfg(has_liquidhb)]
        {
            // SAFETY: `h` holds `2 channels m` values, which the call copies.
            let q = unsafe {
                match build {
                    Firpfbch2Build::Portable => ffi::sdrk_liquid_firpfbch2_port_create(
                        h.as_ptr(),
                        mc,
                        m,
                        c_int::from(false),
                    ),
                    #[cfg(not(target_arch = "wasm32"))]
                    Firpfbch2Build::Fftw => {
                        ffi::sdrk_liquid_firpfbch2_fftw_create(h.as_ptr(), mc, m, c_int::from(true))
                    }
                    #[cfg(target_arch = "wasm32")]
                    Firpfbch2Build::Fftw => core::ptr::null_mut(),
                }
            };
            if q.is_null() {
                None
            } else {
                Some(Self { q, build, channels })
            }
        }
        #[cfg(not(has_liquidhb))]
        {
            let _ = (build, m, mc);
            None
        }
    }

    /// Clears the windows and the parity flag.
    pub fn reset(&mut self) {
        #[cfg(has_liquidhb)]
        // SAFETY: the object is valid until `drop`.
        unsafe {
            match self.build {
                Firpfbch2Build::Portable => ffi::sdrk_liquid_firpfbch2_port_reset(self.q),
                #[cfg(not(target_arch = "wasm32"))]
                Firpfbch2Build::Fftw => ffi::sdrk_liquid_firpfbch2_fftw_reset(self.q),
                #[cfg(target_arch = "wasm32")]
                Firpfbch2Build::Fftw => {}
            }
        }
    }

    /// `x.len() / (M / 2)` hops: `M` channel samples per hop to `y`.
    ///
    /// # Panics
    /// If `x` is not a whole number of hops or `y` is not `2 x.len()` long.
    pub fn execute(&mut self, x: &mut [[f32; 2]], y: &mut [[f32; 2]]) {
        let hop = self.channels / 2;
        assert!(
            x.len() % hop == 0 && y.len() == 2 * x.len(),
            "liquid firpfbch2: lengths"
        );
        let n = c_uint::try_from(x.len() / hop).expect("block too long");
        #[cfg(has_liquidhb)]
        // SAFETY: `x` holds `n M / 2` samples (liquid only reads them) and `y` holds `n M`.
        unsafe {
            match self.build {
                Firpfbch2Build::Portable => ffi::sdrk_liquid_firpfbch2_port_execute_block(
                    self.q,
                    x.as_mut_ptr(),
                    n,
                    y.as_mut_ptr(),
                ),
                #[cfg(not(target_arch = "wasm32"))]
                Firpfbch2Build::Fftw => ffi::sdrk_liquid_firpfbch2_fftw_execute_block(
                    self.q,
                    x.as_mut_ptr(),
                    n,
                    y.as_mut_ptr(),
                ),
                #[cfg(target_arch = "wasm32")]
                Firpfbch2Build::Fftw => {}
            }
        }
        #[cfg(not(has_liquidhb))]
        let _ = n;
    }
}

#[cfg(has_liquidhb)]
impl Drop for LiquidFirpfbch2 {
    fn drop(&mut self) {
        // SAFETY: the object came from the constructor and is destroyed once.
        unsafe {
            match self.build {
                Firpfbch2Build::Portable => ffi::sdrk_liquid_firpfbch2_port_destroy(self.q),
                #[cfg(not(target_arch = "wasm32"))]
                Firpfbch2Build::Fftw => ffi::sdrk_liquid_firpfbch2_fftw_destroy(self.q),
                #[cfg(target_arch = "wasm32")]
                Firpfbch2Build::Fftw => {}
            }
        }
    }
}
