//! The Gaussian generators of liquid-dsp 1.8.3 (MIT): `randnf` and `crandnf` over libc `rand()`
//! (`benches/refs/liquid-randn.c`, cfg `has_liquidrandn`), the information reference of
//! `gauss_noise` (M48).
//!
//! liquid evaluates Box-Muller in single precision (`sqrtf`, `logf`, the sine branch with `sinf`
//! for `randnf` and `cexpf` for `crandnf`) on uniform values of `rand() / RAND_MAX`, with a
//! rejection loop for `u1 == 0`: no random access, and a sequence that depends on the libc of the
//! platform. SDRk's `gauss_noise` runs PCG32 with one draw per real value and the inverse normal
//! CDF of a 1024-entry table in integers, bit-exact on every target and seekable, so liquid does
//! different work (an information row, not a gate).

#[cfg(has_liquidrandn)]
mod ffi {
    use core::ffi::c_uint;

    unsafe extern "C" {
        pub fn sdrk_liquid_randn_seed(seed: c_uint);
        pub fn sdrk_liquid_randnf_block(y: *mut f32, n: c_uint, sigma: f32);
        pub fn sdrk_liquid_crandnf_block(y: *mut [f32; 2], n: c_uint, sigma: f32);
    }
}

/// liquid's `randnf` and `crandnf`, each value scaled by `sigma` (the standard deviation per real
/// component). libc's generator is global: the wrappers are for the measurement binary only.
pub struct LiquidRandn;

impl LiquidRandn {
    /// Whether the library is linked.
    pub const AVAILABLE: bool = cfg!(has_liquidrandn);

    /// Seeds libc's generator (`srand`); nothing without the library.
    pub fn seed(seed: u32) {
        #[cfg(has_liquidrandn)]
        // SAFETY: `srand` only sets the state of libc's generator.
        unsafe {
            ffi::sdrk_liquid_randn_seed(seed);
        }
        #[cfg(not(has_liquidrandn))]
        let _ = seed;
    }

    /// Fills `output` with `randnf() * sigma`; nothing without the library.
    ///
    /// # Panics
    /// If the block is longer than `c_uint::MAX`.
    pub fn randnf_block(sigma: f32, output: &mut [f32]) {
        let n = core::ffi::c_uint::try_from(output.len()).expect("liquid randnf: block too long");
        #[cfg(has_liquidrandn)]
        // SAFETY: the buffer holds `n` values and is only written.
        unsafe {
            ffi::sdrk_liquid_randnf_block(output.as_mut_ptr(), n, sigma);
        }
        #[cfg(not(has_liquidrandn))]
        let _ = (n, sigma);
    }

    /// Fills the complex `output` with `crandnf() * sigma`; nothing without the library.
    ///
    /// # Panics
    /// If the block is longer than `c_uint::MAX`.
    pub fn crandnf_block(sigma: f32, output: &mut [[f32; 2]]) {
        let n = core::ffi::c_uint::try_from(output.len()).expect("liquid crandnf: block too long");
        #[cfg(has_liquidrandn)]
        // SAFETY: the buffer holds `n` pairs (the layout of `float complex`) and is only written.
        unsafe {
            ffi::sdrk_liquid_crandnf_block(output.as_mut_ptr(), n, sigma);
        }
        #[cfg(not(has_liquidrandn))]
        let _ = (n, sigma);
    }
}
