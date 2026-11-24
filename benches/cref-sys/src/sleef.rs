//! SLEEF 3.9.0's vectorized single-precision functions (BSL-1.0, `benches/refs/sleef-ops.c`,
//! cfg `has_sleef`): `log10f`, `atan2f` and `sincosf` with the 1.0-ULP bound (`_u10`) of the
//! vector extension of this binary's level (`sse2`, `sse4`, `avx2`, `avx512f`, `advsimd`), one
//! build of SLEEF per level (`libsleefops-<level>.a`, x86-64 and aarch64). The SIMD references of
//! `power_db`, of the arc tangent of `quad_demod` at `sse2` and `sse4.2`, and of `oscillate`
//! `C32` (D2903).
//!
//! Every function writes nothing without the library; the callers check [`AVAILABLE`].

/// Whether the library is linked.
pub const AVAILABLE: bool = cfg!(has_sleef);

#[cfg(has_sleef)]
mod ffi {
    use core::ffi::c_char;

    unsafe extern "C" {
        pub fn sdrk_sleef_level() -> *const c_char;
        pub fn sdrk_sleef_power_db(input: *const f32, output: *mut f32, n: usize, offset: f32);
        pub fn sdrk_sleef_atan2_32fc(c: *const f32, output: *mut f32, n: usize, gain: f32);
        pub fn sdrk_sleef_oscillate_32fc(
            output: *mut f32,
            n: usize,
            phase: u32,
            increment: u32,
        ) -> u32;
    }
}

/// The SLEEF vector extension of this build (`sse2`, `sse4`, `avx2`, `avx512f`, `advsimd`).
#[must_use]
pub fn level() -> Option<&'static str> {
    #[cfg(has_sleef)]
    {
        // SAFETY: the wrapper returns a pointer to a NUL-terminated string literal.
        let s = unsafe { core::ffi::CStr::from_ptr(ffi::sdrk_sleef_level()) };
        s.to_str().ok()
    }
    #[cfg(not(has_sleef))]
    None
}

/// `output[i] = 10 Sleef_log10f(input[i]) + offset`.
///
/// # Panics
/// If the lengths differ.
pub fn power_db(input: &[f32], output: &mut [f32], offset: f32) {
    assert_eq!(input.len(), output.len(), "SLEEF power_db: length mismatch");
    #[cfg(has_sleef)]
    // SAFETY: both buffers hold `len` floats; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_sleef_power_db(input.as_ptr(), output.as_mut_ptr(), input.len(), offset);
    }
    #[cfg(not(has_sleef))]
    let _ = offset;
}

/// `output[i] = Sleef_atan2f(c[2 i + 1], c[2 i]) * gain` of interleaved complex floats.
///
/// # Panics
/// If `c` is not twice as long as `output`.
pub fn atan2_32fc(c: &[f32], output: &mut [f32], gain: f32) {
    assert_eq!(
        c.len(),
        2 * output.len(),
        "SLEEF atan2_32fc: length mismatch"
    );
    #[cfg(has_sleef)]
    // SAFETY: `c` holds `2 n` floats, the output `n`; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_sleef_atan2_32fc(c.as_ptr(), output.as_mut_ptr(), output.len(), gain);
    }
    #[cfg(not(has_sleef))]
    let _ = gain;
}

/// `output = (cos a, sin a)` of `a = (phase + i increment) as i32 * 2 pi / 2^32` with
/// `Sleef_sincosf`, interleaved; returns the phase after the last sample (`phase` without the
/// library).
///
/// # Panics
/// If the length is odd.
pub fn oscillate_32fc(output: &mut [f32], phase: u32, increment: u32) -> u32 {
    assert!(output.len() % 2 == 0, "SLEEF oscillate_32fc: odd length");
    #[cfg(has_sleef)]
    // SAFETY: the output holds `2 n` floats and is borrowed exclusively.
    unsafe {
        ffi::sdrk_sleef_oscillate_32fc(output.as_mut_ptr(), output.len() / 2, phase, increment)
    }
    #[cfg(not(has_sleef))]
    {
        let _ = increment;
        phase
    }
}
