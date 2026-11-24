//! KFR's element-wise expressions (`benches/refs/kfr-ops.cpp`): the int8 conversions, the
//! complex product and `10 log10(x) + offset`, compiled for the SIMD level of this binary
//! (`libkfrops-<level>.a`, x86-64 only, see `build.rs`). The references at `sse2` where VOLK's
//! machine of the level runs a `generic` protokernel (D2785).

/// Whether the library is linked.
pub const AVAILABLE: bool = cfg!(has_kfrops);

#[cfg(has_kfrops)]
mod ffi {
    unsafe extern "C" {
        pub fn sdrk_kfr_i8_to_f32(input: *const i8, output: *mut f32, n: usize, scale: f32);
        pub fn sdrk_kfr_i8_to_i16(input: *const i8, output: *mut i16, n: usize);
        pub fn sdrk_kfr_complex_mul(a: *const f32, b: *const f32, output: *mut f32, n: usize);
        pub fn sdrk_kfr_power_db(input: *const f32, output: *mut f32, n: usize, offset: f32);
    }
}

/// `output[i] = input[i] * scale`; without the library nothing is written.
///
/// # Panics
/// If the lengths differ.
pub fn i8_to_f32(input: &[i8], output: &mut [f32], scale: f32) {
    assert_eq!(input.len(), output.len(), "KFR i8_to_f32: length mismatch");
    #[cfg(has_kfrops)]
    // SAFETY: both buffers hold `len` elements; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_i8_to_f32(input.as_ptr(), output.as_mut_ptr(), input.len(), scale);
    }
    #[cfg(not(has_kfrops))]
    let _ = scale;
}

/// `output[i] = input[i] << 8`; without the library nothing is written.
///
/// # Panics
/// If the lengths differ.
pub fn i8_to_i16(input: &[i8], output: &mut [i16]) {
    assert_eq!(input.len(), output.len(), "KFR i8_to_i16: length mismatch");
    #[cfg(has_kfrops)]
    // SAFETY: both buffers hold `len` elements; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_i8_to_i16(input.as_ptr(), output.as_mut_ptr(), input.len());
    }
}

/// The complex product of interleaved `a` and `b` (`[re, im]` pairs) into `output`; without the
/// library nothing is written.
///
/// # Panics
/// If the lengths differ or are odd.
pub fn complex_mul(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert!(
        a.len() == b.len() && a.len() == output.len() && a.len() % 2 == 0,
        "KFR complex_mul: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: the three buffers hold `len` floats (`len / 2` complex values); the output is
    // borrowed exclusively, so it overlaps neither input.
    unsafe {
        ffi::sdrk_kfr_complex_mul(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), a.len() / 2);
    }
}

/// `output[i] = 10 log10(input[i]) + offset`; without the library nothing is written.
///
/// # Panics
/// If the lengths differ.
pub fn power_db(input: &[f32], output: &mut [f32], offset: f32) {
    assert_eq!(input.len(), output.len(), "KFR power_db: length mismatch");
    #[cfg(has_kfrops)]
    // SAFETY: both buffers hold `len` floats; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_power_db(input.as_ptr(), output.as_mut_ptr(), input.len(), offset);
    }
    #[cfg(not(has_kfrops))]
    let _ = offset;
}
