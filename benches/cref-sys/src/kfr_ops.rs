//! KFR's element-wise expressions (`benches/refs/kfr-ops.cpp`), compiled for the SIMD level of
//! this binary (`libkfrops-<level>.a`, every platform but the Cortex-M4, see `build.rs`): the int8
//! conversions, the complex products, `10 log10(x) + offset`, the sums, the conversion to
//! offset-binary bytes, the 16-bit and float (de)interleaves, the magnitudes, the conjugate
//! product and its argument, the scale, the complex dot product and the stereo matrix. The
//! references where VOLK's machine of the level runs a `generic` protokernel (`sse2`, D2785) or a
//! protokernel of a lower level (`avx512`, D2900), and where no other SIMD library has the
//! operation (D2901).
//!
//! Every function writes nothing without the library; the callers check [`AVAILABLE`].

/// Whether the library is linked.
pub const AVAILABLE: bool = cfg!(has_kfrops);

#[cfg(has_kfrops)]
mod ffi {
    unsafe extern "C" {
        pub fn sdrk_kfr_i8_to_f32(input: *const i8, output: *mut f32, n: usize, scale: f32);
        pub fn sdrk_kfr_i8_to_i16(input: *const i8, output: *mut i16, n: usize);
        pub fn sdrk_kfr_complex_mul(a: *const f32, b: *const f32, output: *mut f32, n: usize);
        pub fn sdrk_kfr_power_db(input: *const f32, output: *mut f32, n: usize, offset: f32);
        pub fn sdrk_kfr_add_f32(a: *const f32, b: *const f32, output: *mut f32, n: usize);
        pub fn sdrk_kfr_satadd_i16(a: *const i16, b: *const i16, output: *mut i16, n: usize);
        pub fn sdrk_kfr_f32_to_u8(
            input: *const f32,
            output: *mut u8,
            n: usize,
            scale: f32,
            bias: f32,
        );
        pub fn sdrk_kfr_deinterleave_i16(c: *const i16, re: *mut i16, im: *mut i16, n: usize);
        pub fn sdrk_kfr_interleave_i16(re: *const i16, im: *const i16, c: *mut i16, n: usize);
        pub fn sdrk_kfr_interleave_f32(re: *const f32, im: *const f32, c: *mut f32, n: usize);
        pub fn sdrk_kfr_mag_i16(c: *const i16, output: *mut i16, n: usize);
        pub fn sdrk_kfr_mag_sq(c: *const f32, output: *mut f32, n: usize);
        pub fn sdrk_kfr_quad_demod(
            a: *const f32,
            b: *const f32,
            output: *mut f32,
            n: usize,
            gain: f32,
        );
        pub fn sdrk_kfr_multiply_conjugate(
            a: *const f32,
            b: *const f32,
            output: *mut f32,
            n: usize,
        );
        pub fn sdrk_kfr_scale_f32(input: *const f32, output: *mut f32, n: usize, scale: f32);
        pub fn sdrk_kfr_complex_mul_real(a: *const f32, b: *const f32, output: *mut f32, n: usize);
        pub fn sdrk_kfr_dot_complex(a: *const f32, b: *const f32, n: usize, out: *mut f32);
        pub fn sdrk_kfr_stereo_matrix_f32(input: *const f32, output: *mut f32, n: usize);
        pub fn sdrk_kfr_stereo_matrix_i16(input: *const i16, output: *mut i16, n: usize);
    }
}

/// `output[i] = input[i] * scale`.
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

/// `output[i] = input[i] << 8`.
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

/// The complex product of interleaved `a` and `b` (`[re, im]` pairs) into `output`.
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

/// `output[i] = 10 log10(input[i]) + offset`.
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

/// `output[i] = a[i] + b[i]` on floats (a complex sum on the interleaved floats).
///
/// # Panics
/// If the lengths differ.
pub fn add_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert!(
        a.len() == b.len() && a.len() == output.len(),
        "KFR add_f32: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: the three buffers hold `len` floats; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_add_f32(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), a.len());
    }
}

/// `output[i] = satadd(a[i], b[i])` on `i16`.
///
/// # Panics
/// If the lengths differ.
pub fn satadd_i16(a: &[i16], b: &[i16], output: &mut [i16]) {
    assert!(
        a.len() == b.len() && a.len() == output.len(),
        "KFR satadd_i16: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: the three buffers hold `len` values; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_satadd_i16(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), a.len());
    }
}

/// `output[i] = clamp(round(input[i] * scale + bias), 0, 255)` (ties to even).
///
/// # Panics
/// If the lengths differ.
pub fn f32_to_u8(input: &[f32], output: &mut [u8], scale: f32, bias: f32) {
    assert_eq!(input.len(), output.len(), "KFR f32_to_u8: length mismatch");
    #[cfg(has_kfrops)]
    // SAFETY: both buffers hold `len` elements; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_f32_to_u8(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len(),
            scale,
            bias,
        );
    }
    #[cfg(not(has_kfrops))]
    let _ = (scale, bias);
}

/// `re[i] = c[2 i]`, `im[i] = c[2 i + 1]` on `i16`.
///
/// # Panics
/// If `c` is not twice as long as `re` and `im`.
pub fn deinterleave_i16(c: &[i16], re: &mut [i16], im: &mut [i16]) {
    assert!(
        c.len() == 2 * re.len() && re.len() == im.len(),
        "KFR deinterleave_i16: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: `c` holds `2 n` values, `re` and `im` `n` each; the outputs are borrowed
    // exclusively.
    unsafe {
        ffi::sdrk_kfr_deinterleave_i16(c.as_ptr(), re.as_mut_ptr(), im.as_mut_ptr(), re.len());
    }
}

/// `c[2 i] = re[i]`, `c[2 i + 1] = im[i]` on `i16`.
///
/// # Panics
/// If `c` is not twice as long as `re` and `im`.
pub fn interleave_i16(re: &[i16], im: &[i16], c: &mut [i16]) {
    assert!(
        c.len() == 2 * re.len() && re.len() == im.len(),
        "KFR interleave_i16: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: as in `deinterleave_i16`, the roles of the buffers swapped.
    unsafe {
        ffi::sdrk_kfr_interleave_i16(re.as_ptr(), im.as_ptr(), c.as_mut_ptr(), re.len());
    }
}

/// `c[2 i] = re[i]`, `c[2 i + 1] = im[i]` on floats.
///
/// # Panics
/// If `c` is not twice as long as `re` and `im`.
pub fn interleave_f32(re: &[f32], im: &[f32], c: &mut [f32]) {
    assert!(
        c.len() == 2 * re.len() && re.len() == im.len(),
        "KFR interleave_f32: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: as in `interleave_i16`.
    unsafe {
        ffi::sdrk_kfr_interleave_f32(re.as_ptr(), im.as_ptr(), c.as_mut_ptr(), re.len());
    }
}

/// `output[i] = min(round(sqrt(re^2 + im^2)), 32767)` in `f32` of the interleaved `i16` pairs.
///
/// # Panics
/// If `c` is not twice as long as `output`.
pub fn mag_i16(c: &[i16], output: &mut [i16]) {
    assert_eq!(c.len(), 2 * output.len(), "KFR mag_i16: length mismatch");
    #[cfg(has_kfrops)]
    // SAFETY: `c` holds `2 n` values, the output `n`; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_mag_i16(c.as_ptr(), output.as_mut_ptr(), output.len());
    }
}

/// `output[i] = re^2 + im^2` of the interleaved pairs (`cabssqr`).
///
/// # Panics
/// If `c` is not twice as long as `output`.
pub fn mag_sq(c: &[f32], output: &mut [f32]) {
    assert_eq!(c.len(), 2 * output.len(), "KFR mag_sq: length mismatch");
    #[cfg(has_kfrops)]
    // SAFETY: as in `mag_i16`.
    unsafe {
        ffi::sdrk_kfr_mag_sq(c.as_ptr(), output.as_mut_ptr(), output.len());
    }
}

/// `output[i] = carg(a[i] cconj(b[i])) * gain` of interleaved complex floats.
///
/// # Panics
/// If `a` or `b` is not twice as long as `output`.
pub fn quad_demod(a: &[f32], b: &[f32], output: &mut [f32], gain: f32) {
    assert!(
        a.len() == 2 * output.len() && b.len() == a.len(),
        "KFR quad_demod: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: `a` and `b` hold `2 n` floats, the output `n`; the output is borrowed
    // exclusively.
    unsafe {
        ffi::sdrk_kfr_quad_demod(
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            output.len(),
            gain,
        );
    }
    #[cfg(not(has_kfrops))]
    let _ = gain;
}

/// `output = a cconj(b)` of interleaved complex floats.
///
/// # Panics
/// If the lengths differ or are odd.
pub fn multiply_conjugate(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert!(
        a.len() == b.len() && a.len() == output.len() && a.len() % 2 == 0,
        "KFR multiply_conjugate: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: as in `complex_mul`.
    unsafe {
        ffi::sdrk_kfr_multiply_conjugate(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), a.len() / 2);
    }
}

/// `output[i] = input[i] * scale`.
///
/// # Panics
/// If the lengths differ.
pub fn scale_f32(input: &[f32], output: &mut [f32], scale: f32) {
    assert_eq!(input.len(), output.len(), "KFR scale_f32: length mismatch");
    #[cfg(has_kfrops)]
    // SAFETY: both buffers hold `len` floats; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_scale_f32(input.as_ptr(), output.as_mut_ptr(), input.len(), scale);
    }
    #[cfg(not(has_kfrops))]
    let _ = scale;
}

/// The interleaved complex `a` times the real `b` into `output`.
///
/// # Panics
/// If `a` and `output` are not twice as long as `b`.
pub fn complex_mul_real(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert!(
        a.len() == 2 * b.len() && output.len() == a.len(),
        "KFR complex_mul_real: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: `a` and the output hold `2 n` floats, `b` `n`; the output is borrowed
    // exclusively.
    unsafe {
        ffi::sdrk_kfr_complex_mul_real(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), b.len());
    }
}

/// `sum a[i] b[i]` of interleaved complex floats (no conjugate), `[re, im]`; zero without the
/// library.
///
/// # Panics
/// If the lengths differ or are odd.
#[must_use]
pub fn dot_complex(a: &[f32], b: &[f32]) -> [f32; 2] {
    assert!(
        a.len() == b.len() && a.len() % 2 == 0,
        "KFR dot_complex: length mismatch"
    );
    #[cfg_attr(not(has_kfrops), allow(unused_mut))]
    let mut out = [0.0f32; 2];
    #[cfg(has_kfrops)]
    // SAFETY: both inputs hold `len` floats; the result is two floats.
    unsafe {
        ffi::sdrk_kfr_dot_complex(a.as_ptr(), b.as_ptr(), a.len() / 2, out.as_mut_ptr());
    }
    out
}

/// `(l, r) = (s + 2 d, s - 2 d)` of the interleaved pairs `(s, d)`, on floats.
///
/// # Panics
/// If the lengths differ or are odd.
pub fn stereo_matrix_f32(input: &[f32], output: &mut [f32]) {
    assert!(
        input.len() == output.len() && input.len() % 2 == 0,
        "KFR stereo_matrix_f32: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: both buffers hold `len` floats; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_stereo_matrix_f32(input.as_ptr(), output.as_mut_ptr(), input.len() / 2);
    }
}

/// `(l, r) = (satadd(s, d2), satsub(s, d2))`, `d2 = satadd(d, d)`, of the interleaved `i16`
/// pairs `(s, d)`: two saturations where the exact `s +- 2 d` saturates once.
///
/// # Panics
/// If the lengths differ or are odd.
pub fn stereo_matrix_i16(input: &[i16], output: &mut [i16]) {
    assert!(
        input.len() == output.len() && input.len() % 2 == 0,
        "KFR stereo_matrix_i16: length mismatch"
    );
    #[cfg(has_kfrops)]
    // SAFETY: both buffers hold `len` values; the output is borrowed exclusively.
    unsafe {
        ffi::sdrk_kfr_stereo_matrix_i16(input.as_ptr(), output.as_mut_ptr(), input.len() / 2);
    }
}
