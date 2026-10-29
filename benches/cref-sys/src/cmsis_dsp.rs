//! The scalar references of the fallback round from CMSIS-DSP 1.18.0 (Apache-2.0): generic C
//! (`__GNUC_PYTHON__`, `ARM_MATH_LOOPUNROLL`, `-O3 -ffp-contract=off`, the instruction-set flags
//! of the SIMD level, as the biquad references of D16) in `libcmsisdsp-<level>.a`, built by
//! `benches/refs/build.sh` from the sources and the handles of `benches/refs/cmsis-dsp.c`.
//!
//! [`AVAILABLE`] says whether the library is linked (cfg `has_cmsisdsp`). Without it every
//! function of this module panics and every constructor returns `None`, so a caller checks
//! [`AVAILABLE`] first. The raw declarations are in [`ffi`] (only with the library) for callers
//! that want to call CMSIS-DSP without the length checks of the wrappers.
//!
//! Formats: `q7` is `i8`, `q15` is `i16`, `q31` is `i32`, `q63` is `i64`; complex buffers are
//! interleaved (`[re, im]` pairs). The Q15 results of CMSIS-DSP are not all Q15: the
//! magnitude is Q2.14, the squared magnitude and the complex product Q3.13, the dot product a
//! Q34.30 `i64`, the complex dot product Q8.24 per component, the arc tangent Q2.13 radians.

/// Whether `libcmsisdsp-<level>.a` is linked.
pub const AVAILABLE: bool = cfg!(has_cmsisdsp);

/// The raw declarations of the CMSIS-DSP functions and of the handles of `cmsis-dsp.c`. Without
/// the library every function is a stub that panics (see [`AVAILABLE`]).
#[allow(missing_docs, clippy::missing_safety_doc)]
pub mod ffi {
    use core::ffi::c_void;

    /// The `extern "C"` block with the library, panicking stubs of the same signatures without.
    macro_rules! declare {
        ($(fn $name:ident($($a:ident: $t:ty),* $(,)?) $(-> $r:ty)?;)*) => {
            #[cfg(has_cmsisdsp)]
            unsafe extern "C" {
                $(pub fn $name($($a: $t),*) $(-> $r)?;)*
            }
            $(
                #[cfg(not(has_cmsisdsp))]
                pub unsafe fn $name($(_: $t),*) $(-> $r)? {
                    panic!("CMSIS-DSP is not available (see cmsis_dsp::AVAILABLE)")
                }
            )*
        };
    }

    declare! {
        fn arm_q7_to_float(src: *const i8, dst: *mut f32, n: u32);
        fn arm_q15_to_float(src: *const i16, dst: *mut f32, n: u32);
        fn arm_q7_to_q15(src: *const i8, dst: *mut i16, n: u32);
        fn arm_float_to_q15(src: *const f32, dst: *mut i16, n: u32);
        fn arm_float_to_q7(src: *const f32, dst: *mut i8, n: u32);
        fn arm_q15_to_q7(src: *const i16, dst: *mut i8, n: u32);

        fn arm_dot_prod_f32(a: *const f32, b: *const f32, n: u32, result: *mut f32);
        fn arm_dot_prod_q15(a: *const i16, b: *const i16, n: u32, result: *mut i64);
        fn arm_cmplx_dot_prod_f32(
            a: *const f32,
            b: *const f32,
            n: u32,
            re: *mut f32,
            im: *mut f32,
        );
        fn arm_cmplx_dot_prod_q15(
            a: *const i16,
            b: *const i16,
            n: u32,
            re: *mut i32,
            im: *mut i32,
        );

        fn arm_scale_f32(src: *const f32, scale: f32, dst: *mut f32, n: u32);
        fn arm_scale_q15(src: *const i16, scale_fract: i16, shift: i8, dst: *mut i16, n: u32);
        fn arm_mult_f32(a: *const f32, b: *const f32, dst: *mut f32, n: u32);
        fn arm_mult_q15(a: *const i16, b: *const i16, dst: *mut i16, n: u32);
        fn arm_cmplx_mult_real_f32(cmplx: *const f32, real: *const f32, dst: *mut f32, n: u32);
        fn arm_cmplx_mult_real_q15(cmplx: *const i16, real: *const i16, dst: *mut i16, n: u32);
        fn arm_cmplx_mult_cmplx_f32(a: *const f32, b: *const f32, dst: *mut f32, n: u32);
        fn arm_cmplx_mult_cmplx_q15(a: *const i16, b: *const i16, dst: *mut i16, n: u32);
        fn arm_cmplx_conj_f32(src: *const f32, dst: *mut f32, n: u32);

        fn arm_cmplx_mag_squared_f32(src: *const f32, dst: *mut f32, n: u32);
        fn arm_cmplx_mag_squared_q15(src: *const i16, dst: *mut i16, n: u32);
        fn arm_cmplx_mag_f32(src: *const f32, dst: *mut f32, n: u32);
        fn arm_cmplx_mag_q15(src: *const i16, dst: *mut i16, n: u32);
        fn arm_vlog_f32(src: *const f32, dst: *mut f32, n: u32);

        fn arm_sin_cos_f32(theta: f32, sin: *mut f32, cos: *mut f32);
        fn arm_sin_cos_q31(theta: i32, sin: *mut i32, cos: *mut i32);
        fn arm_sin_f32(x: f32) -> f32;
        fn arm_cos_f32(x: f32) -> f32;
        fn arm_sin_q15(x: i16) -> i16;
        fn arm_cos_q15(x: i16) -> i16;
        fn arm_atan2_f32(y: f32, x: f32, result: *mut f32) -> i8;
        fn arm_atan2_q15(y: i16, x: i16, result: *mut i16) -> i8;

        fn arm_cfft_f32(instance: *const c_void, data: *mut f32, ifft: u8, bit_reverse: u8);
        fn arm_cfft_q15(instance: *const c_void, data: *mut i16, ifft: u8, bit_reverse: u8);
        fn arm_rfft_fast_f32(
            instance: *const c_void,
            input: *mut f32,
            output: *mut f32,
            ifft: u8,
        );

        fn sdrk_cmsis_fir_f32_new(taps: *const f32, n_taps: u32, max_block: u32)
        -> *mut c_void;
        fn sdrk_cmsis_fir_f32_run(h: *mut c_void, input: *const f32, output: *mut f32, n: u32);
        fn sdrk_cmsis_fir_q15_new(taps: *const i16, n_taps: u32, max_block: u32)
        -> *mut c_void;
        fn sdrk_cmsis_fir_q15_run(h: *mut c_void, input: *const i16, output: *mut i16, n: u32);
        fn sdrk_cmsis_fir_decimate_f32_new(
            taps: *const f32,
            n_taps: u32,
            factor: u32,
            max_block: u32,
        ) -> *mut c_void;
        fn sdrk_cmsis_fir_decimate_f32_run(
            h: *mut c_void,
            input: *const f32,
            output: *mut f32,
            n: u32,
        );
        fn sdrk_cmsis_fir_decimate_q15_new(
            taps: *const i16,
            n_taps: u32,
            factor: u32,
            max_block: u32,
        ) -> *mut c_void;
        fn sdrk_cmsis_fir_decimate_q15_run(
            h: *mut c_void,
            input: *const i16,
            output: *mut i16,
            n: u32,
        );
        fn sdrk_cmsis_fir_interpolate_f32_new(
            taps: *const f32,
            n_taps: u32,
            factor: u32,
            max_block: u32,
        ) -> *mut c_void;
        fn sdrk_cmsis_fir_interpolate_f32_run(
            h: *mut c_void,
            input: *const f32,
            output: *mut f32,
            n: u32,
        );
        fn sdrk_cmsis_fir_interpolate_q15_new(
            taps: *const i16,
            n_taps: u32,
            factor: u32,
            max_block: u32,
        ) -> *mut c_void;
        fn sdrk_cmsis_fir_interpolate_q15_run(
            h: *mut c_void,
            input: *const i16,
            output: *mut i16,
            n: u32,
        );
        fn sdrk_cmsis_dsp_free(h: *mut c_void);

        fn sdrk_cmsis_cfft_f32_instance(len: u32) -> *const c_void;
        fn sdrk_cmsis_cfft_q15_instance(len: u32) -> *const c_void;
        fn sdrk_cmsis_rfft_fast_f32_new(len: u32) -> *mut c_void;

        fn sdrk_cmsis_sin_cos_f32_n(theta: *const f32, s: *mut f32, c: *mut f32, n: u32);
        fn sdrk_cmsis_sin_cos_q31_n(theta: *const i32, s: *mut i32, c: *mut i32, n: u32);
        fn sdrk_cmsis_sin_f32_n(x: *const f32, y: *mut f32, n: u32);
        fn sdrk_cmsis_cos_f32_n(x: *const f32, y: *mut f32, n: u32);
        fn sdrk_cmsis_sin_q15_n(x: *const i16, y: *mut i16, n: u32);
        fn sdrk_cmsis_cos_q15_n(x: *const i16, y: *mut i16, n: u32);
        fn sdrk_cmsis_atan2_f32_n(iq: *const [f32; 2], out: *mut f32, n: u32);
        fn sdrk_cmsis_atan2_q15_n(iq: *const [i16; 2], out: *mut i16, n: u32);
    }
}

/// The block length as CMSIS-DSP's `uint32_t`.
fn block(n: usize) -> u32 {
    u32::try_from(n).expect("CMSIS-DSP: block too long")
}

/// Calls `$f` of [`ffi`] (which panics without the library).
macro_rules! call {
    ($f:ident($($a:expr),* $(,)?)) => {
        // SAFETY: the wrapper checked the lengths of the buffers against the counts it passes;
        // the CMSIS-DSP function reads and writes exactly those elements.
        unsafe {
            ffi::$f($($a),*)
        }
    };
}

/// Panics unless the two lengths are equal.
fn same(a: usize, b: usize, what: &str) {
    assert_eq!(a, b, "CMSIS-DSP {what}: length mismatch");
}

// ---- Conversions (SupportFunctions) ----

/// `arm_q7_to_float`: `dst[i] = src[i] / 128`.
pub fn q7_to_float(src: &[i8], dst: &mut [f32]) {
    same(src.len(), dst.len(), "q7_to_float");
    call!(arm_q7_to_float(
        src.as_ptr(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_q15_to_float`: `dst[i] = src[i] / 32768`.
pub fn q15_to_float(src: &[i16], dst: &mut [f32]) {
    same(src.len(), dst.len(), "q15_to_float");
    call!(arm_q15_to_float(
        src.as_ptr(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_q7_to_q15`: `dst[i] = src[i] << 8`.
pub fn q7_to_q15(src: &[i8], dst: &mut [i16]) {
    same(src.len(), dst.len(), "q7_to_q15");
    call!(arm_q7_to_q15(
        src.as_ptr(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_float_to_q15`: `dst[i] = sat16(src[i] * 32768)`, rounded half away from zero (the
/// generic C adds +-0.5 before the truncating conversion).
pub fn float_to_q15(src: &[f32], dst: &mut [i16]) {
    same(src.len(), dst.len(), "float_to_q15");
    call!(arm_float_to_q15(
        src.as_ptr(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_float_to_q7`: `dst[i] = sat8(src[i] * 128)`, rounded as [`float_to_q15`].
pub fn float_to_q7(src: &[f32], dst: &mut [i8]) {
    same(src.len(), dst.len(), "float_to_q7");
    call!(arm_float_to_q7(
        src.as_ptr(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_q15_to_q7`: `dst[i] = src[i] >> 8` (truncating).
pub fn q15_to_q7(src: &[i16], dst: &mut [i8]) {
    same(src.len(), dst.len(), "q15_to_q7");
    call!(arm_q15_to_q7(
        src.as_ptr(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

// ---- Dot products ----

/// `arm_dot_prod_f32`: `sum a[i] * b[i]` in f32 (four partial sums with loop unrolling).
#[must_use]
pub fn dot_prod_f32(a: &[f32], b: &[f32]) -> f32 {
    same(a.len(), b.len(), "dot_prod_f32");
    let mut result = 0.0f32;
    call!(arm_dot_prod_f32(
        a.as_ptr(),
        b.as_ptr(),
        block(a.len()),
        &raw mut result
    ));
    result
}

/// `arm_dot_prod_q15`: `sum a[i] * b[i]` in an `i64` (Q34.30, no rounding or saturation).
#[must_use]
pub fn dot_prod_q15(a: &[i16], b: &[i16]) -> i64 {
    same(a.len(), b.len(), "dot_prod_q15");
    let mut result = 0i64;
    call!(arm_dot_prod_q15(
        a.as_ptr(),
        b.as_ptr(),
        block(a.len()),
        &raw mut result
    ));
    result
}

/// `arm_cmplx_dot_prod_f32` of interleaved complex vectors: `sum a[i] * b[i]` (no conjugate),
/// returned as `[re, im]`.
#[must_use]
pub fn cmplx_dot_prod_f32(a: &[[f32; 2]], b: &[[f32; 2]]) -> [f32; 2] {
    same(a.len(), b.len(), "cmplx_dot_prod_f32");
    let (mut re, mut im) = (0.0f32, 0.0f32);
    call!(arm_cmplx_dot_prod_f32(
        a.as_ptr().cast(),
        b.as_ptr().cast(),
        block(a.len()),
        &raw mut re,
        &raw mut im,
    ));
    [re, im]
}

/// `arm_cmplx_dot_prod_q15`: `sum a[i] * b[i]` with `i64` accumulators, returned as Q8.24
/// `[re, im]` (the accumulators shifted right by 6).
#[must_use]
pub fn cmplx_dot_prod_q15(a: &[[i16; 2]], b: &[[i16; 2]]) -> [i32; 2] {
    same(a.len(), b.len(), "cmplx_dot_prod_q15");
    let (mut re, mut im) = (0i32, 0i32);
    call!(arm_cmplx_dot_prod_q15(
        a.as_ptr().cast(),
        b.as_ptr().cast(),
        block(a.len()),
        &raw mut re,
        &raw mut im,
    ));
    [re, im]
}

// ---- Scaling and products ----

/// `arm_scale_f32`: `dst[i] = src[i] * scale`. `src` and `dst` may be the same buffer through
/// [`ffi`]; here they are distinct.
pub fn scale_f32(src: &[f32], scale: f32, dst: &mut [f32]) {
    same(src.len(), dst.len(), "scale_f32");
    call!(arm_scale_f32(
        src.as_ptr(),
        scale,
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_scale_q15`: `dst[i] = sat16((src[i] * scale_fract) >> (15 - shift))` (truncating).
pub fn scale_q15(src: &[i16], scale_fract: i16, shift: i8, dst: &mut [i16]) {
    same(src.len(), dst.len(), "scale_q15");
    call!(arm_scale_q15(
        src.as_ptr(),
        scale_fract,
        shift,
        dst.as_mut_ptr(),
        block(src.len()),
    ));
}

/// `arm_mult_f32`: `dst[i] = a[i] * b[i]`.
pub fn mult_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    same(a.len(), b.len(), "mult_f32");
    same(a.len(), dst.len(), "mult_f32");
    call!(arm_mult_f32(
        a.as_ptr(),
        b.as_ptr(),
        dst.as_mut_ptr(),
        block(a.len())
    ));
}

/// `arm_mult_q15`: `dst[i] = sat16((a[i] * b[i]) >> 15)` (truncating).
pub fn mult_q15(a: &[i16], b: &[i16], dst: &mut [i16]) {
    same(a.len(), b.len(), "mult_q15");
    same(a.len(), dst.len(), "mult_q15");
    call!(arm_mult_q15(
        a.as_ptr(),
        b.as_ptr(),
        dst.as_mut_ptr(),
        block(a.len())
    ));
}

/// `arm_cmplx_mult_real_f32`: `dst[i] = cmplx[i] * real[i]`.
pub fn cmplx_mult_real_f32(cmplx: &[[f32; 2]], real: &[f32], dst: &mut [[f32; 2]]) {
    same(cmplx.len(), real.len(), "cmplx_mult_real_f32");
    same(cmplx.len(), dst.len(), "cmplx_mult_real_f32");
    call!(arm_cmplx_mult_real_f32(
        cmplx.as_ptr().cast(),
        real.as_ptr(),
        dst.as_mut_ptr().cast(),
        block(cmplx.len()),
    ));
}

/// `arm_cmplx_mult_real_q15`: `dst[i] = sat16((cmplx[i] * real[i]) >> 15)` per component.
pub fn cmplx_mult_real_q15(cmplx: &[[i16; 2]], real: &[i16], dst: &mut [[i16; 2]]) {
    same(cmplx.len(), real.len(), "cmplx_mult_real_q15");
    same(cmplx.len(), dst.len(), "cmplx_mult_real_q15");
    call!(arm_cmplx_mult_real_q15(
        cmplx.as_ptr().cast(),
        real.as_ptr(),
        dst.as_mut_ptr().cast(),
        block(cmplx.len()),
    ));
}

/// `arm_cmplx_mult_cmplx_f32`: `dst[i] = a[i] * b[i]` (complex product).
pub fn cmplx_mult_cmplx_f32(a: &[[f32; 2]], b: &[[f32; 2]], dst: &mut [[f32; 2]]) {
    same(a.len(), b.len(), "cmplx_mult_cmplx_f32");
    same(a.len(), dst.len(), "cmplx_mult_cmplx_f32");
    call!(arm_cmplx_mult_cmplx_f32(
        a.as_ptr().cast(),
        b.as_ptr().cast(),
        dst.as_mut_ptr().cast(),
        block(a.len()),
    ));
}

/// `arm_cmplx_mult_cmplx_q15`: the complex product in Q3.13 (each partial product shifted
/// right by 17, truncating).
pub fn cmplx_mult_cmplx_q15(a: &[[i16; 2]], b: &[[i16; 2]], dst: &mut [[i16; 2]]) {
    same(a.len(), b.len(), "cmplx_mult_cmplx_q15");
    same(a.len(), dst.len(), "cmplx_mult_cmplx_q15");
    call!(arm_cmplx_mult_cmplx_q15(
        a.as_ptr().cast(),
        b.as_ptr().cast(),
        dst.as_mut_ptr().cast(),
        block(a.len()),
    ));
}

/// `arm_cmplx_conj_f32`: `dst[i] = conj(src[i])`.
pub fn cmplx_conj_f32(src: &[[f32; 2]], dst: &mut [[f32; 2]]) {
    same(src.len(), dst.len(), "cmplx_conj_f32");
    call!(arm_cmplx_conj_f32(
        src.as_ptr().cast(),
        dst.as_mut_ptr().cast(),
        block(src.len())
    ));
}

// ---- Magnitudes and logarithm ----

/// `arm_cmplx_mag_squared_f32`: `dst[i] = re^2 + im^2`.
pub fn cmplx_mag_squared_f32(src: &[[f32; 2]], dst: &mut [f32]) {
    same(src.len(), dst.len(), "cmplx_mag_squared_f32");
    call!(arm_cmplx_mag_squared_f32(
        src.as_ptr().cast(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_cmplx_mag_squared_q15`: `re^2 + im^2` in Q3.13 (`(re*re >> 17) + (im*im >> 17)`).
pub fn cmplx_mag_squared_q15(src: &[[i16; 2]], dst: &mut [i16]) {
    same(src.len(), dst.len(), "cmplx_mag_squared_q15");
    call!(arm_cmplx_mag_squared_q15(
        src.as_ptr().cast(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_cmplx_mag_f32`: `dst[i] = sqrtf(re^2 + im^2)`.
pub fn cmplx_mag_f32(src: &[[f32; 2]], dst: &mut [f32]) {
    same(src.len(), dst.len(), "cmplx_mag_f32");
    call!(arm_cmplx_mag_f32(
        src.as_ptr().cast(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_cmplx_mag_q15`: the magnitude in Q2.14 (`arm_sqrt_q31` of the halved Q31 sum).
pub fn cmplx_mag_q15(src: &[[i16; 2]], dst: &mut [i16]) {
    same(src.len(), dst.len(), "cmplx_mag_q15");
    call!(arm_cmplx_mag_q15(
        src.as_ptr().cast(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

/// `arm_vlog_f32`: the natural logarithm, a polynomial approximation in generic C.
pub fn vlog_f32(src: &[f32], dst: &mut [f32]) {
    same(src.len(), dst.len(), "vlog_f32");
    call!(arm_vlog_f32(
        src.as_ptr(),
        dst.as_mut_ptr(),
        block(src.len())
    ));
}

// ---- Sine, cosine, arc tangent (loops of cmsis-dsp.c over the scalar functions) ----

/// `arm_sin_cos_f32` per element: `theta` in degrees, table with linear interpolation.
pub fn sin_cos_f32(theta: &[f32], sin: &mut [f32], cos: &mut [f32]) {
    same(theta.len(), sin.len(), "sin_cos_f32");
    same(theta.len(), cos.len(), "sin_cos_f32");
    call!(sdrk_cmsis_sin_cos_f32_n(
        theta.as_ptr(),
        sin.as_mut_ptr(),
        cos.as_mut_ptr(),
        block(theta.len()),
    ));
}

/// `arm_sin_cos_q31` per element: `theta` in [-1, 1) for [-180, 180) degrees, Q31 results.
pub fn sin_cos_q31(theta: &[i32], sin: &mut [i32], cos: &mut [i32]) {
    same(theta.len(), sin.len(), "sin_cos_q31");
    same(theta.len(), cos.len(), "sin_cos_q31");
    call!(sdrk_cmsis_sin_cos_q31_n(
        theta.as_ptr(),
        sin.as_mut_ptr(),
        cos.as_mut_ptr(),
        block(theta.len()),
    ));
}

/// `arm_sin_f32` per element (radians).
pub fn sin_f32(x: &[f32], y: &mut [f32]) {
    same(x.len(), y.len(), "sin_f32");
    call!(sdrk_cmsis_sin_f32_n(
        x.as_ptr(),
        y.as_mut_ptr(),
        block(x.len())
    ));
}

/// `arm_cos_f32` per element (radians).
pub fn cos_f32(x: &[f32], y: &mut [f32]) {
    same(x.len(), y.len(), "cos_f32");
    call!(sdrk_cmsis_cos_f32_n(
        x.as_ptr(),
        y.as_mut_ptr(),
        block(x.len())
    ));
}

/// `arm_sin_q15` per element: `x` in [0, 1) (Q15, negative values wrap) for [0, 2 pi).
pub fn sin_q15(x: &[i16], y: &mut [i16]) {
    same(x.len(), y.len(), "sin_q15");
    call!(sdrk_cmsis_sin_q15_n(
        x.as_ptr(),
        y.as_mut_ptr(),
        block(x.len())
    ));
}

/// `arm_cos_q15` per element, as [`sin_q15`].
pub fn cos_q15(x: &[i16], y: &mut [i16]) {
    same(x.len(), y.len(), "cos_q15");
    call!(sdrk_cmsis_cos_q15_n(
        x.as_ptr(),
        y.as_mut_ptr(),
        block(x.len())
    ));
}

/// `arm_atan2_f32(im, re)` of every complex sample, in radians.
pub fn atan2_f32(iq: &[[f32; 2]], out: &mut [f32]) {
    same(iq.len(), out.len(), "atan2_f32");
    call!(sdrk_cmsis_atan2_f32_n(
        iq.as_ptr(),
        out.as_mut_ptr(),
        block(iq.len())
    ));
}

/// `arm_atan2_q15(im, re)` of every complex Q15 sample, in Q2.13 radians.
pub fn atan2_q15(iq: &[[i16; 2]], out: &mut [i16]) {
    same(iq.len(), out.len(), "atan2_q15");
    call!(sdrk_cmsis_atan2_q15_n(
        iq.as_ptr(),
        out.as_mut_ptr(),
        block(iq.len())
    ));
}

// ---- FIR filters ----

macro_rules! fir_handle {
    (
        $(#[$m:meta])*
        $name:ident, $t:ty, $new:ident, $run:ident, ratio = $ratio:ident
    ) => {
        $(#[$m])*
        pub struct $name {
            #[cfg(has_cmsisdsp)]
            handle: *mut core::ffi::c_void,
            max_block: usize,
            factor: usize,
        }

        impl $name {
            /// Runs the filter over `input` and writes the outputs to `output`; consecutive
            /// calls continue the stream.
            ///
            /// # Panics
            /// If `input` is longer than the `max_block` of the constructor, or `output` does
            /// not have the length of the outputs of `input` (see the type).
            pub fn run(&mut self, input: &[$t], output: &mut [$t]) {
                assert!(input.len() <= self.max_block, "CMSIS-DSP FIR: block too long");
                fir_handle!(@check $ratio, self, input, output);
                let n = block(input.len());
                #[cfg(has_cmsisdsp)]
                // SAFETY: the handle is valid until `drop`; `input` holds `n <= max_block`
                // samples and `output` the outputs they produce (checked above); the borrows
                // cannot overlap.
                unsafe {
                    ffi::$run(self.handle, input.as_ptr(), output.as_mut_ptr(), n);
                }
                #[cfg(not(has_cmsisdsp))]
                let _ = (n, output);
            }
        }

        #[cfg(has_cmsisdsp)]
        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: the handle came from the constructor and is freed once.
                unsafe { ffi::sdrk_cmsis_dsp_free(self.handle) }
            }
        }
    };
    (@check same, $s:ident, $i:ident, $o:ident) => {
        let _ = $s.factor;
        assert_eq!($i.len(), $o.len(), "CMSIS-DSP FIR: length mismatch");
    };
    (@check decimate, $s:ident, $i:ident, $o:ident) => {
        assert!($i.len() % $s.factor == 0, "CMSIS-DSP FIR decimator: block not a multiple of the factor");
        assert_eq!($i.len() / $s.factor, $o.len(), "CMSIS-DSP FIR decimator: output length");
    };
    (@check interpolate, $s:ident, $i:ident, $o:ident) => {
        assert_eq!($i.len() * $s.factor, $o.len(), "CMSIS-DSP FIR interpolator: output length");
    };
}

fir_handle! {
    /// `arm_fir_f32`: `y[n] = sum h[k] x[n - k]` in f32 (CMSIS-DSP's unrolled order of the
    /// sums: eight outputs at a time).
    CmsisFirF32, f32, sdrk_cmsis_fir_f32_new, sdrk_cmsis_fir_f32_run, ratio = same
}

fir_handle! {
    /// `arm_fir_q15`: a 64-bit accumulator per output, `sat16(acc >> 15)` (truncating). An odd
    /// tap count is padded with a zero tap (CMSIS-DSP needs an even count of at least four).
    CmsisFirQ15, i16, sdrk_cmsis_fir_q15_new, sdrk_cmsis_fir_q15_run, ratio = same
}

fir_handle! {
    /// `arm_fir_decimate_f32`: every `factor`-th output of the FIR filter, starting with the
    /// first, `y[m] = y_full[m * factor]` (the phase of SDRk's `fir_decimate`). A block must be a
    /// multiple of the factor.
    CmsisFirDecimateF32, f32, sdrk_cmsis_fir_decimate_f32_new, sdrk_cmsis_fir_decimate_f32_run, ratio = decimate
}

fir_handle! {
    /// `arm_fir_decimate_q15`: as [`CmsisFirDecimateF32`] with the accumulator of
    /// [`CmsisFirQ15`].
    CmsisFirDecimateQ15, i16, sdrk_cmsis_fir_decimate_q15_new, sdrk_cmsis_fir_decimate_q15_run, ratio = decimate
}

fir_handle! {
    /// `arm_fir_interpolate_f32`: the polyphase interpolator, `factor` outputs per input; the
    /// tap count must be a multiple of the factor. The gain of the taps is the caller's.
    CmsisFirInterpolateF32, f32, sdrk_cmsis_fir_interpolate_f32_new, sdrk_cmsis_fir_interpolate_f32_run, ratio = interpolate
}

fir_handle! {
    /// `arm_fir_interpolate_q15`: as [`CmsisFirInterpolateF32`] with the accumulator of
    /// [`CmsisFirQ15`].
    CmsisFirInterpolateQ15, i16, sdrk_cmsis_fir_interpolate_q15_new, sdrk_cmsis_fir_interpolate_q15_run, ratio = interpolate
}

/// The tap count and block limit as `u32`, `None` when out of range.
fn fir_args(taps: usize, max_block: usize) -> Option<(u32, u32)> {
    Some((u32::try_from(taps).ok()?, u32::try_from(max_block).ok()?))
}

impl CmsisFirF32 {
    /// The filter of `taps` in SDRk's order (the handle reverses them for CMSIS-DSP) for blocks
    /// of at most `max_block` samples; `None` without the library or for invalid arguments.
    #[must_use]
    pub fn new(taps: &[f32], max_block: usize) -> Option<Self> {
        let (n, m) = fir_args(taps.len(), max_block)?;
        #[cfg(has_cmsisdsp)]
        {
            // SAFETY: `taps` holds `n` values; the handle copies them.
            let handle = unsafe { ffi::sdrk_cmsis_fir_f32_new(taps.as_ptr(), n, m) };
            (!handle.is_null()).then_some(Self {
                handle,
                max_block,
                factor: 1,
            })
        }
        #[cfg(not(has_cmsisdsp))]
        {
            let _ = (n, m);
            None
        }
    }
}

impl CmsisFirQ15 {
    /// As [`CmsisFirF32::new`] with Q15 taps.
    #[must_use]
    pub fn new(taps: &[i16], max_block: usize) -> Option<Self> {
        let (n, m) = fir_args(taps.len(), max_block)?;
        #[cfg(has_cmsisdsp)]
        {
            // SAFETY: `taps` holds `n` values; the handle copies them.
            let handle = unsafe { ffi::sdrk_cmsis_fir_q15_new(taps.as_ptr(), n, m) };
            (!handle.is_null()).then_some(Self {
                handle,
                max_block,
                factor: 1,
            })
        }
        #[cfg(not(has_cmsisdsp))]
        {
            let _ = (n, m);
            None
        }
    }
}

macro_rules! fir_ratio_new {
    ($name:ident, $t:ty, $new:ident) => {
        impl $name {
            /// The filter of `taps` (SDRk's order) with the integer `factor` (1 to 255) for
            /// blocks of at most `max_block` input samples; `None` without the library or when
            /// CMSIS-DSP rejects the arguments.
            #[must_use]
            pub fn new(taps: &[$t], factor: usize, max_block: usize) -> Option<Self> {
                let (n, m) = fir_args(taps.len(), max_block)?;
                let l = u32::try_from(factor).ok()?;
                #[cfg(has_cmsisdsp)]
                {
                    // SAFETY: `taps` holds `n` values; the handle copies them.
                    let handle = unsafe { ffi::$new(taps.as_ptr(), n, l, m) };
                    (!handle.is_null()).then_some(Self {
                        handle,
                        max_block,
                        factor,
                    })
                }
                #[cfg(not(has_cmsisdsp))]
                {
                    let _ = (n, m, l);
                    None
                }
            }
        }
    };
}

fir_ratio_new!(CmsisFirDecimateF32, f32, sdrk_cmsis_fir_decimate_f32_new);
fir_ratio_new!(CmsisFirDecimateQ15, i16, sdrk_cmsis_fir_decimate_q15_new);
fir_ratio_new!(
    CmsisFirInterpolateF32,
    f32,
    sdrk_cmsis_fir_interpolate_f32_new
);
fir_ratio_new!(
    CmsisFirInterpolateQ15,
    i16,
    sdrk_cmsis_fir_interpolate_q15_new
);

// ---- Transforms ----

/// `arm_cfft_f32` with the constant instance of `arm_const_structs.c` (64, 1024 or 4096
/// points): the unnormalised forward DFT in place (radix-8 by 2 or by 4 stages and the bit
/// reversal).
pub struct CmsisCfftF32 {
    #[cfg(has_cmsisdsp)]
    instance: *const core::ffi::c_void,
    len: usize,
}

impl CmsisCfftF32 {
    /// The instance of `len` points; `None` without the library or for another length.
    #[must_use]
    pub fn new(len: usize) -> Option<Self> {
        let n = u32::try_from(len).ok()?;
        #[cfg(has_cmsisdsp)]
        {
            // SAFETY: returns a pointer to a constant structure or null.
            let instance = unsafe { ffi::sdrk_cmsis_cfft_f32_instance(n) };
            (!instance.is_null()).then_some(Self { instance, len })
        }
        #[cfg(not(has_cmsisdsp))]
        {
            let _ = n;
            None
        }
    }

    /// The forward transform of `data` in place, bit reversal on.
    ///
    /// # Panics
    /// If `data` does not have the length of the instance.
    pub fn forward(&self, data: &mut [[f32; 2]]) {
        assert_eq!(data.len(), self.len, "CMSIS-DSP cfft: length differs");
        #[cfg(has_cmsisdsp)]
        // SAFETY: the instance is a constant structure for `len` points; `data` holds them.
        unsafe {
            ffi::arm_cfft_f32(self.instance, data.as_mut_ptr().cast(), 0, 1);
        }
    }
}

/// `arm_cfft_q15` with the constant instance of `arm_const_structs.c` (64, 1024 or 4096
/// points): the forward DFT in place, scaled down by CMSIS-DSP's stages (the result is the
/// DFT divided by `N`, with truncating shifts).
pub struct CmsisCfftQ15 {
    #[cfg(has_cmsisdsp)]
    instance: *const core::ffi::c_void,
    len: usize,
}

impl CmsisCfftQ15 {
    /// The instance of `len` points; `None` without the library or for another length.
    #[must_use]
    pub fn new(len: usize) -> Option<Self> {
        let n = u32::try_from(len).ok()?;
        #[cfg(has_cmsisdsp)]
        {
            // SAFETY: returns a pointer to a constant structure or null.
            let instance = unsafe { ffi::sdrk_cmsis_cfft_q15_instance(n) };
            (!instance.is_null()).then_some(Self { instance, len })
        }
        #[cfg(not(has_cmsisdsp))]
        {
            let _ = n;
            None
        }
    }

    /// The forward transform of `data` in place, bit reversal on.
    ///
    /// # Panics
    /// If `data` does not have the length of the instance.
    pub fn forward(&self, data: &mut [[i16; 2]]) {
        assert_eq!(data.len(), self.len, "CMSIS-DSP cfft: length differs");
        #[cfg(has_cmsisdsp)]
        // SAFETY: the instance is a constant structure for `len` points; `data` holds them.
        unsafe {
            ffi::arm_cfft_q15(self.instance, data.as_mut_ptr().cast(), 0, 1);
        }
    }
}

/// `arm_rfft_fast_f32`: the real-input forward transform of `N` samples (a power of two from
/// 32 to 4096). CMSIS-DSP packs the output: `out[0] = X[0]`, `out[1] = X[N/2]` (both real),
/// then `X[1]`, ..., `X[N/2 - 1]` as interleaved pairs. The input buffer is overwritten.
pub struct CmsisRfftFastF32 {
    #[cfg(has_cmsisdsp)]
    handle: *mut core::ffi::c_void,
    len: usize,
}

impl CmsisRfftFastF32 {
    /// The instance of `len` points; `None` without the library or for an invalid length.
    #[must_use]
    pub fn new(len: usize) -> Option<Self> {
        let n = u32::try_from(len).ok()?;
        #[cfg(has_cmsisdsp)]
        {
            // SAFETY: allocates and initializes an instance, or returns null.
            let handle = unsafe { ffi::sdrk_cmsis_rfft_fast_f32_new(n) };
            (!handle.is_null()).then_some(Self { handle, len })
        }
        #[cfg(not(has_cmsisdsp))]
        {
            let _ = n;
            None
        }
    }

    /// The forward transform of `input` (overwritten) into the packed `output`.
    ///
    /// # Panics
    /// If a buffer does not have the length of the instance.
    pub fn forward(&mut self, input: &mut [f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.len, "CMSIS-DSP rfft: length differs");
        assert_eq!(output.len(), self.len, "CMSIS-DSP rfft: length differs");
        #[cfg(has_cmsisdsp)]
        // SAFETY: the instance is valid until `drop`; both buffers hold `len` values and do not
        // overlap (two exclusive borrows).
        unsafe {
            ffi::arm_rfft_fast_f32(self.handle, input.as_mut_ptr(), output.as_mut_ptr(), 0);
        }
    }
}

#[cfg(has_cmsisdsp)]
impl Drop for CmsisRfftFastF32 {
    fn drop(&mut self) {
        // SAFETY: the instance came from `sdrk_cmsis_rfft_fast_f32_new` (malloc) and is freed
        // once.
        unsafe { ffi::sdrk_cmsis_dsp_free(self.handle) }
    }
}
