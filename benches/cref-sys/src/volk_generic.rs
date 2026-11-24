//! VOLK 3.3.0's portable `generic` protokernels of the interleave and deinterleave kernels and
//! of the complex product (LGPL-3.0-or-later, `benches/refs/volk-generic.c`, cfg
//! `has_volkgeneric`), compiled with `-O3 -ffp-contract=off` and the instruction set of the
//! level (`libvolkgeneric-<level>.a`): the scalar references of the `interleave` and
//! `deinterleave` fallbacks, and with a phasor lookup the reference of the `C32` `scramble`
//! fallback ([`multiply_32fc`], M49), and with a `±1.0` sequence that of the `C32` `scramble`
//! fallback at `bits = 1` ([`multiply_32fc_32f`], M49.1). VOLK has no 16-bit interleave; [`interleave_16ic`] is the
//! loop of `volk_32f_x2_interleave_32fc_generic` with 16-bit elements. The float-to-integer
//! conversions ([`convert_16i`], [`convert_8i`], [`convert_8u`]) are the references of the
//! `convert_out` fallbacks (D2905), the conjugate product and `atan2f` ([`multiply_conjugate`],
//! [`atan2`]) that of the `quad_demod` fallback (D2906).

#[cfg(has_volkgeneric)]
mod ffi {
    unsafe extern "C" {
        pub fn sdrk_volk_generic_32f_x2_interleave_32fc(
            c: *mut [f32; 2],
            re: *const f32,
            im: *const f32,
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_32fc_deinterleave_32f_x2(
            re: *mut f32,
            im: *mut f32,
            c: *const [f32; 2],
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_16i_x2_interleave_16ic(
            c: *mut [i16; 2],
            re: *const i16,
            im: *const i16,
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_16ic_deinterleave_16i_x2(
            re: *mut i16,
            im: *mut i16,
            c: *const [i16; 2],
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_32fc_x2_multiply_32fc(
            c: *mut [f32; 2],
            a: *const [f32; 2],
            b: *const [f32; 2],
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_32fc_32f_multiply_32fc(
            c: *mut [f32; 2],
            a: *const [f32; 2],
            b: *const f32,
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_32f_s32f_convert_16i(
            out: *mut i16,
            input: *const f32,
            scale: f32,
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_32f_s32f_convert_8i(
            out: *mut i8,
            input: *const f32,
            scale: f32,
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_32f_s32f_x2_convert_8u(
            out: *mut u8,
            input: *const f32,
            scale: f32,
            bias: f32,
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_32fc_x2_multiply_conjugate_32fc(
            c: *mut [f32; 2],
            a: *const [f32; 2],
            b: *const [f32; 2],
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_generic_32fc_s32f_atan2_32f(
            out: *mut f32,
            c: *const [f32; 2],
            normalize: f32,
            n: core::ffi::c_uint,
        );
    }
}

/// Whether the library is linked.
pub const AVAILABLE: bool = cfg!(has_volkgeneric);

/// The common length of three slices as the `unsigned int` of VOLK.
fn common(a: usize, b: usize, c: usize) -> core::ffi::c_uint {
    a.min(b).min(c) as core::ffi::c_uint
}

/// `volk_32f_x2_interleave_32fc_generic`: `out[i] = (re[i], im[i])` for the common length;
/// `false` without the library.
pub fn interleave_32fc(re: &[f32], im: &[f32], out: &mut [[f32; 2]]) -> bool {
    let n = common(re.len(), im.len(), out.len());
    #[cfg(has_volkgeneric)]
    // SAFETY: every buffer holds at least `n` elements; the output is a separate mutable slice.
    unsafe {
        ffi::sdrk_volk_generic_32f_x2_interleave_32fc(
            out.as_mut_ptr(),
            re.as_ptr(),
            im.as_ptr(),
            n,
        );
    }
    #[cfg(not(has_volkgeneric))]
    let _ = n;
    AVAILABLE
}

/// `volk_32fc_deinterleave_32f_x2_generic`: `re[i] = input[i][0]`, `im[i] = input[i][1]` for
/// the common length; `false` without the library.
pub fn deinterleave_32fc(input: &[[f32; 2]], re: &mut [f32], im: &mut [f32]) -> bool {
    let n = common(input.len(), re.len(), im.len());
    #[cfg(has_volkgeneric)]
    // SAFETY: as in `interleave_32fc`; the two outputs are separate mutable slices.
    unsafe {
        ffi::sdrk_volk_generic_32fc_deinterleave_32f_x2(
            re.as_mut_ptr(),
            im.as_mut_ptr(),
            input.as_ptr(),
            n,
        );
    }
    #[cfg(not(has_volkgeneric))]
    let _ = n;
    AVAILABLE
}

/// The loop of `volk_32f_x2_interleave_32fc_generic` on 16-bit values: `out[i] = (re[i],
/// im[i])` for the common length; `false` without the library.
pub fn interleave_16ic(re: &[i16], im: &[i16], out: &mut [[i16; 2]]) -> bool {
    let n = common(re.len(), im.len(), out.len());
    #[cfg(has_volkgeneric)]
    // SAFETY: as in `interleave_32fc`.
    unsafe {
        ffi::sdrk_volk_generic_16i_x2_interleave_16ic(
            out.as_mut_ptr(),
            re.as_ptr(),
            im.as_ptr(),
            n,
        );
    }
    #[cfg(not(has_volkgeneric))]
    let _ = n;
    AVAILABLE
}

/// `volk_16ic_deinterleave_16i_x2_generic`: `re[i] = input[i][0]`, `im[i] = input[i][1]` for
/// the common length; `false` without the library.
pub fn deinterleave_16ic(input: &[[i16; 2]], re: &mut [i16], im: &mut [i16]) -> bool {
    let n = common(input.len(), re.len(), im.len());
    #[cfg(has_volkgeneric)]
    // SAFETY: as in `deinterleave_32fc`.
    unsafe {
        ffi::sdrk_volk_generic_16ic_deinterleave_16i_x2(
            re.as_mut_ptr(),
            im.as_mut_ptr(),
            input.as_ptr(),
            n,
        );
    }
    #[cfg(not(has_volkgeneric))]
    let _ = n;
    AVAILABLE
}

/// `volk_32fc_x2_multiply_32fc_generic`: `out[i] = a[i] * b[i]` (complex, C99 complex
/// arithmetic, not fused) for the common length; `false` without the library.
pub fn multiply_32fc(a: &[[f32; 2]], b: &[[f32; 2]], out: &mut [[f32; 2]]) -> bool {
    let n = common(a.len(), b.len(), out.len());
    #[cfg(has_volkgeneric)]
    // SAFETY: every buffer holds at least `n` complex samples; the output is a separate mutable
    // slice, so it aliases no input.
    unsafe {
        ffi::sdrk_volk_generic_32fc_x2_multiply_32fc(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    }
    #[cfg(not(has_volkgeneric))]
    let _ = n;
    AVAILABLE
}

/// `volk_32fc_32f_multiply_32fc_generic`: `out[i] = a[i] * b[i]` (a complex sample times a real
/// value, two products) for the common length; `false` without the library.
pub fn multiply_32fc_32f(a: &[[f32; 2]], b: &[f32], out: &mut [[f32; 2]]) -> bool {
    let n = common(a.len(), b.len(), out.len());
    #[cfg(has_volkgeneric)]
    // SAFETY: every buffer holds at least `n` elements; the output is a separate mutable slice,
    // so it aliases no input.
    unsafe {
        ffi::sdrk_volk_generic_32fc_32f_multiply_32fc(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    }
    #[cfg(not(has_volkgeneric))]
    let _ = n;
    AVAILABLE
}

/// `volk_32f_s32f_convert_16i_generic`: `out[i] = rintf(clamp(input[i] * scale))` for the
/// common length; `false` without the library.
pub fn convert_16i(input: &[f32], scale: f32, out: &mut [i16]) -> bool {
    let n = common(input.len(), out.len(), usize::MAX);
    #[cfg(has_volkgeneric)]
    // SAFETY: both buffers hold at least `n` elements; the output is a separate mutable slice.
    unsafe {
        ffi::sdrk_volk_generic_32f_s32f_convert_16i(out.as_mut_ptr(), input.as_ptr(), scale, n);
    }
    #[cfg(not(has_volkgeneric))]
    let _ = (n, scale);
    AVAILABLE
}

/// `volk_32f_s32f_convert_8i_generic`: `out[i] = rintf(clamp(input[i] * scale))` for the common
/// length; `false` without the library.
pub fn convert_8i(input: &[f32], scale: f32, out: &mut [i8]) -> bool {
    let n = common(input.len(), out.len(), usize::MAX);
    #[cfg(has_volkgeneric)]
    // SAFETY: as in `convert_16i`.
    unsafe {
        ffi::sdrk_volk_generic_32f_s32f_convert_8i(out.as_mut_ptr(), input.as_ptr(), scale, n);
    }
    #[cfg(not(has_volkgeneric))]
    let _ = (n, scale);
    AVAILABLE
}

/// `volk_32f_s32f_x2_convert_8u_generic`: `out[i] = rintf(clamp(input[i] * scale + bias))` for
/// the common length; `false` without the library.
pub fn convert_8u(input: &[f32], scale: f32, bias: f32, out: &mut [u8]) -> bool {
    let n = common(input.len(), out.len(), usize::MAX);
    #[cfg(has_volkgeneric)]
    // SAFETY: as in `convert_16i`.
    unsafe {
        ffi::sdrk_volk_generic_32f_s32f_x2_convert_8u(
            out.as_mut_ptr(),
            input.as_ptr(),
            scale,
            bias,
            n,
        );
    }
    #[cfg(not(has_volkgeneric))]
    let _ = (n, scale, bias);
    AVAILABLE
}

/// `volk_32fc_x2_multiply_conjugate_32fc_generic`: `out[i] = a[i] * conj(b[i])` for the common
/// length; `false` without the library.
pub fn multiply_conjugate(a: &[[f32; 2]], b: &[[f32; 2]], out: &mut [[f32; 2]]) -> bool {
    let n = common(a.len(), b.len(), out.len());
    #[cfg(has_volkgeneric)]
    // SAFETY: every buffer holds at least `n` complex samples; the output is a separate mutable
    // slice, so it aliases no input.
    unsafe {
        ffi::sdrk_volk_generic_32fc_x2_multiply_conjugate_32fc(
            out.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            n,
        );
    }
    #[cfg(not(has_volkgeneric))]
    let _ = n;
    AVAILABLE
}

/// `volk_32fc_s32f_atan2_32f_generic`: `out[i] = atan2f(c[i][1], c[i][0]) / normalize` (libm's
/// `atan2f`) for the common length; `false` without the library.
pub fn atan2(c: &[[f32; 2]], normalize: f32, out: &mut [f32]) -> bool {
    let n = common(c.len(), out.len(), usize::MAX);
    #[cfg(has_volkgeneric)]
    // SAFETY: both buffers hold at least `n` elements; the output is a separate mutable slice.
    unsafe {
        ffi::sdrk_volk_generic_32fc_s32f_atan2_32f(out.as_mut_ptr(), c.as_ptr(), normalize, n);
    }
    #[cfg(not(has_volkgeneric))]
    let _ = (n, normalize);
    AVAILABLE
}
