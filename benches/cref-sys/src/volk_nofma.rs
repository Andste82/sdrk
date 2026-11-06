//! VOLK 3.3.0's complex product and complex dot product without fused multiply-add
//! (LGPL-3.0-or-later, `benches/refs/volk-nofma.c`, cfg `has_volknofma`): the protokernels of
//! `volk_32fc_x2_multiply_32fc` and `volk_32fc_x2_dot_prod_32fc` that multiply and add in
//! separate operations (`u_avx` on x86-64, `neon` on AArch64), compiled with
//! `-ffp-contract=off` for the levels `avx2`, `avx512` and `neon` (`libvolknofma-<level>.a`).
//! Information for `mul` `C32` at these levels (its gate from D287 until D326) and the gate of
//! `dot_complex` there (D821).

#[cfg(has_volknofma)]
mod ffi {
    unsafe extern "C" {
        pub fn sdrk_volk_nofma_32fc_x2_multiply_32fc(
            c: *mut [f32; 2],
            a: *const [f32; 2],
            b: *const [f32; 2],
            n: core::ffi::c_uint,
        );
        pub fn sdrk_volk_nofma_32fc_x2_multiply_32fc_name() -> *const core::ffi::c_char;
        pub fn sdrk_volk_nofma_32fc_x2_dot_prod_32fc(
            result: *mut [f32; 2],
            a: *const [f32; 2],
            b: *const [f32; 2],
            n: core::ffi::c_uint,
        );
    }
}

/// Whether the library is linked (the level is `avx2`, `avx512` or `neon`).
pub const AVAILABLE: bool = cfg!(has_volknofma);

/// The name of the protokernel, `None` without the library.
#[must_use]
pub fn protokernel() -> Option<&'static str> {
    #[cfg(has_volknofma)]
    {
        // SAFETY: the function returns a pointer to a static NUL-terminated ASCII string.
        let name =
            unsafe { core::ffi::CStr::from_ptr(ffi::sdrk_volk_nofma_32fc_x2_multiply_32fc_name()) };
        name.to_str().ok()
    }
    #[cfg(not(has_volknofma))]
    {
        None
    }
}

/// `out[i] = a[i] * b[i]` (complex) for the common length of the three slices; `false`
/// without the library.
pub fn multiply_32fc(a: &[[f32; 2]], b: &[[f32; 2]], out: &mut [[f32; 2]]) -> bool {
    let n = a.len().min(b.len()).min(out.len());
    #[cfg(has_volknofma)]
    // SAFETY: the three buffers hold at least `n` complex samples each; the output is a
    // separate mutable slice, so it aliases no input.
    unsafe {
        ffi::sdrk_volk_nofma_32fc_x2_multiply_32fc(
            out.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            n as core::ffi::c_uint,
        );
    }
    #[cfg(not(has_volknofma))]
    let _ = n;
    AVAILABLE
}

/// `sum a[i] * b[i]` (complex, not conjugated) over the common length of the two slices, as
/// `[re, im]`; `None` without the library. The protokernel is the one [`protokernel`] names
/// (`volk_32fc_x2_dot_prod_32fc_u_avx` or `_neon`).
#[must_use]
pub fn dot_prod_32fc(a: &[[f32; 2]], b: &[[f32; 2]]) -> Option<[f32; 2]> {
    let n = a.len().min(b.len());
    #[cfg(has_volknofma)]
    {
        let mut result = [0.0f32; 2];
        // SAFETY: both inputs hold at least `n` complex samples; the result is a local.
        unsafe {
            ffi::sdrk_volk_nofma_32fc_x2_dot_prod_32fc(
                &raw mut result,
                a.as_ptr(),
                b.as_ptr(),
                n as core::ffi::c_uint,
            );
        }
        Some(result)
    }
    #[cfg(not(has_volknofma))]
    {
        let _ = n;
        None
    }
}
