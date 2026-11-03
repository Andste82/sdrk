//! VOLK 3.3.0's complex product without fused multiply-add (LGPL-3.0-or-later,
//! `benches/refs/volk-nofma.c`, cfg `has_volknofma`): the protokernel of
//! `volk_32fc_x2_multiply_32fc` that multiplies and adds in separate operations (`u_avx` on
//! x86-64, `neon` on AArch64), compiled with `-ffp-contract=off` for the levels `avx2`,
//! `avx512` and `neon` (`libvolknofma-<level>.a`). The gate of `mul` `C32` at these levels
//! (D287).

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
