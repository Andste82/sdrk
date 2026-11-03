//! VOLK 3.3.0's portable `generic` protokernels of the interleave and deinterleave kernels
//! (LGPL-3.0-or-later, `benches/refs/volk-generic.c`, cfg `has_volkgeneric`), compiled with
//! `-O3 -ffp-contract=off` and the instruction set of the level (`libvolkgeneric-<level>.a`):
//! the scalar references of the `interleave` and `deinterleave` fallbacks. VOLK has no 16-bit
//! interleave; [`interleave_16ic`] is the loop of `volk_32f_x2_interleave_32fc_generic` with
//! 16-bit elements.

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
        ffi::sdrk_volk_generic_32f_x2_interleave_32fc(out.as_mut_ptr(), re.as_ptr(), im.as_ptr(), n);
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
        ffi::sdrk_volk_generic_16i_x2_interleave_16ic(out.as_mut_ptr(), re.as_ptr(), im.as_ptr(), n);
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
