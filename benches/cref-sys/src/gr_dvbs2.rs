//! The per-symbol PL scrambling switch of GNU Radio's `dvbs2_physical_cc` (GPL-3.0-or-later;
//! `benches/refs/gr-dvbs2-plscramble.c`, cfg `has_grdvbs2`), the information reference of
//! `scramble` at `bits = 2` (M49 §5), marked "reference does different work": a `switch` per
//! symbol on a precomputed `int` sequence `R_n` in 0..3 (`out = in · j^{R_n}`), without the PL
//! header, the pilots, the zero samples between the symbols and the generation of `R_n`.

#[cfg(has_grdvbs2)]
mod ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_dvbs2_plscramble(
            input: *const [f32; 2],
            out: *mut [f32; 2],
            rn: *const core::ffi::c_int,
            n: core::ffi::c_int,
        );
    }
}

/// Whether the library is linked.
pub const AVAILABLE: bool = cfg!(has_grdvbs2);

/// `out[i] = input[i] · j^{rn[i]}` for the common length (`rn[i]` in 0..3, other values leave the
/// previous output value, as the C++ switch without a default); `false` without the library.
pub fn plscramble(input: &[[f32; 2]], rn: &[core::ffi::c_int], out: &mut [[f32; 2]]) -> bool {
    let n = input.len().min(rn.len()).min(out.len());
    #[cfg(has_grdvbs2)]
    // SAFETY: the three buffers hold at least `n` elements; the output is a separate mutable
    // slice, so it aliases no input.
    unsafe {
        ffi::sdrk_gr_dvbs2_plscramble(
            input.as_ptr(),
            out.as_mut_ptr(),
            rn.as_ptr(),
            n as core::ffi::c_int,
        );
    }
    #[cfg(not(has_grdvbs2))]
    let _ = n;
    AVAILABLE
}
