//! The VOLK functions used by the SDRk benchmarks (VOLK is LGPL-3.0; it is linked dynamically
//! from the system and is not part of SDRk).
//!
//! Only the signatures that the benchmarks need are declared. `lv_32fc_t` is a pair of `f32`.

#![allow(non_camel_case_types, clippy::missing_safety_doc)]

/// VOLK's `lv_32fc_t`: a complex float, real part first.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct lv_32fc_t {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

/// Whether the library was found at build time.
pub const AVAILABLE: bool = cfg!(not(no_volk));

#[cfg(not(no_volk))]
mod ffi {
    use super::lv_32fc_t;
    use core::ffi::c_uint;

    // VOLK declares every kernel as a function-pointer variable that its dispatcher fills in
    // when the library is loaded (`extern p_xxx volk_xxx;`), not as a function.
    unsafe extern "C" {
        pub static volk_8i_s32f_convert_32f:
            Option<unsafe extern "C" fn(*mut f32, *const i8, f32, c_uint)>;
        pub static volk_16i_s32f_convert_32f:
            Option<unsafe extern "C" fn(*mut f32, *const i16, f32, c_uint)>;
        pub static volk_32f_s32f_convert_16i:
            Option<unsafe extern "C" fn(*mut i16, *const f32, f32, c_uint)>;
        pub static volk_32fc_s32fc_x2_rotator2_32fc: Option<
            unsafe extern "C" fn(
                *mut lv_32fc_t,
                *const lv_32fc_t,
                *const lv_32fc_t,
                *mut lv_32fc_t,
                c_uint,
            ),
        >;
        pub static volk_32fc_32f_dot_prod_32fc:
            Option<unsafe extern "C" fn(*mut lv_32fc_t, *const lv_32fc_t, *const f32, c_uint)>;
        pub static volk_32fc_magnitude_squared_32f:
            Option<unsafe extern "C" fn(*mut f32, *const lv_32fc_t, c_uint)>;
        pub static volk_32fc_x2_multiply_conjugate_32fc: Option<
            unsafe extern "C" fn(*mut lv_32fc_t, *const lv_32fc_t, *const lv_32fc_t, c_uint),
        >;
        pub static volk_32fc_s32f_atan2_32f:
            Option<unsafe extern "C" fn(*mut f32, *const lv_32fc_t, f32, c_uint)>;
    }
}

macro_rules! wrappers {
    ($( $(#[$m:meta])* fn $name:ident($($a:ident: $t:ty),*); )+) => {
        $(
            $(#[$m])*
            #[cfg(not(no_volk))]
            pub unsafe fn $name($($a: $t),*) {
                // SAFETY: the dispatcher of the library fills the pointer when it is loaded;
                // the caller upholds the contract of the VOLK function of this name.
                unsafe {
                    if let Some(f) = ffi::$name {
                        f($($a),*)
                    }
                }
            }
            $(#[$m])*
            #[cfg(no_volk)]
            pub unsafe fn $name($(_: $t),*) {
                unreachable!("VOLK is not available")
            }
        )+
    };
}

wrappers! {
    /// `out[i] = in[i] / scalar` (VOLK multiplies by the reciprocal).
    fn volk_8i_s32f_convert_32f(out: *mut f32, input: *const i8, scalar: f32, n: u32);
    /// `out[i] = in[i] / scalar`.
    fn volk_16i_s32f_convert_32f(out: *mut f32, input: *const i16, scalar: f32, n: u32);
    /// `out[i] = saturate(round(in[i] * scalar))`.
    fn volk_32f_s32f_convert_16i(out: *mut i16, input: *const f32, scalar: f32, n: u32);
    /// Frequency shift by a complex phase increment; `phase` is updated.
    fn volk_32fc_s32fc_x2_rotator2_32fc(
        out: *mut lv_32fc_t,
        input: *const lv_32fc_t,
        phase_inc: *const lv_32fc_t,
        phase: *mut lv_32fc_t,
        n: u32
    );
    /// One complex output: the dot product of complex input and real taps.
    fn volk_32fc_32f_dot_prod_32fc(
        result: *mut lv_32fc_t,
        input: *const lv_32fc_t,
        taps: *const f32,
        n: u32
    );
    /// `out[i] = |in[i]|^2`.
    fn volk_32fc_magnitude_squared_32f(out: *mut f32, input: *const lv_32fc_t, n: u32);
    /// `out[i] = a[i] * conj(b[i])`.
    fn volk_32fc_x2_multiply_conjugate_32fc(
        out: *mut lv_32fc_t,
        a: *const lv_32fc_t,
        b: *const lv_32fc_t,
        n: u32
    );
    /// `out[i] = atan2(im, re) / norm`.
    fn volk_32fc_s32f_atan2_32f(out: *mut f32, input: *const lv_32fc_t, norm: f32, n: u32);
}
