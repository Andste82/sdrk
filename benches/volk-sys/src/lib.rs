//! The VOLK functions used by the SDRk benchmarks (VOLK is LGPL-3.0; `benches/refs/build.sh`
//! builds it from the release tarball, and it is never part of SDRk).
//!
//! Only the signatures that the benchmarks need are declared. `lv_32fc_t` is a pair of `f32`.

#![allow(non_camel_case_types, clippy::missing_safety_doc)]

#[cfg(not(no_volk))]
use core::ffi::CStr;
use core::ffi::{c_char, c_uint};

/// VOLK's `lv_32fc_t`: a complex float, real part first.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct lv_32fc_t {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

/// VOLK's `lv_16sc_t`: a complex 16-bit integer, real part first.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct lv_16sc_t {
    /// Real part.
    pub re: i16,
    /// Imaginary part.
    pub im: i16,
}

/// Whether the library was found at build time.
pub const AVAILABLE: bool = cfg!(not(no_volk));

/// Whether the linked VOLK is VOLK's own build (x86-64, AArch64) with its SIMD protokernels.
/// The comparisons of the first rounds that use the functions of [`native`],
/// [`volk_8i_convert_16i`] and [`volk_32f_x2_dot_prod_32f`] check this constant (or
/// [`HAS_8I_CONVERT_16I`], [`AVAILABLE_REAL_DOT`]) and therefore stay off on wasm32.
pub const NATIVE: bool = AVAILABLE && cfg!(not(target_arch = "wasm32"));

/// Whether every function declared here links: with VOLK on every platform. Since the fallback
/// round the wasm32 library of `benches/refs/volk-wasm.c` exports the generic protokernels of
/// all declarations (VOLK's own dispatch on a machine without SIMD), so a comparison that
/// should also run at `simd128` checks this constant instead of [`NATIVE`].
pub const FULL: bool = AVAILABLE;

/// The signature of `volk_32fc_32f_dot_prod_32fc` and its aligned variant.
pub type DotProd = unsafe extern "C" fn(*mut lv_32fc_t, *const lv_32fc_t, *const f32, c_uint);

/// The signature of `volk_32f_x2_dot_prod_32f` and its aligned variant.
pub type DotProdReal = unsafe extern "C" fn(*mut f32, *const f32, *const f32, c_uint);

/// The signature of `volk_32f_x2_add_32f` and `volk_32f_x2_subtract_32f` (and their aligned
/// variants): `out[i] = a[i] +- b[i]`.
pub type AddSub = unsafe extern "C" fn(*mut f32, *const f32, *const f32, c_uint);

/// The signature of `volk_32fc_deinterleave_32f_x2` and its aligned variant:
/// `re[i]`, `im[i]` from the interleaved complex input.
pub type Deinterleave = unsafe extern "C" fn(*mut f32, *mut f32, *const lv_32fc_t, c_uint);

/// The signature of `volk_32f_x2_interleave_32fc` and its aligned variant.
pub type Interleave = unsafe extern "C" fn(*mut lv_32fc_t, *const f32, *const f32, c_uint);

/// VOLK's `volk_func_desc_t`: the implementations of a kernel on the machine VOLK selected.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct volk_func_desc_t {
    /// `n_impls` implementation names.
    pub impl_names: *const *const c_char,
    /// The architecture mask each implementation needs (the dispatcher ranks by it).
    pub impl_deps: *const i32,
    /// Whether each implementation needs aligned buffers.
    pub impl_alignment: *const bool,
    /// Number of implementations.
    pub n_impls: usize,
}

#[cfg(not(no_volk))]
mod ffi {
    use super::DotProdReal;
    use super::{DotProd, lv_16sc_t, lv_32fc_t, volk_func_desc_t};
    use core::ffi::{c_char, c_uint};

    // VOLK declares every kernel as a function-pointer variable (`extern p_xxx volk_xxx;`)
    // that starts at an initialisation stub: the first call ranks the implementations, stores
    // the chosen one in the variable and calls it. The variables are therefore mutable.
    unsafe extern "C" {
        pub static mut volk_8i_s32f_convert_32f:
            Option<unsafe extern "C" fn(*mut f32, *const i8, f32, c_uint)>;
        pub static mut volk_16i_s32f_convert_32f:
            Option<unsafe extern "C" fn(*mut f32, *const i16, f32, c_uint)>;
        pub static mut volk_8i_convert_16i:
            Option<unsafe extern "C" fn(*mut i16, *const i8, c_uint)>;
        pub static mut volk_32f_s32f_convert_16i:
            Option<unsafe extern "C" fn(*mut i16, *const f32, f32, c_uint)>;
        pub static mut volk_32f_s32f_convert_8i:
            Option<unsafe extern "C" fn(*mut i8, *const f32, f32, c_uint)>;
        pub static mut volk_32f_s32f_x2_convert_8u:
            Option<unsafe extern "C" fn(*mut u8, *const f32, f32, f32, c_uint)>;
        pub static mut volk_32fc_s32fc_x2_rotator2_32fc: Option<
            unsafe extern "C" fn(
                *mut lv_32fc_t,
                *const lv_32fc_t,
                *const lv_32fc_t,
                *mut lv_32fc_t,
                c_uint,
            ),
        >;
        pub static mut volk_32fc_32f_dot_prod_32fc_a: Option<DotProd>;
        pub static mut volk_32fc_32f_dot_prod_32fc: Option<DotProd>;
        pub static mut volk_32fc_magnitude_squared_32f:
            Option<unsafe extern "C" fn(*mut f32, *const lv_32fc_t, c_uint)>;
        pub static mut volk_32fc_magnitude_32f:
            Option<unsafe extern "C" fn(*mut f32, *const lv_32fc_t, c_uint)>;
        pub static mut volk_16ic_magnitude_16i:
            Option<unsafe extern "C" fn(*mut i16, *const lv_16sc_t, c_uint)>;
        pub static mut volk_32fc_x2_multiply_conjugate_32fc: Option<
            unsafe extern "C" fn(*mut lv_32fc_t, *const lv_32fc_t, *const lv_32fc_t, c_uint),
        >;
        pub static mut volk_32fc_s32f_atan2_32f:
            Option<unsafe extern "C" fn(*mut f32, *const lv_32fc_t, f32, c_uint)>;

        pub fn volk_8i_s32f_convert_32f_get_func_desc() -> volk_func_desc_t;
        pub fn volk_16i_s32f_convert_32f_get_func_desc() -> volk_func_desc_t;
        pub fn volk_8i_convert_16i_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32f_s32f_convert_16i_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32f_s32f_convert_8i_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32f_s32f_x2_convert_8u_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32fc_s32fc_x2_rotator2_32fc_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32fc_32f_dot_prod_32fc_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32fc_magnitude_squared_32f_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32fc_magnitude_32f_get_func_desc() -> volk_func_desc_t;
        pub fn volk_16ic_magnitude_16i_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32fc_x2_multiply_conjugate_32fc_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32fc_s32f_atan2_32f_get_func_desc() -> volk_func_desc_t;

        pub fn volk_get_alignment() -> usize;
        pub fn volk_get_machine() -> *const c_char;
        pub fn volk_version() -> *const c_char;
        pub fn volk_compiler_flags() -> *const c_char;
    }

    unsafe extern "C" {
        pub static mut volk_32f_x2_dot_prod_32f_a: Option<DotProdReal>;
        pub static mut volk_32f_x2_dot_prod_32f:
            Option<unsafe extern "C" fn(*mut f32, *const f32, *const f32, c_uint)>;
        pub static mut volk_32fc_x2_dot_prod_32fc: Option<
            unsafe extern "C" fn(*mut lv_32fc_t, *const lv_32fc_t, *const lv_32fc_t, c_uint),
        >;

        pub fn volk_32f_x2_dot_prod_32f_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32fc_x2_dot_prod_32fc_get_func_desc() -> volk_func_desc_t;
    }

    // The kernels of the `stereo_matrix` composition (deinterleave, add, subtract, interleave).
    unsafe extern "C" {
        pub static mut volk_32f_x2_add_32f: Option<super::AddSub>;
        pub static mut volk_32f_x2_subtract_32f: Option<super::AddSub>;
        pub static mut volk_32fc_deinterleave_32f_x2: Option<super::Deinterleave>;
        pub static mut volk_32f_x2_interleave_32fc: Option<super::Interleave>;

        pub fn volk_32f_x2_add_32f_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32f_x2_subtract_32f_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32fc_deinterleave_32f_x2_get_func_desc() -> volk_func_desc_t;
        pub fn volk_32f_x2_interleave_32fc_get_func_desc() -> volk_func_desc_t;
    }
}

macro_rules! wrappers {
    ($( $(#[$m:meta])* fn $name:ident($($a:ident: $t:ty),*); )+) => {
        $(
            $(#[$m])*
            #[cfg(not(no_volk))]
            pub unsafe fn $name($($a: $t),*) {
                // SAFETY: the variable is read once per call (the dispatcher may replace it on
                // the first call, never concurrently here); the caller upholds the contract of
                // the VOLK function of this name.
                unsafe {
                    if let Some(f) = (&raw const ffi::$name).read() {
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
    /// `out[i] = saturate(round(in[i] * scalar))` to `i8`.
    fn volk_32f_s32f_convert_8i(out: *mut i8, input: *const f32, scalar: f32, n: u32);
    /// `out[i] = saturate(round(in[i] * scale + bias))` to `u8`.
    fn volk_32f_s32f_x2_convert_8u(out: *mut u8, input: *const f32, scale: f32, bias: f32, n: u32);
    /// Frequency shift by a complex phase increment; `phase` is updated.
    fn volk_32fc_s32fc_x2_rotator2_32fc(
        out: *mut lv_32fc_t,
        input: *const lv_32fc_t,
        phase_inc: *const lv_32fc_t,
        phase: *mut lv_32fc_t,
        n: u32
    );
    /// `out[i] = |in[i]|^2`.
    fn volk_32fc_magnitude_squared_32f(out: *mut f32, input: *const lv_32fc_t, n: u32);
    /// `out[i] = |in[i]|`.
    fn volk_32fc_magnitude_32f(out: *mut f32, input: *const lv_32fc_t, n: u32);
    /// `out[i] = |in[i]|` for 16-bit integer samples (VOLK scales by 1/32768 and back, and
    /// rounds in single precision).
    fn volk_16ic_magnitude_16i(out: *mut i16, input: *const lv_16sc_t, n: u32);
    /// `out[i] = a[i] * conj(b[i])`.
    fn volk_32fc_x2_multiply_conjugate_32fc(
        out: *mut lv_32fc_t,
        a: *const lv_32fc_t,
        b: *const lv_32fc_t,
        n: u32
    );
    /// `out[i] = atan2(im, re) / norm`.
    fn volk_32fc_s32f_atan2_32f(out: *mut f32, input: *const lv_32fc_t, norm: f32, n: u32);
    /// `*result = sum a[i] * taps[i]` of complex samples with real taps (with the alignment
    /// check of the dispatcher).
    fn volk_32fc_32f_dot_prod_32fc(result: *mut lv_32fc_t, a: *const lv_32fc_t, taps: *const f32, n: u32);
}

wrappers! {
    /// `*result = sum a[i] * b[i]`.
    fn volk_32f_x2_dot_prod_32f(result: *mut f32, a: *const f32, b: *const f32, n: u32);
    /// `*result = sum a[i] * b[i]` of complex vectors (no conjugate).
    fn volk_32fc_x2_dot_prod_32fc(result: *mut lv_32fc_t, a: *const lv_32fc_t, b: *const lv_32fc_t, n: u32);
}

wrappers! {
    /// `out[i] = a[i] + b[i]` (the dispatcher's unaligned entry).
    fn volk_32f_x2_add_32f(out: *mut f32, a: *const f32, b: *const f32, n: u32);
    /// `out[i] = a[i] - b[i]`.
    fn volk_32f_x2_subtract_32f(out: *mut f32, a: *const f32, b: *const f32, n: u32);
    /// `re[i] = in[i].re`, `im[i] = in[i].im`.
    fn volk_32fc_deinterleave_32f_x2(re: *mut f32, im: *mut f32, input: *const lv_32fc_t, n: u32);
    /// `out[i] = (re[i], im[i])`.
    fn volk_32f_x2_interleave_32fc(out: *mut lv_32fc_t, re: *const f32, im: *const f32, n: u32);
}

/// The VOLK functions that the first rounds compared only against VOLK's own build
/// ([`NATIVE`]); since the fallback round they exist on wasm32 too (see [`FULL`]). Without
/// VOLK every wrapper is a failing stub.
pub mod native {
    use super::lv_32fc_t;

    #[cfg(not(no_volk))]
    pub(crate) mod ffi {
        use crate::{lv_32fc_t, volk_func_desc_t};
        use core::ffi::c_uint;

        // Function-pointer variables, as in the parent module.
        unsafe extern "C" {
            pub static mut volk_32fc_32f_multiply_32fc:
                Option<unsafe extern "C" fn(*mut lv_32fc_t, *const lv_32fc_t, *const f32, c_uint)>;
            pub static mut volk_32f_log2_32f:
                Option<unsafe extern "C" fn(*mut f32, *const f32, c_uint)>;
            pub static mut volk_32f_s32f_multiply_32f:
                Option<unsafe extern "C" fn(*mut f32, *const f32, f32, c_uint)>;
            pub static mut volk_32f_s32f_add_32f:
                Option<unsafe extern "C" fn(*mut f32, *const f32, f32, c_uint)>;

            pub fn volk_32fc_32f_multiply_32fc_get_func_desc() -> volk_func_desc_t;
            pub fn volk_32f_log2_32f_get_func_desc() -> volk_func_desc_t;
            pub fn volk_32f_s32f_multiply_32f_get_func_desc() -> volk_func_desc_t;
            pub fn volk_32f_s32f_add_32f_get_func_desc() -> volk_func_desc_t;
        }
    }

    macro_rules! native_wrappers {
        ($( $(#[$m:meta])* fn $name:ident($($a:ident: $t:ty),*); )+) => {
            $(
                $(#[$m])*
                #[cfg(not(no_volk))]
                pub unsafe fn $name($($a: $t),*) {
                    // SAFETY: as for the wrappers of the parent module.
                    unsafe {
                        if let Some(f) = (&raw const ffi::$name).read() {
                            f($($a),*)
                        }
                    }
                }
                $(#[$m])*
                #[cfg(no_volk)]
                pub unsafe fn $name($(_: $t),*) {
                    unreachable!("this VOLK function is not available")
                }
            )+
        };
    }

    native_wrappers! {
        /// `out[i] = a[i] * b[i]`, a complex vector times a real one.
        fn volk_32fc_32f_multiply_32fc(out: *mut lv_32fc_t, a: *const lv_32fc_t, b: *const f32, n: u32);
        /// `out[i] = log2(in[i])`.
        fn volk_32f_log2_32f(out: *mut f32, input: *const f32, n: u32);
        /// `out[i] = in[i] * scalar`.
        fn volk_32f_s32f_multiply_32f(out: *mut f32, input: *const f32, scalar: f32, n: u32);
        /// `out[i] = in[i] + scalar`.
        fn volk_32f_s32f_add_32f(out: *mut f32, input: *const f32, scalar: f32, n: u32);
    }
}

/// Whether the comparison with [`volk_8i_convert_16i`] of the first rounds runs: with VOLK's own
/// build only (see [`NATIVE`]); the function itself links wherever VOLK does ([`FULL`]).
pub const HAS_8I_CONVERT_16I: bool = AVAILABLE && cfg!(not(target_arch = "wasm32"));

/// `out[i] = in[i] * 256`.
#[cfg(not(no_volk))]
pub unsafe fn volk_8i_convert_16i(out: *mut i16, input: *const i8, n: u32) {
    // SAFETY: as in the functions of `wrappers!`.
    unsafe {
        if let Some(f) = (&raw const ffi::volk_8i_convert_16i).read() {
            f(out, input, n);
        }
    }
}

/// Without VOLK.
#[cfg(no_volk)]
pub unsafe fn volk_8i_convert_16i(_: *mut i16, _: *const i8, _: u32) {
    unreachable!("volk_8i_convert_16i is not available")
}

/// The current value of `volk_32fc_32f_dot_prod_32fc_a`, the aligned dot product as GNU Radio's
/// `fir_filter_ccf` calls it: no alignment check per call. Before the first call it is VOLK's
/// initialisation stub, which resolves and calls the implementation.
#[cfg(not(no_volk))]
#[must_use]
pub fn dot_prod_a() -> Option<DotProd> {
    // SAFETY: a plain read of the variable; only VOLK's stub writes it, on the first call.
    unsafe { (&raw const ffi::volk_32fc_32f_dot_prod_32fc_a).read() }
}

/// Without VOLK: `None`.
#[cfg(no_volk)]
#[must_use]
pub fn dot_prod_a() -> Option<DotProd> {
    None
}

/// The current value of `volk_32f_x2_dot_prod_32f_a`, the aligned dot product as GNU Radio's
/// `fir_filter_fff` calls it (see [`dot_prod_a`]).
#[cfg(not(no_volk))]
#[must_use]
pub fn dot_prod_real_a() -> Option<DotProdReal> {
    // SAFETY: a plain read of the variable; only VOLK's stub writes it, on the first call.
    unsafe { (&raw const ffi::volk_32f_x2_dot_prod_32f_a).read() }
}

/// Without VOLK: `None`.
#[cfg(no_volk)]
#[must_use]
pub fn dot_prod_real_a() -> Option<DotProdReal> {
    None
}

/// Whether the comparison with [`dot_prod_real_a`] of the first rounds runs: with VOLK's own
/// build only (see [`NATIVE`]); the function itself links wherever VOLK does ([`FULL`]).
pub const AVAILABLE_REAL_DOT: bool = AVAILABLE && cfg!(not(target_arch = "wasm32"));

/// VOLK's alignment in bytes (the largest alignment of the selected machine's architectures).
#[cfg(not(no_volk))]
#[must_use]
pub fn alignment() -> usize {
    // SAFETY: no arguments; selects the machine on the first call.
    unsafe { ffi::volk_get_alignment() }
}

/// Without VOLK: 1.
#[cfg(no_volk)]
#[must_use]
pub fn alignment() -> usize {
    1
}

/// Version, selected machine and compiler flags of the linked VOLK.
#[derive(Clone, Debug)]
pub struct Info {
    /// `volk_version()`.
    pub version: String,
    /// `volk_get_machine()`.
    pub machine: String,
    /// `volk_compiler_flags()`.
    pub compiler_flags: String,
}

/// The [`Info`] of the linked VOLK, `None` without VOLK.
#[must_use]
pub fn info() -> Option<Info> {
    #[cfg(not(no_volk))]
    {
        let text = |p: *const c_char| -> String {
            if p.is_null() {
                String::new()
            } else {
                // SAFETY: VOLK returns pointers to static NUL-terminated strings.
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        };
        // SAFETY: the three functions take no arguments and return static strings.
        let (version, machine, flags) = unsafe {
            (
                ffi::volk_version(),
                ffi::volk_get_machine(),
                ffi::volk_compiler_flags(),
            )
        };
        Some(Info {
            version: text(version),
            machine: text(machine),
            compiler_flags: text(flags),
        })
    }
    #[cfg(no_volk)]
    {
        None
    }
}

/// One implementation of a VOLK kernel on the selected machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Implementation {
    /// The name a `volk_config` line uses.
    pub name: String,
    /// The architecture mask it needs; the dispatcher prefers the largest.
    pub deps: i32,
    /// Whether it needs aligned buffers.
    pub aligned: bool,
}

/// The implementations of the VOLK function `name` on the selected machine, `None` for a name
/// that is not declared here or without VOLK.
#[must_use]
pub fn implementations(name: &str) -> Option<Vec<Implementation>> {
    #[cfg(not(no_volk))]
    {
        let get: unsafe extern "C" fn() -> volk_func_desc_t = match name {
            "volk_8i_s32f_convert_32f" => ffi::volk_8i_s32f_convert_32f_get_func_desc,
            "volk_16i_s32f_convert_32f" => ffi::volk_16i_s32f_convert_32f_get_func_desc,
            "volk_8i_convert_16i" => ffi::volk_8i_convert_16i_get_func_desc,
            "volk_32f_s32f_convert_16i" => ffi::volk_32f_s32f_convert_16i_get_func_desc,
            "volk_32f_s32f_convert_8i" => ffi::volk_32f_s32f_convert_8i_get_func_desc,
            "volk_32f_s32f_x2_convert_8u" => ffi::volk_32f_s32f_x2_convert_8u_get_func_desc,
            "volk_32fc_s32fc_x2_rotator2_32fc" => {
                ffi::volk_32fc_s32fc_x2_rotator2_32fc_get_func_desc
            }
            "volk_32fc_32f_dot_prod_32fc" => ffi::volk_32fc_32f_dot_prod_32fc_get_func_desc,
            "volk_32fc_magnitude_squared_32f" => ffi::volk_32fc_magnitude_squared_32f_get_func_desc,
            "volk_32fc_magnitude_32f" => ffi::volk_32fc_magnitude_32f_get_func_desc,
            "volk_16ic_magnitude_16i" => ffi::volk_16ic_magnitude_16i_get_func_desc,
            "volk_32fc_x2_multiply_conjugate_32fc" => {
                ffi::volk_32fc_x2_multiply_conjugate_32fc_get_func_desc
            }
            "volk_32fc_s32f_atan2_32f" => ffi::volk_32fc_s32f_atan2_32f_get_func_desc,
            "volk_32fc_32f_multiply_32fc" => native::ffi::volk_32fc_32f_multiply_32fc_get_func_desc,
            "volk_32f_log2_32f" => native::ffi::volk_32f_log2_32f_get_func_desc,
            "volk_32f_s32f_multiply_32f" => native::ffi::volk_32f_s32f_multiply_32f_get_func_desc,
            "volk_32f_s32f_add_32f" => native::ffi::volk_32f_s32f_add_32f_get_func_desc,
            "volk_32f_x2_dot_prod_32f" => ffi::volk_32f_x2_dot_prod_32f_get_func_desc,
            "volk_32f_x2_add_32f" => ffi::volk_32f_x2_add_32f_get_func_desc,
            "volk_32f_x2_subtract_32f" => ffi::volk_32f_x2_subtract_32f_get_func_desc,
            "volk_32fc_deinterleave_32f_x2" => ffi::volk_32fc_deinterleave_32f_x2_get_func_desc,
            "volk_32f_x2_interleave_32fc" => ffi::volk_32f_x2_interleave_32fc_get_func_desc,
            "volk_32fc_x2_dot_prod_32fc" => ffi::volk_32fc_x2_dot_prod_32fc_get_func_desc,
            _ => return None,
        };
        // SAFETY: the descriptor functions take no arguments; the descriptor points to static
        // tables of `n_impls` entries each, with NUL-terminated names.
        let list = unsafe {
            let desc = get();
            (0..desc.n_impls)
                .map(|i| Implementation {
                    name: CStr::from_ptr(*desc.impl_names.add(i))
                        .to_string_lossy()
                        .into_owned(),
                    deps: *desc.impl_deps.add(i),
                    aligned: *desc.impl_alignment.add(i),
                })
                .collect()
        };
        Some(list)
    }
    #[cfg(no_volk)]
    {
        let _ = name;
        None
    }
}

/// The implementation VOLK's dispatcher chooses for aligned buffers without a `volk_config`
/// entry: the aligned one with the largest dependency mask, else the unaligned one with the
/// largest (the first of equals wins, as in `volk_rank_archs`).
#[must_use]
pub fn default_implementation(list: &[Implementation]) -> Option<&Implementation> {
    let best = |aligned: bool| {
        list.iter()
            .filter(|i| i.aligned == aligned)
            .fold(None::<&Implementation>, |best, i| match best {
                Some(b) if b.deps >= i.deps => Some(b),
                _ => Some(i),
            })
    };
    best(true).or_else(|| best(false))
}
