//! Choice of the SIMD level, and the rules for code written for one level.
//!
//! # Level-specific code
//!
//! Most kernels are written once against the `Simd` trait of `fearless_simd`. Where the portable
//! operations are not enough, a kernel may add a path for one level that calls the intrinsics of
//! `core::arch` (decision D1 of the SIMD optimization):
//!
//! - The path is a function declared with `fearless_simd::kernel!`, whose first argument is the
//!   token of its level (`Sse2`, `Sse4_2`, `Avx2`, `Avx512`, `Neon` or `WasmSimd128`). The macro
//!   compiles the body with the target features of that level, so the value intrinsics are safe
//!   to call, both when the level is fixed at compile time and when it is detected at run time
//!   (the feature `std`). Helpers called from the body are `#[inline(always)]`, otherwise they
//!   are compiled without those features. The exceptions are the large helpers of the `C32`
//!   FFT and the stage loop of the `CQ15` FFT, which are `#[inline(always)]` only without debug
//!   assertions, so that unoptimised builds keep small stack frames (see `fft::c32`); with debug
//!   assertions they are correct but slower.
//! - The generic kernel picks the path from the token it was dispatched with:
//!   `simd.level().as_avx512()`, `as_avx2()`, `as_sse4_2()`, [`sse2_only`] on x86,
//!   `as_neon()` on AArch64 and `as_wasm_simd128()` on wasm32 with `simd128`, each under the
//!   `#[cfg]` of its target. The accessors of a weaker level also answer for a stronger one
//!   (`as_avx2()` is `Some` at AVX-512), so the strongest level is asked first. Every other
//!   level runs the generic code, which stays as the fallback of every level-specific path.
//! - The paths of one kernel live in a module per architecture in the kernel's file. `unsafe`
//!   is allowed there only for small helpers that the safe intrinsics cannot express (pointer
//!   loads and stores, interleaving loads and stores such as `ld2`/`st2`, masked loads,
//!   gathers), with `#[allow(unsafe_code)]` on that module and a `// SAFETY:` comment on every
//!   block that names the bounds that make it sound.
//! - Every path has a test that compares it with the generic code (bit-exact, or within the
//!   tolerance of the kernel's contract) over all lengths and alignments it handles. CI runs the
//!   tests of this crate at every level: natively with the level fixed at compile time (SSE2,
//!   x86-64-v2, x86-64-v3) and detected at run time, under Intel SDE (AVX-512 fixed and
//!   detected, and run-time detection on the CPUIDs of SSE2 and SSE4.2), under QEMU (NEON) and
//!   under wasmtime (with and without `simd128`, with relaxed SIMD and with the static tables);
//!   see `docs/testing.md`.
//!
//! The test at the end of this file is the smallest example of the pattern.

use fearless_simd::Level;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::Sse2;

/// The level the kernels run at: detected at run time with the feature `std`, fixed at compile
/// time otherwise.
#[inline(always)]
pub(crate) fn current() -> Level {
    #[cfg(feature = "std")]
    {
        Level::new()
    }
    #[cfg(not(feature = "std"))]
    {
        Level::baseline()
    }
}

/// Runs `$body` with `$simd` bound to the token of the current level.
macro_rules! dispatch {
    ($simd:ident => $body:expr) => {
        fearless_simd::dispatch!($crate::level::current(), $simd => $body)
    };
}
pub(crate) use dispatch;

/// Defines the function `$name($arg: $ty, ...)` that runs `$kernel(simd, $arg, ...)` with `simd`
/// the token of the current level, like [`dispatch!`] but with a smaller fixed cost per call
/// where the level is detected at run time (x86 with the feature `std`).
///
/// `dispatch!` hands the kernel to the function of the level that carries its target features
/// as a closure that captures the arguments by reference, so the caller stores them and pointers
/// to them on the stack and the callee loads them back. The code of the level that the target
/// has anyway (SSE2) is also inlined into the caller, which then saves and restores the registers
/// that this code needs on every call at every level. Here the levels AVX-512, AVX2 and SSE4.2
/// get one `fearless_simd::kernel!` function each, which takes the arguments in registers, and
/// the levels below run in one function out of line. AVX2, the most common level, is tested
/// first with one byte load of [`avx2_cache`] and one compare. For `fft` `C32` at N = 64 on
/// AVX2 detected at run time the fixed cost of the dispatch falls from about 40 to 19
/// instructions per call: what remains is the compare, the call and the stack frame of the
/// callee (`docs/performance.md` gives the current counts against AVX2 fixed at compile time). With the
/// level fixed at compile time, and on the other targets (which detect nothing at run time), the
/// function is `dispatch!`, which then compiles to the kernel itself.
///
/// `$kernel` is a function generic over the token, such as `entry::<_, 64>`, that returns `()`.
macro_rules! dispatch_fn {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) => $kernel:expr
    ) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) {
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                fearless_simd::kernel!(
                    #[inline(always)]
                    fn avx512(t: Avx512, $($arg: $ty),*) {
                        $kernel(t, $($arg),*)
                    }
                );
                fearless_simd::kernel!(
                    #[inline(always)]
                    fn avx2(t: Avx2, $($arg: $ty),*) {
                        $kernel(t, $($arg),*)
                    }
                );
                fearless_simd::kernel!(
                    #[inline(always)]
                    fn sse4_2(t: Sse4_2, $($arg: $ty),*) {
                        $kernel(t, $($arg),*)
                    }
                );
                /// The levels below SSE4.2: SSE2, or the baseline of a target without SSE2 (the
                /// detection gives no other level there).
                #[inline(never)]
                fn below_sse4_2(level: fearless_simd::Level, $($arg: $ty),*) {
                    if let Some(t) = level.as_sse2() {
                        fearless_simd::Simd::vectorize(
                            t,
                            #[inline(always)]
                            || $kernel(t, $($arg),*),
                        )
                    } else {
                        fearless_simd::dispatch!(
                            fearless_simd::Level::baseline(),
                            simd => $kernel(simd, $($arg),*)
                        )
                    }
                }
                // AVX2 is tested on its own, and the other levels on a second read of the
                // level: from one read LLVM merges the tests into a jump table, which costs
                // AVX2 three instructions more per call.
                if let fearless_simd::Level::Avx2(t) = $crate::level::current() {
                    avx2(t, $($arg),*)
                } else {
                    match $crate::level::current() {
                        fearless_simd::Level::Avx512(t) => avx512(t, $($arg),*),
                        fearless_simd::Level::Sse4_2(t) => sse4_2(t, $($arg),*),
                        level => below_sse4_2(level, $($arg),*),
                    }
                }
            }
            #[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
            $crate::level::dispatch!(simd => $kernel(simd, $($arg),*))
        }
    };
}
pub(crate) use dispatch_fn;

/// The SSE2 token if the level is the plain SSE2 baseline of x86 and nothing better (unlike
/// `Level::as_sse2`, which also answers at SSE4.2 and above).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn sse2_only(level: Level) -> Option<Sse2> {
    if level.as_sse4_2().is_some() {
        None
    } else {
        level.as_sse2()
    }
}

/// Whether the token is the plain SSE2 baseline of x86, which has no sign-extending widening
/// (`pmovsx`, SSE4.1): the kernels that widen signed integers use shifts and interleaves instead.
#[inline(always)]
pub(crate) fn is_sse2_baseline<S: fearless_simd::Simd>(simd: S) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        sse2_only(simd.level()).is_some()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = simd;
        false
    }
}

/// Name of the current level.
pub(crate) fn name() -> &'static str {
    let level = current();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if level.as_avx512().is_some() {
            return "avx512";
        }
        if level.as_avx2().is_some() {
            return "avx2";
        }
        if level.as_sse4_2().is_some() {
            return "sse4.2";
        }
        if level.as_sse2().is_some() {
            return "sse2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if level.as_neon().is_some() {
            return "neon";
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if level.as_wasm_simd128().is_some() {
            return "simd128";
        }
    }
    let _ = level;
    "scalar"
}

#[cfg(test)]
mod tests {
    use fearless_simd::Simd as SimdToken;

    use super::name;

    /// The level-specific paths of the example kernel `x + y` on `f32` slices.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    mod x86 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm_add_ps, _mm256_add_ps, _mm512_add_ps};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm_add_ps, _mm256_add_ps, _mm512_add_ps};
        use fearless_simd::prelude::*;
        use fearless_simd::{f32x4, f32x8, f32x16};

        fearless_simd::kernel!(
            pub(super) fn add_avx512(t: Avx512, x: &mut [f32], y: &[f32]) -> usize {
                for (a, b) in x.chunks_exact_mut(16).zip(y.chunks_exact(16)) {
                    let (va, vb) = (f32x16::from_slice(t, a), f32x16::from_slice(t, b));
                    f32x16::simd_from(t, _mm512_add_ps(va.into(), vb.into())).store_slice(a);
                }
                x.len() / 16 * 16
            }
        );

        fearless_simd::kernel!(
            pub(super) fn add_avx2(t: Avx2, x: &mut [f32], y: &[f32]) -> usize {
                for (a, b) in x.chunks_exact_mut(8).zip(y.chunks_exact(8)) {
                    let (va, vb) = (f32x8::from_slice(t, a), f32x8::from_slice(t, b));
                    f32x8::simd_from(t, _mm256_add_ps(va.into(), vb.into())).store_slice(a);
                }
                x.len() / 8 * 8
            }
        );

        fearless_simd::kernel!(
            pub(super) fn add_sse4_2(t: Sse4_2, x: &mut [f32], y: &[f32]) -> usize {
                for (a, b) in x.chunks_exact_mut(4).zip(y.chunks_exact(4)) {
                    let (va, vb) = (f32x4::from_slice(t, a), f32x4::from_slice(t, b));
                    f32x4::simd_from(t, _mm_add_ps(va.into(), vb.into())).store_slice(a);
                }
                x.len() / 4 * 4
            }
        );

        fearless_simd::kernel!(
            pub(super) fn add_sse2(t: Sse2, x: &mut [f32], y: &[f32]) -> usize {
                for (a, b) in x.chunks_exact_mut(4).zip(y.chunks_exact(4)) {
                    let (va, vb) = (f32x4::from_slice(t, a), f32x4::from_slice(t, b));
                    f32x4::simd_from(t, _mm_add_ps(va.into(), vb.into())).store_slice(a);
                }
                x.len() / 4 * 4
            }
        );
    }

    #[cfg(target_arch = "aarch64")]
    mod neon {
        use core::arch::aarch64::vaddq_f32;
        use fearless_simd::f32x4;
        use fearless_simd::prelude::*;

        fearless_simd::kernel!(
            pub(super) fn add_neon(t: Neon, x: &mut [f32], y: &[f32]) -> usize {
                for (a, b) in x.chunks_exact_mut(4).zip(y.chunks_exact(4)) {
                    let (va, vb) = (f32x4::from_slice(t, a), f32x4::from_slice(t, b));
                    f32x4::simd_from(t, vaddq_f32(va.into(), vb.into())).store_slice(a);
                }
                x.len() / 4 * 4
            }
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    mod wasm {
        use core::arch::wasm32::f32x4_add;
        use fearless_simd::f32x4;
        use fearless_simd::prelude::*;

        fearless_simd::kernel!(
            pub(super) fn add_simd128(t: WasmSimd128, x: &mut [f32], y: &[f32]) -> usize {
                for (a, b) in x.chunks_exact_mut(4).zip(y.chunks_exact(4)) {
                    let (va, vb) = (f32x4::from_slice(t, a), f32x4::from_slice(t, b));
                    f32x4::simd_from(t, f32x4_add(va.into(), vb.into())).store_slice(a);
                }
                x.len() / 4 * 4
            }
        );
    }

    /// The level-specific path of the current level, if there is one: its name and the number
    /// of leading elements it handled.
    #[inline(always)]
    fn add_level<S: SimdToken>(simd: S, x: &mut [f32], y: &[f32]) -> (&'static str, usize) {
        let level = simd.level();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(t) = level.as_avx512() {
                return ("avx512", x86::add_avx512(t, x, y));
            }
            if let Some(t) = level.as_avx2() {
                return ("avx2", x86::add_avx2(t, x, y));
            }
            if let Some(t) = level.as_sse4_2() {
                return ("sse4.2", x86::add_sse4_2(t, x, y));
            }
            if let Some(t) = super::sse2_only(level) {
                return ("sse2", x86::add_sse2(t, x, y));
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(t) = level.as_neon() {
            return ("neon", neon::add_neon(t, x, y));
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if let Some(t) = level.as_wasm_simd128() {
            return ("simd128", wasm::add_simd128(t, x, y));
        }
        let _ = (level, x, y);
        ("scalar", 0)
    }

    /// The generic kernel `x += y`: the path of the level takes the full vectors, the generic
    /// code the rest (all of it on a level without a path). Returns the name of the path.
    #[inline(always)]
    fn add<S: SimdToken>(simd: S, x: &mut [f32], y: &[f32]) -> &'static str {
        let (path, done) = add_level(simd, x, y);
        for (a, b) in x.iter_mut().zip(y).skip(done) {
            *a += b;
        }
        path
    }

    #[test]
    fn the_level_specific_path_of_the_current_level_runs_and_matches_the_generic_code() {
        for len in 0..70 {
            let mut x: [f32; 70] = core::array::from_fn(|i| i as f32 * 0.5);
            let y: [f32; 70] = core::array::from_fn(|i| 3.0 - i as f32);
            let mut expected = x;
            for (a, b) in expected.iter_mut().zip(&y).take(len) {
                *a += b;
            }
            let path = dispatch!(simd => add(simd, &mut x[..len], &y[..len]));
            assert_eq!(path, name(), "len {len}");
            assert_eq!(x, expected, "len {len}");
        }
    }

    dispatch_fn!(
        /// [`add`] through [`dispatch_fn!`], with the name of the path in `path`.
        fn add_fn(path: &mut &'static str, x: &mut [f32], y: &[f32]) => add_to
    );

    /// [`add`] with its result in `path`, in the form that [`dispatch_fn!`] takes.
    #[inline(always)]
    fn add_to<S: SimdToken>(simd: S, path: &mut &'static str, x: &mut [f32], y: &[f32]) {
        *path = add(simd, x, y);
    }

    #[test]
    fn dispatch_fn_runs_the_path_of_the_current_level() {
        for len in 0..70 {
            let mut x: [f32; 70] = core::array::from_fn(|i| i as f32 * 0.5);
            let y: [f32; 70] = core::array::from_fn(|i| 3.0 - i as f32);
            let mut expected = x;
            for (a, b) in expected.iter_mut().zip(&y).take(len) {
                *a += b;
            }
            let mut path = "";
            add_fn(&mut path, &mut x[..len], &y[..len]);
            assert_eq!(path, name(), "len {len}");
            assert_eq!(x, expected, "len {len}");
        }
    }
}
