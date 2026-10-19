//! Choice of the SIMD level.

use fearless_simd::Level;

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
