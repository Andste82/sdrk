//! The biquad references: KFR's `iir` (`benches/refs/kfr-biquad.cpp`) and the cascades of
//! CMSIS-DSP in generic C (`benches/refs/cmsis-biquad.c`). Both are linked in the build of the
//! SIMD level of this binary (see `build.rs`); the coefficients are given in SDRk's order and
//! sign, `[b0, b1, b2, a1, a2]` per section with `y = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2`.

/// The SIMD level whose biquad reference libraries this build links (`sse2`, `sse4.2`, `avx2`,
/// `avx512`, `neon`, `simd128`): the level the target features select (see `build.rs`).
pub const BIQUAD_REF_LEVEL: &str = env!("SDRK_BIQUAD_REF_LEVEL");

#[cfg(has_kfr)]
mod kfr_ffi {
    use core::ffi::{c_char, c_void};

    unsafe extern "C" {
        pub fn sdrk_kfr_biquad_f32_new(coefs: *const [f32; 5], count: u32) -> *mut c_void;
        pub fn sdrk_kfr_biquad_f32_run(
            handle: *mut c_void,
            input: *const f32,
            output: *mut f32,
            n: usize,
        );
        pub fn sdrk_kfr_biquad_f32_free(handle: *mut c_void);
        pub fn sdrk_kfr_biquad_arch() -> *const c_char;
    }
}

/// KFR's biquad cascade (`kfr::iir` on an `iir_state<float, 2>`): one real channel, the two
/// sections in the lanes of one vector, transposed direct form II. Consecutive calls continue
/// the stream.
pub struct KfrBiquad {
    #[cfg(has_kfr)]
    handle: *mut core::ffi::c_void,
}

impl KfrBiquad {
    /// The cascade of `coefs`; `None` without KFR or unless there are exactly two sections.
    #[cfg(has_kfr)]
    #[must_use]
    pub fn new(coefs: &[[f32; 5]]) -> Option<Self> {
        let count = u32::try_from(coefs.len()).ok()?;
        // SAFETY: `coefs` holds `count` sections of five floats; the shim copies them.
        let handle = unsafe { kfr_ffi::sdrk_kfr_biquad_f32_new(coefs.as_ptr(), count) };
        (!handle.is_null()).then_some(KfrBiquad { handle })
    }

    /// Without KFR: `None`.
    #[cfg(not(has_kfr))]
    #[must_use]
    pub fn new(coefs: &[[f32; 5]]) -> Option<Self> {
        let _ = coefs;
        None
    }

    /// The architecture KFR was compiled for (`sse2`, `sse42`, `avx2`, `avx512`, `neon64`).
    #[must_use]
    pub fn arch() -> Option<&'static str> {
        #[cfg(has_kfr)]
        {
            // SAFETY: the shim returns a pointer to a NUL-terminated string literal.
            let s = unsafe { core::ffi::CStr::from_ptr(kfr_ffi::sdrk_kfr_biquad_arch()) };
            s.to_str().ok()
        }
        #[cfg(not(has_kfr))]
        None
    }

    /// Filters `input` into `output`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "KFR biquad: length mismatch");
        #[cfg(has_kfr)]
        // SAFETY: the handle is valid until `drop`; both buffers hold `len` floats and do not
        // overlap (an exclusive borrow of the output).
        unsafe {
            kfr_ffi::sdrk_kfr_biquad_f32_run(
                self.handle,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len(),
            );
        }
    }
}

#[cfg(has_kfr)]
impl Drop for KfrBiquad {
    fn drop(&mut self) {
        // SAFETY: the handle came from `sdrk_kfr_biquad_f32_new` and is freed once.
        unsafe { kfr_ffi::sdrk_kfr_biquad_f32_free(self.handle) }
    }
}

#[cfg(has_cmsis)]
mod cmsis_ffi {
    use core::ffi::c_void;

    unsafe extern "C" {
        pub fn sdrk_cmsis_df2t_f32_new(coefs: *const [f32; 5], stages: u32) -> *mut c_void;
        pub fn sdrk_cmsis_df2t_f32_run(
            handle: *mut c_void,
            input: *const f32,
            output: *mut f32,
            n: u32,
        );
        pub fn sdrk_cmsis_stereo_df2t_f32_new(coefs: *const [f32; 5], stages: u32) -> *mut c_void;
        pub fn sdrk_cmsis_stereo_df2t_f32_run(
            handle: *mut c_void,
            input: *const [f32; 2],
            output: *mut [f32; 2],
            n: u32,
        );
        pub fn sdrk_cmsis_df1_q15_new(coefs: *const [i16; 5], stages: u32) -> *mut c_void;
        pub fn sdrk_cmsis_df1_q15_run(
            handle: *mut c_void,
            input: *const i16,
            output: *mut i16,
            n: u32,
        );
        pub fn sdrk_cmsis_biquad_free(handle: *mut c_void);
    }
}

/// Which CMSIS-DSP cascade a [`CmsisBiquad`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CmsisKind {
    Df2tF32,
    StereoDf2tF32,
    Df1Q15,
}

/// A biquad cascade of CMSIS-DSP in generic C, one of `arm_biquad_cascade_df2T_f32` (one real
/// channel), `arm_biquad_cascade_stereo_df2T_f32` (two interleaved channels, the layout of a
/// C32 buffer) and `arm_biquad_cascade_df1_q15` with `postShift = 1` (Q2.14 coefficients, a
/// 64-bit accumulator truncated by 14 bits). Consecutive calls continue the stream.
pub struct CmsisBiquad {
    #[cfg(has_cmsis)]
    handle: *mut core::ffi::c_void,
    kind: CmsisKind,
}

impl CmsisBiquad {
    /// `arm_biquad_cascade_df2T_f32` for `coefs`; `None` without CMSIS-DSP or for more than 255
    /// sections.
    #[must_use]
    pub fn df2t_f32(coefs: &[[f32; 5]]) -> Option<Self> {
        Self::make(CmsisKind::Df2tF32, coefs.as_ptr().cast(), coefs.len())
    }

    /// `arm_biquad_cascade_stereo_df2T_f32` for `coefs`; `None` as for [`Self::df2t_f32`].
    #[must_use]
    pub fn stereo_df2t_f32(coefs: &[[f32; 5]]) -> Option<Self> {
        Self::make(CmsisKind::StereoDf2tF32, coefs.as_ptr().cast(), coefs.len())
    }

    /// `arm_biquad_cascade_df1_q15` with `postShift = 1` for the Q2.14 `coefs`; `None` as for
    /// [`Self::df2t_f32`], and for a feedback coefficient of -2.0 (raw -32768), which CMSIS-DSP
    /// cannot hold negated.
    #[must_use]
    pub fn df1_q15(coefs: &[[i16; 5]]) -> Option<Self> {
        Self::make(CmsisKind::Df1Q15, coefs.as_ptr().cast(), coefs.len())
    }

    #[cfg(has_cmsis)]
    fn make(kind: CmsisKind, coefs: *const core::ffi::c_void, len: usize) -> Option<Self> {
        let stages = u32::try_from(len).ok()?;
        // SAFETY: `coefs` points to `stages` sections of five values of the type that `kind`
        // takes (the public constructors pass such a slice); the shim copies them.
        let handle = unsafe {
            match kind {
                CmsisKind::Df2tF32 => cmsis_ffi::sdrk_cmsis_df2t_f32_new(coefs.cast(), stages),
                CmsisKind::StereoDf2tF32 => {
                    cmsis_ffi::sdrk_cmsis_stereo_df2t_f32_new(coefs.cast(), stages)
                }
                CmsisKind::Df1Q15 => cmsis_ffi::sdrk_cmsis_df1_q15_new(coefs.cast(), stages),
            }
        };
        (!handle.is_null()).then_some(CmsisBiquad { handle, kind })
    }

    #[cfg(not(has_cmsis))]
    fn make(kind: CmsisKind, coefs: *const core::ffi::c_void, len: usize) -> Option<Self> {
        let _ = (kind, coefs, len);
        None
    }

    /// Filters one real channel (`df2t_f32`).
    ///
    /// # Panics
    /// If the cascade is another one, or the lengths differ or exceed `u32::MAX`.
    pub fn run_f32(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(self.kind, CmsisKind::Df2tF32, "CMSIS biquad: not df2T_f32");
        let n = Self::len(input.len(), output.len());
        #[cfg(has_cmsis)]
        // SAFETY: a df2T_f32 handle, valid until `drop`; both buffers hold `n` floats and do
        // not overlap (an exclusive borrow of the output).
        unsafe {
            cmsis_ffi::sdrk_cmsis_df2t_f32_run(self.handle, input.as_ptr(), output.as_mut_ptr(), n);
        }
        #[cfg(not(has_cmsis))]
        let _ = n;
    }

    /// Filters two interleaved channels (`stereo_df2t_f32`), `input.len()` frames.
    ///
    /// # Panics
    /// If the cascade is another one, or the lengths differ or exceed `u32::MAX`.
    pub fn run_stereo(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) {
        assert_eq!(
            self.kind,
            CmsisKind::StereoDf2tF32,
            "CMSIS biquad: not stereo_df2T_f32"
        );
        let n = Self::len(input.len(), output.len());
        #[cfg(has_cmsis)]
        // SAFETY: a stereo handle, valid until `drop`; both buffers hold `n` frames of two
        // floats and do not overlap.
        unsafe {
            cmsis_ffi::sdrk_cmsis_stereo_df2t_f32_run(
                self.handle,
                input.as_ptr(),
                output.as_mut_ptr(),
                n,
            );
        }
        #[cfg(not(has_cmsis))]
        let _ = n;
    }

    /// Filters one Q15 channel (`df1_q15`).
    ///
    /// # Panics
    /// If the cascade is another one, or the lengths differ or exceed `u32::MAX`.
    pub fn run_q15(&mut self, input: &[i16], output: &mut [i16]) {
        assert_eq!(self.kind, CmsisKind::Df1Q15, "CMSIS biquad: not df1_q15");
        let n = Self::len(input.len(), output.len());
        #[cfg(has_cmsis)]
        // SAFETY: a df1_q15 handle, valid until `drop`; both buffers hold `n` values and do not
        // overlap.
        unsafe {
            cmsis_ffi::sdrk_cmsis_df1_q15_run(self.handle, input.as_ptr(), output.as_mut_ptr(), n);
        }
        #[cfg(not(has_cmsis))]
        let _ = n;
    }

    fn len(input: usize, output: usize) -> u32 {
        assert_eq!(input, output, "CMSIS biquad: length mismatch");
        u32::try_from(input).expect("CMSIS biquad: too many samples")
    }
}

#[cfg(has_cmsis)]
impl Drop for CmsisBiquad {
    fn drop(&mut self) {
        // SAFETY: the handle came from a constructor of cmsis-biquad.c and is freed once.
        unsafe { cmsis_ffi::sdrk_cmsis_biquad_free(self.handle) }
    }
}
