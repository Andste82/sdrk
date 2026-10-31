//! The C libraries the SDRk benchmarks compare with, each behind a small owning wrapper:
//!
//! - [`Fftw`]: FFTW 3 in single precision (GPL-2.0-or-later), a forward transform in place or
//!   out of place, planned with `FFTW_ESTIMATE`, so that the plan does not depend on timing.
//! - [`FftwR2c`]: FFTW's real-input forward transform (`N` real samples to `N / 2 + 1` bins),
//!   out of place, planned in the same way.
//! - [`KissFft16`]: KissFFT built with `FIXED_POINT=16` (BSD-3-Clause), out of place; every
//!   stage divides by its radix, so the result is the DFT divided by N.
//! - [`Ne10Int16`]: the int16 NEON transform of Ne10 (BSD-3-Clause), out of place, scaled by
//!   1/N (AArch64 only).
//! - [`GrAgcCc`]: the `agc_cc` loop of GNU Radio (GPL-3.0-or-later), the reference of the `C32`
//!   `agc`.
//! - [`GrPwrSquelchCc`]: the `pwr_squelch_cc` loop of GNU Radio (GPL-3.0-or-later), the
//!   reference of the `C32` `squelch`.
//! - [`KfrBiquad`]: the biquad cascade of KFR 7 (`kfr::iir`, GPL-2.0-or-later), the SIMD
//!   reference of `biquad` and `biquad_real`.
//! - [`CmsisBiquad`]: the biquad cascades of CMSIS-DSP in generic C (Apache-2.0), the scalar
//!   reference of the biquad fallbacks.
//! - [`cmsis_dsp`]: the scalar references of the fallback round from CMSIS-DSP in generic C
//!   (conversions, dot products, scaling and products, magnitudes, logarithm, sine, cosine and
//!   arc tangent, FIR filters, decimators and interpolators, complex and real FFTs), one library
//!   per SIMD level like the biquads.
//! - [`gnuradio`]: the GNU Radio references of the fallback round ([`GrRotator`],
//!   [`GrFrequencyModulatorFc`], [`GrQuadratureDemodCf`], [`GrFmDeemph`]).
//! - [`liquid`]: liquid-dsp's DC blocker ([`LiquidDcBlocker`], [`LiquidDcBlockerCrcf`], MIT).
//! - [`LiquidResamp2Crcf`], [`LiquidResamp2Rrrf`], [`LiquidFirhilbf`]: liquid-dsp's halfband
//!   decimators and Hilbert transform decimator with its SIMD dot products (MIT);
//!   [`AirspyFloat`], [`AirspyInt16`]: libairspy 1.0.10's real-to-complex converters (MIT).
//!
//! `benches/refs/build.sh` builds the static libraries; `build.rs` links the ones it finds. A
//! constructor returns `None` when its library is not linked. All of them use the interleaved
//! complex layout of `repr(C)` pairs.

mod biquad;
pub mod cmsis_dsp;
pub mod gnuradio;
mod halfband;
pub mod liquid;

pub use biquad::{BIQUAD_REF_LEVEL, CmsisBiquad, KfrBiquad};
pub use gnuradio::{GrFmDeemph, GrFrequencyModulatorFc, GrQuadratureDemodCf, GrRotator};
pub use halfband::{
    AirspyFloat, AirspyInt16, LiquidFirhilbf, LiquidResamp2Crcf, LiquidResamp2Rrrf,
};
pub use liquid::{LiquidDcBlocker, LiquidDcBlockerCrcf};

/// FFTW's single-precision forward transform, planned for one pair of buffers: in place
/// (input and output are the same buffer) or out of place.
pub struct Fftw {
    #[cfg(has_fftw)]
    plan: *mut core::ffi::c_void,
    len: usize,
    in_place: bool,
}

#[cfg(has_fftw)]
mod fftw_ffi {
    use core::ffi::{c_int, c_uint, c_void};

    pub const FFTW_FORWARD: c_int = -1;
    pub const FFTW_ESTIMATE: c_uint = 1 << 6;

    unsafe extern "C" {
        pub fn fftwf_plan_dft_1d(
            n: c_int,
            input: *mut [f32; 2],
            output: *mut [f32; 2],
            sign: c_int,
            flags: c_uint,
        ) -> *mut c_void;
        pub fn fftwf_execute_dft(plan: *mut c_void, input: *mut [f32; 2], output: *mut [f32; 2]);
        pub fn fftwf_plan_dft_r2c_1d(
            n: c_int,
            input: *mut f32,
            output: *mut [f32; 2],
            flags: c_uint,
        ) -> *mut c_void;
        pub fn fftwf_execute_dft_r2c(plan: *mut c_void, input: *mut f32, output: *mut [f32; 2]);
        pub fn fftwf_destroy_plan(plan: *mut c_void);
    }
}

impl Fftw {
    /// Plans the forward transform of `data` in place; `None` without FFTW or if planning
    /// fails. FFTW chooses its SIMD codelets from the alignment of the planned buffers, so
    /// every later call must pass buffers with the same alignment (see [`Fftw::forward`]).
    pub fn in_place(data: &mut [[f32; 2]]) -> Option<Self> {
        let p = data.as_mut_ptr();
        Self::plan(data.len(), p, p)
    }

    /// Plans the forward transform from `input` to `output` (as GNU Radio's `fft_vcc` uses
    /// FFTW); `None` without FFTW, if planning fails or if the lengths differ. The alignment
    /// rule of [`Fftw::in_place`] applies to both buffers. For a complex transform out of place
    /// FFTW preserves the input.
    pub fn out_of_place(input: &mut [[f32; 2]], output: &mut [[f32; 2]]) -> Option<Self> {
        if input.len() != output.len() {
            return None;
        }
        Self::plan(input.len(), input.as_mut_ptr(), output.as_mut_ptr())
    }

    /// The plan of `len` points from `input` to `output`, which point to `len` complex values
    /// each (or are the same pointer).
    #[cfg(has_fftw)]
    fn plan(len: usize, input: *mut [f32; 2], output: *mut [f32; 2]) -> Option<Self> {
        let n = core::ffi::c_int::try_from(len).ok()?;
        // SAFETY: both pointers come from live slices of `len` complex values (see the
        // callers); with `FFTW_ESTIMATE` the planner does not write to the arrays.
        let plan = unsafe {
            fftw_ffi::fftwf_plan_dft_1d(
                n,
                input,
                output,
                fftw_ffi::FFTW_FORWARD,
                fftw_ffi::FFTW_ESTIMATE,
            )
        };
        (!plan.is_null()).then_some(Fftw {
            plan,
            len,
            in_place: input == output,
        })
    }

    /// Without FFTW: `None`.
    #[cfg(not(has_fftw))]
    fn plan(len: usize, input: *mut [f32; 2], output: *mut [f32; 2]) -> Option<Self> {
        let _ = (len, input, output);
        None
    }

    /// The unnormalised forward DFT of `data` in place.
    ///
    /// The alignment is not checked, because a check would enter the instruction counts of the
    /// comparison.
    ///
    /// # Safety
    /// `data` must have the alignment of the buffer the plan was made for (modulo 64 bytes, so
    /// that FFTW's `fftwf_alignment_of` agrees); FFTW's new-array execute is undefined
    /// otherwise.
    ///
    /// # Panics
    /// If the plan is out of place or `data` does not have the planned length.
    pub unsafe fn forward(&mut self, data: &mut [[f32; 2]]) {
        assert!(self.in_place, "FFTW: the plan is out of place");
        assert_eq!(data.len(), self.len, "FFTW: length differs from the plan");
        #[cfg(has_fftw)]
        // SAFETY: the plan is valid until `drop` and was made for one buffer of this length;
        // `data` has the alignment of the planned buffer (the caller's contract).
        unsafe {
            fftw_ffi::fftwf_execute_dft(self.plan, data.as_mut_ptr(), data.as_mut_ptr());
        }
    }

    /// The unnormalised forward DFT of `input` into `output`. The alignment is not checked, as
    /// in [`Fftw::forward`].
    ///
    /// # Safety
    /// `input` and `output` must have the alignments of the buffers the plan was made for
    /// (modulo 64 bytes); FFTW's new-array execute is undefined otherwise.
    ///
    /// # Panics
    /// If the plan is in place or a buffer does not have the planned length.
    pub unsafe fn forward_to(&mut self, input: &mut [[f32; 2]], output: &mut [[f32; 2]]) {
        assert!(!self.in_place, "FFTW: the plan is in place");
        assert_eq!(input.len(), self.len, "FFTW: length differs from the plan");
        assert_eq!(output.len(), self.len, "FFTW: length differs from the plan");
        #[cfg(has_fftw)]
        // SAFETY: the plan is valid until `drop` and was made for two distinct buffers of this
        // length (the borrows here cannot alias either); both have the alignments of the
        // planned buffers (the caller's contract).
        unsafe {
            fftw_ffi::fftwf_execute_dft(self.plan, input.as_mut_ptr(), output.as_mut_ptr());
        }
    }
}

#[cfg(has_fftw)]
impl Drop for Fftw {
    fn drop(&mut self) {
        // SAFETY: the plan came from `fftwf_plan_dft_1d` and is destroyed once.
        unsafe { fftw_ffi::fftwf_destroy_plan(self.plan) }
    }
}

/// FFTW's single-precision real-input forward transform, planned for one pair of buffers, out
/// of place (FFTW's default for a real-input transform preserves the input).
pub struct FftwR2c {
    #[cfg(has_fftw)]
    plan: *mut core::ffi::c_void,
    len: usize,
}

impl FftwR2c {
    /// Plans the transform of the `N` real samples of `input` into the `N / 2 + 1` bins of
    /// `output`; `None` without FFTW, if planning fails or if the lengths do not match. The
    /// alignment rule of [`Fftw::in_place`] applies to both buffers.
    #[cfg(has_fftw)]
    pub fn new(input: &mut [f32], output: &mut [[f32; 2]]) -> Option<Self> {
        let len = input.len();
        if output.len() != len / 2 + 1 {
            return None;
        }
        let n = core::ffi::c_int::try_from(len).ok()?;
        // SAFETY: both pointers come from live slices of the planned lengths; with
        // `FFTW_ESTIMATE` the planner does not write to the arrays.
        let plan = unsafe {
            fftw_ffi::fftwf_plan_dft_r2c_1d(
                n,
                input.as_mut_ptr(),
                output.as_mut_ptr(),
                fftw_ffi::FFTW_ESTIMATE,
            )
        };
        (!plan.is_null()).then_some(FftwR2c { plan, len })
    }

    /// Without FFTW: `None`.
    #[cfg(not(has_fftw))]
    pub fn new(input: &mut [f32], output: &mut [[f32; 2]]) -> Option<Self> {
        let _ = (input, output);
        None
    }

    /// The unnormalised forward DFT of the real `input` into the `N / 2 + 1` bins of `output`.
    /// The alignment is not checked, as in [`Fftw::forward`].
    ///
    /// # Safety
    /// `input` and `output` must have the alignments of the buffers the plan was made for
    /// (modulo 64 bytes); FFTW's new-array execute is undefined otherwise.
    ///
    /// # Panics
    /// If a buffer does not have the planned length.
    pub unsafe fn forward(&mut self, input: &mut [f32], output: &mut [[f32; 2]]) {
        assert_eq!(input.len(), self.len, "FFTW: length differs from the plan");
        assert_eq!(
            output.len(),
            self.len / 2 + 1,
            "FFTW: length differs from the plan"
        );
        #[cfg(has_fftw)]
        // SAFETY: the plan is valid until `drop` and was made for two distinct buffers of these
        // lengths (the borrows here cannot alias either); both have the alignments of the
        // planned buffers (the caller's contract).
        unsafe {
            fftw_ffi::fftwf_execute_dft_r2c(self.plan, input.as_mut_ptr(), output.as_mut_ptr());
        }
    }
}

#[cfg(has_fftw)]
impl Drop for FftwR2c {
    fn drop(&mut self) {
        // SAFETY: the plan came from `fftwf_plan_dft_r2c_1d` and is destroyed once.
        unsafe { fftw_ffi::fftwf_destroy_plan(self.plan) }
    }
}

/// KissFFT with 16-bit fixed point, forward, out of place.
pub struct KissFft16 {
    #[cfg(has_kissfft)]
    cfg: *mut core::ffi::c_void,
    len: usize,
}

#[cfg(has_kissfft)]
mod kiss_ffi {
    use core::ffi::{c_int, c_void};

    unsafe extern "C" {
        pub fn kiss_fft_alloc(
            nfft: c_int,
            inverse_fft: c_int,
            mem: *mut c_void,
            lenmem: *mut usize,
        ) -> *mut c_void;
        pub fn kiss_fft(cfg: *mut c_void, fin: *const [i16; 2], fout: *mut [i16; 2]);
        // `kiss_fft_free` is a macro for `free` when KissFFT allocates the configuration.
        pub fn free(ptr: *mut c_void);
    }
}

impl KissFft16 {
    /// The configuration of a forward transform of `len` points; `None` without KissFFT.
    #[cfg(has_kissfft)]
    pub fn new(len: usize) -> Option<Self> {
        let n = core::ffi::c_int::try_from(len).ok()?;
        // SAFETY: with a null `mem` KissFFT allocates the configuration itself.
        let cfg =
            unsafe { kiss_ffi::kiss_fft_alloc(n, 0, core::ptr::null_mut(), core::ptr::null_mut()) };
        (!cfg.is_null()).then_some(KissFft16 { cfg, len })
    }

    /// Without KissFFT: `None`.
    #[cfg(not(has_kissfft))]
    pub fn new(len: usize) -> Option<Self> {
        let _ = len;
        None
    }

    /// `output = DFT(input) / N` (each radix stage divides with rounding).
    ///
    /// # Panics
    /// If a buffer does not have the configured length.
    pub fn forward(&mut self, input: &[[i16; 2]], output: &mut [[i16; 2]]) {
        assert!(
            input.len() == self.len && output.len() == self.len,
            "KissFFT: length differs from the configuration"
        );
        #[cfg(has_kissfft)]
        // SAFETY: the configuration is valid until `drop`; both buffers hold `len` values and
        // do not overlap (one is borrowed shared, the other exclusively).
        unsafe {
            kiss_ffi::kiss_fft(self.cfg, input.as_ptr(), output.as_mut_ptr());
        }
    }
}

#[cfg(has_kissfft)]
impl Drop for KissFft16 {
    fn drop(&mut self) {
        // SAFETY: KissFFT allocated the configuration with `malloc`; it is freed once.
        unsafe { kiss_ffi::free(self.cfg) }
    }
}

/// Ne10's int16 NEON transform, forward, out of place, scaled by 1/N.
pub struct Ne10Int16 {
    #[cfg(has_ne10)]
    cfg: *mut core::ffi::c_void,
    len: usize,
}

#[cfg(has_ne10)]
mod ne10_ffi {
    use core::ffi::c_void;

    unsafe extern "C" {
        pub fn ne10_fft_alloc_c2c_int16(nfft: i32) -> *mut c_void;
        pub fn ne10_fft_c2c_1d_int16_neon(
            fout: *mut [i16; 2],
            fin: *mut [i16; 2],
            cfg: *mut c_void,
            inverse_fft: i32,
            scaled_flag: i32,
        );
        pub fn ne10_fft_destroy_c2c_int16(cfg: *mut c_void);
    }
}

impl Ne10Int16 {
    /// The configuration of a transform of `len` points; `None` without Ne10.
    #[cfg(has_ne10)]
    pub fn new(len: usize) -> Option<Self> {
        let n = i32::try_from(len).ok()?;
        // SAFETY: plain allocation of a configuration for `n` points.
        let cfg = unsafe { ne10_ffi::ne10_fft_alloc_c2c_int16(n) };
        (!cfg.is_null()).then_some(Ne10Int16 { cfg, len })
    }

    /// Without Ne10: `None`.
    #[cfg(not(has_ne10))]
    pub fn new(len: usize) -> Option<Self> {
        let _ = len;
        None
    }

    /// `output = DFT(input) / N`. `input` may be overwritten: Ne10 does not document that it
    /// preserves it.
    ///
    /// # Panics
    /// If a buffer does not have the configured length.
    pub fn forward(&mut self, input: &mut [[i16; 2]], output: &mut [[i16; 2]]) {
        assert!(
            input.len() == self.len && output.len() == self.len,
            "Ne10: length differs from the configuration"
        );
        #[cfg(has_ne10)]
        // SAFETY: the configuration is valid until `drop`; both buffers hold `len` values and
        // do not overlap (two distinct mutable borrows). Ne10 takes the input as mutable and may
        // write to it, which the exclusive borrow of `input` allows.
        unsafe {
            ne10_ffi::ne10_fft_c2c_1d_int16_neon(
                output.as_mut_ptr(),
                input.as_mut_ptr(),
                self.cfg,
                0,
                1,
            );
        }
    }
}

#[cfg(has_ne10)]
impl Drop for Ne10Int16 {
    fn drop(&mut self) {
        // SAFETY: the configuration came from `ne10_fft_alloc_c2c_int16` and is freed once.
        unsafe { ne10_ffi::ne10_fft_destroy_c2c_int16(self.cfg) }
    }
}

/// The `agc_cc` loop of GNU Radio (`benches/refs/gr-agc.c`): four floats (rate, reference,
/// gain, maximum gain) and the interleaved complex layout of `repr(C)` pairs.
pub struct GrAgcCc {
    state: [f32; 4],
}

#[cfg(has_gragc)]
mod gragc_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_agc_cc_scale_n(
            state: *mut [f32; 4],
            output: *mut [f32; 2],
            input: *const [f32; 2],
            n: core::ffi::c_uint,
        );
    }
}

impl GrAgcCc {
    /// The object `agc_cc(rate, reference, gain, max_gain)`; `None` without the library.
    #[must_use]
    pub fn new(rate: f32, reference: f32, gain: f32, max_gain: f32) -> Option<Self> {
        cfg!(has_gragc).then_some(GrAgcCc {
            state: [rate, reference, gain, max_gain],
        })
    }

    /// `scaleN`: scales `input` into `output`.
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than `u32::MAX` samples.
    pub fn scale_n(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) {
        assert_eq!(input.len(), output.len(), "agc_cc: length mismatch");
        let n = u32::try_from(input.len()).expect("agc_cc: too many samples");
        #[cfg(has_gragc)]
        // SAFETY: both buffers hold `n` pairs and do not overlap (an exclusive borrow of the
        // output); the state is the four floats of the C object.
        unsafe {
            gragc_ffi::sdrk_gr_agc_cc_scale_n(
                &raw mut self.state,
                output.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }
        #[cfg(not(has_gragc))]
        let _ = n;
    }
}

/// The `pwr_squelch_cc` block of GNU Radio (`benches/refs/gr-squelch.c`): the object of the C
/// translation, which holds the table of its virtual functions, in storage of this wrapper.
pub struct GrPwrSquelchCc {
    object: Box<[u64; GR_PWR_SQUELCH_WORDS]>,
}

/// Words of storage for the C object (it needs 80 bytes on the 64-bit targets).
const GR_PWR_SQUELCH_WORDS: usize = 16;

#[cfg(has_grsquelch)]
mod grsquelch_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_pwr_squelch_cc_size() -> usize;
        pub fn sdrk_gr_pwr_squelch_cc_init(
            object: *mut u64,
            db: f64,
            alpha: f64,
            ramp: core::ffi::c_int,
            gate: bool,
        );
        pub fn sdrk_gr_pwr_squelch_cc_work(
            object: *mut u64,
            noutput_items: core::ffi::c_int,
            input: *const [f32; 2],
            output: *mut [f32; 2],
        ) -> core::ffi::c_int;
    }
}

impl GrPwrSquelchCc {
    /// The object `pwr_squelch_cc(db, alpha, ramp, gate)`; `None` without the library.
    ///
    /// # Panics
    /// If the C object does not fit the storage of the wrapper.
    #[must_use]
    pub fn new(db: f64, alpha: f64, ramp: i32, gate: bool) -> Option<Self> {
        if !cfg!(has_grsquelch) {
            return None;
        }
        let mut object = Box::new([0u64; GR_PWR_SQUELCH_WORDS]);
        #[cfg(has_grsquelch)]
        // SAFETY: the storage is checked to hold the object, and it is 8-byte aligned.
        unsafe {
            assert!(
                grsquelch_ffi::sdrk_gr_pwr_squelch_cc_size() <= 8 * GR_PWR_SQUELCH_WORDS,
                "pwr_squelch_cc: the object is too large"
            );
            grsquelch_ffi::sdrk_gr_pwr_squelch_cc_init(object.as_mut_ptr(), db, alpha, ramp, gate);
        }
        #[cfg(not(has_grsquelch))]
        let _ = (db, alpha, ramp, gate, &mut object);
        Some(GrPwrSquelchCc { object })
    }

    /// `general_work` over `input`; returns the number of outputs written (all of them while
    /// the block does not gate).
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than `i32::MAX` samples.
    pub fn work(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) -> usize {
        assert_eq!(input.len(), output.len(), "pwr_squelch_cc: length mismatch");
        let n = core::ffi::c_int::try_from(input.len()).expect("pwr_squelch_cc: too many samples");
        #[cfg(has_grsquelch)]
        // SAFETY: both buffers hold `n` pairs and do not overlap (an exclusive borrow of the
        // output); the object was initialized by `new`.
        let written = unsafe {
            grsquelch_ffi::sdrk_gr_pwr_squelch_cc_work(
                self.object.as_mut_ptr(),
                n,
                input.as_ptr(),
                output.as_mut_ptr(),
            )
        };
        #[cfg(not(has_grsquelch))]
        let written = {
            let _ = &mut self.object;
            n
        };
        usize::try_from(written).unwrap_or(0)
    }
}
