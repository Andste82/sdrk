//! The work loop of GNU Radio's `fastnoise_source_f` and `fastnoise_source_c`
//! (GPL-3.0-or-later; `benches/refs/gr-fastnoise.c`, cfg `has_grfastnoise`), the information
//! reference of `gauss_noise` (M48 §5) marked "reference does less work".
//!
//! Per output sample the block draws one xoroshiro128+ value and copies the entry of a pool of
//! Gaussian values that it selects (`d_samples[next & d_bitmask]`): there is no normal mapping
//! per sample, the output repeats the pool values, its tail ends at the largest value of the
//! pool, and there is no random access. The pool is filled once at construction, outside the
//! counted calls.

/// The pool size of GNU Radio Companion's `analog_fastnoise_source_x` block (`samples`, default
/// 8192; the C++ `make` defaults to 16384).
pub const GR_FASTNOISE_POOL: usize = 8192;

/// Words of storage for a C object (40 bytes on the 64-bit targets).
const GR_FASTNOISE_WORDS: usize = 8;

#[cfg(has_grfastnoise)]
mod ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_fastnoise_source_f_size() -> usize;
        pub fn sdrk_gr_fastnoise_source_f_init(
            object: *mut u64,
            ampl: f32,
            seed: u64,
            pool: *mut f32,
            samples: usize,
        );
        pub fn sdrk_gr_fastnoise_source_f_work(
            object: *mut u64,
            noutput_items: core::ffi::c_int,
            out: *mut f32,
        ) -> core::ffi::c_int;
        pub fn sdrk_gr_fastnoise_source_c_size() -> usize;
        pub fn sdrk_gr_fastnoise_source_c_init(
            object: *mut u64,
            ampl: f32,
            seed: u64,
            pool: *mut [f32; 2],
            samples: usize,
        );
        pub fn sdrk_gr_fastnoise_source_c_work(
            object: *mut u64,
            noutput_items: core::ffi::c_int,
            out: *mut [f32; 2],
        ) -> core::ffi::c_int;
    }
}

/// `fastnoise_source_f(GR_GAUSSIAN, ampl, seed, samples)`: the C object and its pool.
pub struct GrFastnoiseSourceF {
    object: Box<[u64; GR_FASTNOISE_WORDS]>,
    // Read by the C object through the pointer it keeps; never moved (a boxed slice).
    #[allow(
        dead_code,
        reason = "owned for the C object, which holds a pointer into it"
    )]
    pool: Box<[f32]>,
}

impl GrFastnoiseSourceF {
    /// The block with amplitude `ampl` (the standard deviation of a real value) and a pool of
    /// `samples` values; `None` without the library.
    ///
    /// # Panics
    /// If the C object does not fit the storage of the wrapper.
    #[must_use]
    pub fn new(ampl: f32, seed: u64, samples: usize) -> Option<Self> {
        if !cfg!(has_grfastnoise) {
            return None;
        }
        let mut object = Box::new([0u64; GR_FASTNOISE_WORDS]);
        let mut pool = vec![0.0f32; samples].into_boxed_slice();
        #[cfg(has_grfastnoise)]
        // SAFETY: the storage is checked to hold the object and is 8-byte aligned; the pool
        // holds `samples` values and lives (unmoved, on the heap) as long as the object.
        unsafe {
            assert!(
                ffi::sdrk_gr_fastnoise_source_f_size() <= 8 * GR_FASTNOISE_WORDS,
                "fastnoise_source_f: the object is too large"
            );
            ffi::sdrk_gr_fastnoise_source_f_init(
                object.as_mut_ptr(),
                ampl,
                seed,
                pool.as_mut_ptr(),
                samples,
            );
        }
        #[cfg(not(has_grfastnoise))]
        let _ = (ampl, seed, &mut object, &mut pool);
        Some(GrFastnoiseSourceF { object, pool })
    }

    /// `work` into `out`.
    ///
    /// # Panics
    /// If `out` holds more than `i32::MAX` samples.
    pub fn work(&mut self, out: &mut [f32]) {
        let n = core::ffi::c_int::try_from(out.len()).expect("fastnoise_source_f: too many");
        #[cfg(has_grfastnoise)]
        // SAFETY: `out` holds `n` values and is exclusively borrowed; the object was
        // initialized by `new` and its pool is owned by `self`.
        unsafe {
            ffi::sdrk_gr_fastnoise_source_f_work(self.object.as_mut_ptr(), n, out.as_mut_ptr());
        }
        #[cfg(not(has_grfastnoise))]
        let _ = (n, &mut self.object);
    }
}

/// `fastnoise_source_c(GR_GAUSSIAN, ampl, seed, samples)`: the C object and its pool. GNU Radio
/// divides `ampl` by `sqrt(2)`, so `ampl = sigma * sqrt(2)` gives `sigma` per component.
pub struct GrFastnoiseSourceC {
    object: Box<[u64; GR_FASTNOISE_WORDS]>,
    #[allow(
        dead_code,
        reason = "owned for the C object, which holds a pointer into it"
    )]
    pool: Box<[[f32; 2]]>,
}

impl GrFastnoiseSourceC {
    /// The block with amplitude `ampl` (the RMS of a complex value) and a pool of `samples`
    /// values; `None` without the library.
    ///
    /// # Panics
    /// If the C object does not fit the storage of the wrapper.
    #[must_use]
    pub fn new(ampl: f32, seed: u64, samples: usize) -> Option<Self> {
        if !cfg!(has_grfastnoise) {
            return None;
        }
        let mut object = Box::new([0u64; GR_FASTNOISE_WORDS]);
        let mut pool = vec![[0.0f32; 2]; samples].into_boxed_slice();
        #[cfg(has_grfastnoise)]
        // SAFETY: as for `GrFastnoiseSourceF::new`; `[f32; 2]` has the layout of
        // `float _Complex`.
        unsafe {
            assert!(
                ffi::sdrk_gr_fastnoise_source_c_size() <= 8 * GR_FASTNOISE_WORDS,
                "fastnoise_source_c: the object is too large"
            );
            ffi::sdrk_gr_fastnoise_source_c_init(
                object.as_mut_ptr(),
                ampl,
                seed,
                pool.as_mut_ptr(),
                samples,
            );
        }
        #[cfg(not(has_grfastnoise))]
        let _ = (ampl, seed, &mut object, &mut pool);
        Some(GrFastnoiseSourceC { object, pool })
    }

    /// `work` into `out`.
    ///
    /// # Panics
    /// If `out` holds more than `i32::MAX` samples.
    pub fn work(&mut self, out: &mut [[f32; 2]]) {
        let n = core::ffi::c_int::try_from(out.len()).expect("fastnoise_source_c: too many");
        #[cfg(has_grfastnoise)]
        // SAFETY: as for `GrFastnoiseSourceF::work`; `out` holds `n` pairs.
        unsafe {
            ffi::sdrk_gr_fastnoise_source_c_work(self.object.as_mut_ptr(), n, out.as_mut_ptr());
        }
        #[cfg(not(has_grfastnoise))]
        let _ = (n, &mut self.object);
    }
}
