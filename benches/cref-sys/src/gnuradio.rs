//! The GNU Radio references of the fallback round (GPL-3.0-or-later, GNU Radio main at
//! ec1bf15, pinned C translations in `benches/refs`, built once per platform with the generic
//! flags of [`crate::GrAgcCc`]):
//!
//! - [`GrRotator`]: the scalar `gr::blocks::rotator::rotate` (`gr-rotator.c`, cfg
//!   `has_grrotator`), the reference of the `C32` `rotate` fallback;
//! - [`GrFrequencyModulatorFc`]: `analog.frequency_modulator_fc` with `gr::fxpt`
//!   (`gr-fm-mod.c`, cfg `has_grfmmod`), the reference of `fm_mod`;
//! - [`GrQuadratureDemodCf`]: `analog.quadrature_demod_cf` with VOLK's generic
//!   `multiply_conjugate` and `gr::fast_atan2f` (`gr-quad-demod.c`, `gr-fast-atan2f.c`, cfg
//!   `has_grquaddemod`), the reference of `quad_demod`;
//! - [`GrFmDeemph`]: the `iir_filter_ffd` of `analog.fm_deemph` (`gr-deemph.c`, cfg
//!   `has_grdeemph`), the reference of `deemphasis`.
//!
//! A constructor returns `None` when its library is not linked.

#[cfg(has_grrotator)]
mod rotator_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_rotator_init(
            state: *mut super::RotatorState,
            phase: *const [f32; 2],
            incr: *const [f32; 2],
        );
        pub fn sdrk_gr_rotator_rotate_n(
            state: *mut super::RotatorState,
            output: *mut [f32; 2],
            input: *const [f32; 2],
            n: core::ffi::c_uint,
        );
    }
}

/// The members of `gr::blocks::rotator` (`d_phase`, `d_phase_incr`, `d_counter`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RotatorState {
    /// `d_phase`.
    pub phase: [f32; 2],
    /// `d_phase_incr`.
    pub phase_incr: [f32; 2],
    /// `d_counter`.
    pub counter: u32,
}

/// GNU Radio's scalar rotator: per sample `out = in * phase`, `phase *= incr`, and every 512th
/// sample `phase /= |phase|`.
pub struct GrRotator {
    state: RotatorState,
}

impl GrRotator {
    /// `rotator()` with `set_phase(phase)` and `set_phase_incr(incr)` (both normalized);
    /// `None` without the library.
    #[must_use]
    pub fn new(phase: [f32; 2], incr: [f32; 2]) -> Option<Self> {
        let mut state = RotatorState::default();
        #[cfg(has_grrotator)]
        // SAFETY: plain initialization of the state from two complex values.
        unsafe {
            rotator_ffi::sdrk_gr_rotator_init(&raw mut state, &raw const phase, &raw const incr);
        }
        #[cfg(not(has_grrotator))]
        let _ = (phase, incr, &mut state);
        cfg!(has_grrotator).then_some(GrRotator { state })
    }

    /// The members after the last call.
    #[must_use]
    pub fn state(&self) -> RotatorState {
        self.state
    }

    /// `out[i] = rotate(in[i])`.
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than `u32::MAX` samples.
    pub fn rotate_n(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) {
        assert_eq!(input.len(), output.len(), "rotator: length mismatch");
        let n = u32::try_from(input.len()).expect("rotator: too many samples");
        #[cfg(has_grrotator)]
        // SAFETY: both buffers hold `n` pairs and do not overlap (an exclusive borrow of the
        // output); the state is the C object.
        unsafe {
            rotator_ffi::sdrk_gr_rotator_rotate_n(
                &raw mut self.state,
                output.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }
        #[cfg(not(has_grrotator))]
        let _ = n;
    }
}

#[cfg(has_grfmmod)]
mod fmmod_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_frequency_modulator_fc_init(state: *mut [f32; 2], sensitivity: f32);
        pub fn sdrk_gr_frequency_modulator_fc_work(
            state: *mut [f32; 2],
            noutput_items: core::ffi::c_int,
            input: *const f32,
            output: *mut [f32; 2],
        ) -> core::ffi::c_int;
    }
}

/// GNU Radio's `frequency_modulator_fc(sensitivity)`: `phase += sensitivity * x` in f32, folded
/// into [-pi, pi) with `fmodf`, output `(cos, sin)` of `gr::fxpt` (a 1024-entry table with
/// linear interpolation). The sensitivity is in radians per sample per unit input, i.e.
/// `2 pi deviation / rate`.
pub struct GrFrequencyModulatorFc {
    /// `d_sensitivity`, `d_phase`.
    state: [f32; 2],
}

impl GrFrequencyModulatorFc {
    /// The block with `d_phase = 0`; `None` without the library.
    #[must_use]
    pub fn new(sensitivity: f32) -> Option<Self> {
        let mut state = [sensitivity, 0.0];
        #[cfg(has_grfmmod)]
        // SAFETY: plain initialization of the two members.
        unsafe {
            fmmod_ffi::sdrk_gr_frequency_modulator_fc_init(&raw mut state, sensitivity);
        }
        #[cfg(not(has_grfmmod))]
        let _ = &mut state;
        cfg!(has_grfmmod).then_some(GrFrequencyModulatorFc { state })
    }

    /// The phase after the last call (`d_phase`, radians).
    #[must_use]
    pub fn phase(&self) -> f32 {
        self.state[1]
    }

    /// `work` over `input`.
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than `i32::MAX` samples.
    pub fn work(&mut self, input: &[f32], output: &mut [[f32; 2]]) {
        assert_eq!(
            input.len(),
            output.len(),
            "frequency_modulator_fc: length mismatch"
        );
        let n = core::ffi::c_int::try_from(input.len())
            .expect("frequency_modulator_fc: too many samples");
        #[cfg(has_grfmmod)]
        // SAFETY: both buffers hold `n` values and do not overlap; the state is the C object.
        unsafe {
            fmmod_ffi::sdrk_gr_frequency_modulator_fc_work(
                &raw mut self.state,
                n,
                input.as_ptr(),
                output.as_mut_ptr(),
            );
        }
        #[cfg(not(has_grfmmod))]
        let _ = n;
    }
}

#[cfg(has_grquaddemod)]
mod quaddemod_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_quadrature_demod_cf_init(state: *mut f32, gain: f32);
        pub fn sdrk_gr_quadrature_demod_cf_work(
            state: *mut f32,
            noutput_items: core::ffi::c_int,
            input: *const [f32; 2],
            output: *mut f32,
        ) -> core::ffi::c_int;
        pub fn sdrk_gr_fast_atan2f(y: f32, x: f32) -> f32;
    }
}

/// GNU Radio's `quadrature_demod_cf(gain)`: `out[i] = gain * fast_atan2f(in[i+1] *
/// conj(in[i]))`. The block has a history of two samples, so `work` takes `n + 1` inputs for
/// `n` outputs: `input[0]` is the last sample of the previous call (zero at the start, as GNU
/// Radio's scheduler fills the history).
pub struct GrQuadratureDemodCf {
    /// `d_gain`.
    gain: f32,
}

impl GrQuadratureDemodCf {
    /// The block; `None` without the library.
    #[must_use]
    pub fn new(gain: f32) -> Option<Self> {
        #[cfg_attr(not(has_grquaddemod), allow(unused_mut))]
        let mut gain = gain;
        #[cfg(has_grquaddemod)]
        // SAFETY: plain initialization of the member.
        unsafe {
            quaddemod_ffi::sdrk_gr_quadrature_demod_cf_init(&raw mut gain, gain);
        }
        cfg!(has_grquaddemod).then_some(GrQuadratureDemodCf { gain })
    }

    /// `work`: `input` holds the history sample and then the `output.len()` new samples.
    ///
    /// # Panics
    /// If `input.len() != output.len() + 1`, if there are more than `i32::MAX` outputs or if the
    /// temporary buffer of the block cannot be allocated.
    pub fn work(&mut self, input: &[[f32; 2]], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            output.len() + 1,
            "quadrature_demod_cf: needs n + 1 inputs"
        );
        let n = core::ffi::c_int::try_from(output.len())
            .expect("quadrature_demod_cf: too many samples");
        #[cfg(has_grquaddemod)]
        {
            // SAFETY: `input` holds `n + 1` pairs and `output` `n` floats; they do not overlap.
            let written = unsafe {
                quaddemod_ffi::sdrk_gr_quadrature_demod_cf_work(
                    &raw mut self.gain,
                    n,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                )
            };
            assert_eq!(written, n, "quadrature_demod_cf: allocation failed");
        }
        #[cfg(not(has_grquaddemod))]
        let _ = (n, &mut self.gain);
    }

    /// `gr::fast_atan2f(y, x)` (the out-of-line function of libgnuradio-runtime); `None`
    /// without the library.
    #[must_use]
    pub fn fast_atan2f(y: f32, x: f32) -> Option<f32> {
        #[cfg(has_grquaddemod)]
        // SAFETY: a pure function of two floats.
        return Some(unsafe { quaddemod_ffi::sdrk_gr_fast_atan2f(y, x) });
        #[cfg(not(has_grquaddemod))]
        {
            let _ = (y, x);
            None
        }
    }
}

#[cfg(has_grdeemph)]
mod deemph_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_fm_deemph_size() -> core::ffi::c_ulong;
        pub fn sdrk_gr_fm_deemph_init(object: *mut u64, fs: f64, tau: f64) -> core::ffi::c_int;
        pub fn sdrk_gr_fm_deemph_free(object: *mut u64);
        pub fn sdrk_gr_fm_deemph_work(
            object: *mut u64,
            noutput_items: core::ffi::c_int,
            input: *const f32,
            output: *mut f32,
        ) -> core::ffi::c_int;
        pub fn sdrk_gr_fm_deemph_taps(object: *const u64, taps: *mut [f64; 3]);
    }
}

/// Words of storage for the C object of [`GrFmDeemph`] (it needs 112 bytes on the 64-bit
/// targets).
const GR_DEEMPH_WORDS: usize = 24;

/// GNU Radio's `fm_deemph(fs, tau)`: the first-order IIR filter `iir_filter_ffd` with the
/// bilinear-transform taps of `fm_emph.py` (`b0 = b1`, a pole at `p1`, a zero at `z = -1`),
/// double taps and accumulator, float input and output. Note that SDRk's `deemphasis` is the
/// one-pole filter `y += alpha (x - y)` without the zero; the work per sample is comparable.
pub struct GrFmDeemph {
    object: Box<[u64; GR_DEEMPH_WORDS]>,
}

impl GrFmDeemph {
    /// The block for the sample rate `fs` (Hz) and the time constant `tau` (s); `None` without
    /// the library or when its storage cannot be allocated.
    ///
    /// # Panics
    /// If the C object does not fit the storage of the wrapper.
    #[must_use]
    pub fn new(fs: f64, tau: f64) -> Option<Self> {
        if !cfg!(has_grdeemph) {
            return None;
        }
        #[cfg_attr(not(has_grdeemph), allow(unused_mut))]
        let mut object = Box::new([0u64; GR_DEEMPH_WORDS]);
        #[cfg(has_grdeemph)]
        // SAFETY: the storage is checked to hold the object, and it is 8-byte aligned.
        unsafe {
            let size = usize::try_from(deemph_ffi::sdrk_gr_fm_deemph_size()).unwrap_or(usize::MAX);
            assert!(
                size <= 8 * GR_DEEMPH_WORDS,
                "fm_deemph: the object is too large"
            );
            if deemph_ffi::sdrk_gr_fm_deemph_init(object.as_mut_ptr(), fs, tau) != 0 {
                deemph_ffi::sdrk_gr_fm_deemph_free(object.as_mut_ptr());
                return None;
            }
        }
        #[cfg(not(has_grdeemph))]
        let _ = (fs, tau, &mut object);
        Some(GrFmDeemph { object })
    }

    /// The taps the constructor computed: `[b0, b1, a1]` with the feedback tap as stored
    /// (negated by the new-style constructor), so `y = b0 x + b1 x1 + a1 y1`.
    #[must_use]
    pub fn taps(&self) -> [f64; 3] {
        #[cfg_attr(not(has_grdeemph), allow(unused_mut))]
        let mut taps = [0.0; 3];
        #[cfg(has_grdeemph)]
        // SAFETY: the object was initialized by `new`.
        unsafe {
            deemph_ffi::sdrk_gr_fm_deemph_taps(self.object.as_ptr(), &raw mut taps);
        }
        taps
    }

    /// `work` over `input`.
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than `i32::MAX` samples.
    pub fn work(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "fm_deemph: length mismatch");
        let n = core::ffi::c_int::try_from(input.len()).expect("fm_deemph: too many samples");
        #[cfg(has_grdeemph)]
        // SAFETY: both buffers hold `n` floats and do not overlap; the object was initialized
        // by `new`.
        unsafe {
            deemph_ffi::sdrk_gr_fm_deemph_work(
                self.object.as_mut_ptr(),
                n,
                input.as_ptr(),
                output.as_mut_ptr(),
            );
        }
        #[cfg(not(has_grdeemph))]
        let _ = (n, &mut self.object);
    }
}

#[cfg(has_grdeemph)]
impl Drop for GrFmDeemph {
    fn drop(&mut self) {
        // SAFETY: the object was initialized by `new`; its buffers are freed once.
        unsafe { deemph_ffi::sdrk_gr_fm_deemph_free(self.object.as_mut_ptr()) }
    }
}
