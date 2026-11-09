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
//!   `has_grdeemph`), the reference of `deemphasis`;
//! - [`GrCrc`]: the generic table-driven CRC `gr::digital::crc` of gr-digital (`gr-crc.c`, cfg
//!   `has_grcrc`), the reference of the CRCs of `sdrk-proto`;
//! - [`GrCostasLoopCc`]: the loop of `digital.costas_loop_cc` with `gr::blocks::control_loop`
//!   (`gr-costas.c`, cfg `has_grcostas`), the reference of `costas`;
//! - [`GrConstellationReceiverCb`], [`GrLinearEqualizerRect`]: `digital.constellation_receiver_cb`
//!   and `digital.linear_equalizer` with `adaptive_algorithm_nlms` on a `constellation_rect`
//!   (`gr-constellation.c`, cfg `has_grconstellation`), the references of the QAM paths of
//!   `costas` and `equalize` (M44);
//! - [`GrLinearEqualizer`]: `digital.linear_equalizer` with `adaptive_algorithm_nlms`, the QPSK
//!   decision and VOLK's generic dot products (`gr-equalizer.c`, cfg `has_grequalizer`), the
//!   reference of the `equalize` fallback.
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

#[cfg(has_grcrc)]
mod crc_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_crc_size() -> core::ffi::c_ulong;
        pub fn sdrk_gr_crc_init(
            object: *mut u64,
            num_bits: core::ffi::c_uint,
            poly: u64,
            initial_value: u64,
            final_xor: u64,
            input_reflected: bool,
            result_reflected: bool,
        ) -> core::ffi::c_int;
        pub fn sdrk_gr_crc_compute(object: *mut u64, data: *const u8, len: usize) -> u64;
    }
}

/// Words of storage for the C object of [`GrCrc`] (the 256-entry `uint64_t` table and the
/// parameters: 2088 bytes on the 64-bit targets, 2080 on wasm32).
const GR_CRC_WORDS: usize = 264;

/// GNU Radio's generic CRC `gr::digital::crc(num_bits, poly, initial_value, final_xor,
/// input_reflected, result_reflected)`: a 256-entry table of `uint64_t` built by the
/// constructor, then one table lookup per byte with a 64-bit remainder.
pub struct GrCrc {
    object: Box<[u64; GR_CRC_WORDS]>,
}

impl GrCrc {
    /// The CRC with the parameters of the CRC catalogue (`poly` MSB first); `None` without the
    /// library or for a width outside 8 to 64 bits.
    ///
    /// # Panics
    /// If the C object does not fit the storage of the wrapper.
    #[must_use]
    pub fn new(
        num_bits: u32,
        poly: u64,
        initial_value: u64,
        final_xor: u64,
        input_reflected: bool,
        result_reflected: bool,
    ) -> Option<Self> {
        if !cfg!(has_grcrc) {
            return None;
        }
        #[cfg_attr(not(has_grcrc), allow(unused_mut))]
        let mut object = Box::new([0u64; GR_CRC_WORDS]);
        #[cfg(has_grcrc)]
        // SAFETY: the storage is checked to hold the object, and it is 8-byte aligned.
        unsafe {
            let size = usize::try_from(crc_ffi::sdrk_gr_crc_size()).unwrap_or(usize::MAX);
            assert!(size <= 8 * GR_CRC_WORDS, "crc: the object is too large");
            if crc_ffi::sdrk_gr_crc_init(
                object.as_mut_ptr(),
                num_bits,
                poly,
                initial_value,
                final_xor,
                input_reflected,
                result_reflected,
            ) != 0
            {
                return None;
            }
        }
        #[cfg(not(has_grcrc))]
        let _ = (
            num_bits,
            poly,
            initial_value,
            final_xor,
            input_reflected,
            result_reflected,
            &mut object,
        );
        Some(GrCrc { object })
    }

    /// `compute(data, len)`: the CRC of `data` (0 without the library).
    #[must_use]
    pub fn compute(&mut self, data: &[u8]) -> u64 {
        #[cfg(has_grcrc)]
        // SAFETY: `data` holds `len` bytes; the object was initialized by `new`.
        return unsafe {
            crc_ffi::sdrk_gr_crc_compute(self.object.as_mut_ptr(), data.as_ptr(), data.len())
        };
        #[cfg(not(has_grcrc))]
        {
            let _ = (data, &mut self.object);
            0
        }
    }
}

#[cfg(has_grcostas)]
mod costas_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_costas_cc_init(
            state: *mut super::CostasLoopState,
            loop_bw: f32,
            order: core::ffi::c_uint,
        );
        pub fn sdrk_gr_costas_cc_work(
            state: *mut super::CostasLoopState,
            output: *mut [f32; 2],
            input: *const [f32; 2],
            n: core::ffi::c_uint,
        );
    }
}

/// The members of `gr::blocks::control_loop` and `gr::digital::costas_loop_cc_impl` as the C
/// translation lays them out.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CostasLoopState {
    /// `d_phase` in radians.
    pub phase: f32,
    /// `d_freq` in radians per sample.
    pub freq: f32,
    /// `d_max_freq`.
    pub max_freq: f32,
    /// `d_min_freq`.
    pub min_freq: f32,
    /// `d_damping`.
    pub damping: f32,
    /// `d_loop_bw`.
    pub loop_bw: f32,
    /// `d_alpha`, the proportional gain.
    pub alpha: f32,
    /// `d_beta`, the integral gain.
    pub beta: f32,
    /// `d_error`.
    pub error: f32,
    /// `d_noise`.
    pub noise: f32,
    /// `d_use_snr`.
    pub use_snr: core::ffi::c_int,
    /// `d_order`.
    pub order: core::ffi::c_uint,
}

/// The loop of GNU Radio's `costas_loop_cc(loop_bw, order)` (one output, no tags, `use_snr`
/// false): per sample `sincosf` of the phase, a complex product, the detector, the clip to
/// `+-1` and the second-order loop of `control_loop`.
pub struct GrCostasLoopCc {
    state: CostasLoopState,
}

impl GrCostasLoopCc {
    /// The block `costas_loop_cc(loop_bw, order)` for the order 2 or 4; `None` without the
    /// library or for another order.
    #[must_use]
    pub fn new(loop_bw: f32, order: u32) -> Option<Self> {
        if !cfg!(has_grcostas) || !matches!(order, 2 | 4) {
            return None;
        }
        #[cfg_attr(not(has_grcostas), allow(unused_mut))]
        let mut state = CostasLoopState::default();
        #[cfg(has_grcostas)]
        // SAFETY: the state is the C object, initialized here.
        unsafe {
            costas_ffi::sdrk_gr_costas_cc_init(&raw mut state, loop_bw, order);
        }
        #[cfg(not(has_grcostas))]
        let _ = loop_bw;
        Some(GrCostasLoopCc { state })
    }

    /// The members of the loop.
    #[must_use]
    pub fn state(&self) -> CostasLoopState {
        self.state
    }

    /// `work` over `input`.
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than `u32::MAX` samples.
    pub fn work(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) {
        assert_eq!(input.len(), output.len(), "costas_loop_cc: length mismatch");
        let n = u32::try_from(input.len()).expect("costas_loop_cc: too many samples");
        #[cfg(has_grcostas)]
        // SAFETY: both buffers hold `n` pairs and do not overlap; the state was initialized by
        // `new`.
        unsafe {
            costas_ffi::sdrk_gr_costas_cc_work(
                &raw mut self.state,
                output.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }
        #[cfg(not(has_grcostas))]
        let _ = n;
    }
}

#[cfg(has_grequalizer)]
mod equalizer_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_linear_equalizer_init(
            state: *mut super::LinearEqualizerState,
            num_taps: core::ffi::c_uint,
            step_size: f32,
            taps: *const [f32; 2],
            capacity: core::ffi::c_uint,
        ) -> core::ffi::c_int;
        pub fn sdrk_gr_linear_equalizer_free(state: *mut super::LinearEqualizerState);
        pub fn sdrk_gr_linear_equalizer_work(
            state: *mut super::LinearEqualizerState,
            output: *mut [f32; 2],
            input: *const [f32; 2],
            n: core::ffi::c_uint,
        );
    }
}

/// The members of the C translation of `linear_equalizer` (`gr-equalizer.c`).
#[repr(C)]
#[derive(Debug)]
pub struct LinearEqualizerState {
    num_taps: core::ffi::c_uint,
    step_size: f32,
    taps: *mut [f32; 2],
    aligned: *mut [f32; 2],
    buffer: *mut [f32; 2],
    capacity: core::ffi::c_uint,
    error: [f32; 2],
    decision: [f32; 2],
    points: [[f32; 2]; 4],
}

impl Default for LinearEqualizerState {
    fn default() -> Self {
        LinearEqualizerState {
            num_taps: 0,
            step_size: 0.0,
            taps: core::ptr::null_mut(),
            aligned: core::ptr::null_mut(),
            buffer: core::ptr::null_mut(),
            capacity: 0,
            error: [0.0; 2],
            decision: [0.0; 2],
            points: [[0.0; 2]; 4],
        }
    }
}

/// GNU Radio's `linear_equalizer(num_taps, 1, adaptive_algorithm_nlms(qpsk, step_size))` with
/// `set_taps(taps)`, without a training sequence (decision-directed from the start): per symbol
/// the FIR of the last `num_taps` samples, the QPSK decision (points `+-sqrt 2 +-j sqrt 2`), the
/// error `d - y`, the input power of the window and the NLMS update of every tap.
pub struct GrLinearEqualizer {
    #[cfg_attr(not(has_grequalizer), allow(dead_code))]
    state: Box<LinearEqualizerState>,
    capacity: usize,
}

// SAFETY: the C object owns its buffers and is used by one thread at a time.
unsafe impl Send for GrLinearEqualizer {}

impl GrLinearEqualizer {
    /// The block for `taps` (natural order) and the normalized step size, for at most
    /// `capacity` samples per call; `None` without the library or when the C side cannot
    /// allocate.
    #[must_use]
    pub fn new(taps: &[[f32; 2]], step_size: f32, capacity: usize) -> Option<Self> {
        if !cfg!(has_grequalizer) || taps.is_empty() {
            return None;
        }
        #[cfg_attr(not(has_grequalizer), allow(unused_mut))]
        let mut state = Box::new(LinearEqualizerState::default());
        #[cfg(has_grequalizer)]
        {
            let n = u32::try_from(taps.len()).ok()?;
            let c = u32::try_from(capacity).ok()?;
            // SAFETY: `taps` holds `n` pairs; the state is the C object, initialized here.
            let r = unsafe {
                equalizer_ffi::sdrk_gr_linear_equalizer_init(
                    &raw mut *state,
                    n,
                    step_size,
                    taps.as_ptr(),
                    c,
                )
            };
            if r != 0 {
                return None;
            }
        }
        #[cfg(not(has_grequalizer))]
        let _ = step_size;
        Some(GrLinearEqualizer { state, capacity })
    }

    /// `work` over `input` (at most the capacity of [`GrLinearEqualizer::new`]).
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than the capacity.
    pub fn work(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) {
        assert_eq!(
            input.len(),
            output.len(),
            "linear_equalizer: length mismatch"
        );
        assert!(
            input.len() <= self.capacity,
            "linear_equalizer: above the capacity"
        );
        let n = u32::try_from(input.len()).expect("linear_equalizer: too many samples");
        #[cfg(has_grequalizer)]
        // SAFETY: both buffers hold `n` pairs (at most the capacity) and do not overlap; the
        // state was initialized by `new`.
        unsafe {
            equalizer_ffi::sdrk_gr_linear_equalizer_work(
                &raw mut *self.state,
                output.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }
        #[cfg(not(has_grequalizer))]
        let _ = n;
    }
}

impl Drop for GrLinearEqualizer {
    fn drop(&mut self) {
        #[cfg(has_grequalizer)]
        // SAFETY: the state was initialized by `new` and is freed once.
        unsafe {
            equalizer_ffi::sdrk_gr_linear_equalizer_free(&raw mut *self.state);
        }
    }
}

#[cfg(has_grconstellation)]
mod constellation_ffi {
    unsafe extern "C" {
        pub fn sdrk_gr_constellation_receiver_cb_new(
            points: *const [f32; 2],
            arity: core::ffi::c_uint,
            real_sectors: core::ffi::c_uint,
            imag_sectors: core::ffi::c_uint,
            width_real: f32,
            width_imag: f32,
            loop_bw: f32,
            fmin: f32,
            fmax: f32,
        ) -> *mut core::ffi::c_void;
        pub fn sdrk_gr_constellation_receiver_cb_free(state: *mut core::ffi::c_void);
        pub fn sdrk_gr_constellation_receiver_cb_alpha(state: *const core::ffi::c_void) -> f32;
        pub fn sdrk_gr_constellation_receiver_cb_beta(state: *const core::ffi::c_void) -> f32;
        pub fn sdrk_gr_constellation_receiver_cb_work(
            state: *mut core::ffi::c_void,
            output: *mut [f32; 2],
            input: *const [f32; 2],
            n: core::ffi::c_uint,
        );
        pub fn sdrk_gr_linear_equalizer_rect_new(
            num_taps: core::ffi::c_uint,
            step_size: f32,
            taps: *const [f32; 2],
            capacity: core::ffi::c_uint,
            points: *const [f32; 2],
            arity: core::ffi::c_uint,
            real_sectors: core::ffi::c_uint,
            imag_sectors: core::ffi::c_uint,
            width_real: f32,
            width_imag: f32,
        ) -> *mut core::ffi::c_void;
        pub fn sdrk_gr_linear_equalizer_rect_free(state: *mut core::ffi::c_void);
        pub fn sdrk_gr_linear_equalizer_rect_work(
            state: *mut core::ffi::c_void,
            output: *mut [f32; 2],
            input: *const [f32; 2],
            n: core::ffi::c_uint,
        );
    }
}

/// A `constellation_rect` of GNU Radio as the C translations take it: the points in symbol
/// order, the sectors per axis and their widths, without normalization.
#[derive(Clone, Copy, Debug)]
pub struct GrRect<'a> {
    /// The points in symbol order (at most 64).
    pub points: &'a [[f32; 2]],
    /// The sectors of the real axis.
    pub real_sectors: u32,
    /// The sectors of the imaginary axis.
    pub imag_sectors: u32,
    /// The width of a real sector.
    pub width_real: f32,
    /// The width of an imaginary sector.
    pub width_imag: f32,
}

/// The loop of GNU Radio's `constellation_receiver_cb(constellation_rect, loop_bw, fmin, fmax)`
/// (no tags, the derotated sample as the output): per sample `sincosf` of the phase, a complex
/// product, the sector decision, `-arg(sample conj(point))` and the second-order loop of
/// `control_loop`.
pub struct GrConstellationReceiverCb {
    #[cfg_attr(not(has_grconstellation), allow(dead_code))]
    object: *mut core::ffi::c_void,
}

// SAFETY: the C object is owned by the wrapper and used by one thread at a time.
unsafe impl Send for GrConstellationReceiverCb {}

impl GrConstellationReceiverCb {
    /// The block for the constellation and the loop; `None` without the library or when the C
    /// side refuses the constellation.
    #[must_use]
    pub fn new(rect: GrRect<'_>, loop_bw: f32, fmin: f32, fmax: f32) -> Option<Self> {
        if !cfg!(has_grconstellation) {
            return None;
        }
        #[cfg(has_grconstellation)]
        {
            let arity = u32::try_from(rect.points.len()).ok()?;
            // SAFETY: `points` holds `arity` pairs; the C side copies them.
            let object = unsafe {
                constellation_ffi::sdrk_gr_constellation_receiver_cb_new(
                    rect.points.as_ptr(),
                    arity,
                    rect.real_sectors,
                    rect.imag_sectors,
                    rect.width_real,
                    rect.width_imag,
                    loop_bw,
                    fmin,
                    fmax,
                )
            };
            if object.is_null() {
                return None;
            }
            Some(GrConstellationReceiverCb { object })
        }
        #[cfg(not(has_grconstellation))]
        {
            let _ = (rect, loop_bw, fmin, fmax);
            None
        }
    }

    /// The gains `(alpha, beta)` of the loop.
    #[must_use]
    pub fn gains(&self) -> (f32, f32) {
        #[cfg(has_grconstellation)]
        // SAFETY: the object was created by `new`.
        unsafe {
            (
                constellation_ffi::sdrk_gr_constellation_receiver_cb_alpha(self.object),
                constellation_ffi::sdrk_gr_constellation_receiver_cb_beta(self.object),
            )
        }
        #[cfg(not(has_grconstellation))]
        (0.0, 0.0)
    }

    /// `work` over `input`.
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than `u32::MAX` samples.
    pub fn work(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) {
        assert_eq!(
            input.len(),
            output.len(),
            "constellation_receiver_cb: length mismatch"
        );
        let n = u32::try_from(input.len()).expect("constellation_receiver_cb: too many samples");
        #[cfg(has_grconstellation)]
        // SAFETY: both buffers hold `n` pairs and do not overlap; the object was created by
        // `new`.
        unsafe {
            constellation_ffi::sdrk_gr_constellation_receiver_cb_work(
                self.object,
                output.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }
        #[cfg(not(has_grconstellation))]
        let _ = n;
    }
}

impl Drop for GrConstellationReceiverCb {
    fn drop(&mut self) {
        #[cfg(has_grconstellation)]
        // SAFETY: the object was created by `new` and is freed once.
        unsafe {
            constellation_ffi::sdrk_gr_constellation_receiver_cb_free(self.object);
        }
    }
}

/// GNU Radio's `linear_equalizer(num_taps, 1, adaptive_algorithm_nlms(constellation_rect,
/// step_size))` with `set_taps(taps)`, decision-directed from the start: per symbol the FIR of the
/// last `num_taps` samples, the sector decision of the constellation, the error `d - y`, the
/// input power of the window and the NLMS update of every tap.
pub struct GrLinearEqualizerRect {
    #[cfg_attr(not(has_grconstellation), allow(dead_code))]
    object: *mut core::ffi::c_void,
    capacity: usize,
}

// SAFETY: the C object owns its buffers and is used by one thread at a time.
unsafe impl Send for GrLinearEqualizerRect {}

impl GrLinearEqualizerRect {
    /// The block for `taps` (natural order), the normalized step size and the constellation,
    /// for at most `capacity` samples per call; `None` without the library or when the C side
    /// cannot allocate or refuses the constellation.
    #[must_use]
    pub fn new(
        taps: &[[f32; 2]],
        step_size: f32,
        capacity: usize,
        rect: GrRect<'_>,
    ) -> Option<Self> {
        if !cfg!(has_grconstellation) || taps.is_empty() {
            return None;
        }
        #[cfg(has_grconstellation)]
        {
            let n = u32::try_from(taps.len()).ok()?;
            let c = u32::try_from(capacity).ok()?;
            let arity = u32::try_from(rect.points.len()).ok()?;
            // SAFETY: `taps` holds `n` pairs, `points` `arity` pairs; the C side copies both.
            let object = unsafe {
                constellation_ffi::sdrk_gr_linear_equalizer_rect_new(
                    n,
                    step_size,
                    taps.as_ptr(),
                    c,
                    rect.points.as_ptr(),
                    arity,
                    rect.real_sectors,
                    rect.imag_sectors,
                    rect.width_real,
                    rect.width_imag,
                )
            };
            if object.is_null() {
                return None;
            }
            Some(GrLinearEqualizerRect { object, capacity })
        }
        #[cfg(not(has_grconstellation))]
        {
            let _ = (step_size, capacity, rect);
            None
        }
    }

    /// `work` over `input` (at most the capacity of [`GrLinearEqualizerRect::new`]).
    ///
    /// # Panics
    /// If the buffers differ in length or hold more than the capacity.
    pub fn work(&mut self, input: &[[f32; 2]], output: &mut [[f32; 2]]) {
        assert_eq!(
            input.len(),
            output.len(),
            "linear_equalizer: length mismatch"
        );
        assert!(
            input.len() <= self.capacity,
            "linear_equalizer: above the capacity"
        );
        let n = u32::try_from(input.len()).expect("linear_equalizer: too many samples");
        #[cfg(has_grconstellation)]
        // SAFETY: both buffers hold `n` pairs (at most the capacity) and do not overlap; the
        // object was created by `new`.
        unsafe {
            constellation_ffi::sdrk_gr_linear_equalizer_rect_work(
                self.object,
                output.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }
        #[cfg(not(has_grconstellation))]
        let _ = n;
    }
}

impl Drop for GrLinearEqualizerRect {
    fn drop(&mut self) {
        #[cfg(has_grconstellation)]
        // SAFETY: the object was created by `new` and is freed once.
        unsafe {
            constellation_ffi::sdrk_gr_linear_equalizer_rect_free(self.object);
        }
    }
}
