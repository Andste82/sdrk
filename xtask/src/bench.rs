//! The shared part of the kernel benchmarks of `benches/kernels`: the kernel table, the plans of
//! the platforms and levels, the build of the measurement binary and of the C references, and
//! the profilers that give the execution counts per instruction and the branch counts of a run.
//! `cargo measure` ([`crate::measure`]) simulates the cycles per sample from them; the
//! instruction counts of `simd128` ([`crate::count`]) use the same plans, binaries and
//! references.
//!
//! The method is differential. The measurement binary (`count run <kernel> <impl> <k>`) performs
//! one warm-up call and then `k` calls on the same buffers. It is run with `k = 0` and with
//! `k = K`; the difference of the two profiles is the work of the `K` calls. Everything outside
//! them (start of the process, allocation, set-up, the warm-up call) cancels. The in-place FFTs
//! restore their input before every call; the profile of the restore alone is subtracted from
//! both sides.
//!
//! How the binary of a level runs and where its profile comes from:
//!
//! | Level | Runs on | Profiler |
//! | --- | --- | --- |
//! | `sse2`, `sse4.2` | Intel SDE with a CPUID of a strict SSE2 or SSE4.2 CPU (`-cpuid_in`, `-chip_check_cpuid`), so that VOLK, rustfft and FFTW choose the code of that level themselves | SDE `-dcfg` |
//! | `avx2` | the host | Valgrind `callgrind` |
//! | `avx512` | Intel SDE (`-spr`) | SDE `-dcfg` |
//! | `neon` | QEMU user mode | the plugin `xtask/plugins/bbcount.c` |
//! | `simd128` | wasmtime | none (no core model; instructions from fuel, [`crate::count`]) |
//!
//! Under SDE only the region between two SSC markers of the binary is profiled. The C references
//! are built by `benches/refs/build.sh` (pinned tarballs) into `benches/target/refs/<platform>`.
//! See `docs/performance.md`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, anyhow, bail};
use serde_json::Value;

/// Calls per measurement in the differential method.
pub const K: u64 = 16;
/// Samples per call unless a kernel fixes its own size.
pub const N: u64 = 4096;
/// Guest address space reserved by QEMU (`-R`): it fixes the memory layout, and with it the
/// instructions of the start-up code, so that identical runs give identical counts (without it
/// they differ by up to 50 000 instructions).
pub const QEMU_RESERVED_VA: &str = "0x200000000";
/// The SSC markers around the counted calls (`benches/kernels`, `region_start`/`region_stop`).
const SDE_REGION: [&str; 6] = [
    "-controller_default_start",
    "0",
    "-start_ssc_mark",
    "5d4b0001:repeat",
    "-stop_ssc_mark",
    "5d4b0002:repeat",
];
// ---------------------------------------------------------------------------------------------
// The measurement plan

/// What a kernel is compared with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reference {
    /// VOLK (pinned per level, GNU Radio's call pattern for the FIR family).
    Volk,
    /// VOLK's protokernel without fused multiply-add, compiled with `-ffp-contract=off` for
    /// the level (`benches/refs/volk-nofma.c`): information next to VOLK's fused dispatch where
    /// that dispatch fuses (the gate of `mul` `C32` from D287 until D326 fused ours as well).
    VolkNoFma,
    /// VOLK's portable `generic` protokernel, compiled with `-O3 -ffp-contract=off` for the
    /// level (`benches/refs/volk-generic.c`): the scalar reference of a fallback entry.
    VolkGeneric,
    /// The loop of a GNU Radio block that is no VOLK kernel (`agc_cc`, `pwr_squelch_cc`),
    /// compiled with the generic flags of GNU Radio's own build.
    GnuRadio,
    /// rustfft, `FftPlanner`.
    RustFft,
    /// FFTW, single precision, out of place (as GNU Radio's `fft_vcc` uses it).
    Fftw,
    /// FFTW in place, which needs more passes than out of place.
    FftwInPlace,
    /// `KissFFT`, 16-bit fixed point.
    KissFft,
    /// Ne10, int16 NEON.
    Ne10,
    /// KFR's biquad cascade (`kfr::iir`), compiled for the level (D15).
    Kfr,
    /// CMSIS-DSP's biquad cascades in generic C, compiled for the level (D15, D16).
    Cmsis,
    /// liquid-dsp: the DC blocker (`iirfilt_*_create_dc_blocker`, portable C) and the halfband
    /// decimators and Hilbert transform decimator (`resamp2`, `firhilbf`) with the SIMD dot
    /// product of the level (the portable one for a fallback entry), the amplitude modulator
    /// `ampmodem`, the channelizer `firpfbch2_crcf` and the Gaussian generators `randnf` and
    /// `crandnf` (`benches/refs/liquid-randn.c`, information only).
    Liquid,
    /// libairspy 1.0.10's real-to-complex converters (`iqconverter_float`, `iqconverter_int16`,
    /// conversion only).
    Libairspy,
    /// `rand_distr`'s `StandardNormal` (Ziggurat) over rand's `SmallRng`, the informational
    /// Ziggurat row of `gauss_noise` (M48 §5, `gauss_noise_f32_ziggurat`).
    RandDistr,
    /// Our own fallback, where no established library exists.
    Fallback,
    /// Our own numbers of another level (run-time detection against that fixed level).
    OwnLevel(&'static str),
    /// Our own kernel on another workload of the table (`ours` of the kernel named): the
    /// work a specialised kernel saves against the general kernel it replaces (D69). It is
    /// information, never the judged comparison; the results name it `ours:<kernel>`.
    OwnKernel(&'static str),
}

impl Reference {
    /// The implementation name of the measurement binary.
    #[must_use]
    pub fn implementation(self) -> &'static str {
        match self {
            Reference::Volk => "volk",
            Reference::VolkNoFma => "volk-nofma",
            Reference::VolkGeneric => "volk-generic",
            Reference::GnuRadio => "gnuradio",
            Reference::RustFft => "rustfft",
            Reference::Fftw => "fftw",
            Reference::FftwInPlace => "fftw-in-place",
            Reference::KissFft => "kissfft",
            Reference::Ne10 => "ne10",
            Reference::Kfr => "kfr",
            Reference::Cmsis => "cmsis",
            Reference::Liquid => "liquid",
            Reference::Libairspy => "airspy",
            Reference::RandDistr => "rand-distr",
            Reference::Fallback => "fallback",
            Reference::OwnLevel(_) | Reference::OwnKernel(_) => "ours",
        }
    }
}

/// The kind of comparison a kernel has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// A VOLK kernel exists.
    Volk,
    /// A VOLK kernel exists, but not in the wasm32 build of VOLK (`benches/refs/volk-wasm.c`
    /// holds only the generic kernels of the entries of [`Family::Volk`]): there our fallback is
    /// the comparison, as information (D12).
    VolkNative,
    /// A VOLK kernel exists, and at the levels with FMA (`avx2`, `avx512`, `neon`) the
    /// dispatcher picks a fused implementation (`volk_32fc_x2_multiply_32fc`: `a_avx2_fma`,
    /// `neonv8`), as ours does there (D326): VOLK's dispatch is the gate as in [`Family::Volk`],
    /// and VOLK's protokernel without fused operations ([`Reference::VolkNoFma`], the gate of
    /// D287) is information at those levels.
    VolkNoFma,
    /// A VOLK kernel exists, and at the levels with FMA (`avx2`, `avx512`, `neon`) the
    /// dispatcher picks a fused implementation (`volk_32fc_x2_dot_prod_32fc`: `a_avx_fma`,
    /// `neonv8`) that the bit-exact contract of ours forbids (`dot_complex`, D821): there VOLK's
    /// protokernel without fused operations ([`Reference::VolkNoFma`]) is the gate, as for `mul`
    /// under D287, and VOLK's dispatch is information; at the other levels as [`Family::Volk`].
    VolkNoFmaGate,
    /// A VOLK kernel does the same work up to a constant offset (`convert_cu8`: VOLK converts
    /// signed bytes only): it is the gate, and our fallback is reported as information.
    VolkAndOwn,
    /// A loop of GNU Radio that is no VOLK kernel (`agc_cc`, `pwr_squelch_cc`,
    /// `frequency_modulator_fc`): it is the gate (where `limits.json` judges it), and our fallback
    /// is reported as information.
    GnuRadio,
    /// The C32 FFT.
    FftC32,
    /// The CQ15 FFT.
    FftCq15,
    /// The real-input FFT: per level (`LevelPlan::rfft`) FFTW's real-input transform or rustfft's
    /// real planner (`realfft`), our fallback as information.
    RealFft,
    /// No established library: our fallback.
    Own,
    /// A workload of an own-kernel comparison ([`Reference::OwnKernel`], D69): measured as
    /// [`Family::Own`], but not a kernel of the library, so it has no entry at a level that is
    /// judged against another build of ours (`avx2-runtime`: the workload runs a kernel whose
    /// run-time path is judged by its own entry).
    Workload,
    /// The f32 and C32 biquads (D15): KFR's SIMD cascade where the level has a build of it
    /// (`LevelPlan::kfr`), CMSIS-DSP's generic C as information; at a level without KFR,
    /// CMSIS-DSP is the gate.
    Kfr,
    /// The Q15 and CQ15 biquads (D15), `rotate`, `fm_mod` and `am_mod` of `CQ15`: no SIMD
    /// library has them on our levels, so our fallback is the comparison; CMSIS-DSP's generic C
    /// (for `rotate`, `fm_mod` and `am_mod` composed of its building blocks) as information.
    Q15Own,
    /// A fallback entry (D16, `<kernel>_fallback`): the measured side is our fallback, compiled
    /// for the level, and the gate is CMSIS-DSP's generic C. Not measured at a level with
    /// run-time detection, whose fallback is the baseline build.
    FallbackVsCmsis,
    /// A fallback entry as [`Family::FallbackVsCmsis`] whose gate is VOLK's portable `generic`
    /// protokernel compiled for the level (`interleave` and `deinterleave`; CMSIS-DSP has no
    /// interleave). For the `CQ15` interleave, which VOLK lacks, the loop of the `f32` kernel
    /// on 16-bit values.
    FallbackVsVolkGeneric,
    /// A fallback entry as [`Family::FallbackVsCmsis`] whose gate is the scalar loop of a GNU
    /// Radio block (`rotate`: `gr::blocks::rotator`, `fm_mod`: `frequency_modulator_fc`,
    /// `deemphasis_fallback`: the IIR loop of `fm_deemph`, `agc`: `agc_cc`, `quad_demod`:
    /// `quadrature_demod_cf`), compiled with the generic flags of GNU Radio's own build;
    /// CMSIS-DSP's generic C as information where the entry has it.
    FallbackVsGnuRadio,
    /// A fallback entry as [`Family::FallbackVsCmsis`] whose gate is liquid-dsp's DC blocker
    /// (`dc_block_fallback`, `dc_block_real_fallback`), its halfband decimator with the
    /// portable dot product (`halfband_decimate_fallback`, `halfband_decimate_real_fallback`) or
    /// its amplitude modulator `ampmodem` (`am_mod_fallback`), or its Gaussian generators
    /// `crandnf` and `randnf` (`gauss_noise_c32_fallback`, `gauss_noise_f32_fallback`, information
    /// only: liquid does different work), its channelizer `firpfbch2_crcf`
    /// (`channelize_fallback`) or its fast convolution `fftfilt_cccf` with its own FFT
    /// (`xcorr_fft_n512_fallback`, `xcorr_fft_n4096_fallback`).
    FallbackVsLiquid,
    /// A fallback entry whose kernel has no library that does the same work
    /// (`gauss_noise_q15_fallback`, M48: no library makes `Q15` Gaussian noise): the measured
    /// side is our fallback, compiled for the level, and the comparison is the same fallback
    /// (ratio 1, information only). The entry exists so that the D321 cycle gate pairs the SIMD
    /// entry of [`Family::Own`] with it. Not measured at a level with run-time detection.
    FallbackOwn,
    /// The halfband decimators of the float types (`halfband_decimate`,
    /// `halfband_decimate_real`): liquid-dsp's `resamp2` with its SIMD dot product of the level
    /// (the portable loop at `simd128`, where liquid has no SIMD loop).
    HalfbandLiquid,
    /// The channelizer `channelize` (M45): liquid-dsp's `firpfbch2_crcf` analyzer with the SIMD
    /// dot product of the level and FFTW (liquid's own FFT on wasm32, which has no FFTW build),
    /// our fallback as information.
    ChannelizeLiquid,
    /// The fast convolution `xcorr_fft` (M50 §5): liquid-dsp's `fftfilt_cccf` with FFTW (liquid's
    /// own FFT on wasm32, which has no FFTW build; information there), our fallback as
    /// information.
    XcorrLiquid,
    /// `gauss_noise` (M48): liquid-dsp's `crandnf` and `randnf` over libc `rand()`
    /// (`benches/refs/liquid-randn.c`), which do other work than the contract (`f32` Box–Muller,
    /// no PCG32, no random access, no bit-exact sequence): the comparison as information (`info`
    /// with the waiver "reference does different work" at every level), our fallback beside it.
    LiquidDifferentWork,
    /// `gauss_noise_c32_fastnoise`, `gauss_noise_f32_fastnoise` (M48): our `gauss_noise` against
    /// the work loop of GNU Radio's `fastnoise_source_c` / `_f` (`benches/refs/gr-fastnoise.c`),
    /// which copies one value of a precomputed pool per sample (`info` with the waiver "reference
    /// does less work" at every level). Not measured at a level with run-time detection, whose
    /// kernel entry is judged against the fixed level.
    GaussNoiseFastnoise,
    /// `gauss_noise_f32_ziggurat` (M48): our `gauss_noise` against `rand_distr`'s `StandardNormal`
    /// (Ziggurat, rejection sampling) as `f32` (`info` with the waiver "reference does different
    /// work" at every level); not measured at a level with run-time detection.
    GaussNoiseZiggurat,
    /// `scramble_cq15_b2`, `scramble_cq15_b8` (M49 §5): our kernel through the facade against the
    /// phasor looked up into a buffer plus CMSIS-DSP's `arm_cmplx_mult_cmplx_q15` (generic C
    /// compiled for the level, Q3.13 output), our fallback as information.
    ScrambleCmsis,
    /// `scramble_c32_b2_dvbs2` (M49 §5): our `scramble` at `bits = 2` against the per-symbol
    /// switch on `R_n` of GNU Radio's `dvbs2_physical_cc` (`benches/refs/gr-dvbs2-plscramble.c`;
    /// `info` with the waiver "reference does different work" at every level). Not measured at a
    /// level with run-time detection, whose kernel entry is judged against the fixed level.
    ScrambleDvbs2,
    /// `real_to_complex` of `C32`: liquid-dsp's `firhilbf` with its SIMD dot product; at `sse2`
    /// libairspy's `iqconverter_float` (SSE2) is the second gate, the entry
    /// [`Family::HilbertAirspy`].
    HilbertLiquid,
    /// `real_to_complex_airspy`, measured at `sse2` only: our `real_to_complex` against libairspy's
    /// `iqconverter_float` (SSE2), the second gate of `real_to_complex` at that level (the
    /// first is liquid-dsp's `firhilbf`; both must pass).
    HilbertAirspy,
    /// `real_to_complex_cq15`: libairspy's `iqconverter_int16`, a lower accuracy class (truncating
    /// `>> 15`, a 32-bit accumulator, the odd samples halved). Its status is the maintainer's
    /// open decision (I23-1), `info` until then; our fallback as information.
    HilbertAirspyQ15,
    /// `real_to_complex_fallback`: liquid-dsp's `firhilbf` with the portable dot product; libairspy's
    /// generic C as information.
    FallbackHilbert,
    /// `real_to_complex_cq15_fallback`: libairspy's `iqconverter_int16` (generic C).
    FallbackVsAirspy,
    /// The fallback entry of a `C32` or real-input FFT (`fft_<n>_fallback`,
    /// `rfft_<n>_fallback`): as [`Family::FallbackVsCmsis`] against `arm_cfft_f32` or
    /// `arm_rfft_fast_f32`, with the input restore of the FFTs subtracted.
    FftFallbackVsCmsis,
    /// The fallback entry of a `CQ15` FFT (`fft_cq15_<n>..._fallback`): as
    /// [`Family::FftFallbackVsCmsis`] against `arm_cfft_q15` (fixed `1/N` scaling with
    /// truncating shifts), with `KissFFT` (scalar C as well) as information.
    FftCq15FallbackVsCmsis,
}

impl Family {
    /// The FFTs, whose measurements restore their input before every call (see `Impl::Restore`
    /// of the measurement binary) and whose sizes are part of the name.
    #[must_use]
    pub fn is_fft(self) -> bool {
        matches!(
            self,
            Family::FftC32
                | Family::FftCq15
                | Family::RealFft
                | Family::FftFallbackVsCmsis
                | Family::FftCq15FallbackVsCmsis
        )
    }

    /// A fallback entry (`<kernel>_fallback`), whose measured side is our fallback.
    #[must_use]
    pub fn is_fallback_entry(self) -> bool {
        matches!(
            self,
            Family::FallbackVsCmsis
                | Family::FallbackVsVolkGeneric
                | Family::FallbackVsGnuRadio
                | Family::FallbackVsLiquid
                | Family::FallbackOwn
                | Family::FallbackHilbert
                | Family::FallbackVsAirspy
                | Family::FftFallbackVsCmsis
                | Family::FftCq15FallbackVsCmsis
        )
    }

    /// The implementation that is measured: our fallback for a fallback entry, our kernel
    /// through the facade otherwise.
    #[must_use]
    pub fn measured(self) -> &'static str {
        if self.is_fallback_entry() {
            "fallback"
        } else {
            "ours"
        }
    }

    /// Whether the kernel has an entry at `plan`'s level.
    #[must_use]
    pub fn measured_at(self, plan: &LevelPlan) -> bool {
        if self == Family::HilbertAirspy {
            return plan.level == "sse2" && plan.baseline.is_none();
        }
        !((self.is_fallback_entry()
            || matches!(
                self,
                Family::Workload
                    | Family::GaussNoiseFastnoise
                    | Family::GaussNoiseZiggurat
                    | Family::ScrambleDvbs2
            ))
            && plan.baseline.is_some())
    }
}

/// A kernel of the benchmark table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelSpec {
    /// Name (as the measurement binary knows it).
    pub name: &'static str,
    /// Its comparisons.
    pub family: Family,
    /// Samples per call.
    pub samples: u64,
    /// Also measured with the [`crate::count::SMALL_N`] block sizes (informational).
    pub small_n: bool,
    /// Also measured with misaligned buffers (informational).
    pub misaligned: bool,
    /// Another kernel of the table whose `ours` is reported next to the comparisons as
    /// information ([`Reference::OwnKernel`]).
    pub own: Option<&'static str>,
}

const fn spec(name: &'static str, family: Family, small_n: bool) -> KernelSpec {
    KernelSpec {
        name,
        family,
        samples: N,
        small_n,
        misaligned: true,
        own: None,
    }
}

/// A kernel with an own-kernel comparison.
const fn spec_own(name: &'static str, family: Family, own: &'static str) -> KernelSpec {
    KernelSpec {
        name,
        family,
        samples: N,
        small_n: false,
        misaligned: true,
        own: Some(own),
    }
}

/// A protocol entry of `sdrk-proto` or a packet block of M37: no backends, so ours and the fallback
/// are the same code; a "sample" is a byte (CRCs), a bit (frame decoder) or a signal sample of one
/// bit (the blocks `frame_encode` and `frame_decode`), and the buffers are plain vectors, so there
/// is no misaligned variant. `family` is [`Family::GnuRadio`] for the CRCs (GNU Radio's
/// `gr::digital::crc`, D527) and [`Family::Own`] (information only) for the framing, which no
/// library has. The RDS blocks of M41 (`rds_demod_block`, one multiplex sample; `rds_decode_block`,
/// one soft bit) and the Morse blocks of M47 (`morse_encode_block`, one output sample;
/// `morse_decode_block`, one input sample) use the same shape: blocks that no library has,
/// information only. So do the channelizer blocks of M45 (`channelize_block`, one input sample: the
/// kernel entry `channelize` with the block around it, whose library comparison is that kernel
/// entry; `channel_select_block`, one output sample, a strided copy; D928) and the two channel
/// banks of M45 test 9 (`channel_bank_channelize`, one channelizer and eight `channel_select`
/// blocks, and `channel_bank_ddc`, eight `ddc` blocks for the same eight channels, one input sample
/// each; the comparison is between the two banks, D930) and the band pulse detector of M45.1
/// (`band_pulse_detect_block`, one lane value; no non-GPL library has it, D962). The QAM slicer of
/// M44 (`symbol_demap_qam16_block`, `symbol_demap_qam64_block`, and of M44.2
/// `symbol_demap_qam256_block`, one input symbol each) is the block around the kernel
/// `qam_soft_bits`, whose soft bits no library computes the same way (D1201, D1412): information
/// too, as are the QAM blocks `qam_mod_block` and `qam_mod_qam256_block` (one output sample) and
/// `qam_demod_block` and `qam_demod_qam256_block` (one input sample), chains of kernels with
/// entries of their own (D1216), the
/// PSK modulator block `psk_mod_block` of M44.1 (one output sample), the building blocks and
/// composites of M44.1 `symbol_map_block` (one output symbol), `iq_offset_block` (one sample),
/// `qam_acquire_block` (one input symbol), `qam_frame_encode_block` (one output sample) and
/// `qam_frame_decode_block` (one input sample), the building blocks of M44.1.1
/// `pulse_shape_block` (one output sample) and `pulse_match_block` (one input sample),
/// `frame_decode_four_block` (one soft bit, `frame_decode` with `rotations` `four`), and the
/// noise source of M48 `noise_block` (one output sample, the block around the kernel entries
/// `gauss_noise_c32` and `gauss_noise_f32`), and the scrambler of M49 `scramble_block` (one
/// sample, the PRBS indices and the kernel entry `scramble_c32_b8`). The bit scramblers of M44.2
/// (`bit_scramble_block`, `bit_descramble_block`, and their in-place forms
/// `bit_scramble_in_place_block`, `bit_descramble_in_place_block`, one soft bit each) are
/// [`Family::GnuRadio`]: GNU Radio's `scrambler_bb` and `descrambler_bb` with the same polynomial
/// are the scalar reference. The SNR estimate of M44.4 (`snr_estimate_block` and
/// `snr_estimate_block_svr`, one input symbol each) is [`Family::GnuRadio`] too: GNU Radio's
/// `mpsk_snr_est_cc` with its estimators M2M4 and SVR, as information in `limits.json`
/// (reference does different work, D2504).
const fn proto(name: &'static str, family: Family) -> KernelSpec {
    KernelSpec {
        name,
        family,
        samples: N,
        small_n: false,
        misaligned: false,
        own: None,
    }
}

/// An entry of `xcorr_fft` (M50), or the block `correlate_block` (`CorrelateBlock` with the
/// default `prbs7` code, `L` 127, `N_fft` 512, the same samples as `xcorr_fft_n512`): the whole
/// transform blocks that fit the standard [`N`]
/// samples of the measurement binary (`samples` = `floor(N / B) B`, `B = N_fft - L + 1`: 10
/// blocks of 386 for `L` 127, `N_fft` 512; one block of 3074 for `L` 1023, `N_fft` 4096), with
/// misaligned buffers. The kernel entries are judged against liquid-dsp's `fftfilt_cccf`
/// ([`Family::XcorrLiquid`], [`Family::FallbackVsLiquid`]; liquid's block length is the power of
/// two nearest to `B`, and it takes the same samples per call, D2619); `correlate_block` is
/// information ([`Family::Own`]).
const fn xcorr(name: &'static str, family: Family, samples: u64) -> KernelSpec {
    KernelSpec {
        name,
        family,
        samples,
        small_n: false,
        misaligned: true,
        own: None,
    }
}

const fn fft(name: &'static str, family: Family, samples: u64) -> KernelSpec {
    KernelSpec {
        name,
        family,
        samples,
        small_n: false,
        misaligned: false,
        own: None,
    }
}

/// The kernels of the benchmark table, in report order.
pub static KERNELS: [KernelSpec; 286] = [
    spec("convert_cs8", Family::Volk, true),
    spec("convert_cs16", Family::Volk, true),
    spec("convert_cu8", Family::VolkAndOwn, true),
    spec("convert_real_rs16", Family::Volk, true),
    spec("convert_cq15_cs8", Family::Volk, true),
    spec("convert_cq15_cu8", Family::VolkAndOwn, true),
    spec("convert_out_cs16", Family::Volk, true),
    spec("convert_out_cs8", Family::Volk, true),
    spec("convert_out_cu8", Family::Volk, true),
    spec("convert_out_real_rs16", Family::Volk, true),
    spec("convert_out_real_rs8", Family::Volk, true),
    spec("convert_out_real_ru8", Family::Volk, true),
    spec("convert_cs8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_cs16_fallback", Family::FallbackVsCmsis, false),
    spec("convert_cu8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_real_rs16_fallback", Family::FallbackVsCmsis, false),
    spec("convert_cq15_cs8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_cq15_cu8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_out_cs16_fallback", Family::FallbackVsCmsis, false),
    spec("convert_out_cs8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_out_cu8_fallback", Family::FallbackVsCmsis, false),
    spec(
        "convert_out_real_rs16_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "convert_out_real_rs8_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "convert_out_real_ru8_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("rotate", Family::Volk, false),
    spec("rotate_cq15", Family::Q15Own, false),
    spec("rotate_fallback", Family::FallbackVsGnuRadio, false),
    spec("rotate_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("scramble_c32_b1", Family::Volk, false),
    spec("scramble_c32_b2", Family::VolkNoFmaGate, false),
    spec("scramble_c32_b8", Family::VolkNoFmaGate, false),
    spec("scramble_cq15_b2", Family::ScrambleCmsis, false),
    spec("scramble_cq15_b8", Family::ScrambleCmsis, false),
    spec(
        "scramble_c32_b1_fallback",
        Family::FallbackVsVolkGeneric,
        false,
    ),
    spec(
        "scramble_c32_b2_fallback",
        Family::FallbackVsVolkGeneric,
        false,
    ),
    spec(
        "scramble_c32_b8_fallback",
        Family::FallbackVsVolkGeneric,
        false,
    ),
    spec("scramble_cq15_b2_fallback", Family::FallbackVsCmsis, false),
    spec("scramble_cq15_b8_fallback", Family::FallbackVsCmsis, false),
    spec("scramble_c32_b2_dvbs2", Family::ScrambleDvbs2, false),
    proto("scramble_block", Family::Own),
    spec("fir", Family::Volk, false),
    spec("fir_decimate", Family::Volk, false),
    spec("fir_interpolate", Family::Volk, false),
    spec("fir_interpolate_short", Family::Volk, false),
    spec("resample", Family::Volk, false),
    spec("fir_interpolate_real", Family::Volk, false),
    spec("resample_real", Family::Volk, false),
    spec("fir_interpolate_real_q15", Family::Own, false),
    spec("resample_real_q15", Family::Own, false),
    spec("fir_cq15", Family::Own, false),
    spec("fir_decimate_cq15", Family::Own, false),
    spec("fir_interpolate_cq15", Family::Own, false),
    spec("resample_cq15", Family::Own, false),
    spec("mag_sq", Family::Volk, false),
    spec("mag_sq_q15", Family::Own, false),
    spec("quad_demod", Family::Volk, false),
    spec("quad_demod_cq15", Family::Own, false),
    spec("quad_demod_fallback", Family::FallbackVsGnuRadio, false),
    spec("quad_demod_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("fm_mod", Family::GnuRadio, false),
    spec("fm_mod_cq15", Family::Q15Own, false),
    spec("fm_mod_fallback", Family::FallbackVsGnuRadio, false),
    spec("fm_mod_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("squelch", Family::GnuRadio, false),
    spec("squelch_cq15", Family::Own, false),
    spec("mag", Family::Volk, false),
    spec("mag_q15", Family::Volk, false),
    spec("window", Family::Volk, false),
    spec("window_cq15", Family::Own, false),
    spec("power_db", Family::Volk, false),
    spec("power_db_q15", Family::Own, false),
    spec("mag_sq_fallback", Family::FallbackVsCmsis, false),
    spec("mag_sq_q15_fallback", Family::FallbackVsCmsis, false),
    spec("mag_fallback", Family::FallbackVsCmsis, false),
    spec("mag_q15_fallback", Family::FallbackVsCmsis, false),
    spec("window_fallback", Family::FallbackVsCmsis, false),
    spec("window_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("power_db_fallback", Family::FallbackVsCmsis, false),
    spec("power_db_q15_fallback", Family::FallbackVsCmsis, false),
    spec("agc", Family::GnuRadio, false),
    spec("agc_cq15", Family::Own, false),
    spec("agc_fallback", Family::FallbackVsGnuRadio, false),
    spec("dc_block_real", Family::Own, false),
    spec("dc_block_real_q15", Family::Own, false),
    spec("deemphasis", Family::Own, false),
    spec("deemphasis_q15", Family::Own, false),
    spec("dc_block", Family::Own, false),
    spec("dc_block_cq15", Family::Own, false),
    spec("dc_block_real_fallback", Family::FallbackVsLiquid, false),
    spec("deemphasis_fallback", Family::FallbackVsGnuRadio, false),
    spec("dc_block_fallback", Family::FallbackVsLiquid, false),
    spec("biquad", Family::Kfr, true),
    spec("biquad_cq15", Family::Q15Own, true),
    spec("biquad_real", Family::Kfr, true),
    spec("biquad_real_q15", Family::Q15Own, true),
    spec("biquad_fallback", Family::FallbackVsCmsis, false),
    spec("biquad_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("biquad_real_fallback", Family::FallbackVsCmsis, false),
    spec("biquad_real_q15_fallback", Family::FallbackVsCmsis, false),
    spec("fir_fallback", Family::FallbackVsCmsis, false),
    spec("fir_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("fir_decimate_fallback", Family::FallbackVsCmsis, false),
    spec("fir_decimate_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("fir_interpolate_fallback", Family::FallbackVsCmsis, false),
    spec(
        "fir_interpolate_short_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "fir_interpolate_cq15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("resample_fallback", Family::FallbackVsCmsis, false),
    spec("resample_cq15_fallback", Family::FallbackVsCmsis, false),
    spec(
        "fir_interpolate_real_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "fir_interpolate_real_q15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("resample_real_fallback", Family::FallbackVsCmsis, false),
    spec("resample_real_q15_fallback", Family::FallbackVsCmsis, false),
    spec("dot", Family::Volk, true),
    spec("dot_cq15", Family::Own, true),
    spec("dot_real", Family::VolkNative, true),
    spec("dot_real_q15", Family::Own, true),
    spec("scale", Family::VolkNative, false),
    spec("scale_cq15", Family::Own, false),
    spec("dot_fallback", Family::FallbackVsCmsis, false),
    spec("dot_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("dot_real_fallback", Family::FallbackVsCmsis, false),
    spec("dot_real_q15_fallback", Family::FallbackVsCmsis, false),
    spec("scale_fallback", Family::FallbackVsCmsis, false),
    spec("scale_cq15_fallback", Family::FallbackVsCmsis, false),
    spec_own(
        "halfband_decimate",
        Family::HalfbandLiquid,
        "halfband_ref_fir_decimate",
    ),
    spec_own(
        "halfband_decimate_cq15",
        Family::Q15Own,
        "halfband_ref_fir_decimate_cq15",
    ),
    spec("halfband_decimate_real", Family::HalfbandLiquid, false),
    spec("halfband_decimate_real_q15", Family::Q15Own, false),
    spec_own(
        "real_to_complex",
        Family::HilbertLiquid,
        "halfband_ref_shifted",
    ),
    spec_own(
        "real_to_complex_cq15",
        Family::HilbertAirspyQ15,
        "halfband_ref_shifted_cq15",
    ),
    spec("real_to_complex_airspy", Family::HilbertAirspy, false),
    spec(
        "halfband_decimate_fallback",
        Family::FallbackVsLiquid,
        false,
    ),
    spec(
        "halfband_decimate_cq15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "halfband_decimate_real_fallback",
        Family::FallbackVsLiquid,
        false,
    ),
    spec(
        "halfband_decimate_real_q15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("real_to_complex_fallback", Family::FallbackHilbert, false),
    spec(
        "real_to_complex_cq15_fallback",
        Family::FallbackVsAirspy,
        false,
    ),
    spec("halfband_ref_fir_decimate", Family::Workload, false),
    spec("halfband_ref_fir_decimate_cq15", Family::Workload, false),
    spec("halfband_ref_shifted", Family::Workload, false),
    spec("halfband_ref_shifted_cq15", Family::Workload, false),
    spec("stereo_matrix", Family::Volk, false),
    spec("stereo_matrix_cq15", Family::Q15Own, false),
    spec("stereo_matrix_fallback", Family::FallbackVsCmsis, false),
    spec(
        "stereo_matrix_cq15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("stereo_pll_fallback", Family::FallbackVsLiquid, false),
    spec("stereo_pll_q15_fallback", Family::FallbackVsCmsis, false),
    spec("preemph_limit", Family::Own, false),
    spec("preemph_limit_q15", Family::Own, false),
    spec("noise_blank", Family::Own, false),
    spec("noise_blank_cq15", Family::Own, false),
    spec("auto_notch", Family::Own, false),
    spec("am_mod", Family::Own, false),
    spec("am_mod_cq15", Family::Q15Own, false),
    spec("am_mod_fallback", Family::FallbackVsLiquid, false),
    spec("am_mod_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("preemph_limit_stereo", Family::Own, false),
    spec("stereo_encode", Family::Own, false),
    proto("crc16_ccitt_false", Family::GnuRadio),
    proto("crc16_x25", Family::GnuRadio),
    proto("crc32", Family::GnuRadio),
    proto("crc24_adsb", Family::GnuRadio),
    proto("frame_decode", Family::Own),
    proto("frame_encode_block", Family::Own),
    proto("frame_decode_block", Family::Own),
    spec("symbol_sync", Family::Own, false),
    spec("symbol_sync_real", Family::Own, false),
    spec("costas", Family::Own, false),
    spec("costas_fallback", Family::FallbackVsGnuRadio, false),
    proto("rds_demod_block", Family::Own),
    proto("rds_decode_block", Family::Own),
    spec("dot_complex", Family::VolkNoFmaGate, true),
    spec("dot_complex_fallback", Family::FallbackVsCmsis, false),
    spec("equalize", Family::Own, false),
    spec("equalize_fallback", Family::FallbackVsGnuRadio, false),
    spec("costas_qam16", Family::Own, false),
    spec("costas_qam16_fallback", Family::FallbackVsGnuRadio, false),
    spec("equalize_qam16", Family::Own, false),
    spec("equalize_qam16_fallback", Family::FallbackVsGnuRadio, false),
    spec("costas_qam64", Family::Own, false),
    spec("costas_qam64_fallback", Family::FallbackVsGnuRadio, false),
    spec("equalize_qam64", Family::Own, false),
    spec("equalize_qam64_fallback", Family::FallbackVsGnuRadio, false),
    spec("rds_shape", Family::Own, false),
    spec("stereo_encode_rds", Family::Own, false),
    proto("morse_encode_block", Family::Own),
    proto("morse_decode_block", Family::Own),
    spec("channelize", Family::ChannelizeLiquid, false),
    spec("channelize_fallback", Family::FallbackVsLiquid, false),
    proto("channelize_block", Family::Own),
    proto("channel_select_block", Family::Own),
    proto("channel_bank_channelize", Family::Own),
    proto("channel_bank_ddc", Family::Own),
    proto("band_pulse_detect_block", Family::Own),
    spec("qam_soft_bits_qam16", Family::Own, false),
    spec("qam_soft_bits_qam64", Family::Own, false),
    spec("qam_soft_bits_qpsk", Family::Own, false),
    spec("qam_soft_bits_qam256", Family::Own, false),
    spec("qam_soft_bits_qam1024", Family::Own, false),
    spec("costas_qam256", Family::Own, false),
    spec("costas_qam256_fallback", Family::FallbackVsGnuRadio, false),
    spec("costas_qam1024", Family::Own, false),
    spec("equalize_qam256", Family::Own, false),
    spec(
        "equalize_qam256_fallback",
        Family::FallbackVsGnuRadio,
        false,
    ),
    spec("equalize_qam1024", Family::Own, false),
    spec("costas_qam1024_fallback", Family::FallbackVsGnuRadio, false),
    spec(
        "equalize_qam1024_fallback",
        Family::FallbackVsGnuRadio,
        false,
    ),
    proto("bit_scramble_block", Family::GnuRadio),
    proto("bit_descramble_block", Family::GnuRadio),
    proto("bit_scramble_in_place_block", Family::GnuRadio),
    proto("bit_descramble_in_place_block", Family::GnuRadio),
    proto("snr_estimate_block", Family::GnuRadio),
    proto("snr_estimate_block_svr", Family::GnuRadio),
    proto("symbol_demap_qam16_block", Family::Own),
    proto("symbol_demap_qam64_block", Family::Own),
    proto("symbol_demap_qam256_block", Family::Own),
    proto("qam_mod_block", Family::Own),
    proto("qam_mod_qam256_block", Family::Own),
    proto("qam_demod_block", Family::Own),
    proto("qam_demod_qam256_block", Family::Own),
    proto("qam_demod_search_block", Family::Own),
    proto("psk_mod_block", Family::Own),
    proto("symbol_map_block", Family::Own),
    proto("iq_offset_block", Family::Own),
    proto("qam_acquire_block", Family::Own),
    proto("qam_frame_encode_block", Family::Own),
    proto("qam_frame_decode_block", Family::Own),
    proto("pulse_shape_block", Family::Own),
    proto("pulse_match_block", Family::Own),
    proto("frame_decode_four_block", Family::Own),
    spec("gauss_noise_c32", Family::LiquidDifferentWork, false),
    spec("gauss_noise_f32", Family::LiquidDifferentWork, false),
    spec("gauss_noise_q15", Family::Own, false),
    proto("noise_block", Family::Own),
    spec("gauss_noise_c32_fallback", Family::FallbackVsLiquid, false),
    spec("gauss_noise_f32_fallback", Family::FallbackVsLiquid, false),
    spec("gauss_noise_q15_fallback", Family::FallbackOwn, false),
    spec(
        "gauss_noise_c32_fastnoise",
        Family::GaussNoiseFastnoise,
        false,
    ),
    spec(
        "gauss_noise_f32_fastnoise",
        Family::GaussNoiseFastnoise,
        false,
    ),
    spec(
        "gauss_noise_f32_ziggurat",
        Family::GaussNoiseZiggurat,
        false,
    ),
    spec("add", Family::Volk, false),
    spec("add_cq15", Family::Volk, false),
    spec("add_real", Family::Volk, false),
    spec("add_real_q15", Family::Volk, false),
    spec("mul", Family::VolkNoFma, false),
    spec("mul_cq15", Family::Own, false),
    spec("mul_real", Family::Volk, false),
    spec("mul_real_q15", Family::Own, false),
    spec("interleave", Family::Volk, false),
    spec("interleave_cq15", Family::Own, false),
    spec("deinterleave", Family::Volk, false),
    spec("deinterleave_cq15", Family::Volk, false),
    spec("add_fallback", Family::FallbackVsCmsis, false),
    spec("add_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("add_real_fallback", Family::FallbackVsCmsis, false),
    spec("add_real_q15_fallback", Family::FallbackVsCmsis, false),
    spec("mul_fallback", Family::FallbackVsCmsis, false),
    spec("mul_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("mul_real_fallback", Family::FallbackVsCmsis, false),
    spec("mul_real_q15_fallback", Family::FallbackVsCmsis, false),
    spec("oscillate", Family::Own, false),
    spec("oscillate_cq15", Family::Own, false),
    spec("oscillate_fallback", Family::FallbackVsCmsis, false),
    spec("oscillate_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("interleave_fallback", Family::FallbackVsVolkGeneric, false),
    spec(
        "interleave_cq15_fallback",
        Family::FallbackVsVolkGeneric,
        false,
    ),
    spec(
        "deinterleave_fallback",
        Family::FallbackVsVolkGeneric,
        false,
    ),
    spec(
        "deinterleave_cq15_fallback",
        Family::FallbackVsVolkGeneric,
        false,
    ),
    fft("fft_64", Family::FftC32, 64),
    fft("fft_1024", Family::FftC32, 1024),
    fft("fft_4096", Family::FftC32, 4096),
    fft("fft_cq15_1024", Family::FftCq15, 1024),
    fft("fft_cq15_1024_default", Family::FftCq15, 1024),
    fft("fft_cq15_1024_full_scale", Family::FftCq15, 1024),
    fft("rfft_1024", Family::RealFft, 1024),
    fft("fft_64_fallback", Family::FftFallbackVsCmsis, 64),
    fft("fft_1024_fallback", Family::FftFallbackVsCmsis, 1024),
    fft("fft_4096_fallback", Family::FftFallbackVsCmsis, 4096),
    fft(
        "fft_cq15_1024_fallback",
        Family::FftCq15FallbackVsCmsis,
        1024,
    ),
    fft(
        "fft_cq15_1024_default_fallback",
        Family::FftCq15FallbackVsCmsis,
        1024,
    ),
    fft("rfft_1024_fallback", Family::FftFallbackVsCmsis, 1024),
    xcorr("xcorr_fft_n512", Family::XcorrLiquid, 10 * 386),
    xcorr("xcorr_fft_n4096", Family::XcorrLiquid, 3074),
    xcorr(
        "xcorr_fft_n512_fallback",
        Family::FallbackVsLiquid,
        10 * 386,
    ),
    xcorr("xcorr_fft_n4096_fallback", Family::FallbackVsLiquid, 3074),
    xcorr("correlate_block", Family::Own, 10 * 386),
];

/// The CPU Intel SDE emulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdeCpu {
    /// x86-64 with SSE2 and nothing above it (own CPUID definition).
    Sse2,
    /// x86-64-v2: up to SSE4.2 and POPCNT (own CPUID definition).
    Sse42,
    /// Sapphire Rapids (`-spr`).
    SapphireRapids,
}

/// Where the measurement binary of a level runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runner {
    /// The host CPU (profiled with Valgrind callgrind).
    Host,
    /// Intel SDE with the CPU of the level.
    Sde(SdeCpu),
    /// QEMU user mode.
    Qemu,
    /// wasmtime.
    Wasmtime,
}

impl Runner {
    /// What it is, for the reports.
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            Runner::Host => "the host CPU",
            Runner::Sde(SdeCpu::Sse2) => "Intel SDE, CPUID of a strict SSE2 x86-64 CPU",
            Runner::Sde(SdeCpu::Sse42) => "Intel SDE, CPUID of an x86-64-v2 (SSE4.2) CPU",
            Runner::Sde(SdeCpu::SapphireRapids) => "Intel SDE -spr",
            Runner::Qemu => "qemu-aarch64",
            Runner::Wasmtime => "wasmtime",
        }
    }
}

/// What gives the execution counts per instruction for the simulated cycles of a level
/// (`cargo measure`, D68): the profiler of the runner of the level, in the environment the
/// binary of the level runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profiler {
    /// callgrind with `--dump-instr=yes --collect-jumps=yes`.
    Callgrind,
    /// Intel SDE `-dcfg` on the CPU of the level; on a level that runs on the host (the
    /// validation of the SDE path, `--cycle-counter sde`) on SDE's Haswell (`-hsw`, AVX2 and
    /// FMA).
    Sde,
    /// QEMU user mode with the plugin `xtask/plugins/bbcount.c`.
    Qemu,
}

impl Profiler {
    /// The profiler of the runner of `plan`, `None` for wasmtime (no core model, D68).
    #[must_use]
    pub fn of(plan: &LevelPlan) -> Option<Profiler> {
        match plan.runner {
            Runner::Host => Some(Profiler::Callgrind),
            Runner::Sde(_) => Some(Profiler::Sde),
            Runner::Qemu => Some(Profiler::Qemu),
            Runner::Wasmtime => None,
        }
    }

    /// The name in the command line (`--cycle-counter`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Profiler::Callgrind => "callgrind",
            Profiler::Sde => "sde",
            Profiler::Qemu => "qemu",
        }
    }

    /// The profiler called `name`.
    #[must_use]
    pub fn parse(name: &str) -> Option<Profiler> {
        [Profiler::Callgrind, Profiler::Sde, Profiler::Qemu]
            .into_iter()
            .find(|p| p.name() == name)
    }

    /// What it is, for the report and the JSON (`block_counts`).
    #[must_use]
    pub fn describe(self, plan: &LevelPlan) -> String {
        match (self, plan.runner) {
            (Profiler::Callgrind, _) => {
                "valgrind callgrind, execution counts per instruction and taken jumps".to_string()
            }
            (Profiler::Sde, Runner::Sde(_)) => format!(
                "Intel SDE -dcfg, block and branch counts ({})",
                plan.runner.describe()
            ),
            (Profiler::Sde, _) => {
                "Intel SDE -dcfg -hsw, block and branch counts (validation of the SDE path)"
                    .to_string()
            }
            (Profiler::Qemu, _) => {
                "qemu-aarch64 with the bbcount plugin, translation-block and transfer counts"
                    .to_string()
            }
        }
    }
}

/// The `volk_config` of a level: which implementations VOLK's dispatcher takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolkPin {
    /// An empty `volk_config`: VOLK's default dispatch on the machine it selects.
    Default,
    /// The defaults of VOLK's `sse2_64_mmx` machine.
    Sse2,
    /// The defaults of VOLK's `sse4_2_64_mmx` machine.
    Sse42,
    /// Implementations that need no more than AVX2 and FMA (no AVX-512).
    Avx2,
    /// The defaults of VOLK's `neonv8` machine (D11).
    Neon,
}

impl VolkPin {
    /// The `volk_config` text.
    #[must_use]
    pub const fn config(self) -> &'static str {
        match self {
            Self::Default => "",
            Self::Sse2 => VOLK_SSE2_CONFIG,
            Self::Sse42 => VOLK_SSE42_CONFIG,
            Self::Avx2 => VOLK_AVX2_CONFIG,
            Self::Neon => VOLK_NEON_CONFIG,
        }
    }
}

/// A level of a platform: how to build the binary, where it runs and what it is compared
/// with.
#[derive(Clone, Copy, Debug)]
pub struct LevelPlan {
    /// Platform name given to `--platform`.
    pub platform: &'static str,
    /// Level name (key of `limits.json`).
    pub level: &'static str,
    /// `cargo --target`, if cross-compiling.
    pub target: Option<&'static str>,
    /// `RUSTFLAGS`.
    pub rustflags: &'static str,
    /// Feature `std` of the kernels crate (run-time detection of the level).
    pub runtime_detect: bool,
    /// Where the binary runs.
    pub runner: Runner,
    /// The platform of the C references (`benches/refs/build.sh`).
    pub refs: &'static str,
    /// The VOLK implementations the dispatcher may choose.
    pub volk_pin: VolkPin,
    /// The judged comparison of the C32 FFT and the informational ones.
    pub fft_c32: (Reference, &'static [Reference]),
    /// The judged comparison of the real-input FFT and the informational ones.
    pub rfft: (Reference, &'static [Reference]),
    /// Informational comparisons of the CQ15 FFT next to `KissFFT`.
    pub fft_cq15_extra: &'static [Reference],
    /// KFR is built for this level and gates the f32 and C32 biquads (D15); without it
    /// CMSIS-DSP does.
    pub kfr: bool,
    /// Judged against our own numbers of this level instead of the libraries.
    pub baseline: Option<&'static str>,
}

const RUSTFFT: (Reference, &[Reference]) = (Reference::RustFft, &[]);
/// rustfft's real-input planner (`realfft`), our fallback as information.
const REALFFT: (Reference, &[Reference]) = (Reference::RustFft, &[Reference::Fallback]);

/// All levels, in the order they run.
pub const PLANS: [LevelPlan; 7] = [
    LevelPlan {
        platform: "x86_64",
        level: "sse2",
        target: None,
        rustflags: "",
        runtime_detect: false,
        runner: Runner::Sde(SdeCpu::Sse2),
        refs: "x86_64",
        volk_pin: VolkPin::Sse2,
        // rustfft has no planner below SSE4.1, so on this CPU it is scalar (D3). FFTW is
        // judged out of place, its usual use; in place it needs more passes.
        fft_c32: (
            Reference::Fftw,
            &[Reference::FftwInPlace, Reference::RustFft],
        ),
        // FFTW's real-input transform out of place, as the complex one (D3).
        rfft: (Reference::Fftw, &[Reference::RustFft, Reference::Fallback]),
        fft_cq15_extra: &[],
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "x86_64",
        level: "sse4.2",
        target: None,
        rustflags: "-C target-cpu=x86-64-v2",
        runtime_detect: false,
        runner: Runner::Sde(SdeCpu::Sse42),
        refs: "x86_64",
        volk_pin: VolkPin::Sse42,
        fft_c32: RUSTFFT,
        // FFTW has no codelets above SSE2, so it runs the code of `sse2` here, and its real-input
        // transform is the stronger reference at this level.
        rfft: (Reference::Fftw, &[Reference::RustFft, Reference::Fallback]),
        fft_cq15_extra: &[],
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "x86_64",
        level: "avx2",
        target: None,
        rustflags: "-C target-cpu=x86-64-v3",
        runtime_detect: false,
        runner: Runner::Host,
        refs: "x86_64",
        volk_pin: VolkPin::Avx2,
        fft_c32: RUSTFFT,
        rfft: REALFFT,
        fft_cq15_extra: &[],
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "x86_64",
        level: "avx2-runtime",
        target: None,
        rustflags: "",
        runtime_detect: true,
        runner: Runner::Host,
        refs: "x86_64",
        volk_pin: VolkPin::Avx2,
        fft_c32: RUSTFFT,
        rfft: REALFFT,
        fft_cq15_extra: &[],
        kfr: true,
        baseline: Some("avx2"),
    },
    LevelPlan {
        platform: "x86_64-avx512",
        level: "avx512",
        // An explicit target keeps `RUSTFLAGS` away from the build scripts, which run on the
        // host (without AVX-512) while the measurement binary runs under SDE.
        target: Some("x86_64-unknown-linux-gnu"),
        rustflags: "-C target-cpu=sapphirerapids",
        runtime_detect: false,
        runner: Runner::Sde(SdeCpu::SapphireRapids),
        refs: "x86_64",
        volk_pin: VolkPin::Default,
        fft_c32: RUSTFFT,
        rfft: REALFFT,
        fft_cq15_extra: &[],
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "aarch64",
        level: "neon",
        target: Some("aarch64-unknown-linux-gnu"),
        rustflags: "",
        runtime_detect: false,
        runner: Runner::Qemu,
        refs: "aarch64",
        volk_pin: VolkPin::Neon,
        fft_c32: (Reference::RustFft, &[Reference::Fftw]),
        rfft: (Reference::RustFft, &[Reference::Fftw, Reference::Fallback]),
        fft_cq15_extra: &[Reference::Ne10],
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "wasm32",
        level: "simd128",
        target: Some("wasm32-wasip1"),
        rustflags: "-C target-feature=+simd128",
        runtime_detect: false,
        runner: Runner::Wasmtime,
        refs: "wasm32",
        volk_pin: VolkPin::Default,
        fft_c32: RUSTFFT,
        rfft: REALFFT,
        fft_cq15_extra: &[],
        // KFR in its Emscripten configuration (sse41 over SIMD128, benches/refs/build.sh, D29).
        kfr: true,
        baseline: None,
    },
];

/// The judged comparison of a kernel on a level and the informational ones.
#[must_use]
pub fn comparisons(plan: &LevelPlan, spec: &KernelSpec) -> (Reference, Vec<Reference>) {
    if let Some(level) = plan.baseline {
        return (Reference::OwnLevel(level), Vec::new());
    }
    let (gate, mut extras) = family_comparisons(plan, spec);
    if let Some(own) = spec.own {
        extras.push(Reference::OwnKernel(own));
    }
    (gate, extras)
}

fn family_comparisons(plan: &LevelPlan, spec: &KernelSpec) -> (Reference, Vec<Reference>) {
    match spec.family {
        Family::VolkNative if plan.refs == "wasm32" => (Reference::Fallback, Vec::new()),
        Family::VolkNoFma if matches!(plan.level, "avx2" | "avx512" | "neon") => {
            (Reference::Volk, vec![Reference::VolkNoFma])
        }
        Family::VolkNoFmaGate if matches!(plan.level, "avx2" | "avx512" | "neon") => {
            (Reference::VolkNoFma, vec![Reference::Volk])
        }
        Family::Volk
        | Family::VolkAndOwn
        | Family::VolkNative
        | Family::VolkNoFma
        | Family::VolkNoFmaGate => {
            let extras = if spec.family == Family::VolkAndOwn {
                vec![Reference::Fallback]
            } else {
                Vec::new()
            };
            (Reference::Volk, extras)
        }
        Family::GnuRadio => (Reference::GnuRadio, vec![Reference::Fallback]),
        Family::FftC32 => (plan.fft_c32.0, plan.fft_c32.1.to_vec()),
        Family::FftCq15 => (Reference::KissFft, plan.fft_cq15_extra.to_vec()),
        Family::RealFft => (plan.rfft.0, plan.rfft.1.to_vec()),
        Family::Own | Family::Workload | Family::FallbackOwn => (Reference::Fallback, Vec::new()),
        Family::Kfr if plan.kfr => (Reference::Kfr, vec![Reference::Cmsis]),
        // Without KFR at the level, CMSIS-DSP gates, as it does for a fallback entry.
        Family::Kfr | Family::FallbackVsCmsis | Family::FftFallbackVsCmsis => {
            (Reference::Cmsis, Vec::new())
        }
        Family::FftCq15FallbackVsCmsis => (Reference::Cmsis, vec![Reference::KissFft]),
        Family::FallbackVsVolkGeneric => (Reference::VolkGeneric, Vec::new()),
        Family::Q15Own => (Reference::Fallback, vec![Reference::Cmsis]),
        Family::FallbackVsGnuRadio => (Reference::GnuRadio, vec![Reference::Cmsis]),
        // libairspy's SSE2 converter is the second gate at the level it is written for: the
        // entry `real_to_complex_airspy`.
        Family::FallbackVsLiquid | Family::HalfbandLiquid | Family::HilbertLiquid => {
            (Reference::Liquid, Vec::new())
        }
        Family::ChannelizeLiquid | Family::XcorrLiquid | Family::LiquidDifferentWork => {
            (Reference::Liquid, vec![Reference::Fallback])
        }
        Family::GaussNoiseFastnoise | Family::ScrambleDvbs2 => (Reference::GnuRadio, Vec::new()),
        Family::GaussNoiseZiggurat => (Reference::RandDistr, Vec::new()),
        Family::ScrambleCmsis => (Reference::Cmsis, vec![Reference::Fallback]),
        Family::HilbertAirspyQ15 => (Reference::Libairspy, vec![Reference::Fallback]),
        Family::FallbackHilbert => (Reference::Liquid, vec![Reference::Libairspy]),
        Family::HilbertAirspy | Family::FallbackVsAirspy => (Reference::Libairspy, Vec::new()),
    }
}

/// The defaults of VOLK 3.3's `sse2_64_mmx` machine for the level `sse2` (the strict SSE2 CPU
/// of SDE selects that machine). VOLK has no SSE2 implementation of `rotator2`, `atan2`,
/// `multiply_conjugate`, `8i_s32f_convert_32f` and `8i_convert_16i`: their portable `generic`
/// code remains. The compiler vectorizes that of the two conversions with SSE2, a fair reference; `rotator2` and
/// `atan2` have no SSE implementation at all and their `generic` code is scalar, so those entries
/// are `info`. `volk_32fc_x2_dot_prod_32fc` starts at SSE3, so its scalar `generic` code
/// remains as well, and `dot_complex` is `info` at this level (D821).
const VOLK_SSE2_CONFIG: &str = "\
volk_8i_s32f_convert_32f generic generic
volk_16i_s32f_convert_32f a_sse u_sse
volk_8i_convert_16i generic generic
volk_32f_s32f_convert_16i a_sse2 u_sse2
volk_32f_s32f_convert_8i a_sse2 u_sse2
volk_32f_s32f_x2_convert_8u a_sse2 u_sse2
volk_32fc_s32fc_x2_rotator2_32fc generic generic
volk_32fc_32f_dot_prod_32fc a_sse u_sse
volk_32fc_magnitude_squared_32f a_sse u_sse
volk_32fc_magnitude_32f a_sse u_sse
volk_16ic_magnitude_16i a_sse generic
volk_32fc_x2_multiply_conjugate_32fc generic generic
volk_32fc_s32f_atan2_32f generic generic
volk_32fc_32f_multiply_32fc a_sse generic
volk_32f_log2_32f generic generic
volk_32f_s32f_multiply_32f a_sse u_sse
volk_32f_s32f_add_32f a_sse u_sse
volk_32f_x2_dot_prod_32f a_sse u_sse
volk_32fc_x2_dot_prod_32fc generic generic
volk_32fc_deinterleave_32f_x2 a_sse generic
volk_32f_x2_add_32f a_sse u_sse
volk_32f_x2_subtract_32f a_sse generic
volk_32f_x2_interleave_32fc a_sse generic
volk_32fc_x2_add_32fc a_sse u_sse
volk_32f_x2_multiply_32f a_sse u_sse
volk_32fc_x2_multiply_32fc generic generic
volk_16i_x2_add_saturated_16i a_sse2 u_sse2
volk_16ic_deinterleave_16i_x2 a_sse2 generic
";

/// The defaults of VOLK 3.3's `sse4_2_64_mmx` machine for the level `sse4.2`.
const VOLK_SSE42_CONFIG: &str = "\
volk_8i_s32f_convert_32f a_sse4_1 u_sse4_1
volk_16i_s32f_convert_32f a_sse4_1 u_sse4_1
volk_8i_convert_16i a_sse4_1 u_sse4_1
volk_32f_s32f_convert_16i a_sse2 u_sse2
volk_32f_s32f_convert_8i a_sse2 u_sse2
volk_32f_s32f_x2_convert_8u a_sse2 u_sse2
volk_32fc_s32fc_x2_rotator2_32fc generic generic
volk_32fc_32f_dot_prod_32fc a_sse u_sse
volk_32fc_magnitude_squared_32f a_sse3 u_sse3
volk_32fc_magnitude_32f a_sse3 u_sse3
volk_16ic_magnitude_16i a_sse3 generic
volk_32fc_x2_multiply_conjugate_32fc a_sse3 u_sse3
volk_32fc_s32f_atan2_32f generic generic
volk_32fc_32f_multiply_32fc a_sse generic
volk_32f_log2_32f a_sse4_1 u_sse4_1
volk_32f_s32f_multiply_32f a_sse u_sse
volk_32f_s32f_add_32f a_sse u_sse
volk_32f_x2_dot_prod_32f a_sse4_1 u_sse4_1
volk_32fc_x2_dot_prod_32fc a_sse3 u_sse3
volk_32fc_deinterleave_32f_x2 a_sse generic
volk_32f_x2_add_32f a_sse u_sse
volk_32f_x2_subtract_32f a_sse generic
volk_32f_x2_interleave_32fc a_sse generic
volk_32fc_x2_add_32fc a_sse u_sse
volk_32f_x2_multiply_32f a_sse u_sse
volk_32fc_x2_multiply_32fc a_sse3 u_sse3
volk_16i_x2_add_saturated_16i a_sse2 u_sse2
volk_16ic_deinterleave_16i_x2 a_ssse3 generic
";

/// The VOLK implementations that need no more than AVX2 and FMA, for the levels `avx2` and
/// `avx2-runtime`. Without this, a runner with AVX-512 makes VOLK dispatch to its AVX-512
/// implementations (`rotator2`, `atan2`, `dot_prod`, the conversions have them) and the AVX2
/// build of our kernels would be compared with a different instruction set than on a machine
/// without AVX-512.
const VOLK_AVX2_CONFIG: &str = "\
volk_8i_s32f_convert_32f a_avx2 u_avx2
volk_16i_s32f_convert_32f a_avx2 u_avx2
volk_8i_convert_16i a_avx2 u_avx2
volk_32f_s32f_convert_16i a_avx2 u_avx2
volk_32f_s32f_convert_8i a_avx2 u_avx2
volk_32f_s32f_x2_convert_8u a_avx2_fma u_avx2_fma
volk_32fc_s32fc_x2_rotator2_32fc a_avx u_avx
volk_32fc_32f_dot_prod_32fc a_avx2_fma u_avx2_fma
volk_32fc_magnitude_squared_32f a_avx u_avx
volk_32fc_magnitude_32f a_avx u_avx
volk_16ic_magnitude_16i a_avx2 u_avx2
volk_32fc_x2_multiply_conjugate_32fc a_avx u_avx
volk_32fc_s32f_atan2_32f a_avx2_fma u_avx2_fma
volk_32fc_32f_multiply_32fc a_avx generic
volk_32f_log2_32f a_avx2_fma u_avx2_fma
volk_32f_s32f_multiply_32f a_avx u_avx
volk_32f_s32f_add_32f a_avx u_avx
volk_32f_x2_dot_prod_32f a_avx2_fma u_avx2_fma
volk_32fc_x2_dot_prod_32fc a_avx_fma u_avx_fma
volk_32fc_deinterleave_32f_x2 a_avx u_avx
volk_32f_x2_add_32f a_avx u_avx
volk_32f_x2_subtract_32f a_avx u_avx
volk_32f_x2_interleave_32fc a_avx u_avx
volk_32fc_x2_add_32fc a_avx u_avx
volk_32f_x2_multiply_32f a_avx u_avx
volk_32fc_x2_multiply_32fc a_avx2_fma u_avx2_fma
volk_16i_x2_add_saturated_16i a_avx2 u_avx2
volk_16ic_deinterleave_16i_x2 a_avx2 u_avx2
";

/// The default dispatch of VOLK 3.3's `neonv8` machine (what QEMU's CPU selects), pinned so
/// that a change of the machine or of VOLK's ranking shows up as a missing implementation
/// instead of a silently different reference (D11). The aligned dot product of the FIR family
/// is `a_neon`: `neonv8` needs no alignment, and the aligned entry point prefers aligned
/// implementations.
const VOLK_NEON_CONFIG: &str = "\
volk_8i_s32f_convert_32f neonv8 neonv8
volk_16i_s32f_convert_32f neonv8 neonv8
volk_8i_convert_16i neon neon
volk_32f_s32f_convert_16i neonv8 neonv8
volk_32f_s32f_convert_8i neonv8 neonv8
volk_32f_s32f_x2_convert_8u neonv8 neonv8
volk_32fc_s32fc_x2_rotator2_32fc neon neon
volk_32fc_32f_dot_prod_32fc a_neon neonv8
volk_32fc_magnitude_squared_32f neonv8 neonv8
volk_32fc_magnitude_32f neonv8 neonv8
volk_16ic_magnitude_16i neonv8 neonv8
volk_32fc_x2_multiply_conjugate_32fc neonv8 neonv8
volk_32fc_s32f_atan2_32f neonv8 neonv8
volk_32fc_32f_multiply_32fc neonv8 neonv8
volk_32f_log2_32f neonv8 neonv8
volk_32f_s32f_multiply_32f neonv8 neonv8
volk_32f_s32f_add_32f neonv8 neonv8
volk_32f_x2_dot_prod_32f neonv8 neonv8
volk_32fc_x2_dot_prod_32fc neonv8 neonv8
volk_32fc_deinterleave_32f_x2 neonv8 neonv8
volk_32f_x2_add_32f neonv8 neonv8
volk_32f_x2_subtract_32f neonv8 neonv8
volk_32f_x2_interleave_32fc neonv8 neonv8
volk_32fc_x2_add_32fc neonv8 neonv8
volk_32f_x2_multiply_32f neonv8 neonv8
volk_32fc_x2_multiply_32fc neonv8 neonv8
volk_16i_x2_add_saturated_16i neonv8 neonv8
volk_16ic_deinterleave_16i_x2 neonv8 neonv8
";

/// The VOLK functions whose names VOLK reads as environment variables to override its
/// dispatch (`volk_rank_archs`); the measured processes run without them and without
/// `VOLK_GENERIC`.
const VOLK_FUNCTIONS: [&str; 28] = [
    "volk_8i_s32f_convert_32f",
    "volk_16i_s32f_convert_32f",
    "volk_8i_convert_16i",
    "volk_32f_s32f_convert_16i",
    "volk_32f_s32f_convert_8i",
    "volk_32f_s32f_x2_convert_8u",
    "volk_32fc_s32fc_x2_rotator2_32fc",
    "volk_32fc_32f_dot_prod_32fc",
    "volk_32fc_magnitude_squared_32f",
    "volk_32fc_magnitude_32f",
    "volk_16ic_magnitude_16i",
    "volk_32fc_x2_multiply_conjugate_32fc",
    "volk_32fc_s32f_atan2_32f",
    "volk_32fc_32f_multiply_32fc",
    "volk_32f_log2_32f",
    "volk_32f_s32f_multiply_32f",
    "volk_32f_s32f_add_32f",
    "volk_32f_x2_dot_prod_32f",
    "volk_32fc_x2_dot_prod_32fc",
    "volk_32fc_deinterleave_32f_x2",
    "volk_32f_x2_add_32f",
    "volk_32f_x2_subtract_32f",
    "volk_32f_x2_interleave_32fc",
    "volk_32fc_x2_add_32fc",
    "volk_32f_x2_multiply_32f",
    "volk_32fc_x2_multiply_32fc",
    "volk_16i_x2_add_saturated_16i",
    "volk_16ic_deinterleave_16i_x2",
];

/// CPUID of SDE for the level `sse2`: `GenuineIntel`, family 15, SSE2 and nothing above it; XSAVE
/// with x87 and SSE state only. The XSAVE bits matter: on a CPU without OSXSAVE, the
/// `cpu_features` library inside VOLK falls back to the host's `/proc/cpuinfo` for the SSE
/// flags, and VOLK would select a machine the emulated CPU does not have (SDE's own `-p4p`
/// gives the `sse4_1` machine; `-p4` does not run 64-bit code).
const CPUID_SSE2: &str = "\
# CPUID_VERSION = 1.0
# x86-64 with SSE2 and nothing above it, for cargo measure.
00000000 ******** => 0000000d 756e6547 6c65746e 49656e69
00000001 ******** => 00000f68 00010800 0c000000 078bfbff
0000000d 00000000 => 00000003 00000240 00000240 00000000
0000000d 00000001 => 00000000 00000000 00000000 00000000
80000000 ******** => 80000001 00000000 00000000 00000000
80000001 ******** => 00000000 00000000 00000001 20100800
";

/// CPUID of SDE for the level `sse4.2`: x86-64-v2 (SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CX16,
/// LAHF), with XSAVE for x87 and SSE state as in [`CPUID_SSE2`].
const CPUID_SSE42: &str = "\
# CPUID_VERSION = 1.0
# x86-64-v2 (up to SSE4.2 and POPCNT), for cargo measure.
00000000 ******** => 0000000d 756e6547 6c65746e 49656e69
00000001 ******** => 000106a5 00010800 0c982201 078bfbff
0000000d 00000000 => 00000003 00000240 00000240 00000000
0000000d 00000001 => 00000000 00000000 00000000 00000000
80000000 ******** => 80000001 00000000 00000000 00000000
80000001 ******** => 00000000 00000000 00000001 20100800
";

/// Writes the CPUID definition of the level `sse2` or `sse4.2` for SDE's `-cpuid_in` to `path`
/// (`cargo xtask sde-cpuid`), so that CI runs the tests on the CPU the cycle profiles use.
///
/// # Errors
/// Fails on another level or if the file cannot be written.
pub fn write_cpuid(level: &str, path: &Path) -> anyhow::Result<bool> {
    let text = match level {
        "sse2" => CPUID_SSE2,
        "sse4.2" => CPUID_SSE42,
        other => bail!("no CPUID definition for the level `{other}` (sse2 or sse4.2)"),
    };
    std::fs::write(path, text).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(true)
}

// ---------------------------------------------------------------------------------------------
// Runs of the measurement binary

/// Environment and tools of the runs of the measurement binary of one level.
pub(crate) struct Session {
    pub(crate) plan: LevelPlan,
    pub(crate) binary: PathBuf,
    /// `VOLK_CONFIGPATH` with the pinned `volk_config` of the level.
    pub(crate) volk_config: PathBuf,
    /// The `lib` directory of the C references (VOLK is a shared library there).
    refs_lib: PathBuf,
    /// The CPUID definition of SDE, for the levels that need one.
    cpuid: Option<PathBuf>,
    /// Scratch directory of the level (the profiles of the runs).
    scratch: PathBuf,
    /// QEMU user mode and the sysroot of the target libc.
    qemu: Option<(PathBuf, PathBuf)>,
    /// `target/tools` of the work tree (the QEMU plugins).
    tools: PathBuf,
    /// The source of the block-count plugin of QEMU (`xtask/plugins/bbcount.c`).
    bbcount_source: PathBuf,
}

/// The arguments of `count run` with `k` calls, written with as many digits as `calls`: both
/// runs of a measurement then have arguments of the same length. Their length moves the
/// start-up code of the process by a few instructions, and only identical start-ups cancel
/// exactly (otherwise the length of the environment decided the last digit of some counts).
pub(crate) fn run_args(
    kernel: &str,
    which: &str,
    k: u64,
    calls: u64,
    variant: &Variant,
) -> Vec<String> {
    let width = calls.to_string().len();
    let mut args: Vec<String> = ["run", kernel, which, &format!("{k:0width$}")]
        .iter()
        .map(ToString::to_string)
        .collect();
    args.extend(variant.options());
    args
}

/// One run of the measurement binary: the arguments and the `volk_config` directory.
pub(crate) struct Run<'a> {
    pub(crate) args: Vec<String>,
    pub(crate) volk_config: &'a Path,
}

impl Session {
    /// The command for `program` with the environment of the measured process: the pinned
    /// `volk_config`, the library path of the references, and without the variables that
    /// override VOLK's dispatch.
    pub(crate) fn command(&self, program: impl AsRef<std::ffi::OsStr>, run: &Run<'_>) -> Command {
        let mut cmd = Command::new(program);
        cmd.env("VOLK_CONFIGPATH", run.volk_config)
            .env_remove("VOLK_GENERIC");
        for f in VOLK_FUNCTIONS {
            cmd.env_remove(f);
        }
        if matches!(self.plan.runner, Runner::Host | Runner::Sde(_)) {
            cmd.env("LD_LIBRARY_PATH", &self.refs_lib);
        }
        if matches!(self.plan.runner, Runner::Qemu) {
            // qemu-user reads every `QEMU_<OPTION>` variable as a command-line option
            // (`QEMU_VERSION` is `-version`, which prints the version and runs nothing), so
            // none may reach it from the caller's environment.
            for (name, _) in std::env::vars_os() {
                if is_qemu_option_variable(&name) {
                    cmd.env_remove(name);
                }
            }
        }
        cmd
    }

    /// The emulator in front of the binary: program and arguments.
    fn launcher(&self) -> anyhow::Result<(PathBuf, Vec<std::ffi::OsString>)> {
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        let program = match self.plan.runner {
            Runner::Host => return Ok((self.binary.clone(), Vec::new())),
            Runner::Sde(_) => {
                args.extend(self.sde_cpu_args()?);
                args.push("--".into());
                sde()
            }
            Runner::Qemu => {
                let (qemu, sysroot) = self.qemu.clone().ok_or_else(|| anyhow!("qemu"))?;
                args.extend(["-R", QEMU_RESERVED_VA, "-L"].map(Into::into));
                args.push(sysroot.into());
                args.push("-E".into());
                args.push(format!("LD_LIBRARY_PATH={}", self.refs_lib.display()).into());
                qemu
            }
            Runner::Wasmtime => {
                args.push("run".into());
                PathBuf::from("wasmtime")
            }
        };
        args.push(self.binary.clone().into());
        Ok((program, args))
    }

    /// The CPU options of SDE for this level: `-spr` for `avx512`, the CPUID definition of
    /// `sse2` and `sse4.2` with its check, and `-hsw` for a level that runs on the host (the
    /// validation of the SDE profile).
    fn sde_cpu_args(&self) -> anyhow::Result<Vec<std::ffi::OsString>> {
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        match (self.plan.runner, &self.cpuid) {
            (Runner::Sde(SdeCpu::SapphireRapids), _) => args.push("-spr".into()),
            (Runner::Sde(_), Some(def)) => {
                args.push("-cpuid_in".into());
                args.push(def.into());
                // Any instruction above the CPUID stops the run: no library may use
                // code of a higher level than the one it is compared at.
                args.extend(["-chip_check_cpuid", "1"].map(Into::into));
            }
            (Runner::Sde(cpu), None) => bail!("no CPUID definition for {cpu:?}"),
            (Runner::Host, _) => args.push("-hsw".into()),
            (runner, _) => bail!(
                "the level {} runs on {runner:?}, not on x86",
                self.plan.level
            ),
        }
        Ok(args)
    }

    /// Runs the measurement binary without a profiler and returns its standard output.
    pub(crate) fn plain(&self, run: &Run<'_>) -> anyhow::Result<String> {
        let (program, args) = self.launcher()?;
        let out = self
            .command(program, run)
            .args(args)
            .args(&run.args)
            .stderr(Stdio::inherit())
            .output()
            .context("cannot run the measurement binary")?;
        if !out.status.success() {
            bail!("the measurement binary failed: {:?}", run.args);
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// The execution counts per instruction of `which` on `kernel` from `profiler`, for level 2
    /// of the measurement policy: the counts of a run with `calls` calls minus those of a run
    /// with none, so that only the counted calls remain (the differential method of
    /// the instruction counts of [`crate::count`], per instruction). The runs have the environment
    /// of the level ([`Session::command`]: the pinned `volk_config`, the references; the CPU of
    /// SDE, the QEMU options).
    fn profile(
        &self,
        kernel: &str,
        which: &str,
        variant: &Variant,
        profiler: Profiler,
    ) -> anyhow::Result<crate::mca::Counts> {
        let calls = K * (N / variant.samples).max(1);
        match profiler {
            Profiler::Callgrind => self.callgrind_profile(kernel, which, variant),
            Profiler::Sde => {
                let full = self.sde_blocks(kernel, which, calls, calls, variant)?;
                let empty = self.sde_blocks(kernel, which, 0, calls, variant)?;
                crate::blockcount::expand(&crate::blockcount::difference(&full, &empty))
            }
            Profiler::Qemu => {
                let full = self.qemu_blocks(kernel, which, calls, calls, variant)?;
                let empty = self.qemu_blocks(kernel, which, 0, calls, variant)?;
                Ok(crate::mca::difference(&full, &empty))
            }
        }
    }

    /// The block counts of one run under SDE `-dcfg` (every image of the process, between the
    /// SSC markers of the counted calls).
    fn sde_blocks(
        &self,
        kernel: &str,
        which: &str,
        k: u64,
        calls: u64,
        variant: &Variant,
    ) -> anyhow::Result<crate::blockcount::BlockCounts> {
        let base = self.scratch.join(format!("dcfg-{kernel}-{which}-{k}"));
        let mut file = base.clone().into_os_string();
        file.push(".dcfg.json");
        let file = PathBuf::from(file);
        // SDE would read an existing DCFG of the same name as the basis of the new one.
        match std::fs::remove_file(&file) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(e).context("cannot remove an old DCFG");
            }
            _ => {}
        }
        let run = Run {
            args: run_args(kernel, which, k, calls, variant),
            volk_config: &self.volk_config,
        };
        let out = self
            .command(sde(), &run)
            .args(self.sde_cpu_args()?)
            // Only the region between the SSC markers: outside it the
            // process does work that depends on the number of calls (the check of the
            // output), which the measurement leaves out.
            .args(SDE_REGION)
            .args(["-dcfg", "-dcfg:out_base_name"])
            .arg(&base)
            .args([
                "-dcfg:dcfg_file_suffix",
                ".dcfg.json",
                "-dcfg:read_dcfg_if_avail",
                "0",
                "-dcfg:analysis",
                "0",
                "-dcfg:source",
                "0",
                "-dcfg:symbols",
                "0",
                "--",
            ])
            .arg(&self.binary)
            .args(&run.args)
            .stdout(Stdio::null())
            .output()
            .context("cannot run sde64")?;
        if !out.status.success() {
            bail!(
                "sde64 -dcfg failed for {:?}: {}",
                run.args,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("SDE wrote no DCFG {}", file.display()))?;
        let blocks = crate::blockcount::parse_dcfg(&text, |object| {
            crate::blockcount::elf(object).map_or(0, |e| e.link_base())
        })?;
        let _ = std::fs::remove_file(&file);
        Ok(blocks)
    }

    /// The instruction counts of one run under QEMU with the plugin `bbcount`.
    fn qemu_blocks(
        &self,
        kernel: &str,
        which: &str,
        k: u64,
        calls: u64,
        variant: &Variant,
    ) -> anyhow::Result<crate::mca::Counts> {
        let (qemu, sysroot) = self
            .qemu
            .clone()
            .ok_or_else(|| anyhow!("the level {} is not counted under QEMU", self.plan.level))?;
        let plugin = bbcount_plugin(&self.tools, &self.bbcount_source)?;
        let log = self
            .scratch
            .join(format!("bbcount-{kernel}-{which}-{k}.txt"));
        let run = Run {
            args: run_args(kernel, which, k, calls, variant),
            volk_config: &self.volk_config,
        };
        let out = self
            .command(&qemu, &run)
            .args(["-R", QEMU_RESERVED_VA, "-L"])
            .arg(&sysroot)
            .arg("-E")
            .arg(format!("LD_LIBRARY_PATH={}", self.refs_lib.display()))
            .arg("-plugin")
            .arg(&plugin)
            .args(["-d", "plugin", "-D"])
            .arg(&log)
            .arg(&self.binary)
            .args(&run.args)
            .stdout(Stdio::null())
            .output()
            .context("cannot run qemu-aarch64")?;
        if !out.status.success() {
            bail!(
                "qemu-aarch64 with bbcount failed for {:?}: {}",
                run.args,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let text = std::fs::read_to_string(&log)
            .with_context(|| format!("the bbcount plugin wrote no {}", log.display()))?;
        let counts = crate::blockcount::parse_bbcount(&text, |object, offset| {
            crate::blockcount::elf(object)
                .ok()?
                .address_of_offset(offset)
        })?;
        let _ = std::fs::remove_file(&log);
        Ok(counts)
    }

    /// [`Session::profile`] under callgrind.
    fn callgrind_profile(
        &self,
        kernel: &str,
        which: &str,
        variant: &Variant,
    ) -> anyhow::Result<crate::mca::Counts> {
        let calls = K * (N / variant.samples).max(1);
        let mut runs = Vec::new();
        for k in [0, calls] {
            let file = self
                .scratch
                .join(format!("mca-{kernel}-{which}-{k}.callgrind"));
            let run = Run {
                args: run_args(kernel, which, k, calls, variant),
                volk_config: &self.volk_config,
            };
            let out = self
                .command("valgrind", &run)
                .args([
                    "-q",
                    "--tool=callgrind",
                    "--dump-instr=yes",
                    "--collect-jumps=yes",
                    "--compress-pos=no",
                    "--compress-strings=no",
                ])
                .arg(format!("--callgrind-out-file={}", file.display()))
                .arg(&self.binary)
                .args(&run.args)
                .stdout(Stdio::null())
                .output()
                .context("cannot run valgrind")?;
            if !out.status.success() {
                bail!(
                    "callgrind failed for {:?}: {}",
                    run.args,
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("cannot read {}", file.display()))?;
            runs.push(crate::mca::parse_callgrind(&text));
        }
        let [empty, full] = <[crate::mca::Counts; 2]>::try_from(runs)
            .map_err(|_| anyhow!("two callgrind runs expected"))?;
        Ok(crate::mca::difference(&full, &empty))
    }

    /// The branch counts of the counted calls of `which` on `kernel` from Valgrind's
    /// `cachegrind --branch-sim=yes` (D1611): a run with the calls minus a run without them, in
    /// the environment of the runs of the level ([`Session::command`]). The binary runs on
    /// the host CPU; only the build fixes the level of our code, and the pinned `volk_config`
    /// that of VOLK.
    fn branch_sim(
        &self,
        kernel: &str,
        which: &str,
        variant: &Variant,
    ) -> anyhow::Result<crate::branches::Branches> {
        let calls = K * (N / variant.samples).max(1);
        let mut runs = Vec::new();
        for k in [0, calls] {
            let file = self
                .scratch
                .join(format!("branches-{kernel}-{which}-{k}.cachegrind"));
            let run = Run {
                args: run_args(kernel, which, k, calls, variant),
                volk_config: &self.volk_config,
            };
            let out = self
                .command("valgrind", &run)
                .args([
                    "-q",
                    "--tool=cachegrind",
                    "--cache-sim=no",
                    "--branch-sim=yes",
                ])
                .arg(format!("--cachegrind-out-file={}", file.display()))
                .arg(&self.binary)
                .args(&run.args)
                .stdout(Stdio::null())
                .output()
                .context("cannot run valgrind")?;
            if !out.status.success() {
                bail!(
                    "cachegrind failed for {:?}: {}",
                    run.args,
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("cannot read {}", file.display()))?;
            let _ = std::fs::remove_file(&file);
            runs.push(crate::branches::parse_cachegrind(&text)?);
        }
        let [empty, full] = <[crate::branches::Branches; 2]>::try_from(runs)
            .map_err(|_| anyhow!("two cachegrind runs expected"))?;
        full.minus(&empty)
    }

    /// A JSON line of the measurement binary (`info`, `volk-info`, `volk-impls`).
    pub(crate) fn query(&self, args: &[&str]) -> anyhow::Result<Value> {
        let out = self.plain(&Run {
            args: args.iter().map(ToString::to_string).collect(),
            volk_config: &self.volk_config,
        })?;
        serde_json::from_str(out.trim()).with_context(|| format!("output of {args:?}: {out}"))
    }
}

/// The buffers of one measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Variant {
    /// Samples per call.
    pub(crate) samples: u64,
    /// Misaligned by one sample.
    pub(crate) misaligned: bool,
    /// Whether `--n` is passed (the FFT sizes are fixed by the kernel).
    pub(crate) explicit_n: bool,
}

impl Variant {
    /// The standard buffers of `spec`.
    pub(crate) fn standard(spec: &KernelSpec) -> Self {
        Variant {
            samples: spec.samples,
            misaligned: false,
            explicit_n: false,
        }
    }

    fn options(&self) -> Vec<String> {
        let mut o = Vec::new();
        if self.explicit_n {
            o.extend(["--n".to_string(), self.samples.to_string()]);
        }
        if self.misaligned {
            o.push("--misaligned".to_string());
        }
        o
    }

    /// The label of an informational variant.
    pub(crate) fn label(&self) -> String {
        match (self.misaligned, self.explicit_n) {
            (true, _) => "misaligned by one sample".to_string(),
            (false, true) => format!("n = {}", self.samples),
            (false, false) => String::new(),
        }
    }
}

/// Whether `name` is a variable that qemu-user would read as one of its options.
fn is_qemu_option_variable(name: &std::ffi::OsStr) -> bool {
    name.to_str().is_some_and(|n| n.starts_with("QEMU_"))
}

fn sde() -> PathBuf {
    std::env::var_os("SDRK_SDE").map_or_else(|| PathBuf::from("sde64"), PathBuf::from)
}

/// The QEMU of the `neon` runs (a build with plugins) and the sysroot of the target libc. See
/// `docs/performance.md` for how to set them up.
fn qemu_tools() -> (PathBuf, PathBuf) {
    let qemu = std::env::var_os("SDRK_QEMU_AARCH64")
        .map_or_else(|| PathBuf::from("qemu-aarch64"), PathBuf::from);
    let sysroot = std::env::var_os("SDRK_AARCH64_SYSROOT")
        .map_or_else(|| PathBuf::from("/usr/aarch64-linux-gnu"), PathBuf::from);
    (qemu, sysroot)
}

/// Builds the `insn` plugin of the QEMU tree into `target/tools/`. The source is taken from the
/// tree named by `SDRK_QEMU_SRC` (needs `include/qemu/qemu-plugin.h` and
/// `tests/tcg/plugins/insn.c`; the location has moved between QEMU releases) or downloaded
/// from the QEMU repository at the tag `SDRK_QEMU_TAG` (default `v10.2.1`). Needs a C compiler
/// and the glib development files (`libglib2.0-dev`, `pkg-config`).
pub(crate) fn build_insn_plugin(tools: &Path, plugin: &Path) -> anyhow::Result<()> {
    let src_dir = qemu_source(
        tools,
        &["tests/tcg/plugins/insn.c", "include/qemu/qemu-plugin.h"],
        true,
    )?;
    compile_plugin(&src_dir, &src_dir.join("tests/tcg/plugins/insn.c"), plugin)
}

/// The block-count plugin of QEMU (`xtask/plugins/bbcount.c`) built into `tools`, against the
/// `qemu-plugin.h` of the QEMU tree of [`build_insn_plugin`]. It is rebuilt when the source
/// changes (its SHA-256 is kept next to the library).
///
/// # Errors
/// The source cannot be read, the header cannot be fetched or the compiler fails.
pub(crate) fn bbcount_plugin(tools: &Path, source: &Path) -> anyhow::Result<PathBuf> {
    use sha2::Digest as _;
    let plugin = tools.join("libbbcount.so");
    let stamp = tools.join("libbbcount.so.sha256");
    let text =
        std::fs::read(source).with_context(|| format!("cannot read {}", source.display()))?;
    let want = format!("{:x}", sha2::Sha256::digest(&text));
    if plugin.exists() && std::fs::read_to_string(&stamp).is_ok_and(|s| s.trim() == want) {
        return Ok(plugin);
    }
    let src_dir = qemu_source(tools, &["include/qemu/qemu-plugin.h"], false)?;
    compile_plugin(&src_dir, source, &plugin)?;
    std::fs::write(&stamp, want)?;
    Ok(plugin)
}

/// The QEMU source tree with `files`: `SDRK_QEMU_SRC`, or the files downloaded into
/// `tools/qemu-src` from the tag `SDRK_QEMU_TAG` (default `v10.2.1`, the QEMU of the
/// development container); with `refresh` they are downloaded even if present.
fn qemu_source(tools: &Path, files: &[&str], refresh: bool) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(tools)?;
    if let Some(dir) = std::env::var_os("SDRK_QEMU_SRC") {
        Ok(PathBuf::from(dir))
    } else {
        let tag = std::env::var("SDRK_QEMU_TAG").unwrap_or_else(|_| "v10.2.1".to_string());
        let dir = tools.join("qemu-src");
        for file in files {
            let dest = dir.join(file);
            if !refresh && dest.exists() {
                continue;
            }
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let url = format!("https://raw.githubusercontent.com/qemu/qemu/{tag}/{file}");
            let status = Command::new("curl")
                .args(["-sfL", &url, "-o"])
                .arg(&dest)
                .status()
                .context("cannot run curl to fetch the QEMU plugin source")?;
            if !status.success() {
                bail!("cannot download {url}; set SDRK_QEMU_SRC to a QEMU source tree");
            }
        }
        Ok(dir)
    }
}

/// Compiles the QEMU plugin `source` into `plugin` against the headers of `src_dir`. Needs a C
/// compiler and the glib development files (`libglib2.0-dev`, `pkg-config`).
fn compile_plugin(src_dir: &Path, source: &Path, plugin: &Path) -> anyhow::Result<()> {
    let glib = Command::new("pkg-config")
        .args(["--cflags", "glib-2.0"])
        .output()
        .context("pkg-config is needed to build the QEMU plugin")?;
    if !glib.status.success() {
        bail!("pkg-config cannot find glib-2.0 (install libglib2.0-dev)");
    }
    let cflags = String::from_utf8_lossy(&glib.stdout).into_owned();
    let status = Command::new("gcc")
        .args(["-shared", "-fPIC", "-O2", "-o"])
        .arg(plugin)
        .arg(source)
        .arg(format!("-I{}", src_dir.join("include/qemu").display()))
        .arg(format!("-I{}", src_dir.join("include").display()))
        .args(cflags.split_whitespace())
        .status()
        .context("cannot run gcc")?;
    if !status.success() {
        bail!("building the QEMU plugin {} failed", source.display());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Driver

/// Builds the C references of `platform` (a no-op when they are up to date) and returns their
/// prefix.
pub(crate) fn references(root: &Path, platform: &str) -> anyhow::Result<PathBuf> {
    let prefix = root
        .join("benches")
        .join("target")
        .join("refs")
        .join(platform);
    let status = Command::new("bash")
        .arg(root.join("benches").join("refs").join("build.sh"))
        .arg(platform)
        .arg(&prefix)
        .status()
        .context("cannot run bash for benches/refs/build.sh")?;
    if !status.success() {
        bail!("building the C references for {platform} failed (benches/refs/build.sh)");
    }
    Ok(prefix)
}

/// The flags of a measurement build for `CARGO_ENCODED_RUSTFLAGS`: `flags` plus a remapping of
/// the paths of the work tree and of the Cargo home. Without it these paths end up in the
/// panic locations of the binary, and their length shifts its data and with it the counts (by
/// 0.01 instructions per sample, between two checkouts at different paths). The encoded form
/// keeps paths with spaces in one argument.
fn rustflags(flags: &str, root: &Path) -> std::ffi::OsString {
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".cargo")));
    encoded_rustflags(flags, root, cargo_home.as_deref())
}

/// [`rustflags`] with the Cargo home given.
fn encoded_rustflags(flags: &str, root: &Path, cargo_home: Option<&Path>) -> std::ffi::OsString {
    let mut args: Vec<std::ffi::OsString> = flags.split_whitespace().map(Into::into).collect();
    let remaps = [(Some(root), "/sdrk"), (cargo_home, "/cargo")];
    for (from, to) in remaps {
        if let Some(from) = from {
            let mut arg = std::ffi::OsString::from("--remap-path-prefix=");
            arg.push(from);
            arg.push("=");
            arg.push(to);
            args.push(arg);
        }
    }
    args.join(std::ffi::OsStr::new("\u{1f}"))
}

fn build(root: &Path, plan: &LevelPlan, refs: &Path) -> anyhow::Result<PathBuf> {
    let target_dir = root.join("benches").join("target").join(plan.level);
    let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cmd.current_dir(root.join("benches"))
        .args(["build", "--release", "-p", "kernels", "--target-dir"])
        .arg(&target_dir);
    if let Some(target) = plan.target {
        cmd.args(["--target", target]);
    }
    if plan.runtime_detect {
        cmd.args(["--features", "std"]);
    }
    // Only the references of benches/refs/build.sh, never a VOLK of the system: the
    // distribution's libvolk is built differently (-O2, LTO, ORC) and gives other counts.
    cmd.env_remove("RUSTFLAGS")
        .env("CARGO_ENCODED_RUSTFLAGS", rustflags(plan.rustflags, root))
        .env("SDRK_REFS_DIR", refs)
        .env("PKG_CONFIG_PATH", refs.join("lib").join("pkgconfig"))
        .env_remove("PKG_CONFIG_LIBDIR")
        .env_remove("PKG_CONFIG_SYSROOT_DIR");
    if plan.target.is_some() {
        cmd.env("PKG_CONFIG_ALLOW_CROSS", "1");
    }
    if plan.target == Some("aarch64-unknown-linux-gnu") {
        cmd.env(
            "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER",
            "aarch64-linux-gnu-gcc",
        );
    }
    println!(
        "==> building the measurement binary for {}/{}",
        plan.platform, plan.level
    );
    let status = cmd.status().context("cannot run cargo")?;
    if !status.success() {
        bail!("building benches/kernels failed");
    }
    let mut path = target_dir;
    if let Some(target) = plan.target {
        path.push(target);
    }
    path.push("release");
    path.push(if plan.target.is_some_and(|t| t.starts_with("wasm")) {
        "count.wasm"
    } else {
        "count"
    });
    Ok(path)
}

/// Writes `text` to `dir/volk/volk_config` and returns `dir` (the value of `VOLK_CONFIGPATH`).
pub(crate) fn volk_config(dir: PathBuf, text: &str) -> anyhow::Result<PathBuf> {
    let file = dir.join("volk").join("volk_config");
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&file, text)?;
    Ok(dir)
}

/// Builds the measurement binary of `plan` and sets up the session of its runs.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the Windows file system calls are large; xtask is a host tool"
)]
pub(crate) fn session(root: &Path, plan: &LevelPlan, refs: &Path) -> anyhow::Result<Session> {
    let binary = build(root, plan, refs)?;
    let scratch = root.join("benches").join("target").join(plan.level);
    std::fs::create_dir_all(&scratch)?;
    let cpuid = match plan.runner {
        Runner::Sde(SdeCpu::Sse2) => Some((CPUID_SSE2, "cpuid-sse2.def")),
        Runner::Sde(SdeCpu::Sse42) => Some((CPUID_SSE42, "cpuid-sse4.2.def")),
        _ => None,
    }
    .map(|(text, name)| -> anyhow::Result<PathBuf> {
        let path = scratch.join(name);
        std::fs::write(&path, text)?;
        Ok(path)
    })
    .transpose()?;
    let session = Session {
        plan: *plan,
        binary,
        volk_config: volk_config(scratch.join("volk-pinned"), plan.volk_pin.config())?,
        refs_lib: refs.join("lib"),
        cpuid,
        scratch: scratch.clone(),
        qemu: (plan.runner == Runner::Qemu).then(qemu_tools),
        tools: root.join("target").join("tools"),
        bbcount_source: root.join("xtask").join("plugins").join("bbcount.c"),
    };
    Ok(session)
}

/// The instruction profiles of one entry for level 2: ours and the judged comparison.
#[derive(Clone, Debug)]
pub struct EntryProfiles {
    /// The entry of the benchmark table.
    pub kernel: String,
    /// The comparison (as in the results file).
    pub compared_with: String,
    /// Samples processed by the counted calls (the divisor of the per-sample figures).
    pub samples: u64,
    /// Our side (the fallback for a fallback entry).
    pub ours: crate::mca::Counts,
    /// The judged comparison.
    pub other: crate::mca::Counts,
}

/// The branch counts of one count entry (D1611): ours and the judged comparison.
#[derive(Clone, Debug)]
pub struct EntryBranches {
    /// The comparison (as in the results file).
    pub compared_with: String,
    /// Samples processed by the counted calls (the divisor of the per-sample figures).
    pub samples: u64,
    /// Our side (the fallback for a fallback entry).
    pub ours: crate::branches::Branches,
    /// The judged comparison.
    pub other: crate::branches::Branches,
}

/// The branch counts of `kernels` at `level` of `platform` from `cachegrind --branch-sim=yes`
/// on the benchmark workload of each entry (D1611): per entry, the branches and mispredictions
/// of the counted calls of our side and of the judged comparison. Only the x86-64 levels that
/// Valgrind runs ([`crate::branches::simulated_level`]); the FFTs have the restore of their
/// input subtracted. An entry that cannot be counted is returned with its error.
///
/// # Errors
/// An unknown platform or level, a level without branch simulation, or a tool that is missing
/// or fails before the first entry.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the plan table are large; xtask is a host tool"
)]
pub fn branch_profiles(
    root: &Path,
    platform: &str,
    level: &str,
    kernels: &[String],
) -> anyhow::Result<Vec<(String, anyhow::Result<EntryBranches>)>> {
    let plan = PLANS
        .iter()
        .find(|p| p.platform == platform && p.level == level)
        .ok_or_else(|| anyhow!("the platform `{platform}` has no level `{level}`"))?;
    if plan.baseline.is_some() {
        bail!("the level {level} is judged against our own numbers, not against a reference");
    }
    if !matches!(plan.runner, Runner::Host | Runner::Sde(_))
        || !crate::branches::simulated_level(level)
    {
        bail!("the level {level} has no branch simulation (D1611: x86-64 builds under Valgrind)");
    }
    let refs = references(root, plan.refs)?;
    let session = session(root, plan, &refs)?;
    let mut out = Vec::new();
    for spec in KERNELS
        .iter()
        .filter(|s| kernels.iter().any(|k| k == s.name))
    {
        if !spec.family.measured_at(plan) {
            out.push((
                spec.name.to_string(),
                Err(anyhow!("not measured at the level {level}")),
            ));
            continue;
        }
        let entry = || -> anyhow::Result<EntryBranches> {
            let (gate, _) = comparisons(plan, spec);
            let info = session.query(&["info", spec.name])?;
            let compared_with = info
                .get("impls")
                .and_then(|i| i.get(gate.implementation()))
                .and_then(Value::as_str)
                .unwrap_or(gate.implementation())
                .to_string();
            println!(
                "==> {level} {} (branch simulation of cachegrind, ours vs {compared_with})",
                spec.name
            );
            let standard = Variant::standard(spec);
            let restore = if spec.family.is_fft() {
                Some(session.branch_sim(spec.name, "restore", &standard)?)
            } else {
                None
            };
            let side = |which: &str| -> anyhow::Result<crate::branches::Branches> {
                let b = session.branch_sim(spec.name, which, &standard)?;
                match &restore {
                    Some(r) => b.minus(r),
                    None => Ok(b),
                }
            };
            Ok(EntryBranches {
                compared_with,
                samples: K * (N / standard.samples).max(1) * standard.samples,
                ours: side(spec.family.measured())?,
                other: side(gate.implementation())?,
            })
        };
        out.push((spec.name.to_string(), entry()));
    }
    Ok(out)
}

/// The instruction profiles of `kernels` at `level` of `platform` (level 2 of the measurement
/// policy): per entry, the execution counts per instruction of the counted calls for our side
/// and for the judged comparison, taken by `profiler` in the environment of the runs of the
/// level (D68); at a level judged against our own build of another level (`avx2-runtime`), the
/// other side is our entry in the binary of that level (D2702). The FFTs have the
/// restore of their input subtracted. An entry that cannot be profiled (not measured at the
/// level, a run that fails) is returned with its error, so that the report can name it.
///
/// # Errors
/// An unknown platform, level or kernel, a profiler that does not fit the counter of the level
/// (`sde` fits callgrind levels too, for its validation), or a tool that is missing or fails
/// before the first entry.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the plan table are large; xtask is a host tool"
)]
pub fn profiles(
    root: &Path,
    platform: &str,
    level: &str,
    kernels: &[String],
    profiler: Profiler,
) -> anyhow::Result<Vec<(String, anyhow::Result<EntryProfiles>)>> {
    let plan = PLANS
        .iter()
        .find(|p| p.platform == platform && p.level == level)
        .ok_or_else(|| anyhow!("the platform `{platform}` has no level `{level}`"))?;
    match (profiler, plan.runner) {
        (Profiler::Callgrind, Runner::Host)
        | (Profiler::Sde, Runner::Sde(_) | Runner::Host)
        | (Profiler::Qemu, Runner::Qemu) => {}
        _ => bail!(
            "the level {level} runs on {}; its execution counts cannot come from {}",
            plan.runner.describe(),
            profiler.name()
        ),
    }
    let refs = references(root, plan.refs)?;
    let session = session(root, plan, &refs)?;
    // A level judged against our own build of another level (`avx2-runtime` against `avx2`,
    // D2702): the other side is the same entry in the binary of that level.
    let baseline = plan
        .baseline
        .map(|base| -> anyhow::Result<Session> {
            let base_plan = PLANS
                .iter()
                .find(|p| p.platform == platform && p.level == base)
                .ok_or_else(|| anyhow!("the platform `{platform}` has no level `{base}`"))?;
            self::session(root, base_plan, &refs)
        })
        .transpose()?;
    let mut out = Vec::new();
    for spec in KERNELS
        .iter()
        .filter(|s| kernels.iter().any(|k| k == s.name))
    {
        if !spec.family.measured_at(plan) {
            out.push((
                spec.name.to_string(),
                Err(anyhow!("not measured at the level {level}")),
            ));
            continue;
        }
        let entry = || -> anyhow::Result<EntryProfiles> {
            let (gate, _) = comparisons(plan, spec);
            let compared_with = if let Reference::OwnLevel(base) = gate {
                format!("ours at {base}")
            } else {
                let info = session.query(&["info", spec.name])?;
                info.get("impls")
                    .and_then(|i| i.get(gate.implementation()))
                    .and_then(Value::as_str)
                    .unwrap_or(gate.implementation())
                    .to_string()
            };
            println!(
                "==> {level} {} (instruction profiles from {}, ours vs {compared_with})",
                spec.name,
                profiler.name()
            );
            let standard = Variant::standard(spec);
            // The in-place FFTs restore their input before every call: the restore, profiled in
            // the binary of each side, is subtracted.
            let restore = |s: &Session| -> anyhow::Result<Option<crate::mca::Counts>> {
                spec.family
                    .is_fft()
                    .then(|| s.profile(spec.name, "restore", &standard, profiler))
                    .transpose()
            };
            let profile = |s: &Session,
                           restore: &Option<crate::mca::Counts>,
                           which: &str|
             -> anyhow::Result<crate::mca::Counts> {
                let p = s.profile(spec.name, which, &standard, profiler)?;
                Ok(match restore {
                    Some(r) => crate::mca::difference(&p, r),
                    None => p,
                })
            };
            let ours_restore = restore(&session)?;
            let ours = profile(&session, &ours_restore, spec.family.measured())?;
            let other = match &baseline {
                Some(base) => profile(base, &restore(base)?, spec.family.measured())?,
                None => profile(&session, &ours_restore, gate.implementation())?,
            };
            Ok(EntryProfiles {
                kernel: spec.name.to_string(),
                compared_with,
                samples: K * (N / standard.samples).max(1) * standard.samples,
                ours,
                other,
            })
        };
        out.push((spec.name.to_string(), entry()));
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::count::{Limits, Status};

    #[test]
    fn qemu_option_variables_are_recognised() {
        for name in ["QEMU_VERSION", "QEMU_LD_PREFIX", "QEMU_"] {
            assert!(
                is_qemu_option_variable(std::ffi::OsStr::new(name)),
                "{name}"
            );
        }
        for name in [
            "SDRK_QEMU_AARCH64",
            "QEMU",
            "qemu_version",
            "LD_LIBRARY_PATH",
        ] {
            assert!(
                !is_qemu_option_variable(std::ffi::OsStr::new(name)),
                "{name}"
            );
        }
    }

    #[test]
    fn the_plan_covers_the_four_platforms_with_established_references() {
        for p in ["x86_64", "x86_64-avx512", "aarch64", "wasm32"] {
            assert!(PLANS.iter().any(|plan| plan.platform == p), "{p}");
        }
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        assert_eq!(
            comparisons(plan("sse2"), kernel("fft_1024")),
            (
                Reference::Fftw,
                vec![Reference::FftwInPlace, Reference::RustFft]
            )
        );
        assert_eq!(
            comparisons(plan("sse4.2"), kernel("fft_1024")).0,
            Reference::RustFft
        );
        assert_eq!(
            comparisons(plan("sse2"), kernel("rfft_1024")),
            (
                Reference::Fftw,
                vec![Reference::RustFft, Reference::Fallback]
            )
        );
        assert_eq!(
            comparisons(plan("avx2"), kernel("rfft_1024")),
            (Reference::RustFft, vec![Reference::Fallback])
        );
        assert_eq!(
            comparisons(plan("neon"), kernel("rotate")).0,
            Reference::Volk
        );
        assert_eq!(comparisons(plan("neon"), kernel("rotate")).1, Vec::new());
        assert_eq!(
            comparisons(plan("simd128"), kernel("mag_sq")).0,
            Reference::Volk
        );
        assert_eq!(
            comparisons(plan("sse2"), kernel("agc")),
            (Reference::GnuRadio, vec![Reference::Fallback])
        );
        assert_eq!(
            comparisons(plan("avx2-runtime"), kernel("agc")).0,
            Reference::OwnLevel("avx2")
        );
        assert_eq!(
            comparisons(plan("simd128"), kernel("agc_cq15")).0,
            Reference::Fallback
        );
        assert_eq!(
            comparisons(plan("neon"), kernel("squelch")),
            (Reference::GnuRadio, vec![Reference::Fallback])
        );
        assert_eq!(
            comparisons(plan("sse2"), kernel("squelch_cq15")).0,
            Reference::Fallback
        );
        assert_eq!(
            comparisons(plan("neon"), kernel("fft_cq15_1024")),
            (Reference::KissFft, vec![Reference::Ne10])
        );
        assert_eq!(
            comparisons(plan("avx2"), kernel("fm_mod")),
            (Reference::GnuRadio, vec![Reference::Fallback])
        );
        assert_eq!(
            comparisons(plan("avx2-runtime"), kernel("fir")).0,
            Reference::OwnLevel("avx2")
        );
        // VOLK's wasm32 build has no `volk_32f_x2_dot_prod_32f` and no `volk_32f_s32f_multiply_32f`.
        for k in ["dot_real", "scale"] {
            assert_eq!(comparisons(plan("avx2"), kernel(k)).0, Reference::Volk);
            assert_eq!(
                comparisons(plan("neon"), kernel(k)),
                (Reference::Volk, Vec::new())
            );
            assert_eq!(
                comparisons(plan("simd128"), kernel(k)),
                (Reference::Fallback, Vec::new())
            );
        }
        assert_eq!(
            comparisons(plan("simd128"), kernel("dot")).0,
            Reference::Volk
        );
    }

    #[test]
    fn the_oscillators_have_scalar_references_for_their_fallbacks() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "neon", "simd128"] {
            for k in ["rotate_cq15", "fm_mod_cq15"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, vec![Reference::Cmsis]),
                    "{k} at {level}"
                );
                let fallback = kernel(&format!("{k}_fallback"));
                assert_eq!(fallback.family.measured(), "fallback");
                assert_eq!(
                    comparisons(plan(level), fallback),
                    (Reference::Cmsis, Vec::new())
                );
            }
            for k in ["rotate_fallback", "fm_mod_fallback"] {
                assert_eq!(kernel(k).family.measured(), "fallback");
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::GnuRadio, vec![Reference::Cmsis])
                );
            }
        }
        assert!(
            !kernel("rotate_fallback")
                .family
                .measured_at(plan("avx2-runtime"))
        );
    }

    #[test]
    fn the_biquads_are_judged_against_kfr_cmsis_and_their_fallback_entries() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            for k in ["biquad", "biquad_real"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Kfr, vec![Reference::Cmsis]),
                    "{k} at {level}"
                );
            }
        }
        assert_eq!(
            comparisons(plan("neon"), kernel("biquad_cq15")),
            (Reference::Fallback, vec![Reference::Cmsis])
        );
        let fallback = kernel("biquad_real_q15_fallback");
        assert_eq!(fallback.family.measured(), "fallback");
        assert_eq!(kernel("biquad_real").family.measured(), "ours");
        assert_eq!(
            comparisons(plan("avx512"), fallback),
            (Reference::Cmsis, Vec::new())
        );
        assert!(fallback.family.measured_at(plan("simd128")));
        assert!(!fallback.family.measured_at(plan("avx2-runtime")));
        assert!(
            kernel("biquad_real")
                .family
                .measured_at(plan("avx2-runtime"))
        );
    }

    #[test]
    fn the_halfband_entries_are_judged_against_external_references() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            // liquid-dsp's SIMD dot product is the gate of the float kernels.
            assert_eq!(
                comparisons(plan(level), kernel("halfband_decimate")),
                (
                    Reference::Liquid,
                    vec![Reference::OwnKernel("halfband_ref_fir_decimate")]
                ),
                "{level}"
            );
            assert_eq!(
                comparisons(plan(level), kernel("halfband_decimate_real")),
                (Reference::Liquid, Vec::new())
            );
            // libairspy's SSE2 converter is the second gate at `sse2` only (its own entry).
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex")),
                (
                    Reference::Liquid,
                    vec![Reference::OwnKernel("halfband_ref_shifted")]
                )
            );
            assert_eq!(
                kernel("real_to_complex_airspy")
                    .family
                    .measured_at(plan(level)),
                level == "sse2"
            );
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex_airspy")),
                (Reference::Libairspy, Vec::new())
            );
            // No library has a Q15 halfband decimator: our fallback gates, the CMSIS-DSP
            // composition is information; `real_to_complex_cq15` has libairspy (D70).
            assert_eq!(
                comparisons(plan(level), kernel("halfband_decimate_real_q15")),
                (Reference::Fallback, vec![Reference::Cmsis])
            );
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex_cq15")),
                (
                    Reference::Libairspy,
                    vec![
                        Reference::Fallback,
                        Reference::OwnKernel("halfband_ref_shifted_cq15")
                    ]
                )
            );
            // The fallback entries: liquid-dsp's portable dot product, the CMSIS-DSP
            // composition, libairspy's generic code.
            for k in [
                "halfband_decimate_fallback",
                "halfband_decimate_real_fallback",
            ] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Liquid, Vec::new())
                );
                assert_eq!(kernel(k).family.measured(), "fallback");
            }
            for k in [
                "halfband_decimate_cq15_fallback",
                "halfband_decimate_real_q15_fallback",
            ] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, Vec::new())
                );
            }
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex_fallback")),
                (Reference::Liquid, vec![Reference::Libairspy])
            );
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex_cq15_fallback")),
                (Reference::Libairspy, Vec::new())
            );
        }
        for k in [
            "halfband_decimate_fallback",
            "real_to_complex_cq15_fallback",
            "halfband_ref_fir_decimate",
            "halfband_ref_shifted_cq15",
        ] {
            assert!(!kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
            assert!(kernel(k).family.measured_at(plan("avx2")), "{k}");
        }
        // The own-kernel comparison runs `ours` of the other kernel, and every kernel it names
        // is in the table.
        assert_eq!(Reference::OwnKernel("x").implementation(), "ours");
        assert_eq!(Reference::Libairspy.implementation(), "airspy");
        for spec in &KERNELS {
            if let Some(own) = spec.own {
                assert!(KERNELS.iter().any(|k| k.name == own), "{own}");
            }
        }
    }

    #[test]
    fn the_stereo_entries_are_judged_against_the_references_of_d108_to_d111() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            // The VOLK composition gates `stereo_matrix` at every level (generic kernels on
            // `simd128`).
            let (gate, extras) = comparisons(plan(level), kernel("stereo_matrix"));
            assert_eq!(gate, Reference::Volk, "{level}");
            assert_eq!(extras, Vec::new(), "{level}");
            // No library has a SIMD CQ15 matrix: our fallback, CMSIS-DSP as information.
            assert_eq!(
                comparisons(plan(level), kernel("stereo_matrix_cq15")),
                (Reference::Fallback, vec![Reference::Cmsis]),
                "{level}"
            );
            // The fallbacks: CMSIS-DSP on planar data; liquid-dsp's NCO for the f32 loop.
            for k in [
                "stereo_matrix_fallback",
                "stereo_matrix_cq15_fallback",
                "stereo_pll_q15_fallback",
            ] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, Vec::new()),
                    "{k} at {level}"
                );
            }
            assert_eq!(
                comparisons(plan(level), kernel("stereo_pll_fallback")),
                (Reference::Liquid, Vec::new()),
                "{level}"
            );
        }
        for k in [
            "stereo_matrix_fallback",
            "stereo_matrix_cq15_fallback",
            "stereo_pll_fallback",
            "stereo_pll_q15_fallback",
        ] {
            assert_eq!(kernel(k).family.measured(), "fallback", "{k}");
            assert!(!kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
            assert!(kernel(k).family.measured_at(plan("simd128")), "{k}");
        }
        for k in ["stereo_matrix", "stereo_matrix_cq15"] {
            assert_eq!(kernel(k).family.measured(), "ours", "{k}");
            assert!(kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
        // `stereo_pll` has no SIMD branch: no entry without the suffix.
        assert!(KERNELS.iter().all(|k| k.name != "stereo_pll"));
        assert!(KERNELS.iter().all(|k| k.name != "stereo_pll_q15"));
    }

    #[test]
    fn the_recursion_fallbacks_are_judged_against_liquid_and_gnu_radio() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "avx2", "avx512", "neon", "simd128"] {
            for k in ["dc_block_fallback", "dc_block_real_fallback"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Liquid, Vec::new())
                );
            }
            assert_eq!(
                comparisons(plan(level), kernel("deemphasis_fallback")),
                (Reference::GnuRadio, vec![Reference::Cmsis])
            );
            assert_eq!(
                comparisons(plan(level), kernel("deemphasis")),
                (Reference::Fallback, Vec::new())
            );
        }
        assert_eq!(kernel("deemphasis_fallback").family.measured(), "fallback");
        assert!(
            !kernel("dc_block_fallback")
                .family
                .measured_at(plan("avx2-runtime"))
        );
    }

    #[test]
    fn preemph_limit_is_information_against_the_fallback_at_every_level() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            for k in ["preemph_limit", "preemph_limit_q15"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, Vec::new()),
                    "{k} at {level}"
                );
            }
        }
        for k in ["preemph_limit", "preemph_limit_q15"] {
            assert_eq!(
                comparisons(plan("avx2-runtime"), kernel(k)).0,
                Reference::OwnLevel("avx2"),
                "{k}"
            );
            assert!(kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
    }

    #[test]
    fn the_am_mod_entries_have_the_comparisons_of_their_decision() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            // No SIMD library has an AM modulator: our fallback, CMSIS-DSP's composition beside
            // the `CQ15` one.
            assert_eq!(
                comparisons(plan(level), kernel("am_mod")),
                (Reference::Fallback, Vec::new()),
                "am_mod at {level}"
            );
            assert_eq!(
                comparisons(plan(level), kernel("am_mod_cq15")),
                (Reference::Fallback, vec![Reference::Cmsis]),
                "am_mod_cq15 at {level}"
            );
            // The fallbacks: liquid-dsp's `ampmodem` and the CMSIS-DSP composition.
            assert_eq!(
                comparisons(plan(level), kernel("am_mod_fallback")),
                (Reference::Liquid, Vec::new()),
                "am_mod_fallback at {level}"
            );
            assert_eq!(
                comparisons(plan(level), kernel("am_mod_cq15_fallback")),
                (Reference::Cmsis, Vec::new()),
                "am_mod_cq15_fallback at {level}"
            );
        }
        for k in ["am_mod", "am_mod_cq15"] {
            assert_eq!(kernel(k).family.measured(), "ours", "{k}");
            assert!(kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
        for k in ["am_mod_fallback", "am_mod_cq15_fallback"] {
            assert_eq!(kernel(k).family.measured(), "fallback", "{k}");
            assert!(!kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
        // `ssb_mod` is a block of existing kernels, without an entry of its own.
        assert!(KERNELS.iter().all(|s| !s.name.starts_with("ssb_mod")));
    }

    #[test]
    fn the_noise_reduction_kernels_are_information_against_the_fallback_at_every_level() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let names = ["noise_blank", "noise_blank_cq15", "auto_notch"];
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            for k in names {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, Vec::new()),
                    "{k} at {level}"
                );
            }
        }
        for k in names {
            assert_eq!(
                comparisons(plan("avx2-runtime"), kernel(k)).0,
                Reference::OwnLevel("avx2"),
                "{k}"
            );
            assert!(kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
        // Not accelerated: no `_fallback` entry next to the facade entry.
        for k in names {
            let fallback = format!("{k}_fallback");
            assert!(KERNELS.iter().all(|s| s.name != fallback), "{fallback}");
        }
    }

    #[test]
    fn the_stereo_transmit_kernels_are_information_against_the_fallback_at_every_level() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let names = [
            "preemph_limit_stereo",
            "stereo_encode",
            "rds_shape",
            "stereo_encode_rds",
        ];
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            for k in names {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, Vec::new()),
                    "{k} at {level}"
                );
            }
        }
        for k in names {
            assert_eq!(
                comparisons(plan("avx2-runtime"), kernel(k)).0,
                Reference::OwnLevel("avx2"),
                "{k}"
            );
            assert!(kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
        // Not accelerated: no `_fallback` entry next to the facade entry.
        for k in names {
            let fallback = format!("{k}_fallback");
            assert!(KERNELS.iter().all(|s| s.name != fallback), "{fallback}");
        }
    }

    #[test]
    fn the_protocol_entries_are_gated_against_gnu_radio_or_information() {
        // The RDS blocks of M41 (D678) are information like the framing.
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let crcs = [
            "crc16_ccitt_false",
            "crc16_x25",
            "crc32",
            "crc24_adsb",
            "bit_scramble_block",
            "bit_descramble_block",
            "bit_scramble_in_place_block",
            "bit_descramble_in_place_block",
            "snr_estimate_block",
            "snr_estimate_block_svr",
        ];
        let framing = [
            "frame_decode",
            "frame_encode_block",
            "frame_decode_block",
            "rds_demod_block",
            "rds_decode_block",
            "morse_encode_block",
            "morse_decode_block",
            "channelize_block",
            "channel_select_block",
            "channel_bank_channelize",
            "channel_bank_ddc",
            "band_pulse_detect_block",
            "symbol_demap_qam16_block",
            "symbol_demap_qam64_block",
            "symbol_demap_qam256_block",
            "qam_mod_block",
            "qam_mod_qam256_block",
            "qam_demod_block",
            "qam_demod_qam256_block",
            "qam_demod_search_block",
            "frame_decode_four_block",
            "psk_mod_block",
            "symbol_map_block",
            "iq_offset_block",
            "qam_acquire_block",
            "qam_frame_encode_block",
            "qam_frame_decode_block",
            "pulse_shape_block",
            "pulse_match_block",
            "noise_block",
            "scramble_block",
        ];
        let names = crcs.iter().chain(&framing).copied().collect::<Vec<_>>();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            // The CRCs against GNU Radio's `gr::digital::crc` (D527) and the bit scramblers
            // against its `scrambler_bb` and `descrambler_bb` (M44.2), the SNR estimate against its
            // `mpsk_snr_est_cc` (M44.4), the framing against our
            // own code (no library has it).
            for k in crcs {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::GnuRadio, vec![Reference::Fallback]),
                    "{k} at {level}"
                );
            }
            for k in framing {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, Vec::new()),
                    "{k} at {level}"
                );
            }
        }
        for k in names {
            assert_eq!(
                comparisons(plan("avx2-runtime"), kernel(k)).0,
                Reference::OwnLevel("avx2"),
                "{k}"
            );
            assert!(!kernel(k).misaligned && !kernel(k).small_n, "{k}");
            assert!(
                KERNELS.iter().all(|s| s.name != format!("{k}_fallback")),
                "{k}"
            );
        }
    }

    #[test]
    fn the_symbol_sync_entries_are_information_against_the_fallback_at_every_level() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        // `symbol_sync` (`C32`) and `symbol_sync_real` (`f32`): no library runs the same loop
        // (D562), so both are counted against our own fallback.
        let names = ["symbol_sync", "symbol_sync_real"];
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            for k in names {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, Vec::new()),
                    "{k} at {level}"
                );
            }
        }
        for k in names {
            assert_eq!(
                comparisons(plan("avx2-runtime"), kernel(k)).0,
                Reference::OwnLevel("avx2"),
                "{k}"
            );
            assert!(kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
            assert!(kernel(k).misaligned && !kernel(k).small_n, "{k}");
            // Not accelerated: no `_fallback` entry next to the facade entry.
            let fallback = format!("{k}_fallback");
            assert!(KERNELS.iter().all(|s| s.name != fallback), "{fallback}");
        }
        // `qam_soft_bits` (I44-1, D1412): the SIMD paths against our own fallback (no library
        // computes the same soft bits, D1201), misaligned as the other kernels.
        for k in [
            "qam_soft_bits_qpsk",
            "qam_soft_bits_qam16",
            "qam_soft_bits_qam64",
            "qam_soft_bits_qam256",
            "qam_soft_bits_qam1024",
        ] {
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, Vec::new()),
                    "{k} at {level}"
                );
            }
            assert_eq!(
                comparisons(plan("avx2-runtime"), kernel(k)).0,
                Reference::OwnLevel("avx2"),
                "{k}"
            );
            assert!(kernel(k).misaligned && !kernel(k).small_n, "{k}");
        }
        // The block `symbol_demap` has entries only for its QAM constellations, which run the
        // kernel `qam_soft_bits` with the block around it (M44, D1201, D1412; 256-QAM of M44.2,
        // D1524).
        assert!(
            KERNELS
                .iter()
                .filter(|s| s.name.starts_with("symbol_demap"))
                .map(|s| s.name)
                .eq([
                    "symbol_demap_qam16_block",
                    "symbol_demap_qam64_block",
                    "symbol_demap_qam256_block"
                ])
        );
    }

    #[test]
    fn the_costas_entries_are_information_and_the_fallback_against_gnu_radio() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        // `costas`: not accelerated, the facade entry is counted against our own fallback;
        // `costas_fallback` against GNU Radio's `costas_loop_cc` (D607).
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            assert_eq!(
                comparisons(plan(level), kernel("costas")),
                (Reference::Fallback, Vec::new()),
                "costas at {level}"
            );
            assert_eq!(
                comparisons(plan(level), kernel("costas_fallback")).0,
                Reference::GnuRadio,
                "costas_fallback at {level}"
            );
        }
        assert_eq!(
            comparisons(plan("avx2-runtime"), kernel("costas")).0,
            Reference::OwnLevel("avx2"),
        );
        assert!(kernel("costas").family.measured_at(plan("avx2-runtime")));
        assert!(
            !kernel("costas_fallback")
                .family
                .measured_at(plan("avx2-runtime"))
        );
        assert_eq!(kernel("costas_fallback").family.measured(), "fallback");
        for k in ["costas", "costas_fallback"] {
            assert!(kernel(k).misaligned && !kernel(k).small_n, "{k}");
        }
        // M44: the QAM path of the carrier loop and of the equalizer, the facade entries against
        // our own fallback, the fallbacks against GNU Radio's `constellation_receiver_cb` and
        // `linear_equalizer` on a `constellation_rect` (D1210).
        for (simd, fallback) in [
            ("costas_qam16", "costas_qam16_fallback"),
            ("equalize_qam16", "equalize_qam16_fallback"),
            ("costas_qam64", "costas_qam64_fallback"),
            ("equalize_qam64", "equalize_qam64_fallback"),
            ("costas_qam256", "costas_qam256_fallback"),
            ("equalize_qam256", "equalize_qam256_fallback"),
            ("costas_qam1024", "costas_qam1024_fallback"),
            ("equalize_qam1024", "equalize_qam1024_fallback"),
        ] {
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), kernel(simd)),
                    (Reference::Fallback, Vec::new()),
                    "{simd} at {level}"
                );
                assert_eq!(
                    comparisons(plan(level), kernel(fallback)).0,
                    Reference::GnuRadio,
                    "{fallback} at {level}"
                );
            }
            assert!(kernel(simd).family.measured_at(plan("avx2-runtime")));
            assert!(!kernel(fallback).family.measured_at(plan("avx2-runtime")));
            assert_eq!(kernel(fallback).family.measured(), "fallback");
        }
        let limits = Limits::parse(include_str!("../../benches/limits.json")).unwrap();
        for name in [
            "costas_qam16",
            "costas_qam16_fallback",
            "equalize_qam16",
            "equalize_qam16_fallback",
            "costas_qam64",
            "costas_qam64_fallback",
            "equalize_qam64",
            "equalize_qam64_fallback",
            "costas_qam256",
            "costas_qam256_fallback",
            "equalize_qam256",
            "equalize_qam256_fallback",
            "costas_qam1024",
            "costas_qam1024_fallback",
            "equalize_qam1024",
            "equalize_qam1024_fallback",
        ] {
            for p in PLANS.iter().filter(|p| kernel(name).family.measured_at(p)) {
                let Some(limit) = limits.get(name, p.level) else {
                    assert!(
                        p.baseline.is_some(),
                        "{name} at {}: no entry in limits.json",
                        p.level
                    );
                    continue;
                };
                if limit.status == Status::Info {
                    assert!(limit.waiver.is_some(), "{name} at {}", p.level);
                }
            }
        }
        // The filter design `rrc` is not a kernel and has no entry (D606).
        assert!(KERNELS.iter().all(|s| !s.name.starts_with("rrc")));
    }

    #[test]
    fn the_m43_entries_have_the_comparisons_of_their_decisions() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        // `dot_complex`: VOLK's protokernel without FMA where VOLK's dispatch fuses (D821), with
        // the dispatch as information; VOLK's dispatch elsewhere.
        for level in ["avx2", "avx512"] {
            assert_eq!(
                comparisons(plan(level), kernel("dot_complex")),
                (Reference::VolkNoFma, vec![Reference::Volk]),
                "dot_complex at {level}"
            );
        }
        assert_eq!(
            comparisons(plan("neon"), kernel("dot_complex")),
            (Reference::VolkNoFma, vec![Reference::Volk])
        );
        for level in ["sse2", "sse4.2", "simd128"] {
            assert_eq!(
                comparisons(plan(level), kernel("dot_complex")),
                (Reference::Volk, Vec::new()),
                "dot_complex at {level}"
            );
        }
        assert_eq!(
            comparisons(plan("avx2-runtime"), kernel("dot_complex")).0,
            Reference::OwnLevel("avx2")
        );
        assert!(kernel("dot_complex").small_n);
        // `dot_complex_fallback` against CMSIS-DSP's `arm_cmplx_dot_prod_f32` (D822).
        let fallback = kernel("dot_complex_fallback");
        assert_eq!(fallback.family.measured(), "fallback");
        // `equalize`: the facade against our fallback (information, D823); the fallback entry
        // against GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` (D838).
        let eq = kernel("equalize");
        let eq_fallback = kernel("equalize_fallback");
        assert_eq!(eq.family.measured(), "ours");
        assert_eq!(eq_fallback.family.measured(), "fallback");
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            assert_eq!(
                comparisons(plan(level), fallback),
                (Reference::Cmsis, Vec::new()),
                "dot_complex_fallback at {level}"
            );
            assert_eq!(
                comparisons(plan(level), eq),
                (Reference::Fallback, Vec::new()),
                "equalize at {level}"
            );
            assert_eq!(
                comparisons(plan(level), eq_fallback).0,
                Reference::GnuRadio,
                "equalize_fallback at {level}"
            );
        }
        assert!(eq.family.measured_at(plan("avx2-runtime")));
        assert!(!eq_fallback.family.measured_at(plan("avx2-runtime")));
        assert!(!fallback.family.measured_at(plan("avx2-runtime")));
        // Every level of the four entries is listed in `benches/limits.json`, the ones that are
        // not judged with a waiver.
        let limits = Limits::parse(include_str!("../../benches/limits.json")).unwrap();
        for name in [
            "dot_complex",
            "dot_complex_fallback",
            "equalize",
            "equalize_fallback",
        ] {
            for p in PLANS.iter().filter(|p| kernel(name).family.measured_at(p)) {
                let Some(limit) = limits.get(name, p.level) else {
                    assert!(
                        p.baseline.is_some(),
                        "{name} at {}: no entry in limits.json",
                        p.level
                    );
                    continue;
                };
                if limit.status == Status::Info {
                    assert!(limit.waiver.is_some(), "{name} at {}", p.level);
                }
            }
        }
        // The VOLK pins name the complex dot product on every pinned level.
        for pin in [VolkPin::Sse2, VolkPin::Sse42, VolkPin::Avx2, VolkPin::Neon] {
            assert!(
                pin.config().contains("volk_32fc_x2_dot_prod_32fc "),
                "{pin:?}"
            );
        }
        assert!(VOLK_FUNCTIONS.contains(&"volk_32fc_x2_dot_prod_32fc"));
    }

    #[test]
    fn the_fast_convolution_is_judged_against_liquid_on_every_level() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let limits = Limits::parse(include_str!("../../benches/limits.json")).unwrap();
        for n in ["n512", "n4096"] {
            let name = format!("xcorr_fft_{n}");
            let simd = kernel(&name);
            let fallback = kernel(&format!("{name}_fallback"));
            assert_eq!(simd.family.measured(), "ours");
            assert_eq!(fallback.family.measured(), "fallback");
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), simd),
                    (Reference::Liquid, vec![Reference::Fallback]),
                    "{name} at {level}"
                );
                assert_eq!(
                    comparisons(plan(level), fallback),
                    (Reference::Liquid, Vec::new()),
                    "{name}_fallback at {level}"
                );
            }
            assert!(!fallback.family.measured_at(plan("avx2-runtime")));
            for spec in [simd, fallback] {
                for p in PLANS.iter().filter(|p| spec.family.measured_at(p)) {
                    let limit = limits.get(spec.name, p.level);
                    assert!(
                        limit.is_some(),
                        "{} at {}: no entry in limits.json",
                        spec.name,
                        p.level
                    );
                    if let Some(limit) = limit.filter(|l| l.status == Status::Info) {
                        assert!(limit.waiver.is_some(), "{} at {}", spec.name, p.level);
                    }
                }
            }
        }
        // No FFTW build on wasm32: liquid's own FFT is no SIMD reference of `simd128`.
        let wasm = limits.get("xcorr_fft_n512", "simd128").unwrap();
        assert_eq!(wasm.status, Status::Info);
    }

    #[test]
    fn the_channelizer_is_judged_against_liquid_on_every_level() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let simd = kernel("channelize");
        let fallback = kernel("channelize_fallback");
        assert_eq!(simd.family.measured(), "ours");
        assert_eq!(fallback.family.measured(), "fallback");
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            assert_eq!(
                comparisons(plan(level), simd),
                (Reference::Liquid, vec![Reference::Fallback]),
                "channelize at {level}"
            );
            assert_eq!(
                comparisons(plan(level), fallback),
                (Reference::Liquid, Vec::new()),
                "channelize_fallback at {level}"
            );
        }
        assert_eq!(
            comparisons(plan("avx2-runtime"), simd).0,
            Reference::OwnLevel("avx2")
        );
        assert!(!fallback.family.measured_at(plan("avx2-runtime")));
        let limits = Limits::parse(include_str!("../../benches/limits.json")).unwrap();
        for name in ["channelize", "channelize_fallback"] {
            for p in PLANS.iter().filter(|p| kernel(name).family.measured_at(p)) {
                let limit = limits.get(name, p.level);
                assert!(
                    limit.is_some(),
                    "{name} at {}: no entry in limits.json",
                    p.level
                );
                if let Some(limit) = limit.filter(|l| l.status == Status::Info) {
                    assert!(limit.waiver.is_some(), "{name} at {}", p.level);
                }
            }
        }
    }

    #[test]
    fn scramble_is_judged_against_the_library_compositions_on_every_level() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let limits = Limits::parse(include_str!("../../benches/limits.json")).unwrap();
        // `C32`: the phasor lookup plus VOLK's product without FMA where the level has FMA, VOLK's
        // dispatcher elsewhere (M49 §5); the fallbacks against VOLK's generic product.
        for name in ["scramble_c32_b2", "scramble_c32_b8"] {
            for level in ["avx2", "avx512", "neon"] {
                assert_eq!(
                    comparisons(plan(level), kernel(name)).0,
                    Reference::VolkNoFma,
                    "{name} at {level}"
                );
            }
            for level in ["sse2", "sse4.2", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), kernel(name)).0,
                    Reference::Volk,
                    "{name} at {level}"
                );
            }
            let fallback = kernel(&format!("{name}_fallback"));
            assert_eq!(fallback.family.measured(), "fallback");
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), fallback),
                    (Reference::VolkGeneric, Vec::new()),
                    "{name}_fallback at {level}"
                );
            }
        }
        // `CQ15`: the same lookup plus CMSIS-DSP's `arm_cmplx_mult_cmplx_q15`, the fallback as
        // information beside it.
        for name in ["scramble_cq15_b2", "scramble_cq15_b8"] {
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), kernel(name)),
                    (Reference::Cmsis, vec![Reference::Fallback]),
                    "{name} at {level}"
                );
                assert_eq!(
                    comparisons(plan(level), kernel(&format!("{name}_fallback"))),
                    (Reference::Cmsis, Vec::new()),
                    "{name}_fallback at {level}"
                );
            }
            assert_eq!(
                comparisons(plan("avx2-runtime"), kernel(name)).0,
                Reference::OwnLevel("avx2")
            );
        }
        // GNU Radio's DVB-S2 switch: information (reference does different work), not measured
        // with run-time detection.
        let dvbs2 = kernel("scramble_c32_b2_dvbs2");
        assert_eq!(dvbs2.family.measured(), "ours");
        assert!(!dvbs2.family.measured_at(plan("avx2-runtime")));
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            assert_eq!(
                comparisons(plan(level), dvbs2),
                (Reference::GnuRadio, Vec::new()),
                "scramble_c32_b2_dvbs2 at {level}"
            );
            let limit = limits.get("scramble_c32_b2_dvbs2", level).unwrap();
            assert_eq!(limit.status, Status::Info, "at {level}");
            assert!(
                limit
                    .waiver
                    .as_deref()
                    .is_some_and(|w| w.contains("reference does different work"))
            );
        }
        for name in [
            "scramble_c32_b1",
            "scramble_c32_b2",
            "scramble_c32_b8",
            "scramble_cq15_b2",
            "scramble_cq15_b8",
            "scramble_c32_b1_fallback",
            "scramble_c32_b2_fallback",
            "scramble_c32_b8_fallback",
            "scramble_cq15_b2_fallback",
            "scramble_cq15_b8_fallback",
            "scramble_c32_b2_dvbs2",
            "scramble_block",
        ] {
            for p in PLANS.iter().filter(|p| kernel(name).family.measured_at(p)) {
                assert!(
                    limits.get(name, p.level).is_some(),
                    "{name} at {}: no entry",
                    p.level
                );
            }
        }
    }

    #[test]
    fn scramble_m49_1_rows_are_fair_and_gated() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let limits = Limits::parse(include_str!("../../benches/limits.json")).unwrap();
        // `bits = 1` (M49.1 §5): the `±1.0` sequence plus VOLK's `volk_32fc_32f_multiply_32fc`
        // through the dispatcher on every level (no FMA in it), the fallback against its generic
        // protokernel. With `scramble_c32_b8` the rows are fair and gated at every level with a
        // SIMD reference (D2092 revised); `sse2` (b8: no SSE2 product in VOLK) and `simd128`
        // (D12) stay information.
        let b1 = kernel("scramble_c32_b1");
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            assert_eq!(
                comparisons(plan(level), b1).0,
                Reference::Volk,
                "b1 at {level}"
            );
            assert_eq!(
                comparisons(plan(level), kernel("scramble_c32_b1_fallback")),
                (Reference::VolkGeneric, Vec::new()),
                "b1 fallback at {level}"
            );
        }
        for (name, info) in [
            ("scramble_c32_b1", &["simd128"][..]),
            ("scramble_c32_b8", &["sse2", "simd128"][..]),
            ("scramble_c32_b1_fallback", &[][..]),
            ("scramble_c32_b8_fallback", &[][..]),
        ] {
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                let limit = limits.get(name, level).unwrap();
                let want = if info.contains(&level) {
                    Status::Info
                } else {
                    Status::Release
                };
                assert_eq!(limit.status, want, "{name} at {level}");
                assert!(
                    !limit
                        .waiver
                        .as_deref()
                        .is_some_and(|w| w.contains("reference does less work")),
                    "{name} at {level}"
                );
            }
        }
        // `CQ15` at `bits = 8` stays information: CMSIS-DSP truncates to Q3.13 (D2092).
        for name in ["scramble_cq15_b8", "scramble_cq15_b8_fallback"] {
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                let limit = limits.get(name, level).unwrap();
                assert_eq!(limit.status, Status::Info, "{name} at {level}");
                assert!(
                    limit
                        .waiver
                        .as_deref()
                        .is_some_and(|w| w.contains("reference does less work")),
                    "{name} at {level}"
                );
            }
        }
    }

    #[test]
    fn gauss_noise_is_information_against_liquid_on_every_level() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let limits = Limits::parse(include_str!("../../benches/limits.json")).unwrap();
        for name in ["gauss_noise_c32", "gauss_noise_f32"] {
            let spec = kernel(name);
            assert_eq!(spec.family.measured(), "ours");
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), spec),
                    (Reference::Liquid, vec![Reference::Fallback]),
                    "{name} at {level}"
                );
            }
            assert_eq!(
                comparisons(plan("avx2-runtime"), spec).0,
                Reference::OwnLevel("avx2")
            );
        }
        // `Q15` noise has no library row: the SIMD paths against our own fallback.
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            assert_eq!(
                comparisons(plan(level), kernel("gauss_noise_q15")),
                (Reference::Fallback, Vec::new()),
                "gauss_noise_q15 at {level}"
            );
        }
        // Its fallback entry, the pair of the D321 cycle gate, compares the fallback with
        // itself.
        let spec = kernel("gauss_noise_q15_fallback");
        assert_eq!(spec.family.measured(), "fallback");
        assert!(!spec.family.measured_at(plan("avx2-runtime")));
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            assert_eq!(
                comparisons(plan(level), spec),
                (Reference::Fallback, Vec::new()),
                "gauss_noise_q15_fallback at {level}"
            );
        }
        for name in ["gauss_noise_c32_fallback", "gauss_noise_f32_fallback"] {
            let spec = kernel(name);
            assert_eq!(spec.family.measured(), "fallback");
            assert!(!spec.family.measured_at(plan("avx2-runtime")));
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), spec),
                    (Reference::Liquid, Vec::new()),
                    "{name} at {level}"
                );
            }
        }
        // The rows of GNU Radio's `fastnoise_source` (less work) and of rand_distr's Ziggurat
        // (different work): entries of their own, so that the cycles are simulated against them.
        for (name, reference) in [
            ("gauss_noise_c32_fastnoise", Reference::GnuRadio),
            ("gauss_noise_f32_fastnoise", Reference::GnuRadio),
            ("gauss_noise_f32_ziggurat", Reference::RandDistr),
        ] {
            let spec = kernel(name);
            assert_eq!(spec.family.measured(), "ours");
            assert!(!spec.family.measured_at(plan("avx2-runtime")));
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), spec),
                    (reference, Vec::new()),
                    "{name} at {level}"
                );
            }
        }
        assert_eq!(Reference::RandDistr.implementation(), "rand-distr");
        for name in [
            "gauss_noise_c32",
            "gauss_noise_f32",
            "gauss_noise_q15",
            "noise_block",
            "gauss_noise_c32_fallback",
            "gauss_noise_f32_fallback",
            "gauss_noise_q15_fallback",
            "gauss_noise_c32_fastnoise",
            "gauss_noise_f32_fastnoise",
            "gauss_noise_f32_ziggurat",
        ] {
            for p in PLANS.iter().filter(|p| kernel(name).family.measured_at(p)) {
                let limit = limits.get(name, p.level);
                let limit = limit.unwrap_or_else(|| panic!("{name} at {}: no entry", p.level));
                // Information at every level against liquid and rand_distr, which do different
                // work, and GNU Radio's `fastnoise_source`, which does less (M48 §5), and for the
                // block, which no library has; the run-time-detection build of the kernel
                // entries is judged against the `avx2` build of the SIMD path.
                if p.level == "avx2-runtime" && name.starts_with("gauss_noise") {
                    assert_eq!(limit.status, Status::Release, "{name} at {}", p.level);
                    continue;
                }
                assert_eq!(limit.status, Status::Info, "{name} at {}", p.level);
                assert!(limit.waiver.is_some(), "{name} at {}", p.level);
            }
        }
    }

    #[test]
    fn the_fft_fallback_entries_are_judged_against_cmsis_with_the_restore_subtracted() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            for k in [
                "fft_64_fallback",
                "fft_1024_fallback",
                "fft_4096_fallback",
                "rfft_1024_fallback",
            ] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, Vec::new()),
                    "{k} at {level}"
                );
            }
            for k in ["fft_cq15_1024_fallback", "fft_cq15_1024_default_fallback"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, vec![Reference::KissFft]),
                    "{k} at {level}"
                );
            }
        }
        for k in [
            "fft_1024_fallback",
            "fft_cq15_1024_fallback",
            "rfft_1024_fallback",
        ] {
            let family = kernel(k).family;
            assert!(family.is_fft(), "{k}");
            assert_eq!(family.measured(), "fallback", "{k}");
            assert!(!family.measured_at(plan("avx2-runtime")), "{k}");
        }
        assert_eq!(kernel("fft_1024").family.measured(), "ours");
    }

    #[test]
    fn the_dot_and_scale_fallbacks_are_judged_against_cmsis_dsp() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for k in [
            "dot_fallback",
            "dot_cq15_fallback",
            "dot_real_fallback",
            "dot_real_q15_fallback",
            "scale_fallback",
            "scale_cq15_fallback",
        ] {
            assert_eq!(kernel(k).family.measured(), "fallback", "{k}");
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, Vec::new()),
                    "{k} at {level}"
                );
            }
            assert!(!kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
    }

    #[test]
    fn the_m30_entries_have_the_comparisons_of_their_decisions() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        let levels = ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"];
        // VOLK has the function: the gate at every level.
        for k in [
            "add",
            "add_cq15",
            "add_real",
            "add_real_q15",
            "mul_real",
            "interleave",
            "deinterleave",
            "deinterleave_cq15",
        ] {
            for level in levels {
                assert_eq!(
                    comparisons(plan(level), kernel(k)).0,
                    Reference::Volk,
                    "{k} at {level}"
                );
            }
        }
        // `mul`: VOLK's dispatch, fused where ours is (D326); its protokernel without FMA as
        // information at those levels.
        for level in levels {
            let (gate, extras) = comparisons(plan(level), kernel("mul"));
            assert_eq!(gate, Reference::Volk, "mul at {level}");
            if matches!(level, "avx2" | "avx512" | "neon") {
                assert_eq!(
                    extras.first(),
                    Some(&Reference::VolkNoFma),
                    "mul at {level}"
                );
            } else {
                assert!(!extras.contains(&Reference::VolkNoFma), "mul at {level}");
            }
        }
        // No library has the function: our fallback, as information.
        for k in [
            "mul_cq15",
            "mul_real_q15",
            "interleave_cq15",
            "oscillate",
            "oscillate_cq15",
        ] {
            for level in levels {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, Vec::new()),
                    "{k} at {level}"
                );
            }
        }
        // The fallbacks against CMSIS-DSP's generic C, not at `avx2-runtime`.
        for k in [
            "add_fallback",
            "add_cq15_fallback",
            "add_real_fallback",
            "add_real_q15_fallback",
            "mul_fallback",
            "mul_cq15_fallback",
            "mul_real_fallback",
            "mul_real_q15_fallback",
            "oscillate_fallback",
            "oscillate_cq15_fallback",
        ] {
            assert_eq!(kernel(k).family.measured(), "fallback", "{k}");
            for level in levels {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, Vec::new()),
                    "{k} at {level}"
                );
            }
            assert!(!kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
        // The `interleave` and `deinterleave` fallbacks against VOLK's `generic` protokernels,
        // not at `avx2-runtime`.
        for k in [
            "interleave_fallback",
            "interleave_cq15_fallback",
            "deinterleave_fallback",
            "deinterleave_cq15_fallback",
        ] {
            assert_eq!(kernel(k).family.measured(), "fallback", "{k}");
            for level in levels {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::VolkGeneric, Vec::new()),
                    "{k} at {level}"
                );
            }
            assert!(!kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
    }

    #[test]
    fn both_runs_of_a_measurement_have_arguments_of_the_same_length() {
        let variant = Variant {
            samples: 64,
            misaligned: true,
            explicit_n: true,
        };
        let empty = run_args("fir", "ours", 0, 1024, &variant);
        let full = run_args("fir", "ours", 1024, 1024, &variant);
        assert_eq!(
            empty,
            ["run", "fir", "ours", "0000", "--n", "64", "--misaligned"]
        );
        assert_eq!(
            full,
            ["run", "fir", "ours", "1024", "--n", "64", "--misaligned"]
        );
    }

    #[test]
    fn the_measurement_build_remaps_the_paths_of_the_checkout() {
        let flags = encoded_rustflags(
            "-C target-cpu=x86-64-v3",
            Path::new("/work/my tree"),
            Some(Path::new("/home/u/.cargo")),
        );
        let args: Vec<&str> = flags.to_str().unwrap_or_default().split('\u{1f}').collect();
        assert_eq!(
            args,
            [
                "-C",
                "target-cpu=x86-64-v3",
                "--remap-path-prefix=/work/my tree=/sdrk",
                "--remap-path-prefix=/home/u/.cargo=/cargo",
            ]
        );
        let without_flags = encoded_rustflags("", Path::new("/w"), None);
        assert_eq!(without_flags, "--remap-path-prefix=/w=/sdrk");
    }

    #[test]
    fn the_kernel_table_names_the_kernels_of_the_measurement_binary() {
        let source = include_str!("../../benches/kernels/src/lib.rs");
        let table = source
            .split("pub const KERNELS")
            .nth(1)
            .and_then(|t| t.split("];").next())
            .unwrap();
        let mut binary: Vec<&str> = table.split('"').skip(1).step_by(2).collect();
        binary.sort_unstable();
        let mut ours: Vec<&str> = KERNELS.iter().map(|k| k.name).collect();
        ours.sort_unstable();
        assert_eq!(binary, ours);
    }

    #[test]
    fn the_kernel_table_is_consistent() {
        let names: Vec<&str> = KERNELS.iter().map(|k| k.name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
        for spec in &KERNELS {
            let fallback = spec.family.is_fft() && spec.family.is_fallback_entry();
            let complex = matches!(spec.family, Family::FftC32 | Family::FftCq15)
                || (fallback && spec.name.starts_with("fft_"));
            let real =
                spec.family == Family::RealFft || (fallback && spec.name.starts_with("rfft_"));
            assert_eq!(
                fallback,
                spec.name.ends_with("_fallback") && (complex || real)
            );
            assert_eq!(complex, spec.name.starts_with("fft_"), "{}", spec.name);
            assert_eq!(real, spec.name.starts_with("rfft_"), "{}", spec.name);
            let fft = spec.family.is_fft();
            assert_eq!(fft, complex || real);
            // The FFT sizes are fixed by the name; their buffers are always aligned.
            assert!(!(fft && (spec.misaligned || spec.small_n)), "{}", spec.name);
        }
    }

    #[test]
    fn the_cpuid_definitions_hide_everything_above_their_level() {
        let leaf1_ecx = |text: &str| -> u32 {
            let line = text.lines().find(|l| l.starts_with("00000001")).unwrap();
            u32::from_str_radix(line.split_whitespace().nth(5).unwrap(), 16).unwrap()
        };
        // SSE3 (bit 0), SSSE3 (9), SSE4.1 (19), SSE4.2 (20), AVX (28): none at sse2.
        let sse2 = leaf1_ecx(CPUID_SSE2);
        assert_eq!(sse2 & (1 | 1 << 9 | 1 << 19 | 1 << 20 | 1 << 28), 0);
        // OSXSAVE (27) keeps cpu_features off the host's /proc/cpuinfo.
        assert_ne!(sse2 & 1 << 27, 0);
        let v2 = leaf1_ecx(CPUID_SSE42);
        assert_eq!(
            v2 & (1 | 1 << 9 | 1 << 19 | 1 << 20 | 1 << 23),
            1 | 1 << 9 | 1 << 19 | 1 << 20 | 1 << 23
        );
        assert_eq!(v2 & (1 << 28 | 1 << 12), 0, "no AVX, no FMA");
    }
}
