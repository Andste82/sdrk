//! `correlate`: the sliding correlation of a complex stream with a PRBS code or a caller
//! reference (M50 §6), around the kernel `xcorr_fft`.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::fft::FftPlan;
use sdrk_core::kernels::xcorr::{
    MAX_FFT_SIZE, MIN_FFT_SIZE, XcorrFft, XcorrFftState, xcorr_spectrum,
};
use sdrk_core::timing::Timing;
use sdrk_proto::prbs::{Poly, Prbs};

use super::{PRBS_OPTIONS, PRBS_POLYS, prbs_choice, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// The largest `samples_per_chip` of the PRBS code.
pub const CORRELATE_MAX_SAMPLES_PER_CHIP: usize = 16;

/// The choice index of `prbs7`, the default code.
const CODE_DEFAULT: u16 = 1;

/// Description of the `correlate` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "correlate",
    summary: "Sliding correlation of a complex stream with a PRBS code by FFT fast convolution; \
        the peak marks the end of the code.",
    details: "Output n is y[n] = sum_k conj(r[k]) x[n - L + 1 + k], the correlation of the input \
        with the reference r of L samples: the PRBS code (chip value 1 - 2b of the bits of \
        prbs5 to prbs15 from seed, one period of 2^n - 1 chips) with every chip repeated \
        samples_per_chip times. A code that starts at input index m peaks at output index m + L \
        - 1 (the delay L - 1). The kernel xcorr_fft convolves with the conjugated, reversed \
        reference by overlap-save with an N-point FFT: N - L + 1 outputs per transform block, \
        the remainder of a call waits for the next input. fft_size 0 chooses the smallest power \
        of two of at least 4 L (at least 16, at most 65536); a manual size must be a power of \
        two from 16 to 65536 and at least 2 L. Storage: the spectrum and the scratch (N complex \
        values each), the history (L - 1) and the twiddles (N / 2). Memory and cost: 4 L costs \
        about 1.4 times less per sample than 2 L at about twice the memory (a 1023-chip code at \
        2 samples per chip: about 176 KiB with the automatic 8192 against 96 KiB with 4096). \
        Tolerance class (SNR at least 100 dB against the f64 correlation); the output does not \
        depend on the cut of the stream.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The correlation, one value per input sample, delayed by L - 1.",
    }],
    params: &[
        ParamDesc {
            name: "code",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: PRBS_OPTIONS,
                default: CODE_DEFAULT,
            },
            typical: "prbs7 (127 chips); prbs10 (1023 chips) for more processing gain",
            mutability: Mutability::Build,
            description: "The PRBS polynomial of the code: prbs5, prbs7, prbs9, prbs10, prbs11 \
                or prbs15; the code is one period of 2^n - 1 chips.",
        },
        ParamDesc {
            name: "seed",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 32767,
                default: Some(1),
            },
            typical: "1; the seed of the transmitter",
            mutability: Mutability::Build,
            description: "Initial register of the PRBS, 1 to 32767; only the low n bits are \
                used and they must not all be zero.",
        },
        ParamDesc {
            name: "samples_per_chip",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: CORRELATE_MAX_SAMPLES_PER_CHIP as i64,
                default: Some(1),
            },
            typical: "1 at the chip rate; 2 or 4 with oversampling",
            mutability: Mutability::Build,
            description: "Samples per chip, 1 to 16: every chip of the code is repeated this \
                many times in the reference.",
        },
        ParamDesc {
            name: "fft_size",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: MAX_FFT_SIZE as i64,
                default: Some(0),
            },
            typical: "0 (automatic, 4 L); the smallest power of two of at least 2 L to save memory",
            mutability: Mutability::Build,
            description: "FFT size N of the overlap-save: 0 for automatic (the smallest power \
                of two of at least 4 L, from 16 to 65536), else a power of two from 16 to 65536 \
                and at least 2 L.",
        },
    ],
    required: &[],
};

/// The FFT size of a reference of `reference_len` samples: `fft_size` 0 is automatic, the
/// smallest power of two of at least `4 L`, at least 16 and at most 65536; any other value must
/// be a power of two from 16 to 65536 and at least `2 L`.
///
/// ```
/// use sdrk_blocks::correlate_fft_size;
/// assert_eq!(correlate_fft_size(127, 0)?, 512);
/// assert_eq!(correlate_fft_size(2, 0)?, 16);
/// assert_eq!(correlate_fft_size(32_767, 0)?, 65_536); // capped: 4 L would be 131072
/// assert_eq!(correlate_fft_size(127, 256)?, 256);
/// assert!(correlate_fft_size(127, 128).is_err()); // below 2 L
/// assert!(correlate_fft_size(127, 300).is_err()); // not a power of two
/// assert!(correlate_fft_size(32_769, 0).is_err()); // 2 L above 65536
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
///
/// # Errors
/// `InvalidParameter` (`reference`) for an empty reference; `InvalidParameter` (`fft_size`) for
/// a value other than 0 that is not a power of two from 16 to 65536; `OutOfRange` (`fft_size`)
/// with the required size (the smallest power of two of at least `2 L` and 16) as `min` when the
/// size is below `2 L` or, for the automatic size, when `2 L` is above 65536.
pub fn correlate_fft_size(reference_len: usize, fft_size: usize) -> Result<usize> {
    if reference_len == 0 {
        return Err(Error::InvalidParameter {
            name: "reference",
            reason: "the reference must have at least one sample",
        });
    }
    if fft_size != 0
        && (!fft_size.is_power_of_two() || !(MIN_FFT_SIZE..=MAX_FFT_SIZE).contains(&fft_size))
    {
        return Err(Error::InvalidParameter {
            name: "fft_size",
            reason: "fft_size must be 0 (automatic) or a power of two from 16 to 65536",
        });
    }
    let twice = reference_len.saturating_mul(2);
    let required = twice
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
        .max(MIN_FFT_SIZE);
    let n = if fft_size == 0 {
        reference_len
            .saturating_mul(4)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .clamp(MIN_FFT_SIZE, MAX_FFT_SIZE)
    } else {
        fft_size
    };
    if n < twice {
        return Err(Error::OutOfRange {
            name: "fft_size",
            min: required as f64,
            max: MAX_FFT_SIZE as f64,
            got: fft_size as f64,
        });
    }
    Ok(n)
}

/// Configuration of [`CorrelateBlock`] (the PRBS code of the reference and the FFT size).
///
/// ```
/// use sdrk_blocks::CorrelateConfig;
/// use sdrk_proto::prbs::Poly;
/// let rate = sdrk::Rate::hz(48_000)?;
/// let cfg = CorrelateConfig::default(); // prbs7, seed 1, one sample per chip, automatic N
/// assert_eq!((cfg.reference_len()?, cfg.resolved_fft_size()?), (127, 512));
/// let oversampled = CorrelateConfig { code: Poly::Prbs10, samples_per_chip: 2, ..cfg };
/// assert_eq!((oversampled.reference_len()?, oversampled.resolved_fft_size()?), (2046, 8192));
/// // The smallest size of at least 2 L halves the memory of the transforms.
/// assert!(CorrelateConfig { fft_size: 4096, ..oversampled }.validate(rate).is_ok());
/// assert!(CorrelateConfig { fft_size: 2048, ..oversampled }.validate(rate).is_err());
/// // 32 masks to 0 in the 5-bit register of prbs5.
/// assert!(CorrelateConfig { code: Poly::Prbs5, seed: 32, ..cfg }.validate(rate).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorrelateConfig {
    /// The PRBS polynomial of the code (`Prbs5`, `Prbs7`, `Prbs9`, `Prbs10`, `Prbs11` or
    /// `Prbs15`); the code is one period of `2^n - 1` chips.
    pub code: Poly,
    /// Initial register of the PRBS, 1 to 32767; the low `n` bits must not all be zero.
    pub seed: u32,
    /// Samples per chip, 1 to [`CORRELATE_MAX_SAMPLES_PER_CHIP`].
    pub samples_per_chip: usize,
    /// FFT size `N`: 0 for automatic, else a power of two from 16 to 65536 and at least `2 L`.
    pub fft_size: usize,
}

impl Default for CorrelateConfig {
    fn default() -> Self {
        Self {
            code: Poly::Prbs7,
            seed: 1,
            samples_per_chip: 1,
            fft_size: 0,
        }
    }
}

fn check(name: &str, value: ParamValue) -> Result<()> {
    DESCRIPTION
        .param(name)
        .ok_or(Error::UnknownParameter)?
        .check(&value)
}

impl CorrelateConfig {
    /// The PRBS generator of the code at its start.
    fn prbs(&self) -> Result<Prbs> {
        Prbs::new(self.code, self.seed).map_err(|_| Error::InvalidParameter {
            name: "seed",
            reason: "the low n bits of seed (n the degree of the code) are all zero",
        })
    }

    /// The chips of the code: one period, `2^n - 1`.
    fn chips(&self) -> Result<usize> {
        if prbs_choice(self.code).is_none() {
            return Err(Error::InvalidParameter {
                name: "code",
                reason: "the code must be prbs5, prbs7, prbs9, prbs10, prbs11 or prbs15",
            });
        }
        Ok((1usize << self.code.degree()) - 1)
    }

    /// The length `L` of the reference: `(2^n - 1) samples_per_chip`.
    ///
    /// # Errors
    /// `InvalidParameter` (`code`) for a polynomial outside the choice; `OutOfRange`
    /// (`samples_per_chip`) outside 1 to 16.
    pub fn reference_len(&self) -> Result<usize> {
        check(
            "samples_per_chip",
            ParamValue::I64(i64::try_from(self.samples_per_chip).unwrap_or(i64::MAX)),
        )?;
        Ok(self.chips()? * self.samples_per_chip)
    }

    /// The FFT size `N` the block uses ([`correlate_fft_size`] of the reference length).
    ///
    /// # Errors
    /// The errors of [`reference_len`](Self::reference_len) and [`correlate_fft_size`].
    pub fn resolved_fft_size(&self) -> Result<usize> {
        correlate_fft_size(self.reference_len()?, self.fft_size)
    }

    /// Checks the configuration (the sample rate does not matter): the ranges, a code of the
    /// choice, a `seed` whose low `n` bits are not all zero (`InvalidParameter` of `seed`) and
    /// the FFT size (`InvalidParameter` or `OutOfRange` of `fft_size`, see
    /// [`correlate_fft_size`]).
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        check("seed", ParamValue::I64(i64::from(self.seed)))?;
        check(
            "fft_size",
            ParamValue::I64(i64::try_from(self.fft_size).unwrap_or(i64::MAX)),
        )?;
        self.prbs()?;
        self.resolved_fft_size().map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "code" => {
                self.code = PRBS_POLYS
                    .get(usize::from(value.as_choice()?))
                    .copied()
                    .ok_or(Error::WrongParamType)?;
            }
            "seed" => self.seed = u32::try_from(value.as_i64()?).unwrap_or(0),
            "samples_per_chip" => self.samples_per_chip = value.as_usize()?,
            "fft_size" => self.fft_size = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "code" => Ok(ParamValue::Choice(
                prbs_choice(self.code).unwrap_or(CODE_DEFAULT),
            )),
            "seed" => Ok(ParamValue::I64(i64::from(self.seed))),
            "samples_per_chip" => Ok(ParamValue::I64(
                i64::try_from(self.samples_per_chip).unwrap_or(i64::MAX),
            )),
            "fft_size" => Ok(ParamValue::I64(
                i64::try_from(self.fft_size).unwrap_or(i64::MAX),
            )),
            _ => unknown(),
        }
    }
}

/// The state type of the block.
type State<S> = XcorrFftState<
    C32,
    <S as Storage>::Buf<C32>,
    <S as Storage>::Buf<C32>,
    <S as Storage>::Buf<C32>,
    <S as Storage>::Buf<C32>,
>;

/// The bytes of the buffers of a reference of `l` samples with an `n`-point FFT, in the order of
/// the allocation: the spectrum (`n`), the history (`l - 1`), the scratch (`n`) and the twiddles
/// (`n / 2`), all `C32`.
fn plan_bytes(l: usize, n: usize) -> Result<usize> {
    let mut plan = Plan::default();
    plan.add::<C32>(n)?;
    plan.add::<C32>(l - 1)?;
    plan.add::<C32>(n)?;
    plan.add::<C32>(n / 2)?;
    Ok(plan.bytes())
}

/// Writes the taps `h[k] = conj(r[L - 1 - k])` of the reference `r` (`taps.len() == r.len()`).
fn taps_from_reference(reference: &[C32], taps: &mut [C32]) {
    for (t, r) in taps.iter_mut().zip(reference.iter().rev()) {
        *t = r.conj();
    }
}

/// The sliding correlation of a `C32` stream with a reference `r` of `L` samples,
/// `y[n] = sum_k conj(r[k]) x[n - L + 1 + k]`, around the kernel `xcorr_fft` (M50 §6).
///
/// [`new`](Self::new) takes the reference from the configuration: the PRBS code
/// (`sdrk_proto::prbs::Prbs::new(code, seed)`, chip `i` the `i`-th `next_bit`, chip value
/// `1 - 2b`, imaginary part 0; one period of `2^n - 1` chips) with every chip repeated
/// `samples_per_chip` times. [`with_reference`](Self::with_reference) takes any `C32` reference
/// from the caller (Rust API only; the pipeline has no list parameters). The block builds the
/// taps `h[k] = conj(r[L - 1 - k])` once, writes their spectrum with
/// `xcorr_spectrum` (the `Fallback` FFT) and keeps no reference. A reference that starts at
/// input index `m` peaks at output index `m + L - 1`.
///
/// Each `process` calls the kernel through the facade `Sdrk` once: it runs the whole transform
/// blocks of `B = N - L + 1` samples that fit the input and the output room and leaves a
/// remainder of the input unconsumed. Timing: ratio 1, delay `L - 1`, `min_input` `B`.
///
/// **Memory and cost.** The storage is the spectrum and the scratch (`N` values each), the
/// history (`L - 1`) and the twiddles (`N / 2`), all `C32`. The automatic size `4 L` costs about
/// 1.4 times less per sample than the smallest size `2 L` (two `N`-point FFTs per `N - L + 1`
/// outputs) at about twice the memory: a 1023-chip code at 2 samples per chip (`L` 2046) takes
/// 176 KiB with `N` 8192 (spectrum 64 KiB, scratch 64 KiB, history 16 KiB, twiddles 32 KiB) and
/// 96 KiB with `fft_size` 4096. A larger `N` than `4 L` saves at most about 10 % for `L` up to
/// 4095 at many times the memory. In the Rust API `XcorrFftState::from_spectrum` with a
/// `static` spectrum and `FftPlan::from_static_4096` puts the spectrum and the twiddles in
/// flash.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, CorrelateBlock, CorrelateConfig};
/// use sdrk::{Rate, C32};
/// let rate = Rate::hz(48_000)?;
/// let cfg = CorrelateConfig::default(); // prbs7: L 127, N 512, B 386
/// let mut mem = AlignedBytes([0u8; 12_288]);
/// let bytes = CorrelateBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let block = CorrelateBlock::new(&cfg, rate, &mut arena)?;
/// assert_eq!((block.reference_len(), block.fft_size(), block.block_size()), (127, 512, 386));
///
/// // A reference of four samples from the caller: N 16, B 13.
/// let r = [C32::new(1.0, 0.0), C32::new(0.0, 1.0), C32::new(-1.0, 0.0), C32::new(0.0, -1.0)];
/// let mut mem = AlignedBytes([0u8; 512]);
/// let bytes = CorrelateBlock::<ArenaStorage<'_>>::reference_storage_bytes(4, 0)?;
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut corr = CorrelateBlock::with_reference(&r, 0, &mut arena)?;
/// let mut x = [C32::default(); 26];
/// x[5..9].copy_from_slice(&r); // the reference at input index 5
/// let mut y = [C32::default(); 26];
/// let p = corr.process(&x, &mut y)?;
/// assert_eq!((p.consumed, p.produced), (26, 26)); // two blocks of 13
/// assert!((y[8].re - 4.0).abs() < 1e-5 && y[8].im.abs() < 1e-5); // the peak at 5 + L - 1
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CorrelateBlock<S: Storage> {
    config: CorrelateConfig,
    /// The reference came from the configuration (not from `with_reference`).
    from_code: bool,
    state: State<S>,
}

impl<S: Storage> core::fmt::Debug for CorrelateBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CorrelateBlock")
            .field("config", &self.config)
            .field("from_code", &self.from_code)
            .field("reference_len", &self.state.taps_len())
            .finish_non_exhaustive()
    }
}

impl<S: Storage> CorrelateBlock<S> {
    /// Builds the state for a reference of `l` samples and an `n`-point FFT: `fill` writes the
    /// taps into the first `l` values of the scratch (it may use the next `l` as room), their
    /// spectrum goes into the spectrum buffer.
    fn build(
        l: usize,
        n: usize,
        storage: &mut S,
        fill: impl FnOnce(&mut [C32], &mut [C32]),
    ) -> Result<State<S>> {
        let mut spectrum = storage.alloc::<C32>(n)?;
        let history = storage.alloc::<C32>(l - 1)?;
        let mut scratch = storage.alloc::<C32>(n)?;
        let twiddles = storage.alloc::<C32>(n / 2)?;
        let plan = FftPlan::new(n, twiddles)?;
        {
            let (taps, rest) = scratch
                .as_mut()
                .split_at_mut_checked(l)
                .ok_or(Error::Core(sdrk_core::Error::Internal("correlate scratch")))?;
            fill(taps, rest);
            xcorr_spectrum(taps, spectrum.as_mut(), &plan)?;
        }
        Ok(XcorrFftState::from_spectrum(
            spectrum, l, history, scratch, plan,
        )?)
    }

    /// Creates the block with the PRBS code of the configuration as the reference; the
    /// spectrum, the history, the scratch and the twiddles come from `storage`.
    ///
    /// # Errors
    /// The errors of [`CorrelateConfig::validate`] and of the storage.
    pub fn new(config: &CorrelateConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let (l, n) = (config.reference_len()?, config.resolved_fft_size()?);
        let mut prbs = config.prbs()?;
        let spc = config.samples_per_chip;
        let state = Self::build(l, n, storage, |taps, rest| {
            if let Some(r) = rest.get_mut(..taps.len()) {
                for chip in r.chunks_exact_mut(spc) {
                    let v = 1.0 - 2.0 * f32::from(prbs.next_bit());
                    chip.fill(C32::new(v, 0.0));
                }
                taps_from_reference(r, taps);
            }
        })?;
        Ok(Self {
            config: *config,
            from_code: true,
            state,
        })
    }

    /// Creates the block with a reference from the caller (any `C32` values, `L` at least 1)
    /// and the FFT size `fft_size` (0 for automatic, see [`correlate_fft_size`]). The parameters
    /// `code`, `seed` and `samples_per_chip` of such a block are not readable
    /// (`InvalidParameter`); `fft_size` reads as given.
    ///
    /// # Errors
    /// The errors of [`correlate_fft_size`] and of the storage.
    pub fn with_reference(reference: &[C32], fft_size: usize, storage: &mut S) -> Result<Self> {
        let l = reference.len();
        let n = correlate_fft_size(l, fft_size)?;
        let state = Self::build(l, n, storage, |taps, _| {
            taps_from_reference(reference, taps);
        })?;
        Ok(Self {
            config: CorrelateConfig {
                fft_size,
                ..CorrelateConfig::default()
            },
            from_code: false,
            state,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address: the spectrum and the scratch (`N` each), the history (`L - 1`) and the twiddles
    /// (`N / 2`), all `C32`.
    ///
    /// ```
    /// use sdrk_blocks::{ArenaStorage, CorrelateBlock, CorrelateConfig};
    /// let rate = sdrk::Rate::hz(48_000)?;
    /// let bytes = CorrelateBlock::<ArenaStorage<'_>>::storage_bytes(&CorrelateConfig::default(), rate)?;
    /// assert_eq!(bytes, (512 + 126 + 512 + 256) * 8);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    ///
    /// # Errors
    /// The errors of [`CorrelateConfig::validate`].
    pub fn storage_bytes(config: &CorrelateConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        plan_bytes(config.reference_len()?, config.resolved_fft_size()?)
    }

    /// Exact bytes of storage of [`with_reference`](Self::with_reference) with a reference of
    /// `reference_len` samples.
    ///
    /// # Errors
    /// The errors of [`correlate_fft_size`].
    pub fn reference_storage_bytes(reference_len: usize, fft_size: usize) -> Result<usize> {
        let n = correlate_fft_size(reference_len, fft_size)?;
        plan_bytes(reference_len, n)
    }

    /// Output rate (equal to the input rate).
    ///
    /// # Errors
    /// The errors of [`CorrelateConfig::validate`].
    pub fn rate_out(config: &CorrelateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The reference length `L`.
    pub fn reference_len(&self) -> usize {
        self.state.taps_len()
    }

    /// The FFT size `N`.
    pub fn fft_size(&self) -> usize {
        self.state.fft_size()
    }

    /// The block size `B = N - L + 1`: the outputs of one transform block and the `min_input`.
    pub fn block_size(&self) -> usize {
        self.state.block_size()
    }

    /// The kernel state (the spectrum of the taps, the history and the plan).
    pub fn state(&self) -> &State<S> {
        &self.state
    }
}

impl<S: Storage> Block for CorrelateBlock<S>
where
    Sdrk: XcorrFft<C32>,
{
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        Ok(Sdrk.xcorr_fft(&mut self.state, input, output)?)
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.state.ratio(),
            kind: self.state.ratio_kind(),
            delay: self.state.group_delay(),
        }
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        if DESCRIPTION.param(name).is_some() {
            Err(Error::ReadOnlyParameter)
        } else {
            unknown()
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        match name {
            "code" | "seed" | "samples_per_chip" if !self.from_code => {
                Err(Error::InvalidParameter {
                    name: "reference",
                    reason: "the block was built from a caller reference (with_reference)",
                })
            }
            _ => self.config.get(name),
        }
    }

    fn min_input(&self) -> usize {
        self.state.block_size()
    }
}
