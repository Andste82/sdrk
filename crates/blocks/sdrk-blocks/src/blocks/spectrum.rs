//! `spectrum`: windowed, averaged power spectrum in dB relative to full scale.

use sdrk::{C32, CQ15, Rate, Sdrk};
use sdrk_core::RealSample;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::kernels::spectrum::{MagSq, PowerDb, Window};
use sdrk_core::window::{Symmetry, blackman_harris, hann, kaiser};

use super::{check_count, head, tail, unknown};
use crate::block::{FrameBlock, FrameMeta, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// Smallest FFT size.
const MIN_FFT: usize = 16;
/// Largest FFT size.
const MAX_FFT: usize = 65_536;
/// Largest value of `averaging`.
const MAX_AVERAGING: i64 = 1024;
/// Largest value of `interval_samples`.
const MAX_INTERVAL: i64 = 1 << 30;

/// Description of the `spectrum` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "spectrum",
    summary: "Computes averaged power spectra in dB relative to full scale as frames.",
    details: "Takes fft_size consecutive samples, multiplies them by a window (periodic Hann, \
        four-term Blackman-Harris or Kaiser), transforms them with an FFT, averages the power \
        over `averaging` windows and converts it to dB such that a full-scale complex tone at a \
        bin centre reads 0 dB. The frame holds fft_size values in the order -fs/2 to +fs/2 \
        (DC at index fft_size / 2). Consecutive windows start interval_samples apart: smaller \
        than fft_size means overlapping windows, larger means samples are skipped. The input \
        type CQ15 uses a block-floating-point FFT; its levels are accurate to about 0.2 dB \
        down to -60 dB. Pitfall: the first frame is complete only after fft_size * averaging \
        samples (less with overlap).",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "frame",
        types: &[PortType::Frame {
            len_param: "fft_size",
        }],
        description: "Power in dB relative to full scale, fft_size values, DC in the middle.",
    }],
    params: &[
        ParamDesc {
            name: "fft_size",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: MIN_FFT as i64,
                max: MAX_FFT as i64,
                default: Some(1024),
            },
            typical: "1024; 4096 for finer resolution",
            mutability: Mutability::Build,
            description: "Number of FFT points, a power of two from 16 to 65536.",
        },
        ParamDesc {
            name: "window",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: &["hann", "blackman_harris", "kaiser"],
                default: 0,
            },
            typical: "hann",
            mutability: Mutability::Build,
            description: "Window function: hann, blackman_harris or kaiser.",
        },
        ParamDesc {
            name: "kaiser_beta",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 50.0,
                default: Some(8.6),
            },
            typical: "8.6",
            mutability: Mutability::Build,
            description: "Shape parameter of the Kaiser window (used only for window = kaiser).",
        },
        ParamDesc {
            name: "averaging",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_AVERAGING,
                default: Some(1),
            },
            typical: "1; 10 for a smoother display",
            mutability: Mutability::Runtime,
            description: "Number of windows whose power is averaged into one frame.",
        },
        ParamDesc {
            name: "interval_samples",
            unit: Unit::Samples,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_INTERVAL,
                default: None,
            },
            typical: "default fft_size; fft_size / 2 for 50 % overlap",
            mutability: Mutability::Runtime,
            description: "Distance between the starts of consecutive windows in samples. The \
                default is fft_size. Smaller values overlap the windows.",
        },
    ],
    required: &[],
};

/// Window function of the [`SpectrumBlock`].
///
/// ```
/// use sdrk_blocks::SpectrumWindow;
/// assert_eq!(SpectrumWindow::Hann.index(), 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectrumWindow {
    /// Periodic Hann window.
    Hann,
    /// Four-term Blackman-Harris window.
    BlackmanHarris,
    /// Kaiser window with the configured `kaiser_beta`.
    Kaiser,
}

impl SpectrumWindow {
    /// Index into the options of the `window` parameter.
    pub const fn index(self) -> u16 {
        match self {
            Self::Hann => 0,
            Self::BlackmanHarris => 1,
            Self::Kaiser => 2,
        }
    }

    fn from_index(i: u16) -> Result<Self> {
        match i {
            0 => Ok(Self::Hann),
            1 => Ok(Self::BlackmanHarris),
            2 => Ok(Self::Kaiser),
            _ => Err(Error::OutOfRange {
                name: "window",
                min: 0.0,
                max: 2.0,
                got: f64::from(i),
            }),
        }
    }
}

/// Configuration of [`SpectrumBlock`].
///
/// ```
/// use sdrk_blocks::SpectrumConfig;
/// let cfg = SpectrumConfig { fft_size: 4096, averaging: 4, ..SpectrumConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(2_400_000)?).is_ok());
/// assert_eq!(cfg.interval(), 4096);
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectrumConfig {
    /// FFT size, a power of two from 16 to 65536.
    pub fft_size: usize,
    /// Window function.
    pub window: SpectrumWindow,
    /// Shape parameter of the Kaiser window.
    pub kaiser_beta: f64,
    /// Number of windows averaged into one frame, 1 to 1024.
    pub averaging: usize,
    /// Distance between window starts in samples; `None` means `fft_size`.
    pub interval_samples: Option<usize>,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            window: SpectrumWindow::Hann,
            kaiser_beta: 8.6,
            averaging: 1,
            interval_samples: None,
        }
    }
}

impl SpectrumConfig {
    /// The distance between window starts with the default applied.
    pub fn interval(&self) -> usize {
        self.interval_samples.unwrap_or(self.fft_size)
    }

    /// Checks the configuration.
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        if !self.fft_size.is_power_of_two() {
            return Err(Error::InvalidParameter {
                name: "fft_size",
                reason: "the fft size must be a power of two",
            });
        }
        check_count("fft_size", self.fft_size, MIN_FFT, MAX_FFT)?;
        check_count("averaging", self.averaging, 1, MAX_AVERAGING as usize)?;
        check_count(
            "interval_samples",
            self.interval(),
            1,
            MAX_INTERVAL as usize,
        )?;
        if !(self.kaiser_beta >= 0.0 && self.kaiser_beta <= 50.0) {
            return Err(Error::OutOfRange {
                name: "kaiser_beta",
                min: 0.0,
                max: 50.0,
                got: self.kaiser_beta,
            });
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// ```
    /// use sdrk_blocks::{ParamValue, SpectrumConfig, SpectrumWindow};
    /// let mut cfg = SpectrumConfig::default();
    /// cfg.set("window", ParamValue::Choice(1))?;
    /// assert_eq!(cfg.window, SpectrumWindow::BlackmanHarris);
    /// cfg.set("interval_samples", ParamValue::I64(512))?;
    /// assert_eq!(cfg.interval(), 512);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "fft_size" => self.fft_size = value.as_usize()?,
            "window" => self.window = SpectrumWindow::from_index(value.as_choice()?)?,
            "kaiser_beta" => self.kaiser_beta = value.as_f64()?,
            "averaging" => self.averaging = value.as_usize()?,
            "interval_samples" => self.interval_samples = Some(value.as_usize()?),
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter (`interval_samples` reports the default `fft_size` while
    /// it is not set).
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "fft_size" => Ok(ParamValue::I64(self.fft_size as i64)),
            "window" => Ok(ParamValue::Choice(self.window.index())),
            "kaiser_beta" => Ok(ParamValue::F64(self.kaiser_beta)),
            "averaging" => Ok(ParamValue::I64(self.averaging as i64)),
            "interval_samples" => Ok(ParamValue::I64(self.interval() as i64)),
            _ => unknown(),
        }
    }
}

/// Sample types of the spectrum block: how the FFT output turns into power.
pub trait SpectrumSample: BlockSample {
    /// FFT scaling used for this sample type.
    const SCALING: Scaling;

    /// Adds the power of the FFT bins to `acc`. `exponent` is the FFT scale exponent;
    /// `tmp` is a work buffer of the same length.
    fn add_power(bins: &[Self], exponent: i32, acc: &mut [f32], tmp: &mut [f32]) -> Result<()>;
}

impl SpectrumSample for C32 {
    const SCALING: Scaling = Scaling::Default;

    fn add_power(bins: &[C32], _exponent: i32, acc: &mut [f32], tmp: &mut [f32]) -> Result<()> {
        <Sdrk as MagSq<C32>>::mag_sq(&Sdrk, bins, tmp)?;
        for (a, p) in acc.iter_mut().zip(tmp.iter()) {
            *a += *p;
        }
        Ok(())
    }
}

impl SpectrumSample for CQ15 {
    const SCALING: Scaling = Scaling::BlockFloat;

    /// Power in `f32` from the CQ15 bins, `(re^2 + im^2) 2^(2 exponent - 30)`; the Q15
    /// `mag_sq` kernel saturates for large bins and rounds weak bins to zero.
    fn add_power(bins: &[CQ15], exponent: i32, acc: &mut [f32], _tmp: &mut [f32]) -> Result<()> {
        let scale = libm::ldexpf(1.0, 2 * exponent - 30);
        for (a, b) in acc.iter_mut().zip(bins) {
            let (re, im) = (f32::from(b.re.0), f32::from(b.im.0));
            *a += (re * re + im * im) * scale;
        }
        Ok(())
    }
}

/// Averaged power spectrum for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, FrameBlock, SpectrumBlock, SpectrumConfig};
/// use sdrk::{Rate, C32};
/// let cfg = SpectrumConfig { fft_size: 16, ..SpectrumConfig::default() };
/// let rate = Rate::hz(16_000)?;
/// let bytes = SpectrumBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = SpectrumBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// // A tone in bin 4 (4 kHz) with amplitude 1.
/// let tone: Vec<C32> = (0..16)
///     .map(|n| {
///         let p = 2.0 * core::f64::consts::PI * 4.0 * n as f64 / 16.0;
///         C32::new(p.cos() as f32, p.sin() as f32)
///     })
///     .collect();
/// let mut frame = [0.0f32; 16];
/// let (used, meta) = block.push(&tone, &mut frame)?;
/// assert_eq!(used, 16);
/// assert!(meta.is_some());
/// assert!(frame[8 + 4].abs() < 0.01); // 0 dBFS at +4 kHz (DC is at index 8)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SpectrumBlock<T: SpectrumSample, S: Storage> {
    config: SpectrumConfig,
    plan: FftPlan<T, S::Buf<T>>,
    input: S::Buf<T>,
    bins: S::Buf<T>,
    window: S::Buf<T::Real>,
    acc: S::Buf<f32>,
    offset_db: f32,
    fill: usize,
    skip: usize,
    window_start: u64,
    first_sample: u64,
    count: usize,
}

impl<T: SpectrumSample, S: Storage> core::fmt::Debug for SpectrumBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpectrumBlock")
            .field("config", &self.config)
            .field("fill", &self.fill)
            .field("skip", &self.skip)
            .field("window_start", &self.window_start)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl<T: SpectrumSample, S: Storage> SpectrumBlock<T, S> {
    /// Creates the block; the buffers and the twiddle table come from `storage`.
    pub fn new(config: &SpectrumConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let n = config.fft_size;
        let input = storage.alloc::<T>(n)?;
        let bins = storage.alloc::<T>(n)?;
        let plan = FftPlan::new(n, storage.alloc::<T>(n / 2)?)?;
        let mut window = storage.alloc::<T::Real>(n)?;
        match config.window {
            SpectrumWindow::Hann => hann(window.as_mut(), Symmetry::Periodic)?,
            SpectrumWindow::BlackmanHarris => {
                blackman_harris(window.as_mut(), Symmetry::Periodic)?;
            }
            SpectrumWindow::Kaiser => {
                kaiser(window.as_mut(), config.kaiser_beta, Symmetry::Periodic)?;
            }
        }
        let acc = storage.alloc::<f32>(n)?;
        let sum: f64 = window.as_ref().iter().map(|w| w.to_f64()).sum();
        let reference = sum; // N * coherent gain = sum of the window
        let offset_db = (-20.0 * libm::log10(reference)) as f32;
        Ok(Self {
            config: *config,
            plan,
            input,
            bins,
            window,
            acc,
            offset_db,
            fill: 0,
            skip: 0,
            window_start: 0,
            first_sample: 0,
            count: 0,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &SpectrumConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        let n = config.fft_size;
        let mut plan = Plan::default();
        plan.add::<T>(n)?;
        plan.add::<T>(n)?;
        plan.add::<T>(n / 2)?;
        plan.add::<T::Real>(n)?;
        plan.add::<f32>(n)?;
        Ok(plan.bytes())
    }

    /// Frame rate is not a sample rate; this returns the input rate for symmetry with the
    /// stream blocks.
    pub fn rate_out(config: &SpectrumConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: SpectrumSample, S: Storage> SpectrumBlock<T, S>
where
    Sdrk: Window<T> + Fft<T> + PowerDb<f32>,
{
    /// Transforms the completed window and adds its power to the accumulator.
    fn accumulate(&mut self, tmp: &mut [f32]) -> Result<()> {
        <Sdrk as Window<T>>::window(
            &Sdrk,
            self.input.as_ref(),
            self.bins.as_mut(),
            self.window.as_ref(),
        )?;
        let scale = <Sdrk as Fft<T>>::fft(
            &Sdrk,
            &self.plan,
            self.bins.as_mut(),
            Direction::Forward,
            T::SCALING,
        )?;
        T::add_power(self.bins.as_ref(), scale.exponent, self.acc.as_mut(), tmp)
    }

    /// Turns the accumulator into a finished frame and clears it.
    fn finish(&mut self, frame: &mut [f32]) -> Result<FrameMeta> {
        let averaged = self.count;
        let div = averaged as f32;
        for a in self.acc.as_mut() {
            *a /= div;
        }
        <Sdrk as PowerDb<f32>>::power_db(&Sdrk, self.acc.as_ref(), frame, self.offset_db)?;
        frame.rotate_left(self.config.fft_size / 2);
        self.acc.as_mut().fill(0.0);
        self.count = 0;
        Ok(FrameMeta {
            first_sample: self.first_sample,
            averaged: averaged as u32,
            exponent: 0,
        })
    }
}

impl<T: SpectrumSample, S: Storage> FrameBlock for SpectrumBlock<T, S>
where
    Sdrk: Window<T> + Fft<T> + PowerDb<f32>,
{
    type In = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn frame_len(&self) -> usize {
        self.config.fft_size
    }

    fn push(&mut self, input: &[T], frame: &mut [f32]) -> Result<(usize, Option<FrameMeta>)> {
        let n = self.config.fft_size;
        if frame.len() != n {
            return Err(Error::Core(sdrk_core::Error::LengthMismatch {
                expected: n,
                got: frame.len(),
            }));
        }
        let internal = || Error::Core(sdrk_core::Error::Internal("spectrum window"));
        let mut consumed = 0usize;
        loop {
            let skipped = self.skip.min(input.len() - consumed);
            self.skip -= skipped;
            consumed += skipped;
            if self.skip > 0 {
                return Ok((consumed, None));
            }
            let take = (n - self.fill).min(input.len() - consumed);
            let dst = self
                .input
                .as_mut()
                .get_mut(self.fill..self.fill + take)
                .ok_or_else(internal)?;
            dst.copy_from_slice(head(tail(input, consumed)?, take)?);
            self.fill += take;
            consumed += take;
            if self.fill < n {
                return Ok((consumed, None));
            }
            // A window is complete. The frame buffer serves as the work buffer of the power.
            if self.count == 0 {
                self.first_sample = self.window_start;
            }
            self.accumulate(frame)?;
            self.count += 1;
            let step = self.config.interval();
            self.window_start += step as u64;
            if step < n {
                self.input.as_mut().copy_within(step..n, 0);
                self.fill = n - step;
            } else {
                self.fill = 0;
                self.skip = step - n;
            }
            if self.count >= self.config.averaging {
                let meta = self.finish(frame)?;
                return Ok((consumed, Some(meta)));
            }
        }
    }

    fn reset(&mut self) {
        self.fill = 0;
        self.skip = 0;
        self.window_start = 0;
        self.first_sample = 0;
        self.count = 0;
        self.acc.as_mut().fill(0.0);
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "averaging" => {
                self.config.averaging = value.as_usize()?;
                Ok(())
            }
            "interval_samples" => {
                self.config.interval_samples = Some(value.as_usize()?);
                Ok(())
            }
            _ => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }
}
