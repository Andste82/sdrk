//! `real_to_iq`: shift by fs/4 and halfband decimation of a real signal.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::halfband::{RealToComplex, RealToComplexState};

use super::divide_rate;
use super::halfband_decimate::{HalfbandDecimateConfig, PARAMS, timing_of};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::Result;
use crate::sample::{BlockSample, TunedElem};
use crate::storage::Storage;

/// Description of the `real_to_iq` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "real_to_iq",
    summary: "Turns a real signal into a complex one at half the rate (fs/4 shift and halfband).",
    details: "Multiplies the real input by 1, -j, -1, j (a shift down by a quarter of the input \
        rate) and applies a halfband low-pass with a decimation by 2, in one step. A real \
        component at the frequency f (0 to fs / 2) appears in the complex output at f - fs / 4; \
        the output centre (0 Hz) is the input frequency fs / 4, and the mirror component at -f \
        is removed by the filter. A station at the real frequency f is tuned with a ddc \
        frequency of f - fs / 4. The usable band is fs / 4 plus or minus (fs / 4 - transition / \
        2) around the output centre. The group delay is (taps - 1) / 2 input samples. Use it \
        for real-sampling devices such as the Airspy R2 and Mini.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32, PortType::Q15],
        description: "Real samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples at half the input rate (f32 gives C32, Q15 gives CQ15).",
    }],
    params: PARAMS,
    required: &[],
};

/// Shift by fs/4 and halfband decimation for real input: `f32` to `C32`, `Q15` to `CQ15`.
///
/// The output centre is the input frequency fs / 4. With the sample counter `n` since
/// construction or `reset`, the input is multiplied by `(-j)^n` first.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, HalfbandDecimateConfig, RealToIqBlock};
/// use sdrk::{Rate, C32};
/// let cfg = HalfbandDecimateConfig::default();
/// let rate = Rate::hz(48_000)?;
/// let mut mem = vec![0u8; RealToIqBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = RealToIqBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = vec![C32::default(); 200];
/// // A tone at fs/4 (1, 0, -1, 0, ...) is the output centre: it becomes a constant of half the
/// // amplitude. (A constant input, at 0 Hz, would land on the band edge at -fs/4.)
/// let x: Vec<f32> = (0..400).map(|n| [1.0, 0.0, -1.0, 0.0][n % 4]).collect();
/// let p = block.process(&x, &mut out)?;
/// assert_eq!(p.produced, 200);
/// assert!((out[199].re - 0.5).abs() < 1e-3 && out[199].im.abs() < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RealToIqBlock<T: BlockSample, S: Storage> {
    config: HalfbandDecimateConfig,
    state: RealToComplexState<T, S::Buf<T::Real>, S::Buf<T>>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for RealToIqBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RealToIqBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> RealToIqBlock<T, S> {
    /// Designs the taps and creates the block; taps and history come from `storage`.
    pub fn new(config: &HalfbandDecimateConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let taps = config.alloc_taps::<T::Real, S>(rate_in, storage, <T as TunedElem>::tap)?;
        let hist = storage.alloc::<T>(taps.as_ref().len() - 1)?;
        Ok(Self {
            config: config.resolved(rate_in),
            state: RealToComplexState::new(taps, hist)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &HalfbandDecimateConfig, rate_in: Rate) -> Result<usize> {
        config.plan_bytes::<T::Real, T>(rate_in)
    }

    /// Output rate for the configuration (half the input rate).
    pub fn rate_out(config: &HalfbandDecimateConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, 2)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.state.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for RealToIqBlock<T, S>
where
    Sdrk: RealToComplex<T>,
{
    type In = T::Real;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T::Real], output: &mut [T]) -> Result<Progress> {
        Ok(<Sdrk as RealToComplex<T>>::real_to_complex(
            &Sdrk,
            &mut self.state,
            input,
            output,
        )?)
    }

    fn timing(&self) -> BlockTiming {
        timing_of(&self.state)
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        runtime_param(&DESCRIPTION, name, &value).map(|_| ())
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        2
    }
}
