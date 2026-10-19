//! One factory per block kind.
//!
//! Every kind implements the small [`Kind`] trait (default config, `set`, the figures of a
//! configuration, the construction of the node). The generic [`Factory`] turns that into a
//! [`BlockFactory`]: JSON parameters to `ParamValue`, `set` for each given parameter, the
//! required-parameter check, the rate-dependent validation and the cost figures.

use std::any::Any;
use std::marker::PhantomData;

use sdrk::{C32, CQ15, Delay, Q15, Rate, Ratio};
use sdrk_blocks::design::kaiserord;
use sdrk_blocks::{
    BlockDescription, BlockSample, ConvertBlock, ConvertConfig, ConvertOutBlock, ConvertOutConfig,
    DcBlockBlock, DcBlockConfig, DdcBlock, DdcConfig, DesignSpec, DucBlock, DucConfig, FirBlock,
    FirConfig, FmDemodBlock, FmDemodConfig, FmModBlock, FmModConfig, ParamValue, PortType,
    ResamplerBlock, ResamplerConfig, ResamplerRealBlock, ScaleBlock, ScaleConfig, SpectrumBlock,
    SpectrumConfig, TUNED_PHASES,
};
use serde_json::{Map, Value};

use crate::model::Precision;
use crate::node::{FrameNode, Node, StreamNode};
use crate::params::{check_params, to_param};
use crate::registry::{BlockFactory, Checked, PortKind};
use crate::storage::VecStorage;
use crate::validate::{DiagCode, Diagnostic};

type BlockResult<T> = Result<T, sdrk_blocks::Error>;

/// The numbers a configuration yields for a given input rate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Figures {
    pub rate_out: Rate,
    pub memory_bytes: usize,
    pub cost: f64,
    pub taps: usize,
    pub delay: Delay,
    pub min_input: usize,
    pub output: PortKind,
}

/// A block kind, as far as the generic factory needs to know it.
pub(crate) trait Kind: Send + Sync + 'static {
    /// The configuration type of the block.
    type Cfg: Clone + Send + Sync + 'static;
    /// The kind name, as in the block description.
    const KIND: &'static str;
    /// Names of the kernels the block calls.
    fn kernels() -> &'static [&'static str];
    /// The configuration with every default.
    fn default_cfg() -> Self::Cfg;
    /// Sets one parameter (kind and absolute range are checked).
    fn set(cfg: &mut Self::Cfg, name: &str, value: ParamValue) -> BlockResult<()>;
    /// Validates the configuration for the input and computes the figures.
    fn plan(
        cfg: &Self::Cfg,
        input: PortType,
        rate: Rate,
        precision: Precision,
    ) -> BlockResult<Figures>;
    /// Builds the node.
    fn build(
        cfg: &Self::Cfg,
        input: PortType,
        rate: Rate,
        precision: Precision,
        storage: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>>;
}

/// The configuration together with everything `build` needs.
struct Held<C> {
    cfg: C,
    input: PortType,
    rate: Rate,
    precision: Precision,
}

/// The generic factory.
pub(crate) struct Factory<K: Kind> {
    desc: &'static BlockDescription,
    _kind: PhantomData<fn() -> K>,
}

impl<K: Kind> Factory<K> {
    pub(crate) fn boxed(desc: &'static BlockDescription) -> Box<dyn BlockFactory> {
        Box::new(Self {
            desc,
            _kind: PhantomData,
        })
    }
}

fn diag(path: String, code: DiagCode, message: String) -> Diagnostic {
    Diagnostic {
        path,
        code,
        message,
    }
}

/// Diagnostic for an error of `Kind::set` (path relative to the node).
fn set_error(name: &str, e: &sdrk_blocks::Error) -> Diagnostic {
    let path = format!("params.{name}");
    let code = match e {
        sdrk_blocks::Error::OutOfRange { .. } => DiagCode::OutOfRange,
        sdrk_blocks::Error::WrongParamType => DiagCode::WrongType,
        sdrk_blocks::Error::UnknownParameter => DiagCode::UnknownParam,
        _ => DiagCode::InvalidValue,
    };
    diag(path, code, e.to_string())
}

/// Diagnostic for an error of `Kind::plan` (path relative to the node).
fn plan_error(e: &sdrk_blocks::Error) -> Diagnostic {
    use sdrk_blocks::Error as E;
    let (path, code) = match e {
        E::OutOfRange { name, .. } => (format!("params.{name}"), DiagCode::RateUnsupported),
        E::InvalidParameter { name, .. } => (format!("params.{name}"), DiagCode::InvalidValue),
        E::Storage { .. } => (String::new(), DiagCode::LimitExceeded),
        _ => (String::new(), DiagCode::RateUnsupported),
    };
    diag(path, code, e.to_string())
}

/// Reads the parameters of a node into a configuration.
fn parse_params<K: Kind>(
    desc: &'static BlockDescription,
    params: &Map<String, Value>,
) -> Result<K::Cfg, Vec<Diagnostic>> {
    let mut diags = Vec::new();
    check_params(desc, params.iter(), "params", &mut diags);
    if !diags.is_empty() {
        return Err(diags);
    }
    let mut cfg = K::default_cfg();
    for (name, value) in params {
        if let Some(p) = desc.param(name) {
            if let Ok(pv) = to_param(p, value) {
                if let Err(e) = K::set(&mut cfg, name, pv) {
                    diags.push(set_error(name, &e));
                }
            }
        }
    }
    if diags.is_empty() {
        Ok(cfg)
    } else {
        Err(diags)
    }
}

impl<K: Kind> BlockFactory for Factory<K> {
    fn description(&self) -> &'static BlockDescription {
        self.desc
    }

    fn check(
        &self,
        params: &Map<String, Value>,
        input: PortType,
        rate_in: Rate,
        precision: Precision,
    ) -> Result<Checked, Vec<Diagnostic>> {
        let cfg = parse_params::<K>(self.desc, params)?;
        let accepted = self
            .desc
            .inputs
            .first()
            .is_some_and(|p| p.types.contains(&input));
        if !accepted {
            return Err(vec![diag(
                String::new(),
                DiagCode::TypeMismatch,
                format!("`{}` does not accept {input:?} input", self.desc.kind),
            )]);
        }
        let f = K::plan(&cfg, input, rate_in, precision).map_err(|e| vec![plan_error(&e)])?;
        Ok(Checked {
            rate_out: f.rate_out,
            memory_bytes: f.memory_bytes,
            cost: f.cost,
            output: f.output,
            taps: f.taps,
            delay: f.delay,
            min_input: f.min_input,
            config: Box::new(Held {
                cfg,
                input,
                rate: rate_in,
                precision,
            }) as Box<dyn Any + Send + Sync>,
        })
    }

    fn kernels(&self) -> &'static [&'static str] {
        K::kernels()
    }

    fn build(
        &self,
        checked: &Checked,
        storage: &mut VecStorage,
    ) -> Result<Box<dyn Node>, Diagnostic> {
        let held = checked
            .config
            .downcast_ref::<Held<K::Cfg>>()
            .ok_or_else(|| {
                diag(
                    String::new(),
                    DiagCode::InvalidValue,
                    "the checked configuration belongs to another kind".to_string(),
                )
            })?;
        K::build(&held.cfg, held.input, held.rate, held.precision, storage).map_err(|e| {
            let mut d = plan_error(&e);
            if matches!(e, sdrk_blocks::Error::Storage { .. }) {
                d.code = DiagCode::LimitExceeded;
            }
            d
        })
    }
}

/// Runs `$body` with the type alias `$t` set to `C32` or `CQ15`, according to the precision.
macro_rules! with_complex {
    ($prec:expr, $t:ident, $body:expr) => {
        match $prec {
            Precision::F32 => {
                type $t = C32;
                $body
            }
            Precision::Q15 => {
                type $t = CQ15;
                $body
            }
        }
    };
}

/// Runs `$body` with the type alias `$t` set to `f32` or `Q15`.
macro_rules! with_real {
    ($prec:expr, $t:ident, $body:expr) => {
        match $prec {
            Precision::F32 => {
                type $t = f32;
                $body
            }
            Precision::Q15 => {
                type $t = Q15;
                $body
            }
        }
    };
}

pub(crate) fn complex_port(precision: Precision) -> PortType {
    match precision {
        Precision::F32 => PortType::C32,
        Precision::Q15 => PortType::CQ15,
    }
}

pub(crate) fn real_port(precision: Precision) -> PortType {
    match precision {
        Precision::F32 => PortType::F32,
        Precision::Q15 => PortType::Q15,
    }
}

fn is_complex(t: PortType) -> bool {
    matches!(t, PortType::C32 | PortType::CQ15)
}

fn mismatch(what: &'static str) -> sdrk_blocks::Error {
    sdrk_blocks::Error::InvalidParameter {
        name: "input",
        reason: what,
    }
}

fn delay_half(taps: usize) -> Delay {
    Delay::new(taps.saturating_sub(1) as u64, 2).unwrap_or(Delay::ZERO)
}

/// Samples and bytes of the smallest packing group of a streaming format.
fn packing_group(format: &sdrk_device::SampleFormat) -> (usize, usize) {
    (1..=64usize)
        .find_map(|n| format.bytes_for_samples(n).ok().map(|b| (n, b)))
        .unwrap_or((1, 1))
}

fn stream(t: PortType) -> PortKind {
    PortKind::Stream(t)
}

// ---------------------------------------------------------------- convert

pub(crate) struct ConvertKind;

impl Kind for ConvertKind {
    type Cfg = ConvertConfig;

    const KIND: &'static str = "convert";
    fn kernels() -> &'static [&'static str] {
        &["convert"]
    }
    fn default_cfg() -> ConvertConfig {
        ConvertConfig::default()
    }
    fn set(cfg: &mut ConvertConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(
        cfg: &ConvertConfig,
        input: PortType,
        rate: Rate,
        p: Precision,
    ) -> BlockResult<Figures> {
        if input != PortType::Raw {
            return Err(mismatch("convert needs raw bytes"));
        }
        let rate_out = ConvertBlock::<C32, VecStorage>::rate_out(cfg, rate)?;
        Ok(Figures {
            rate_out,
            memory_bytes: ConvertBlock::<C32, VecStorage>::storage_bytes(cfg, rate)?,
            cost: 2.0,
            taps: 0,
            delay: Delay::ZERO,
            min_input: packing_group(&cfg.format).1,
            output: stream(complex_port(p)),
        })
    }
    fn build(
        cfg: &ConvertConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(ConvertBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// ------------------------------------------------------------ convert_out

pub(crate) struct ConvertOutKind;

impl Kind for ConvertOutKind {
    type Cfg = ConvertOutConfig;

    const KIND: &'static str = "convert_out";
    fn kernels() -> &'static [&'static str] {
        &["convert_out"]
    }
    fn default_cfg() -> ConvertOutConfig {
        ConvertOutConfig::default()
    }
    fn set(cfg: &mut ConvertOutConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(
        cfg: &ConvertOutConfig,
        input: PortType,
        rate: Rate,
        _p: Precision,
    ) -> BlockResult<Figures> {
        if !is_complex(input) {
            return Err(mismatch("convert_out needs complex samples"));
        }
        Ok(Figures {
            rate_out: ConvertOutBlock::<C32, VecStorage>::rate_out(cfg, rate)?,
            memory_bytes: ConvertOutBlock::<C32, VecStorage>::storage_bytes(cfg, rate)?,
            cost: 2.0,
            taps: 0,
            delay: Delay::ZERO,
            min_input: packing_group(&cfg.format).0,
            output: stream(PortType::Raw),
        })
    }
    fn build(
        cfg: &ConvertOutConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(ConvertOutBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// ------------------------------------------------------------------ scale

pub(crate) struct ScaleKind;

impl Kind for ScaleKind {
    type Cfg = ScaleConfig;

    const KIND: &'static str = "scale";
    fn kernels() -> &'static [&'static str] {
        &["scale"]
    }
    fn default_cfg() -> ScaleConfig {
        ScaleConfig::default()
    }
    fn set(cfg: &mut ScaleConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(cfg: &ScaleConfig, input: PortType, rate: Rate, p: Precision) -> BlockResult<Figures> {
        if !is_complex(input) {
            return Err(mismatch("scale needs complex samples"));
        }
        let max = with_complex!(p, T, <T as BlockSample>::MAX_GAIN_DB);
        if cfg.gain_db > max {
            return Err(sdrk_blocks::Error::OutOfRange {
                name: "gain_db",
                min: -120.0,
                max,
                got: cfg.gain_db,
            });
        }
        Ok(Figures {
            rate_out: ScaleBlock::<C32, VecStorage>::rate_out(cfg, rate)?,
            memory_bytes: ScaleBlock::<C32, VecStorage>::storage_bytes(cfg, rate)?,
            cost: 1.0,
            taps: 0,
            delay: Delay::ZERO,
            min_input: 1,
            output: stream(complex_port(p)),
        })
    }
    fn build(
        cfg: &ScaleConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(ScaleBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// ---------------------------------------------------------------- dc_block

pub(crate) struct DcBlockKind;

impl Kind for DcBlockKind {
    type Cfg = DcBlockConfig;

    const KIND: &'static str = "dc_block";
    fn kernels() -> &'static [&'static str] {
        &["dc_block"]
    }
    fn default_cfg() -> DcBlockConfig {
        DcBlockConfig::default()
    }
    fn set(cfg: &mut DcBlockConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(
        cfg: &DcBlockConfig,
        input: PortType,
        rate: Rate,
        p: Precision,
    ) -> BlockResult<Figures> {
        if !is_complex(input) {
            return Err(mismatch("dc_block needs complex samples"));
        }
        Ok(Figures {
            rate_out: DcBlockBlock::<C32, VecStorage>::rate_out(cfg, rate)?,
            memory_bytes: DcBlockBlock::<C32, VecStorage>::storage_bytes(cfg, rate)?,
            cost: 3.0,
            taps: 0,
            delay: Delay::ZERO,
            min_input: 1,
            output: stream(complex_port(p)),
        })
    }
    fn build(
        cfg: &DcBlockConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(DcBlockBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// --------------------------------------------------------------------- fir

pub(crate) struct FirKind;

impl Kind for FirKind {
    type Cfg = FirConfig;

    const KIND: &'static str = "fir";
    fn kernels() -> &'static [&'static str] {
        &["fir"]
    }
    fn default_cfg() -> FirConfig {
        FirConfig::default()
    }
    fn set(cfg: &mut FirConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(cfg: &FirConfig, input: PortType, rate: Rate, p: Precision) -> BlockResult<Figures> {
        if !is_complex(input) {
            return Err(mismatch("fir needs complex samples"));
        }
        let taps = cfg.num_taps(rate)?;
        with_complex!(
            p,
            T,
            Ok(Figures {
                rate_out: FirBlock::<T, VecStorage>::rate_out(cfg, rate)?,
                memory_bytes: FirBlock::<T, VecStorage>::storage_bytes(cfg, rate)?,
                cost: taps as f64,
                taps,
                delay: delay_half(taps),
                min_input: 1,
                output: stream(complex_port(p)),
            })
        )
    }
    fn build(
        cfg: &FirConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(FirBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// --------------------------------------------------------------------- ddc

pub(crate) struct DdcKind;

impl Kind for DdcKind {
    type Cfg = DdcConfig;

    const KIND: &'static str = "ddc";
    fn kernels() -> &'static [&'static str] {
        &["rotate", "fir_decimate", "ddc"]
    }
    fn default_cfg() -> DdcConfig {
        DdcConfig::default()
    }
    fn set(cfg: &mut DdcConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(cfg: &DdcConfig, input: PortType, rate: Rate, p: Precision) -> BlockResult<Figures> {
        if !is_complex(input) {
            return Err(mismatch("ddc needs complex samples"));
        }
        let taps = cfg.num_taps(rate)?;
        with_complex!(
            p,
            T,
            Ok(Figures {
                rate_out: DdcBlock::<T, VecStorage>::rate_out(cfg, rate)?,
                memory_bytes: DdcBlock::<T, VecStorage>::storage_bytes(cfg, rate)?,
                cost: 4.0 + 4.0 * taps as f64 / cfg.decimation as f64,
                taps,
                delay: delay_half(taps),
                min_input: cfg.decimation,
                output: stream(complex_port(p)),
            })
        )
    }
    fn build(
        cfg: &DdcConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(DdcBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// --------------------------------------------------------------------- duc

pub(crate) struct DucKind;

impl Kind for DucKind {
    type Cfg = DucConfig;

    const KIND: &'static str = "duc";
    fn kernels() -> &'static [&'static str] {
        &["fir_interpolate", "rotate", "duc"]
    }
    fn default_cfg() -> DucConfig {
        DucConfig::default()
    }
    fn set(cfg: &mut DucConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(cfg: &DucConfig, input: PortType, rate: Rate, p: Precision) -> BlockResult<Figures> {
        if !is_complex(input) {
            return Err(mismatch("duc needs complex samples"));
        }
        let taps = cfg.num_taps(rate)?;
        let l = cfg.interpolation as u64;
        with_complex!(
            p,
            T,
            Ok(Figures {
                rate_out: DucBlock::<T, VecStorage>::rate_out(cfg, rate)?,
                memory_bytes: DucBlock::<T, VecStorage>::storage_bytes(cfg, rate)?,
                cost: 4.0 * l as f64 + 4.0 * taps as f64,
                taps,
                delay: Delay::new(taps.saturating_sub(1) as u64, 2 * l).unwrap_or(Delay::ZERO),
                min_input: 1,
                output: stream(complex_port(p)),
            })
        )
    }
    fn build(
        cfg: &DucConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(DucBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// ---------------------------------------------------------------- fm_demod

pub(crate) struct FmDemodKind;

impl Kind for FmDemodKind {
    type Cfg = FmDemodConfig;

    const KIND: &'static str = "fm_demod";
    fn kernels() -> &'static [&'static str] {
        &["quad_demod", "deemphasis", "resample_real", "fm_demod"]
    }
    fn default_cfg() -> FmDemodConfig {
        FmDemodConfig::default()
    }
    fn set(cfg: &mut FmDemodConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(
        cfg: &FmDemodConfig,
        input: PortType,
        rate: Rate,
        p: Precision,
    ) -> BlockResult<Figures> {
        if !is_complex(input) {
            return Err(mismatch("fm_demod needs complex samples"));
        }
        let taps = cfg.audio_taps(rate)?;
        let d = cfg.audio_decimation;
        with_complex!(
            p,
            T,
            Ok(Figures {
                rate_out: FmDemodBlock::<T, VecStorage>::rate_out(cfg, rate)?,
                memory_bytes: FmDemodBlock::<T, VecStorage>::storage_bytes(cfg, rate)?,
                cost: 10.0 + taps as f64 / d as f64,
                taps,
                delay: Delay::new(taps.max(1) as u64, 2).unwrap_or(Delay::ZERO),
                min_input: d,
                output: stream(real_port(p)),
            })
        )
    }
    fn build(
        cfg: &FmDemodConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(FmDemodBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// ------------------------------------------------------------------ fm_mod

pub(crate) struct FmModKind;

impl Kind for FmModKind {
    type Cfg = FmModConfig;

    const KIND: &'static str = "fm_mod";
    fn kernels() -> &'static [&'static str] {
        &["fm_mod"]
    }
    fn default_cfg() -> FmModConfig {
        FmModConfig::default()
    }
    fn set(cfg: &mut FmModConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(cfg: &FmModConfig, input: PortType, rate: Rate, p: Precision) -> BlockResult<Figures> {
        if is_complex(input) {
            return Err(mismatch("fm_mod needs real samples"));
        }
        Ok(Figures {
            rate_out: FmModBlock::<C32, VecStorage>::rate_out(cfg, rate)?,
            memory_bytes: FmModBlock::<C32, VecStorage>::storage_bytes(cfg, rate)?,
            cost: 6.0,
            taps: 0,
            delay: Delay::ZERO,
            min_input: 1,
            output: stream(complex_port(p)),
        })
    }
    fn build(
        cfg: &FmModConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(StreamNode::new(FmModBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}

// --------------------------------------------------------------- resampler

pub(crate) struct ResamplerKind;

/// Taps, group delay and smallest input of the resampler for a configuration, from the same
/// design rules as the block (`ResamplerConfig::validate` has already accepted it).
fn resampler_figures(
    cfg: &ResamplerConfig,
    rate: Rate,
    rate_out: Rate,
) -> BlockResult<(usize, Delay, usize, f64)> {
    let fin = rate.as_f64();
    let fout = rate_out.as_f64();
    let low = fin.min(fout);
    let spec = DesignSpec {
        cutoff_hz: cfg.cutoff_hz.unwrap_or(0.45 * low),
        transition_hz: cfg.transition_hz.unwrap_or(0.1 * low),
        attenuation_db: cfg.attenuation_db,
    };
    let ratio = Ratio::new(
        rate_out.num().saturating_mul(rate.den()),
        rate.num().saturating_mul(rate_out.den()),
    )?;
    if cfg.tunable {
        let phases = TUNED_PHASES as f64;
        let (n_hi, _beta) = kaiserord(spec.attenuation_db, spec.transition_hz, phases * fin)?;
        let k = n_hi.div_ceil(TUNED_PHASES);
        let delay = Delay::new(
            (TUNED_PHASES * k).saturating_sub(1) as u64,
            2 * TUNED_PHASES as u64,
        )
        .unwrap_or(Delay::ZERO);
        // The step is the input per output sample, corrected by up to 1000 ppm.
        let step = fin / fout * 1.002;
        let min_input = step as usize + 4;
        let cost = k as f64 * ratio.num() as f64 / ratio.den() as f64 + 4.0;
        Ok((TUNED_PHASES * k, delay, min_input, cost))
    } else {
        let (l, m) = (ratio.num(), ratio.den());
        let taps = spec.num_taps(l as f64 * fin, cfg.max_taps)?;
        let delay = Delay::new(taps.saturating_sub(1) as u64, 2 * l).unwrap_or(Delay::ZERO);
        let cost = taps as f64 / m as f64;
        Ok((taps, delay, m as usize, cost))
    }
}

impl Kind for ResamplerKind {
    type Cfg = ResamplerConfig;

    const KIND: &'static str = "resampler";
    fn kernels() -> &'static [&'static str] {
        &["resample", "resample_real", "dot", "dot_real"]
    }
    fn default_cfg() -> ResamplerConfig {
        ResamplerConfig::default()
    }
    fn set(cfg: &mut ResamplerConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(
        cfg: &ResamplerConfig,
        input: PortType,
        rate: Rate,
        p: Precision,
    ) -> BlockResult<Figures> {
        let complex = is_complex(input);
        let (rate_out, memory_bytes) = if complex {
            with_complex!(
                p,
                T,
                (
                    ResamplerBlock::<T, VecStorage>::rate_out(cfg, rate)?,
                    ResamplerBlock::<T, VecStorage>::storage_bytes(cfg, rate)?
                )
            )
        } else {
            with_real!(
                p,
                R,
                (
                    ResamplerRealBlock::<R, VecStorage>::rate_out(cfg, rate)?,
                    ResamplerRealBlock::<R, VecStorage>::storage_bytes(cfg, rate)?
                )
            )
        };
        let (taps, delay, min_input, cost) = resampler_figures(cfg, rate, rate_out)?;
        Ok(Figures {
            rate_out,
            memory_bytes,
            cost,
            taps,
            delay,
            min_input,
            output: stream(input),
        })
    }
    fn build(
        cfg: &ResamplerConfig,
        input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        match (is_complex(input), p) {
            (true, Precision::F32) => resampler_c32(cfg, rate, st),
            (true, Precision::Q15) => resampler_cq15(cfg, rate, st),
            (false, Precision::F32) => resampler_f32(cfg, rate, st),
            (false, Precision::Q15) => resampler_q15(cfg, rate, st),
        }
    }
}

// One function per sample type: the four blocks differ in size, and a single function that
// builds all of them would need the stack of all four at once.

macro_rules! resampler_builder {
    ($name:ident, $block:ident, $t:ty) => {
        fn $name(
            cfg: &ResamplerConfig,
            rate: Rate,
            st: &mut VecStorage,
        ) -> BlockResult<Box<dyn Node>> {
            Ok(Box::new(StreamNode::new($block::<$t, _>::new(
                cfg, rate, st,
            )?)))
        }
    };
}

resampler_builder!(resampler_c32, ResamplerBlock, C32);
resampler_builder!(resampler_cq15, ResamplerBlock, CQ15);
resampler_builder!(resampler_f32, ResamplerRealBlock, f32);
resampler_builder!(resampler_q15, ResamplerRealBlock, Q15);

// ---------------------------------------------------------------- spectrum

pub(crate) struct SpectrumKind;

impl Kind for SpectrumKind {
    type Cfg = SpectrumConfig;

    const KIND: &'static str = "spectrum";
    fn kernels() -> &'static [&'static str] {
        &["window", "fft", "power_db"]
    }
    fn default_cfg() -> SpectrumConfig {
        SpectrumConfig::default()
    }
    fn set(cfg: &mut SpectrumConfig, name: &str, value: ParamValue) -> BlockResult<()> {
        cfg.set(name, value)
    }
    fn plan(
        cfg: &SpectrumConfig,
        input: PortType,
        rate: Rate,
        p: Precision,
    ) -> BlockResult<Figures> {
        if !is_complex(input) {
            return Err(mismatch("spectrum needs complex samples"));
        }
        let n = cfg.fft_size as f64;
        Ok(Figures {
            rate_out: SpectrumBlock::<C32, VecStorage>::rate_out(cfg, rate)?,
            memory_bytes: with_complex!(
                p,
                T,
                SpectrumBlock::<T, VecStorage>::storage_bytes(cfg, rate)?
            ),
            cost: (2.5 * n * n.log2() + 3.0 * n) / cfg.interval() as f64,
            taps: 0,
            delay: Delay::ZERO,
            min_input: 1,
            output: PortKind::Frames { len: cfg.fft_size },
        })
    }
    fn build(
        cfg: &SpectrumConfig,
        _input: PortType,
        rate: Rate,
        p: Precision,
        st: &mut VecStorage,
    ) -> BlockResult<Box<dyn Node>> {
        with_complex!(
            p,
            T,
            Ok(Box::new(FrameNode::new(SpectrumBlock::<T, _>::new(
                cfg, rate, st
            )?)))
        )
    }
}
