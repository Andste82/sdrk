//! The block registry: kinds, factories and the entry points of validation and build.

use std::any::Any;
use std::collections::BTreeMap;

use sdrk::{Delay, Rate};
use sdrk_blocks::{BlockDescription, PortType, all_descriptions};
use serde_json::{Map, Value};

use crate::analysis;
use crate::error::{Error, Result};
use crate::kinds::{
    ConvertKind, ConvertOutKind, DcBlockKind, DdcKind, DucKind, Factory, FirKind, FmDemodKind,
    FmModKind, Kind, ResamplerKind, ScaleKind, SpectrumKind, SquelchKind,
};
use crate::model::{PipelineSpec, Precision};
use crate::node::Node;
use crate::runner::Pipeline;
use crate::storage::VecStorage;
use crate::validate::{Diagnostic, Report};

/// What a node produces: a stream of one sample type or frames of `f32` values.
///
/// ```
/// use sdrk_blocks::PortType;
/// use sdrk_pipeline::PortKind;
/// assert_ne!(PortKind::Stream(PortType::C32), PortKind::Frames { len: 1024 });
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortKind {
    /// A stream of the given sample type.
    Stream(PortType),
    /// Frames of `len` `f32` values.
    Frames {
        /// Number of values in a frame.
        len: usize,
    },
}

/// The result of checking the parameters of one node: the figures the build needs and the
/// parsed configuration (opaque).
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::PortType;
/// use sdrk_pipeline::{Precision, Registry};
/// let registry = Registry::standard();
/// let factory = registry.factory("ddc").unwrap();
/// let params = serde_json::json!({ "decimation": 10 });
/// let checked = factory
///     .check(params.as_object().unwrap(), PortType::C32, Rate::hz(2_400_000)?, Precision::F32)
///     .unwrap();
/// assert_eq!(checked.rate_out, Rate::hz(240_000)?);
/// assert!(checked.cost > 0.0);
/// # Ok::<(), sdrk::Error>(())
/// ```
pub struct Checked {
    /// Output rate of the node.
    pub rate_out: Rate,
    /// Bytes of block storage (`storage_bytes`).
    pub memory_bytes: usize,
    /// Estimated multiply-accumulates per input sample of the node (see `docs/pipeline.md`).
    pub cost: f64,
    /// What the node produces.
    pub output: PortKind,
    /// Number of filter taps the node uses (0 if it has none); limited by
    /// `Limits::max_taps_per_node`.
    pub taps: usize,
    /// Group delay in input samples of the node.
    pub delay: Delay,
    /// Smallest input count that is guaranteed to produce an output (sizes the buffers).
    pub min_input: usize,
    pub(crate) config: Box<dyn Any + Send + Sync>,
}

impl core::fmt::Debug for Checked {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Checked")
            .field("rate_out", &self.rate_out)
            .field("memory_bytes", &self.memory_bytes)
            .field("cost", &self.cost)
            .field("output", &self.output)
            .field("taps", &self.taps)
            .field("delay", &self.delay)
            .field("min_input", &self.min_input)
            .finish_non_exhaustive()
    }
}

/// Creates the nodes of one block kind.
pub trait BlockFactory: Send + Sync {
    /// The description of the block kind.
    fn description(&self) -> &'static BlockDescription;
    /// Parses the params (already JSON-checked) into a config and validates it for this input
    /// rate and precision. `input` is the port type arriving at this node (it lets `resampler`
    /// choose the complex or the real block). The paths of the diagnostics are relative to the
    /// node, e.g. `params.frequency_hz`; an empty path means the node itself.
    fn check(
        &self,
        params: &Map<String, Value>,
        input: PortType,
        rate_in: Rate,
        precision: Precision,
    ) -> Result<Checked, Vec<Diagnostic>>;
    /// Names of the kernels the block calls (used by backend preferences in a later milestone).
    fn kernels(&self) -> &'static [&'static str];
    /// Builds the node from a checked configuration.
    fn build(
        &self,
        checked: &Checked,
        storage: &mut VecStorage,
    ) -> Result<Box<dyn Node>, Diagnostic>;
}

/// Limits that `validate` and `build` enforce.
///
/// ```
/// use sdrk_pipeline::Limits;
/// let l = Limits::default();
/// assert_eq!(l.max_nodes, 256);
/// assert_eq!(l.max_taps_per_node, 16_383);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Most nodes (automatic ones included). Default 256.
    pub max_nodes: usize,
    /// Most bytes of memory (buffers, queues and block storage). Default 256 MiB.
    pub max_memory_bytes: usize,
    /// Largest estimated multiply-accumulates per pipeline input sample. Default infinite.
    pub max_cost_per_input_sample: f64,
    /// Most filter taps in one node. Default 16 383.
    pub max_taps_per_node: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_nodes: 256,
            max_memory_bytes: 256 * 1024 * 1024,
            max_cost_per_input_sample: f64::INFINITY,
            max_taps_per_node: 16_383,
        }
    }
}

/// Options of `validate` and `build`.
///
/// ```
/// use sdrk_pipeline::{BuildOptions, Limits};
/// let opts = BuildOptions { limits: Limits { max_nodes: 8, ..Limits::default() } };
/// assert_eq!(opts.limits.max_nodes, 8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BuildOptions {
    /// The resource limits.
    pub limits: Limits,
}

/// All block kinds that a description can use.
///
/// ```
/// use sdrk_pipeline::Registry;
/// let registry = Registry::standard();
/// assert_eq!(registry.descriptions().count(), 12);
/// assert!(registry.factory("fm_demod").is_some());
/// ```
pub struct Registry {
    factories: BTreeMap<&'static str, Box<dyn BlockFactory>>,
}

impl core::fmt::Debug for Registry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.factories.keys()).finish()
    }
}

/// Adds the factory of kind `K` if `sdrk-blocks` has a description for it.
fn register<K: Kind>(factories: &mut BTreeMap<&'static str, Box<dyn BlockFactory>>) {
    if let Some(desc) = all_descriptions()
        .iter()
        .copied()
        .find(|d| d.kind == K::KIND)
    {
        factories.insert(desc.kind, Factory::<K>::boxed(desc));
    }
}

impl Registry {
    /// A registry with every block of `sdrk-blocks`.
    pub fn standard() -> Self {
        let mut factories: BTreeMap<&'static str, Box<dyn BlockFactory>> = BTreeMap::new();
        register::<ConvertKind>(&mut factories);
        register::<ConvertOutKind>(&mut factories);
        register::<ScaleKind>(&mut factories);
        register::<DcBlockKind>(&mut factories);
        register::<FirKind>(&mut factories);
        register::<DdcKind>(&mut factories);
        register::<DucKind>(&mut factories);
        register::<FmDemodKind>(&mut factories);
        register::<FmModKind>(&mut factories);
        register::<ResamplerKind>(&mut factories);
        register::<SpectrumKind>(&mut factories);
        register::<SquelchKind>(&mut factories);
        Self { factories }
    }

    /// The factory of a kind.
    pub fn factory(&self, kind: &str) -> Option<&dyn BlockFactory> {
        self.factories.get(kind).map(|f| &**f)
    }

    /// The descriptions of all registered kinds, sorted by kind.
    pub fn descriptions(&self) -> impl Iterator<Item = &'static BlockDescription> + '_ {
        self.factories.values().map(|f| f.description())
    }

    /// The versioned JSON Schema (draft 2020-12) of the pipeline description.
    pub fn json_schema(&self) -> Value {
        crate::schema::json_schema(self)
    }

    /// Validates a description in two stages (structure, then graph and rates) and reports
    /// every problem found. Nothing is built; the figures of the report are estimates from
    /// the configurations.
    pub fn validate(&self, spec: &PipelineSpec, opts: &BuildOptions) -> Report {
        analysis::analyze(self, spec, opts).report
    }

    /// Parse and validate in one step. Syntax errors, unknown fields and a wrong schema version
    /// become diagnostics (`Syntax` with line and column, `UnknownField`, `SchemaVersion`)
    /// with their path.
    pub fn validate_json(&self, json: &str, opts: &BuildOptions) -> Report {
        match crate::structure::parse(json) {
            Ok(spec) => self.validate(&spec, opts),
            Err(report) => report,
        }
    }

    /// Validates and builds the pipeline. All memory is allocated here.
    ///
    /// # Errors
    /// The report, if validation found errors or a block cannot be built.
    pub fn build(&self, spec: &PipelineSpec, opts: &BuildOptions) -> Result<Pipeline, Report> {
        let analysis = analysis::analyze(self, spec, opts);
        match analysis.plan {
            Some(plan) if analysis.report.errors.is_empty() => {
                crate::build::build(self, spec, plan, analysis.report)
            }
            _ => Err(analysis.report),
        }
    }

    /// Parses, validates and builds a JSON description in one step.
    ///
    /// # Errors
    /// [`Error::Build`] with the report if anything is wrong.
    pub fn build_json(&self, json: &str, opts: &BuildOptions) -> Result<Pipeline> {
        match crate::structure::parse(json) {
            Ok(spec) => self.build(&spec, opts).map_err(Error::Build),
            Err(report) => Err(Error::Build(report)),
        }
    }
}
