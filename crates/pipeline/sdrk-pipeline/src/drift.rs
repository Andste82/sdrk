//! Clock-drift helper: drives the `ppm` parameter of a tunable `resample` node from the fill
//! level of a stream output.

use serde_json::Value;

use sdrk_blocks::{DriftConfig, DriftController};

use crate::error::{Error, Result};
use crate::runner::Pipeline;
use crate::validate::{DiagCode, Report};

/// Keeps the queue of a stream output at a target fill level by steering the `ppm` parameter of
/// a tunable `resample` node with a [`DriftController`].
///
/// The application calls [`update`](Self::update) before every `push` to the input and before
/// every read of the output (see `update` for why). See `docs/blocks.md` (section "Clock drift")
/// for the control law.
///
/// ```
/// use sdrk_blocks::DriftConfig;
/// use sdrk_pipeline::{BuildOptions, InputData, PipelineSpec, Registry};
/// use sdrk_pipeline::drift::PipelineDrift;
///
/// let mut spec = PipelineSpec::from_json(include_str!("../../../../examples/flows/resample_audio.json"))?;
/// if let Some(node) = spec.nodes.get_mut("to_cd") {
///     node.params.insert("tunable".into(), true.into());
/// }
/// let mut pipeline = Registry::standard().build(&spec, &BuildOptions::default()).map_err(|r| r.to_json())?;
/// let config = DriftConfig::new(44_100.0, 4_410.0);
/// let mut drift = PipelineDrift::new(&pipeline, "audio_out", "to_cd", config)?;
/// pipeline.push("audio_in", InputData::F32(&[0.0; 480]))?;
/// // The first call only initialises the controller and returns the initial ppm.
/// assert_eq!(drift.update(&mut pipeline, 0.0, 0)?, 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct PipelineDrift {
    controller: DriftController,
    output: String,
    node: String,
}

/// The error for a node or output that cannot be used: one `invalid_value` diagnostic.
#[inline(never)]
fn reject(path: String, message: String) -> Error {
    let mut report = Report::default();
    report.error(path, DiagCode::InvalidValue, message);
    Error::Build(report)
}

impl PipelineDrift {
    /// Creates the helper. `output` is the stream output whose queue is the buffer; `node` is a
    /// `resample` node with `tunable = true` on the path to that output.
    ///
    /// ```
    /// use sdrk_blocks::DriftConfig;
    /// use sdrk_pipeline::{BuildOptions, Registry};
    /// use sdrk_pipeline::drift::PipelineDrift;
    ///
    /// // The example flow is not tunable, so the node is refused.
    /// let pipeline = Registry::standard().build_json(
    ///     include_str!("../../../../examples/flows/resample_audio.json"),
    ///     &BuildOptions::default(),
    /// )?;
    /// let config = DriftConfig::new(44_100.0, 4_410.0);
    /// assert!(PipelineDrift::new(&pipeline, "audio_out", "to_cd", config).is_err());
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownOutput`, `UnknownNode`, `Build` (a report with one `invalid_value` error: the node
    /// is not a `resample` node, is not tunable, is not on the path to the output, or the output
    /// is not a stream output) and `Block` (the configuration is invalid).
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `Err(...)` separately; the real frame is small"
    )]
    pub fn new(pipeline: &Pipeline, output: &str, node: &str, config: DriftConfig) -> Result<Self> {
        let oi = pipeline.output_pos(output)?;
        let ni = *pipeline
            .node_index
            .get(node)
            .ok_or_else(|| Error::UnknownNode(node.to_string()))?;
        let out = pipeline
            .outputs
            .get(oi)
            .ok_or(Error::Internal("output index"))?;
        if !matches!(out.kind, crate::runner::OutKind::Stream { .. }) {
            return Err(reject(
                format!("outputs.{output}"),
                format!("output `{output}` is not a stream output"),
            ));
        }
        let path = format!("nodes.{node}");
        if !out.path.contains(&ni) {
            return Err(reject(
                path,
                format!("node `{node}` is not on the path to output `{output}`"),
            ));
        }
        let spec = pipeline.spec.nodes.get(node);
        if spec.is_none_or(|s| s.kind != "resample") {
            return Err(reject(
                path,
                format!("node `{node}` is not a `resample` node"),
            ));
        }
        let tunable = spec
            .and_then(|s| s.params.get("tunable"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !tunable {
            return Err(reject(
                format!("nodes.{node}.params.tunable"),
                format!("node `{node}` is not tunable; set `tunable` to true"),
            ));
        }
        let controller = DriftController::new(config).map_err(|source| Error::Block {
            node: node.to_string(),
            source,
        })?;
        Ok(Self {
            controller,
            output: output.to_string(),
            node: node.to_string(),
        })
    }

    /// Reads `available(output)` plus `extra_fill` (samples the application holds outside the
    /// pipeline, for example in the sound-card buffer), runs the controller and sets `ppm` on
    /// the node. Returns the value set. Does not allocate.
    ///
    /// Calling contract: call `update` immediately before every `push` to the input that feeds
    /// `output` and immediately before every read of `output` (in a sound-card callback: update
    /// first, then take the block). The queue changes only at those two events, so each call
    /// sees the level that held during the interval since the previous call, and the
    /// controller's time-weighted smoothing (`smoothing_s`) averages the queue over time. That
    /// average moves continuously as the input and output clocks slip against each other.
    /// Calling `update` only after each read instead samples the queue at one fixed phase of
    /// the output clock: when input and output blocks have about the same period (480 in at
    /// 48 kHz, 441 out at 44.1 kHz) the reading jumps by a whole output block each time the
    /// two clocks slip by one period, and the controller answers every jump with a spike of
    /// several hundred ppm (D314).
    ///
    /// # Errors
    /// The errors of [`Pipeline::available`] and [`Pipeline::set_param`].
    pub fn update(
        &mut self,
        pipeline: &mut Pipeline,
        now_s: f64,
        extra_fill: usize,
    ) -> Result<f64> {
        let fill = pipeline.available(&self.output)?.saturating_add(extra_fill);
        let ppm = self.controller.update(now_s, fill as f64);
        pipeline.set_param(&self.node, "ppm", Value::from(ppm))?;
        Ok(ppm)
    }

    /// The controller, for reading its output or changing the initial ppm.
    pub fn controller(&self) -> &DriftController {
        &self.controller
    }

    /// Mutable access to the controller (`set_initial_ppm`, `reset`).
    pub fn controller_mut(&mut self) -> &mut DriftController {
        &mut self.controller
    }
}
