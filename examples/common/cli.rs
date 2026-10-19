//! Argument parsing and flow handling shared by the two example binaries (included with
//! `#[path]`). Arguments are parsed with `std::env::args` only.
#![allow(dead_code)]

use std::collections::BTreeMap;

use sdrk_device::SampleFormat;
use sdrk_pipeline::{
    BuildOptions, FORMAT_NAMES, FormatName, FormatSpec, PipelineSpec, Registry, Report, Settings,
};
use serde_json::Value;

/// Why a run failed; decides the exit code.
#[derive(Debug)]
pub enum Failure {
    /// Wrong command line (exit code 2).
    Usage(String),
    /// The flow is invalid; carries the diagnostics JSON (exit code 1).
    Flow(String),
    /// Any other failure (exit code 1).
    Run(String),
}

impl From<sdrk_io::Error> for Failure {
    fn from(e: sdrk_io::Error) -> Self {
        Self::Run(e.to_string())
    }
}

impl From<sdrk_device::Error> for Failure {
    fn from(e: sdrk_device::Error) -> Self {
        Self::Run(e.to_string())
    }
}

impl From<sdrk_pipeline::Error> for Failure {
    fn from(e: sdrk_pipeline::Error) -> Self {
        match e {
            sdrk_pipeline::Error::Build(report) => Self::Flow(report.to_json()),
            other => Self::Run(other.to_string()),
        }
    }
}

impl From<std::io::Error> for Failure {
    fn from(e: std::io::Error) -> Self {
        Self::Run(format!("i/o error: {e}"))
    }
}

/// Parsed `--key value` options; a key may repeat.
#[derive(Debug, Default)]
pub struct Options {
    values: BTreeMap<String, Vec<String>>,
}

impl Options {
    /// Parses `--key value` and `--key=value` for the keys in `known`.
    pub fn parse(args: &[String], known: &[&str]) -> Result<Self, Failure> {
        let mut values: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            let Some(rest) = arg.strip_prefix("--") else {
                return Err(Failure::Usage(format!("unexpected argument `{arg}`")));
            };
            let (key, inline) = match rest.split_once('=') {
                Some((k, v)) => (k, Some(v.to_string())),
                None => (rest, None),
            };
            if !known.contains(&key) {
                return Err(Failure::Usage(format!("unknown option `--{key}`")));
            }
            let value = match inline {
                Some(v) => v,
                None => it
                    .next()
                    .cloned()
                    .ok_or_else(|| Failure::Usage(format!("option `--{key}` needs a value")))?,
            };
            values.entry(key.to_string()).or_default().push(value);
        }
        Ok(Self { values })
    }

    /// The last value of an option.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values
            .get(key)
            .and_then(|v| v.last())
            .map(String::as_str)
    }

    /// All values of an option, in order.
    pub fn all(&self, key: &str) -> &[String] {
        self.values.get(key).map_or(&[], Vec::as_slice)
    }

    /// The value of a required option.
    pub fn required(&self, key: &str) -> Result<&str, Failure> {
        self.get(key)
            .ok_or_else(|| Failure::Usage(format!("missing option `--{key}`")))
    }
}

/// Reads, validates and parses a flow; `default_text` is used without a path. YAML is chosen
/// by the file extension.
pub fn load_spec(
    registry: &Registry,
    path: Option<&str>,
    default_text: &str,
    opts: &BuildOptions,
) -> Result<PipelineSpec, Failure> {
    let (text, yaml) = match path {
        Some(p) => (
            std::fs::read_to_string(p)
                .map_err(|e| Failure::Run(format!("cannot read `{p}`: {e}")))?,
            std::path::Path::new(p)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml")),
        ),
        None => (default_text.to_string(), false),
    };
    let report = if yaml {
        registry.validate_yaml(&text, opts)
    } else {
        registry.validate_json(&text, opts)
    };
    if !report.is_ok() {
        return Err(Failure::Flow(report.to_json()));
    }
    if yaml {
        Ok(PipelineSpec::from_yaml(&text)?)
    } else {
        Ok(PipelineSpec::from_json(&text)?)
    }
}

/// Builds the pipeline; an invalid description becomes a `Failure::Flow` with the diagnostics.
pub fn build(
    registry: &Registry,
    spec: &PipelineSpec,
    opts: &BuildOptions,
) -> Result<sdrk_pipeline::Pipeline, Failure> {
    registry
        .build(spec, opts)
        .map_err(|report: Report| Failure::Flow(report.to_json()))
}

/// Applies `node.param=value` (the value is JSON, or a plain string if it is not JSON) or
/// `settings.field=value` to the description.
pub fn apply_set(spec: &mut PipelineSpec, assignment: &str) -> Result<(), Failure> {
    let bad = |m: &str| Failure::Usage(format!("--set `{assignment}`: {m}"));
    let (key, raw) = assignment
        .split_once('=')
        .ok_or_else(|| bad("expected node.param=value"))?;
    let (node, param) = key
        .split_once('.')
        .ok_or_else(|| bad("expected node.param=value"))?;
    let value: Value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
    if node == "settings" {
        let mut v = serde_json::to_value(spec.settings).map_err(|e| Failure::Run(e.to_string()))?;
        if let Some(map) = v.as_object_mut() {
            map.insert(param.to_string(), value);
        }
        spec.settings = serde_json::from_value::<Settings>(v)
            .map_err(|e| bad(&format!("invalid setting: {e}")))?;
        return Ok(());
    }
    let n = spec
        .nodes
        .get_mut(node)
        .ok_or_else(|| bad(&format!("the flow has no node `{node}`")))?;
    n.params.insert(param.to_string(), value);
    Ok(())
}

/// The model entry for a format constant, if it has a name.
pub fn format_spec_of(format: &SampleFormat) -> Option<FormatSpec> {
    FORMAT_NAMES
        .iter()
        .filter_map(|n| FormatName::from_name(n))
        .find(|n| n.format() == *format)
        .map(FormatSpec::Name)
}

/// Prints a failure and returns the exit code.
pub fn report_failure(failure: &Failure, usage: &str) -> std::process::ExitCode {
    match failure {
        Failure::Usage(m) => {
            eprintln!("error: {m}\n\n{usage}");
            std::process::ExitCode::from(2)
        }
        Failure::Flow(json) => {
            eprintln!("{json}");
            std::process::ExitCode::from(1)
        }
        Failure::Run(m) => {
            eprintln!("error: {m}");
            std::process::ExitCode::from(1)
        }
    }
}

/// Smallest number of samples that fills whole packing groups.
pub fn granule_samples(format: &SampleFormat) -> usize {
    [1usize, 2, 4, 8]
        .into_iter()
        .find(|&g| format.bytes_for_samples(g).is_ok())
        .unwrap_or(1)
}

/// The name of the only input of the flow, or the one given with `--input-name`.
pub fn input_name(spec: &PipelineSpec, given: Option<&str>) -> Result<String, Failure> {
    if let Some(n) = given {
        return Ok(n.to_string());
    }
    let mut names = spec.inputs.keys();
    match (names.next(), names.next()) {
        (Some(n), None) => Ok(n.clone()),
        _ => Err(Failure::Usage(
            "the flow does not have exactly one input; name it with --input-name".into(),
        )),
    }
}
