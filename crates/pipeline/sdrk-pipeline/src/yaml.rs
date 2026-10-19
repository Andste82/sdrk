//! YAML input (feature `yaml`): the same model as JSON, comments allowed.

use serde_json::Value;

use crate::error::{Error, Result};
use crate::model::PipelineSpec;
use crate::registry::{BuildOptions, Registry};
use crate::runner::Pipeline;
use crate::validate::{DiagCode, Report};

/// The text of a YAML error with its position.
fn describe(error: &serde_saphyr::Error) -> String {
    match error.location() {
        Some(at) => format!("line {}, column {}: {error}", at.line(), at.column()),
        None => error.to_string(),
    }
}

impl PipelineSpec {
    /// Parses a description from YAML.
    ///
    /// The model is the same as for JSON, including `deny_unknown_fields`; comments are
    /// allowed. There is no YAML writer: JSON is the canonical format.
    ///
    /// ```
    /// use sdrk_pipeline::PipelineSpec;
    /// let spec = PipelineSpec::from_yaml(
    ///     "# a gain stage\nsdrk: 1\nname: tone\ninputs:\n  iq: { type: c32, rate_hz: 48000 }\n",
    /// )?;
    /// assert_eq!(spec.name, "tone");
    /// let err = PipelineSpec::from_yaml("sdrk: 1\nname: [oops\n").unwrap_err();
    /// assert!(err.to_string().contains("line 2"));
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// [`Error::Yaml`] with the line and column of the problem.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_saphyr::from_str(yaml).map_err(|e| Error::Yaml(describe(&e)))
    }
}

impl Registry {
    /// Parse and validate YAML text. Syntax errors become a `syntax` diagnostic with line and
    /// column; the other diagnostics have the same codes and paths as for JSON.
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, Registry};
    /// let yaml = include_str!("../../../../examples/flows/fm_receiver.yaml");
    /// assert!(Registry::standard().validate_yaml(yaml, &BuildOptions::default()).is_ok());
    /// assert!(!Registry::standard().validate_yaml("sdrk: 2", &BuildOptions::default()).is_ok());
    /// ```
    pub fn validate_yaml(&self, yaml: &str, opts: &BuildOptions) -> Report {
        match parse_yaml(yaml) {
            Ok(spec) => self.validate(&spec, opts),
            Err(report) => report,
        }
    }

    /// Parses, validates and builds a YAML description in one step.
    ///
    /// ```
    /// use sdrk_pipeline::{BuildOptions, Registry};
    /// let yaml = include_str!("../../../../examples/flows/fm_receiver.yaml");
    /// let p = Registry::standard().build_yaml(yaml, &BuildOptions::default())?;
    /// assert_eq!(p.report().nodes.len(), 4);
    /// # Ok::<(), sdrk_pipeline::Error>(())
    /// ```
    ///
    /// # Errors
    /// [`Error::Build`] with the report if anything is wrong.
    pub fn build_yaml(&self, yaml: &str, opts: &BuildOptions) -> Result<Pipeline> {
        match parse_yaml(yaml) {
            Ok(spec) => self.build(&spec, opts).map_err(Error::Build),
            Err(report) => Err(Error::Build(report)),
        }
    }
}

fn parse_yaml(yaml: &str) -> Result<PipelineSpec, Report> {
    let value: Value = serde_saphyr::from_str(yaml).map_err(|e| {
        let mut report = Report::default();
        report.error(
            "",
            DiagCode::Syntax,
            format!("invalid YAML at {}", describe(&e)),
        );
        report
    })?;
    crate::structure::parse_value(value)
}
