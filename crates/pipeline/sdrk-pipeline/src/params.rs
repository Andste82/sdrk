//! Conversion between JSON values and block parameter values.

use sdrk_blocks::{ParamDesc, ParamKind, ParamValue};
use serde_json::{Number, Value};

use crate::model::{FormatName, FormatSpec};
use crate::structure::join;
use crate::validate::{DiagCode, Diagnostic};
use sdrk_blocks::BlockDescription;

/// Why a JSON value is not a valid parameter value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ParamIssue {
    /// The JSON type does not fit; the text names the expected type.
    WrongType(&'static str),
    /// Right type, but not a possible value (non-integral number, unknown option or name).
    Invalid(String),
    /// A number outside the absolute limits of the parameter.
    Range(sdrk_blocks::Error),
}

impl ParamIssue {
    /// Human-readable text for diagnostics and errors.
    pub(crate) fn message(&self, value: &Value) -> String {
        match self {
            Self::WrongType(expected) => {
                format!("expected {expected}, got {}", json_type(value))
            }
            Self::Invalid(m) => m.clone(),
            Self::Range(e) => e.to_string(),
        }
    }
}

/// Name of the JSON type of a value.
pub(crate) fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Converts a JSON value to the value of the parameter `desc` and checks its absolute limits.
///
/// The success path does not allocate.
pub(crate) fn to_param(desc: &ParamDesc, value: &Value) -> Result<ParamValue, ParamIssue> {
    let pv = match (&desc.kind, value) {
        (ParamKind::F64 { .. }, Value::Number(n)) => {
            ParamValue::F64(n.as_f64().ok_or(ParamIssue::WrongType("a number"))?)
        }
        (ParamKind::F64 { .. }, _) => return Err(ParamIssue::WrongType("a number")),
        (ParamKind::I64 { .. }, Value::Number(n)) => ParamValue::I64(integer(n)?),
        (ParamKind::I64 { .. }, _) => return Err(ParamIssue::WrongType("an integer")),
        (ParamKind::Bool { .. }, Value::Bool(b)) => ParamValue::Bool(*b),
        (ParamKind::Bool { .. }, _) => return Err(ParamIssue::WrongType("a boolean")),
        (ParamKind::Choice { options, .. }, Value::String(s)) => {
            match options.iter().position(|o| *o == s.as_str()) {
                Some(i) => ParamValue::Choice(i as u16),
                None => {
                    return Err(ParamIssue::Invalid(format!(
                        "`{s}` is not one of: {}",
                        options.join(", ")
                    )));
                }
            }
        }
        (ParamKind::Choice { .. }, _) => return Err(ParamIssue::WrongType("a string")),
        (ParamKind::Format, Value::String(s)) => match FormatName::from_name(s) {
            Some(n) => ParamValue::Format(n.format()),
            None => {
                return Err(ParamIssue::Invalid(format!(
                    "`{s}` is not a format name; use one of {}",
                    crate::model::FORMAT_NAMES.join(", ")
                )));
            }
        },
        (ParamKind::Format, Value::Object(_)) => {
            let spec: FormatSpec = serde_json::from_value(value.clone())
                .map_err(|e| ParamIssue::Invalid(e.to_string()))?;
            ParamValue::Format(spec.to_format().map_err(ParamIssue::Invalid)?)
        }
        (ParamKind::Format, _) => return Err(ParamIssue::WrongType("a format name or object")),
        _ => return Err(ParamIssue::WrongType("a supported value")),
    };
    desc.check(&pv).map_err(ParamIssue::Range)?;
    Ok(pv)
}

/// An integral JSON number as `i64` (`10` and `10.0` are accepted, `10.5` is not).
fn integer(n: &Number) -> Result<i64, ParamIssue> {
    if let Some(i) = n.as_i64() {
        return Ok(i);
    }
    if n.is_u64() {
        return Err(ParamIssue::Invalid(format!(
            "{n} does not fit into 64 bits"
        )));
    }
    let f = n.as_f64().ok_or(ParamIssue::WrongType("an integer"))?;
    if f.fract() == 0.0 && f.abs() < 9.0e15 {
        Ok(f as i64)
    } else {
        Err(ParamIssue::Invalid(format!("{n} is not an integer")))
    }
}

/// The JSON form of a parameter value (for the configuration readout).
pub(crate) fn to_json(desc: &ParamDesc, value: ParamValue) -> Value {
    match value {
        ParamValue::F64(v) => Number::from_f64(v).map_or(Value::Null, Value::Number),
        ParamValue::I64(v) => Value::Number(v.into()),
        ParamValue::Bool(v) => Value::Bool(v),
        ParamValue::Choice(i) => match &desc.kind {
            ParamKind::Choice { options, .. } => options
                .get(usize::from(i))
                .map_or(Value::Null, |o| Value::String((*o).to_string())),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

/// Checks the parameter names, types and absolute ranges of a node against its description and
/// appends one diagnostic per problem; `prefix` is the path of the `params` object.
pub(crate) fn check_params<'a>(
    desc: &BlockDescription,
    params: impl Iterator<Item = (&'a String, &'a Value)>,
    prefix: &str,
    out: &mut Vec<Diagnostic>,
) {
    let params: Vec<(&String, &Value)> = params.collect();
    for required in desc.required {
        if !params.iter().any(|(k, _)| k.as_str() == *required) {
            out.push(Diagnostic {
                path: join(prefix, required),
                code: DiagCode::MissingParam,
                message: format!("the parameter `{required}` is required by `{}`", desc.kind),
            });
        }
    }
    for (name, value) in params {
        let path = join(prefix, name);
        let Some(p) = desc.param(name) else {
            out.push(Diagnostic {
                path,
                code: DiagCode::UnknownParam,
                message: format!(
                    "`{}` has no parameter `{name}`; it has: {}",
                    desc.kind,
                    desc.params
                        .iter()
                        .map(|p| p.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });
            continue;
        };
        if let Err(issue) = to_param(p, value) {
            let code = match &issue {
                ParamIssue::WrongType(_) => DiagCode::WrongType,
                ParamIssue::Invalid(_) => DiagCode::InvalidValue,
                ParamIssue::Range(_) => DiagCode::OutOfRange,
            };
            out.push(Diagnostic {
                path,
                code,
                message: issue.message(value),
            });
        }
    }
}
