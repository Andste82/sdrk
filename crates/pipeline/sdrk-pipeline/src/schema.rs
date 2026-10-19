//! The JSON Schema (draft 2020-12) of the pipeline description.
//!
//! `schemars` derives the static part (the model). The `nodes` part is generated from the
//! block descriptions of the registry, so a new block or parameter shows up in the schema
//! without any further code.

use std::fmt::Write as _;

use schemars::generate::SchemaSettings;
use sdrk_blocks::{BlockDescription, Mutability, ParamDesc, ParamKind};
use serde_json::{Map, Value, json};

use crate::model::{FORMAT_NAMES, MAX_DESCRIPTION_CHARS, NAME_PATTERN, PipelineSpec};
use crate::registry::Registry;

/// `$id` of the schema of format version 1.
pub const SCHEMA_ID: &str = "https://github.com/Andste82/sdrk/schema/sdrk-pipeline-v1.json";

/// Path of the committed schema file, relative to the workspace root.
pub const SCHEMA_FILE: &str = "schema/sdrk-pipeline-v1.json";

#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn param_schema(p: &ParamDesc) -> Value {
    let unit = p.unit.symbol();
    let when = match p.mutability {
        Mutability::Build => "Build-time",
        _ => "Runtime",
    };
    let mut text = p.description.trim_end().to_string();
    if !unit.is_empty() {
        let _ = write!(text, " Unit: {unit}.");
    }
    let _ = write!(text, " Typical: {}. {when}.", p.typical);
    let mut m = Map::new();
    match p.kind {
        ParamKind::F64 { min, max, default } => {
            m.insert("type".into(), json!("number"));
            m.insert("minimum".into(), json!(min));
            m.insert("maximum".into(), json!(max));
            if let Some(d) = default {
                m.insert("default".into(), json!(d));
            }
        }
        ParamKind::I64 { min, max, default } => {
            m.insert("type".into(), json!("integer"));
            m.insert("minimum".into(), json!(min));
            m.insert("maximum".into(), json!(max));
            if let Some(d) = default {
                m.insert("default".into(), json!(d));
            }
        }
        ParamKind::Bool { default } => {
            m.insert("type".into(), json!("boolean"));
            m.insert("default".into(), json!(default));
        }
        ParamKind::Choice { options, default } => {
            m.insert("type".into(), json!("string"));
            m.insert("enum".into(), json!(options));
            if let Some(d) = options.get(usize::from(default)) {
                m.insert("default".into(), json!(d));
            }
        }
        ParamKind::Format => {
            m.insert(
                "anyOf".into(),
                json!([
                    { "type": "string", "enum": FORMAT_NAMES },
                    { "type": "object", "$ref": "#/$defs/CustomFormat" }
                ]),
            );
        }
        _ => {}
    }
    m.insert("description".into(), json!(text));
    Value::Object(m)
}

#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
fn node_schema(desc: &BlockDescription) -> Value {
    let mut params = Map::new();
    for p in desc.params {
        params.insert(p.name.to_string(), param_schema(p));
    }
    let mut params_schema = Map::new();
    params_schema.insert("type".into(), json!("object"));
    params_schema.insert("additionalProperties".into(), json!(false));
    params_schema.insert("properties".into(), Value::Object(params));
    if !desc.required.is_empty() {
        params_schema.insert("required".into(), json!(desc.required));
    }
    let mut required = vec!["kind"];
    if !desc.required.is_empty() {
        required.push("params");
    }
    json!({
        "type": "object",
        "description": desc.summary,
        "additionalProperties": false,
        "properties": {
            "kind": { "const": desc.kind },
            "params": Value::Object(params_schema),
            "group": {
                "type": "string",
                "pattern": NAME_PATTERN,
                "description": "The group of the node (a key of `groups`); nodes without one are in `main`."
            },
            "backend": { "$ref": "#/$defs/BackendSpec" },
            "description": { "type": "string", "maxLength": MAX_DESCRIPTION_CHARS }
        },
        "required": required
    })
}

/// Rebuilds every object with sorted keys.
pub(crate) fn sorted(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(entries.into_iter().map(|(k, v)| (k, sorted(v))).collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sorted).collect()),
        other => other,
    }
}

#[allow(
    clippy::large_stack_frames,
    reason = "clippy counts the temporary of every `?` separately; the real frame is small"
)]
pub(crate) fn json_schema(registry: &Registry) -> Value {
    let generator = SchemaSettings::draft2020_12().into_generator();
    let root = generator.into_root_schema_for::<PipelineSpec>();
    let mut schema = serde_json::to_value(&root).unwrap_or(Value::Null);
    if let Some(obj) = schema.as_object_mut() {
        obj.insert("$id".into(), json!(SCHEMA_ID));
        obj.insert(
            "title".into(),
            json!("sdrk pipeline description, format version 1"),
        );
        if let Some(defs) = obj.get_mut("$defs").and_then(Value::as_object_mut) {
            defs.remove("NodeSpec");
        }
        if let Some(props) = obj.get_mut("properties").and_then(Value::as_object_mut) {
            props.insert(
                "sdrk".into(),
                json!({ "const": 1, "description": "Format version; must be 1." }),
            );
            let kinds: Vec<Value> = registry.descriptions().map(node_schema).collect();
            props.insert(
                "nodes".into(),
                json!({
                    "type": "object",
                    "description": "The processing nodes by name.",
                    "default": {},
                    "propertyNames": { "pattern": NAME_PATTERN },
                    "additionalProperties": { "oneOf": kinds }
                }),
            );
            for key in ["inputs", "outputs", "groups"] {
                if let Some(section) = props.get_mut(key).and_then(Value::as_object_mut) {
                    section.insert("propertyNames".into(), json!({ "pattern": NAME_PATTERN }));
                }
            }
        }
    }
    sorted(schema)
}

impl Registry {
    /// The schema as committed in `schema/sdrk-pipeline-v1.json`: pretty-printed, keys
    /// sorted, LF line endings, one trailing newline.
    pub fn json_schema_text(&self) -> String {
        let mut text = serde_json::to_string_pretty(&self.json_schema()).unwrap_or_default();
        text.push('\n');
        text
    }
}
