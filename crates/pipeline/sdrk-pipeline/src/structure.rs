//! Stage 1a of `validate_json`: the structure of the JSON document.
//!
//! Works on the parsed `serde_json::Value`, so that every problem is reported with the path of
//! the offending key and all problems are found in one pass. When the structure is sound the
//! value is converted into the typed model.

use serde_json::{Map, Value};

use crate::model::{
    FORMAT_NAMES, MAX_CHUNK_LIMIT, MAX_DESCRIPTION_CHARS, MAX_EVENT_QUEUE, MAX_PENDING_PARAMS,
    PipelineSpec, parse_rate_text,
};
use crate::params::json_type;
use crate::validate::{DiagCode, Report};

/// Joins a path and a key with a dot.
pub(crate) fn join(parent: &str, key: &str) -> String {
    if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    }
}

struct Checker {
    report: Report,
}

impl Checker {
    fn object<'v>(
        &mut self,
        value: &'v Value,
        path: &str,
        allowed: &[&str],
    ) -> Option<&'v Map<String, Value>> {
        let Some(map) = value.as_object() else {
            self.report.error(
                path,
                DiagCode::WrongType,
                format!("expected an object, got {}", json_type(value)),
            );
            return None;
        };
        for key in map.keys() {
            if !allowed.contains(&key.as_str()) {
                self.report.error(
                    join(path, key),
                    DiagCode::UnknownField,
                    format!("unknown field `{key}`; allowed: {}", allowed.join(", ")),
                );
            }
        }
        Some(map)
    }

    fn string<'v>(&mut self, value: &'v Value, path: &str) -> Option<&'v str> {
        let s = value.as_str();
        if s.is_none() {
            self.report.error(
                path,
                DiagCode::WrongType,
                format!("expected a string, got {}", json_type(value)),
            );
        }
        s
    }

    fn description(&mut self, value: &Value, path: &str) {
        if let Some(s) = self.string(value, path) {
            if s.chars().count() > MAX_DESCRIPTION_CHARS {
                self.report.error(
                    path,
                    DiagCode::OutOfRange,
                    format!("a description has at most {MAX_DESCRIPTION_CHARS} characters"),
                );
            }
        }
    }

    /// A non-negative integer; `None` after reporting a problem.
    fn count(&mut self, value: &Value, path: &str) -> Option<u64> {
        let Value::Number(n) = value else {
            self.report.error(
                path,
                DiagCode::WrongType,
                format!("expected an integer, got {}", json_type(value)),
            );
            return None;
        };
        if let Some(u) = n.as_u64() {
            return Some(u);
        }
        if n.as_i64().is_some() {
            self.report.error(
                path,
                DiagCode::OutOfRange,
                format!("{n} must not be negative"),
            );
        } else {
            self.report.error(
                path,
                DiagCode::InvalidValue,
                format!("{n} is not an integer"),
            );
        }
        None
    }

    fn one_of(&mut self, value: &Value, path: &str, options: &[&str]) -> bool {
        match self.string(value, path) {
            Some(s) if options.contains(&s) => true,
            Some(s) => {
                self.report.error(
                    path,
                    DiagCode::InvalidValue,
                    format!("`{s}` is not one of: {}", options.join(", ")),
                );
                false
            }
            None => false,
        }
    }

    fn required<'v>(
        &mut self,
        map: &'v Map<String, Value>,
        key: &str,
        path: &str,
    ) -> Option<&'v Value> {
        let found = map.get(key);
        if found.is_none() {
            self.report.error(
                join(path, key),
                DiagCode::MissingParam,
                format!("the field `{key}` is required"),
            );
        }
        found
    }

    fn rate(&mut self, value: &Value, path: &str) {
        match value {
            Value::Number(_) => {
                if let Some(u) = self.count(value, path) {
                    if u == 0 {
                        self.report
                            .error(path, DiagCode::OutOfRange, "the rate must be positive");
                    }
                }
            }
            Value::String(s) => {
                if let Err(m) = parse_rate_text(s) {
                    self.report.error(path, DiagCode::InvalidValue, m);
                }
            }
            _ => self.report.error(
                path,
                DiagCode::WrongType,
                format!(
                    "expected a positive integer or \"num/den\", got {}",
                    json_type(value)
                ),
            ),
        }
    }

    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn format(&mut self, value: &Value, path: &str) {
        match value {
            Value::String(s) => {
                if !FORMAT_NAMES.contains(&s.as_str()) {
                    self.report.error(
                        path,
                        DiagCode::InvalidValue,
                        format!(
                            "`{s}` is not a format name; use one of {}",
                            FORMAT_NAMES.join(", ")
                        ),
                    );
                }
            }
            Value::Object(_) => {
                let allowed = [
                    "container",
                    "bits",
                    "align",
                    "layout",
                    "endian",
                    "full_scale",
                    "swap_iq",
                ];
                let Some(map) = self.object(value, path, &allowed) else {
                    return;
                };
                let enums: [(&str, &[&str]); 4] = [
                    (
                        "container",
                        &[
                            "u8", "i8", "i16", "i32", "f32", "packed12", "packed10", "bits1",
                        ],
                    ),
                    ("align", &["lsb", "msb"]),
                    ("layout", &["interleaved", "planar", "real"]),
                    ("endian", &["le", "be"]),
                ];
                for (key, options) in enums {
                    if let Some(v) = self.required(map, key, path) {
                        self.one_of(v, &join(path, key), options);
                    }
                }
                if let Some(v) = self.required(map, "bits", path) {
                    if let Some(bits) = self.count(v, &join(path, "bits")) {
                        if bits > 255 {
                            self.report.error(
                                join(path, "bits"),
                                DiagCode::OutOfRange,
                                "bits must be at most 255",
                            );
                        }
                    }
                }
                if let Some(v) = map.get("full_scale") {
                    self.full_scale(v, &join(path, "full_scale"));
                }
                if let Some(v) = map.get("swap_iq") {
                    if !v.is_boolean() {
                        self.report.error(
                            join(path, "swap_iq"),
                            DiagCode::WrongType,
                            format!("expected a boolean, got {}", json_type(v)),
                        );
                    }
                }
            }
            _ => self.report.error(
                path,
                DiagCode::WrongType,
                format!(
                    "expected a format name or an object, got {}",
                    json_type(value)
                ),
            ),
        }
    }

    fn full_scale(&mut self, value: &Value, path: &str) {
        match value {
            Value::String(s) if s == "power_of_two" => {}
            Value::Object(map) if map.len() == 1 && map.contains_key("explicit") => {
                if let Some(v) = map.get("explicit") {
                    if let Some(u) = self.count(v, &join(path, "explicit")) {
                        if u == 0 || u > u64::from(u32::MAX) {
                            self.report.error(
                                join(path, "explicit"),
                                DiagCode::OutOfRange,
                                "the explicit full scale is 1 to 4294967295",
                            );
                        }
                    }
                }
            }
            _ => self.report.error(
                path,
                DiagCode::InvalidValue,
                "expected \"power_of_two\" or {\"explicit\": <integer>}",
            ),
        }
    }

    fn settings(&mut self, value: &Value) {
        let Some(map) = self.object(
            value,
            "settings",
            &["precision", "max_chunk", "event_queue", "pending_params"],
        ) else {
            return;
        };
        for (key, max) in [
            ("event_queue", MAX_EVENT_QUEUE),
            ("pending_params", MAX_PENDING_PARAMS),
        ] {
            if let Some(v) = map.get(key) {
                if let Some(u) = self.count(v, &join("settings", key)) {
                    if u == 0 || u > max as u64 {
                        self.report.error(
                            join("settings", key),
                            DiagCode::OutOfRange,
                            format!("{key} is 1 to {max}, got {u}"),
                        );
                    }
                }
            }
        }
        if let Some(v) = map.get("precision") {
            self.one_of(v, "settings.precision", &["f32", "q15"]);
        }
        if let Some(v) = map.get("max_chunk") {
            if let Some(u) = self.count(v, "settings.max_chunk") {
                if u == 0 || u > MAX_CHUNK_LIMIT as u64 {
                    self.report.error(
                        "settings.max_chunk",
                        DiagCode::OutOfRange,
                        format!("max_chunk is 1 to {MAX_CHUNK_LIMIT}, got {u}"),
                    );
                }
            }
        }
    }

    fn inputs(&mut self, value: &Value) {
        let Value::Object(entries) = value else {
            self.report.error(
                "inputs",
                DiagCode::WrongType,
                format!("expected an object, got {}", json_type(value)),
            );
            return;
        };
        for (name, entry) in entries {
            let path = join("inputs", name);
            let Some(map) =
                self.object(entry, &path, &["format", "type", "rate_hz", "description"])
            else {
                continue;
            };
            if let Some(v) = self.required(map, "rate_hz", &path) {
                self.rate(v, &join(&path, "rate_hz"));
            }
            if let Some(v) = map.get("format") {
                self.format(v, &join(&path, "format"));
            }
            if let Some(v) = map.get("type") {
                self.one_of(v, &join(&path, "type"), &["c32", "cq15", "f32", "q15"]);
            }
            if let Some(v) = map.get("description") {
                self.description(v, &join(&path, "description"));
            }
        }
    }

    fn nodes(&mut self, value: &Value) {
        let Value::Object(entries) = value else {
            self.report.error(
                "nodes",
                DiagCode::WrongType,
                format!("expected an object, got {}", json_type(value)),
            );
            return;
        };
        for (name, entry) in entries {
            let path = join("nodes", name);
            let Some(map) = self.object(
                entry,
                &path,
                &["kind", "params", "group", "backend", "description"],
            ) else {
                continue;
            };
            if let Some(v) = map.get("group") {
                self.string(v, &join(&path, "group"));
            }
            if let Some(v) = map.get("backend") {
                self.backend(v, &join(&path, "backend"));
            }
            if let Some(v) = self.required(map, "kind", &path) {
                self.string(v, &join(&path, "kind"));
            }
            if let Some(v) = map.get("params") {
                if !v.is_object() {
                    self.report.error(
                        join(&path, "params"),
                        DiagCode::WrongType,
                        format!("expected an object, got {}", json_type(v)),
                    );
                }
            }
            if let Some(v) = map.get("description") {
                self.description(v, &join(&path, "description"));
            }
        }
    }

    fn backend(&mut self, value: &Value, path: &str) {
        let Some(map) = self.object(value, path, &["prefer", "require"]) else {
            return;
        };
        for key in ["prefer", "require"] {
            if let Some(v) = map.get(key) {
                self.string(v, &join(path, key));
            }
        }
        if map.len() != 1 {
            self.report.error(
                path,
                DiagCode::InvalidValue,
                "give exactly one of `prefer` and `require`",
            );
        }
    }

    fn groups(&mut self, value: &Value) {
        let Value::Object(entries) = value else {
            self.report.error(
                "groups",
                DiagCode::WrongType,
                format!("expected an object, got {}", json_type(value)),
            );
            return;
        };
        for (name, entry) in entries {
            let path = join("groups", name);
            let Some(map) = self.object(entry, &path, &["description"]) else {
                continue;
            };
            if let Some(v) = map.get("description") {
                self.description(v, &join(&path, "description"));
            }
        }
    }

    fn edges(&mut self, value: &Value) {
        let Value::Array(items) = value else {
            self.report.error(
                "edges",
                DiagCode::WrongType,
                format!("expected an array, got {}", json_type(value)),
            );
            return;
        };
        for (i, item) in items.iter().enumerate() {
            let path = format!("edges[{i}]");
            let Some(map) = self.object(item, &path, &["from", "to"]) else {
                continue;
            };
            for key in ["from", "to"] {
                if let Some(v) = self.required(map, key, &path) {
                    self.string(v, &join(&path, key));
                }
            }
        }
    }

    fn outputs(&mut self, value: &Value) {
        let Value::Object(entries) = value else {
            self.report.error(
                "outputs",
                DiagCode::WrongType,
                format!("expected an object, got {}", json_type(value)),
            );
            return;
        };
        for (name, entry) in entries {
            let path = join("outputs", name);
            let Some(map) = self.object(
                entry,
                &path,
                &[
                    "type",
                    "format",
                    "capacity",
                    "queue",
                    "drop",
                    "underrun",
                    "description",
                ],
            ) else {
                continue;
            };
            if let Some(v) = map.get("underrun") {
                self.one_of(v, &join(&path, "underrun"), &["zeros", "short"]);
            }
            if let Some(v) = self.required(map, "type", &path) {
                self.one_of(v, &join(&path, "type"), &["stream", "frames"]);
            }
            if let Some(v) = map.get("format") {
                self.format(v, &join(&path, "format"));
            }
            for key in ["capacity", "queue"] {
                if let Some(v) = map.get(key) {
                    self.count(v, &join(&path, key));
                }
            }
            if let Some(v) = map.get("drop") {
                self.one_of(
                    v,
                    &join(&path, "drop"),
                    &["error", "drop_oldest", "drop_newest"],
                );
            }
            if let Some(v) = map.get("description") {
                self.description(v, &join(&path, "description"));
            }
        }
    }

    fn document(&mut self, value: &Value) {
        let allowed = [
            "sdrk",
            "name",
            "description",
            "settings",
            "inputs",
            "groups",
            "nodes",
            "edges",
            "outputs",
        ];
        let Some(map) = self.object(value, "", &allowed) else {
            return;
        };
        match map.get("sdrk") {
            None => self.report.error(
                "sdrk",
                DiagCode::SchemaVersion,
                "the field `sdrk` (format version) is required; use 1",
            ),
            Some(v) if v.as_u64() == Some(1) => {}
            Some(v) => self.report.error(
                "sdrk",
                DiagCode::SchemaVersion,
                format!("unsupported format version {v}; this library reads version 1"),
            ),
        }
        if let Some(v) = self.required(map, "name", "") {
            self.string(v, "name");
        }
        if let Some(v) = map.get("description") {
            self.description(v, "description");
        }
        if let Some(v) = map.get("settings") {
            self.settings(v);
        }
        if let Some(v) = map.get("inputs") {
            self.inputs(v);
        }
        if let Some(v) = map.get("groups") {
            self.groups(v);
        }
        if let Some(v) = map.get("nodes") {
            self.nodes(v);
        }
        if let Some(v) = map.get("edges") {
            self.edges(v);
        }
        if let Some(v) = map.get("outputs") {
            self.outputs(v);
        }
    }
}

/// Syntax diagnostic for a `serde_json` error, with line and column.
pub(crate) fn syntax_report(error: &serde_json::Error) -> Report {
    let mut report = Report::default();
    report.error(
        "",
        DiagCode::Syntax,
        format!(
            "invalid JSON at line {}, column {}: {}",
            error.line(),
            error.column(),
            error
        ),
    );
    report
}

/// Parses JSON text into the typed model, or returns a report with every structural problem.
pub(crate) fn parse(json: &str) -> Result<PipelineSpec, Report> {
    let value: Value = serde_json::from_str(json).map_err(|e| syntax_report(&e))?;
    parse_value(value)
}

/// Checks the structure of a parsed document (from JSON or YAML) and converts it into the
/// typed model.
pub(crate) fn parse_value(value: Value) -> Result<PipelineSpec, Report> {
    let mut checker = Checker {
        report: Report::default(),
    };
    checker.document(&value);
    if !checker.report.errors.is_empty() {
        return Err(checker.report);
    }
    serde_json::from_value::<PipelineSpec>(value).map_err(|e| {
        let mut report = Report::default();
        report.error(
            "",
            DiagCode::Syntax,
            format!("the document does not fit the model: {e}"),
        );
        report
    })
}
