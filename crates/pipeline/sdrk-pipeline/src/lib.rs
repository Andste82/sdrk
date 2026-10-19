//! SDRk pipeline: signal-processing graphs described as data.
//!
//! Layer `crates/pipeline` (heap allowed). A pipeline is a JSON document (inputs, nodes, edges,
//! outputs). This crate parses and validates it with machine-readable diagnostics, builds it
//! into a [`Pipeline`] of type-erased [`Node`]s over the heap-free blocks of `sdrk-blocks`, and
//! runs it in push mode. Memory is allocated once, in `build`; processing does not allocate.
//!
//! ```
//! use sdrk_pipeline::{BuildOptions, InputData, Registry};
//!
//! let json = r#"{
//!   "sdrk": 1,
//!   "name": "tone-level",
//!   "settings": { "precision": "f32", "max_chunk": 1024 },
//!   "inputs": { "rf": { "format": "cu8", "rate_hz": 48000 } },
//!   "nodes": { "gain": { "kind": "scale", "params": { "gain_db": -6 } } },
//!   "edges": [ { "from": "rf", "to": "gain" }, { "from": "gain", "to": "out" } ],
//!   "outputs": { "out": { "type": "stream" } }
//! }"#;
//! let registry = Registry::standard();
//! let report = registry.validate_json(json, &BuildOptions::default());
//! assert!(report.is_ok(), "{}", report.to_json());
//! let mut pipeline = registry.build_json(json, &BuildOptions::default())?;
//! pipeline.push("rf", InputData::Raw(&[255, 128, 255, 128]))?;
//! let mut out = [sdrk::C32::default(); 4];
//! assert_eq!(pipeline.read_stream("out", &mut out)?, 2);
//! # Ok::<(), sdrk_pipeline::Error>(())
//! ```
#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]

mod analysis;
mod build;
mod error;
mod kinds;
mod model;
mod node;
mod params;
mod registry;
mod runner;
mod schema;
mod storage;
mod structure;
mod validate;

pub use error::{Error, Result};
pub use model::{
    AlignSpec, ContainerSpec, CustomFormat, DEFAULT_MAX_CHUNK, DEFAULT_QUEUE, DropPolicy, EdgeSpec,
    EndianSpec, FORMAT_NAMES, FORMAT_VERSION, FormatName, FormatSpec, FullScaleSpec, InputSpec,
    InputType, LayoutSpec, MAX_CHUNK_LIMIT, MAX_DESCRIPTION_CHARS, NAME_PATTERN, NodeSpec,
    OutputSpec, OutputType, PipelineSpec, Precision, RateHz, Settings, is_valid_name,
};
pub use node::{FrameNode, Node, NodeElem, NodeInput, NodeOutput, StreamNode};
pub use registry::{BlockFactory, BuildOptions, Checked, Limits, PortKind, Registry};
pub use runner::{InputData, NodeStats, Pipeline};
pub use schema::{SCHEMA_FILE, SCHEMA_ID};
pub use storage::VecStorage;
pub use validate::{DiagCode, Diagnostic, NodeReport, Report};
