//! Machine-readable block descriptions.
//!
//! Every block has a [`BlockDescription`]: what it does, its ports and its parameters with
//! units, ranges, defaults and mutability. The descriptions are `static` data; the pipeline
//! builds its JSON schema and its documentation from them, and the config types of the blocks
//! use the same tables to check names, kinds and ranges.

use crate::block::ParamValue;
use crate::error::{Error, Result};

/// Everything a user (or an AI agent) needs to know about a block kind.
///
/// ```
/// use sdrk_blocks::all_descriptions;
/// let ddc = all_descriptions().iter().find(|d| d.kind == "ddc").unwrap();
/// assert!(ddc.summary.ends_with('.'));
/// assert!(ddc.required.contains(&"decimation"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockDescription {
    /// Unique `snake_case` name of the kind, e.g. `"ddc"`.
    pub kind: &'static str,
    /// One sentence (at most 120 characters, ends with a period).
    pub summary: &'static str,
    /// A paragraph: what the block does, typical use, pitfalls.
    pub details: &'static str,
    /// Input ports.
    pub inputs: &'static [PortDesc],
    /// Output ports.
    pub outputs: &'static [PortDesc],
    /// Parameters (build-time and run-time).
    pub params: &'static [ParamDesc],
    /// Names of the parameters that must be given (they have no default at all).
    pub required: &'static [&'static str],
}

impl BlockDescription {
    /// The description of the parameter `name`.
    ///
    /// ```
    /// use sdrk_blocks::all_descriptions;
    /// let fir = all_descriptions().iter().find(|d| d.kind == "fir").unwrap();
    /// assert!(fir.param("cutoff_hz").is_some());
    /// assert!(fir.param("nope").is_none());
    /// ```
    pub fn param(&self, name: &str) -> Option<&'static ParamDesc> {
        self.params.iter().find(|p| p.name == name)
    }
}

/// A port of a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortDesc {
    /// Port name.
    pub name: &'static str,
    /// The sample types the port accepts or produces.
    pub types: &'static [PortType],
    /// What flows through the port.
    pub description: &'static str,
}

/// The type of the data on a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PortType {
    /// Complex `f32`.
    C32,
    /// Complex Q15.
    CQ15,
    /// Real `f32`.
    F32,
    /// Real Q15.
    Q15,
    /// Bytes in a [`SampleFormat`](sdrk_device::SampleFormat).
    Raw,
    /// A frame of `f32` values whose length is the value of the named parameter.
    Frame {
        /// Name of the parameter that holds the frame length.
        len_param: &'static str,
    },
}

/// A parameter of a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamDesc {
    /// `snake_case` name; a unit suffix is part of it (`frequency_hz`).
    pub name: &'static str,
    /// Unit of the value.
    pub unit: Unit,
    /// Type, range and default.
    pub kind: ParamKind,
    /// Human text, e.g. `"100e3 for FM broadcast"`.
    pub typical: &'static str,
    /// Whether the parameter can change while the block runs.
    pub mutability: Mutability,
    /// What the parameter does.
    pub description: &'static str,
}

impl ParamDesc {
    /// Checks the type and range of `value`.
    ///
    /// # Errors
    /// `WrongParamType` if the value has the wrong type, `OutOfRange` if a number is outside
    /// the absolute limits of the parameter (NaN is out of range).
    ///
    /// ```
    /// use sdrk_blocks::{all_descriptions, Error, ParamValue};
    /// let ddc = all_descriptions().iter().find(|d| d.kind == "ddc").unwrap();
    /// let decimation = ddc.param("decimation").unwrap();
    /// assert!(decimation.check(&ParamValue::I64(10)).is_ok());
    /// assert!(matches!(decimation.check(&ParamValue::I64(0)), Err(Error::OutOfRange { .. })));
    /// assert_eq!(decimation.check(&ParamValue::Bool(true)), Err(Error::WrongParamType));
    /// ```
    pub fn check(&self, value: &ParamValue) -> Result<()> {
        match (&self.kind, value) {
            (ParamKind::F64 { min, max, .. }, ParamValue::F64(v)) => {
                if *v >= *min && *v <= *max {
                    Ok(())
                } else {
                    Err(Error::OutOfRange {
                        name: self.name,
                        min: *min,
                        max: *max,
                        got: *v,
                    })
                }
            }
            (ParamKind::I64 { min, max, .. }, ParamValue::I64(v)) => {
                if v >= min && v <= max {
                    Ok(())
                } else {
                    Err(Error::OutOfRange {
                        name: self.name,
                        min: *min as f64,
                        max: *max as f64,
                        got: *v as f64,
                    })
                }
            }
            (ParamKind::Bool { .. }, ParamValue::Bool(_))
            | (ParamKind::Format, ParamValue::Format(_)) => Ok(()),
            (ParamKind::Choice { options, .. }, ParamValue::Choice(i)) => {
                if usize::from(*i) < options.len() {
                    Ok(())
                } else {
                    Err(Error::OutOfRange {
                        name: self.name,
                        min: 0.0,
                        max: options.len().saturating_sub(1) as f64,
                        got: f64::from(*i),
                    })
                }
            }
            _ => Err(Error::WrongParamType),
        }
    }

    /// The fixed default of the parameter, if it has one.
    ///
    /// ```
    /// use sdrk_blocks::{all_descriptions, ParamValue};
    /// let ddc = all_descriptions().iter().find(|d| d.kind == "ddc").unwrap();
    /// assert_eq!(ddc.param("frequency_hz").unwrap().default_value(), Some(ParamValue::F64(0.0)));
    /// assert_eq!(ddc.param("cutoff_hz").unwrap().default_value(), None); // depends on the rate
    /// ```
    pub fn default_value(&self) -> Option<ParamValue> {
        match self.kind {
            ParamKind::F64 { default, .. } => default.map(ParamValue::F64),
            ParamKind::I64 { default, .. } => default.map(ParamValue::I64),
            ParamKind::Bool { default } => Some(ParamValue::Bool(default)),
            ParamKind::Choice { default, .. } => Some(ParamValue::Choice(default)),
            ParamKind::Format => None,
        }
    }
}

/// Unit of a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unit {
    /// Dimensionless (counts, flags, choices).
    None,
    /// Hertz.
    Hz,
    /// Seconds.
    Seconds,
    /// Microseconds.
    Microseconds,
    /// Decibels.
    Decibel,
    /// Parts per million.
    Ppm,
    /// Samples.
    Samples,
    /// A plain ratio.
    Ratio,
}

impl Unit {
    /// Short text for tables: `"Hz"`, `"dB"`, …; empty for [`Unit::None`].
    ///
    /// ```
    /// assert_eq!(sdrk_blocks::Unit::Decibel.symbol(), "dB");
    /// ```
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Hz => "Hz",
            Self::Seconds => "s",
            Self::Microseconds => "us",
            Self::Decibel => "dB",
            Self::Ppm => "ppm",
            Self::Samples => "samples",
            Self::Ratio => "ratio",
        }
    }
}

/// When a parameter can be changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Mutability {
    /// Only at construction.
    Build,
    /// Also while the block runs (applies from the next `process` call).
    Runtime,
}

/// Type, range and default of a parameter.
///
/// `min` and `max` are the absolute limits that a schema can check. Limits and defaults that
/// depend on the input rate (`frequency_hz` within ±fs/2, `cutoff_hz` default `0.4 fs / decimation`)
/// are applied by `validate` and `new` of the block's config; their `default` is `None` and the
/// rule is written in `typical` and `description`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ParamKind {
    /// A floating-point number.
    F64 {
        /// Smallest value.
        min: f64,
        /// Largest value.
        max: f64,
        /// Default; `None` for a required parameter or a rate-dependent default.
        default: Option<f64>,
    },
    /// An integer.
    I64 {
        /// Smallest value.
        min: i64,
        /// Largest value.
        max: i64,
        /// Default; `None` for a required parameter.
        default: Option<i64>,
    },
    /// A flag.
    Bool {
        /// Default.
        default: bool,
    },
    /// One of a fixed list of names.
    Choice {
        /// The names; a value is the index into this list.
        options: &'static [&'static str],
        /// Index of the default.
        default: u16,
    },
    /// A [`SampleFormat`](sdrk_device::SampleFormat) given by name (e.g. `"cu8"`); there is no default.
    Format,
}

/// Every block kind of this crate, in a fixed order.
///
/// ```
/// let kinds: Vec<_> = sdrk_blocks::all_descriptions().iter().map(|d| d.kind).collect();
/// assert_eq!(kinds.len(), 19);
/// assert!(kinds.contains(&"resampler"));
/// ```
pub fn all_descriptions() -> &'static [&'static BlockDescription] {
    static ALL: [&BlockDescription; 19] = [
        &crate::blocks::convert::DESCRIPTION,
        &crate::blocks::convert_out::DESCRIPTION,
        &crate::blocks::scale::DESCRIPTION,
        &crate::blocks::dc_block::DESCRIPTION,
        &crate::blocks::fir::DESCRIPTION,
        &crate::blocks::ddc::DESCRIPTION,
        &crate::blocks::duc::DESCRIPTION,
        &crate::blocks::fm_demod::DESCRIPTION,
        &crate::blocks::fm_mod::DESCRIPTION,
        &crate::blocks::fm_stereo::DESCRIPTION,
        &crate::blocks::resampler::DESCRIPTION,
        &crate::blocks::spectrum::DESCRIPTION,
        &crate::blocks::squelch::DESCRIPTION,
        &crate::blocks::agc::DESCRIPTION,
        &crate::blocks::am_demod::DESCRIPTION,
        &crate::blocks::ssb_demod::DESCRIPTION,
        &crate::blocks::biquad::DESCRIPTION,
        &crate::blocks::halfband::DESCRIPTION,
        &crate::blocks::real_to_complex::DESCRIPTION,
    ];
    &ALL
}
