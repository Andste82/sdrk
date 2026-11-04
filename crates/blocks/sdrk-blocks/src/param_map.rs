//! `ParamMap`: flat parameter names of a composite block mapped to the paths of its stages.

use crate::block::{Block, ParamValue};
use crate::error::{Error, Result};

/// A table of flat parameter names and the paths they stand for in a composition of
/// [`Chain`](crate::Chain)s and [`Fork`](crate::Fork)s (`("audio_cutoff_hz", "1.1.0.cutoff_hz")`).
///
/// A composite block that offers flat parameter names keeps a `const` map and forwards its own
/// `set_param` and `param` through [`ParamMap::set`] and [`ParamMap::get`]; its
/// `description()` stays its own. A flat name that is not in the map is `UnknownParameter`; the
/// error of the inner block is returned unchanged. The first entry of a name wins.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{
///     AlignedBytes, ArenaStorage, DelayConfig, DelayRealBlock, Error, Fork, ParamMap, ParamValue,
/// };
/// const MAP: ParamMap = ParamMap::new(&[("long", "0.samples"), ("short", "1.samples")]);
/// let rate = Rate::hz(48_000)?;
/// let mut mem = AlignedBytes([0u8; 4096]);
/// let mut arena = ArenaStorage::new(&mut mem.0);
/// let a = DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 4 }, rate, &mut arena)?;
/// let b = DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 1 }, rate, &mut arena)?;
/// let fork = Fork::new(a, b, 16, &mut arena)?;
/// assert_eq!(MAP.path("long"), Some("0.samples"));
/// assert_eq!(MAP.get(&fork, "short")?, ParamValue::I64(1));
/// assert_eq!(MAP.get(&fork, "samples"), Err(Error::UnknownParameter));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamMap {
    entries: &'static [(&'static str, &'static str)],
}

impl ParamMap {
    /// A map of `(flat name, path)` entries.
    ///
    /// ```
    /// use sdrk_blocks::ParamMap;
    /// const MAP: ParamMap = ParamMap::new(&[("cutoff_hz", "1.0.cutoff_hz")]);
    /// assert_eq!(MAP.entries().len(), 1);
    /// ```
    pub const fn new(entries: &'static [(&'static str, &'static str)]) -> Self {
        Self { entries }
    }

    /// The `(flat name, path)` entries in the order they were given.
    ///
    /// ```
    /// use sdrk_blocks::ParamMap;
    /// const MAP: ParamMap = ParamMap::new(&[("gain", "0.gain_db")]);
    /// assert_eq!(MAP.entries(), &[("gain", "0.gain_db")]);
    /// ```
    pub const fn entries(&self) -> &'static [(&'static str, &'static str)] {
        self.entries
    }

    /// The path of the flat name `flat`, or `None` if the map does not have it.
    ///
    /// ```
    /// use sdrk_blocks::ParamMap;
    /// const MAP: ParamMap = ParamMap::new(&[("cutoff_hz", "1.0.cutoff_hz")]);
    /// assert_eq!(MAP.path("cutoff_hz"), Some("1.0.cutoff_hz"));
    /// assert_eq!(MAP.path("1.0.cutoff_hz"), None);
    /// ```
    pub fn path(&self, flat: &str) -> Option<&'static str> {
        self.entries
            .iter()
            .find(|(name, _)| *name == flat)
            .map(|(_, path)| *path)
    }

    /// Sets the parameter `flat` of `block` through its path.
    ///
    /// # Errors
    /// `UnknownParameter` if `flat` is not in the map; otherwise the error of
    /// `block.set_param`, unchanged.
    ///
    /// ```
    /// use sdrk::{Rate, C32};
    /// use sdrk_blocks::{
    ///     AlignedBytes, ArenaStorage, Chain, Error, ParamMap, ParamValue, ScaleBlock, ScaleConfig,
    /// };
    /// const MAP: ParamMap = ParamMap::new(&[("gain_db", "1.gain_db")]);
    /// let rate = Rate::hz(48_000)?;
    /// let mut mem = AlignedBytes([0u8; 4096]);
    /// let mut arena = ArenaStorage::new(&mut mem.0);
    /// let a = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), rate, &mut arena)?;
    /// let b = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), rate, &mut arena)?;
    /// let mut chain = Chain::new(a, b, 16, &mut arena)?;
    /// MAP.set(&mut chain, "gain_db", ParamValue::F64(-6.0))?;
    /// assert_eq!(MAP.get(&chain, "gain_db")?, ParamValue::F64(-6.0));
    /// assert_eq!(MAP.set(&mut chain, "gain", ParamValue::F64(0.0)), Err(Error::UnknownParameter));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn set<B: Block + ?Sized>(
        &self,
        block: &mut B,
        flat: &str,
        value: ParamValue,
    ) -> Result<()> {
        let path = self.path(flat).ok_or(Error::UnknownParameter)?;
        block.set_param(path, value)
    }

    /// Reads the parameter `flat` of `block` through its path.
    ///
    /// # Errors
    /// `UnknownParameter` if `flat` is not in the map; otherwise the error of `block.param`,
    /// unchanged.
    ///
    /// ```
    /// use sdrk::{Rate, C32};
    /// use sdrk_blocks::{AlignedBytes, ArenaStorage, Chain, ParamMap, ParamValue, ScaleBlock, ScaleConfig};
    /// const MAP: ParamMap = ParamMap::new(&[("first_gain_db", "0.gain_db")]);
    /// let rate = Rate::hz(48_000)?;
    /// let mut mem = AlignedBytes([0u8; 4096]);
    /// let mut arena = ArenaStorage::new(&mut mem.0);
    /// let cfg = ScaleConfig { gain_db: -3.0 };
    /// let a = ScaleBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
    /// let b = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), rate, &mut arena)?;
    /// let chain = Chain::new(a, b, 16, &mut arena)?;
    /// assert_eq!(MAP.get(&chain, "first_gain_db")?, ParamValue::F64(-3.0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get<B: Block + ?Sized>(&self, block: &B, flat: &str) -> Result<ParamValue> {
        let path = self.path(flat).ok_or(Error::UnknownParameter)?;
        block.param(path)
    }
}
