//! `ParamMap`: flat parameter names of a composite block mapped to the paths of its stages.

use crate::block::{Block, ParamValue};
use crate::error::{Error, Result};

/// A table of flat parameter names and the paths they stand for in a composition of
/// [`Chain`](crate::Chain)s and [`Fork`](crate::Fork)s (`("audio_cutoff_hz", "1.1.0.cutoff_hz")`).
///
/// A composite block that offers flat parameter names keeps a `const` map and forwards its own
/// `set_param` and `param` through [`ParamMap::set`] and [`ParamMap::get`]; its
/// `description()` stays its own. A flat name that is not in the map is `UnknownParameter`; the
/// error of the inner block is returned unchanged.
///
/// One flat name may stand for several paths (M44.1.1 §4, the naming rule of `docs/blocks.md`:
/// a parameter appears once in the composite and goes, with the same value, to every building
/// block that has it): its entries are its paths in the order of the table. [`ParamMap::set`]
/// checks every path first and then writes every path; [`ParamMap::get`] reads the first.
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

    /// The first path of the flat name `flat`, or `None` if the map does not have it.
    ///
    /// ```
    /// use sdrk_blocks::ParamMap;
    /// const MAP: ParamMap = ParamMap::new(&[("cutoff_hz", "1.0.cutoff_hz")]);
    /// assert_eq!(MAP.path("cutoff_hz"), Some("1.0.cutoff_hz"));
    /// assert_eq!(MAP.path("1.0.cutoff_hz"), None);
    /// ```
    pub fn path(&self, flat: &str) -> Option<&'static str> {
        self.paths(flat).next()
    }

    /// Every path of the flat name `flat`, in the order of the table (none if the map does not
    /// have it).
    ///
    /// ```
    /// use sdrk_blocks::ParamMap;
    /// const MAP: ParamMap = ParamMap::new(&[("gain_db", "0.gain_db"), ("gain_db", "1.gain_db")]);
    /// assert!(MAP.paths("gain_db").eq(["0.gain_db", "1.gain_db"]));
    /// assert_eq!(MAP.paths("gain").count(), 0);
    /// ```
    pub fn paths<'a>(&self, flat: &'a str) -> impl Iterator<Item = &'static str> + 'a {
        self.entries
            .iter()
            .filter(move |(name, _)| *name == flat)
            .map(|(_, path)| *path)
    }

    /// Whether `path` is a path of the map (of any flat name).
    ///
    /// ```
    /// use sdrk_blocks::ParamMap;
    /// const MAP: ParamMap = ParamMap::new(&[("cutoff_hz", "1.0.cutoff_hz")]);
    /// assert!(MAP.has_path("1.0.cutoff_hz"));
    /// assert!(!MAP.has_path("cutoff_hz"));
    /// ```
    pub fn has_path(&self, path: &str) -> bool {
        self.entries.iter().any(|(_, p)| *p == path)
    }

    /// Sets the parameter `flat` of `target` through all its paths: every path is checked
    /// first ([`ParamPaths::check_path_param`]), then written in the order of the table. If a
    /// check fails, nothing is written; if a write still fails (a building block whose
    /// `set_param` checks more than its `check_param`), the paths written before it get the
    /// value of the first path back, which they held before (the paths of one flat name hold
    /// the same value), so a `set` reaches all of them or none.
    ///
    /// # Errors
    /// `UnknownParameter` if `flat` is not in the map; otherwise the first error of a building
    /// block, unchanged.
    ///
    /// ```
    /// use sdrk::{Rate, C32};
    /// use sdrk_blocks::{
    ///     AlignedBytes, ArenaStorage, Chain, Error, ParamMap, ParamValue, ScaleBlock, ScaleConfig,
    /// };
    /// const MAP: ParamMap = ParamMap::new(&[("gain_db", "0.gain_db"), ("gain_db", "1.gain_db")]);
    /// let rate = Rate::hz(48_000)?;
    /// let mut mem = AlignedBytes([0u8; 4096]);
    /// let mut arena = ArenaStorage::new(&mut mem.0);
    /// let a = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), rate, &mut arena)?;
    /// let b = ScaleBlock::<C32, _>::new(&ScaleConfig::default(), rate, &mut arena)?;
    /// let mut chain = Chain::new(a, b, 16, &mut arena)?;
    /// MAP.set(&mut chain, "gain_db", ParamValue::F64(-6.0))?;
    /// assert_eq!(MAP.get(&chain, "gain_db")?, ParamValue::F64(-6.0));
    /// assert_eq!(chain.param("1.gain_db")?, ParamValue::F64(-6.0));
    /// assert_eq!(MAP.set(&mut chain, "gain", ParamValue::F64(0.0)), Err(Error::UnknownParameter));
    /// # use sdrk_blocks::Block;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn set<T: ParamPathsMut + ?Sized>(
        &self,
        target: &mut T,
        flat: &str,
        value: ParamValue,
    ) -> Result<()> {
        let first = self.path(flat).ok_or(Error::UnknownParameter)?;
        for path in self.paths(flat) {
            target.check_path_param(path, &value)?;
        }
        let before = target.path_param(first)?;
        for (written, path) in self.paths(flat).enumerate() {
            if let Err(e) = target.set_path_param(path, value) {
                for back in self.paths(flat).take(written) {
                    // The value the path held before: it takes it again.
                    let _ = target.set_path_param(back, before);
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Reads the parameter `flat` of `target` through its first path.
    ///
    /// # Errors
    /// `UnknownParameter` if `flat` is not in the map; otherwise the error of the building
    /// block, unchanged.
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
    pub fn get<T: ParamPaths + ?Sized>(&self, target: &T, flat: &str) -> Result<ParamValue> {
        let path = self.path(flat).ok_or(Error::UnknownParameter)?;
        target.path_param(path)
    }
}

/// The building blocks of a composite, reached by paths (`"1.cutoff_hz"`: the parameter
/// `cutoff_hz` of the stage `1`): what a [`ParamMap`] reads and checks.
///
/// Every [`Block`] is one: a path is a parameter name, which [`Chain`](crate::Chain) and
/// [`Fork`](crate::Fork) route to their stages. A composite block that holds its building
/// blocks in fields of its own routes the paths of its map through a private view of them, so
/// that its building blocks are reached through its public parameters alone.
pub trait ParamPaths {
    /// The current value of the parameter at `path`.
    ///
    /// # Errors
    /// `UnknownParameter` for a path that reaches no parameter; otherwise the error of the
    /// building block.
    fn path_param(&self, path: &str) -> Result<ParamValue>;

    /// Checks whether the building block at `path` would take `value`
    /// ([`Block::check_param`]), without changing anything.
    ///
    /// # Errors
    /// As [`Block::check_param`].
    fn check_path_param(&self, path: &str, value: &ParamValue) -> Result<()>;
}

/// [`ParamPaths`] that can also be written: what [`ParamMap::set`] writes.
pub trait ParamPathsMut: ParamPaths {
    /// Sets the parameter at `path` ([`Block::set_param`]).
    ///
    /// # Errors
    /// As [`Block::set_param`].
    fn set_path_param(&mut self, path: &str, value: ParamValue) -> Result<()>;
}

impl<B: Block + ?Sized> ParamPaths for B {
    fn path_param(&self, path: &str) -> Result<ParamValue> {
        self.param(path)
    }
    fn check_path_param(&self, path: &str, value: &ParamValue) -> Result<()> {
        self.check_param(path, value)
    }
}

impl<B: Block + ?Sized> ParamPathsMut for B {
    fn set_path_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        self.set_param(path, value)
    }
}
