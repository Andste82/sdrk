//! NRZI line coding with the AX.25 convention: a 0 bit toggles the line level, a 1 bit keeps
//! it. The receiver reads a 1 where two consecutive levels are equal and a 0 where they
//! differ, so the coding is independent of the polarity of the channel (an FSK receiver may
//! swap mark and space without harm).
//!
//! [`encode`] maps one bit to the next line level; [`Decoder`] holds the last level and maps
//! each received level back to a bit. Both work one bit at a time, so a block can run them in
//! its own per-bit loop without a buffer.
//!
//! ```
//! use sdrk_proto::nrzi::{Decoder, encode};
//!
//! let bits = [true, false, false, true, true, false];
//! let mut level = false;
//! let mut dec = Decoder::new(false);
//! for &b in &bits {
//!     level = encode(level, b);
//!     assert_eq!(dec.decode(level), b);
//! }
//! ```

/// The line level after sending `bit` on a line at level `level`: a 0 bit toggles the level,
/// a 1 bit keeps it.
///
/// ```
/// use sdrk_proto::nrzi::encode;
/// assert!(encode(true, true));
/// assert!(!encode(true, false));
/// assert!(encode(false, false));
/// ```
#[inline]
pub const fn encode(level: bool, bit: bool) -> bool {
    level ^ !bit
}

/// The NRZI decoder: it keeps the last line level.
///
/// ```
/// use sdrk_proto::nrzi::Decoder;
///
/// let mut dec = Decoder::new(false);
/// assert!(dec.decode(false)); // same level: 1
/// assert!(!dec.decode(true)); // change: 0
/// assert!(dec.level());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Decoder {
    level: bool,
}

impl Decoder {
    /// A decoder whose previous level is `level` (the first decoded bit compares against it).
    pub const fn new(level: bool) -> Self {
        Self { level }
    }

    /// The bit of the received line level `level`: 1 when it equals the previous level, 0 when
    /// it differs. `level` becomes the previous level.
    #[inline]
    pub fn decode(&mut self, level: bool) -> bool {
        let bit = level == self.level;
        self.level = level;
        bit
    }

    /// The last line level.
    pub const fn level(&self) -> bool {
        self.level
    }

    /// Sets the previous level to `level`.
    pub fn reset(&mut self, level: bool) {
        self.level = level;
    }
}
