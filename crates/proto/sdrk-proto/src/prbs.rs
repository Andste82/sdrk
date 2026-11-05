//! Pseudo-random binary sequences (PRBS) for bit error rate tests.
//!
//! [`Prbs`] is a Fibonacci linear-feedback shift register. The register holds the last `n`
//! output bits (`n` = the degree of the polynomial); each step XORs the two tapped stages, the
//! result is both the output bit and the new first stage:
//!
//! | [`Poly`] | Polynomial | Taps (stages) | Period |
//! | --- | --- | --- | --- |
//! | `Prbs9` | `x^9 + x^5 + 1` | 9, 5 | 511 |
//! | `Prbs15` | `x^15 + x^14 + 1` | 15, 14 | 32767 |
//!
//! The seed is the initial register content (bit `k` = stage `k + 1`, only the low `n` bits
//! are used) and must be non-zero, because the all-zero register never leaves zero.
//!
//! ```
//! use sdrk_proto::prbs::{Poly, Prbs};
//!
//! let mut p = Prbs::new(Poly::Prbs9, 0x1ff)?;
//! let first: Vec<u8> = (0..8).map(|_| p.next_bit()).collect();
//! assert_eq!(first, [0, 0, 0, 0, 0, 1, 1, 1]);
//! # Ok::<(), sdrk_proto::Error>(())
//! ```

use crate::error::{Error, Result};

/// The generator polynomial of a [`Prbs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Poly {
    /// `x^9 + x^5 + 1`, period 511.
    Prbs9,
    /// `x^15 + x^14 + 1`, period 32767.
    Prbs15,
}

impl Poly {
    /// The degree `n` of the polynomial (the register length in bits).
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// assert_eq!(Poly::Prbs9.degree(), 9);
    /// assert_eq!(Poly::Prbs15.degree(), 15);
    /// ```
    pub const fn degree(self) -> u32 {
        match self {
            Self::Prbs9 => 9,
            Self::Prbs15 => 15,
        }
    }

    /// The period of the sequence, `2^n - 1` bits.
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// assert_eq!(Poly::Prbs15.period(), 32767);
    /// ```
    pub const fn period(self) -> u32 {
        (1u32 << self.degree()) - 1
    }

    /// The second tap (the first tap is stage `n`).
    const fn second_tap(self) -> u32 {
        match self {
            Self::Prbs9 => 5,
            Self::Prbs15 => 14,
        }
    }

    const fn mask(self) -> u32 {
        self.period()
    }

    /// The feedback bit of the register content `state` (bit `k` = stage `k + 1`).
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// // stage 9 set, stage 5 clear: the feedback is 1
    /// assert_eq!(Poly::Prbs9.feedback(1 << 8), 1);
    /// assert_eq!(Poly::Prbs9.feedback(0x1ff), 0);
    /// ```
    pub const fn feedback(self, state: u32) -> u8 {
        (((state >> (self.degree() - 1)) ^ (state >> (self.second_tap() - 1))) & 1) as u8
    }

    /// Shifts `bit` into the register content `state` as the new first stage.
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// assert_eq!(Poly::Prbs9.shift(0x1ff, 0), 0x1fe);
    /// ```
    pub const fn shift(self, state: u32, bit: u8) -> u32 {
        ((state << 1) | (bit & 1) as u32) & self.mask()
    }
}

/// A PRBS generator (Fibonacci LFSR); see the module documentation.
///
/// ```
/// use sdrk_proto::prbs::{Poly, Prbs};
///
/// let mut p = Prbs::new(Poly::Prbs15, 1)?;
/// let ones: u32 = (0..32767).map(|_| u32::from(p.next_bit())).sum();
/// assert_eq!(ones, 16384); // a maximal-length sequence has 2^(n-1) ones per period
/// assert!(Prbs::new(Poly::Prbs9, 0).is_err());
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prbs {
    poly: Poly,
    state: u32,
}

impl Prbs {
    /// Creates a generator with the initial register content `seed` (low `n` bits).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] if the low `n` bits of `seed` are all zero.
    ///
    /// ```
    /// use sdrk_proto::prbs::{Poly, Prbs};
    /// assert!(Prbs::new(Poly::Prbs9, 0x1ff).is_ok());
    /// assert!(Prbs::new(Poly::Prbs9, 0x200).is_err()); // only the low 9 bits count
    /// ```
    pub fn new(poly: Poly, seed: u32) -> Result<Self> {
        let state = seed & poly.mask();
        if state == 0 {
            return Err(Error::InvalidArgument("prbs seed must be non-zero"));
        }
        Ok(Self { poly, state })
    }

    /// The polynomial.
    ///
    /// ```
    /// use sdrk_proto::prbs::{Poly, Prbs};
    /// assert_eq!(Prbs::new(Poly::Prbs9, 1)?.poly(), Poly::Prbs9);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn poly(&self) -> Poly {
        self.poly
    }

    /// The current register content (bit `k` = stage `k + 1`).
    ///
    /// ```
    /// use sdrk_proto::prbs::{Poly, Prbs};
    /// assert_eq!(Prbs::new(Poly::Prbs9, 0x3ff)?.state(), 0x1ff);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn state(&self) -> u32 {
        self.state
    }

    /// The next bit (0 or 1): the XOR of the tapped stages, which also becomes stage 1.
    ///
    /// ```
    /// use sdrk_proto::prbs::{Poly, Prbs};
    /// let mut p = Prbs::new(Poly::Prbs9, 1 << 8)?;
    /// assert_eq!(p.next_bit(), 1);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn next_bit(&mut self) -> u8 {
        let bit = self.poly.feedback(self.state);
        self.state = self.poly.shift(self.state, bit);
        bit
    }
}
