//! Pseudo-random binary sequences (PRBS) for bit error rate tests.
//!
//! [`Prbs`] is a Fibonacci linear-feedback shift register. The register holds the last `n`
//! output bits (`n` = the degree of the polynomial); each step XORs the two tapped stages, the
//! result is both the output bit and the new first stage:
//!
//! | [`Poly`] | Polynomial | Taps (stages) | Period |
//! | --- | --- | --- | --- |
//! | `Prbs5` | `x^5 + x^3 + 1` | 5, 3 | 31 |
//! | `Prbs7` | `x^7 + x^6 + 1` | 7, 6 | 127 |
//! | `Prbs9` | `x^9 + x^5 + 1` | 9, 5 | 511 |
//! | `Prbs10` | `x^10 + x^7 + 1` | 10, 7 | 1023 |
//! | `Prbs11` | `x^11 + x^9 + 1` | 11, 9 | 2047 |
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
    /// `x^5 + x^3 + 1`, period 31 (M49).
    Prbs5,
    /// `x^7 + x^6 + 1`, period 127 (M49).
    Prbs7,
    /// `x^10 + x^7 + 1`, period 1023 (M49).
    Prbs10,
    /// `x^11 + x^9 + 1`, period 2047 (M49).
    Prbs11,
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
            Self::Prbs5 => 5,
            Self::Prbs7 => 7,
            Self::Prbs9 => 9,
            Self::Prbs10 => 10,
            Self::Prbs11 => 11,
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

    /// The second tap `t` of `x^n + x^t + 1` (the first tap is stage `n`).
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// assert_eq!(Poly::Prbs5.second_tap(), 3);
    /// assert_eq!(Poly::Prbs15.second_tap(), 14);
    /// ```
    pub const fn second_tap(self) -> u32 {
        match self {
            Self::Prbs5 => 3,
            Self::Prbs7 => 6,
            Self::Prbs9 => 5,
            Self::Prbs10 => 7,
            Self::Prbs11 => 9,
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

    /// The next `count` bits (0 to 32) as one word, the first bit as the most significant of
    /// the `count` bits: the same bits and the same final state as `count` calls of
    /// [`next_bit`](Self::next_bit), with `0` for `count = 0`.
    ///
    /// The register is stepped up to `t` bits at a time (`t` the second tap): the next `m ≤ t`
    /// bits depend only on stages that are already in the register (bit `j` is stage `n − j`
    /// XOR stage `t − j`), so they come from two shifts and one XOR of the register content.
    ///
    /// A `count` above 32 is taken as 32.
    ///
    /// ```
    /// use sdrk_proto::prbs::{Poly, Prbs};
    /// let mut a = Prbs::new(Poly::Prbs9, 0x1ff)?;
    /// let mut b = a.clone();
    /// let word = a.next_bits(8);
    /// let bits: u32 = (0..8).fold(0, |w, _| (w << 1) | u32::from(b.next_bit()));
    /// assert_eq!(word, bits);
    /// assert_eq!(word, 0b0000_0111);
    /// assert_eq!(a, b);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn next_bits(&mut self, count: u32) -> u32 {
        let n = self.poly.degree();
        let t = self.poly.second_tap();
        let mut left = count.min(32);
        let mut word = 0u32;
        while left > 0 {
            let m = left.min(t);
            let low = (1u32 << m) - 1;
            let bits = ((self.state >> (n - m)) ^ (self.state >> (t - m))) & low;
            self.state = ((self.state << m) | bits) & self.poly.mask();
            word = (word << m) | bits;
            left -= m;
        }
        word
    }
}
