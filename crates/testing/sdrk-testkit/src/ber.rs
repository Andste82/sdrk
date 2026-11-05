//! A bit error rate meter for PRBS test signals (M38).
//!
//! [`BerMeter`] takes hard bits (0 or 1; any non-zero value counts as 1) of a PRBS from
//! [`sdrk_proto::prbs`] at an unknown phase.
//!
//! - **Acquisition:** every received bit is shifted into a receive register; the next bit is
//!   predicted from that register (the self-synchronizing form of the LFSR), so the meter
//!   locks to any phase of the sequence. Before the register is full, the prediction uses the
//!   seed, so a stream that starts at the seed locks after its first 32 bits. After 32 correct
//!   predictions in a row (with a non-zero register, so a stuck all-zero input never locks)
//!   the meter is locked; the 32 bits themselves are not counted.
//! - **Counting:** once locked, the meter runs its own copy of the register (fed with its own
//!   predictions, so a received error does not propagate) and counts every received bit in
//!   [`BerMeter::bits`] and every mismatch in [`BerMeter::errors`].
//! - **Slip:** when 64 or more of the last (up to) 128 counted bits are errors, the meter
//!   counts a [`BerMeter::slips`] and returns to acquisition. The bits and errors counted
//!   before the slip was detected stay counted.
//!
//! ```
//! use sdrk_proto::prbs::{Poly, Prbs};
//! use sdrk_testkit::ber::BerMeter;
//! let mut tx = Prbs::new(Poly::Prbs9, 0x1ff)?;
//! let mut meter = BerMeter::new(Poly::Prbs9, 0x1ff)?;
//! for n in 0..1032 {
//!     let bit = tx.next_bit();
//!     meter.push(if n == 500 { bit ^ 1 } else { bit });
//! }
//! assert_eq!((meter.bits(), meter.errors(), meter.slips()), (1000, 1, 0));
//! # Ok::<(), sdrk_proto::Error>(())
//! ```

use sdrk_proto::prbs::{Poly, Prbs};

/// Correct predictions in a row that lock the meter.
pub const SYNC_BITS: u32 = 32;
/// Length of the slip window in bits.
pub const SLIP_WINDOW: u32 = 128;
/// Errors within the slip window that count as a slip.
pub const SLIP_ERRORS: u32 = 64;

/// The two-sided normal quantile of 95 % (`z` of the Wilson interval).
const Z95: f64 = 1.959_963_984_540_054;

/// A PRBS bit error rate meter; see the module documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BerMeter {
    poly: Poly,
    /// The last received bits (acquisition register).
    rx: u32,
    /// The meter's own register while locked.
    local: u32,
    locked: bool,
    run: u32,
    bits: u64,
    errors: u64,
    slips: u64,
    /// Error flags of the last counted bits (bit 0 = the newest).
    window: u128,
    window_len: u32,
}

impl BerMeter {
    /// A meter for the sequence `poly`; `seed` is the initial register (see the module
    /// documentation).
    ///
    /// # Errors
    /// [`sdrk_proto::Error::InvalidArgument`] if the low `n` bits of `seed` are zero.
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// use sdrk_testkit::ber::BerMeter;
    /// assert!(BerMeter::new(Poly::Prbs15, 0).is_err());
    /// ```
    pub fn new(poly: Poly, seed: u32) -> Result<Self, sdrk_proto::Error> {
        let state = Prbs::new(poly, seed)?.state();
        Ok(Self {
            poly,
            rx: state,
            local: state,
            locked: false,
            run: 0,
            bits: 0,
            errors: 0,
            slips: 0,
            window: 0,
            window_len: 0,
        })
    }

    /// Takes one hard bit.
    ///
    /// ```
    /// use sdrk_proto::prbs::{Poly, Prbs};
    /// use sdrk_testkit::ber::BerMeter;
    /// let mut tx = Prbs::new(Poly::Prbs15, 77)?;
    /// let mut meter = BerMeter::new(Poly::Prbs15, 1)?; // another phase: locks after 15 + 32 bits
    /// for _ in 0..47 { meter.push(tx.next_bit()); }
    /// assert!(meter.is_locked());
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn push(&mut self, bit: u8) {
        let bit = u8::from(bit != 0);
        if self.locked {
            let expected = self.poly.feedback(self.local);
            self.local = self.poly.shift(self.local, expected);
            self.rx = self.poly.shift(self.rx, bit);
            let err = bit != expected;
            self.bits += 1;
            self.errors += u64::from(err);
            self.window = (self.window << 1) | u128::from(err);
            self.window_len = (self.window_len + 1).min(SLIP_WINDOW);
            if self.window.count_ones() >= SLIP_ERRORS {
                self.slips += 1;
                self.locked = false;
                self.run = 0;
            }
        } else {
            let expected = self.poly.feedback(self.rx);
            self.rx = self.poly.shift(self.rx, bit);
            // an all-zero register is no state of the sequence (a stuck all-zero input would
            // otherwise predict itself)
            if bit == expected && self.rx != 0 {
                self.run += 1;
            } else {
                self.run = 0;
            }
            if self.run >= SYNC_BITS {
                self.locked = true;
                self.local = self.rx;
                self.window = 0;
                self.window_len = 0;
            }
        }
    }

    /// Takes a slice of hard bits.
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// use sdrk_testkit::ber::BerMeter;
    /// let mut meter = BerMeter::new(Poly::Prbs9, 1)?;
    /// meter.push_bits(&[0; 100]); // the all-zero stream never locks
    /// assert!(!meter.is_locked());
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn push_bits(&mut self, bits: &[u8]) {
        for &b in bits {
            self.push(b);
        }
    }

    /// True while the meter is locked to the sequence.
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// use sdrk_testkit::ber::BerMeter;
    /// assert!(!BerMeter::new(Poly::Prbs9, 1)?.is_locked());
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Bits counted (received while locked).
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// use sdrk_testkit::ber::BerMeter;
    /// assert_eq!(BerMeter::new(Poly::Prbs9, 1)?.bits(), 0);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn bits(&self) -> u64 {
        self.bits
    }

    /// Errors counted.
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// use sdrk_testkit::ber::BerMeter;
    /// assert_eq!(BerMeter::new(Poly::Prbs9, 1)?.errors(), 0);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn errors(&self) -> u64 {
        self.errors
    }

    /// Slips detected (each one a return to acquisition).
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// use sdrk_testkit::ber::BerMeter;
    /// assert_eq!(BerMeter::new(Poly::Prbs9, 1)?.slips(), 0);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn slips(&self) -> u64 {
        self.slips
    }

    /// The bit error rate `errors / bits` (0.0 before the first counted bit).
    ///
    /// ```
    /// use sdrk_proto::prbs::Poly;
    /// use sdrk_testkit::ber::BerMeter;
    /// assert_eq!(BerMeter::new(Poly::Prbs9, 1)?.ber(), 0.0);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn ber(&self) -> f64 {
        if self.bits == 0 {
            0.0
        } else {
            self.errors as f64 / self.bits as f64
        }
    }

    /// The 95 % Wilson score interval `(low, high)` of the bit error rate (`(0.0, 1.0)` before
    /// the first counted bit).
    ///
    /// ```
    /// use sdrk_proto::prbs::{Poly, Prbs};
    /// use sdrk_testkit::ber::BerMeter;
    /// let mut tx = Prbs::new(Poly::Prbs9, 1)?;
    /// let mut meter = BerMeter::new(Poly::Prbs9, 1)?;
    /// for _ in 0..1032 { meter.push(tx.next_bit()); }
    /// let (lo, hi) = meter.confidence_95();
    /// assert!(lo < 1e-12);
    /// assert!(hi > 0.0 && hi < 0.004); // about z^2 / n for zero errors
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn confidence_95(&self) -> (f64, f64) {
        if self.bits == 0 {
            return (0.0, 1.0);
        }
        let n = self.bits as f64;
        let p = self.errors as f64 / n;
        let z2 = Z95 * Z95;
        let den = 1.0 + z2 / n;
        let centre = (p + z2 / (2.0 * n)) / den;
        let half = Z95 / den * libm::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n));
        ((centre - half).max(0.0), (centre + half).min(1.0))
    }
}
