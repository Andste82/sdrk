//! Data whitening: XOR with a pseudo-random byte sequence, so that long runs of equal bits in
//! the data do not reach the modulator (they would starve the clock recovery and shift the
//! mean frequency of an FSK signal). Whitening twice with the same sequence restores the data.
//!
//! [`Pn9`] is the PN9 sequence of many sub-GHz transceivers (for example the CC1101 and the
//! CC1310 family): a 9-bit register with the polynomial `x^9 + x^5 + 1`, initial value
//! [`PN9_SEED`] `0x1FF`. Each byte of the sequence is the low 8 bits of the register (bit 0
//! is the least significant bit of the byte); then the register is shifted 8 times, each time
//! right by one with `bit0 XOR bit5` entering as the new bit 8. The sequence starts with
//! `FF E1 1D 9A ED 85 33 24` and repeats after [`PN9_PERIOD`] = 511 bytes.
//!
//! ```
//! use sdrk_proto::whiten::Pn9;
//!
//! let mut data = *b"hello";
//! Pn9::new().apply(&mut data);
//! assert_eq!(data[0], b'h' ^ 0xFF);
//! Pn9::new().apply(&mut data); // the same sequence again restores the data
//! assert_eq!(&data, b"hello");
//! ```

/// The initial register value of [`Pn9`].
pub const PN9_SEED: u16 = 0x1FF;

/// The period of the PN9 byte sequence in bytes (`2^9 - 1`).
pub const PN9_PERIOD: usize = 511;

/// One period of the PN9 byte sequence, built by the register of the module documentation.
static PN9_BYTES: [u8; PN9_PERIOD] = pn9_bytes();

const fn pn9_bytes() -> [u8; PN9_PERIOD] {
    let mut table = [0u8; PN9_PERIOD];
    let mut state = PN9_SEED;
    let mut k = 0;
    while k < PN9_PERIOD {
        #[allow(clippy::indexing_slicing)] // const fn: `k < PN9_PERIOD`
        {
            table[k] = (state & 0xFF) as u8;
        }
        let mut s = 0;
        while s < 8 {
            let bit = (state ^ (state >> 5)) & 1;
            state = (state >> 1) | (bit << 8);
            s += 1;
        }
        k += 1;
    }
    table
}

/// The byte `k` of the PN9 sequence (counted from the seed; any `k`, the sequence repeats).
///
/// ```
/// assert_eq!(sdrk_proto::whiten::pn9_byte(1), 0xE1);
/// assert_eq!(sdrk_proto::whiten::pn9_byte(511), 0xFF);
/// ```
#[inline]
pub fn pn9_byte(k: usize) -> u8 {
    PN9_BYTES.get(k % PN9_PERIOD).copied().unwrap_or(0)
}

/// A PN9 whitening sequence (see the module documentation); it remembers its position, so a
/// stream can be whitened piece by piece.
///
/// ```
/// use sdrk_proto::whiten::Pn9;
///
/// let mut pn9 = Pn9::new();
/// let first: Vec<u8> = (0..8).map(|_| pn9.next_byte()).collect();
/// assert_eq!(first, [0xFF, 0xE1, 0x1D, 0x9A, 0xED, 0x85, 0x33, 0x24]);
/// assert_eq!(pn9.position(), 8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Pn9 {
    /// Position in the byte sequence, `0 .. PN9_PERIOD`.
    pos: usize,
}

impl Pn9 {
    /// The sequence at its start (the register holds [`PN9_SEED`]).
    ///
    /// ```
    /// assert_eq!(sdrk_proto::whiten::Pn9::new().position(), 0);
    /// ```
    pub const fn new() -> Self {
        Self { pos: 0 }
    }

    /// The number of bytes produced since the start, modulo [`PN9_PERIOD`].
    ///
    /// ```
    /// use sdrk_proto::whiten::Pn9;
    /// let mut pn9 = Pn9::new();
    /// pn9.apply(&mut [0u8; 600]);
    /// assert_eq!(pn9.position(), 600 - 511);
    /// ```
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Back to the start of the sequence.
    ///
    /// ```
    /// use sdrk_proto::whiten::Pn9;
    /// let mut pn9 = Pn9::new();
    /// pn9.next_byte();
    /// pn9.reset();
    /// assert_eq!(pn9.next_byte(), 0xFF);
    /// ```
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// The next byte of the sequence.
    ///
    /// ```
    /// use sdrk_proto::whiten::Pn9;
    /// let mut pn9 = Pn9::new();
    /// assert_eq!((pn9.next_byte(), pn9.next_byte()), (0xFF, 0xE1));
    /// ```
    pub fn next_byte(&mut self) -> u8 {
        let b = pn9_byte(self.pos);
        self.pos = (self.pos + 1) % PN9_PERIOD;
        b
    }

    /// XORs `data` with the next `data.len()` bytes of the sequence (whitening and
    /// de-whitening are the same operation).
    ///
    /// ```
    /// use sdrk_proto::whiten::Pn9;
    /// let mut data = [0u8; 4];
    /// Pn9::new().apply(&mut data);
    /// assert_eq!(data, [0xFF, 0xE1, 0x1D, 0x9A]);
    /// ```
    pub fn apply(&mut self, data: &mut [u8]) {
        let mut rest = data;
        while !rest.is_empty() {
            let seq = PN9_BYTES.get(self.pos..).unwrap_or_default();
            let n = rest.len().min(seq.len());
            if n == 0 {
                // Unreachable: `pos` stays below `PN9_PERIOD`. Stop rather than loop forever.
                self.pos = 0;
                return;
            }
            let (head, tail) = rest.split_at_mut(n);
            for (d, s) in head.iter_mut().zip(seq) {
                *d ^= s;
            }
            self.pos = (self.pos + n) % PN9_PERIOD;
            rest = tail;
        }
    }
}
