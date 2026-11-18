//! `bit_scramble`: the self-synchronizing (multiplicative) scrambler of a soft-bit stream, the
//! transmit side of `bit_descramble` (M44.2, the maintainer's decision on I44.2-6).
//!
//! The register and the polynomials are shared with `bit_descramble`
//! ([`BitScrambleConfig`], [`BitPolynomial`]).

use sdrk::{Progress, Rate};

use super::symbol_map::{CONSTELLATION_OPTIONS, SymbolMapConstellation};
use super::{unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::Storage;

/// The names of the `polynomial` choice, in index order.
pub(crate) const POLYNOMIAL_OPTIONS: &[&str] = &["x23_x18", "x23_x5", "x17_x12", "x58_x39"];

/// The `polynomial` parameter of `bit_scramble`, `bit_descramble` and the QAM frame composites.
pub(crate) const POLYNOMIAL_PARAM: ParamDesc = ParamDesc {
    name: "polynomial",
    unit: Unit::None,
    kind: ParamKind::Choice {
        options: POLYNOMIAL_OPTIONS,
        default: 0,
    },
    typical: "x23_x18 (ITU-T V.32 to V.34); x17_x12 for G3RUH packet radio",
    mutability: Mutability::Build,
    description: "The polynomial of the self-synchronizing scrambler, 1 + x^-b + x^-a: x23_x18 \
        (1 + x^-18 + x^-23, the call-mode scrambler of ITU-T V.32, V.32bis and V.34), x23_x5 \
        (1 + x^-5 + x^-23, their answer mode), x17_x12 (1 + x^-12 + x^-17, the G3RUH 9600 bd \
        packet radio scrambler) or x58_x39 (1 + x^-39 + x^-58, the 64b/66b scrambler of IEEE \
        802.3 clause 49). The descrambler locks after a bits per symbol bits.",
};

/// The `constellation` parameter of `bit_scramble` and `bit_descramble`: the symbols whose bits
/// the register runs over.
pub(crate) const SCRAMBLE_CONSTELLATION_PARAM: ParamDesc = ParamDesc {
    name: "constellation",
    unit: Unit::None,
    kind: ParamKind::Choice {
        options: CONSTELLATION_OPTIONS,
        default: 0,
    },
    typical: "bpsk for a plain bit stream; the constellation of the modulator after it",
    mutability: Mutability::Build,
    description: "The constellation whose symbols carry the bits (the names of symbol_map: \
        bpsk 1 bit per symbol, qpsk 2, 8psk 3, qam16 4, qam64 6, qam256 8, qam1024 10). The \
        delays of the polynomial count symbols: bit k of a symbol is scrambled with bit k of \
        the symbols a and b before it, the bit polynomial 1 + x^-(b m) + x^-(a m) for m bits \
        per symbol, so that a quarter turn of a square constellation (which swaps and inverts \
        bits within a symbol) commutes with the scrambler. bpsk is the plain bit scrambler.",
};

/// The names of the `scrambler` choice of the QAM frame composites, in index order.
const SCRAMBLER_OPTIONS: &[&str] = &["multiplicative", "none"];

/// The `scrambler` parameter of `qam_frame_encode` and `qam_frame_decode`: whether the composite
/// holds the stage `bit_scramble` / `bit_descramble` (a parameter of the structure, D1823).
pub(crate) const SCRAMBLER_PARAM: ParamDesc = ParamDesc {
    name: "scrambler",
    unit: Unit::None,
    kind: ParamKind::Choice {
        options: SCRAMBLER_OPTIONS,
        default: 0,
    },
    typical: "multiplicative; none for a link to a receiver without the descrambler",
    mutability: Mutability::Build,
    description: "The scrambler of the bits on the air: multiplicative (bit_scramble in the \
        transmitter, bit_descramble in the receiver, with polynomial and the constellation; the \
        whole bit stream, frames and idle signal, looks random on the air, so the blind \
        equalizer of the dense constellations converges) or none (the stage is left out). Both \
        ends of a link must use the same.",
};

/// The `scrambler` choice of the QAM frame composites: whether they hold the stage
/// `bit_scramble` / `bit_descramble`.
///
/// ```
/// use sdrk_blocks::Scrambler;
/// assert_eq!(Scrambler::default(), Scrambler::Multiplicative);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Scrambler {
    /// The self-synchronizing scrambler of `bit_scramble` (the default).
    #[default]
    Multiplicative,
    /// No scrambler: the stage is left out (`none`).
    Off,
}

impl Scrambler {
    pub(crate) const fn index(self) -> u16 {
        match self {
            Self::Multiplicative => 0,
            Self::Off => 1,
        }
    }

    pub(crate) const fn from_index(i: u16) -> Self {
        if i == 1 {
            Self::Off
        } else {
            Self::Multiplicative
        }
    }
}

/// Description of the `bit_scramble` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "bit_scramble",
    summary: "Scrambles a soft-bit stream with a self-synchronizing (multiplicative) scrambler \
        so that it looks random on the air.",
    details: "Every output bit is the input bit XOR the output bits a and b symbols earlier \
        (polynomial, constellation; the register starts at zero and after a reset), so long runs \
        and short periods of the data (preambles, sync words, idle patterns) become a \
        pseudo-random stream, as a blind equalizer and a timing loop of the receiver need. The \
        bits are soft bits: above 0 is a 1; the output keeps the magnitude of the input and \
        takes the sign of the scrambled bit (a 0 or NaN input stays 0 or NaN and counts as a \
        0). bit_descramble undoes it without any frame reference: it locks by itself after a \
        times the bits per symbol of correct bits, and a wrong bit on the air gives 3 wrong \
        bits after it (at the bit and a and b symbols later). Useful for any bit stream; the \
        transmit side of qam_frame_encode with its scrambler on. The rate does not change.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft bits, one sample per bit; above 0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The scrambled soft bits, one per input bit; above 0 is a 1.",
    }],
    params: &[POLYNOMIAL_PARAM, SCRAMBLE_CONSTELLATION_PARAM],
    required: &[],
};

/// The polynomial of a self-synchronizing scrambler, `1 + x^-b + x^-a` (parameter
/// `polynomial` of `bit_scramble` and `bit_descramble`).
///
/// ```
/// use sdrk_blocks::BitPolynomial;
/// assert_eq!(BitPolynomial::default(), BitPolynomial::X23X18);
/// assert_eq!(BitPolynomial::X23X18.taps(), (23, 18));
/// assert_eq!(BitPolynomial::X58X39.taps(), (58, 39));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum BitPolynomial {
    /// `1 + x^-18 + x^-23`: the call-mode scrambler of ITU-T V.32, V.32bis and V.34 (the
    /// default).
    #[default]
    X23X18,
    /// `1 + x^-5 + x^-23`: the answer-mode scrambler of ITU-T V.32, V.32bis and V.34.
    X23X5,
    /// `1 + x^-12 + x^-17`: the scrambler of the G3RUH 9600 bd packet radio modem.
    X17X12,
    /// `1 + x^-39 + x^-58`: the scrambler of the 64b/66b code of IEEE 802.3 clause 49.
    X58X39,
}

impl BitPolynomial {
    /// The delays `(a, b)` of `1 + x^-b + x^-a`, `a > b`.
    pub const fn taps(self) -> (usize, usize) {
        match self {
            Self::X23X18 => (23, 18),
            Self::X23X5 => (23, 5),
            Self::X17X12 => (17, 12),
            Self::X58X39 => (58, 39),
        }
    }

    pub(crate) const fn index(self) -> u16 {
        match self {
            Self::X23X18 => 0,
            Self::X23X5 => 1,
            Self::X17X12 => 2,
            Self::X58X39 => 3,
        }
    }

    pub(crate) const fn from_index(i: u16) -> Self {
        match i {
            1 => Self::X23X5,
            2 => Self::X17X12,
            3 => Self::X58X39,
            _ => Self::X23X18,
        }
    }
}

/// Configuration of [`BitScrambleBlock`] and [`BitDescrambleBlock`](crate::BitDescrambleBlock).
///
/// ```
/// use sdrk_blocks::{BitPolynomial, BitScrambleConfig, ParamValue, SymbolMapConstellation};
/// let mut cfg = BitScrambleConfig::default();
/// assert_eq!(cfg.delays(), (23, 18));
/// cfg.set("constellation", ParamValue::Choice(5))?; // qam256: 8 bits per symbol
/// assert_eq!(cfg.constellation, SymbolMapConstellation::Qam256);
/// assert_eq!(cfg.delays(), (184, 144));
/// cfg.set("polynomial", ParamValue::Choice(2))?;
/// assert_eq!(cfg.polynomial, BitPolynomial::X17X12);
/// assert_eq!(cfg.get("polynomial")?, ParamValue::Choice(2));
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitScrambleConfig {
    /// The polynomial (default `x23_x18`).
    pub polynomial: BitPolynomial,
    /// The constellation whose symbols carry the bits: the delays of the polynomial count its
    /// symbols (default `bpsk`, one bit per symbol: the plain bit scrambler).
    pub constellation: SymbolMapConstellation,
}

impl BitScrambleConfig {
    /// The delays of the register in bits: `(a m, b m)` for the polynomial `1 + x^-b + x^-a`
    /// and `m` bits per symbol of the constellation.
    pub const fn delays(&self) -> (usize, usize) {
        let (a, b) = self.polynomial.taps();
        let m = self.constellation.bits_per_symbol();
        (a * m, b * m)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "polynomial" => self.polynomial = BitPolynomial::from_index(value.as_choice()?),
            "constellation" => {
                self.constellation = SymbolMapConstellation::from_index(value.as_choice()?);
            }
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "polynomial" => Ok(ParamValue::Choice(self.polynomial.index())),
            "constellation" => Ok(ParamValue::Choice(self.constellation.index())),
            _ => unknown(),
        }
    }
}

/// Words of the ring of [`Register`]: 1024 bits, more than the longest delay (58 symbols of 10
/// bits) plus one step of 64 bits.
const RING_WORDS: usize = 16;
/// Bits of the ring.
const RING_BITS: usize = RING_WORDS * 64;
/// Bits per step of the register.
const STEP: usize = 64;

/// The bits of the register, newest last, as a ring of `RING_BITS` bits: bit `t` of the stream
/// is bit `t % 64` of word `(t / 64) % RING_WORDS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Register {
    ring: [u64; RING_WORDS],
    /// The ring position of the next bit.
    pos: u16,
    /// The delays `a m` and `b m` in bits.
    a: u16,
    b: u16,
}

/// The `k` lowest bits set (`k` 1 to 64).
#[inline]
fn low_mask(k: usize) -> u64 {
    u64::MAX >> (64 - k)
}

/// The hard bits of up to 64 soft bits: bit `j` is `x[j] > 0`, and the mask of the soft bits
/// that are not 0 or NaN (`x[j] > 0 || x[j] < 0`).
#[inline]
fn hard_bits(x: &[f32]) -> (u64, u64) {
    let (mut gt, mut lt) = (0u64, 0u64);
    for (j, &v) in x.iter().enumerate() {
        gt |= u64::from(v > 0.0) << j;
        lt |= u64::from(v < 0.0) << j;
    }
    (gt, gt | lt)
}

/// `y[j] = x[j]` with its sign bit flipped where bit `j` of `p` is set.
#[inline]
fn flip_signs(p: u64, x: &[f32], y: &mut [f32]) {
    for (j, (y, &x)) in y.iter_mut().zip(x).enumerate() {
        let s = ((p >> j) as u32 & 1) << 31;
        *y = f32::from_bits(x.to_bits() ^ s);
    }
}

impl Register {
    /// The zero register of the delays of `config`.
    pub(crate) fn new(config: BitScrambleConfig) -> Self {
        let (a, b) = config.delays();
        Self {
            ring: [0; RING_WORDS],
            pos: 0,
            // At most 580 (58 symbols of 10 bits).
            a: a as u16,
            b: b as u16,
        }
    }

    /// Clears the bits (the state after construction).
    pub(crate) fn reset(&mut self) {
        self.ring = [0; RING_WORDS];
        self.pos = 0;
    }

    fn word(&self, w: usize) -> u64 {
        self.ring.get(w % RING_WORDS).copied().unwrap_or(0)
    }

    /// The `k` (1 to 64) bits from ring position `p` on, the first in bit 0.
    #[inline]
    fn get(&self, at: usize, len: usize) -> u64 {
        let (word, bit) = (at / 64, at % 64);
        // `<< 1 << (63 - bit)` is `<< (64 - bit)` for `bit` above 0 and gives 0 for `bit` 0.
        let bits = (self.word(word) >> bit) | ((self.word(word + 1) << 1) << (63 - bit));
        bits & low_mask(len)
    }

    /// Writes the `k` (1 to 64) bits of `v` (above them 0) at ring position `p`.
    #[inline]
    fn put(&mut self, at: usize, len: usize, bits: u64) {
        let (word, bit) = (at / 64, at % 64);
        let mask = low_mask(len);
        if let Some(x) = self.ring.get_mut(word % RING_WORDS) {
            *x = (*x & !(mask << bit)) | (bits << bit);
        }
        // `>> 1 >> (63 - bit)` is `>> (64 - bit)` for `bit` above 0 and gives 0 for `bit` 0.
        if let Some(x) = self.ring.get_mut((word + 1) % RING_WORDS) {
            *x = (*x & !((mask >> 1) >> (63 - bit))) | ((bits >> 1) >> (63 - bit));
        }
    }

    /// The ring position `back` bits before the next bit.
    #[inline]
    fn back(&self, back: u16) -> usize {
        (usize::from(self.pos) + RING_BITS - usize::from(back)) % RING_BITS
    }

    /// Moves the position on by `k` bits.
    #[inline]
    fn advance(&mut self, k: usize) {
        self.pos = ((usize::from(self.pos) + k) % RING_BITS) as u16;
    }

    /// Scrambles up to `min(b m, 64)` soft bits: the parity of the output bits `a m` and
    /// `b m` before each bit flips its sign, and the output bits go into the register.
    #[inline]
    fn scramble_step(&mut self, x: &[f32], y: &mut [f32]) {
        let k = x.len();
        let p = self.get(self.back(self.a), k) ^ self.get(self.back(self.b), k);
        let (gt, nonzero) = hard_bits(x);
        // The output bit: the input bit XOR the parity, 0 for a 0 or NaN input.
        self.put(usize::from(self.pos), k, (gt ^ p) & nonzero);
        self.advance(k);
        flip_signs(p, x, y);
    }

    /// Descrambles up to 64 soft bits: the input bits go into the register first (a delay
    /// below 64 bits reaches into the same step), then the parity of the input bits `a m` and
    /// `b m` before each bit flips its sign.
    #[inline]
    fn descramble_step(&mut self, x: &[f32], y: &mut [f32]) {
        let k = x.len();
        let (gt, _) = hard_bits(x);
        let at = usize::from(self.pos);
        self.put(at, k, gt);
        let p = self.get(self.back(self.a), k) ^ self.get(self.back(self.b), k);
        self.advance(k);
        flip_signs(p, x, y);
    }

    /// Scrambles `x` into `y` (the same length).
    pub(crate) fn scramble(&mut self, x: &[f32], y: &mut [f32]) {
        let step = usize::from(self.b).min(STEP);
        for (x, y) in x.chunks(step).zip(y.chunks_mut(step)) {
            self.scramble_step(x, y);
        }
    }

    /// Descrambles `x` into `y` (the same length).
    pub(crate) fn descramble(&mut self, x: &[f32], y: &mut [f32]) {
        for (x, y) in x.chunks(STEP).zip(y.chunks_mut(STEP)) {
            self.descramble_step(x, y);
        }
    }

    /// Scrambles `data` in place.
    pub(crate) fn scramble_in_place(&mut self, data: &mut [f32]) {
        let step = usize::from(self.b).min(STEP);
        let mut x = [0.0f32; STEP];
        for y in data.chunks_mut(step) {
            let x = x.get_mut(..y.len()).unwrap_or_default();
            x.copy_from_slice(y);
            self.scramble_step(x, y);
        }
    }

    /// Descrambles `data` in place.
    pub(crate) fn descramble_in_place(&mut self, data: &mut [f32]) {
        let mut x = [0.0f32; STEP];
        for y in data.chunks_mut(STEP) {
            let x = x.get_mut(..y.len()).unwrap_or_default();
            x.copy_from_slice(y);
            self.descramble_step(x, y);
        }
    }
}

/// The self-synchronizing scrambler: `f32` soft bits in, `f32` soft bits out, one per bit.
///
/// With the delays `A = a m` and `B = b m` in bits ([`BitScrambleConfig::delays`]: the polynomial
/// `1 + x^-b + x^-a` over symbols of `m` bits), output bit `n` is `s[n] = x[n] ^ s[n - A] ^
/// s[n - B]` with `x[n]` the input bit (`> 0` is a 1) and `s` the output bits before it (0 before
/// the start and after [`reset`](Block::reset)); the output sample is the input sample with its
/// sign flipped where the parity `s[n - A] ^ s[n - B]` is 1, so it keeps the magnitude (a 0 or
/// NaN input stays 0 or NaN and counts as a 0 bit in the register). The register runs over up to
/// 64 bits per step (at most `B` for the feedback), and the result does not depend on how the
/// input is cut. A call takes `min(input.len(), output.len())` bits. Timing: constant ratio 1,
/// delay 0. Storage: none (the register of 1024 bits is in the block).
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, BitDescrambleBlock, BitScrambleBlock, BitScrambleConfig, Block};
/// let cfg = BitScrambleConfig::default();
/// let rate = Rate::hz(9600)?;
/// let mut tx = BitScrambleBlock::new(&cfg, rate, &mut ArenaStorage::new(&mut []))?;
/// let mut rx = BitDescrambleBlock::new(&cfg, rate, &mut ArenaStorage::new(&mut []))?;
/// // a long run of ones becomes a pseudo-random stream
/// let bits = [1.0f32; 100];
/// let mut air = [0.0f32; 100];
/// tx.process(&bits, &mut air)?;
/// assert!(air.iter().any(|&v| v < 0.0));
/// let mut back = [0.0f32; 100];
/// rx.process(&air, &mut back)?;
/// assert_eq!(back, bits);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct BitScrambleBlock<S: Storage> {
    reg: Register,
    config: BitScrambleConfig,
    _storage: core::marker::PhantomData<fn() -> S>,
}

impl<S: Storage> core::fmt::Debug for BitScrambleBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BitScrambleBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> BitScrambleBlock<S> {
    /// Creates the block (it needs no storage).
    ///
    /// # Errors
    /// None today; the signature is that of every block.
    pub fn new(config: &BitScrambleConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Ok(Self {
            reg: Register::new(*config),
            config: *config,
            _storage: core::marker::PhantomData,
        })
    }

    /// Exact bytes of storage the block needs: 0.
    ///
    /// # Errors
    /// None today.
    pub fn storage_bytes(_config: &BitScrambleConfig, _rate_in: Rate) -> Result<usize> {
        Ok(0)
    }

    /// Output rate: the input rate.
    ///
    /// # Errors
    /// None today.
    pub fn rate_out(_config: &BitScrambleConfig, rate_in: Rate) -> Result<Rate> {
        Ok(rate_in)
    }

    /// The configuration of the block.
    pub fn config(&self) -> BitScrambleConfig {
        self.config
    }

    /// Scrambles `data` in place: the same as [`Block::process`] with `data` as input and output
    /// (the call of `qam_frame_encode` on the bits between its stages).
    ///
    /// ```
    /// use sdrk::Rate;
    /// use sdrk_blocks::{ArenaStorage, BitScrambleBlock, BitScrambleConfig, Block};
    /// let cfg = BitScrambleConfig::default();
    /// let rate = Rate::hz(9600)?;
    /// let mut a = BitScrambleBlock::new(&cfg, rate, &mut ArenaStorage::new(&mut []))?;
    /// let mut b = BitScrambleBlock::new(&cfg, rate, &mut ArenaStorage::new(&mut []))?;
    /// let x: Vec<f32> = (0..50).map(|i| if i % 3 == 0 { 1.0 } else { -0.5 }).collect();
    /// let mut y = vec![0.0f32; 50];
    /// a.process(&x, &mut y)?;
    /// let mut z = x.clone();
    /// b.process_in_place(&mut z);
    /// assert_eq!(y, z);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process_in_place(&mut self, data: &mut [f32]) {
        self.reg.scramble_in_place(data);
    }
}

impl<S: Storage> Block for BitScrambleBlock<S> {
    type In = f32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        self.reg
            .scramble(super::head(input, n)?, super::head_mut(output, n)?);
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {
        self.reg.reset();
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(_) => Err(Error::ReadOnlyParameter),
            None => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
