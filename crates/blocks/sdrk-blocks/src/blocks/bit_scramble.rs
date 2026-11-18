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
use crate::storage::{Plan, Storage};

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

/// The sign bit of the hard bit of a soft bit: `0x8000_0000` for a 1 (above 0), 0 for a 0
/// (0, below 0 or NaN).
#[inline]
fn sign_of(v: f32) -> u32 {
    u32::from(v > 0.0) << 31
}

/// `x` with its sign bit flipped by the parity of the bits `a` and `b` (as sign bits).
#[inline]
fn flip(x: f32, a: f32, b: f32) -> f32 {
    f32::from_bits(x.to_bits() ^ sign_of(a) ^ sign_of(b))
}

/// Soft bits per piece of the in-place descrambler (a copy on the stack, 1 KiB).
const PIECE: usize = 256;

/// The register of the scrambler and the descrambler: the last `A` soft bits (the output bits of
/// the scrambler, the input bits of the descrambler) in a ring from the storage, the oldest at
/// `pos`, zero at the start and after a reset. A call reads the ring for its first `A` bits and
/// its own bits after them, so that the bulk of a call is one element-wise loop over slices
/// (the hard bits as sign bits: a comparison, two `xor` per bit, D1532).
pub(crate) struct Register<B> {
    ring: B,
    pos: u16,
    /// The delays `A = a m` and `B = b m` in bits (`A` is the length of the ring).
    a: u16,
    b: u16,
}

impl<B: AsRef<[f32]> + AsMut<[f32]>> Register<B> {
    /// The zero register of the delays of `config` over `ring` (`A` values).
    pub(crate) fn new(config: BitScrambleConfig, mut ring: B) -> Self {
        let (a, b) = config.delays();
        ring.as_mut().fill(0.0);
        Self {
            ring,
            pos: 0,
            // At most 580 (58 symbols of 10 bits).
            a: a as u16,
            b: b as u16,
        }
    }

    /// Clears the bits (the state after construction).
    pub(crate) fn reset(&mut self) {
        self.ring.as_mut().fill(0.0);
        self.pos = 0;
    }

    /// The positions in the ring of the bits `A` and `B` before the first bit of a call.
    #[inline]
    fn starts(&self) -> (usize, usize) {
        let (a, b) = (usize::from(self.a), usize::from(self.b));
        let pos = usize::from(self.pos);
        (pos, (pos + a - b) % a)
    }

    /// The value at ring position `at`, and the position after it.
    #[inline]
    fn next(&self, at: usize) -> (f32, usize) {
        let ring = self.ring.as_ref();
        let v = ring.get(at).copied().unwrap_or(0.0);
        let at = at + 1;
        (v, if at == ring.len() { 0 } else { at })
    }

    /// Puts the bits of a call, `bits`, into the ring (its last `A` of them).
    fn remember(&mut self, bits: &[f32]) {
        let a = usize::from(self.a);
        let ring = self.ring.as_mut();
        if bits.len() >= a {
            if let Some(last) = bits.get(bits.len() - a..) {
                ring.copy_from_slice(last);
            }
            self.pos = 0;
            return;
        }
        let pos = usize::from(self.pos);
        let (first, second) = bits.split_at(bits.len().min(a - pos));
        if let Some(r) = ring.get_mut(pos..pos + first.len()) {
            r.copy_from_slice(first);
        }
        if let Some(r) = ring.get_mut(..second.len()) {
            r.copy_from_slice(second);
        }
        self.pos = ((pos + bits.len()) % a) as u16;
    }

    /// Scrambles `data` in place: `s[n] = x[n] ^ s[n - A] ^ s[n - B]`.
    pub(crate) fn scramble_in_place(&mut self, data: &mut [f32]) {
        self.scramble_with::<true>(&[], data);
    }

    /// Scrambles `x` into `y` (the same length), without a copy of `x` into `y` first (a copy of
    /// a whole call is a `memcpy`, D1532).
    pub(crate) fn scramble(&mut self, x: &[f32], y: &mut [f32]) {
        self.scramble_with::<false>(x, y);
    }

    /// The scrambler: the input is `out` itself for `IN_PLACE`, else `input` (of the length of
    /// `out`).
    /// After the first `A` bits it runs in pieces of `B` bits rounded down to whole vectors of 8
    /// (`B` below 8: `B`), each one element-wise loop that reads the output of the pieces before
    /// it (the feedback reaches `B` bits back).
    #[inline]
    fn scramble_with<const IN_PLACE: bool>(&mut self, input: &[f32], out: &mut [f32]) {
        let (da, db) = (usize::from(self.a), usize::from(self.b));
        let head = out.len().min(da);
        let (mut at_a, mut at_b) = self.starts();
        for i in 0..head {
            let (va, next_a) = self.next(at_a);
            at_a = next_a;
            let vb = if i >= db {
                out.get(i - db).copied().unwrap_or(0.0)
            } else {
                let (v, next_b) = self.next(at_b);
                at_b = next_b;
                v
            };
            let v = if IN_PLACE { out.get(i) } else { input.get(i) };
            let v = v.copied().unwrap_or(0.0);
            if let Some(o) = out.get_mut(i) {
                *o = flip(v, va, vb);
            }
        }
        let piece = if db >= 8 { db & !7 } else { db };
        let mut k = head;
        while k < out.len() {
            let len = piece.min(out.len() - k);
            let (done, rest) = out.split_at_mut(k);
            let (Some(sa), Some(sb), Some(dst)) = (
                done.get(k - da..k - da + len),
                done.get(k - db..k - db + len),
                rest.get_mut(..len),
            ) else {
                return;
            };
            if IN_PLACE {
                for ((o, &va), &vb) in dst.iter_mut().zip(sa).zip(sb) {
                    *o = flip(*o, va, vb);
                }
            } else if let Some(xs) = input.get(k..k + len) {
                for (((o, &v), &va), &vb) in dst.iter_mut().zip(xs).zip(sa).zip(sb) {
                    *o = flip(v, va, vb);
                }
            }
            k += len;
        }
        // `out` holds the output bits, the register of the scrambler.
        let bits: &[f32] = out;
        self.remember(bits);
    }

    /// Descrambles `x` into `y` (the same length): `y[n] = x[n] ^ x[n - A] ^ x[n - B]`, one
    /// element-wise loop after the first `A` bits.
    pub(crate) fn descramble(&mut self, input: &[f32], out: &mut [f32]) {
        let (da, db) = (usize::from(self.a), usize::from(self.b));
        let head = input.len().min(da);
        let (mut at_a, mut at_b) = self.starts();
        for (i, (o, &v)) in out.iter_mut().zip(input).take(head).enumerate() {
            let (va, next_a) = self.next(at_a);
            at_a = next_a;
            let vb = if i >= db {
                input.get(i - db).copied().unwrap_or(0.0)
            } else {
                let (w, next_b) = self.next(at_b);
                at_b = next_b;
                w
            };
            *o = flip(v, va, vb);
        }
        if let (Some(dst), Some(xs), Some(sb)) =
            (out.get_mut(da..), input.get(da..), input.get(da - db..))
        {
            for (((o, &v), &va), &vb) in dst.iter_mut().zip(xs).zip(input).zip(sb) {
                *o = flip(v, va, vb);
            }
        }
        self.remember(input);
    }

    /// Descrambles `data` in place, through a copy of each piece of [`PIECE`] bits.
    pub(crate) fn descramble_in_place(&mut self, data: &mut [f32]) {
        let mut copy = [0.0f32; PIECE];
        for y in data.chunks_mut(PIECE) {
            let x = copy.get_mut(..y.len()).unwrap_or_default();
            x.copy_from_slice(y);
            self.descramble(x, y);
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
/// NaN input stays 0 or NaN and counts as a 0 bit in the register). The register is a ring of
/// the last `A` output bits as soft bits; after its first `A` bits a call runs in pieces of `B`
/// bits, element-wise (the feedback reaches `B` bits back), and the result does not depend on how
/// the input is cut. A call takes `min(input.len(), output.len())` bits. Timing: constant ratio
/// 1, delay 0. Storage: the ring, `A` `f32` (at most 580).
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, BitDescrambleBlock, BitScrambleBlock, BitScrambleConfig, Block};
/// let cfg = BitScrambleConfig::default();
/// let rate = Rate::hz(9600)?;
/// let mut mem = [0u8; 4096];
/// let mut st = ArenaStorage::new(&mut mem);
/// let mut tx = BitScrambleBlock::new(&cfg, rate, &mut st)?;
/// let mut rx = BitDescrambleBlock::new(&cfg, rate, &mut st)?;
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
    reg: Register<S::Buf<f32>>,
    config: BitScrambleConfig,
}

impl<S: Storage> core::fmt::Debug for BitScrambleBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BitScrambleBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> BitScrambleBlock<S> {
    /// Creates the block; the ring of the register comes from `storage`.
    ///
    /// # Errors
    /// `Storage` when the storage is too small.
    pub fn new(config: &BitScrambleConfig, _rate_in: Rate, storage: &mut S) -> Result<Self> {
        Ok(Self {
            reg: Register::new(*config, storage.alloc::<f32>(config.delays().0)?),
            config: *config,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more): the ring of `A` `f32`.
    ///
    /// # Errors
    /// None today.
    pub fn storage_bytes(config: &BitScrambleConfig, _rate_in: Rate) -> Result<usize> {
        let mut plan = Plan::default();
        plan.add::<f32>(config.delays().0)?;
        Ok(plan.bytes())
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
    /// let mut mem = [0u8; 4096];
    /// let mut st = ArenaStorage::new(&mut mem);
    /// let mut a = BitScrambleBlock::new(&cfg, rate, &mut st)?;
    /// let mut b = BitScrambleBlock::new(&cfg, rate, &mut st)?;
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
