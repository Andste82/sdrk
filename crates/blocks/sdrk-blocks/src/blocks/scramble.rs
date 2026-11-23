//! `scramble`: keyed phase scrambling and descrambling of a complex stream with a PRBS key.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::scramble::Scramble;
use sdrk_proto::prbs::{Poly, Prbs};

use super::{PRBS_OPTIONS, PRBS_POLYS, head, head_mut, prbs_choice, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::sample::BlockSample;
use crate::storage::{Plan, Storage};

/// The most samples one call of [`ScrambleBlock::process`] takes: the phase indices of a call
/// live in the struct (one byte per sample).
pub const SCRAMBLE_CHUNK: usize = 256;

/// The choice index of `prbs15`, the default key.
const KEY_DEFAULT: u16 = 5;

/// Description of the `scramble` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "scramble",
    summary: "Multiplies a complex stream by unit phasors from a PRBS key, or removes them \
        (inverse).",
    details: "Phase scrambling with the kernel scramble: sample n is multiplied by exp(j 2 pi \
        p[n] / 2^bits), where p[n] is made of the next bits bits of the PRBS key (the first bit \
        is the most significant). bits 1 gives the signs plus or minus 1, bits 2 the four \
        quadrants (the DVB-S2 physical-layer scrambling pattern), both exact; bits 3 to 8 add \
        a table phasor. With inverse the block multiplies by the conjugate phasors, so a \
        descrambler with the same key, seed and bits that starts at the same sample \
        returns the input: exactly at bits 1 and 2 (a Q15 component -1.0 returns as \
        -32767/32768), within 2 LSB (cq15) or about 140 dB (c32) at bits 3 to 8. The PRBS \
        position runs on across calls; reset restarts it from seed. A public PRBS key \
        gives spectral whitening and the separation of signals, not secrecy. Pitfall: there \
        is no key synchronisation; scrambler and descrambler must start at the same sample.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "The scrambled (or descrambled) samples, same type and rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "bits",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 8,
                default: Some(1),
            },
            typical: "1 for sign scrambling; 2 for the four quadrants as DVB-S2",
            mutability: Mutability::Build,
            description: "Phase bits per sample, 1 to 8: the phasors are the 2^bits roots of \
                unity, and each sample takes bits bits of the PRBS key.",
        },
        ParamDesc {
            name: "inverse",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false at the transmitter; true at the receiver",
            mutability: Mutability::Build,
            description: "Descramble: multiply by the conjugate phasors of the same key.",
        },
        ParamDesc {
            name: "key",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: PRBS_OPTIONS,
                default: KEY_DEFAULT,
            },
            typical: "prbs15; a shorter key repeats after 2^n - 1 bits",
            mutability: Mutability::Build,
            description: "The PRBS polynomial of the key: prbs5, prbs7, prbs9, prbs10, prbs11 \
                or prbs15 (period 2^n - 1 bits).",
        },
        ParamDesc {
            name: "seed",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 32767,
                default: Some(1),
            },
            typical: "1; any other value starts the key elsewhere",
            mutability: Mutability::Build,
            description: "Initial register of the PRBS key, 1 to 32767; only the low n bits \
                are used and they must not all be zero.",
        },
    ],
    required: &[],
};

/// Configuration of [`ScrambleBlock`].
///
/// ```
/// use sdrk_blocks::ScrambleConfig;
/// use sdrk_proto::prbs::Poly;
/// let rate = sdrk::Rate::hz(48_000)?;
/// let cfg = ScrambleConfig { bits: 2, ..ScrambleConfig::default() };
/// assert!(cfg.validate(rate).is_ok());
/// assert!(ScrambleConfig { bits: 9, ..cfg }.validate(rate).is_err());
/// // 32 masks to 0 in the 5-bit register of prbs5.
/// assert!(ScrambleConfig { key: Poly::Prbs5, seed: 32, ..cfg }.validate(rate).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrambleConfig {
    /// Phase bits per sample, 1 to 8.
    pub bits: u32,
    /// Descramble (the conjugate phasors).
    pub inverse: bool,
    /// The PRBS polynomial of the key (`Prbs5`, `Prbs7`, `Prbs9`, `Prbs10`, `Prbs11` or
    /// `Prbs15`).
    pub key: Poly,
    /// Initial register of the key, 1 to 32767; the low `n` bits must not all be zero.
    pub seed: u32,
}

impl Default for ScrambleConfig {
    fn default() -> Self {
        Self {
            bits: 1,
            inverse: false,
            key: Poly::Prbs15,
            seed: 1,
        }
    }
}

fn check(name: &str, value: ParamValue) -> Result<()> {
    DESCRIPTION
        .param(name)
        .ok_or(Error::UnknownParameter)?
        .check(&value)
}

impl ScrambleConfig {
    /// Checks the configuration (the sample rate does not matter): the ranges, a key of the
    /// choice, and a `seed` whose low `n` bits are not all zero (`InvalidParameter` of
    /// `seed`).
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        check("bits", ParamValue::I64(i64::from(self.bits)))?;
        check("seed", ParamValue::I64(i64::from(self.seed)))?;
        if prbs_choice(self.key).is_none() {
            return Err(Error::InvalidParameter {
                name: "key",
                reason: "the key must be prbs5, prbs7, prbs9, prbs10, prbs11 or prbs15",
            });
        }
        self.prbs().map(|_| ())
    }

    /// The PRBS generator of the key at its start.
    fn prbs(&self) -> Result<Prbs> {
        Prbs::new(self.key, self.seed).map_err(|_| Error::InvalidParameter {
            name: "seed",
            reason: "the low n bits of seed (n the degree of the key) are all zero",
        })
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "bits" => self.bits = u32::try_from(value.as_i64()?).unwrap_or(0),
            "inverse" => self.inverse = value.as_bool()?,
            "key" => {
                self.key = PRBS_POLYS
                    .get(usize::from(value.as_choice()?))
                    .copied()
                    .ok_or(Error::WrongParamType)?;
            }
            "seed" => self.seed = u32::try_from(value.as_i64()?).unwrap_or(0),
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "bits" => Ok(ParamValue::I64(i64::from(self.bits))),
            "inverse" => Ok(ParamValue::Bool(self.inverse)),
            "key" => Ok(ParamValue::Choice(
                prbs_choice(self.key).unwrap_or(KEY_DEFAULT),
            )),
            "seed" => Ok(ParamValue::I64(i64::from(self.seed))),
            _ => unknown(),
        }
    }
}

/// Keyed phase scrambler (or descrambler) for `C32` and `CQ15` around the kernel `scramble`.
///
/// Sample `n` (counted from the construction or the last `reset`) is multiplied by
/// `exp(j 2π p[n] / 2^bits)` (the conjugate for `inverse`), where `p[n]` is made of the PRBS
/// bits `n · bits` to `n · bits + bits − 1` of the key, the first one as the most significant
/// bit. The PRBS (`sdrk_proto::prbs`, polynomial `key`, initial register `seed`) runs on
/// across calls; `reset` restarts it. A call takes at most [`SCRAMBLE_CHUNK`] samples, limited
/// by the output room. Ratio 1, delay 0, no storage.
///
/// **Not encryption.** The key is a public PRBS: scrambling gives spectral whitening and the
/// separation of signals with different keys, not secrecy. A scrambler and a descrambler with
/// the same configuration (but `inverse`) agree when both start at the same sample; there is
/// no key synchronisation.
///
/// The indices are produced word-parallel (M49.1 §4): the key bits come up to 32 at a time from
/// a 64-bit window of the key sequence with one shift and one XOR (the recurrence of the
/// polynomial stretched by a power of two), and are split into the indices of four samples at a
/// time with constant shifts; the bits and indices that a call does not use are kept for the
/// next call, so the indices equal those of [`Prbs::next_bit`] and do not depend on how the
/// stream is cut.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, ScrambleBlock, ScrambleConfig};
/// use sdrk::{Rate, C32};
/// let rate = Rate::hz(48_000)?;
/// let tx_cfg = ScrambleConfig { bits: 2, ..ScrambleConfig::default() };
/// let rx_cfg = ScrambleConfig { inverse: true, ..tx_cfg };
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut tx = ScrambleBlock::<C32, _>::new(&tx_cfg, rate, &mut arena)?;
/// let mut rx = ScrambleBlock::<C32, _>::new(&rx_cfg, rate, &mut arena)?;
/// let x = [C32::new(0.5, -0.25); 8];
/// let (mut y, mut z) = ([C32::default(); 8], [C32::default(); 8]);
/// tx.process(&x, &mut y)?;
/// rx.process(&y, &mut z)?;
/// assert_eq!(z, x); // exact at bits 1 and 2
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ScrambleBlock<T: BlockSample, S: Storage> {
    config: ScrambleConfig,
    /// The index production at the start of the key (for `reset`).
    start: KeyIndices,
    /// The index production at the next sample.
    key: KeyIndices,
    /// The phase indices of the current call.
    phase: [u8; SCRAMBLE_CHUNK],
    _marker: PhantomData<fn() -> (T, S)>,
}

/// The samples of one index group: the indices are split from the key bits four at a time.
const GROUP: usize = 4;

/// Word-parallel production of the phase indices of a PRBS key (M49.1 §4).
///
/// The key sequence `s` of `x^n + x^t + 1` satisfies `s[i] = s[i − n] ^ s[i − t]` and therefore
/// (squaring the polynomial over GF(2)) also `s[i] = s[i − L] ^ s[i − T]` with `L = 2^j n`,
/// `T = 2^j t` for every `j`. With the last 64 key bits in a window (`L ≤ 64`), the next
/// `m ≤ T` bits depend only on bits already in the window, so they come from two shifts and one
/// XOR: `((win >> (L − m)) ^ (win >> (T − m))) & (2^m − 1)`, the first bit the most
/// significant. The window starts from the register of [`Prbs`] (the last `n` bits) extended
/// backwards with `s[i − n] = s[i] ^ s[i − t]`. The bits are buffered and split into the indices
/// of [`GROUP`] samples at a time (`GROUP · bits ≤ 32` bits, constant shifts per `bits`); the
/// indices of a group that a call does not use are kept for the next call.
#[derive(Debug, Clone, Copy)]
struct KeyIndices {
    /// The last 64 key bits, the newest in bit 0 (bit `k` is the bit `k + 1` steps back).
    win: u64,
    /// `L − m` (the shift of the long lag).
    shift_long: u32,
    /// `T − m` (the shift of the short lag).
    shift_short: u32,
    /// `m`, the key bits per step (at most 32).
    step: u32,
    /// `2^m − 1`.
    step_mask: u64,
    /// Key bits drawn but not split yet: the low `have` bits, the next one the highest.
    acc: u64,
    /// The number of valid bits in `acc` (below `GROUP · bits` after a group).
    have: u32,
    /// The indices of the last group; the last `pending` of them are not used yet.
    group: [u8; GROUP],
    /// The number of unused indices in `group` (0 to `GROUP − 1`).
    pending: usize,
}

impl KeyIndices {
    /// The index production at the register of `prbs`.
    fn new(prbs: &Prbs) -> Self {
        let poly = prbs.poly();
        let (n, t) = (poly.degree(), poly.second_tap());
        // Backwards: bit k (k ≥ n) is s[c − 1 − k] = s[c − 1 − k + n] ^ s[c − 1 − k + n − t].
        let mut win = u64::from(prbs.state());
        for k in n..64 {
            let bit = ((win >> (k - n)) ^ (win >> (k - n + t))) & 1;
            win |= bit << k;
        }
        // The largest power of two with L = 2^j n ≤ 64 (n ≥ 5, so j ≥ 2).
        let mut j = 0;
        while (n << (j + 1)) <= 64 {
            j += 1;
        }
        let (long, short) = (n << j, t << j);
        let step = short.min(32);
        Self {
            win,
            shift_long: long - step,
            shift_short: short - step,
            step,
            step_mask: (1u64 << step) - 1,
            acc: 0,
            have: 0,
            group: [0; GROUP],
            pending: 0,
        }
    }

    /// The next [`GROUP`] indices of `B` bits each.
    #[inline]
    fn next_group<const B: u32>(&mut self) -> [u8; GROUP] {
        let need = B * GROUP as u32;
        // Before a step `have < need ≤ 32` and `step ≤ 32`, so the bits fit the 64-bit buffer.
        while self.have < need {
            let new =
                ((self.win >> self.shift_long) ^ (self.win >> self.shift_short)) & self.step_mask;
            self.win = (self.win << self.step) | new;
            self.acc = (self.acc << self.step) | new;
            self.have += self.step;
        }
        self.have -= need;
        let bits = self.acc >> self.have;
        let mask = (1u64 << B) - 1;
        // `B ≤ 8`, so every index fits a byte.
        [
            ((bits >> (3 * B)) & mask) as u8,
            ((bits >> (2 * B)) & mask) as u8,
            ((bits >> B) & mask) as u8,
            (bits & mask) as u8,
        ]
    }

    /// Writes the next `out.len()` indices of `B` bits each.
    fn fill<const B: u32>(&mut self, out: &mut [u8]) {
        // The state in a local (the stores to `out` would otherwise reload it every group).
        let mut s = *self;
        let used = GROUP - s.pending;
        let mut taken = 0;
        for (o, &p) in out.iter_mut().zip(s.group.iter().skip(used)) {
            *o = p;
            taken += 1;
        }
        s.pending -= taken;
        if let Some(rest) = out.get_mut(taken..) {
            let mut chunks = rest.chunks_exact_mut(GROUP);
            for c in &mut chunks {
                c.copy_from_slice(&s.next_group::<B>());
            }
            let tail = chunks.into_remainder();
            if !tail.is_empty() {
                s.group = s.next_group::<B>();
                for (o, &p) in tail.iter_mut().zip(s.group.iter()) {
                    *o = p;
                }
                s.pending = GROUP - tail.len();
            }
        }
        *self = s;
    }
}

impl<T: BlockSample, S: Storage> ScrambleBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &ScrambleConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let start = KeyIndices::new(&config.prbs()?);
        Ok(Self {
            config: *config,
            key: start,
            start,
            phase: [0; SCRAMBLE_CHUNK],
            _marker: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero: the phase indices are in the struct).
    pub fn storage_bytes(config: &ScrambleConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// Output rate (equal to the input rate).
    pub fn rate_out(config: &ScrambleConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }
}

impl<T: BlockSample, S: Storage> Block for ScrambleBlock<T, S>
where
    Sdrk: Scramble<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len()).min(SCRAMBLE_CHUNK);
        let bits = self.config.bits;
        let phase = head_mut(&mut self.phase, n)?;
        match bits {
            1 => self.key.fill::<1>(phase),
            2 => self.key.fill::<2>(phase),
            3 => self.key.fill::<3>(phase),
            4 => self.key.fill::<4>(phase),
            5 => self.key.fill::<5>(phase),
            6 => self.key.fill::<6>(phase),
            7 => self.key.fill::<7>(phase),
            8 => self.key.fill::<8>(phase),
            _ => return Err(Error::Core(sdrk_core::Error::Internal("scramble bits"))),
        }
        <Sdrk as Scramble<T>>::scramble(
            &Sdrk,
            head(input, n)?,
            phase,
            bits,
            self.config.inverse,
            head_mut(output, n)?,
        )?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    fn reset(&mut self) {
        self.key = self.start;
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
