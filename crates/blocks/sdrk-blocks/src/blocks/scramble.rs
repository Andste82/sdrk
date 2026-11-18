//! `scramble`: keyed phase scrambling and descrambling of a complex stream with a PRBS key.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::scramble::Scramble;
use sdrk_proto::prbs::{Poly, Prbs};

use super::{head, head_mut, unit_timing, unknown};
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

/// The names of the PRBS keys, in the order of the `key` choice.
const KEY_OPTIONS: &[&str] = &["prbs5", "prbs7", "prbs9", "prbs10", "prbs11", "prbs15"];

/// The polynomials of [`KEY_OPTIONS`], in the same order.
const KEY_POLYS: [Poly; 6] = [
    Poly::Prbs5,
    Poly::Prbs7,
    Poly::Prbs9,
    Poly::Prbs10,
    Poly::Prbs11,
    Poly::Prbs15,
];

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
        descrambler with the same key, key_seed and bits that starts at the same sample \
        returns the input: exactly at bits 1 and 2 (a Q15 component -1.0 returns as \
        -32767/32768), within 2 LSB (cq15) or about 140 dB (c32) at bits 3 to 8. The PRBS \
        position runs on across calls; reset restarts it from key_seed. A public PRBS key \
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
                options: KEY_OPTIONS,
                default: KEY_DEFAULT,
            },
            typical: "prbs15; a shorter key repeats after 2^n - 1 bits",
            mutability: Mutability::Build,
            description: "The PRBS polynomial of the key: prbs5, prbs7, prbs9, prbs10, prbs11 \
                or prbs15 (period 2^n - 1 bits).",
        },
        ParamDesc {
            name: "key_seed",
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
/// assert!(ScrambleConfig { key: Poly::Prbs5, key_seed: 32, ..cfg }.validate(rate).is_err());
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
    pub key_seed: u32,
}

impl Default for ScrambleConfig {
    fn default() -> Self {
        Self {
            bits: 1,
            inverse: false,
            key: Poly::Prbs15,
            key_seed: 1,
        }
    }
}

fn check(name: &str, value: ParamValue) -> Result<()> {
    DESCRIPTION
        .param(name)
        .ok_or(Error::UnknownParameter)?
        .check(&value)
}

/// The choice index of a key polynomial.
fn key_index(poly: Poly) -> Option<u16> {
    KEY_POLYS
        .iter()
        .position(|&p| p == poly)
        .and_then(|i| u16::try_from(i).ok())
}

impl ScrambleConfig {
    /// Checks the configuration (the sample rate does not matter): the ranges, a key of the
    /// choice, and a `key_seed` whose low `n` bits are not all zero (`InvalidParameter` of
    /// `key_seed`).
    pub fn validate(&self, _rate_in: Rate) -> Result<()> {
        check("bits", ParamValue::I64(i64::from(self.bits)))?;
        check("key_seed", ParamValue::I64(i64::from(self.key_seed)))?;
        if key_index(self.key).is_none() {
            return Err(Error::InvalidParameter {
                name: "key",
                reason: "the key must be prbs5, prbs7, prbs9, prbs10, prbs11 or prbs15",
            });
        }
        self.prbs().map(|_| ())
    }

    /// The PRBS generator of the key at its start.
    fn prbs(&self) -> Result<Prbs> {
        Prbs::new(self.key, self.key_seed).map_err(|_| Error::InvalidParameter {
            name: "key_seed",
            reason: "the low n bits of key_seed (n the degree of the key) are all zero",
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
                self.key = KEY_POLYS
                    .get(usize::from(value.as_choice()?))
                    .copied()
                    .ok_or(Error::WrongParamType)?;
            }
            "key_seed" => self.key_seed = u32::try_from(value.as_i64()?).unwrap_or(0),
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
                key_index(self.key).unwrap_or(KEY_DEFAULT),
            )),
            "key_seed" => Ok(ParamValue::I64(i64::from(self.key_seed))),
            _ => unknown(),
        }
    }
}

/// Keyed phase scrambler (or descrambler) for `C32` and `CQ15` around the kernel `scramble`.
///
/// Sample `n` (counted from the construction or the last `reset`) is multiplied by
/// `exp(j 2π p[n] / 2^bits)` (the conjugate for `inverse`), where `p[n]` is made of the PRBS
/// bits `n · bits` to `n · bits + bits − 1` of the key, the first one as the most significant
/// bit. The PRBS (`sdrk_proto::prbs`, polynomial `key`, initial register `key_seed`) runs on
/// across calls; `reset` restarts it. A call takes at most [`SCRAMBLE_CHUNK`] samples, limited
/// by the output room. Ratio 1, delay 0, no storage.
///
/// **Not encryption.** The key is a public PRBS: scrambling gives spectral whitening and the
/// separation of signals with different keys, not secrecy. A scrambler and a descrambler with
/// the same configuration (but `inverse`) agree when both start at the same sample; there is
/// no key synchronisation.
///
/// The key is drawn 32 bits at a time ([`Prbs::next_bits`], which steps the register up to `t`
/// bits at a time, `t` the second tap of the polynomial, and gives the bits of
/// [`Prbs::next_bit`]); the bits of a word that a call does not use are kept for the next call,
/// so the indices do not depend on how the stream is cut.
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
    /// The key at its start (for `reset`).
    start: Prbs,
    /// The key at the next sample.
    prbs: Prbs,
    /// Key bits drawn but not used yet: the low `have` bits of `acc`, the next one the highest.
    acc: u64,
    /// The number of valid bits in `acc` (below `bits` after a call).
    have: u32,
    /// The phase indices of the current call.
    phase: [u8; SCRAMBLE_CHUNK],
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T: BlockSample, S: Storage> ScrambleBlock<T, S> {
    /// Creates the block; it needs no storage.
    pub fn new(config: &ScrambleConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let start = config.prbs()?;
        Ok(Self {
            config: *config,
            prbs: start.clone(),
            start,
            acc: 0,
            have: 0,
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
        let mask = (1u64 << bits) - 1;
        // The buffer in locals (the stores to `phase` would otherwise reload it every sample).
        let (mut acc, mut have) = (self.acc, self.have);
        for p in phase.iter_mut() {
            if have < bits {
                // At most `bits − 1 ≤ 7` bits are left, so 32 more fit the 64-bit buffer.
                acc = (acc << 32) | u64::from(self.prbs.next_bits(32));
                have += 32;
            }
            have -= bits;
            // `bits` is at most 8, so the index fits a byte.
            *p = ((acc >> have) & mask) as u8;
        }
        (self.acc, self.have) = (acc, have);
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
        self.prbs = self.start.clone();
        self.acc = 0;
        self.have = 0;
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
