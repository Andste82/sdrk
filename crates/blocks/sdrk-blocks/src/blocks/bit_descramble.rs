//! `bit_descramble`: the self-synchronizing (multiplicative) descrambler of a soft-bit stream, the
//! receive side of `bit_scramble` (M44.2, the maintainer's decision on I44.2-6).

use sdrk::{Progress, Rate};

use super::bit_scramble::{
    BitScrambleConfig, POLYNOMIAL_PARAM, Register, SCRAMBLE_CONSTELLATION_PARAM,
};
use super::{unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::{Error, Result};
use crate::storage::Storage;

/// Description of the `bit_descramble` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "bit_descramble",
    summary: "Undoes bit_scramble: the self-synchronizing (multiplicative) descrambler of a \
        soft-bit stream.",
    details: "Every output bit is the input bit XOR the input bits a and b symbols earlier \
        (polynomial, constellation; the register starts at zero and after a reset), the inverse \
        of bit_scramble with the same parameters. It needs no frame reference: after a times \
        the bits per symbol of correct input bits from any start it gives the bits that went \
        into the scrambler, and a wrong input bit gives 3 wrong output bits (at the bit and a \
        and b symbols later). The bits are soft bits: above 0 is a 1; the output keeps the \
        magnitude of the input (its reliability) and takes the sign of the descrambled bit (a 0 \
        or NaN input stays 0 or NaN and counts as a 0 bit in the register). With the \
        constellation of the demodulator before it, a quarter turn of a square constellation \
        commutes with it, so frame_decode with rotations four after it still resolves the \
        rotation. The receive side of qam_frame_decode with its scrambler on. The rate does not \
        change.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Scrambled soft bits, one sample per bit; above 0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The descrambled soft bits, one per input bit; above 0 is a 1.",
    }],
    params: &[POLYNOMIAL_PARAM, SCRAMBLE_CONSTELLATION_PARAM],
    required: &[],
};

/// The self-synchronizing descrambler: `f32` soft bits in, `f32` soft bits out, one per bit.
///
/// With the delays `A` and `B` in bits of [`BitScrambleConfig::delays`], output bit `n` is
/// `x[n] ^ x[n - A] ^ x[n - B]` with `x` the input bits (`> 0` is a 1; 0 before the start and after
/// [`reset`](Block::reset)); the output sample is the input sample with its sign flipped where the
/// parity `x[n - A] ^ x[n - B]` is 1, so it keeps the magnitude. The inverse of
/// [`BitScrambleBlock`](crate::BitScrambleBlock) with the same configuration, from any start after
/// `A` bits. The register is a ring of the last `A` input bits as soft bits; after its first `A`
/// bits a call is one element-wise loop, and the result does not depend on how the input is cut.
/// A call takes `min(input.len(), output.len())` bits. Timing: constant ratio 1, delay 0.
/// Storage: the ring, `A` `f32` (at most 580).
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, BitDescrambleBlock, BitScrambleConfig, Block};
/// let cfg = BitScrambleConfig::default();
/// let mut mem = [0u8; 4096];
/// let mut st = ArenaStorage::new(&mut mem);
/// let mut rx = BitDescrambleBlock::new(&cfg, Rate::hz(9600)?, &mut st)?;
/// // one 1 on the air gives three 1 out: at the bit, 18 and 23 bits later
/// let mut air = [-1.0f32; 30];
/// air[2] = 1.0;
/// let mut out = [0.0f32; 30];
/// rx.process(&air, &mut out)?;
/// let ones: Vec<usize> = (0..30).filter(|&i| out[i] > 0.0).collect();
/// assert_eq!(ones, [2, 20, 25]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct BitDescrambleBlock<S: Storage> {
    reg: Register<S::Buf<f32>>,
    config: BitScrambleConfig,
}

impl<S: Storage> core::fmt::Debug for BitDescrambleBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BitDescrambleBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> BitDescrambleBlock<S> {
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
    /// address (an unaligned slice may need up to 15 bytes more): the ring of `A` `f32`, as
    /// [`BitScrambleBlock::storage_bytes`](crate::BitScrambleBlock::storage_bytes).
    ///
    /// # Errors
    /// None today.
    pub fn storage_bytes(config: &BitScrambleConfig, rate_in: Rate) -> Result<usize> {
        super::bit_scramble::BitScrambleBlock::<S>::storage_bytes(config, rate_in)
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

    /// Descrambles `data` in place: the same as [`Block::process`] with `data` as input and
    /// output (the call of `qam_frame_decode` on the soft bits between its stages).
    ///
    /// ```
    /// use sdrk::Rate;
    /// use sdrk_blocks::{ArenaStorage, BitDescrambleBlock, BitScrambleConfig, Block};
    /// let cfg = BitScrambleConfig::default();
    /// let rate = Rate::hz(9600)?;
    /// let mut mem = [0u8; 4096];
    /// let mut st = ArenaStorage::new(&mut mem);
    /// let mut a = BitDescrambleBlock::new(&cfg, rate, &mut st)?;
    /// let mut b = BitDescrambleBlock::new(&cfg, rate, &mut st)?;
    /// let x: Vec<f32> = (0..90).map(|i| if i % 7 < 3 { 0.25 } else { -2.0 }).collect();
    /// let mut y = vec![0.0f32; 90];
    /// a.process(&x, &mut y)?;
    /// let mut z = x.clone();
    /// b.process_in_place(&mut z);
    /// assert_eq!(y, z);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn process_in_place(&mut self, data: &mut [f32]) {
        self.reg.descramble_in_place(data);
    }
}

impl<S: Storage> Block for BitDescrambleBlock<S> {
    type In = f32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        self.reg
            .descramble(super::head(input, n)?, super::head_mut(output, n)?);
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
