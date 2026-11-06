//! `rds_encode`: the RDS group carousel and control packets to the shaped biphase baseband,
//! driven by the sample clock of the multiplex.

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::rds::{RDS_TABLE_LEN, RdsShape, RdsShapeState, rds_pulse_table};
use sdrk_proto::Packet;
use sdrk_proto::rds::{BLOCK_BITS, Carousel, GROUP_BITS, Group, PS_LEN, RT_LEN, group_blocks26};

use super::{head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::{Clocked, Packets, StreamElem};
use crate::storage::{Plan, Storage};

/// First byte of a packet that sets the programme service name (followed by exactly 8 bytes).
pub const RDS_PACKET_PS: u8 = 0x01;
/// First byte of a packet that sets the radiotext (followed by 0 to 64 bytes; none stops the
/// radiotext).
pub const RDS_PACKET_RT: u8 = 0x02;
/// First byte of a packet that queues one raw group (followed by exactly 8 bytes: the words of
/// blocks A, B, C, D big-endian).
pub const RDS_PACKET_GROUP: u8 = 0x03;

/// Lowest clock rate of `rds_encode` in Hz (twice the bandwidth of the RDS pulse, the limit of
/// the kernel `rds_shape`).
const MIN_RATE_HZ: f64 = 4750.0;

/// Description of the `rds_encode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "rds_encode",
    summary: "RDS encoder: the group carousel and control packets to the shaped biphase RDS \
        baseband at the clock rate.",
    details: "Port a is the clock: any stream at the rate of the multiplex (normally the \
        stereo audio after the up-sampling resample); its values are ignored and one output \
        sample is produced per clock sample. Port b takes control packets by their first \
        byte: 0x01 and exactly 8 bytes set the programme service name (groups 0A); 0x02 and 0 \
        to 64 bytes set the radiotext (groups 2A; a shorter text gets a carriage return and \
        spaces, a changed text toggles the A/B flag, an empty one stops the groups 2A); 0x03 \
        and exactly 8 bytes queue one raw group (the words of blocks A, B, C, D big-endian, \
        sent unchanged before the next carousel group; at most 16 wait, a further one waits \
        in the packet queue with the packets after it). Any other packet is ignored and \
        counted in the reading ignored_packets. Packets are taken at the start of every call \
        and before every new group; they take effect with the next group (every 87.6 ms). \
        Without packets the carousel sends the name (8 spaces at first) and, once set, the \
        radiotext, alternately. Each group is 4 blocks of 26 bits (information word and \
        checkword, offsets A, B, C or C', D), differentially coded and shaped with the kernel \
        rds_shape at 1187.5 bit/s, the first bit centred at the first clock sample; the output \
        peaks at about 1.06. Feed it to port b of fm_stereo_rds_mod. Pitfalls: the clock rate \
        must be at least 4750 Hz (fm_stereo_rds_mod needs 120 kHz); the reported delay is 0 \
        because the data timing does not matter relative to the audio.",
    inputs: &[
        PortDesc {
            name: "clock",
            types: &[PortType::C32, PortType::CQ15, PortType::F32, PortType::Q15],
            description: "Any stream at the multiplex rate; only its sample count is used.",
        },
        PortDesc {
            name: "control",
            types: &[PortType::Packets],
            description: "Control packets: 0x01 name, 0x02 radiotext, 0x03 raw group.",
        },
    ],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "The RDS baseband (peak about 1.06) at the clock rate, for fm_stereo_rds_mod.",
    }],
    params: &[
        ParamDesc {
            name: "pi",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: 65_535,
                default: None,
            },
            typical: "53539 (0xD123)",
            mutability: Mutability::Build,
            description: "Programme identification code, sent in block A of every carousel \
                group.",
        },
        ParamDesc {
            name: "pty",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: 31,
                default: Some(0),
            },
            typical: "0 (none), 10 (pop music)",
            mutability: Mutability::Build,
            description: "Programme type, sent in block B of every carousel group.",
        },
        ParamDesc {
            name: "tp",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false",
            mutability: Mutability::Build,
            description: "Traffic programme flag, sent in block B of every carousel group.",
        },
    ],
    required: &["pi"],
};

/// Configuration of [`RdsEncodeBlock`].
///
/// ```
/// use sdrk_blocks::RdsEncodeConfig;
/// let cfg = RdsEncodeConfig { pi: 0xD123, ..RdsEncodeConfig::default() };
/// assert_eq!((cfg.pty, cfg.tp), (0, false));
/// assert!(cfg.validate(sdrk::Rate::hz(192_000)?).is_ok());
/// assert!(RdsEncodeConfig { pty: 32, ..cfg }.validate(sdrk::Rate::hz(192_000)?).is_err());
/// // Below 4750 Hz the pulse does not fit the rate.
/// assert!(cfg.validate(sdrk::Rate::hz(4_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RdsEncodeConfig {
    /// Programme identification (required in a flow; 0 by default here).
    pub pi: u16,
    /// Programme type (0 to 31).
    pub pty: u8,
    /// Traffic programme flag.
    pub tp: bool,
}

impl RdsEncodeConfig {
    /// Checks the configuration against the clock rate.
    ///
    /// # Errors
    /// `OutOfRange` for `pty` above 31 or a rate below 4750 Hz.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        for p in DESCRIPTION.params {
            p.check(&self.get(p.name)?)?;
        }
        if rate_in.as_f64() < MIN_RATE_HZ {
            return Err(Error::OutOfRange {
                name: "rate_in",
                min: MIN_RATE_HZ,
                max: f64::MAX,
                got: rate_in.as_f64(),
            });
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "pi" => self.pi = u16::try_from(value.as_i64()?).map_err(|_| Error::WrongParamType)?,
            "pty" => self.pty = u8::try_from(value.as_i64()?).map_err(|_| Error::WrongParamType)?,
            "tp" => self.tp = value.as_bool()?,
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
            "pi" => Ok(ParamValue::I64(i64::from(self.pi))),
            "pty" => Ok(ParamValue::I64(i64::from(self.pty))),
            "tp" => Ok(ParamValue::Bool(self.tp)),
            _ => unknown(),
        }
    }
}

/// The bits of the group being sent and the differential coder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Coder {
    /// The four 26-bit blocks of the group.
    blocks: [u32; 4],
    /// The next bit of the group (`GROUP_BITS`: the group is sent, fetch the next one).
    pos: usize,
    /// The last transmitted bit `t_{k-1}`.
    t: bool,
}

impl Coder {
    const fn new() -> Self {
        Self {
            blocks: [0; 4],
            pos: GROUP_BITS,
            t: false,
        }
    }

    /// The amplitude `2 t_k - 1` of the next bit. When the group is sent, the packets that
    /// wait are taken first (see [`Control::take`]), then the next group of the carousel.
    fn next(&mut self, control: &mut Control, packets: &[Packet], taken: &mut usize) -> i8 {
        if self.pos >= GROUP_BITS {
            control.take(packets, taken);
            self.blocks = group_blocks26(control.carousel.next_group());
            self.pos = 0;
        }
        let block = self.blocks.get(self.pos / BLOCK_BITS).copied().unwrap_or(0);
        let d = (block >> (BLOCK_BITS - 1 - self.pos % BLOCK_BITS)) & 1 != 0;
        self.pos += 1;
        self.t ^= d;
        if self.t { 1 } else { -1 }
    }
}

/// The carousel and the count of ignored packets: what the control packets change.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Control {
    carousel: Carousel,
    ignored: u64,
}

impl Control {
    /// Applies one control packet; `false` when it must wait (a raw group with a full queue).
    fn apply(&mut self, packet: &Packet) -> bool {
        let ok = match packet.data() {
            [RDS_PACKET_PS, rest @ ..] if rest.len() == PS_LEN => {
                let mut ps = [b' '; PS_LEN];
                ps.copy_from_slice(rest);
                self.carousel.set_ps(&ps);
                true
            }
            [RDS_PACKET_RT, rest @ ..] if rest.len() <= RT_LEN => {
                self.carousel.set_rt(rest).is_ok()
            }
            [RDS_PACKET_GROUP, a0, a1, b0, b1, c0, c1, d0, d1] => {
                if self.carousel.is_raw_full() {
                    return false;
                }
                self.carousel
                    .push_raw(Group::new([
                        u16::from_be_bytes([*a0, *a1]),
                        u16::from_be_bytes([*b0, *b1]),
                        u16::from_be_bytes([*c0, *c1]),
                        u16::from_be_bytes([*d0, *d1]),
                    ]))
                    .is_ok()
            }
            _ => false,
        };
        if !ok {
            self.ignored = self.ignored.saturating_add(1);
        }
        true
    }

    /// Applies the packets from `*taken` on in order and advances `*taken`, up to the first one
    /// that must wait.
    fn take(&mut self, packets: &[Packet], taken: &mut usize) {
        while let Some(p) = packets.get(*taken) {
            if !self.apply(p) {
                break;
            }
            *taken += 1;
        }
    }
}

/// RDS encoder: a clock stream `A` (port a, values ignored) with control packets (port b) in,
/// the shaped biphase RDS baseband (`f32`, one sample per clock sample) out.
///
/// The block holds a [`Carousel`] (`sdrk_proto::rds`) for the programme service name and the
/// radiotext, the bits of the group being sent and the differential coder, and calls the kernel
/// `rds_shape` with the pulse table in its storage ([`RDS_TABLE_LEN`] `f32` values). Packets of
/// port b are consumed in order at the start of every call and again before every new group
/// within the call: `0x01` + 8 bytes sets the name, `0x02` + 0 to 64 bytes the radiotext,
/// `0x03` + 8 bytes queues a raw group (A, B, C, D big-endian); a raw group packet that finds the
/// queue of the carousel full stops the consumption and waits (with the packets after it) for
/// the next group or call. Every other packet is consumed, ignored and counted
/// ([`RdsEncodeBlock::ignored_packets`], reading `ignored_packets`). The carousel reads the texts
/// and the queue only when it gives the next group, so the output does not depend on how the
/// clock is cut into calls, as long as each packet reaches the block at the same clock sample.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, AlignedBytes, Block, Packet, PacketFlags, RdsEncodeBlock, RdsEncodeConfig};
/// let cfg = RdsEncodeConfig { pi: 0xD123, pty: 10, tp: false };
/// let rate = Rate::hz(192_000)?;
/// let bytes = RdsEncodeBlock::<f32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = AlignedBytes([0u8; 4112]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = RdsEncodeBlock::<f32, _>::new(&cfg, rate, &mut arena)?;
/// let name = Packet::new(0, PacketFlags::NONE, b"\x01SDRK TST")?;
/// let bad = Packet::new(0, PacketFlags::NONE, b"\x07?")?;
/// let clock = [0.0f32; 512];
/// let mut out = [0.0f32; 512];
/// let p = block.process((&clock, &[name, bad]), &mut out)?;
/// assert_eq!((p.consumed, p.produced, block.consumed_packets()), (512, 512, 2));
/// assert_eq!(block.ignored_packets(), 1);
/// assert_eq!(block.carousel().ps(), b"SDRK TST");
/// assert!(out.iter().all(|y| y.abs() <= 2.0) && out.iter().any(|&y| y != 0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RdsEncodeBlock<A: StreamElem, S: Storage> {
    config: RdsEncodeConfig,
    control: Control,
    shape: RdsShapeState<S::Buf<f32>>,
    coder: Coder,
    taken: usize,
    _clock: PhantomData<fn() -> A>,
}

impl<A: StreamElem, S: Storage> core::fmt::Debug for RdsEncodeBlock<A, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RdsEncodeBlock")
            .field("config", &self.config)
            .field("control", &self.control)
            .field("coder", &self.coder)
            .finish_non_exhaustive()
    }
}

impl<A: StreamElem, S: Storage> RdsEncodeBlock<A, S> {
    /// Creates the block: the carousel of `config` (name 8 spaces, no radiotext) and the pulse
    /// table of `rds_shape` in `storage`.
    ///
    /// # Errors
    /// The errors of [`RdsEncodeConfig::validate`]; `Storage` when `storage` is too small.
    pub fn new(config: &RdsEncodeConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let carousel = Carousel::new(config.pi, config.pty, config.tp)?;
        let mut table = storage.alloc::<f32>(RDS_TABLE_LEN)?;
        rds_pulse_table(table.as_mut())?;
        Ok(Self {
            config: *config,
            control: Control {
                carousel,
                ignored: 0,
            },
            shape: RdsShapeState::new(table, rate_in)?,
            coder: Coder::new(),
            taken: 0,
            _clock: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs: the pulse table of [`RDS_TABLE_LEN`] `f32`.
    ///
    /// # Errors
    /// The errors of [`RdsEncodeConfig::validate`].
    pub fn storage_bytes(config: &RdsEncodeConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<f32>(RDS_TABLE_LEN)?;
        Ok(plan.bytes())
    }

    /// Output rate for the configuration (equal to the clock rate).
    ///
    /// # Errors
    /// The errors of [`RdsEncodeConfig::validate`].
    pub fn rate_out(config: &RdsEncodeConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// The packets that were ignored (unknown first byte or wrong length) since construction or
    /// [`Block::reset`].
    pub fn ignored_packets(&self) -> u64 {
        self.control.ignored
    }

    /// The carousel: the name, the radiotext and the raw groups that wait.
    pub fn carousel(&self) -> &Carousel {
        &self.control.carousel
    }

    /// The index of the next data bit the shaper will fetch (the bits fetched since
    /// construction or [`Block::reset`]).
    pub fn next_bit(&self) -> u64 {
        self.shape.next_bit()
    }
}

impl<A: StreamElem, S: Storage> Block for RdsEncodeBlock<A, S> {
    type In = Clocked<A, Packets>;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: (&[A], &[Packet]), output: &mut [f32]) -> Result<Progress> {
        let (clock, packets) = input;
        let Self {
            control,
            shape,
            coder,
            taken,
            ..
        } = self;
        *taken = 0;
        control.take(packets, taken);
        let n = clock.len().min(output.len());
        let out = head_mut(output, n)?;
        let mut bits = || coder.next(control, packets, taken);
        <Sdrk as RdsShape>::rds_shape(&Sdrk, shape, &mut bits, out)?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
    }

    /// Restarts the bit stream: the shaper at sample 0, the differential coder at 0, the group
    /// being sent dropped, the carousel at group 0A segment 0, the count of ignored packets 0.
    /// The name, the radiotext (with its A/B flag) and the queued raw groups stay.
    fn reset(&mut self) {
        self.shape.reset();
        self.coder = Coder::new();
        self.control.carousel.restart();
        self.control.ignored = 0;
        self.taken = 0;
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(_) => Err(Error::ReadOnlyParameter),
            None => Err(Error::UnknownParameter),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }

    fn consumed_packets(&self) -> usize {
        self.taken
    }

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        (index == 0).then_some(("ignored_packets", self.control.ignored as f64))
    }
}
