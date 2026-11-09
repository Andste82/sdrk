//! `sensor_encode`: sensor readings as JSON text packets to the pulse trains of the sensor
//! devices (M46 extension, D1124): the mirror image of `sensor_decode`. The block reads the
//! JSON object of a reading (the form `sensor_decode` writes), picks the encoder of its model
//! from [`sdrk_proto::sensors::ENCODERS`] and writes the pulse train of the device as packets of
//! [`sdrk_proto::pulses`] (a long train is cut into packets with the continued flag).

use sdrk::{Delay, Progress, Ratio, RatioKind};
use sdrk_proto::pulses::{self, MAX_PAIRS, Pair};
use sdrk_proto::sensors::{EncodeError, MAX_TRAIN_PAIRS, Reading, TrainSink, encoder_by_model};
use sdrk_proto::{Packet, PacketFlags};

use super::unknown;
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// The most repeats of a train the parameter `repeats` allows.
pub const SENSOR_ENCODE_MAX_REPEATS: i64 = 64;

/// The most packets one reading gives: the longest train ([`MAX_TRAIN_PAIRS`] pairs) cut into
/// packets of [`MAX_PAIRS`] pairs; this is the output ratio of the block (D1128).
pub const SENSOR_ENCODE_PACKETS: usize = MAX_TRAIN_PAIRS.div_ceil(MAX_PAIRS);

/// Description of the `sensor_encode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "sensor_encode",
    summary: "Encodes sensor readings (JSON text packets) into the pulse trains of the sensor \
        devices.",
    details: "Reads one JSON object per packet, the form that sensor_decode writes (fields \
        model, id, channel, battery_ok and the values of the device such as temperature_C, \
        humidity, wind_avg_km_h, wind_max_km_h, wind_dir_deg, rain_mm; time, freq_MHz, rssi_dB, \
        mic and repeats are read and ignored), maps the model to the encoder of the device \
        (wh2, nexus, oregon_v21 with its models Oregon-THN132N and Oregon-THGR122NX, wh1080, \
        lacrosse_tx29, lacrosse_tx35, bresser_3ch with the model Bresser-3CH) and writes the pulse train of \
        the device with the timings of its decoder: one packet per 30 pairs, the first ones \
        with the continued flag, all carrying the index of the reading packet, at the sample \
        rate 1 MHz (one sample per microsecond, so that every nominal timing is an integer). \
        One train holds all repeats of the message with the repeat gap of the device between \
        them and ends with a gap of 0; the repeats default to the figure of the device (wh2 3, \
        nexus 4, oregon_v21 2, wh1080 2, lacrosse_tx29 2, lacrosse_tx35 2, bresser_3ch 22) when repeats is 0. \
        A train has at most 1200 pairs (the capacity of sensor_decode), so a device takes at \
        most 27 repeats (bresser_3ch) and more is counted in too_long. A value is rounded to \
        the resolution of the device; a value outside its range, a field the device does not \
        carry and a missing required field reject the reading (counted in rejected). A packet \
        that is not the JSON text of a reading is counted in invalid, a model that no encoder \
        has in unknown_model; all four kinds are consumed and give no output. The block \
        needs room for the whole train (up to 40 packets) and otherwise waits, so a reading is \
        written whole or not at all. Feed the output to port b of pulse_key to key a \
        transmitter. The encoders inherit what the decoders assume (wh1080 band and pulse gap, \
        the repeat gaps); a nexus frame that passes the check of the Rubicson protocol is \
        refused. Pitfall: lacrosse_tx29 and lacrosse_tx35 are \
        FSK trains (header Fsk) and need pulse_key in mode fsk, then fsk_mod and a gate; \
        the others are on-off trains for pulse_key in mode ook; wh1080 is not confirmed on \
        air. Pitfall: a \
        bresser_3ch message is about 1100 pairs, so a reading is about 37 packets. Cost: 200 \
        per reading plus 1 per pair.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Packets],
        description: "One packet per reading: the JSON text of the reading.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "The pulse train of the reading (a train longer than 30 pairs is several \
            packets, the first ones with the continued flag), the index of the reading packet.",
    }],
    params: &[ParamDesc {
        name: "repeats",
        unit: Unit::None,
        kind: ParamKind::I64 {
            min: 0,
            max: SENSOR_ENCODE_MAX_REPEATS,
            default: Some(0),
        },
        typical: "0 (the figure of the device); 22 for bresser_3ch, 3 for wh2",
        mutability: Mutability::Build,
        description: "Copies of the message in one train; 0 takes the figure of the device. \
            More than fit 1200 pairs reject the reading (too_long).",
    }],
    required: &[],
};

/// Configuration of [`SensorEncodeBlock`].
///
/// ```
/// use sdrk_blocks::{SensorEncodeConfig, ParamValue};
/// let mut cfg = SensorEncodeConfig::default();
/// assert_eq!(cfg.repeats, 0);
/// cfg.set("repeats", ParamValue::I64(5))?;
/// assert_eq!(cfg.repeats, 5);
/// assert!(cfg.set("repeats", ParamValue::I64(65)).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SensorEncodeConfig {
    /// Copies of the message in one train (0: the figure of the device).
    pub repeats: u32,
}

impl SensorEncodeConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` of `repeats`.
    pub fn validate(&self) -> Result<()> {
        for p in DESCRIPTION.params {
            p.check(&self.get(p.name)?)?;
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
            "repeats" => {
                self.repeats = u32::try_from(value.as_i64()?).map_err(|_| Error::WrongParamType)?;
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
            "repeats" => Ok(ParamValue::I64(i64::from(self.repeats))),
            _ => unknown(),
        }
    }
}

/// Sensor encoder: [`Packets`] of JSON readings in, pulse train [`Packets`] out.
///
/// ```
/// use sdrk_blocks::{ArenaStorage, AlignedBytes, Block, Packet, SensorEncodeBlock, SensorEncodeConfig};
/// use sdrk_proto::PacketFlags;
/// use sdrk_proto::pulses;
/// let cfg = SensorEncodeConfig::default();
/// let mut mem = vec![0u8; SensorEncodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut enc = SensorEncodeBlock::new(&cfg, &mut arena)?;
/// let text = br#"{"model":"nexus","id":92,"channel":2,"temperature_C":-3.5,"humidity":71}"#;
/// let reading = Packet::new(7, PacketFlags::NONE, text)?;
/// let mut out = [Packet::default(); 40];
/// let p = enc.process(&[reading], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (1, 5)); // 4 frames of 37 pairs: 148 pairs, 5 packets
/// assert_eq!(out[0].index, 7);
/// assert!(out[0].flags.contains(pulses::CONTINUED) && !out[4].flags.contains(pulses::CONTINUED));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SensorEncodeBlock<S: Storage> {
    config: SensorEncodeConfig,
    /// The pairs of the train being cut into packets.
    pairs: S::Buf<Pair>,
    encoded: u64,
    invalid: u64,
    unknown_model: u64,
    rejected: u64,
    too_long: u64,
}

impl<S: Storage> core::fmt::Debug for SensorEncodeBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SensorEncodeBlock")
            .field("config", &self.config)
            .field("encoded", &self.encoded)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> SensorEncodeBlock<S> {
    /// Creates the block.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration, `Storage` if `storage` is too small.
    pub fn new(config: &SensorEncodeConfig, storage: &mut S) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: *config,
            pairs: storage.alloc::<Pair>(MAX_TRAIN_PAIRS)?,
            encoded: 0,
            invalid: 0,
            unknown_model: 0,
            rejected: 0,
            too_long: 0,
        })
    }

    /// Exact bytes of storage the block needs.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &SensorEncodeConfig) -> Result<usize> {
        config.validate()?;
        let mut plan = Plan::default();
        plan.add::<Pair>(MAX_TRAIN_PAIRS)?;
        Ok(plan.bytes())
    }

    /// Readings written as pulse trains so far.
    pub fn encoded(&self) -> u64 {
        self.encoded
    }

    /// Packets that were not the JSON text of a reading.
    pub fn invalid(&self) -> u64 {
        self.invalid
    }

    /// Readings of a model that no encoder has.
    pub fn unknown_model(&self) -> u64 {
        self.unknown_model
    }

    /// Readings that an encoder rejected (a value out of range, a field the device does not
    /// carry, a missing field).
    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// Readings whose train does not fit 1200 pairs.
    pub fn too_long(&self) -> u64 {
        self.too_long
    }

    /// The train of one reading packet: `Err` is the counter to add (the packet is consumed),
    /// `Ok` the number of pairs in `self.pairs` with the encoder's header.
    fn encode(&mut self, text: &[u8]) -> core::result::Result<(usize, pulses::Header), Count> {
        let text = core::str::from_utf8(text).map_err(|_| Count::Invalid)?;
        let reading = Reading::from_json(text).map_err(|e| match e {
            sdrk_proto::sensors::JsonError::UnknownModel => Count::UnknownModel,
            _ => Count::Invalid,
        })?;
        let encoder = encoder_by_model(reading.model).ok_or(Count::UnknownModel)?;
        let repeats = match self.config.repeats {
            0 => encoder.default_repeats(),
            n => n,
        };
        let mut sink = TrainSink::new(self.pairs.as_mut());
        encoder
            .encode(&reading, repeats, &mut sink)
            .map_err(|e| match e {
                EncodeError::BufferTooSmall => Count::TooLong,
                EncodeError::UnknownModel => Count::UnknownModel,
                _ => Count::Rejected,
            })?;
        Ok((sink.len(), encoder.header()))
    }
}

/// The counter a consumed packet without output adds to.
#[derive(Clone, Copy)]
enum Count {
    Invalid,
    UnknownModel,
    Rejected,
    TooLong,
}

impl<S: Storage> Block for SensorEncodeBlock<S> {
    type In = Packets;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[Packet], output: &mut [Packet]) -> Result<Progress> {
        let mut consumed = 0;
        let mut produced = 0;
        for packet in input {
            let (len, header) = match self.encode(packet.data()) {
                Ok(t) => t,
                Err(c) => {
                    match c {
                        Count::Invalid => self.invalid += 1,
                        Count::UnknownModel => self.unknown_model += 1,
                        Count::Rejected => self.rejected += 1,
                        Count::TooLong => self.too_long += 1,
                    }
                    consumed += 1;
                    continue;
                }
            };
            // The whole train or nothing: wait for room.
            let need = len.div_ceil(MAX_PAIRS).max(1);
            if output.len().saturating_sub(produced) < need {
                break;
            }
            consumed += 1;
            let pairs = self.pairs.as_ref().get(..len).unwrap_or(&[]);
            let room = output.get_mut(produced..).unwrap_or(&mut []);
            for (slot, p) in room.iter_mut().zip(pulses::packets(
                packet.index,
                PacketFlags::NONE,
                &header,
                pairs,
            )) {
                if let Ok(p) = p {
                    *slot = p;
                    produced += 1;
                } else {
                    self.rejected += 1;
                    break;
                }
            }
            self.encoded += 1;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let r = Ratio::new(SENSOR_ENCODE_PACKETS as u64, 1).unwrap_or(Ratio::ONE);
        BlockTiming {
            ratio: r,
            kind: RatioKind::Packets { max_per_input: r },
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.encoded = 0;
        self.invalid = 0;
        self.unknown_model = 0;
        self.rejected = 0;
        self.too_long = 0;
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

    fn reading(&self, index: usize) -> Option<(&'static str, f64)> {
        match index {
            0 => Some(("encoded", self.encoded as f64)),
            1 => Some(("invalid", self.invalid as f64)),
            2 => Some(("unknown_model", self.unknown_model as f64)),
            3 => Some(("rejected", self.rejected as f64)),
            4 => Some(("too_long", self.too_long as f64)),
            _ => None,
        }
    }
}
