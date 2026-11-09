//! Sensor readings (M46 §2): the trait of a device decoder, the reading and its JSON form, the
//! sink that collects readings and the registry of decoders by name.
//!
//! A decoder takes a pulse train of `band_pulse_detect`, configures a slicer of
//! [`crate::slicers`], finds its sync pattern and length in the rows, checks the integrity field
//! (a checksum or CRC; a device without one needs two equal repeats and records
//! [`Mic::Repeat`]) and fills [`Reading`]s. A value outside the range of the device rejects the
//! message ([`DecodeResult::OutOfRange`]). Decoders hold no state and never allocate; they are
//! zero-sized types listed in [`DECODERS`].
//!
//! ```
//! use sdrk_proto::pulses::{self, Modulation};
//! use sdrk_proto::sensors::{self, DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
//! use sdrk_proto::PacketFlags;
//!
//! /// A decoder that accepts every train with at least two pairs.
//! struct Demo;
//! impl SensorDecoder for Demo {
//!     fn name(&self) -> &'static str { "demo" }
//!     fn modulation(&self) -> Modulation { Modulation::Ook }
//!     fn decode(&self, train: &pulses::PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
//!         if train.len() < 2 { return DecodeResult::NoMatch; }
//!         let r = Reading::new("demo", 1, Mic::Repeat, &train.header());
//!         if out.push(r) { DecodeResult::Decoded(1) } else { DecodeResult::NoMatch }
//!     }
//! }
//!
//! let header = pulses::Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: 0,
//!     level_db10: 0, noise_db10: 0, rate_hz: 250_000 };
//! let pairs = [pulses::Pair { pulse: 10, gap: 10 }, pulses::Pair { pulse: 10, gap: 0 }];
//! let packet = pulses::encode_packet(0, PacketFlags::NONE, &header, &pairs)?;
//! let train = pulses::decode(packet.data())?;
//! let mut sink = ReadingSink::new();
//! assert_eq!(Demo.decode(&train, &mut sink), DecodeResult::Decoded(1));
//! assert_eq!(sink.len(), 1);
//! assert!(sensors::by_name("demo").is_none()); // the registry lists the built-in decoders only
//! assert!(sensors::by_name("wh2").is_some());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod bresser_3ch;
mod lacrosse_tx29;
mod nexus;
mod oregon_v21;
mod reading;
mod util;
mod wh1080;
mod wh2;

pub use bresser_3ch::Bresser3ch;
pub use lacrosse_tx29::LacrosseTx29;
pub use nexus::Nexus;
pub use oregon_v21::OregonV21;
pub use reading::{JsonBuf, Mic, Reading};
pub use wh2::Wh2;
pub use wh1080::Wh1080;

use crate::pulses::{Modulation, PulseTrain};

/// The most readings one train can give (a train may carry several repeats).
pub const MAX_READINGS: usize = 4;

/// The outcome of [`SensorDecoder::decode`], from best to worst: a train with several rows
/// reports the best of its rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecodeResult {
    /// This many readings were written to the sink.
    Decoded(usize),
    /// The integrity field of the best candidate did not match.
    BadIntegrity,
    /// The integrity field matched but a value is outside the range of the device.
    OutOfRange,
    /// A candidate started (sync pattern found) but the row ended too early.
    Truncated,
    /// The train is not a message of this device.
    NoMatch,
}

impl DecodeResult {
    fn rank(self) -> u8 {
        match self {
            Self::Decoded(_) => 4,
            Self::BadIntegrity => 3,
            Self::OutOfRange => 2,
            Self::Truncated => 1,
            Self::NoMatch => 0,
        }
    }

    /// The better of two results (the first one when they are equally good); decoders use it
    /// to combine the rows of a train.
    #[must_use]
    pub fn best(self, other: Self) -> Self {
        match (self, other) {
            (Self::Decoded(a), Self::Decoded(b)) => Self::Decoded(a + b),
            _ if other.rank() > self.rank() => other,
            _ => self,
        }
    }

    /// The number of readings, 0 unless decoded.
    pub fn count(self) -> usize {
        match self {
            Self::Decoded(n) => n,
            _ => 0,
        }
    }
}

/// Collects the readings of one train (at most [`MAX_READINGS`]).
#[derive(Debug, Clone)]
pub struct ReadingSink {
    items: [Option<Reading>; MAX_READINGS],
    len: usize,
}

impl Default for ReadingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadingSink {
    /// An empty sink.
    pub const fn new() -> Self {
        Self {
            items: [None; MAX_READINGS],
            len: 0,
        }
    }

    /// Removes every reading.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Adds a reading; false (and nothing added) when the sink is full.
    pub fn push(&mut self, reading: Reading) -> bool {
        match self.items.get_mut(self.len) {
            Some(slot) => {
                *slot = Some(reading);
                self.len += 1;
                true
            }
            None => false,
        }
    }

    /// The number of readings.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sink is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The readings in order.
    pub fn iter(&self) -> impl Iterator<Item = &Reading> {
        self.items.iter().take(self.len).flatten()
    }

    /// Reading `i`.
    pub fn get(&self, i: usize) -> Option<&Reading> {
        if i < self.len {
            self.items.get(i)?.as_ref()
        } else {
            None
        }
    }
}

/// A device decoder (see the [module documentation](self)).
pub trait SensorDecoder: Sync {
    /// The name in the registry and in the `decoders` parameter, lowercase.
    fn name(&self) -> &'static str;

    /// The modulation of the device; trains of the other modulation are not offered to it.
    fn modulation(&self) -> Modulation;

    /// Decodes `train` and pushes the readings to `out`; the radio fields of the readings come
    /// from the train's header.
    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult;
}

/// The built-in decoders in a fixed order: 433 MHz (`wh2`, `nexus`, `oregon_v21`, `wh1080`),
/// 868 MHz (`lacrosse_tx29`), then `bresser_3ch` (433 MHz), added last so that the bits of the
/// `decoders` mask of the block keep their meaning.
pub static DECODERS: &[&dyn SensorDecoder] = &[
    &Wh2,
    &Nexus,
    &OregonV21,
    &Wh1080,
    &LacrosseTx29,
    &Bresser3ch,
];

/// The decoder of that name.
pub fn by_name(name: &str) -> Option<&'static dyn SensorDecoder> {
    DECODERS.iter().copied().find(|d| d.name() == name)
}
