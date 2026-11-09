//! `LaCrosse` IT+ `TX35DTH-IT` temperature and humidity sensor (868 MHz, FSK, 105 us per bit).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied; decision D1144). The device uses the
//! frame of the `TX29` ([`super::LacrosseTx29`]: the 24 bit pattern 0xA2DD49, five bytes with
//! the model nibble 9, a 6 bit id, the new-battery flag, three decimal digits of the
//! temperature with offset 40, the humidity with 106 = none and the weak battery bit on top,
//! CRC-8 polynomial 0x31 init 0) at a slower line rate: NRZ at about 9.58 kbit/s, 105 us per
//! bit. The two models are told apart by the bit rate alone: the slicer tolerates 20 % of a
//! bit per run, so a train at the other rate misses the pattern. The document has no separate
//! humidity rule for this device, so the rule of the `TX29` applies (decision D1144).

use super::encoder::{EncodeError, SensorEncoder, TrainSink};
use super::lacrosse_tx29::{REPEAT_GAP_US, RESET_US, decode_frames, encode_frame};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Modulation, PulseTrain};

/// The decoder and encoder of the TX35DTH-IT.
#[derive(Debug, Clone, Copy, Default)]
pub struct LacrosseTx35;

/// The bit time in microseconds (9.58 kbit/s is 104.4 us; the document gives 105 us).
const BIT_US: u32 = 105;

impl SensorDecoder for LacrosseTx35 {
    fn name(&self) -> &'static str {
        "lacrosse_tx35"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Fsk
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        decode_frames(train, out, "lacrosse_tx35", &[105.0], RESET_US)
    }
}

impl SensorEncoder for LacrosseTx35 {
    fn name(&self) -> &'static str {
        "lacrosse_tx35"
    }

    fn models(&self) -> &'static [&'static str] {
        &["lacrosse_tx35"]
    }

    fn modulation(&self) -> Modulation {
        Modulation::Fsk
    }

    fn mic(&self) -> Mic {
        Mic::Crc
    }

    fn default_repeats(&self) -> u32 {
        2
    }

    fn repeat_gap_us(&self) -> u32 {
        REPEAT_GAP_US
    }

    /// The frame of the `TX29` ([`super::LacrosseTx29`], including the
    /// [`EncodeError::Unrepresentable`] case of a CRC byte that ends in more than seven zero
    /// bits) as NRZ bits of 105 us.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encode_frame(r, self.models(), BIT_US, repeats, out)
    }
}
