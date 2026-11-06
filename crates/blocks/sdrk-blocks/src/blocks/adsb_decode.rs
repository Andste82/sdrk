//! `adsb_decode`: Mode S / ADS-B downlink frames from complex samples at 2 MHz (pulse-position
//! modulation, CRC-24, an ICAO address cache for the address/parity formats), built from the
//! kernel `mag`. The preamble test, the phase correction and the address cache follow the 2 MHz
//! demodulator of dump1090 (`demod_2000.c`, `mode_s.c`, `icao_filter.c`; D721 to D723).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::spectrum::Mag;
use sdrk_proto::adsb::{self, LONG_FRAME_BYTES, SHORT_FRAME_BYTES};
use sdrk_proto::crc::crc24_adsb;
use sdrk_proto::{Packet, PacketFlags};

use super::{head, head_mut, tail, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// The flag of an address/parity frame (DF 0, 4, 5, 16, 20, 21, 24, and DF 11 with a nonzero
/// CRC syndrome): bit 16, the first protocol bit of [`PacketFlags`]. Its parity cannot be
/// checked on its own; the frame is output only when its syndrome, the address it was sent
/// with, is in the ICAO address cache of the block (D722).
pub const ADSB_ADDRESS_PARITY: PacketFlags = PacketFlags(1 << 16);

/// Slots of each of the two tables of the ICAO address cache (a power of two; D722).
pub const ADSB_ICAO_SLOTS: usize = 2048;

/// Addresses each table of the ICAO address cache holds at most (half its slots, so a lookup
/// probes few slots; D722).
pub const ADSB_ICAO_CAPACITY: usize = ADSB_ICAO_SLOTS / 2;

/// Input samples between two flips of the ICAO address cache: 60 s at 2 MHz (dump1090
/// `MODES_ICAO_FILTER_TTL` 60000 ms), so an address lives 60 to 120 s of input (D722).
pub const ADSB_ICAO_TTL: u64 = 60 * RATE_HZ;

/// Input samples per `mag` call at most (the size of the work part of the magnitude buffer).
pub const ADSB_CHUNK: usize = 2048;

/// The only accepted sample rate in Hz.
const RATE_HZ: u64 = 2_000_000;

/// Samples of the noise window whose median is the noise level.
const MED_LEN: usize = 256;

/// The rank of the median in the window: the 128th smallest value (the lower median, D710).
const MED_RANK: usize = MED_LEN / 2;

/// Samples of the preamble.
const PREAMBLE: usize = 16;

/// Bits of a long frame.
const LONG_BITS: usize = 8 * LONG_FRAME_BYTES;

/// Bits of a short frame.
const SHORT_BITS: usize = 8 * SHORT_FRAME_BYTES;

/// Samples from the preamble start to the end of a long frame.
const FRAME_LEN: usize = PREAMBLE + 2 * LONG_BITS;

/// Samples kept between pieces: the noise window and a long frame less one sample.
const KEEP: usize = MED_LEN + FRAME_LEN - 1;

/// Length of the magnitude buffer.
const BUF_LEN: usize = KEEP + ADSB_CHUNK;

/// Bins of the noise histogram: bin 0 holds everything below 2^-24 (and NaN), bins 1 to 1023
/// cover 2^-24 to 2^8 in steps of 1/32 octave, the last bin also everything above.
const BINS: usize = 1024;

/// The bin of a value is `(bits >> BIN_SHIFT) - BIN_BASE` (an `f32` keeps 5 mantissa bits).
const BIN_SHIFT: u32 = 18;

/// The exponent field of 2^-24 (103) in units of 1/32 octave, less 1.
const BIN_BASE: u32 = (103 << 5) - 1;

/// The smallest value of bin 1, 2^-24.
const BIN_MIN: f32 = 1.0 / 16_777_216.0;

/// The downlink formats that are decoded (dump1090 `decodeModesMessage`): DF 0, 4, 5, 11, 16,
/// 17, 18, 20, 21 and 24, one bit each.
const VALID_DF: u32 = (1 << 0)
    | (1 << 4)
    | (1 << 5)
    | (1 << 11)
    | (1 << 16)
    | (1 << 17)
    | (1 << 18)
    | (1 << 20)
    | (1 << 21)
    | (1 << 24);

/// Bit 24 marks a used slot of the ICAO address cache (an address has 24 bits).
const ICAO_USED: u32 = 1 << 24;

/// Description of the `adsb_decode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "adsb_decode",
    summary: "Decodes Mode S and ADS-B downlink frames from complex samples at exactly 2 MHz.",
    details: "Works on the magnitude m of the input (mag), after the 2 MHz demodulator of \
        dump1090. A preamble at sample p (pulses at p, p+2, p+7 and p+9) passes when m[p] > \
        m[p+1] < m[p+2] > m[p+3] < m[p], m[p+4], m[p+5] and m[p+6] < m[p], m[p+7] > m[p+8] < \
        m[p+9] > m[p+6], when m[p+4], m[p+5] and m[p+11] to m[p+14] lie below (m[p] + m[p+2] + \
        m[p+7] + m[p+9]) / 6, and when the mean of the four pulses is at least min_snr_db \
        above the median of the 256 samples before p (the noise level, the exact lower \
        median, tracked in a histogram of 1/32 octave bins). Bit k is 1 when m[p+16+2k] > \
        m[p+17+2k] (pulse-position modulation, 1 microsecond per bit); the first 5 bits are \
        the downlink format DF: DF 16 and above have 112 bits, below 56. Only DF 0, 4, 5, 11, \
        16, 17, 18, 20, 21 and 24 are decoded. DF 17 and 18 (extended squitter) are output \
        as 14 bytes with the flag CRC_OK when their CRC-24 residue is 0, DF 11 (all-call \
        reply) as 7 bytes with CRC_OK when its residue is 0; these frames also put their \
        address into an ICAO address cache (two tables flipped every 60 s of input, so an \
        address stays 60 to 120 s). The address/parity frames (DF 0, 4, 5, 16, 20, 21, 24 and \
        DF 11 with a nonzero residue) are output with the flag bit 16 \
        (ADSB_ADDRESS_PARITY) only when their residue, the address they were sent with, is in \
        the cache; everything else is dropped. When a candidate fails and its preamble shows \
        energy beside the pulses, the bits are decoded once more after the phase correction \
        of dump1090. After an output frame the search continues behind it, after a dropped \
        candidate at the next sample. The packet index is the input sample index of p, \
        counted since construction or reset; a frame is output once the 240 samples from p \
        are in. Pitfall: only 2 MHz is accepted (rate_unsupported otherwise); resample other \
        rates first. Pitfall: an address/parity frame is dropped until a DF 11, 17 or 18 \
        frame of the same aircraft has been received. No error correction and no position \
        decoding.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex baseband at exactly 2 MHz, centred on 1090 MHz.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "The frames: 14 bytes for the long formats, 7 bytes for the short ones.",
    }],
    params: &[ParamDesc {
        name: "min_snr_db",
        unit: Unit::Decibel,
        kind: ParamKind::F64 {
            min: 0.0,
            max: 60.0,
            default: Some(6.0),
        },
        typical: "6; 10 to suppress weak and false preambles",
        mutability: Mutability::Build,
        description: "How far the mean pulse amplitude of a preamble must lie above the median \
            magnitude of the 256 samples before it, in dB (amplitude ratio 10^(dB / 20)).",
    }],
    required: &[],
};

/// Configuration of [`AdsbDecodeBlock`].
///
/// ```
/// use sdrk_blocks::AdsbDecodeConfig;
/// let cfg = AdsbDecodeConfig::default();
/// assert_eq!(cfg.min_snr_db, 6.0);
/// assert!(cfg.validate(sdrk::Rate::hz(2_000_000)?).is_ok());
/// assert!(cfg.validate(sdrk::Rate::hz(2_400_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdsbDecodeConfig {
    /// Least preamble level above the noise median in dB (0 to 60).
    pub min_snr_db: f64,
}

impl Default for AdsbDecodeConfig {
    fn default() -> Self {
        Self { min_snr_db: 6.0 }
    }
}

impl AdsbDecodeConfig {
    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `OutOfRange` for a `min_snr_db` outside 0 to 60; `Core(Unsupported)` for a sample rate
    /// other than 2 MHz.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        if !(0.0..=60.0).contains(&self.min_snr_db) {
            return Err(Error::OutOfRange {
                name: "min_snr_db",
                min: 0.0,
                max: 60.0,
                got: self.min_snr_db,
            });
        }
        if rate_in != Rate::hz(RATE_HZ)? {
            return Err(Error::Core(sdrk_core::Error::Unsupported(
                "adsb_decode needs a sample rate of exactly 2 MHz",
            )));
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
            "min_snr_db" => self.min_snr_db = value.as_f64()?,
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
            "min_snr_db" => Ok(ParamValue::F64(self.min_snr_db)),
            _ => unknown(),
        }
    }
}

/// The histogram bin of a magnitude.
#[inline]
fn bin(x: f32) -> usize {
    if x >= BIN_MIN {
        let b = (x.to_bits() >> BIN_SHIFT).saturating_sub(BIN_BASE) as usize;
        b.min(BINS - 1)
    } else {
        0
    }
}

/// The smallest value of bin `b` (0 for bin 0).
fn bin_floor(b: usize) -> f32 {
    if b == 0 {
        0.0
    } else {
        f32::from_bits((b as u32 + BIN_BASE) << BIN_SHIFT)
    }
}

/// The mean of the four pulses when the preamble `w` passes the test of dump1090
/// (`demod_2000.c`, `demodulate2000`): the ordering of the first ten samples, and the samples
/// 4, 5 and 11 to 14 below a sixth of the sum of the pulses (D721). A NaN sample fails a
/// comparison and so the test.
#[inline]
fn preamble(w: &[f32; PREAMBLE]) -> Option<f32> {
    let [
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        _,
        p11,
        p12,
        p13,
        p14,
        _,
    ] = *w;
    if !(p0 > p1
        && p1 < p2
        && p2 > p3
        && p3 < p0
        && p4 < p0
        && p5 < p0
        && p6 < p0
        && p7 > p8
        && p8 < p9
        && p9 > p6)
    {
        return None;
    }
    let sum = ((p0 + p2) + p7) + p9;
    let high = sum / 6.0;
    let ok = p4 < high && p5 < high && p11 < high && p12 < high && p13 < high && p14 < high;
    ok.then_some(0.25 * sum)
}

/// dump1090 `detectOutOfPhase`: energy after the pulses 2 and 9 or before the pulses 7 and 0
/// (`prev` is the sample before the preamble `w`).
#[inline]
fn out_of_phase(w: &[f32; PREAMBLE], prev: f32) -> bool {
    let at = |k: usize| w.get(k).copied().unwrap_or(0.0);
    at(3) > at(2) / 3.0 || at(10) > at(9) / 3.0 || at(6) > at(7) / 3.0 || prev > at(1) / 3.0
}

/// dump1090 `applyPhaseCorrection` on `a`: `a[0]` is the sample before the preamble, `a[1 +
/// i]` the sample `p + i`. From the energy beside the preamble pulses it decides whether the
/// sampling is early or late and moves part of each bit sample to the neighbouring one, bit
/// by bit (from the end when late, from the start when early).
fn phase_correct(a: &mut [f32; 1 + FRAME_LEN]) {
    let at = |a: &[f32; 1 + FRAME_LEN], i: usize| a.get(i + 1).copied().unwrap_or(0.0);
    let on = at(a, 0) + at(a, 2) + at(a, 7) + at(a, 9);
    let early = 2.0 * (a.first().copied().unwrap_or(0.0) + at(a, 6));
    let late = 2.0 * (at(a, 3) + at(a, 10));
    let scale = |v: &mut f32, f: f32| *v *= f;
    if early > late {
        let r = early / (early + on);
        let (up, down) = (1.0 + r, 1.0 - r);
        if let Some(v) = a.get_mut(FRAME_LEN) {
            scale(v, up);
        }
        // Pairs (j, j + 1) from the last bit down to the second; a[j] is the sample j - 1.
        let mut j = FRAME_LEN - 2;
        while j > PREAMBLE {
            let one = at(a, j) > at(a, j + 1);
            if let Some(v) = a.get_mut(j) {
                scale(v, if one { down } else { up });
            }
            j -= 2;
        }
    } else {
        let r = late / (late + on);
        if !r.is_finite() {
            return;
        }
        let (up, down) = (1.0 + r, 1.0 - r);
        if let Some(v) = a.get_mut(PREAMBLE + 1) {
            scale(v, up);
        }
        let mut j = PREAMBLE;
        while j < FRAME_LEN - 2 {
            let one = at(a, j) > at(a, j + 1);
            if let Some(v) = a.get_mut(j + 3) {
                scale(v, if one { up } else { down });
            }
            j += 2;
        }
    }
}

/// The bits of the sample pairs `s` (bit 1 when the first sample is larger) into `out`,
/// most significant bit first, 8 pairs per byte.
fn bits_to_bytes(s: &[f32], out: &mut [u8]) {
    for (byte, chunk) in out.iter_mut().zip(s.chunks_exact(16)) {
        let mut v = 0u8;
        for pair in chunk.chunks_exact(2) {
            if let [a, b] = pair {
                v = (v << 1) | u8::from(a > b);
            }
        }
        *byte = v;
    }
}

/// dump1090 `icaoHash`: the Jenkins one-at-a-time hash of the three address bytes, reduced to
/// a slot of a table.
fn icao_hash(addr: u32) -> usize {
    let mut h = 0u32;
    for byte in [addr & 0xFF, (addr >> 8) & 0xFF, (addr >> 16) & 0xFF] {
        h = h.wrapping_add(byte);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h as usize & (ADSB_ICAO_SLOTS - 1)
}

/// The ICAO address cache of dump1090 (`icao_filter.c`): two open-addressed tables with linear
/// probing in caller storage; addresses go into the active one; at every flip, each
/// [`ADSB_ICAO_TTL`] input samples since construction or reset, the other table is cleared and
/// becomes the active one. A lookup searches both. The flips are applied lazily at the input
/// index of the frame that uses the cache, so the cache does not depend on the chunking (D722).
struct IcaoCache<B> {
    /// Both tables, `ADSB_ICAO_SLOTS` slots each; a used slot holds `ICAO_USED | address`.
    slots: B,
    /// Addresses in each table.
    count: [usize; 2],
    /// The active table (0 or 1).
    active: usize,
    /// Input index of the next flip.
    next_flip: u64,
}

impl<B: AsRef<[u32]> + AsMut<[u32]>> IcaoCache<B> {
    fn clear(&mut self) {
        self.slots.as_mut().fill(0);
        self.count = [0, 0];
        self.active = 0;
        self.next_flip = ADSB_ICAO_TTL;
    }

    fn table(&self, t: usize) -> &[u32] {
        let from = t * ADSB_ICAO_SLOTS;
        self.slots
            .as_ref()
            .get(from..from + ADSB_ICAO_SLOTS)
            .unwrap_or(&[])
    }

    /// Applies the flips due at input index `pos`.
    fn expire(&mut self, pos: u64) {
        if pos < self.next_flip {
            return;
        }
        let flips = (pos - self.next_flip) / ADSB_ICAO_TTL + 1;
        self.next_flip += flips * ADSB_ICAO_TTL;
        let tables = if flips >= 2 {
            0..2
        } else {
            1 - self.active..2 - self.active
        };
        for t in tables {
            let from = t * ADSB_ICAO_SLOTS;
            if let Some(table) = self.slots.as_mut().get_mut(from..from + ADSB_ICAO_SLOTS) {
                table.fill(0);
            }
            if let Some(c) = self.count.get_mut(t) {
                *c = 0;
            }
        }
        if flips % 2 == 1 {
            self.active = 1 - self.active;
        }
    }

    /// The slot of `addr` in table `t`, or the free slot where it would go (`None` when the
    /// table is full and does not hold it).
    fn probe(&self, t: usize, addr: u32) -> Option<(usize, bool)> {
        let table = self.table(t);
        let want = ICAO_USED | addr;
        let h0 = icao_hash(addr);
        (0..ADSB_ICAO_SLOTS).find_map(|k| {
            let i = (h0 + k) & (ADSB_ICAO_SLOTS - 1);
            match table.get(i).copied() {
                Some(0) => Some((i, false)),
                Some(v) if v == want => Some((i, true)),
                _ => None,
            }
        })
    }

    /// Whether `addr` is in either table, at input index `pos`.
    fn contains(&mut self, addr: u32, pos: u64) -> bool {
        self.expire(pos);
        (0..2).any(|t| matches!(self.probe(t, addr), Some((_, true))))
    }

    /// Puts `addr` into the active table at input index `pos`; a full table (half its slots
    /// used) takes no new address.
    fn add(&mut self, addr: u32, pos: u64) {
        self.expire(pos);
        let t = self.active;
        let count = self.count.get(t).copied().unwrap_or(ADSB_ICAO_CAPACITY);
        if let Some((i, false)) = self.probe(t, addr)
            && count < ADSB_ICAO_CAPACITY
        {
            if let Some(slot) = self.slots.as_mut().get_mut(t * ADSB_ICAO_SLOTS + i) {
                *slot = ICAO_USED | addr;
            }
            if let Some(c) = self.count.get_mut(t) {
                *c += 1;
            }
        }
    }
}

/// The second decoding of a candidate (D723): `x` holds the sample before the preamble and
/// the 240 samples of the candidate; a copy gets the phase correction and is decoded.
#[cold]
#[inline(never)]
fn retry<B: AsRef<[u32]> + AsMut<[u32]>>(
    x: &[f32],
    cache: &mut IcaoCache<B>,
    pos: u64,
    slot: &mut Packet,
) -> Result<Option<usize>> {
    let mut a = [0.0f32; 1 + FRAME_LEN];
    a.copy_from_slice(head(x, 1 + FRAME_LEN)?);
    phase_correct(&mut a);
    decode(tail(&a, 1 + PREAMBLE)?, cache, pos, slot)
}

/// The bit decisions, DF and CRC of the frame whose bit pairs are `s` (dump1090
/// `decodeModesMessage` without error correction, D722); an accepted frame of the candidate at
/// input index `pos` goes to `slot` and its length in samples is returned.
#[cold]
#[inline(never)]
fn decode<B: AsRef<[u32]> + AsMut<[u32]>>(
    s: &[f32],
    cache: &mut IcaoCache<B>,
    pos: u64,
    slot: &mut Packet,
) -> Result<Option<usize>> {
    let mut bytes = [0u8; LONG_FRAME_BYTES];
    bits_to_bytes(head(s, 16)?, head_mut(&mut bytes, 1)?);
    let df = adsb::df(&bytes);
    if VALID_DF & (1u32 << df) == 0 {
        return Ok(None);
    }
    let n = if df >= 16 {
        LONG_FRAME_BYTES
    } else {
        SHORT_FRAME_BYTES
    };
    let frame = head_mut(&mut bytes, n)?;
    bits_to_bytes(s, frame);
    // The syndrome (dump1090 `modesChecksum`): the CRC-24 of the frame without its last 24
    // bits, xor those bits; 0 for a correct parity, the address for an address/parity frame.
    let (body, parity) = frame.split_at(n - 3);
    let &[p0, p1, p2] = parity else {
        return Err(internal());
    };
    let residue = crc24_adsb(body) ^ u32::from_be_bytes([0, p0, p1, p2]);
    let flags = match df {
        17 | 18 | 11 if residue == 0 => {
            // A checked frame: its address is the sender's.
            cache.add(adsb::icao(frame), pos);
            PacketFlags::CRC_OK
        }
        17 | 18 => return Ok(None),
        _ if cache.contains(residue, pos) => ADSB_ADDRESS_PARITY,
        _ => return Ok(None),
    };
    *slot = Packet::new(pos, flags, frame)?;
    Ok(Some(PREAMBLE + 16 * n))
}

/// ADS-B decoder: `C32` at 2 MHz in, [`Packets`] out.
///
/// Per piece of at most [`ADSB_CHUNK`] input samples the block calls `mag` once into its
/// magnitude buffer; the search runs per sample in Rust: the preamble test, the update of the
/// noise histogram (one bin in, one out, the median bin moved, D710; the exact median is read
/// from the median bin only for a preamble that passes the other tests) and, for an accepted
/// preamble, the bit decisions, the CRC-24 and the lookup in the ICAO address cache (a
/// second decoding after the phase correction for a failed candidate that looks out of
/// phase). The buffer keeps the last 495 magnitudes (the noise window and one frame) and the
/// cache flips at fixed input indices, so the output does not depend on how the input is
/// split. The noise window starts filled with zeros and the cache empty. When a call fills the
/// last free output slot it stops right after that packet; the remaining candidates are
/// searched in the next call, also when it brings no input.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{AdsbDecodeBlock, AdsbDecodeConfig, ArenaStorage, Block, Packet};
/// use sdrk_proto::PacketFlags;
/// let rate = Rate::hz(2_000_000)?;
/// let cfg = AdsbDecodeConfig::default();
/// let bytes = AdsbDecodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut dec = AdsbDecodeBlock::new(&cfg, rate, &mut arena)?;
/// // A DF 17 frame at sample 300: preamble pulses, then one pulse per bit.
/// let frame = [0x8Du8, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98];
/// let mut x = vec![C32::new(0.01, 0.0); 1000];
/// for k in [0, 2, 7, 9] {
///     x[300 + k] = C32::new(0.5, 0.0);
/// }
/// for k in 0..112 {
///     let bit = (frame[k / 8] >> (7 - k % 8)) & 1;
///     x[316 + 2 * k + usize::from(bit == 0)] = C32::new(0.5, 0.0);
/// }
/// let mut out = [Packet::default(); 4];
/// let p = dec.process(&x, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (1000, 1));
/// assert_eq!(out[0].data(), &frame);
/// assert_eq!((out[0].index, out[0].flags), (300, PacketFlags::CRC_OK));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AdsbDecodeBlock<S: Storage> {
    config: AdsbDecodeConfig,
    /// `10^(min_snr_db / 20)`, rounded once to `f32`.
    factor: f32,
    /// Magnitudes; index `next` is the next candidate `p`.
    mag: S::Buf<f32>,
    /// Valid magnitudes in `mag`.
    filled: usize,
    /// Buffer index of the next candidate.
    next: usize,
    /// Input index of the next candidate.
    pos: u64,
    /// No preamble is tested before this input index (behind the last output frame).
    skip_until: u64,
    /// The noise histogram of the 256 magnitudes before the next candidate.
    bins: S::Buf<u16>,
    /// The bin of the median.
    med: usize,
    /// Values of the window in the bins below `med`.
    below: usize,
    /// The ICAO address cache.
    cache: IcaoCache<S::Buf<u32>>,
}

impl<S: Storage> core::fmt::Debug for AdsbDecodeBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AdsbDecodeBlock")
            .field("config", &self.config)
            .field("pos", &self.pos)
            .finish_non_exhaustive()
    }
}

fn internal() -> Error {
    Error::Core(sdrk_core::Error::Internal("adsb_decode window"))
}

impl<S: Storage> AdsbDecodeBlock<S> {
    /// Creates the block; the magnitude buffer and the histogram come from `storage`.
    ///
    /// # Errors
    /// As [`AdsbDecodeConfig::validate`]; `Storage` if the storage is too small.
    pub fn new(config: &AdsbDecodeConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let mag = storage.alloc::<f32>(BUF_LEN)?;
        let bins = storage.alloc::<u16>(BINS)?;
        let slots = storage.alloc::<u32>(2 * ADSB_ICAO_SLOTS)?;
        let mut block = Self {
            config: *config,
            factor: libm::pow(10.0, config.min_snr_db / 20.0) as f32,
            mag,
            filled: 0,
            next: 0,
            pos: 0,
            skip_until: 0,
            bins,
            med: 0,
            below: 0,
            cache: IcaoCache {
                slots,
                count: [0, 0],
                active: 0,
                next_flip: ADSB_ICAO_TTL,
            },
        };
        block.clear();
        Ok(block)
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`AdsbDecodeConfig::validate`].
    pub fn storage_bytes(config: &AdsbDecodeConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<f32>(BUF_LEN)?;
        plan.add::<u16>(BINS)?;
        plan.add::<u32>(2 * ADSB_ICAO_SLOTS)?;
        Ok(plan.bytes())
    }

    /// Input samples whose preamble test is done (the index of the next candidate).
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// The current noise level: the lower median (the 128th smallest value) of the 256
    /// magnitudes before the next candidate (NaN counts as 0).
    pub fn noise_level(&self) -> f32 {
        self.median(self.next).unwrap_or(0.0)
    }

    /// The exact lower median of the window before buffer index `j`: the values of the median
    /// bin are gathered (the histogram gives their rank offset) and the one of the remaining
    /// rank is selected. Called only for a preamble that passes the shape test and the coarse
    /// level test (D710).
    #[cold]
    #[inline(never)]
    fn median(&self, j: usize) -> Result<f32> {
        let from = j.checked_sub(MED_LEN).ok_or_else(internal)?;
        let window = head(tail(self.mag.as_ref(), from)?, MED_LEN)?;
        let mut vals = [0.0f32; MED_LEN];
        let mut n = 0usize;
        for &v in window {
            if bin(v) == self.med {
                *vals.get_mut(n).ok_or_else(internal)? = if v.is_nan() { 0.0 } else { v };
                n += 1;
            }
        }
        let k = MED_RANK.checked_sub(self.below + 1).ok_or_else(internal)?;
        let vals = vals.get_mut(..n).ok_or_else(internal)?;
        if k >= vals.len() {
            return Err(internal());
        }
        let (_, v, _) = vals.select_nth_unstable_by(k, f32::total_cmp);
        Ok(*v)
    }

    /// The state after construction: a window of 256 zeros, no candidate tested.
    fn clear(&mut self) {
        self.mag.as_mut().fill(0.0);
        self.filled = MED_LEN;
        self.next = MED_LEN;
        self.pos = 0;
        self.skip_until = 0;
        let bins = self.bins.as_mut();
        bins.fill(0);
        if let Some(b) = bins.first_mut() {
            *b = MED_LEN as u16;
        }
        self.med = 0;
        self.below = 0;
        self.cache.clear();
    }

    /// Moves the noise window by one sample: `add` comes in, `remove` goes out.
    #[inline]
    fn slide(&mut self, add: f32, remove: f32) -> Result<()> {
        let (a, r) = (bin(add), bin(remove));
        let bins = self.bins.as_mut();
        let ha = bins.get_mut(a).ok_or_else(internal)?;
        *ha = ha.saturating_add(1);
        let hr = bins.get_mut(r).ok_or_else(internal)?;
        *hr = hr.saturating_sub(1);
        if a < self.med {
            self.below += 1;
        }
        if r < self.med {
            self.below = self.below.saturating_sub(1);
        }
        // Restore below < MED_RANK <= below + bins[med].
        while self.below >= MED_RANK {
            self.med = self.med.checked_sub(1).ok_or_else(internal)?;
            let h = usize::from(*bins.get(self.med).ok_or_else(internal)?);
            self.below = self.below.saturating_sub(h);
        }
        loop {
            let h = usize::from(*bins.get(self.med).ok_or_else(internal)?);
            if self.below + h >= MED_RANK {
                break;
            }
            self.below += h;
            self.med += 1;
        }
        Ok(())
    }

    /// Tests the candidate at buffer index `j` (the window, the sample before it and the frame
    /// are in the buffer); writes the packet of an accepted frame to `slot` and returns its
    /// length in samples. A candidate that fails after the preamble test and the level test is
    /// decoded once more after the phase correction of dump1090 when its preamble looks out of
    /// phase (D723).
    #[inline]
    fn candidate(&mut self, j: usize, slot: &mut Packet) -> Result<Option<usize>> {
        let m = self.mag.as_ref();
        let w: &[f32; PREAMBLE] = head(tail(m, j)?, PREAMBLE)?
            .try_into()
            .map_err(|_| internal())?;
        let Some(level) = preamble(w) else {
            return Ok(None);
        };
        // The median is at least the lower edge of its bin: a level below that edge times the
        // factor fails without the exact median.
        if level
            .partial_cmp(&(bin_floor(self.med) * self.factor))
            .is_none_or(core::cmp::Ordering::is_lt)
            || level < self.median(j)? * self.factor
        {
            return Ok(None);
        }
        let bits = head(tail(m, j + PREAMBLE)?, 2 * LONG_BITS)?;
        if let Some(len) = decode(bits, &mut self.cache, self.pos, slot)? {
            return Ok(Some(len));
        }
        let from = j.checked_sub(1).ok_or_else(internal)?;
        let prev = *m.get(from).ok_or_else(internal)?;
        if !out_of_phase(w, prev) {
            return Ok(None);
        }
        retry(
            head(tail(m, from)?, 1 + FRAME_LEN)?,
            &mut self.cache,
            self.pos,
            slot,
        )
    }
}

impl<S: Storage> Block for AdsbDecodeBlock<S>
where
    Sdrk: Mag<C32>,
{
    type In = C32;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [Packet]) -> Result<Progress> {
        let mut consumed = 0usize;
        let mut produced = 0usize;
        if output.is_empty() {
            return Ok(Progress::default());
        }
        loop {
            while self.next + FRAME_LEN <= self.filled {
                let Some(slot) = output.get_mut(produced) else {
                    return Ok(Progress { consumed, produced });
                };
                let j = self.next;
                if self.pos >= self.skip_until
                    && let Some(len) = self.candidate(j, slot)?
                {
                    produced += 1;
                    self.skip_until = self.pos + len as u64;
                }
                let m = self.mag.as_ref();
                let add = *m.get(j).ok_or_else(internal)?;
                let remove = *m.get(j - MED_LEN).ok_or_else(internal)?;
                self.slide(add, remove)?;
                self.next += 1;
                self.pos += 1;
            }
            if consumed == input.len() || produced == output.len() {
                break;
            }
            if self.filled == BUF_LEN {
                // Keep the noise window of the next candidate and everything after it.
                let from = self.next - MED_LEN;
                self.mag.as_mut().copy_within(from..self.filled, 0);
                self.filled -= from;
                self.next -= from;
            }
            let n = (input.len() - consumed).min(BUF_LEN - self.filled);
            let x = head(tail(input, consumed)?, n)?;
            let dst = head_mut(
                self.mag
                    .as_mut()
                    .get_mut(self.filled..)
                    .ok_or_else(internal)?,
                n,
            )?;
            <Sdrk as Mag<C32>>::mag(&Sdrk, x, dst)?;
            self.filled += n;
            consumed += n;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let r = Ratio::new(1, (PREAMBLE + 2 * SHORT_BITS) as u64).unwrap_or(Ratio::ONE);
        BlockTiming {
            ratio: r,
            kind: RatioKind::Packets { max_per_input: r },
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.clear();
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn cache() -> IcaoCache<Vec<u32>> {
        let mut c = IcaoCache {
            slots: std::vec![0u32; 2 * ADSB_ICAO_SLOTS],
            count: [0, 0],
            active: 0,
            next_flip: 0,
        };
        c.clear();
        c
    }

    #[test]
    fn an_address_lives_60_to_120_s_of_input() {
        let mut c = cache();
        // Added at 0: kept over the first flip at 60 s, gone at the second at 120 s.
        c.add(0xAB_CDEF, 0);
        assert!(c.contains(0xAB_CDEF, ADSB_ICAO_TTL - 1));
        assert!(c.contains(0xAB_CDEF, 2 * ADSB_ICAO_TTL - 1));
        assert!(!c.contains(0xAB_CDEF, 2 * ADSB_ICAO_TTL));
        // Added just before a flip: kept for one TTL more only.
        let mut c = cache();
        c.add(0x00_0000, ADSB_ICAO_TTL - 1);
        assert!(c.contains(0x00_0000, 2 * ADSB_ICAO_TTL - 1));
        assert!(!c.contains(0x00_0000, 2 * ADSB_ICAO_TTL));
        // A long gap clears both tables at once.
        let mut c = cache();
        c.add(1, 10);
        c.add(2, ADSB_ICAO_TTL + 10);
        assert!(c.contains(1, ADSB_ICAO_TTL + 20) && c.contains(2, ADSB_ICAO_TTL + 20));
        assert!(!c.contains(2, 10 * ADSB_ICAO_TTL) && !c.contains(1, 10 * ADSB_ICAO_TTL));
        assert_eq!(c.next_flip, 11 * ADSB_ICAO_TTL);
        assert!(!c.contains(0x12_3456, 0));
    }

    #[test]
    fn a_full_table_takes_no_new_address() {
        let mut c = cache();
        for a in 0..ADSB_ICAO_CAPACITY as u32 {
            c.add(a * 7919, 5);
            c.add(a * 7919, 6);
        }
        assert_eq!(c.count, [ADSB_ICAO_CAPACITY, 0]);
        assert!((0..ADSB_ICAO_CAPACITY as u32).all(|a| c.contains(a * 7919, 7)));
        c.add(0xFF_FFFF, 8);
        assert!(!c.contains(0xFF_FFFF, 9));
        // After a flip the other table takes new addresses; the old ones stay.
        c.add(0xFF_FFFF, ADSB_ICAO_TTL);
        assert!(c.contains(0xFF_FFFF, ADSB_ICAO_TTL) && c.contains(7919, ADSB_ICAO_TTL));
        assert_eq!(c.count, [ADSB_ICAO_CAPACITY, 1]);
    }

    #[test]
    fn the_phase_correction_leaves_an_in_phase_frame_alone() {
        // Pulses only at the preamble and bit samples: no energy beside them, so the scale
        // factors are 1 (and an all-zero frame stays zero).
        let mut a = [0.0f32; 1 + FRAME_LEN];
        for k in [0usize, 2, 7, 9] {
            a[1 + k] = 0.5;
        }
        for k in 0..LONG_BITS {
            a[1 + PREAMBLE + 2 * k + k % 2] = 0.5;
        }
        let before = a;
        phase_correct(&mut a);
        assert_eq!(a, before);
        let w: [f32; PREAMBLE] = a[1..=PREAMBLE].try_into().unwrap();
        assert!(!out_of_phase(&w, 0.0));
        let mut z = [0.0f32; 1 + FRAME_LEN];
        phase_correct(&mut z);
        assert!(z.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn the_hash_is_that_of_dump1090() {
        // Jenkins one-at-a-time over the bytes low to high, as `icaoHash` of dump1090.
        let mut h: u32 = 0;
        for b in [0xEFu32, 0xCD, 0xAB] {
            h = h.wrapping_add(b);
            h = h.wrapping_add(h << 10);
            h ^= h >> 6;
        }
        h = h.wrapping_add(h << 3);
        h ^= h >> 11;
        h = h.wrapping_add(h << 15);
        assert_eq!(icao_hash(0xAB_CDEF), h as usize % ADSB_ICAO_SLOTS);
    }
}
