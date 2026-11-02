//! Suite case of M27: the block `Chain` on the target.
//!
//! `chain_fm_receiver` runs the `fm_receiver` flow of the examples (`convert` from `cu8`, `ddc`
//! at +600 kHz with 100 kHz cutoff and decimation 10, `fm_demod` with 75 kHz deviation, 50 us
//! de-emphasis and audio decimation 5) as one `chain!` in `Q15` precision on 0.05 s of the FM
//! test signal of M08 and compares a digest of the audio with [`CHAIN_FM_RECEIVER_DIGEST`], the
//! digest of the same flow run by the pipeline on the host
//! (`crates/pipeline/sdrk-pipeline/tests/chain_digest.rs`).

#![allow(clippy::too_many_lines)]

use num_complex::Complex64;
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, Block, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig,
    FmDemodBlock, FmDemodConfig, chain,
};
use sdrk_core::{CQ15, Q15, Rate};
use sdrk_device::formats;

use super::ensure;
use crate::arena::Arena;
use crate::e2e::{
    BASEBAND_RATE_HZ, CARRIER_OFFSET_HZ, CHUNK, DEEMPHASIS_TAU_S, DEVIATION_HZ, IF_RATE_HZ,
    fm_source,
};
use crate::suite::Failure;

const CASE: &str = "chain_fm_receiver";

/// Samples of the case: 0.05 s at 2.4 MS/s.
pub const CHAIN_FM_RECEIVER_SAMPLES: usize = 120_000;
/// Capacity of the intermediate buffers of the chain in samples.
const CHAIN_CHUNK: usize = 256;
/// Number of audio samples the flow produces for [`CHAIN_FM_RECEIVER_SAMPLES`].
pub const CHAIN_FM_RECEIVER_AUDIO: usize = CHAIN_FM_RECEIVER_SAMPLES / 50;
/// FNV-1a (64 bit) of the little-endian bytes of the `Q15` audio the pipeline produces for the
/// case, see the module documentation.
pub const CHAIN_FM_RECEIVER_DIGEST: u64 = 0x65fb_0151_2026_34f4;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over the bytes of `Q15` samples (little-endian, two bytes each), continuing `hash`.
pub fn digest_q15(mut hash: u64, samples: &[Q15]) -> u64 {
    for s in samples {
        for byte in s.0.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// The start value of [`digest_q15`].
pub const DIGEST_START: u64 = FNV_OFFSET;

/// Runs the chain and returns the number of audio samples and their digest.
pub fn chain_fm_receiver_digest(arena: &mut Arena<'_>) -> Result<(usize, u64), Failure> {
    let if_rate = Rate::hz(IF_RATE_HZ)?;
    let bb_rate = Rate::hz(BASEBAND_RATE_HZ)?;
    let ddc_cfg = DdcConfig {
        frequency_hz: CARRIER_OFFSET_HZ,
        decimation: 10,
        cutoff_hz: Some(100_000.0),
        max_chunk: CHAIN_CHUNK,
        ..DdcConfig::default()
    };
    let demod_cfg = FmDemodConfig {
        deviation_hz: DEVIATION_HZ,
        deemphasis_us: DEEMPHASIS_TAU_S * 1e6,
        audio_decimation: 5,
        ..FmDemodConfig::default()
    };
    let mut c = arena.carver();
    let bytes_needed = DdcBlock::<CQ15, ArenaStorage<'_>>::storage_bytes(&ddc_cfg, if_rate)?
        + FmDemodBlock::<CQ15, ArenaStorage<'_>>::storage_bytes(&demod_cfg, bb_rate)?
        + 2 * CHAIN_CHUNK * core::mem::size_of::<CQ15>()
        + 6 * ARENA_ALIGN;
    let mut storage = ArenaStorage::new(c.take::<u8>(bytes_needed)?);
    let convert = ConvertBlock::<CQ15, _>::new(
        &ConvertConfig {
            format: formats::CU8,
        },
        if_rate,
        &mut storage,
    )?;
    let ddc = DdcBlock::<CQ15, _>::new(&ddc_cfg, if_rate, &mut storage)?;
    let demod = FmDemodBlock::<CQ15, _>::new(&demod_cfg, bb_rate, &mut storage)?;
    let mut rx = chain!(&mut storage, CHAIN_CHUNK; convert, ddc, demod);

    let scratch = c.take::<Complex64>(CHUNK)?;
    let mut source = fm_source(scratch, formats::CU8);
    let bytes = c.take::<u8>(2 * CHUNK)?;
    let out = c.take::<Q15>(CHUNK / 50 + 4)?;
    let (mut count, mut hash) = (0usize, DIGEST_START);
    for block in 0..CHAIN_FM_RECEIVER_SAMPLES / CHUNK {
        source((block * CHUNK) as u64, bytes)?;
        let mut rest: &[u8] = bytes;
        while !rest.is_empty() {
            let p = rx.process(rest, out)?;
            if p.consumed == 0 && p.produced == 0 {
                return Err(Failure::new(CASE, "chain made no progress"));
            }
            let made = out.get(..p.produced).unwrap_or_default();
            hash = digest_q15(hash, made);
            count += p.produced;
            rest = rest.get(p.consumed..).unwrap_or_default();
        }
    }
    Ok((count, hash))
}

pub(super) fn chain_fm_receiver(arena: &mut Arena<'_>) -> Result<(), Failure> {
    let (count, hash) = chain_fm_receiver_digest(arena)?;
    ensure(count == CHAIN_FM_RECEIVER_AUDIO, || {
        Failure::new(CASE, "wrong audio sample count").detail(format_args!("{count}"))
    })?;
    ensure(hash == CHAIN_FM_RECEIVER_DIGEST, || {
        Failure::new(CASE, "audio differs from the pipeline run on the host")
            .detail(format_args!("digest {hash:016x}"))
    })
}
