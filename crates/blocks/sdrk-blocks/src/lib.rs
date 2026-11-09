//! Blocks of SDRk: filter design, storage abstraction and self-describing signal-processing
//! blocks for the receive and transmit chains and for spectra.
//!
//! Layer `crates/blocks`, heap-free. A block is built from a config, the input sample rate and
//! a [`Storage`]; it uses the facade [`sdrk::Sdrk`] for every kernel call, so backends speed
//! up the blocks automatically. Every block kind has a machine-readable
//! [`BlockDescription`] (see [`all_descriptions`]). Packet ports ([`Packets`], [`Clocked`])
//! carry the [`Packet`]s of `sdrk-proto` (re-exported here).
//!
//! ```
//! use sdrk::{Rate, C32};
//! use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, DdcBlock, DdcConfig};
//!
//! // DDC from 2.4 MHz to 240 kHz around a carrier at +600 kHz.
//! let cfg = DdcConfig { frequency_hz: 600e3, decimation: 10, ..DdcConfig::default() };
//! let rate = Rate::hz(2_400_000)?;
//! let bytes = DdcBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
//! let mut mem = vec![0u8; bytes + 16];
//! let mut arena = ArenaStorage::new(&mut mem);
//! let mut ddc = DdcBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
//! let mut out = vec![C32::default(); 100];
//! let p = ddc.process(&[C32::new(0.5, 0.0); 1000], &mut out)?;
//! assert_eq!((p.consumed, p.produced), (1000, 100));
//! # let _ = AlignedBytes([0u8; 1]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![no_std]
#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]

mod block;
mod blocks;
mod chain;
pub mod describe;
pub mod design;
mod drift;
mod error;
mod fork;
mod param_map;
mod ports;
mod sample;
mod storage;

pub use block::{Block, BlockTiming, FrameMeta, ParamValue};
pub use blocks::add::{AddBlock, AddRealBlock};
pub use blocks::adsb_decode::{
    ADSB_ADDRESS_PARITY, ADSB_CHUNK, ADSB_ICAO_CAPACITY, ADSB_ICAO_SLOTS, ADSB_ICAO_TTL,
    AdsbDecodeBlock, AdsbDecodeConfig,
};
pub use blocks::afc::{AFC_PIECE, AfcBlock, AfcConfig};
pub use blocks::agc::{AgcBlock, AgcConfig};
pub use blocks::am_demod::{AM_DEMOD_CHUNK, AmDemodBlock, AmDemodConfig};
pub use blocks::am_mod::{AmModBlock, AmModConfig};
pub use blocks::auto_notch::{AutoNotchConfig, AutoNotchRealBlock};
pub use blocks::band_pulse_detect::{
    BAND_PULSE_CHUNK, BAND_PULSE_LEAD, BAND_PULSE_MAX_LANES, BAND_PULSE_MIN_RATE_HZ,
    BAND_PULSE_WARMUP, BandPulseDetectBlock, BandPulseDetectConfig, PulseModulation,
};
pub use blocks::biquad::{BiquadBlock, BiquadConfig, BiquadElem, BiquadRealBlock};
pub use blocks::channel_select::{ChannelSelectBlock, ChannelSelectConfig};
pub use blocks::channelize::{ChannelizeBlock, ChannelizeConfig};
pub use blocks::convert::{ConvertBlock, ConvertConfig, ConvertRealBlock};
pub use blocks::convert_out::{ConvertOutBlock, ConvertOutConfig};
pub use blocks::costas::{CostasBlock, CostasConfig};
pub use blocks::dc_block::{DcBlockBlock, DcBlockConfig, DcBlockRealBlock};
pub use blocks::ddc::{DdcBlock, DdcConfig};
pub use blocks::decimate::{DecimateBlock, DecimateConfig, DecimateRealBlock};
pub use blocks::deemphasis::{DeemphasisBlock, DeemphasisConfig, DeemphasisRealBlock};
pub use blocks::delay::{DelayBlock, DelayConfig, DelayRealBlock};
pub use blocks::duc::{DucBlock, DucConfig};
pub use blocks::equalize::{EqualizeBlock, EqualizeConfig};
pub use blocks::fir::{FirBlock, FirConfig};
pub use blocks::fm_demod::{FM_DEMOD_CHUNK, FmDemodBlock, FmDemodConfig};
pub use blocks::fm_mod::{FmModBlock, FmModConfig};
pub use blocks::fm_stereo_demod::{FM_STEREO_DEMOD_CHUNK, FmStereoDemodBlock, FmStereoDemodConfig};
pub use blocks::fm_stereo_mod::{FmStereoModBlock, FmStereoModConfig};
pub use blocks::fm_stereo_rds_mod::{FmStereoRdsModBlock, FmStereoRdsModConfig};
pub use blocks::frame_decode::{FrameDecodeBlock, FrameDecodeConfig, FrameRotations};
pub use blocks::frame_encode::{FrameEncodeBlock, FrameEncodeConfig, FrameIdle};
pub use blocks::fsk_demod::{
    FSK_DEMOD_CHUNK, FskDemodBlock, FskDemodConfig, TED_GAIN_2FSK, TED_GAIN_4FSK,
};
pub use blocks::fsk_mod::{FSK_MOD_CHUNK, FskModBlock, FskModConfig};
pub use blocks::halfband_decimate::{
    HalfbandDecimateBlock, HalfbandDecimateConfig, HalfbandDecimateRealBlock,
};
pub use blocks::hdlc_decode::{HdlcDecodeBlock, HdlcDecodeConfig};
pub use blocks::hdlc_encode::{HdlcEncodeBlock, HdlcEncodeConfig};
pub use blocks::imag_part::ImagPartBlock;
pub use blocks::interpolate::{InterpolateBlock, InterpolateConfig, InterpolateRealBlock};
pub use blocks::mag::{MagBlock, MagConfig};
pub use blocks::make_complex::MakeComplexBlock;
pub use blocks::morse_decode::{MorseDecodeBlock, MorseDecodeConfig};
pub use blocks::morse_encode::{MorseEncodeBlock, MorseEncodeConfig};
pub use blocks::mul::{MulBlock, MulRealBlock};
pub use blocks::nco::{NcoBlock, NcoConfig, NcoRealBlock};
pub use blocks::noise_blank::{NoiseBlankBlock, NoiseBlankConfig};
pub use blocks::ook_demod::{
    OOK_DEMOD_CHUNK, OOK_LOW_RATIO, OOK_MIN_SWING, OOK_THRESHOLD, OokDemodBlock, OokDemodConfig,
    TED_GAIN_OOK,
};
pub use blocks::ook_mod::{OOK_MOD_CHUNK, OokModBlock, OokModConfig};
pub use blocks::preemph_limit::{PreemphLimitBlock, PreemphLimitConfig, PreemphLimitElem};
pub use blocks::psk_demod::{PSK_DEMOD_CHUNK, PskDemodBlock, PskDemodConfig};
pub use blocks::psk_mod::{PSK_MOD_CHUNK, PskModBlock, PskModConfig, PskVariant};
pub use blocks::qam_demod::{QAM_DEMOD_CHUNK, QamDemodBlock, QamDemodConfig};
pub use blocks::qam_mod::{QAM_MOD_CHUNK, QamModBlock, QamModConfig};
pub use blocks::quad_demod::{QuadDemodBlock, QuadDemodConfig};
pub use blocks::rds_decode::{RdsDecodeBlock, RdsDecodeConfig};
pub use blocks::rds_demod::{
    RDS_DEMOD_CHUNK, RDS_DEMOD_MIN_RATE_HZ, RdsDemodBlock, RdsDemodConfig,
};
pub use blocks::rds_encode::{
    RDS_PACKET_GROUP, RDS_PACKET_PS, RDS_PACKET_RT, RdsEncodeBlock, RdsEncodeConfig,
};
pub use blocks::real_part::RealPartBlock;
pub use blocks::real_to_iq::RealToIqBlock;
pub use blocks::resample::{
    RESAMPLE_TUNED_PHASES, ResampleBlock, ResampleConfig, ResampleRealBlock,
};
pub use blocks::rotate::{RotateBlock, RotateConfig};
pub use blocks::scale::{ScaleBlock, ScaleConfig};
pub use blocks::sensor_decode::{
    SENSOR_ALL_DECODERS, SENSOR_CROSS_CHANNEL_S, SENSOR_SEEN, SENSOR_TRAIN_PAIRS,
    SensorDecodeBlock, SensorDecodeConfig,
};
pub use blocks::slice::{PSK8_GRAY, SliceBlock, SliceConfig, SliceConstellation, SliceElem};
pub use blocks::spectrum::{SpectrumBlock, SpectrumConfig, SpectrumSample, SpectrumWindow};
pub use blocks::squelch::{SquelchBlock, SquelchConfig};
pub use blocks::ssb_demod::{SSB_DEMOD_CHUNK, SsbDemodBlock, SsbDemodConfig, SsbSideband};
pub use blocks::ssb_mod::{SSB_MOD_CHUNK, SsbModBlock, SsbModConfig};
pub use blocks::stereo_matrix::StereoMatrixBlock;
pub use blocks::stereo_pll::{StereoPllBlock, StereoPllConfig};
pub use blocks::symbol_sync::{SymbolSyncBlock, SymbolSyncConfig};
pub use blocks::to_complex::ToComplexBlock;
pub use chain::{CHAIN_DESCRIPTION, Chain};
pub use describe::{
    BlockDescription, LaneElem, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
    all_descriptions,
};
pub use design::{BiquadResponse, DesignSpec, DesignWindow};
pub use drift::{DriftConfig, DriftController};
pub use error::{Error, Result};
pub use fork::{FORK_DESCRIPTION, Fork};
pub use param_map::ParamMap;
pub use ports::{
    ChainEnd, ChainItem, Clocked, Frame, FrameOut, Lanes, LanesMut, LanesRef, Packets, Ports,
    StreamElem, Two,
};
pub use sample::{BlockReal, BlockSample, TunedElem};
pub use sdrk_core::kernels::costas::CostasConstellation;
pub use sdrk_core::kernels::equalize::{EqConstellation, EqMode};
pub use sdrk_core::kernels::symbol_sync::Ted;
pub use sdrk_proto::frame::Whitening;
pub use sdrk_proto::{Packet, PacketFlags};
pub use storage::{ARENA_ALIGN, AlignedBytes, ArenaStorage, Storage};
