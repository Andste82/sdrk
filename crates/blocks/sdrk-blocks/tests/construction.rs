//! Construction of every block in an arena of exactly `storage_bytes` bytes, block sizes and
//! thread-safety of the public types.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::assert_is_empty,
    clippy::range_plus_one,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::multiple_bound_locations,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::print_stderr
)]

mod common;

use common::{check_exact_storage, rate};
use sdrk::{C32, CQ15, Q15, Rate};
use sdrk_blocks::{
    AgcBlock, AgcConfig, AlignedBytes, AmDemodBlock, AmDemodConfig, ArenaStorage, BiquadBlock,
    BiquadConfig, BiquadElem, BiquadRealBlock, BiquadResponse, Block, BlockSample, ConvertBlock,
    ConvertConfig, ConvertOutBlock, ConvertOutConfig, DcBlockBlock, DcBlockConfig,
    DcBlockRealBlock, DdcBlock, DdcConfig, DucBlock, DucConfig, Error, FirBlock, FirConfig,
    FmDemodBlock, FmDemodConfig, FmModBlock, FmModConfig, FmStereoDemodBlock, FmStereoDemodConfig,
    FmStereoModBlock, FmStereoModConfig, Frame, PreemphLimitBlock, PreemphLimitConfig,
    ResampleBlock, ResampleConfig, ResampleRealBlock, ScaleBlock, ScaleConfig, SpectrumBlock,
    SpectrumConfig, SpectrumSample, SpectrumWindow, SquelchBlock, SquelchConfig, SsbDemodBlock,
    SsbDemodConfig, SymbolDemapBlock, SymbolDemapConfig, SymbolDemapConstellation, SymbolDemapElem,
    SymbolSyncBlock, SymbolSyncConfig, Ted,
};
use sdrk_blocks::{
    CostasBlock, CostasConfig, FrameRotations, FskDemodBlock, IqOffsetBlock, IqOffsetConfig,
    IqPart, MorseDecodeBlock, MorseDecodeConfig, MorseEncodeBlock, MorseEncodeConfig,
    OokDemodBlock, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig, PskVariant,
    PulseMatchBlock, PulseMatchConfig, PulseShapeBlock, PulseShapeConfig, QamAcquireBlock,
    QamAcquireConfig, QamDemodBlock, QamDemodConfig, QamFrameDecodeBlock, QamFrameDecodeConfig,
    QamFrameEncodeBlock, QamFrameEncodeConfig, QamModBlock, QamModConfig, RdsDecodeBlock,
    RdsDecodeConfig, RdsDemodBlock, RdsDemodConfig, SymbolMapBlock, SymbolMapConfig,
    SymbolMapConstellation, SymbolMapVariant,
};
use sdrk_device::{SdrkError, formats};

type A<'a> = ArenaStorage<'a>;

fn convert_exact<T: BlockSample>()
where
    sdrk::Sdrk: sdrk_core::kernels::convert::Convert<T>,
{
    for format in [
        formats::CU8,
        formats::CS16,
        formats::PACKED12,
        formats::CF32,
    ] {
        let cfg = ConvertConfig { format };
        let r = rate(1_000_000);
        let bytes = ConvertBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 0);
        check_exact_storage(bytes, |a| ConvertBlock::<T, _>::new(&cfg, r, a).map(|_| ()));
    }
}

fn convert_out_exact<T: BlockSample>()
where
    sdrk::Sdrk: sdrk_core::kernels::convert::ConvertOut<T>,
{
    for format in [
        formats::CS8,
        formats::CS16,
        formats::PACKED10,
        formats::CF32,
    ] {
        let cfg = ConvertOutConfig { format };
        let r = rate(1_000_000);
        let bytes = ConvertOutBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| {
            ConvertOutBlock::<T, _>::new(&cfg, r, a).map(|_| ())
        });
    }
}

fn simple_exact<T: BlockSample>() {
    let r = rate(48_000);
    let cfg = ScaleConfig::default();
    check_exact_storage(
        ScaleBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap(),
        |a| ScaleBlock::<T, _>::new(&cfg, r, a).map(|_| ()),
    );
    let cfg = DcBlockConfig::default();
    check_exact_storage(
        DcBlockBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap(),
        |a| DcBlockBlock::<T, _>::new(&cfg, r, a).map(|_| ()),
    );
    let cfg = SquelchConfig::default();
    let bytes = SquelchBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
    assert_eq!(bytes, 0);
    check_exact_storage(bytes, |a| SquelchBlock::<T, _>::new(&cfg, r, a).map(|_| ()));
    let cfg = AgcConfig::default();
    let bytes = AgcBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
    assert_eq!(bytes, 0);
    check_exact_storage(bytes, |a| AgcBlock::<T, _>::new(&cfg, r, a).map(|_| ()));
    let cfg = FmModConfig::default();
    let r = rate(240_000);
    check_exact_storage(
        FmModBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap(),
        |a| FmModBlock::<T, _>::new(&cfg, r, a).map(|_| ()),
    );
}

fn filters_exact<T: BlockSample>() {
    let r = rate(48_000);
    let cfg = FirConfig {
        cutoff_hz: 4_000.0,
        transition_hz: 1_000.0,
        ..FirConfig::default()
    };
    let bytes = FirBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
    assert!(bytes > 0);
    check_exact_storage(bytes, |a| FirBlock::<T, _>::new(&cfg, r, a).map(|_| ()));

    let cfg = DdcConfig {
        decimation: 4,
        max_chunk: 100,
        ..DdcConfig::default()
    };
    let bytes = DdcBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
    check_exact_storage(bytes, |a| DdcBlock::<T, _>::new(&cfg, r, a).map(|_| ()));

    for (dc_cutoff_hz, audio_decimation) in [(30.0, 1), (0.0, 1), (30.0, 6), (0.0, 4)] {
        let cfg = AmDemodConfig {
            dc_cutoff_hz,
            audio_decimation,
            audio_cutoff_hz: 2_000.0,
        };
        let bytes = AmDemodBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| AmDemodBlock::<T, _>::new(&cfg, r, a).map(|_| ()));
    }

    for audio_decimation in [1, 4] {
        let cfg = SsbDemodConfig {
            audio_decimation,
            ..SsbDemodConfig::default()
        };
        let bytes = SsbDemodBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| {
            SsbDemodBlock::<T, _>::new(&cfg, r, a).map(|_| ())
        });
    }

    let cfg = DucConfig {
        interpolation: 3,
        max_chunk: 90,
        ..DucConfig::default()
    };
    let r = rate(16_000);
    let bytes = DucBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
    check_exact_storage(bytes, |a| DucBlock::<T, _>::new(&cfg, r, a).map(|_| ()));

    for decimation in [1, 5] {
        let cfg = FmDemodConfig {
            audio_decimation: decimation,
            ..FmDemodConfig::default()
        };
        let r = rate(240_000);
        let bytes = FmDemodBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| FmDemodBlock::<T, _>::new(&cfg, r, a).map(|_| ()));
    }
}

fn stereo_exact<T: BlockSample + BiquadElem>()
where
    T::Real: BiquadElem,
{
    for (decimation, deemphasis_us) in [(1, 50.0), (10, 50.0), (10, 0.0)] {
        let cfg = FmStereoDemodConfig {
            audio_decimation: decimation,
            deemphasis_us,
            ..FmStereoDemodConfig::default()
        };
        let r = rate(480_000);
        let bytes = FmStereoDemodBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| {
            FmStereoDemodBlock::<T, _>::new(&cfg, r, a).map(|_| ())
        });
    }
}

/// The biquad blocks of the complex type `T` and of the real type `R` take exactly the
/// storage they report: 5 coefficients and 4 history values per section.
fn biquad_exact<T: BlockSample + BiquadElem, R: sdrk_blocks::BlockReal + BiquadElem>() {
    let r = rate(48_000);
    for (response, order, cutoff_hz) in [
        (BiquadResponse::Lowpass, 2, 1_000.0),
        (BiquadResponse::Highpass, 4, 300.0),
        (BiquadResponse::Lowpass, 8, 2_000.0),
        (BiquadResponse::Bandpass, 2, 3_000.0),
        (BiquadResponse::Notch, 2, 1_000.0),
    ] {
        let cfg = BiquadConfig {
            response,
            order,
            cutoff_hz,
            ..BiquadConfig::default()
        };
        let bytes = BiquadBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert!(bytes > 0);
        check_exact_storage(bytes, |a| BiquadBlock::<T, _>::new(&cfg, r, a).map(|_| ()));
        let bytes = BiquadRealBlock::<R, A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert!(bytes > 0);
        check_exact_storage(bytes, |a| {
            BiquadRealBlock::<R, _>::new(&cfg, r, a).map(|_| ())
        });
    }
}

fn resample_blocks_exact<T: BlockSample>() {
    for (tunable, out_hz) in [(false, 32_000.0), (true, 44_100.0), (true, 48_000.0)] {
        let cfg = ResampleConfig {
            rate_out_hz: out_hz,
            tunable,
            max_taps: 4096,
            ..ResampleConfig::default()
        };
        let r = rate(48_000);
        let bytes = ResampleBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert!(bytes > 0);
        check_exact_storage(bytes, |a| {
            ResampleBlock::<T, _>::new(&cfg, r, a).map(|_| ())
        });
    }
}

fn spectrum_exact<T: SpectrumSample>()
where
    sdrk::Sdrk: sdrk_core::kernels::spectrum::Window<T>
        + sdrk_core::kernels::fft::Fft<T>
        + sdrk_core::kernels::spectrum::PowerDb<f32>,
{
    for (size, window) in [
        (16, SpectrumWindow::Hann),
        (256, SpectrumWindow::BlackmanHarris),
        (1024, SpectrumWindow::Kaiser),
    ] {
        let cfg = SpectrumConfig {
            fft_size: size,
            window,
            ..SpectrumConfig::default()
        };
        let r = rate(1_000_000);
        let bytes = SpectrumBlock::<T, A<'_>>::storage_bytes(&cfg, r).unwrap();
        check_exact_storage(bytes, |a| {
            SpectrumBlock::<T, _>::new(&cfg, r, a).map(|_| ())
        });
    }
}

#[test]
fn complex_blocks_c32() {
    convert_exact::<C32>();
    convert_out_exact::<C32>();
    simple_exact::<C32>();
    filters_exact::<C32>();
    biquad_exact::<C32, f32>();
    stereo_exact::<C32>();
    resample_blocks_exact::<C32>();
    spectrum_exact::<C32>();
}

#[test]
fn complex_blocks_cq15() {
    convert_exact::<CQ15>();
    convert_out_exact::<CQ15>();
    simple_exact::<CQ15>();
    filters_exact::<CQ15>();
    biquad_exact::<CQ15, Q15>();
    stereo_exact::<CQ15>();
    resample_blocks_exact::<CQ15>();
    spectrum_exact::<CQ15>();
}

/// `preemph_limit` needs no storage, with and without pre-emphasis (mono and stereo).
fn preemph_limit_exact<R: sdrk_blocks::PreemphLimitElem>() {
    for (preemphasis_us, rate_hz) in [(50.0, 240_000), (530.0, 48_000), (0.0, 8_000)] {
        let cfg = PreemphLimitConfig {
            preemphasis_us,
            ..PreemphLimitConfig::default()
        };
        let r = rate(rate_hz);
        let bytes = PreemphLimitBlock::<R, A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 0);
        check_exact_storage(bytes, |a| {
            PreemphLimitBlock::<R, _>::new(&cfg, r, a).map(|_| ())
        });
    }
}

fn dc_block_real_exact<R: sdrk_blocks::BlockReal>() {
    let r = rate(48_000);
    let cfg = DcBlockConfig::default();
    let bytes = DcBlockRealBlock::<R, A<'_>>::storage_bytes(&cfg, r).unwrap();
    assert_eq!(bytes, 0);
    check_exact_storage(bytes, |a| {
        DcBlockRealBlock::<R, _>::new(&cfg, r, a).map(|_| ())
    });
}

/// The blocks of M40 take exactly what `storage_bytes` reports (the arena is used up, one byte
/// less fails); `costas` needs none.
#[test]
fn psk_blocks_storage_is_exact() {
    let r = rate(9600);
    let cfg = CostasConfig::default();
    let bytes = CostasBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
    assert_eq!(bytes, 0);
    check_exact_storage(bytes, |a| CostasBlock::new(&cfg, r, a).map(|_| ()));
    // M40, then 8PSK and the variants of M40.1 (`offset` adds its ring of `sps / 2`)
    for (order, sps, span, differential, variant) in [
        (2, 2, 4, false, PskVariant::Plain),
        (4, 8, 10, true, PskVariant::Plain),
        (4, 64, 32, false, PskVariant::Plain),
        (8, 8, 10, false, PskVariant::Plain),
        (4, 8, 10, true, PskVariant::Pi4),
        (4, 2, 4, false, PskVariant::Offset),
        (4, 64, 32, false, PskVariant::Offset),
    ] {
        let cfg = PskModConfig {
            constellation: c_of(order),
            samples_per_symbol: sps,
            span,
            differential,
            variant,
            ..PskModConfig::default()
        };
        check_exact_storage(PskModBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap(), |a| {
            PskModBlock::new(&cfg, r, a).map(|_| ())
        });
    }
    for (order, symbol_rate_hz, fs, span, differential, variant) in [
        (2, 4800.0, 9600, 4, false, PskVariant::Plain),
        (4, 25_000.0, 200_000, 10, false, PskVariant::Plain),
        (4, 1200.0, 48_000, 32, false, PskVariant::Plain),
        (8, 25_000.0, 200_000, 10, false, PskVariant::Plain),
        (8, 1200.0, 48_000, 32, true, PskVariant::Plain),
        (4, 25_000.0, 200_000, 10, true, PskVariant::Pi4),
        (4, 25_000.0, 200_000, 10, false, PskVariant::Offset),
        (4, 50.0, 51_200, 4, false, PskVariant::Offset),
    ] {
        let cfg = PskDemodConfig {
            constellation: c_of(order),
            symbol_rate_hz,
            span,
            differential,
            variant,
            ..PskDemodConfig::default()
        };
        let r = rate(fs);
        check_exact_storage(
            PskDemodBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap(),
            |a| PskDemodBlock::new(&cfg, r, a).map(|_| ()),
        );
    }
    // M44.1: symbol_map (no storage) and iq_offset (its ring of half a symbol); M44.1.1:
    // pulse_shape and pulse_match (taps and history)
    for constellation in [
        SymbolMapConstellation::Bpsk,
        SymbolMapConstellation::Psk8,
        SymbolMapConstellation::Qam64,
    ] {
        let cfg = SymbolMapConfig {
            constellation,
            ..SymbolMapConfig::default()
        };
        let bytes = SymbolMapBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 0);
        check_exact_storage(bytes, |a| SymbolMapBlock::new(&cfg, r, a).map(|_| ()));
    }
    for (sps, part) in [(2, IqPart::Imag), (8, IqPart::Real), (1024, IqPart::Imag)] {
        let cfg = IqOffsetConfig {
            samples_per_symbol: sps,
            part,
        };
        let bytes = IqOffsetBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 4 * (sps / 2));
        check_exact_storage(bytes, |a| IqOffsetBlock::new(&cfg, r, a).map(|_| ()));
    }
    for correct in [false, true] {
        let cfg = QamAcquireConfig { correct };
        let bytes = QamAcquireBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 8 * (512 + 1024));
        check_exact_storage(bytes, |a| QamAcquireBlock::new(&cfg, r, a).map(|_| ()));
    }
    for (sps, span) in [(2, 4), (8, 10), (64, 32)] {
        let cfg = PulseShapeConfig {
            samples_per_symbol: sps,
            span,
            ..PulseShapeConfig::default()
        };
        let n = span * sps + 1;
        let bytes = PulseShapeBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 4 * n + 8 * n.div_ceil(sps));
        check_exact_storage(bytes, |a| PulseShapeBlock::new(&cfg, r, a).map(|_| ()));
    }
    for (fs, symbol_rate_hz, span) in [
        (9600, 4800.0, 4),
        (200_000, 25_000.0, 10),
        (1_024_000, 1000.0, 32),
    ] {
        let cfg = PulseMatchConfig {
            symbol_rate_hz,
            span,
            ..PulseMatchConfig::default()
        };
        let r = rate(fs);
        let n = span * (fs as usize / symbol_rate_hz as usize) + 1;
        let bytes = PulseMatchBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 4 * n + 8 * (n - 1));
        check_exact_storage(bytes, |a| PulseMatchBlock::new(&cfg, r, a).map(|_| ()));
    }
    // M44: qam_mod and qam_demod (with and without the equalizer)
    for (order, sps, span) in [(16, 8, 10), (64, 2, 4), (16, 64, 32)] {
        let cfg = QamModConfig {
            constellation: c_of(order),
            samples_per_symbol: sps,
            span,
            ..QamModConfig::default()
        };
        let r = rate(100_000);
        check_exact_storage(QamModBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap(), |a| {
            QamModBlock::new(&cfg, r, a).map(|_| ())
        });
    }
    for (order, symbol_rate_hz, fs, span, equalizer_taps) in [
        (16, 25_000.0, 200_000, 10, 11),
        (64, 25_000.0, 200_000, 10, 15),
        (16, 1200.0, 48_000, 32, 0),
        (64, 50.0, 51_200, 4, 63),
    ] {
        let cfg = QamDemodConfig {
            constellation: c_of(order),
            symbol_rate_hz,
            span,
            equalizer_taps,
            ..QamDemodConfig::default()
        };
        let r = rate(fs);
        check_exact_storage(
            QamDemodBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap(),
            |a| QamDemodBlock::new(&cfg, r, a).map(|_| ()),
        );
    }
    // M44.1 §7: the composites qam_frame_encode and qam_frame_decode
    for (order, sps) in [(16usize, 8usize), (64, 4)] {
        let mut cfg = QamFrameEncodeConfig::default();
        cfg.qam.constellation = c_of(order as u32);
        cfg.qam.samples_per_symbol = sps;
        let r = rate(100_000);
        check_exact_storage(
            QamFrameEncodeBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap(),
            |a| QamFrameEncodeBlock::new(&cfg, r, a).map(|_| ()),
        );
    }
    for (order, equalizer_taps) in [(16u32, 15usize), (64, 0)] {
        let mut cfg = QamFrameDecodeConfig::default();
        cfg.demod.constellation = c_of(order);
        cfg.demod.symbol_rate_hz = 25_000.0;
        cfg.demod.equalizer_taps = equalizer_taps;
        let r = rate(200_000);
        check_exact_storage(
            QamFrameDecodeBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap(),
            |a| QamFrameDecodeBlock::new(&cfg, r, a).map(|_| ()),
        );
    }
    // M41: rds_demod at three rates, rds_decode needs no storage
    for fs in [128_000u64, 240_000, 1_000_000] {
        let cfg = RdsDemodConfig::default();
        let r = rate(fs);
        check_exact_storage(
            RdsDemodBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap(),
            |a| RdsDemodBlock::new(&cfg, r, a).map(|_| ()),
        );
    }
    let cfg = RdsDecodeConfig::default();
    check_exact_storage(
        RdsDecodeBlock::<A<'_>>::storage_bytes(&cfg, rate(1187)).unwrap(),
        |a| RdsDecodeBlock::new(&cfg, rate(1187), a).map(|_| ()),
    );
}

#[test]
fn dc_block_real_blocks() {
    dc_block_real_exact::<f32>();
    dc_block_real_exact::<Q15>();
}

#[test]
fn preemph_limit_blocks() {
    preemph_limit_exact::<f32>();
    preemph_limit_exact::<Q15>();
    preemph_limit_exact::<C32>();
}

/// `fm_stereo_mod` needs no storage, at the lowest rate and at 192 kHz; below 120 kHz it is
/// rejected.
#[test]
fn fm_stereo_mod_blocks() {
    for rate_hz in [120_000, 192_000, 1_000_000] {
        let cfg = FmStereoModConfig::default();
        let r = rate(rate_hz);
        let bytes = FmStereoModBlock::<A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert_eq!(bytes, 0);
        check_exact_storage(bytes, |a| FmStereoModBlock::new(&cfg, r, a).map(|_| ()));
    }
    let cfg = FmStereoModConfig::default();
    assert!(matches!(
        FmStereoModBlock::<A<'_>>::storage_bytes(&cfg, rate(96_000)),
        Err(Error::OutOfRange {
            name: "rate_in",
            ..
        })
    ));
}

fn real_resample_blocks<R: sdrk_blocks::BlockReal>() {
    for (tunable, out_hz) in [(false, 12_000.0), (true, 48_000.0)] {
        let cfg = ResampleConfig {
            rate_out_hz: out_hz,
            tunable,
            ..ResampleConfig::default()
        };
        let r = rate(48_000);
        let bytes = ResampleRealBlock::<R, A<'_>>::storage_bytes(&cfg, r).unwrap();
        assert!(bytes > 0);
        check_exact_storage(bytes, |a| {
            ResampleRealBlock::<R, _>::new(&cfg, r, a).map(|_| ())
        });
    }
}

#[test]
fn real_resample_blocks_storage() {
    real_resample_blocks::<f32>();
    real_resample_blocks::<Q15>();
}

#[test]
fn an_unaligned_arena_needs_at_most_fifteen_more_bytes() {
    let r = rate(48_000);
    let cfg = FirConfig {
        cutoff_hz: 4_000.0,
        transition_hz: 1_000.0,
        ..FirConfig::default()
    };
    let bytes = FirBlock::<C32, A<'_>>::storage_bytes(&cfg, r).unwrap();
    for offset in 0..16 {
        let mut mem = AlignedBytes([0u8; 8192]);
        let mut arena = ArenaStorage::new(&mut mem.0[offset..offset + bytes + 15]);
        FirBlock::<C32, _>::new(&cfg, r, &mut arena).unwrap();
    }
}

#[test]
fn the_error_is_a_storage_error_of_kind_capacity_exceeded() {
    let r = rate(48_000);
    let cfg = FirConfig::default();
    let mut tiny = AlignedBytes([0u8; 16]);
    let mut arena = ArenaStorage::new(&mut tiny.0);
    let e = FirBlock::<C32, _>::new(&cfg, r, &mut arena).unwrap_err();
    assert!(matches!(e, Error::Storage { .. }));
    assert_eq!(e.kind(), sdrk_device::ErrorKind::CapacityExceeded);
}

/// The limit of a primitive block in bytes: all big data lives in the storage, the structs hold
/// only a few slices and counters. The limit follows the stack limit of 4 KiB of R3 (clippy
/// `large_stack_frames`): `chain!` holds up to 8 blocks inline, 8 x 512 bytes.
const BLOCK_LIMIT: usize = 512;
/// The limit of a composite block (a block that holds other blocks inline) in bytes, the
/// maintainer's decision on I44.1-1 (D1573).
const COMPOSITE_LIMIT: usize = 2048;

#[test]
fn no_block_is_larger_than_512_bytes() {
    fn check<T>(name: &str) {
        let size = std::mem::size_of::<T>();
        assert!(size <= BLOCK_LIMIT, "{name} is {size} bytes");
    }
    macro_rules! both {
        ($($b:ident),*) => {$(
            check::<$b<C32, A<'static>>>(concat!(stringify!($b), "<C32>"));
            check::<$b<CQ15, A<'static>>>(concat!(stringify!($b), "<CQ15>"));
            check::<$b<C32, sdrk_testkit::VecStorage>>(concat!(stringify!($b), "<C32, Vec>"));
            check::<$b<CQ15, sdrk_testkit::VecStorage>>(concat!(stringify!($b), "<CQ15, Vec>"));
        )*};
    }
    both!(
        ConvertBlock,
        ConvertOutBlock,
        ScaleBlock,
        DcBlockBlock,
        FirBlock,
        DdcBlock,
        DucBlock,
        FmDemodBlock,
        FmStereoDemodBlock,
        FmModBlock,
        ResampleBlock,
        SpectrumBlock,
        SquelchBlock,
        AgcBlock,
        AmDemodBlock,
        SsbDemodBlock,
        BiquadBlock,
        FmStereoDemodBlock
    );
    check::<PreemphLimitBlock<f32, A<'static>>>("PreemphLimitBlock<f32>");
    check::<PreemphLimitBlock<Q15, A<'static>>>("PreemphLimitBlock<Q15>");
    check::<PreemphLimitBlock<Q15, sdrk_testkit::VecStorage>>("PreemphLimitBlock<Q15, Vec>");
    check::<PreemphLimitBlock<C32, A<'static>>>("PreemphLimitBlock<C32>");
    check::<FmStereoModBlock<A<'static>>>("FmStereoModBlock");
    check::<SymbolSyncBlock<C32, A<'static>>>("SymbolSyncBlock<C32>");
    check::<SymbolSyncBlock<f32, sdrk_testkit::VecStorage>>("SymbolSyncBlock<f32, Vec>");
    check::<SymbolDemapBlock<C32, A<'static>>>("SymbolDemapBlock<C32>");
    check::<SymbolDemapBlock<f32, sdrk_testkit::VecStorage>>("SymbolDemapBlock<f32, Vec>");
    check::<CostasBlock<A<'static>>>("CostasBlock");
    check::<PskDemodBlock<A<'static>>>("PskDemodBlock");
    check::<PskDemodBlock<sdrk_testkit::VecStorage>>("PskDemodBlock<Vec>");
    check::<SymbolMapBlock<A<'static>>>("SymbolMapBlock");
    check::<SymbolMapBlock<sdrk_testkit::VecStorage>>("SymbolMapBlock<Vec>");
    check::<IqOffsetBlock<A<'static>>>("IqOffsetBlock");
    check::<IqOffsetBlock<sdrk_testkit::VecStorage>>("IqOffsetBlock<Vec>");
    check::<PulseShapeBlock<A<'static>>>("PulseShapeBlock");
    check::<PulseShapeBlock<sdrk_testkit::VecStorage>>("PulseShapeBlock<Vec>");
    check::<PulseMatchBlock<A<'static>>>("PulseMatchBlock");
    check::<PulseMatchBlock<sdrk_testkit::VecStorage>>("PulseMatchBlock<Vec>");
    check::<QamAcquireBlock<A<'static>>>("QamAcquireBlock");
    check::<QamAcquireBlock<sdrk_testkit::VecStorage>>("QamAcquireBlock<Vec>");
    check::<DcBlockRealBlock<f32, A<'static>>>("DcBlockRealBlock<f32>");
    check::<DcBlockRealBlock<Q15, sdrk_testkit::VecStorage>>("DcBlockRealBlock<Q15, Vec>");
    check::<BiquadRealBlock<f32, A<'static>>>("BiquadRealBlock<f32>");
    check::<BiquadRealBlock<Q15, A<'static>>>("BiquadRealBlock<Q15>");
    check::<BiquadRealBlock<f32, sdrk_testkit::VecStorage>>("BiquadRealBlock<f32, Vec>");
    check::<BiquadRealBlock<Q15, sdrk_testkit::VecStorage>>("BiquadRealBlock<Q15, Vec>");
    check::<ResampleRealBlock<f32, A<'static>>>("ResampleRealBlock<f32>");
    check::<ResampleRealBlock<Q15, A<'static>>>("ResampleRealBlock<Q15>");
    check::<ResampleRealBlock<Q15, sdrk_testkit::VecStorage>>("ResampleRealBlock<Q15, Vec>");
    // Every other primitive block (the building blocks of the composites among them: `equalize`,
    // `interpolate`, `frame_encode`, `frame_decode`), except `MorseDecodeBlock` (648 bytes on
    // `main` since M47, found by this list; open item I44.1-6 for the maintainer).
    macro_rules! complex {
        ($($b:ident),*) => {$(
            check::<sdrk_blocks::$b<C32, A<'static>>>(concat!(stringify!($b), "<C32>"));
            check::<sdrk_blocks::$b<CQ15, A<'static>>>(concat!(stringify!($b), "<CQ15>"));
        )*};
    }
    macro_rules! real {
        ($($b:ident),*) => {$(
            check::<sdrk_blocks::$b<f32, A<'static>>>(concat!(stringify!($b), "<f32>"));
            check::<sdrk_blocks::$b<Q15, A<'static>>>(concat!(stringify!($b), "<Q15>"));
        )*};
    }
    macro_rules! plain {
        ($($b:ident),*) => {$(
            check::<sdrk_blocks::$b<A<'static>>>(stringify!($b));
            check::<sdrk_blocks::$b<sdrk_testkit::VecStorage>>(concat!(stringify!($b), "<Vec>"));
        )*};
    }
    complex!(
        AddBlock,
        AmModBlock,
        DecimateBlock,
        DeemphasisBlock,
        HalfbandDecimateBlock,
        ImagPartBlock,
        InterpolateBlock,
        MagBlock,
        MakeComplexBlock,
        MulBlock,
        NcoBlock,
        NcoRealBlock,
        NoiseBlankBlock,
        NoiseBlock,
        QuadDemodBlock,
        RealPartBlock,
        RealToIqBlock,
        RotateBlock,
        ScrambleBlock,
        SsbModBlock,
        StereoMatrixBlock,
        StereoPllBlock,
        ToComplexBlock
    );
    real!(
        AddRealBlock,
        ConvertRealBlock,
        DecimateRealBlock,
        DeemphasisRealBlock,
        HalfbandDecimateRealBlock,
        InterpolateRealBlock,
        MulRealBlock,
        NoiseRealBlock
    );
    plain!(
        AdsbDecodeBlock,
        AdsbParseBlock,
        AfcBlock,
        AutoNotchRealBlock,
        BandPulseDetectBlock,
        ChannelizeBlock,
        EqualizeBlock,
        FmStereoRdsModBlock,
        FrameDecodeBlock,
        FrameEncodeBlock,
        FskModBlock,
        HdlcDecodeBlock,
        HdlcEncodeBlock,
        MorseEncodeBlock,
        OokModBlock,
        PulseKeyBlock,
        RdsDecodeBlock,
        RdsDemodBlock,
        SensorDecodeBlock,
        SensorEncodeBlock
    );
    check::<sdrk_blocks::ChannelSelectBlock>("ChannelSelectBlock");
    check::<sdrk_blocks::RdsEncodeBlock<f32, A<'static>>>("RdsEncodeBlock<f32>");
    check::<sdrk_blocks::RdsEncodeBlock<Q15, A<'static>>>("RdsEncodeBlock<Q15>");
}

/// The composite blocks, the blocks that hold other blocks inline (the explicit list of the
/// limit [`COMPOSITE_LIMIT`]; every other block is a primitive block within [`BLOCK_LIMIT`]).
#[test]
fn no_composite_block_is_larger_than_2048_bytes() {
    fn check<T>(name: &str) {
        let size = std::mem::size_of::<T>();
        assert!(size <= COMPOSITE_LIMIT, "{name} is {size} bytes");
        eprintln!("{name}: {size} bytes");
    }
    macro_rules! composites {
        ($($b:ident),*) => {$(
            check::<$b<A<'static>>>(stringify!($b));
            check::<$b<sdrk_testkit::VecStorage>>(concat!(stringify!($b), "<Vec>"));
        )*};
    }
    composites!(
        PskModBlock,
        QamModBlock,
        QamDemodBlock,
        QamFrameEncodeBlock,
        QamFrameDecodeBlock,
        FskDemodBlock,
        OokDemodBlock
    );
}

fn assert_send<T: Send>() {}

/// A lock policy for the send and sync checks of `LockWatch`.
struct Policy;

impl sdrk_blocks::LockPolicy for Policy {
    const CHECK: u16 = 4096;
    const TOLERANCE: f64 = 2e-4;
    const WEIGHT_CHECKS: u8 = 2;
    const GAP_WINDOWS: u8 = 6;
}
fn assert_sync<T: Sync>() {}

#[test]
fn public_types_are_send() {
    macro_rules! blocks {
        ($($b:ident),*) => {$(
            assert_send::<$b<C32, A<'static>>>();
            assert_send::<$b<CQ15, A<'static>>>();
            assert_send::<$b<CQ15, sdrk_testkit::VecStorage>>();
            assert_sync::<$b<C32, A<'static>>>();
        )*};
    }
    blocks!(
        ConvertBlock,
        ConvertOutBlock,
        ScaleBlock,
        DcBlockBlock,
        FirBlock,
        DdcBlock,
        DucBlock,
        FmDemodBlock,
        FmModBlock,
        ResampleBlock,
        SpectrumBlock,
        SquelchBlock,
        AgcBlock,
        AmDemodBlock,
        SsbDemodBlock,
        BiquadBlock
    );
    assert_send::<PreemphLimitBlock<f32, A<'static>>>();
    assert_send::<PreemphLimitBlock<Q15, sdrk_testkit::VecStorage>>();
    assert_sync::<PreemphLimitBlock<Q15, A<'static>>>();
    assert_send::<PreemphLimitConfig>();
    assert_send::<PreemphLimitBlock<C32, sdrk_testkit::VecStorage>>();
    assert_sync::<PreemphLimitBlock<C32, A<'static>>>();
    assert_send::<FmStereoModBlock<A<'static>>>();
    assert_send::<FmStereoModBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<FmStereoModBlock<A<'static>>>();
    assert_send::<FmStereoModConfig>();
    assert_sync::<FmStereoModConfig>();
    assert_send::<SymbolSyncBlock<C32, A<'static>>>();
    assert_send::<SymbolSyncBlock<f32, sdrk_testkit::VecStorage>>();
    assert_sync::<SymbolSyncBlock<C32, A<'static>>>();
    assert_sync::<SymbolSyncBlock<f32, A<'static>>>();
    assert_send::<SymbolSyncConfig>();
    assert_sync::<SymbolSyncConfig>();
    assert_send::<Ted>();
    assert_send::<SymbolDemapBlock<C32, A<'static>>>();
    assert_send::<SymbolDemapBlock<f32, sdrk_testkit::VecStorage>>();
    assert_sync::<SymbolDemapBlock<C32, A<'static>>>();
    assert_sync::<SymbolDemapBlock<f32, A<'static>>>();
    assert_send::<SymbolDemapConfig>();
    assert_send::<CostasBlock<A<'static>>>();
    assert_sync::<CostasBlock<sdrk_testkit::VecStorage>>();
    assert_send::<CostasConfig>();
    assert_sync::<CostasConfig>();
    assert_send::<PskModBlock<A<'static>>>();
    assert_send::<PskModBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<PskModBlock<A<'static>>>();
    assert_send::<PskModConfig>();
    assert_sync::<PskModConfig>();
    assert_send::<PskDemodBlock<A<'static>>>();
    assert_send::<PskDemodBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<PskDemodBlock<A<'static>>>();
    assert_send::<PskDemodConfig>();
    assert_sync::<PskDemodConfig>();
    assert_send::<PskVariant>();
    assert_send::<QamModBlock<A<'static>>>();
    assert_send::<QamModBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<QamModBlock<A<'static>>>();
    assert_send::<QamModConfig>();
    assert_sync::<QamModConfig>();
    assert_send::<QamDemodBlock<A<'static>>>();
    assert_send::<QamDemodBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<QamDemodBlock<A<'static>>>();
    assert_send::<QamDemodConfig>();
    assert_sync::<QamDemodConfig>();
    assert_send::<QamFrameEncodeBlock<A<'static>>>();
    assert_sync::<QamFrameEncodeBlock<sdrk_testkit::VecStorage>>();
    assert_send::<QamFrameEncodeConfig>();
    assert_send::<QamFrameDecodeBlock<A<'static>>>();
    assert_sync::<QamFrameDecodeBlock<sdrk_testkit::VecStorage>>();
    assert_send::<QamFrameDecodeConfig>();
    assert_send::<sdrk_blocks::FrameWatch>();
    assert_sync::<sdrk_blocks::FrameWatch>();
    assert_send::<RdsDemodBlock<A<'static>>>();
    assert_send::<RdsDemodBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<RdsDemodBlock<A<'static>>>();
    assert_send::<RdsDemodConfig>();
    assert_sync::<RdsDemodConfig>();
    assert_send::<RdsDecodeBlock<A<'static>>>();
    assert_send::<RdsDecodeBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<RdsDecodeBlock<A<'static>>>();
    assert_send::<RdsDecodeConfig>();
    assert_sync::<RdsDecodeConfig>();
    assert_send::<MorseEncodeBlock<A<'static>>>();
    assert_send::<MorseEncodeBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<MorseEncodeBlock<A<'static>>>();
    assert_send::<MorseDecodeBlock<A<'static>>>();
    assert_send::<MorseDecodeBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<MorseDecodeBlock<A<'static>>>();
    assert_send::<MorseEncodeConfig>();
    assert_sync::<MorseEncodeConfig>();
    assert_send::<MorseDecodeConfig>();
    assert_sync::<MorseDecodeConfig>();
    assert_sync::<PskVariant>();
    assert_send::<FrameRotations>();
    assert_sync::<FrameRotations>();
    assert_sync::<SymbolDemapConfig>();
    assert_send::<SymbolMapBlock<A<'static>>>();
    assert_send::<SymbolMapBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<SymbolMapBlock<A<'static>>>();
    assert_send::<SymbolMapConfig>();
    assert_sync::<SymbolMapConstellation>();
    assert_sync::<SymbolMapVariant>();
    assert_send::<IqOffsetBlock<A<'static>>>();
    assert_send::<IqOffsetBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<IqOffsetBlock<A<'static>>>();
    assert_sync::<IqOffsetConfig>();
    assert_send::<PulseShapeBlock<A<'static>>>();
    assert_send::<PulseShapeBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<PulseShapeBlock<A<'static>>>();
    assert_sync::<PulseShapeConfig>();
    assert_send::<PulseMatchBlock<A<'static>>>();
    assert_send::<PulseMatchBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<PulseMatchBlock<A<'static>>>();
    assert_sync::<PulseMatchConfig>();
    assert_send::<QamAcquireBlock<A<'static>>>();
    assert_send::<QamAcquireBlock<sdrk_testkit::VecStorage>>();
    assert_sync::<QamAcquireBlock<A<'static>>>();
    assert_send::<QamAcquireConfig>();
    assert_sync::<QamAcquireConfig>();
    assert_send::<sdrk_blocks::LockWatch<Policy>>();
    assert_sync::<sdrk_blocks::LockWatch<Policy>>();
    assert_send::<sdrk_blocks::LockState>();
    assert_send::<sdrk_blocks::LockTransition>();
    assert_send::<sdrk_blocks::AcquireWindow>();
    assert_sync::<IqPart>();
    assert_send::<SymbolDemapConstellation>();
    fn symbol_demap_elem<T: SymbolDemapElem + Send + Sync>() {}
    symbol_demap_elem::<f32>();
    symbol_demap_elem::<C32>();
    assert_send::<DcBlockRealBlock<f32, A<'static>>>();
    assert_send::<DcBlockRealBlock<Q15, sdrk_testkit::VecStorage>>();
    assert_sync::<DcBlockRealBlock<Q15, A<'static>>>();
    assert_send::<BiquadRealBlock<f32, A<'static>>>();
    assert_send::<BiquadRealBlock<Q15, sdrk_testkit::VecStorage>>();
    assert_sync::<BiquadRealBlock<Q15, A<'static>>>();
    assert_send::<BiquadConfig>();
    assert_send::<ResampleRealBlock<f32, A<'static>>>();
    assert_send::<ResampleRealBlock<Q15, sdrk_testkit::VecStorage>>();
    assert_send::<ConvertConfig>();
    assert_send::<ConvertOutConfig>();
    assert_send::<ScaleConfig>();
    assert_send::<DcBlockConfig>();
    assert_send::<FirConfig>();
    assert_send::<DdcConfig>();
    assert_send::<DucConfig>();
    assert_send::<FmDemodConfig>();
    assert_send::<FmStereoDemodConfig>();
    assert_send::<FmModConfig>();
    assert_send::<ResampleConfig>();
    assert_send::<SpectrumConfig>();
    assert_send::<SquelchConfig>();
    assert_send::<AgcConfig>();
    assert_send::<AmDemodConfig>();
    assert_send::<SsbDemodConfig>();
    assert_send::<Error>();
    assert_send::<A<'static>>();
    assert_send::<sdrk_blocks::BlockDescription>();
    assert_sync::<sdrk_blocks::BlockDescription>();
    assert_send::<sdrk_blocks::ParamValue>();
    assert_send::<sdrk_blocks::BlockTiming>();
    assert_send::<sdrk_blocks::FrameMeta>();
    assert_send::<sdrk_blocks::design::DesignSpec>();
    // The traits require Send of the blocks as well.
    fn is_block<B: Block>() {
        assert_send::<B>();
    }
    is_block::<FirBlock<C32, A<'static>>>();
    fn is_frame<B: Block<Out = Frame>>() {
        assert_send::<B>();
    }
    is_frame::<SpectrumBlock<CQ15, A<'static>>>();
    let _ = Rate::hz(1);
}

/// The constellation of a numeric order of the tables of this file (2, 4, 8, 16 or 64).
fn c_of(order: u32) -> SymbolMapConstellation {
    match order {
        2 => SymbolMapConstellation::Bpsk,
        4 => SymbolMapConstellation::Qpsk,
        8 => SymbolMapConstellation::Psk8,
        16 => SymbolMapConstellation::Qam16,
        64 => SymbolMapConstellation::Qam64,
        other => unreachable!("no constellation of the order {other}"),
    }
}
