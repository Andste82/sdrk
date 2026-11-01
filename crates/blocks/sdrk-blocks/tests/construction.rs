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
    clippy::cast_possible_truncation
)]

mod common;

use common::{check_exact_storage, rate};
use sdrk::{C32, CQ15, Q15, Rate};
use sdrk_blocks::{
    AgcBlock, AgcConfig, AlignedBytes, AmDemodBlock, AmDemodConfig, ArenaStorage, BiquadBlock,
    BiquadConfig, BiquadElem, BiquadRealBlock, BiquadResponse, Block, BlockSample, ConvertBlock,
    ConvertConfig, ConvertOutBlock, ConvertOutConfig, DcBlockBlock, DcBlockConfig, DdcBlock,
    DdcConfig, DucBlock, DucConfig, Error, FirBlock, FirConfig, FmDemodBlock, FmDemodConfig,
    FmModBlock, FmModConfig, FmStereoDemodBlock, FmStereoDemodConfig, FrameBlock, ResampleBlock,
    ResampleConfig, ResampleRealBlock, ScaleBlock, ScaleConfig, SpectrumBlock, SpectrumConfig,
    SpectrumSample, SpectrumWindow, SquelchBlock, SquelchConfig, SsbDemodBlock, SsbDemodConfig,
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

#[test]
fn no_block_is_larger_than_512_bytes() {
    // All big data lives in the storage; the structs hold only a few slices and counters.
    fn check<T>(name: &str) {
        let size = std::mem::size_of::<T>();
        assert!(size <= 512, "{name} is {size} bytes");
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
    check::<BiquadRealBlock<f32, A<'static>>>("BiquadRealBlock<f32>");
    check::<BiquadRealBlock<Q15, A<'static>>>("BiquadRealBlock<Q15>");
    check::<BiquadRealBlock<f32, sdrk_testkit::VecStorage>>("BiquadRealBlock<f32, Vec>");
    check::<BiquadRealBlock<Q15, sdrk_testkit::VecStorage>>("BiquadRealBlock<Q15, Vec>");
    check::<ResampleRealBlock<f32, A<'static>>>("ResampleRealBlock<f32>");
    check::<ResampleRealBlock<Q15, A<'static>>>("ResampleRealBlock<Q15>");
    check::<ResampleRealBlock<Q15, sdrk_testkit::VecStorage>>("ResampleRealBlock<Q15, Vec>");
}

fn assert_send<T: Send>() {}
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
    fn is_frame<B: FrameBlock>() {
        assert_send::<B>();
    }
    is_frame::<SpectrumBlock<CQ15, A<'static>>>();
    let _ = Rate::hz(1);
}
