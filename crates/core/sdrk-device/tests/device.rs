//! Table-driven tests of the `sdrk-device` public API.
#![allow(clippy::unwrap_used)]

use sdrk_device::formats;
use sdrk_device::*;

fn send_sync<T: Send + Sync>() {}

#[test]
fn rate_and_ratio() {
    let r = Rate::new(2_400_000, 1)
        .unwrap()
        .scale(Ratio::new(1, 10).unwrap())
        .unwrap();
    assert_eq!(r, Rate::new(240_000, 1).unwrap());
    assert_eq!(Rate::new(48_000, 1).unwrap(), Rate::new(96_000, 2).unwrap());
    assert_eq!(Rate::new(0, 1), Err(Error::InvalidRate { num: 0, den: 1 }));
    assert_eq!(Rate::new(1, 0), Err(Error::InvalidRate { num: 1, den: 0 }));
    assert!(Rate::hz(0).is_err());
    assert_eq!(
        Ratio::new(0, 1).unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        Rate::hz(3).unwrap().scale(Ratio::new(u64::MAX, 1).unwrap()),
        Err(Error::Overflow("rate scale"))
    );
    assert!((Rate::new(1, 4).unwrap().as_f64() - 0.25).abs() < 1e-12);
}

#[test]
fn ratio_ops() {
    let r = Ratio::new(1, 10)
        .unwrap()
        .mul(Ratio::new(1, 5).unwrap())
        .unwrap();
    assert_eq!((r.num(), r.den()), (1, 50));
    let t = Ratio::new(1, 10).unwrap();
    assert_eq!(t.map_index(0).unwrap(), 0);
    assert_eq!(t.map_index(9).unwrap(), 0);
    assert_eq!(t.map_index(10).unwrap(), 1);
    assert_eq!(t.map_index(1 << 63).unwrap(), (1u64 << 63) / 10);
    assert_eq!(
        Ratio::new(2, 1).unwrap().map_index(u64::MAX),
        Err(Error::Overflow("map index"))
    );
    let inv = t.inverse();
    assert_eq!((inv.num(), inv.den()), (10, 1));
    assert_eq!(Ratio::ONE.mul(t).unwrap(), t);
    assert!(
        Ratio::new(u64::MAX, 1)
            .unwrap()
            .mul(Ratio::new(u64::MAX, 1).unwrap())
            .is_err()
    );
}

#[test]
fn ns_round_trip() {
    let r = Rate::new(2_400_000, 1).unwrap();
    for secs in [1u64, 3600, 10 * 365 * 86_400] {
        let ns = secs * 1_000_000_000;
        let n = r.samples_in_ns(ns).unwrap();
        assert_eq!(n, secs * 2_400_000);
        assert_eq!(r.ns_for_samples(n).unwrap(), ns);
    }
    assert_eq!(r.ns_for_samples(1).unwrap(), 416);
    assert!(r.ns_for_samples(u64::MAX).is_err());
    let slow = Rate::new(1, u64::MAX).unwrap();
    assert!(slow.ns_for_samples(u64::MAX).is_err());
    assert!(
        Rate::new(u64::MAX, 1)
            .unwrap()
            .samples_in_ns(u64::MAX)
            .is_err()
    );
}

#[test]
fn time_of() {
    let a = TimeAnchor {
        index: SampleIndex(1_000_000),
        unix_ns: 1_700_000_000_000_000_000,
        source: TimeSource::Device,
    };
    let r = Rate::new(2_400_000, 1).unwrap();
    assert_eq!(
        a.time_of(SampleIndex(3_400_000), r).unwrap(),
        1_700_000_001_000_000_000
    );
    assert_eq!(a.time_of(SampleIndex(1_000_000), r).unwrap(), a.unix_ns);
    assert_eq!(
        a.time_of(SampleIndex(0), r).unwrap(),
        1_700_000_000_000_000_000 - 416_666_667
    );
    assert_eq!(
        a.time_of(SampleIndex(u64::MAX - 1), r),
        Err(Error::Overflow("time of"))
    );
    let slow = Rate::new(1, u64::MAX).unwrap();
    assert!(a.time_of(SampleIndex(u64::MAX), slow).is_err());
}

#[test]
fn sample_index() {
    assert_eq!(SampleIndex(u64::MAX).checked_add(1), None);
    assert_eq!(
        SampleIndex(u64::MAX).saturating_add(5),
        SampleIndex(u64::MAX)
    );
    assert_eq!(SampleIndex(9).checked_sub(SampleIndex(4)), Some(5));
    assert!(SampleIndex(1) < SampleIndex(2));
    assert_eq!(SampleIndex::default(), SampleIndex(0));
}

#[test]
fn byte_counts() {
    assert_eq!(formats::CU8.bytes_for_samples(1000).unwrap(), 2000);
    assert_eq!(formats::CS16.bytes_for_samples(1000).unwrap(), 4000);
    assert_eq!(formats::PACKED12.bytes_for_samples(1000).unwrap(), 3000);
    assert_eq!(formats::PACKED10.bytes_for_samples(1000).unwrap(), 2500);
    assert!(matches!(
        formats::PACKED10.bytes_for_samples(999),
        Err(Error::InvalidFormat(_))
    ));
    assert_eq!(formats::CF32.bytes_for_samples(3).unwrap(), 24);
    assert_eq!(formats::RF32.bytes_for_samples(3).unwrap(), 12);
    assert_eq!(formats::RS16.bytes_for_samples(3).unwrap(), 6);
    assert_eq!(formats::I2S32_MSB.bytes_for_samples(3).unwrap(), 24);
    assert_eq!(formats::CS16_PLANAR.bytes_for_samples(3).unwrap(), 12);
    let bits_c = SampleFormat::new(
        Container::Bits1,
        1,
        true,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Little,
    )
    .unwrap();
    let bits_r = SampleFormat::new(
        Container::Bits1,
        1,
        true,
        Layout::Real,
        Align::Lsb,
        Endian::Little,
    )
    .unwrap();
    assert_eq!(bits_c.bytes_for_samples(4).unwrap(), 1);
    assert!(bits_c.bytes_for_samples(3).is_err());
    assert_eq!(bits_r.bytes_for_samples(8).unwrap(), 1);
    assert!(bits_r.bytes_for_samples(4).is_err());
    let p12r = SampleFormat::new(
        Container::Packed12,
        12,
        true,
        Layout::Real,
        Align::Lsb,
        Endian::Little,
    )
    .unwrap();
    assert_eq!(p12r.bytes_for_samples(4).unwrap(), 6);
    assert!(p12r.bytes_for_samples(3).is_err());
    let p10r = SampleFormat::new(
        Container::Packed10,
        10,
        true,
        Layout::Real,
        Align::Lsb,
        Endian::Little,
    )
    .unwrap();
    assert_eq!(p10r.bytes_for_samples(4).unwrap(), 5);
    assert!(p10r.bytes_for_samples(2).is_err());
    assert!(matches!(
        formats::CS16.bytes_for_samples(usize::MAX),
        Err(Error::Overflow(_))
    ));
    assert!(matches!(
        formats::CU8.bytes_for_samples(usize::MAX / 2 + 1),
        Err(Error::Overflow(_))
    ));
    assert!(matches!(
        formats::CF32.bytes_for_samples(usize::MAX / 2),
        Err(Error::Overflow(_))
    ));
}

#[test]
fn samples_in_bytes() {
    assert_eq!(formats::CU8.samples_in_bytes(2001), 1000);
    assert_eq!(formats::CS16.samples_in_bytes(4003), 1000);
    assert_eq!(formats::PACKED12.samples_in_bytes(3002), 1000);
    assert_eq!(formats::PACKED10.samples_in_bytes(2504), 1000);
    assert_eq!(formats::PACKED10.samples_in_bytes(4), 0);
    assert_eq!(formats::RF32.samples_in_bytes(7), 1);
    assert_eq!(formats::CS16_PLANAR.samples_in_bytes(9), 2);
    assert_eq!(formats::CU8.samples_in_bytes(usize::MAX), usize::MAX / 2);
    // round trip for valid counts
    for f in [
        formats::CU8,
        formats::CS8,
        formats::CS16,
        formats::PACKED12,
        formats::PACKED10,
        formats::CF32,
        formats::RS16,
    ] {
        let b = f.bytes_for_samples(40).unwrap();
        assert_eq!(f.samples_in_bytes(b), 40);
    }
}

#[test]
fn full_scale() {
    assert_eq!(formats::CU8.full_scale_value(), 128.0);
    assert_eq!(formats::CS8.full_scale_value(), 128.0);
    assert_eq!(formats::CS16.full_scale_value(), 32768.0);
    assert_eq!(formats::CS12_LSB16.full_scale_value(), 2048.0);
    assert_eq!(formats::SC16_Q11.full_scale_value(), 2048.0);
    assert_eq!(formats::PACKED10.full_scale_value(), 512.0);
    assert_eq!(formats::I2S32_MSB.full_scale_value(), 8_388_608.0);
    assert_eq!(formats::CF32.full_scale_value(), 1.0);
    let big = SampleFormat::new(
        Container::I32,
        32,
        true,
        Layout::Real,
        Align::Lsb,
        Endian::Little,
    )
    .unwrap();
    assert_eq!(big.full_scale_value(), 2_147_483_648.0);
}

#[test]
fn constants_match_validating_constructor() {
    use Align::*;
    use Container::*;
    use Layout::*;
    let l = Endian::Little;
    let n = SampleFormat::new;
    assert_eq!(formats::CU8, n(U8, 8, false, Interleaved, Lsb, l).unwrap());
    assert_eq!(formats::CS8, n(I8, 8, true, Interleaved, Lsb, l).unwrap());
    assert_eq!(
        formats::CS16,
        n(I16, 16, true, Interleaved, Lsb, l).unwrap()
    );
    assert_eq!(
        formats::CS12_LSB16,
        n(I16, 12, true, Interleaved, Lsb, l).unwrap()
    );
    assert_eq!(
        formats::SC16_Q11,
        n(I16, 12, true, Interleaved, Lsb, l)
            .unwrap()
            .with_full_scale(FullScale::Explicit(2048))
            .unwrap()
    );
    assert_eq!(
        formats::CS16_PLANAR,
        n(I16, 16, true, Planar, Lsb, l).unwrap()
    );
    assert_eq!(
        formats::PACKED12,
        n(Packed12, 12, true, Interleaved, Lsb, l).unwrap()
    );
    assert_eq!(
        formats::PACKED10,
        n(Packed10, 10, true, Interleaved, Lsb, l).unwrap()
    );
    assert_eq!(
        formats::I2S32_MSB,
        n(I32, 24, true, Interleaved, Msb, l).unwrap()
    );
    assert_eq!(
        formats::CF32,
        n(F32, 32, true, Interleaved, Lsb, l).unwrap()
    );
    assert_eq!(formats::RF32, n(F32, 32, true, Real, Lsb, l).unwrap());
    assert_eq!(formats::RS16, n(I16, 16, true, Real, Lsb, l).unwrap());
}

#[test]
fn getters_and_builders() {
    let f = formats::SC16_Q11.with_swap_iq(true);
    assert!(f.swap_iq());
    assert_eq!(f.container(), Container::I16);
    assert_eq!(f.bits(), 12);
    assert!(f.signed());
    assert_eq!(f.layout(), Layout::Interleaved);
    assert_eq!(f.align(), Align::Lsb);
    assert_eq!(f.endian(), Endian::Little);
    assert_eq!(f.full_scale(), FullScale::Explicit(2048));
    assert_eq!(formats::CU8.values_per_sample(), 2);
    assert_eq!(formats::RF32.values_per_sample(), 1);
    let be = SampleFormat::new(
        Container::I16,
        16,
        true,
        Layout::Real,
        Align::Lsb,
        Endian::Big,
    )
    .unwrap();
    assert_eq!(be.endian(), Endian::Big);
}

#[test]
fn invalid_combinations() {
    use Align::*;
    use Container::*;
    use Layout::*;
    let l = Endian::Little;
    let n = SampleFormat::new;
    assert!(n(U8, 8, true, Interleaved, Lsb, l).is_err());
    assert!(n(I16, 16, false, Interleaved, Lsb, l).is_err());
    assert!(n(I8, 8, false, Interleaved, Lsb, l).is_err());
    assert!(n(I32, 32, false, Interleaved, Lsb, l).is_err());
    assert!(n(F32, 16, true, Interleaved, Lsb, l).is_err());
    assert!(n(F32, 32, false, Interleaved, Lsb, l).is_err());
    assert!(n(Packed12, 12, true, Planar, Lsb, l).is_err());
    assert!(n(Packed10, 10, true, Planar, Lsb, l).is_err());
    assert!(n(Bits1, 1, true, Planar, Lsb, l).is_err());
    assert!(n(Packed12, 10, true, Interleaved, Lsb, l).is_err());
    assert!(n(Packed10, 12, true, Interleaved, Lsb, l).is_err());
    assert!(n(Bits1, 2, true, Interleaved, Lsb, l).is_err());
    assert!(n(U8, 9, false, Interleaved, Lsb, l).is_err());
    assert!(n(I8, 9, true, Interleaved, Lsb, l).is_err());
    assert!(n(I16, 17, true, Interleaved, Lsb, l).is_err());
    assert!(n(I32, 33, true, Interleaved, Lsb, l).is_err());
    assert!(n(I16, 0, true, Interleaved, Lsb, l).is_err());
    assert_eq!(
        formats::CS16.with_full_scale(FullScale::Explicit(0)),
        Err(Error::InvalidFormat("explicit full scale must be positive"))
    );
}

#[test]
fn block_flags() {
    let mut f = BlockFlags::empty();
    assert_eq!(f, BlockFlags::default());
    assert_eq!(f.bits(), 0);
    f.insert(BlockFlags::GAP_BEFORE);
    f.insert(BlockFlags::LATE);
    assert!(f.contains(BlockFlags::GAP_BEFORE.union(BlockFlags::LATE)));
    assert!(!f.contains(BlockFlags::UNDERRUN));
    assert!(f.contains(BlockFlags::empty()));
    f.remove(BlockFlags::GAP_BEFORE);
    assert!(!f.contains(BlockFlags::GAP_BEFORE));
    assert_eq!(f, BlockFlags::LATE);
    let all = [
        BlockFlags::GAP_BEFORE,
        BlockFlags::OVERFLOW,
        BlockFlags::UNDERRUN,
        BlockFlags::BURST_START,
        BlockFlags::BURST_END,
        BlockFlags::LATE,
        BlockFlags::RETUNED,
    ];
    let mut acc = 0u16;
    for a in all {
        assert_eq!(a.bits().count_ones(), 1);
        assert_eq!(acc & a.bits(), 0);
        acc |= a.bits();
    }
    assert_eq!(BlockFlags::from_bits_truncate(0xFFFF).bits(), acc);
    assert_eq!(
        BlockFlags::from_bits_truncate(BlockFlags::LATE.bits()),
        BlockFlags::LATE
    );
    let m = BlockMeta::default();
    assert_eq!(m.start, SampleIndex(0));
    assert_eq!(m.anchor, None);
}

#[test]
fn send_sync_all() {
    send_sync::<ErrorKind>();
    send_sync::<Error>();
    send_sync::<Container>();
    send_sync::<Layout>();
    send_sync::<Align>();
    send_sync::<Endian>();
    send_sync::<FullScale>();
    send_sync::<SampleFormat>();
    send_sync::<SampleIndex>();
    send_sync::<Rate>();
    send_sync::<Ratio>();
    send_sync::<TimeSource>();
    send_sync::<TimeAnchor>();
    send_sync::<BlockFlags>();
    send_sync::<BlockMeta>();
}

#[test]
fn error_display_and_kind() {
    let cases = [
        (Error::InvalidFormat("x y"), ErrorKind::InvalidArgument),
        (
            Error::InvalidRate { num: 0, den: 1 },
            ErrorKind::InvalidArgument,
        ),
        (Error::Overflow("z"), ErrorKind::OutOfRange),
        (
            Error::BufferTooSmall { needed: 4, got: 2 },
            ErrorKind::BufferTooSmall,
        ),
    ];
    for (e, k) in cases {
        assert_eq!(e.kind(), k);
        let s = e.to_string();
        assert_ne!(s, "");
        assert!(!s.ends_with('.'));
        assert_eq!(s, s.to_lowercase(), "{s}");
        assert!(core::error::Error::source(&e).is_none());
    }
    assert_eq!(
        Error::BufferTooSmall { needed: 4, got: 2 }.to_string(),
        "buffer too small: needed 4 bytes, got 2"
    );
}

#[test]
fn error_kind_display() {
    let all = [
        (ErrorKind::InvalidArgument, "invalid_argument"),
        (ErrorKind::LengthMismatch, "length_mismatch"),
        (ErrorKind::BufferTooSmall, "buffer_too_small"),
        (ErrorKind::OutOfRange, "out_of_range"),
        (ErrorKind::Unsupported, "unsupported"),
        (ErrorKind::CapacityExceeded, "capacity_exceeded"),
        (ErrorKind::InvalidState, "invalid_state"),
        (ErrorKind::ResourceLimit, "resource_limit"),
        (ErrorKind::Parse, "parse"),
        (ErrorKind::Validation, "validation"),
        (ErrorKind::Io, "io"),
        (ErrorKind::Device, "device"),
        (ErrorKind::Timeout, "timeout"),
        (ErrorKind::Disconnected, "disconnected"),
        (ErrorKind::Internal, "internal"),
    ];
    for (k, s) in all {
        assert_eq!(k.to_string(), s);
    }
}

struct Loop;
impl RxStream for Loop {
    type Error = Error;
    fn format(&self) -> SampleFormat {
        formats::CU8
    }
    fn sample_rate(&self) -> Rate {
        Rate::hz(1000).unwrap()
    }
    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Error> {
        let n = formats::CU8.samples_in_bytes(buf.len()) * 2;
        buf.iter_mut().take(n).for_each(|b| *b = 7);
        Ok((n, BlockMeta::default()))
    }
}
impl TxStream for Loop {
    type Error = Error;
    fn format(&self) -> SampleFormat {
        formats::CU8
    }
    fn sample_rate(&self) -> Rate {
        Rate::hz(1000).unwrap()
    }
    fn write(&mut self, buf: &[u8], _m: &BlockMeta) -> Result<usize, Error> {
        Ok(buf.len())
    }
    fn poll_status(&mut self) -> Result<Option<BlockMeta>, Error> {
        Ok(None)
    }
}

#[test]
fn stream_traits_are_implementable() {
    let mut s = Loop;
    let mut b = [0u8; 5];
    let (n, m) = s.read(&mut b).unwrap();
    assert_eq!((n, m), (4, BlockMeta::default()));
    assert_eq!(TxStream::write(&mut s, &b, &m).unwrap(), 5);
    assert_eq!(s.poll_status().unwrap(), None);
    assert_eq!(RxStream::sample_rate(&s), TxStream::sample_rate(&s));
}
