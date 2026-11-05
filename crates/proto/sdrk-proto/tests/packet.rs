//! `Packet`, `PacketFlags`, the error type and the `Send`/`Sync` assertions of `sdrk-proto`
//! (R4.6, R5).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use sdrk_device::{ErrorKind, SdrkError};
use sdrk_proto::frame::{Decoder, FrameBits};
use sdrk_proto::{Error, MAX_PACKET_BYTES, Packet, PacketFlags, ProtoError};

#[test]
fn new_accepts_up_to_max_packet_bytes() {
    for len in [0, 1, 255, MAX_PACKET_BYTES] {
        let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let p = Packet::new(42, PacketFlags::CRC_OK, &data).unwrap();
        assert_eq!(p.data(), data.as_slice());
        assert_eq!(p.len(), len);
        assert_eq!(p.is_empty(), len == 0);
        assert_eq!(p.index, 42);
        assert_eq!(p.flags, PacketFlags::CRC_OK);
    }
    let err = Packet::new(0, PacketFlags::NONE, &[0; MAX_PACKET_BYTES + 1]).unwrap_err();
    assert_eq!(
        err,
        Error::TooLong {
            len: MAX_PACKET_BYTES + 1,
            max: MAX_PACKET_BYTES
        }
    );
    assert_eq!(err.kind(), ErrorKind::OutOfRange);
    assert_eq!(err.to_string(), "too long: 257 bytes, at most 256 allowed");
    assert!(Packet::new(0, PacketFlags::NONE, &vec![0; 70_000]).is_err());
}

#[test]
fn equality_is_semantic() {
    let a = Packet::new(1, PacketFlags::NONE, b"ab").unwrap();
    let b = Packet::new(1, PacketFlags::NONE, b"ab").unwrap();
    assert_eq!(a, b);
    assert_ne!(a, Packet::new(1, PacketFlags::NONE, b"ab\0").unwrap());
    assert_ne!(a, Packet::new(2, PacketFlags::NONE, b"ab").unwrap());
    assert_eq!(Packet::default(), Packet::EMPTY);
    assert!(Packet::EMPTY.is_empty());
    assert_eq!(
        format!("{a:?}"),
        "Packet { index: 1, flags: PacketFlags(0x0), data: [97, 98], .. }"
    );
}

#[test]
fn flags() {
    assert_eq!(PacketFlags::CRC_OK.bits(), 1);
    assert_eq!(PacketFlags::CRC_FAIL.bits(), 2);
    assert_eq!(PacketFlags::CORRECTED.bits(), 4);
    assert_eq!(PacketFlags::TRUNCATED.bits(), 8);
    let mut f = PacketFlags::from_bits(0x1_0000);
    f |= PacketFlags::TRUNCATED;
    assert!(f.contains(PacketFlags::TRUNCATED));
    assert!(f.contains(PacketFlags::from_bits(0x1_0000)));
    assert!(!f.contains(PacketFlags::CRC_OK | PacketFlags::TRUNCATED));
    assert!(f.contains(PacketFlags::NONE));
    assert!(PacketFlags::default().is_empty());
    assert_eq!(f.0, 0x1_0008);
}

#[test]
fn error_kinds_and_messages() {
    let e: ProtoError = Error::BufferTooSmall { needed: 3, got: 2 };
    assert_eq!(e.kind(), ErrorKind::BufferTooSmall);
    assert_eq!(e.to_string(), "buffer too small: needed 3, got 2");
    let e = Error::InvalidArgument("x");
    assert_eq!(e.kind(), ErrorKind::InvalidArgument);
    assert_eq!(e.to_string(), "invalid argument: x");
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Packet>();
    assert_send_sync::<PacketFlags>();
    assert_send_sync::<Error>();
    assert_send_sync::<Decoder>();
    assert_send_sync::<FrameBits>();
}

#[test]
fn packet_fits_a_small_stack() {
    // A packet is copied by value through ports and queues; keep it small (R3.4).
    assert!(core::mem::size_of::<Packet>() <= 280);
}
