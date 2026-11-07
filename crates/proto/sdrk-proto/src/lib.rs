//! Protocol logic of SDRk without signal processing: the packet type, CRCs, a simple test
//! framing, PN9 data whitening, the PRBS generators of the bit error rate tests, the Radio
//! Data System (`rds`: checkwords, syndromes, groups, programme service name and radiotext),
//! NRZI and HDLC framing with the AX.25 address header (APRS), the fields and parity of
//! Mode S frames (ADS-B), and Morse code (`morse`: the table, an encoder and a decoder).
//!
//! Layer `crates/proto`, heap-free; depends on `sdrk-device` only. Digital modes end in bytes:
//! a decoder turns bits into [`Packet`]s, an encoder turns packets back into bits. The blocks
//! of `sdrk-blocks` and the pipeline carry these packets on packet ports.
//!
//! ```
//! use sdrk_proto::{crc, frame, Packet, PacketFlags, MAX_PACKET_BYTES};
//!
//! let p = Packet::new(0, PacketFlags::NONE, b"123456789")?;
//! assert_eq!(crc::crc16_ccitt_false(p.data()), 0x29B1);
//! assert!(p.len() <= MAX_PACKET_BYTES);
//! assert_eq!(frame::frame_bits(p.len(), 4), 8 * (4 + 4 + 1 + 9 + 2));
//! # Ok::<(), sdrk_proto::Error>(())
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

pub mod adsb;
pub mod ax25;
pub mod crc;
mod error;
pub mod frame;
pub mod hdlc;
pub mod morse;
pub mod nrzi;
mod packet;
pub mod prbs;
pub mod rds;
pub mod whiten;

pub use error::{Error, ProtoError, Result};
pub use packet::{MAX_PACKET_BYTES, Packet, PacketFlags};
