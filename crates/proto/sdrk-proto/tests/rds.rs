//! `sdrk_proto::rds` (M41 test 1): checkwords and syndromes of all offsets, a known group of the
//! reference signal (`rds/reference_short`), the assembly of PS and RT in `RdsState`, the A/B
//! flag and the character mapping.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use sdrk_proto::rds::{
    self, GENERATOR, Group, Offset, RdsState, VALID_SHIFT, VERSION_B, Version, map_char,
};
use sdrk_proto::{Packet, PacketFlags};
use sdrk_testkit::golden::{self, Array};

/// `a(x) b(x)` over GF(2).
fn clmul(a: u128, b: u128) -> u128 {
    (0..64)
        .filter(|i| b >> i & 1 != 0)
        .fold(0, |r, i| r ^ (a << i))
}

/// `v(x) mod g(x)`.
fn pmod(mut v: u128) -> u16 {
    let g = u128::from(GENERATOR);
    while v >> 10 != 0 {
        let top = v.ilog2();
        v ^= g << (top - 10);
    }
    v as u16
}

/// `x^e mod g(x)`.
fn xpow(e: u32) -> u128 {
    (0..e).fold(1u128, |r, _| u128::from(pmod(r << 1)))
}

#[test]
fn offset_words_and_syndromes_of_all_five_offsets() {
    let table = [
        (Offset::A, 0x0FC, 0x3D8),
        (Offset::B, 0x198, 0x3D4),
        (Offset::C, 0x168, 0x25C),
        (Offset::CPrime, 0x350, 0x3CC),
        (Offset::D, 0x1B4, 0x258),
    ];
    for (o, word, syn) in table {
        assert_eq!(o.word(), word);
        assert_eq!(o.syndrome(), syn);
        // The syndrome of the offset word alone, and of every valid block.
        assert_eq!(rds::syndrome(u32::from(word)), syn, "{o:?}");
        for info in [0u16, 1, 0x8000, 0xD123, 0xFFFF, 0x2140] {
            let block = rds::block26(info, o);
            assert_eq!(block >> 10, u32::from(info));
            assert_eq!(rds::syndrome(block), syn, "{o:?} {info:#x}");
            assert_eq!(Offset::from_syndrome(rds::syndrome(block)), Some(o));
        }
    }
    assert_eq!(Offset::ALL.len(), 5);
}

#[test]
fn checkword_is_the_remainder_plus_the_offset_word() {
    for info in (0..=u16::MAX).step_by(251) {
        for o in Offset::ALL {
            let want = pmod(u128::from(info) << 10) ^ o.word();
            assert_eq!(rds::checkword(info, o), want);
        }
    }
    // The plain remainder of a valid block is its offset word, not the listed syndrome.
    let block = rds::block26(0xD123, Offset::A);
    assert_eq!(pmod(u128::from(block)), 0x0FC);
}

#[test]
fn syndrome_is_the_block_times_x_to_the_325() {
    // x^325 is the inverse of x^16 modulo g (g has the order 341).
    assert_eq!(pmod(clmul(xpow(325), xpow(16))), 1);
    assert_eq!(xpow(341), 1);
    let x325 = xpow(325);
    let mut v: u32 = 0x2AB_CDEF;
    for _ in 0..2000 {
        v = v.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let block = v & 0x3FF_FFFF;
        assert_eq!(rds::syndrome(block), pmod(clmul(u128::from(block), x325)));
        // Bits above the block are ignored.
        assert_eq!(rds::syndrome(block | 0xFC00_0000), rds::syndrome(block));
    }
}

#[test]
fn every_burst_of_up_to_ten_bits_is_detected() {
    let block = rds::block26(0xD123, Offset::B);
    for len in 1..=10u32 {
        let first = 1u32 << (len - 1);
        for pattern in first..(1 << len) {
            if pattern & 1 == 0 {
                continue;
            }
            for shift in 0..=26 - len {
                let bad = block ^ (pattern << shift);
                assert_ne!(rds::syndrome(bad), Offset::B.syndrome());
            }
        }
    }
}

#[test]
fn offset_order() {
    assert_eq!(Offset::A.next(), Offset::B);
    assert_eq!(Offset::B.next(), Offset::C);
    assert_eq!(Offset::C.next(), Offset::D);
    assert_eq!(Offset::CPrime.next(), Offset::D);
    assert_eq!(Offset::D.next(), Offset::A);
    for o in Offset::ALL {
        assert!(o.next().may_follow(o));
    }
    assert!(Offset::CPrime.may_follow(Offset::B));
    assert!(!Offset::CPrime.may_follow(Offset::A));
    assert!(!Offset::A.may_follow(Offset::A));
    let pos: Vec<usize> = Offset::ALL.iter().map(|o| o.position()).collect();
    assert_eq!(pos, [0, 1, 2, 2, 3]);
}

fn reference_groups() -> Vec<[u16; 4]> {
    let case = golden::load("rds", "reference_short").unwrap();
    let Array::I32(blocks) = case.array("blocks").unwrap() else {
        panic!("blocks must be i32");
    };
    blocks
        .chunks_exact(4)
        .map(|g| core::array::from_fn(|i| g[i] as u16))
        .collect()
}

#[test]
fn a_known_group_of_the_reference_parses_to_pi_pty_and_a_ps_segment() {
    let groups = reference_groups();
    assert_eq!(groups.len(), 32);
    // Group 2 of the carousel: 0A, segment 1 "RK".
    let g = Group::new(groups[2]);
    assert_eq!(g.pi(), 0xD123);
    assert_eq!(g.group_type(), (0, Version::A));
    assert_eq!(g.pty(), 10);
    assert!(!g.tp());
    assert_eq!(g.blocks[1] & 0x3, 1);
    assert_eq!(g.blocks[2], 0xE0E0);
    assert_eq!(g.blocks[3].to_be_bytes(), *b"RK");
    // Group 3: 2A, segment 1 " fro".
    let g = Group::new(groups[3]);
    assert_eq!(g.group_type(), (2, Version::A));
    assert_eq!(g.blocks[1] & 0xF, 1);
    let mut s = RdsState::new();
    for g in &groups {
        s.update(&Group::new(*g));
    }
    assert_eq!(s.ps(), "SDRK TST");
    assert_eq!(s.rt(), "Hello from sdrk!");
    assert_eq!(s.pi(), Some(0xD123));
    assert_eq!(s.pty(), Some(10));
    assert_eq!(s.tp(), Some(false));
}

fn ps_group(pi: u16, seg: u16, pair: [u8; 2]) -> Group {
    Group::new([pi, 0x0148 | seg, 0xE0E0, u16::from_be_bytes(pair)])
}

fn rt_group(seg: u16, ab: bool, text: [u8; 4]) -> Group {
    let b = 0x2140 | if ab { 0x10 } else { 0 } | seg;
    Group::new([
        0xD123,
        b,
        u16::from_be_bytes([text[0], text[1]]),
        u16::from_be_bytes([text[2], text[3]]),
    ])
}

#[test]
fn ps_needs_all_four_segments_in_any_order() {
    let mut s = RdsState::new();
    for (seg, pair) in [(3, *b"ST"), (0, *b"SD"), (2, *b" T")] {
        s.update(&ps_group(0xD123, seg, pair));
        assert_eq!(s.ps(), "");
        assert!(!s.ps_ready());
    }
    s.update(&ps_group(0xD123, 1, *b"RK"));
    assert_eq!(s.ps(), "SDRK TST");
    // A segment with an invalid block D is ignored; a valid one replaces the characters.
    let mut bad = ps_group(0xD123, 0, *b"XX");
    bad.valid[3] = false;
    s.update(&bad);
    assert_eq!(s.ps(), "SDRK TST");
    s.update(&ps_group(0xD123, 0, *b"AB"));
    assert_eq!(s.ps(), "ABRK TST");
    // Group 0B carries PS as well (block C' repeats the PI).
    s.update(&Group::new([
        0xD123,
        0x0948 | 1,
        0xD123,
        u16::from_be_bytes(*b"CD"),
    ]));
    assert_eq!(s.ps(), "ABCD TST");
}

#[test]
fn rt_assembles_and_ends_at_the_carriage_return() {
    let text = *b"Hello from sdrk!\r                                               ";
    let mut s = RdsState::new();
    for seg in (0..16).rev() {
        assert_eq!(s.rt(), "", "before segment {seg}");
        let at = 4 * seg as usize;
        s.update(&rt_group(seg, false, text[at..at + 4].try_into().unwrap()));
    }
    assert_eq!(s.rt(), "Hello from sdrk!");
    // Only the segments up to the carriage return are needed.
    let mut s = RdsState::new();
    for seg in 0..5u16 {
        assert!(!s.rt_ready());
        let at = 4 * seg as usize;
        s.update(&rt_group(seg, false, text[at..at + 4].try_into().unwrap()));
    }
    assert!(s.rt_ready());
    assert_eq!(s.rt(), "Hello from sdrk!");
    // Without a carriage return all 16 segments make 64 characters.
    let mut s = RdsState::new();
    for seg in 0..16u16 {
        s.update(&rt_group(seg, false, *b"abcd"));
    }
    assert_eq!(s.rt().len(), 64);
    // A group 2A with an invalid block C is ignored.
    let mut s = RdsState::new();
    let mut g = rt_group(0, false, *b"Hi\r ");
    g.valid[2] = false;
    s.update(&g);
    assert_eq!(s.rt(), "");
    g.valid[2] = true;
    s.update(&g);
    assert_eq!(s.rt(), "Hi");
}

#[test]
fn the_ab_flag_clears_the_radiotext() {
    let mut s = RdsState::new();
    s.update(&rt_group(0, false, *b"Old "));
    s.update(&rt_group(1, false, *b"text"));
    s.update(&rt_group(2, false, *b"\r   "));
    assert_eq!(s.rt(), "Old text");
    // Segment 0 of a new text (flag B): segments 1 and 2 of the old one no longer count.
    s.update(&rt_group(0, true, *b"New "));
    assert_eq!(s.rt(), "");
    s.update(&rt_group(1, true, *b"one\r"));
    assert_eq!(s.rt(), "New one");
    // The same flag again keeps the text.
    s.update(&rt_group(1, true, *b"two\r"));
    assert_eq!(s.rt(), "New two");
}

#[test]
fn group_2b_carries_two_characters_per_segment() {
    let mut s = RdsState::new();
    for (seg, pair) in [(0u16, *b"Ab"), (1, *b"c\r")] {
        s.update(&Group::new([
            0xD123,
            0x2940 | seg,
            0xD123,
            u16::from_be_bytes(pair),
        ]));
    }
    assert_eq!(s.rt(), "Abc");
    // Switching to 2A clears the 2B text.
    s.update(&rt_group(1, false, *b"wxyz"));
    assert_eq!(s.rt(), "");
}

#[test]
fn characters_outside_ascii_become_question_marks() {
    assert_eq!(map_char(0x20), b' ');
    assert_eq!(map_char(0x7E), b'~');
    for b in [0x00, 0x0D, 0x1F, 0x7F, 0x80, 0xE0, 0xFF] {
        assert_eq!(map_char(b), b'?', "{b:#x}");
    }
    let mut s = RdsState::new();
    s.update(&ps_group(0xD123, 0, [0x8E, b'a']));
    s.update(&ps_group(0xD123, 1, [0x7F, 0x0D]));
    s.update(&ps_group(0xD123, 2, *b"bc"));
    s.update(&ps_group(0xD123, 3, [0x00, b'd']));
    assert_eq!(s.ps(), "?a??bc?d");
    s.update(&rt_group(0, false, [0xC4, b'x', 0x01, b'\r']));
    assert_eq!(s.rt(), "?x?");
}

#[test]
fn a_new_pi_clears_the_state_and_an_unreadable_pi_is_ignored() {
    let mut s = RdsState::new();
    for (seg, pair) in [(0, *b"SD"), (1, *b"RK"), (2, *b" T"), (3, *b"ST")] {
        s.update(&ps_group(0xD123, seg, pair));
    }
    s.update(&rt_group(0, false, *b"Hi\r "));
    assert_eq!(s.ps(), "SDRK TST");
    // Block A invalid and version A: the group is ignored.
    let mut g = ps_group(0x1111, 0, *b"XX");
    g.valid[0] = false;
    s.update(&g);
    assert_eq!(s.ps(), "SDRK TST");
    assert_eq!(s.pi(), Some(0xD123));
    // Another station.
    s.update(&ps_group(0x1111, 0, *b"XX"));
    assert_eq!(s.pi(), Some(0x1111));
    assert_eq!(s.ps(), "");
    assert_eq!(s.rt(), "");
    // A version B group gives the PI from block C' when block A is invalid.
    let mut g = Group::new([0, 0x0948, 0x2222, u16::from_be_bytes(*b"ZZ")]);
    g.valid[0] = false;
    s.update(&g);
    assert_eq!(s.pi(), Some(0x2222));
    s.reset();
    assert_eq!(s, RdsState::new());
    assert_eq!(s.pi(), None);
}

#[test]
fn packets_round_trip() {
    let mut g = Group::new([0xD123, 0x0948, 0xD123, 0x4142]);
    let p = g.to_packet(77).unwrap();
    assert_eq!(p.index, 77);
    assert_eq!(p.data(), &[0xD1, 0x23, 0x09, 0x48, 0xD1, 0x23, 0x41, 0x42]);
    assert!(p.flags.contains(PacketFlags::CRC_OK));
    assert!(p.flags.contains(PacketFlags(0xF << VALID_SHIFT)));
    assert!(p.flags.contains(PacketFlags(VERSION_B)));
    assert_eq!(Group::from_packet(&p), Some(g));
    g.valid = [true, false, true, true];
    let p = g.to_packet(0).unwrap();
    assert!(p.flags.contains(PacketFlags::CRC_FAIL));
    assert!(!p.flags.contains(PacketFlags::CRC_OK));
    assert!(!p.flags.contains(PacketFlags(VERSION_B)));
    assert_eq!(Group::from_packet(&p), Some(g));
    assert_eq!(g.validity_bits(), 0b1101 << VALID_SHIFT);
    assert_eq!(
        Group::from_packet(&Packet::new(0, PacketFlags::NONE, b"short").unwrap()),
        None
    );
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn public_types_are_send_and_sync() {
    assert_send_sync::<Offset>();
    assert_send_sync::<Version>();
    assert_send_sync::<Group>();
    assert_send_sync::<RdsState>();
}
