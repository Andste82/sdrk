//! `sdrk_proto::rds::Carousel` (M42 §1): every carousel group parses back through `RdsState` to
//! the text that was set, the carousel of the M41 reference signal, the raw-group queue, the A/B
//! flag and the padding of the radiotext.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use sdrk_proto::rds::{
    self, Carousel, Group, NO_AF, Offset, RAW_QUEUE, RdsState, Version, group_blocks26,
};
use sdrk_testkit::rds::carousel_group;
use sdrk_testkit::rng::Pcg32;

fn feed(c: &mut Carousel, rx: &mut RdsState, groups: usize) {
    for _ in 0..groups {
        let words = c.next_group();
        // Every block of the group carries a valid checkword for its offset.
        let blocks = group_blocks26(words);
        for (i, (b, w)) in blocks.iter().zip(words).enumerate() {
            assert_eq!(b >> 10, u32::from(w));
            let o = Offset::from_syndrome(rds::syndrome(*b)).unwrap();
            assert_eq!(o.position(), i);
        }
        rx.update(&Group::new(words));
    }
}

#[test]
fn the_carousel_is_the_one_of_the_reference_signal() {
    let mut c = Carousel::new(0xD123, 10, false).unwrap();
    c.set_ps(b"SDRK TST");
    c.set_rt(b"Hello from sdrk!").unwrap();
    for g in 0..64 {
        assert_eq!(c.next_group(), carousel_group(g), "group {g}");
    }
}

#[test]
fn every_group_parses_back_to_the_text_that_was_set() {
    let mut rng = Pcg32::new(580, 1);
    for round in 0..20u32 {
        let pi = (rng.next_u32() & 0xFFFF) as u16;
        let pty = (rng.next_u32() % 32) as u8;
        let tp = rng.next_u32() & 1 == 1;
        let mut c = Carousel::new(pi, pty, tp).unwrap();
        let mut ps = [0u8; 8];
        for b in &mut ps {
            *b = 0x20 + (rng.next_u32() % 95) as u8;
        }
        let len = 1 + rng.below(64) as usize;
        let rt: Vec<u8> = (0..len)
            .map(|_| 0x20 + (rng.next_u32() % 95) as u8)
            .collect();
        c.set_ps(&ps);
        c.set_rt(&rt).unwrap();
        let mut rx = RdsState::new();
        // Four 0A and sixteen 2A segments need 32 groups.
        feed(&mut c, &mut rx, 32);
        assert_eq!(rx.ps().as_bytes(), &ps, "round {round}");
        assert_eq!(rx.rt().as_bytes(), &rt[..], "round {round}");
        assert_eq!(rx.pi(), Some(pi));
        assert_eq!(rx.pty(), Some(pty));
        assert_eq!(rx.tp(), Some(tp));
    }
}

#[test]
fn the_group_contents_follow_the_standard() {
    let mut c = Carousel::new(0x1234, 31, true).unwrap();
    c.set_ps(b"ABCDEFGH");
    c.set_rt(b"x").unwrap();
    for g in 0..8u16 {
        let w = c.next_group();
        let grp = Group::new(w);
        assert_eq!(grp.pi(), 0x1234);
        assert!(grp.tp());
        assert_eq!(grp.pty(), 31);
        if g % 2 == 0 {
            assert_eq!(grp.group_type(), (0, Version::A));
            // TA 0, M/S 1, DI 0, the address.
            assert_eq!(w[1] & 0x1F, 0x08 | (g / 2));
            assert_eq!(w[2], NO_AF);
            let seg = usize::from(g / 2);
            assert_eq!(
                w[3].to_be_bytes(),
                [b"ABCDEFGH"[2 * seg], b"ABCDEFGH"[2 * seg + 1]]
            );
        } else {
            assert_eq!(grp.group_type(), (2, Version::A));
            assert_eq!(w[1] & 0x1F, g / 2);
        }
    }
    // The radiotext "x" is padded with a carriage return and spaces.
    let mut want = [b' '; 64];
    want[0] = b'x';
    want[1] = 0x0D;
    assert_eq!(c.rt(), &want);
    // A text of exactly 64 bytes has no carriage return.
    c.set_rt(&[b'y'; 64]).unwrap();
    assert_eq!(c.rt(), &[b'y'; 64]);
}

#[test]
fn without_a_radiotext_only_groups_0a_are_sent() {
    let mut c = Carousel::new(0xD123, 10, false).unwrap();
    for g in 0..10u16 {
        let w = c.next_group();
        assert_eq!(w[1] >> 12, 0);
        assert_eq!(w[1] & 0x3, g % 4);
        assert_eq!(w[3], 0x2020);
    }
    c.set_rt(b"on").unwrap();
    assert!(c.rt_active());
    // The last group was a 0A, so a 2A segment 0 follows.
    assert_eq!(c.next_group()[1], 0x2140);
    assert_eq!(c.next_group()[1] >> 12, 0);
    c.set_rt(b"").unwrap();
    assert!(!c.rt_active());
    for _ in 0..4 {
        assert_eq!(c.next_group()[1] >> 12, 0);
    }
}

#[test]
fn raw_groups_come_first_in_order_and_do_not_advance_the_carousel() {
    let mut c = Carousel::new(0xD123, 10, false).unwrap();
    c.set_ps(b"SDRK TST");
    c.set_rt(b"Hello from sdrk!").unwrap();
    assert_eq!(c.next_group(), carousel_group(0));
    for i in 0..RAW_QUEUE as u16 {
        c.push_raw(Group::new([0xD123, 0x4000 | i, i, !i])).unwrap();
    }
    assert!(c.is_raw_full());
    let err = c.push_raw(Group::new([1, 2, 3, 4])).unwrap_err();
    assert!(matches!(err, sdrk_proto::Error::BufferTooSmall { .. }));
    for i in 0..RAW_QUEUE as u16 {
        assert_eq!(c.next_group(), [0xD123, 0x4000 | i, i, !i]);
        if i == 3 {
            // Room again: the new group goes to the end of the queue.
            c.push_raw(Group::new([7, 7, 7, 7])).unwrap();
        }
    }
    assert_eq!(c.next_group(), [7, 7, 7, 7]);
    assert_eq!(c.raw_len(), 0);
    for g in 1..40 {
        assert_eq!(c.next_group(), carousel_group(g));
    }
}

#[test]
fn a_changed_text_toggles_the_ab_flag_and_restarts_at_segment_0() {
    let mut c = Carousel::new(0xD123, 10, false).unwrap();
    c.set_rt(b"first").unwrap();
    assert!(!c.ab_flag());
    let mut rx = RdsState::new();
    feed(&mut c, &mut rx, 32);
    assert_eq!(rx.rt(), "first");
    // Half a text, then a new one: the receiver clears on the flag change and shows only the
    // new text.
    c.next_group();
    c.next_group();
    c.next_group();
    c.set_rt(b"second text").unwrap();
    assert!(c.ab_flag());
    let mut seen = Vec::new();
    for _ in 0..32 {
        let w = c.next_group();
        if w[1] >> 12 == 2 {
            assert_ne!(w[1] & 0x10, 0);
            seen.push(w[1] & 0xF);
        }
        rx.update(&Group::new(w));
    }
    assert_eq!(seen, (0..16).collect::<Vec<_>>());
    assert_eq!(rx.rt(), "second text");
    // The same text again does not toggle; an empty text and the same text again neither.
    c.set_rt(b"second text").unwrap();
    c.set_rt(b"").unwrap();
    c.set_rt(b"second text").unwrap();
    assert!(c.ab_flag());
    c.set_rt(b"third").unwrap();
    assert!(!c.ab_flag());
    // A text above 64 bytes is rejected and changes nothing.
    assert!(matches!(
        c.set_rt(&[b'z'; 65]),
        Err(sdrk_proto::Error::TooLong { len: 65, max: 64 })
    ));
    assert_eq!(&c.rt()[..6], b"third\r");
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn the_carousel_is_send_and_sync() {
    assert_send_sync::<Carousel>();
}
