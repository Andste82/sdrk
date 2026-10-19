//! The storage abstraction: `ArenaStorage` and `VecStorage`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
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

use common::Aligned;
use sdrk_blocks::{ARENA_ALIGN, AlignedBytes, ArenaStorage, Error, Storage};
use sdrk_core::{C32, CQ15};
use sdrk_testkit::VecStorage;

#[test]
fn the_arena_hands_out_zeroed_aligned_buffers_in_order() {
    let mut mem = AlignedBytes([0xAAu8; 128]);
    let mut arena = ArenaStorage::new(&mut mem.0);
    let a = arena.alloc::<u8>(3).unwrap();
    assert_eq!(a, [0, 0, 0]);
    assert_eq!(arena.used(), 3);
    let b = arena.alloc::<f64>(2).unwrap();
    assert_eq!(b.as_ptr() as usize % 8, 0);
    assert_eq!(arena.used(), 8 + 16, "3 bytes, 5 of padding, 16 bytes");
    let c = arena.alloc::<CQ15>(3).unwrap();
    assert_eq!(c.as_ptr() as usize % 2, 0);
    assert!(c.iter().all(|s| *s == CQ15::default()));
    assert_eq!(arena.used() + arena.remaining(), 128);
    // The buffers are disjoint and usable at the same time.
    a[0] = 1;
    b[0] = 2.0;
    c[0].re.0 = 3;
    assert_eq!((a[0], b[0], c[0].re.0), (1, 2.0, 3));
}

#[test]
fn memory_that_was_used_before_is_cleared() {
    let mut mem = AlignedBytes([0xFFu8; 64]);
    let mut arena = ArenaStorage::new(&mut mem.0);
    assert!(
        arena
            .alloc::<C32>(8)
            .unwrap()
            .iter()
            .all(|s| s.re == 0.0 && s.im == 0.0)
    );
}

#[test]
fn an_allocation_that_does_not_fit_fails_and_changes_nothing() {
    let mut mem = AlignedBytes([0u8; 32]);
    let mut arena = ArenaStorage::new(&mut mem.0);
    arena.alloc::<u8>(1).unwrap();
    let e = arena.alloc::<f32>(8).unwrap_err();
    assert_eq!(
        e,
        Error::Storage {
            needed: 3 + 32,
            available: 31
        }
    );
    assert_eq!((arena.used(), arena.remaining()), (1, 31));
    // A smaller request still works afterwards.
    assert_eq!(arena.alloc::<f32>(7).unwrap().len(), 7);
    assert_eq!(arena.used(), 32);
    // Exhausted: only empty buffers fit.
    assert!(arena.alloc::<u8>(1).is_err());
    assert_eq!(arena.alloc::<u8>(0).unwrap().len(), 0);
}

#[test]
fn absurd_sizes_are_errors_not_panics() {
    let mut mem = AlignedBytes([0u8; 16]);
    let mut arena = ArenaStorage::new(&mut mem.0);
    assert!(matches!(
        arena.alloc::<f64>(usize::MAX),
        Err(Error::Storage { .. })
    ));
    assert!(matches!(
        arena.alloc::<u8>(usize::MAX),
        Err(Error::Storage { .. })
    ));
    let mut empty = ArenaStorage::new(&mut []);
    assert_eq!(empty.alloc::<u8>(0).unwrap().len(), 0);
    assert!(empty.alloc::<u8>(1).is_err());
}

#[test]
fn an_unaligned_start_costs_padding() {
    let mut mem = Aligned::new(64);
    let bytes = mem.bytes(64);
    let mut arena = ArenaStorage::new(&mut bytes[1..]);
    let buf = arena.alloc::<f32>(2).unwrap();
    assert_eq!(buf.as_ptr() as usize % 4, 0);
    assert_eq!(arena.used(), 3 + 8);
}

#[test]
fn arena_align_is_sixteen_and_the_helper_is_aligned() {
    assert_eq!(ARENA_ALIGN, 16);
    let mem = AlignedBytes([0u8; 5]);
    assert_eq!(mem.0.as_ptr() as usize % ARENA_ALIGN, 0);
    assert_eq!(core::mem::align_of::<AlignedBytes<1>>(), 16);
}

#[test]
fn vec_storage_gives_zeroed_vectors_of_the_requested_length() {
    let mut s = VecStorage;
    let v = s.alloc::<C32>(5).unwrap();
    assert_eq!(v.len(), 5);
    assert!(v.iter().all(|x| x.re == 0.0 && x.im == 0.0));
    assert!(s.alloc::<u8>(0).unwrap().is_empty());
}
