//! `Fork::process` and a chain with a pair between its stages never allocate: a counting
//! global allocator watches 1000 calls (M31 test 6). The allocator is per test binary, so this
//! is a file of its own.
//!
//! `GlobalAlloc` is an unsafe trait, so this test opts out of the crate-wide ban on unsafe code.
#![allow(
    unsafe_code,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::large_stack_arrays,
    clippy::large_stack_frames,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::range_plus_one
)]

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use common::{Aligned, rate, real_signal};
use sdrk::C32;
use sdrk_blocks::design::BiquadResponse;
use sdrk_blocks::{
    ArenaStorage, BiquadConfig, BiquadRealBlock, Block, DelayConfig, DelayRealBlock, Error, Fork,
    ParamMap, ParamValue, StereoPllBlock, StereoPllConfig, chain,
};

thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static ON: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

fn note() {
    let _ = ON.try_with(|on| {
        if on.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}

// SAFETY: every call is forwarded unchanged to the system allocator; the bookkeeping only
// touches thread-local `Cell`s and never allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn count_allocations<R>(f: impl FnOnce() -> R) -> (R, usize) {
    COUNT.with(|c| c.set(0));
    ON.with(|on| on.set(true));
    let r = f();
    ON.with(|on| on.set(false));
    (r, COUNT.with(Cell::get))
}

const FS: u64 = 192_000;

const MAP: ParamMap = ParamMap::new(&[("pilot_q", "0.1.q"), ("stereo", "1.stereo")]);

fn pilot_cfg() -> BiquadConfig {
    BiquadConfig {
        response: BiquadResponse::Bandpass,
        cutoff_hz: 19_000.0,
        order: 2,
        q: 5.0,
    }
}

/// The front of a stereo decoder: the multiplex signal delayed by 3 samples and its pilot
/// band-pass (aligned by a delay line of 3 samples), joined in the pilot loop.
fn build_decoder<'a>(
    arena: &mut ArenaStorage<'a>,
) -> Result<impl Block<In = f32, Out = C32> + 'a, Error> {
    let mpx = DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 3 }, rate(FS), arena)?;
    let pilot = BiquadRealBlock::<f32, _>::new(&pilot_cfg(), rate(FS), arena)?;
    let pll = StereoPllBlock::<C32, _>::new(&StereoPllConfig::default(), rate(FS), arena)?;
    let fork = Fork::new(mpx, pilot, 64, arena)?;
    Ok(chain!(&mut *arena, 64; fork, pll))
}

#[test]
fn fork_and_a_chain_with_a_pair_do_not_allocate() {
    let mut mem = Aligned::new(1 << 14);
    let mut arena = ArenaStorage::new(mem.bytes(1 << 14));
    let mut dec = build_decoder(&mut arena).unwrap();
    let input = real_signal::<f32>(4096, 478, 19_000.0 / FS as f64);
    let mut out = vec![C32::default(); 512];
    // warm up outside the measurement
    dec.process(&input, &mut out).unwrap();

    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0;
        for i in 0..1000usize {
            let start = (i * 13) % 512;
            let end = start + 1 + (i * 37) % 3000;
            produced += dec
                .process(&input[start..end], &mut out[..1 + i % 500])
                .unwrap()
                .produced;
            if i % 100 == 0 {
                MAP.set(
                    &mut dec,
                    "pilot_q",
                    ParamValue::F64(4.0 + (i / 100) as f64 * 0.1),
                )
                .unwrap();
                let _ = MAP.get(&dec, "stereo").unwrap();
            }
            if i % 250 == 249 {
                dec.reset();
            }
        }
        produced
    });
    assert!(produced > 1000, "{produced}");
    assert_eq!(allocations, 0);
}

#[test]
fn a_bare_fork_does_not_allocate() {
    let mut mem = Aligned::new(1 << 12);
    let mut arena = ArenaStorage::new(mem.bytes(1 << 12));
    let a =
        DelayRealBlock::<f32, _>::new(&DelayConfig { samples: 0 }, rate(FS), &mut arena).unwrap();
    let b = BiquadRealBlock::<f32, _>::new(&pilot_cfg(), rate(FS), &mut arena).unwrap();
    let mut fork = Fork::new(a, b, 32, &mut arena).unwrap();
    let input = real_signal::<f32>(2048, 479, 0.1);
    let (mut x, mut y) = (vec![0.0f32; 300], vec![0.0f32; 300]);
    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0;
        for i in 0..1000usize {
            let start = (i * 7) % 1024;
            let end = start + (i * 31) % 1000;
            let n = i % 300;
            produced += fork
                .process(&input[start..end], (&mut x[..n], &mut y[..n]))
                .unwrap()
                .produced;
            if i % 300 == 299 {
                fork.reset();
            }
        }
        produced
    });
    assert!(produced > 1000, "{produced}");
    assert_eq!(allocations, 0);
}
