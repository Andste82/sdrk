//! `Chain::process` never allocates: a counting global allocator watches 1000 calls.
//!
//! `GlobalAlloc` is an unsafe trait, so this test opts out of the crate-wide ban on unsafe code.
#![allow(
    unsafe_code,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::large_stack_arrays,
    clippy::large_stack_frames,
    clippy::range_plus_one
)]

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use common::{Aligned, rate};
use sdrk::{C32, CQ15};
use sdrk_blocks::{
    ArenaStorage, Block, Chain, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig, FmDemodBlock,
    FmDemodConfig, FrameOut, ParamValue, SpectrumBlock, SpectrumConfig, chain,
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

#[test]
fn process_set_param_and_reset_do_not_allocate() {
    let mut mem = Aligned::new(1 << 16);
    let mut arena = ArenaStorage::new(mem.bytes(1 << 16));
    let mut rx = build_receiver(&mut arena).unwrap();
    let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 7 % 251) as u8).collect();
    let mut audio = vec![0.0f32; 512];
    // warm up outside the measurement
    rx.process(&bytes, &mut audio).unwrap();

    let (produced, allocations) = count_allocations(|| {
        let mut produced = 0;
        for i in 0..1000usize {
            let start = (i * 13) % 512;
            let end = start + 1 + (i * 37) % 3000;
            produced += rx
                .process(&bytes[start..end], &mut audio[..1 + i % 500])
                .unwrap()
                .produced;
            if i % 100 == 0 {
                rx.set_param("1.0.frequency_hz", ParamValue::F64(5_000.0 + i as f64))
                    .unwrap();
            }
            if i % 250 == 249 {
                rx.reset();
            }
        }
        produced
    });
    assert!(produced > 1000, "{produced}");
    assert_eq!(allocations, 0);
}

fn build_receiver<'a>(
    arena: &mut ArenaStorage<'a>,
) -> Result<impl Block<In = u8, Out = f32> + 'a, sdrk_blocks::Error> {
    let conv = ConvertBlock::<C32, _>::new(&ConvertConfig::default(), rate(240_000), arena)?;
    let ddc = DdcBlock::<C32, _>::new(
        &DdcConfig {
            frequency_hz: 10_000.0,
            decimation: 4,
            max_chunk: 64,
            ..DdcConfig::default()
        },
        rate(240_000),
        arena,
    )?;
    let fm = FmDemodBlock::<C32, _>::new(&FmDemodConfig::default(), rate(60_000), arena)?;
    Ok(chain!(&mut *arena, 64; conv, ddc, fm))
}

#[test]
fn a_chain_into_a_frame_block_does_not_allocate() {
    let mut mem = Aligned::new(1 << 16);
    let mut arena = ArenaStorage::new(mem.bytes(1 << 16));
    let conv =
        ConvertBlock::<CQ15, _>::new(&ConvertConfig::default(), rate(48_000), &mut arena).unwrap();
    let spec = SpectrumBlock::<CQ15, _>::new(
        &SpectrumConfig {
            fft_size: 64,
            ..SpectrumConfig::default()
        },
        rate(48_000),
        &mut arena,
    )
    .unwrap();
    let mut c: Chain<_, _, ArenaStorage<'_>> = Chain::new(conv, spec, 128, &mut arena).unwrap();
    let bytes: Vec<u8> = (0..2048u32).map(|i| (i * 11 % 253) as u8).collect();
    let mut frame = vec![0.0f32; c.frame_len()];
    let (frames, allocations) = count_allocations(|| {
        let mut frames = 0;
        for _ in 0..200 {
            let mut pending = &bytes[..];
            loop {
                let mut meta = None;
                let p = c
                    .process(
                        pending,
                        FrameOut {
                            values: &mut frame,
                            meta: &mut meta,
                        },
                    )
                    .unwrap();
                frames += usize::from(meta.is_some());
                pending = &pending[p.consumed..];
                if meta.is_none() && p.consumed == 0 {
                    break;
                }
            }
        }
        frames
    });
    assert!(frames > 100);
    assert_eq!(allocations, 0);
}
