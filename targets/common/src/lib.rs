//! Shared suite driver of the SDRk emulator targets.
//!
//! A target sets up its platform, provides a [`core::fmt::Write`] sink for the
//! protocol lines and calls [`run`]. The host (`cargo xtask emu`) decides pass or
//! fail only from the protocol lines, so the driver prints them and nothing else.
#![no_std]

use core::fmt::Write;

use sdrk_testkit::{Arena, TestCase, Tier, WriteReporter, run_suite};

/// Size of the arena every target places in its own static memory (256 KiB).
pub const ARENA_SIZE: usize = 256 * 1024;

/// Size of the arena of the quick profile (64 KiB): quick cases are sized for it, and a larger
/// arena would make some of them run bigger (and, with software floating point, much slower)
/// variants of their sweeps, for example the 4096-point DFT of the reference.
pub const QUICK_ARENA_SIZE: usize = 64 * 1024;

/// The tier selected at build time: feature `full`, else quick.
pub const TIER: Tier = if cfg!(feature = "full") {
    Tier::Full
} else {
    Tier::Quick
};

/// Runs `cases` at the build-time [`TIER`] over `arena_memory`, printing the protocol to
/// `out`. The full profile uses the whole memory, the quick profile its first
/// [`QUICK_ARENA_SIZE`] bytes. Returns `true` if no case failed.
pub fn run<W: Write>(out: W, arena_memory: &mut [u8], cases: &[TestCase]) -> bool {
    let len = if TIER == Tier::Full {
        arena_memory.len()
    } else {
        arena_memory.len().min(QUICK_ARENA_SIZE)
    };
    let (memory, _) = arena_memory.split_at_mut(len);
    let mut arena = Arena::new(memory);
    let mut reporter = WriteReporter::new(out);
    run_suite(cases, TIER, &mut arena, &mut reporter).failed == 0
}

/// Prints the panic report line `SDRK-PANIC <message>` on one line.
pub fn report_panic<W: Write>(out: &mut W, info: &core::panic::PanicInfo<'_>) {
    struct OneLine<'a, W: Write>(&'a mut W);
    impl<W: Write> Write for OneLine<'_, W> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for ch in s.chars() {
                self.0
                    .write_char(if ch == '\n' || ch == '\r' { ' ' } else { ch })?;
            }
            Ok(())
        }
    }
    let _ = out.write_str("\nSDRK-PANIC ");
    let _ = write!(OneLine(out), "{info}");
    let _ = out.write_char('\n');
}
