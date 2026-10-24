//! SDRk on-target suite for QEMU `mps3-an547` (Cortex-M55), output and exit through semihosting.
#![no_std]
#![no_main]

use core::fmt::{self, Write};
use core::panic::PanicInfo;

use cortex_m_rt::entry;
use cortex_m_semihosting::debug::{self, EXIT_FAILURE, EXIT_SUCCESS};
use cortex_m_semihosting::hio;
use sdrk_target_common::{ARENA_SIZE, report_panic, run};
use sdrk_testkit::all_cases;

/// The arena of the suite (256 KiB). A target is an application, so a static buffer is fine.
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

/// The semihosting console as a `fmt::Write` sink.
struct Console(hio::HostStream);

impl Console {
    fn open() -> Option<Self> {
        hio::hstdout().ok().map(Console)
    }
}

impl Write for Console {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.write_all(s.as_bytes()).map_err(|_| fmt::Error)
    }
}

fn exit(ok: bool) -> ! {
    debug::exit(if ok { EXIT_SUCCESS } else { EXIT_FAILURE });
    loop {
        cortex_m::asm::wfi();
    }
}

#[entry]
fn main() -> ! {
    // SAFETY: `main` runs once and nothing else touches `ARENA`.
    let arena = unsafe { &mut *core::ptr::addr_of_mut!(ARENA) };
    let Some(console) = Console::open() else {
        exit(false)
    };
    let ok = run(console, arena, all_cases());
    exit(ok)
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    if let Some(mut console) = Console::open() {
        report_panic(&mut console, info);
    }
    exit(false)
}
