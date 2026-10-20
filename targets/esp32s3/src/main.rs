//! SDRk on-target suite for the ESP32-S3 (Xtensa LX7), output on UART0.
//!
//! The image is wrapped by `espflash save-image --merge`; the Espressif QEMU fork runs it and
//! `cargo xtask emu` reads the protocol lines from standard output.
//!
//! Without the cargo feature `hardware` (the default) `esp_hal::init` is not called: it brings up
//! the 480 MHz PLL and waits for the analog lock that the emulator does not model, so the target
//! only keeps the runtime of `esp-hal` (startup code, linker script) and the CPU stays on the
//! clock the ROM bootloader left. What `init` would otherwise do for this program is done by
//! hand: the watchdogs that the bootloader leaves running are switched off and the
//! floating-point coprocessor is enabled (without `CPENABLE` the first `f32` instruction raises
//! an exception that the runtime cannot handle in this state).
//!
//! With the feature `hardware` the target is meant for a real board: `main` calls `esp_hal::init`
//! (CPU clock, interrupts, timers, watchdog set-up of the HAL) and does none of the manual
//! set-up. That build is only compiled in CI; it cannot run in the emulator.
#![no_std]
#![no_main]
#![feature(asm_experimental_arch)]

use core::panic::PanicInfo;

use esp_println::Printer;
use sdrk_target_common::{ARENA_SIZE, report_panic, run};
use sdrk_testkit::all_cases;

// The application descriptor that the ESP-IDF second stage bootloader expects.
esp_bootloader_esp_idf::esp_app_desc!();

/// The arena of the suite (256 KiB). A target is an application, so a static buffer is fine.
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

/// The manual set-up of the emulator: no `esp_hal::init`.
#[cfg(not(feature = "hardware"))]
mod emulator {
    use core::ptr::write_volatile;

    /// Register addresses of the ESP32-S3 (technical reference manual, chapters on the watchdogs;
    /// the offsets are those of the `esp32s3` peripheral access crate).
    const TIMG0: usize = 0x6001_f000;
    const TIMG1: usize = 0x6002_0000;
    const TIMG_WDTCONFIG0: usize = 0x48;
    const TIMG_WDTWPROTECT: usize = 0x64;
    const TIMG_WDT_KEY: u32 = 0x50d8_3aa1;
    const RTC_CNTL: usize = 0x6000_8000;
    const RTC_WDTCONFIG0: usize = 0x98;
    const RTC_WDTWPROTECT: usize = 0xb0;
    const RTC_WDT_KEY: u32 = 0x50d8_3aa1;
    const RTC_SWD_CONF: usize = 0xb4;
    const RTC_SWD_WPROTECT: usize = 0xb8;
    const RTC_SWD_KEY: u32 = 0x8f1d_312a;
    /// Bit 30 of `SWD_CONF`: disable the super watchdog.
    const SWD_DISABLE: u32 = 1 << 30;

    fn write(address: usize, value: u32) {
        // SAFETY: the addresses are memory-mapped registers of the ESP32-S3 that the code above
        // names; the write has no effect on memory the program owns.
        unsafe { write_volatile(address as *mut u32, value) };
    }

    /// Switches off the two timer-group watchdogs, the RTC watchdog and the super watchdog.
    fn disable_watchdogs() {
        for group in [TIMG0, TIMG1] {
            write(group + TIMG_WDTWPROTECT, TIMG_WDT_KEY);
            write(group + TIMG_WDTCONFIG0, 0);
            write(group + TIMG_WDTWPROTECT, 0);
        }
        write(RTC_CNTL + RTC_WDTWPROTECT, RTC_WDT_KEY);
        write(RTC_CNTL + RTC_WDTCONFIG0, 0);
        write(RTC_CNTL + RTC_WDTWPROTECT, 0);
        write(RTC_CNTL + RTC_SWD_WPROTECT, RTC_SWD_KEY);
        write(RTC_CNTL + RTC_SWD_CONF, SWD_DISABLE);
        write(RTC_CNTL + RTC_SWD_WPROTECT, 0);
    }

    /// Switches the watchdogs off and enables the floating-point coprocessor.
    pub fn prepare() {
        disable_watchdogs();
        // SAFETY: setting bit 0 of CPENABLE enables the floating-point coprocessor.
        unsafe { core::arch::asm!("wsr.cpenable {0}", "rsync", in(reg) 1u32) };
    }
}

#[esp_hal::main]
fn main() -> ! {
    #[cfg(feature = "hardware")]
    let _peripherals = esp_hal::init(esp_hal::Config::default());
    #[cfg(not(feature = "hardware"))]
    emulator::prepare();
    // SAFETY: `main` runs once and nothing else touches `ARENA`.
    let arena = unsafe { &mut *core::ptr::addr_of_mut!(ARENA) };
    run(Printer, arena, all_cases());
    // The host stops the emulator when it has read `SDRK-SUITE END`.
    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    report_panic(&mut Printer, info);
    loop {
        core::hint::spin_loop();
    }
}
