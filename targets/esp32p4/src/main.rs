//! SDRk on-target suite for the ESP32-P4 (RISC-V rv32imafc), output on UART0.
//!
//! The image is wrapped by `espflash save-image --merge`; `esp-emu` runs it and `cargo xtask
//! emu` reads the protocol lines from standard output.
#![no_std]
#![no_main]

use core::panic::PanicInfo;

use esp_println::Printer;
use sdrk_target_common::{ARENA_SIZE, report_panic, run};
use sdrk_testkit::all_cases;

// The application descriptor that the ESP-IDF second stage bootloader expects.
esp_bootloader_esp_idf::esp_app_desc!();

/// The arena of the suite (256 KiB). A target is an application, so a static buffer is fine.
static mut ARENA: [u8; ARENA_SIZE] = [0; ARENA_SIZE];

#[esp_hal::main]
fn main() -> ! {
    let _peripherals = esp_hal::init(esp_hal::Config::default());
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
