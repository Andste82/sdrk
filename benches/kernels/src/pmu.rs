//! Hardware performance counters through `perf_event_open` (Linux, x86-64 and AArch64): the
//! core cycles and the retired instructions of a region, counted in user mode on the thread
//! that calls it. No emulator, no model and no wall clock; it needs the permission for
//! `perf_event_open` (root, `CAP_PERFMON`, or `kernel.perf_event_paranoid` of at most 2) and a
//! PMU the kernel exposes (not given in most virtual machines and containers).

use std::fs::File;
use std::io::Read;
use std::os::fd::FromRawFd;

#[cfg(target_arch = "x86_64")]
const SYS_PERF_EVENT_OPEN: i64 = 298;
#[cfg(target_arch = "aarch64")]
const SYS_PERF_EVENT_OPEN: i64 = 241;

const PERF_TYPE_HARDWARE: u32 = 0;
const HW_CPU_CYCLES: u64 = 0;
const HW_INSTRUCTIONS: u64 = 1;
/// `disabled`, `exclude_kernel` and `exclude_hv` of `perf_event_attr`.
const FLAGS: u64 = (1 << 0) | (1 << 5) | (1 << 6);
const IOC_ENABLE: u64 = 0x2400;
const IOC_DISABLE: u64 = 0x2401;
const IOC_RESET: u64 = 0x2403;

unsafe extern "C" {
    fn syscall(number: i64, ...) -> i64;
    fn ioctl(fd: i32, request: u64, ...) -> i32;
}

/// The counters of one region.
pub struct Counters {
    cycles: File,
    instructions: File,
}

/// What a region counted.
#[derive(Clone, Copy, Debug)]
pub struct Counts {
    pub cycles: u64,
    pub instructions: u64,
}

fn open(config: u64) -> Result<File, String> {
    // `perf_event_attr` of ABI version 5, 128 bytes: type, size, config, then zeros up to the
    // flags at byte 40.
    let mut attr = [0u64; 16];
    attr[0] = u64::from(PERF_TYPE_HARDWARE) | (128u64 << 32);
    attr[1] = config;
    attr[5] = FLAGS;
    // SAFETY: `attr` is a valid, 128-byte `perf_event_attr` that outlives the call; the
    // arguments are pid 0 (this thread), cpu -1 (any), no group and no flags.
    let fd = unsafe { syscall(SYS_PERF_EVENT_OPEN, attr.as_ptr(), 0i32, -1i32, -1i32, 0u64) };
    if fd < 0 {
        return Err(format!(
            "perf_event_open: {} (perf_event_paranoid = {})",
            std::io::Error::last_os_error(),
            std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
                .map_or_else(|_| "?".into(), |s| s.trim().to_string())
        ));
    }
    // SAFETY: `fd` is a new file descriptor that nothing else owns.
    Ok(unsafe { File::from_raw_fd(fd as i32) })
}

impl Counters {
    pub fn new() -> Result<Self, String> {
        Ok(Counters {
            cycles: open(HW_CPU_CYCLES)?,
            instructions: open(HW_INSTRUCTIONS)?,
        })
    }

    fn ctl(&self, request: u64) {
        use std::os::fd::AsRawFd;
        for f in [&self.cycles, &self.instructions] {
            // SAFETY: the descriptors are open perf events; these requests take no argument.
            unsafe { ioctl(f.as_raw_fd(), request, 0) };
        }
    }

    #[inline(always)]
    pub fn start(&self) {
        self.ctl(IOC_RESET);
        self.ctl(IOC_ENABLE);
    }

    #[inline(always)]
    pub fn stop(&mut self) -> Counts {
        self.ctl(IOC_DISABLE);
        let read = |f: &mut File| {
            let mut b = [0u8; 8];
            f.read_exact(&mut b).expect("read of a perf counter");
            u64::from_ne_bytes(b)
        };
        Counts {
            cycles: read(&mut self.cycles),
            instructions: read(&mut self.instructions),
        }
    }
}
