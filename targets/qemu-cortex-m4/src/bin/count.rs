//! Instruction-count workload of the fallback kernels on the Cortex-M4F (`cargo xtask count
//! --platform cortex-m4`).
//!
//! QEMU passes `<kernel> <k>` as the semihosting command line; the program runs the kernel
//! once as a warm-up and then `k` times on the same buffers. The host counts the instructions
//! of the whole run with the QEMU `insn` plugin, and the differential method of the count
//! command (`k = 16` minus `k = 0`) removes boot, warm-up and printing.
#![no_std]
#![no_main]

use core::fmt::Write;
use core::hint::black_box;
use core::panic::PanicInfo;

use cortex_m_rt::entry;
use cortex_m_semihosting::debug::{self, EXIT_FAILURE, EXIT_SUCCESS};
use cortex_m_semihosting::{hio, nr, syscall1};
use sdrk::kernels::agc::{Agc, AgcState};
use sdrk::kernels::convert::{Convert, ConvertOut};
use sdrk::kernels::decim::{FirDecimState, FirDecimate};
use sdrk::kernels::demod::{DcBlockReal, DcBlockRealState, QuadDemod, QuadDemodState};
use sdrk::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk::kernels::filter::{Fir, FirState};
use sdrk::kernels::mix::{Nco, Rotate};
use sdrk::kernels::modulate::{FmMod, FmModState};
use sdrk::kernels::spectrum::MagSq;
use sdrk::{C32, CQ15, Fallback, Q15, Rate};
use sdrk_device::formats;

/// Samples per call of the streaming kernels.
const N: usize = 1024;
const FIR_TAPS: usize = 64;
const DECIM_TAPS: usize = 127;
const DECIM: usize = 10;

/// The kernels of the report: the names are the keys of `benches/results/cortex-m4.json`.
const KERNELS: [&str; 15] = [
    "convert_cs8",
    "convert_cs16",
    "convert_out_cs16",
    "rotate",
    "fir",
    "fir_decimate",
    "mag_sq",
    "quad_demod",
    "fm_mod",
    "fft_1024",
    "fft_cq15_1024",
    "agc",
    "agc_cq15",
    "dc_block_real",
    "dc_block_real_q15",
];

/// A small linear congruential generator; the values only have to be varied.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 8_388_608.0 - 1.0
    }
}

fn noise(rng: &mut Lcg, out: &mut [C32], amp: f32) {
    for s in out {
        *s = C32::new(rng.unit() * amp, rng.unit() * amp);
    }
}

/// The amplitude of sample `i` of the `agc` input: it steps from 0.01 to 0.3 to 0.003 in
/// thirds, so the gain moves in both directions within a call (as in `benches/kernels`).
fn agc_amplitude(i: usize) -> f32 {
    [0.01, 0.3, 0.003][(3 * i / N).min(2)]
}

/// The `agc` parameters of `benches/kernels`: reference 0.5, attack 0.005, decay 0.05, gain
/// limits -20 and 60 dB, initial gain 0 dB, 48 kHz.
fn agc_state<T: sdrk::ComplexSample>() -> Option<AgcState<T>> {
    AgcState::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, Rate::hz(48_000).ok()?).ok()
}

fn exit(ok: bool) -> ! {
    debug::exit(if ok { EXIT_SUCCESS } else { EXIT_FAILURE });
    loop {
        cortex_m::asm::wfi();
    }
}

/// The last two words of the semihosting command line: the kernel name and `k`.
fn arguments(buf: &mut [u8; 128]) -> Option<(&str, usize)> {
    let mut block = [buf.as_mut_ptr() as usize, buf.len()];
    // SAFETY: SYS_GET_CMDLINE takes a pointer to a buffer and its size and writes the length
    // back into the second word. The block is passed by a mutable pointer: the host writes it,
    // so passing a shared reference (`syscall(.., &block)`) would let the compiler keep the old
    // length (seen at `opt-level = "s"`).
    let status = unsafe { syscall1(nr::GET_CMDLINE, block.as_mut_ptr() as usize) };
    if status != 0 {
        return None;
    }
    let len = block[1].min(buf.len());
    let text = core::str::from_utf8(buf.get(..len)?).ok()?;
    let mut words = text.split_whitespace().rev();
    let k = words.next()?.parse().ok()?;
    let name = words.next()?;
    Some((name, k))
}

#[inline(never)]
fn run_convert_cs8(rng: &mut Lcg, k: usize) -> bool {
    let mut bytes = [0u8; 2 * N];
    for b in &mut bytes {
        *b = rng.next() as u8;
    }
    let mut out = [C32::default(); N];
    for _ in 0..=k {
        let _ = black_box(Fallback.convert(&formats::CS8, black_box(&bytes), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_convert_cs16(rng: &mut Lcg, k: usize) -> bool {
    let mut bytes = [0u8; 4 * N];
    for b in &mut bytes {
        *b = rng.next() as u8;
    }
    let mut out = [C32::default(); N];
    for _ in 0..=k {
        let _ = black_box(Fallback.convert(&formats::CS16, black_box(&bytes), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_convert_out_cs16(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [0u8; 4 * N];
    for _ in 0..=k {
        let _ = black_box(Fallback.convert_out(&formats::CS16, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_rotate(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [C32::default(); N];
    let mut nco = Nco::from_increment(0x0123_4567);
    for _ in 0..=k {
        let _ = black_box(Fallback.rotate(&mut nco, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_fir(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [C32::default(); N];
    let Ok(mut state) = FirState::new(
        [1.0 / FIR_TAPS as f32; FIR_TAPS],
        [C32::default(); FIR_TAPS - 1],
    ) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fir(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_fir_decimate(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [C32::default(); N / DECIM + 2];
    let Ok(mut state) = FirDecimState::new(
        [1.0 / DECIM_TAPS as f32; DECIM_TAPS],
        [C32::default(); DECIM_TAPS - 1],
        DECIM,
    ) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fir_decimate(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_mag_sq(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.9);
    let mut out = [0f32; N];
    for _ in 0..=k {
        let _ = black_box(Fallback.mag_sq(black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_quad_demod(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    noise(rng, &mut input, 0.8);
    let mut out = [0f32; N];
    let mut state = QuadDemodState::<C32>::new();
    for _ in 0..=k {
        let _ = black_box(Fallback.quad_demod(&mut state, black_box(&input), &mut out, 1.0));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_fm_mod(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [0f32; N];
    for s in &mut input {
        *s = rng.unit();
    }
    let mut out = [C32::default(); N];
    let Ok(rate) = Rate::hz(240_000) else {
        return false;
    };
    let Ok(mut state) = FmModState::new(75_000.0, rate) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fm_mod(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_fft_1024(rng: &mut Lcg, k: usize) -> bool {
    let mut data = [C32::default(); 1024];
    noise(rng, &mut data, 0.5);
    let Ok(plan) = FftPlan::new(1024, [C32::default(); 512]) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::Default));
        black_box(&data);
    }
    true
}

#[inline(never)]
fn run_fft_cq15_1024(rng: &mut Lcg, k: usize) -> bool {
    let mut data = [CQ15::default(); 1024];
    for s in &mut data {
        s.re = Q15((rng.unit() * 8000.0) as i16);
        s.im = Q15((rng.unit() * 8000.0) as i16);
    }
    let Ok(plan) = FftPlan::new(1024, [CQ15::default(); 512]) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::BlockFloat));
        black_box(&data);
    }
    true
}

#[inline(never)]
fn run_agc(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [C32::default(); N];
    for (i, s) in input.iter_mut().enumerate() {
        let amp = agc_amplitude(i);
        *s = C32::new(rng.unit() * amp, rng.unit() * amp);
    }
    let mut out = [C32::default(); N];
    let Some(mut state) = agc_state::<C32>() else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.agc(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_agc_cq15(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [CQ15::default(); N];
    for (i, s) in input.iter_mut().enumerate() {
        let amp = agc_amplitude(i) * 32768.0;
        s.re = Q15((rng.unit() * amp) as i16);
        s.im = Q15((rng.unit() * amp) as i16);
    }
    let mut out = [CQ15::default(); N];
    let Some(mut state) = agc_state::<CQ15>() else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.agc(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_dc_block_real(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [0f32; N];
    for s in &mut input {
        *s = rng.unit() * 0.2 + 0.3;
    }
    let mut out = [0f32; N];
    let Ok(mut state) = DcBlockRealState::new(0.999_f32) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.dc_block_real(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

#[inline(never)]
fn run_dc_block_real_q15(rng: &mut Lcg, k: usize) -> bool {
    let mut input = [Q15::default(); N];
    for s in &mut input {
        *s = Q15(((rng.unit() * 0.2 + 0.3) * 32768.0) as i16);
    }
    let mut out = [Q15::default(); N];
    let Ok(mut state) = DcBlockRealState::new(Q15(32_735)) else {
        return false;
    };
    for _ in 0..=k {
        let _ = black_box(Fallback.dc_block_real(&mut state, black_box(&input), &mut out));
        black_box(&out);
    }
    true
}

/// Runs kernel `name`: one warm-up call and `k` counted calls. Each kernel has a function of its
/// own that is never inlined, so that the code generated for one kernel does not depend on the
/// others in the binary (in one function, adding a kernel changed the counts of the FIR and FFT
/// by up to 10 %).
fn run(name: &str, k: usize) -> bool {
    let mut rng = Lcg(0x1234_5678);
    let rng = &mut rng;
    match name {
        "convert_cs8" => run_convert_cs8(rng, k),
        "convert_cs16" => run_convert_cs16(rng, k),
        "convert_out_cs16" => run_convert_out_cs16(rng, k),
        "rotate" => run_rotate(rng, k),
        "fir" => run_fir(rng, k),
        "fir_decimate" => run_fir_decimate(rng, k),
        "mag_sq" => run_mag_sq(rng, k),
        "quad_demod" => run_quad_demod(rng, k),
        "fm_mod" => run_fm_mod(rng, k),
        "fft_1024" => run_fft_1024(rng, k),
        "fft_cq15_1024" => run_fft_cq15_1024(rng, k),
        "agc" => run_agc(rng, k),
        "agc_cq15" => run_agc_cq15(rng, k),
        "dc_block_real" => run_dc_block_real(rng, k),
        "dc_block_real_q15" => run_dc_block_real_q15(rng, k),
        _ => false,
    }
}

#[entry]
fn main() -> ! {
    let mut buf = [0u8; 128];
    let Some((name, k)) = arguments(&mut buf) else {
        if let Ok(mut out) = hio::hstdout() {
            let _ = writeln!(
                out,
                "sdrk-count error: no `<kernel> <k>` on the command line"
            );
        }
        exit(false)
    };
    let ok = run(name, k);
    if let Ok(mut out) = hio::hstdout() {
        let _ = writeln!(
            out,
            "sdrk-count {name} k={k} ok={ok} kernels={}",
            KERNELS.len()
        );
    }
    exit(ok)
}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    exit(false)
}
