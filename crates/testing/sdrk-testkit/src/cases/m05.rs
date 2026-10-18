//! Suite cases of the first kernels: `convert`, `rotate` and `fir`.
//!
//! Every case compares `Fallback` (and the facade) with `Reference`; no files
//! are read, so the cases run on microcontrollers too.

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::convert::{Convert, ConvertReal};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::num::q15_from_f64;
use sdrk_core::{
    C32, CQ15, ComplexSample, Fallback, Progress, Q15, Rate, Reference, Sample, Timing,
};
use sdrk_device::{Layout, SampleFormat, formats};

use crate::arena::Arena;
use crate::encode::{decode_reference, encode};
use crate::r#gen;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::{bit_exact, compare_complex};
use crate::rng::Pcg32;
use crate::suite::Failure;

fn ensure(cond: bool, f: impl FnOnce() -> Failure) -> Result<(), Failure> {
    if cond { Ok(()) } else { Err(f()) }
}

/// The formats the conversion kernels support (everything but `Bits1`).
const FORMATS: [SampleFormat; 12] = [
    formats::CU8,
    formats::CS8,
    formats::CS16,
    formats::CS12_LSB16,
    formats::SC16_Q11,
    formats::CS16_PLANAR,
    formats::PACKED12,
    formats::PACKED10,
    formats::I2S32_MSB,
    formats::CF32,
    formats::RF32,
    formats::RS16,
];

fn random_cq15(rng: &mut Pcg32, scale: f64) -> CQ15 {
    let mut v = || q15_from_f64((rng.next_f64() * 2.0 - 1.0) * scale);
    CQ15 { re: v(), im: v() }
}

fn random_c32(rng: &mut Pcg32, scale: f64) -> C32 {
    C32::new(
        ((rng.next_f64() * 2.0 - 1.0) * scale) as f32,
        ((rng.next_f64() * 2.0 - 1.0) * scale) as f32,
    )
}

fn check_real_format(
    format: &SampleFormat,
    raw: &[u8],
    oracle: &[Complex64],
    fail: &impl Fn(&'static str) -> Failure,
) -> Result<(), Failure> {
    const N: usize = 128;
    let (mut fa, mut fb, mut fc) = ([0f32; N], [0f32; N], [0f32; N]);
    let n = ConvertReal::<f32>::convert_real(&Fallback, format, raw, &mut fa)?;
    ensure(n == N, || fail("real sample count"))?;
    ConvertReal::<f32>::convert_real(&Reference, format, raw, &mut fb)?;
    ConvertReal::<f32>::convert_real(&Sdrk, format, raw, &mut fc)?;
    ensure(bit_exact(&fa, &fb).is_none(), || {
        fail("real f32: fallback differs from reference")
    })?;
    ensure(bit_exact(&fa, &fc).is_none(), || {
        fail("real f32: facade differs")
    })?;
    let (mut qa, mut qb, mut qc) = ([Q15(0); N], [Q15(0); N], [Q15(0); N]);
    ConvertReal::<Q15>::convert_real(&Fallback, format, raw, &mut qa)?;
    ConvertReal::<Q15>::convert_real(&Reference, format, raw, &mut qb)?;
    ConvertReal::<Q15>::convert_real(&Sdrk, format, raw, &mut qc)?;
    ensure(bit_exact(&qa, &qb).is_none(), || {
        fail("real q15: fallback differs from reference")
    })?;
    ensure(bit_exact(&qa, &qc).is_none(), || {
        fail("real q15: facade differs")
    })?;
    for ((f, q), want) in fa.iter().zip(qa.iter()).zip(oracle) {
        ensure(*f == want.re as f32, || {
            fail("real f32 differs from the oracle")
        })?;
        ensure(*q == q15_from_f64(want.re), || {
            fail("real q15 differs from the oracle")
        })?;
    }
    Ok(())
}

pub(super) fn convert_all_formats_vs_encoder(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "convert_all_formats_vs_encoder";
    const N: usize = 128;
    let mut c = arena.carver();
    let samples = c.take::<Complex64>(N)?;
    let oracle = c.take::<Complex64>(N)?;
    let bytes = c.take::<u8>(N * 8)?;
    let (c32_fb, c32_rf, c32_fac) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
    let (q_fb, q_rf, q_fac) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    let mut rng = Pcg32::new(501, 5);
    for s in samples.iter_mut() {
        *s = Complex64::new(
            (rng.next_f64() * 2.0 - 1.0) * 0.99,
            (rng.next_f64() * 2.0 - 1.0) * 0.99,
        );
    }
    for (index, format) in FORMATS.iter().enumerate() {
        let fail =
            |msg: &'static str| Failure::new(CASE, msg).detail(format_args!("format {index}"));
        let used = encode(format, samples, bytes)?;
        let raw = bytes.get(..used).unwrap_or_default();
        ensure(decode_reference(format, raw, oracle)? == N, || {
            fail("oracle sample count")
        })?;
        if format.layout() == Layout::Real {
            check_real_format(format, raw, oracle, &fail)?;
            continue;
        }
        ensure(
            Convert::<C32>::convert(&Fallback, format, raw, c32_fb)? == N,
            || fail("sample count"),
        )?;
        Convert::<C32>::convert(&Reference, format, raw, c32_rf)?;
        Convert::<C32>::convert(&Sdrk, format, raw, c32_fac)?;
        ensure(bit_exact(c32_fb, c32_rf).is_none(), || {
            fail("c32: fallback differs from reference")
        })?;
        ensure(bit_exact(c32_fb, c32_fac).is_none(), || {
            fail("c32: facade differs")
        })?;
        Convert::<CQ15>::convert(&Fallback, format, raw, q_fb)?;
        Convert::<CQ15>::convert(&Reference, format, raw, q_rf)?;
        Convert::<CQ15>::convert(&Sdrk, format, raw, q_fac)?;
        ensure(bit_exact(q_fb, q_rf).is_none(), || {
            fail("cq15: fallback differs from reference")
        })?;
        ensure(bit_exact(q_fb, q_fac).is_none(), || {
            fail("cq15: facade differs")
        })?;
        for ((f, q), want) in c32_fb.iter().zip(q_fb.iter()).zip(oracle.iter()) {
            let same = f.re == want.re as f32 && f.im == want.im as f32;
            ensure(same, || fail("c32 differs from the oracle"))?;
            let q_same = q.re == q15_from_f64(want.re) && q.im == q15_from_f64(want.im);
            ensure(q_same, || fail("cq15 differs from the oracle"))?;
        }
    }
    Ok(())
}

pub(super) fn rotate_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "rotate_c32_vs_reference";
    const N: usize = 1024;
    let mut c = arena.carver();
    let (input, fb, rf, fac) = (
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
    );
    let want = c.take::<Complex64>(N)?;
    let mut rng = Pcg32::new(502, 1);
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 0.9);
    }
    for (inc, phase) in [
        (0x1F2E_3D4Cu32, 0u32),
        (0xC000_0001, 0xFFFF_FF00),
        (1 << 30, 77),
    ] {
        let (mut a, mut b, mut d) = (
            Nco::from_increment(inc),
            Nco::from_increment(inc),
            Nco::from_increment(inc),
        );
        for n in [&mut a, &mut b, &mut d] {
            n.set_phase(phase);
        }
        Rotate::<C32>::rotate(&Fallback, &mut a, input, fb)?;
        Rotate::<C32>::rotate(&Reference, &mut b, input, rf)?;
        Rotate::<C32>::rotate(&Sdrk, &mut d, input, fac)?;
        for (w, r) in want.iter_mut().zip(rf.iter()) {
            *w = r.to_c64();
        }
        let cmp = compare_complex(want, fb.iter().map(|s| s.to_c64()), 0);
        ensure(cmp.snr_db >= 120.0, || {
            Failure::new(CASE, "below 120 dB")
                .detail(format_args!("inc {inc:#x}: {} dB", cmp.snr_db))
        })?;
        ensure(bit_exact(fb, fac).is_none(), || {
            Failure::new(CASE, "facade differs")
        })?;
        ensure(a == b && a == d, || {
            Failure::new(CASE, "phase advanced differently")
        })?;
    }
    Ok(())
}

pub(super) fn rotate_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "rotate_cq15_vs_reference";
    const N: usize = 1024;
    let mut c = arena.carver();
    let (input, fb, rf, fac) = (
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
    );
    let want = c.take::<Complex64>(N)?;
    let mut rng = Pcg32::new(503, 1);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 0.5);
    }
    for (inc, phase) in [
        (0x0CCC_CCCDu32, 0u32),
        (0xC000_0001, 0xFFFF_FF00),
        (0x7FFF_FFFF, 9),
    ] {
        let (mut a, mut b, mut d) = (
            Nco::from_increment(inc),
            Nco::from_increment(inc),
            Nco::from_increment(inc),
        );
        for n in [&mut a, &mut b, &mut d] {
            n.set_phase(phase);
        }
        Rotate::<CQ15>::rotate(&Fallback, &mut a, input, fb)?;
        Rotate::<CQ15>::rotate(&Reference, &mut b, input, rf)?;
        Rotate::<CQ15>::rotate(&Sdrk, &mut d, input, fac)?;
        for (w, r) in want.iter_mut().zip(rf.iter()) {
            *w = r.to_c64();
        }
        let cmp = compare_complex(want, fb.iter().map(|s| s.to_c64()), 0);
        ensure(cmp.snr_db >= 80.0, || {
            Failure::new(CASE, "below 80 dB")
                .detail(format_args!("inc {inc:#x}: {} dB", cmp.snr_db))
        })?;
        ensure(bit_exact(fb, fac).is_none(), || {
            Failure::new(CASE, "facade differs")
        })?;
        ensure(a == b && a == d, || {
            Failure::new(CASE, "phase advanced differently")
        })?;
    }
    Ok(())
}

fn same_rate_step<T: ComplexSample>(
    input: &[T],
    output: &mut [T],
    kernel: impl FnOnce(&[T], &mut [T]) -> sdrk_core::Result<()>,
) -> sdrk_core::Result<Progress> {
    let n = input.len().min(output.len());
    let internal = || sdrk_core::Error::Internal("slice split");
    let (i, _) = input.split_at_checked(n).ok_or_else(internal)?;
    let (o, _) = output.split_at_mut_checked(n).ok_or_else(internal)?;
    kernel(i, o)?;
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

pub(super) fn rotate_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const N: usize = 1024;
    let mut rng = Pcg32::new(504, 2);
    {
        let mut c = arena.carver();
        let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 1.0);
        }
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(41, 61),
            || Nco::from_increment(0x1234_5678),
            |nco, i, o| same_rate_step(i, o, |i, o| Rotate::<C32>::rotate(&Sdrk, nco, i, o)),
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 1.0);
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(42, 47),
        || Nco::from_increment(0xDEAD_BEEF),
        |nco, i, o| same_rate_step(i, o, |i, o| Rotate::<CQ15>::rotate(&Sdrk, nco, i, o)),
    )?;
    Ok(())
}

/// Hamming-windowed sinc low-pass with `N` taps and cutoff `fc` (cycles/sample), DC gain 1.
fn lowpass<const N: usize>(fc: f64) -> [f64; N] {
    let mut h = [0.0f64; N];
    let mid = (N - 1) as f64 / 2.0;
    let mut sum = 0.0;
    for (k, t) in h.iter_mut().enumerate() {
        let x = k as f64 - mid;
        let sinc = if x == 0.0 {
            1.0
        } else {
            libm::sin(2.0 * core::f64::consts::PI * fc * x) / (core::f64::consts::PI * x)
        };
        let window =
            0.54 - 0.46 * libm::cos(2.0 * core::f64::consts::PI * k as f64 / (N - 1) as f64);
        *t = sinc * window;
        sum += *t;
    }
    for t in &mut h {
        *t /= sum;
    }
    h
}

pub(super) fn fir_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_c32_vs_reference";
    const N: usize = 1024;
    const TAPS: usize = 31;
    let mut c = arena.carver();
    let (input, fb, rf, fac) = (
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
    );
    let want = c.take::<Complex64>(N)?;
    let mut rng = Pcg32::new(505, 1);
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 0.9);
    }
    let taps = lowpass::<TAPS>(0.2).map(|t| t as f32);
    let new_state = || FirState::<C32, _, _>::new(taps, [C32::ZERO; TAPS - 1]);
    let (mut a, mut b, mut d) = (new_state()?, new_state()?, new_state()?);
    Fir::<C32>::fir(&Fallback, &mut a, input, fb)?;
    Fir::<C32>::fir(&Reference, &mut b, input, rf)?;
    Fir::<C32>::fir(&Sdrk, &mut d, input, fac)?;
    for (w, r) in want.iter_mut().zip(rf.iter()) {
        *w = r.to_c64();
    }
    let cmp = compare_complex(want, fb.iter().map(|s| s.to_c64()), 0);
    ensure(cmp.snr_db >= 110.0, || {
        Failure::new(CASE, "below 110 dB").detail(format_args!("{} dB", cmp.snr_db))
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs")
    })?;
    ensure(
        a.history() == b.history() && a.history() == d.history(),
        || Failure::new(CASE, "histories differ"),
    )
}

pub(super) fn fir_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_cq15_vs_reference";
    const N: usize = 1024;
    const TAPS: usize = 31;
    let mut c = arena.carver();
    let (input, fb, rf, fac) = (
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
    );
    let mut rng = Pcg32::new(506, 1);
    let taps = lowpass::<TAPS>(0.2).map(q15_from_f64);
    for extremes in [false, true] {
        if extremes {
            r#gen::extreme::full_scale_alternating(input);
        } else {
            for s in input.iter_mut() {
                *s = random_cq15(&mut rng, 1.0);
            }
        }
        let new_state = || FirState::<CQ15, _, _>::new(taps, [CQ15::ZERO; TAPS - 1]);
        let (mut a, mut b, mut d) = (new_state()?, new_state()?, new_state()?);
        Fir::<CQ15>::fir(&Fallback, &mut a, input, fb)?;
        Fir::<CQ15>::fir(&Reference, &mut b, input, rf)?;
        Fir::<CQ15>::fir(&Sdrk, &mut d, input, fac)?;
        ensure(bit_exact(fb, rf).is_none(), || {
            Failure::new(CASE, "not bit-exact to the reference")
        })?;
        ensure(bit_exact(fb, fac).is_none(), || {
            Failure::new(CASE, "facade differs")
        })?;
        ensure(
            a.history() == b.history() && a.history() == d.history(),
            || Failure::new(CASE, "histories differ"),
        )?;
    }
    Ok(())
}

pub(super) fn fir_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const N: usize = 1024;
    const TAPS: usize = 17;
    let mut rng = Pcg32::new(507, 2);
    {
        let mut c = arena.carver();
        let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 1.0);
        }
        let taps = lowpass::<TAPS>(0.15).map(|t| t as f32);
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(43, 53),
            || FirState::<C32, _, _>::new(taps, [C32::ZERO; TAPS - 1]),
            |state, i, o| match state {
                Ok(st) => same_rate_step(i, o, |i, o| Fir::<C32>::fir(&Sdrk, st, i, o)),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 1.0);
    }
    let taps = lowpass::<TAPS>(0.15).map(q15_from_f64);
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(44, 59),
        || FirState::<CQ15, _, _>::new(taps, [CQ15::ZERO; TAPS - 1]),
        |state, i, o| match state {
            Ok(st) => same_rate_step(i, o, |i, o| Fir::<CQ15>::fir(&Sdrk, st, i, o)),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

pub(super) fn fir_impulse_response(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_impulse_response";
    const TAPS: usize = 9;
    const N: usize = 24;
    let _ = arena;
    let h32 = [0.25f32, -0.5, 0.125, 0.75, -1.0, 0.5, 0.0625, -0.25, 0.5];
    let hq = [
        8192i16, -16384, 4096, 24576, -32768, 16384, 2048, -8192, 16384,
    ]
    .map(Q15);
    let mut impulse32 = [C32::ZERO; N];
    impulse32[0] = C32::new(1.0, -1.0);
    let mut out32 = [C32::ZERO; N];
    let mut st = FirState::<C32, _, _>::new(h32, [C32::ZERO; TAPS - 1])?;
    Fir::<C32>::fir(&Sdrk, &mut st, &impulse32, &mut out32)?;
    for (k, y) in out32.iter().enumerate() {
        let h = h32.get(k).copied().unwrap_or(0.0);
        ensure(y.re == h && y.im == -h, || {
            Failure::new(CASE, "c32 impulse response differs").detail(format_args!("index {k}"))
        })?;
    }
    // A Q15 impulse of 32767 reproduces the taps up to the 2^-15 scale error.
    let mut impulse_q = [CQ15::ZERO; N];
    impulse_q[0] = CQ15 {
        re: Q15(32767),
        im: Q15(-32768),
    };
    let mut out_q = [CQ15::ZERO; N];
    let mut st = FirState::<CQ15, _, _>::new(hq, [CQ15::ZERO; TAPS - 1])?;
    Fir::<CQ15>::fir(&Sdrk, &mut st, &impulse_q, &mut out_q)?;
    for (k, y) in out_q.iter().enumerate() {
        let h = i32::from(hq.get(k).map_or(0, |q| q.0));
        let want_re = sdrk_core::num::sat16(sdrk_core::num::round_shift(i64::from(h) * 32767, 15));
        let want_im = sdrk_core::num::sat16(sdrk_core::num::round_shift(-i64::from(h) * 32768, 15));
        ensure(y.re.0 == want_re && y.im.0 == want_im, || {
            Failure::new(CASE, "cq15 impulse response differs").detail(format_args!("index {k}"))
        })?;
    }
    ensure(st.group_delay() == sdrk_core::Delay::new(8, 2)?, || {
        Failure::new(CASE, "wrong group delay")
    })
}

pub(super) fn extremes_convert_rotate_fir(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_convert_rotate_fir";
    const N: usize = 512;
    let mut c = arena.carver();
    let (c32, c32_out) = (c.take::<C32>(N)?, c.take::<C32>(N)?);
    let (cq, cq_out) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    let bytes = c.take::<u8>(N * 8)?;
    let (again, again_q) = (c.take::<C32>(N)?, c.take::<CQ15>(N)?);
    let rate = Rate::hz(1000)?;

    // Silence, full scale and f32 specials through rotate and fir.
    r#gen::extreme::f32_specials_complex(c32);
    let mut nco = Nco::new(123.0, rate)?;
    Rotate::<C32>::rotate(&Sdrk, &mut nco, c32, c32_out)?;
    ensure(
        c32_out.iter().any(|s| s.re.is_nan() || s.im.is_nan()),
        || Failure::new(CASE, "NaN did not propagate through rotate"),
    )?;
    let taps = [0.25f32, 0.5, 0.25];
    let mut st = FirState::<C32, _, _>::new(taps, [C32::ZERO; 2])?;
    Fir::<C32>::fir(&Sdrk, &mut st, c32, c32_out)?;
    ensure(
        c32_out.iter().any(|s| s.re.is_nan() || s.im.is_nan()),
        || Failure::new(CASE, "NaN did not propagate through fir"),
    )?;

    for fill in 0..2 {
        if fill == 0 {
            r#gen::extreme::silence(cq);
        } else {
            r#gen::extreme::full_scale_alternating(cq);
        }
        let mut nco = Nco::from_increment(0x4000_0001);
        Rotate::<CQ15>::rotate(&Sdrk, &mut nco, cq, cq_out)?;
        if fill == 0 {
            ensure(cq_out.iter().all(|s| *s == CQ15::ZERO), || {
                Failure::new(CASE, "silence did not stay silent")
            })?;
        }
        let mut gain = FirState::<CQ15, _, _>::new([Q15(32767); 8], [CQ15::ZERO; 7])?;
        Fir::<CQ15>::fir(&Sdrk, &mut gain, cq, cq_out)?;
        let mut reference = FirState::<CQ15, _, _>::new([Q15(32767); 8], [CQ15::ZERO; 7])?;
        let mut expect = [CQ15::ZERO; 8];
        Fir::<CQ15>::fir(
            &Reference,
            &mut reference,
            cq.get(..8).unwrap_or_default(),
            &mut expect,
        )?;
        ensure(cq_out.get(..8) == Some(&expect[..]), || {
            Failure::new(CASE, "saturating fir differs from the reference")
        })?;
    }

    // Empty slices are fine and leave the state alone.
    let mut nco = Nco::from_increment(5);
    Rotate::<CQ15>::rotate(&Sdrk, &mut nco, &[], &mut [])?;
    ensure(nco.phase() == 0, || {
        Failure::new(CASE, "empty rotate moved the phase")
    })?;

    // Whole conversions of random bytes never fail and agree with the reference.
    let mut rng = Pcg32::new(508, 3);
    for b in bytes.iter_mut() {
        *b = (rng.next_u32() >> 24) as u8;
    }
    for format in FORMATS.iter().filter(|f| f.layout() != Layout::Real) {
        let raw = bytes
            .get(..format.bytes_for_samples(N)?)
            .unwrap_or_default();
        Convert::<C32>::convert(&Fallback, format, raw, c32_out)?;
        Convert::<C32>::convert(&Reference, format, raw, again)?;
        ensure(bit_exact(c32_out, again).is_none(), || {
            Failure::new(CASE, "random bytes: c32 differs from the reference")
        })?;
        Convert::<CQ15>::convert(&Fallback, format, raw, cq_out)?;
        Convert::<CQ15>::convert(&Reference, format, raw, again_q)?;
        ensure(bit_exact(cq_out, again_q).is_none(), || {
            Failure::new(CASE, "random bytes: cq15 differs from the reference")
        })?;
    }
    Ok(())
}
