//! Tests of the liquid-dsp `nco_crcf` reference of M24 (`LIQUID_VCO_INTERP`, `liquid-nco.c`):
//! its table against a re-computation of `nco_crcf_create`, its sine and cosine against `f64`,
//! and the pilot loop on a clean pilot. Run through the benchmark workspace with
//! `SDRK_REFS_DIR` set as xtask sets it; without the library the tests pass without checking.

use cref_sys::LiquidNcoPll;
use std::f64::consts::PI;

const RATE: f64 = 480_000.0;

/// `nco_crcf_fp_sin`: the sine of a 32-bit phase word in float.
fn fp_sin(x: u32) -> f32 {
    (x as f32 * std::f32::consts::PI / 2_147_483_648.0).sin()
}

/// The table of `nco_crcf_create` for `LIQUID_VCO_INTERP`, as the source computes it.
fn table() -> Vec<[f32; 2]> {
    const SIZE: usize = 1024;
    let (q, h) = (SIZE / 4, SIZE / 2);
    let mut t = vec![[0.0f32; 2]; SIZE];
    let d_theta = (u32::MAX / SIZE as u32) as i32;
    let mut theta = 0u32;
    for i in 0..q {
        let value = fp_sin(theta);
        let next = fp_sin(theta.wrapping_add(d_theta as u32));
        let skew = (next - value) / d_theta as f32;
        t[i] = [value, skew];
        t[i + h] = [-value, -skew];
        theta = theta.wrapping_add(d_theta as u32);
    }
    t[q] = [1.0, -t[q - 1][1]];
    t[q + h] = [-t[q][0], t[q - 1][1]];
    for i in 1..q {
        let ip = i + q;
        let (value, skew) = (t[q - i][0], t[q - i - 1][1]);
        t[ip] = [value, -skew];
        t[ip + h] = [-value, skew];
    }
    t
}

#[test]
fn the_table_is_the_one_of_nco_crcf_create() {
    let Some(nco) = LiquidNcoPll::new(19_000.0, RATE) else {
        eprintln!("liquid-dsp nco is not linked; skipped");
        return;
    };
    let got = nco.table();
    let want = table();
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            [g[0].to_bits(), g[1].to_bits()],
            [w[0].to_bits(), w[1].to_bits()],
            "entry {i}: {g:?} against {w:?}"
        );
    }
}

#[test]
fn sine_and_cosine_follow_the_phase() {
    let Some(mut nco) = LiquidNcoPll::new(19_000.0, RATE) else {
        eprintln!("liquid-dsp nco is not linked; skipped");
        return;
    };
    let mut worst = 0.0f64;
    for k in 0..4096 {
        let phi = 2.0 * PI * (f64::from(k) + 0.37) / 4096.0;
        nco.set_phase(phi as f32);
        let (s, c) = nco.sincos();
        // The phase word the double modulo of `_constrain` produced, not the float input.
        let exact = 2.0 * PI * f64::from(nco.phase_word()) / 4_294_967_296.0;
        worst = worst
            .max((f64::from(s) - exact.sin()).abs())
            .max((f64::from(c) - exact.cos()).abs());
    }
    // 1024 entries with linear interpolation: about 4.7e-6 (106.5 dB below full scale).
    assert!(worst < 6e-6, "worst error {worst}");
    assert!(
        worst > 1e-6,
        "worst error {worst}: expected the coarse table"
    );
}

#[test]
fn the_loop_locks_on_a_clean_pilot() {
    let Some(mut nco) = LiquidNcoPll::new(19_000.0, RATE) else {
        eprintln!("liquid-dsp nco is not linked; skipped");
        return;
    };
    // The gains of `stereo_pll` for pilot level 0.1 at 480 kHz, in radians, with the widest loop
    // the stereo kernel allows (natural_hz 190): liquid's `_constrain` reduces a correction
    // modulo two pi in `float` (a negative one becomes `2 pi - |c|` rounded to a multiple of
    // 4.8e-7 rad), so a narrower loop (natural_hz 20: 8 dB, 50: 27 dB, measured on this pilot)
    // does not track as the loop of `stereo_pll` does. That is a property of the reference.
    let kd = 0.05;
    let wn = 2.0 * PI * 190.0 / RATE;
    let zeta = 1.0 / 2f64.sqrt();
    let kp = (2.0 * zeta * wn / kd) as f32;
    let ki = (wn * wn / kd) as f32;
    let n = 96_000;
    let f = 19_000.5;
    let theta = |k: usize| 2.0 * PI * ((f * k as f64 / RATE).fract()) + 1.0;
    let mpx: Vec<f32> = (0..n)
        .map(|k| (0.5 * (2.0 * PI * k as f64 * 1000.0 / RATE).sin()) as f32)
        .collect();
    let pilot: Vec<f32> = (0..n).map(|k| (0.1 * theta(k).sin()) as f32).collect();
    let mut out = vec![[0.0f32; 2]; n];
    nco.pll_block(&mpx, &pilot, &mut out, kp, ki);
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for k in n / 2..n {
        let want = f64::from(mpx[k]) * (2.0 * theta(k)).sin();
        sig += want * want;
        err += (want - f64::from(out[k][1])).powi(2);
        assert_eq!(out[k][0], mpx[k]);
    }
    let snr = 10.0 * (sig / err).log10();
    assert!(snr > 30.0, "locked pilot: {snr} dB");
}
