//! Generates the cosine tables used by the NCO mixer (`rotate` kernel), the inverse-CDF table of
//! the `gauss_noise` kernel and, with the feature `static-twiddles`, the 4096-point FFT twiddle
//! tables.
//!
//! `COS_F32[k] = cos(2 pi k / 4096)` and `COS_Q15[k]` is the same value rounded
//! to Q15 (round half up, saturating), both computed in f64 with the pure-Rust
//! `libm`, so every host and target gets bit-identical tables.

// A build script talks to cargo through stdout and may use `std`.
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::fmt::Write as _;
use std::path::PathBuf;

const SIZE: usize = 4096;

fn q15_from_f64(x: f64) -> i16 {
    let v = libm::floor(x * 32768.0 + 0.5);
    if v >= 32767.0 {
        32767
    } else if v <= -32768.0 {
        -32768
    } else {
        v as i16
    }
}

/// `(re, im)` of `e^(-j 2 pi k / n)`; the same expression as `kernels::fft::twiddle_f64`.
fn twiddle(k: usize, n: usize) -> (f64, f64) {
    let theta = 2.0 * core::f64::consts::PI * (k as f64 / n as f64);
    (libm::cos(theta), -libm::sin(theta))
}

/// The static FFT twiddle tables of feature `static-twiddles`.
fn emit_twiddles(src: &mut String) {
    const N: usize = 4096;
    src.push_str("\n/// `e^(-j 2 pi k / 4096)` for `k < 2048` as `C32` (16 KiB).\n");
    writeln!(
        src,
        "pub static TWIDDLES_C32_4096: [crate::C32; {}] = [",
        N / 2
    )
    .unwrap();
    for k in 0..N / 2 {
        let (re, im) = twiddle(k, N);
        writeln!(
            src,
            "    crate::C32 {{ re: {:?}_f32, im: {:?}_f32 }},",
            re as f32, im as f32
        )
        .unwrap();
    }
    src.push_str("];\n\n/// The same table rounded to Q15 (8 KiB).\n");
    writeln!(
        src,
        "pub static TWIDDLES_CQ15_4096: [crate::CQ15; {}] = [",
        N / 2
    )
    .unwrap();
    for k in 0..N / 2 {
        let (re, im) = twiddle(k, N);
        writeln!(
            src,
            "    crate::CQ15 {{ re: crate::Q15({}), im: crate::Q15({}) }},",
            q15_from_f64(re),
            q15_from_f64(im)
        )
        .unwrap();
    }
    src.push_str("];\n");
}

/// Evaluates the polynomial with the coefficients `c` (lowest degree first) at `x` (Horner).
fn horner(c: &[f64; 8], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &k| acc * x + k)
}

/// The inverse normal CDF `Φ⁻¹(p)` for `p` in `(0, 1)`: Wichura's algorithm AS 241 (`PPND16`,
/// Applied Statistics 37 (1988) 477–484), relative error about 1e-16, in f64 with `libm`.
// The coefficients keep the digits of the published algorithm, so the transcription can be
// checked against it digit by digit.
#[allow(clippy::excessive_precision)]
fn ppnd16(p: f64) -> f64 {
    const A: [f64; 8] = [
        3.387_132_872_796_366_608_0e0,
        1.331_416_678_917_843_774_5e2,
        1.971_590_950_306_551_442_7e3,
        1.373_169_376_550_946_112_5e4,
        4.592_195_393_154_987_145_7e4,
        6.726_577_092_700_870_085_3e4,
        3.343_057_558_358_812_810_5e4,
        2.509_080_928_730_122_672_7e3,
    ];
    const B: [f64; 8] = [
        1.0,
        4.231_333_070_160_091_125_2e1,
        6.871_870_074_920_579_083_0e2,
        5.394_196_021_424_751_107_7e3,
        2.121_379_430_158_659_586_7e4,
        3.930_789_580_009_271_061_0e4,
        2.872_908_573_572_194_267_4e4,
        5.226_495_278_852_854_561_0e3,
    ];
    const C: [f64; 8] = [
        1.423_437_110_749_683_577_34e0,
        4.630_337_846_156_545_295_90e0,
        5.769_497_221_460_691_405_50e0,
        3.647_848_324_763_204_605_04e0,
        1.270_458_252_452_368_382_58e0,
        2.417_807_251_774_506_117_70e-1,
        2.272_384_498_926_918_458_33e-2,
        7.745_450_142_783_414_076_40e-4,
    ];
    const D: [f64; 8] = [
        1.0,
        2.053_191_626_637_758_821_87e0,
        1.676_384_830_183_803_849_40e0,
        6.897_673_349_851_000_045_50e-1,
        1.481_039_764_274_800_745_90e-1,
        1.519_866_656_361_645_719_66e-2,
        5.475_938_084_995_344_946_00e-4,
        1.050_750_071_644_416_843_24e-9,
    ];
    const E: [f64; 8] = [
        6.657_904_643_501_103_777_20e0,
        5.463_784_911_164_114_369_90e0,
        1.784_826_539_917_291_335_80e0,
        2.965_605_718_285_048_912_30e-1,
        2.653_218_952_657_612_309_30e-2,
        1.242_660_947_388_078_438_60e-3,
        2.711_555_568_743_487_578_15e-5,
        2.010_334_399_292_288_132_65e-7,
    ];
    const F: [f64; 8] = [
        1.0,
        5.998_322_065_558_879_376_90e-1,
        1.369_298_809_227_358_053_10e-1,
        1.487_536_129_085_061_485_25e-2,
        7.868_691_311_456_132_591_00e-4,
        1.846_318_317_510_054_681_80e-5,
        1.421_511_758_316_445_888_70e-7,
        2.044_263_103_389_939_785_64e-15,
    ];
    let q = p - 0.5;
    if libm::fabs(q) <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * horner(&A, r) / horner(&B, r);
    }
    let r = libm::sqrt(-libm::log(if q < 0.0 { p } else { 1.0 - p }));
    let v = if r <= 5.0 {
        horner(&C, r - 1.6) / horner(&D, r - 1.6)
    } else {
        horner(&E, r - 5.0) / horner(&F, r - 5.0)
    };
    if q < 0.0 { -v } else { v }
}

/// The inverse-CDF table of the `gauss_noise` kernel (M48 §2.2): 32 octaves `L` of the
/// quantile `q = x / 2^33` in `(0, 1/2)`, 32 segments each. Knot `K[L][j] = -Φ⁻¹((32 + j) ·
/// 2^(L - 38))` for `j = 0 ..= 32` (with `K[31][32] = 0` at `q = 1/2`), `B = floor(K · 2^17 +
/// 0.5)` in Q17, and entry `(L << 5) | j` packs `(B[L][j] << 12) | (B[L][j] - B[L][j + 1])`.
///
/// The asserts fail the build when a property of the contract does not hold: continuity across
/// the octaves, the first knot 830729, non-increasing knots, slopes in `1 ..= 4095`, the last
/// knot 0, and the rounding-tie guard (every `K · 2^17` at least 1e-4 LSB away from a `.5`
/// boundary, so any correct f64 inverse normal gives the same integers).
fn emit_gauss_icdf(src: &mut String) {
    let mut knots = vec![[0_i64; 33]; 32];
    let mut tie_min = f64::INFINITY;
    for (l, row) in knots.iter_mut().enumerate() {
        for (j, knot) in row.iter_mut().enumerate() {
            if l == 31 && j == 32 {
                continue; // q = 1/2: the knot is exactly 0.
            }
            let p = libm::ldexp((32 + j) as f64, l as i32 - 38);
            let y = -ppnd16(p) * 131_072.0;
            let tie = libm::fabs(y - libm::floor(y) - 0.5);
            tie_min = tie_min.min(tie);
            *knot = libm::floor(y + 0.5) as i64;
        }
    }
    assert!(
        tie_min >= 1e-4,
        "gauss_icdf: a knot lies {tie_min} LSB from a rounding tie"
    );
    assert_eq!(knots[0][0], 830_729, "gauss_icdf: the first knot");
    assert_eq!(knots[31][32], 0, "gauss_icdf: the last knot");
    for l in 0..31 {
        assert_eq!(
            knots[l][32],
            knots[l + 1][0],
            "gauss_icdf: continuity at octave {l}"
        );
    }
    src.push_str(
        "\n/// The inverse-CDF table of the `gauss_noise` kernel (M48): entry `(L << 5) | j` packs \
         the Q17\n/// knot `B[L][j]` (bits 12 and up) and the slope `B[L][j] - B[L][j + 1]` \
         (bits 0 to 11), see\n/// `kernels::noise::normal_q17`.\n",
    );
    src.push_str("pub static GAUSS_ICDF: [u32; 1024] = [\n");
    for (l, row) in knots.iter().enumerate() {
        for j in 0..32 {
            let base = row[j];
            let slope = base - row[j + 1];
            assert!(
                (1..4096).contains(&slope),
                "gauss_icdf: slope {slope} at octave {l} segment {j}"
            );
            assert!(
                base < 1 << 20,
                "gauss_icdf: knot {base} needs more than 20 bits"
            );
            let entry = u32::try_from((base << 12) | slope).unwrap();
            writeln!(src, "    0x{entry:08x},").unwrap();
        }
    }
    src.push_str("];\n");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut src = String::new();
    src.push_str("/// `cos(2 pi k / 4096)` as f32 for `k` in `0..=4096` (the last entry closes the period).\n");
    writeln!(src, "pub static COS_F32: [f32; {}] = [", SIZE + 1).unwrap();
    for k in 0..=SIZE {
        let c = libm::cos(2.0 * core::f64::consts::PI * k as f64 / SIZE as f64);
        writeln!(src, "    {:?}_f32,", c as f32).unwrap();
    }
    src.push_str("];\n\n");
    src.push_str("/// The same table rounded to Q15 raw values.\n");
    writeln!(src, "pub static COS_Q15: [i16; {}] = [", SIZE + 1).unwrap();
    for k in 0..=SIZE {
        let c = libm::cos(2.0 * core::f64::consts::PI * k as f64 / SIZE as f64);
        writeln!(src, "    {},", q15_from_f64(c)).unwrap();
    }
    src.push_str("];\n");
    emit_gauss_icdf(&mut src);
    if std::env::var_os("CARGO_FEATURE_STATIC_TWIDDLES").is_some() {
        emit_twiddles(&mut src);
    }
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("tables.rs");
    std::fs::write(out, src).unwrap();
}
