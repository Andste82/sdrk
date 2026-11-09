//! Square QAM constellations (M44 §1): the mapping of `qam16` and `qam64` shared by the
//! modulator, the slicer, the equalizer and the carrier loop.
//!
//! A symbol of `M = 16` or `64` points carries `k = log2 M` bits, most significant first. The
//! first `k / 2` bits select the in-phase level, the last `k / 2` the quadrature level, each by
//! the reflected Gray code of the level index: level index `i` (`0 .. m - 1`, `m = √M`) has the
//! code `i ^ (i >> 1)` and the level value `2 i - (m - 1)`. The points are scaled by `1 / √E`,
//! `E = 2 (M - 1) / 3`, so the average power is 1. The decision is a grid slicer per axis, as
//! GNU Radio's `constellation_rect`: the level index nearest to `(y √E + (m - 1)) / 2`, clamped to
//! `0 .. m - 1`. A rotation by 90° maps the bit groups `(I, Q)` of a symbol to
//! `(Q with its first bit inverted, I)` ([`rotate_symbol`]).
//!
//! ```
//! use sdrk_core::{kernels::constellation::Qam, C32};
//! let q = Qam::Qam16;
//! assert_eq!((q.points(), q.bits_per_symbol(), q.levels_per_axis()), (16, 4, 4));
//! // Symbol 0b1110: I bits 11 (level index 2, +1), Q bits 10 (level index 3, +3).
//! let p = q.point(0b1110);
//! assert!((p.re - 1.0 / 10f32.sqrt()).abs() < 1e-7 && (p.im - 3.0 / 10f32.sqrt()).abs() < 1e-7);
//! assert_eq!(q.decide_symbol(C32::new(0.3, 0.9)), 0b1110);
//! assert!((q.cma_r2() - 1.32).abs() < 1e-12);
//! ```

use crate::types::C32;

/// The level values of 16-QAM in level-index order, `(2 i - 3) / √10`, each the `f32` of the
/// `f64` value.
#[allow(
    clippy::excessive_precision,
    reason = "the literals are the f64 values l / sqrt(10), rounded once to f32 by the cast"
)]
pub const QAM16_LEVELS: [f32; 4] = [
    -0.948_683_298_050_513_8_f64 as f32,
    -0.316_227_766_016_837_94_f64 as f32,
    0.316_227_766_016_837_94_f64 as f32,
    0.948_683_298_050_513_8_f64 as f32,
];

/// The level values of 64-QAM in level-index order, `(2 i - 7) / √42`, each the `f32` of the
/// `f64` value.
#[allow(
    clippy::excessive_precision,
    reason = "the literals are the f64 values l / sqrt(42), rounded once to f32 by the cast"
)]
pub const QAM64_LEVELS: [f32; 8] = [
    -1.080_123_449_734_643_2_f64 as f32,
    -0.771_516_749_810_459_5_f64 as f32,
    -0.462_910_049_886_275_7_f64 as f32,
    -0.154_303_349_962_091_9_f64 as f32,
    0.154_303_349_962_091_9_f64 as f32,
    0.462_910_049_886_275_7_f64 as f32,
    0.771_516_749_810_459_5_f64 as f32,
    1.080_123_449_734_643_2_f64 as f32,
];

/// `√10` rounded once to `f32`: the factor from the 16-QAM points to the integer grid.
#[allow(
    clippy::excessive_precision,
    reason = "the literal is the f64 of sqrt(10), rounded once to f32 by the cast"
)]
pub const QAM16_SQRT_E: f32 = 3.162_277_660_168_379_5_f64 as f32;
/// `√42` rounded once to `f32`: the factor from the 64-QAM points to the integer grid.
#[allow(
    clippy::excessive_precision,
    reason = "the literal is the f64 of sqrt(42), rounded once to f32 by the cast"
)]
pub const QAM64_SQRT_E: f32 = 6.480_740_698_407_86_f64 as f32;

/// `√10 / 2` rounded once to `f32`: the factor of the grid decision of 16-QAM.
#[allow(
    clippy::excessive_precision,
    reason = "the literal is the f64 of sqrt(10) / 2, rounded once to f32 by the cast"
)]
const QAM16_HALF_SQRT_E: f32 = 1.581_138_830_084_189_8_f64 as f32;
/// `√42 / 2` rounded once to `f32`: the factor of the grid decision of 64-QAM.
#[allow(
    clippy::excessive_precision,
    reason = "the literal is the f64 of sqrt(42) / 2, rounded once to f32 by the cast"
)]
const QAM64_HALF_SQRT_E: f32 = 3.240_370_349_203_93_f64 as f32;

/// The reflected Gray code of the level index `i`, `i ^ (i >> 1)`.
///
/// ```
/// use sdrk_core::kernels::constellation::gray;
/// assert_eq!([0, 1, 2, 3].map(gray), [0, 1, 3, 2]);
/// ```
#[inline]
pub const fn gray(i: u32) -> u32 {
    i ^ (i >> 1)
}

/// The level index whose Gray code is `g` (the inverse of [`gray`] for codes below `2^8`).
///
/// ```
/// use sdrk_core::kernels::constellation::{gray, level_of_code};
/// for i in 0..8 {
///     assert_eq!(level_of_code(gray(i)), i);
/// }
/// ```
#[inline]
pub const fn level_of_code(g: u32) -> u32 {
    let mut i = g;
    i ^= i >> 1;
    i ^= i >> 2;
    i ^= i >> 4;
    i
}

/// Builds the point table of a square QAM with `m` levels per axis (`b` bits per axis) from its
/// level values: entry `s` is the point of symbol `s`.
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
const fn points_of<const N: usize>(levels: &[f32], b: u32) -> [C32; N] {
    let mut out = [C32 { re: 0.0, im: 0.0 }; N];
    let mask = (1u32 << b) - 1;
    let mut s = 0;
    while s < N {
        let i = level_of_code((s as u32 >> b) & mask) as usize;
        let q = level_of_code(s as u32 & mask) as usize;
        out[s] = C32 {
            re: levels[i],
            im: levels[q],
        };
        s += 1;
    }
    out
}

/// The 16-QAM points, entry `s` the point of the symbol `s` (bits `b0 b1 b2 b3`, `b0` the most
/// significant: `b0 b1` the in-phase code, `b2 b3` the quadrature code).
pub const QAM16_POINTS: [C32; 16] = points_of::<16>(&QAM16_LEVELS, 2);
/// The 64-QAM points, entry `s` the point of the symbol `s` (bits `b0 .. b5`, `b0` the most
/// significant: `b0 b1 b2` the in-phase code, `b3 b4 b5` the quadrature code).
pub const QAM64_POINTS: [C32; 64] = points_of::<64>(&QAM64_LEVELS, 3);

/// `Σ |p|⁴` over the points of a square QAM with `m` levels per axis, on the integer grid
/// (level values `2 i - (m - 1)`).
const fn sum_fourth_power(m: i64) -> i64 {
    let mut sum = 0;
    let mut a = -(m - 1);
    while a < m {
        let mut b = -(m - 1);
        while b < m {
            let p = a * a + b * b;
            sum += p * p;
            b += 2;
        }
        a += 2;
    }
    sum
}

/// `R2 = E|s|⁴ / E|s|²` of the unit-power square QAM with `m` levels per axis: on the integer
/// grid `E|s|² = E = 2 (M - 1) / 3`, so `R2 = Σ|p|⁴ / (M E²) = 9 Σ|p|⁴ / (4 M (M - 1)²)`, one
/// division of two exact integers.
const fn r2_of(m: i64) -> f64 {
    let big_m = m * m;
    (9 * sum_fourth_power(m)) as f64 / (4 * big_m * (big_m - 1) * (big_m - 1)) as f64
}

/// A square QAM constellation of M44.
///
/// ```
/// use sdrk_core::kernels::constellation::Qam;
/// assert_eq!(Qam::Qam64.bits_per_symbol(), 6);
/// assert!((Qam::Qam64.cma_r2() - 1.381).abs() < 5e-4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Qam {
    /// 16-QAM: 4 bits per symbol, 4 levels per axis, `E = 10`.
    Qam16,
    /// 64-QAM: 6 bits per symbol, 8 levels per axis, `E = 42`.
    Qam64,
}

impl Qam {
    /// The number of points `M`.
    #[inline]
    pub const fn points(self) -> usize {
        match self {
            Self::Qam16 => 16,
            Self::Qam64 => 64,
        }
    }

    /// The bits per symbol `k = log2 M`.
    #[inline]
    pub const fn bits_per_symbol(self) -> usize {
        match self {
            Self::Qam16 => 4,
            Self::Qam64 => 6,
        }
    }

    /// The bits per axis `k / 2`.
    #[inline]
    pub const fn bits_per_axis(self) -> usize {
        self.bits_per_symbol() / 2
    }

    /// The levels per axis `m = √M`.
    #[inline]
    pub const fn levels_per_axis(self) -> usize {
        match self {
            Self::Qam16 => 4,
            Self::Qam64 => 8,
        }
    }

    /// The average power `E = 2 (M - 1) / 3` of the integer grid (10 and 42).
    #[inline]
    pub const fn energy(self) -> f64 {
        match self {
            Self::Qam16 => 10.0,
            Self::Qam64 => 42.0,
        }
    }

    /// `√E` rounded once to `f32` ([`QAM16_SQRT_E`], [`QAM64_SQRT_E`]): `y √E` puts a unit-power
    /// sample on the integer grid of the levels `2 i - (m - 1)`.
    #[inline]
    pub const fn sqrt_energy(self) -> f32 {
        match self {
            Self::Qam16 => QAM16_SQRT_E,
            Self::Qam64 => QAM64_SQRT_E,
        }
    }

    /// The level values per axis in level-index order ([`QAM16_LEVELS`], [`QAM64_LEVELS`]).
    #[inline]
    pub const fn levels(self) -> &'static [f32] {
        match self {
            Self::Qam16 => &QAM16_LEVELS,
            Self::Qam64 => &QAM64_LEVELS,
        }
    }

    /// The points in symbol order ([`QAM16_POINTS`], [`QAM64_POINTS`]).
    #[inline]
    pub const fn point_table(self) -> &'static [C32] {
        match self {
            Self::Qam16 => &QAM16_POINTS,
            Self::Qam64 => &QAM64_POINTS,
        }
    }

    /// The point of `symbol` (its low `k` bits, most significant first).
    ///
    /// ```
    /// use sdrk_core::kernels::constellation::Qam;
    /// let p = Qam::Qam16.point(0);
    /// assert!(p.re < -0.9 && p.im < -0.9); // bits 00 00: level index 0 on both axes
    /// ```
    #[inline]
    pub fn point(self, symbol: u32) -> C32 {
        let t = self.point_table();
        let s = symbol as usize & (t.len() - 1);
        t.get(s).copied().unwrap_or_default()
    }

    /// The CMA radius `R2 = E|s|⁴ / E|s|²` of the unit-power constellation, computed exactly
    /// from the integer grid (16-QAM `33 / 25 = 1.32`, 64-QAM `29 / 21 ≈ 1.381`).
    #[inline]
    pub const fn cma_r2(self) -> f64 {
        match self {
            Self::Qam16 => r2_of(4),
            Self::Qam64 => r2_of(8),
        }
    }

    /// The level index of one axis value `v` of a unit-power sample: the nearest integer to
    /// `(v √E + (m - 1)) / 2` (computed as `floor(v (√E / 2) + m / 2)`, `√E / 2` rounded once to
    /// `f32`, so a sample exactly between two levels takes the upper one), clamped to
    /// `0 .. m - 1`. Infinities clamp to the outer levels; NaN gives level index 0.
    ///
    /// ```
    /// use sdrk_core::kernels::constellation::Qam;
    /// let q = Qam::Qam16;
    /// assert_eq!([-5.0, -0.7, -0.1, 0.0, 0.1, 0.7, 5.0].map(|v| q.decide_level(v)), [0, 0, 1, 2, 2, 3, 3]);
    /// assert_eq!(q.decide_level(f32::NAN), 0);
    /// ```
    #[inline]
    pub fn decide_level(self, v: f32) -> u32 {
        let (h, half_m, top) = match self {
            Self::Qam16 => (QAM16_HALF_SQRT_E, 2.0, 3.0),
            Self::Qam64 => (QAM64_HALF_SQRT_E, 4.0, 7.0),
        };
        // `f32::max` returns the other operand for NaN, so NaN goes to 0; a large value clamps
        // before the conversion.
        let t = (v * h + half_m).max(0.0).min(top);
        // `t` lies in `0 ..= m - 1`, so the truncation is the floor.
        t as u32
    }

    /// The symbol (low `k` bits, most significant first) of the point nearest to `y` with the
    /// grid decision of [`Qam::decide_level`] on both axes.
    #[inline]
    pub fn decide_symbol(self, y: C32) -> u32 {
        let b = self.bits_per_axis() as u32;
        (gray(self.decide_level(y.re)) << b) | gray(self.decide_level(y.im))
    }

    /// The point nearest to `y`, the grid decision of [`Qam::decide_level`] on both axes.
    ///
    /// ```
    /// use sdrk_core::{kernels::constellation::Qam, C32};
    /// let d = Qam::Qam64.decide(C32::new(0.2, -2.0));
    /// assert_eq!(d, C32::new(Qam::Qam64.levels()[4], Qam::Qam64.levels()[0]));
    /// ```
    #[inline]
    pub fn decide(self, y: C32) -> C32 {
        let l = self.levels();
        let at = |i: u32| l.get(i as usize).copied().unwrap_or(0.0);
        C32::new(at(self.decide_level(y.re)), at(self.decide_level(y.im)))
    }
}

/// The symbol of a square constellation with `bits_per_axis` bits per axis (1 for QPSK, 2 for
/// 16-QAM, 3 for 64-QAM) after `quarter_turns` rotations by +90° (taken modulo 4): one rotation,
/// `j y`, maps the bit groups `(I, Q)` to `(Q with its most significant bit inverted, I)` (the
/// in-phase level index `i` moves to the quadrature axis, the quadrature level index `q` becomes
/// `m - 1 - q` on the in-phase axis, and the Gray code of `m - 1 - q` is that of `q` with its
/// first bit inverted). Bits above `2 bits_per_axis` are dropped. `bits_per_axis` 0 or above
/// 15 gives 0.
///
/// ```
/// use sdrk_core::kernels::constellation::rotate_symbol;
/// // QPSK (b0 = re > 0, b1 = im > 0): 1 + j (11) turns into -1 + j (01), then -1 - j (00).
/// assert_eq!(rotate_symbol(0b11, 1, 1), 0b01);
/// assert_eq!(rotate_symbol(0b11, 1, 2), 0b00);
/// assert_eq!(rotate_symbol(0b1110, 2, 4), 0b1110);
/// ```
#[inline]
pub const fn rotate_symbol(symbol: u32, bits_per_axis: u32, quarter_turns: u32) -> u32 {
    if bits_per_axis == 0 || bits_per_axis > 15 {
        return 0;
    }
    let mask = (1u32 << bits_per_axis) - 1;
    let msb = 1u32 << (bits_per_axis - 1);
    let mut i = (symbol >> bits_per_axis) & mask;
    let mut q = symbol & mask;
    let mut r = quarter_turns % 4;
    while r > 0 {
        let ni = q ^ msb;
        q = i;
        i = ni;
        r -= 1;
    }
    (i << bits_per_axis) | q
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    const BOTH: [Qam; 2] = [Qam::Qam16, Qam::Qam64];

    /// `j^r y`.
    fn turn(y: C32, r: u32) -> C32 {
        (0..r % 4).fold(y, |y, _| C32::new(-y.im, y.re))
    }

    #[test]
    fn levels_and_factors_are_the_f32_of_the_exact_values() {
        for q in BOTH {
            let e = q.energy();
            let m = q.levels_per_axis();
            assert_eq!(e, 2.0 * (q.points() as f64 - 1.0) / 3.0);
            assert_eq!(q.sqrt_energy().to_bits(), (libm::sqrt(e) as f32).to_bits());
            for (i, &l) in q.levels().iter().enumerate() {
                let exact = (2.0 * i as f64 - (m as f64 - 1.0)) / libm::sqrt(e);
                assert_eq!(l.to_bits(), (exact as f32).to_bits(), "{q:?} level {i}");
            }
        }
        assert_eq!(
            QAM16_HALF_SQRT_E.to_bits(),
            ((libm::sqrt(10.0) / 2.0) as f32).to_bits()
        );
        assert_eq!(
            QAM64_HALF_SQRT_E.to_bits(),
            ((libm::sqrt(42.0) / 2.0) as f32).to_bits()
        );
    }

    #[test]
    fn every_point_follows_the_mapping() {
        for q in BOTH {
            let b = q.bits_per_axis() as u32;
            let m = q.levels_per_axis() as u32;
            let sqrt_e = libm::sqrt(q.energy());
            assert_eq!(q.point_table().len(), q.points());
            for s in 0..q.points() as u32 {
                // The level index whose Gray code is each group, found by search.
                let find = |g: u32| (0..m).find(|&i| (i ^ (i >> 1)) == g).unwrap();
                let i = find(s >> b);
                let qi = find(s & ((1 << b) - 1));
                let p = q.point(s);
                let want = |i: u32| ((2.0 * f64::from(i) - f64::from(m - 1)) / sqrt_e) as f32;
                assert_eq!(p.re.to_bits(), want(i).to_bits(), "{q:?} {s}");
                assert_eq!(p.im.to_bits(), want(qi).to_bits(), "{q:?} {s}");
                assert_eq!(q.decide_symbol(p), s, "{q:?} {s}");
                assert_eq!(q.decide(p), p, "{q:?} {s}");
            }
            // Neighbouring levels differ in exactly one bit (Gray).
            for i in 0..m - 1 {
                assert_eq!((gray(i) ^ gray(i + 1)).count_ones(), 1);
            }
        }
    }

    #[test]
    fn the_average_power_is_one_and_r2_matches() {
        for (q, r2) in [(Qam::Qam16, 1.32), (Qam::Qam64, 1.381)] {
            let n = q.points() as f64;
            let (p2, p4) = q.point_table().iter().fold((0.0, 0.0), |(a, b), p| {
                let m = f64::from(p.re) * f64::from(p.re) + f64::from(p.im) * f64::from(p.im);
                (a + m, b + m * m)
            });
            assert!((p2 / n - 1.0).abs() < 1e-6, "{q:?} power {}", p2 / n);
            assert!((p4 / p2 - q.cma_r2()).abs() < 1e-5, "{q:?}");
            // 3 digits, as in the specification.
            assert!((q.cma_r2() - r2).abs() < 5e-4, "{q:?} {}", q.cma_r2());
        }
        assert_eq!(Qam::Qam16.cma_r2(), 33.0 / 25.0);
        assert_eq!(Qam::Qam64.cma_r2(), 29.0 / 21.0);
    }

    #[test]
    fn the_decision_switches_at_the_boundaries() {
        for q in BOTH {
            let m = q.levels_per_axis() as u32;
            let sqrt_e = libm::sqrt(q.energy());
            // Boundary between level index i - 1 and i: y √E = 2 i - m.
            for i in 1..m {
                let b = (2.0 * f64::from(i) - f64::from(m)) / sqrt_e;
                let d = 1e-5;
                assert_eq!(q.decide_level((b - d) as f32), i - 1, "{q:?} below {i}");
                assert_eq!(q.decide_level((b + d) as f32), i, "{q:?} above {i}");
            }
            // The middle boundary (0) and its signed zeros go up.
            assert_eq!(q.decide_level(0.0), m / 2);
            assert_eq!(q.decide_level(-0.0), m / 2);
            // Clamping and the non-finite inputs.
            for (v, want) in [
                (-1e30, 0),
                (1e30, m - 1),
                (f32::NEG_INFINITY, 0),
                (f32::INFINITY, m - 1),
                (f32::NAN, 0),
            ] {
                assert_eq!(q.decide_level(v), want, "{q:?} {v}");
            }
        }
    }

    #[test]
    fn the_rotation_map_matches_the_points() {
        for q in BOTH {
            let b = q.bits_per_axis() as u32;
            for s in 0..q.points() as u32 {
                for r in 0..4 {
                    let t = rotate_symbol(s, b, r);
                    assert_eq!(q.point(t), turn(q.point(s), r), "{q:?} {s} r{r}");
                    // `r` rotations and `4 - r` give the identity.
                    assert_eq!(rotate_symbol(t, b, 4 - r), s);
                }
            }
        }
        // QPSK, one bit per axis: the points (±1 ± j) / √2 with bit 1 on the positive side.
        let a = core::f32::consts::FRAC_1_SQRT_2;
        let qpsk = |s: u32| {
            let side = |bit: u32| if bit == 1 { a } else { -a };
            C32::new(side(s >> 1), side(s & 1))
        };
        for s in 0..4 {
            for r in 0..4 {
                assert_eq!(
                    qpsk(rotate_symbol(s, 1, r)),
                    turn(qpsk(s), r),
                    "qpsk {s} r{r}"
                );
            }
        }
        assert_eq!(rotate_symbol(0b11, 0, 1), 0);
        assert_eq!(rotate_symbol(0b11, 16, 1), 0);
    }
}
