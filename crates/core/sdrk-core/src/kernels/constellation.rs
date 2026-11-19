//! Square QAM constellations (M44 §1, M44.2 §1): the mapping of `qpsk` (4-QAM), `qam16`,
//! `qam64`, `qam256` and `qam1024` shared by the modulator, the slicer, the equalizer and the
//! carrier loop, and the kernel [`QamSoftBits`] of the slicer (M44 §2, M44.2 §2).
//!
//! A symbol of `M = 4^b` points (`b` = 1 to 5 bits per axis) carries `k = 2 b` bits, most
//! significant first. The first `b` bits select the in-phase level, the last `b` the quadrature
//! level, each by the reflected Gray code of the level index: level index `i` (`0 .. m - 1`,
//! `m = 2^b`) has the code `i ^ (i >> 1)` and the level value `2 i - (m - 1)`. The points are
//! scaled by `1 / √E`, `E = 2 (M - 1) / 3` (2, 10, 42, 170 and 682), so the average power is 1.
//! The 4-QAM grid is the QPSK of `psk_mod` point for point, `((2 b0 - 1) / √2, (2 b1 - 1) / √2)`.
//! The decision is a grid slicer per axis, as GNU Radio's `constellation_rect`: the level index
//! nearest to `(y √E + (m - 1)) / 2`, clamped to `0 .. m - 1`. A rotation by 90° maps the bit
//! groups `(I, Q)` of a symbol to `(Q with its first bit inverted, I)` ([`rotate_symbol`]).
//!
//! Every order is one instance of the same code over the bits per axis `B` (a `const` generic
//! parameter): the level values, the decision and the soft bits are written once (M44.2 §2, §3).
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
//! assert_eq!(Qam::Qam1024.bits_per_symbol(), 10);
//! ```

use crate::error::Result;
use crate::types::{C32, ComplexSample};

/// `√E` in `f64` per bits per axis (index `b`, entry 0 unused): `√2`, `√10`, `√42`, `√170`
/// and `√682`, each the correctly rounded `f64` (the bits of `libm::sqrt`, checked by a test).
#[allow(
    clippy::excessive_precision,
    reason = "the literals are the correctly rounded f64 values of sqrt(E)"
)]
const SQRT_E_F64: [f64; 6] = [
    1.0,
    core::f64::consts::SQRT_2,
    3.162_277_660_168_379_5,
    6.480_740_698_407_86,
    13.038_404_810_405_298,
    26.115_129_714_401_192,
];

/// The level values of a square grid with `M` levels per axis in level-index order,
/// `(2 i - (M - 1)) / √E`, each the `f32` of the `f64` quotient.
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
const fn levels_of<const M: usize>(sqrt_e: f64) -> [f32; M] {
    let mut out = [0.0f32; M];
    let mut i = 0;
    while i < M {
        out[i] = (((2 * i) as f64 - (M - 1) as f64) / sqrt_e) as f32;
        i += 1;
    }
    out
}

/// The level values of 4-QAM (QPSK) in level-index order, `∓1 / √2`.
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
pub const QPSK_LEVELS: [f32; 2] = levels_of::<2>(SQRT_E_F64[1]);
/// The level values of 16-QAM in level-index order, `(2 i - 3) / √10`.
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
pub const QAM16_LEVELS: [f32; 4] = levels_of::<4>(SQRT_E_F64[2]);
/// The level values of 64-QAM in level-index order, `(2 i - 7) / √42`.
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
pub const QAM64_LEVELS: [f32; 8] = levels_of::<8>(SQRT_E_F64[3]);
/// The level values of 256-QAM in level-index order, `(2 i - 15) / √170`.
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
pub const QAM256_LEVELS: [f32; 16] = levels_of::<16>(SQRT_E_F64[4]);
/// The level values of 1024-QAM in level-index order, `(2 i - 31) / √682`.
#[allow(
    clippy::indexing_slicing,
    reason = "const evaluation only: an index out of range fails the build, never at run time"
)]
pub const QAM1024_LEVELS: [f32; 32] = levels_of::<32>(SQRT_E_F64[5]);

/// `√2` rounded once to `f32`: the factor from the 4-QAM points to the integer grid.
pub const QPSK_SQRT_E: f32 = SQRT_E_F64[1] as f32;
/// `√10` rounded once to `f32`: the factor from the 16-QAM points to the integer grid.
pub const QAM16_SQRT_E: f32 = SQRT_E_F64[2] as f32;
/// `√42` rounded once to `f32`: the factor from the 64-QAM points to the integer grid.
pub const QAM64_SQRT_E: f32 = SQRT_E_F64[3] as f32;
/// `√170` rounded once to `f32`: the factor from the 256-QAM points to the integer grid.
pub const QAM256_SQRT_E: f32 = SQRT_E_F64[4] as f32;
/// `√682` rounded once to `f32`: the factor from the 1024-QAM points to the integer grid.
pub const QAM1024_SQRT_E: f32 = SQRT_E_F64[5] as f32;

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

/// Builds the point table of a square QAM with `b` bits per axis from its level values: entry
/// `s` is the point of symbol `s`.
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

/// The 4-QAM (QPSK) points, entry `s` the point of the symbol `s` (bits `b0 b1`, `b0` the most
/// significant: `b0` the in-phase code, `b1` the quadrature code).
pub const QPSK_POINTS: [C32; 4] = points_of::<4>(&QPSK_LEVELS, 1);
/// The 16-QAM points, entry `s` the point of the symbol `s` (bits `b0 b1 b2 b3`, `b0` the most
/// significant: `b0 b1` the in-phase code, `b2 b3` the quadrature code).
pub const QAM16_POINTS: [C32; 16] = points_of::<16>(&QAM16_LEVELS, 2);
/// The 64-QAM points, entry `s` the point of the symbol `s` (bits `b0 .. b5`, `b0` the most
/// significant: `b0 b1 b2` the in-phase code, `b3 b4 b5` the quadrature code).
pub const QAM64_POINTS: [C32; 64] = points_of::<64>(&QAM64_LEVELS, 3);
/// The 256-QAM points (2 KiB), entry `s` the point of the symbol `s` (bits `b0 .. b7`, `b0` the
/// most significant: `b0 .. b3` the in-phase code, `b4 .. b7` the quadrature code).
pub static QAM256_POINTS: [C32; 256] = points_of::<256>(&QAM256_LEVELS, 4);
/// The 1024-QAM points (8 KiB), entry `s` the point of the symbol `s` (bits `b0 .. b9`, `b0`
/// the most significant: `b0 .. b4` the in-phase code, `b5 .. b9` the quadrature code).
pub static QAM1024_POINTS: [C32; 1024] = points_of::<1024>(&QAM1024_LEVELS, 5);

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

/// The square grid with `B` bits per axis (1 to 5): its constants and the forms of its grid
/// decision, one generic implementation for every order (M44.2 §1, §3).
pub(crate) struct Grid<const B: usize>;

impl<const B: usize> Grid<B> {
    /// The levels per axis `m = 2^B`.
    const M: usize = 1 << B;
    /// The level values in level-index order.
    const LEVELS: &'static [f32] = match B {
        1 => &QPSK_LEVELS,
        2 => &QAM16_LEVELS,
        3 => &QAM64_LEVELS,
        4 => &QAM256_LEVELS,
        _ => &QAM1024_LEVELS,
    };
    /// `√E / 2` rounded once to `f32`: the factor of the grid decision.
    #[allow(
        clippy::indexing_slicing,
        reason = "const evaluation only: an index out of range fails the build, never at run time"
    )]
    const HALF_SQRT_E: f32 = (SQRT_E_F64[B] / 2.0) as f32;
    /// `m / 2`, the offset of the grid decision.
    const HALF_M: f32 = (Self::M / 2) as f32;
    /// The top level index `m - 1` as `f32`.
    const TOP: f32 = (Self::M - 1) as f32;

    /// `t = v (√E / 2) + m / 2`, the argument of every form of the decision.
    #[inline]
    fn arg(v: f32) -> f32 {
        v * Self::HALF_SQRT_E + Self::HALF_M
    }

    /// The level index of [`Qam::decide_level`].
    #[inline]
    fn level(v: f32) -> u32 {
        // `f32::max` returns the other operand for NaN, so NaN goes to 0; a large value clamps
        // before the conversion.
        let t = Self::arg(v).max(0.0).min(Self::TOP);
        // `t` lies in `0 ..= m - 1`, so the truncation is the floor.
        t as u32
    }

    /// The level value of [`Qam::decide_level`] for one axis value `v`, from the level table at
    /// the index `clamp(t as i32, 0, m - 1)`: the conversion truncates toward zero, which is
    /// the floor for `t >= 0`, and every `t < 0` (and NaN, which converts to 0) clamps to 0 as
    /// `floor(max(t, 0))` does, every `t >= m` to `m - 1`. The clamp in `f32` (a maximum and a
    /// minimum, 8 cycles on the chain of the loops at SSE2 on `skylake`) becomes two integer
    /// selects after the conversion (D1602).
    #[inline]
    fn by_index(v: f32) -> f32 {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "the saturating conversion is the floor of the clamped value; m - 1 < 32"
        )]
        let i = (Self::arg(v) as i32).clamp(0, (Self::M - 1) as i32);
        #[allow(clippy::cast_sign_loss, reason = "clamped to 0 ..= m - 1")]
        Self::LEVELS.get(i as usize).copied().unwrap_or(0.0)
    }

    /// The level value of [`Qam::decide_level`] for one axis value `v`, from the level table.
    #[inline]
    fn by_table(v: f32) -> f32 {
        Self::LEVELS
            .get(Self::level(v) as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// The level value of [`Qam::decide_level`] for one axis value `v`, the same bits as
    /// `levels()[decide_level(v)]`: from the same `t = v (√E / 2) + m / 2` a tree of `B` levels
    /// of comparisons `t >= k` selects the level (`floor` of the clamped `t` is the largest `k`
    /// with `t >= k`; NaN fails every comparison and takes level index 0), so the decision has no
    /// conversion to an integer and no table load on the chain of the loops (D1417). Level `s`
    /// of the tree (from the leaves, `s` = 0 to `B - 1`) picks between the neighbouring groups
    /// of `2^s` levels `2 j` and `2 j + 1` by `t >= (2 j + 1) 2^s`. On AArch64 and AVX2 every
    /// select is one instruction (`fcsel`, `vblendvps`). On x86 every select is
    /// [`core::hint::select_unpredictable`]: as plain selects LLVM compiled some of them at AVX2
    /// and AVX-512 as jumps on the decided symbol, which a core mispredicts on about half of
    /// the random symbols, and the branch-free tree costs less in cycles plus mispredictions
    /// (D1611, D1614). AArch64 compiles the plain selects without a jump.
    #[inline]
    fn by_select(v: f32) -> f32 {
        let t = Self::arg(v);
        // The groups of the current level of the tree, `n` of them, each of `span` levels.
        let mut w = [0.0f32; 32];
        for (w, &l) in w.iter_mut().zip(Self::LEVELS) {
            *w = l;
        }
        let (mut n, mut span) = (Self::M, 1usize);
        while n > 1 {
            for j in 0..n / 2 {
                let lo = w.get(2 * j).copied().unwrap_or(0.0);
                let hi = w.get(2 * j + 1).copied().unwrap_or(0.0);
                if let Some(o) = w.get_mut(j) {
                    *o = pick(t >= ((2 * j + 1) * span) as f32, hi, lo);
                }
            }
            n /= 2;
            span *= 2;
        }
        w.first().copied().unwrap_or(0.0)
    }

    /// The level value of one axis, the same bits in every form, chosen per order and level by
    /// cycles plus mispredictions (D1611): for 256-QAM and 1024-QAM [`Grid::by_index`] on every
    /// target (its tree of 15 or 31 selects per axis and the level table, whose bound check
    /// stays a jump, simulated slower, D1505); for 16-QAM and 64-QAM [`Qam::decide_m44`] (the
    /// forms of M44, D1525); for 4-QAM [`Grid::by_select`] on AArch64 and
    /// AVX2 (D1417), [`Grid::by_index`] on x86 without AVX2 (D1602), [`Grid::by_table`]
    /// elsewhere (Cortex-M4 and `wasm32` are counted in instructions).
    #[inline]
    pub(crate) fn value(v: f32) -> f32 {
        if B == 2 {
            Qam::Qam16.decide_m44(v)
        } else if B == 3 {
            Qam::Qam64.decide_m44(v)
        } else if B >= 4 {
            Self::by_index(v)
        } else if cfg!(any(target_arch = "aarch64", target_feature = "avx2")) {
            Self::by_select(v)
        } else if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            Self::by_index(v)
        } else {
            Self::by_table(v)
        }
    }
}

/// One select of the decision trees ([`Grid::by_select`], [`select16`], [`select64`]):
/// [`core::hint::select_unpredictable`] on x86 (as plain selects LLVM compiled some of them at
/// AVX2 and AVX-512 as jumps on the decided symbol, D1611, D1614), a plain select elsewhere
/// (AArch64 compiles it without a jump).
#[inline]
fn pick(c: bool, a: f32, b: f32) -> f32 {
    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        core::hint::select_unpredictable(c, a, b)
    } else if c {
        a
    } else {
        b
    }
}

/// The select tree of 16-QAM ([`Grid::by_select`] for `B` = 2 written out as in M44, the same
/// selects, D1417), see [`Qam::decide_m44`].
#[inline]
pub(crate) fn select16(v: f32) -> f32 {
    let [l0, l1, l2, l3] = QAM16_LEVELS;
    let t = v * Grid::<2>::HALF_SQRT_E + 2.0;
    let lo = pick(t >= 1.0, l1, l0);
    let hi = pick(t >= 3.0, l3, l2);
    pick(t >= 2.0, hi, lo)
}

/// The select tree of 64-QAM ([`Grid::by_select`] for `B` = 3 written out as in M44, the same
/// selects, D1417), see [`Qam::decide_m44`].
#[inline]
pub(crate) fn select64(v: f32) -> f32 {
    let [l0, l1, l2, l3, l4, l5, l6, l7] = QAM64_LEVELS;
    let t = v * Grid::<3>::HALF_SQRT_E + 4.0;
    let q0 = pick(t >= 1.0, l1, l0);
    let q1 = pick(t >= 3.0, l3, l2);
    let q2 = pick(t >= 5.0, l5, l4);
    let q3 = pick(t >= 7.0, l7, l6);
    let lo = pick(t >= 2.0, q1, q0);
    let hi = pick(t >= 6.0, q3, q2);
    pick(t >= 4.0, hi, lo)
}

/// `$body` with the grid `$g` of the bits per axis of `$q` (a [`Qam`]): one instance per order.
macro_rules! with_grid {
    ($q:expr, $g:ident => $body:expr) => {
        match $q {
            Qam::Qpsk => {
                type $g = Grid<1>;
                $body
            }
            Qam::Qam16 => {
                type $g = Grid<2>;
                $body
            }
            Qam::Qam64 => {
                type $g = Grid<3>;
                $body
            }
            Qam::Qam256 => {
                type $g = Grid<4>;
                $body
            }
            Qam::Qam1024 => {
                type $g = Grid<5>;
                $body
            }
        }
    };
}

/// A square QAM constellation of M44 and M44.2, `M = 4^b` points with `b` = 1 to 5 bits per
/// axis. 4-QAM is [`Qam::Qpsk`] (the constellation `qpsk` of the blocks; there is no `qam4`).
///
/// ```
/// use sdrk_core::kernels::constellation::Qam;
/// assert_eq!(Qam::Qam64.bits_per_symbol(), 6);
/// assert!((Qam::Qam64.cma_r2() - 1.381).abs() < 5e-4);
/// assert_eq!(Qam::from_bits_per_axis(4), Some(Qam::Qam256));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Qam {
    /// 4-QAM, the QPSK grid on the diagonals: 2 bits per symbol, 2 levels per axis, `E = 2`.
    Qpsk,
    /// 16-QAM: 4 bits per symbol, 4 levels per axis, `E = 10`.
    Qam16,
    /// 64-QAM: 6 bits per symbol, 8 levels per axis, `E = 42`.
    Qam64,
    /// 256-QAM: 8 bits per symbol, 16 levels per axis, `E = 170`.
    Qam256,
    /// 1024-QAM: 10 bits per symbol, 32 levels per axis, `E = 682`.
    Qam1024,
}

impl Qam {
    /// Every order, smallest first.
    pub const ALL: [Qam; 5] = [
        Self::Qpsk,
        Self::Qam16,
        Self::Qam64,
        Self::Qam256,
        Self::Qam1024,
    ];

    /// The square QAM with `b` bits per axis (1 to 5), `None` for any other `b`.
    #[inline]
    pub const fn from_bits_per_axis(b: usize) -> Option<Self> {
        match b {
            1 => Some(Self::Qpsk),
            2 => Some(Self::Qam16),
            3 => Some(Self::Qam64),
            4 => Some(Self::Qam256),
            5 => Some(Self::Qam1024),
            _ => None,
        }
    }

    /// The bits per axis `b` (1 to 5).
    #[inline]
    pub const fn bits_per_axis(self) -> usize {
        match self {
            Self::Qpsk => 1,
            Self::Qam16 => 2,
            Self::Qam64 => 3,
            Self::Qam256 => 4,
            Self::Qam1024 => 5,
        }
    }

    /// The number of points `M = 4^b`.
    #[inline]
    pub const fn points(self) -> usize {
        1 << self.bits_per_symbol()
    }

    /// The bits per symbol `k = log2 M = 2 b`.
    #[inline]
    pub const fn bits_per_symbol(self) -> usize {
        2 * self.bits_per_axis()
    }

    /// The levels per axis `m = √M = 2^b`.
    #[inline]
    pub const fn levels_per_axis(self) -> usize {
        1 << self.bits_per_axis()
    }

    /// The average power `E = 2 (M - 1) / 3` of the integer grid (2, 10, 42, 170 and 682).
    #[inline]
    pub const fn energy(self) -> f64 {
        (2 * (self.points() - 1) / 3) as f64
    }

    /// `√E` rounded once to `f32` ([`QPSK_SQRT_E`] to [`QAM1024_SQRT_E`]): `y √E` puts a
    /// unit-power sample on the integer grid of the levels `2 i - (m - 1)`.
    #[inline]
    pub const fn sqrt_energy(self) -> f32 {
        match self {
            Self::Qpsk => QPSK_SQRT_E,
            Self::Qam16 => QAM16_SQRT_E,
            Self::Qam64 => QAM64_SQRT_E,
            Self::Qam256 => QAM256_SQRT_E,
            Self::Qam1024 => QAM1024_SQRT_E,
        }
    }

    /// The level values per axis in level-index order ([`QPSK_LEVELS`] to [`QAM1024_LEVELS`]).
    #[inline]
    pub const fn levels(self) -> &'static [f32] {
        match self {
            Self::Qpsk => &QPSK_LEVELS,
            Self::Qam16 => &QAM16_LEVELS,
            Self::Qam64 => &QAM64_LEVELS,
            Self::Qam256 => &QAM256_LEVELS,
            Self::Qam1024 => &QAM1024_LEVELS,
        }
    }

    /// The points in symbol order ([`QPSK_POINTS`] to [`QAM1024_POINTS`]).
    #[inline]
    pub const fn point_table(self) -> &'static [C32] {
        match self {
            Self::Qpsk => &QPSK_POINTS,
            Self::Qam16 => &QAM16_POINTS,
            Self::Qam64 => &QAM64_POINTS,
            Self::Qam256 => &QAM256_POINTS,
            Self::Qam1024 => &QAM1024_POINTS,
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
    /// from the integer grid (4-QAM 1, 16-QAM `33 / 25 = 1.32`, 64-QAM `29 / 21 ≈ 1.381`,
    /// 256-QAM `593 / 425 ≈ 1.3953`, 1024-QAM `477 / 341 ≈ 1.3988`).
    #[inline]
    pub const fn cma_r2(self) -> f64 {
        r2_of(self.levels_per_axis() as i64)
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
        with_grid!(self, G => G::level(v))
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
        // 16-QAM and 64-QAM in the form of M44 (see [`Qam::decide_m44`]).
        match self {
            Self::Qam16 | Self::Qam64 => C32::new(self.decide_m44(y.re), self.decide_m44(y.im)),
            q => with_grid!(q, G => C32::new(G::value(y.re), G::value(y.im))),
        }
    }

    /// The level value of one axis of 16-QAM or 64-QAM ([`Grid::value`] for `B` = 2 and 3, the
    /// same bits), written as in M44: [`Qam::select_m44`] on AArch64 and AVX2 (D1417),
    /// [`Grid::by_index`] on x86 without AVX2 (D1602), [`Grid::by_table`] elsewhere. A method of
    /// [`Qam`] with both trees behind one `match`, as in M44, not the generic [`Grid`]: through
    /// the generic functions the same selects made LLVM compile the loops of `equalize` and
    /// `costas` around them differently (up to 46 % slower), and as two separate functions still
    /// 3 to 8 % slower on `znver3`, `znver4` and `cortex-a72` (the 64-QAM loop of `costas` and the
    /// 16-QAM loop of `equalize` vectorized the two axes on AArch64); in this form the loops
    /// compile to the code of M44 (the measure step of M44.2, D1525). Only for 16-QAM and 64-QAM:
    /// the callers ([`Qam::decide`], [`Grid::value`] for `B` = 2 and 3 and the decision of
    /// `equalize`) pass no other order. For another order the result is not specified (the
    /// select tree of AArch64 and AVX2 gives 64-QAM levels, the other targets [`Grid::value`]);
    /// the arms stay as they are because a change of this function moved the code of the loops
    /// around it (D1525; the review of M44.2, D1538).
    #[inline]
    pub(crate) fn decide_m44(self, v: f32) -> f32 {
        if cfg!(any(target_arch = "aarch64", target_feature = "avx2")) {
            self.select_m44(v)
        } else if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            match self {
                Self::Qam16 => Grid::<2>::by_index(v),
                Self::Qam64 => Grid::<3>::by_index(v),
                q => with_grid!(q, G => G::value(v)),
            }
        } else {
            match self {
                Self::Qam16 => Grid::<2>::by_table(v),
                Self::Qam64 => Grid::<3>::by_table(v),
                q => with_grid!(q, G => G::value(v)),
            }
        }
    }

    /// [`Grid::by_select`] of 16-QAM (64-QAM for every other order) written out as in M44 (the
    /// same selects, D1417).
    #[inline]
    pub(crate) fn select_m44(self, v: f32) -> f32 {
        // The trees of [`select16`] and [`select64`] written out here once more: calling them
        // made LLVM pair the two axes of the 64-QAM loop of `costas` into vectors on AArch64
        // (+7.6 % on `cortex-a72`, D1525).
        let pick = |c: bool, a: f32, b: f32| {
            if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
                core::hint::select_unpredictable(c, a, b)
            } else if c {
                a
            } else {
                b
            }
        };
        if self == Self::Qam16 {
            let [l0, l1, l2, l3] = QAM16_LEVELS;
            let t = v * Grid::<2>::HALF_SQRT_E + 2.0;
            let lo = pick(t >= 1.0, l1, l0);
            let hi = pick(t >= 3.0, l3, l2);
            pick(t >= 2.0, hi, lo)
        } else {
            let [l0, l1, l2, l3, l4, l5, l6, l7] = QAM64_LEVELS;
            let t = v * Grid::<3>::HALF_SQRT_E + 4.0;
            let q0 = pick(t >= 1.0, l1, l0);
            let q1 = pick(t >= 3.0, l3, l2);
            let q2 = pick(t >= 5.0, l5, l4);
            let q3 = pick(t >= 7.0, l7, l6);
            let lo = pick(t >= 2.0, q1, q0);
            let hi = pick(t >= 6.0, q3, q2);
            pick(t >= 4.0, hi, lo)
        }
    }

    /// The point nearest to `y` by the level table on both axes: the bits of [`Qam::decide`],
    /// the form of the 64-QAM equalizer on AArch64 (D1602).
    #[inline]
    pub(crate) fn decide_table(self, y: C32) -> C32 {
        with_grid!(self, G => C32::new(G::by_table(y.re), G::by_table(y.im)))
    }
}

/// The piecewise-linear max-log soft bits of a square QAM (M44 §2, M44.2 §2): the kernel of
/// `symbol_demap` `qam16` to `qam1024` and of `qam_demod`.
///
/// Per axis, in-phase first (the components in memory order, `re` then `im` of every symbol),
/// with `b` bits per axis and `v = component * √E` (one `f32` product, `√E` =
/// [`Qam::sqrt_energy`], rounded once), the `b` soft bits of the axis are the recursion
///
/// ```text
/// t_0 = v
/// t_i = 2^(b - i) - |t_(i - 1)|     for i = 1 .. b - 1
/// s_i = scale * t_i
/// ```
///
/// every step one rounded `f32` operation, without FMA:
///
/// - 4-QAM (QPSK), one bit per axis: `scale * v`;
/// - 16-QAM, two bits per axis: `scale * v`, `scale * (2 - |v|)`;
/// - 64-QAM, three bits per axis: `scale * v`, `scale * (4 - |v|)`, `scale * (2 - |4 - |v||)`
///   (`|4 - a|` is `|a - 4|` exactly, the form of M44);
/// - 256-QAM and 1024-QAM the next steps with 8, 4, 2 and 16, 8, 4, 2.
///
/// A soft bit above 0 means bit 1 (the Gray mapping of [`Qam`]: the first bit of an axis is 1
/// on the positive side, bit `i` on the levels whose distance to the knee `±2^(b - i)` of the
/// step before is below `2^(b - i - 1)`). NaN and infinite components give NaN and infinite
/// soft bits. The output holds `bits_per_symbol` soft bits per input symbol, so its length must
/// be `input.len() * qam.bits_per_symbol()`.
///
/// Every implementation gives the bits of the fallback (the class is bit-exact against this
/// `f32` form); the `Reference` evaluates the same definition in `f64` with the exact `√E`
/// and rounds once, the achievable optimum (D1413).
///
/// ```
/// use sdrk_core::kernels::constellation::{Qam, QamSoftBits};
/// use sdrk_core::{Fallback, C32};
/// let q = Qam::Qam16;
/// let mut bits = [0.0f32; 4];
/// // The point of symbol 0b1110: in-phase level +1 (bits 11), quadrature level +3 (bits 10).
/// Fallback.qam_soft_bits(q, &[q.point(0b1110)], &mut bits, 1.0)?;
/// assert_eq!(bits.map(|b| b > 0.0), [true, true, true, false]);
/// assert!(Fallback.qam_soft_bits(q, &[q.point(0)], &mut [0.0; 3], 1.0).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait QamSoftBits<T: ComplexSample> {
    /// The soft bits of every symbol of `input` (see the trait). `output.len()` must be
    /// `input.len() * qam.bits_per_symbol()`, otherwise `LengthMismatch` with that length as
    /// `expected`.
    fn qam_soft_bits(
        &self,
        qam: Qam,
        input: &[T],
        output: &mut [T::Real],
        scale: f32,
    ) -> Result<()>;
}

/// `Ok` when `output_len` is `input_len * qam.bits_per_symbol()`, else `LengthMismatch` with
/// that product as `expected` (shared by the implementations of [`QamSoftBits`]).
///
/// # Errors
/// `LengthMismatch` (also when the product overflows).
#[inline]
pub fn check_soft_bits_len(qam: Qam, input_len: usize, output_len: usize) -> Result<()> {
    match input_len.checked_mul(qam.bits_per_symbol()) {
        Some(expected) if expected == output_len => Ok(()),
        expected => Err(crate::error::Error::LengthMismatch {
            expected: expected.unwrap_or(usize::MAX),
            got: output_len,
        }),
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

    const ALL: [Qam; 5] = Qam::ALL;

    /// The level value of `v` by every form of the decision of `q`: select tree, index, table.
    fn forms(q: Qam, v: f32) -> [f32; 3] {
        with_grid!(q, G => [G::by_select(v), G::by_index(v), G::by_table(v)])
    }

    /// `√E / 2` of the grid of `q` as the decision uses it.
    fn half_sqrt_e(q: Qam) -> f32 {
        with_grid!(q, G => G::HALF_SQRT_E)
    }

    /// `j^r y`.
    fn turn(y: C32, r: u32) -> C32 {
        (0..r % 4).fold(y, |y, _| C32::new(-y.im, y.re))
    }

    #[test]
    fn levels_and_factors_are_the_f32_of_the_exact_values() {
        for q in ALL {
            let e = q.energy();
            let m = q.levels_per_axis();
            assert_eq!(e, 2.0 * (q.points() as f64 - 1.0) / 3.0);
            assert_eq!(q.sqrt_energy().to_bits(), (libm::sqrt(e) as f32).to_bits());
            for (i, &l) in q.levels().iter().enumerate() {
                let exact = (2.0 * i as f64 - (m as f64 - 1.0)) / libm::sqrt(e);
                assert_eq!(l.to_bits(), (exact as f32).to_bits(), "{q:?} level {i}");
            }
        }
        for q in ALL {
            let e = q.energy();
            assert_eq!(
                SQRT_E_F64[q.bits_per_axis()].to_bits(),
                libm::sqrt(e).to_bits()
            );
            assert_eq!(
                half_sqrt_e(q).to_bits(),
                ((libm::sqrt(e) / 2.0) as f32).to_bits(),
                "{q:?}"
            );
            assert_eq!(Qam::from_bits_per_axis(q.bits_per_axis()), Some(q));
            assert_eq!(q.points(), 1 << (2 * q.bits_per_axis()));
        }
        assert_eq!(ALL.map(Qam::energy), [2.0, 10.0, 42.0, 170.0, 682.0]);
        assert_eq!(Qam::from_bits_per_axis(0), None);
        assert_eq!(Qam::from_bits_per_axis(6), None);
    }

    #[test]
    fn qpsk_is_the_qpsk_of_psk_mod() {
        // ((2 b0 - 1) / sqrt 2, (2 b1 - 1) / sqrt 2), bit 1 on the positive side.
        let a = core::f32::consts::FRAC_1_SQRT_2;
        let side = |bit: u32| if bit == 1 { a } else { -a };
        for s in 0..4 {
            let p = Qam::Qpsk.point(s);
            assert_eq!(p.re.to_bits(), side(s >> 1).to_bits(), "{s}");
            assert_eq!(p.im.to_bits(), side(s & 1).to_bits(), "{s}");
        }
    }

    #[test]
    fn every_point_follows_the_mapping() {
        for q in ALL {
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
        for (q, r2) in [
            (Qam::Qpsk, 1.0),
            (Qam::Qam16, 1.32),
            (Qam::Qam64, 1.381),
            (Qam::Qam256, 1.3953),
            (Qam::Qam1024, 1.3988),
        ] {
            let n = q.points() as f64;
            let (p2, p4) = q.point_table().iter().fold((0.0, 0.0), |(a, b), p| {
                let m = f64::from(p.re) * f64::from(p.re) + f64::from(p.im) * f64::from(p.im);
                (a + m, b + m * m)
            });
            assert!((p2 / n - 1.0).abs() < 1e-6, "{q:?} power {}", p2 / n);
            assert!((p4 / p2 - q.cma_r2()).abs() < 1e-5, "{q:?}");
            // The digits of the specification.
            assert!((q.cma_r2() - r2).abs() < 5e-5, "{q:?} {}", q.cma_r2());
        }
        assert_eq!(Qam::Qpsk.cma_r2(), 1.0);
        assert_eq!(Qam::Qam256.cma_r2(), 593.0 / 425.0);
        assert_eq!(Qam::Qam1024.cma_r2(), 477.0 / 341.0);
        assert_eq!(Qam::Qam16.cma_r2(), 33.0 / 25.0);
        assert_eq!(Qam::Qam64.cma_r2(), 29.0 / 21.0);
    }

    #[test]
    fn the_decision_by_comparisons_equals_the_level_table() {
        // Every exponent and both signs with low, high and random significands, and the
        // specials: the selected level has the bits of the table entry of `decide_level`.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut values = std::vec![
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0
        ];
        for exp in 0u32..=255 {
            for sign in [0u32, 1] {
                let top = (sign << 31) | (exp << 23);
                for low in 0u32..512 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    values.extend([
                        f32::from_bits(top | low),
                        f32::from_bits(top | (0x7F_FFFF - low)),
                        f32::from_bits(top | (seed as u32 & 0x7F_FFFF)),
                    ]);
                }
            }
        }
        for q in ALL {
            let m = q.levels_per_axis() as u32;
            let sqrt_e = libm::sqrt(q.energy());
            // The boundaries `y √E = 2 i - m` and their neighbours in `f32`.
            for i in 0..=2 * m {
                let b = ((f64::from(i) - f64::from(m)) / sqrt_e) as f32;
                for k in 0..64u32 {
                    values.extend([
                        f32::from_bits(b.to_bits() + k),
                        f32::from_bits(b.to_bits().wrapping_sub(k)),
                    ]);
                }
            }
            for &v in &values {
                let want = q.levels()[q.decide_level(v) as usize];
                for (form, got) in forms(q, v).into_iter().enumerate() {
                    assert_eq!(got.to_bits(), want.to_bits(), "{q:?} form {form} {v:e}");
                }
                let got = q.decide(C32::new(v, v));
                assert_eq!(got.re.to_bits(), want.to_bits(), "{q:?} {v:e}");
                assert_eq!(got.im.to_bits(), want.to_bits(), "{q:?} {v:e}");
            }
        }
    }

    #[test]
    fn the_decision_switches_at_the_boundaries() {
        for q in ALL {
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
        for q in ALL {
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
