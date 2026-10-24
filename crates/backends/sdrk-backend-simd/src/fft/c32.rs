//! The complex FFT for `C32`; see the module documentation of `fft`.
//!
//! # Layout
//!
//! The transform works in place on the `2 N` floats of the `N` samples. `L` is the number of
//! `f32` lanes of the vectors of one width (4, 8 or 16; AVX-512 uses 8 or 4 lanes for the
//! sizes below 256). Between the passes the data is in blocks of `L` complex numbers stored as
//! `L` real parts followed by `L` imaginary parts (split form), in the bit-reversed order of a
//! decimation-in-time transform:
//!
//! 1. [`first_pass`]: the bit reversal, the first `log2 L` stages and the conversion to split
//!    form in one pass over `L x L` tiles. With `N = 2^b` and `L = 2^q`, the sample
//!    `a N / L + t L + c` (`a`, `c < L`) is in row `a`, column `c` of tile `t`. The first `q`
//!    stages are `L`-point DFTs over `a`, one per column; the DFT of column `c` of tile `t` is the
//!    block `rev_q(c) T + rev(t)` (`T` tiles, `rev` reverses `b - 2 q` bits), which is row
//!    `rev_q(c)` of tile `rev(t)`. So a tile is loaded row by row, transformed across its rows,
//!    transposed, and stored in place of its mirror tile `rev(t)` (and the mirror in its place).
//! 2. radix-4 passes ([`pass4`]) with three complex products per butterfly, after one radix-2 pass
//!    ([`pass2`]) if the number of remaining stages is odd. A pass of span `S` multiplies the lane
//!    `k` of its four quarters by `1`, `W_S^2k`, `W_S^k` and `W_S^3k`. Where the vectors of the
//!    eight inputs of a radix-8 butterfly and its twiddles fit the registers (NEON with 32
//!    registers of 4 lanes, wasm32), radix-8 passes ([`pass8`]) take the stages that remain
//!    after as few radix-4 passes as make their number a multiple of 3; they need fewer loads,
//!    stores and loop instructions per stage.
//! 3. the last pass stores interleaved samples (and scales the inverse transform by `1/N`).
//!
//! For `N = L^2` (16 with 4 lanes, 64 with 8) the tile is the whole transform: [`codelet`]
//! multiplies the transposed tile by its twiddles and runs the remaining `L`-point DFT across the
//! rows in registers.
//!
//! # Twiddles
//!
//! With the feature `fft-tables` the passes of spans up to 4096 read their twiddles from static
//! per-span tables in split form (`build.rs`): `W_S^k` for a span does not depend on `N`, and for
//! power-of-two sizes the tables are bit-identical with the plan's twiddles (`W^3k` is exact
//! instead of a product; radix-8 passes compute `W^5k` and `W^7k` as one product). The plan's
//! table then serves only the spans above 4096. Without the feature every twiddle of the passes
//! comes from the plan: contiguous loads where the table stride is 1, scalar loads otherwise, and
//! products of two exact entries for the rest (`W^3k = W^k W^2k` in radix-4 passes; `W^5k`,
//! `W^6k`, `W^7k` from `W^k .. W^4k` in radix-8 passes). The constants of the `L`-point DFTs
//! across the rows of a tile (`W_16^t`) are constants of the program in both cases.
//!
//! # Levels
//!
//! The code is written once against `fearless_simd`; [`Lanes`] holds the data movement that a
//! level may do better, with [`Portable`] as the implementation of every level and the fallback
//! of the others:
//!
//! - AVX2 (`x86::InLane`): the rows of a tile are deinterleaved with one `vshufps` per vector
//!   into the lane order [`IN_LANE`] and the tile is transposed with in-lane shuffles (24 per
//!   8 x 8 matrix instead of 48); the 8 x 8 codelet keeps that lane order up to its store.
//! - NEON (`neon::Ld2`): interleaved rows and results move with `ld2`/`st2`.
//! - AVX-512 uses the portable operations (its two-source permutations are single instructions)
//!   with 16 lanes from 256 points, 8 lanes below.
//!
//! NEON and wasm32 use radix-8 passes; the sizes 16 to 256 get a transform of their own in which
//! the size is a constant (see `Fft::fft`).
//!
//! # Builds with debug assertions
//!
//! The tiles live in arrays of 16 vectors and the `L`-point DFTs are unrolled, which the optimiser
//! turns into registers only if the whole transform is one function. So the functions of the
//! transform are `#[inline(always)]`, but only without debug assertions: without optimisation
//! (the dev profile of a crate that uses this one outside this workspace), forced inlining would
//! give every slot of every inlined copy its own place in one frame, over a megabyte, more than
//! the stack of a wasm32 program or a thread. With debug assertions each function keeps its own
//! frame (at most about 10 KiB without optimisation, with 16-lane vectors) and the optimiser, if
//! there is one, inlines as it sees fit; a function that it leaves out of line is compiled without
//! the target features of the level (correct, only slower). The small `kernel!` functions stay
//! `#[inline(always)]`.

use core::iter::Zip;
use core::marker::PhantomData;
use core::slice::Iter;

use fearless_simd::{Simd as SimdToken, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, prelude::*};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, FftScale, Scaling};
use sdrk_core::{C32, Error, Result};

use crate::Simd;
use crate::util::check_len;

#[cfg(feature = "fft-tables")]
mod tables {
    // The generated file is plain data in the shortest round-trip notation of the build script,
    // so the literal lints do not apply.
    #![allow(clippy::unreadable_literal, clippy::excessive_precision)]
    include!(concat!(env!("OUT_DIR"), "/fft_tables.rs"));
}

/// Runs `$body` once for each `$i` below `$n` and below the bound 4, 8 or 16 with `$i` a constant,
/// without a loop. The registers of a tile live in arrays that stay in registers only if every
/// index is a constant, which a loop would leave to the unrolling heuristics of the compiler. The
/// bodies are kept to a call of a small function, so that the copies stay small.
macro_rules! unroll {
    ($i:ident < $n:expr; 4 => $body:block) => {
        unroll!(@each $i $n; $body; 0 1 2 3)
    };
    ($i:ident < $n:expr; 8 => $body:block) => {
        unroll!(@each $i $n; $body; 0 1 2 3 4 5 6 7)
    };
    ($i:ident < $n:expr; 16 => $body:block) => {
        unroll!(@each $i $n; $body; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    };
    (@each $i:ident $n:expr; $body:block; $($k:literal)*) => {{
        let n: usize = $n;
        $(
            if $k < n {
                let $i: usize = $k;
                $body
            }
        )*
    }};
}

/// A complex vector in split form: `L` real parts and `L` imaginary parts.
type Cv<V> = (V, V);

/// The vector width of one transform and the data movement that depends on the level.
trait Lanes<S: SimdToken>: Copy {
    /// `L` real or imaginary parts.
    type V: SimdFloat<S, Element = f32>;
    /// The same bytes as `L / 2` complex numbers in `f64` lanes.
    type D: SimdFloat<S, Element = f64, ByteVector = <Self::V as SimdBase<S>>::ByteVector>;
    /// `2 L` floats: `L` complex numbers in split form (a block) or interleaved (a row of a tile
    /// before the first pass, a block after the last).
    type Block: bytemuck::Pod;
    /// `L` floats (the real or the imaginary parts of a block).
    type Lane: bytemuck::Pod;

    /// The lanes as a vector.
    fn lanes(simd: S, x: &Self::Lane) -> Self::V;

    /// The block in split form.
    fn load(simd: S, b: &Self::Block) -> Cv<Self::V>;

    /// Stores split vectors as a block in split form.
    fn store(v: Cv<Self::V>, b: &mut Self::Block);

    /// Loads `L` interleaved complex numbers in split form. The order of the lanes is the
    /// implementation's; [`Lanes::transpose`] undoes it.
    fn load_row(simd: S, b: &Self::Block) -> Cv<Self::V>;

    /// Transposes a tile of `L` rows: if on entry `rows[r]` holds row `r` as loaded by
    /// [`Lanes::load_row`], on exit `rows[k]` lane `j` holds the element of row `rev j` and column
    /// `rev k`, where `rev` reverses `log2 L` bits.
    fn transpose(simd: S, rows: &mut [Self::V; 16]);

    /// Stores split vectors as `L` interleaved complex numbers.
    fn store_interleaved(v: Cv<Self::V>, b: &mut Self::Block);

    /// Whether the codelet (`N = L^2`) keeps its 8 lanes in the order [`IN_LANE`] between its
    /// transposition and its store, with [`Lanes::codelet_transpose`] and
    /// [`Lanes::codelet_store`]; otherwise they are in natural order.
    const CODELET_IN_LANE: bool = false;

    /// [`Lanes::transpose`] for the codelet: as `transpose`, but with the lanes `j` of the result
    /// in the order of the codelet.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn codelet_transpose(simd: S, rows: &mut [Self::V; 16]) {
        Self::transpose(simd, rows);
    }

    /// [`Lanes::store_interleaved`] for split vectors in the lane order of the codelet.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn codelet_store(v: Cv<Self::V>, b: &mut Self::Block) {
        Self::store_interleaved(v, b);
    }
}

/// The order of the 8 lanes that the in-lane shuffles of AVX2 give: one `vshufps` deinterleaves
/// two vectors of interleaved complex numbers into this order, and `vunpcklps`/`vunpckhps` turn it
/// back into interleaved complex numbers in natural order.
const IN_LANE: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

/// [`Lanes`] with the portable operations of `fearless_simd` (natural lane order).
#[derive(Clone, Copy)]
struct Portable<V, D>(PhantomData<(V, D)>);

impl<S, V, D> Lanes<S> for Portable<V, D>
where
    S: SimdToken,
    V: SimdFloat<S, Element = f32>,
    V::Array: bytemuck::Pod,
    D: SimdFloat<S, Element = f64, ByteVector = <V as SimdBase<S>>::ByteVector>,
{
    type V = V;
    type D = D;
    type Block = [V::Array; 2];
    type Lane = V::Array;

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn lanes(simd: S, x: &V::Array) -> V {
        V::load_array_ref(simd, x)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn load(simd: S, [re, im]: &Self::Block) -> Cv<V> {
        (V::load_array_ref(simd, re), V::load_array_ref(simd, im))
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn store(v: Cv<V>, [re, im]: &mut Self::Block) {
        v.0.store_array(re);
        v.1.store_array(im);
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn load_row(simd: S, b: &Self::Block) -> Cv<V> {
        let (a, b) = Self::load(simd, b);
        a.deinterleave(b)
    }

    /// `log2 L` stages of `zip_low`/`zip_high` between rows at distance 1, 2, 4, ...
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn transpose(_simd: S, rows: &mut [V; 16]) {
        let l = V::LEN;
        unroll!(stage < l.trailing_zeros() as usize; 4 => {
            unroll!(r < l; 16 => {
                zip_rows::<S, V>(rows, 1 << stage, r);
            });
        });
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn store_interleaved(v: Cv<V>, b: &mut Self::Block) {
        Self::store(v.0.interleave(v.1), b);
    }
}

/// The AVX2 data movement of the tile pass: in-lane shuffles only (`vshufps`, `vunpck*ps`,
/// `vperm2f128`), with a lane order that the transposition undoes for free.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256, _mm256_permute2f128_ps, _mm256_shuffle_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256, _mm256_permute2f128_ps, _mm256_shuffle_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps,
    };
    use fearless_simd::{Avx2, Level, f32x8, f64x4, prelude::*};

    use super::{Cv, Lanes, Portable};

    /// The AVX2 token if the level is AVX2 and not AVX-512 (whose two-source permutations make
    /// the portable shuffles single instructions).
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(super) fn avx2_only(level: Level) -> Option<Avx2> {
        if level.as_avx512().is_some() {
            None
        } else {
            level.as_avx2()
        }
    }

    fearless_simd::kernel!(
        /// The real and imaginary parts of the 8 interleaved complex numbers `a`, `b` (4 each),
        /// lane `l` holding the number `PI[l]`, `PI = IN_LANE`.
        #[inline(always)]
        fn deinterleave(t: Avx2, a: f32x8<Avx2>, b: f32x8<Avx2>) -> (f32x8<Avx2>, f32x8<Avx2>) {
            let (a, b): (__m256, __m256) = (a.into(), b.into());
            (
                f32x8::simd_from(t, _mm256_shuffle_ps::<0x88>(a, b)),
                f32x8::simd_from(t, _mm256_shuffle_ps::<0xDD>(a, b)),
            )
        }
    );

    fearless_simd::kernel!(
        /// `vunpcklps` and `vunpckhps`.
        #[inline(always)]
        fn unpack(avx2: Avx2, a: f32x8<Avx2>, b: f32x8<Avx2>) -> (f32x8<Avx2>, f32x8<Avx2>) {
            let (a, b): (__m256, __m256) = (a.into(), b.into());
            (
                f32x8::simd_from(avx2, _mm256_unpacklo_ps(a, b)),
                f32x8::simd_from(avx2, _mm256_unpackhi_ps(a, b)),
            )
        }
    );

    fearless_simd::kernel!(
        /// The low and the high 64-bit pairs of each 128-bit lane of `a` and `b`.
        #[inline(always)]
        fn pairs(avx2: Avx2, a: f32x8<Avx2>, b: f32x8<Avx2>) -> (f32x8<Avx2>, f32x8<Avx2>) {
            let (a, b): (__m256, __m256) = (a.into(), b.into());
            (
                f32x8::simd_from(avx2, _mm256_shuffle_ps::<0x44>(a, b)),
                f32x8::simd_from(avx2, _mm256_shuffle_ps::<0xEE>(a, b)),
            )
        }
    );

    fearless_simd::kernel!(
        /// The low 128-bit lanes of `a` and `b`, and the high ones.
        #[inline(always)]
        fn halves(avx2: Avx2, a: f32x8<Avx2>, b: f32x8<Avx2>) -> (f32x8<Avx2>, f32x8<Avx2>) {
            let (a, b): (__m256, __m256) = (a.into(), b.into());
            (
                f32x8::simd_from(avx2, _mm256_permute2f128_ps::<0x20>(a, b)),
                f32x8::simd_from(avx2, _mm256_permute2f128_ps::<0x31>(a, b)),
            )
        }
    );

    /// Transposes the 8 x 8 matrix `r` (8 unpacks, 8 shuffles, 8 lane permutations).
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn transpose8(avx2: Avx2, r: &mut [f32x8<Avx2>; 8]) {
        let [r0, r1, r2, r3, r4, r5, r6, r7] = *r;
        let ((t0, t1), (t2, t3)) = (unpack(avx2, r0, r1), unpack(avx2, r2, r3));
        let ((t4, t5), (t6, t7)) = (unpack(avx2, r4, r5), unpack(avx2, r6, r7));
        let ((u0, u1), (u2, u3)) = (pairs(avx2, t0, t2), pairs(avx2, t1, t3));
        let ((u4, u5), (u6, u7)) = (pairs(avx2, t4, t6), pairs(avx2, t5, t7));
        let ((c0, c4), (c1, c5)) = (halves(avx2, u0, u4), halves(avx2, u1, u5));
        let ((c2, c6), (c3, c7)) = (halves(avx2, u2, u6), halves(avx2, u3, u7));
        *r = [c0, c1, c2, c3, c4, c5, c6, c7];
    }

    /// [`Lanes`] of AVX2: a row is deinterleaved with one `vshufps` per vector into the lane order
    /// `PI` of [`deinterleave`]; the tile is transposed with in-lane shuffles, which also restore
    /// the natural order (24 shuffles per 8 x 8 matrix and 2 per row, against 48 and 4 with the
    /// portable operations).
    #[derive(Clone, Copy)]
    pub(super) struct InLane;

    impl Lanes<Avx2> for InLane {
        type V = f32x8<Avx2>;
        type D = f64x4<Avx2>;
        type Block = [[f32; 8]; 2];
        type Lane = [f32; 8];

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn lanes(simd: Avx2, x: &[f32; 8]) -> f32x8<Avx2> {
            <Portable<f32x8<Avx2>, f64x4<Avx2>> as Lanes<Avx2>>::lanes(simd, x)
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn load(simd: Avx2, b: &Self::Block) -> Cv<f32x8<Avx2>> {
            <Portable<f32x8<Avx2>, f64x4<Avx2>> as Lanes<Avx2>>::load(simd, b)
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn store(v: Cv<f32x8<Avx2>>, b: &mut Self::Block) {
            <Portable<f32x8<Avx2>, f64x4<Avx2>> as Lanes<Avx2>>::store(v, b);
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn load_row(simd: Avx2, b: &Self::Block) -> Cv<f32x8<Avx2>> {
            let (a, b) = Self::load(simd, b);
            deinterleave(simd, a, b)
        }

        /// With `rows[r]` lane `l` = element `(r, PI[l])`: the transposition of the rows in
        /// bit-reversed order gives in vector `l` lane `j` the element `(rev j, PI[l])`, which is
        /// the result for `k = rev(PI[l])`.
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn transpose(simd: Avx2, rows: &mut [f32x8<Avx2>; 16]) {
            let mut t = [
                rows[0], rows[4], rows[2], rows[6], rows[1], rows[5], rows[3], rows[7],
            ];
            transpose8(simd, &mut t);
            // `out[k] = t[PI[rev k]]` (`PI` is its own inverse).
            [rows[0], rows[1], rows[2], rows[3]] = [t[0], t[2], t[4], t[6]];
            [rows[4], rows[5], rows[6], rows[7]] = [t[1], t[3], t[5], t[7]];
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn store_interleaved(v: Cv<f32x8<Avx2>>, b: &mut Self::Block) {
            <Portable<f32x8<Avx2>, f64x4<Avx2>> as Lanes<Avx2>>::store_interleaved(v, b);
        }

        const CODELET_IN_LANE: bool = true;

        /// As [`InLane::transpose`] with the rows taken in the order `rev(PI[r])`, which puts
        /// the element of row `rev(PI[j])` into lane `j`.
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn codelet_transpose(simd: Avx2, rows: &mut [f32x8<Avx2>; 16]) {
            let mut t = [
                rows[0], rows[4], rows[1], rows[5], rows[2], rows[6], rows[3], rows[7],
            ];
            transpose8(simd, &mut t);
            [rows[0], rows[1], rows[2], rows[3]] = [t[0], t[2], t[4], t[6]];
            [rows[4], rows[5], rows[6], rows[7]] = [t[1], t[3], t[5], t[7]];
        }

        /// `vunpcklps` and `vunpckhps` give the numbers `0 .. 4` and `4 .. 8` from lanes in the
        /// order `PI`.
        #[cfg_attr(not(debug_assertions), inline(always))]
        fn codelet_store(v: Cv<f32x8<Avx2>>, b: &mut Self::Block) {
            Self::store(unpack(v.0.token(), v.0, v.1), b);
        }
    }
}

/// The NEON data movement: rows and results move between interleaved samples and split vectors
/// with the structure loads and stores `ld2` and `st2` (one instruction for two vectors, against
/// two loads or stores and two `uzp`/`zip`).
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use core::arch::aarch64::{float32x4x2_t, vld2q_f32, vst2q_f32};

    use fearless_simd::{Neon, f32x4, f64x2, prelude::*};

    use super::{Cv, Lanes, Portable};

    fearless_simd::kernel!(
        /// The real and imaginary parts of the 4 interleaved complex numbers of `src`.
        #[inline(always)]
        fn ld2(neon: Neon, src: &[f32; 8]) -> (f32x4<Neon>, f32x4<Neon>) {
            // SAFETY: `src` holds the 8 floats that `vld2q_f32` reads.
            let v = unsafe { vld2q_f32(src.as_ptr()) };
            (f32x4::simd_from(neon, v.0), f32x4::simd_from(neon, v.1))
        }
    );

    fearless_simd::kernel!(
        /// Stores 4 complex numbers given as real and imaginary parts interleaved into `dst`.
        #[inline(always)]
        fn st2(_neon: Neon, re: f32x4<Neon>, im: f32x4<Neon>, dst: &mut [f32; 8]) {
            // SAFETY: `dst` holds the 8 floats that `vst2q_f32` writes.
            unsafe { vst2q_f32(dst.as_mut_ptr(), float32x4x2_t(re.into(), im.into())) }
        }
    );

    /// [`Lanes`] of NEON: [`Portable`] with `ld2` for the rows and `st2` for the results.
    #[derive(Clone, Copy)]
    pub(super) struct Ld2;

    type Base = Portable<f32x4<Neon>, f64x2<Neon>>;

    impl Lanes<Neon> for Ld2 {
        type V = f32x4<Neon>;
        type D = f64x2<Neon>;
        type Block = [[f32; 4]; 2];
        type Lane = [f32; 4];

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn lanes(simd: Neon, x: &[f32; 4]) -> f32x4<Neon> {
            <Base as Lanes<Neon>>::lanes(simd, x)
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn load(simd: Neon, b: &Self::Block) -> Cv<f32x4<Neon>> {
            <Base as Lanes<Neon>>::load(simd, b)
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn store(v: Cv<f32x4<Neon>>, b: &mut Self::Block) {
            <Base as Lanes<Neon>>::store(v, b);
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn load_row(simd: Neon, b: &Self::Block) -> Cv<f32x4<Neon>> {
            match b.as_flattened().try_into() {
                Ok(src) => ld2(simd, src),
                Err(_) => <Base as Lanes<Neon>>::load_row(simd, b),
            }
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn transpose(simd: Neon, rows: &mut [f32x4<Neon>; 16]) {
            <Base as Lanes<Neon>>::transpose(simd, rows);
        }

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn store_interleaved(v: Cv<f32x4<Neon>>, b: &mut Self::Block) {
            match b.as_flattened_mut().try_into() {
                Ok(dst) => st2(v.0.token(), v.0, v.1, dst),
                Err(_) => <Base as Lanes<Neon>>::store_interleaved(v, b),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Small helpers

/// Stores `(p, q)` as `(x[i], x[j])`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn put_pair<T>(x: &mut [T], i: usize, j: usize, (p, q): (T, T)) {
    if let Some(s) = x.get_mut(i) {
        *s = p;
    }
    if let Some(s) = x.get_mut(j) {
        *s = q;
    }
}

/// The step of the zip network of [`Portable::transpose`] for row `r` at distance `d`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn zip_rows<S: SimdToken, V: SimdFloat<S, Element = f32>>(rows: &mut [V; 16], d: usize, r: usize) {
    if let (0, Some(&a), Some(&b)) = (r & d, rows.get(r), rows.get(r + d)) {
        put_pair(rows, r, r + d, a.interleave(b));
    }
}

/// `x` with its lowest `bits` bits reversed (and the others cleared).
#[cfg_attr(not(debug_assertions), inline(always))]
fn rev(x: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS - bits)
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn cadd<S: SimdToken, V: SimdFloat<S, Element = f32>>(a: Cv<V>, b: Cv<V>) -> Cv<V> {
    (a.0 + b.0, a.1 + b.1)
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn csub<S: SimdToken, V: SimdFloat<S, Element = f32>>(a: Cv<V>, b: Cv<V>) -> Cv<V> {
    (a.0 - b.0, a.1 - b.1)
}

/// `w a` (forward) or `conj(w) a` (inverse).
///
/// The differences are written as one product and one fused multiply-subtract where the level
/// has one: `mul_sub` (`a b - c`, `vfmsub` with FMA on x86) everywhere but on AArch64, where
/// NEON has only `c - a b` (`fmls`), which `(-a).mul_add(b, c)` gives without the negation.
#[cfg_attr(not(debug_assertions), inline(always))]
fn cmul<S: SimdToken, V: SimdFloat<S, Element = f32>, const INV: bool>(
    w: Cv<V>,
    a: Cv<V>,
) -> Cv<V> {
    if INV {
        (
            w.1.mul_add(a.1, w.0 * a.0),
            diff::<S, V>(w.0, a.1, w.1, a.0),
        )
    } else {
        (
            diff::<S, V>(w.0, a.0, w.1, a.1),
            w.1.mul_add(a.0, w.0 * a.1),
        )
    }
}

/// `x y - u v` for [`cmul`].
#[cfg_attr(not(debug_assertions), inline(always))]
fn diff<S: SimdToken, V: SimdFloat<S, Element = f32>>(x: V, y: V, u: V, v: V) -> V {
    if cfg!(target_arch = "aarch64") {
        (-u).mul_add(v, x * y)
    } else {
        x.mul_sub(y, u * v)
    }
}

/// `(a - j b, a + j b)` (forward) or `(a + j b, a - j b)` (inverse): the second half of a
/// radix-4 butterfly, with the rotation folded into the sums.
#[cfg_attr(not(debug_assertions), inline(always))]
fn rot_pair<S: SimdToken, V: SimdFloat<S, Element = f32>, const INV: bool>(
    a: Cv<V>,
    b: Cv<V>,
) -> (Cv<V>, Cv<V>) {
    let p = (a.0 + b.1, a.1 - b.0);
    let q = (a.0 - b.1, a.1 + b.0);
    if INV { (q, p) } else { (p, q) }
}

// ---------------------------------------------------------------------------------------------
// Twiddles

/// The plan's twiddles and how to find the twiddles of a pass.
#[derive(Clone, Copy)]
struct Twiddles<'a> {
    /// The plan's table as floats: `W_N^k` is the complex number at index `k * stride`.
    plan: &'a [f32],
    stride: usize,
    n: usize,
    /// Whether the static span tables are used (feature `fft-tables`).
    tables: bool,
}

/// `(re, im)` of the plan's complex number at `idx` (zero outside the table).
#[cfg_attr(not(debug_assertions), inline(always))]
fn entry(tw: &[f32], idx: usize) -> (f32, f32) {
    match tw.get(2 * idx..2 * idx + 2) {
        Some([re, im]) => (*re, *im),
        _ => (0.0, 0.0),
    }
}

/// The plan's complex number at `idx` as one 64-bit word (zero outside the table).
#[cfg_attr(not(debug_assertions), inline(always))]
fn entry_bits(tw: &[[f32; 2]], idx: usize) -> f64 {
    tw.get(idx)
        .map_or(0.0, |&x| f64::from_bits(bytemuck::cast::<[f32; 2], u64>(x)))
}

/// The static tables `(A_S, B_S)` of the span `S = 2^log2` as `(re, im)` halves.
#[cfg(feature = "fft-tables")]
#[cfg_attr(not(debug_assertions), inline(always))]
fn span_table(log2: u32) -> Option<[(&'static [f32], &'static [f32]); 2]> {
    let (a, b) = tables::span(log2)?;
    Some([a.split_at(a.len() / 2), b.split_at(b.len() / 2)])
}

#[cfg(not(feature = "fft-tables"))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn span_table(_log2: u32) -> Option<[(&'static [f32], &'static [f32]); 2]> {
    None
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
/// `W_S^k` of the static tables for the span `S = 2^log2` as `(re, im)` halves (`k < S / 2`, and
/// `k < S / 4` for `S = 4096`): the twiddles of the post-processing of an `rfft` of `S` points
/// (see `real`). `None` without the feature `fft-tables` and above 4096 points.
#[inline(always)]
pub(super) fn real_twiddles(log2: u32) -> Option<(&'static [f32], &'static [f32])> {
    span_table(log2).map(|[a, _]| a)
}

/// Where the twiddles of one pass come from.
#[derive(Clone, Copy)]
enum PassTwiddles<'a> {
    /// Split slices of the static tables: `W_S^k`, `W_S^2k` and `W_S^3k` for `k` from 0.
    Table([(&'a [f32], &'a [f32]); 3]),
    /// The plan's table: `W_S^k` at index `k * s`, `W_S^2k` at `2 k s`.
    Plan { tw: &'a [f32], s: usize },
}

/// Where the twiddles of one radix-8 pass come from.
#[derive(Clone, Copy)]
enum Radix8Twiddles<'a> {
    /// Split slices of the static tables: `W_S^k`, `W_S^2k`, `W_S^3k`, `W_S^4k` and `W_S^6k`.
    Table([(&'a [f32], &'a [f32]); 5]),
    /// The plan's table: `W_S^k` at index `k * s`.
    Plan { tw: &'a [f32], s: usize },
}

impl<'a> Twiddles<'a> {
    /// The twiddles of the radix-4 pass of span `span`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn radix4(&self, span: usize) -> PassTwiddles<'a> {
        if self.tables {
            let log2 = span.trailing_zeros();
            if let (Some([(a_re, a_im), (b_re, b_im)]), Some([half, _])) =
                (span_table(log2), span_table(log2 - 1))
            {
                return PassTwiddles::Table([(a_re, a_im), half, (b_re, b_im)]);
            }
        }
        PassTwiddles::Plan {
            tw: self.plan,
            s: self.n / span * self.stride,
        }
    }

    /// The twiddles of the radix-8 pass of span `span`: from the tables `W^k` (`A_S`), `W^2k`
    /// (`A_S/2`), `W^3k` (`B_S`), `W^4k` (`A_S/4`) and `W^6k` (`B_S/2`).
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn radix8(&self, span: usize) -> Radix8Twiddles<'a> {
        if self.tables {
            let log2 = span.trailing_zeros();
            if let (Some([a, b]), Some([a2, b2]), Some([a4, _])) =
                (span_table(log2), span_table(log2 - 1), span_table(log2 - 2))
            {
                return Radix8Twiddles::Table([a, a2, b, a4, b2]);
            }
        }
        Radix8Twiddles::Plan {
            tw: self.plan,
            s: self.n / span * self.stride,
        }
    }

    /// The twiddles of the radix-2 pass of span `span` (in the slot of `W_S^k`).
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn radix2(&self, span: usize) -> PassTwiddles<'a> {
        if self.tables
            && let Some([a, b]) = span_table(span.trailing_zeros())
        {
            return PassTwiddles::Table([a, a, b]);
        }
        PassTwiddles::Plan {
            tw: self.plan,
            s: self.n / span * self.stride,
        }
    }
}

/// `W_16^t = e^(-j 2 pi t / 16)` for `t < 8`, rounded to f32 (the test `roots_are_rounded_libm`
/// checks them against `libm`; `t = 0` and `t = 4` are exact).
const ROOTS_16: [(f32, f32); 8] = [
    (1.0, 0.0),
    (0.923_879_5, -0.382_683_43),
    (
        core::f32::consts::FRAC_1_SQRT_2,
        -core::f32::consts::FRAC_1_SQRT_2,
    ),
    (0.382_683_43, -0.923_879_5),
    (0.0, -1.0),
    (-0.382_683_43, -0.923_879_5),
    (
        -core::f32::consts::FRAC_1_SQRT_2,
        -core::f32::consts::FRAC_1_SQRT_2,
    ),
    (-0.923_879_5, -0.382_683_43),
];

/// `W_l^t` for `t < l / 2` (`l` a power of two up to 16, conjugated for the inverse), the
/// constants of the `l`-point DFTs across the rows of a tile or of a radix-8 butterfly. Constants
/// of the program rather than plan values, so that they can be operands in memory instead of
/// occupying registers.
#[cfg_attr(not(debug_assertions), inline(always))]
fn roots<S: SimdToken, V: SimdFloat<S, Element = f32>, const INV: bool>(
    simd: S,
    l: usize,
) -> [Cv<V>; 8] {
    let zero = V::splat(simd, 0.0);
    let mut w = [(zero, zero); 8];
    for (t, slot) in w.iter_mut().enumerate().take(l / 2) {
        if let Some(&(re, im)) = ROOTS_16.get(t * 16 / l) {
            *slot = (
                V::splat(simd, re),
                V::splat(simd, if INV { -im } else { im }),
            );
        }
    }
    w
}

/// [`PassTwiddles`] for the lane groups `j < mb` of one width: the table slices as `mb` lane
/// arrays each.
#[derive(Clone, Copy)]
enum LaneTwiddles<'a, L> {
    Table([(&'a [L], &'a [L]); 3]),
    Plan { tw: &'a [f32], s: usize },
}

/// The first `len` floats of `x` as lane arrays (empty if `x` is shorter).
#[cfg_attr(not(debug_assertions), inline(always))]
fn lane_slice<L: bytemuck::Pod>(x: &[f32], len: usize) -> &[L] {
    x.get(..len)
        .and_then(|x| bytemuck::try_cast_slice(x).ok())
        .unwrap_or(&[])
}

impl<'a> PassTwiddles<'a> {
    /// The twiddles of the first `mb` lane groups of `P`.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn lanes<S: SimdToken, P: Lanes<S>>(&self, mb: usize) -> LaneTwiddles<'a, P::Lane> {
        let len = mb * P::V::LEN;
        let cast = |x: &'a [f32]| lane_slice::<P::Lane>(x, len);
        match *self {
            Self::Table(t) => LaneTwiddles::Table(t.map(|(re, im)| (cast(re), cast(im)))),
            Self::Plan { tw, s } => LaneTwiddles::Plan { tw, s },
        }
    }
}

/// The plan's twiddles `W^k` for the lanes `k = k0 .. k0 + L` from the complex numbers `k s` of
/// `tw`, loaded as 64-bit words (real and imaginary part together) and separated by one
/// deinterleave.
#[cfg_attr(not(debug_assertions), inline(always))]
fn gather<S: SimdToken, P: Lanes<S>>(simd: S, tw: &[[f32; 2]], k0: usize, s: usize) -> Cv<P::V> {
    let h = P::D::LEN;
    let lo = P::D::from_fn(simd, |i| entry_bits(tw, (k0 + i) * s));
    let hi = P::D::from_fn(simd, |i| entry_bits(tw, (k0 + h + i) * s));
    lo.bitcast::<P::V>().deinterleave(hi.bitcast::<P::V>())
}

/// The longest prefix of `tw` that is a whole number of `T`, as values of `T`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn plan_view<T: bytemuck::Pod>(tw: &[f32]) -> &[T] {
    let size = core::mem::size_of::<T>() / core::mem::size_of::<f32>();
    tw.get(..tw.len() / size.max(1) * size)
        .and_then(|x| bytemuck::try_cast_slice(x).ok())
        .unwrap_or(&[])
}

/// The twiddles of one lane group of a pass from an item of the iterator that walks the source
/// of the twiddles: `K` complex vectors (`W^(e k)`, `e = 1 ..= K`, for the lanes `k`).
///
/// The passes walk zipped slices (or a range of lane groups) and convert each item with this
/// trait instead of a closure of `map`: the iterators keep the single index of a zip of slices,
/// and the conversion, which holds the vector code, is always inlined and so compiled with the
/// target features of the level (a closure may be compiled out of line without them).
trait Twiddle<S: SimdToken, P: Lanes<S>, T, const K: usize> {
    fn twiddles(&self, simd: S, item: T) -> [Cv<P::V>; K];
}

/// Twiddles from the static span tables (items: the zipped lane arrays of the 3 or 5 tables).
struct FromTables;

/// Twiddles from the plan's table with stride 1 (items: the block `j`, the blocks `2 j, 2 j + 1`
/// and for radix-8 passes the blocks `4 j .. 4 j + 4` with `j`); `complex` is the table as complex
/// numbers, for `W^3k` of radix-8 passes.
struct FromBlocks<'a> {
    complex: &'a [[f32; 2]],
}

/// Twiddles gathered from the plan's table with the stride `s` (items: the lane groups `j`).
struct Gathered<'a> {
    tw: &'a [[f32; 2]],
    s: usize,
}

/// `[W^k, W^2k, W^3k]` with `W^3k = W^k W^2k`.
impl<S: SimdToken, P: Lanes<S>> Twiddle<S, P, usize, 3> for Gathered<'_> {
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn twiddles(&self, simd: S, j: usize) -> [Cv<P::V>; 3] {
        let k0 = j * P::V::LEN;
        let w1 = gather::<S, P>(simd, self.tw, k0, self.s);
        let w2 = gather::<S, P>(simd, self.tw, k0, 2 * self.s);
        [w1, w2, cmul::<S, P::V, false>(w1, w2)]
    }
}

/// `W^(e k)` for `e = 1 .. 8`: `e <= 4` from the complex numbers `e k s` (all below `N / 2`),
/// the others products of two of them.
impl<S: SimdToken, P: Lanes<S>> Twiddle<S, P, usize, 7> for Gathered<'_> {
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn twiddles(&self, simd: S, j: usize) -> [Cv<P::V>; 7] {
        let (k0, tw, s) = (j * P::V::LEN, self.tw, self.s);
        let w1 = gather::<S, P>(simd, tw, k0, s);
        let w2 = gather::<S, P>(simd, tw, k0, 2 * s);
        let w3 = gather::<S, P>(simd, tw, k0, 3 * s);
        let w4 = gather::<S, P>(simd, tw, k0, 4 * s);
        [
            w1,
            w2,
            w3,
            w4,
            cmul::<S, P::V, false>(w1, w4),
            cmul::<S, P::V, false>(w2, w4),
            cmul::<S, P::V, false>(w3, w4),
        ]
    }
}

/// The twiddles `[W^k, W^2k, W^3k]` of the lanes `k = j L .. (j + 1) L` of a radix-4 pass with
/// the table stride 1: `W^k` is the block `w1` of the plan's table (`L` interleaved complex
/// numbers), `W^2k` every other number of the two blocks `w2`, and `W^3k = W^k W^2k`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn plan_contiguous<S: SimdToken, P: Lanes<S>>(
    simd: S,
    w1: &P::Block,
    w2: &[P::Block],
) -> [Cv<P::V>; 3] {
    let (a, b) = P::load(simd, w1);
    let w1 = a.deinterleave(b);
    let w2 = even::<S, P>(simd, w2, 0).deinterleave(even::<S, P>(simd, w2, 1));
    [w1, w2, cmul::<S, P::V, false>(w1, w2)]
}

/// The complex numbers `0, 2, 4, ...` of the block `w[i]` of interleaved numbers (`L / 2` of them,
/// interleaved in one vector; zero if there is no such block).
#[cfg_attr(not(debug_assertions), inline(always))]
fn even<S: SimdToken, P: Lanes<S>>(simd: S, w: &[P::Block], i: usize) -> P::V {
    match w.get(i) {
        Some(b) => {
            let (a, b) = P::load(simd, b);
            a.bitcast::<P::D>()
                .unzip_low(b.bitcast::<P::D>())
                .bitcast::<P::V>()
        }
        None => P::V::splat(simd, 0.0),
    }
}

/// One table of split twiddles walked lane group by lane group: real and imaginary parts.
type Pairs<'a, L> = Zip<Iter<'a, L>, Iter<'a, L>>;

/// The pairs `(re, im)` of a split table as an iterator.
#[cfg_attr(not(debug_assertions), inline(always))]
fn pairs<'a, L>((re, im): (&'a [L], &'a [L])) -> Pairs<'a, L> {
    re.iter().zip(im)
}

/// The zipped lane arrays `(re, im)` of the three tables of a radix-4 pass.
type Item3<'a, L> = (((&'a L, &'a L), (&'a L, &'a L)), (&'a L, &'a L));

/// The zipped lane arrays `(re, im)` of the five tables of a radix-8 pass.
type Item5<'a, L> = (
    (
        ((&'a L, &'a L), (&'a L, &'a L)),
        ((&'a L, &'a L), (&'a L, &'a L)),
    ),
    (&'a L, &'a L),
);

/// The split vectors of a pair of lane arrays.
#[cfg_attr(not(debug_assertions), inline(always))]
fn lane_pair<S: SimdToken, P: Lanes<S>>(simd: S, (re, im): (&P::Lane, &P::Lane)) -> Cv<P::V> {
    (P::lanes(simd, re), P::lanes(simd, im))
}

/// `[W^k, W^2k, W^3k]`.
impl<'a, S: SimdToken, P: Lanes<S>> Twiddle<S, P, Item3<'a, P::Lane>, 3> for FromTables {
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn twiddles(&self, simd: S, ((w1, w2), w3): Item3<'a, P::Lane>) -> [Cv<P::V>; 3] {
        [
            lane_pair::<S, P>(simd, w1),
            lane_pair::<S, P>(simd, w2),
            lane_pair::<S, P>(simd, w3),
        ]
    }
}

/// `W^k`, `W^2k`, `W^3k`, `W^4k`, `W^6k` from the tables, `W^5k = W^k W^4k` and
/// `W^7k = W^3k W^4k`.
impl<'a, S: SimdToken, P: Lanes<S>> Twiddle<S, P, Item5<'a, P::Lane>, 7> for FromTables {
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn twiddles(&self, simd: S, (((w1, w2), (w3, w4)), w6): Item5<'a, P::Lane>) -> [Cv<P::V>; 7] {
        let w1 = lane_pair::<S, P>(simd, w1);
        let w3 = lane_pair::<S, P>(simd, w3);
        let w4 = lane_pair::<S, P>(simd, w4);
        [
            w1,
            lane_pair::<S, P>(simd, w2),
            w3,
            w4,
            cmul::<S, P::V, false>(w1, w4),
            lane_pair::<S, P>(simd, w6),
            cmul::<S, P::V, false>(w3, w4),
        ]
    }
}

/// [`plan_contiguous`].
impl<'a, S: SimdToken, P: Lanes<S>> Twiddle<S, P, (&'a P::Block, &'a [P::Block]), 3>
    for FromBlocks<'_>
{
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn twiddles(&self, simd: S, (w1, w2): (&'a P::Block, &'a [P::Block])) -> [Cv<P::V>; 3] {
        plan_contiguous::<S, P>(simd, w1, w2)
    }
}

/// [`plan_contiguous8`].
impl<'a, S: SimdToken, P: Lanes<S>>
    Twiddle<S, P, (usize, ((&'a P::Block, &'a [P::Block]), &'a [P::Block])), 7> for FromBlocks<'_>
{
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn twiddles(
        &self,
        simd: S,
        (j, ((w1, w2), w4)): (usize, ((&'a P::Block, &'a [P::Block]), &'a [P::Block])),
    ) -> [Cv<P::V>; 7] {
        plan_contiguous8::<S, P>(simd, (w1, w2, w4), self.complex, j)
    }
}

// ---------------------------------------------------------------------------------------------
// Passes

/// The rows of a tile in split form: `L` real parts and `L` imaginary parts.
type Tile<V> = ([V; 16], [V; 16]);

/// In-register DIF radix-2 `n`-point DFT across the rows `x[0 .. n]` (`n` a power of two up to
/// 16; natural order in, bit-reversed order out), with the constants `w[t] = W_n^t`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn dft_rows<S: SimdToken, V: SimdFloat<S, Element = f32>, const INV: bool>(
    x: &mut Tile<V>,
    w: &[Cv<V>; 8],
    n: usize,
) {
    unroll!(stage < n.trailing_zeros() as usize; 4 => {
        unroll!(p < n / 2; 8 => {
            dft_butterfly::<S, V, INV>(x, w, n, n >> (stage + 1), p);
        });
    });
}

/// The first stage of [`dft_rows`]. After it the rows of each half `x[k n / 2 .. (k + 1) n / 2]`
/// are an `n / 2`-point DFT of their own ([`dft_half`]): a transform that stores the results of
/// one half before it computes the other needs fewer registers at a time.
#[cfg_attr(not(debug_assertions), inline(always))]
fn dft_first_stage<S: SimdToken, V: SimdFloat<S, Element = f32>, const INV: bool>(
    x: &mut Tile<V>,
    w: &[Cv<V>; 8],
    n: usize,
) {
    unroll!(p < n / 2; 8 => {
        dft_butterfly::<S, V, INV>(x, w, n, n / 2, p);
    });
}

/// The stages of [`dft_rows`] after the first for the rows of the half `k`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn dft_half<S: SimdToken, V: SimdFloat<S, Element = f32>, const INV: bool>(
    x: &mut Tile<V>,
    w: &[Cv<V>; 8],
    n: usize,
    k: usize,
) {
    let (quarter, first) = (n / 4, k * n / 4);
    unroll!(stage < n.trailing_zeros() as usize - 1; 4 => {
        let h = quarter >> stage;
        unroll!(q < quarter; 8 => {
            dft_butterfly::<S, V, INV>(x, w, n, h, first + q);
        });
    });
}

/// Butterfly `p` of the stage of half size `h` of the `l`-point [`dft_rows`]: the rows
/// `(g + j, g + j + h)` with `j = p mod h`, `g = 2 h floor(p / h)`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn dft_butterfly<S: SimdToken, V: SimdFloat<S, Element = f32>, const INV: bool>(
    x: &mut Tile<V>,
    w: &[Cv<V>; 8],
    l: usize,
    h: usize,
    p: usize,
) {
    let (j, g) = (p % h, p / h * 2 * h);
    let (i0, i1) = (g + j, g + j + h);
    if let (Some(&ur), Some(&ui), Some(&vr), Some(&vi)) =
        (x.0.get(i0), x.1.get(i0), x.0.get(i1), x.1.get(i1))
    {
        let y = rotate::<S, V, INV>((ur - vr, ui - vi), w, l, j * (l / (2 * h)));
        put_pair(&mut x.0, i0, i1, (ur + vr, y.0));
        put_pair(&mut x.1, i0, i1, (ui + vi, y.1));
    }
}

/// `W_l^t d` (conjugated for the inverse) with the constants `w` of [`roots`].
#[cfg_attr(not(debug_assertions), inline(always))]
fn rotate<S: SimdToken, V: SimdFloat<S, Element = f32>, const INV: bool>(
    d: Cv<V>,
    w: &[Cv<V>; 8],
    l: usize,
    t: usize,
) -> Cv<V> {
    if t == 0 {
        d
    } else if 4 * t == l {
        // `W_l^(l/4) = -j` (`+j` for the inverse).
        if INV { (-d.1, d.0) } else { (d.1, -d.0) }
    } else if let (true, Some(&(c, _)), Some(&(m, _))) =
        (8 * t == l || 8 * t == 3 * l, w.get(l / 8), w.get(3 * l / 8))
    {
        // `W_8 = c (1 - j)` and `W_8^3 = m (1 + j)` with `c = cos(pi / 4) = -m` (conjugated
        // for the inverse): one product and two fused operations with the real constants `c`
        // and `m` instead of a complex product.
        let (a, b) = d;
        let p = c * a;
        match (INV, 8 * t == l) {
            (false, true) => (c.mul_add(b, p), c.mul_sub(b, p)),
            (false, false) => (c.mul_sub(b, p), m.mul_sub(b, p)),
            (true, true) => (m.mul_add(b, p), c.mul_add(b, p)),
            (true, false) => (m.mul_sub(b, p), m.mul_add(b, p)),
        }
    } else {
        w.get(t).map_or(d, |&c| cmul::<S, V, false>(c, d))
    }
}

#[allow(unsafe_code)]
mod grid {
    use core::marker::PhantomData;
    use core::ptr::NonNull;

    /// A slice of `rows * cols` blocks as a row-major matrix. A tile of the first pass is one
    /// column, at a distance of `cols` blocks per row; so are the blocks of one lane group of a
    /// radix-2, -4 or -8 pass in every span, with the halves, quarters or eighths of the spans as
    /// the rows. A column is checked once; its `rows` elements then need no bounds check (`a <
    /// rows` is a constant comparison where `rows` is the number of lanes, and
    /// [`Column::chunks`] counts chunks of rows instead).
    pub(super) struct Grid<'a, B> {
        ptr: NonNull<B>,
        rows: usize,
        cols: usize,
        _blocks: PhantomData<&'a mut [B]>,
    }

    impl<'a, B> Grid<'a, B> {
        /// The first `rows * (len / rows)` blocks of `blocks` as `rows` rows.
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn new(blocks: &'a mut [B], rows: usize) -> Self {
            let cols = blocks.len().checked_div(rows).unwrap_or(0);
            Self {
                ptr: NonNull::from(blocks).cast(),
                rows,
                cols,
                _blocks: PhantomData,
            }
        }

        /// The number of columns.
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn cols(&self) -> usize {
            self.cols
        }

        /// Column `c`, or `None` outside the matrix.
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn column(&mut self, c: usize) -> Option<Column<'_, B>> {
            if c >= self.cols {
                return None;
            }
            Some(Column {
                // SAFETY: `c < cols <= len` of the slice that `ptr` comes from.
                ptr: unsafe { self.ptr.add(c) },
                rows: self.rows,
                cols: self.cols,
                _grid: PhantomData,
            })
        }
    }

    /// One column of a [`Grid`], borrowed mutably from it.
    pub(super) struct Column<'g, B> {
        /// The element in row 0.
        ptr: NonNull<B>,
        rows: usize,
        cols: usize,
        _grid: PhantomData<&'g mut [B]>,
    }

    impl<B> Column<'_, B> {
        /// The element in row `a`.
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn get(&self, a: usize) -> Option<&B> {
            if a >= self.rows {
                return None;
            }
            // SAFETY: `ptr` is element `c` with `c < cols`, so element `a * cols + c` is below
            // `rows * cols <= len` of the grid's slice, which `self` borrows mutably; the
            // reference lives as long as `&self`.
            Some(unsafe { self.ptr.add(a * self.cols).as_ref() })
        }

        /// The element in row `a`, mutable.
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn get_mut(&mut self, a: usize) -> Option<&mut B> {
            if a >= self.rows {
                return None;
            }
            // SAFETY: as in `get`; `&mut self` makes the reference unique.
            Some(unsafe { self.ptr.add(a * self.cols).as_mut() })
        }

        /// The elements of the rows `K i .. K i + K` for `i < rows / K`, one array per `i`.
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn chunks<const K: usize>(&mut self) -> Chunks<'_, B, K> {
            Chunks {
                ptr: self.ptr,
                cols: self.cols,
                next: 0,
                len: self.rows / K,
                _column: PhantomData,
            }
        }
    }

    /// The iterator of [`Column::chunks`]. Unlike `get_mut` in a loop over the rows, it needs no
    /// bounds check per element, only the count of the chunks.
    pub(super) struct Chunks<'c, B, const K: usize> {
        /// The element in row 0 of the column.
        ptr: NonNull<B>,
        cols: usize,
        /// The index of the next chunk.
        next: usize,
        /// The number of chunks.
        len: usize,
        _column: PhantomData<&'c mut [B]>,
    }

    impl<'c, B, const K: usize> Iterator for Chunks<'c, B, K> {
        type Item = [&'c mut B; K];

        #[cfg_attr(not(debug_assertions), inline(always))]
        fn next(&mut self) -> Option<Self::Item> {
            if self.next >= self.len {
                return None;
            }
            let first = self.next * K;
            self.next += 1;
            let (ptr, cols) = (self.ptr, self.cols);
            Some(core::array::from_fn(|i| {
                // SAFETY: the row `first + i` is below `len * K <= rows`, so as in `get` the
                // element is in the grid's slice, which the column borrows mutably for `'c`
                // through the iterator. The chunks do not overlap and each is handed out once, so
                // the references are unique.
                unsafe { ptr.add((first + i) * cols).as_mut() }
            }))
        }
    }
}

use grid::{Column, Grid};

/// Loads a tile (row `a` is `col[a]`), transforms it across the rows and transposes it: on exit
/// `x.0[k]`, `x.1[k]` are the block of row `k` of the mirror tile (with the lanes in the order of
/// the codelet for `CODELET`).
#[cfg_attr(not(debug_assertions), inline(always))]
fn tile<S: SimdToken, P: Lanes<S>, const INV: bool, const CODELET: bool>(
    simd: S,
    col: &Column<'_, P::Block>,
    w: &[Cv<P::V>; 8],
) -> Tile<P::V> {
    let l = P::V::LEN;
    let zero = P::V::splat(simd, 0.0);
    let mut x = ([zero; 16], [zero; 16]);
    // The first stage of the DFT as the rows arrive, then each half on its own: fewer rows live
    // at a time than with all loads first.
    unroll!(a < l / 2; 8 => {
        load_row::<S, P>(simd, col, &mut x, a);
        load_row::<S, P>(simd, col, &mut x, a + l / 2);
        dft_butterfly::<S, P::V, INV>(&mut x, w, l, l / 2, a);
    });
    dft_half::<S, P::V, INV>(&mut x, w, l, 0);
    dft_half::<S, P::V, INV>(&mut x, w, l, 1);
    if CODELET {
        // The imaginary parts first: the register allocation of the codelet then spills less
        // (AVX2, N = 64: 5.84 instead of 5.89 instructions per sample).
        P::codelet_transpose(simd, &mut x.1);
        P::codelet_transpose(simd, &mut x.0);
    } else {
        P::transpose(simd, &mut x.0);
        P::transpose(simd, &mut x.1);
    }
    x
}

/// Row `a` of the tile `col` into `x`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn load_row<S: SimdToken, P: Lanes<S>>(
    simd: S,
    col: &Column<'_, P::Block>,
    x: &mut Tile<P::V>,
    a: usize,
) {
    if let (Some(b), Some(re), Some(im)) = (col.get(a), x.0.get_mut(a), x.1.get_mut(a)) {
        (*re, *im) = P::load_row(simd, b);
    }
}

/// Stores a transformed tile as the blocks `col[k]`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn store_tile<S: SimdToken, P: Lanes<S>>(x: &Tile<P::V>, col: &mut Column<'_, P::Block>) {
    unroll!(k < P::V::LEN; 16 => {
        store_row::<S, P>(x, col, k);
    });
}

/// Row `k` of `x` into the block `col[k]`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn store_row<S: SimdToken, P: Lanes<S>>(x: &Tile<P::V>, col: &mut Column<'_, P::Block>, k: usize) {
    if let (Some(&re), Some(&im), Some(b)) = (x.0.get(k), x.1.get(k), col.get_mut(k)) {
        P::store((re, im), b);
    }
}

/// The bit reversal, the first `log2 L` stages and the conversion to split form (`N >= L^2`).
#[cfg_attr(not(debug_assertions), inline(always))]
fn first_pass<S: SimdToken, P: Lanes<S>, const INV: bool>(
    simd: S,
    blocks: &mut [P::Block],
    w: &[Cv<P::V>; 8],
) {
    let mut grid = Grid::new(blocks, P::V::LEN);
    let tiles = grid.cols();
    // `mirror` counts in bit-reversed order. From `t - 1` to `t` the lowest
    // `trailing_zeros(t) + 1` bits of the count flip, so the same number of the highest bits of
    // `mirror` flip (`tiles` is a power of two): a few instructions without a branch, where a
    // carry loop costs about 15 per tile (sse2, N = 1024: 25.67 -> 25.30 instructions per sample).
    let mut mirror = 0;
    for t in 0..tiles {
        if t > 0 {
            mirror ^= tiles - ((tiles / 2) >> t.trailing_zeros());
        }
        if mirror < t {
            continue;
        }
        let Some(mut col) = grid.column(t) else {
            continue;
        };
        let x = tile::<S, P, INV, false>(simd, &col, w);
        if mirror == t {
            store_tile::<S, P>(&x, &mut col);
        } else if let Some(mut mirror_col) = grid.column(mirror) {
            let y = tile::<S, P, INV, false>(simd, &mirror_col, w);
            store_tile::<S, P>(&x, &mut mirror_col);
            if let Some(mut col) = grid.column(t) {
                store_tile::<S, P>(&y, &mut col);
            }
        }
    }
}

/// One radix-2 stage of span `2 m` on split blocks (`mb = m / L` blocks per half). With the halves
/// as the rows of a grid of `mb` columns, the butterflies of the lane group `j` in every span are
/// the pairs of rows of the column `j`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn pass2<S: SimdToken, P: Lanes<S>, const INV: bool>(
    simd: S,
    blocks: &mut [P::Block],
    mb: usize,
    tw: PassTwiddles<'_>,
) {
    let rows = blocks.len() / mb.max(1);
    let mut grid = Grid::new(blocks, rows);
    match tw.lanes::<S, P>(mb) {
        LaneTwiddles::Table([(re, im), ..]) => {
            for (j, (re, im)) in (0..mb).zip(re.iter().zip(im)) {
                let w = (P::lanes(simd, re), P::lanes(simd, im));
                if let Some(mut col) = grid.column(j) {
                    radix2_lanes::<S, P, INV>(simd, &mut col, w);
                }
            }
        }
        LaneTwiddles::Plan { tw, s } => {
            let tw = plan_view::<[f32; 2]>(tw);
            for j in 0..mb {
                let w = gather::<S, P>(simd, tw, j * P::V::LEN, s);
                if let Some(mut col) = grid.column(j) {
                    radix2_lanes::<S, P, INV>(simd, &mut col, w);
                }
            }
        }
    }
}

/// The butterflies of [`pass2`] of one lane group in every span (the column `col`), with the
/// twiddles `w`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn radix2_lanes<S: SimdToken, P: Lanes<S>, const INV: bool>(
    simd: S,
    col: &mut Column<'_, P::Block>,
    w: Cv<P::V>,
) {
    for [b0, b1] in col.chunks::<2>() {
        let a = P::load(simd, b0);
        let t = cmul::<S, P::V, INV>(w, P::load(simd, b1));
        P::store(cadd::<S, P::V>(a, t), b0);
        P::store(csub::<S, P::V>(a, t), b1);
    }
}

/// Stores a result of a pass: in split form, or for the last pass (`LAST`) as interleaved samples
/// scaled by `g` (inverse only).
#[cfg_attr(not(debug_assertions), inline(always))]
fn store_result<S: SimdToken, P: Lanes<S>, const INV: bool, const LAST: bool>(
    v: Cv<P::V>,
    b: &mut P::Block,
    g: P::V,
) {
    if LAST {
        P::store_interleaved(if INV { (v.0 * g, v.1 * g) } else { v }, b);
    } else {
        P::store(v, b);
    }
}

/// The radix-4 butterflies of the blocks `b0 .. b3` (the same lanes of the four quarters of a
/// span) with the twiddles `[W^k, W^2k, W^3k]`. `LAST` stores interleaved samples scaled by `g`
/// (inverse only).
#[cfg_attr(not(debug_assertions), inline(always))]
fn butterfly4<S: SimdToken, P: Lanes<S>, const INV: bool, const LAST: bool>(
    simd: S,
    [w1, w2, w3]: [Cv<P::V>; 3],
    [b0, b1, b2, b3]: [&mut P::Block; 4],
    g: P::V,
) {
    let x0 = P::load(simd, b0);
    let t1 = cmul::<S, P::V, INV>(w2, P::load(simd, b1));
    let t2 = cmul::<S, P::V, INV>(w1, P::load(simd, b2));
    let t3 = cmul::<S, P::V, INV>(w3, P::load(simd, b3));
    let (a, b) = (cadd::<S, P::V>(x0, t1), csub::<S, P::V>(x0, t1));
    let (c, d) = (cadd::<S, P::V>(t2, t3), csub::<S, P::V>(t2, t3));
    let (y1, y3) = rot_pair::<S, P::V, INV>(b, d);
    store_result::<S, P, INV, LAST>(cadd::<S, P::V>(a, c), b0, g);
    store_result::<S, P, INV, LAST>(y1, b1, g);
    store_result::<S, P, INV, LAST>(csub::<S, P::V>(a, c), b2, g);
    store_result::<S, P, INV, LAST>(y3, b3, g);
}

/// One radix-4 pass of span `4 m` on split blocks (`mb = m / L` blocks per quarter). The last pass
/// (`LAST`) stores interleaved samples scaled by `scale` (inverse only).
#[cfg_attr(not(debug_assertions), inline(always))]
fn pass4<S: SimdToken, P: Lanes<S>, const INV: bool, const LAST: bool>(
    simd: S,
    blocks: &mut [P::Block],
    mb: usize,
    tw: PassTwiddles<'_>,
    scale: f32,
) {
    let g = P::V::splat(simd, scale);
    match tw.lanes::<S, P>(mb) {
        LaneTwiddles::Table([w1, w2, w3]) => {
            let tw = pairs(w1).zip(pairs(w2)).zip(pairs(w3));
            radix4::<S, P, INV, LAST, _, _>(simd, blocks, mb, tw, &FromTables, g);
        }
        LaneTwiddles::Plan { tw, s: 1 } => {
            let table = plan_view::<P::Block>(tw);
            let tw = table.iter().zip(table.chunks_exact(2));
            let from = FromBlocks { complex: &[] };
            radix4::<S, P, INV, LAST, _, _>(simd, blocks, mb, tw, &from, g);
        }
        LaneTwiddles::Plan { tw, s } => {
            let from = Gathered {
                tw: plan_view(tw),
                s,
            };
            radix4::<S, P, INV, LAST, _, _>(simd, blocks, mb, 0..mb, &from, g);
        }
    }
}

/// [`pass4`] with the twiddles of the lane groups `j < mb` from the items of `tw`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn radix4<S, P, const INV: bool, const LAST: bool, I, C>(
    simd: S,
    blocks: &mut [P::Block],
    mb: usize,
    tw: I,
    from: &C,
    g: P::V,
) where
    S: SimdToken,
    P: Lanes<S>,
    I: Iterator,
    C: Twiddle<S, P, I::Item, 3>,
{
    if blocks.len() == 4 * mb {
        // One span: walk the four quarters together.
        let (h01, h23) = blocks.split_at_mut(2 * mb);
        let (q0, q1) = h01.split_at_mut(mb);
        let (q2, q3) = h23.split_at_mut(mb);
        let quarters = q0.iter_mut().zip(q1).zip(q2).zip(q3);
        for (w, (((b0, b1), b2), b3)) in tw.zip(quarters) {
            let w = from.twiddles(simd, w);
            butterfly4::<S, P, INV, LAST>(simd, w, [b0, b1, b2, b3], g);
        }
        return;
    }
    // Several spans: the twiddles of one lane group serve the same lanes of every span. With the
    // quarters as the rows of a grid of `mb` columns, these are the chunks of four rows of the
    // column `j`.
    let rows = blocks.len() / mb.max(1);
    let mut grid = Grid::new(blocks, rows);
    for (j, w) in (0..mb).zip(tw) {
        let w = from.twiddles(simd, w);
        let Some(mut col) = grid.column(j) else {
            return;
        };
        for quarters in col.chunks::<4>() {
            butterfly4::<S, P, INV, LAST>(simd, w, quarters, g);
        }
    }
}

/// The radix-8 butterflies of the blocks `b[0 .. 8]` (the same lanes of the eight sub-transforms
/// of a span) with the twiddles `w[e - 1] = W^(e k)` and the constants `w8[t] = W_8^t`. With the
/// sub-transforms `x_e` in bit-reversed order, the result is the 8-point DFT of
/// `u[e] = W^(e k) x_rev(e)`, which [`dft_rows`] leaves in bit-reversed order again. `LAST` stores
/// interleaved samples scaled by `g` (inverse only).
#[cfg_attr(not(debug_assertions), inline(always))]
fn butterfly8<S: SimdToken, P: Lanes<S>, const INV: bool, const LAST: bool>(
    simd: S,
    w: [Cv<P::V>; 7],
    [b0, b1, b2, b3, b4, b5, b6, b7]: [&mut P::Block; 8],
    w8: &[Cv<P::V>; 8],
    g: P::V,
) {
    let [w1, w2, w3, w4, w5, w6, w7] = w;
    let zero = P::V::splat(simd, 0.0);
    let mut x = ([zero; 16], [zero; 16]);
    (x.0[0], x.1[0]) = P::load(simd, b0);
    (x.0[1], x.1[1]) = cmul::<S, P::V, INV>(w1, P::load(simd, b4));
    (x.0[2], x.1[2]) = cmul::<S, P::V, INV>(w2, P::load(simd, b2));
    (x.0[3], x.1[3]) = cmul::<S, P::V, INV>(w3, P::load(simd, b6));
    (x.0[4], x.1[4]) = cmul::<S, P::V, INV>(w4, P::load(simd, b1));
    (x.0[5], x.1[5]) = cmul::<S, P::V, INV>(w5, P::load(simd, b5));
    (x.0[6], x.1[6]) = cmul::<S, P::V, INV>(w6, P::load(simd, b3));
    (x.0[7], x.1[7]) = cmul::<S, P::V, INV>(w7, P::load(simd, b7));
    dft_rows::<S, P::V, INV>(&mut x, w8, 8);
    let [y0, y1, y2, y3, y4, y5, y6, y7, ..] = x.0;
    let [z0, z1, z2, z3, z4, z5, z6, z7, ..] = x.1;
    store_result::<S, P, INV, LAST>((y0, z0), b0, g);
    store_result::<S, P, INV, LAST>((y1, z1), b4, g);
    store_result::<S, P, INV, LAST>((y2, z2), b2, g);
    store_result::<S, P, INV, LAST>((y3, z3), b6, g);
    store_result::<S, P, INV, LAST>((y4, z4), b1, g);
    store_result::<S, P, INV, LAST>((y5, z5), b5, g);
    store_result::<S, P, INV, LAST>((y6, z6), b3, g);
    store_result::<S, P, INV, LAST>((y7, z7), b7, g);
}

/// The eight eighths of `group`, `mb` blocks each.
#[cfg_attr(not(debug_assertions), inline(always))]
fn eighths<B>(group: &mut [B], mb: usize) -> Option<[&mut [B]; 8]> {
    let (h0, h1) = group.split_at_mut_checked(4 * mb)?;
    let (q0, q1) = h0.split_at_mut_checked(2 * mb)?;
    let (q2, q3) = h1.split_at_mut_checked(2 * mb)?;
    let (e0, e1) = q0.split_at_mut_checked(mb)?;
    let (e2, e3) = q1.split_at_mut_checked(mb)?;
    let (e4, e5) = q2.split_at_mut_checked(mb)?;
    let (e6, e7) = q3.split_at_mut_checked(mb)?;
    Some([e0, e1, e2, e3, e4, e5, e6, e7])
}

/// The twiddles `W^(e k)`, `e = 1 .. 8`, of the lanes `k = j L .. (j + 1) L` of a radix-8 pass
/// with the table stride 1: `W^k` is the block `w1` of the plan's table, `W^2k` every other number
/// of the two blocks `w2`, `W^4k` every fourth number of the four blocks `w4`, `W^3k` the complex
/// numbers `3 k` of `complex`, and the others products of two of them.
#[cfg_attr(not(debug_assertions), inline(always))]
fn plan_contiguous8<S: SimdToken, P: Lanes<S>>(
    simd: S,
    (w1, w2, w4): (&P::Block, &[P::Block], &[P::Block]),
    complex: &[[f32; 2]],
    j: usize,
) -> [Cv<P::V>; 7] {
    let (a, b) = P::load(simd, w1);
    let w1 = a.deinterleave(b);
    let w2 = even::<S, P>(simd, w2, 0).deinterleave(even::<S, P>(simd, w2, 1));
    // Every fourth number: every other number of the even numbers.
    let q0 = even::<S, P>(simd, w4, 0)
        .bitcast::<P::D>()
        .unzip_low(even::<S, P>(simd, w4, 1).bitcast::<P::D>());
    let q1 = even::<S, P>(simd, w4, 2)
        .bitcast::<P::D>()
        .unzip_low(even::<S, P>(simd, w4, 3).bitcast::<P::D>());
    let w4 = q0.bitcast::<P::V>().deinterleave(q1.bitcast::<P::V>());
    let w3 = gather::<S, P>(simd, complex, j * P::V::LEN, 3);
    [
        w1,
        w2,
        w3,
        w4,
        cmul::<S, P::V, false>(w1, w4),
        cmul::<S, P::V, false>(w2, w4),
        cmul::<S, P::V, false>(w3, w4),
    ]
}

/// One radix-8 pass of span `8 m` on split blocks (`mb = m / L` blocks per eighth). The last pass
/// (`LAST`) stores interleaved samples scaled by `scale` (inverse only).
#[cfg_attr(not(debug_assertions), inline(always))]
fn pass8<S: SimdToken, P: Lanes<S>, const INV: bool, const LAST: bool>(
    simd: S,
    blocks: &mut [P::Block],
    mb: usize,
    tw: Radix8Twiddles<'_>,
    scale: f32,
) {
    let g = P::V::splat(simd, scale);
    let w8 = roots::<S, P::V, INV>(simd, 8);
    match tw {
        Radix8Twiddles::Table(t) => {
            let len = mb * P::V::LEN;
            let [w1, w2, w3, w4, w6] =
                t.map(|(re, im)| (lane_slice::<P::Lane>(re, len), lane_slice(im, len)));
            let tw = (pairs(w1).zip(pairs(w2)))
                .zip(pairs(w3).zip(pairs(w4)))
                .zip(pairs(w6));
            radix8::<S, P, INV, LAST, _, _>(simd, blocks, mb, tw, &FromTables, &w8, g);
        }
        Radix8Twiddles::Plan { tw, s: 1 } => {
            let (table, complex) = (plan_view::<P::Block>(tw), plan_view(tw));
            let tw = (table.iter().zip(table.chunks_exact(2)))
                .zip(table.chunks_exact(4))
                .enumerate();
            let from = FromBlocks { complex };
            radix8::<S, P, INV, LAST, _, _>(simd, blocks, mb, tw, &from, &w8, g);
        }
        Radix8Twiddles::Plan { tw, s } => {
            let from = Gathered {
                tw: plan_view(tw),
                s,
            };
            radix8::<S, P, INV, LAST, _, _>(simd, blocks, mb, 0..mb, &from, &w8, g);
        }
    }
}

/// [`pass8`] with the twiddles of the lane groups `j < mb` from the items of `tw`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn radix8<S, P, const INV: bool, const LAST: bool, I, C>(
    simd: S,
    blocks: &mut [P::Block],
    mb: usize,
    tw: I,
    from: &C,
    w8: &[Cv<P::V>; 8],
    g: P::V,
) where
    S: SimdToken,
    P: Lanes<S>,
    I: Iterator,
    C: Twiddle<S, P, I::Item, 7>,
{
    if blocks.len() == 8 * mb {
        // One span: walk the eight eighths together.
        let Some([e0, e1, e2, e3, e4, e5, e6, e7]) = eighths(blocks, mb) else {
            return;
        };
        let lo = e0.iter_mut().zip(e1).zip(e2.iter_mut().zip(e3));
        let hi = e4.iter_mut().zip(e5).zip(e6.iter_mut().zip(e7));
        for (w, (((b0, b1), (b2, b3)), ((b4, b5), (b6, b7)))) in tw.zip(lo.zip(hi)) {
            let w = from.twiddles(simd, w);
            butterfly8::<S, P, INV, LAST>(simd, w, [b0, b1, b2, b3, b4, b5, b6, b7], w8, g);
        }
        return;
    }
    // Several spans: the twiddles of one lane group serve the same lanes of every span (the
    // chunks of eight rows of the column `j`, as in [`radix4`]).
    let rows = blocks.len() / mb.max(1);
    let mut grid = Grid::new(blocks, rows);
    for (j, w) in (0..mb).zip(tw) {
        let w = from.twiddles(simd, w);
        let Some(mut col) = grid.column(j) else {
            return;
        };
        for eighths in col.chunks::<8>() {
            butterfly8::<S, P, INV, LAST>(simd, w, eighths, w8, g);
        }
    }
}

/// The twiddles `W_N^(c k)` of the rows `c = 1 .. L` of the codelet (`N = L^2`, `L <= 8`) for the
/// columns `k < L` in the lane order of the codelet, row `c` at index `c - 1`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn codelet_twiddles<S: SimdToken, P: Lanes<S>>(simd: S, tw: &Twiddles<'_>) -> [Cv<P::V>; 7] {
    type V<S, P> = <P as Lanes<S>>::V;
    let l = V::<S, P>::LEN;
    let zero = V::<S, P>::splat(simd, 0.0);
    let mut out = [(zero, zero); 7];
    #[cfg(feature = "fft-tables")]
    if tw.tables {
        // A row of the table has the layout of a block in split form.
        let table: &[f32] = if l == 4 {
            &tables::CODELET_16
        } else if P::CODELET_IN_LANE {
            &tables::CODELET_64_IN_LANE
        } else {
            &tables::CODELET_64
        };
        for (slot, row) in out.iter_mut().zip(plan_view::<P::Block>(table)) {
            *slot = P::load(simd, row);
        }
        return out;
    }
    for (c, slot) in (1..l).zip(out.iter_mut()) {
        *slot = (
            V::<S, P>::from_fn(simd, |lane| codelet_entry::<S, P>(tw, c, lane).0),
            V::<S, P>::from_fn(simd, |lane| codelet_entry::<S, P>(tw, c, lane).1),
        );
    }
    out
}

/// `W_N^(c k)` from the plan's table for the codelet lane `lane` (column `k`), with
/// `W^j = -W^(j - N/2)` for the indices beyond the table.
#[cfg_attr(not(debug_assertions), inline(always))]
fn codelet_entry<S: SimdToken, P: Lanes<S>>(
    tw: &Twiddles<'_>,
    c: usize,
    lane: usize,
) -> (f32, f32) {
    let k = match IN_LANE.get(lane) {
        Some(&k) if P::CODELET_IN_LANE => k,
        _ => lane,
    };
    let half = tw.n / 2;
    let j = c * k % tw.n;
    let (re, im) = entry(tw.plan, (j % half) * tw.stride);
    if j >= half { (-re, -im) } else { (re, im) }
}

/// The whole transform of `N = L^2` points in registers: the tile, the twiddles, the second
/// `L`-point DFT across the rows and the interleaved store.
#[cfg_attr(not(debug_assertions), inline(always))]
fn codelet<S: SimdToken, P: Lanes<S>, const INV: bool>(
    simd: S,
    blocks: &mut [P::Block],
    tw: &Twiddles<'_>,
    w: &[Cv<P::V>; 8],
) {
    let l = P::V::LEN;
    let mut grid = Grid::new(blocks, l);
    let Some(col) = grid.column(0) else {
        return;
    };
    let t = tile::<S, P, INV, true>(simd, &col, w);
    let zero = P::V::splat(simd, 0.0);
    let mut x = ([zero; 16], [zero; 16]);
    let wc = codelet_twiddles::<S, P>(simd, tw);
    unroll!(c < l; 8 => {
        codelet_row::<S, P, INV>(&t, &mut x, &wc, c);
    });
    // The second DFT half by half, each half stored before the next (AVX2, N = 64: 5.70 instead
    // of 5.78 instructions per sample, fewer spills of the 16 registers of the tile).
    dft_first_stage::<S, P::V, INV>(&mut x, w, l);
    let g = P::V::splat(simd, 1.0 / tw.n as f32);
    dft_half::<S, P::V, INV>(&mut x, w, l, 0);
    unroll!(r < l / 2; 8 => {
        codelet_store::<S, P, INV>(&x, blocks, g, r);
    });
    dft_half::<S, P::V, INV>(&mut x, w, l, 1);
    unroll!(r < l / 2; 8 => {
        codelet_store::<S, P, INV>(&x, blocks, g, l / 2 + r);
    });
}

/// Row `c` of the second DFT of the codelet: the block of column `c`, which [`tile`] left in row
/// `rev c` of `t`, times `W_N^(c k)` (`wc[c - 1]`).
#[cfg_attr(not(debug_assertions), inline(always))]
fn codelet_row<S: SimdToken, P: Lanes<S>, const INV: bool>(
    t: &Tile<P::V>,
    x: &mut Tile<P::V>,
    wc: &[Cv<P::V>; 7],
    c: usize,
) {
    let bits = P::V::LEN.trailing_zeros();
    if let (Some(&br), Some(&bi), Some(re), Some(im)) = (
        t.0.get(rev(c, bits)),
        t.1.get(rev(c, bits)),
        x.0.get_mut(c),
        x.1.get_mut(c),
    ) {
        (*re, *im) = match c.checked_sub(1).and_then(|i| wc.get(i)) {
            Some(&w) => cmul::<S, P::V, INV>(w, (br, bi)),
            None => (br, bi),
        };
    }
}

/// Stores the result row `r` of the codelet (frequencies `L rev(r) .. L rev(r) + L`) scaled by `g`
/// (inverse only).
#[cfg_attr(not(debug_assertions), inline(always))]
fn codelet_store<S: SimdToken, P: Lanes<S>, const INV: bool>(
    x: &Tile<P::V>,
    blocks: &mut [P::Block],
    g: P::V,
    r: usize,
) {
    let bits = P::V::LEN.trailing_zeros();
    if let (Some(&re), Some(&im), Some(b)) = (x.0.get(r), x.1.get(r), blocks.get_mut(rev(r, bits)))
    {
        P::codelet_store(if INV { (re * g, im * g) } else { (re, im) }, b);
    }
}

/// The transform at the width of `P` (`N >= L^2`).
#[cfg_attr(not(debug_assertions), inline(always))]
fn run<S: SimdToken, P: Lanes<S>, const INV: bool, const RADIX8: bool>(
    simd: S,
    data: &mut [f32],
    tw: Twiddles<'_>,
) {
    let n = tw.n;
    let l = P::V::LEN;
    let Ok(blocks) = bytemuck::try_cast_slice_mut::<f32, P::Block>(data) else {
        return;
    };
    let w = roots::<S, P::V, INV>(simd, l);
    if n == l * l && l <= 8 {
        codelet::<S, P, INV>(simd, blocks, &tw, &w);
        return;
    }
    first_pass::<S, P, INV>(simd, blocks, &w);
    let g = 1.0 / n as f32;
    let mut m = l;
    if RADIX8 {
        // Radix-4 passes until the number of remaining stages is a multiple of 3 (at least 2
        // remain), then radix-8.
        while !(n / m).trailing_zeros().is_multiple_of(3) {
            if 4 * m == n {
                pass4::<S, P, INV, true>(simd, blocks, m / l, tw.radix4(4 * m), g);
                return;
            }
            pass4::<S, P, INV, false>(simd, blocks, m / l, tw.radix4(4 * m), 1.0);
            m *= 4;
        }
        while 8 * m < n {
            pass8::<S, P, INV, false>(simd, blocks, m / l, tw.radix8(8 * m), 1.0);
            m *= 8;
        }
        pass8::<S, P, INV, true>(simd, blocks, m / l, tw.radix8(8 * m), g);
        return;
    }
    if (n / l).trailing_zeros() % 2 == 1 {
        pass2::<S, P, INV>(simd, blocks, m / l, tw.radix2(2 * m));
        m *= 2;
    }
    while 4 * m < n {
        pass4::<S, P, INV, false>(simd, blocks, m / l, tw.radix4(4 * m), 1.0);
        m *= 4;
    }
    pass4::<S, P, INV, true>(simd, blocks, m / l, tw.radix4(4 * m), g);
}

/// Whether 4-lane vectors use radix-8 passes: on wasm32, but not on SSE, where the 16 `xmm`
/// registers spill (sse2 1024: 25.6 with radix-4, 27.5 with radix-8). NEON chooses radix-8
/// itself; AVX-512 does not (its 16-lane twiddles do not fit beside the data: 5.68 against 5.85
/// at 1024).
const RADIX8_4_LANES: bool = cfg!(target_arch = "wasm32");

/// Picks the widest vectors with `N >= L^2`.
#[cfg_attr(not(debug_assertions), inline(always))]
fn transform<S: SimdToken, const INV: bool>(simd: S, data: &mut [f32], tw: Twiddles<'_>) {
    let n = tw.n;
    let l = S::f32s::LEN;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = x86::avx2_only(simd.level())
        && n >= 64
    {
        run::<_, x86::InLane, INV, false>(avx2, data, tw);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = simd.level().as_neon() {
        run::<_, neon::Ld2, INV, true>(neon, data, tw);
        return;
    }
    if l >= 16 && n >= 256 {
        run::<S, Portable<f32x16<S>, f64x8<S>>, INV, false>(simd, data, tw);
    } else if l >= 8 && n >= 64 {
        run::<S, Portable<f32x8<S>, f64x4<S>>, INV, false>(simd, data, tw);
    } else {
        run::<S, Portable<f32x4<S>, f64x2<S>>, INV, RADIX8_4_LANES>(simd, data, tw);
    }
}

/// The transform of `data` with the plan's table `plan` (`W_N^k` at complex index `k * stride`)
/// for the size `N` (`data` must hold `2 N` floats), or for the `data.len() / 2` samples of `data`
/// if `N = 0`; `tables` selects the static span tables where the feature `fft-tables` has them.
/// With a constant size the compiler resolves every choice of the passes and every loop count.
#[cfg_attr(not(debug_assertions), inline(always))]
fn fft_c32_of<S: SimdToken, const N: usize>(
    simd: S,
    data: &mut [f32],
    plan: &[f32],
    stride: usize,
    inverse: bool,
    tables: bool,
) {
    let n = if N > 0 { N } else { data.len() / 2 };
    let Some(data) = data.get_mut(..2 * n) else {
        return;
    };
    let tw = Twiddles {
        plan,
        stride,
        n,
        tables: tables && cfg!(feature = "fft-tables"),
    };
    if inverse {
        transform::<S, true>(simd, data, tw);
    } else {
        transform::<S, false>(simd, data, tw);
    }
}

/// The transforms of [`Fft::fft`]: one function per size up to 256, each with the size as a
/// constant, and one for the other sizes.
///
/// The sizes up to 256, where the fixed costs of a call weigh most, each get a transform of their
/// own in which the size is a constant (instructions per sample on AVX2 for N = 16, 64, 256: 12.4,
/// 6.8, 10.0 -> 9.8, 5.9, 8.3; on sse2 15.0, 22.2, 22.9 -> 12.9, 16.1, 22.1).
///
/// The functions are not generic (unlike `fft`, which is generic over the plan's storage), and
/// they are `#[inline]` only without debug assertions: a crate built with debug assertions then
/// calls the copy compiled in this crate instead of compiling its own (in this workspace, which
/// optimises this crate in every profile, an optimised copy), while an optimised caller can still
/// inline the transform of its size (AVX2, N = 64: 5.8 instead of 6.5 instructions per sample).
mod sized {
    use fearless_simd::Simd as SimdToken;

    use super::fft_c32_of;
    use crate::level::dispatch_fn;

    macro_rules! sized {
        ($($name:ident $size:literal),*) => {
            $(
                dispatch_fn!(
                    #[cfg_attr(not(debug_assertions), inline)]
                    pub(super) fn $name(floats: &mut [f32], tw: &[f32], stride: usize, inverse: bool)
                        => entry::<_, $size>
                );
            )*
        };
    }
    sized!(n16 16, n32 32, n64 64, n128 128, n256 256);

    dispatch_fn!(
        pub(super) fn any(floats: &mut [f32], tw: &[f32], stride: usize, inverse: bool)
            => entry::<_, 0>
    );

    /// The transform of the level `simd` for the size `N` (any size if `N = 0`).
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn entry<S: SimdToken, const N: usize>(
        simd: S,
        floats: &mut [f32],
        tw: &[f32],
        stride: usize,
        inverse: bool,
    ) {
        fft_c32_of::<S, N>(simd, floats, tw, stride, inverse, true);
    }
}

impl Fft<C32> for Simd {
    fn fft<Tw: AsRef<[C32]>>(
        &self,
        plan: &FftPlan<C32, Tw>,
        data: &mut [C32],
        dir: Direction,
        scaling: Scaling,
    ) -> Result<FftScale> {
        fft_c32(plan, data, dir, scaling, true)
    }
}

/// The transform: with `sized`, the sizes from 16 to 256 points take their own sized
/// transforms, every other size (and every size without `sized`) the transform of any size.
/// The inner transform of `rfft` passes `false`: as a second caller of the sized transforms it
/// would keep them out of line of `fft` (a call more per transform, 0.1 instructions per sample
/// at N = 64).
#[inline(always)]
pub(super) fn fft_c32<Tw: AsRef<[C32]>>(
    plan: &FftPlan<C32, Tw>,
    data: &mut [C32],
    dir: Direction,
    scaling: Scaling,
    sized: bool,
) -> Result<FftScale> {
    if scaling == Scaling::BlockFloat {
        return Err(Error::Unsupported(
            "block floating point is not available for c32",
        ));
    }
    let n = plan.len();
    check_len(n, data.len())?;
    let inverse = dir == Direction::Inverse;
    let tw: &[f32] = bytemuck::cast_slice(plan.twiddles());
    let floats: &mut [f32] = bytemuck::cast_slice_mut(data);
    let stride = plan.twiddle_stride();
    // On `log2 N`, a dense range, the dispatch is one jump table; on the powers of two
    // themselves the compiler first maps them to a range (AVX2, N = 64: 4 instructions more).
    match if sized { plan.log2_len() } else { 0 } {
        4 => sized::n16(floats, tw, stride, inverse),
        5 => sized::n32(floats, tw, stride, inverse),
        6 => sized::n64(floats, tw, stride, inverse),
        7 => sized::n128(floats, tw, stride, inverse),
        8 => sized::n256(floats, tw, stride, inverse),
        _ => sized::any(floats, tw, stride, inverse),
    }
    Ok(FftScale {
        exponent: if inverse { plan.log2_len() as i32 } else { 0 },
    })
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
    use sdrk_core::{C32, Fallback};

    use super::{ROOTS_16, fft_c32_of};
    use crate::level::dispatch;

    /// `n` values of a fixed pseudo-random sequence in `[-1, 1)`.
    fn noise(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from((state >> 16) as u16) / 32768.0 - 1.0
            })
            .collect()
    }

    /// The data movement of AVX2 against the portable operations, bit for bit: the rows of a tile
    /// (load and transposition), the transposition of the codelet (lanes in the order `IN_LANE`)
    /// and its interleaved store.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn the_avx2_lanes_match_the_portable_lanes() {
        use fearless_simd::{Avx2, f32x8, f64x4, prelude::*};

        use super::{IN_LANE, Lanes, Portable, x86::InLane};

        type Q = Portable<f32x8<Avx2>, f64x4<Avx2>>;
        let Some(avx2) = crate::level::current().as_avx2() else {
            return;
        };
        let lanes = |v: f32x8<Avx2>| -> [f32; 8] { *v };
        for seed in 0..8 {
            let data = noise(seed, 128);
            let mut rows = [[[0.0f32; 8]; 2]; 8];
            for (row, x) in rows.iter_mut().zip(data.chunks_exact(16)) {
                row[0].copy_from_slice(&x[..8]);
                row[1].copy_from_slice(&x[8..]);
            }
            let zero = f32x8::splat(avx2, 0.0);
            let (mut p, mut q) = (([zero; 16], [zero; 16]), ([zero; 16], [zero; 16]));
            for (r, row) in rows.iter().enumerate() {
                (p.0[r], p.1[r]) = InLane::load_row(avx2, row);
                (q.0[r], q.1[r]) = Q::load_row(avx2, row);
            }
            let mut c = p;
            InLane::transpose(avx2, &mut p.0);
            InLane::transpose(avx2, &mut p.1);
            InLane::codelet_transpose(avx2, &mut c.0);
            InLane::codelet_transpose(avx2, &mut c.1);
            Q::transpose(avx2, &mut q.0);
            Q::transpose(avx2, &mut q.1);
            for k in 0..8 {
                assert_eq!(lanes(p.0[k]), lanes(q.0[k]), "seed {seed} row {k}");
                assert_eq!(lanes(p.1[k]), lanes(q.1[k]), "seed {seed} row {k}");
                let natural = lanes(q.0[k]);
                let permuted: [f32; 8] = core::array::from_fn(|j| natural[IN_LANE[j]]);
                assert_eq!(lanes(c.0[k]), permuted, "seed {seed} codelet row {k}");
                let (mut stored, mut want) = ([[0.0f32; 8]; 2], [[0.0f32; 8]; 2]);
                InLane::codelet_store((c.0[k], c.1[k]), &mut stored);
                Q::store_interleaved((q.0[k], q.1[k]), &mut want);
                assert_eq!(stored, want, "seed {seed} codelet store {k}");
            }
        }
    }

    /// The `ld2`/`st2` movement of NEON against the portable operations, bit for bit.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_neon_lanes_match_the_portable_lanes() {
        use fearless_simd::{Neon, f32x4, f64x2};

        use super::{Lanes, Portable, neon::Ld2};

        type Q = Portable<f32x4<Neon>, f64x2<Neon>>;
        let Some(neon) = crate::level::current().as_neon() else {
            return;
        };
        let lanes = |v: f32x4<Neon>| -> [f32; 4] { *v };
        for seed in 0..8 {
            let data = noise(seed, 8);
            let mut row = [[0.0f32; 4]; 2];
            row[0].copy_from_slice(&data[..4]);
            row[1].copy_from_slice(&data[4..]);
            let (p, q) = (Ld2::load_row(neon, &row), Q::load_row(neon, &row));
            assert_eq!(
                (lanes(p.0), lanes(p.1)),
                (lanes(q.0), lanes(q.1)),
                "seed {seed}"
            );
            let (mut stored, mut want) = ([[0.0f32; 4]; 2], [[0.0f32; 4]; 2]);
            Ld2::store_interleaved(p, &mut stored);
            Q::store_interleaved(q, &mut want);
            assert_eq!(stored, want, "seed {seed}");
            assert_eq!(stored, row, "seed {seed}");
        }
    }

    #[test]
    fn roots_are_rounded_libm() {
        // `W_16^0 = 1` and `W_16^4 = -j` are exact (the DFTs never multiply by them; libm gives
        // `cos(pi / 2) = 6.1e-17`).
        for (t, &(re, im)) in ROOTS_16.iter().enumerate().filter(|&(t, _)| t % 4 != 0) {
            let theta = 2.0 * core::f64::consts::PI * (t as f64 / 16.0);
            assert_eq!(re, libm::cos(theta) as f32, "t={t}");
            assert_eq!(im, -libm::sin(theta) as f32, "t={t}");
        }
    }

    /// The SNR of `got` against `want` in dB.
    fn snr_db(want: &[C32], got: &[C32]) -> f64 {
        let (mut sig, mut err) = (0f64, 0f64);
        for (w, g) in want.iter().zip(got) {
            sig += f64::from(w.re).powi(2) + f64::from(w.im).powi(2);
            err += f64::from(w.re - g.re).powi(2) + f64::from(w.im - g.im).powi(2);
        }
        10.0 * (sig / err).log10()
    }

    /// Both sources of twiddles (the static span tables with the feature `fft-tables`, the plan's
    /// table always) at every size and table stride, against the fallback.
    #[test]
    fn the_plan_twiddles_and_the_span_tables_give_the_transform() {
        let big = FftPlan::<C32, _>::new(4096, std::vec![C32::default(); 2048]).unwrap();
        for log2 in 4..=16 {
            let n = 1usize << log2;
            let own = std::vec![C32::default(); n / 2];
            let own = FftPlan::<C32, _>::new(n, own).unwrap();
            let x: Vec<C32> = noise(log2, 2 * n)
                .chunks_exact(2)
                .map(|v| C32::new(v[0], v[1]))
                .collect();
            for dir in [Direction::Forward, Direction::Inverse] {
                let mut want = x.clone();
                Fallback
                    .fft(&own, &mut want, dir, Scaling::Default)
                    .unwrap();
                let mut strides = std::vec![(own.twiddles().to_vec(), 1)];
                if n <= 4096 {
                    let table = big.twiddles().iter().step_by(4096 / n).copied().collect();
                    strides.push((table, 4096 / n));
                    strides.push((big.twiddles().to_vec(), 4096 / n));
                }
                for (table, stride) in &strides {
                    // A strided table is passed whole, as `from_static_4096` does.
                    let stride = if table.len() == n / 2 { 1 } else { *stride };
                    let tw: &[f32] = bytemuck::cast_slice(table);
                    for tables in [false, true] {
                        let mut got = x.clone();
                        let data: &mut [f32] = bytemuck::cast_slice_mut(&mut got);
                        let inverse = dir == Direction::Inverse;
                        dispatch!(simd => fft_c32_of::<_, 0>(simd, data, tw, stride, inverse, tables));
                        let snr = snr_db(&want, &got);
                        assert!(
                            snr > 120.0,
                            "n={n} {dir:?} stride {stride} tables {tables}: {snr:.1} dB"
                        );
                    }
                }
            }
        }
    }
}
