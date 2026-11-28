//! The target features of the hot loops of some fallback kernels (decision D2840, internal
//! since D3010).
//!
//! The fallback is compiled with the target features of the build. Until D3007 the SIMD
//! backend's run-time dispatch build ran these kernels through [`FallbackAt`] with the hot loops
//! compiled for AVX2; that route is removed, and the crate no longer exports these types.
//! [`Fallback`](crate::Fallback) runs `FallbackAt(Native)`, the build's own out-of-line
//! functions. The generic forms stay because removing the parameter changed the code that
//! LLVM builds: `fir_cq15_fallback` took 0.2 % to 0.6 % more cycles per sample at `avx2`,
//! `avx512` and `neon` (CI run 46103932474), and no fallback may get slower on AArch64.
//!
//! The kernels: `fir` `CQ15`, `fir_interpolate` `CQ15` and `fir_interpolate_real` `Q15`.

/// Where the hot loops of the kernels of [`FallbackAt`] are compiled, and which of the forms
/// that depend on the target features they take. The forms give the same bits on every target
/// (exact integer sums); only their speed depends on the target.
pub trait Target: Copy {
    /// Whether this is [`Native`]: the hot loops are the build's own out-of-line functions,
    /// and [`Target::hot`] is not called.
    const NATIVE: bool;
    /// Whether the code runs with AVX2 (and the rest of `x86-64-v3`): the forms that a build
    /// with `target_feature = "avx2"` takes.
    const AVX2: bool;
    /// Runs `f`, one hot loop, out of line in a function compiled with the target features of
    /// this target (the implementation passes an `#[inline(always)]` closure).
    fn hot<R>(self, f: impl FnOnce() -> R) -> R;
}

/// The target features of the build (what [`Fallback`](crate::Fallback) runs).
#[derive(Clone, Copy, Debug, Default)]
pub struct Native;

impl Target for Native {
    const NATIVE: bool = true;
    const AVX2: bool = cfg!(all(target_arch = "x86_64", target_feature = "avx2"));
    #[allow(
        clippy::inline_always,
        reason = "not called: the build's own functions run"
    )]
    #[inline(always)]
    fn hot<R>(self, f: impl FnOnce() -> R) -> R {
        f()
    }
}

/// The fallback of `fir` `CQ15`, `fir_interpolate` `CQ15` and `fir_interpolate_real` `Q15`
/// with its hot loops compiled for the target `T` (see the module documentation). The outputs
/// are those of [`Fallback`](crate::Fallback) bit for bit on every target.
#[derive(Clone, Copy, Debug, Default)]
pub struct FallbackAt<T>(pub T);

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    //! A target that takes the AVX2 forms and the out-of-line path of [`Target::hot`] gives the
    //! bits of [`Fallback`] on every target (on x86-64 below AVX2 these are other forms than the
    //! build's own).

    use std::vec;
    use std::vec::Vec;

    use super::{FallbackAt, Target};
    use crate::kernels::filter::{Fir, FirState};
    use crate::kernels::interp::{
        FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
    };
    use crate::{CQ15, Fallback, Q15};

    /// The AVX2 forms, the loops out of line through `hot`, compiled for the build.
    #[derive(Clone, Copy)]
    struct Forms;

    impl Target for Forms {
        const NATIVE: bool = false;
        const AVX2: bool = true;
        #[inline(never)]
        fn hot<R>(self, f: impl FnOnce() -> R) -> R {
            f()
        }
    }

    fn next(s: &mut u32) -> u32 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *s >> 8
    }

    fn q15(s: &mut u32) -> Q15 {
        match next(s) % 8 {
            0 => Q15(i16::MIN),
            1 => Q15(i16::MAX),
            _ => Q15(next(s) as i16),
        }
    }

    fn cq15(s: &mut u32) -> CQ15 {
        CQ15 {
            re: q15(s),
            im: q15(s),
        }
    }

    /// Call lengths that cover the history outputs, whole groups and their remainders.
    const CALLS: [usize; 6] = [0, 1, 7, 33, 200, 5];

    #[test]
    fn fir_cq15_matches_the_fallback() {
        let mut s = 1;
        for n in [1usize, 2, 7, 16, 31, 64, 129, 300] {
            let taps: Vec<Q15> = (0..n).map(|_| q15(&mut s)).collect();
            let mut a =
                FirState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::default(); n - 1]).unwrap();
            let mut b = FirState::<CQ15, _, _>::new(taps, vec![CQ15::default(); n - 1]).unwrap();
            for len in CALLS {
                let x: Vec<CQ15> = (0..len).map(|_| cq15(&mut s)).collect();
                let (mut ya, mut yb) = (vec![CQ15::default(); len], vec![CQ15::default(); len]);
                Fallback.fir(&mut a, &x, &mut ya).unwrap();
                FallbackAt(Forms).fir(&mut b, &x, &mut yb).unwrap();
                assert_eq!(ya, yb, "{n} taps, {len} samples");
            }
        }
    }

    #[test]
    fn fir_interpolate_matches_the_fallback() {
        let mut s = 2;
        for (n, l) in [
            (1usize, 1usize),
            (5, 2),
            (32, 4),
            (63, 3),
            (127, 8),
            (300, 2),
            (600, 3),
        ] {
            let taps: Vec<Q15> = (0..n).map(|_| q15(&mut s)).collect();
            let h = n.div_ceil(l);
            let ch = || vec![CQ15::default(); h];
            let rh = || vec![Q15(0); h];
            let mut ca = FirInterpState::<CQ15, _, _>::new(taps.clone(), ch(), l).unwrap();
            let mut cb = FirInterpState::<CQ15, _, _>::new(taps.clone(), ch(), l).unwrap();
            let mut ra = FirInterpRealState::<Q15, _, _>::new(taps.clone(), rh(), l).unwrap();
            let mut rb = FirInterpRealState::<Q15, _, _>::new(taps, rh(), l).unwrap();
            for (k, len) in CALLS.into_iter().enumerate() {
                // Output room that sometimes ends inside the outputs of a sample.
                let room = len * l - (k % 2) * (l / 2);
                let x: Vec<CQ15> = (0..len).map(|_| cq15(&mut s)).collect();
                let (mut ya, mut yb) = (vec![CQ15::default(); room], vec![CQ15::default(); room]);
                let pa = Fallback.fir_interpolate(&mut ca, &x, &mut ya).unwrap();
                let pb = FallbackAt(Forms)
                    .fir_interpolate(&mut cb, &x, &mut yb)
                    .unwrap();
                assert_eq!((pa.consumed, pa.produced), (pb.consumed, pb.produced));
                assert_eq!(ya, yb, "CQ15 {n} taps, L {l}, {len} samples");
                let x: Vec<Q15> = (0..len).map(|_| q15(&mut s)).collect();
                let (mut ya, mut yb) = (vec![Q15(0); room], vec![Q15(0); room]);
                let pa = Fallback.fir_interpolate_real(&mut ra, &x, &mut ya).unwrap();
                let pb = FallbackAt(Forms)
                    .fir_interpolate_real(&mut rb, &x, &mut yb)
                    .unwrap();
                assert_eq!((pa.consumed, pa.produced), (pb.consumed, pb.produced));
                assert_eq!(ya, yb, "Q15 {n} taps, L {l}, {len} samples");
            }
        }
    }
}
