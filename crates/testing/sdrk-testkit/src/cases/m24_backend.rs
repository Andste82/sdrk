//! Suite case of the SIMD backend for the M24 kernel `stereo_matrix`, run through the facade:
//! `backend_vs_fallback_stereo_matrix`.
//!
//! Both sample types at the lengths of `m13` (the tails of every vector width), bit for bit
//! against the fallback; `CQ15` with the corner values of the saturation (`-32768`, `32767`)
//! in `s` and `d`. The case uses no files, so it also runs on the emulators of the
//! microcontroller targets, where the facade is the fallback.

use sdrk::Sdrk;
use sdrk_core::kernels::stereo::StereoMatrix;
use sdrk_core::{C32, CQ15, Fallback, Q15};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use super::m13::{fail, lengths};
use crate::arena::Arena;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

const CORNERS: [i16; 6] = [-32768, -32767, -1, 0, 32766, 32767];

fn corner(k: usize) -> Q15 {
    Q15(CORNERS.get(k % CORNERS.len()).copied().unwrap_or(0))
}

pub(super) fn backend_vs_fallback_stereo_matrix(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_stereo_matrix";
    let mut rng = Pcg32::new(2410, 1);
    for n in lengths(arena.remaining(), 3 * 8 + 8) {
        let mut c = arena.carver();
        let x = c.take::<C32>(n)?;
        let (want, got) = (c.take::<C32>(n)?, c.take::<C32>(n)?);
        for s in x.iter_mut() {
            *s = random_c32(&mut rng, 0.9);
        }
        StereoMatrix::<C32>::stereo_matrix(&Fallback, x, want)?;
        StereoMatrix::<C32>::stereo_matrix(&Sdrk, x, got)?;
        ensure(bit_exact(want, got).is_none(), || {
            fail(CASE, "c32 differs from the fallback", n)
        })?;
    }
    for n in lengths(arena.remaining(), 3 * 4 + 4) {
        let mut c = arena.carver();
        let x = c.take::<CQ15>(n)?;
        let (want, got) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
        for (k, s) in x.iter_mut().enumerate() {
            *s = if k % 3 == 0 {
                CQ15 {
                    re: corner(k / 3),
                    im: corner(k / 3 + k / 18),
                }
            } else {
                random_cq15(&mut rng, 1.0)
            };
        }
        StereoMatrix::<CQ15>::stereo_matrix(&Fallback, x, want)?;
        StereoMatrix::<CQ15>::stereo_matrix(&Sdrk, x, got)?;
        ensure(bit_exact(want, got).is_none(), || {
            fail(CASE, "cq15 differs from the fallback", n)
        })?;
    }
    Ok(())
}
