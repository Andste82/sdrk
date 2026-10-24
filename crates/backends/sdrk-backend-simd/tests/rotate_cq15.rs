//! `rotate` `CQ15`: bit-exact with the fallback for every length, including the zero-padded
//! last vector, and for every kind of increment.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::{CQ15, Fallback};

#[test]
fn every_length_is_bit_exact_with_the_fallback() {
    let mut rng = Rng::new(50);
    for n in 0..=70 {
        for inc in [0u32, 1, 3 << 30, 0x0123_4567, rng.next_u32()] {
            let x = cq15_noise(&mut rng, n);
            let mut a = Nco::from_increment(inc);
            a.set_phase(rng.next_u32());
            let mut b = a;
            let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
            Fallback.rotate(&mut a, &x, &mut want).unwrap();
            Simd.rotate(&mut b, &x, &mut got).unwrap();
            assert_eq!(want, got, "n={n} inc={inc:#x}");
            assert_eq!(a, b);
        }
    }
}
