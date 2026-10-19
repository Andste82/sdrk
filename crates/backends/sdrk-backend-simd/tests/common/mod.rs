//! Shared helpers of the backend tests: a small PRNG and data generators.
#![allow(dead_code, clippy::indexing_slicing, clippy::unwrap_used)]

use sdrk_core::{C32, CQ15, Q15};

/// PCG32, the same generator as the testkit (seeded, deterministic).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(0);
        r.next_u32();
        r.0 = r.0.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        (f64::from(self.next_u32()) / 2_147_483_648.0 - 1.0) as f32
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n.max(1)
    }
}

pub fn c32_noise(rng: &mut Rng, n: usize, amp: f32) -> Vec<C32> {
    (0..n)
        .map(|_| C32::new(rng.unit() * amp, rng.unit() * amp))
        .collect()
}

pub fn cq15_noise(rng: &mut Rng, n: usize) -> Vec<CQ15> {
    (0..n)
        .map(|_| CQ15 {
            re: Q15(rng.next_u32() as i16),
            im: Q15(rng.next_u32() as i16),
        })
        .collect()
}

/// Lengths that exercise every tail of the vector loops.
pub const LENGTHS: [usize; 12] = [0, 1, 2, 3, 7, 15, 16, 17, 31, 255, 257, 4096];

/// Splits `n` into random chunk sizes (each at least 1).
pub fn chunks(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut left = n;
    while left > 0 {
        let c = (1 + rng.below(70)).min(left);
        out.push(c);
        left -= c;
    }
    out
}

/// Bit pattern of an f32 slice (so NaN payloads compare too).
pub fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}
