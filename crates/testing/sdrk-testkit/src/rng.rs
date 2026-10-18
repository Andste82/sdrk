//! PCG32 (PCG-XSH-RR 64/32) pseudo-random numbers.
//!
//! Deterministic, tiny and identical on every target. The algorithm follows
//! `pcg_basic.c` of the PCG reference implementation by Melissa O'Neill.

const MULT: u64 = 6_364_136_223_846_793_005;

/// A PCG32 generator.
///
/// ```
/// use sdrk_testkit::rng::Pcg32;
/// let mut a = Pcg32::new(1, 2);
/// let mut b = Pcg32::new(1, 2);
/// assert_eq!(a.next_u32(), b.next_u32());
/// assert!(a.next_f64() < 1.0);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Creates a generator from a seed and a stream number (`pcg32_srandom_r`).
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(seed);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
    }

    /// Next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.step();
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Next 64 random bits; the high half is drawn first.
    pub fn next_u64(&mut self) -> u64 {
        let hi = u64::from(self.next_u32());
        let lo = u64::from(self.next_u32());
        (hi << 32) | lo
    }

    /// Uniform value in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Unbiased value in `0..n` (Lemire's multiply-and-reject method).
    /// Returns 0 for `n == 0`.
    ///
    /// ```
    /// use sdrk_testkit::rng::Pcg32;
    /// let mut rng = Pcg32::new(7, 7);
    /// assert!((0..1000).all(|_| rng.below(10) < 10));
    /// ```
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let mut m = u64::from(self.next_u32()) * u64::from(n);
        let mut low = m as u32;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = u64::from(self.next_u32()) * u64::from(n);
                low = m as u32;
            }
        }
        (m >> 32) as u32
    }

    /// Standard normal value (Box-Muller, cosine branch only). Always consumes
    /// exactly four `next_u32` draws.
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2)
    }

    /// Skips `delta` calls of [`Pcg32::next_u32`] in `O(log delta)` steps
    /// (`pcg_advance_lcg_64`). Lets a source be a pure function of the sample
    /// index.
    ///
    /// ```
    /// use sdrk_testkit::rng::Pcg32;
    /// let mut a = Pcg32::new(3, 4);
    /// let mut b = a.clone();
    /// for _ in 0..1000 { a.next_u32(); }
    /// b.advance(1000);
    /// assert_eq!(a, b);
    /// ```
    pub fn advance(&mut self, mut delta: u64) {
        let mut cur_mult = MULT;
        let mut cur_plus = self.inc;
        let mut acc_mult: u64 = 1;
        let mut acc_plus: u64 = 0;
        while delta > 0 {
            if delta & 1 == 1 {
                acc_mult = acc_mult.wrapping_mul(cur_mult);
                acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
            }
            cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
            cur_mult = cur_mult.wrapping_mul(cur_mult);
            delta >>= 1;
        }
        self.state = acc_mult.wrapping_mul(self.state).wrapping_add(acc_plus);
    }
}
