//! The RDS reference signal of M41 in Rust: a re-synthesis of the `NumPy` modulator
//! `tests/golden/sdrk_golden/cases/rds.py`, operation for operation.
//!
//! Rust tests must not call Python, so the long end-to-end input (6 s and more) is produced
//! here. The model is that of the module docstring of `rds.py` (M41 section 5):
//!
//! 1. The carousel of [`carousel_group`]: PI `0xD123`, PTY 10, TP 0, alternately group 0A (PS
//!    `SDRK TST`, block B with TA 0, M/S 1, DI 0, block C `0xE0E0`) and group 2A (radiotext
//!    `Hello from sdrk!`, a carriage return, spaces to 64 characters, A/B flag 0); period 32
//!    groups.
//! 2. Each 16-bit word is followed by its checkword ([`checkword`], computed here, independent
//!    of `sdrk_proto`), 26 bits per block, most significant bit first.
//! 3. Differential coding `t_k = t_{k-1} xor d_k`, `t_{-1} = 0`, amplitude `a_k = 2 t_k - 1`.
//! 4. The biphase pulse `p = h(x + 1/4) - h(x - 1/4)` with
//!    `h(x) = 8 cos(4 pi x) / (pi (1 - 64 x^2))` (`h(+-1/8) = 2`), `x = t / Tb`, cut to
//!    `-2 <= x < 2`, divided by its largest magnitude on the sample grid; bit `k` is centred at
//!    `k Tb`. At 240 kHz the grid is `x = n / 3840` and sample `m` sees the four bits
//!    `k = floor(19 m / 3840) - 1 + j` at `n = 19 m - 3840 k`.
//! 5. The multiplex `0.45 (L + R) + 0.45 (L - R) sin(2 phi) + 0.1 sin(phi) + 0.04 b sin(3 phi)`
//!    with `L` 1 kHz, `R` 2 kHz and the pilot phase `phi` of 19 kHz, every sine from a table
//!    of `sin(2 pi r / 240)`.
//! 6. FM with 75 kHz per unit: the multiplex held for two samples at 480 kHz, the phase
//!    accumulated from 0 with the increment `2 pi 75000 / 480000 mpx`, the output
//!    `0.5 exp(j phase)`. Noise is added by the caller (`channel::Awgn`), [`to_cs16`] rounds
//!    with the R7 rule.
//!
//! The only differences to the Python files come from `libm` against the C library in the
//! table values and the FM phase (an ulp now and then): the test `golden_m41` compares the
//! model with the committed case `rds/reference_short` (float values within its
//! `max_abs_err`, the `cs16` samples within `rf_max_lsb`).
//!
//! The pulse table (15360 `f64`) and the sine table (240 `f64`) live in caller storage
//! ([`TABLE_LEN`] values), so the model is heap-free and small.
//!
//! ```
//! use num_complex::Complex64;
//! use sdrk_testkit::rds::{TABLE_LEN, RdsReference};
//!
//! let mut table = vec![0.0; TABLE_LEN];
//! let mut sig = RdsReference::new(&mut table[..], 0.04)?;
//! let mut rf = [Complex64::new(0.0, 0.0); 64];
//! sig.fill(0, &mut rf)?;
//! assert_eq!(rf[0], Complex64::new(0.5, 0.0));
//! assert!(sig.fill(0, &mut rf).is_err()); // calls must be consecutive
//! # Ok::<(), sdrk_core::Error>(())
//! ```

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk_core::{CQ15, Error, Q15};

/// Multiplex sample rate in Hz.
pub const MPX_RATE_HZ: u64 = 240_000;
/// RF sample rate in Hz.
pub const RF_RATE_HZ: u64 = 480_000;
/// Programme identification of the carousel.
pub const PI_CODE: u16 = 0xD123;
/// Programme type of the carousel.
pub const PTY: u8 = 10;
/// Programme service name of the carousel.
pub const PS: &[u8; 8] = b"SDRK TST";
/// Radiotext of the carousel (without the carriage return and padding).
pub const RT_TEXT: &[u8] = b"Hello from sdrk!";
/// Groups of one carousel period.
pub const CAROUSEL_PERIOD: u64 = 32;
/// Bits of one carousel period.
pub const PERIOD_BITS: u64 = CAROUSEL_PERIOD * 104;
/// The RDS level of the multiplex (4 %).
pub const RDS_LEVEL: f64 = 0.04;
/// Peak frequency deviation per unit of the multiplex in Hz.
pub const DEVIATION_HZ: f64 = 75_000.0;
/// Amplitude of the RF carrier (0.5 of full scale).
pub const RF_AMPLITUDE: f64 = 0.5;
/// Numerator and denominator of `t / Tb = m NUM / DEN` at 240 kHz.
pub const GRID_NUM: i64 = 19;
/// See [`GRID_NUM`].
pub const GRID_DEN: i64 = 3840;
/// Entries of the pulse table: `x = n / 3840` for `-7680 <= n < 7680`.
pub const PULSE_LEN: usize = 4 * GRID_DEN as usize;
/// Values of the caller storage of [`RdsReference`]: the pulse table, then the sine table.
pub const TABLE_LEN: usize = PULSE_LEN + SIN_LEN;

/// The offset words A, B, C, C', D.
pub const OFFSET_WORDS: [u16; 5] = [0x0FC, 0x198, 0x168, 0x350, 0x1B4];

const GENERATOR: u32 = 0x5B9;
const MAIN_LEVEL: f64 = 0.45;
const PILOT_LEVEL: f64 = 0.1;
const SIN_LEN: usize = 240;
const PERIOD_WORDS: usize = PERIOD_BITS.div_ceil(64) as usize;

/// The checkword of `info` with the offset word `offset_word`: `info(x) x^10 mod g(x)` plus the
/// offset word (bitwise long division).
///
/// ```
/// use sdrk_testkit::rds::{checkword, OFFSET_WORDS};
/// assert_eq!(checkword(0, OFFSET_WORDS[0]), 0x0FC);
/// assert_eq!(checkword(1, 0), 0x1B9);
/// ```
pub fn checkword(info: u16, offset_word: u16) -> u16 {
    let mut v = u32::from(info) << 10;
    for bit in (10..26).rev() {
        if (v >> bit) & 1 != 0 {
            v ^= GENERATOR << (bit - 10);
        }
    }
    (v as u16) ^ offset_word
}

/// The 64 radiotext bytes: the text, a carriage return, spaces.
fn radiotext(i: usize) -> u8 {
    let n = RT_TEXT.len();
    match i.cmp(&n) {
        core::cmp::Ordering::Less => RT_TEXT.get(i).copied().unwrap_or(b' '),
        core::cmp::Ordering::Equal => 0x0D,
        core::cmp::Ordering::Greater => b' ',
    }
}

fn ps_char(i: usize) -> u8 {
    PS.get(i).copied().unwrap_or(b' ')
}

/// The four information words of group `g` of the carousel (`g` counted from 0, endless).
///
/// ```
/// use sdrk_testkit::rds::carousel_group;
/// assert_eq!(carousel_group(0), [0xD123, 0x0148, 0xE0E0, u16::from_be_bytes(*b"SD")]);
/// assert_eq!(carousel_group(1)[1], 0x2140);
/// assert_eq!(carousel_group(33), carousel_group(1));
/// ```
pub fn carousel_group(g: u64) -> [u16; 4] {
    let common = (u16::from(PTY) << 5) & 0x03E0;
    if g.is_multiple_of(2) {
        let seg = ((g / 2) % 4) as usize;
        let b = common | (1 << 3) | seg as u16;
        let d = u16::from_be_bytes([ps_char(2 * seg), ps_char(2 * seg + 1)]);
        [PI_CODE, b, 0xE0E0, d]
    } else {
        let seg = ((g / 2) % 16) as usize;
        let b = (2 << 12) | common | seg as u16;
        let c = u16::from_be_bytes([radiotext(4 * seg), radiotext(4 * seg + 1)]);
        let d = u16::from_be_bytes([radiotext(4 * seg + 2), radiotext(4 * seg + 3)]);
        [PI_CODE, b, c, d]
    }
}

/// The offset word of block `i` (0 to 3) of a carousel group (all groups are version A).
fn offset_of(i: usize) -> u16 {
    let word = match i {
        0 => OFFSET_WORDS.first(),
        1 => OFFSET_WORDS.get(1),
        2 => OFFSET_WORDS.get(2),
        _ => OFFSET_WORDS.get(4),
    };
    word.copied().unwrap_or(0)
}

/// The data bit `d_k` of the carousel (before differential coding).
///
/// ```
/// use sdrk_testkit::rds::data_bit;
/// // 0xD123 = 1101 0001 ...
/// let first: Vec<u8> = (0..4).map(data_bit).collect();
/// assert_eq!(first, [1, 1, 0, 1]);
/// ```
pub fn data_bit(k: u64) -> u8 {
    let g = k / 104;
    let within = (k % 104) as usize;
    let (block, bit) = (within / 26, within % 26);
    let info = carousel_group(g).get(block).copied().unwrap_or(0);
    let word = (u32::from(info) << 10) | u32::from(checkword(info, offset_of(block)));
    ((word >> (25 - bit)) & 1) as u8
}

/// `h(x)` at `x = n / den` (see the module documentation).
fn h(n: i64, den: i64) -> f64 {
    let d = den * den - 64 * n * n;
    if d == 0 {
        return 2.0;
    }
    let r = (2 * n).rem_euclid(den);
    let c = libm::cos(2.0 * PI * r as f64 / den as f64);
    8.0 * c * (den * den) as f64 / (PI * d as f64)
}

/// The reference signal as a stream (see the module documentation).
#[derive(Debug, Clone)]
pub struct RdsReference<S> {
    table: S,
    peak: f64,
    tx: [u64; PERIOD_WORDS],
    period_parity: u8,
    rds_level: f64,
    phase: f64,
    next_rf: u64,
}

impl<S: AsRef<[f64]> + AsMut<[f64]>> RdsReference<S> {
    /// The signal with the RDS level `rds_level` (0.04 for the reference, 0.0 for the same
    /// multiplex without RDS); `table` holds [`TABLE_LEN`] values and is filled here.
    ///
    /// # Errors
    /// `InvalidParameter` if `table` has the wrong length or `rds_level` is not finite.
    pub fn new(table: S, rds_level: f64) -> Result<Self, Error> {
        if table.as_ref().len() != TABLE_LEN {
            return Err(Error::InvalidParameter(
                "the rds table must hold TABLE_LEN values",
            ));
        }
        if !rds_level.is_finite() {
            return Err(Error::InvalidParameter("the rds level must be finite"));
        }
        let mut s = Self {
            table,
            peak: 0.0,
            tx: [0; PERIOD_WORDS],
            period_parity: 0,
            rds_level,
            phase: 0.0,
            next_rf: 0,
        };
        let q = GRID_DEN / 4;
        let mut peak = 0.0f64;
        let (pulse, sines) = s.table.as_mut().split_at_mut(PULSE_LEN);
        for (i, v) in pulse.iter_mut().enumerate() {
            let n = i as i64 - 2 * GRID_DEN;
            *v = h(n + q, GRID_DEN) - h(n - q, GRID_DEN);
            // `max(abs(v))` of Python keeps the first of equal values; the value is the same.
            if libm::fabs(*v) > peak {
                peak = libm::fabs(*v);
            }
        }
        for v in pulse.iter_mut() {
            *v /= peak;
        }
        for (r, v) in sines.iter_mut().enumerate() {
            *v = libm::sin(2.0 * PI * r as f64 / SIN_LEN as f64);
        }
        s.peak = peak;
        let mut t = 0u8;
        for k in 0..PERIOD_BITS {
            t ^= data_bit(k);
            if let Some(w) = s.tx.get_mut((k / 64) as usize) {
                *w |= u64::from(t) << (k % 64);
            }
        }
        s.period_parity = t;
        Ok(s)
    }

    /// The largest magnitude of the pulse on the grid before normalization.
    pub fn pulse_peak(&self) -> f64 {
        self.peak
    }

    /// The normalized pulse at `x = n / 3840` (`-7680 <= n < 7680`, else 0).
    pub fn pulse(&self, n: i64) -> f64 {
        usize::try_from(n + 2 * GRID_DEN)
            .ok()
            .filter(|&i| i < PULSE_LEN)
            .and_then(|i| self.table.as_ref().get(i))
            .copied()
            .unwrap_or(0.0)
    }

    /// The transmitted (differentially coded) bit `t_k`.
    pub fn tx_bit(&self, k: u64) -> u8 {
        let within = k % PERIOD_BITS;
        let periods_odd = ((k / PERIOD_BITS) & 1) as u8;
        let bit = self
            .tx
            .get((within / 64) as usize)
            .map_or(0, |w| ((w >> (within % 64)) & 1) as u8);
        bit ^ (periods_odd & self.period_parity)
    }

    /// The RDS baseband `b(m)` at the multiplex sample `m`.
    pub fn baseband(&self, m: u64) -> f64 {
        let m = m as i64;
        let k0 = (m * GRID_NUM).div_euclid(GRID_DEN) - 1;
        let mut out = 0.0;
        for j in 0..4 {
            let k = k0 + j;
            let term = if k >= 0 {
                let a = 2.0 * f64::from(self.tx_bit(k as u64)) - 1.0;
                a * self.pulse(m * GRID_NUM - GRID_DEN * k)
            } else {
                0.0
            };
            out += term;
        }
        out
    }

    fn tone(&self, k: u64, m: u64) -> f64 {
        let r = (k.wrapping_mul(m) % SIN_LEN as u64) as usize;
        self.table
            .as_ref()
            .get(PULSE_LEN + r)
            .copied()
            .unwrap_or(0.0)
    }

    /// The multiplex at the sample `m` (240 kHz).
    pub fn mpx(&self, m: u64) -> f64 {
        // Exact integer turns: (f / 1000) m mod 240 with m reduced first (no overflow).
        let mr = m % SIN_LEN as u64;
        let left = self.tone(1, mr);
        let right = self.tone(2, mr);
        let s1 = self.tone(19, mr);
        let s2 = self.tone(38, mr);
        let s3 = self.tone(57, mr);
        let b = self.baseband(m);
        MAIN_LEVEL * (left + right)
            + MAIN_LEVEL * (left - right) * s2
            + PILOT_LEVEL * s1
            + self.rds_level * b * s3
    }

    /// The index of the next RF sample [`RdsReference::fill`] expects.
    pub fn next_index(&self) -> u64 {
        self.next_rf
    }

    /// Starts again at RF sample 0.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.next_rf = 0;
    }

    /// Writes the RF samples `start ..` (480 kHz, amplitude 0.5, no noise) into `out`.
    ///
    /// # Errors
    /// `InvalidParameter` if `start` is not [`RdsReference::next_index`] (the FM phase is
    /// accumulated, so the calls must be consecutive; [`RdsReference::reset`] starts again).
    pub fn fill(&mut self, start: u64, out: &mut [Complex64]) -> Result<(), Error> {
        if start != self.next_rf {
            return Err(Error::InvalidParameter(
                "the rds reference is generated in consecutive calls",
            ));
        }
        let k = 2.0 * PI * DEVIATION_HZ / RF_RATE_HZ as f64;
        let mut cached = (u64::MAX, 0.0);
        for (i, o) in out.iter_mut().enumerate() {
            let j = start + i as u64;
            let m = j / 2;
            if cached.0 != m {
                cached = (m, self.mpx(m));
            }
            *o = Complex64::new(
                RF_AMPLITUDE * libm::cos(self.phase),
                RF_AMPLITUDE * libm::sin(self.phase),
            );
            self.phase += cached.1 * k;
        }
        self.next_rf = start + out.len() as u64;
        Ok(())
    }
}

/// One component rounded with the R7 rule: `floor(32768 v + 0.5)` clamped, NaN 0.
fn q15_of(v: f64) -> Q15 {
    if v.is_nan() {
        return Q15(0);
    }
    Q15(libm::floor(v * 32768.0 + 0.5).clamp(-32768.0, 32767.0) as i16)
}

/// A sample rounded to `cs16` with the R7 rule (the rounding of `rds.py`).
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::rds::to_cs16;
/// let q = to_cs16(Complex64::new(0.5, -2.0));
/// assert_eq!((q.re.0, q.im.0), (16384, -32768));
/// ```
pub fn to_cs16(x: Complex64) -> CQ15 {
    CQ15 {
        re: q15_of(x.re),
        im: q15_of(x.im),
    }
}
