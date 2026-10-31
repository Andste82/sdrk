"""Cases for the real_to_complex kernel (M23): ``u[n] = x[n] (-j)^n``, then the halfband decimator."""

import numpy as np
import scipy.signal

from .. import design, q15, scalar_math
from ..registry import Array, case

N = 63
LEN = 16384
BETA60 = float(scipy.signal.kaiser_beta(60.0))


def real_input(rng):
    """Real noise sigma 0.2 plus two tones at 0.11 fs and 0.37 fs with amplitude 0.3 (float64)."""
    n = np.arange(LEN)
    x = rng.standard_normal(LEN) * 0.2
    x += 0.3 * scalar_math.cos(2.0 * np.pi * 0.11 * n + 0.3)
    x += 0.3 * scalar_math.cos(2.0 * np.pi * 0.37 * n + 1.1)
    return x


def shift_f64(x):
    """``x[n] (-j)^n`` as complex128."""
    pattern = np.array([1.0, -1.0j, -1.0, 1.0j])
    return x.astype(np.float64) * pattern[np.arange(len(x)) % 4]


def shift_q15(x):
    """The integer shift: ``(x, 0), (0, -x), (-x, 0), (0, x)`` for ``n mod 4 = 0..3``, the negation saturating
    (``-(-32768) = 32767``). Returns (re, im) as int64 arrays."""
    x = np.asarray(x, dtype=np.int64)
    neg = q15.sat16(-x)
    n4 = np.arange(len(x)) % 4
    re = np.where(n4 == 0, x, np.where(n4 == 2, neg, 0))
    im = np.where(n4 == 1, neg, np.where(n4 == 3, x, 0))
    return re.astype(np.int64), im.astype(np.int64)


def decimate_q15(v, h):
    full = np.convolve(np.asarray(v, dtype=np.int64), np.asarray(h, dtype=np.int64))[: len(v)][::2]
    return q15.sat16(q15.round_shift(full, 15))


@case(kernel="real_to_complex", name="n63_c32_16384", seed=434)
def n63_c32(rng):
    h = design.halfband_taps(N, BETA60).astype(np.float32)
    x = real_input(rng).astype(np.float32)
    u = shift_f64(x)
    expected = scalar_math.fir_exact(h, u, 2)
    return dict(
        params={"taps": N, "attenuation_db": 60.0, "beta": BETA60},
        arrays={"input": Array("f32", x), "taps": Array("f32", h), "expected": Array("cf64", expected)},
        # The regression guard (accuracy policy step 3): the golden SNR of the f32 fallback
        # (142.1 dB) minus 1 dB.
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "snr_db_guard": 141.1, "skip": 0},
        notes="u = x (-j)^n in float64, y[m] = sum_k h[k] u[2 m - k] (exact sums of scalar_math.fir_exact) with the f32 taps of halfband_taps(63, "
        "kaiser_beta(60)); real noise sigma 0.2 plus tones at 0.11 fs and 0.37 fs with amplitude 0.3",
    )


@case(kernel="real_to_complex", name="n63_cq15_16384", seed=435)
def n63_cq15(rng):
    h = q15.from_float(design.halfband_taps(N, BETA60))
    x = q15.from_float(real_input(rng))
    # The saturated negation: -32768 at every phase of n mod 4.
    for start in (200, 1001, 5000):
        x[start : start + 8] = -32768
    re, im = shift_q15(x)
    return dict(
        params={"taps": N, "attenuation_db": 60.0, "beta": BETA60},
        arrays={
            "input": Array("i16", x),
            "taps": Array("i16", h),
            "expected": Array("ci16", q15.pack(decimate_q15(re, h), decimate_q15(im, h))),
        },
        tolerance={"class": "bit_exact"},
        notes="integer model: u = (x, 0), (0, -x), (-x, 0), (0, x) for n mod 4 = 0..3 with the saturating negation "
        "(-(-32768) = 32767), int64 convolution with the Q15 halfband taps, every second output, one rounding and "
        "saturation per component; the input holds runs of -32768 at 200, 1001 and 5000 (every phase of n mod 4)",
    )
