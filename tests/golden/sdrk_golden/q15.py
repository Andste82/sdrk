"""Independent integer model of the SDRk Q15 conventions (rule R7).

All functions work on ``numpy.int64`` arrays (or scalars) and mirror the names
of ``sdrk_core::num``.
"""

import numpy as np


def sat16(x):
    """Saturate to [-32768, 32767]."""
    return np.clip(x, -32768, 32767).astype(np.int64)


def round_shift(x, s):
    """Round half up, then arithmetic right shift (numpy >> on int64 is arithmetic)."""
    return x if s == 0 else (x + (1 << (s - 1))) >> s


def from_float(x):
    """float64 array -> int64 Q15 raw. NaN -> 0, +-inf -> +-limit, computed in f64."""
    x = np.asarray(x, dtype=np.float64)
    v = np.floor(np.where(np.isnan(x), 0.0, x) * 32768.0 + 0.5)
    v = np.where(np.isposinf(x), 32767.0, np.where(np.isneginf(x), -32768.0, v))
    v = np.clip(v, -32768.0, 32767.0)
    return sat16(v.astype(np.int64))


def mul(a, b):
    """Q15 multiply: sat16((a * b + (1 << 14)) >> 15)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return sat16(round_shift(a * b, 15))


def cmul(ar, ai, br, bi):
    """Complex Q15 multiply with one rounding and saturation per component."""
    ar, ai, br, bi = (np.asarray(v, dtype=np.int64) for v in (ar, ai, br, bi))
    return (
        sat16(round_shift(ar * br - ai * bi, 15)),
        sat16(round_shift(ar * bi + ai * br, 15)),
    )


def to_float(q):
    """Q15 raw -> float64 (exact)."""
    return np.asarray(q).astype(np.float64) / 32768.0


def pack(re, im):
    """Stack real and imaginary parts into an (n, 2) array (the ``ci16`` layout)."""
    return np.stack([np.asarray(re, dtype=np.int64), np.asarray(im, dtype=np.int64)], axis=1)


def self_test():
    """Check the R7 spot values; raises AssertionError on any deviation."""
    assert int(mul(-32768, -32768)) == 32767
    assert int(mul(-32768, 32767)) == -32767
    assert int(mul(1, 1)) == 0
    assert int(mul(16384, 16384)) == 8192
    f = from_float(np.array([1.0, -1.0, np.nan, np.inf, -np.inf, 1 / 65536, -1 / 65536]))
    assert f.tolist() == [32767, -32768, 0, 32767, -32768, 1, 0], f.tolist()
    assert int(round_shift(np.int64(-3), 1)) == -1
    assert int(round_shift(np.int64(5), 0)) == 5
    re, im = cmul(-32768, -32768, -32768, -32768)
    assert int(re) == 0 and int(im) == 32767
    assert float(to_float(np.int64(-32768))) == -1.0
