"""Cases for the dot kernels (sliding dot products of a window with real taps)."""

import numpy as np

from .. import q15
from ..registry import Array, case

ROWS = 256


def sliding(x, taps):
    """Dot product of every window ``x[i : i + n]`` with ``taps``, i = 0 .. ROWS - 1 (float64)."""
    n = len(taps)
    return np.array([np.dot(x[i : i + n], taps) for i in range(ROWS)])


def c32_case(rng, n):
    x = ((rng.standard_normal(n + ROWS - 1) + 1j * rng.standard_normal(n + ROWS - 1)) * 0.3).astype(np.complex64)
    taps = rng.uniform(-0.5, 0.5, n).astype(np.float32)
    expected = sliding(x.astype(np.complex128), taps.astype(np.float64))
    return dict(
        params={"taps": n, "rows": ROWS},
        arrays={"input": Array("cf32", x), "taps": Array("f32", taps), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 120.0, "skip": 0},
        notes="expected[i] = sum_k input[i + k] * taps[k] in float64, 256 windows",
    )


def f32_case(rng, n):
    x = (rng.standard_normal(n + ROWS - 1) * 0.3).astype(np.float32)
    taps = rng.uniform(-0.5, 0.5, n).astype(np.float32)
    expected = sliding(x.astype(np.float64), taps.astype(np.float64))
    return dict(
        params={"taps": n, "rows": ROWS},
        arrays={"input": Array("f32", x), "taps": Array("f32", taps), "expected": Array("f64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 120.0, "skip": 0},
        notes="expected[i] = sum_k input[i + k] * taps[k] in float64, 256 windows",
    )


def q15_dot(v, h):
    out = np.zeros(ROWS, dtype=np.int64)
    h = np.asarray(h, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    n = len(h)
    for i in range(ROWS):
        acc = int(np.sum(v[i : i + n] * h))
        out[i] = int(q15.sat16(q15.round_shift(acc, 15)))
    return out


def cq15_case(rng, n):
    x = (rng.standard_normal(n + ROWS - 1) + 1j * rng.standard_normal(n + ROWS - 1)) * 0.3
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    taps = q15.from_float(rng.uniform(-0.5, 0.5, n))
    expected = q15.pack(q15_dot(xr, taps), q15_dot(xi, taps))
    return dict(
        params={"taps": n, "rows": ROWS},
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "taps": Array("i16", taps), "expected": Array("ci16", expected)},
        tolerance={"class": "bit_exact"},
        notes="Q30 accumulation in int64, one rounding and saturation per component, 256 windows",
    )


def q15_case(rng, n):
    x = q15.from_float(rng.standard_normal(n + ROWS - 1) * 0.3)
    taps = q15.from_float(rng.uniform(-0.5, 0.5, n))
    return dict(
        params={"taps": n, "rows": ROWS},
        arrays={"input": Array("i16", x), "taps": Array("i16", taps), "expected": Array("i16", q15_dot(x, taps))},
        tolerance={"class": "bit_exact"},
        notes="Q30 accumulation in int64, one rounding and saturation, 256 windows",
    )


def _register(kernel, name, seed, fn, n):
    @case(kernel=kernel, name=name, seed=seed)
    def _case(rng):
        return fn(rng, n)


for _n, _seed in ((7, 300), (64, 301), (1023, 302)):
    _register("dot", f"c32_{_n}", _seed, c32_case, _n)
for _n, _seed in ((7, 303), (64, 304), (1023, 305)):
    _register("dot", f"cq15_{_n}", _seed, cq15_case, _n)
for _n, _seed in ((7, 306), (1023, 307)):
    _register("dot", f"f32_{_n}", _seed, f32_case, _n)
for _n, _seed in ((7, 308), (1023, 309)):
    _register("dot", f"q15_{_n}", _seed, q15_case, _n)
