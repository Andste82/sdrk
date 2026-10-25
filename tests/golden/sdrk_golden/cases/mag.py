"""Cases for the mag kernel (envelope, the magnitude of complex samples)."""

import math

import numpy as np

from .. import q15
from ..registry import Array, case
from .fir import noise


def mag_q15_scalar(re, im):
    """Integer rule: the rounded square root of ``re^2 + im^2`` (a tie is impossible), not
    saturated. ``math.isqrt`` is the exact floor of the square root."""
    s = re * re + im * im
    r = math.isqrt(s)
    return r + (1 if s - r * r > r else 0)


@case(kernel="mag", name="c32_4096", seed=413)
def c32_4096(rng):
    x = noise(rng, 4096, 0.5)
    listed = [0, 1, -1, 1j, -1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    x[: len(listed)] = listed
    x = x.astype(np.complex64)
    expected = np.abs(x.astype(np.complex128))
    return dict(
        params={},
        arrays={"input": Array("cf32", x), "expected": Array("f64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 130.0, "skip": 0},
        notes="noise sigma 0.5 per component; the first nine samples are 0, +-1, +-1j and "
        "+-1 +-1j; expected np.abs of the f32 input in float64",
    )


@case(kernel="mag", name="cq15_4096", seed=414)
def cq15_4096(rng):
    x = noise(rng, 4096, 0.5)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    corners = [
        (-32768, -32768), (32767, 32767), (-32768, 32767), (32767, -32768),
        (-32768, 0), (0, -32768), (32767, 0), (0, 32767), (0, 0), (1, 0), (0, 1), (1, 1),
        (181, 181), (-182, 181), (23170, 23170), (23170, 23171), (3, 4), (-5, 12),
    ]
    for i, (a, b) in enumerate(corners):
        xr[i], xi[i] = a, b
    y = np.array([mag_q15_scalar(a, b) for a, b in zip(xr.tolist(), xi.tolist())], dtype=np.int64)
    return dict(
        params={},
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("i16", q15.sat16(y))},
        tolerance={"class": "bit_exact"},
        notes="sat16(r + (s - r^2 > r)) with s = re^2 + im^2 and r = math.isqrt(s); the first "
        "eighteen samples are corner values, the four full-scale corners saturate",
    )
