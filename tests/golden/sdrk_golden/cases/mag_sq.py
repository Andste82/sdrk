"""Cases for the mag_sq kernel."""

import numpy as np

from .. import q15
from ..registry import Array, case
from .fir import noise


@case(kernel="mag_sq", name="c32_4096", seed=60)
def c32_4096(rng):
    x = noise(rng, 4096, 0.5).astype(np.complex64)
    re, im = x.real.astype(np.float64), x.imag.astype(np.float64)
    return dict(
        params={},
        arrays={"input": Array("cf32", x), "expected": Array("f64", re * re + im * im)},
        tolerance={"class": "tolerance", "snr_db_min": 140.0, "skip": 0},
        notes="re^2 + im^2 in float64 from the f32 input",
    )


@case(kernel="mag_sq", name="cq15_4096", seed=61)
def cq15_4096(rng):
    x = noise(rng, 4096, 0.5)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    corners = [(-32768, -32768), (32767, 32767), (-32768, 0), (0, 32767), (0, 0), (1, 1), (181, 181), (-182, 181)]
    for i, (a, b) in enumerate(corners):
        xr[i], xi[i] = a, b
    y = q15.sat16(q15.round_shift(xr * xr + xi * xi, 15))
    return dict(
        params={},
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("i16", y)},
        tolerance={"class": "bit_exact"},
        notes="sat16(round_shift(re^2 + im^2, 15)); the first eight samples are corner values, full scale saturates",
    )
