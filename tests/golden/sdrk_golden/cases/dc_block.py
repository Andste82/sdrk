"""Cases for the dc_block kernel (one-pole DC blocker on complex signals)."""

import numpy as np
import scipy.signal

from .. import q15
from ..registry import Array, case
from .fir import noise

DC = 0.3 + 0.2j


@case(kernel="dc_block", name="r0p999_c32_8192", seed=52)
def r0p999_c32(rng):
    x = (noise(rng, 8192, 0.2) + DC).astype(np.complex64)
    r = float(np.float32(0.999))
    expected = scipy.signal.lfilter([1.0, -1.0], [1.0, -r], x.astype(np.complex128))
    return dict(
        params={"r": r},
        arrays={"input": Array("cf32", x), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 100.0, "skip": 0},
        notes="y[n] = x[n] - x[n-1] + r y[n-1] per component, r cast to f32, DC offset 0.3 + 0.2j",
    )


def block_q15(v, r):
    x_prev = y_prev = 0
    out = np.zeros(len(v), dtype=np.int64)
    for i, x in enumerate(v.tolist()):
        y = int(q15.sat16((x - x_prev) + q15.round_shift(r * y_prev, 15)))
        out[i] = y
        x_prev, y_prev = x, y
    return out


@case(kernel="dc_block", name="r0p999_cq15_8192", seed=53)
def r0p999_cq15(rng):
    x = noise(rng, 8192, 0.2) + DC
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    r = int(q15.from_float(0.999))
    return dict(
        params={"r_raw": r},
        arrays={
            "input": Array("ci16", q15.pack(xr, xi)),
            "expected": Array("ci16", q15.pack(block_q15(xr, r), block_q15(xi, r))),
        },
        tolerance={"class": "bit_exact"},
        notes="exact Q15 recursion per component: sat16((x - x_prev) + round_shift(r y_prev, 15))",
    )
