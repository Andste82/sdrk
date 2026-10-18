"""Cases for the scale kernel."""

import numpy as np

from .. import q15
from ..registry import Array, case


@case(kernel="scale", name="c32_random_4096", seed=1)
def scale_c32(rng):
    x = ((rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) * 0.25).astype(np.complex64)
    gain = 0.75
    y = x.astype(np.complex128) * gain
    return dict(
        params={"gain": gain},
        arrays={"input": Array("cf32", x), "expected": Array("cf64", y)},
        tolerance={"class": "tolerance", "snr_db_min": 140.0, "skip": 0},
        notes="output[n] = input[n] * gain",
    )


@case(kernel="scale", name="cq15_random_4096", seed=5)
def scale_cq15(rng):
    re = q15.from_float(0.9 * rng.uniform(-1, 1, 4096))
    im = q15.from_float(0.9 * rng.uniform(-1, 1, 4096))
    gain_raw = 24576
    return dict(
        params={"gain_raw": gain_raw},
        arrays={
            "input": Array("ci16", q15.pack(re, im)),
            "expected": Array("ci16", q15.pack(q15.mul(re, gain_raw), q15.mul(im, gain_raw))),
        },
        tolerance={"class": "bit_exact"},
        notes="output[n] = input[n] * gain_raw / 32768 per component, Q15 rounding",
    )
