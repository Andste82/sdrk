"""Cases for the deemphasis kernel (single-pole low-pass on real audio)."""

import numpy as np
import scipy.signal

from .. import q15
from ..registry import Array, case

TAU = 50e-6
FS = 48000


def alpha_exact():
    return 1.0 - np.exp(-1.0 / (FS * TAU))


@case(kernel="deemphasis", name="tau50us_fs48k_f32_8192", seed=50)
def tau50us_f32(rng):
    x = (rng.standard_normal(8192) * 0.3).astype(np.float32)
    a = float(np.float32(alpha_exact()))
    expected = scipy.signal.lfilter([a], [1.0, -(1.0 - a)], x.astype(np.float64))
    return dict(
        params={"tau_s": TAU, "rate_hz": FS, "alpha": a},
        arrays={"input": Array("f32", x), "expected": Array("f64", expected)},
        # The regression guard (accuracy policy step 3): the golden SNR of the f32 fallback
        # minus 1 dB.
        tolerance={
            "class": "tolerance",
            "snr_db_min": 100.0,
            "snr_db_guard": 144.4,
            "skip": 0,
        },
        notes="alpha = 1 - exp(-1 / (fs tau)) cast to f32; y[n] = y[n-1] + alpha (x[n] - y[n-1])",
    )


@case(kernel="deemphasis", name="tau50us_fs48k_q15_8192", seed=51)
def tau50us_q15(rng):
    x = q15.from_float(rng.standard_normal(8192) * 0.3)
    a = int(q15.from_float(alpha_exact()))
    y = 0
    out = np.zeros(len(x), dtype=np.int64)
    for i, v in enumerate(x.tolist()):
        d = v - y
        y = int(q15.sat16(y + q15.round_shift(a * d, 15)))
        out[i] = y
    return dict(
        params={"tau_s": TAU, "rate_hz": FS, "alpha_raw": a},
        arrays={"input": Array("i16", x), "expected": Array("i16", out)},
        tolerance={"class": "bit_exact"},
        notes="exact Q15 recursion y = sat16(y + round_shift(alpha * (x - y), 15))",
    )
