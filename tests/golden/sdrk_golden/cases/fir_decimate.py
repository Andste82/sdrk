"""Cases for the fir_decimate kernel (complex signal, real taps, decimation by M)."""

import numpy as np
import scipy.signal

from .. import q15
from ..registry import Array, case
from .fir import noise


def c32_case(x, h, m, snr, notes):
    h32 = np.asarray(h).astype(np.float32)
    full = scipy.signal.lfilter(h32.astype(np.float64), [1.0], x.astype(np.complex128))
    return dict(
        params={"taps": int(len(h32)), "factor": m},
        arrays={"input": Array("cf32", x), "taps": Array("f32", h32), "expected": Array("cf64", full[::m])},
        tolerance={"class": "tolerance", "snr_db_min": snr, "skip": 0},
        notes=notes,
    )


@case(kernel="fir_decimate", name="m10_n63_c32_20000", seed=40)
def m10_n63_c32(rng):
    x = noise(rng, 20000, 0.3).astype(np.complex64)
    return c32_case(x, scipy.signal.firwin(63, 0.08), 10, 110.0, "y[m] = lfilter(h, 1, x)[m * 10], zero history")


@case(kernel="fir_decimate", name="m10_n63_cq15_20000", seed=41)
def m10_n63_cq15(rng):
    h = q15.from_float(scipy.signal.firwin(63, 0.08))
    x = noise(rng, 20000, 0.3)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    n = len(xr)

    def decimate(v):
        full = np.convolve(v, h)[:n][::10]
        return q15.sat16(q15.round_shift(full, 15))

    return dict(
        params={"taps": 63, "factor": 10},
        arrays={
            "input": Array("ci16", q15.pack(xr, xi)),
            "taps": Array("i16", h),
            "expected": Array("ci16", q15.pack(decimate(xr), decimate(xi))),
        },
        tolerance={"class": "bit_exact"},
        notes="integer convolution in int64, every 10th output, one rounding and saturation per component",
    )


@case(kernel="fir_decimate", name="m3_n31_c32_9000", seed=42)
def m3_n31_c32(rng):
    x = noise(rng, 9000, 0.3).astype(np.complex64)
    return c32_case(x, scipy.signal.firwin(31, 0.25), 3, 110.0, "y[m] = lfilter(h, 1, x)[m * 3], zero history")
