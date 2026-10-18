"""Cases for the fir_interpolate kernels: ``upfirdn(h, x, L, 1)`` restricted to the outputs of the input."""

import numpy as np
import scipy.signal

from .. import q15
from ..registry import Array, case
from .fir import noise
from .resample import out_count, polyphase_q15

UP = 5
TAPS = 241
SAMPLES = 4000


@case(kernel="fir_interpolate", name="l5_n241_c32_4000", seed=90)
def l5_n241_c32(rng):
    h = (UP * scipy.signal.firwin(TAPS, 0.18)).astype(np.float32)
    x = noise(rng, SAMPLES, 0.3).astype(np.complex64)
    y = scipy.signal.upfirdn(h.astype(np.float64), x.astype(np.complex128), UP, 1)[: UP * SAMPLES]
    assert len(y) == out_count(SAMPLES, UP, 1) == UP * SAMPLES
    return dict(
        params={"taps": TAPS, "factor": UP, "outputs": UP * SAMPLES},
        arrays={"input": Array("cf32", x), "taps": Array("f32", h), "expected": Array("cf64", y)},
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "skip": 0},
        notes="taps 5 * firwin(241, 0.18) with the interpolation gain; every input sample yields exactly 5 outputs",
    )


@case(kernel="fir_interpolate", name="l5_n241_cq15_4000", seed=91)
def l5_n241_cq15(rng):
    # The taps are divided by L, so that every polyphase branch sums to at most 1 (a unit-gain
    # low-pass): the output has the gain loss of a factor 5 that a caller would restore afterwards.
    h = q15.from_float(UP * scipy.signal.firwin(TAPS, 0.18) / UP)
    x = noise(rng, SAMPLES, 0.3)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    y = q15.pack(polyphase_q15(xr, h, UP, 1), polyphase_q15(xi, h, UP, 1))
    assert len(y) == UP * SAMPLES
    return dict(
        params={"taps": TAPS, "factor": UP, "outputs": UP * SAMPLES},
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "taps": Array("i16", h), "expected": Array("ci16", y)},
        tolerance={"class": "bit_exact"},
        notes="taps divided by 5 (gain loss of 5); integer polyphase model, int64 accumulation, one rounding per output",
    )


@case(kernel="fir_interpolate_real", name="l5_n241_f32_4000", seed=94)
def l5_n241_f32(rng):
    h = (UP * scipy.signal.firwin(TAPS, 0.18)).astype(np.float32)
    x = (rng.standard_normal(SAMPLES) * 0.3).astype(np.float32)
    y = scipy.signal.upfirdn(h.astype(np.float64), x.astype(np.float64), UP, 1)[: UP * SAMPLES]
    return dict(
        params={"taps": TAPS, "factor": UP, "outputs": UP * SAMPLES},
        arrays={"input": Array("f32", x), "taps": Array("f32", h), "expected": Array("f64", y)},
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "skip": 0},
        notes="real signal, interpolation by 5 with gain 5 in the taps",
    )


@case(kernel="fir_interpolate_real", name="l5_n241_q15_4000", seed=95)
def l5_n241_q15(rng):
    h = q15.from_float(scipy.signal.firwin(TAPS, 0.18))
    x = q15.from_float(rng.standard_normal(SAMPLES) * 0.3)
    y = polyphase_q15(x, h, UP, 1)
    return dict(
        params={"taps": TAPS, "factor": UP, "outputs": UP * SAMPLES},
        arrays={"input": Array("i16", x), "taps": Array("i16", h), "expected": Array("i16", y)},
        tolerance={"class": "bit_exact"},
        notes="real Q15 signal; taps without the gain L (gain loss of 5); integer polyphase model",
    )
