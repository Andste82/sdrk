"""Cases for the fir kernel (complex signal, real taps)."""

import numpy as np
import scipy.signal

from .. import q15
from ..registry import Array, case


def noise(rng, n, sigma):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma


def c32_case(x, h, snr, notes):
    h32 = np.asarray(h).astype(np.float32)
    expected = scipy.signal.lfilter(h32.astype(np.float64), [1.0], x.astype(np.complex128))
    return dict(
        params={"taps": int(len(h32))},
        arrays={"input": Array("cf32", x), "taps": Array("f32", h32), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": snr, "skip": 0},
        notes=notes,
    )


def fir_q15(x_re, x_im, h):
    h = np.asarray(h, dtype=np.int64)
    n = len(x_re)
    re = q15.sat16(q15.round_shift(np.convolve(np.asarray(x_re, dtype=np.int64), h)[:n], 15))
    im = q15.sat16(q15.round_shift(np.convolve(np.asarray(x_im, dtype=np.int64), h)[:n], 15))
    return q15.pack(re, im)


@case(kernel="fir", name="lowpass63_c32_8192", seed=30)
def lowpass63_c32(rng):
    x = noise(rng, 8192, 0.3).astype(np.complex64)
    return c32_case(x, scipy.signal.firwin(63, 0.2), 110.0, "y[n] = sum h[k] x[n-k], zero history")


@case(kernel="fir", name="lowpass63_cq15_8192", seed=31)
def lowpass63_cq15(rng):
    h = q15.from_float(scipy.signal.firwin(63, 0.2))
    x = noise(rng, 8192, 0.25)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    return dict(
        params={"taps": 63},
        arrays={
            "input": Array("ci16", q15.pack(xr, xi)),
            "taps": Array("i16", h),
            "expected": Array("ci16", fir_q15(xr, xi, h)),
        },
        tolerance={"class": "bit_exact"},
        notes="Q30 accumulation in int64, one rounding and saturation per component",
    )


@case(kernel="fir", name="long255_c32_16384", seed=32)
def long255_c32(rng):
    x = noise(rng, 16384, 0.3).astype(np.complex64)
    return c32_case(
        x,
        scipy.signal.firwin(255, 0.05, window=("kaiser", 8.0)),
        110.0,
        "long filter, kaiser window",
    )


@case(kernel="fir", name="single_tap_c32_1024", seed=33)
def single_tap_c32(rng):
    x = noise(rng, 1024, 0.3).astype(np.complex64)
    h = np.array([0.5], dtype=np.float32)
    expected = (x.astype(np.complex128) * 0.5).astype(np.complex64)
    return dict(
        params={"taps": 1},
        arrays={"input": Array("cf32", x), "taps": Array("f32", h), "expected": Array("cf32", expected)},
        tolerance={"class": "bit_exact"},
        notes="a single tap of 0.5 halves every sample exactly",
    )


@case(kernel="fir", name="gain_overflow_cq15_2048", seed=34)
def gain_overflow_cq15(rng):
    n = 2048
    h = q15.from_float(np.full(8, 0.25))
    xr = np.full(n, 32767, dtype=np.int64)
    xi = np.full(n, -32768, dtype=np.int64)
    noise_part = rng.integers(-32768, 32768, size=(64, 2), dtype=np.int64)
    xr[:64], xi[:64] = noise_part[:, 0], noise_part[:, 1]
    return dict(
        params={"taps": 8},
        arrays={
            "input": Array("ci16", q15.pack(xr, xi)),
            "taps": Array("i16", h),
            "expected": Array("ci16", fir_q15(xr, xi, h)),
        },
        tolerance={"class": "bit_exact"},
        notes="eight taps of 0.25 (gain 2.0) on a DC input saturate to full scale",
    )
