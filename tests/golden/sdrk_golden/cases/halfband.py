"""Cases for the halfband_decimate and halfband_decimate_real kernels (M23): the same filter as
``fir_decimate`` with factor 2, taps with every even offset from the centre equal to zero."""

import numpy as np
import scipy.signal

from .. import design, q15
from ..registry import Array, case
from .fir import noise

N = 23
BETA60 = float(scipy.signal.kaiser_beta(60.0))


def taps_f32(n=N):
    """Halfband taps (float64 design) rounded to float32; the zero taps stay 0."""
    return design.halfband_taps(n, BETA60).astype(np.float32)


def taps_q15(n=N):
    """Halfband taps as Q15 integers; the zero taps stay 0."""
    return q15.from_float(design.halfband_taps(n, BETA60))


def decimate_q15(v, h):
    """Integer model of ``fir_decimate`` with factor 2: every second output of the int64 convolution,
    one rounding and saturation."""
    full = np.convolve(np.asarray(v, dtype=np.int64), np.asarray(h, dtype=np.int64))[: len(v)][::2]
    return q15.sat16(q15.round_shift(full, 15))


@case(kernel="halfband_decimate", name="n23_c32_8192", seed=430)
def n23_c32(rng):
    h = taps_f32()
    x = noise(rng, 8192, 0.3).astype(np.complex64)
    full = scipy.signal.lfilter(h.astype(np.float64), [1.0], x.astype(np.complex128))
    return dict(
        params={"taps": N, "attenuation_db": 60.0, "beta": BETA60},
        arrays={"input": Array("cf32", x), "taps": Array("f32", h), "expected": Array("cf64", full[::2])},
        # The regression guard (accuracy policy step 3): the golden SNR of the f32 fallback
        # (142.9 dB) minus 1 dB.
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "snr_db_guard": 141.9, "skip": 0},
        notes="halfband_taps(23, kaiser_beta(60)) rounded to f32; y[m] = lfilter(h, 1, x)[2 m] in float64, "
        "zero history; complex noise sigma 0.3",
    )


@case(kernel="halfband_decimate", name="n23_cq15_8192", seed=431)
def n23_cq15(rng):
    h = taps_q15()
    x = noise(rng, 8192, 0.3)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    return dict(
        params={"taps": N, "attenuation_db": 60.0, "beta": BETA60},
        arrays={
            "input": Array("ci16", q15.pack(xr, xi)),
            "taps": Array("i16", h),
            "expected": Array("ci16", q15.pack(decimate_q15(xr, h), decimate_q15(xi, h))),
        },
        tolerance={"class": "bit_exact"},
        notes="the integer model of fir_decimate with factor 2 and the quantized halfband taps (zero taps stay 0): "
        "int64 convolution, every second output, one rounding and saturation per component",
    )


@case(kernel="halfband_decimate_real", name="n23_f32_8192", seed=432)
def n23_f32(rng):
    h = taps_f32()
    x = (rng.standard_normal(8192) * 0.3).astype(np.float32)
    full = scipy.signal.lfilter(h.astype(np.float64), [1.0], x.astype(np.float64))
    return dict(
        params={"taps": N, "attenuation_db": 60.0, "beta": BETA60},
        arrays={"input": Array("f32", x), "taps": Array("f32", h), "expected": Array("f64", full[::2])},
        # The regression guard (accuracy policy step 3): the golden SNR of the f32 fallback
        # (143.2 dB) minus 1 dB.
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "snr_db_guard": 142.1, "skip": 0},
        notes="halfband_taps(23, kaiser_beta(60)) rounded to f32; y[m] = lfilter(h, 1, x)[2 m] in float64, "
        "zero history; real noise sigma 0.3",
    )


@case(kernel="halfband_decimate_real", name="n23_q15_8192", seed=433)
def n23_q15(rng):
    h = taps_q15()
    x = q15.from_float(rng.standard_normal(8192) * 0.3)
    return dict(
        params={"taps": N, "attenuation_db": 60.0, "beta": BETA60},
        arrays={"input": Array("i16", x), "taps": Array("i16", h), "expected": Array("i16", decimate_q15(x, h))},
        tolerance={"class": "bit_exact"},
        notes="the integer model of fir_decimate with factor 2 and the quantized halfband taps (zero taps stay 0), "
        "real input: int64 convolution, every second output, one rounding and saturation; quantized noise sigma 0.3",
    )


def corner_input(h, rng, n=2048):
    """A Q15 input that drives the sums of the halfband to both saturation limits and to the edge of the
    rounding: windows aligned with the signs of the taps (the sum of the magnitudes is above 1, so the sums
    reach +-32767 and +-32768 after the saturation), runs of -32768 and of +32767 (the sums of a constant run
    are about 32768 * sum(h)), full-scale alternation, and noise in between. Returns the raw int64 values."""
    x = q15.from_float(rng.standard_normal(n) * 0.3)
    sign = np.sign(h.astype(np.int64))
    # For the output m the window is x[2 m - k], k = 0 .. N - 1 (``h`` is symmetric, so reversed is the same).
    for m0, flip in ((20, 1), (60, -1), (101, 1)):
        for k in range(len(h)):
            x[2 * m0 - k] = 32767 if sign[k] * flip > 0 else -32768
    x[300:340] = -32768
    x[400:440] = 32767
    x[500:540] = np.where(np.arange(40) % 2 == 0, -32768, 32767)
    x[600:640] = np.where(np.arange(40) % 4 < 2, -32768, 32767)
    return x


@case(kernel="halfband_decimate", name="n23_cq15_corners_2048", seed=452)
def n23_cq15_corners(rng):
    h = taps_q15()
    xr, xi = corner_input(h, rng), corner_input(h, rng)
    # The second component is a delayed, inverted copy of the first in part, so that the two saturate on
    # different outputs.
    xi[700:900] = q15.sat16(-xr[700:900])
    yr, yi = decimate_q15(xr, h), decimate_q15(xi, h)
    assert (yr == 32767).any() and (yr == -32768).any() and (yi == 32767).any() and (yi == -32768).any()
    return dict(
        params={"taps": N, "attenuation_db": 60.0, "beta": BETA60},
        arrays={
            "input": Array("ci16", q15.pack(xr, xi)),
            "taps": Array("i16", h),
            "expected": Array("ci16", q15.pack(yr, yi)),
        },
        tolerance={"class": "bit_exact"},
        notes="the integer model of fir_decimate with factor 2 (int64 convolution, one rounding and saturation per "
        "component); the input holds windows aligned with the signs of the taps (both saturation limits), runs of "
        "-32768 and +32767 and full-scale alternations",
    )


@case(kernel="halfband_decimate_real", name="n23_q15_corners_2048", seed=453)
def n23_q15_corners(rng):
    h = taps_q15()
    x = corner_input(h, rng)
    y = decimate_q15(x, h)
    assert (y == 32767).any() and (y == -32768).any()
    return dict(
        params={"taps": N, "attenuation_db": 60.0, "beta": BETA60},
        arrays={"input": Array("i16", x), "taps": Array("i16", h), "expected": Array("i16", y)},
        tolerance={"class": "bit_exact"},
        notes="the integer model of fir_decimate with factor 2 and the quantized halfband taps, real input; the "
        "input holds windows aligned with the signs of the taps (both saturation limits), runs of -32768 and "
        "+32767 and full-scale alternations",
    )
