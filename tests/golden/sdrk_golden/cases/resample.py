"""Cases for the resample kernels (rational L/M polyphase resampler)."""

import numpy as np
import scipy.signal

from .. import q15
from ..registry import Array, case
from .fir import noise


def out_count(n, up, down):
    """Number of outputs that depend only on the ``n`` received input samples.

    K = floor((n * L - 1) / M) + 1, the number of j >= 0 with floor(j * M / L) <= n - 1.
    """
    return (n * up - 1) // down + 1


def c32_case(x, h, up, down, snr, notes):
    h32 = np.asarray(h).astype(np.float32)
    k = out_count(len(x), up, down)
    y = scipy.signal.upfirdn(h32.astype(np.float64), x.astype(np.complex128), up, down)[:k]
    assert len(y) == k
    return dict(
        params={"taps": int(len(h32)), "up": up, "down": down, "outputs": k},
        arrays={"input": Array("cf32", x), "taps": Array("f32", h32), "expected": Array("cf64", y)},
        tolerance={"class": "tolerance", "snr_db_min": snr, "skip": 0},
        notes=notes + "; K = floor((n * L - 1) / M) + 1 outputs",
    )


def polyphase_q15(v, h, up, down):
    """Integer model: loop over the outputs, int64 accumulation, one rounding per output."""
    h = np.asarray(h, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    k = out_count(len(v), up, down)
    out = np.zeros(k, dtype=np.int64)
    for j in range(k):
        t = j * down
        p, b = t % up, t // up
        taps = h[p::up]
        idx = b - np.arange(len(taps))
        ok = idx >= 0
        acc = int(np.sum(taps[ok] * v[idx[ok]]))
        out[j] = int(q15.sat16(q15.round_shift(acc, 15)))
    return out


@case(kernel="resample", name="l1m5_n121_c32_10000", seed=43)
def l1m5_n121_c32(rng):
    x = noise(rng, 10000, 0.3).astype(np.complex64)
    return c32_case(x, 1 * scipy.signal.firwin(121, 0.18), 1, 5, 110.0, "decimation by 5, gain L = 1")


@case(kernel="resample", name="l3m2_n96_c32_6000", seed=44)
def l3m2_n96_c32(rng):
    x = noise(rng, 6000, 0.3).astype(np.complex64)
    return c32_case(x, 3 * scipy.signal.firwin(96, 1 / 3 * 0.9), 3, 2, 110.0, "rate 3/2, gain L = 3 in the taps")


@case(kernel="resample", name="l147m160_n1470_c32_16000", seed=45)
def l147m160_n1470_c32(rng):
    x = noise(rng, 16000, 0.3).astype(np.complex64)
    return c32_case(
        x, 147 * scipy.signal.firwin(1470, 0.9 / 160), 147, 160, 100.0, "rate 147/160 (44.1 kHz to 48 kHz style)"
    )


@case(kernel="resample", name="l3m2_n96_cq15_6000", seed=46)
def l3m2_n96_cq15(rng):
    # The taps would need the gain L = 3, which does not fit into Q15 for a unit-gain
    # low-pass; the taps are scaled by 1/3, so this case has a DC gain of 1 instead of 3
    # (a gain loss of a factor 3 that the caller would have to restore after the resampler).
    h = q15.from_float(3 * scipy.signal.firwin(96, 1 / 3 * 0.9) / 3)
    x = noise(rng, 6000, 0.3)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    k = out_count(len(xr), 3, 2)
    y = q15.pack(polyphase_q15(xr, h, 3, 2), polyphase_q15(xi, h, 3, 2))
    return dict(
        params={"taps": 96, "up": 3, "down": 2, "outputs": k},
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "taps": Array("i16", h), "expected": Array("ci16", y)},
        tolerance={"class": "bit_exact"},
        notes="taps scaled by 1/3 (gain loss of 3); integer polyphase model; K = floor((n * L - 1) / M) + 1",
    )


@case(kernel="resample_real", name="l1m5_n121_f32_10000", seed=47)
def l1m5_n121_f32(rng):
    x = (rng.standard_normal(10000) * 0.3).astype(np.float32)
    h32 = scipy.signal.firwin(121, 0.18).astype(np.float32)
    k = out_count(len(x), 1, 5)
    y = scipy.signal.upfirdn(h32.astype(np.float64), x.astype(np.float64), 1, 5)[:k]
    assert len(y) == k
    return dict(
        params={"taps": 121, "up": 1, "down": 5, "outputs": k},
        arrays={"input": Array("f32", x), "taps": Array("f32", h32), "expected": Array("f64", y)},
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "skip": 0},
        notes="real signal, decimation by 5",
    )
