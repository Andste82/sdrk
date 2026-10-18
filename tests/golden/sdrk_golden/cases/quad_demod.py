"""Cases for the quad_demod kernel (FM discriminator)."""

import numpy as np

from .. import q15
from ..registry import Array, case
from .fir import noise


def fm_signal(rng, n):
    k = np.arange(n)
    phase = np.cumsum(2 * np.pi * 0.05 * np.sin(2 * np.pi * 0.002 * k))
    return np.exp(1j * phase) + noise(rng, n, 1e-3)


def discriminate(x):
    """arg(x[n] * conj(x[n-1])) with x[-1] = 0, in float64."""
    x = np.asarray(x, dtype=np.complex128)
    prev = np.concatenate([[0.0 + 0.0j], x[:-1]])
    return np.angle(x * np.conj(prev))


@case(kernel="quad_demod", name="fm_tone_c32_20000", seed=48)
def fm_tone_c32(rng):
    x = fm_signal(rng, 20000).astype(np.complex64)
    return dict(
        params={"gain": 1.0},
        arrays={"input": Array("cf32", x), "expected": Array("f64", discriminate(x))},
        tolerance={"class": "tolerance", "snr_db_min": 100.0, "skip": 0},
        notes="FM tone, deviation 0.05 cycles per sample, gain 1.0 (radians)",
    )


@case(kernel="quad_demod", name="fm_tone_cq15_20000", seed=49)
def fm_tone_cq15(rng):
    x = 0.9 * fm_signal(rng, 20000)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    xq = q15.to_float(xr) + 1j * q15.to_float(xi)
    gain_raw = 32767
    expected = discriminate(xq) / np.pi * (gain_raw / 32768)
    return dict(
        params={"gain_raw": gain_raw},
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("f64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 70.0, "skip": 0},
        notes="amplitude 0.9, output unit phase / pi times the Q15 gain 32767/32768",
    )
