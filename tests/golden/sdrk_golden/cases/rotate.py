"""Cases for the rotate kernel (NCO mixer with a 32-bit phase accumulator)."""

import numpy as np

from .. import q15
from ..registry import Array, case

TWO32 = 2**32


def increment(freq):
    """``inc = int(round(frac(f) * 2**32)) % 2**32`` (round half to even, like Rust ``rint``)."""
    frac = freq - np.floor(freq)
    return int(round(frac * TWO32)) % TWO32


def rotation(n, inc, phase0):
    """``exp(j * 2pi * phase[n] / 2**32)`` with an exact modular phase."""
    idx = np.arange(n, dtype=np.uint64)
    phase = (np.uint64(phase0) + idx * np.uint64(inc)) % np.uint64(TWO32)
    theta = 2.0 * np.pi * phase.astype(np.float64) / float(TWO32)
    return np.exp(1j * theta)


def noise(rng, n, sigma):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma


def c32_case(rng, n, freq, phase0, sigma=0.3):
    x = noise(rng, n, sigma).astype(np.complex64)
    inc = increment(freq)
    expected = x.astype(np.complex128) * rotation(n, inc, phase0)
    return dict(
        params={"freq": freq, "inc": inc, "phase0": phase0},
        arrays={"input": Array("cf32", x), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 120.0, "skip": 0},
        notes="output[n] = input[n] * exp(j*2*pi*phase[n]/2^32), phase[n] = phase0 + n*inc (mod 2^32)",
    )


@case(kernel="rotate", name="c32_noise_f0p1234_8192", seed=20)
def c32_noise(rng):
    return c32_case(rng, 8192, 0.1234, 0)


@case(kernel="rotate", name="c32_negfreq_8192", seed=21)
def c32_negfreq(rng):
    return c32_case(rng, 8192, -0.3, 123456789)


@case(kernel="rotate", name="c32_wrap_4096", seed=22)
def c32_wrap(rng):
    return c32_case(rng, 4096, 0.4999, TWO32 - 1000)


def cq15_case(x_re, x_im, freq, phase0, snr):
    n = len(x_re)
    inc = increment(freq)
    xf = q15.to_float(x_re) + 1j * q15.to_float(x_im)
    y = xf * rotation(n, inc, phase0)
    # The Q15 kernel saturates, so the ideal output is clipped to the Q15 range as well
    # (a rotated sample of magnitude above 1 can leave the range on one component).
    top = 32767.0 / 32768.0
    expected = np.clip(y.real, -1.0, top) + 1j * np.clip(y.imag, -1.0, top)
    return dict(
        params={"freq": freq, "inc": inc, "phase0": phase0},
        arrays={"input": Array("ci16", q15.pack(x_re, x_im)), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": snr, "skip": 0},
        notes="expected is the f64 rotation of the quantized input, clipped to the Q15 range",
    )


@case(kernel="rotate", name="cq15_noise_f0p05_8192", seed=23)
def cq15_noise(rng):
    n = 8192
    x = noise(rng, n, 0.25)
    return cq15_case(q15.from_float(x.real), q15.from_float(x.imag), 0.05, 0, 80.0)


@case(kernel="rotate", name="cq15_fullscale_tone_4096", seed=24)
def cq15_fullscale(rng):
    n = 4096
    amp = 1.0 - 2.0**-15
    # Random starting phase so the seed is meaningful.
    start = rng.uniform(0.0, 2.0 * np.pi)
    tone = amp * np.exp(1j * (2.0 * np.pi * 0.0123 * np.arange(n) + start))
    return cq15_case(q15.from_float(tone.real), q15.from_float(tone.imag), -0.2371, 987654321, 75.0)
