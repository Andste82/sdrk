"""Cases for the ``oscillate`` kernel (a complex exponential from the 32-bit NCO phase).

``out[n] = exp(j * 2pi * (k * phase[n]) / 2^32)`` with ``phase[n] = phase0 + n * inc`` and the
product ``k * phase[n]`` taken modulo 2^32. The C32 expected values are the float64 ``exp`` (scalar ``cos`` and ``sin``) of the
exact modular phase; the CQ15 expected values come from the integer model of the table
interpolation of ``rotate`` (``COS_Q15``, 12-bit index, 15-bit fraction, ``sin`` at the phase
minus a quarter turn).
"""

import hashlib

import numpy as np

from .. import q15, scalar_math
from ..registry import Array, case
from .rotate import increment
from .stereo import COS_F32_SHA256, COS_Q15, COS_Q15_SHA256

TWO32 = 1 << 32
MASK = TWO32 - 1
QUARTER = 1 << 30
N = 8192
RATE_HZ = 192000
TONE_HZ = 19000.0
PHASE0 = 0x12345678

COS_Q15_ARRAY = np.asarray(COS_Q15, dtype=np.int64)


def scaled_phases(n, inc, phase0, harmonic):
    """``(harmonic * (phase0 + i * inc)) mod 2^32`` as exact Python integers in an int64 array."""
    return np.array([(harmonic * (phase0 + i * inc)) & MASK for i in range(n)], dtype=np.int64)


def interp_q15(p):
    idx = p >> 20
    f15 = (p >> 5) & 0x7FFF
    a, b = COS_Q15_ARRAY[idx], COS_Q15_ARRAY[idx + 1]
    return q15.sat16(a + q15.round_shift((b - a) * f15, 15))


def params(harmonic, inc):
    return {
        "freq": TONE_HZ / RATE_HZ,
        "inc": inc,
        "phase0": PHASE0,
        "harmonic": harmonic,
        "cos_f32_sha256": COS_F32_SHA256,
        "cos_q15_sha256": COS_Q15_SHA256,
    }


def c32_case(harmonic):
    inc = increment(TONE_HZ / RATE_HZ)
    phase = scaled_phases(N, inc, PHASE0, harmonic)
    expected = scalar_math.expj(2.0 * np.pi * phase.astype(np.float64) / float(TWO32))
    return dict(
        params=params(harmonic, inc),
        arrays={"expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 120.0, "skip": 0},
        notes="expected[n] = exp(j*2*pi*(k*phase[n] mod 2^32)/2^32) in float64, phase[n] = phase0 + "
        "n*inc, 19000 Hz at 192 kHz, k = harmonic",
    )


def cq15_case(harmonic):
    inc = increment(TONE_HZ / RATE_HZ)
    phase = scaled_phases(N, inc, PHASE0, harmonic)
    re = interp_q15(phase)
    im = interp_q15((phase - QUARTER) & MASK)
    return dict(
        params=params(harmonic, inc),
        arrays={"expected": Array("ci16", q15.pack(re, im))},
        tolerance={"class": "bit_exact"},
        notes="table model of the Q15 rotate interpolation at k*phase[n] (cos at the phase, sin at "
        "the phase minus a quarter turn); 19000 Hz at 192 kHz, k = harmonic",
    )


@case(kernel="oscillate", name="osc_c32_k1_8192", seed=467)
def osc_c32_k1(rng):
    return c32_case(1)


@case(kernel="oscillate", name="osc_c32_k2_8192", seed=467)
def osc_c32_k2(rng):
    return c32_case(2)


@case(kernel="oscillate", name="osc_cq15_k1_8192", seed=467)
def osc_cq15_k1(rng):
    return cq15_case(1)


@case(kernel="oscillate", name="osc_cq15_k3_8192", seed=467)
def osc_cq15_k3(rng):
    return cq15_case(3)
