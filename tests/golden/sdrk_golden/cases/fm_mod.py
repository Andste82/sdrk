"""Cases for the fm_mod kernel: integrate the scaled input in integers, then ``exp(j 2 pi phase / 2^32)``."""

import numpy as np

from .. import q15
from ..registry import Array, case

N = 8192
DEVIATION_HZ = 75e3
RATE_HZ = 240e3
K = DEVIATION_HZ / RATE_HZ * 2**32


def expected_signal(x, k=K):
    """phase = cumsum(floor(x * k + 0.5)) mod 2^32 (int64 sums cannot overflow here); output exp(j * ...)."""
    steps = np.floor(x.astype(np.float64) * k + 0.5).astype(np.int64)
    phase = np.cumsum(steps) % (1 << 32)
    return np.exp(1j * 2.0 * np.pi * phase.astype(np.float64) / 2.0**32)


def params():
    return {"deviation_hz": DEVIATION_HZ, "rate_hz": RATE_HZ, "k": K, "samples": N}


@case(kernel="fm_mod", name="dev75k_fs240k_c32_8192", seed=92)
def dev75k_c32(rng):
    x = (rng.standard_normal(N) * 0.3).astype(np.float32)
    return dict(
        params=params(),
        arrays={"input": Array("f32", x), "expected": Array("cf64", expected_signal(x))},
        tolerance={"class": "tolerance", "snr_db_min": 120.0, "skip": 0},
        notes="phase += floor(x * k + 0.5) with k = deviation / rate * 2^32, exact wrap modulo 2^32, unit amplitude output",
    )


@case(kernel="fm_mod", name="dev75k_fs240k_cq15_8192", seed=93)
def dev75k_cq15(rng):
    q = q15.from_float(rng.standard_normal(N) * 0.3)
    x = q.astype(np.float64) / 32768.0
    return dict(
        params=params(),
        arrays={"input": Array("i16", q), "expected": Array("cf64", expected_signal(x))},
        tolerance={"class": "tolerance", "snr_db_min": 80.0, "skip": 0},
        notes="Q15 input as x = q / 32768; the Q15 table output has amplitude 1 - 2^-15",
    )


# A deviation that makes k * x fractional for almost every input value (with the 75 kHz / 240 kHz
# of the cases above, k * x is an integer for almost every float32 x, so the rounding rule of the
# phase step would not be visible). A wrong rounding costs a bias of half a phase unit per sample,
# which is about 108 dB below the signal after 8192 samples.
ODD_DEVIATION_HZ = 33333.3
ODD_RATE_HZ = 192000.0
ODD_K = ODD_DEVIATION_HZ / ODD_RATE_HZ * 2**32


def odd_params():
    return {"deviation_hz": ODD_DEVIATION_HZ, "rate_hz": ODD_RATE_HZ, "k": ODD_K, "samples": N}


@case(kernel="fm_mod", name="dev33k_fs192k_c32_8192", seed=96)
def dev33k_c32(rng):
    x = rng.uniform(-1.0, 1.0, N).astype(np.float32)
    return dict(
        params=odd_params(),
        arrays={"input": Array("f32", x), "expected": Array("cf64", expected_signal(x, ODD_K))},
        tolerance={"class": "tolerance", "snr_db_min": 120.0, "skip": 0},
        notes="fractional k * x for almost every sample: checks the round-half-up of the phase step",
    )


@case(kernel="fm_mod", name="dev33k_fs192k_cq15_8192", seed=97)
def dev33k_cq15(rng):
    q = q15.from_float(rng.uniform(-1.0, 1.0, N))
    x = q.astype(np.float64) / 32768.0
    return dict(
        params=odd_params(),
        arrays={"input": Array("i16", q), "expected": Array("cf64", expected_signal(x, ODD_K))},
        tolerance={"class": "tolerance", "snr_db_min": 80.0, "skip": 0},
        notes="Q15 input with fractional k * x; the Q15 table output has amplitude 1 - 2^-15",
    )
