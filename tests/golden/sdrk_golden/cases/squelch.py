"""Cases for the squelch kernel (power gate with hysteresis).

The gate decision is discrete, so the expected values come from exact models of the arithmetic
fixed by the specification: the float32 model (``f32model``) for C32 and the integer model for
CQ15. The coefficient and the levels are computed in float64 first, as the kernel does.
"""

import math

import numpy as np

from .. import f32model as fm
from .. import q15, scalar_math
from ..registry import Array, case
from .fir import noise

RATE_HZ = 48000
TAU_S = 0.002
THRESHOLD_DB = -40.0
HYSTERESIS_DB = 3.0

# Complex tone bursts: (first sample, one past the last, level in dBFS).
BURSTS = [(4000, 10000, -20.0), (15000, 18000, -20.0), (20000, 22000, -41.0)]
TONE_CYCLES_PER_SAMPLE = 0.05


def alpha64(rate_hz, tau_s):
    return 1.0 - math.exp(-1.0 / (rate_hz * tau_s))


def level64(db):
    return 10.0 ** (db / 10.0)


def q30(db):
    return int(math.floor(level64(db) * 2.0**30 + 0.5))


def bursts_signal(rng):
    """Noise sigma 0.001 plus the tone bursts, as complex128 (quantized by the callers)."""
    n = 24000
    x = noise(rng, n, 0.001)
    k = np.arange(n)
    tone = scalar_math.expj(2.0 * np.pi * ((TONE_CYCLES_PER_SAMPLE * k) % 1.0))
    for first, end, db in BURSTS:
        x[first:end] += (10.0 ** (db / 20.0)) * tone[first:end]
    return x


def squelch_f32(x, schedule):
    """Float32 model. ``schedule`` maps a sample index to (threshold_db, hysteresis_db)
    applied from that sample on; index 0 must be present."""
    alpha = fm.f32(alpha64(RATE_HZ, TAU_S))
    p = fm.f32(0.0)
    is_open = False
    p_open = p_close = None
    out = np.zeros(len(x), dtype=np.complex64)
    for i, v in enumerate(x.tolist()):
        if i in schedule:
            thr, hyst = schedule[i]
            p_open, p_close = fm.f32(level64(thr)), fm.f32(level64(thr - hyst))
        re, im = fm.f32(v.real), fm.f32(v.imag)
        q = fm.add(fm.mul(re, re), fm.mul(im, im))
        if fm.is_finite(q):
            p = fm.flush_denormal(fm.add(p, fm.mul(alpha, fm.sub(q, p))))
            if p >= p_open:
                is_open = True
            elif p < p_close:
                is_open = False
        if is_open:
            out[i] = v
    return out


def squelch_q15(xr, xi, schedule):
    """Integer model: ``xr`` and ``xi`` are Q15 raw values."""
    alpha = int(q15.from_float(alpha64(RATE_HZ, TAU_S)))
    p = 0
    is_open = False
    p_open = p_close = None
    out = np.zeros((len(xr), 2), dtype=np.int64)
    for i, (re, im) in enumerate(zip(xr.tolist(), xi.tolist())):
        if i in schedule:
            thr, hyst = schedule[i]
            p_open, p_close = q30(thr), q30(thr - hyst)
        q = re * re + im * im
        p = p + int(q15.round_shift(alpha * (q - p), 15))
        if p >= p_open:
            is_open = True
        elif p < p_close:
            is_open = False
        if is_open:
            out[i] = (re, im)
    return out


def params(**extra):
    base = {
        "threshold_db": THRESHOLD_DB,
        "hysteresis_db": HYSTERESIS_DB,
        "tau_s": TAU_S,
        "rate_hz": RATE_HZ,
    }
    base.update(extra)
    return base


@case(kernel="squelch", name="bursts_c32_24000", seed=410)
def bursts_c32(rng):
    x = bursts_signal(rng).astype(np.complex64)
    expected = squelch_f32(x, {0: (THRESHOLD_DB, HYSTERESIS_DB)})
    return dict(
        params=params(),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="noise sigma 0.001, tone bursts -20 dBFS (4000-9999, 15000-17999) and -41 dBFS "
        "(20000-21999, between the close and the open level); exact float32 model",
    )


@case(kernel="squelch", name="bursts_cq15_24000", seed=411)
def bursts_cq15(rng):
    x = bursts_signal(rng)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    expected = squelch_q15(xr, xi, {0: (THRESHOLD_DB, HYSTERESIS_DB)})
    return dict(
        params=params(),
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("ci16", expected)},
        tolerance={"class": "bit_exact"},
        notes="same signal as bursts_c32_24000 quantized to Q15; exact integer model "
        "(Q30 power, alpha as a Q15 raw value)",
    )


@case(kernel="squelch", name="retune_c32_8192", seed=412)
def retune_c32(rng):
    x = noise(rng, 8192, 0.01).astype(np.complex64)
    schedule = {0: (-30.0, HYSTERESIS_DB), 4096: (-45.0, HYSTERESIS_DB)}
    expected = squelch_f32(x, schedule)
    return dict(
        params=params(
            threshold_db=-30.0,
            retune_at=4096,
            retune_threshold_db=-45.0,
            retune_hysteresis_db=HYSTERESIS_DB,
        ),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="noise sigma 0.01 per component (power 2e-4, -37 dBFS; the milestone text calls "
        "this -40 dBFS); threshold -30 dB until sample 4095, -45 dB from "
        "sample 4096 (set_thresholds before that sample); exact float32 model",
    )
