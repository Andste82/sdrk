"""Cases for the am_mod kernel (AM modulator, M35 section 1).

The kernel is bit-exact in both types, so the expected values come from exact models of the
arithmetic fixed by the specification: the float32 model (``f32model``, one operation at a time:
clamp with NaN giving 0, ``t = depth x'``, ``u = 1 + t``, ``re = carrier u``) for C32 and the
integer model with plain Python integers for CQ15 (``c_q``, ``m_q`` = ``floor(v 32768 + 0.5)``
limited to 32767, ``t = round_shift(m_q x, 15)``, ``u = 32768 + t``,
``re = sat16(round_shift(c_q u, 15))``). The tone uses ``math.cos`` of an exactly reduced phase and
the noise comes from ``rng.standard_normal``; nothing that feeds a committed byte uses NumPy
transcendentals, so the files are identical on every CPU.
"""

import math

import numpy as np

from .. import f32model as fm
from .. import q15
from ..registry import Array, case

N = 4096
RATE_HZ = 48000
TONE_HZ = 1000
TONE_AMP = 0.7
NOISE_SIGMA = 0.2
CARRIER = 0.5
DEPTH = 0.8
# Values beyond +-1 (the first six fixed, the rest random with a magnitude in (1, 4)) and NaN.
BEYOND = 32
NANS = 8
FIXED_BEYOND = [math.inf, -math.inf, 3.4028234663852886e38, -3.4028234663852886e38, 1.0 + 2.0**-23, -(1.0 + 2.0**-23)]


def signal(rng):
    """The tone plus noise in float64 with the specials at random positions.

    Returns the signal and the sorted positions of the values beyond +-1 and of the NaNs.
    """
    tone = np.array([TONE_AMP * math.cos(2.0 * math.pi * ((TONE_HZ * k) % RATE_HZ) / RATE_HZ) for k in range(N)])
    x = tone + rng.standard_normal(N) * NOISE_SIGMA
    pos = rng.choice(N, BEYOND + NANS, replace=False)
    signs = rng.choice([-1.0, 1.0], BEYOND - len(FIXED_BEYOND))
    mags = rng.uniform(1.0, 4.0, BEYOND - len(FIXED_BEYOND))
    beyond = FIXED_BEYOND + (signs * mags).tolist()
    for p, v in zip(pos[:BEYOND].tolist(), beyond):
        x[p] = v
    for p in pos[BEYOND:].tolist():
        x[p] = math.nan
    return x, sorted(pos[:BEYOND].tolist()), sorted(pos[BEYOND:].tolist())


def gain_q15(v):
    """``round(v 32768)`` limited to 32767, round half up in float64 (``v`` is the f32 value)."""
    return min(int(math.floor(float(fm.f32(v)) * 32768.0 + 0.5)), 32767)


def am_mod_f32(x, carrier, depth):
    """Float32 model; ``x`` is a float32 array. Returns the real parts (float32)."""
    c, m = fm.f32(carrier), fm.f32(depth)
    one, lo, hi = fm.f32(1.0), fm.f32(-1.0), fm.f32(1.0)
    out = np.zeros(len(x), dtype=np.float32)
    for i, v in enumerate(x.tolist()):
        v = fm.f32(v)
        xc = fm.f32(0.0) if np.isnan(v) else min(max(v, lo), hi)
        t = fm.mul(m, xc)
        u = fm.add(one, t)
        out[i] = fm.mul(c, u)
    return out


def am_mod_q15(xs, c_q, m_q):
    """Integer model with Python integers; ``xs`` are Q15 raw values. Returns the real parts."""
    out = []
    for x in xs:
        t = (m_q * x + (1 << 14)) >> 15
        u = 32768 + t
        out.append(max(-32768, min(32767, (c_q * u + (1 << 14)) >> 15)))
    return np.array(out, dtype=np.int64)


def params(beyond, nans, **extra):
    return {
        "rate_hz": RATE_HZ,
        "tone_hz": TONE_HZ,
        "tone_amplitude": TONE_AMP,
        "noise_sigma": NOISE_SIGMA,
        "carrier": CARRIER,
        "depth": DEPTH,
        "beyond_positions": beyond,
        "nan_positions": nans,
        **extra,
    }


@case(kernel="am_mod", name="c32_4096", seed=480)
def am_mod_c32(rng):
    x, beyond, nans = signal(rng)
    x = x.astype(np.float32)
    assert all(abs(float(x[p])) > 1.0 for p in beyond) and all(np.isnan(x[p]) for p in nans)
    re = am_mod_f32(x, CARRIER, DEPTH)
    out = np.empty(N, dtype=np.complex64)
    out.real = re
    out.imag = np.zeros(N, dtype=np.float32)
    # The clamp reaches both ends of the envelope: carrier (1 - depth) and carrier (1 + depth).
    assert float(re.min()) == float(fm.mul(CARRIER, fm.add(1.0, fm.mul(DEPTH, -1.0))))
    assert float(re.max()) == float(fm.mul(CARRIER, fm.add(1.0, fm.mul(DEPTH, 1.0))))
    return dict(
        params=params(beyond, nans),
        arrays={"input": Array("f32", x), "expected": Array("cf32", out)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="48 kHz; a 1 kHz tone 0.7 plus real noise sigma 0.2, then 32 values beyond +-1 (+-inf, "
        "+-f32::MAX, +-(1 + 2^-23) and 26 random magnitudes in (1, 4)) and 8 NaN at the listed "
        "positions; carrier 0.5, depth 0.8; exact float32 model: x' = clamp(x, -1, 1) with NaN "
        "giving 0, t = depth x', u = 1 + t, re = carrier u, im = +0",
    )


@case(kernel="am_mod", name="cq15_4096", seed=480)
def am_mod_cq15(rng):
    x, beyond, nans = signal(rng)
    xs = q15.from_float(x)
    c_q, m_q = gain_q15(CARRIER), gain_q15(DEPTH)
    re = am_mod_q15(xs.tolist(), c_q, m_q)
    assert int(xs.min()) == -32768 and int(xs.max()) == 32767
    return dict(
        params=params(beyond, nans, carrier_q=c_q, depth_q=m_q),
        arrays={"input": Array("i16", xs), "expected": Array("ci16", q15.pack(re, np.zeros(N, dtype=np.int64)))},
        tolerance={"class": "bit_exact"},
        notes="the signal of c32_4096 quantized to Q15 (the values beyond +-1 saturate, NaN gives 0); "
        "integer model: c_q = 16384, m_q = 26214 (floor(v 32768 + 0.5) of the f32 values, limited to "
        "32767), t = round_shift(m_q x, 15), u = 32768 + t, re = sat16(round_shift(c_q u, 15)), im = 0",
    )
