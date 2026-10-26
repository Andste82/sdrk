"""Cases for the agc kernel (automatic gain control, a multiplicative feedback loop whose error
is normalised by the output level, as the ``agc3_cc`` of GNU Radio).

``C32`` is compared with an f64 model of the specification that uses the f64 values of the f32
constants; ``CQ15`` is bit-exact against an integer model with the gain in Q.32.
"""

import math

import numpy as np

from .. import f32model as fm
from .. import q15, scalar_math
from ..registry import Array, case
from .fir import noise

RATE_HZ = 48000
REFERENCE = 0.5
ATTACK_S = 0.005
RELEASE_S = 0.05
MIN_DB, MAX_DB, INITIAL_DB = -20.0, 60.0, 0.0

# Complex tone of 1 kHz, 8000 samples per step, amplitudes of the three steps.
AMPLITUDES = [0.01, 0.3, 0.003]
STEP = 8000
NOISE_SIGMA = 1e-4


def rate64(rate_hz, tau_s):
    return 1.0 - math.exp(-1.0 / (rate_hz * tau_s))


def lin(db):
    return 10.0 ** (db / 20.0)


def signal(rng):
    n = STEP * len(AMPLITUDES)
    k = np.arange(n)
    tone = scalar_math.expj(2.0 * np.pi * ((k * 1000.0 / RATE_HZ) % 1.0))
    amp = np.repeat(np.array(AMPLITUDES), STEP)
    return amp * tone + noise(rng, n, NOISE_SIGMA)


def agc_f64(x):
    """Model of the C32 algorithm in f64 with the f64 values of the f32 constants."""
    f = lambda v: float(np.float32(v))  # noqa: E731
    ref = f(REFERENCE)
    lo, hi = f(REFERENCE) * 0.25, f(REFERENCE) * 1.5
    ra, rr = f(rate64(RATE_HZ, ATTACK_S)), f(rate64(RATE_HZ, RELEASE_S))
    gmin, gmax = f(lin(MIN_DB)), f(lin(MAX_DB))
    g = f(lin(INITIAL_DB))
    out = np.zeros(len(x), dtype=np.complex128)
    for i, v in enumerate(x.tolist()):
        y = complex(g * v.real, g * v.imag)
        out[i] = y
        m = math.sqrt(y.real * y.real + y.imag * y.imag)
        if m > 0.0:
            e = ref - m
            if e < 0:
                d = min(m, hi)
                g = g * d / (d + ra * (m - ref))
            else:
                d = max(m, lo)
                g = g * (d + rr * e) / d
            g = min(max(g, gmin), gmax)
    return out


def agc_f32(x):
    """Float32 model of the arithmetic of the Rust fallback (``fallback/agc.rs``).

    The output level is ``g * |x|`` and the release is evaluated in the two rearranged forms of
    the fallback, with the constants ``c = r_r ref`` and ``b = r_r / lo`` computed in f64 from
    the f64 values of the f32 constants and rounded once to f32; the attack and the first
    release form clamp on both sides, the second release form only to the maximum (it never
    lowers the gain).
    """
    ref = fm.f32(REFERENCE)
    lo, hi = fm.mul(ref, fm.f32(0.25)), fm.mul(ref, fm.f32(1.5))
    ra, rr = fm.f32(rate64(RATE_HZ, ATTACK_S)), fm.f32(rate64(RATE_HZ, RELEASE_S))
    c, b = fm.f32(float(rr) * float(ref)), fm.f32(float(rr) / float(lo))
    gmin, gmax = fm.f32(lin(MIN_DB)), fm.f32(lin(MAX_DB))
    g = fm.f32(lin(INITIAL_DB))
    out = np.zeros(len(x), dtype=np.complex64)
    for i, v in enumerate(x.tolist()):
        xr, xi = fm.f32(v.real), fm.f32(v.imag)
        out[i] = complex(float(fm.mul(g, xr)), float(fm.mul(g, xi)))
        a = fm.f32(np.sqrt(fm.add(fm.mul(xr, xr), fm.mul(xi, xi))))
        m = fm.mul(g, a)
        if m > ref:
            d = min(m, hi)
            g = fm.div(fm.mul(g, d), fm.add(d, fm.mul(ra, fm.sub(m, ref))))
            g = min(max(g, gmin), gmax)
        elif m >= lo:
            g = min(max(fm.add(g, fm.sub(fm.div(c, a), fm.mul(rr, g))), gmin), gmax)
        elif m > 0.0:
            g = min(fm.add(g, fm.mul(g, fm.mul(b, fm.sub(ref, m)))), gmax)
    return out


def sqrt_round(s):
    """Rounded integer square root (no saturation): ``r + (s - r^2 > r)``."""
    r = math.isqrt(s)
    return r + (1 if s - r * r > r else 0)


def agc_q15(xr, xi):
    """Integer model: gain ``g`` in Q.32, the rates in Q.48, divisions rounded half up."""
    ref_raw = int(math.floor(REFERENCE * 32768.0 + 0.5))
    lo_raw, hi_raw = ref_raw >> 2, ref_raw + (ref_raw >> 1)
    ra = int(math.floor(rate64(RATE_HZ, ATTACK_S) * 2.0**48 + 0.5))
    rr = int(math.floor(rate64(RATE_HZ, RELEASE_S) * 2.0**48 + 0.5))
    q32 = lambda db: int(math.floor(lin(db) * 2.0**32 + 0.5))  # noqa: E731
    gmin, gmax, g = q32(MIN_DB), q32(MAX_DB), q32(INITIAL_DB)
    assert ref_raw >= 1024 and ra <= 2**47 and rr <= 2**47 and gmax < 2**47
    out = np.zeros((len(xr), 2), dtype=np.int64)
    for i, (re, im) in enumerate(zip(xr.tolist(), xi.tolist())):
        yr = int(q15.sat16(q15.round_shift(re * g, 32)))
        yi = int(q15.sat16(q15.round_shift(im * g, 32)))
        assert abs(re * g) < 2**63 and abs(im * g) < 2**63
        out[i] = (yr, yi)
        if re == 0 and im == 0:
            continue
        m = sqrt_round(yr * yr + yi * yi)
        assert m <= 46341
        e = ref_raw - m
        if e < 0:
            d = min(m, hi_raw)
            den = (d << 48) - ra * e
            num = (g * d) << 48
            assert 2 * num < 2**126 and 2 * den < 2**126
            nxt = (2 * num + den) // (2 * den)
        else:
            d = max(m, lo_raw)
            den = d << 48
            num = g * rr * e
            assert 2 * num < 2**126
            nxt = g + (2 * num + den) // (2 * den)
        g = min(max(nxt, gmin), gmax)
    return out


def params():
    return {
        "rate_hz": RATE_HZ,
        "reference": REFERENCE,
        "attack_s": ATTACK_S,
        "release_s": RELEASE_S,
        "min_gain_db": MIN_DB,
        "max_gain_db": MAX_DB,
        "initial_gain_db": INITIAL_DB,
    }


@case(kernel="agc", name="steps_c32_24000", seed=417)
def steps_c32(rng):
    x = signal(rng).astype(np.complex64)
    expected = agc_f64(x.astype(np.complex128))
    # The float32 loop of the kernel must stay close to the f64 model; if it did not, the
    # pre-approved bit-exact fallback of the milestone would apply (measured: 125.5 dB, and the model gives the values of the fallback bit for bit).
    f32_out = agc_f32(x).astype(np.complex128)
    drift = 10.0 * np.log10(np.sum(np.abs(expected) ** 2) / np.sum(np.abs(expected - f32_out) ** 2))
    assert drift > 100.0, f"float32 and f64 AGC models drift apart: {drift:.1f} dB"
    return dict(
        params=params(),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", expected.astype(np.complex64))},
        tolerance={"class": "tolerance", "snr_db_min": 80.0, "skip": 0},
        notes="1 kHz complex tone, amplitude 0.01, 0.3 and 0.003 (8000 samples each) plus noise "
        "sigma 1e-4; the f64 model of the C32 algorithm with the f64 values of the f32 constants, rounded once to f32 (a cf64 file would exceed the 512 KiB limit of a case)",
    )


@case(kernel="agc", name="steps_cq15_24000", seed=418)
def steps_cq15(rng):
    x = signal(rng)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    return dict(
        params=params(),
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("ci16", agc_q15(xr, xi))},
        tolerance={"class": "bit_exact"},
        notes="same signal as steps_c32_24000 quantized to Q15; integer model with the gain in "
        "Q.32 and the step computed in 128-bit integers",
    )
