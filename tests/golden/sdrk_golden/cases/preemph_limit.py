"""Cases for the preemph_limit kernel (first-order pre-emphasis and peak limiter).

The limiter compares and switches, so the expected values come from exact models of the
arithmetic fixed by the specification (M25 section 2): the float32 model (``f32model``) for f32
and plain Python integers for Q15. The coefficients are computed in float64 with ``math`` first,
as the kernel does. Nothing here uses NumPy for arithmetic that feeds a committed byte (no
``sin``/``exp``/``lfilter``), so the files are identical on every CPU. The noise comes from
``rng.standard_normal``, which is the same on every platform.
"""

import math
import struct

import numpy as np

from .. import design as sdrk_design
from .. import f32model as fm
from .. import q15
from ..registry import Array, case

RATE_HZ = 48000
TAU_S = 50e-6
HOLD_S = 0.02
RELEASE_S = 0.2

# Largest absolute f32 input: a finite input beyond it is clamped (M25 section 2, step 1).
INPUT_LIMIT = 1024.0

# Saturation of the Q15 pre-emphasized value in Q30 (2^62), never reached by a stable filter.
V_LIMIT = 1 << 62


def bits(x):
    """The IEEE-754 bits of a float64 as an integer (exact in JSON)."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def hold_samples(hold_s, rate_hz):
    return int(math.floor(hold_s * rate_hz + 0.5))


def release_factor(release_s, rate_hz):
    return math.exp(-1.0 / (rate_hz * release_s))


def preemph_f32(x, coefs, limit, hold_s, release_s, rate_hz, retune=None):
    """Float32 model of the kernel. ``retune`` is ``(index, new_limit)``: ``set_limit`` before
    that sample. Returns ``(output, stats)``; stats counts the samples per limiter state."""
    b0, b1, a1 = (fm.f32(c) for c in coefs)
    r = fm.f32(release_factor(release_s, rate_hz))
    hold_len = hold_samples(hold_s, rate_hz)
    t = fm.f32(limit)
    zero = fm.f32(0.0)
    x1 = v1 = env = zero
    hold = 0
    out = np.zeros(len(x), dtype=np.float32)
    stats = {"limited": 0, "held": 0, "released": 0, "idle": 0}
    for i, sample in enumerate(np.asarray(x, dtype=np.float32).tolist()):
        if retune is not None and i == retune[0]:
            t = fm.f32(retune[1])
        xs = fm.f32(max(-INPUT_LIMIT, min(INPUT_LIMIT, sample))) if math.isfinite(sample) else zero
        v = fm.sub(fm.add(fm.mul(b0, xs), fm.mul(b1, x1)), fm.mul(a1, v1))
        x1 = xs
        v1 = fm.flush_denormal(v)
        m = abs(v)
        if m >= env:
            env, hold = m, hold_len
        elif hold > 0:
            hold -= 1
            stats["held"] += 1
        else:
            env = fm.flush_denormal(fm.mul(env, r))
            stats["released"] += 1
        if env > t:
            y = fm.mul(v, fm.div(t, env))
            stats["limited"] += 1
        else:
            y = v
            stats["idle"] += 1
        # clamp(y, -T, T); a NaN stays a NaN
        if y > t:
            y = t
        elif y < -t:
            y = -t
        out[i] = y
    return out, stats


def div_round(n, d):
    """sign(n) * floor((|n| + floor(d / 2)) / d) for d > 0."""
    q = (abs(n) + d // 2) // d
    return -q if n < 0 else q


def preemph_q15(xs, coefs, limit, hold_s, release_s, rate_hz):
    """Integer model of the kernel; ``xs`` are Q15 raw values. Returns ``(output, stats)``."""
    b0, b1, a1 = (int(math.floor(c * 2.0**24 + 0.5)) for c in coefs)
    r = int(math.floor(release_factor(release_s, rate_hz) * 2.0**31 + 0.5))
    hold_len = hold_samples(hold_s, rate_hz)
    t30 = int(math.floor(limit * 2.0**30 + 0.5))
    t15 = min(int(math.floor(limit * 32768.0 + 0.5)), 32767)
    x1 = v1 = env = hold = 0
    out = np.zeros(len(xs), dtype=np.int64)
    stats = {"limited": 0, "held": 0, "released": 0, "idle": 0, "max_env": 0}
    for i, x in enumerate(int(v) for v in xs):
        s = (b0 * x + b1 * x1) * (1 << 15) - a1 * v1
        v = (s + (1 << 23)) >> 24
        v = max(-V_LIMIT, min(V_LIMIT, v))
        x1, v1 = x, v
        m = abs(v)
        if m >= env:
            env, hold = m, hold_len
        elif hold > 0:
            hold -= 1
            stats["held"] += 1
        else:
            env = (env * r + (1 << 30)) >> 31
            stats["released"] += 1
        stats["max_env"] = max(stats["max_env"], env)
        if env > t30:
            y = div_round(v * t30, env)
            stats["limited"] += 1
        else:
            y = v
            stats["idle"] += 1
        o = max(-32768, min(32767, (y + (1 << 14)) >> 15))
        out[i] = max(-t15, min(t15, o))
    return out, stats


def noise(rng, n, sigma):
    return rng.standard_normal(n) * sigma


def limit_input(rng):
    """Noise sigma 0.05 plus tone bursts above the limit: 1 kHz amplitude 1.5 over samples
    500 to 1499 and 8 kHz amplitude 1.5 over samples 3500 to 4299 (float32)."""
    x = noise(rng, 8192, 0.05)
    for first, end, freq in ((500, 1500, 1000.0), (3500, 4300, 8000.0)):
        for k in range(first, end):
            x[k] += 1.5 * math.sin(2.0 * math.pi * ((freq * k / RATE_HZ) % 1.0))
    return x.astype(np.float32)


def params(tau_s, limit, coefs, **extra):
    base = {
        "tau_s": tau_s,
        "rate_hz": RATE_HZ,
        "limit": limit,
        "hold_s": HOLD_S,
        "release_s": RELEASE_S,
        "coefs": [float(c) for c in coefs],
        "b0_bits": bits(coefs[0]),
        "b1_bits": bits(coefs[1]),
        "a1_bits": bits(coefs[2]),
    }
    base.update(extra)
    return base


def coefs_of(tau_s, rate_hz=RATE_HZ):
    return sdrk_design.preemphasis_coefs(tau_s, float(rate_hz))


@case(kernel="preemph_limit", name="preemph_f32_8192", seed=444)
def preemph_f32_case(rng):
    x = noise(rng, 8192, 0.05).astype(np.float32)
    coefs = coefs_of(TAU_S)
    expected, stats = preemph_f32(x, coefs, 1.0, HOLD_S, RELEASE_S, RATE_HZ)
    assert stats["limited"] == 0, stats  # the limiter stays idle
    return dict(
        params=params(TAU_S, 1.0, coefs),
        arrays={"input": Array("f32", x), "expected": Array("f32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="48 kHz, tau 50 us, limit 1.0, hold 0.02 s, release 0.2 s, noise sigma 0.05; the "
        "limiter stays idle (pure pre-emphasis); exact float32 model",
    )


@case(kernel="preemph_limit", name="limit_f32_8192", seed=445)
def limit_f32_case(rng):
    x = limit_input(rng)
    coefs = coefs_of(TAU_S)
    expected, stats = preemph_f32(x, coefs, 0.5, HOLD_S, RELEASE_S, RATE_HZ)
    assert stats["limited"] > 500 and stats["held"] > 500 and stats["released"] > 500, stats
    assert float(np.max(np.abs(expected))) <= 0.5
    return dict(
        params=params(TAU_S, 0.5, coefs),
        arrays={"input": Array("f32", x), "expected": Array("f32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="as preemph_f32_8192 with limit 0.5; tone bursts at 1 kHz (samples 500-1499) and "
        "8 kHz (3500-4299) with amplitude 1.5 over noise sigma 0.05, so the limiter attacks, holds "
        "and releases; exact float32 model",
    )


@case(kernel="preemph_limit", name="limit_q15_8192", seed=446)
def limit_q15_case(rng):
    # The input of limit_f32_8192 (seed 445), quantized.
    x = limit_input(np.random.default_rng(445))
    xs = q15.from_float(x.astype(np.float64))
    coefs = coefs_of(TAU_S)
    expected, stats = preemph_q15(xs, coefs, 0.5, HOLD_S, RELEASE_S, RATE_HZ)
    assert stats["limited"] > 500 and stats["held"] > 500 and stats["released"] > 500, stats
    return dict(
        params=params(TAU_S, 0.5, coefs),
        arrays={"input": Array("i16", xs), "expected": Array("i16", expected)},
        tolerance={"class": "bit_exact"},
        notes="the input of limit_f32_8192 (seed 445) quantized to Q15, limit 0.5; exact integer "
        "model (Q24 coefficients, Q30 value, Q31 release factor)",
    )


@case(kernel="preemph_limit", name="preemph_q15_8192", seed=447)
def preemph_q15_case(rng):
    tau = 530e-6
    x = noise(rng, 8192, 0.2)
    xs = q15.from_float(x)
    coefs = coefs_of(tau)
    expected, stats = preemph_q15(xs, coefs, 1.0, HOLD_S, RELEASE_S, RATE_HZ)
    assert stats["max_env"] > (1 << 34), stats  # the Q30 value is far above full scale
    return dict(
        params=params(tau, 1.0, coefs),
        arrays={"input": Array("i16", xs), "expected": Array("i16", expected)},
        tolerance={"class": "bit_exact"},
        notes="48 kHz, tau 530 us (gain up to 36 dB inside the kernel), limit 1.0, hold 0.02 s, "
        "release 0.2 s, noise sigma 0.2 quantized to Q15; exact integer model",
    )


@case(kernel="preemph_limit", name="nonfinite_f32_4096", seed=448)
def nonfinite_f32_case(rng):
    # The first 4096 samples of the input of preemph_f32_8192 (seed 444) with 20 non-finite values.
    x = noise(np.random.default_rng(444), 8192, 0.05).astype(np.float32)[:4096]
    positions = sorted(int(p) for p in rng.choice(4096, size=20, replace=False))
    specials = [np.nan, np.inf, -np.inf]
    for k, pos in enumerate(positions):
        x[pos] = specials[k % 3]
    coefs = coefs_of(TAU_S)
    expected, stats = preemph_f32(x, coefs, 1.0, HOLD_S, RELEASE_S, RATE_HZ)
    assert np.all(np.isfinite(expected))
    return dict(
        params=params(TAU_S, 1.0, coefs, positions=positions),
        arrays={"input": Array("f32", x), "expected": Array("f32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="the first 4096 samples of the input of preemph_f32_8192 with NaN, +inf and -inf "
        "(in turn) at the 20 listed positions; a non-finite input counts as 0; exact float32 model",
    )


@case(kernel="preemph_limit", name="clamp_f32_4096", seed=0)
def clamp_f32_case(_rng):
    # The first 4096 samples of the input of preemph_f32_8192 (seed 444) with 24 finite values
    # beyond the input limit: 1025, 2000, 1e30, 1e36, 3e38 and the largest float32, each with
    # both signs, twice.
    x = noise(np.random.default_rng(444), 8192, 0.05).astype(np.float32)[:4096]
    positions = [100 + 163 * k for k in range(24)]
    big = [1025.0, 2000.0, 1e30, 1e36, 3e38, float(np.finfo(np.float32).max)]
    for k, pos in enumerate(positions):
        x[pos] = np.float32(big[k % 6] * (1.0 if (k // 6) % 2 == 0 else -1.0) * (1.0 if k % 2 == 0 else -1.0))
    coefs = coefs_of(TAU_S)
    expected, stats = preemph_f32(x, coefs, 1.0, HOLD_S, RELEASE_S, RATE_HZ)
    assert np.all(np.isfinite(expected))
    return dict(
        params=params(TAU_S, 1.0, coefs, positions=positions, input_limit=INPUT_LIMIT),
        arrays={"input": Array("f32", x), "expected": Array("f32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="the first 4096 samples of the input of preemph_f32_8192 with 24 finite values beyond the input "
        "limit 1024 (1025, 2000, 1e30, 1e36, 3e38 and the largest float32, both signs) at the listed positions; "
        "a finite input is clamped to +-1024; exact float32 model",
    )


@case(kernel="preemph_limit", name="retune_f32_8192", seed=449)
def retune_f32_case(rng):
    x = limit_input(rng)
    coefs = coefs_of(TAU_S)
    expected, stats = preemph_f32(x, coefs, 1.0, HOLD_S, RELEASE_S, RATE_HZ, retune=(4096, 0.5))
    assert stats["limited"] > 500 and stats["held"] > 500, stats
    return dict(
        params=params(TAU_S, 1.0, coefs, retune_at=4096, retune_limit=0.5),
        arrays={"input": Array("f32", x), "expected": Array("f32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="the construction of limit_f32_8192 with this case's own seed; limit 1.0 for samples "
        "0-4095 and 0.5 from sample 4096 (set_limit before that sample; the 8 kHz burst spans "
        "the change); exact float32 model",
    )
