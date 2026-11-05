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


# ---------------------------------------------------------------------------------------------
# Stereo variant (M36 section 1): one envelope and one gain for both channels
# ---------------------------------------------------------------------------------------------

# Start of the f64 regression guard (decision D111): the whole signal.
GUARD_SKIP = 0


def sanitize_f32(sample):
    """The input rule of the f32 kernel: NaN and infinities become 0, a finite value is clamped
    to +-1024 (D200, D460)."""
    return fm.f32(max(-INPUT_LIMIT, min(INPUT_LIMIT, sample))) if math.isfinite(sample) else fm.f32(0.0)


def clamp_to(y, t):
    if y > t:
        return t
    if y < -t:
        return -t
    return y


def preemph_stereo_f32(left, right, coefs, limit, hold_s, release_s, rate_hz):
    """Float32 model of ``preemph_limit_stereo``. Returns ``(out_left, out_right, stats)``;
    stats counts the samples per limiter state and the limited samples whose envelope peak came
    from the left (``peak_left``) or the right channel (``peak_right``)."""
    b0, b1, a1 = (fm.f32(c) for c in coefs)
    r = fm.f32(release_factor(release_s, rate_hz))
    hold_len = hold_samples(hold_s, rate_hz)
    t = fm.f32(limit)
    zero = fm.f32(0.0)
    x1 = [zero, zero]
    v1 = [zero, zero]
    env = zero
    hold = 0
    n = len(left)
    out = [np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]
    stats = {"limited": 0, "held": 0, "released": 0, "idle": 0, "peak_left": 0, "peak_right": 0}
    xs_l = np.asarray(left, dtype=np.float32).tolist()
    xs_r = np.asarray(right, dtype=np.float32).tolist()
    for i in range(n):
        v = [zero, zero]
        for ch, sample in enumerate((xs_l[i], xs_r[i])):
            xs = sanitize_f32(sample)
            v[ch] = fm.sub(fm.add(fm.mul(b0, xs), fm.mul(b1, x1[ch])), fm.mul(a1, v1[ch]))
            x1[ch] = xs
            v1[ch] = fm.flush_denormal(v[ch])
        ml, mr = abs(v[0]), abs(v[1])
        m = ml if ml > mr else mr  # a select, not max (D461)
        if m >= env:
            env, hold = m, hold_len
        elif hold > 0:
            hold -= 1
            stats["held"] += 1
        else:
            env = fm.flush_denormal(fm.mul(env, r))
            stats["released"] += 1
        if env > t:
            g = fm.div(t, env)
            y = [fm.mul(v[0], g), fm.mul(v[1], g)]
            stats["limited"] += 1
            stats["peak_left" if ml > mr else "peak_right"] += 1
        else:
            y = v
            stats["idle"] += 1
        out[0][i] = clamp_to(y[0], t)
        out[1][i] = clamp_to(y[1], t)
    return out[0], out[1], stats


def preemph_stereo_f64(left, right, coefs, limit, hold_s, release_s, rate_hz):
    """The same loop in float64 (float64 coefficients, release factor and limit; no flush of
    denormals): the regression guard of decision D111. Returns a list of complex values."""
    b0, b1, a1 = coefs
    r = release_factor(release_s, rate_hz)
    hold_len = hold_samples(hold_s, rate_hz)
    t = float(limit)
    x1 = [0.0, 0.0]
    v1 = [0.0, 0.0]
    env = 0.0
    hold = 0
    out = []
    for sl, sr in zip(np.asarray(left, dtype=np.float64).tolist(), np.asarray(right, dtype=np.float64).tolist()):
        v = [0.0, 0.0]
        for ch, sample in enumerate((sl, sr)):
            xs = max(-INPUT_LIMIT, min(INPUT_LIMIT, sample)) if math.isfinite(sample) else 0.0
            v[ch] = (b0 * xs + b1 * x1[ch]) - a1 * v1[ch]
            x1[ch] = xs
            v1[ch] = v[ch]
        ml, mr = abs(v[0]), abs(v[1])
        m = ml if ml > mr else mr
        if m >= env:
            env, hold = m, hold_len
        elif hold > 0:
            hold -= 1
        else:
            env = env * r
        if env > t:
            g = t / env
            y = [v[0] * g, v[1] * g]
        else:
            y = v
        out.append(complex(clamp_to(y[0], t), clamp_to(y[1], t)))
    return out


def snr_db(reference, test):
    """``10 log10(sum |ref|^2 / sum |ref - test|^2)`` with exact sums."""
    num = math.fsum(abs(r) ** 2 for r in reference)
    den = math.fsum(abs(r - t) ** 2 for r, t in zip(reference, test))
    return math.inf if den == 0.0 else 10.0 * math.log10(num / den)


def guard_db(model, f64):
    """The SNR of the model against the float64 loop from ``GUARD_SKIP`` on, minus 1 dB, rounded
    down to 0.1 dB (D111, D115)."""
    snr = snr_db(f64[GUARD_SKIP:], model[GUARD_SKIP:])
    assert math.isfinite(snr), "the float32 model equals the float64 loop; use the sentinel of D372"
    return math.floor((snr - 1.0) * 10.0) / 10.0


def complex_of(left, right):
    """A complex64 array from the float32 parts (never as ``re + 1j * im``)."""
    z = np.zeros(len(left), dtype=np.complex64)
    z.real = np.asarray(left, dtype=np.float32)
    z.imag = np.asarray(right, dtype=np.float32)
    return z


def stereo_guard(left, right, out_l, out_r, coefs, limit):
    f64 = preemph_stereo_f64(left, right, coefs, limit, HOLD_S, RELEASE_S, RATE_HZ)
    model = [complex(float(a), float(b)) for a, b in zip(out_l.tolist(), out_r.tolist())]
    return guard_db(model, f64)


# Bursts of the 10 kHz tone on the left channel of stereo_limit_c32_16384: (first, end).
STEREO_BURSTS = ((2000, 4000), (9000, 11000))


@case(kernel="preemph_limit", name="stereo_limit_c32_16384", seed=485)
def stereo_limit_c32_case(rng):
    n = 16384
    limit = 0.9
    right = noise(rng, n, 0.3)
    left = np.zeros(n)
    for k in range(n):
        left[k] = 0.5 * math.sin(2.0 * math.pi * ((1000 * k % RATE_HZ) / RATE_HZ))
    for first, end in STEREO_BURSTS:
        for k in range(first, end):
            left[k] += 0.9 * math.sin(2.0 * math.pi * ((10000 * k % RATE_HZ) / RATE_HZ))
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    coefs = coefs_of(TAU_S)
    out_l, out_r, stats = preemph_stereo_f32(left, right, coefs, limit, HOLD_S, RELEASE_S, RATE_HZ)
    # The limiter attacks on either channel, holds and releases.
    assert stats["peak_left"] > 500 and stats["peak_right"] > 500, stats
    # The pre-emphasized noise (sigma about 1.1) keeps the limiter busy almost everywhere.
    assert stats["held"] > 500 and stats["released"] > 500, stats
    assert float(np.max(np.abs(out_l))) <= 0.9 and float(np.max(np.abs(out_r))) <= 0.9
    guard = stereo_guard(left, right, out_l, out_r, coefs, limit)
    return dict(
        params=params(TAU_S, limit, coefs, bursts=[list(b) for b in STEREO_BURSTS], guard_skip=GUARD_SKIP,
                      snr_db_guard=guard),
        arrays={"input": Array("cf32", complex_of(left, right)), "expected": Array("cf32", complex_of(out_l, out_r))},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="stereo limiter (re left, im right), 48 kHz, tau 50 us, limit 0.9, hold 0.02 s, release "
        "0.2 s; left: 1 kHz 0.5 plus 10 kHz 0.9 over the listed bursts, right: noise sigma 0.3; the "
        "envelope peak comes from either channel while limiting; exact float32 model; snr_db_guard "
        "compares with the float64 loop (D111)",
    )


@case(kernel="preemph_limit", name="stereo_equal_c32_8192", seed=486)
def stereo_equal_c32_case(rng):
    # The input of limit_f32_8192 (seed 445) in both channels.
    x = limit_input(np.random.default_rng(445))
    coefs = coefs_of(TAU_S)
    mono, mono_stats = preemph_f32(x, coefs, 0.5, HOLD_S, RELEASE_S, RATE_HZ)
    out_l, out_r, stats = preemph_stereo_f32(x, x, coefs, 0.5, HOLD_S, RELEASE_S, RATE_HZ)
    assert out_l.tobytes() == mono.tobytes() and out_r.tobytes() == mono.tobytes()
    assert stats["limited"] == mono_stats["limited"] > 500, stats
    guard = stereo_guard(x, x, out_l, out_r, coefs, 0.5)
    return dict(
        params=params(TAU_S, 0.5, coefs, guard_skip=GUARD_SKIP, snr_db_guard=guard),
        arrays={"input": Array("cf32", complex_of(x, x)), "expected": Array("cf32", complex_of(mono, mono))},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="the input of limit_f32_8192 (seed 445) in both channels, limit 0.5; the expected output "
        "is the expected output of limit_f32_8192 in both channels (the generator asserts that the "
        "stereo model gives the same bits); snr_db_guard compares with the float64 loop (D111)",
    )
