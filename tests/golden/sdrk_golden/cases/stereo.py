"""Cases for the FM stereo kernels (``stereo_pll`` and ``stereo_matrix``).

The pilot loop is a feedback system with a discrete lock decision, so the expected values come from
exact models of the arithmetic fixed by the specification (rule R8.3 exception, as for ``squelch``):
the float32 model (``f32model``) for ``f32`` and the integer model for ``Q15``. Every value that
feeds a committed byte is computed with plain Python, ``math.fsum``-free scalar arithmetic and the
scalar ``sin`` and ``cos`` of ``scalar_math`` (no BLAS, no SIMD transcendental).

The regression guard of decision D111 runs the same loop in float64 (exact ``sin`` and ``cos``
instead of the tables, no flush of denormals) on the inputs of every ``lock_*`` and ``fade_*``
case. ``snr_db_guard`` is the SNR of the difference channel of the model under test against that
float64 output from sample ``guard_skip`` on, minus 1 dB, rounded down to 0.1 dB. The float64
output itself is not committed: the Rust test runs its own float64 loop on the stored inputs.
"""

import hashlib
import math
import struct

import numpy as np

from .. import f32model as fm
from .. import q15, scalar_math
from ..registry import Array, case

TWO32 = 1 << 32
MASK = TWO32 - 1
QUARTER = 1 << 30

PILOT_HZ = 19000.0
PILOT_TRUE_HZ = 19000.5
PILOT_START_RAD = 1.0
PILOT_AMP = 0.1
NATURAL_HZ = 50.0
RATE_HZ = 192000
GUARD_SKIP = 8192
FADE_FIRST, FADE_END = 8192, 12288


# ---------------------------------------------------------------------------------------------
# Tables and coefficients
# ---------------------------------------------------------------------------------------------


def _cos_tables():
    c64 = scalar_math.cos(2.0 * np.pi * np.arange(4097) / 4096)
    t32 = c64.astype(np.float32)
    tq = [int(v) for v in q15.from_float(c64)]
    return t32, tq


COS_F32, COS_Q15 = _cos_tables()
COS_F32_LIST = [fm.f32(v) for v in COS_F32.tolist()]
COS_F32_SHA256 = hashlib.sha256(COS_F32.astype("<f4").tobytes()).hexdigest()
COS_Q15_SHA256 = hashlib.sha256(np.asarray(COS_Q15, dtype="<i2").tobytes()).hexdigest()


def bits(x):
    """The IEEE-754 bits of a float64 as an integer (exact in JSON)."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def coefficients(pilot_hz, natural_hz, pilot_level, rate_hz):
    """The f64 coefficients of ``StereoPllState::new`` in the same order of operations."""
    fs = float(rate_hz)
    two_pi = 2.0 * math.pi
    kd = pilot_level / 2.0
    zeta = 1.0 / math.sqrt(2.0)
    wn = two_pi * natural_hz / fs
    return {
        "kp": 2.0 * zeta * wn / (kd * two_pi),
        "ki": wn * wn / (kd * two_pi),
        "imax": 50.0 / fs,
        "beta": 1.0 - math.exp(-1.0 / (fs * 0.05)),
        "on": 0.4 * kd,
        "off": 0.3 * kd,
        "step0": int(math.floor(pilot_hz / fs * 4294967296.0 + 0.5)),
    }


def coefficient_params(co):
    return {
        "kp_bits": bits(co["kp"]),
        "ki_bits": bits(co["ki"]),
        "imax_bits": bits(co["imax"]),
        "beta_bits": bits(co["beta"]),
        "on_bits": bits(co["on"]),
        "off_bits": bits(co["off"]),
        "step0": co["step0"],
    }


# ---------------------------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------------------------


def _turns(num, den, n):
    """``frac(num * k / den)`` for k in 0..n with exact integer arithmetic, as float64."""
    return np.array([((num * k) % den) / den for k in range(n)], dtype=np.float64)


def signals(rng, n, fade):
    """(mpx, pilot) as float64: the multiplex signal of section 2 of the milestone plus noise.

    L is a 1 kHz tone, R a 3 kHz tone (amplitude 1), the pilot 0.1 at 19 000.5 Hz with the start
    phase 1.0 rad; the pilot amplitude of the ``fade`` variant falls linearly from 0.1 to 0 over
    samples 8192..12287 and stays 0. ``mpx`` gets real noise sigma 0.01, ``pilot`` (the pilot
    term alone) independent noise sigma 0.005.
    """
    noise_mpx = rng.standard_normal(n) * 0.01
    noise_pilot = rng.standard_normal(n) * 0.005
    left = scalar_math.sin(2.0 * np.pi * _turns(1000, RATE_HZ, n))
    right = scalar_math.sin(2.0 * np.pi * _turns(3000, RATE_HZ, n))
    theta = 2.0 * np.pi * _turns(38001, 2 * RATE_HZ, n) + PILOT_START_RAD
    pilot_tone = scalar_math.sin(theta)
    carrier38 = scalar_math.sin(2.0 * theta)
    amp = np.full(n, PILOT_AMP)
    if fade:
        k = np.arange(n)
        ramp = PILOT_AMP * (1.0 - (k - FADE_FIRST) / (FADE_END - FADE_FIRST))
        amp = np.where(k < FADE_FIRST, PILOT_AMP, np.where(k < FADE_END, ramp, 0.0))
    mpx = 0.45 * (left + right) + 0.45 * (left - right) * carrier38 + amp * pilot_tone + noise_mpx
    pilot = amp * pilot_tone + noise_pilot
    return mpx, pilot


# ---------------------------------------------------------------------------------------------
# float32 model
# ---------------------------------------------------------------------------------------------

F32_ONE_OVER_2_20 = fm.f32(1.0 / 1048576.0)


def _interp_f32(p):
    idx = p >> 20
    f = fm.mul(fm.f32(p & 0xFFFFF), F32_ONE_OVER_2_20)
    a, b = COS_F32_LIST[idx], COS_F32_LIST[idx + 1]
    return fm.add(a, fm.mul(fm.sub(b, a), f))


def stereo_pll_f32(mpx, pilot, co, disable_at=None, enable_at=None):
    """Float32 model of section 3. Returns (output as complex64 pairs, stereo flags)."""
    kp, ki, imax = fm.f32(co["kp"]), fm.f32(co["ki"]), fm.f32(co["imax"])
    beta, on, off = fm.f32(co["beta"]), fm.f32(co["on"]), fm.f32(co["off"])
    step0 = co["step0"]
    phase, integ, level = 0, fm.f32(0.0), fm.f32(0.0)
    stereo, enabled = False, True
    out = np.zeros(len(mpx), dtype=np.complex64)
    flags = np.zeros(len(mpx), dtype=np.uint8)
    zero = fm.f32(0.0)
    for n, (x, p) in enumerate(zip(mpx.tolist(), pilot.tolist())):
        if disable_at is not None and n == disable_at:
            enabled = False
        if enable_at is not None and n == enable_at:
            enabled = True
        x, p = fm.f32(x), fm.f32(p)
        c = _interp_f32(phase)
        s = _interp_f32((phase - QUARTER) & MASK)
        s2 = _interp_f32(((phase * 2) - QUARTER) & MASK)
        if fm.is_finite(p):
            e = fm.mul(p, c)
            q = fm.mul(p, s)
            level = fm.flush_denormal(fm.add(level, fm.mul(beta, fm.sub(q, level))))
            if level >= on:
                stereo = True
            elif level < off:
                stereo = False
        else:
            e = zero
        diff = fm.mul(x, s2) if (stereo and enabled) else zero
        out[n] = complex(float(x), float(diff))
        flags[n] = 1 if stereo else 0
        integ = fm.flush_denormal(fm.add(integ, fm.mul(ki, e)))
        if integ < -imax:
            integ = -imax
        elif integ > imax:
            integ = imax
        f = fm.add(fm.mul(kp, e), integ)
        corr = int(math.floor(float(f) * 4294967296.0 + 0.5))
        phase = (phase + step0 + corr) & MASK
    return out, flags


# ---------------------------------------------------------------------------------------------
# integer model
# ---------------------------------------------------------------------------------------------


def _interp_q15(p):
    idx = p >> 20
    f15 = (p >> 5) & 0x7FFF
    a, b = COS_Q15[idx], COS_Q15[idx + 1]
    return int(q15.sat16(a + int(q15.round_shift((b - a) * f15, 15))))


def q15_constants(co):
    return {
        "kp": int(math.floor(co["kp"] * 2.0**34 + 0.5)),
        "ki": int(math.floor(co["ki"] * 2.0**34 + 0.5)),
        "imax": int(math.floor(co["imax"] * 2.0**64 + 0.5)),
        "beta": int(math.floor(co["beta"] * 2.0**31 + 0.5)),
        "on": int(math.floor(co["on"] * 2.0**61 + 0.5)),
        "off": int(math.floor(co["off"] * 2.0**61 + 0.5)),
    }


def stereo_pll_q15(xs, ps, co):
    """Integer model of section 3: ``xs``, ``ps`` are Q15 raw values. Returns (n, 2) int64."""
    k = q15_constants(co)
    step0 = co["step0"]
    phase, integ, level, stereo = 0, 0, 0, False
    out = np.zeros((len(xs), 2), dtype=np.int64)
    flags = np.zeros(len(xs), dtype=np.uint8)
    for n, (x, p) in enumerate(zip(xs.tolist(), ps.tolist())):
        c = _interp_q15(phase)
        s = _interp_q15((phase - QUARTER) & MASK)
        s2 = _interp_q15(((phase * 2) - QUARTER) & MASK)
        e, q = p * c, p * s
        level = level + k["beta"] * (q - int(q15.round_shift(level, 31)))
        if level >= k["on"]:
            stereo = True
        elif level < k["off"]:
            stereo = False
        diff = int(q15.sat16(int(q15.round_shift(x * s2, 15)))) if stereo else 0
        out[n] = (x, diff)
        flags[n] = 1 if stereo else 0
        integ = min(max(integ + k["ki"] * e, -k["imax"]), k["imax"])
        f = k["kp"] * e + integ
        phase = (phase + step0 + (int(q15.round_shift(f, 32)) & MASK)) & MASK
    return out, flags


# ---------------------------------------------------------------------------------------------
# float64 regression guard (decision D111)
# ---------------------------------------------------------------------------------------------


def stereo_pll_f64(mpx, pilot, co):
    """The loop of section 3 in float64 with exact sine and cosine; returns the ``im`` channel."""
    kp, ki, imax, beta = co["kp"], co["ki"], co["imax"], co["beta"]
    on, off, step0 = co["on"], co["off"], co["step0"]
    phase, integ, level, stereo = 0, 0.0, 0.0, False
    out = []
    for x, p in zip(mpx.tolist(), pilot.tolist()):
        theta = 2.0 * math.pi * (phase / 4294967296.0)
        s2 = math.sin(2.0 * theta)
        if math.isfinite(p):
            e = p * math.cos(theta)
            q = p * math.sin(theta)
            level = level + beta * (q - level)
            if level >= on:
                stereo = True
            elif level < off:
                stereo = False
        else:
            e = 0.0
        out.append(x * s2 if stereo else 0.0)
        integ = min(max(integ + ki * e, -imax), imax)
        f = kp * e + integ
        phase = (phase + step0 + int(math.floor(f * 4294967296.0 + 0.5))) & MASK
    return out


def snr_db(reference, test):
    """``10 log10(sum ref^2 / sum (ref - test)^2)`` with exact sums."""
    num = math.fsum(r * r for r in reference)
    den = math.fsum((r - t) * (r - t) for r, t in zip(reference, test))
    return math.inf if den == 0.0 else 10.0 * math.log10(num / den)


def guard_db(model_im, f64_im):
    """The lowest SNR of the model under test minus 1 dB, rounded down to 0.1 dB."""
    snr = snr_db(f64_im[GUARD_SKIP:], model_im[GUARD_SKIP:])
    return math.floor((snr - 1.0) * 10.0) / 10.0


# ---------------------------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------------------------


def pll_params(pilot_level, co, **extra):
    base = {
        "rate_hz": RATE_HZ,
        "pilot_hz": PILOT_HZ,
        "natural_hz": NATURAL_HZ,
        "pilot_level": pilot_level,
        "cos_f32_sha256": COS_F32_SHA256,
        "cos_q15_sha256": COS_Q15_SHA256,
        "guard_skip": GUARD_SKIP,
    }
    base.update(coefficient_params(co))
    base.update(extra)
    return base


@case(kernel="stereo_pll", name="lock_f32_16384", seed=438)
def lock_f32(rng):
    co = coefficients(PILOT_HZ, NATURAL_HZ, 0.1, RATE_HZ)
    mpx, pilot = signals(rng, 16384, fade=False)
    x32, p32 = mpx.astype(np.float32), pilot.astype(np.float32)
    out, _ = stereo_pll_f32(x32.astype(np.float64), p32.astype(np.float64), co)
    guard = guard_db(out.imag.tolist(), stereo_pll_f64(x32.astype(np.float64), p32.astype(np.float64), co))
    return dict(
        params=pll_params(0.1, co, snr_db_guard=guard),
        arrays={"mpx": Array("f32", x32), "pilot": Array("f32", p32), "expected": Array("cf32", out)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="192 kHz, natural_hz 50, pilot_level 0.1; mpx = multiplex signal of section 2 (L 1 kHz, "
        "R 3 kHz, amplitude 1, pilot 0.1 at 19000.5 Hz, start phase 1.0 rad) plus noise sigma 0.01; "
        "pilot = the pilot term plus noise sigma 0.005; exact float32 model; snr_db_guard compares "
        "the difference channel from sample 8192 with the float64 loop (D111)",
    )


@case(kernel="stereo_pll", name="lock_q15_16384", seed=439)
def lock_q15(rng):
    co = coefficients(PILOT_HZ, NATURAL_HZ, 0.0625, RATE_HZ)
    mpx, pilot = signals(rng, 16384, fade=False)
    xs, ps = q15.from_float(mpx * 0.625), q15.from_float(pilot * 0.625)
    out, _ = stereo_pll_q15(xs, ps, co)
    f64 = stereo_pll_f64(xs / 32768.0, ps / 32768.0, co)
    guard = guard_db((out[:, 1] / 32768.0).tolist(), f64)
    return dict(
        params=pll_params(0.0625, co, snr_db_guard=guard),
        arrays={"mpx": Array("i16", xs), "pilot": Array("i16", ps), "expected": Array("ci16", out)},
        tolerance={"class": "bit_exact"},
        notes="the signals of lock_f32_16384 (own noise draw) scaled by 0.625 and quantized; "
        "pilot_level 0.0625; exact integer model (Q30 detector, Q61 level, integrator in 2^-64 turn)",
    )


@case(kernel="stereo_pll", name="fade_f32_16384", seed=440)
def fade_f32(rng):
    co = coefficients(PILOT_HZ, NATURAL_HZ, 0.1, RATE_HZ)
    mpx, pilot = signals(rng, 16384, fade=True)
    x32, p32 = mpx.astype(np.float32), pilot.astype(np.float32)
    out, flags = stereo_pll_f32(x32.astype(np.float64), p32.astype(np.float64), co)
    guard = guard_db(out.imag.tolist(), stereo_pll_f64(x32.astype(np.float64), p32.astype(np.float64), co))
    return dict(
        params=pll_params(0.1, co, snr_db_guard=guard),
        arrays={
            "mpx": Array("f32", x32),
            "pilot": Array("f32", p32),
            "expected": Array("cf32", out),
            "stereo": Array("u8", flags),
        },
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="as lock_f32_16384, the pilot amplitude falls linearly from 0.1 to 0 over samples "
        "8192-12287 and stays 0; stereo holds the lock decision after each sample (0 or 1)",
    )


@case(kernel="stereo_pll", name="disable_f32_8192", seed=441)
def disable_f32(rng):
    co = coefficients(PILOT_HZ, NATURAL_HZ, 0.1, RATE_HZ)
    mpx, pilot = signals(rng, 8192, fade=False)
    x32, p32 = mpx.astype(np.float32), pilot.astype(np.float32)
    out, _ = stereo_pll_f32(x32.astype(np.float64), p32.astype(np.float64), co, disable_at=4096)
    return dict(
        params=pll_params(0.1, co, disable_at=4096),
        arrays={"mpx": Array("f32", x32), "pilot": Array("f32", p32), "expected": Array("cf32", out)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="as lock_f32 with 8192 samples; set_enabled(false) before sample 4096 (the "
        "difference channel is zero from there on while the loop keeps running)",
    )


@case(kernel="stereo_matrix", name="c32_4096", seed=442)
def matrix_c32(rng):
    x = rng.uniform(-0.6, 0.6, (4096, 2)).astype(np.float32)
    s, d = x[:, 0], x[:, 1]
    d2 = d + d
    expected = np.stack([s + d2, s - d2], axis=1).astype(np.float32)
    return dict(
        params={},
        arrays={
            "input": Array("cf32", x[:, 0] + 1j * x[:, 1]),
            "expected": Array("cf32", expected[:, 0] + 1j * expected[:, 1]),
        },
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="uniform noise in +-0.6 per component; l = s + (d + d), r = s - (d + d) in float32",
    )


@case(kernel="stereo_matrix", name="cq15_4096", seed=443)
def matrix_cq15(rng):
    x = rng.integers(-32768, 32768, (4096, 2)).astype(np.int64)
    s, d = x[:, 0], x[:, 1]
    expected = np.stack([q15.sat16(s + 2 * d), q15.sat16(s - 2 * d)], axis=1)
    return dict(
        params={},
        arrays={"input": Array("ci16", x), "expected": Array("ci16", expected)},
        tolerance={"class": "bit_exact"},
        notes="uniform raw values over the whole range (saturates often); l = sat16(s + 2d), "
        "r = sat16(s - 2d) with d2 exact in int64",
    )


@case(kernel="stereo_pll", name="disable_enable_f32_16384", seed=454)
def disable_enable_f32(rng):
    co = coefficients(PILOT_HZ, NATURAL_HZ, 0.1, RATE_HZ)
    mpx, pilot = signals(rng, 16384, fade=False)
    x32, p32 = mpx.astype(np.float32), pilot.astype(np.float32)
    out, _ = stereo_pll_f32(
        x32.astype(np.float64), p32.astype(np.float64), co, disable_at=10000, enable_at=13000
    )
    return dict(
        params=pll_params(0.1, co, disable_at=10000, enable_at=13000),
        arrays={"mpx": Array("f32", x32), "pilot": Array("f32", p32), "expected": Array("cf32", out)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="not in the table of the milestone (decision D113): as lock_f32_16384, set_enabled(false) "
        "before sample 10000 and set_enabled(true) before sample 13000, both after the lock "
        "(about sample 5000), so the case shows the zeroed and the restored difference channel",
    )
