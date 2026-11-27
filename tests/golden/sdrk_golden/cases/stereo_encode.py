"""Cases for the ``stereo_encode`` kernel (stereo audio to the FM multiplex signal, M36 section 2).

The sines are those of the ``C32`` oscillator of ``oscillate`` (D3006), so the expected values
come from the exact float32 model (``f32model``) of the arithmetic fixed by the specification,
with the cosine table of ``cases/stereo.py``: the second-order Taylor step from the table point
``a`` below the phase, ``sin(a + d) = S + d (C - S h)`` with ``C = cos a``, ``S = sin a`` (the
entries ``i`` and ``(i + 3072) mod 4096`` of the top 12 bits ``i``), ``d`` the low 20 bits times
``2 pi / 2^32`` rounded to float32 and ``h = d / 2``; ``s1 = sin(phi)``, ``s2 = sin(2 phi)`` at the
doubled phase, ``m = (L + R) 0.5``, ``d = (L - R) 0.5``,
``mpx = (main m + (main d) s2) + pilot s1``. The tones use ``math.sin`` of exactly reduced
phases, so the files are identical on every CPU.

``snr_db_guard`` (the accuracy guard of decision D34, as D111 for the loops) is the SNR of the
model output against the same formula in float64 with exact sines of the exact phase, minus 1 dB,
rounded down to 0.1 dB. The float64 output is not committed; the Rust test computes its own.
"""

import math

import numpy as np

from .. import f32model as fm
from ..registry import Array, case
from .rotate import increment
from .stereo import COS_F32_LIST, COS_F32_SHA256

TWO32 = 1 << 32
MASK = TWO32 - 1

RATE_HZ = 192000
PILOT_HZ = 19000.0
MAIN = 0.9
PILOT = 0.09
PHASE0 = 0x40000000

# ``2 pi / 2^32`` rounded to float32: the angle of one unit of the phase (``PHASE_UNIT_F32``).
PHASE_UNIT_F32 = fm.f32(2.0 * math.pi / TWO32)
HALF = fm.f32(0.5)


def _nco_sin_f32(p):
    """The sine of the ``C32`` oscillator of ``oscillate`` at the phase ``p``: the Taylor step
    ``S + d (C - S h)`` in float32 without FMA."""
    idx = p >> 20
    c = COS_F32_LIST[idx]
    s = COS_F32_LIST[(idx + 3072) & 4095]
    d = fm.mul(fm.f32(p & 0xFFFFF), PHASE_UNIT_F32)
    h = fm.mul(d, HALF)
    return fm.add(s, fm.mul(d, fm.sub(c, fm.mul(s, h))))


def tone(n, freq, amp):
    """``amp sin(2 pi frac(freq k / fs))`` with the fraction in exact integer arithmetic, float32."""
    return np.array(
        [amp * math.sin(2.0 * math.pi * ((freq * k % RATE_HZ) / RATE_HZ)) for k in range(n)],
        dtype=np.float32,
    )


def encode_f32(left, right, main, pilot, inc, phase0):
    """Float32 model of ``stereo_encode``; returns the output and the phase after the last sample."""
    main = fm.f32(main)
    pilot = fm.f32(pilot)
    half = fm.f32(0.5)
    phase = phase0
    out = np.zeros(len(left), dtype=np.float32)
    for i, (sl, sr) in enumerate(zip(np.asarray(left, dtype=np.float32).tolist(),
                                     np.asarray(right, dtype=np.float32).tolist())):
        s1 = _nco_sin_f32(phase)
        s2 = _nco_sin_f32((2 * phase) & MASK)
        m = fm.mul(fm.add(sl, sr), half)
        d = fm.mul(fm.sub(sl, sr), half)
        total = fm.add(fm.mul(main, m), fm.mul(fm.mul(main, d), s2))
        out[i] = fm.add(total, fm.mul(pilot, s1))
        phase = (phase + inc) & MASK
    return out, phase


def encode_f64(left, right, main, pilot, inc, phase0):
    """The same formula in float64 with exact sines of the exact phase."""
    out = []
    for i, (sl, sr) in enumerate(zip(np.asarray(left, dtype=np.float64).tolist(),
                                     np.asarray(right, dtype=np.float64).tolist())):
        phi = 2.0 * math.pi * (((phase0 + i * inc) & MASK) / TWO32)
        m = (sl + sr) * 0.5
        d = (sl - sr) * 0.5
        out.append(main * m + main * d * math.sin(2.0 * phi) + pilot * math.sin(phi))
    return out


def guard_db(model, f64):
    num = math.fsum(r * r for r in f64)
    den = math.fsum((r - t) ** 2 for r, t in zip(f64, model))
    snr = 10.0 * math.log10(num / den)
    return math.floor((snr - 1.0) * 10.0) / 10.0


@case(kernel="stereo_encode", name="c32_16384", seed=487)
def c32_case(_rng):
    n = 16384
    left = tone(n, 1000, 0.6)
    right = tone(n, 3000, 0.4)
    inc = increment(PILOT_HZ / RATE_HZ)
    out, phase_end = encode_f32(left, right, float(np.float32(MAIN)), float(np.float32(PILOT)), inc, PHASE0)
    assert float(np.max(np.abs(out))) <= 0.99
    f64 = encode_f64(left, right, float(np.float32(MAIN)), float(np.float32(PILOT)), inc, PHASE0)
    guard = guard_db(out.astype(np.float64).tolist(), f64)
    x = np.zeros(n, dtype=np.complex64)
    x.real = left
    x.imag = right
    return dict(
        params={
            "rate_hz": RATE_HZ,
            "pilot_hz": PILOT_HZ,
            "main": MAIN,
            "pilot": PILOT,
            "phase0": PHASE0,
            "inc": inc,
            "phase_end": phase_end,
            "cos_f32_sha256": COS_F32_SHA256,
            "snr_db_guard": guard,
        },
        arrays={"input": Array("cf32", x), "expected": Array("f32", out)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="192 kHz; re left 1 kHz 0.6, im right 3 kHz 0.4; pilot 19 kHz with the start phase "
        "0x40000000, main 0.9, pilot 0.09; exact float32 model with the cosine table and the "
        "Taylor step of the C32 oscillate; snr_db_guard compares with the float64 formula with exact sines "
        "(D34). The seed is not used (no random input)",
    )


# ---------------------------------------------------------------------------------------------
# stereo_encode_rds (M42 section 4)
# ---------------------------------------------------------------------------------------------

RDS_MAIN = 0.87
RDS_LEVEL = 0.04


def encode_rds_f32(left, right, rds, main, pilot, level, inc, phase0):
    """Float32 model of ``stereo_encode_rds``: the multiplex of ``encode_f32`` plus
    ``(level b) s3`` with ``s3 = sin(3 phi)`` at the tripled phase, added
    last. Returns the output and the phase after the last sample."""
    base, phase_end = encode_f32(left, right, main, pilot, inc, phase0)
    level = fm.f32(level)
    out = np.zeros(len(base), dtype=np.float32)
    phase = phase0
    for i, (m, b) in enumerate(zip(base.tolist(), np.asarray(rds, dtype=np.float32).tolist())):
        s3 = _nco_sin_f32((3 * phase) & MASK)
        out[i] = fm.add(m, fm.mul(fm.mul(level, b), s3))
        phase = (phase + inc) & MASK
    return out, phase_end


def encode_rds_f64(left, right, rds, main, pilot, level, inc, phase0):
    """The same formula in float64 with exact sines of the exact phase."""
    base = encode_f64(left, right, main, pilot, inc, phase0)
    out = []
    for i, (m, b) in enumerate(zip(base, np.asarray(rds, dtype=np.float64).tolist())):
        phi = 2.0 * math.pi * (((phase0 + i * inc) & MASK) / TWO32)
        out.append(m + level * b * math.sin(3.0 * phi))
    return out


@case(kernel="stereo_encode_rds", name="c32_16384", seed=581)
def c32_rds_case(rng):
    from . import rds as rdsmod

    n = 16384
    left = tone(n, 1000, 0.6)
    right = tone(n, 3000, 0.4)
    nb = rdsmod.bits_needed(n, RATE_HZ)
    amps = (2 * rng.integers(0, 2, size=nb) - 1).astype(np.float64)
    b = rdsmod.rds_baseband(amps, 0, n, RATE_HZ).astype(np.float32)
    inc = increment(PILOT_HZ / RATE_HZ)
    main, pilot, level = (float(np.float32(v)) for v in (RDS_MAIN, PILOT, RDS_LEVEL))
    out, phase_end = encode_rds_f32(left, right, b, main, pilot, level, inc, PHASE0)
    assert float(np.max(np.abs(out))) <= 1.0
    f64 = encode_rds_f64(left, right, b, main, pilot, level, inc, PHASE0)
    guard = guard_db(out.astype(np.float64).tolist(), f64)
    x = np.zeros(n, dtype=np.complex64)
    x.real = left
    x.imag = right
    return dict(
        params={
            "rate_hz": RATE_HZ,
            "pilot_hz": PILOT_HZ,
            "main": RDS_MAIN,
            "pilot": PILOT,
            "rds": RDS_LEVEL,
            "phase0": PHASE0,
            "inc": inc,
            "phase_end": phase_end,
            "cos_f32_sha256": COS_F32_SHA256,
            "snr_db_guard": guard,
        },
        arrays={
            "input": Array("cf32", x),
            "rds": Array("f32", b),
            "expected": Array("f32", out),
        },
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="192 kHz; re left 1 kHz 0.6, im right 3 kHz 0.4; rds: the RDS baseband of "
        "cases/rds.py (rds_baseband at 192 kHz) of random amplitudes +-1, rounded to float32; "
        "pilot 19 kHz with the start phase 0x40000000, main 0.87, pilot 0.09, rds 0.04; exact "
        "float32 model of stereo_encode plus (rds b) sin(3 phi) from the oscillator of oscillate "
        "at the tripled phase, added last; snr_db_guard compares with the float64 formula with "
        "exact sines (D34)",
    )
