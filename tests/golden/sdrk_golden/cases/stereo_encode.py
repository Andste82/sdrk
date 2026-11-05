"""Cases for the ``stereo_encode`` kernel (stereo audio to the FM multiplex signal, M36 section 2).

The sines come from the cosine table of ``rotate`` with its interpolation, so the expected
values come from the exact float32 model (``f32model``) of the arithmetic fixed by the
specification, with the table and the interpolation of ``cases/stereo.py`` (the model of
``stereo_pll``): ``s1 = sin(phi)`` at the phase minus a quarter turn, ``s2 = sin(2 phi)`` at the
doubled phase minus a quarter turn, ``m = (L + R) 0.5``, ``d = (L - R) 0.5``,
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
from .stereo import COS_F32_SHA256, _interp_f32

TWO32 = 1 << 32
MASK = TWO32 - 1
QUARTER = 1 << 30

RATE_HZ = 192000
PILOT_HZ = 19000.0
MAIN = 0.9
PILOT = 0.09
PHASE0 = 0x40000000


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
        s1 = _interp_f32((phase - QUARTER) & MASK)
        s2 = _interp_f32(((2 * phase) - QUARTER) & MASK)
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
        "0x40000000, main 0.9, pilot 0.09; exact float32 model with the cosine table of rotate "
        "and its interpolation; snr_db_guard compares with the float64 formula with exact sines "
        "(D34). The seed is not used (no random input)",
    )
