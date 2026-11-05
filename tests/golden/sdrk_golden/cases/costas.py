"""Cases for the costas kernel (carrier recovery for BPSK and QPSK, M40 section 2).

The Costas loop is a feedback system with discrete decisions (the signs of the order-4 detector and
the lock detector), so the expected values come from an exact float32 model of the arithmetic fixed
by the specification (rule R8.3 exception, as for ``stereo_pll``): the derotation with the cosine
table of ``rotate`` and its interpolation, the detector, the loop filter and the lock detector in
float32 with the ``f32model`` helpers, one operation at a time, and the phase step
``floor(v 2^32 + 0.5)`` in float64, with the rules the kernel documents in ``docs/kernels.md``:
``sign(0) = 0``, a non-finite detector output counts as 0, and a non-finite lock metric leaves the
lock detector unchanged (D603); the order-4 lock metric is ``-cos(4 arg y)`` (D601).

The signals use ``math.sin`` and ``math.cos`` one value at a time and the noise comes from
``rng.standard_normal``; nothing that feeds a committed byte uses NumPy transcendentals or BLAS, so
the files are identical on every CPU.

The regression guard of decision D111 runs the same loop in float64 (exact ``sin`` and ``cos`` of
the ``u32`` phase instead of the table, no flush of denormals) on the stored inputs.
``snr_db_guard`` is the SNR of the model output against that float64 output from symbol
``guard_skip`` on, minus 1 dB, rounded down to 0.1 dB. The float64 output is not committed: the
Rust test runs its own float64 loop.
"""

import math
import struct

import numpy as np

from .. import f32model as fm
from ..registry import Array, case
from .stereo import COS_F32_SHA256, _interp_f32

TWO32 = 1 << 32
MASK = TWO32 - 1
QUARTER = 1 << 30

# The block defaults of the costas block (D604).
LOOP_BW = 0.02
DAMPING = 0.7071
MAX_OFFSET = 0.05

LOCK_ALPHA = fm.f32(0.01)
LOCK_ON = fm.f32(0.5)
LOCK_OFF = fm.f32(0.3)
FOUR = fm.f32(4.0)
ZERO = fm.f32(0.0)
GUARD_SKIP = 0


def f32_bits(x):
    """The IEEE-754 bits of a float32 as an integer."""
    return struct.unpack("<I", struct.pack("<f", float(fm.f32(x))))[0]


def gains(loop_bw, damping):
    """The f64 loop constants of ``CostasState::new`` in the same order of operations."""
    theta = loop_bw / (damping + 1.0 / (4.0 * damping))
    d = (1.0 + (2.0 * damping) * theta) + theta * theta
    kp = (((4.0 * damping) * theta) / d) / math.tau
    ki = ((4.0 * (theta * theta)) / d) / math.tau
    return kp, ki


def _sign(x):
    if x > 0.0:
        return fm.f32(1.0)
    if x < 0.0:
        return fm.f32(-1.0)
    return ZERO


def costas_f32(x, order, kp64, ki64, max64):
    """Float32 model of section 2. Returns (output complex64, lock flags, final state)."""
    # The limit is flushed like the state it bounds (R6.3): a subnormal limit becomes 0.
    kp, ki, mx = fm.f32(kp64), fm.f32(ki64), fm.flush_denormal(fm.f32(max64))
    phase, freq, level, locked = 0, ZERO, ZERO, False
    out = np.zeros(len(x), dtype=np.complex64)
    flags = np.zeros(len(x), dtype=np.uint8)
    for n, v in enumerate(x.tolist()):
        xr, xi = fm.f32(v.real), fm.f32(v.imag)
        c = _interp_f32(phase)
        s = _interp_f32((phase - QUARTER) & MASK)
        yr = fm.add(fm.mul(xr, c), fm.mul(xi, s))
        yi = fm.sub(fm.mul(xi, c), fm.mul(xr, s))
        out[n] = complex(float(yr), float(yi))
        if order == 2:
            e = fm.mul(yr, yi)
        else:
            e = fm.sub(fm.mul(_sign(yr), yi), fm.mul(_sign(yi), yr))
        if not fm.is_finite(e):
            e = ZERO
        freq = fm.flush_denormal(fm.add(freq, fm.mul(ki, e)))
        if freq > mx:
            freq = mx
        if freq < -mx:
            freq = -mx
        step = fm.add(freq, fm.mul(kp, e))
        phase = (phase + (int(math.floor(float(step) * 4294967296.0 + 0.5)) & MASK)) & MASK
        a, b = fm.mul(yr, yr), fm.mul(yi, yi)
        p, d = fm.add(a, b), fm.sub(a, b)
        if order == 2:
            m = fm.div(d, p)
        else:
            m = fm.div(fm.sub(fm.mul(fm.mul(FOUR, a), b), fm.mul(d, d)), fm.mul(p, p))
        if fm.is_finite(m):
            level = fm.flush_denormal(fm.add(level, fm.mul(LOCK_ALPHA, fm.sub(m, level))))
            if level > LOCK_ON:
                locked = True
            elif level < LOCK_OFF:
                locked = False
        flags[n] = 1 if locked else 0
    return out, flags, (phase, freq, level, locked)


def costas_f64(x, order, kp, ki, mx):
    """The loop of section 2 in float64 with exact sine and cosine of the ``u32`` phase."""
    phase, freq = 0, 0.0
    out = []
    for v in x.tolist():
        theta = 2.0 * math.pi * (phase / 4294967296.0)
        c, s = math.cos(theta), math.sin(theta)
        xr, xi = float(v.real), float(v.imag)
        yr, yi = xr * c + xi * s, xi * c - xr * s
        out.append(complex(yr, yi))
        if order == 2:
            e = yr * yi
        else:
            sg = lambda u: 1.0 if u > 0.0 else (-1.0 if u < 0.0 else 0.0)  # noqa: E731
            e = sg(yr) * yi - sg(yi) * yr
        freq = min(max(freq + ki * e, -mx), mx)
        phase = (phase + int(math.floor((freq + kp * e) * 4294967296.0 + 0.5))) & MASK
    return out


def guard_db(model, f64_out):
    """The SNR of the model against the float64 loop minus 1 dB, rounded down to 0.1 dB."""
    ref = f64_out[GUARD_SKIP:]
    test = [complex(v) for v in model.tolist()[GUARD_SKIP:]]
    num = math.fsum(abs(r) ** 2 for r in ref)
    den = math.fsum(abs(r - t) ** 2 for r, t in zip(ref, test))
    snr = math.inf if den == 0.0 else 10.0 * math.log10(num / den)
    return math.floor((snr - 1.0) * 10.0) / 10.0


def psk_signal(rng, n, order, phase0, offset, ebn0_db):
    """Symbols (Gray, bit 1 positive) rotated by ``phase0 + 2 pi offset k`` plus complex noise.

    BPSK ``2 b - 1``; QPSK ``((2 b0 - 1) + j (2 b1 - 1)) / sqrt(2)``; unit symbol energy, so the
    noise has ``sigma^2 = 1 / (2 (bits per symbol) Eb/N0)`` per component.
    """
    bps = 1 if order == 2 else 2
    data = rng.integers(0, 2, n * bps).tolist()
    nre, nim = rng.standard_normal(n).tolist(), rng.standard_normal(n).tolist()
    sigma = math.sqrt(1.0 / (2.0 * bps * 10.0 ** (ebn0_db / 10.0)))
    r = 1.0 / math.sqrt(2.0)
    x = []
    for k in range(n):
        if order == 2:
            sym = complex(2 * data[k] - 1, 0.0)
        else:
            sym = complex((2 * data[2 * k] - 1) * r, (2 * data[2 * k + 1] - 1) * r)
        # The phase in turns, reduced exactly before the scalar sine and cosine.
        turns = (phase0 / math.tau + offset * k) % 1.0
        w = complex(math.cos(math.tau * turns), math.sin(math.tau * turns))
        v = sym * w
        x.append(complex(v.real + sigma * nre[k], v.imag + sigma * nim[k]))
    return np.array(x, dtype=np.complex128).astype(np.complex64), sigma


def run_case(rng, order, offset, ebn0_db, notes):
    n = 8192
    phase0 = float(rng.uniform(0.0, math.tau))
    x, sigma = psk_signal(rng, n, order, phase0, offset, ebn0_db)
    kp, ki = gains(LOOP_BW, DAMPING)
    out, flags, (phase, freq, level, locked) = costas_f32(x, order, kp, ki, MAX_OFFSET)
    guard = guard_db(out, costas_f64(x, order, float(fm.f32(kp)), float(fm.f32(ki)), float(fm.f32(MAX_OFFSET))))
    # The loop must have locked and stay locked over the second half.
    assert locked and all(flags[n // 2:]), order
    if offset != 0.0:
        assert abs(float(freq) - offset) < 0.002, float(freq)
    return dict(
        params={
            "order": order,
            "loop_bw": LOOP_BW,
            "damping": DAMPING,
            "max_offset": MAX_OFFSET,
            "kp_bits": f32_bits(kp),
            "ki_bits": f32_bits(ki),
            "max_bits": f32_bits(MAX_OFFSET),
            "phase0_rad": phase0,
            "offset_cycles_per_symbol": offset,
            "ebn0_db": ebn0_db,
            "sigma": sigma,
            "final_phase": phase,
            "final_freq_bits": f32_bits(freq),
            "final_level_bits": f32_bits(level),
            "final_locked": 1 if locked else 0,
            "first_locked": int(np.argmax(flags)),
            "cos_f32_sha256": COS_F32_SHA256,
            "guard_skip": GUARD_SKIP,
            "snr_db_guard": guard,
        },
        arrays={"input": Array("cf32", x), "expected": Array("cf32", out), "locked": Array("u8", flags)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes=notes,
    )


@case(kernel="costas", name="bpsk_lock_c32_8192", seed=550)
def bpsk_lock(rng):
    return run_case(
        rng, 2, 0.0, 10.0,
        "BPSK (2 b - 1), one sample per symbol, rotated by a random start phase phase0_rad, complex "
        "noise for Eb/N0 10 dB; order 2 with the block defaults (loop_bw 0.02, damping 0.7071, "
        "max_offset 0.05); exact float32 model over the whole input; locked is the lock flag after "
        "every symbol, first_locked the first symbol with the flag on; snr_db_guard compares with "
        "the float64 loop (D111)",
    )


@case(kernel="costas", name="qpsk_lock_c32_8192", seed=551)
def qpsk_lock(rng):
    return run_case(
        rng, 4, 0.0, 10.0,
        "QPSK ((2 b0 - 1) + j (2 b1 - 1)) / sqrt(2), one sample per symbol, rotated by a random start "
        "phase phase0_rad, complex noise for Eb/N0 10 dB; order 4 with the block defaults; exact "
        "float32 model over the whole input; locked is the lock flag after every symbol; "
        "snr_db_guard compares with the float64 loop (D111)",
    )


@case(kernel="costas", name="qpsk_offset_c32_8192", seed=552)
def qpsk_offset(rng):
    return run_case(
        rng, 4, 0.01, 12.0,
        "QPSK as qpsk_lock_c32_8192 with a carrier offset of 0.01 cycles per symbol (symbol k rotated "
        "by phase0 + 2 pi 0.01 k), Eb/N0 12 dB; order 4 with the block defaults; the frequency "
        "integrator ends within 0.002 of the offset (asserted by the generator); snr_db_guard "
        "compares with the float64 loop (D111)",
    )
