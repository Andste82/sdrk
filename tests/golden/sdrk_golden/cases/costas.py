"""Cases for the costas kernel (carrier recovery for BPSK, QPSK and 8PSK, M40 section 2, M40.1 section 1).

The Costas loop is a feedback system with discrete decisions (the signs of the order-4 detector and
the lock detector), so the expected values come from an exact float32 model of the arithmetic fixed
by the specification (rule R8.3 exception, as for ``stereo_pll``): the derotation with the cosine
table of ``rotate`` and its interpolation, the detector, the loop filter and the lock detector in
float32 with the ``f32model`` helpers, one operation at a time, and the phase step
``floor(v 2^32 + 0.5)`` in float64, with the rules the kernel documents in ``docs/kernels.md``:
``sign(0) = 0``, a non-finite detector output counts as 0, and a non-finite lock metric leaves the
lock detector unchanged (D603); the order-4 lock metric is ``-cos(4 arg y)`` (D601). Order 8
(M40.1) decides the 8PSK point ``d`` of the half-open octant of ``y`` from the signs and the
comparison of the magnitudes (D641), measures ``e = y.im dr - y.re di`` and uses the metric
``cos(8 err) = T4(u)``, computed as ``((8 w) (w - 1)) + 1`` with ``w = u u``,
``u = ((r r) / p) 2 - 1`` and ``r = y.re dr + y.im di`` (D640, D659). 16-QAM (M44 section 3)
decides the grid point ``d`` of ``y`` per axis (``floor(v (sqrt E / 2) + m / 2)`` clamped to
``0 .. m - 1``, NaN to 0, D1202), measures the exact phase ``e = atan2f(i, r)`` of ``y conj d``
with ``r = y.re dr + y.im di`` and ``i = y.im dr - y.re di`` (the ``atan2f`` of FreeBSD's msun,
the algorithm of the Rust ``libm`` crate, transcribed operation by operation in ``atan2f.py``,
D1207) and uses the decision error ``c = 1 - g ((ex ex) + (ey ey))`` with ``ex = y.re - dr``,
``ey = y.im - di`` and ``g = 6 / delta^2 = 1.5 E`` (15 and 63), at least -1 and none for
``y = 0`` (D1400).

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
from ..atan2f import atan2f
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
TWO = fm.f32(2.0)
EIGHT = fm.f32(8.0)
ONE = fm.f32(1.0)
ZERO = fm.f32(0.0)
# The parts of the 8PSK points exp(j (2k + 1) pi / 8), rounded once to float32.
C8 = fm.f32(math.cos(math.pi / 8.0))
S8 = fm.f32(math.sin(math.pi / 8.0))
# The Gray mapping of M40.1 section 1: g = b0 4 + b1 2 + b2 -> symbol index k.
GRAY_TO_INDEX8 = (0, 1, 3, 2, 7, 6, 4, 5)
GUARD_SKIP = 0

# The square QAM of M44 section 1: levels per axis m and the power E of the integer grid.
QAM = {"qam16": (4, 10), "qam64": (8, 42)}


def qam_levels_f32(name):
    """The float32 level values ``(2 i - (m - 1)) / sqrt E`` in level-index order."""
    m, e = QAM[name]
    return [fm.f32((2 * i - (m - 1)) / math.sqrt(e)) for i in range(m)]


def qam_decide_f32(name, v):
    """The grid level value of one float32 component (D1202): ``floor(v (sqrt E / 2) + m / 2)``
    clamped to ``0 .. m - 1`` (NaN to level 0)."""
    m, e = QAM[name]
    t = fm.add(fm.mul(v, fm.f32(math.sqrt(e) / 2.0)), fm.f32(m / 2.0))
    if np.isnan(t) or t < 0.0:
        t = fm.f32(0.0)
    if t > m - 1:
        t = fm.f32(m - 1)
    return qam_levels_f32(name)[int(math.floor(float(t)))]


def qam_decide_f64(name, v):
    """The grid level of a float64 component: nearest level, the same boundaries rule."""
    m, e = QAM[name]
    t = min(max(v * (math.sqrt(e) / 2.0) + m / 2.0, 0.0), m - 1)
    return (2 * int(math.floor(t)) - (m - 1)) / math.sqrt(e)


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


def _octant8(yr, yi):
    """The index of the half-open octant ``[k pi / 4, (k + 1) pi / 4)`` of ``y`` (D641).

    The quadrant from the signs (``re > 0, im >= 0``; ``re <= 0, im > 0``; ``re < 0, im <= 0``;
    ``re >= 0, im < 0``), the octant from the magnitudes; zero and NaN give 0. Works on float32
    and float64 values alike (the comparisons are exact).
    """
    ar, ai = abs(yr), abs(yi)
    if yr > 0.0 and yi >= 0.0:
        return 0 if ar > ai else 1
    if yr <= 0.0 and yi > 0.0:
        return 2 if ai > ar else 3
    if yr < 0.0 and yi <= 0.0:
        return 4 if ar > ai else 5
    if yr >= 0.0 and yi < 0.0:
        return 6 if ai > ar else 7
    return 0


def _point8_f32(k):
    """The float32 parts of the 8PSK point ``k``."""
    return (
        (C8, S8), (S8, C8), (-S8, C8), (-C8, S8), (-C8, -S8), (-S8, -C8), (S8, -C8), (C8, -S8)
    )[k]


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
        elif order == 4:
            e = fm.sub(fm.mul(_sign(yr), yi), fm.mul(_sign(yi), yr))
        elif order == 8:
            dr, di = _point8_f32(_octant8(yr, yi))
            e = fm.sub(fm.mul(yi, dr), fm.mul(yr, di))
        else:
            dr, di = qam_decide_f32(order, yr), qam_decide_f32(order, yi)
            r = fm.add(fm.mul(yr, dr), fm.mul(yi, di))
            e = atan2f(fm.sub(fm.mul(yi, dr), fm.mul(yr, di)), r)
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
        elif order == 4:
            m = fm.div(fm.sub(fm.mul(fm.mul(FOUR, a), b), fm.mul(d, d)), fm.mul(p, p))
        elif order == 8:
            dr, di = _point8_f32(_octant8(yr, yi))
            r = fm.add(fm.mul(yr, dr), fm.mul(yi, di))
            u = fm.sub(fm.mul(fm.div(fm.mul(r, r), p), TWO), ONE)
            w = fm.mul(u, u)
            m = fm.add(fm.mul(fm.mul(EIGHT, w), fm.sub(w, ONE)), ONE)
        else:
            # The decision error 1 - g |y - d|^2, g = 6 / delta^2 = 1.5 E, at least -1; no metric
            # for y = 0 (D1400).
            dr, di = qam_decide_f32(order, yr), qam_decide_f32(order, yi)
            ex, ey = fm.sub(yr, dr), fm.sub(yi, di)
            e2 = fm.add(fm.mul(ex, ex), fm.mul(ey, ey))
            m = fm.sub(ONE, fm.mul(fm.f32(1.5 * QAM[order][1]), e2))
            if float(yr) == 0.0 and float(yi) == 0.0:
                m = fm.f32(math.nan)
            elif fm.is_finite(m) and m < -ONE:
                m = -ONE
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
        elif order == 4:
            sg = lambda u: 1.0 if u > 0.0 else (-1.0 if u < 0.0 else 0.0)  # noqa: E731
            e = sg(yr) * yi - sg(yi) * yr
        elif order == 8:
            a8 = (2 * _octant8(yr, yi) + 1) * math.pi / 8.0
            e = yi * math.cos(a8) - yr * math.sin(a8)
        else:
            dr, di = qam_decide_f64(order, yr), qam_decide_f64(order, yi)
            e = math.atan2(yi * dr - yr * di, yr * dr + yi * di)
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

    BPSK ``2 b - 1``; QPSK ``((2 b0 - 1) + j (2 b1 - 1)) / sqrt(2)``; 8PSK
    ``exp(j (2 pi k / 8 + pi / 8))`` with ``k`` from the Gray code ``b0 4 + b1 2 + b2`` (M40.1
    section 1); unit symbol energy, so the noise has ``sigma^2 = 1 / (2 (bits per symbol) Eb/N0)``
    per component.
    """
    bps = {2: 1, 4: 2, 8: 3}[order]
    data = rng.integers(0, 2, n * bps).tolist()
    nre, nim = rng.standard_normal(n).tolist(), rng.standard_normal(n).tolist()
    sigma = math.sqrt(1.0 / (2.0 * bps * 10.0 ** (ebn0_db / 10.0)))
    r = 1.0 / math.sqrt(2.0)
    x = []
    for k in range(n):
        if order == 2:
            sym = complex(2 * data[k] - 1, 0.0)
        elif order == 4:
            sym = complex((2 * data[2 * k] - 1) * r, (2 * data[2 * k + 1] - 1) * r)
        else:
            g = data[3 * k] * 4 + data[3 * k + 1] * 2 + data[3 * k + 2]
            a8 = (2 * GRAY_TO_INDEX8[g] + 1) * math.pi / 8.0
            sym = complex(math.cos(a8), math.sin(a8))
        # The phase in turns, reduced exactly before the scalar sine and cosine.
        turns = (phase0 / math.tau + offset * k) % 1.0
        w = complex(math.cos(math.tau * turns), math.sin(math.tau * turns))
        v = sym * w
        x.append(complex(v.real + sigma * nre[k], v.imag + sigma * nim[k]))
    return np.array(x, dtype=np.complex128).astype(np.complex64), sigma


def qam_signal(rng, n, name, phase0, offset, ebn0_db):
    """Unit-power square QAM with the Gray mapping of M44 section 1 (the first ``k / 2`` bits the
    in-phase level index by its Gray code, the last ``k / 2`` the quadrature one), rotated by
    ``phase0 + 2 pi offset k`` plus complex noise with ``sigma^2 = 1 / (2 k Eb/N0)``."""
    m, e = QAM[name]
    b = m.bit_length() - 1
    k_bits = 2 * b
    data = rng.integers(0, 2, n * k_bits).tolist()
    nre, nim = rng.standard_normal(n).tolist(), rng.standard_normal(n).tolist()
    sigma = math.sqrt(1.0 / (2.0 * k_bits * 10.0 ** (ebn0_db / 10.0)))
    gray_to_index = {i ^ (i >> 1): i for i in range(m)}
    x = []
    for k in range(n):
        bits = data[k * k_bits : (k + 1) * k_bits]
        gi = int("".join(str(v) for v in bits[:b]), 2)
        gq = int("".join(str(v) for v in bits[b:]), 2)
        li, lq = gray_to_index[gi], gray_to_index[gq]
        sym = complex(2 * li - (m - 1), 2 * lq - (m - 1)) / math.sqrt(e)
        turns = (phase0 / math.tau + offset * k) % 1.0
        w = complex(math.cos(math.tau * turns), math.sin(math.tau * turns))
        v = sym * w
        x.append(complex(v.real + sigma * nre[k], v.imag + sigma * nim[k]))
    return np.array(x, dtype=np.complex128).astype(np.complex64), sigma


def run_case(rng, order, offset, ebn0_db, notes, phase0=None):
    n = 8192
    if phase0 is None:
        phase0 = float(rng.uniform(0.0, math.tau))
    if order in QAM:
        x, sigma = qam_signal(rng, n, order, phase0, offset, ebn0_db)
    else:
        x, sigma = psk_signal(rng, n, order, phase0, offset, ebn0_db)
    kp, ki = gains(LOOP_BW, DAMPING)
    out, flags, (phase, freq, level, locked) = costas_f32(x, order, kp, ki, MAX_OFFSET)
    guard = guard_db(out, costas_f64(x, order, float(fm.f32(kp)), float(fm.f32(ki)), float(fm.f32(MAX_OFFSET))))
    # The loop must have locked and stay locked over the second half.
    assert locked and all(flags[n // 2:]), order
    if offset != 0.0:
        assert abs(float(freq) - offset) < 0.002, float(freq)
    first = {"constellation": order} if order in QAM else {"order": order}
    return dict(
        params={
            **first,
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


@case(kernel="costas", name="8psk_lock_c32_8192", seed=560)
def psk8_lock(rng):
    return run_case(
        rng, 8, 0.0, 14.0,
        "8PSK exp(j (2 pi k / 8 + pi / 8)) with k from the Gray code b0 4 + b1 2 + b2 (M40.1 "
        "section 1), one sample per symbol, rotated by a random start phase phase0_rad, complex "
        "noise for Eb/N0 14 dB; order 8 (the decision-directed detector against the point of the "
        "octant of y) with the block defaults; exact float32 model over the whole input; locked is "
        "the lock flag after every symbol; snr_db_guard compares with the float64 loop (D111)",
    )


@case(kernel="costas", name="dd_qam16_c32_8192", seed=600)
def dd_qam16(rng):
    return run_case(
        rng, "qam16", 0.0005, 16.0,
        "16-QAM with the Gray mapping of M44 section 1 (unit average power), one sample per symbol, "
        "rotated by the start phase phase0_rad (0.3 rad) and a carrier offset of 0.0005 cycles per "
        "symbol, complex noise for Eb/N0 16 dB; constellation qam16 (the decision-directed detector "
        "arg(y conj d) against the grid decision, atan2f of msun) with the block defaults; exact "
        "float32 model over the whole input; locked is the lock flag after every symbol; "
        "snr_db_guard compares with the float64 loop (D111)",
        phase0=0.3,
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
