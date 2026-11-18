"""Cases for the qam_soft_bits kernel (the piecewise-linear max-log soft bits of every square QAM,
M44 section 2, M44.2 section 2, open item I44-1, decisions D1412 and D1413).

The kernel is bit-exact against its float32 form: per component ``x`` (in memory order, ``re``
then ``im`` of every symbol) with ``b`` bits per axis, ``t_0 = x * sqrt_e`` with ``sqrt_e`` the
float32 of ``sqrt(E)`` (E = 2, 10, 42, 170 or 682) and ``t_i = 2^(b - i) - |t_(i - 1)|`` for
``i = 1 .. b - 1``, the soft bits ``scale * t_i``: ``scale * v``, ``scale * (2 - |v|)`` for
16-QAM, ``scale * v``, ``scale * (4 - |v|)``, ``scale * (2 - |4 - |v||)`` for 64-QAM (``|4 - a|``
is ``|a - 4|`` exactly, the form of M44), every operation one rounded float32 operation. The model evaluates them with NumPy ``float32`` element by element on
arrays (one rounded operation per element, nothing fused), so it equals Rust ``f32`` code without
FMA on every CPU.

The regression guard of decision D111 evaluates the same definition in float64 with the exact
``sqrt(E)`` (plain Python floats, ``math.sqrt``): ``snr_db_guard`` is the SNR of the model output
against it minus 1 dB, rounded down to 0.1 dB; the Rust test computes its own float64 values (the
``Reference`` of the kernel). ``snr_db_optimum`` (information) is the SNR of the float64 values
rounded once to float32, the best an ``f32`` output can reach.
"""

import math
import struct

import numpy as np

from ..registry import Array, case

F32 = np.float32
# (E, levels per axis, bits per axis) of every square order; 4-QAM is qpsk (M44.2).
QAM = {
    "qpsk": (2, 2, 1),
    "qam16": (10, 4, 2),
    "qam64": (42, 8, 3),
    "qam256": (170, 16, 4),
    "qam1024": (682, 32, 5),
}


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def model_f32(order, x, scale):
    """The float32 form for the components ``x`` (float32 array); returns the soft bits in output
    order (``b`` per component)."""
    e, _, b = QAM[order]
    c = F32(math.sqrt(e))
    s = F32(scale)
    with np.errstate(all="ignore"):
        t = x * c
        cols = [s * t]
        for i in range(1, b):
            t = F32(2 ** (b - i)) - np.abs(t)
            cols.append(s * t)
    return np.stack(cols, axis=1).reshape(-1).astype(F32)


def definition_f64(order, x, scale):
    """The definition in float64 with the exact ``sqrt(E)``, one value per soft bit."""
    e, _, b = QAM[order]
    c = math.sqrt(e)
    s = float(F32(scale))
    out = []
    for xv in x.tolist():
        t = float(xv) * c
        bits = [t]
        for i in range(1, b):
            t = float(2 ** (b - i)) - abs(t)
            bits.append(t)
        out.extend(s * v for v in bits)
    return out


def snr_db(ref, got):
    num = math.fsum(r * r for r in ref)
    den = math.fsum((r - g) ** 2 for r, g in zip(ref, got))
    assert den > 0.0
    return 10.0 * math.log10(num / den)


def run_case(rng, order, n, scale, noise=0.2):
    e, m, _ = QAM[order]
    levels = [(2 * i - (m - 1)) / math.sqrt(e) for i in range(m)]
    # Listed components first: zeros, the levels, the boundaries of the soft bits (|v| = 2, 4, 6,
    # where a soft bit is 0) and values outside the grid.
    listed = [0.0, -0.0] + levels
    for k in range(2, m, 2):
        listed += [k / math.sqrt(e), -k / math.sqrt(e)]
    listed += [3.0, -3.0, 1e-30, -1e-30]
    if len(listed) % 2:
        listed.append(0.5)
    comps = np.empty(2 * n, dtype=F32)
    comps[: len(listed)] = np.array(listed, dtype=F32)
    rest = 2 * n - len(listed)
    pts = rng.choice(np.array(levels), rest) + rng.uniform(-noise, noise, rest)
    comps[len(listed) :] = pts.astype(F32)
    x = np.empty(n, dtype=np.complex64)
    x.real = comps[0::2]
    x.imag = comps[1::2]
    expected = model_f32(order, comps, scale)
    ref = definition_f64(order, comps, scale)
    got = [float(v) for v in expected.tolist()]
    guard = math.floor((snr_db(ref, got) - 1.0) * 10.0) / 10.0
    assert guard > 130.0, guard
    optimum = snr_db(ref, [float(F32(r)) for r in ref])
    return dict(
        params={
            "constellation": order,
            "scale_bits": f32_bits(F32(scale)),
            "listed_components": len(listed),
            "snr_db_guard": guard,
            "snr_db_optimum": round(optimum, 1),
        },
        arrays={"input": Array("cf32", x), "expected": Array("f32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes=f"{order} soft bits of {n} symbols, {expected.size // n} per symbol in output order "
        "(the components in memory order, re then im); the scale is the float32 of "
        f"{scale} (scale_bits); the first {len(listed)} components are 0, -0, the levels, the soft-bit "
        "boundaries +-k / sqrt(E) for even k, +-3 and +-1e-30, the rest random levels plus uniform "
        f"noise in [-{noise}, {noise}); snr_db_guard compares with the float64 definition with the exact "
        "sqrt(E) (D111), snr_db_optimum is that definition rounded once to float32",
    )


@case(kernel="qam_soft_bits", name="qam16_c32_4096", seed=775)
def qam16_c32_4096(rng):
    return run_case(rng, "qam16", 4096, 5.3)


@case(kernel="qam_soft_bits", name="qam64_c32_4096", seed=776)
def qam64_c32_4096(rng):
    return run_case(rng, "qam64", 4096, 0.7)


# M44.2: the generic instances of 1, 4 and 5 bits per axis. The noise is about the 16-QAM noise of 0.2
# in units of the level spacing 2 / sqrt(10), so every order sees the same noise per cell.
@case(kernel="qam_soft_bits", name="qpsk_c32_4096", seed=820)
def qpsk_c32_4096(rng):
    return run_case(rng, "qpsk", 4096, 2.5, noise=0.45)


@case(kernel="qam_soft_bits", name="qam256_c32_4096", seed=821)
def qam256_c32_4096(rng):
    return run_case(rng, "qam256", 4096, 0.4, noise=0.05)


@case(kernel="qam_soft_bits", name="qam1024_c32_4096", seed=822)
def qam1024_c32_4096(rng):
    return run_case(rng, "qam1024", 4096, 1.9, noise=0.025)
