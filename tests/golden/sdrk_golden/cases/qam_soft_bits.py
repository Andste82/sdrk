"""Cases for the qam_soft_bits kernel (the piecewise-linear max-log soft bits of 16- and 64-QAM,
M44 section 2, open item I44-1, decisions D1412 and D1413).

The kernel is bit-exact against its float32 form: per component ``x`` (in memory order, ``re``
then ``im`` of every symbol) ``v = x * sqrt_e`` with ``sqrt_e`` the float32 of ``sqrt(E)``
(E = 10 or 42) and ``a = |v|``, the soft bits ``scale * v``, ``scale * (2 - a)`` (16-QAM) or
``scale * v``, ``scale * (4 - a)``, ``scale * (2 - |a - 4|)`` (64-QAM), every operation one
rounded float32 operation. The model evaluates them with NumPy ``float32`` element by element on
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
QAM = {"qam16": (10, 4), "qam64": (42, 8)}


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def model_f32(order, x, scale):
    """The float32 form for the components ``x`` (float32 array); returns the soft bits in output
    order (two or three per component)."""
    e, _ = QAM[order]
    c = F32(math.sqrt(e))
    s = F32(scale)
    with np.errstate(all="ignore"):
        v = x * c
        a = np.abs(v)
        if order == "qam16":
            cols = [s * v, s * (F32(2.0) - a)]
        else:
            cols = [s * v, s * (F32(4.0) - a), s * (F32(2.0) - np.abs(a - F32(4.0)))]
    return np.stack(cols, axis=1).reshape(-1).astype(F32)


def definition_f64(order, x, scale):
    """The definition in float64 with the exact ``sqrt(E)``, one value per soft bit."""
    e, _ = QAM[order]
    c = math.sqrt(e)
    s = float(F32(scale))
    out = []
    for xv in x.tolist():
        v = float(xv) * c
        a = abs(v)
        if order == "qam16":
            bits = [v, 2.0 - a]
        else:
            bits = [v, 4.0 - a, 2.0 - abs(a - 4.0)]
        out.extend(s * b for b in bits)
    return out


def snr_db(ref, got):
    num = math.fsum(r * r for r in ref)
    den = math.fsum((r - g) ** 2 for r, g in zip(ref, got))
    assert den > 0.0
    return 10.0 * math.log10(num / den)


def run_case(rng, order, n, scale):
    e, m = QAM[order]
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
    pts = rng.choice(np.array(levels), rest) + rng.uniform(-0.2, 0.2, rest)
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
        "noise in [-0.2, 0.2); snr_db_guard compares with the float64 definition with the exact "
        "sqrt(E) (D111), snr_db_optimum is that definition rounded once to float32",
    )


@case(kernel="qam_soft_bits", name="qam16_c32_4096", seed=775)
def qam16_c32_4096(rng):
    return run_case(rng, "qam16", 4096, 5.3)


@case(kernel="qam_soft_bits", name="qam64_c32_4096", seed=776)
def qam64_c32_4096(rng):
    return run_case(rng, "qam64", 4096, 0.7)
