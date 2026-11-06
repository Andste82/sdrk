"""Block cases of the PSK variants (M40.1): the max-log soft bits of the ``slice`` constellation ``8psk``.

The soft bits are a fixed float32 computation (M40.1 section 1, D642), so the expected values come
from an exact float32 model with the ``f32model`` helpers, one operation at a time: the points
``exp(j (2k + 1) pi / 8)`` with each part rounded once to float32 (``math.cos`` and ``math.sin`` of
the float64 angle), the squared distance ``dr dr + di di`` with ``dr = y.re - p.re`` and
``di = y.im - p.im``, for every bit the minimum over the points whose Gray bit is 0 and over those
whose bit is 1, each starting at the first point of its set in index order and taking a later point
only when it is strictly nearer, and the soft bit ``scale ((m0 - m1) * 0.25)``. The Gray code of
point ``k`` is the inverse of the table ``g -> k`` of section 1. The inputs are deterministic (no
random numbers, seed 0 as for the design cases): the 8 points, the 8 decision boundaries
``k pi / 4`` and both sets at amplitude 0.5.
"""

import math

import numpy as np

from .. import f32model as fm
from ..registry import Array, case

# M40.1 section 1: g = b0 4 + b1 2 + b2 -> symbol index k.
GRAY_TO_INDEX8 = (0, 1, 3, 2, 7, 6, 4, 5)
INDEX_TO_GRAY8 = tuple(GRAY_TO_INDEX8.index(k) for k in range(8))
QUARTER = fm.f32(0.25)


def points8():
    """The float32 parts of the 8PSK points in index order."""
    out = []
    for k in range(8):
        a = (2 * k + 1) * math.pi / 8.0
        out.append((fm.f32(math.cos(a)), fm.f32(math.sin(a))))
    return out


def psk8_bits(scale, yr, yi):
    """The three max-log soft bits ``b0, b1, b2`` of the float32 symbol ``(yr, yi)``."""
    dist = []
    for pr, pi in points8():
        dr, di = fm.sub(yr, pr), fm.sub(yi, pi)
        dist.append(fm.add(fm.mul(dr, dr), fm.mul(di, di)))
    out = []
    for bit in range(3):
        m = [None, None]
        for k, d in enumerate(dist):
            b = (INDEX_TO_GRAY8[k] >> (2 - bit)) & 1
            if m[b] is None or d < m[b]:
                m[b] = d
        out.append(fm.mul(scale, fm.mul(fm.sub(m[0], m[1]), QUARTER)))
    return out


@case(kernel="blocks", name="slice_8psk_c32", seed=0)
def slice_8psk_c32(rng):
    del rng  # deterministic corner values
    scale = 2.5
    x = []
    for amp in (1.0, 0.5):
        for k in range(16):
            # Even k: the boundaries k pi / 8 = (k / 2) pi / 4; odd k: the points.
            a = k * math.pi / 8.0
            x.append(complex(amp * math.cos(a), amp * math.sin(a)))
    x = np.array(x, dtype=np.complex128).astype(np.complex64)
    s = fm.f32(scale)
    expected = []
    for v in x.tolist():
        expected.extend(psk8_bits(s, fm.f32(v.real), fm.f32(v.imag)))
    expected = np.array(expected, dtype=np.float32)
    # Every point gives its own Gray code.
    for k in range(8):
        bits = expected[3 * (2 * k + 1):3 * (2 * k + 2)]
        g = int(bits[0] > 0) * 4 + int(bits[1] > 0) * 2 + int(bits[2] > 0)
        assert GRAY_TO_INDEX8[g] == k, (k, bits)
    return dict(
        params={"constellation": "8psk", "scale": scale, "symbols": len(x), "bits_per_symbol": 3},
        arrays={"input": Array("cf32", x), "expected": Array("f32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="slice constellation 8psk (M40.1 section 1): the inputs are exp(j k pi / 8) for k = 0 "
        "to 15 at amplitude 1, then at 0.5 (even k the decision boundaries k pi / 4 of the octants, "
        "odd k the 8 points), each part rounded once to float32; expected are the three soft bits "
        "b0, b1, b2 per symbol, scale ((m0 - m1) * 0.25) with m0 (m1) the smallest squared distance "
        "to the float32 points whose Gray bit is 0 (1), minima in index order with a strict "
        "comparison; exact float32 model",
    )
