"""Bit-exact cross-check of sdrk_core::num against the Python Q15 model."""

import numpy as np

from .. import q15
from ..registry import Array, case

# The 96 special inputs of from_float_4096, listed explicitly. Every entry is
# subsequently rounded to float32 (the type the Rust function takes).
SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
    1 / 65536, -1 / 65536, 3 / 65536, -3 / 65536,
    float("nan"), float("inf"), float("-inf"),
    1 - 2**-15, -(1 - 2**-15),
    2**-16 + 2**-40, 2**-16 - 2**-40, -(2**-16 + 2**-40), -(2**-16 - 2**-40),
    2**-16 + 2**-30, 2**-16 - 2**-30, -(2**-16 + 2**-30), -(2**-16 - 2**-30),
    # exact half-LSB ties (k + 0.5) / 32768 for several k
    5 / 65536, -5 / 65536, 7 / 65536, -7 / 65536, 9 / 65536, -9 / 65536,
    65535 / 65536, -65535 / 65536, 32767 / 65536, -32767 / 65536,
    # values around the upper and lower saturation limits
    1 - 2**-16, 1 - 2**-16 - 2**-24, 1 + 2**-16, 1 + 2**-15, 1.0000001,
    -1 - 2**-16, -1 + 2**-16, -1 - 2**-15, -1.0000001,
    1.5, -1.5, 2.0, -2.0, 1e10, -1e10, 3.4028235e38, -3.4028235e38,
    # tiny values, denormals and exact LSB steps
    1e-30, -1e-30, 1.17549435e-38, -1.17549435e-38, 1e-45, -1e-45,
    1 / 32768, -1 / 32768, 2 / 32768, -2 / 32768, 0.25, -0.25, 0.125, -0.125,
    0.999969482421875, -0.999969482421875, 0.9999695, 0.99997,
    0.3, -0.3, 0.7, -0.7, 0.1, -0.1,
    16383.5 / 32768, -16383.5 / 32768, 16384.5 / 32768, -16384.5 / 32768,
    0.49999, -0.49999, 0.50001, -0.50001, 1e-5, -1e-5, 3e-5, -3e-5,
    1.2, -1.2, 0.9999, -0.9999,
    11 / 65536, -11 / 65536, 13 / 65536, -13 / 65536, 0.75, -0.75,
]
assert len(SPECIALS) == 96, len(SPECIALS)


@case(kernel="q15_ops", name="from_float_4096", seed=2)
def from_float_4096(rng):
    x = np.concatenate([rng.uniform(-1.2, 1.2, 4000), np.array(SPECIALS, dtype=np.float64)])
    x32 = x.astype(np.float32)
    expected = q15.from_float(x32.astype(np.float64))
    return dict(
        params={},
        arrays={"input": Array("f32", x32), "expected": Array("i16", expected)},
        tolerance={"class": "bit_exact"},
        notes="expected[n] = q15_from_f32(input[n]); rounding happens in f64",
    )


@case(kernel="q15_ops", name="mul_pairs_4096", seed=3)
def mul_pairs_4096(rng):
    a = rng.integers(-32768, 32768, 4096)
    b = rng.integers(-32768, 32768, 4096)
    edge = [
        (-32768, -32768), (-32768, 32767), (32767, 32767), (1, 1),
        (-1, 1), (16384, 16384), (0, int(b[6])), (-32768, 1),
    ]
    for i, (ea, eb) in enumerate(edge):
        a[i], b[i] = ea, eb
    return dict(
        params={},
        arrays={
            "a": Array("i16", a),
            "b": Array("i16", b),
            "expected": Array("i16", q15.mul(a, b)),
        },
        tolerance={"class": "bit_exact"},
        notes="expected[n] = q15_mul(a[n], b[n])",
    )


@case(kernel="q15_ops", name="cmul_pairs_4096", seed=4)
def cmul_pairs_4096(rng):
    ar, ai, br, bi = (rng.integers(-32768, 32768, 4096) for _ in range(4))
    edge = [
        (-32768, -32768, -32768, -32768), (-32768, 0, -32768, 0), (32767, 32767, 32767, 32767),
        (0, 32767, 0, 32767), (1, 1, 1, 1), (-32768, 32767, 32767, -32768),
    ]
    for i, (a, b, c, d) in enumerate(edge):
        ar[i], ai[i], br[i], bi[i] = a, b, c, d
    er, ei = q15.cmul(ar, ai, br, bi)
    return dict(
        params={},
        arrays={
            "a": Array("ci16", q15.pack(ar, ai)),
            "b": Array("ci16", q15.pack(br, bi)),
            "expected": Array("ci16", q15.pack(er, ei)),
        },
        tolerance={"class": "bit_exact"},
        notes="expected[n] = cq15_mul(a[n], b[n])",
    )
