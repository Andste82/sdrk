"""Cases for the dot_complex kernel (complex samples by complex weights, M43, decisions D800, D801).

The kernel is bit-exact: its contract fixes the float32 order (``docs/kernels.md``), so the expected
values come from an exact float32 model of that order (rule R8.3 exception, as for the feedback
kernels): eight partial sums per component starting at ``+0.0``, element ``i`` added to partial sum
``i mod 8``, the product ``(x.re w.re - x.im w.im, x.re w.im + x.im w.re)`` with every product
rounded before the sum, then per component ``s[k] = acc[k] + acc[k + 4]``, ``t[k] = s[k] + s[k + 2]``
and ``t[0] + t[1]``. The model evaluates every operation with NumPy ``float32`` arithmetic, either on
scalars or element by element on ``float32`` arrays (one rounded operation per element, nothing fused,
no reduction function), so it equals Rust ``f32`` code without FMA on every CPU.

Each case holds ``rows`` sliding windows: ``expected[r] = dot_complex(input[r : r + taps], weights)``.

The regression guard of decision D111 sums the same products in float64 (``math.fsum`` of the
real and imaginary parts of the float64 products, i.e. the exactly rounded sum).
``snr_db_guard`` is the SNR of the model output against that float64 sum over all rows minus 1 dB,
rounded down to 0.1 dB; the Rust test computes its own float64 sum.
"""

import math

import numpy as np

from ..registry import Array, case

PARTIALS = 8
F32 = np.float32


def dot_complex_f32(xr, xi, wr, wi):
    """The contract order for a batch of windows.

    ``xr``, ``xi``: float32 arrays of shape (rows, n); ``wr``, ``wi``: float32 arrays of length
    ``n``. Returns the float32 arrays of the real and imaginary results (length ``rows``).
    """
    rows, n = xr.shape
    with np.errstate(all="ignore"):
        acc_r = np.zeros((rows, PARTIALS), dtype=F32)
        acc_i = np.zeros((rows, PARTIALS), dtype=F32)
        for i in range(n):
            k = i % PARTIALS
            a_r, a_i = xr[:, i], xi[:, i]
            b_r, b_i = F32(wr[i]), F32(wi[i])
            pr = a_r * b_r - a_i * b_i
            pi = a_r * b_i + a_i * b_r
            acc_r[:, k] = acc_r[:, k] + pr
            acc_i[:, k] = acc_i[:, k] + pi
        return tree(acc_r), tree(acc_i)


def tree(acc):
    """``(s0 + s2) + (s1 + s3)`` with ``s_k = acc_k + acc_{k + 4}`` per row (float32)."""
    s = [acc[:, k] + acc[:, k + 4] for k in range(4)]
    return (s[0] + s[2]) + (s[1] + s[3])


def dot_complex_f64(x, w):
    """The exactly rounded float64 sum of the float64 products of one window (lists of complex)."""
    re = math.fsum(a.real * b.real - a.imag * b.imag for a, b in zip(x, w))
    im = math.fsum(a.real * b.imag + a.imag * b.real for a, b in zip(x, w))
    return complex(re, im)


def run_case(rng, n, rows):
    x = ((rng.standard_normal(n + rows - 1) + 1j * rng.standard_normal(n + rows - 1)) * 0.3).astype(np.complex64)
    w = (rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)).astype(np.complex64)
    xr = np.stack([x.real[r : r + n] for r in range(rows)]).astype(F32)
    xi = np.stack([x.imag[r : r + n] for r in range(rows)]).astype(F32)
    yr, yi = dot_complex_f32(xr, xi, w.real.astype(F32), w.imag.astype(F32))
    expected = np.empty(rows, dtype=np.complex64)
    expected.real = yr
    expected.imag = yi
    xs = [complex(v) for v in x.tolist()]
    ws = [complex(v) for v in w.tolist()]
    ref = [dot_complex_f64(xs[r : r + n], ws) for r in range(rows)]
    got = [complex(v) for v in expected.tolist()]
    num = math.fsum(abs(v) ** 2 for v in ref)
    den = math.fsum(abs(a - b) ** 2 for a, b in zip(ref, got))
    assert den > 0.0
    guard = math.floor((10.0 * math.log10(num / den) - 1.0) * 10.0) / 10.0
    assert guard > 100.0, guard
    return dict(
        params={"taps": n, "rows": rows, "partials": PARTIALS, "snr_db_guard": guard},
        arrays={"input": Array("cf32", x), "weights": Array("cf32", w), "expected": Array("cf32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes=f"expected[r] = dot_complex(input[r : r + {n}], weights) for r = 0 .. {rows - 1} (not conjugated), "
        f"the order of the contract: 8 partial sums per component, element i to partial sum i mod 8 "
        f"({n} = 8 * {n // 8} + {n % 8}, so the last block is partial), the product (x.re w.re - x.im w.im, "
        "x.re w.im + x.im w.re) without FMA, then (s0 + s2) + (s1 + s3) with s_k = acc_k + acc_(k+4); "
        "input complex normal 0.3 per component, weights uniform in [-0.5, 0.5) per component; "
        "snr_db_guard compares with the exactly rounded float64 sum (D111)",
    )


@case(kernel="dot_complex", name="c32_13", seed=596)
def c32_13(rng):
    return run_case(rng, 13, 256)


@case(kernel="dot_complex", name="c32_1021", seed=597)
def c32_1021(rng):
    return run_case(rng, 1021, 64)
