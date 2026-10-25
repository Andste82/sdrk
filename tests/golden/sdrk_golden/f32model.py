"""Exact float32 model for kernels with discrete decisions (rule R8.3 exception).

Every helper computes with NumPy ``float32`` scalars, one operation at a time. NumPy rounds each
scalar operation to float32 and never fuses a multiply with an add, so a model written with these
helpers performs exactly the arithmetic of a Rust ``f32`` expression compiled without FMA. Cases
that use it are declared ``bit_exact`` and carry ``"model": "float32"`` in their JSON.
"""

import numpy as np

F32 = np.float32

# The Rust literal 1e-30 in an f32 comparison is the nearest float32 to 1e-30.
_FLUSH_LIMIT = F32(1e-30)


def f32(x):
    """Round ``x`` to a float32 scalar (round to nearest even)."""
    with np.errstate(all="ignore"):
        return F32(x)


def add(a, b):
    """``a + b`` rounded once to float32."""
    with np.errstate(all="ignore"):
        return F32(a) + F32(b)


def sub(a, b):
    """``a - b`` rounded once to float32."""
    with np.errstate(all="ignore"):
        return F32(a) - F32(b)


def mul(a, b):
    """``a * b`` rounded once to float32."""
    with np.errstate(all="ignore"):
        return F32(a) * F32(b)


def is_finite(a):
    """True when ``a`` is neither NaN nor infinite."""
    return bool(np.isfinite(F32(a)))


def flush_denormal(x):
    """0.0 when ``|x| < 1e-30`` (NaN stays NaN), otherwise ``x``; mirrors ``num::flush_denormal``."""
    x = F32(x)
    return F32(0.0) if abs(x) < _FLUSH_LIMIT else x


def self_test():
    """Check the helpers against hand-computed values; raises AssertionError on any deviation."""
    # Every result is a float32 scalar.
    for v in (f32(0.1), add(0.5, 0.25), sub(1, 2), mul(3, 4), flush_denormal(1.0)):
        assert isinstance(v, np.float32), type(v)
    # 0.1f + 0.2f is exactly 0.3f in float32 (bits 0x3E99999A); in float64 it is not 0.3.
    assert add(f32(0.1), f32(0.2)).tobytes() == f32(0.3).tobytes()
    assert float(f32(0.1)) + float(f32(0.2)) != float(f32(0.3))
    # f32(0.1) is 13421773 * 2^-27.
    assert float(f32(0.1)) == 13421773 * 2.0**-27
    # No fusing: (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11 (tie to even), so the
    # separate multiply and subtract give 0, a fused multiply-add would give 2^-24.
    x = add(1.0, 2.0**-12)
    assert float(sub(mul(x, x), 1.0 + 2.0**-11)) == 0.0
    # Rounding of a single operation: 1 + 2^-24 is a tie and rounds to even, i.e. to 1.
    assert float(add(1.0, 2.0**-24)) == 1.0
    assert float(add(1.0, 3.0 * 2.0**-24)) == 1.0 + 2.0**-22
    # Overflow gives infinity without raising; the helpers are quiet about it.
    assert float(mul(3e38, 10.0)) == float("inf")
    assert not is_finite(mul(3e38, 10.0)) and not is_finite(float("nan")) and is_finite(1e38)
    # flush_denormal: below 1e-30 in magnitude is 0.0, the limit itself and NaN are kept.
    assert float(flush_denormal(9e-31)) == 0.0 and float(flush_denormal(-9e-31)) == 0.0
    assert float(flush_denormal(f32(1e-30))) == float(f32(1e-30))
    assert float(flush_denormal(2e-30)) == float(f32(2e-30))
    assert bool(np.isnan(flush_denormal(float("nan"))))
    assert float(flush_denormal(-0.5)) == -0.5
