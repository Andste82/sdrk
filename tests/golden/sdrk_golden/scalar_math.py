"""Element-wise transcendental functions evaluated one value at a time with the C library.

NumPy dispatches ``log10``, ``sin``, ``exp`` and ``arctan2`` to SIMD implementations that differ
between CPU generations (for example AVX-512 versus AVX2), so the results can differ in the last
bits from machine to machine. Golden vectors must be byte-identical everywhere, so the cases that
feed a transcendental function into a committed vector use these scalar versions instead.
"""

import math

import numpy as np


def _map(fn, x):
    a = np.asarray(x, dtype=np.float64)
    return np.fromiter((fn(v) for v in a.ravel()), dtype=np.float64, count=a.size).reshape(a.shape)


def log10(x):
    return _map(math.log10, x)


def sin(x):
    return _map(math.sin, x)


def cos(x):
    return _map(math.cos, x)


def expj(theta):
    """``exp(1j * theta)`` as ``cos(theta) + 1j * sin(theta)``."""
    t = np.asarray(theta, dtype=np.float64)
    return _map(math.cos, t) + 1j * _map(math.sin, t)


def angle(z):
    """``arg(z)`` through ``atan2``."""
    z = np.asarray(z, dtype=np.complex128)
    re, im = z.real.ravel(), z.imag.ravel()
    out = np.fromiter((math.atan2(i, r) for r, i in zip(re, im)), dtype=np.float64, count=re.size)
    return out.reshape(z.shape)


def cos(x):
    return _map(math.cos, x)


def exp(x):
    return _map(math.exp, x)


def pow10(x):
    """``10 ** x`` through ``pow``."""
    return _map(lambda v: math.pow(10.0, v), x)


def geomspace(start, stop, num):
    """``numpy.geomspace`` for positive endpoints, from ``log10`` and ``pow`` evaluated per value."""
    lin = np.linspace(math.log10(start), math.log10(stop), num)
    out = pow10(lin)
    out[0], out[-1] = start, stop
    return out


def dot(a, b):
    """Dot product of two vectors with a fixed summation order, independent of the BLAS in use.

    Every product is rounded to float64 and the sum is the exactly rounded ``math.fsum`` of the
    products, so the result does not depend on the kernel that BLAS dispatches for the CPU. ``a`` may
    be complex, ``b`` must be real.
    """
    a = np.asarray(a)
    b = np.asarray(b, dtype=np.float64)
    if np.iscomplexobj(a):
        return complex(dot(a.real, b), dot(a.imag, b))
    return math.fsum((a.astype(np.float64) * b).tolist())


def fir_exact(h, x, decimation=1):
    """``y[m] = sum_k h[k] x[decimation m - k]`` (zero history) with every sum correctly rounded.

    ``scipy.signal.lfilter`` and ``np.convolve`` add the products through BLAS kernels whose summation
    order depends on the CPU, so their last bit differs from machine to machine. Here ``h`` and ``x``
    hold float32 values (or integers below 2**24 times a sign), so every product is exact in float64 and
    ``math.fsum`` returns the correctly rounded sum whatever the order. A complex ``x`` is filtered per
    component. Zero taps are skipped (they add exact zeros)."""
    h = np.asarray(h, dtype=np.float64)
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return fir_exact(h, x.real, decimation) + 1j * fir_exact(h, x.imag, decimation)
    taps = [(k, float(c)) for k, c in enumerate(h) if c != 0.0]
    xs = x.astype(np.float64).tolist()
    count = (len(xs) + decimation - 1) // decimation
    out = np.empty(count, dtype=np.float64)
    for m in range(count):
        n = decimation * m
        out[m] = math.fsum(c * xs[n - k] for k, c in taps if n - k >= 0)
    return out
