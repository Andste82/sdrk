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
