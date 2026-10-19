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
