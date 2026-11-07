"""Cases for the channelize kernel (M45): the polyphase analysis filter bank, 2x oversampled.

Definition (``docs/kernels.md``): with the prototype ``h`` of ``N = M L`` taps, the hop ``D = M / 2``
and ``x[n] = 0`` for ``n < 0``, output step ``m`` is taken at ``t = (m + 1) D - 1`` and channel ``k`` is

    y_k[m] = sum_r h[r] x[t - r] exp(-j 2 pi k (t - r) / M).

The expected values are computed from that sum directly (no FFT, no polyphase decomposition): the
products ``h[r] x[n]`` of float32 values are exact in float64, each is multiplied by
``cos(2 pi q / M)`` or ``sin(2 pi q / M)`` (``q = k n mod M``, ``math.cos`` / ``math.sin``) and the
products are added with ``math.fsum``, so the result does not depend on the CPU.
"""

import math

import numpy as np

from .. import design as sdrk_design
from .. import scalar_math
from ..registry import Array, case


def scene(rng, n, channels, noise):
    """One tone per channel (offset within +-0.3 of a spacing, random phase, amplitude 1 / M) plus
    complex Gaussian noise of ``noise`` per component, as complex64."""
    idx = np.arange(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.complex128)
    for k in range(channels):
        f = (k + rng.uniform(-0.3, 0.3)) / channels
        phi = rng.uniform(0.0, 2.0 * math.pi)
        phase = 2.0 * math.pi * np.mod(f * idx, 1.0) + phi
        x += scalar_math.expj(phase) / channels
    x += (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * noise
    return x.astype(np.complex64)


def channelize_exact(h32, x, channels):
    """The definition evaluated with correctly rounded sums (see the module docstring)."""
    m = channels
    d = m // 2
    taps = len(h32)
    h = np.asarray(h32, dtype=np.float64)
    xr = np.asarray(x.real, dtype=np.float64)
    xi = np.asarray(x.imag, dtype=np.float64)
    cos_t = np.array([math.cos(2.0 * math.pi * q / m) for q in range(m)])
    sin_t = np.array([math.sin(2.0 * math.pi * q / m) for q in range(m)])
    steps = len(x) // d
    ks = np.arange(m, dtype=np.int64)[:, None]
    out = np.empty(steps * m, dtype=np.complex128)
    for step in range(steps):
        t = (step + 1) * d - 1
        r = np.arange(taps)
        n = t - r
        valid = n >= 0
        r, n = r[valid], n[valid]
        a = h[r] * xr[n]  # exact: float32 times float32
        b = h[r] * xi[n]
        q = (ks * n[None, :]) % m
        c, s = cos_t[q], sin_t[q]
        # (a + j b)(c - j s) = (a c + b s) + j (b c - a s)
        re_terms = np.concatenate([a * c, b * s], axis=1)
        im_terms = np.concatenate([b * c, -(a * s)], axis=1)
        for k in range(m):
            out[step * m + k] = complex(math.fsum(re_terms[k].tolist()), math.fsum(im_terms[k].tolist()))
    return out


def kernel_case(rng, channels, taps_per_branch, n, guard):
    h32 = sdrk_design.channelizer_prototype(channels, taps_per_branch, 0.5, 1.0, 60.0).astype(np.float32)
    x = scene(rng, n, channels, 0.01)
    y = channelize_exact(h32, x, channels)
    return dict(
        params={
            "channels": channels,
            "taps_per_branch": taps_per_branch,
            "passband": 0.5,
            "stopband": 1.0,
            "attenuation_db": 60.0,
        },
        arrays={"input": Array("cf32", x), "taps": Array("f32", h32), "expected": Array("cf64", y)},
        tolerance={"class": "tolerance", "snr_db_min": 100.0, "snr_db_guard": guard, "skip": 0},
        notes=(
            "y_k[m] = sum_r h[r] x[(m+1) M/2 - 1 - r] exp(-j 2 pi k ((m+1) M/2 - 1 - r) / M), step-major "
            "(M channels per step, FFT order), zero history; h = the float32 prototype with edges 0.5 / 1.0, "
            "60 dB; exactly rounded sums"
        ),
    )


@case(kernel="channelize", name="m16_l8_c32_4096", seed=660)
def m16_l8(rng):
    # Regression guard: the fallback reaches 140.8 dB (the optimum, the expected values rounded to
    # cf32, is about 150 dB).
    return kernel_case(rng, 16, 8, 4096, 139.8)


@case(kernel="channelize", name="m64_l8_c32_4096", seed=661)
def m64_l8(rng):
    # Regression guard: the fallback reaches 139.3 dB.
    return kernel_case(rng, 64, 8, 4096, 138.3)
