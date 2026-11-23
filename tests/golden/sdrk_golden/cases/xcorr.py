"""Cases for the FFT fast convolution ``xcorr_fft`` (M50): a ``C32`` stream convolved with fixed
complex taps by overlap-save.

Definition (``docs/kernels.md``): with the taps ``h`` (length ``L``) and ``x[n] = 0`` for ``n < 0``,
output ``n`` is ``y[n] = sum_k h[k] x[n - k]``, that is ``numpy.convolve(x, h)[:rows]``. Each input
is four transform blocks long (``B = N - L + 1``, ``rows = 4 B``).

The expected values are that convolution, independent of any FFT, evaluated with correctly rounded
sums: the real products of the float32 components are exact in float64 (``hr xr``, ``hi xi``,
``hr xi``, ``hi xr``), and each output component is the ``math.fsum`` of its products (``re = sum hr
xr - hi xi``, ``im = sum hr xi + hi xr``), so the files do not depend on the CPU or on a BLAS
library. The generator asserts that the result equals ``numpy.convolve`` within 1e-12 of the
largest output.

The code cases embed one period of a PRBS code (chip ``i`` the ``i``-th bit of
``sdrk_golden.prbs.Prbs(code, 1)``, chip value ``1 - 2 b``, shared conventions §B.1) at input offset
``offset`` in complex Gaussian noise, and the taps are the code conjugated and reversed, so the
correlation peak is at output ``offset + L - 1``; the JSON stores that index and the ratio of the
peak magnitude to the median output magnitude as information.

Draw order: the offset (code cases), then the noise or the input, then the taps (``rand``).
"""

import math

import numpy as np

from ..prbs import Prbs
from ..registry import Array, case
from .design import rrc_taps

SNR_MIN = 100.0


def conv_exact(x, h, rows):
    """``numpy.convolve(x, h)[:rows]`` with correctly rounded sums (see the module docstring)."""
    xr = np.asarray(x.real, dtype=np.float64)
    xi = np.asarray(x.imag, dtype=np.float64)
    hr = np.asarray(h.real, dtype=np.float64)
    hi = np.asarray(h.imag, dtype=np.float64)
    out = np.empty(rows, dtype=np.complex128)
    for n in range(rows):
        k = np.arange(max(0, n - (len(x) - 1)), min(len(h) - 1, n) + 1)
        s = n - k
        re = np.concatenate([hr[k] * xr[s], -(hi[k] * xi[s])])
        im = np.concatenate([hr[k] * xi[s], hi[k] * xr[s]])
        out[n] = complex(math.fsum(re.tolist()), math.fsum(im.tolist()))
    check = np.convolve(x.astype(np.complex128), h.astype(np.complex128))[:rows]
    assert np.max(np.abs(out - check)) <= 1e-12 * np.max(np.abs(out)), "fsum and numpy.convolve differ"
    return out


def chips(code, length):
    """One period of the PRBS code as chip values ``1 - 2 b``."""
    bits = Prbs(code, 1).bits(length)
    return np.array([1.0 - 2.0 * b for b in bits], dtype=np.float64)


def noise(rng, n, power):
    """Complex Gaussian noise of the total power ``power`` per sample."""
    sigma = math.sqrt(power / 2.0)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma


def result(x, h, n_fft, params, notes, guard):
    """The case dict (input ``cf32``, taps ``cf32``, expected ``cf64`` of the exact convolution, the
    contract floor and the regression guard ``guard``, the fallback SNR minus 1 dB rounded down to
    0.1 dB) and the expected values."""
    x = np.asarray(x, dtype=np.complex64)
    h = np.asarray(h, dtype=np.complex64)
    taps = len(h)
    b = n_fft - taps + 1
    assert len(x) == 4 * b
    y = conv_exact(x, h, len(x))
    tolerance = {"class": "tolerance", "snr_db_min": SNR_MIN, "snr_db_guard": guard, "skip": 0}
    return dict(
        params={"taps_len": taps, "fft_size": n_fft, "block_size": b, **params},
        arrays={"input": Array("cf32", x), "taps": Array("cf32", h), "expected": Array("cf64", y)},
        tolerance=tolerance,
        notes=notes,
    ), y


def code_case(rng, code, taps, n_fft, noise_power, guard):
    b = n_fft - taps + 1
    n_in = 4 * b
    offset = int(rng.integers(taps, n_in - 2 * taps))
    c = chips(code, taps)
    x = noise(rng, n_in, noise_power)
    x[offset : offset + taps] += c
    h = np.conj(c[::-1])
    out, y = result(
        x,
        h,
        n_fft,
        {"code": code, "seed": 1, "offset": offset, "noise_power": noise_power},
        f"one period of {code} (seed 1, chip 1 - 2b) at input offset {offset} in complex Gaussian noise of "
        f"power {noise_power} per sample; taps = the code conjugated and reversed; expected = the exact "
        "convolution numpy.convolve(x, h)[:rows]; zero history",
        guard,
    )
    mag = np.abs(y)
    peak = int(np.argmax(mag))
    assert peak == offset + taps - 1, (peak, offset + taps - 1)
    out["params"]["peak_index"] = peak
    out["params"]["peak_to_median"] = round(float(mag[peak] / np.median(mag)), 6)
    return out


@case(kernel="xcorr_fft", name="code_127_n512", seed=700)
def code_127_n512(rng):
    # Chip SNR 0 dB. Regression guard: the fallback reaches 134.54 dB (the reference 145.35 dB).
    return code_case(rng, "prbs7", 127, 512, 1.0, 133.5)


@case(kernel="xcorr_fft", name="code_1023_n4096", seed=701)
def code_1023_n4096(rng):
    # Chip SNR -10 dB: the processing gain of 30.1 dB puts the peak power about 20 dB above the
    # mean output noise power. Regression guard: the fallback reaches 132.56 dB (the reference
    # 143.84 dB).
    return code_case(rng, "prbs10", 1023, 4096, 10.0, 131.5)


@case(kernel="xcorr_fft", name="rrc_65_n256", seed=702)
def rrc_65_n256(rng):
    # QPSK symbols (+-0.5 per component, exact) at 4 samples per symbol shaped by the RRC pulse
    # (rolloff 0.35, span 16 symbols, 65 taps, unit energy); the kernel taps are the same pulse
    # in float32 (imaginary part 0), the matched filter.
    taps, n_fft, sps = 65, 256, 4
    b = n_fft - taps + 1
    n_in = 4 * b
    pulse = rrc_taps(0.35, sps, 16)
    assert len(pulse) == taps
    symbols = (n_in + taps) // sps + 1
    sym = 0.5 * (2.0 * rng.integers(0, 2, symbols) - 1.0) + 0.5j * (2.0 * rng.integers(0, 2, symbols) - 1.0)
    up = np.zeros(symbols * sps, dtype=np.complex128)
    up[::sps] = sym
    # Shaped in float64 with correctly rounded sums, starting one pulse into the stream.
    x = conv_exact(up, pulse.astype(np.complex128), taps + n_in)[taps:]
    out, _ = result(
        x,
        pulse.astype(np.float32).astype(np.complex64),
        n_fft,
        {"rolloff": 0.35, "sps": sps, "span": 16},
        "QPSK (+-0.5 per component) at 4 samples per symbol shaped by the unit-energy RRC (rolloff 0.35, "
        "span 16), cast to cf32; taps = the RRC in float32 as complex (the matched filter); expected = the "
        "exact convolution numpy.convolve(x, h)[:rows]; zero history",
        # Regression guard: the fallback reaches 135.94 dB (the reference 147.00 dB).
        134.9,
    )
    return out


@case(kernel="xcorr_fft", name="rand_300_n1024", seed=703)
def rand_300_n1024(rng):
    # General convolution: complex Gaussian input of unit power and complex Gaussian taps of total
    # energy 1.
    taps, n_fft = 300, 1024
    b = n_fft - taps + 1
    x = noise(rng, 4 * b, 1.0)
    h = noise(rng, taps, 1.0 / taps)
    out, _ = result(
        x,
        h,
        n_fft,
        {},
        "complex Gaussian input (power 1 per sample) and complex Gaussian taps (energy 1 in total); "
        "expected = the exact convolution numpy.convolve(x, h)[:rows]; zero history",
        # Regression guard: the fallback reaches 133.29 dB (the reference 142.76 dB).
        132.2,
    )
    return out
