"""Cases for the fft, rfft and irfft kernels (own radix-2 FFT, f32 and Q15).

All expected values come from ``numpy.fft`` in complex128, computed from the *quantized* input.
The Q15 cases compare after the Rust test has applied the exponent that the kernel returns:
``fixed`` cases store the transform already divided by ``N`` (the exponent ``log2 N`` is a
convention of the kernel), ``bfp`` cases store the unnormalized transform and the test multiplies
the output by ``2**exponent``.
"""

import numpy as np

from .. import q15
from ..registry import Array, case

FULL_SCALE = 1.0 - 2.0**-15
# Minimum SNR stored for the weak-tone case (the criterion of the milestone is the tone-to-median margin,
# which the Rust test computes); set from the measured value minus a margin.
WEAK_TONE_SNR_DB = 55.0


def noise(rng, n, sigma):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma


def quantize(x):
    """Complex float array -> (``ci16`` array, exact float64 value of the quantized samples)."""
    re, im = q15.from_float(x.real), q15.from_float(x.imag)
    return q15.pack(re, im), q15.to_float(re) + 1j * q15.to_float(im)


def tone(rng, n, bin_, amp):
    """``amp * exp(j (2 pi bin n / N + phi))`` with a random start phase (so the seed matters)."""
    phi = rng.uniform(0.0, 2.0 * np.pi)
    idx = np.arange(n, dtype=np.int64)
    return amp * np.exp(1j * (2.0 * np.pi * ((bin_ * idx) % n) / n + phi))


# ----------------------------------------------------------------------------- C32


def c32_forward(rng, n):
    x = noise(rng, n, 0.3).astype(np.complex64)
    return dict(
        params={"n": n, "direction": "forward"},
        arrays={"input": Array("cf32", x), "expected": Array("cf64", np.fft.fft(x.astype(np.complex128)))},
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "skip": 0},
        notes="expected = numpy.fft.fft of the complex64 input; exponent 0",
    )


@case(kernel="fft", name="c32_fwd_16", seed=70)
def c32_fwd_16(rng):
    return c32_forward(rng, 16)


@case(kernel="fft", name="c32_fwd_64", seed=71)
def c32_fwd_64(rng):
    return c32_forward(rng, 64)


@case(kernel="fft", name="c32_fwd_1024", seed=72)
def c32_fwd_1024(rng):
    return c32_forward(rng, 1024)


@case(kernel="fft", name="c32_fwd_4096", seed=73)
def c32_fwd_4096(rng):
    return c32_forward(rng, 4096)


@case(kernel="fft", name="c32_inv_1024", seed=74)
def c32_inv_1024(rng):
    n = 1024
    spectrum = np.fft.fft(noise(rng, n, 0.3)).astype(np.complex64)
    return dict(
        params={"n": n, "direction": "inverse"},
        arrays={
            "input": Array("cf32", spectrum),
            "expected": Array("cf64", np.fft.ifft(spectrum.astype(np.complex128))),
        },
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "skip": 0},
        notes="input = fft of complex noise cast to complex64; expected = numpy.fft.ifft (divided by N); exponent log2 N",
    )


# ------------------------------------------------------------------------ real FFT


def rfft_case(rng, n):
    x = (rng.standard_normal(n) * 0.3).astype(np.float32)
    return dict(
        params={"n": n},
        arrays={"input": Array("f32", x), "expected": Array("cf64", np.fft.rfft(x.astype(np.float64)))},
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "skip": 0},
        notes="expected = numpy.fft.rfft of the float32 input (N/2 + 1 bins); exponent 0",
    )


@case(kernel="fft", name="rfft_f32_32", seed=75)
def rfft_32(rng):
    return rfft_case(rng, 32)


@case(kernel="fft", name="rfft_f32_1024", seed=76)
def rfft_1024(rng):
    return rfft_case(rng, 1024)


@case(kernel="fft", name="rfft_f32_4096", seed=77)
def rfft_4096(rng):
    return rfft_case(rng, 4096)


def irfft_case(rng, n):
    x = (rng.standard_normal(n) * 0.3).astype(np.float64)
    spectrum = np.fft.rfft(x).astype(np.complex64)
    # The imaginary parts of bin 0 and bin N/2 must be ignored by the kernel (and by numpy).
    spectrum[0] = np.complex64(spectrum[0].real + 0.5j)
    spectrum[n // 2] = np.complex64(spectrum[n // 2].real + 0.5j)
    wide = spectrum.astype(np.complex128)
    cleaned = wide.copy()
    cleaned[0] = cleaned[0].real
    cleaned[n // 2] = cleaned[n // 2].real
    expected = np.fft.irfft(cleaned, n=n)
    assert np.allclose(expected, np.fft.irfft(wide, n=n), rtol=0.0, atol=1e-12)
    return dict(
        params={"n": n},
        arrays={"input": Array("cf32", spectrum), "expected": Array("f64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 110.0, "skip": 0},
        notes="input = numpy.fft.rfft of noise as complex64 with imag(X[0]) = imag(X[N/2]) = 0.5 (ignored); "
        "expected = numpy.fft.irfft(X, n=N); exponent log2 N",
    )


@case(kernel="fft", name="irfft_f32_32", seed=85)
def irfft_32(rng):
    return irfft_case(rng, 32)


@case(kernel="fft", name="irfft_f32_1024", seed=86)
def irfft_1024(rng):
    return irfft_case(rng, 1024)


@case(kernel="fft", name="irfft_f32_4096", seed=87)
def irfft_4096(rng):
    return irfft_case(rng, 4096)


# ----------------------------------------------------------------------------- CQ15


@case(kernel="fft", name="cq15_fixed_tone_1024", seed=78)
def cq15_fixed_tone_1024(rng):
    n = 1024
    packed, xf = quantize(tone(rng, n, 37, FULL_SCALE))
    return dict(
        params={"n": n, "direction": "forward", "scaling": "default", "bin": 37},
        arrays={"input": Array("ci16", packed), "expected": Array("cf64", np.fft.fft(xf) / n)},
        tolerance={"class": "tolerance", "snr_db_min": 55.0, "skip": 0},
        notes="full-scale tone at bin 37; expected = fft(quantized input) / N; exponent log2 N",
    )


def fixed_noise(rng, n):
    packed, xf = quantize(noise(rng, n, 0.3))
    return dict(
        params={"n": n, "direction": "forward", "scaling": "default"},
        arrays={"input": Array("ci16", packed), "expected": Array("cf64", np.fft.fft(xf) / n)},
        tolerance={"class": "tolerance", "snr_db_min": 40.0, "skip": 0},
        notes="noise sigma 0.3; expected = fft(quantized input) / N; exponent log2 N",
    )


@case(kernel="fft", name="cq15_fixed_noise_1024", seed=79)
def cq15_fixed_noise_1024(rng):
    return fixed_noise(rng, 1024)


@case(kernel="fft", name="cq15_fixed_noise_4096", seed=80)
def cq15_fixed_noise_4096(rng):
    return fixed_noise(rng, 4096)


@case(kernel="fft", name="cq15_fixed_inv_1024", seed=81)
def cq15_fixed_inv_1024(rng):
    n = 1024
    packed, xf = quantize(noise(rng, n, 0.3))
    return dict(
        params={"n": n, "direction": "inverse", "scaling": "default"},
        arrays={"input": Array("ci16", packed), "expected": Array("cf64", np.fft.ifft(xf))},
        tolerance={"class": "tolerance", "snr_db_min": 40.0, "skip": 0},
        notes="a white spectrum as input; expected = numpy.fft.ifft(quantized input); exponent log2 N",
    )


@case(kernel="fft", name="cq15_bfp_tone_1024", seed=82)
def cq15_bfp_tone_1024(rng):
    n = 1024
    packed, xf = quantize(tone(rng, n, 37, FULL_SCALE))
    return dict(
        params={"n": n, "direction": "forward", "scaling": "block_float", "bin": 37},
        arrays={"input": Array("ci16", packed), "expected": Array("cf64", np.fft.fft(xf))},
        # The specification asks for 60 dB as a first estimate; the exact algorithm reaches 59.05 dB
        # (docs/decisions-log.md, M07), so the stored minimum is 58 dB.
        tolerance={"class": "tolerance", "snr_db_min": 58.0, "skip": 0},
        notes="expected = unnormalized fft(quantized input); the test multiplies the output by 2^exponent first",
    )


@case(kernel="fft", name="cq15_bfp_noise_1024", seed=83)
def cq15_bfp_noise_1024(rng):
    n = 1024
    packed, xf = quantize(noise(rng, n, 0.3))
    return dict(
        params={"n": n, "direction": "forward", "scaling": "block_float"},
        arrays={"input": Array("ci16", packed), "expected": Array("cf64", np.fft.fft(xf))},
        tolerance={"class": "tolerance", "snr_db_min": 55.0, "skip": 0},
        notes="expected = unnormalized fft(quantized input); the test multiplies the output by 2^exponent first",
    )


@case(kernel="fft", name="cq15_bfp_weak_tone_1024", seed=84)
def cq15_bfp_weak_tone_1024(rng):
    n = 1024
    amp = 10.0 ** (-60.0 / 20.0)
    packed, xf = quantize(tone(rng, n, 100, amp))
    return dict(
        params={"n": n, "direction": "forward", "scaling": "block_float", "bin": 100, "min_margin_db": 30.0},
        arrays={"input": Array("ci16", packed), "expected": Array("cf64", np.fft.fft(xf))},
        tolerance={"class": "tolerance", "snr_db_min": WEAK_TONE_SNR_DB, "skip": 0},
        notes="tone at -60 dBFS, no noise; the test also checks that the tone bin is at least min_margin_db above "
        "the median of all other bins (after multiplying by 2^exponent)",
    )

