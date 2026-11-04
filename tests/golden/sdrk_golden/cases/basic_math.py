"""Cases for the element-wise kernels ``add``, ``add_real``, ``mul``, ``mul_real`` and ``interleave``.

The float32 results are the single IEEE operations of NumPy ``float32`` arrays (a product is
rounded before the difference or sum, nothing is fused) and the Q15 results come from the integer
model in ``q15.py`` (rule R7). All cases are bit-exact except ``mul_c32_4096``: the SIMD backend
fuses the complex product on AVX2, AVX-512 and NEON (D326), so that case has an SNR floor against
the float32 model, which the fallback and the reference meet bit for bit.
"""

import numpy as np

from .. import q15
from ..registry import Array, case

N = 4096
SIGMA = 0.4
CORNERS = 64

# Corner values of the Q15 and f32 inputs (full scale, one LSB, zero).
Q15_CORNERS = np.array([-32768, -32767, -1, 0, 1, 32766, 32767], dtype=np.int64)
F32_CORNERS = np.array([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], dtype=np.float32)


def noise_f32(rng, n):
    return (rng.standard_normal(n) * SIGMA).astype(np.float32)


def pack_c32(re, im):
    """A complex64 array from float32 parts, bit for bit (no complex arithmetic on the parts)."""
    out = np.empty(len(re), dtype=np.complex64)
    out.real = re
    out.imag = im
    return out


def noise_c32(rng, n):
    return pack_c32(noise_f32(rng, n), noise_f32(rng, n))


def noise_q15(rng, n):
    return q15.from_float(rng.standard_normal(n) * SIGMA)


def saturating_q15(rng, n):
    """Noise whose first ``CORNERS`` samples are at plus or minus full scale."""
    x = noise_q15(rng, n)
    x[:CORNERS] = np.where(rng.integers(0, 2, CORNERS) == 1, 32767, -32768)
    return x


def corners_q15(rng, n):
    """Noise whose first ``CORNERS`` samples are drawn from the corner values."""
    x = noise_q15(rng, n)
    x[:CORNERS] = rng.choice(Q15_CORNERS, CORNERS)
    return x


def corners_f32(rng, n):
    x = noise_f32(rng, n)
    x[:CORNERS] = rng.choice(F32_CORNERS, CORNERS)
    return x


def c32_mul(a, b):
    """``re = ar*br - ai*bi``, ``im = ar*bi + ai*br`` with float32 products rounded first."""
    re = (a.real * b.real) - (a.imag * b.imag)
    im = (a.real * b.imag) + (a.imag * b.real)
    return pack_c32(re, im)


def with_corner_pairs(a_re, a_im, b_re, b_im):
    """Put the pairs (-1)*(-1) and (-1)*(-1 + j*-1) at the start (Q15 raw values)."""
    a_re[:2], a_im[:2] = -32768, 0
    b_re[:2], b_im[:2] = -32768, 0
    a_re[1], b_im[1] = -32768, -32768


# ---------------------------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------------------------


@case(kernel="add", name="add_c32_4096", seed=463)
def add_c32(rng):
    a, b = noise_c32(rng, N), noise_c32(rng, N)
    return dict(
        params={},
        arrays={"a": Array("cf32", a), "b": Array("cf32", b), "expected": Array("cf32", a + b)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="expected = a + b per component in float32; noise sigma 0.4",
    )


@case(kernel="add", name="add_cq15_4096", seed=463)
def add_cq15(rng):
    a_re, a_im = saturating_q15(rng, N), saturating_q15(rng, N)
    b_re, b_im = saturating_q15(rng, N), saturating_q15(rng, N)
    return dict(
        params={},
        arrays={
            "a": Array("ci16", q15.pack(a_re, a_im)),
            "b": Array("ci16", q15.pack(b_re, b_im)),
            "expected": Array("ci16", q15.pack(q15.add(a_re, b_re), q15.add(a_im, b_im))),
        },
        tolerance={"class": "bit_exact"},
        notes="expected = sat16(a + b) per component; noise sigma 0.4 quantized, the first 64 samples "
        "of every component are at plus or minus full scale",
    )


@case(kernel="add_real", name="add_f32_4096", seed=464)
def add_f32(rng):
    a, b = noise_f32(rng, N), noise_f32(rng, N)
    return dict(
        params={},
        arrays={"a": Array("f32", a), "b": Array("f32", b), "expected": Array("f32", a + b)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="expected = a + b in float32; noise sigma 0.4",
    )


@case(kernel="add_real", name="add_q15_4096", seed=464)
def add_q15(rng):
    a, b = saturating_q15(rng, N), saturating_q15(rng, N)
    return dict(
        params={},
        arrays={"a": Array("i16", a), "b": Array("i16", b), "expected": Array("i16", q15.add(a, b))},
        tolerance={"class": "bit_exact"},
        notes="expected = sat16(a + b); noise sigma 0.4 quantized, the first 64 samples at plus or "
        "minus full scale",
    )


# ---------------------------------------------------------------------------------------------
# mul
# ---------------------------------------------------------------------------------------------


@case(kernel="mul", name="mul_c32_4096", seed=465)
def mul_c32(rng):
    a, b = noise_c32(rng, N), noise_c32(rng, N)
    a.real[:CORNERS] = rng.choice(F32_CORNERS, CORNERS)
    a.imag[:CORNERS] = rng.choice(F32_CORNERS, CORNERS)
    b.real[:CORNERS] = rng.choice(F32_CORNERS, CORNERS)
    b.imag[:CORNERS] = rng.choice(F32_CORNERS, CORNERS)
    a[0], b[0] = np.complex64(-1 + 0j), np.complex64(-1 + 0j)
    a[1], b[1] = np.complex64(-1 + 0j), np.complex64(-1 - 1j)
    return dict(
        params={},
        arrays={"a": Array("cf32", a), "b": Array("cf32", b), "expected": Array("cf32", c32_mul(a, b))},
        tolerance={"class": "tolerance", "snr_db_min": 140.0, "skip": 0},
        model="float32",
        notes="re = ar*br - ai*bi, im = ar*bi + ai*br with the products rounded to float32 first "
        "(no fused multiply-add), which the fallback and the reference give bit for bit; the SIMD "
        "backend fuses one product of each part on AVX2, AVX-512 and NEON (D326), hence the SNR "
        "floor; noise sigma 0.4, the first 64 samples are corner values",
    )


@case(kernel="mul", name="mul_cq15_4096", seed=465)
def mul_cq15(rng):
    a_re, a_im = noise_q15(rng, N), noise_q15(rng, N)
    b_re, b_im = noise_q15(rng, N), noise_q15(rng, N)
    for v in (a_re, a_im, b_re, b_im):
        v[:CORNERS] = rng.choice(Q15_CORNERS, CORNERS)
    with_corner_pairs(a_re, a_im, b_re, b_im)
    re, im = q15.cmul(a_re, a_im, b_re, b_im)
    return dict(
        params={},
        arrays={
            "a": Array("ci16", q15.pack(a_re, a_im)),
            "b": Array("ci16", q15.pack(b_re, b_im)),
            "expected": Array("ci16", q15.pack(re, im)),
        },
        tolerance={"class": "bit_exact"},
        notes="one rounding and saturation per component on the i64 sums of products; noise sigma "
        "0.4 quantized, the first 64 samples are corner values (-1 * -1 gives 32767)",
    )


@case(kernel="mul_real", name="mul_f32_4096", seed=466)
def mul_f32(rng):
    a, b = corners_f32(rng, N), corners_f32(rng, N)
    return dict(
        params={},
        arrays={"a": Array("f32", a), "b": Array("f32", b), "expected": Array("f32", a * b)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="expected = a * b in float32; noise sigma 0.4, the first 64 samples are corner values",
    )


@case(kernel="mul_real", name="mul_q15_4096", seed=466)
def mul_q15(rng):
    a, b = corners_q15(rng, N), corners_q15(rng, N)
    a[0], b[0] = -32768, -32768
    return dict(
        params={},
        arrays={"a": Array("i16", a), "b": Array("i16", b), "expected": Array("i16", q15.mul(a, b))},
        tolerance={"class": "bit_exact"},
        notes="expected = sat16((a * b + 2^14) >> 15); noise sigma 0.4 quantized, the first 64 "
        "samples are corner values",
    )


# ---------------------------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------------------------


@case(kernel="interleave", name="interleave_c32_1024", seed=468)
def interleave_c32(rng):
    re, im = noise_f32(rng, 1024), noise_f32(rng, 1024)
    both = pack_c32(re, im)
    only_re = pack_c32(re, np.zeros(1024, dtype=np.float32))
    return dict(
        params={},
        arrays={
            "re": Array("f32", re),
            "im": Array("f32", im),
            "expected": Array("cf32", both),
            "expected_no_im": Array("cf32", only_re),
        },
        tolerance={"class": "bit_exact"},
        notes="expected[n] = (re[n], im[n]); expected_no_im[n] = (re[n], 0); deinterleave of expected "
        "gives re and im back",
    )


@case(kernel="interleave", name="interleave_cq15_1024", seed=468)
def interleave_cq15(rng):
    re, im = noise_q15(rng, 1024), noise_q15(rng, 1024)
    return dict(
        params={},
        arrays={
            "re": Array("i16", re),
            "im": Array("i16", im),
            "expected": Array("ci16", q15.pack(re, im)),
            "expected_no_im": Array("ci16", q15.pack(re, np.zeros(1024, dtype=np.int64))),
        },
        tolerance={"class": "bit_exact"},
        notes="expected[n] = (re[n], im[n]); expected_no_im[n] = (re[n], 0); deinterleave of expected "
        "gives re and im back",
    )
