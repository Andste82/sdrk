"""Cases for the scramble kernel (M49): keyed phase scrambling ``x · exp(j 2π k / 2^b)``.

The expected values come from the f64 reference of the contract (``docs/kernels.md``): the index
``k = p & (2^b - 1)`` (``inverse``: ``(-p) & (2^b - 1)``) is split into the quadrant ``q`` and
the residue ``r`` (``q = k >> (b - 2)``, ``r = k & (2^(b-2) - 1)`` for ``b >= 2``; ``q = 2k``,
``r = 0`` for ``b = 1``); for ``r != 0`` the input is multiplied by the f64 phasor
``(cos θ, sin θ)``, ``θ = 2π r / 2^b`` (``numpy.cos``/``numpy.sin``), then the exact quadrant map
``j^q`` (negation and swap) is applied. The cosine-table model of ``rotate`` is not needed.

- ``C32`` cases with ``b <= 2`` are ``bit_exact``: the quadrant map is exact in float32
  (``cf32`` expected).
- The ``CQ15`` case with ``b = 2`` is ``bit_exact`` with the integer model (``q15.neg_sat``, the
  saturating negation); its input includes the components -32768 and 32767.
- The ``tolerance`` cases (``b = 3`` and ``b = 8``) store the f64 result (``cf64``, for ``CQ15``
  clipped to the Q15 range like the ``rotate`` cases) and the SNR floors of the ``rotate``
  contract: 120 dB (``C32``) and 80 dB (``CQ15``).

Draw order: the input samples, then the phase bytes (``rng.integers``, ``u8``).

DVB-S2 cross-check: the physical-layer scrambling multiplies payload symbols by ``j^{R_n}``,
``R_n`` in {0, 1, 2, 3}, which is exactly ``exp(j 2π R_n / 4)``; ``scramble`` with ``bits = 2``
and ``p = R_n`` reproduces it exactly in both types (the quadrant map, as GNU Radio
``dvbs2_physical_cc`` does it by swap and negation). The ``R_n`` generation is not part of the
kernel.
"""

import numpy as np

from .. import q15
from ..registry import Array, case

N = 2048
SNR_C32 = 120.0
SNR_CQ15 = 80.0


def indices(phase, bits, inverse):
    """The masked (and for ``inverse`` negated) phasor indices of the phase bytes."""
    k = phase.astype(np.int64)
    if inverse:
        k = -k
    return k & ((1 << bits) - 1)


def split(k, bits):
    """Quadrant ``q`` and residue ``r`` of the indices ``k``."""
    if bits == 1:
        return 2 * k, np.zeros_like(k)
    s = bits - 2
    return k >> s, k & ((1 << s) - 1)


def quadrant(re, im, q, neg):
    """``(re, im)`` times ``j^q`` by negation (``neg``) and swap."""
    q = q & 3
    out_re = np.where(q == 0, re, np.where(q == 1, neg(im), np.where(q == 2, neg(re), im)))
    out_im = np.where(q == 0, im, np.where(q == 1, re, np.where(q == 2, neg(im), neg(re))))
    return out_re, out_im


def reference_f64(x, phase, bits, inverse):
    """The f64 reference of the contract for complex input ``x`` (complex128)."""
    q, r = split(indices(phase, bits, inverse), bits)
    theta = 2.0 * np.pi * r.astype(np.float64) / float(1 << bits)
    c, s = np.cos(theta), np.sin(theta)
    re, im = x.real.copy(), x.imag.copy()
    rot = r != 0
    re_r = re * c - im * s
    im_r = re * s + im * c
    re = np.where(rot, re_r, re)
    im = np.where(rot, im_r, im)
    out_re, out_im = quadrant(re, im, q, np.negative)
    # Assembled from the parts so that signed zeros keep their sign (``re + 1j * im`` does not).
    out = np.empty(len(out_re), dtype=np.complex128)
    out.real = out_re
    out.imag = out_im
    return out


def c32_input(rng, specials):
    """Complex Gaussian noise of sigma 0.5 as complex64; with ``specials`` the first eight
    samples are signed zeros, tiny, subnormal and large values (bit-exact cases only)."""
    x = ((rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 0.5).astype(np.complex64)
    if specials:
        re = x.real.copy()
        im = x.imag.copy()
        sv = np.array(
            [
                (0.0, -0.0),
                (-0.0, 0.0),
                (-0.0, -0.0),
                (1e-45, -1e-45),
                (-1.2e-38, 3.0e38),
                (-3.0e38, 1.0),
                (1.0, -1.0),
                (0.5, 0.0),
            ],
            dtype=np.float32,
        )
        re[: len(sv)] = sv[:, 0]
        im[: len(sv)] = sv[:, 1]
        # Assemble from the parts so that signed zeros keep their sign.
        x = np.empty(N, dtype=np.complex64)
        x.real = re
        x.imag = im
    return x


def c32_case(rng, bits, inverse, exact):
    x = c32_input(rng, exact)
    phase = rng.integers(0, 1 << bits, size=N, dtype=np.uint8)
    y = reference_f64(x.astype(np.complex128), phase, bits, inverse)
    if exact:
        expected = np.empty(N, dtype=np.complex64)
        expected.real = y.real.astype(np.float32)
        expected.imag = y.imag.astype(np.float32)
        arr = Array("cf32", expected)
        tol = {"class": "bit_exact"}
    else:
        arr = Array("cf64", y)
        tol = {"class": "tolerance", "snr_db_min": SNR_C32, "skip": 0}
    return dict(
        params={"bits": bits, "inverse": inverse},
        arrays={"input": Array("cf32", x), "phase": Array("u8", phase), "expected": arr},
        tolerance=tol,
        notes=f"C32, bits {bits}, inverse {inverse}: quadrant map j^q of x (r = 0) or of the f64 "
        "product with exp(j 2 pi r / 2^b) (r != 0)",
    )


@case(kernel="scramble", name="bpsk_c32_2048", seed=690)
def bpsk_c32(rng):
    # The 0/1 phase sequence comes from numpy.random.default_rng(seed) (R8.2).
    return c32_case(rng, 1, False, True)


@case(kernel="scramble", name="qpsk_c32_2048", seed=691)
def qpsk_c32(rng):
    return c32_case(rng, 2, False, True)


@case(kernel="scramble", name="oct_c32_2048", seed=692)
def oct_c32(rng):
    return c32_case(rng, 3, False, False)


@case(kernel="scramble", name="qpsk_inv_c32_2048", seed=693)
def qpsk_inv_c32(rng):
    return c32_case(rng, 2, True, True)


@case(kernel="scramble", name="byte_inv_c32_2048", seed=699)
def byte_inv_c32(rng):
    return c32_case(rng, 8, True, False)


# The Q15 corners in the first samples of the CQ15 cases.
CORNERS = [
    (-32768, -32768),
    (-32768, 32767),
    (32767, -32768),
    (32767, 32767),
    (-32768, 0),
    (0, -32768),
    (-1, 1),
    (0, 0),
]


@case(kernel="scramble", name="qpsk_cq15_2048", seed=694)
def qpsk_cq15(rng):
    bits = 2
    x = rng.integers(-32768, 32768, size=(N, 2), dtype=np.int64)
    x[: len(CORNERS)] = np.array(CORNERS, dtype=np.int64)
    phase = rng.integers(0, 1 << bits, size=N, dtype=np.uint8)
    q, _ = split(indices(phase, bits, False), bits)
    re, im = quadrant(x[:, 0], x[:, 1], q, q15.neg_sat)
    return dict(
        params={"bits": bits, "inverse": False},
        arrays={
            "input": Array("ci16", q15.pack(x[:, 0], x[:, 1])),
            "phase": Array("u8", phase),
            "expected": Array("ci16", q15.pack(re, im)),
        },
        tolerance={"class": "bit_exact"},
        notes="CQ15, bits 2: the quadrant map j^q with the saturating negation (q15.neg_sat, "
        "-(-32768) = 32767); uniform full-range input with the Q15 corners first",
    )


@case(kernel="scramble", name="byte_cq15_2048", seed=698)
def byte_cq15(rng):
    bits = 8
    xf = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 0.3
    re, im = q15.from_float(xf.real), q15.from_float(xf.imag)
    corners = np.array(CORNERS, dtype=np.int64)
    re[: len(CORNERS)] = corners[:, 0]
    im[: len(CORNERS)] = corners[:, 1]
    phase = rng.integers(0, 1 << bits, size=N, dtype=np.uint8)
    y = reference_f64(q15.to_float(re) + 1j * q15.to_float(im), phase, bits, False)
    # The Q15 kernel saturates, so the ideal output is clipped to the Q15 range as well.
    top = 32767.0 / 32768.0
    expected = np.clip(y.real, -1.0, top) + 1j * np.clip(y.imag, -1.0, top)
    return dict(
        params={"bits": bits, "inverse": False},
        arrays={
            "input": Array("ci16", q15.pack(re, im)),
            "phase": Array("u8", phase),
            "expected": Array("cf64", expected),
        },
        tolerance={"class": "tolerance", "snr_db_min": SNR_CQ15, "skip": 0},
        notes="CQ15, bits 8: f64 reference of the quantized input (noise sigma 0.3, the Q15 "
        "corners first), clipped to the Q15 range",
    )
