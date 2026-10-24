"""Cases for the convert_out kernel (C32/CQ15 or f32/Q15 samples -> raw device bytes).

The float path (``from_c32``) follows the specification: ``v = x * full_scale + offset`` in float64
(NaN counts as amplitude 0), ``raw`` is ``v`` rounded to the nearest integer with ties to even
(``numpy.rint``), saturated to the range of the format. The integer path (``from_cq15``) uses the
Q15 rules, written here as exact integer divisions rounded to the nearest integer with ties to even:
U8 ``(q + 255 * 2^(15 - bits)) / 2^(16 - bits)``, signed power-of-two full scale with at most
16 bits ``q / 2^(16 - bits)``, more than 16 bits ``q << (bits - 16)`` and an explicit full scale
``n``: ``q n / 32768`` (U8: ``(q n + 255 * 16384) / 32768``). Alignment, packing, swap, planar and
byte order reuse the packing code of ``convert.py``, which is written from ``docs/device.md``.
"""

import numpy as np

from .. import q15
from ..registry import Array, case
from .convert import INTEGER_FORMATS, N, arrange, full_scale_of, pack_bytes

# Same formats as the convert cases; the two variants follow the table in convert.py.
FORMATS = INTEGER_FORMATS + [
    ("cu8_swapiq", "u8", 8, False, "interleaved", "lsb", "little", True, None),
    ("cs16_bigendian", "i16", 16, True, "interleaved", "lsb", "big", False, None),
]


def raw_range(bits, signed):
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def div_even(a, d):
    """Integer division a / d (d > 0) rounded to the nearest integer, ties to even."""
    q = a // d
    r2 = 2 * (a - q * d)
    return q + ((r2 > d) | ((r2 == d) & (q % 2 == 1)))


def float_raw(x, container, bits, signed, full_scale):
    """Effective raw values of f32 amplitudes: rint(x * full_scale + offset) (ties to even), saturated."""
    a = np.where(np.isnan(x), 0.0, x.astype(np.float64))
    off = 127.5 if container == "u8" else 0.0
    v = np.rint(a * full_scale_of(bits, full_scale) + off)
    lo, hi = raw_range(bits, signed)
    return np.clip(v, lo, hi).astype(np.int64)


def q15_raw(q, container, bits, signed, full_scale):
    """Effective raw values of Q15 samples by the integer rules (ties to even)."""
    q = q.astype(np.int64)
    lo, hi = raw_range(bits, signed)
    if container == "u8":
        if full_scale is None:
            s = 15 - bits
            raw = div_even(q + (255 << s), 1 << (s + 1))
        else:
            n = int(full_scale)
            raw = div_even(q * n + 255 * 16384, 32768)
    elif full_scale is not None:
        n = int(full_scale)
        raw = div_even(q * n, 32768)
    elif bits <= 16:
        raw = div_even(q, 1 << (16 - bits))
    else:
        raw = q << (bits - 16)
    return np.clip(raw, lo, hi)


def to_complex(x):
    """Interleaved (re, im) f32 values as complex64 (assigned per part, so inf and NaN survive)."""
    z = np.empty(len(x) // 2, dtype=np.complex64)
    z.real = x[0::2]
    z.imag = x[1::2]
    return z


def float_bytes(stream):
    return np.asarray(stream, dtype="<f4").tobytes()


def params_of(spec, samples):
    _name, container, bits, signed, layout, align, endian, swap, full_scale = spec
    return {
        "container": container,
        "bits": bits,
        "signed": signed,
        "layout": layout,
        "align": align,
        "endian": endian,
        "swap_iq": swap,
        "full_scale": full_scale,
        "samples": samples,
    }


def draw_c32(rng, spec):
    """Random values in (-1.05, 1.05) with NaN, +-1, +-inf and values on the +-1/2 LSB boundaries."""
    _name, container, bits, signed, layout, _align, _endian, _swap, full_scale = spec
    count = N if layout == "real" else 2 * N
    x = rng.uniform(-1.05, 1.05, size=count).astype(np.float32)
    fs = full_scale_of(bits, full_scale)
    off = 127.5 if container == "u8" else 0.0
    lo, hi = raw_range(bits, signed)
    n = rng.integers(lo + 1, hi, size=6)
    ties = ((n - 0.5 - off) / fs).astype(np.float32)  # v is exactly halfway between two integers there
    specials = np.concatenate(
        [np.array([np.nan, 1.0, -1.0, np.inf, -np.inf, 0.0], dtype=np.float32), ties, -ties]
    )
    idx = rng.permutation(count)[: len(specials)]
    x[idx] = specials
    return x


def draw_q15(rng, spec):
    layout = spec[4]
    count = N if layout == "real" else 2 * N
    q = rng.integers(-32768, 32768, size=count, dtype=np.int64)
    q[0] = -32768
    q[1] = 32767
    return q


def encode(vals_raw, spec):
    """Raw effective values (I part then Q part, or the real values) -> device bytes."""
    _name, container, bits, signed, layout, align, endian, swap, _fs = spec
    stream, _i, _q = arrange(vals_raw, layout, swap)
    return pack_bytes(stream, container, bits, signed, align, endian)


def build_c32(rng, spec):
    name, container, bits, signed, layout, _align, _endian, _swap, full_scale = spec
    x = draw_c32(rng, spec)
    if layout == "real":
        arr = Array("f32", x)
        vals = float_raw(x, container, bits, signed, full_scale)
    else:
        arr = Array("cf32", to_complex(x))
        vals = np.concatenate(
            [float_raw(x[0::2], container, bits, signed, full_scale), float_raw(x[1::2], container, bits, signed, full_scale)]
        )
    return dict(
        params=params_of(spec, N),
        arrays={"input": arr, "expected": Array("u8", np.frombuffer(encode(vals, spec), dtype=np.uint8))},
        tolerance={"class": "bit_exact"},
        notes="float path: x * full_scale + offset in float64 rounded to nearest, ties to even, NaN as 0, saturated; bytes as in docs/device.md",
    )


def build_cq15(rng, spec):
    name, container, bits, signed, layout, _align, _endian, _swap, full_scale = spec
    q = draw_q15(rng, spec)
    if layout == "real":
        arr = Array("i16", q)
        vals = q15_raw(q, container, bits, signed, full_scale)
    else:
        arr = Array("ci16", q15.pack(q[0::2], q[1::2]))
        vals = np.concatenate(
            [q15_raw(q[0::2], container, bits, signed, full_scale), q15_raw(q[1::2], container, bits, signed, full_scale)]
        )
    return dict(
        params=params_of(spec, N),
        arrays={"input": arr, "expected": Array("u8", np.frombuffer(encode(vals, spec), dtype=np.uint8))},
        tolerance={"class": "bit_exact"},
        notes="integer path: Q15 division by a power of two or by the explicit full scale rounded to nearest, ties to even, or left shift by the bit depth",
    )


def _register(index, spec):
    case(kernel="convert_out", name=f"{spec[0]}_from_c32_{N}", seed=200 + index)(
        lambda rng, spec=spec: build_c32(rng, spec)
    )
    case(kernel="convert_out", name=f"{spec[0]}_from_cq15_{N}", seed=220 + index)(
        lambda rng, spec=spec: build_cq15(rng, spec)
    )


for _index, _spec in enumerate(FORMATS):
    _register(_index, _spec)


def build_float_container(rng, layout, from_q15):
    """F32 containers: the f32 bits are copied (Q15 input gives the exact q / 32768)."""
    count = N if layout == "real" else 2 * N
    if from_q15:
        q = rng.integers(-32768, 32768, size=count, dtype=np.int64)
        q[0], q[1] = -32768, 32767
        stream = q.astype(np.float64) / 32768.0
        arr = Array("i16", q) if layout == "real" else Array("ci16", q15.pack(q[0::2], q[1::2]))
        stream = stream.astype(np.float32)
    else:
        x = (rng.standard_normal(count) * 0.7).astype(np.float32)
        specials = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0, -1.0, 3.0, 1e-40], dtype=np.float32)
        x[rng.permutation(count)[: len(specials)]] = specials
        stream = x
        arr = Array("f32", x) if layout == "real" else Array("cf32", to_complex(x))
    spec = ("f32", "f32", 32, True, layout, "lsb", "little", False, None)
    return dict(
        params=params_of(spec, N),
        arrays={"input": arr, "expected": Array("u8", np.frombuffer(float_bytes(stream), dtype=np.uint8))},
        tolerance={"class": "bit_exact"},
        notes="f32 containers copy the value (NaN, inf and -0.0 unchanged); Q15 input gives q / 32768 exactly",
    )


@case(kernel="convert_out", name="cf32_from_c32_256", seed=212)
def cf32_from_c32(rng):
    return build_float_container(rng, "interleaved", False)


@case(kernel="convert_out", name="rf32_from_c32_256", seed=213)
def rf32_from_c32(rng):
    return build_float_container(rng, "real", False)


@case(kernel="convert_out", name="cf32_from_cq15_256", seed=232)
def cf32_from_cq15(rng):
    return build_float_container(rng, "interleaved", True)


@case(kernel="convert_out", name="rf32_from_cq15_256", seed=233)
def rf32_from_cq15(rng):
    return build_float_container(rng, "real", True)
