"""Cases for the convert kernel (raw device bytes -> C32/CQ15 or f32/Q15).

The packing below is written from ``docs/device.md`` and shares no code with the
Rust implementation. Raw values are drawn uniformly over the full integer range
of the format; the first two values of every case are forced to the range limits.
"""

import numpy as np

from .. import q15
from ..registry import Array, case

N = 256

# (name, container, bits, signed, layout, align, endian, swap_iq, full_scale, seed)
# Order and seeds follow the constant table of docs/device.md (seed = 100 + position).
INTEGER_FORMATS = [
    ("cu8", "u8", 8, False, "interleaved", "lsb", "little", False, None),
    ("cs8", "i8", 8, True, "interleaved", "lsb", "little", False, None),
    ("cs16", "i16", 16, True, "interleaved", "lsb", "little", False, None),
    ("cs12_lsb16", "i16", 12, True, "interleaved", "lsb", "little", False, None),
    ("sc16_q11", "i16", 12, True, "interleaved", "lsb", "little", False, 2048),
    ("cs16_planar", "i16", 16, True, "planar", "lsb", "little", False, None),
    ("packed12", "packed12", 12, True, "interleaved", "lsb", "little", False, None),
    ("packed10", "packed10", 10, True, "interleaved", "lsb", "little", False, None),
    ("i2s32_msb", "i32", 24, True, "interleaved", "msb", "little", False, None),
    # CF32 (9) and RF32 (10) are handled below.
    ("rs16", "i16", 16, True, "real", "lsb", "little", False, None),
]

CONTAINER_BITS = {"u8": 8, "i8": 8, "i16": 16, "i32": 32}
PACKED_GROUP = {"packed12": (2, 3), "packed10": (4, 5)}


def effective_range(container, bits, signed):
    if not signed:
        return 0, 1 << bits
    return -(1 << (bits - 1)), 1 << (bits - 1)


def draw_values(rng, container, bits, signed, count):
    lo, hi = effective_range(container, bits, signed)
    vals = rng.integers(lo, hi, size=count, dtype=np.int64)
    vals[0] = lo
    vals[1] = hi - 1
    return vals


def arrange(vals, layout, swap):
    """Stream order of the values: returns (stream, i_part, q_part or None)."""
    if layout == "real":
        return vals, vals, None
    n = len(vals) // 2
    i_part, q_part = vals[:n], vals[n:]
    first, second = (q_part, i_part) if swap else (i_part, q_part)
    if layout == "planar":
        stream = np.concatenate([first, second])
    else:
        stream = np.empty(2 * n, dtype=np.int64)
        stream[0::2] = first
        stream[1::2] = second
    return stream, i_part, q_part


def to_words(stream, container, bits, signed, align):
    """Effective values -> unsigned container words (python ints in an int64 array)."""
    cbits = CONTAINER_BITS[container]
    if container == "u8":
        return stream << (cbits - bits) if align == "msb" else stream
    if align == "msb":
        stream = stream << (cbits - bits)
    return stream & ((1 << cbits) - 1)


def pack_bytes(stream, container, bits, signed, align, endian):
    if container in PACKED_GROUP:
        gv, gb = PACKED_GROUP[container]
        assert len(stream) % gv == 0
        out = bytearray()
        for g in range(len(stream) // gv):
            acc = 0
            for k in range(gv):
                acc |= (int(stream[g * gv + k]) & ((1 << bits) - 1)) << (k * bits)
            out += acc.to_bytes(gb, "little")
        return bytes(out)
    words = to_words(stream, container, bits, signed, align)
    nbytes = CONTAINER_BITS[container] // 8
    order = "big" if endian == "big" else "little"
    return b"".join(int(w).to_bytes(nbytes, order) for w in words)


def full_scale_of(bits, full_scale):
    return float(full_scale) if full_scale is not None else float(1 << (bits - 1))


def amplitude(e, container, bits, full_scale):
    off = 127.5 if container == "u8" else 0.0
    return (e.astype(np.float64) - off) / full_scale_of(bits, full_scale)


def to_q15(e, container, bits, full_scale):
    e = e.astype(np.int64)
    if container == "u8":
        d = 2 * e - 255
        if full_scale is None:
            return q15.sat16(d << (15 - bits))
        n = int(full_scale)
        return q15.sat16((d * 32768 + n) // (2 * n))
    if full_scale is not None:
        n = int(full_scale)
        return q15.sat16((e * 65536 + n) // (2 * n))
    if bits <= 16:
        return q15.sat16(e << (16 - bits))
    return q15.sat16(q15.round_shift(e, bits - 16))


def build_integer(rng, spec):
    name, container, bits, signed, layout, align, endian, swap, full_scale = spec
    count = N if layout == "real" else 2 * N
    vals = draw_values(rng, container, bits, signed, count)
    stream, i_part, q_part = arrange(vals, layout, swap)
    raw = pack_bytes(stream, container, bits, signed, align, endian)
    params = {
        "container": container,
        "bits": bits,
        "signed": signed,
        "layout": layout,
        "align": align,
        "endian": endian,
        "swap_iq": swap,
        "full_scale": full_scale,
        "samples": N,
    }
    if layout == "real":
        exp_f = amplitude(vals, container, bits, full_scale).astype(np.float32)
        exp_q = to_q15(vals, container, bits, full_scale)
    else:
        re = amplitude(i_part, container, bits, full_scale)
        im = amplitude(q_part, container, bits, full_scale)
        exp_f = (re + 1j * im).astype(np.complex64)
        exp_q = q15.pack(
            to_q15(i_part, container, bits, full_scale),
            to_q15(q_part, container, bits, full_scale),
        )
    return dict(
        params=params,
        arrays={
            "input": Array("u8", np.frombuffer(raw, dtype=np.uint8)),
            "expected_f32": Array("f32" if layout == "real" else "cf32", exp_f),
            "expected_q15": Array("i16" if layout == "real" else "ci16", exp_q),
        },
        tolerance={"class": "bit_exact"},
        notes="raw bytes decoded to amplitude (f32) and to Q15 by the exact integer rules",
    )


def _register_integer(spec, seed):
    case(kernel="convert", name=f"{spec[0]}_{N}", seed=seed)(lambda rng, spec=spec: build_integer(rng, spec))


for _pos, _spec in enumerate(INTEGER_FORMATS):
    # Positions 9 and 10 of the constant table are CF32 and RF32, so RS16 is position 11.
    _position = _pos if _pos < 9 else _pos + 2
    _register_integer(_spec, 100 + _position)

_register_integer(
    ("cu8_swapiq", "u8", 8, False, "interleaved", "lsb", "little", True, None),
    120,
)
_register_integer(
    ("cs16_bigendian", "i16", 16, True, "interleaved", "lsb", "big", False, None),
    121,
)


def build_float(rng, layout):
    count = N if layout == "real" else 2 * N
    vals = rng.standard_normal(count).astype(np.float32)
    specials = np.array(
        [0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -1.0, 3.0, -2.5, 1e-40, 0.99997, -1.00002],
        dtype=np.float32,
    )
    idx = rng.permutation(count)[: len(specials)]
    vals[idx] = specials
    f64 = vals.astype(np.float64)
    if layout == "real":
        stream = vals
        exp_f = vals
        exp_q = q15.from_float(f64)
        exp_f_dtype, exp_q_dtype = "f32", "i16"
    else:
        stream = vals  # interleaved: re, im, re, im
        exp_f = np.empty(count // 2, dtype=np.complex64)
        exp_f.real = vals[0::2]
        exp_f.imag = vals[1::2]
        exp_q = q15.pack(q15.from_float(f64[0::2]), q15.from_float(f64[1::2]))
        exp_f_dtype, exp_q_dtype = "cf32", "ci16"
    params = {
        "container": "f32",
        "bits": 32,
        "signed": True,
        "layout": layout,
        "align": "lsb",
        "endian": "little",
        "swap_iq": False,
        "full_scale": None,
        "samples": N,
    }
    return dict(
        params=params,
        arrays={
            "input": Array("u8", np.frombuffer(stream.astype("<f4").tobytes(), dtype=np.uint8)),
            "expected_f32": Array(exp_f_dtype, exp_f),
            "expected_q15": Array(exp_q_dtype, exp_q),
        },
        tolerance={"class": "bit_exact"},
        notes="f32 containers pass through unchanged (NaN, inf, -0.0); Q15 uses the f64 rounding rule",
    )


@case(kernel="convert", name="cf32_256", seed=109)
def convert_cf32(rng):
    return build_float(rng, "interleaved")


@case(kernel="convert", name="rf32_256", seed=110)
def convert_rf32(rng):
    return build_float(rng, "real")
