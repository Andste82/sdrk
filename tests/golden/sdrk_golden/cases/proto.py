"""Cases for ``sdrk-proto``: the four CRCs and the test framing (M37).

The expected CRCs come from the Python standard library where it has the CRC
(``binascii.crc_hqx`` with init 0xFFFF is CRC-16/CCITT-FALSE, ``binascii.crc32`` is
CRC-32/ISO-HDLC) and from bitwise models of the catalogue parameters otherwise (CRC-16/X-25,
the Mode S CRC-24). The models are checked against the catalogue check values before any case is
written. NumPy only draws the messages.

Layout of a CRC case: ``data`` holds the messages back to back (``u8``), ``lengths`` their
lengths and ``expected`` their CRCs (``i32``: the bit pattern of the unsigned CRC, so CRC-32
values above 2^31 appear negative). A frame case holds the payloads back to back, their
lengths, the preamble length of each frame and the concatenated frame bits (``u8``, 0 or 1).
"""

import binascii

import numpy as np

from ..registry import Array, case

SEED = 479
SYNC_WORD = 0x1ACFFC1D
MAX_PAYLOAD = 253
MODE_S_FRAME = bytes.fromhex("8D4840D6202CC371C32CE0576098")


def crc16_ccitt_false(d):
    return binascii.crc_hqx(d, 0xFFFF)


def crc16_x25(d):
    """Bitwise model: poly 0x1021 reflected (0x8408), init 0xFFFF, xorout 0xFFFF."""
    crc = 0xFFFF
    for b in d:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def crc32(d):
    return binascii.crc32(d) & 0xFFFFFFFF


def crc24_adsb(d):
    """Bitwise model: poly 0xFFF409, init 0, MSB first, no xorout."""
    crc = 0
    for b in d:
        crc ^= b << 16
        for _ in range(8):
            crc = ((crc << 1) ^ 0xFFF409) if crc & 0x800000 else crc << 1
            crc &= 0xFFFFFF
    return crc


def _self_test():
    check = b"123456789"
    assert crc16_ccitt_false(check) == 0x29B1
    assert crc16_x25(check) == 0x906E
    assert crc32(check) == 0xCBF43926
    assert crc24_adsb(MODE_S_FRAME) == 0
    assert crc24_adsb(MODE_S_FRAME[:11]) == 0x576098


_self_test()


def as_i32(values):
    """Unsigned 32-bit values as their ``int32`` bit patterns."""
    return np.array(values, dtype=np.uint32).view(np.int32)


def messages(rng, count, max_len):
    """``count`` random messages with lengths 0 ... ``max_len``, plus the fixed check inputs."""
    msgs = [b"", b"123456789", MODE_S_FRAME, bytes(max_len), b"\xff" * max_len]
    lengths = rng.integers(0, max_len + 1, size=count)
    for n in lengths.tolist():
        msgs.append(rng.integers(0, 256, size=n, dtype=np.uint8).tobytes())
    return msgs


def crc_case(func, notes):
    def run(rng):
        msgs = messages(rng, 64, 256)
        data = np.frombuffer(b"".join(msgs), dtype=np.uint8)
        return dict(
            params={"messages": len(msgs)},
            arrays={
                "data": Array("u8", data),
                "lengths": Array("i32", np.array([len(m) for m in msgs], dtype=np.int32)),
                "expected": Array("i32", as_i32([func(m) for m in msgs])),
            },
            tolerance={"class": "bit_exact"},
            notes=notes,
        )

    return run


case("proto", "crc16_ccitt_false", SEED)(
    crc_case(crc16_ccitt_false, "binascii.crc_hqx(d, 0xFFFF); 69 messages of 0 to 256 bytes")
)
case("proto", "crc16_x25", SEED)(
    crc_case(crc16_x25, "bitwise model of CRC-16/X-25; 69 messages of 0 to 256 bytes")
)
case("proto", "crc32", SEED)(
    crc_case(crc32, "binascii.crc32; 69 messages of 0 to 256 bytes (i32 bit patterns)")
)
case("proto", "crc24_adsb", SEED)(
    crc_case(crc24_adsb, "bitwise model of the Mode S CRC-24; 69 messages of 0 to 256 bytes")
)


def frame_bits(payload, preamble_bytes):
    """The framing of ``sdrk_proto::frame``: bytes of the frame, MSB first, as 0/1 values."""
    assert len(payload) <= MAX_PAYLOAD
    body = bytes([len(payload)]) + payload
    crc = crc16_ccitt_false(body)
    frame = (
        bytes([0x55] * preamble_bytes)
        + SYNC_WORD.to_bytes(4, "big")
        + body
        + crc.to_bytes(2, "big")
    )
    return np.unpackbits(np.frombuffer(frame, dtype=np.uint8))


@case("proto", "frame_encode", SEED)
def frame_encode(rng):
    lengths = [0, 1, MAX_PAYLOAD] + rng.integers(0, MAX_PAYLOAD + 1, size=13).tolist()
    preambles = [4, 0, 4] + rng.integers(0, 33, size=13).tolist()
    payloads = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes() for n in lengths]
    bits = np.concatenate([frame_bits(p, k) for p, k in zip(payloads, preambles)])
    return dict(
        params={"frames": len(payloads)},
        arrays={
            "payload": Array("u8", np.frombuffer(b"".join(payloads), dtype=np.uint8)),
            "lengths": Array("i32", np.array(lengths, dtype=np.int32)),
            "preamble_bytes": Array("i32", np.array(preambles, dtype=np.int32)),
            "bits": Array("u8", bits),
        },
        tolerance={"class": "bit_exact"},
        notes="16 frames (payloads 0, 1, 253 and 13 random lengths; preambles 4, 0, 4 and 13 "
        "random of 0 to 32 bytes); CRC-16/CCITT-FALSE from binascii.crc_hqx; bits MSB first",
    )
