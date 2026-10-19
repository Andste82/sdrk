#!/usr/bin/env python3
"""Writes the example messages of docs/wire-format.md as binary files.

An independent implementation of the 32-byte header (Python's `struct`), so that the Rust
encoder is tested against bytes it did not produce. Run `python3 generate.py` in this directory;
`python3 generate.py --check` verifies the committed files.
"""
import struct
import sys
from pathlib import Path

SAMPLES, FRAME, EVENT, HELLO, PARAM_SET, PARAM_ACK = 1, 2, 3, 4, 5, 6
RAW, C32, CQ15, F32, Q15, JSON = 0, 1, 2, 3, 4, 5
OVERFLOW = 1 << 1


def message(kind, flags, stream, payload_type, start, count, averaged, payload):
    header = b"SDRK" + bytes([1, kind]) + struct.pack("<H", flags) + struct.pack("<H", stream)
    header += bytes([payload_type, 0]) + struct.pack("<Q", start)
    header += struct.pack("<I", count) + struct.pack("<I", len(payload)) + struct.pack("<I", averaged)
    assert len(header) == 32
    return header + payload


def text(s):
    return s.encode("utf-8")


FILES = {
    "samples.bin": message(
        SAMPLES, OVERFLOW, 3, C32, 1000, 2, 0, struct.pack("<4f", 1.0, -2.0, 0.5, 0.25)
    ),
    "frame.bin": message(
        FRAME, 0, 5, F32, 4096, 4, 8, struct.pack("<4f", 0.0, -10.5, -20.25, 3.0)
    ),
    "event.bin": (lambda p: message(EVENT, 0, 2, JSON, 12345, len(p), 0, p))(
        text('{"index":12345,"kind":{"type":"gap","missing":7}}')
    ),
    "hello.bin": (lambda p: message(HELLO, 0, 1, JSON, 0, len(p), 0, p))(
        text('{"stream":1,"name":"audio","type":"f32","rate_hz":"48000/1","anchor":null}')
    ),
    "param_set.bin": (lambda p: message(PARAM_SET, 0, 0, JSON, 0, len(p), 0, p))(
        text(
            '{"node":"tune","name":"frequency_hz","value":610000.0,'
            '"at":{"type":"input_index","input":"rf","index":100}}'
        )
    ),
    "param_ack.bin": (lambda p: message(PARAM_ACK, 0, 0, JSON, 100, len(p), 0, p))(
        text('{"node":"tune","name":"frequency_hz","index":100,"late":false}')
    ),
}


def main():
    here = Path(__file__).resolve().parent
    check = "--check" in sys.argv[1:]
    bad = False
    for name, data in FILES.items():
        path = here / name
        if check:
            if not path.exists() or path.read_bytes() != data:
                print(f"{name}: out of date")
                bad = True
        else:
            path.write_bytes(data)
            print(f"wrote {name} ({len(data)} bytes)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
