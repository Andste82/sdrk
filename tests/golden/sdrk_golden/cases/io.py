"""Cases for the file I/O of ``sdrk-io``: SigMF metadata, WAV files and ISO 8601 timestamps.

Everything here uses only the Python standard library (``json``, ``wave``, ``struct``,
``datetime``) for the expected bytes; NumPy only draws the sample values. The Rust writers must
produce identical bytes and the Rust readers must parse them.
"""

import datetime
import io
import json
import struct
import wave

import numpy as np

from ..registry import Array, case

EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
NS = 10**9


# ---------------------------------------------------------------------------------------- SigMF


def iso8601(unix_ns):
    """``YYYY-MM-DDTHH:MM:SS[.fraction]Z``, the fraction without trailing zeros, omitted if zero."""
    secs, frac = divmod(unix_ns, NS)
    dt = EPOCH + datetime.timedelta(seconds=secs)
    text = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if frac:
        text += "." + f"{frac:09d}".rstrip("0")
    return text + "Z"


def sigmf_meta(params):
    """The expected ``.sigmf-meta`` bytes for the steps and annotations of ``params``.

    A new capture starts at the first write and whenever a write has a gap (its start differs
    from the end of the previous write), an anchor, the ``retuned`` or ``gap_before`` flag, or a
    frequency that differs from the frequency of the open capture.
    """
    num, den = params["rate_num"], params["rate_den"]
    captures = []
    position = 0
    end = None
    frequency = None
    open_frequency = None
    for step in params["steps"]:
        if step["op"] == "frequency":
            frequency = step["hz"]
            continue
        start, count = step["start"], step["samples"]
        flags = step.get("flags", [])
        anchor = step.get("anchor")
        new = (
            not captures
            or end != start
            or "gap_before" in flags
            or "retuned" in flags
            or anchor is not None
            or frequency != open_frequency
        )
        if new:
            cap = {"core:sample_start": position}
            if start != position:
                cap["core:global_index"] = start
            if anchor is not None:
                delta = start - anchor["index"]
                cap["core:datetime"] = iso8601(anchor["unix_ns"] + (delta * NS * den) // num)
            if frequency is not None:
                cap["core:frequency"] = frequency
            captures.append(cap)
            open_frequency = frequency
        position += count
        end = start + count
    annotations = []
    for a in params["annotations"]:
        item = {"core:sample_start": a["sample_start"], "core:sample_count": a["sample_count"]}
        for key in ("freq_lower_edge", "freq_upper_edge", "label", "comment"):
            if key in a:
                item["core:" + key] = a[key]
        annotations.append(item)
    rate = num // den if num % den == 0 else num / den
    glob = {
        "core:datatype": params["datatype"],
        "core:sample_rate": rate,
        "core:version": "1.2.0",
    }
    if "description" in params:
        glob["core:description"] = params["description"]
    doc = {"annotations": annotations, "captures": captures, "global": glob}
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


@case(kernel="io", name="sigmf_cu8_meta", seed=0)
def sigmf_cu8_meta(rng):
    params = {
        "datatype": "cu8",
        "rate_num": 2400000,
        "rate_den": 1,
        "description": "gap, retune and anchor test",
        "steps": [
            {"op": "frequency", "hz": 100000000.0},
            {
                "op": "write",
                "start": 0,
                "samples": 1000,
                "anchor": {"index": 0, "unix_ns": 1700000000123456789},
            },
            {"op": "write", "start": 1000, "samples": 500},
            {"op": "write", "start": 1700, "samples": 300},
            {"op": "frequency", "hz": 101000000.0},
            {"op": "write", "start": 2000, "samples": 100},
            {"op": "write", "start": 2100, "samples": 100, "flags": ["retuned"]},
            {
                "op": "write",
                "start": 2200,
                "samples": 50,
                "anchor": {"index": 2200, "unix_ns": 1700000001000000000},
            },
        ],
        "annotations": [
            {
                "sample_start": 0,
                "sample_count": 100,
                "freq_lower_edge": -50000.0,
                "freq_upper_edge": 50000.0,
                "label": "carrier",
            },
            {"sample_start": 1500, "sample_count": 300, "comment": "burst after the gap"},
            {"sample_start": 1900, "sample_count": 10},
        ],
    }
    return dict(
        params=params,
        arrays={"expected": Array("u8", np.frombuffer(sigmf_meta(params), dtype=np.uint8))},
        tolerance={"class": "bit_exact"},
        notes="expected .sigmf-meta bytes from Python json (sorted keys, indent 2, LF)",
    )


@case(kernel="io", name="sigmf_cf32_fractional_meta", seed=0)
def sigmf_cf32_fractional_meta(rng):
    params = {
        "datatype": "cf32_le",
        "rate_num": 88201,
        "rate_den": 2,
        "steps": [
            {
                "op": "write",
                "start": 500,
                "samples": 64,
                "anchor": {"index": 500, "unix_ns": 951825600000000000},
            },
            {"op": "write", "start": 564, "samples": 64},
        ],
        "annotations": [],
    }
    return dict(
        params=params,
        arrays={"expected": Array("u8", np.frombuffer(sigmf_meta(params), dtype=np.uint8))},
        tolerance={"class": "bit_exact"},
        notes="fractional sample rate 44100.5, start index 500, no annotations",
    )


# ------------------------------------------------------------------------------------------ WAV

GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def wav_bytes(tag, channels, rate, bits, payload, extensible=False, list_chunk=False):
    """A WAV file built with ``struct``."""
    block_align = channels * bits // 8
    if extensible:
        fmt = struct.pack(
            "<HHIIHHHHI", 0xFFFE, channels, rate, rate * block_align, block_align, bits, 22, bits, 3
        )
        fmt += struct.pack("<H", tag) + GUID_TAIL
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block_align, block_align, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if list_chunk:
        info = b"INFO" + b"ISFT" + struct.pack("<I", 5) + b"sdrk\x00" + b"\x00"
        body += b"LIST" + struct.pack("<I", 4 + 8 + 5) + info
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


@case(kernel="io", name="wav_pcm16_mono", seed=0)
def wav_pcm16_mono(rng):
    samples = rng.integers(-32768, 32768, 16, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(samples.astype("<i2").tobytes())
    return dict(
        params={"format": "rs16", "channels": 1, "rate": 8000, "iq": False},
        arrays={
            "samples": Array("i16", samples),
            "expected": Array("u8", np.frombuffer(buf.getvalue(), dtype=np.uint8)),
        },
        tolerance={"class": "bit_exact"},
        notes="header and payload written with the Python wave module",
    )


@case(kernel="io", name="wav_float32_stereo", seed=0)
def wav_float32_stereo(rng):
    iq = (rng.uniform(-1, 1, 16) + 1j * rng.uniform(-1, 1, 16)).astype(np.complex64)
    payload = np.stack([iq.real, iq.imag], axis=1).astype("<f4").tobytes()
    raw = wav_bytes(3, 2, 48000, 32, payload)
    return dict(
        params={"format": "cf32", "channels": 2, "rate": 48000, "iq": True},
        arrays={
            "samples": Array("cf32", iq),
            "expected": Array("u8", np.frombuffer(raw, dtype=np.uint8)),
        },
        tolerance={"class": "bit_exact"},
        notes="canonical 44-byte header built with struct (IEEE float, 16-byte fmt chunk)",
    )


@case(kernel="io", name="wav_extensible_cs16_list", seed=0)
def wav_extensible_cs16_list(rng):
    iq = rng.integers(-32768, 32768, (16, 2), dtype=np.int16)
    raw = wav_bytes(1, 2, 96000, 16, iq.astype("<i2").tobytes(), extensible=True, list_chunk=True)
    return dict(
        params={"format": "cs16", "channels": 2, "rate": 96000, "iq": True},
        arrays={
            "samples": Array("ci16", iq),
            "file": Array("u8", np.frombuffer(raw, dtype=np.uint8)),
        },
        tolerance={"class": "bit_exact"},
        notes="WAVE_FORMAT_EXTENSIBLE with a LIST chunk (odd size, pad byte) before data; read only",
    )


@case(kernel="io", name="wav_extensible_rf32_mono", seed=0)
def wav_extensible_rf32_mono(rng):
    x = rng.uniform(-1, 1, 16).astype("<f4")
    raw = wav_bytes(3, 1, 44100, 32, x.tobytes(), extensible=True)
    return dict(
        params={"format": "rf32", "channels": 1, "rate": 44100, "iq": False},
        arrays={
            "samples": Array("f32", x),
            "file": Array("u8", np.frombuffer(raw, dtype=np.uint8)),
        },
        tolerance={"class": "bit_exact"},
        notes="WAVE_FORMAT_EXTENSIBLE float mono; read only",
    )


# ---------------------------------------------------------------------------------------- ISO 8601

TIMESTAMPS = [
    "1970-01-01T00:00:00Z",
    "1970-01-01T00:00:00.000000001Z",
    "1970-01-01T00:00:01Z",
    "1969-12-31T23:59:59Z",
    "1969-12-31T23:59:59.5Z",
    "1999-12-31T23:59:59Z",
    "2000-01-01T00:00:00Z",
    "2000-02-29T12:00:00.5Z",
    "2000-03-01T00:00:00Z",
    "1900-02-28T23:59:59Z",
    "1900-03-01T00:00:00Z",
    "2004-02-29T00:00:00Z",
    "2023-11-14T22:13:20Z",
    "2023-11-14T22:13:20.1Z",
    "2023-11-14T22:13:20.12Z",
    "2023-11-14T22:13:20.123Z",
    "2023-11-14T22:13:20.1234Z",
    "2023-11-14T22:13:20.12345Z",
    "2023-11-14T22:13:20.123456Z",
    "2023-11-14T22:13:20.1234567Z",
    "2023-11-14T22:13:20.12345678Z",
    "2023-11-14T22:13:20.123456789Z",
    "2021-06-18T23:17:51.163959Z",
    "2024-12-31T23:59:59.999999999Z",
    "2025-01-01T00:00:00Z",
    "2038-01-19T03:14:07Z",
    "2038-01-19T03:14:08Z",
    "2100-02-28T23:59:59Z",
    "2100-03-01T00:00:00Z",
    "2262-04-11T23:47:16.854775807Z",
]

INVALID_TIMESTAMPS = [
    "",
    "2023-11-14",
    "2023-11-14T22:13:20",
    "2023-11-14 22:13:20Z",
    "2023-11-14T22:13:20+01:00",
    "2023-13-01T00:00:00Z",
    "2023-00-10T00:00:00Z",
    "2023-02-29T00:00:00Z",
    "2100-02-29T00:00:00Z",
    "2023-04-31T00:00:00Z",
    "2023-01-01T24:00:00Z",
    "2023-01-01T00:60:00Z",
    "2023-01-01T00:00:60Z",
    "2023-01-01T00:00:00.Z",
    "2023-01-01T00:00:00.5",
    "2023-01-01T00:00:00Zjunk",
    "2023-1-1T00:00:00Z",
    "0000-01-01T00:00:00Z",
    "2263-01-01T00:00:00Z",
    "year-mo-daTHH:MM:SSZ",
]


def unix_ns_of(text):
    """Nanoseconds since the epoch, computed with ``datetime``."""
    main, _, frac = text[:-1].partition(".")
    dt = datetime.datetime.strptime(main, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    secs = (dt - EPOCH) // datetime.timedelta(seconds=1)
    nanos = int((frac + "000000000")[:9]) if frac else 0
    return secs * NS + nanos


@case(kernel="io", name="iso8601_table", seed=0)
def iso8601_table(rng):
    entries = [{"text": t, "unix_ns": unix_ns_of(t)} for t in TIMESTAMPS]
    assert len(entries) == 30
    return dict(
        params={"cases": entries, "invalid": INVALID_TIMESTAMPS},
        arrays={},
        tolerance={"class": "bit_exact"},
        notes="expected nanoseconds from Python datetime; the invalid strings must be rejected",
    )
