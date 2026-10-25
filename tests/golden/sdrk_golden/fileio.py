"""Writer and reader for the on-disk golden format (format_version 1)."""

import json
import os
import platform

import numpy as np

FORMAT_VERSION = 1

# dtype name -> (numpy little-endian dtype, components per element)
_NP = {
    "u8": ("<u1", 1),
    "i8": ("<i1", 1),
    "i16": ("<i2", 1),
    "i32": ("<i4", 1),
    "f32": ("<f4", 1),
    "f64": ("<f8", 1),
    "ci16": ("<i2", 2),
    "cf32": ("<f4", 2),
    "cf64": ("<f8", 2),
}


def encode_array(dtype, data):
    """Return (bytes, element_count) for ``data`` stored as ``dtype``."""
    if dtype not in _NP:
        raise ValueError(f"unknown dtype {dtype}")
    npdt, comps = _NP[dtype]
    arr = np.asarray(data)
    if dtype.startswith("cf"):
        arr = np.asarray(arr, dtype=np.complex128 if dtype == "cf64" else np.complex64)
        arr = np.stack([arr.real, arr.imag], axis=1)
    elif dtype == "ci16":
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("ci16 data must have shape (n, 2)")
    if arr.ndim != (2 if comps == 2 else 1):
        raise ValueError(f"bad shape {arr.shape} for {dtype}")
    if np.issubdtype(np.dtype(npdt), np.integer):
        info = np.iinfo(np.dtype(npdt))
        if arr.size and (arr.min() < info.min or arr.max() > info.max):
            raise ValueError(f"value out of range for {dtype}")
    out = np.ascontiguousarray(arr, dtype=np.dtype(npdt))
    return out.tobytes(), arr.shape[0]


def decode_array(dtype, raw):
    """Inverse of ``encode_array`` (used by tests of the generator itself)."""
    npdt, comps = _NP[dtype]
    arr = np.frombuffer(raw, dtype=np.dtype(npdt))
    if comps == 2:
        arr = arr.reshape(-1, 2)
        if dtype != "ci16":
            arr = arr[:, 0] + 1j * arr[:, 1]
    return arr


def provenance():
    import scipy

    return {
        "generator": "tests/golden/generate.py",
        "python": ".".join(platform.python_version_tuple()[:2]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def render_case(kernel, name, seed, result):
    """Render one case to a dict {filename: bytes}."""
    files = {}
    arrays = {}
    for aname, arr in result["arrays"].items():
        raw, n = encode_array(arr.dtype, arr.data)
        fname = f"{name}.{aname}.bin"
        files[fname] = raw
        arrays[aname] = {"file": fname, "dtype": arr.dtype, "len": n}
    doc = {
        "format_version": FORMAT_VERSION,
        "kernel": kernel,
        "case": name,
        "seed": seed,
        "params": result.get("params", {}),
        "arrays": arrays,
        "tolerance": result["tolerance"],
        "notes": result.get("notes", ""),
        "provenance": provenance(),
    }
    if "model" in result:
        doc["model"] = result["model"]
    files[f"{name}.json"] = (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return files


def write_files(directory, files):
    """Write ``{filename: bytes}`` into ``directory`` (created if needed)."""
    os.makedirs(directory, exist_ok=True)
    for fname, raw in files.items():
        mode = "wb"
        with open(os.path.join(directory, fname), mode) as f:
            f.write(raw)


def read_case(directory, name):
    """Read a case back: returns (json dict, {array name: numpy array})."""
    with open(os.path.join(directory, f"{name}.json"), "r", encoding="utf-8") as f:
        doc = json.load(f)
    arrays = {}
    for aname, meta in doc["arrays"].items():
        with open(os.path.join(directory, meta["file"]), "rb") as f:
            arrays[aname] = decode_array(meta["dtype"], f.read())
    return doc, arrays
