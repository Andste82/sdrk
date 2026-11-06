"""The cross-check of the RDS encoder of M42 (test 4) against the reference modulator of M41.

Eight random groups (``default_rng(585)``; block B has bit 11, the version B flag, cleared in the
even and set in the odd groups, so both the offset C and the offset C' occur) are turned into
the RDS baseband with the functions of ``cases/rds.py`` (M41 section 5): each information word
followed by its checkword with the offsets A, B, C (C' for version B) and D, 26 bits per block,
most significant bit first; the differential coding ``t_k = t_{k-1} xor d_k`` from
``t_{-1} = 0``; the amplitudes ``a_k = 2 t_k - 1``; and the exact baseband ``rds_baseband`` at
192 kHz (the closed-form pulse on the exact grid of the rate, 3072 points per bit), bit 0
centred at sample 0. The block ``rds_encode`` gets the same groups as raw group packets
(``0x03`` and the four words big-endian) before its first sample and must give the same
baseband within ``max_abs_err``.

``model_max_abs_err`` is the distance of the float32 model of the kernel ``rds_shape``
(``cases/rds_shape.py``, the 256-point table with linear interpolation) from the stored
baseband: the bound that the block can reach. The specification asks for 1e-5
(``spec_max_abs_err``); that bound needs a finer table (open item I42-1), so the tolerance is the
measured value rounded up to three significant digits.
"""

import math

import numpy as np

from ..registry import Array, case
from . import rds, rds_shape

RATE_HZ = 192000
GROUPS = 8
WINDOW = 4096
LATE_START = 100000
SPEC_MAX_ABS_ERR = 1e-5


def group_bits(words):
    """The 104 data bits of one group (``uint8``, MSB first), C' when block B marks version B."""
    third = "Cp" if words[1] & 0x0800 else "C"
    out = []
    for w, o in zip(words, ["A", "B", third, "D"]):
        block = (int(w) << 10) | rds.checkword(int(w), o)
        out.extend((block >> (25 - i)) & 1 for i in range(26))
    return out


def round_up(v, digits=3):
    """``v > 0`` rounded up to ``digits`` significant digits."""
    e = math.floor(math.log10(v)) - (digits - 1)
    return float(f"{math.ceil(v / 10.0**e)}e{e}")


@case(kernel="rds_encode", name="cross_check_585", seed=585)
def cross_check(rng):
    words = rng.integers(0, 1 << 16, size=(GROUPS, 4), dtype=np.int64)
    for g in range(GROUPS):
        if g % 2 == 0:
            words[g, 1] &= ~0x0800
        else:
            words[g, 1] |= 0x0800
    d = np.array([b for g in range(GROUPS) for b in group_bits(words[g])], dtype=np.uint8)
    a = rds.amplitudes(rds.differential(d))
    table = rds.pulse_table(RATE_HZ)
    early = rds.rds_baseband(a, 0, WINDOW, RATE_HZ, table)
    late = rds.rds_baseband(a, LATE_START, WINDOW, RATE_HZ, table)
    early32, late32 = early.astype(np.float32), late.astype(np.float32)
    # The float32 model of the kernel on the same amplitudes.
    t64, _ = rds_shape.pulse_table_f64()
    t32 = t64.astype(np.float32)
    a8 = a.astype(np.int8)
    m_early = rds_shape.shape_f32(t32, a8, 0, WINDOW, float(RATE_HZ))
    m_late = rds_shape.shape_f32(t32, a8, LATE_START, WINDOW, float(RATE_HZ))
    err = max(
        float(np.max(np.abs(m_early.astype(np.float64) - early32.astype(np.float64)))),
        float(np.max(np.abs(m_late.astype(np.float64) - late32.astype(np.float64)))),
    )
    assert err < 2e-4, err
    return dict(
        params={
            "rate_hz": RATE_HZ,
            "groups": GROUPS,
            "late_start": LATE_START,
            "pulse_peak": table[3],
            "grid_den": table[1],
            "model_max_abs_err": err,
            "spec_max_abs_err": SPEC_MAX_ABS_ERR,
        },
        arrays={
            "words": Array("i32", words.reshape(-1)),
            "data_bits": Array("u8", d),
            "baseband": Array("f32", early),
            "baseband_late": Array("f32", late),
        },
        tolerance={"class": "max_abs_err", "max_abs_err": round_up(err), "skip": 0},
        notes="test 4 of M42: words: 8 random groups (A, B, C, D per group; version B in the odd "
        "groups); data_bits: their 832 bits with the offsets A, B, C or C', D (bit-exact); "
        "baseband: samples 0..4095 and baseband_late: samples late_start..late_start+4095 of the "
        "exact RDS baseband of cases/rds.py at 192 kHz for these bits (float64 rounded once to "
        "float32); model_max_abs_err: the distance of the float32 model of rds_shape from them, "
        "the tolerance rounds it up (the 1e-5 of the specification, spec_max_abs_err, is open "
        "item I42-1)",
    )
