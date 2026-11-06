"""Cases for the ``rds_shape`` kernel (M42 section 3: the shaped biphase RDS baseband).

The pulse table and the conventions are those of the RDS reference modulator of M41
(``cases/rds.py``, decision D664): ``h(x) = 8 cos(4 pi x) / (pi (1 - 64 x^2))`` with
``x = t / Tb`` and the limit ``h(+-1/8) = 2``, the pulse ``p(x) = h(x + 1/4) - h(x - 1/4)``
(first chip ``+a_k``), bit ``k`` centred at ``x = k`` from the first bit at ``x = 0``.

The kernel samples the pulse on a grid of 256 points per bit over ``-2 <= x <= 2``: 1025 entries,
the last one ``p(2)`` so that the last interval can be interpolated, divided by the largest
magnitude on that grid and rounded once to float32. The f64 table is stored in the case; the
expected output is the exact float32 model (``f32model``) of the kernel on the float32 table:

- position ``x = (n 1187.5) / fs`` in float64 (one multiplication, one division),
  ``k0 = floor(x) - 1``; bits ``k < 0`` are 0, bit ``k >= 0`` is ``bits[k]``;
- for ``j = 0 .. 3``: ``u = ((x - (k0 + j)) + 2) 256`` in float64, ``i = min(floor(u), 1023)``,
  ``fr = float32(u - i)``, ``v = t[i] + (t[i + 1] - t[i]) fr`` in float32;
- ``acc = 0``, then ``acc = acc + a_j v`` for ``j`` ascending, in float32.

``closed_form_max_abs_err`` is the largest difference of the model output from the exact
baseband of ``cases/rds.py`` (its pulse at the exact grid of 192 kHz, 3072 points per bit) for the
same bits: the error of the 256-point table with linear interpolation (D771).
"""

import math

import numpy as np

from .. import f32model as fm
from ..registry import Array, case
from . import rds

#: Grid points per bit of the kernel table.
PER_BIT = 256
#: Entries of the kernel table: 4 bits of 256 points plus the end point.
TABLE_LEN = 4 * PER_BIT + 1
RATE_HZ = 192000
N = 16384


def pulse_table_f64():
    """The normalized kernel table in float64 and the grid peak before normalization."""
    q = PER_BIT // 4
    raw = [
        rds.h(n + q, PER_BIT) - rds.h(n - q, PER_BIT)
        for n in range(-2 * PER_BIT, 2 * PER_BIT + 1)
    ]
    peak = max(abs(v) for v in raw)
    return np.array([v / peak for v in raw], dtype=np.float64), peak


def shape_f32(table32, bits, n0, count, fs):
    """Float32 model of ``rds_shape`` for the samples ``n0 .. n0 + count`` (bits ``k < 0`` are 0)."""
    t = [fm.f32(v) for v in table32]
    zero = fm.f32(0.0)
    out = np.zeros(count, dtype=np.float32)
    for m in range(count):
        n = n0 + m
        x = (float(n) * 1187.5) / fs
        k0 = int(math.floor(x)) - 1
        acc = zero
        for j in range(4):
            k = k0 + j
            a = fm.f32(int(bits[k])) if k >= 0 else zero
            u = ((x - float(k)) + 2.0) * 256.0
            i = min(int(math.floor(u)), TABLE_LEN - 2)
            fr = fm.f32(u - float(i))
            v = fm.add(t[i], fm.mul(fm.sub(t[i + 1], t[i]), fr))
            acc = fm.add(acc, fm.mul(a, v))
        out[m] = acc
    return out


def bits_needed(count, fs):
    """Bits the samples ``0 .. count`` use (``k0 + 3`` of the last sample, plus one)."""
    return int(math.floor(((count - 1) * 1187.5) / fs)) + 3


@case(kernel="rds_shape", name="table_and_bits_f32_16384", seed=580)
def table_and_bits(rng):
    table, peak = pulse_table_f64()
    table32 = table.astype(np.float32)
    nb = bits_needed(N, float(RATE_HZ))
    bits = (2 * rng.integers(0, 2, size=nb) - 1).astype(np.int8)
    out = shape_f32(table32, bits, 0, N, float(RATE_HZ))
    # The exact baseband of the reference modulator for the same bits.
    exact = rds.rds_baseband(bits.astype(np.float64), 0, N, RATE_HZ)
    err = float(np.max(np.abs(out.astype(np.float64) - exact)))
    assert err < 2e-4, err
    assert float(np.max(np.abs(table))) == 1.0
    return dict(
        params={
            "rate_hz": RATE_HZ,
            "bit_rate_hz": 1187.5,
            "per_bit": PER_BIT,
            "table_len": TABLE_LEN,
            "grid_peak": peak,
            "bits": nb,
            "closed_form_max_abs_err": err,
        },
        arrays={
            "table": Array("f64", table),
            "bits": Array("i8", bits),
            "expected": Array("f32", out),
        },
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="192 kHz; random amplitudes +-1 (the bits after the differential coding) for every "
        "bit the 16384 samples use; table: the pulse p(x) = h(x + 1/4) - h(x - 1/4) of cases/rds.py "
        "at x = -2 + i / 256 for i = 0..1024, divided by its largest magnitude on that grid "
        "(grid_peak); expected: the exact float32 model of the kernel on the table rounded to "
        "float32 (module docstring); closed_form_max_abs_err: the largest difference of expected "
        "from the exact baseband of cases/rds.py for the same bits (D771)",
    )
