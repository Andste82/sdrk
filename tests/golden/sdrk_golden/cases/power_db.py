"""Cases for the power_db kernel."""

import numpy as np

from ..registry import Array, case

OFFSET_DB = 1.5


@case(kernel="power_db", name="f32_4096", seed=62)
def f32_4096(rng):
    p = (10.0 ** rng.uniform(-25.0, 1.0, 4096)).astype(np.float32)
    # Special entries: zero, negative zero, negative values, NaN, the floor and below it.
    specials = [0.0, -0.0, -1.0, np.nan, -1e-25, 1e-20, 1e-30, 1e-40, 1.0, 2.0]
    p[: len(specials)] = np.asarray(specials, dtype=np.float32)
    p64 = p.astype(np.float64)
    floor_applied = np.where(np.isnan(p64) | (p64 < 1e-20), 1e-20, p64)
    expected = 10.0 * np.log10(floor_applied) + OFFSET_DB
    return dict(
        params={"offset_db": OFFSET_DB, "special_entries": len(specials)},
        arrays={"input": Array("f32", p), "expected": Array("f64", expected)},
        tolerance={"class": "max_abs_err", "max_abs_err": 1e-4, "skip": 0},
        notes="10 log10(max(p, 1e-20)) + 1.5; the first ten entries are 0, -0, -1, NaN, -1e-25, 1e-20, 1e-30, 1e-40, 1, 2",
    )
