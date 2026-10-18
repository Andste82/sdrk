"""Case registry: the ``@case`` decorator and the global ``CASES`` list."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

MAX_CASE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024

DTYPES = ("u8", "i8", "i16", "i32", "f32", "f64", "ci16", "cf32", "cf64")


@dataclass
class Array:
    """An array with the on-disk dtype name; ``data`` is converted by the writer."""

    dtype: str
    data: object


@dataclass
class Case:
    kernel: str
    name: str
    seed: int
    func: Callable

    def run(self):
        rng = np.random.default_rng(self.seed)
        return self.func(rng)


CASES: list[Case] = []


def case(kernel, name, seed):
    """Register a case function ``f(rng) -> dict(params, arrays, tolerance, notes)``."""

    def deco(func):
        if any(c.kernel == kernel and c.name == name for c in CASES):
            raise ValueError(f"duplicate case {kernel}/{name}")
        CASES.append(Case(kernel, name, seed, func))
        return func

    return deco
