"""Cases for the window generators and the window kernel."""

import numpy as np
import scipy.signal

from .. import q15
from ..registry import Array, case
from .fir import noise


def generator_case(spec, n, fftbins, params):
    w = scipy.signal.get_window(spec, n, fftbins=fftbins)
    return dict(
        params=dict(params, n=n, symmetry="periodic" if fftbins else "symmetric"),
        arrays={"expected": Array("f64", w)},
        tolerance={"class": "max_abs_err", "max_abs_err": 2e-7, "skip": 0},
        notes="scipy.signal.get_window in float64; the Rust generator writes f32",
    )


@case(kernel="window", name="hann_periodic_1024", seed=54)
def hann_periodic(rng):
    return generator_case("hann", 1024, True, {"window": "hann"})


@case(kernel="window", name="hann_symmetric_1023", seed=55)
def hann_symmetric(rng):
    return generator_case("hann", 1023, False, {"window": "hann"})


@case(kernel="window", name="blackmanharris_periodic_4096", seed=56)
def blackmanharris_periodic(rng):
    return generator_case("blackmanharris", 4096, True, {"window": "blackman_harris"})


@case(kernel="window", name="kaiser8_symmetric_255", seed=57)
def kaiser8_symmetric(rng):
    return generator_case(("kaiser", 8.0), 255, False, {"window": "kaiser", "beta": 8.0})


@case(kernel="window", name="apply_hann_c32_1024", seed=58)
def apply_hann_c32(rng):
    w = scipy.signal.get_window("hann", 1024, fftbins=True).astype(np.float32)
    x = noise(rng, 1024, 0.3).astype(np.complex64)
    w64 = w.astype(np.float64)
    expected = np.empty(len(x), dtype=np.complex64)
    expected.real = (x.real.astype(np.float64) * w64).astype(np.float32)
    expected.imag = (x.imag.astype(np.float64) * w64).astype(np.float32)
    return dict(
        params={"window": "hann"},
        arrays={
            "input": Array("cf32", x),
            "window": Array("f32", w),
            "expected": Array("cf32", expected),
        },
        tolerance={"class": "bit_exact"},
        notes="one f32 multiply per component (the f64 product of two f32 values is exact, rounded once)",
    )


@case(kernel="window", name="apply_hann_cq15_1024", seed=59)
def apply_hann_cq15(rng):
    w = q15.from_float(scipy.signal.get_window("hann", 1024, fftbins=True))
    x = noise(rng, 1024, 0.3)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    return dict(
        params={"window": "hann"},
        arrays={
            "input": Array("ci16", q15.pack(xr, xi)),
            "window": Array("i16", w),
            "expected": Array("ci16", q15.pack(q15.mul(xr, w), q15.mul(xi, w))),
        },
        tolerance={"class": "bit_exact"},
        notes="Q15 multiply per component",
    )
