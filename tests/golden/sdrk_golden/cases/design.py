"""Cases for the filter design of sdrk-blocks (``scipy.signal.firwin``, ``kaiserord``)."""

import numpy as np
import scipy.signal

from ..registry import Array, case

WINDOWS = ("hann", "hamming", "blackman")


def firwin_case(window_name, window, num_taps, cutoff, notes):
    taps = scipy.signal.firwin(num_taps, cutoff, window=window, fs=1.0)
    params = {"cutoff_hz": cutoff, "fs_hz": 1.0, "num_taps": num_taps, "window": window_name}
    if isinstance(window, tuple):
        params["beta"] = window[1]
    return dict(
        params=params,
        arrays={"expected": Array("f64", taps)},
        tolerance={"class": "max_abs_err", "max_abs_err": 1e-12, "skip": 0},
        notes=notes,
    )


def _register_firwin(window_name, num_taps, cutoff):
    @case(kernel="design", name=f"firwin_{window_name}_{num_taps}", seed=0)
    def _case(rng):
        return firwin_case(
            window_name,
            window_name,
            num_taps,
            cutoff,
            "scipy.signal.firwin(N, cutoff, window, fs=1.0), symmetric window, DC gain 1",
        )


for _w in WINDOWS:
    _register_firwin(_w, 31, 0.1)
    _register_firwin(_w, 128, 0.23)


def _register_kaiser(beta, tag):
    @case(kernel="design", name=f"firwin_kaiser_{tag}", seed=0)
    def _case(rng):
        return firwin_case(
            "kaiser",
            ("kaiser", beta),
            101,
            0.15,
            "scipy.signal.firwin(101, 0.15, window=('kaiser', beta), fs=1.0)",
        )


for _beta, _tag in ((0.5, "0p5"), (5.0, "5p0"), (8.6, "8p6")):
    _register_kaiser(_beta, _tag)


@case(kernel="design", name="kaiserord_table", seed=0)
def kaiserord_table(rng):
    ripple = np.linspace(20.0, 100.0, 20)
    width = np.geomspace(0.01, 0.2, 20)
    taps = []
    beta = []
    for a, w in zip(ripple, width):
        n, b = scipy.signal.kaiserord(a, w)
        n = int(n)
        taps.append(n + 1 if n % 2 == 0 else n)
        beta.append(b)
    return dict(
        params={"pairs": 20, "fs_hz": 2.0},
        arrays={
            "ripple_db": Array("f64", ripple),
            "width": Array("f64", width),
            "num_taps": Array("i32", np.array(taps, dtype=np.int32)),
            "expected": Array("f64", np.array(beta)),
        },
        tolerance={"class": "max_abs_err", "max_abs_err": 1e-12, "skip": 0},
        notes=(
            "scipy.signal.kaiserord(ripple, width); num_taps (forced odd) must match exactly, "
            "expected is beta; width = 2 transition / fs, so fs = 2 makes transition_hz equal to width"
        ),
    )
