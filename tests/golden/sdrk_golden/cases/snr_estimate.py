"""Block case of ``snr_estimate`` (M44.4 §1) against an f64 model of the estimator.

The model is written from the specification of the block, independently of the Rust code:

* the reading table: for ``b`` bits per axis, the integer grid of the odd levels ``v`` (``m = 2^b``
  levels per axis, power ``E = 2 (m^2 - 1) / 3`` per symbol), Gaussian noise of variance
  ``E / (2 rho)`` per axis and the decision regions between the even integers, the exact means
  ``E[x d]`` and ``E[d^2]`` (from the normal distribution function) give the reading
  ``10 log10(g^2 E[d^2] / N)`` with ``g = E[x d] / E[d^2]`` and ``N = E[(x - d)^2] - (E[x d] -
  E[d^2])^2 / E[d^2]``, at Es/N0 from -10 to 50 dB in steps of 0.5 dB (computed here with SciPy);
* per window of ``window`` symbols, at the amplitude ``a`` of the previous window: the level
  coordinate ``t = v sqrt(E) / (2 a) + m / 2`` per axis, the level ``i = floor(clamp(t, 0, m -
  1))``, the decision ``d = 2 i - (m - 1)`` and the error ``e = 2 (t - i) - 1``; the reading
  ``10 log10(g^2 D / N)`` with ``g = 1 + C / D`` and ``N = sum e^2 - C^2 / D`` (``C = sum e d``,
  ``D = sum d^2`` over both axes); the estimate is the Es/N0 at which the table reads that
  value, linear between the nodes, the floor -10 dB below the first node and the reading less
  the offset of the last node above it;
* the amplitude of the next window: ``a g`` when the estimate is at least 10.5, 16.5, 21.5, 26.5 or
  30.5 dB (``qpsk`` to ``qam1024``), otherwise ``sqrt(S)`` with the M2M4 signal power ``S =
  sqrt((2 M2^2 - M4) / (2 - ka))`` (``ka = E|s|^4 / (E|s|^2)^2`` of the grid) held between ``M2 /
  11`` and ``M2`` (``M2 / 11`` when the square is not positive); the first window only gives
  that amplitude.

Everything is computed in float64 from the float32 input. The block decides in float32: a symbol
within about 1e-6 of a decision boundary can take the other level there, which moves the
estimate of a window of 512 symbols by up to about 0.005 dB (one such symbol in window 17 of
this case: 0.0038 dB), so the tolerance is 0.01 dB.
"""

import numpy as np
from scipy.special import erfc

from ..registry import Array, case

FLOOR_DB = -10.0
STEP_DB = 0.5
NODES = 121
FIT_FROM_DB = {1: 10.5, 2: 16.5, 3: 21.5, 4: 26.5, 5: 30.5}


def normal_cdf(x):
    return 0.5 * erfc(-x / np.sqrt(2.0))


def normal_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def reading_db(b, snr_db):
    """The reading of the decision estimate over white Gaussian noise at an Es/N0 of ``snr_db``."""
    m = 1 << b
    v = 2.0 * np.arange(m) - (m - 1)
    energy = 2.0 * (m * m - 1) / 3.0
    var = energy / (2.0 * 10.0 ** (snr_db / 10.0))
    sd = np.sqrt(var)
    lo_edge = np.concatenate(([-np.inf], v[1:] - 1.0))
    hi_edge = np.concatenate((v[:-1] + 1.0, [np.inf]))
    exd = edd = 0.0
    for x in v:
        lo = (lo_edge - x) / sd
        hi = (hi_edge - x) / sd
        p = normal_cdf(hi) - normal_cdf(lo)
        pdf_lo = np.where(np.isfinite(lo), normal_pdf(np.where(np.isfinite(lo), lo, 0.0)), 0.0)
        pdf_hi = np.where(np.isfinite(hi), normal_pdf(np.where(np.isfinite(hi), hi, 0.0)), 0.0)
        ex = x * p + sd * (pdf_lo - pdf_hi)
        exd += np.sum(v * ex)
        edd += np.sum(v * v * p)
    exd /= m
    edd /= m
    exx = np.mean(v * v) + var
    gain = exd / edd
    noise = exx - 2.0 * exd + edd - (exd - edd) ** 2 / edd
    return 10.0 * np.log10(gain * gain * edd / noise)


def table(b):
    return np.array([reading_db(b, FLOOR_DB + STEP_DB * i) for i in range(NODES)])


def invert(tab, r):
    if r <= tab[0]:
        return FLOOR_DB
    if r >= tab[-1]:
        return FLOOR_DB + STEP_DB * (NODES - 1) + (r - tab[-1])
    i = int(np.searchsorted(tab, r, side="right"))
    return FLOOR_DB + STEP_DB * ((i - 1) + (r - tab[i - 1]) / (tab[i] - tab[i - 1]))


def estimate(x, b, window):
    """The estimates of the windows after the first of the symbols ``x`` (complex128)."""
    m = 1 << b
    energy = 2.0 * (m * m - 1) / 3.0
    v = 2.0 * np.arange(m) - (m - 1)
    ka = np.mean(np.add.outer(v * v, v * v) ** 2) / energy**2
    tab = table(b)
    amplitude = 0.0
    out = []
    for w in range(len(x) // window):
        y = x[w * window : (w + 1) * window]
        p = np.abs(y) ** 2
        m2 = np.mean(p)
        m4 = np.mean(p * p)
        s2 = (2.0 * m2 * m2 - m4) / (2.0 - ka)
        signal = min(max(np.sqrt(s2), m2 / 11.0), m2) if s2 > 0 else m2 / 11.0
        nxt = np.sqrt(signal)
        if amplitude > 0.0:
            k = np.sqrt(energy) / (2.0 * amplitude)
            err = cross = dec = 0.0
            for part in (y.real, y.imag):
                t = part * k + m / 2.0
                i = np.floor(np.clip(t, 0.0, m - 1.0))
                d = 2.0 * i - (m - 1)
                e = 2.0 * (t - i) - 1.0
                err += np.sum(e * e)
                cross += np.sum(e * d)
                dec += np.sum(d * d)
            gain = 1.0 + cross / dec
            noise = err - cross * cross / dec
            snr = invert(tab, 10.0 * np.log10(gain * gain * dec / noise))
            out.append(snr)
            if snr >= FIT_FROM_DB[b]:
                nxt = amplitude * gain
        amplitude = nxt
    return np.array(out)


@case(kernel="blocks", name="snr_estimate_qam1024_c32", seed=901)
def snr_estimate_qam1024_c32(rng):
    b, window, per = 5, 512, 6
    m = 1 << b
    energy = 2.0 * (m * m - 1) / 3.0
    n = 3 * per * window
    levels = 2.0 * rng.integers(0, m, size=(2, n)) - (m - 1)
    s = (levels[0] + 1j * levels[1]) / np.sqrt(energy)
    # Es/N0 34 dB (the fitted gain), then 20 dB (the flat part of the table, the moments) and
    # 8 dB (the steep part), six windows each; amplitude 0.6.
    snr = np.repeat([34.0, 20.0, 8.0], per * window)
    sd = np.sqrt(10.0 ** (-snr / 10.0) / 2.0)
    noise = sd * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x = (0.6 * (s + noise)).astype(np.complex64)
    expected = estimate(x.astype(np.complex128), b, window)
    assert len(expected) == 3 * per - 1
    # Each part reads its Es/N0 (the spread of a window of 512 symbols: about 0.3 dB at 34 dB,
    # 2 dB in the flat part of the table at 20 dB, 1.5 dB at 8 dB).
    assert abs(np.mean(expected[1 : per - 1]) - 34.0) < 0.5
    assert abs(np.mean(expected[per + 1 : 2 * per - 1]) - 20.0) < 3.0
    assert abs(np.mean(expected[2 * per + 1 :]) - 8.0) < 2.0
    return dict(
        params={"constellation": "qam1024", "window": window, "windows": 3 * per},
        arrays={"input": Array("cf32", x), "expected": Array("f64", expected)},
        tolerance={"class": "max_abs_err", "max_abs_err": 0.01, "skip": 0},
        model="float64",
        notes="snr_estimate with qam1024 and a window of 512 symbols: input are 18 windows of "
        "random 1024-QAM symbols (unit-power grid, amplitude 0.6) with complex Gaussian noise "
        "at Es/N0 34 dB (windows 0 to 5), 20 dB (6 to 11) and 8 dB (12 to 17); expected is the "
        "estimate in dB after each window from window 1 on (window 0 only measures the "
        "amplitude), from the f64 model of the estimator: decisions at the amplitude of the "
        "previous window, the least-squares gain, the inverse of the reading table over white "
        "Gaussian noise (SciPy), the amplitude from the fitted gain from 30.5 dB on and from "
        "M2M4 (held between M2 / 11 and M2) below; within 0.01 dB (a float32 decision at a "
        "boundary)",
    )
