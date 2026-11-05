"""Cases for the auto_notch kernel (normalized LMS line enhancer, M33 section 2).

The loop feeds back, so the expected values come from the exact float32 model (``f32model``) of
the arithmetic fixed by the specification (rule R8.3 exception, as for ``squelch``): the
reference vector ``u[i] = x[n - delay - i]`` (zero before the first sample), the prediction
``y = sum w[i] u[i]`` and the power ``P = sum u[i] u[i]``, each from ``+0.0`` in ascending
``i`` (D370), the output ``e = x[n] - y`` and the update ``w[i] = flush_denormal(w[i] + f u[i])``
with ``f = (mu e) / (P + 1e-6)`` computed once; a non-finite ``f`` skips the update and a
non-finite ``e`` gives ``+0.0`` (D422). Non-finite inputs do not occur in these cases.

The tones use ``math.sin`` of an exactly reduced phase, one value at a time, and the noise comes
from ``rng.standard_normal``; nothing that feeds a committed byte uses NumPy transcendentals or
BLAS, so the files are identical on every CPU.

The regression guard of decision D111 runs the same loop in float64 (no flush of denormals) on
the stored input; ``snr_db_guard`` is the SNR of the model output against the float64 output
from sample ``guard_skip`` on, minus 1 dB, rounded down to 0.1 dB.
"""

import math

import numpy as np

from .. import f32model as fm
from ..registry import Array, case

RATE_HZ = 8000
NOISE_SIGMA = 0.1
TONE1_HZ, TONE1_AMP = 1000, 0.5
TONE2_HZ, TONE2_AMP = 1700, 0.3
SWEEP_END_HZ = 1100
TAPS = 32
DELAY = 8
MU = 0.01
EPS = 1e-6
GUARD_SKIP = 0


def _tone(amp, turns):
    return np.array([amp * math.sin(2.0 * math.pi * t) for t in turns], dtype=np.float64)


def fixed_turns(freq_hz, n):
    """``frac(freq n / fs)`` with exact integer arithmetic."""
    return [((freq_hz * k) % RATE_HZ) / RATE_HZ for k in range(n)]


def sweep_turns(f0_hz, f1_hz, n):
    """The phase in turns of a tone whose frequency rises linearly from ``f0`` at sample 0 to
    ``f1`` at sample ``n``: ``(f0 k + (f1 - f0) k^2 / (2 n)) / fs``, reduced exactly."""
    den = 2 * n * RATE_HZ
    return [((2 * n * f0_hz * k + (f1_hz - f0_hz) * k * k) % den) / den for k in range(n)]


def signal(rng, n, sweep):
    """Real white noise sigma 0.1 plus 1000 Hz 0.5 (sweeping to 1100 Hz) and 1700 Hz 0.3."""
    noise = rng.standard_normal(n) * NOISE_SIGMA
    t1 = sweep_turns(TONE1_HZ, SWEEP_END_HZ, n) if sweep else fixed_turns(TONE1_HZ, n)
    return noise + _tone(TONE1_AMP, t1) + _tone(TONE2_AMP, fixed_turns(TONE2_HZ, n))


def auto_notch_f32(x, taps, delay, mu):
    """Float32 model; ``x`` is a float32 array. Returns the output as a float32 array."""
    zero = fm.f32(0.0)
    mu, eps = fm.f32(mu), fm.f32(EPS)
    w = [zero] * taps
    # The stored inputs, with taps + delay zeros in front (the zero history of the state).
    buf = [zero] * (taps + delay)
    out = np.zeros(len(x), dtype=np.float32)
    for n, raw in enumerate(x.tolist()):
        raw = fm.f32(raw)
        finite = fm.is_finite(raw)
        xv = raw if finite else zero
        base = taps + n
        u = [buf[base - i] for i in range(taps)]
        y, power = zero, zero
        for wi, ui in zip(w, u):
            y = fm.add(y, fm.mul(wi, ui))
            power = fm.add(power, fm.mul(ui, ui))
        if finite:
            e = fm.sub(xv, y)
            f = fm.div(fm.mul(mu, e), fm.add(power, eps))
            if fm.is_finite(f):
                w = [fm.flush_denormal(fm.add(wi, fm.mul(f, ui))) for wi, ui in zip(w, u)]
            out[n] = e if fm.is_finite(e) else zero
        buf.append(xv)
    return out


def auto_notch_f64(x, taps, delay, mu):
    """The same loop in float64 (no flush); ``x`` is a list of floats."""
    w = [0.0] * taps
    buf = [0.0] * (taps + delay)
    out = []
    for n, xv in enumerate(x):
        base = taps + n
        u = [buf[base - i] for i in range(taps)]
        y = sum(wi * ui for wi, ui in zip(w, u))
        power = sum(ui * ui for ui in u)
        e = xv - y
        f = (mu * e) / (power + EPS)
        w = [wi + f * ui for wi, ui in zip(w, u)]
        out.append(e)
        buf.append(xv)
    return out


def snr_db(reference, test):
    """``10 log10(sum ref^2 / sum (ref - test)^2)`` with exact sums."""
    num = math.fsum(r * r for r in reference)
    den = math.fsum((r - t) * (r - t) for r, t in zip(reference, test))
    return math.inf if den == 0.0 else 10.0 * math.log10(num / den)


def guard_db(model, f64):
    snr = snr_db(f64[GUARD_SKIP:], model[GUARD_SKIP:])
    return math.floor((snr - 1.0) * 10.0) / 10.0


def tone_power(y, freq_turns):
    """The power of the component of ``y`` at the given phases (a single-bin DFT)."""
    c = math.fsum(v * math.cos(2.0 * math.pi * t) for v, t in zip(y, freq_turns))
    s = math.fsum(v * math.sin(2.0 * math.pi * t) for v, t in zip(y, freq_turns))
    return 2.0 * (c * c + s * s) / (len(y) * len(y))


def check_notch(x, out, sweep):
    """The tones are attenuated by more than 20 dB over the last quarter."""
    n = len(x)
    t1 = sweep_turns(TONE1_HZ, SWEEP_END_HZ, n) if sweep else fixed_turns(TONE1_HZ, n)
    t2 = fixed_turns(TONE2_HZ, n)
    tail = slice(3 * n // 4, n)
    xs, ys = x.astype(np.float64).tolist()[tail], out.astype(np.float64).tolist()[tail]
    for t in (t1[tail], t2[tail]):
        before, after = tone_power(xs, t), tone_power(ys, t)
        assert 10.0 * math.log10(before / after) > 20.0, (before, after)


def an_params(**extra):
    base = {"rate_hz": RATE_HZ, "taps": TAPS, "delay": DELAY, "mu": MU, "guard_skip": GUARD_SKIP}
    base.update(extra)
    return base


def an_case(rng, sweep):
    x = signal(rng, 16384, sweep).astype(np.float32)
    out = auto_notch_f32(x, TAPS, DELAY, MU)
    check_notch(x, out, sweep)
    f64 = auto_notch_f64(x.astype(np.float64).tolist(), TAPS, DELAY, MU)
    guard = guard_db(out.astype(np.float64).tolist(), f64)
    return x, out, guard


@case(kernel="auto_notch", name="an_tone_f32_16384", seed=474)
def an_tone_f32(rng):
    x, out, guard = an_case(rng, sweep=False)
    return dict(
        params=an_params(snr_db_guard=guard),
        arrays={"input": Array("f32", x), "expected": Array("f32", out)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="8 kHz; real white noise sigma 0.1 plus tones 1000 Hz 0.5 and 1700 Hz 0.3 (sine, "
        "start phase 0); taps 32, delay 8, mu 0.01; exact float32 model (sums from +0.0 in "
        "ascending i, D370); the generator asserts that both tones lose more than 20 dB over "
        "the last quarter; snr_db_guard compares with the float64 loop (D111)",
    )


@case(kernel="auto_notch", name="an_sweep_f32_16384", seed=475)
def an_sweep_f32(rng):
    x, out, guard = an_case(rng, sweep=True)
    return dict(
        params=an_params(sweep_end_hz=SWEEP_END_HZ, snr_db_guard=guard),
        arrays={"input": Array("f32", x), "expected": Array("f32", out)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="as an_tone_f32_16384 (own draw), the 1000 Hz tone sweeps linearly to 1100 Hz "
        "over the 16384 samples (phase (f0 k + (f1 - f0) k^2 / (2 n)) / fs turns, reduced "
        "exactly)",
    )
