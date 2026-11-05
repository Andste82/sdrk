"""Cases for the symbol_sync kernel (symbol timing recovery, M38 section 1).

The timing loop is a feedback system (every strobe position depends on the error of the previous
strobe), so the expected values come from an exact float32 model of the arithmetic fixed by the
specification (rule R8.3 exception, as for ``squelch``): the cubic Lagrange interpolation and the
timing error in float32 with the ``f32model`` helpers, one operation at a time, and the loop
filter and the strobe positions in float64, with the rules the kernel documents in
``docs/kernels.md``: the position as an integer index into the samples (three zero history
samples followed by the input) and a fraction ``mu`` (D530), the first strobe at input sample 1,
the Gardner mid sample at ``mu - omega 0.5`` from the state after the previous strobe (D535),
``e = 0`` for the first strobe and for a non-finite error (D533), ``sign(0) = 0`` (D536), the
flush of the integrator (D537) and the limit of the phase step (D534, never reached here).

The signals use ``math.sin`` and ``math.cos`` one value at a time and the noise comes from
``rng.standard_normal``; nothing that feeds a committed byte uses NumPy transcendentals or BLAS,
so the files are identical on every CPU.

The regression guard of decision D111 runs the same loop in float64 (float64 interpolation with
the float64 constants, float64 errors, ``mu`` not rounded) on the stored inputs.
``snr_db_guard`` is the SNR of the model output against that float64 output over the outputs both
produce, from output ``guard_skip`` on, minus 1 dB, rounded down to 0.1 dB. The float64 output is
not committed: the Rust test runs its own float64 loop.
"""

import math
import struct

import numpy as np

from .. import f32model as fm
from ..registry import Array, case

# The block defaults of M38 section 2.
LOOP_BW = 0.01
DAMPING = 0.7071
MAX_DEVIATION = 0.005
TED_GAIN = 1.0

ROLLOFF = 0.35
# Symbols per side of the truncated raised-cosine pulse.
SPAN = 16

GUARD_SKIP = 0

THIRD = fm.div(1.0, 3.0)
HALF = fm.f32(0.5)
SIXTH = fm.div(1.0, 6.0)
ZERO = fm.f32(0.0)


def bits(x):
    """The IEEE-754 bits of a float64 as an integer (exact in JSON)."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def gains(loop_bw, damping, max_deviation, ted_gain, sps):
    """The loop constants of ``SymbolSyncState::new`` in the order of D537."""
    theta = loop_bw / (damping + 1.0 / (4.0 * damping))
    d = (1.0 + (2.0 * damping) * theta) + theta * theta
    kp = ((((4.0 * damping) * theta) / d) / ted_gain) * sps
    ki = (((4.0 * (theta * theta)) / d) / ted_gain) * sps
    return kp, ki, max_deviation * sps


def flush_f64(x):
    """``num::flush_denormal_f64``: 0.0 when ``|x| < 1e-30``."""
    return 0.0 if abs(x) < 1e-30 else x


def clamp(x, lo, hi):
    """``x.max(lo).min(hi)`` of Rust for finite values."""
    return min(max(x, lo), hi)


# ---------------------------------------------------------------------------------------------
# Arithmetic per number type: float32 (the model) and float64 (the guard)
# ---------------------------------------------------------------------------------------------


class F32Ops:
    """The float32 model: every operation rounded once to float32."""

    zero = ZERO

    @staticmethod
    def interp(xm1, x0, x1, x2, mu):
        c1 = fm.sub(fm.add(fm.sub(-fm.mul(xm1, THIRD), fm.mul(x0, HALF)), x1), fm.mul(x2, SIXTH))
        c2 = fm.add(fm.sub(fm.mul(xm1, HALF), x0), fm.mul(x1, HALF))
        c3 = fm.add(fm.sub(fm.add(-fm.mul(xm1, SIXTH), fm.mul(x0, HALF)), fm.mul(x1, HALF)), fm.mul(x2, SIXTH))
        return fm.add(fm.mul(fm.add(fm.mul(fm.add(fm.mul(c3, mu), c2), mu), c1), mu), x0)

    mu = staticmethod(fm.f32)
    add = staticmethod(fm.add)
    sub = staticmethod(fm.sub)
    mul = staticmethod(fm.mul)
    is_finite = staticmethod(fm.is_finite)

    @staticmethod
    def sign(x):
        return fm.f32(1.0) if x > ZERO else (fm.f32(-1.0) if x < ZERO else ZERO)


class F64Ops:
    """The float64 guard: the same formulas in Python floats."""

    zero = 0.0

    @staticmethod
    def interp(xm1, x0, x1, x2, mu):
        c1 = ((-(xm1 * (1.0 / 3.0)) - x0 * 0.5) + x1) - x2 * (1.0 / 6.0)
        c2 = (xm1 * 0.5 - x0) + x1 * 0.5
        c3 = ((-(xm1 * (1.0 / 6.0)) + x0 * 0.5) - x1 * 0.5) + x2 * (1.0 / 6.0)
        return ((c3 * mu + c2) * mu + c1) * mu + x0

    @staticmethod
    def mu(x):
        return x

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def is_finite(x):
        return math.isfinite(x)

    @staticmethod
    def sign(x):
        return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def symbol_sync(ops, comps, ted, sps, kp, ki, lim):
    """The loop of section 1 over the whole input in one call.

    ``comps`` is a list of component sequences (one for ``f32``, ``re`` and ``im`` for ``C32``)
    of the input, already in the number type of ``ops``. Returns the outputs as a list of
    component tuples and the absolute strobe positions (input index plus fraction).
    """
    nc = len(comps)
    # b = three zero history samples followed by the input (D530).
    b = [[ops.zero] * 3 + list(c) for c in comps]
    length = len(b[0])
    i, mu, omega, integ = 4, 0.0, sps, 0.0
    started, mid_ready = False, False
    y_prev = d_prev = mid = tuple([ops.zero] * nc)
    out, positions = [], []

    def interp(at, frac):
        m = ops.mu(frac)
        return tuple(ops.interp(c[at - 1], c[at], c[at + 1], c[at + 2], m) for c in b)

    while True:
        if ted == "gardner" and started and not mid_ready:
            v = mu - omega * 0.5
            f = math.floor(v)
            im = i + int(f)
            if im + 2 >= length:
                break
            mid = interp(im, v - f)
            mid_ready = True
        if i + 2 >= length:
            break
        y = interp(i, mu)
        e = ops.zero
        if ted == "gardner":
            if started:
                e = ops.mul(ops.sub(y_prev[0], y[0]), mid[0])
                for k in range(1, nc):
                    e = ops.add(e, ops.mul(ops.sub(y_prev[k], y[k]), mid[k]))
            mid_ready = False
        else:
            d = tuple(ops.sign(v) for v in y)
            if started:
                a = ops.mul(d_prev[0], y[0])
                c = ops.mul(d[0], y_prev[0])
                for k in range(1, nc):
                    a = ops.add(a, ops.mul(d_prev[k], y[k]))
                    c = ops.add(c, ops.mul(d[k], y_prev[k]))
                e = ops.sub(a, c)
            d_prev = d
        y_prev = y
        started = True
        out.append(y)
        positions.append((i - 3) + mu)
        e = float(e) if ops.is_finite(e) else 0.0
        integ = clamp(flush_f64(integ + ki * e), -lim, lim)
        omega = sps + integ
        q = omega * 0.25
        step = kp * e
        assert -q < step < q, "the phase step limit of D534 was reached"
        v = (mu + omega) + clamp(step, -q, q)
        f = math.floor(v)
        i += int(f)
        mu = v - f
    return out, positions


# ---------------------------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------------------------


def raised_cosine(t, beta):
    """The raised-cosine pulse at ``t`` symbols, ``p(0) = 1``."""
    if t == 0.0:
        return 1.0
    sinc = math.sin(math.pi * t) / (math.pi * t)
    den = 1.0 - (2.0 * beta * t) ** 2
    if abs(den) < 1e-9:
        u = 1.0 / (2.0 * beta)
        return (math.pi / 4.0) * math.sin(math.pi * u) / (math.pi * u)
    return sinc * math.cos(math.pi * beta * t) / den


def shaped(n, symbols, sps, offset, delta):
    """``sum_k a_k p((n delta - k sps - offset) / sps)`` for ``n < n``, ``a_k = symbols[k]``
    (complex), symbol 0 at the transmit position ``offset``; ``delta = 1 + ppm 1e-6`` is the
    clock drift of D540 (receive sample ``n`` is the transmit position ``n delta``). The symbol
    list starts ``SPAN`` symbols before 0, so the signal is complete from the first sample on.
    Returns the list of complex samples, summed in ascending ``k`` per sample."""
    x = [0j] * n
    for idx, a in enumerate(symbols):
        k = idx - SPAN
        centre = k * sps + offset
        lo = max(0, math.ceil((centre - SPAN * sps) / delta))
        hi = min(n - 1, math.floor((centre + SPAN * sps) / delta))
        for m in range(lo, hi + 1):
            x[m] += a * raised_cosine((m * delta - centre) / sps, ROLLOFF)
    return x


def centres(count, sps, offset, delta):
    """The receive positions of the symbols ``0 .. count``."""
    return [(k * sps + offset) / delta for k in range(count)]


def timing_error(positions, sym_pos, sps):
    """The largest distance (in symbols) of the last 256 strobes from the nearest symbol."""
    worst, j = 0.0, 0
    for p in positions[-256:]:
        while j + 1 < len(sym_pos) and sym_pos[j + 1] <= p:
            j += 1
        near = min(abs(p - sym_pos[j]), abs(sym_pos[min(j + 1, len(sym_pos) - 1)] - p))
        worst = max(worst, near / sps)
    return worst


# ---------------------------------------------------------------------------------------------
# Model runs and the float64 guard
# ---------------------------------------------------------------------------------------------


def run_case(x, complex_input, ted, sps):
    """Runs the float32 model and the float64 guard on the stored input ``x`` (complex64 or
    float32). Returns the expected array, the positions and the guard."""
    kp, ki, lim = gains(LOOP_BW, DAMPING, MAX_DEVIATION, TED_GAIN, sps)
    if complex_input:
        comps32 = [[fm.f32(v) for v in x.real.tolist()], [fm.f32(v) for v in x.imag.tolist()]]
    else:
        comps32 = [[fm.f32(v) for v in x.tolist()]]
    out, positions = symbol_sync(F32Ops, comps32, ted, sps, kp, ki, lim)
    comps64 = [[float(v) for v in c] for c in comps32]
    out64, _ = symbol_sync(F64Ops, comps64, ted, sps, kp, ki, lim)
    n = min(len(out), len(out64))
    assert abs(len(out) - len(out64)) <= 1, (len(out), len(out64))
    num = math.fsum(sum(float(v) ** 2 for v in y) for y in out64[GUARD_SKIP:n])
    den = math.fsum(
        sum((float(a) - float(b)) ** 2 for a, b in zip(y, z))
        for y, z in zip(out64[GUARD_SKIP:n], out[GUARD_SKIP:n])
    )
    assert den > 0.0
    guard = math.floor((10.0 * math.log10(num / den) - 1.0) * 10.0) / 10.0
    if complex_input:
        expected = np.empty(len(out), dtype=np.complex64)
        expected.real = np.array([y[0] for y in out], dtype=np.float32)
        expected.imag = np.array([y[1] for y in out], dtype=np.float32)
    else:
        expected = np.array([y[0] for y in out], dtype=np.float32)
    co = {"kp_bits": bits(kp), "ki_bits": bits(ki), "max_integ_bits": bits(lim)}
    return expected, positions, guard, co


def sync_params(sps, ted, produced, guard, co, **extra):
    base = {
        "sps": sps,
        "ted": ted,
        "loop_bw": LOOP_BW,
        "damping": DAMPING,
        "max_deviation": MAX_DEVIATION,
        "ted_gain": TED_GAIN,
        "produced": produced,
        "guard_skip": GUARD_SKIP,
        "snr_db_guard": guard,
    }
    base.update(co)
    base.update(extra)
    return base


def complex64(x):
    """A complex64 array from a list of complex (each part rounded once)."""
    out = np.empty(len(x), dtype=np.complex64)
    out.real = np.array([v.real for v in x], dtype=np.float64).astype(np.float32)
    out.imag = np.array([v.imag for v in x], dtype=np.float64).astype(np.float32)
    return out


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


@case(kernel="symbol_sync", name="gardner_c32_sps4p1_16384", seed=490)
def gardner_c32(rng):
    n, sps, ppm, ebn0_db = 16384, 4.1, 40.0, 20.0
    delta = 1.0 + ppm * 1e-6
    offset = float(rng.uniform(0.0, sps))
    count = int(n * delta / sps) + 2 * SPAN + 2
    data = rng.integers(0, 2, count).tolist()
    symbols = [complex(2 * b - 1, 0) for b in data]
    clean = shaped(n, symbols, sps, offset, delta)
    power = math.fsum(abs(v) ** 2 for v in clean) / n
    sigma = math.sqrt(power * sps / (2.0 * 1 * 10.0 ** (ebn0_db / 10.0)))
    nre, nim = rng.standard_normal(n), rng.standard_normal(n)
    noisy = [complex(v.real + sigma * a, v.imag + sigma * b) for v, a, b in zip(clean, nre.tolist(), nim.tolist())]
    x = complex64(noisy)
    expected, positions, guard, co = run_case(x, True, "gardner", sps)
    err = timing_error(positions, centres(count, sps, offset, delta), sps)
    assert err < 0.1, err
    return dict(
        params=sync_params(
            sps, "gardner", len(expected), guard, co,
            rolloff=ROLLOFF, span=SPAN, drift_ppm=ppm, ebn0_db=ebn0_db, offset=offset,
            sigma=sigma, timing_error_last_256=round(err, 6),
        ),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="BPSK +-1 shaped with a raised cosine (roll-off 0.35, p(0) = 1, 16 symbols per side), "
        "sps 4.1, symbol k at the transmit position offset + k sps, 40 ppm clock drift (receive sample "
        "n is the transmit position n (1 + 40e-6), D540), complex noise for Eb/N0 20 dB from the "
        "measured power per sample; Gardner with the block defaults (loop_bw 0.01, damping 0.7071, "
        "max_deviation 0.005, ted_gain 1); exact float32 model of D530 to D537 over the whole input "
        "in one call; produced is the output count; timing_error_last_256 is the largest distance of "
        "the last 256 strobes from a symbol centre in symbols (information); snr_db_guard compares "
        "with the float64 loop (D111)",
    )


@case(kernel="symbol_sync", name="gardner_f32_sps8_8192", seed=491)
def gardner_f32(rng):
    n, sps, sigma = 8192, 8, 0.1
    offset = float(rng.uniform(0.0, sps))
    count = n // sps + 3
    data = rng.integers(0, 2, count).tolist()
    noise = rng.standard_normal(n).tolist()
    # Symbol j (level 2 b_j - 1) covers the time [offset + (j - 1) sps, offset + j sps); the
    # levels before the first symbol continue it. Sample n is the mean over the time
    # [n - sps, n] (a boxcar of one symbol after an NRZ waveform, the shape of a 2-FSK
    # discriminator output), computed exactly from the overlaps.
    x = []
    for m in range(n):
        lo, hi = m - sps, float(m)
        acc = 0.0
        j0 = max(0, math.floor((lo - offset) / sps) + 1)
        j = j0
        while True:
            a = offset + (j - 1) * sps if j > 0 else -math.inf
            b_end = offset + j * sps
            if a >= hi:
                break
            overlap = min(hi, b_end) - max(lo, a)
            if overlap > 0.0:
                acc += (2.0 * data[j] - 1.0) * overlap
            j += 1
        x.append(acc / sps + sigma * noise[m])
    x = np.array(x, dtype=np.float64).astype(np.float32)
    expected, positions, guard, co = run_case(x, False, "gardner", float(sps))
    # The symbol j is complete (the peak of the boxcar) at the end of its interval.
    sym = [offset + j * sps for j in range(count)]
    err = timing_error(positions, sym, sps)
    assert err < 0.15, err
    return dict(
        params=sync_params(
            float(sps), "gardner", len(expected), guard, co, offset=offset, sigma=sigma,
            timing_error_last_256=round(err, 6),
        ),
        arrays={"input": Array("f32", x), "expected": Array("f32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="real NRZ levels +-1 of random bits, symbol j covering the time [offset + (j - 1) 8, "
        "offset + 8 j), averaged over a sliding window of one symbol (sample n = the exact mean over "
        "[n - 8, n], the boxcar-filtered shape of a 2-FSK discriminator output; the symbol peaks at "
        "offset + 8 j), plus real noise of sigma 0.1; sps 8; Gardner with the block defaults; exact "
        "float32 model over the whole input in one call; timing_error_last_256 in symbols "
        "(information); snr_db_guard compares with the float64 loop (D111)",
    )


@case(kernel="symbol_sync", name="mm_c32_sps2_8192", seed=492)
def mm_c32(rng):
    n, sps, timing = 8192, 2.0, 0.37
    offset = timing * sps
    count = int(n / sps) + 2 * SPAN + 2
    data = rng.integers(0, 2, 2 * count).tolist()
    r = 1.0 / math.sqrt(2.0)
    symbols = [complex((2 * data[2 * k] - 1) * r, (2 * data[2 * k + 1] - 1) * r) for k in range(count)]
    x = complex64(shaped(n, symbols, sps, offset, 1.0))
    expected, positions, guard, co = run_case(x, True, "mm", sps)
    err = timing_error(positions, centres(count, sps, offset, 1.0), sps)
    assert err < 0.1, err
    return dict(
        params=sync_params(
            sps, "mm", len(expected), guard, co, rolloff=ROLLOFF, span=SPAN,
            timing_offset_symbols=timing, timing_error_last_256=round(err, 6),
        ),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", expected)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="QPSK (Gray, bit 2k on the real and 2k + 1 on the imaginary part, each (2b - 1) / sqrt(2)) "
        "shaped with the raised cosine of gardner_c32_sps4p1_16384, sps 2, symbol k at the sample "
        "position (k + 0.37) sps (timing offset 0.37 symbols), no noise, no drift; Mueller-Mueller "
        "with the block defaults; exact float32 model over the whole input in one call; "
        "timing_error_last_256 in symbols (information); snr_db_guard compares with the float64 loop "
        "(D111)",
    )
