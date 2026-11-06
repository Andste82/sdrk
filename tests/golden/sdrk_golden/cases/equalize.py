"""Cases for the equalize kernel (adaptive equalizer, CMA and decision-directed normalized LMS,
decision feedback and a decision-directed phase tracker; M43 section 1 as amended by D825 to D833).

The equalizer is a feedback system with discrete decisions, so the expected values come from an
exact float32 model of the arithmetic fixed by the contract (rule R8.3 exception, as for
``symbol_sync``; decisions D803 to D807 and D826 to D833). Per symbol, with ``N`` forward and ``M``
feedback taps, the joint vector ``h`` (the forward history ``h[0..N]``, newest first, then the
fed-back decisions ``h[N..N + M]``) and the weights ``w``:

1. the forward history moves by one and ``h[0] = x`` (0 when ``x`` is not finite); ``y`` is the
   ``dot_complex`` of ``h`` and ``w`` over all ``N + M`` elements in its order (the model of
   ``cases/dot_complex.py``: eight partial sums per component, element ``i`` to partial sum
   ``i mod 8``, the fixed tree);
2. if ``x`` and ``y`` are finite: ``(c, s)`` = the interpolated cosine and sine of the tracker
   phase from the table of ``rotate`` (the rule of the ``costas`` cases; exactly ``(1, 0)`` at
   phase 0), ``z = (y.re c + y.im s, y.im c - y.re s)``, the decision ``d`` on ``z`` (QPSK the signs
   per component on ``+-1/sqrt 2`` as float32, ``-0.0`` and NaN count as not negative; 8PSK the
   half-open octant point of ``costas`` order 8), ``ed = z - d``, ``p = ed.re ed.re + ed.im ed.im``
   and, if ``p`` is finite, ``mse = flush(mse + 0.01 (p - mse))``;
3. CMA ``f = (y.re y.re + y.im y.im) - 1`` and ``e = (clip(y.re f), clip(y.im f))`` (clip to
   ``[-1, 1]``) with ``mu_cma`` and the feedback value 0; decision mode ``e = (ed.re c - ed.im s,
   ed.re s + ed.im c)`` with ``mu_dd`` and the feedback value ``(d.re c - d.im s, d.re s + d.im c)``;
4. ``P`` = the sum of ``h.re h.re + h.im h.im`` over the ``N + M`` elements in ascending order from
   0 (``Pf`` its value after the ``N`` forward elements), ``m = mu / (P + 1e-6)``,
   ``g = (m e.re, m e.im)``;
5. if ``Pf`` is not 0 and ``g`` is finite: ``w[i] = flush(w[i] - g conj(h[i]))`` per component with
   ``g conj(h) = (g.re h.re + g.im h.im, g.im h.re - g.re h.im)`` (back to the start weights if a
   weight overflows, never reached here);
6. the tracker: ``pd = z.im d.re - z.re d.im`` (0 if not finite), ``freq = clip(flush(freq + ki pd),
   0.125)``, ``phase = phase + floor((freq + kp pd) 2^32 + 0.5)`` modulo ``2^32``, with ``kp`` and
   ``ki`` the float32 values of the ``costas`` gains for ``phase_bw`` and the damping 0.7071;
7. the fed-back decisions move by one and ``h[N]`` = the feedback value (0 for a non-finite ``x``
   or ``y`` and when ``Pf`` is 0);
8. the output is ``z`` (0 for a non-finite ``x`` or ``y``);
9. in CMA mode the counter advances and decision mode starts after ``cma_symbols`` symbols
   (never for ``2^32 - 1``).

The start state has the centre forward tap 1, the other weights 0, a zero history, ``mse`` 1 and
the tracker at phase and frequency 0. The model evaluates every operation in NumPy ``float32``, on
scalars or element by element on ``float32`` arrays of the ``N + M`` taps (one rounded operation
per element, nothing fused, no reduction function; the window power is a scalar loop), so it
equals Rust ``f32`` code without FMA on every CPU.

The signals: symbols with unit energy through a fixed symbol-spaced complex channel (the sums in
plain Python complex arithmetic, in ascending tap order), turned by a constant carrier phase, plus
complex noise from ``rng.standard_normal`` for the stated Eb/N0 per transmitted symbol energy
(``sigma^2 = 1 / (2 bits_per_symbol Eb/N0)``). The channel gains use ``math.cos`` and ``math.sin``
of constants; nothing that feeds a committed byte uses NumPy transcendentals or BLAS.

The regression guard of decision D111 runs the same loop in float64 (the filter sum in ascending
``i``, the float64 points ``1/sqrt 2`` and ``cos(pi/8)``, ``sin(pi/8)``, the exact cosine and sine of
the ``u32`` tracker phase, the step sizes and gains as the float64 values of their float32, the
flush at 1e-30) on the stored inputs. ``snr_db_guard`` is the SNR of the model output against that
float64 output from symbol ``guard_skip`` on, minus 1 dB, rounded down to 0.1 dB; the Rust test runs
its own float64 loop.
"""

import math
import struct

import numpy as np

from .. import f32model as fm
from ..registry import Array, case
from .costas import C8, GRAY_TO_INDEX8, S8, _octant8, _point8_f32, gains
from .dot_complex import PARTIALS
from .stereo import _interp_f32

F32 = np.float32
ZERO = F32(0.0)
ONE = F32(1.0)
ALPHA = F32(0.01)
QPSK_A = F32(1.0 / math.sqrt(2.0))
FLUSH = F32(1e-30)
NLMS_EPS = F32(1e-6)
CMA_CLIP = F32(1.0)
MAX_FREQ = F32(0.125)
DAMPING = 0.7071
NEVER = 2**32 - 1
TWO32 = 1 << 32
MASK = TWO32 - 1
QUARTER = 1 << 30
GUARD_SKIP = 0
# The 8PSK points of the float64 guard: cos(pi / 8) and sin(pi / 8), written out so that the Rust
# guard uses the same float64 values.
C8_F64 = 0.9238795325112867
S8_F64 = 0.3826834323650898
QPSK_A_F64 = 0.7071067811865476


def f32_bits(x):
    """The IEEE-754 bits of a float32 as an integer."""
    return struct.unpack("<I", struct.pack("<f", float(F32(x))))[0]


def flush_arr(v):
    """``flush_denormal`` element by element (NaN stays NaN)."""
    return np.where(np.abs(v) < FLUSH, ZERO, v).astype(F32)


def clip(v, limit):
    """``v`` limited to ``[-limit, limit]`` as two selects (NaN stays NaN)."""
    v = limit if v > limit else v
    return -limit if v < -limit else v


def start_weights(n, taps):
    wr = np.zeros(n, dtype=F32)
    wi = np.zeros(n, dtype=F32)
    wr[taps // 2] = ONE
    return wr, wi


def decide_f32(constellation, yr, yi):
    if constellation == "qpsk":
        return (-QPSK_A if yr < ZERO else QPSK_A), (-QPSK_A if yi < ZERO else QPSK_A)
    if constellation == "8psk":
        return _point8_f32(_octant8(yr, yi))
    raise ValueError(constellation)


def dot_f32(hr, hi, wr, wi):
    """``dot_complex(u, w)`` in the contract order for one window (float32 scalars out)."""
    pr = hr * wr - hi * wi
    pi = hr * wi + hi * wr
    acc_r = np.zeros(PARTIALS, dtype=F32)
    acc_i = np.zeros(PARTIALS, dtype=F32)
    for start in range(0, len(pr), PARTIALS):
        br, bi = pr[start : start + PARTIALS], pi[start : start + PARTIALS]
        m = len(br)
        acc_r[:m] = acc_r[:m] + br
        acc_i[:m] = acc_i[:m] + bi

    def tree(a):
        s0, s1, s2, s3 = a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]
        return (s0 + s2) + (s1 + s3)

    return tree(acc_r), tree(acc_i)


def tracker_gains(phase_bw):
    """The float32 tracker gains (the ``costas`` formula with the damping 0.7071)."""
    kp, ki = gains(phase_bw, DAMPING)
    return fm.f32(kp), fm.f32(ki)


def equalize_f32(x, taps, feedback, constellation, mu_cma, mu_dd, cma_symbols, phase_bw):
    """The float32 model over the whole input in one call. Returns the output (complex64), the
    final weights (complex64), the final ``mse`` (float32), the final mode and the final tracker
    phase and frequency."""
    mu_c, mu_d = F32(mu_cma), F32(mu_dd)
    kp, ki = tracker_gains(phase_bw)
    n = taps + feedback
    wr, wi = start_weights(n, taps)
    hr = np.zeros(n, dtype=F32)
    hi = np.zeros(n, dtype=F32)
    mode, count, mse = ("cma" if cma_symbols != 0 else "decision"), 0, ONE
    phase, freq = 0, ZERO
    xr, xi = x.real.astype(F32), x.imag.astype(F32)
    out_r = np.zeros(len(x), dtype=F32)
    out_i = np.zeros(len(x), dtype=F32)
    with np.errstate(all="ignore"):
        for k in range(len(x)):
            a, b = xr[k], xi[k]
            finite = bool(np.isfinite(a) and np.isfinite(b))
            hr[1:taps] = hr[: taps - 1].copy()
            hi[1:taps] = hi[: taps - 1].copy()
            hr[0], hi[0] = (a, b) if finite else (ZERO, ZERO)
            yr, yi = dot_f32(hr, hi, wr, wi)
            fr_, fi_ = ZERO, ZERO
            if finite and np.isfinite(yr) and np.isfinite(yi):
                if phase == 0:
                    c, s = ONE, ZERO
                else:
                    c = _interp_f32(phase)
                    s = _interp_f32((phase - QUARTER) & MASK)
                zr = yr * c + yi * s
                zi = yi * c - yr * s
                dr, di = decide_f32(constellation, zr, zi)
                er, ei = zr - dr, zi - di
                p = er * er + ei * ei
                if np.isfinite(p):
                    mse = fm.flush_denormal(mse + ALPHA * (p - mse))
                if mode == "cma":
                    f = (yr * yr + yi * yi) - ONE
                    er, ei, mu = clip(yr * f, CMA_CLIP), clip(yi * f, CMA_CLIP), mu_c
                else:
                    er, ei = er * c - ei * s, er * s + ei * c
                    fr_, fi_ = dr * c - di * s, dr * s + di * c
                    mu = mu_d
                power = ZERO
                forward = ZERO
                for j in range(n):
                    power = power + (hr[j] * hr[j] + hi[j] * hi[j])
                    if j + 1 == taps:
                        forward = power
                m = mu / (power + NLMS_EPS)
                gr, gi = m * er, m * ei
                if forward == ZERO:
                    fr_, fi_ = ZERO, ZERO
                if forward != ZERO and np.isfinite(gr) and np.isfinite(gi):
                    cr = gr * hr + gi * hi
                    ci = gi * hr - gr * hi
                    nr, ni = flush_arr(wr - cr), flush_arr(wi - ci)
                    if np.all(np.isfinite(nr)) and np.all(np.isfinite(ni)):
                        wr, wi = nr, ni
                    else:
                        wr, wi = start_weights(n, taps)
                pd = zi * dr - zr * di
                if not np.isfinite(pd):
                    pd = ZERO
                freq = clip(fm.flush_denormal(freq + ki * pd), MAX_FREQ)
                step = freq + kp * pd
                phase = (phase + (int(math.floor(float(step) * 4294967296.0 + 0.5)) & MASK)) & MASK
                out_r[k], out_i[k] = zr, zi
            if feedback > 0:
                hr[taps + 1 :] = hr[taps : n - 1].copy()
                hi[taps + 1 :] = hi[taps : n - 1].copy()
                hr[taps], hi[taps] = fr_, fi_
            if mode == "cma" and cma_symbols != NEVER:
                count += 1
                if count >= cma_symbols:
                    mode = "decision"
    out = np.empty(len(x), dtype=np.complex64)
    out.real, out.imag = out_r, out_i
    w = np.empty(n, dtype=np.complex64)
    w.real, w.imag = wr, wi
    return out, w, F32(mse), mode, phase, F32(freq)


def flush_f64(v):
    return 0.0 if abs(v) < 1e-30 else v


def decide_f64(constellation, yr, yi):
    if constellation == "qpsk":
        a = QPSK_A_F64
        return (-a if yr < 0.0 else a), (-a if yi < 0.0 else a)
    k = _octant8(yr, yi)
    pts = (
        (C8_F64, S8_F64), (S8_F64, C8_F64), (-S8_F64, C8_F64), (-C8_F64, S8_F64),
        (-C8_F64, -S8_F64), (-S8_F64, -C8_F64), (S8_F64, -C8_F64), (C8_F64, -S8_F64),
    )
    return pts[k]


def equalize_f64(x, taps, feedback, constellation, mu_cma, mu_dd, cma_symbols, phase_bw):
    """The float64 guard loop (same steps, the filter and power sums in ascending ``i``, the
    exact cosine and sine of the tracker phase)."""
    mu_c, mu_d = float(F32(mu_cma)), float(F32(mu_dd))
    kp, ki = (float(v) for v in tracker_gains(phase_bw))
    n = taps + feedback
    w = [[0.0, 0.0] for _ in range(n)]
    w[taps // 2][0] = 1.0
    h = [[0.0, 0.0] for _ in range(n)]
    mode, count, mse = ("cma" if cma_symbols != 0 else "decision"), 0, 1.0
    phase, freq = 0, 0.0
    out = []
    for v in x.tolist():
        a, b = float(v.real), float(v.imag)
        h = [[a, b]] + h[: taps - 1] + h[taps:]
        yr, yi = 0.0, 0.0
        for (ur, ui), (pr, pi) in zip(h, w):
            yr += ur * pr - ui * pi
            yi += ur * pi + ui * pr
        theta = 2.0 * math.pi * (phase / 4294967296.0)
        c, s = math.cos(theta), math.sin(theta)
        zr, zi = yr * c + yi * s, yi * c - yr * s
        dr, di = decide_f64(constellation, zr, zi)
        er, ei = zr - dr, zi - di
        mse = flush_f64(mse + 0.01 * ((er * er + ei * ei) - mse))
        fed = [0.0, 0.0]
        if mode == "cma":
            f = (yr * yr + yi * yi) - 1.0
            er, ei, mu = min(max(yr * f, -1.0), 1.0), min(max(yi * f, -1.0), 1.0), mu_c
        else:
            er, ei = er * c - ei * s, er * s + ei * c
            fed = [dr * c - di * s, dr * s + di * c]
            mu = mu_d
        power = 0.0
        for ur, ui in h:
            power += ur * ur + ui * ui
        live = any(ur != 0.0 or ui != 0.0 for ur, ui in h[:taps])
        m = mu / (power + 1e-6) if live else 0.0
        if not live:
            fed = [0.0, 0.0]
        gr, gi = m * er, m * ei
        for (ur, ui), wk in zip(h, w):
            wk[0] = flush_f64(wk[0] - (gr * ur + gi * ui))
            wk[1] = flush_f64(wk[1] - (gi * ur - gr * ui))
        pd = zi * dr - zr * di
        freq = min(max(freq + ki * pd, -0.125), 0.125)
        phase = (phase + int(math.floor((freq + kp * pd) * 4294967296.0 + 0.5))) & MASK
        if feedback > 0:
            h = h[:taps] + [fed] + h[taps : n - 1]
        out.append(complex(zr, zi))
        if mode == "cma" and cma_symbols != NEVER:
            count += 1
            if count >= cma_symbols:
                mode = "decision"
    return out


def guard_db(model, f64_out):
    ref = f64_out[GUARD_SKIP:]
    test = [complex(v) for v in model.tolist()[GUARD_SKIP:]]
    num = math.fsum(abs(r) ** 2 for r in ref)
    den = math.fsum(abs(r - t) ** 2 for r, t in zip(ref, test))
    assert den > 0.0
    return math.floor((10.0 * math.log10(num / den) - 1.0) * 10.0) / 10.0


# ---------------------------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------------------------


def polar(mag, rad):
    return complex(mag * math.cos(rad), mag * math.sin(rad))


def symbols(rng, n, order):
    """Unit-energy symbols: QPSK ``((2 b0 - 1) + j (2 b1 - 1)) / sqrt 2``, 8PSK
    ``exp(j (2 pi k / 8 + pi / 8))`` with ``k`` from the Gray code ``b0 4 + b1 2 + b2``."""
    bps = {4: 2, 8: 3}[order]
    data = rng.integers(0, 2, n * bps).tolist()
    r = 1.0 / math.sqrt(2.0)
    out = []
    for k in range(n):
        if order == 4:
            out.append(complex((2 * data[2 * k] - 1) * r, (2 * data[2 * k + 1] - 1) * r))
        else:
            g = data[3 * k] * 4 + data[3 * k + 1] * 2 + data[3 * k + 2]
            out.append(polar(1.0, (2 * GRAY_TO_INDEX8[g] + 1) * math.pi / 8.0))
    return out, bps


def channel(sym, taps, phase, sigma, rng):
    """``x[n] = e^{j phase} sum_(lag, g) g s[n - lag] + noise[n]`` (symbols outside the list are 0;
    a negative lag is a precursor), the sum in the order of ``taps``."""
    n = len(sym)
    nre, nim = rng.standard_normal(n).tolist(), rng.standard_normal(n).tolist()
    rot = polar(1.0, phase)
    x = []
    for k in range(n):
        acc = 0j
        for lag, g in taps:
            j = k - lag
            if 0 <= j < n:
                acc += g * sym[j]
        v = acc * rot
        x.append(complex(v.real + sigma * nre[k], v.imag + sigma * nim[k]))
    return np.array(x, dtype=np.complex128).astype(np.complex64)


def dispersion(v):
    """The mean of ``(|y|^2 - 1)^2`` (the CMA cost), float64."""
    return math.fsum((abs(complex(y)) ** 2 - 1.0) ** 2 for y in v) / len(v)


def eq_params(taps, feedback, constellation, mu_cma, mu_dd, cma_symbols, phase_bw, end, guard, **extra):
    mse, mode, phase, freq = end
    base = {
        "taps": taps,
        "feedback_taps": feedback,
        "constellation": constellation,
        "mu_cma": mu_cma,
        "mu_dd": mu_dd,
        "cma_symbols": cma_symbols,
        "phase_bw": phase_bw,
        "final_mse_bits": f32_bits(mse),
        "final_mode": mode,
        "final_phase": phase,
        "final_frequency_bits": f32_bits(freq),
        "guard_skip": GUARD_SKIP,
        "snr_db_guard": guard,
    }
    base.update(extra)
    return base


def channel_param(taps):
    return [[lag, abs(g), math.atan2(g.imag, g.real)] for lag, g in taps]


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


@case(kernel="equalize", name="cma_qpsk_c32_16384", seed=590)
def cma_qpsk(rng):
    n, taps, feedback, mu_cma, mu_dd, ebn0_db, phase_bw = 16384, 11, 0, 0.05, 0.02, 20.0, 0.02
    phase = float(rng.uniform(0.0, math.tau))
    sym, bps = symbols(rng, n, 4)
    chan = [(-1, polar(0.1, 2.0)), (0, 1.0 + 0j), (1, polar(0.35, 0.8)), (2, polar(0.15, -2.0))]
    sigma = math.sqrt(1.0 / (2.0 * bps * 10.0 ** (ebn0_db / 10.0)))
    x = channel(sym, chan, phase, sigma, rng)
    args = (taps, feedback, "qpsk", mu_cma, mu_dd, NEVER, phase_bw)
    out, w, mse, mode, ph, fr = equalize_f32(x, *args)
    guard = guard_db(out, equalize_f64(x, *args))
    d_in, d_out = dispersion(x[-4096:]), dispersion(out[-4096:])
    # CMA restores the constant modulus: the dispersion of the last 4096 outputs is well below
    # that of the channel output; the tracker turns the output onto the QPSK points, so the
    # decision error power ends below -20 dB.
    assert mode == "cma" and d_out < 0.5 * d_in and float(mse) < 0.01, (d_in, d_out, float(mse))
    return dict(
        params=eq_params(
            *args[:3], mu_cma, mu_dd, NEVER, phase_bw, (mse, mode, ph, fr), guard,
            ebn0_db=ebn0_db, sigma=sigma, phase_rad=phase, channel=channel_param(chan),
            dispersion_in_last_4096=round(d_in, 6), dispersion_out_last_4096=round(d_out, 6),
        ),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", out), "weights": Array("cf32", w)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="QPSK ((2 b0 - 1) + j (2 b1 - 1)) / sqrt(2), one sample per symbol, through the symbol-spaced "
        "channel 0.1 e^(j2) z^1 + 1 + 0.35 e^(j0.8) z^-1 + 0.15 e^(-j2) z^-2 (channel: [lag, magnitude, "
        "phase]), turned by a random constant carrier phase (phase_rad), plus complex noise for Eb/N0 "
        "20 dB per transmitted symbol energy; equalize with 11 taps, no feedback, mu_cma 0.05 "
        "(normalized), cma_symbols 2^32 - 1 (CMA only, never switches) and the phase tracker at "
        "phase_bw 0.02; exact float32 model over the whole input in one call; weights are the final "
        "weights, final_mse_bits, final_mode, final_phase and final_frequency_bits the final state; "
        "the dispersion mean((|y|^2 - 1)^2) of the last 4096 inputs and outputs is information; "
        "snr_db_guard compares with the float64 loop (D111)",
    )


@case(kernel="equalize", name="dd_8psk_c32_16384", seed=591)
def dd_8psk(rng):
    n, taps, feedback, mu_cma, mu_dd, ebn0_db, phase_bw = 16384, 11, 3, 0.05, 0.02, 22.0, 0.01
    phase = 0.25
    sym, bps = symbols(rng, n, 8)
    chan = [(0, 1.0 + 0j), (1, polar(0.2, 0.7)), (2, polar(0.08, -1.9))]
    sigma = math.sqrt(1.0 / (2.0 * bps * 10.0 ** (ebn0_db / 10.0)))
    x = channel(sym, chan, phase, sigma, rng)
    args = (taps, feedback, "8psk", mu_cma, mu_dd, 0, phase_bw)
    out, w, mse, mode, ph, fr = equalize_f32(x, *args)
    guard = guard_db(out, equalize_f64(x, *args))
    # The centre tap delays by (taps - 1) / 2 symbols; after convergence every decision of the
    # second half is the transmitted symbol (the tracker removes the carrier phase).
    delay = (taps - 1) // 2
    errors = 0
    for k in range(n // 2, n):
        d = decide_f64("8psk", float(out[k].real), float(out[k].imag))
        errors += abs(complex(*d) - sym[k - delay]) > 1e-6
    assert mode == "decision" and float(mse) < 0.01 and errors == 0, (float(mse), errors)
    return dict(
        params=eq_params(
            *args[:3], mu_cma, mu_dd, 0, phase_bw, (mse, mode, ph, fr), guard,
            ebn0_db=ebn0_db, sigma=sigma, phase_rad=phase, channel=channel_param(chan), delay=delay,
            symbol_errors_second_half=errors,
        ),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", out), "weights": Array("cf32", w)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="8PSK exp(j (2 pi k / 8 + pi / 8)) with k from the Gray code b0 4 + b1 2 + b2 (M40.1), one "
        "sample per symbol, through the symbol-spaced channel 1 + 0.2 e^(j0.7) z^-1 + 0.08 e^(-j1.9) z^-2 "
        "(channel: [lag, magnitude, phase]), turned by the carrier phase 0.25 rad, plus complex noise "
        "for Eb/N0 22 dB per transmitted symbol energy; equalize with 11 forward and 3 feedback taps (a "
        "decision feedback equalizer), mu_dd 0.02 (normalized), cma_symbols 0 (decision mode from the "
        "start) and the phase tracker at phase_bw 0.01; exact float32 model over the whole input in one "
        "call; weights are the final 14 weights (11 forward, 3 feedback), final_mse_bits, final_mode, "
        "final_phase and final_frequency_bits the final state; the generator asserts that every "
        "decision of the second half is the transmitted symbol delayed by 5 (symbol_errors_second_half) "
        "and that mse ends below 0.01; snr_db_guard compares with the float64 loop (D111)",
    )
