"""Cases for the blocks fsk_mod and fsk_demod (M39 sections 2 and 3).

``blocks/fsk_mod_gfsk_c32`` (seed 530): the chain of ``fsk_mod`` in float64: the soft bits
mapped to the levels +-1, shaped by the pulse of the block (the boxcar of ``sps`` ones convolved
with ``design.gaussian(bt, sps, span)``, scaled to sum ``sps`` in float64 and rounded once to
float32, as the block stores its taps, D576) with ``upfirdn(h, levels, sps, 1)`` restricted to
``sps`` outputs per symbol, then the integer phase of ``fm_mod`` (``floor(x k + 0.5)`` with
``k = deviation / rate 2^32``, summed modulo 2^32 from phase 0) and ``exp(j 2 pi phase / 2^32)``
one value at a time.

``blocks/fsk_demod_2fsk_f32`` (seed 531): a continuous-phase 2-FSK signal (rectangular
frequency pulses, h = 1) with clock drift and complex noise; the chain of ``fsk_demod`` in
float64 up to the timing loop: the discriminator ``arg(x[n] conj(x[n - 1]))`` (``x[-1] = 0``)
times the float32 gain ``fs / (2 pi deviation)``, the boxcar of ``round(sps)`` float32 taps
``1 / round(sps)`` (zero history), then the float32 model of ``symbol_sync`` of M38 (the input
rounded once to float32, the gains of D565 with the ``ted_gain`` of ``fsk_demod``, D571).
``quad_demod`` and ``resample_real`` are tolerance kernels, so the block is compared with an SNR
floor instead of bit for bit.

Every value that feeds a committed byte is computed one value at a time with the C library
(``math``) and plain Python sums, never with NumPy transcendentals, BLAS or SIMD reductions, so
the files are identical on every CPU.
"""

import bisect
import math

import numpy as np
import scipy.signal

from .. import design, scalar_math
from ..registry import Array, case
from . import symbol_sync as ss

TWO_POW_32 = 4294967296.0


# ---------------------------------------------------------------------------------------------
# fsk_mod


def fsk_mod_taps(bt, sps, span):
    """The pulse of ``fsk_mod``: ``c[k]`` = the sum of the Gaussian taps ``k - sps + 1 ..= k``
    (left to right), scaled by ``sps / sum c`` in float64, rounded once to float32."""
    g = design.gaussian(bt, sps, span).tolist()
    n = (span + 1) * sps
    conv = []
    for k in range(n):
        c = 0.0
        for j in range(max(0, k - sps + 1), min(k, len(g) - 1) + 1):
            c += g[j]
        conv.append(c)
    total = 0.0
    for c in conv:
        total += c
    scale = sps / total
    return np.array([c * scale for c in conv], dtype=np.float64).astype(np.float32)


def fm_phase_signal(x, k):
    """``exp(j 2 pi phase / 2^32)`` with ``phase = sum floor(x k + 0.5)`` modulo 2^32."""
    phase = 0
    out = np.empty(len(x), dtype=np.complex128)
    for i, v in enumerate(x.tolist()):
        phase = (phase + int(math.floor(v * k + 0.5))) % (1 << 32)
        t = 2.0 * math.pi * phase / TWO_POW_32
        out[i] = complex(math.cos(t), math.sin(t))
    return out


@case(kernel="blocks", name="fsk_mod_gfsk_c32", seed=530)
def fsk_mod_gfsk_c32(rng):
    bits_n, sps, bt, span = 1024, 8, 0.5, 4
    bit_rate, deviation = 9600.0, 2400.0
    rate_out = bit_rate * sps
    bits = rng.integers(0, 2, bits_n)
    soft = np.where(bits == 1, 1.0, -1.0).astype(np.float32)
    levels = np.where(soft > 0.0, 1.0, -1.0)
    h = fsk_mod_taps(bt, sps, span)
    shaped = scipy.signal.upfirdn(h.astype(np.float64), levels, sps, 1)[: bits_n * sps]
    k = deviation / rate_out * TWO_POW_32
    y = fm_phase_signal(shaped, k)
    # A long run of equal bits reaches the deviation and no more (up to the float32 rounding
    # of the taps): the phase turns by at most 2 pi deviation / rate per sample.
    steps = scalar_math.angle(y[1:] * np.conj(y[:-1])) * rate_out / (2.0 * math.pi)
    peak = float(np.max(np.abs(steps)))
    assert abs(peak - deviation) < deviation * 1e-6, peak
    return dict(
        params={
            "levels": 2,
            "samples_per_symbol": sps,
            "bit_rate_hz": bit_rate,
            "rate_out_hz": rate_out,
            "deviation_hz": deviation,
            "bt": bt,
            "span": span,
            "taps": len(h),
            "k": k,
            "outputs": bits_n * sps,
        },
        arrays={
            "input": Array("f32", soft),
            "taps": Array("f32", h),
            "expected": Array("cf64", y),
        },
        tolerance={"class": "tolerance", "snr_db_min": 90.0, "skip": 0},
        notes="GMSK: 1024 random soft bits +-1 at 9600 bit/s, levels 2, sps 8, bt 0.5, span 4, "
        "deviation 2400 Hz (symbol rate / 4); float64 chain: level +1 for a bit above 0 and -1 "
        "otherwise, upfirdn(h, levels, 8, 1) restricted to 8 outputs per bit with h = the boxcar of "
        "8 ones convolved with design.gaussian(0.5, 8, 4), scaled to sum 8 and rounded to f32 (the "
        "taps of the block, D576), then phase += floor(x k + 0.5) modulo 2^32 with k = 2400 / 76800 "
        "2^32 from phase 0 and exp(j 2 pi phase / 2^32) with math.cos and math.sin",
    )


# ---------------------------------------------------------------------------------------------
# fsk_demod

# The Gardner slope per symbol of the boxcar-filtered 2-FSK discriminator output, the ted_gain
# that fsk_demod gives its symbol_sync (TED_GAIN_2FSK, D571).
TED_GAIN_2FSK = 1.48


def cpfsk(n, data, sps, offset, delta, cycles_per_sample):
    """Continuous-phase FSK with rectangular frequency pulses at the receive samples ``0 .. n``.

    Symbol ``j`` (level ``2 data[j] - 1``) covers the transmit time ``[offset + (j - 1) sps,
    offset + j sps)`` (symbol 0 also everything before it); receive sample ``m`` is the transmit
    position ``m delta`` (clock drift, D540). The phase in cycles is ``cycles_per_sample`` times
    the integral of the level from the transmit time 0 (the whole symbols as a running sum),
    reduced modulo 1 cycle before ``math.cos`` and ``math.sin``.
    """
    levels = [2.0 * b - 1.0 for b in data]
    out = []
    j, start, acc_whole = 0, 0.0, 0.0
    for m in range(n):
        t = m * delta
        while t >= offset + j * sps:
            end = offset + j * sps
            acc_whole += levels[j] * (end - start)
            start = end
            j += 1
        acc = acc_whole + levels[j] * (t - start)
        cyc = math.fmod(acc * cycles_per_sample, 1.0)
        out.append(complex(math.cos(2.0 * math.pi * cyc), math.sin(2.0 * math.pi * cyc)))
    return out


def discriminate(x, gain):
    """``arg(x[n] conj(x[n - 1])) gain`` with ``x[-1] = 0``, per value in float64."""
    out = []
    pr, pi = 0.0, 0.0
    for v in x:
        re, im = float(v.real), float(v.imag)
        # x conj(prev) = (re pr + im pi) + j (im pr - re pi)
        out.append(math.atan2(im * pr - re * pi, re * pr + im * pi) * gain)
        pr, pi = re, im
    return out


def boxcar(x, tap, n):
    """``y[m] = sum_{k < n} tap x[m - k]`` (zero history), summed from ``k = 0`` up."""
    out = []
    for m in range(len(x)):
        acc = 0.0
        for k in range(n):
            if m - k < 0:
                break
            acc += tap * x[m - k]
        out.append(acc)
    return out


def sps_of(fs, symbol_rate_hz):
    """``sps`` as ``SymbolSyncConfig`` derives it: the symbol rate on the 1/1000 Hz grid, the
    reduced fraction ``1000 fs / mhz`` as ``den / num`` in float64."""
    mhz = math.floor(symbol_rate_hz * 1000.0 + 0.5)
    num, den = mhz, 1000 * fs
    g = math.gcd(num, den)
    return float(den // g) / float(num // g)


@case(kernel="blocks", name="fsk_demod_2fsk_f32", seed=531)
def fsk_demod_2fsk_f32(rng):
    n, fs, symbol_rate, deviation = 16384, 48000, 4700.0, 2350.0
    ppm, ebn0_db = 50.0, 20.0
    sps = sps_of(fs, symbol_rate)
    delta = 1.0 + ppm * 1e-6
    offset = float(rng.uniform(0.0, sps))
    count = int(n * delta / sps) + 4
    data = rng.integers(0, 2, count).tolist()
    clean = cpfsk(n, data, sps, offset, delta, deviation / fs)
    # Unit power per sample, one bit per symbol: sigma^2 = sps / (2 Eb/N0) per component.
    sigma = math.sqrt(sps / (2.0 * 10.0 ** (ebn0_db / 10.0)))
    nre, nim = rng.standard_normal(n).tolist(), rng.standard_normal(n).tolist()
    x = ss.complex64([complex(v.real + sigma * a, v.imag + sigma * b) for v, a, b in zip(clean, nre, nim)])

    # The float64 chain up to the timing loop.
    taps = round(sps)
    gain = float(np.float32(fs / (2.0 * math.pi * deviation)))
    tap = float(np.float32(1.0 / taps))
    disc = discriminate(x.astype(np.complex128).tolist(), gain)
    filt = boxcar(disc, tap, taps)

    # The float32 model of symbol_sync (and the float64 loop for the guard of D111).
    kp, ki, lim = ss.gains(ss.LOOP_BW, ss.DAMPING, ss.MAX_DEVIATION, TED_GAIN_2FSK, sps)
    comps32 = [[ss.fm.f32(v) for v in filt]]
    out, positions = ss.symbol_sync(ss.F32Ops, comps32, "gardner", sps, kp, ki, lim)
    out64, _ = ss.symbol_sync(ss.F64Ops, [list(filt)], "gardner", sps, kp, ki, lim)
    expected = np.array([y[0] for y in out], dtype=np.float32)

    # The window of the boxcar (the discriminator samples m - taps + 1 ..= m span the time
    # (m - taps, m]) lies inside symbol j for m from its end - (sps - taps) to its end; the
    # middle of that plateau, received at the position divided by delta, is the symbol peak.
    sym = [(offset + j * sps - (sps - taps) / 2.0) / delta for j in range(count)]
    err = ss.timing_error(positions, sym, sps)
    assert err < 0.1, err
    # After the lock every hard decision is the transmitted bit (Eb/N0 20 dB).
    errors = 0
    for p, v in list(zip(positions, out))[200:]:
        j = bisect.bisect_left(sym, p)
        if j > 0 and (j == count or p - sym[j - 1] < sym[j] - p):
            j -= 1
        errors += int((float(v[0]) > 0.0) != (data[j] == 1))
    assert errors == 0, errors
    m = min(len(out), len(out64))
    num = math.fsum(float(y[0]) ** 2 for y in out64[:m])
    den = math.fsum((float(a[0]) - float(b[0])) ** 2 for a, b in zip(out64[:m], out[:m]))
    snr_f64_loop = math.floor(10.0 * math.log10(num / den) * 10.0) / 10.0
    return dict(
        params={
            "levels": 2,
            "rate_hz": fs,
            "symbol_rate_hz": symbol_rate,
            "deviation_hz": deviation,
            "loop_bw": ss.LOOP_BW,
            "sps": sps,
            "sps_bits": ss.bits(sps),
            "taps": taps,
            "ted_gain": TED_GAIN_2FSK,
            "kp_bits": ss.bits(kp),
            "ki_bits": ss.bits(ki),
            "max_integ_bits": ss.bits(lim),
            "drift_ppm": ppm,
            "ebn0_db": ebn0_db,
            "offset": offset,
            "sigma": sigma,
            "produced": len(expected),
            "timing_error_last_256": round(err, 6),
            "snr_db_f64_loop": snr_f64_loop,
        },
        arrays={"input": Array("cf32", x), "expected": Array("f32", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 80.0, "skip": 0},
        model="float32",
        notes="continuous-phase 2-FSK (h = 1, rectangular frequency pulses, level 2 b - 1) at 48 kHz, "
        "4700 Bd (sps 480 / 47), deviation 2350 Hz, symbol j over the transmit time [offset + (j - 1) "
        "sps, offset + j sps), 50 ppm clock drift (receive sample n is the transmit position n (1 + "
        "50e-6), D540), unit amplitude plus complex noise for Eb/N0 20 dB (sigma^2 = sps / (2 Eb/N0) "
        "per component); float64 chain: atan2(x[n] conj(x[n - 1])) (x[-1] = 0) times the f32 gain fs "
        "/ (2 pi 2350), the boxcar of 10 f32 taps 1 / 10 (zero history), rounded once to f32, then the "
        "float32 model of symbol_sync (cases/symbol_sync.py; Gardner, loop_bw 0.01, damping 0.7071, "
        "max_deviation 0.005, gains of D565 with ted_gain 1.48 = TED_GAIN_2FSK of D571) over the whole "
        "input in one call; the 2-FSK soft bit is the symbol value; produced is the output count; "
        "timing_error_last_256 is the largest distance of the last 256 strobes from a symbol peak in "
        "symbols (information); snr_db_f64_loop is the SNR of the model against the same loop in "
        "float64 on the float64 chain (information, D34)",
    )
