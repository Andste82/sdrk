"""Cases for the blocks of sdrk-blocks: chains written with NumPy and SciPy.

The design of the taps goes through ``sdrk_golden.design`` (SciPy), the filtering through
``lfilter`` and ``upfirdn``; the tuned resampler has an f64 and an integer model of the
specification (32.32 position, 128 phases, linear interpolation between phases).
"""

import math

import numpy as np
import scipy.signal

from .. import design, q15, scalar_math
from ..registry import Array, case
from .dc_block import block_q15
from .mag import mag_q15_scalar

# ---------------------------------------------------------------------------------------------
# ddc


def noise(rng, n, sigma):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma


@case(kernel="blocks", name="ddc_c32_fs2m4_f600k_d10", seed=310)
def ddc_c32(rng):
    fs, freq, dec = 2.4e6, 600e3, 10
    n = 20000
    t = np.arange(n)
    x = 0.5 * scalar_math.expj(2 * np.pi * (freq + 50e3) * t / fs) + noise(rng, n, 0.05)
    x = x.astype(np.complex64)
    cutoff, transition, att = 0.4 * fs / dec, 0.1 * fs / dec, 60.0
    h = design.lowpass(cutoff, transition, att, fs).astype(np.float32).astype(np.float64)
    rotated = x.astype(np.complex128) * scalar_math.expj(-2 * np.pi * freq * t / fs)
    expected = scipy.signal.upfirdn(h, rotated, 1, dec)[: -(-n // dec)]
    return dict(
        params={
            "rate_hz": fs,
            "frequency_hz": freq,
            "decimation": dec,
            "cutoff_hz": cutoff,
            "transition_hz": transition,
            "attenuation_db": att,
            "taps": len(h),
        },
        arrays={"input": Array("cf32", x), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 100.0, "skip": 0},
        notes=(
            "rotate by exp(-j 2 pi f n / fs) in float64, taps from kaiserord + firwin(kaiser) rounded to f32, "
            "upfirdn(h, y, 1, 10) restricted to ceil(n / 10) outputs; tone at f + 50 kHz"
        ),
    )


# ---------------------------------------------------------------------------------------------
# fm_demod


@case(kernel="blocks", name="fm_demod_c32", seed=311)
def fm_demod_c32(rng):
    fs, dev, tau_us, cutoff, dec = 240e3, 75e3, 50.0, 15e3, 5
    n = 4800
    k = np.arange(n)
    audio_hz, amp, offset = 1000.0, 0.9, 5e3
    beta = dev * amp / audio_hz
    phase = 2 * np.pi * np.modf(offset * k / fs)[0] + beta * (1.0 - scalar_math.cos(2 * np.pi * np.modf(audio_hz * k / fs)[0]))
    x = scalar_math.expj(phase).astype(np.complex64)
    xd = x.astype(np.complex128)
    prev = np.concatenate([[0.0 + 0.0j], xd[:-1]])
    gain = float(np.float32(fs / (2 * np.pi * dev)))
    disc = scalar_math.angle(xd * np.conj(prev)) * gain
    alpha = float(np.float32(1.0 - math.exp(-1.0 / (fs * tau_us * 1e-6))))
    deemph = scipy.signal.lfilter([alpha], [1.0, alpha - 1.0], disc)
    h = design.lowpass(cutoff, cutoff / 4.0, 60.0, fs).astype(np.float32).astype(np.float64)
    audio = scipy.signal.upfirdn(h, deemph, 1, dec)[: (n - 1) // dec + 1]
    return dict(
        params={
            "rate_hz": fs,
            "deviation_hz": dev,
            "deemphasis_us": tau_us,
            "audio_cutoff_hz": cutoff,
            "audio_decimation": dec,
            "alpha": alpha,
            "gain": gain,
            "taps": len(h),
        },
        arrays={"input": Array("cf32", x), "expected": Array("f64", audio)},
        tolerance={"class": "tolerance", "snr_db_min": 90.0, "skip": 0},
        notes=(
            "analytic FM tone (1 kHz, 75 kHz deviation scaled by 0.9, carrier offset 5 kHz) at 240 kHz; "
            "angle(x conj(x[-1])) * gain, one-pole de-emphasis via lfilter, audio filter "
            "(kaiserord + firwin, transition cutoff / 4) and upfirdn by 5"
        ),
    )


# ---------------------------------------------------------------------------------------------
# resampler


def resampler_input(rng, n):
    """Complex noise (sigma 0.2) low-passed by firwin(63, 0.3)."""
    return scipy.signal.lfilter(scipy.signal.firwin(63, 0.3), [1.0], noise(rng, n, 0.2))


RATIONAL_N = 8000
RATIONAL_MAX_TAPS = 8192


@case(kernel="blocks", name="resampler_rational_c32_48k_to_44k1", seed=312)
def resampler_rational_c32(rng):
    fin, fout = 48000.0, 44100.0
    up, down = 147, 160
    x = resampler_input(rng, RATIONAL_N).astype(np.complex64)
    low = min(fin, fout)
    cutoff, transition, att = 0.45 * low, 0.1 * low, 60.0
    h = (up * design.lowpass(cutoff, transition, att, up * fin)).astype(np.float32).astype(np.float64)
    count = (RATIONAL_N * up - 1) // down + 1
    expected = scipy.signal.upfirdn(h, x.astype(np.complex128), up, down)[:count]
    assert len(expected) == count
    return dict(
        params={
            "rate_in_hz": fin,
            "rate_out_hz": fout,
            "up": up,
            "down": down,
            "cutoff_hz": cutoff,
            "transition_hz": transition,
            "attenuation_db": att,
            "max_taps": RATIONAL_MAX_TAPS,
            "taps": len(h),
            "outputs": count,
        },
        arrays={"input": Array("cf32", x), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 100.0, "skip": 0},
        notes="upfirdn(L * taps, x, 147, 160) with the taps designed at L * fs_in; K = floor((n L - 1) / M) + 1 outputs",
    )


TUNED_N = 6000
TUNED_RATE = 48000.0
TUNED_PPM = 125.0


def tuned_setup():
    low = TUNED_RATE
    cutoff, transition, att = 0.45 * low, 0.1 * low, 60.0
    k, bank = design.tuned_bank(cutoff, transition, att, TUNED_RATE)
    step = design.tuned_step(TUNED_RATE, TUNED_RATE, TUNED_PPM)
    return cutoff, transition, att, k, bank, step


def tuned_params(cutoff, transition, att, k, step, outputs):
    return {
        "rate_in_hz": TUNED_RATE,
        "rate_out_hz": TUNED_RATE,
        "ppm": TUNED_PPM,
        "cutoff_hz": cutoff,
        "transition_hz": transition,
        "attenuation_db": att,
        "phases": design.PHASES,
        "taps_per_phase": k,
        "step": step,
        "outputs": outputs,
    }


@case(kernel="blocks", name="resampler_tuned_c32_ppm", seed=313)
def resampler_tuned_c32(rng):
    x = resampler_input(rng, TUNED_N).astype(np.complex64)
    cutoff, transition, att, k, bank, step = tuned_setup()
    bank32 = bank.astype(np.float32).astype(np.float64)
    xd = np.concatenate([np.zeros(k - 1, dtype=np.complex128), x.astype(np.complex128)])

    def window(n, shift=0):
        # x[n + shift - i] for i = 0 .. K - 1 (zero history in front)
        idx = n + shift - np.arange(k) + (k - 1)
        return xd[idx]

    out = []
    for n, p, mu, wrap in design.tuned_positions(TUNED_N, step):
        a = scalar_math.dot(window(n), bank32[p])
        b = scalar_math.dot(window(n, 1), bank32[0]) if wrap else scalar_math.dot(window(n), bank32[p + 1])
        m = np.float32(mu / 65536.0).astype(np.float64)
        out.append(a * (1.0 - m) + b * m)
    expected = np.array(out)
    return dict(
        params=tuned_params(cutoff, transition, att, k, step, len(expected)),
        arrays={"input": Array("cf32", x), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 100.0, "skip": 0},
        notes=(
            "tuned mode of the specification in float64: 128-phase bank, 32.32 position j * step, "
            "phase = top 7 bits, mu = next 16 bits, bank[128] = bank[0] one sample later"
        ),
    )


@case(kernel="blocks", name="resampler_tuned_cq15_ppm", seed=314)
def resampler_tuned_cq15(rng):
    x = resampler_input(rng, TUNED_N)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    cutoff, transition, att, k, bank, step = tuned_setup()
    bankq = q15.from_float(bank)
    assert int(np.abs(bankq).max()) < 32768 and float(np.abs(bank).max()) <= (1 - 2.0**-15) * 1.01

    def run(v):
        vp = np.concatenate([np.zeros(k - 1, dtype=np.int64), np.asarray(v, dtype=np.int64)])

        def dot(phase, n, shift=0):
            idx = n + shift - np.arange(k) + (k - 1)
            acc = int(np.sum(bankq[phase].astype(np.int64) * vp[idx]))
            return int(q15.sat16(q15.round_shift(acc, 15)))

        res = []
        for n, p, mu, wrap in design.tuned_positions(TUNED_N, step):
            a = dot(p, n)
            b = dot(0, n, 1) if wrap else dot(p + 1, n)
            res.append(int(q15.sat16(q15.round_shift(a * (65536 - mu) + b * mu, 16))))
        return np.array(res, dtype=np.int64)

    yr, yi = run(xr), run(xi)
    return dict(
        params=tuned_params(cutoff, transition, att, k, step, len(yr)),
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("ci16", q15.pack(yr, yi))},
        tolerance={"class": "bit_exact"},
        notes=(
            "integer model of the tuned mode: Q15 bank, dot = sat16(round_shift(sum, 15)), "
            "y = sat16(round_shift(a (65536 - mu) + b mu, 16))"
        ),
    )


# ---------------------------------------------------------------------------------------------
# spectrum

FFT = 1024
BIN = 100
AVG = 4
TONE_AMP = 1.0 - 2.0**-15


def spectrum_signal(rng):
    n = FFT * AVG
    k = np.arange(n)
    return TONE_AMP * scalar_math.expj(2 * np.pi * BIN * k / FFT) + noise(rng, n, 0.01)


def spectrum_db(x):
    """Periodic Hann window, FFT, mean power over the windows, dB relative to (N * coherent gain)^2, fftshift."""
    w = 0.5 - 0.5 * scalar_math.cos(2 * np.pi * np.arange(FFT) / FFT)
    cg = w.mean()
    power = np.zeros(FFT)
    for i in range(AVG):
        seg = x[i * FFT : (i + 1) * FFT] * w
        power += np.abs(np.fft.fft(seg)) ** 2
    power /= AVG
    db = 10.0 * scalar_math.log10(np.maximum(power, 1e-20) / (FFT * cg) ** 2)
    return np.fft.fftshift(db)


def spectrum_params():
    return {"fft_size": FFT, "window": "hann", "averaging": AVG, "bin": BIN, "interval_samples": FFT}


@case(kernel="blocks", name="spectrum_c32_1024_hann_avg4", seed=315)
def spectrum_c32(rng):
    x = spectrum_signal(rng).astype(np.complex64)
    return dict(
        params=spectrum_params(),
        arrays={"input": Array("cf32", x), "expected": Array("f64", spectrum_db(x.astype(np.complex128)))},
        tolerance={"class": "max_abs_err", "max_abs_err": 0.01, "skip": 0, "min_ref_db": -100.0, "ref_unit": "db"},
        notes="tone at bin 100.0 (amplitude 1 - 2^-15) plus noise sigma 0.01; 4 windows of 1024 samples",
    )


@case(kernel="blocks", name="spectrum_cq15_1024_bfp", seed=316)
def spectrum_cq15(rng):
    x = spectrum_signal(rng)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    xq = q15.to_float(xr) + 1j * q15.to_float(xi)
    return dict(
        params=spectrum_params(),
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("f64", spectrum_db(xq))},
        tolerance={"class": "max_abs_err", "max_abs_err": 0.2, "skip": 0, "min_ref_db": -60.0, "ref_unit": "db"},
        notes="the same signal quantized to Q15; the numpy result of the quantized values, compared above -60 dB",
    )


# ---------------------------------------------------------------------------------------------
# am_demod (M20): mag -> dc_block_real -> (no audio filter at audio_decimation 1)

AM_RATE = 48000.0
AM_N = 16000
AM_AUDIO_HZ, AM_INDEX, AM_CARRIER, AM_PHASE = 1000.0, 0.5, 0.5, 0.3
AM_DC_CUTOFF_HZ = 30.0
AM_NOISE_SIGMA = 1e-3


def am_signal(rng):
    """am_tone: carrier (1 + index cos(2 pi f n / fs)) exp(j phase) plus complex noise."""
    k = np.arange(AM_N)
    cycles = (AM_AUDIO_HZ * k / AM_RATE) % 1.0
    env = AM_CARRIER * (1.0 + AM_INDEX * scalar_math.cos(2.0 * np.pi * cycles))
    return env * scalar_math.expj(np.full(AM_N, AM_PHASE)) + noise(rng, AM_N, AM_NOISE_SIGMA)


def am_params(**extra):
    return {
        "rate_hz": AM_RATE,
        "audio_hz": AM_AUDIO_HZ,
        "index": AM_INDEX,
        "carrier": AM_CARRIER,
        "phase": AM_PHASE,
        "dc_cutoff_hz": AM_DC_CUTOFF_HZ,
        "audio_cutoff_hz": 5000.0,
        "audio_decimation": 1,
        **extra,
    }


@case(kernel="blocks", name="am_demod_c32", seed=419)
def am_demod_c32(rng):
    x = am_signal(rng).astype(np.complex64)
    r = float(np.float32(scalar_math.exp(-2.0 * np.pi * AM_DC_CUTOFF_HZ / AM_RATE)))
    env = np.abs(x.astype(np.complex128))
    expected = scipy.signal.lfilter([1.0, -1.0], [1.0, -r], env)
    return dict(
        params=am_params(r=r),
        arrays={"input": Array("cf32", x), "expected": Array("f64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 90.0, "skip": 0},
        notes=(
            "am_tone (carrier 0.5, 1 kHz, index 0.5, phase 0.3) plus complex noise sigma 1e-3 per "
            "component at 48 kHz; np.abs of the f32 input, then the DC blocker "
            "y[n] = x[n] - x[n-1] + r y[n-1] via lfilter with r = exp(-2 pi 30 / fs) cast to f32; "
            "audio_decimation 1 has no audio filter (as fm_demod), so there is no upfirdn stage"
        ),
    )


@case(kernel="blocks", name="am_demod_cq15", seed=420)
def am_demod_cq15(rng):
    x = am_signal(rng)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    r = int(q15.from_float(scalar_math.exp(-2.0 * np.pi * AM_DC_CUTOFF_HZ / AM_RATE)))
    env = np.array(
        [int(q15.sat16(mag_q15_scalar(a, b))) for a, b in zip(xr.tolist(), xi.tolist())],
        dtype=np.int64,
    )
    return dict(
        params=am_params(r_raw=r),
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("i16", block_q15(env, r))},
        tolerance={"class": "bit_exact"},
        notes=(
            "the signal of am_demod_c32 quantized to Q15; integer chain: envelope "
            "sat16(round(sqrt(re^2 + im^2))) with math.isqrt, then the Q15 DC blocker "
            "sat16((x - x_prev) + round_shift(r y_prev, 15)) with r = round(exp(-2 pi 30 / fs) 32768)"
        ),
    )


# ---------------------------------------------------------------------------------------------
# am_demod with the audio filter (M20): mag -> dc_block_real -> upfirdn / resample_real

AMD_DECIMATION = 4
AMD_AUDIO_CUTOFF_HZ = 2500.0
# Samples [GAP_START, GAP_END) are exactly zero (a closed squelch): the audio of the block is
# exactly zero there and the DC blocker starts again from a cleared state.
AMD_GAP = (6000, 8000)


def am_gap_signal(rng):
    x = am_signal(rng)
    x[AMD_GAP[0] : AMD_GAP[1]] = 0.0
    return x


def am_gap_params(**extra):
    return am_params(
        **{
            "audio_cutoff_hz": AMD_AUDIO_CUTOFF_HZ,
            "audio_decimation": AMD_DECIMATION,
            "gap_start": AMD_GAP[0],
            "gap_end": AMD_GAP[1],
            **extra,
        }
    )


def audio_lowpass():
    return design.lowpass(AMD_AUDIO_CUTOFF_HZ, AMD_AUDIO_CUTOFF_HZ / 4.0, 60.0, AM_RATE)


def runs_of_nonzero(env):
    """The maximal runs ``[a, b)`` where ``env != 0``."""
    nz = np.concatenate([[False], np.asarray(env) != 0, [False]])
    edges = np.flatnonzero(nz[1:] != nz[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


@case(kernel="blocks", name="am_demod_c32_dec4_gap", seed=436)
def am_demod_c32_dec4_gap(rng):
    x = am_gap_signal(rng).astype(np.complex64)
    r = float(np.float32(scalar_math.exp(-2.0 * np.pi * AM_DC_CUTOFF_HZ / AM_RATE)))
    h = audio_lowpass().astype(np.float32).astype(np.float64)
    env = np.abs(x.astype(np.complex128))
    dc = np.zeros(AM_N)
    for a, b in runs_of_nonzero(env):  # a zero envelope sample clears the DC blocker
        dc[a:b] = scipy.signal.lfilter([1.0, -1.0], [1.0, -r], env[a:b])
    expected = scipy.signal.upfirdn(h, dc, 1, AMD_DECIMATION)[: (AM_N - 1) // AMD_DECIMATION + 1]
    return dict(
        params=am_gap_params(r=r, taps=len(h)),
        arrays={"input": Array("cf32", x), "expected": Array("f64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 90.0, "skip": 0},
        notes=(
            "the signal of am_demod_c32 with samples 6000 to 7999 set to zero; np.abs of the f32 input, "
            "the DC blocker via lfilter on every run of non-zero envelope (an envelope of exactly zero "
            "outputs zero and clears the blocker), the audio filter (kaiserord + firwin, transition "
            "cutoff / 4, 2500 Hz, taps rounded to f32) and upfirdn by 4"
        ),
    )


@case(kernel="blocks", name="am_demod_cq15_dec4_gap", seed=437)
def am_demod_cq15_dec4_gap(rng):
    x = am_gap_signal(rng)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    r = int(q15.from_float(scalar_math.exp(-2.0 * np.pi * AM_DC_CUTOFF_HZ / AM_RATE)))
    h = np.asarray(q15.from_float(audio_lowpass()), dtype=np.int64)
    assert int(np.abs(h).max()) < 32768
    env = np.array(
        [int(q15.sat16(mag_q15_scalar(a, b))) for a, b in zip(xr.tolist(), xi.tolist())],
        dtype=np.int64,
    )
    dc = np.zeros(AM_N, dtype=np.int64)
    for a, b in runs_of_nonzero(env):
        dc[a:b] = block_q15(env[a:b], r)
    # resample_real with L = 1: y[j] = sat16(round_shift(sum_i h[i] v[j M - i], 15))
    count = (AM_N - 1) // AMD_DECIMATION + 1
    padded = np.concatenate([np.zeros(len(h) - 1, dtype=np.int64), dc])
    out = np.zeros(count, dtype=np.int64)
    for j in range(count):
        window = padded[j * AMD_DECIMATION : j * AMD_DECIMATION + len(h)][::-1]
        out[j] = int(q15.sat16(q15.round_shift(int(np.sum(h * window)), 15)))
    return dict(
        params=am_gap_params(r_raw=r, taps=len(h)),
        arrays={"input": Array("ci16", q15.pack(xr, xi)), "expected": Array("i16", out)},
        tolerance={"class": "bit_exact"},
        notes=(
            "the signal of am_demod_cq15 with samples 6000 to 7999 set to zero; integer chain: envelope, "
            "Q15 DC blocker restarted after every zero envelope sample, audio taps round(h 32768), "
            "decimation by 4 with one rounding per output: sat16(round_shift(sum, 15))"
        ),
    )
