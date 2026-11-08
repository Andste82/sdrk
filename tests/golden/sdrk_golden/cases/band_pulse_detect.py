"""Cases for the block ``band_pulse_detect`` (M45.1): OOK transmissions with their exact pulse and
gap widths and frequencies.

The cases do not model the detector (its envelope, thresholds and state machines are the block's
own design, D956 to D960); they hold the exact definition the detector is measured against
(accuracy policy step 1): the times of the half-amplitude edges of every pulse and the frequency of
the carrier, written from the coding rules of M45.1 section 5 without looking at the Rust code
(``sdrk_testkit::ism``). A pulse rises and falls linearly over ``RAMP_US`` (2 us), centred on its
nominal edge, so the half-amplitude points are the nominal edges and a width is the nominal one.

Codings (all times in microseconds): PWM ``(short, long, gap)``: bit 1 a pulse of ``short``, bit 0
one of ``long``, each followed by ``gap``. PPM ``(pulse, short_gap, long_gap)``: every bit a pulse
of ``pulse``, bit 0 followed by ``short_gap``, bit 1 by ``long_gap``. Manchester ``half``: bit 1 is
on-off, bit 0 off-on, each half ``half``; adjacent halves of the same level join, the leading off
half of a first bit 0 is not sent. The last gap of a train is 0.

Layout: ``input`` the samples (``cf32``); per transmission ``starts`` the time of its first edge
in input samples (``f64``), ``freqs`` its offset from the band centre in Hz (``f64``),
``pair_counts`` its number of (pulse, gap) pairs (``i32``); ``pulses`` and ``gaps`` all pairs back
to back in steps of the detector input (``f64``). Parameters: the input rate, the lanes of the
channelizer in front (1: the input is a lane view of width 1 at the step rate), the step rate,
and the tolerances of a test: ``max_abs_err`` of every width in steps (the tolerance), the
frequency within ``max_freq_err_hz`` (spec test 3) and the start within ``max_start_err_steps``.
"""

import math

import numpy as np

from .. import scalar_math
from ..registry import Array, case

RAMP_US = 2.0


def runs_of(coding, bits):
    """(pulse, gap) pairs in microseconds of one train; the last gap is 0."""
    kind = coding[0]
    pairs = []
    if kind == "pwm":
        _, short, long_, gap = coding
        pairs = [(short if b else long_, gap) for b in bits]
    elif kind == "ppm":
        _, pulse, short_gap, long_gap = coding
        pairs = [(pulse, long_gap if b else short_gap) for b in bits]
    elif kind == "manchester":
        _, half = coding
        levels = []
        for b in bits:
            levels += [1, 0] if b else [0, 1]
        while levels and levels[0] == 0:
            levels.pop(0)
        runs = []
        for v in levels:
            if runs and runs[-1][0] == v:
                runs[-1][1] += half
            else:
                runs.append([v, half])
        if runs[-1][0] == 0:
            runs.pop()
        for k in range(0, len(runs), 2):
            gap = runs[k + 1][1] if k + 1 < len(runs) else 0.0
            pairs.append((runs[k][1], gap))
    else:
        raise ValueError(kind)
    last = pairs[-1]
    pairs[-1] = (last[0], 0.0)
    return pairs


def keyed(n, fs, start, pairs_us, amplitude):
    """The amplitude of a train at the samples ``0 .. n``: 0 or ``amplitude`` with linear edges
    of ``RAMP_US`` centred on the nominal edges (``start`` in samples)."""
    t = np.arange(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)
    ramp = RAMP_US * 1e-6 * fs
    at = start
    for pulse, gap in pairs_us:
        on, off = at, at + pulse * 1e-6 * fs
        up = np.clip((t - (on - 0.5 * ramp)) / ramp, 0.0, 1.0)
        down = np.clip(((off + 0.5 * ramp) - t) / ramp, 0.0, 1.0)
        a = np.maximum(a, np.minimum(up, down))
        at = off + gap * 1e-6 * fs
    return amplitude * a


def render(rng, fs, n, noise_rms, txs):
    """Complex Gaussian noise of ``noise_rms`` (per component ``noise_rms / sqrt 2``) plus the
    transmissions ``(start, freq, amplitude, pairs)``, each with a random carrier phase."""
    idx = np.arange(n, dtype=np.float64)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (noise_rms / math.sqrt(2.0))
    for start, freq, amp, pairs in txs:
        phi = rng.uniform(0.0, 2.0 * math.pi)
        phase = 2.0 * math.pi * np.mod(freq / fs * idx, 1.0) + phi
        x = x + keyed(n, fs, start, pairs, amp) * scalar_math.expj(phase)
    return x.astype(np.complex64)


def build(rng, fs, lanes, step_rate, plan, first_s, period_s, noise_rms, snr_db, total_s):
    """A case of the transmissions ``plan`` (coding, bits, frequency), one every ``period_s``
    from ``first_s`` on with a random sub-sample start; ``snr_db`` the level range above the noise
    in a channel spacing (``step_rate / 2``; for a single lane the whole lane)."""
    n = int(round(total_s * fs))
    band = fs if lanes == 1 else step_rate / 2.0
    noise_in_band = noise_rms**2 * band / fs
    txs, starts, freqs, counts, pulses, gaps, levels = [], [], [], [], [], [], []
    for k, (coding, bits, freq) in enumerate(plan):
        start = (first_s + period_s * k) * fs + rng.uniform(0.0, 1.0)
        snr = rng.uniform(*snr_db)
        amp = math.sqrt(noise_in_band * 10.0 ** (snr / 10.0))
        pairs = runs_of(coding, bits)
        txs.append((start, freq, amp, pairs))
        starts.append(start)
        freqs.append(freq)
        levels.append(snr)
        counts.append(len(pairs))
        pulses += [p * 1e-6 * step_rate for p, _ in pairs]
        gaps += [g * 1e-6 * step_rate for _, g in pairs]
        # The burst ends after a gap of 10 times its longest pulse (at most 10 ms): the next
        # transmission starts later.
        end = start + sum(p + g for p, g in pairs) * 1e-6 * fs
        quiet = (min(10e-3, 10e-6 * max(p for p, _ in pairs)) + 0.5e-3) * fs
        assert end + quiet < (first_s + period_s * (k + 1)) * fs
        assert end + quiet < n
    x = render(rng, fs, n, noise_rms, txs)
    arrays = {
        "input": Array("cf32", x),
        "starts": Array("f64", np.array(starts, dtype=np.float64)),
        "freqs": Array("f64", np.array(freqs, dtype=np.float64)),
        "levels_db": Array("f64", np.array(levels, dtype=np.float64)),
        "pair_counts": Array("i32", np.array(counts, dtype=np.int32)),
        "pulses": Array("f64", np.array(pulses, dtype=np.float64)),
        "gaps": Array("f64", np.array(gaps, dtype=np.float64)),
    }
    return arrays


def bits(rng, n):
    return [bool(b) for b in rng.integers(0, 2, size=n).tolist()]


@case("band_pulse_detect", "ook_lane", 629)
def ook_lane(rng):
    fs = 250e3
    codings = [
        ("pwm", 200.0, 400.0, 200.0),
        ("ppm", 200.0, 400.0, 800.0),
        ("manchester", 250.0),
        ("pwm", 100.0, 300.0, 150.0),
        ("ppm", 120.0, 240.0, 480.0),
        ("manchester", 400.0),
    ]
    plan = [(c, bits(rng, 16), rng.uniform(-40e3, 40e3)) for c in codings]
    arrays = build(rng, fs, 1, fs, plan, 0.010, 0.026, 1e-3, (30.0, 36.0), 0.010 + 0.026 * 6)
    return dict(
        params={
            "fs": fs,
            "lanes": 1,
            "step_rate": fs,
            "noise_rms": 1e-3,
            "transmissions": len(plan),
            "max_freq_err_hz": 10e3,
            "max_start_err_steps": 2.0,
        },
        arrays=arrays,
        tolerance={"class": "max_abs_err", "max_abs_err": 1.0, "skip": 0},
        notes="one lane at 250 kHz (a lane view of width 1): 6 OOK trains of 16 bits (PWM "
        "200/400/200 and 100/300/150 us, PPM 200/400/800 and 120/240/480 us, Manchester 250 and "
        "400 us halves), one every 26 ms from 10 ms on, offsets uniform within +-40 kHz, 30 to 36 dB "
        "above the noise of the lane; 2 us linear edges centred on the nominal edges; the exact "
        "widths in steps and every width within 1 step",
    )


@case("band_pulse_detect", "ook_band", 630)
def ook_band(rng):
    fs = 2e6
    lanes = 16
    step = 2.0 * fs / lanes
    spacing = fs / lanes
    pwm = ("pwm", 100.0, 200.0, 100.0)
    # A channel boundary near the centre and one off it, the two band edges of the 433 MHz band
    # (433.05 and 434.79 MHz around 433.92 MHz).
    freqs = [0.5 * spacing, -3.5 * spacing, -870e3, 870e3]
    plan = [(pwm, bits(rng, 10), f) for f in freqs]
    arrays = build(rng, fs, lanes, step, plan, 0.006, 0.0058, 0.0316, (30.0, 36.0), 0.006 + 0.0058 * 4)
    return dict(
        params={
            "fs": fs,
            "lanes": lanes,
            "taps_per_branch": 8,
            "step_rate": step,
            "noise_rms": 0.0316,
            "transmissions": len(plan),
            "max_freq_err_hz": 10e3,
            "max_start_err_steps": 12.0,
        },
        arrays=arrays,
        tolerance={"class": "max_abs_err", "max_abs_err": 1.0, "skip": 0},
        notes="2 MHz around the band centre (the 433 MHz flow: 16 channels of 125 kHz, the "
        "channelize defaults, 250 kHz steps), band noise of -30 dBFS: 4 PWM trains of 10 bits "
        "(100/200/100 us) one every 5.8 ms from 6 ms on, at +62.5 kHz and -437.5 kHz (channel "
        "boundaries) and -870 / +870 kHz (the band edges), 30 to 36 dB above the noise "
        "in a channel spacing; the exact widths in steps and every width within 1 step; the start "
        "within 12 steps (the delay of the channelizer is 8 steps)",
    )
