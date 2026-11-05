"""Cases for the ssb_mod block (M35 section 3): the Weaver chain of ssb_demod mirrored.

Per sample the block computes ``interleave(re = x)`` (the real audio as a complex signal),
``rotate`` by ``-s c`` (NCO 1), the complex low-pass ``fir`` with cutoff ``b`` and the taps
multiplied by 2 before quantization, and ``rotate`` by ``+s c`` (NCO 2), with
``c = (low_hz + high_hz) / 2``, ``b = (high_hz - low_hz) / 2`` and ``s = +1`` (USB) or ``-1``
(LSB). Both oscillators run at the audio rate and start at phase 0.

The sideband flips the sign of both oscillators, as in ``ssb_demod`` (decision D431): with NCO 1
fixed at ``-c`` an LSB tone at ``f`` would come out at ``f - 2 c`` instead of ``-f``, and the LSB
output is the complex conjugate of the USB output. The block works in pieces of
256 samples, which changes nothing in the result (the oscillator phases and the filter history
continue).

C32 cases: the chain in float64 (the exact modular 32-bit phase of each NCO through ``exp``,
``upfirdn`` with the taps rounded to f32), ``tolerance`` with an SNR floor of 90 dB. CQ15 case:
the integer chain (the rotate table model of ``blocks.py``, the integer convolution of the fir
kernel with one rounding and saturation per output), ``bit_exact``. The tap design is the one of
``ssb_demod`` (``design.lowpass``: kaiserord and a Kaiser firwin).
"""

import math

import numpy as np
import scipy.signal

from .. import design, q15, scalar_math
from ..registry import Array, case
from .blocks import decimate_q15, nco_increment, nco_rotation, rotate_q15

RATE_HZ = 48000
N = 8192
LOW_HZ, HIGH_HZ = 300.0, 3000.0
TRANSITION_HZ, ATTENUATION_DB = 200.0, 60.0
MAX_TAPS = 1023
TAP_GAIN = 2.0
TONES = ((700, 0.3), (1900, 0.2))
# The checks of the generator average over a whole number of periods of every tone (480 samples
# hold 7 periods of 700 Hz and 19 of 1900 Hz) after the filter has settled.
CHECK_START, CHECK_PERIOD, CHECK_PERIODS = 960, 480, 15


def audio():
    """Two real cosine tones (700 Hz 0.3, 1900 Hz 0.2) from phase 0, exactly reduced phases."""
    out = np.zeros(N)
    for f, a in TONES:
        out += np.array([a * math.cos(2.0 * math.pi * ((f * k) % RATE_HZ) / RATE_HZ) for k in range(N)])
    return out


def setup(sign):
    """Taps (float64, times 2) and the phase increments of NCO 1 and NCO 2."""
    centre = (LOW_HZ + HIGH_HZ) / 2.0
    half = (HIGH_HZ - LOW_HZ) / 2.0
    h = TAP_GAIN * design.lowpass(half, TRANSITION_HZ, ATTENUATION_DB, RATE_HZ)
    assert len(h) <= MAX_TAPS
    return h, nco_increment(-sign * centre, RATE_HZ), nco_increment(sign * centre, RATE_HZ)


def params(sideband, h, inc_1, inc_2):
    return {
        "rate_hz": RATE_HZ,
        "sideband": sideband,
        "low_hz": LOW_HZ,
        "high_hz": HIGH_HZ,
        "transition_hz": TRANSITION_HZ,
        "attenuation_db": ATTENUATION_DB,
        "max_taps": MAX_TAPS,
        "taps": len(h),
        "delay": (len(h) - 1) // 2,
        "tap_gain": TAP_GAIN,
        "inc_1": inc_1,
        "inc_2": inc_2,
        "tone_hz": [f for f, _ in TONES],
        "tone_amplitude": [a for _, a in TONES],
    }


def check_sidebands(y, sign):
    """Every tone comes out at ``s f`` with its amplitude (1 %), the mirror at ``-s f`` is 50 dB lower."""
    k = np.arange(CHECK_START, CHECK_START + CHECK_PERIOD * CHECK_PERIODS)
    seg = np.asarray(y)[k]
    for f, a in TONES:
        wanted = abs(np.mean(seg * scalar_math.expj(-sign * 2.0 * np.pi * ((f * k) % RATE_HZ) / RATE_HZ)))
        mirror = abs(np.mean(seg * scalar_math.expj(sign * 2.0 * np.pi * ((f * k) % RATE_HZ) / RATE_HZ)))
        assert abs(wanted - a) < 0.01 * a, (f, wanted)
        assert mirror < a * 10.0 ** (-50.0 / 20.0), (f, mirror)


def chain_c32(x, sign):
    """The f64 chain on the f32 input; returns the cf64 output, the taps and the increments."""
    h, inc_1, inc_2 = setup(sign)
    h32 = h.astype(np.float32).astype(np.float64)
    shifted = x.astype(np.float64) * nco_rotation(N, inc_1)
    filtered = scipy.signal.upfirdn(h32, shifted, 1, 1)[:N]
    return filtered * nco_rotation(N, inc_2), h, inc_1, inc_2


def c32_case(sign, sideband, notes):
    x = audio().astype(np.float32)
    y, h, inc_1, inc_2 = chain_c32(x, sign)
    check_sidebands(y, sign)
    return dict(
        params=params(sideband, h, inc_1, inc_2),
        arrays={"input": Array("f32", x), "expected": Array("cf64", y)},
        tolerance={"class": "tolerance", "snr_db_min": 90.0, "skip": 0},
        notes=notes,
    )


@case(kernel="blocks", name="ssb_mod_usb_c32", seed=481)
def ssb_mod_usb_c32(rng):
    return c32_case(
        +1,
        "usb",
        "48 kHz; real tones 700 Hz 0.3 and 1900 Hz 0.2 (cosines from phase 0), defaults (band 300 to "
        "3000 Hz, transition 200 Hz, 60 dB, max_taps 1023); float64 chain: x times exp(j 2 pi phase_1 / "
        "2^32) (NCO 1 at -1650 Hz, exact modular 32-bit phase from 0), upfirdn(h, ., 1, 1) restricted to "
        "n outputs with h = 2 firwin(kaiser) rounded to f32 (cutoff 1350 Hz), times exp(j 2 pi phase_2 / "
        "2^32) (NCO 2 at +1650 Hz); the tones come out at +700 and +1900 Hz",
    )


@case(kernel="blocks", name="ssb_mod_lsb_c32", seed=481)
def ssb_mod_lsb_c32(rng):
    return c32_case(
        -1,
        "lsb",
        "the input and the chain of ssb_mod_usb_c32 with sideband lsb: NCO 1 at +1650 Hz and NCO 2 at "
        "-1650 Hz (both signs flip, D431), so the tones come out at -700 and -1900 Hz",
    )


@case(kernel="blocks", name="ssb_mod_usb_cq15", seed=482)
def ssb_mod_usb_cq15(rng):
    xr = q15.from_float(audio())
    h, inc_1, inc_2 = setup(+1)
    hq = np.asarray(q15.from_float(h), dtype=np.int64)
    assert int(np.abs(hq).max()) < 32768 and float(np.abs(h).max()) <= (1 - 2.0**-15) * 1.01
    rr, ri = rotate_q15(xr, np.zeros(N, dtype=np.int64), inc_1)
    yr, yi = decimate_q15(rr, hq, 1), decimate_q15(ri, hq, 1)
    zr, zi = rotate_q15(yr, yi, inc_2)
    check_sidebands(q15.to_float(zr) + 1j * q15.to_float(zi), +1)
    return dict(
        params=params("usb", h, inc_1, inc_2),
        arrays={"input": Array("i16", xr), "expected": Array("ci16", q15.pack(zr, zi))},
        tolerance={"class": "bit_exact"},
        notes="the input of ssb_mod_usb_c32 quantized to Q15, sideband usb; integer chain: (x, 0), rotate "
        "(Q15 cosine table of 4097 entries, linear interpolation with a 15-bit fraction, complex Q15 "
        "multiply), fir (taps round(2 h 32768), int64 sum, one rounding and saturation per output), "
        "rotate by NCO 2",
    )
