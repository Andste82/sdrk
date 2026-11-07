"""Cases for the filter design of sdrk-blocks (``scipy.signal.firwin``, ``kaiserord``)."""

import math

import numpy as np
import scipy.signal

from .. import design as sdrk_design
from .. import scalar_math
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
    width = scalar_math.geomspace(0.01, 0.2, 20)
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


# Biquad design (M22)

BIQUAD_FS = 1.0


def _register_biquad(response, q):
    @case(kernel="design", name=f"biquad_{response}", seed=0)
    def _case(rng):
        coefs = sdrk_design.biquad_coefs(response, 0.1, q, BIQUAD_FS)
        return dict(
            params={"response": response, "f0_hz": 0.1, "q": q, "fs_hz": BIQUAD_FS},
            arrays={"expected": Array("f64", np.array(coefs))},
            tolerance={"class": "max_abs_err", "max_abs_err": 1e-14, "skip": 0},
            notes="RBJ Audio EQ Cookbook section [b0, b1, b2, a1, a2] / a0 in float64",
        )


for _resp, _q in (("lowpass", 0.7071), ("highpass", 0.7071), ("bandpass", 2.0), ("notch", 10.0)):
    _register_biquad(_resp, _q)


def _register_butterworth(tag, response, order, cutoff):
    @case(kernel="design", name=f"butterworth_{tag}", seed=0)
    def _case(rng):
        rows = sdrk_design.butterworth_sos(response, order, cutoff, BIQUAD_FS)
        # The cascade must have the response of scipy.signal.butter (bilinear transform
        # prewarped at the cutoff).
        btype = "low" if response == "lowpass" else "high"
        ref = scipy.signal.butter(order, cutoff, btype=btype, fs=BIQUAD_FS, output="sos")
        _, h_mine = scipy.signal.sosfreqz(sdrk_design.sos_matrix(rows), worN=512)
        _, h_ref = scipy.signal.sosfreqz(ref, worN=512)
        assert np.max(np.abs(h_mine - h_ref)) <= 1e-9, np.max(np.abs(h_mine - h_ref))
        return dict(
            params={"response": response, "order": order, "cutoff_hz": cutoff, "fs_hz": BIQUAD_FS},
            arrays={"expected": Array("f64", np.array(rows).reshape(-1))},
            tolerance={"class": "max_abs_err", "max_abs_err": 1e-14, "skip": 0},
            notes=(
                "order/2 cookbook sections with Q_k = 1 / (2 cos((2k - 1) pi / (2 order))), lowest Q "
                "first, 5 values per section; sosfreqz equals that of scipy.signal.butter(output='sos') "
                "within 1e-9 at 512 frequencies (asserted by the generator)"
            ),
        )


for _tag, _resp, _order, _cut in (
    ("lp2", "lowpass", 2, 0.05),
    ("lp4", "lowpass", 4, 0.05),
    ("lp8", "lowpass", 8, 0.05),
    ("hp6", "highpass", 6, 0.2),
):
    _register_butterworth(_tag, _resp, _order, _cut)


# Halfband design (M23)


def _register_halfband(n):
    @case(kernel="design", name=f"halfband_n{n}", seed=0)
    def _case(rng):
        beta = float(scipy.signal.kaiser_beta(60.0))
        taps = sdrk_design.halfband_taps(n, beta)
        return dict(
            params={"num_taps": n, "attenuation_db": 60.0, "beta": beta, "fs_hz": 1.0},
            arrays={"expected": Array("f64", taps)},
            tolerance={"class": "max_abs_err", "max_abs_err": 1e-12, "skip": 0},
            notes=(
                "scipy.signal.firwin(N, 0.25, window=('kaiser', beta), fs=1) with beta = kaiser_beta(60), "
                "every tap at an even offset from the centre set to exactly 0, then scaled to sum 1"
            ),
        )


for _n in (11, 23, 63):
    _register_halfband(_n)


@case(kernel="design", name="halfband_len_table", seed=0)
def halfband_len_table(rng):
    attenuation = np.linspace(40.0, 100.0, 10)
    transition = scalar_math.geomspace(0.02, 0.2, 10)
    lens = np.array([sdrk_design.halfband_len(t, a, 1.0) for a, t in zip(attenuation, transition)], dtype=np.int32)
    assert all(int(n) % 4 == 3 for n in lens)
    return dict(
        params={"pairs": 10, "fs_hz": 1.0},
        arrays={
            "attenuation_db": Array("f64", attenuation),
            "transition_hz": Array("f64", transition),
            "expected": Array("i32", lens),
        },
        tolerance={"class": "bit_exact"},
        notes=(
            "scipy.signal.kaiserord(attenuation, 2 transition / fs), the tap count rounded up to the next "
            "N with N mod 4 == 3; attenuation 40 to 100 dB (linspace), transition 0.02 to 0.2 (geomspace)"
        ),
    )


# Pre-emphasis design (M25)


def _register_preemphasis(tau_us, fs_hz, tag):
    @case(kernel="design", name=f"preemphasis_{tag}", seed=0)
    def _case(rng):
        tau = tau_us * 1e-6
        coefs = sdrk_design.preemphasis_coefs(tau, float(fs_hz))
        # SciPy is the reference: H(s) = (1 + s tau) / (1 + s tau_p) mapped by the bilinear
        # transform without prewarping. The stored values are the formulas in plain floats.
        tau_p = 1.0 / (2.0 * math.pi * (fs_hz / 2.4))
        b, a = scipy.signal.bilinear([tau, 1.0], [tau_p, 1.0], fs=float(fs_hz))
        assert np.max(np.abs(np.array([b[0], b[1], a[1]]) - np.array(coefs))) <= 1e-12
        assert abs(a[0] - 1.0) <= 1e-15
        return dict(
            params={"tau_s": tau, "fs_hz": float(fs_hz)},
            arrays={"expected": Array("f64", np.array(coefs))},
            tolerance={"class": "max_abs_err", "max_abs_err": 1e-14, "skip": 0},
            notes="[b0, b1, a1] of (1 + s tau) / (1 + s tau_p), tau_p = 1 / (2 pi fs / 2.4), bilinear "
            "transform without prewarping (K = 2 fs); formulas in float64, equal to "
            "scipy.signal.bilinear within 1e-12",
        )


for _tau_us, _fs, _tag in (
    (50, 48000, "50us_48k"),
    (75, 48000, "75us_48k"),
    (530, 48000, "530us_48k"),
    (50, 240000, "50us_240k"),
):
    _register_preemphasis(_tau_us, _fs, _tag)


# Gaussian pulse of GFSK and GMSK (M39 §1)


@case(kernel="design", name="gaussian_bt0p5_sps8_span4", seed=0)
def _gaussian_bt0p5_sps8_span4(rng):
    bt, sps, span = 0.5, 8, 4
    return dict(
        params={"bt": bt, "sps": sps, "span": span, "num_taps": span * sps + 1},
        arrays={"expected": Array("f64", sdrk_design.gaussian(bt, sps, span))},
        tolerance={"class": "max_abs_err", "max_abs_err": 1e-12, "skip": 0},
        notes="exp(-(2 pi^2 bt^2 / ln 2) (t/T)^2) with t/T = (n - (N - 1) / 2) / sps, "
        "N = span sps + 1, normalised to sum 1; float64",
    )


# Root-raised-cosine design (M40)

# The singular points |t| = 1 / (4 rolloff) are detected with this tolerance on
# |1 - (4 rolloff t)^2|, the rule of design::rrc.
RRC_SINGULAR = 1e-9


def rrc_taps(rolloff, sps, span):
    """The root-raised-cosine taps, one tap at a time with the scalar math module, unit energy."""
    n = span * sps + 1
    b = rolloff
    raw = []
    for k in range(n):
        t = (k - (n - 1) / 2.0) / sps
        if t == 0.0:
            h = 1.0 - b + 4.0 * b / math.pi
        elif abs(1.0 - (4.0 * b * t) ** 2) < RRC_SINGULAR:
            a = math.pi / (4.0 * b)
            h = b / math.sqrt(2.0) * ((1.0 + 2.0 / math.pi) * math.sin(a) + (1.0 - 2.0 / math.pi) * math.cos(a))
        else:
            h = (math.sin(math.pi * t * (1.0 - b)) + 4.0 * b * t * math.cos(math.pi * t * (1.0 + b))) / (
                math.pi * t * (1.0 - (4.0 * b * t) ** 2)
            )
        raw.append(h)
    norm = math.sqrt(math.fsum(h * h for h in raw))
    return np.array([h / norm for h in raw], dtype=np.float64)


def _register_rrc(tag, rolloff, sps, span, singular):
    @case(kernel="design", name=f"rrc_{tag}_sps{sps}_span{span}", seed=0)
    def _case(rng):
        taps = rrc_taps(rolloff, sps, span)
        # The number of taps on the singular points |t| = 1 / (4 rolloff); the case list states it.
        hits = sum(
            1
            for k in range(span * sps + 1)
            if (k - span * sps / 2.0) != 0.0
            and abs(1.0 - (4.0 * rolloff * (k - span * sps / 2.0) / sps) ** 2) < RRC_SINGULAR
        )
        assert hits == singular, (tag, hits)
        return dict(
            params={"rolloff": rolloff, "samples_per_symbol": sps, "span": span, "num_taps": span * sps + 1,
                    "singular_taps": singular},
            arrays={"expected": Array("f64", taps)},
            tolerance={"class": "max_abs_err", "max_abs_err": 1e-12, "skip": 0},
            notes=(
                "root-raised-cosine impulse response, t = (k - span sps / 2) / sps symbols, "
                "h = (sin(pi t (1 - b)) + 4 b t cos(pi t (1 + b))) / (pi t (1 - (4 b t)^2)), closed forms "
                "h(0) = 1 - b + 4 b / pi and, where |1 - (4 b t)^2| < 1e-9, "
                "b / sqrt(2) ((1 + 2 / pi) sin(pi / (4 b)) + (1 - 2 / pi) cos(pi / (4 b))); "
                "normalized to sum h^2 = 1; computed tap by tap with the math module"
            ),
        )


for _tag, _b, _sps, _span, _sing in (("0p35", 0.35, 4, 10, 0), ("0p5", 0.5, 8, 8, 2), ("1p0", 1.0, 10, 6, 0)):
    _register_rrc(_tag, _b, _sps, _span, _sing)


# Channelizer prototype (M45)


def _register_channelizer(channels, taps_per_branch, passband, stopband, tag):
    @case(kernel="design", name=f"channelizer_m{channels}_l{taps_per_branch}_{tag}", seed=0)
    def _case(rng):
        taps = sdrk_design.channelizer_prototype(channels, taps_per_branch, passband, stopband, 60.0)
        return dict(
            params={
                "channels": channels,
                "taps_per_branch": taps_per_branch,
                "passband": passband,
                "stopband": stopband,
                "attenuation_db": 60.0,
            },
            arrays={"expected": Array("f64", taps)},
            tolerance={"class": "max_abs_err", "max_abs_err": 1e-12, "skip": 0},
            notes=(
                "scipy.signal.firwin(M L, (passband + stopband) / (2 M), window=('kaiser', kaiser_beta(60)), "
                "fs=1): the channelizer prototype, edges as fractions of the channel spacing fs / M, DC gain 1"
            ),
        )


_register_channelizer(16, 8, 0.5, 1.0, "0p5_1p0")
_register_channelizer(16, 24, 0.4, 0.6, "0p4_0p6")


@case(kernel="design", name="channelizer_taps_table", seed=663)
def channelizer_taps_table(rng):
    channels = np.array([4 << int(rng.integers(0, 7)) for _ in range(20)], dtype=np.int32)
    passband = rng.uniform(0.05, 1.2, 20)
    stopband = np.array([rng.uniform(p + 0.02, 1.5) for p in passband])
    attenuation = rng.uniform(20.0, 120.0, 20)
    taps = np.array(
        [sdrk_design.channelizer_taps(int(m), p, s, a) for m, p, s, a in zip(channels, passband, stopband, attenuation)],
        dtype=np.int32,
    )
    per_branch = np.array([-(-int(n) // int(m)) for n, m in zip(taps, channels)], dtype=np.int32)
    return dict(
        params={"pairs": 20},
        arrays={
            "channels": Array("i32", channels),
            "passband": Array("f64", passband),
            "stopband": Array("f64", stopband),
            "attenuation_db": Array("f64", attenuation),
            "expected": Array("i32", taps),
            "taps_per_branch": Array("i32", per_branch),
        },
        tolerance={"class": "bit_exact"},
        notes=(
            "scipy.signal.kaiserord(attenuation, 2 (stopband - passband) / M)[0], the tap count as returned "
            "(not forced odd), and taps_per_branch = ceil(taps / M); M a power of two 4 to 256, passband "
            "0.05 to 1.2, stopband up to 1.5, attenuation 20 to 120 dB"
        ),
    )
