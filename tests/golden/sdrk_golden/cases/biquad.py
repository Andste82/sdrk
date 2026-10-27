"""Cases for the biquad and biquad_real kernels (cascaded second-order sections, M22)."""

import numpy as np
import scipy.signal

from .. import design, q15
from ..registry import Array, case
from .fir import noise

FS = 1.0


def sos_of(coefs):
    """The ``scipy.signal`` SOS matrix of flat ``[b0, b1, b2, a1, a2]`` rows (values as float64)."""
    rows = np.asarray(coefs, dtype=np.float64).reshape(-1, 5)
    return np.array([[r[0], r[1], r[2], 1.0, r[3], r[4]] for r in rows])


def f32_coefs(rows):
    """Coefficient rows rounded to float32, flat."""
    return np.array(rows, dtype=np.float64).reshape(-1).astype(np.float32)


def q14_coefs(rows):
    """Coefficient rows as Q2.14 integers, flat."""
    return np.array([design.q14(v) for row in rows for v in row], dtype=np.int64)


def df1_q15(x, coefs):
    """The integer model of section ``k`` after section ``k - 1``: direct form I with Q2.14
    coefficients, ``y = sat16(round_shift(acc, 14))`` per section (the model of the kernel)."""
    out = [int(v) for v in x]
    for b0, b1, b2, a1, a2 in np.asarray(coefs, dtype=np.int64).reshape(-1, 5).tolist():
        x1 = x2 = y1 = y2 = 0
        nxt = []
        for v in out:
            acc = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y = int(q15.sat16(q15.round_shift(acc, 14)))
            nxt.append(y)
            x2, x1, y2, y1 = x1, v, y1, y
        out = nxt
    return np.array(out, dtype=np.int64)


def df2t_f64(x, coefs_by_sample):
    """Transposed direct form II in float64, switching the coefficient row per sample."""
    s1 = s2 = 0.0
    out = np.zeros(len(x))
    for n, v in enumerate(np.asarray(x, dtype=np.float64).tolist()):
        b0, b1, b2, a1, a2 = coefs_by_sample(n)
        y = b0 * v + s1
        s1 = (b1 * v + s2) - a1 * y
        s2 = b2 * v - a2 * y
        out[n] = y
    return out


def sosfilt_real(coefs, x):
    return scipy.signal.sosfilt(sos_of(coefs), np.asarray(x, dtype=np.float64))


@case(kernel="biquad", name="butter4_lp_c32_8192", seed=424)
def butter4_lp_c32(rng):
    coefs = f32_coefs(design.butterworth_sos("lowpass", 4, 0.1, FS))
    x = noise(rng, 8192, 0.3).astype(np.complex64)
    expected = sosfilt_real(coefs, x.real) + 1j * sosfilt_real(coefs, x.imag)
    return dict(
        params={"sections": 2, "response": "lowpass", "order": 4, "cutoff_hz": 0.1, "fs_hz": FS},
        arrays={"input": Array("cf32", x), "coefs": Array("f32", coefs), "expected": Array("cf64", expected)},
        tolerance={"class": "tolerance", "snr_db_min": 100.0, "skip": 0},
        notes="butterworth_sos(lowpass, 4, 0.1) rounded to f32; scipy.signal.sosfilt in float64 with the "
        "f32 coefficient values, per component; complex noise sigma 0.3",
    )


@case(kernel="biquad", name="butter4_lp_cq15_8192", seed=425)
def butter4_lp_cq15(rng):
    coefs = q14_coefs(design.butterworth_sos("lowpass", 4, 0.1, FS))
    x = noise(rng, 8192, 0.3)
    xr, xi = q15.from_float(x.real), q15.from_float(x.imag)
    return dict(
        params={"sections": 2, "response": "lowpass", "order": 4, "cutoff_hz": 0.1, "fs_hz": FS},
        arrays={
            "input": Array("ci16", q15.pack(xr, xi)),
            "coefs": Array("i16", coefs),
            "expected": Array("ci16", q15.pack(df1_q15(xr, coefs), df1_q15(xi, coefs))),
        },
        tolerance={"class": "bit_exact"},
        notes="butterworth_sos(lowpass, 4, 0.1) as Q2.14; direct form I per section, "
        "sat16(round_shift(b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2, 14)); quantized complex noise sigma 0.3",
    )


def real_f32_case(rng, rows, notes, snr):
    coefs = f32_coefs(rows)
    x = (rng.standard_normal(8192) * 0.3).astype(np.float32)
    return dict(
        params={"sections": len(rows), "fs_hz": FS},
        arrays={"input": Array("f32", x), "coefs": Array("f32", coefs), "expected": Array("f64", sosfilt_real(coefs, x))},
        tolerance={"class": "tolerance", "snr_db_min": snr, "skip": 0},
        notes=notes,
    )


@case(kernel="biquad_real", name="notch_q10_f32_8192", seed=426)
def notch_q10_f32(rng):
    rows = [design.biquad_coefs("notch", 0.05, 10.0, FS)]
    return real_f32_case(
        rng, rows, "biquad_coefs(notch, 0.05, 10) rounded to f32; scipy.signal.sosfilt in float64; real noise sigma 0.3", 90.0
    )


@case(kernel="biquad_real", name="bandpass_q2_f32_8192", seed=427)
def bandpass_q2_f32(rng):
    rows = [design.biquad_coefs("bandpass", 0.1, 2.0, FS)]
    return real_f32_case(
        rng, rows, "biquad_coefs(bandpass, 0.1, 2) rounded to f32; scipy.signal.sosfilt in float64; real noise sigma 0.3", 100.0
    )


@case(kernel="biquad_real", name="butter2_hp_q15_8192", seed=428)
def butter2_hp_q15(rng):
    coefs = q14_coefs(design.butterworth_sos("highpass", 2, 0.01, FS))
    x = q15.from_float(rng.standard_normal(8192) * 0.3)
    return dict(
        params={"sections": 1, "response": "highpass", "order": 2, "cutoff_hz": 0.01, "fs_hz": FS},
        arrays={"input": Array("i16", x), "coefs": Array("i16", coefs), "expected": Array("i16", df1_q15(x, coefs))},
        tolerance={"class": "bit_exact"},
        notes="butterworth_sos(highpass, 2, 0.01) as Q2.14; direct form I with one rounding per section; "
        "quantized real noise sigma 0.3",
    )


@case(kernel="biquad_real", name="retune_f32_8192", seed=429)
def retune_f32(rng):
    first = f32_coefs(design.butterworth_sos("lowpass", 2, 0.05, FS))
    second = f32_coefs(design.butterworth_sos("lowpass", 2, 0.15, FS))
    x = (rng.standard_normal(8192) * 0.3).astype(np.float32)
    rows = [first.astype(np.float64).tolist(), second.astype(np.float64).tolist()]
    expected = df2t_f64(x, lambda n: rows[0] if n < 4096 else rows[1])
    return dict(
        params={"sections": 1, "switch_at": 4096, "fs_hz": FS},
        arrays={
            "input": Array("f32", x),
            "coefs": Array("f32", np.concatenate([first, second])),
            "expected": Array("f64", expected),
        },
        tolerance={"class": "tolerance", "snr_db_min": 90.0, "skip": 0},
        notes="lowpass order 2 with cutoff 0.05 for samples 0..4095 and 0.15 from 4096 (set_coefs at that "
        "sample, history kept); transposed direct form II in float64 with the f32 coefficient values, "
        "switching the coefficients at 4096 and keeping s1 and s2; coefs holds both rows (first, second)",
    )
