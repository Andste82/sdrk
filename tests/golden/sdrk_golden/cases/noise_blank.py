"""Cases for the noise_blank kernel (impulse blanker with lookahead and hold, M33 section 1).

The impulse decision is discrete, so the expected values come from exact models of the
arithmetic fixed by the specification (rule R8.3 exception, as for ``squelch``): the float32
model (``f32model``) for C32 and the integer model with plain Python integers for CQ15. The
models follow the rules of the kernel as documented in ``docs/kernels.md``, including the three
rules of decisions D363 to D365 (open item I33-1): an impulse moves the average towards the
threshold power ``k p``, an average of zero starts at the first power that is no impulse (for
CQ15 "p > 0" means ``p >> 31 > 0``), and an impulse blanks ``D + H + 1`` output samples.

The tone and the impulse phases use ``math.sin`` and ``math.cos`` one value at a time and the
noise comes from ``rng.standard_normal``; nothing that feeds a committed byte uses NumPy
transcendentals or BLAS, so the files are identical on every CPU.

The regression guard of decision D111 runs the same loop in float64 (no flush of denormals)
on the stored inputs. ``snr_db_guard`` is the SNR of the model output against that float64
output from sample ``guard_skip`` on, minus 1 dB, rounded down to 0.1 dB. When both outputs
are identical (the usual result: the blanker only copies or zeroes samples, so the outputs
differ only when a decision flips) the SNR is infinite and the case stores
``snr_db_guard = GUARD_IDENTICAL_DB`` instead (decision D372); the Rust test then requires
identical outputs.
"""

import collections
import math
import struct

import numpy as np

from .. import f32model as fm
from .. import q15
from ..registry import Array, case

RATE_HZ = 192000
THRESHOLD_DB = 15.0
TAU_S = 0.01
HOLD_S = 0.0002
LOOKAHEAD_S = 0.00005

NOISE_SIGMA = 0.05
TONE_HZ = 10000
TONE_AMP = 0.3
IMPULSES = 50
IMPULSE_LEN = 4
IMPULSE_AMP = 3.0
# The impulses start at 50 distinct slots of 128 samples (from sample 128 on) plus a random
# offset of 0 to 63, so two impulses are at least 64 samples apart, more than one blanking
# span (D + 4 + H = 52 samples at the parameters above).
IMPULSE_SLOT = 128
IMPULSE_JITTER = 64

# Scale of the CQ15 case before quantization (decision D371): the impulses (3 plus the tone and
# the noise) must stay below full scale, otherwise their saturated power (at most 2) would no
# longer exceed the threshold power (31.6 times the average, about 3).
CQ15_SCALE = 0.25

# The stored guard when the model output equals the float64 output exactly (infinite SNR).
GUARD_IDENTICAL_DB = 300.0
GUARD_SKIP = 0


def bits(x):
    """The IEEE-754 bits of a float64 as an integer (exact in JSON)."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def coefficients(threshold_db, tau_s, hold_s, lookahead_s, rate_hz):
    """The f64 coefficients of ``NoiseBlankState::new`` (and the lengths of the block)."""
    fs = float(rate_hz)
    alpha = 1.0 - math.exp(-1.0 / (fs * tau_s))
    k = math.pow(10.0, threshold_db / 10.0)
    return {
        "alpha": alpha,
        "k": k,
        "alpha_q31": int(math.floor(alpha * 2.0**31 + 0.5)),
        "k_q16": int(math.floor(k * 2.0**16 + 0.5)),
        "hold": int(math.floor(hold_s * fs + 0.5)),
        "lookahead": int(math.floor(lookahead_s * fs + 0.5)),
    }


CO = coefficients(THRESHOLD_DB, TAU_S, HOLD_S, LOOKAHEAD_S, RATE_HZ)


# ---------------------------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------------------------


def clean_signal(rng, n):
    """Complex noise (sigma 0.05 per component) plus a 10 kHz complex tone of amplitude 0.3."""
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * NOISE_SIGMA
    re, im = np.empty(n), np.empty(n)
    for k in range(n):
        theta = 2.0 * math.pi * (((TONE_HZ * k) % RATE_HZ) / RATE_HZ)
        re[k] = TONE_AMP * math.cos(theta)
        im[k] = TONE_AMP * math.sin(theta)
    return noise + (re + 1j * im)


def impulse_signal(rng, n):
    """``clean_signal`` plus 50 impulses of 4 samples, amplitude 3 and a random phase each.

    Returns the signal and the sorted first samples of the impulses.
    """
    x = clean_signal(rng, n)
    slots = rng.choice(np.arange(1, n // IMPULSE_SLOT - 1), IMPULSES, replace=False)
    offsets = rng.integers(0, IMPULSE_JITTER, IMPULSES)
    phases = rng.uniform(0.0, 2.0 * math.pi, IMPULSES)
    starts = []
    for slot, off, ph in sorted(zip(slots.tolist(), offsets.tolist(), phases.tolist())):
        first = slot * IMPULSE_SLOT + off
        starts.append(first)
        v = complex(IMPULSE_AMP * math.cos(ph), IMPULSE_AMP * math.sin(ph))
        x[first : first + IMPULSE_LEN] += v
    return x, starts


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------


def _complex64(re, im):
    """A complex64 array from float32 parts (never ``re + 1j * im``)."""
    out = np.empty(len(re), dtype=np.complex64)
    out.real = np.asarray(re, dtype=np.float32)
    out.imag = np.asarray(im, dtype=np.float32)
    return out


def noise_blank_f32(x, co):
    """Float32 model. ``x`` is complex64. Returns (output complex64, blanked count)."""
    alpha, k = fm.f32(co["alpha"]), fm.f32(co["k"])
    d, span = co["lookahead"], co["lookahead"] + co["hold"] + 1
    zero = fm.f32(0.0)
    p, count, blanked = zero, 0, 0
    ring = collections.deque([(zero, zero)] * d)
    out_re = np.zeros(len(x), dtype=np.float32)
    out_im = np.zeros(len(x), dtype=np.float32)
    for i, (r, m) in enumerate(zip(x.real.tolist(), x.imag.tolist())):
        re, im = fm.f32(r), fm.f32(m)
        q = fm.add(fm.mul(re, re), fm.mul(im, im))
        kp = fm.mul(k, p)
        impulse = (not fm.is_finite(q)) or (p > zero and q > kp)
        if impulse:
            p = fm.flush_denormal(fm.add(p, fm.mul(alpha, fm.sub(kp, p))))
        elif p > zero:
            p = fm.flush_denormal(fm.add(p, fm.mul(alpha, fm.sub(q, p))))
        else:
            p = fm.flush_denormal(q)
        if d == 0:
            y = (re, im)
        else:
            y = ring.popleft()
            ring.append((re, im))
        if impulse:
            count = span
        if count > 0:
            count -= 1
            blanked += 1
        else:
            out_re[i], out_im[i] = y
    return _complex64(out_re, out_im), blanked


def noise_blank_q15(xs, co):
    """Integer model: ``xs`` is an (n, 2) array of Q15 raw values, the power is Q30, the
    average Q61, alpha Q31 and k Q16. Returns (output (n, 2) int64, blanked count)."""
    alpha, k = co["alpha_q31"], co["k_q16"]
    d, span = co["lookahead"], co["lookahead"] + co["hold"] + 1
    p, count, blanked = 0, 0, 0
    ring = collections.deque([(0, 0)] * d)
    out = np.zeros((len(xs), 2), dtype=np.int64)
    for i, (re, im) in enumerate(xs.tolist()):
        q = re * re + im * im
        p30 = p >> 31
        impulse = p30 > 0 and (q << 16) > k * p30
        if impulse:
            p = p + alpha * (((k * p30) >> 16) - int(q15.round_shift(p, 31)))
        elif p30 > 0:
            p = p + alpha * (q - int(q15.round_shift(p, 31)))
        else:
            p = q << 31
        if d == 0:
            y = (re, im)
        else:
            y = ring.popleft()
            ring.append((re, im))
        if impulse:
            count = span
        if count > 0:
            count -= 1
            blanked += 1
        else:
            out[i] = y
    return out, blanked


def noise_blank_f64(x, co):
    """The same loop in float64 (no flush); ``x`` is a list of complex. Returns a list."""
    alpha, k = co["alpha"], co["k"]
    d, span = co["lookahead"], co["lookahead"] + co["hold"] + 1
    p, count = 0.0, 0
    ring = collections.deque([0j] * d)
    out = []
    for v in x:
        q = v.real * v.real + v.imag * v.imag
        kp = k * p
        impulse = (not math.isfinite(q)) or (p > 0.0 and q > kp)
        if impulse:
            p = p + alpha * (kp - p)
        elif p > 0.0:
            p = p + alpha * (q - p)
        else:
            p = q
        if d == 0:
            y = v
        else:
            y = ring.popleft()
            ring.append(v)
        if impulse:
            count = span
        if count > 0:
            count -= 1
            out.append(0j)
        else:
            out.append(y)
    return out


def snr_db(reference, test):
    """``10 log10(sum |ref|^2 / sum |ref - test|^2)`` with exact sums."""
    num = math.fsum(abs(r) ** 2 for r in reference)
    den = math.fsum(abs(r - t) ** 2 for r, t in zip(reference, test))
    return math.inf if den == 0.0 else 10.0 * math.log10(num / den)


def guard_db(model, f64):
    """The SNR of the model against the float64 loop minus 1 dB, rounded down to 0.1 dB, or
    ``GUARD_IDENTICAL_DB`` when the outputs are identical."""
    snr = snr_db(f64[GUARD_SKIP:], model[GUARD_SKIP:])
    if math.isinf(snr):
        return GUARD_IDENTICAL_DB
    return math.floor((snr - 1.0) * 10.0) / 10.0


def check_blanking(blanked, starts, co):
    """Every impulse is blanked once, with ``D + 4 + H`` samples, and nothing else is."""
    want = len(starts) * (co["lookahead"] + IMPULSE_LEN + co["hold"])
    assert blanked == want, (blanked, want)


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


def nb_params(**extra):
    base = {
        "rate_hz": RATE_HZ,
        "threshold_db": THRESHOLD_DB,
        "tau_s": TAU_S,
        "hold_s": HOLD_S,
        "lookahead_s": LOOKAHEAD_S,
        "lookahead": CO["lookahead"],
        "hold": CO["hold"],
        "alpha_bits": bits(CO["alpha"]),
        "k_bits": bits(CO["k"]),
        "alpha_q31": CO["alpha_q31"],
        "k_q16": CO["k_q16"],
        "guard_skip": GUARD_SKIP,
    }
    base.update(extra)
    return base


@case(kernel="noise_blank", name="nb_c32_16384", seed=471)
def nb_c32(rng):
    x, starts = impulse_signal(rng, 16384)
    x = x.astype(np.complex64)
    out, blanked = noise_blank_f32(x, CO)
    check_blanking(blanked, starts, CO)
    f64 = noise_blank_f64(x.astype(np.complex128).tolist(), CO)
    guard = guard_db(out.astype(np.complex128).tolist(), f64)
    return dict(
        params=nb_params(blanked=blanked, impulse_starts=starts, snr_db_guard=guard),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", out)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="192 kHz; complex noise sigma 0.05 per component plus a 10 kHz complex tone 0.3, "
        "plus 50 impulses of 4 samples, amplitude 3 and a random phase each (first samples in "
        "impulse_starts, at least 64 samples apart); threshold 15 dB, tau 0.01 s, hold 0.0002 s "
        "(H = 38), lookahead 0.00005 s (D = 10); exact float32 model with the rules of D363 to "
        "D365; blanked is the count of zero outputs (D + 4 + H per impulse); snr_db_guard "
        "compares with the float64 loop (D111, D372)",
    )


@case(kernel="noise_blank", name="nb_cq15_16384", seed=472)
def nb_cq15(rng):
    x, starts = impulse_signal(rng, 16384)
    x = x * CQ15_SCALE
    xs = q15.pack(q15.from_float(x.real), q15.from_float(x.imag))
    out, blanked = noise_blank_q15(xs, CO)
    check_blanking(blanked, starts, CO)
    f64 = noise_blank_f64([complex(r / 32768.0, i / 32768.0) for r, i in xs.tolist()], CO)
    model = [complex(r / 32768.0, i / 32768.0) for r, i in out.tolist()]
    guard = guard_db(model, f64)
    return dict(
        params=nb_params(
            blanked=blanked, impulse_starts=starts, scale=CQ15_SCALE, snr_db_guard=guard
        ),
        arrays={"input": Array("ci16", xs), "expected": Array("ci16", out)},
        tolerance={"class": "bit_exact"},
        notes="the signal of nb_c32_16384 (own draw) scaled by 0.25 (D371, so the impulses do "
        "not saturate) and quantized; exact integer model (Q30 power, Q61 average, alpha Q31, "
        "k Q16, impulse test q 2^16 > k_q (p >> 31)) with the rules of D363 to D365",
    )


@case(kernel="noise_blank", name="nb_clean_c32_4096", seed=473)
def nb_clean_c32(rng):
    x = clean_signal(rng, 4096).astype(np.complex64)
    out, blanked = noise_blank_f32(x, CO)
    d = CO["lookahead"]
    delayed = np.zeros(len(x), dtype=np.complex64)
    delayed[d:] = x[: len(x) - d]
    assert blanked == 0
    assert out.tobytes() == delayed.tobytes()
    return dict(
        params=nb_params(blanked=0),
        arrays={"input": Array("cf32", x), "expected": Array("cf32", delayed)},
        tolerance={"class": "bit_exact"},
        model="float32",
        notes="the noise and the tone of nb_c32_16384 without impulses; the expected output is "
        "the input delayed by D = 10 samples (zeros first); the generator asserts that the "
        "float32 model gives the same",
    )
