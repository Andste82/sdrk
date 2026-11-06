"""The RDS reference signal of M41 section 5: an independent NumPy model of an RDS modulator.

The chain follows IEC 62106 and the conventions that M42 fixes for the transmitter:

1. **Groups** (``carousel``): PI ``0xD123``, PTY 10, TP 0, alternately one group 0A (programme
   service name ``SDRK TST``, segments 0 to 3 in turn; block B with TA 0, M/S 1, DI 0, block C
   ``0xE0E0``, the "no alternative frequencies" filler) and one group 2A (radiotext
   ``Hello from sdrk!`` followed by a carriage return and padded with spaces to 64 characters,
   segments 0 to 15 in turn, A/B flag 0). The carousel starts with 0A segment 0 and repeats
   after 32 groups.
2. **Blocks** (``checkword``, ``bits``): each information word is followed by its checkword
   (the remainder of ``info(x) x^10`` modulo ``g(x) = 0x5B9`` plus the offset word A, B, C, D),
   26 bits per block, sent most significant bit first.
3. **Differential coding** (``differential``): ``t_k = t_{k-1} xor d_k`` with ``t_{-1} = 0``,
   amplitude ``a_k = 2 t_k - 1``.
4. **Biphase symbols with the data-shaping filter** (``pulse_table``, ``rds_baseband``): the
   pulse of one bit is ``p(t) = h(t + Tb/4) - h(t - Tb/4)`` (first chip ``+a_k``), where ``h`` is
   the impulse response of ``H(f) = cos(pi f Tb / 4)`` for ``|f| <= 2 / Tb``. In closed form
   ``h(t) = 2 alpha cos(2 pi t B) / (alpha^2 - (2 pi t)^2)`` with ``alpha = pi Tb / 4`` and
   ``B = 2 / Tb``; with ``x = t / Tb`` and without the factor ``1 / Tb`` this is
   ``h(x) = 8 cos(4 pi x) / (pi (1 - 64 x^2))`` and ``h(+-1/8) = 2`` (the limit). The pulse is
   cut to ``-2 Tb <= t < 2 Tb`` and normalized to a peak of 1. Bit ``k`` is centred at
   ``k Tb`` (the first bit at ``t = 0``): ``b(t) = sum_k a_k p(t - k Tb)``.
5. **Multiplex** (``multiplex``) at 240 kHz: ``0.45 (L + R) + 0.45 (L - R) sin(2 phi)
   + 0.1 sin(phi) + 0.04 b sin(3 phi)`` with ``L = sin(2 pi 1000 t)``, ``R = sin(2 pi 2000 t)`` and
   the pilot phase ``phi = 2 pi 19000 t`` (the levels of M24; the RDS carrier is locked to three
   times the pilot phase, 4 % of the multiplex; the peak is 1.04).
6. **FM** (``FmModulator``): 75 kHz per unit of the multiplex; the multiplex is held for two
   output samples (zero-order hold to 480 kHz) and the phase is accumulated at 480 kHz,
   ``phase_0 = 0``, ``phase_{j+1} = phase_j + K mpx[j // 2]`` with
   ``K = 2 pi 75000 / 480000``; this is the same as modulating at 240 kHz and writing every
   sample twice at the doubled rate. The output is ``0.5 exp(j phase)`` (0.5 of full scale)
   plus complex white Gaussian noise of ``sigma`` per component, rounded to ``cs16`` with the
   R7 rule ``floor(32768 v + 0.5)``. The SNR is per sample over the 480 kHz bandwidth:
   ``SNR = 0.25 / (2 sigma^2)``.

Every transcendental value comes from a table filled with the scalar ``math`` functions at
exactly reduced integer arguments (``cos(2 pi r / D)``, ``sin(2 pi r / 240)``) or from the
scalar ``math.cos``/``math.sin`` of the accumulated phase, so the files are byte-identical on
every CPU and a Rust model with ``libm`` can follow the same operations (testkit
``sdrk_testkit::rds``). The functions are pure, so the RDS transmitter of M42 can reuse them.
"""

import json
import math

import numpy as np

from .. import q15, scalar_math
from ..registry import Array, case

GENERATOR = 0x5B9
OFFSET_WORDS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "Cp": 0x350, "D": 0x1B4}
PI_CODE = 0xD123
PTY = 10
TP = 0
PS = b"SDRK TST"
RT_TEXT = b"Hello from sdrk!"
CAROUSEL_PERIOD = 32

MPX_RATE_HZ = 240000
RF_RATE_HZ = 480000
BIT_RATE_NUM, BIT_RATE_DEN = 2375, 2  # 1187.5 bit/s
DEVIATION_HZ = 75000.0
LEFT_HZ, RIGHT_HZ, PILOT_HZ = 1000, 2000, 19000
MAIN_LEVEL, PILOT_LEVEL, RDS_LEVEL = 0.45, 0.1, 0.04
RF_AMPLITUDE = 0.5
SEED = 570

SHORT_MPX = 4096
LATE_START = 1000003
LATE_LEN = 2048


# ---------------------------------------------------------------------------------------------
# groups and bits
# ---------------------------------------------------------------------------------------------


def remainder(v):
    """``v(x) mod g(x)`` for a non-negative integer ``v`` (bit ``k`` = ``x^k``)."""
    while v.bit_length() > 10:
        v ^= GENERATOR << (v.bit_length() - 11)
    return v


def checkword(info, offset):
    """The checkword of the 16-bit ``info`` at the offset ``"A"``, ``"B"``, ``"C"``, ``"Cp"``, ``"D"``."""
    return remainder(info << 10) ^ OFFSET_WORDS[offset]


def radiotext():
    """The 64 radiotext bytes: the text, a carriage return, spaces."""
    t = RT_TEXT + b"\r"
    return t + b" " * (64 - len(t))


def group(g):
    """Group ``g`` of the carousel as (four information words, four offsets)."""
    b_common = (TP << 10) | (PTY << 5)
    if g % 2 == 0:
        seg = (g // 2) % 4
        b = (0 << 12) | b_common | (0 << 4) | (1 << 3) | (0 << 2) | seg
        d = (PS[2 * seg] << 8) | PS[2 * seg + 1]
        words = [PI_CODE, b, 0xE0E0, d]
    else:
        seg = (g // 2) % 16
        rt = radiotext()
        b = (2 << 12) | b_common | (0 << 4) | seg
        c = (rt[4 * seg] << 8) | rt[4 * seg + 1]
        d = (rt[4 * seg + 2] << 8) | rt[4 * seg + 3]
        words = [PI_CODE, b, c, d]
    return words, ["A", "B", "C", "D"]


def carousel(n):
    """The first ``n`` groups: an ``(n, 4)`` int64 array of information words."""
    return np.array([group(g)[0] for g in range(n)], dtype=np.int64).reshape(n, 4)


def checkwords(n):
    """The checkwords of the first ``n`` groups, ``(n, 4)``."""
    out = []
    for g in range(n):
        words, offsets = group(g)
        out.append([checkword(w, o) for w, o in zip(words, offsets)])
    return np.array(out, dtype=np.int64).reshape(n, 4)


def bits(n):
    """The data bits ``d_k`` of the first ``n`` groups (``uint8``, 104 per group, MSB first)."""
    out = []
    for g in range(n):
        words, offsets = group(g)
        for w, o in zip(words, offsets):
            block = (w << 10) | checkword(w, o)
            out.extend((block >> (25 - i)) & 1 for i in range(26))
    return np.array(out, dtype=np.uint8)


def differential(d):
    """``t_k = t_{k-1} xor d_k`` with ``t_{-1} = 0``."""
    return np.bitwise_xor.accumulate(np.asarray(d, dtype=np.uint8))


# ---------------------------------------------------------------------------------------------
# baseband
# ---------------------------------------------------------------------------------------------


def bit_grid(rate):
    """``(N, D)`` with ``t / Tb = m N / D`` exactly for the sample ``m`` at ``rate`` Hz."""
    num, den = BIT_RATE_NUM, BIT_RATE_DEN * rate
    g = math.gcd(num, den)
    num, den = num // g, den // g
    if den % 4 != 0:
        raise ValueError("the quarter bit must be a whole number of grid steps")
    return num, den


def h(n, den):
    """``h(x)`` at ``x = n / den`` (``n`` an integer): ``8 cos(4 pi x) / (pi (1 - 64 x^2))``.

    The cosine is taken at the exactly reduced argument ``2 pi r / den``, ``r = 2 n mod den``;
    ``1 - 64 x^2`` is ``(den^2 - 64 n^2) / den^2`` in exact integers; the limit at
    ``x = +-1/8`` is 2.
    """
    d = den * den - 64 * n * n
    if d == 0:
        return 2.0
    r = (2 * n) % den
    c = math.cos(2.0 * math.pi * r / den)
    return 8.0 * c * float(den * den) / (math.pi * float(d))


def pulse_table(rate):
    """The normalized pulse at the grid points: ``(N, D, table, peak)``.

    ``table[n + 2 D]`` is ``p(n / D Tb) / peak`` for ``-2 D <= n < 2 D``, with
    ``p = h(n + D/4) - h(n - D/4)`` and ``peak = max |p|`` over the same grid (the grid peak is
    below the continuous peak by less than 1e-9 relative at 240 kHz).
    """
    num, den = bit_grid(rate)
    q = den // 4
    raw = [h(n + q, den) - h(n - q, den) for n in range(-2 * den, 2 * den)]
    peak = max(abs(v) for v in raw)
    return num, den, np.array([v / peak for v in raw], dtype=np.float64), peak


def amplitudes(tx_bits):
    """``a_k = 2 t_k - 1`` as float64."""
    return 2.0 * np.asarray(tx_bits, dtype=np.float64) - 1.0


def rds_baseband(a, m0, count, rate, table=None):
    """``b(m)`` for ``m = m0 .. m0 + count``: ``sum_k a_k p(t_m - k Tb)`` over the four bits whose
    pulse covers ``t_m`` (``k`` ascending from ``k0 = floor(m N / D) - 1``, bits ``k < 0`` add
    0.0), accumulated from 0.0. ``a`` must hold every bit up to ``k0 + 3`` of the last sample.
    """
    num, den, tab, _ = table if table is not None else pulse_table(rate)
    m = np.arange(m0, m0 + count, dtype=np.int64)
    k0 = (m * num) // den - 1
    if count and int(k0[-1]) + 3 >= len(a):
        raise ValueError("not enough bits for the requested samples")
    out = np.zeros(count, dtype=np.float64)
    for j in range(4):
        k = k0 + j
        n = m * num - den * k
        term = np.where(k >= 0, a[np.maximum(k, 0)] * tab[n + 2 * den], 0.0)
        out = out + term
    return out


def bits_needed(m_end, rate):
    """Bits needed for the samples ``m < m_end``."""
    num, den = bit_grid(rate)
    return ((m_end - 1) * num) // den + 3


# ---------------------------------------------------------------------------------------------
# multiplex and FM
# ---------------------------------------------------------------------------------------------


def sin_table(rate):
    """``sin(2 pi r / Q)`` for ``r = 0 .. Q``, ``Q = rate / 1000`` (every tone is a multiple of 1 kHz)."""
    qn = rate // 1000
    return qn, np.array([math.sin(2.0 * math.pi * r / qn) for r in range(qn)], dtype=np.float64)


def multiplex(m0, baseband, rate=MPX_RATE_HZ, rds_level=RDS_LEVEL):
    """The multiplex at the samples ``m0 ..`` for the RDS baseband ``baseband`` (float64).

    ``0.45 (l + r) + 0.45 (l - r) s2 + 0.1 s1 + rds_level b s3`` evaluated left to right, each
    sine from the table at ``(f / 1000) m mod Q``.
    """
    qn, tab = sin_table(rate)
    m = np.arange(m0, m0 + len(baseband), dtype=np.int64)

    def tone(f):
        return tab[((f // 1000) * m) % qn]

    left, right = tone(LEFT_HZ), tone(RIGHT_HZ)
    s1, s2, s3 = tone(PILOT_HZ), tone(2 * PILOT_HZ), tone(3 * PILOT_HZ)
    return (
        MAIN_LEVEL * (left + right)
        + MAIN_LEVEL * (left - right) * s2
        + PILOT_LEVEL * s1
        + rds_level * baseband * s3
    )


FM_K = 2.0 * math.pi * DEVIATION_HZ / RF_RATE_HZ


class FmModulator:
    """The FM stage of step 6, chunk by chunk (the phase carries over)."""

    def __init__(self):
        self.phase = 0.0

    def run(self, mpx):
        """``(re, im)`` float64 of the ``2 len(mpx)`` output samples, amplitude 0.5, no noise."""
        inc = np.repeat(np.asarray(mpx, dtype=np.float64), 2) * FM_K
        acc = np.cumsum(np.concatenate(([self.phase], inc)))
        phase = acc[:-1]
        self.phase = float(acc[-1])
        return RF_AMPLITUDE * scalar_math.cos(phase), RF_AMPLITUDE * scalar_math.sin(phase)


def to_cs16(re, im):
    """``(n, 2)`` int64 Q15 raw values of the R7 rule."""
    return np.stack([q15.from_float(re), q15.from_float(im)], axis=1)


def noise_sigma(snr_db):
    """Per-component sigma for the SNR per sample over 480 kHz of the 0.5 amplitude carrier."""
    return math.sqrt(RF_AMPLITUDE * RF_AMPLITUDE / (2.0 * 10.0 ** (snr_db / 10.0)))


# ---------------------------------------------------------------------------------------------
# case and long reference
# ---------------------------------------------------------------------------------------------


@case(kernel="rds", name="reference_short", seed=SEED)
def reference_short(rng):
    groups = carousel(CAROUSEL_PERIOD)
    d = bits(CAROUSEL_PERIOD)
    t = differential(d)
    table = pulse_table(MPX_RATE_HZ)
    n_bits = bits_needed(LATE_START + LATE_LEN, MPX_RATE_HZ)
    reps = -(-n_bits // len(d))
    a = amplitudes(differential(np.tile(d, reps)))
    bb = rds_baseband(a, 0, SHORT_MPX, MPX_RATE_HZ, table)
    mpx = multiplex(0, bb)
    re, im = FmModulator().run(mpx)
    bb_late = rds_baseband(a, LATE_START, LATE_LEN, MPX_RATE_HZ, table)
    mpx_late = multiplex(LATE_START, bb_late)
    peak = table[3]
    return dict(
        params={
            "pi": PI_CODE,
            "pty": PTY,
            "tp": TP,
            "ps": PS.decode("ascii"),
            "rt": RT_TEXT.decode("ascii"),
            "groups": CAROUSEL_PERIOD,
            "mpx_rate_hz": MPX_RATE_HZ,
            "rf_rate_hz": RF_RATE_HZ,
            "deviation_hz": DEVIATION_HZ,
            "rds_level": RDS_LEVEL,
            "pilot_level": PILOT_LEVEL,
            "main_level": MAIN_LEVEL,
            "rf_amplitude": RF_AMPLITUDE,
            "pulse_peak": peak,
            "grid_num": table[0],
            "grid_den": table[1],
            "late_start": LATE_START,
            "rf_max_lsb": 1,
        },
        arrays={
            "blocks": Array("i32", groups.reshape(-1)),
            "checkwords": Array("i32", checkwords(CAROUSEL_PERIOD).reshape(-1)),
            "data_bits": Array("u8", d),
            "tx_bits": Array("u8", t),
            "baseband": Array("f32", bb),
            "mpx": Array("f32", mpx),
            "rf": Array("ci16", to_cs16(re, im)),
            "baseband_late": Array("f32", bb_late),
            "mpx_late": Array("f32", mpx_late),
        },
        tolerance={"class": "max_abs_err", "max_abs_err": 1e-6, "skip": 0},
        notes="the RDS reference signal of M41 section 5 (module docstring of cases/rds.py), no "
        "noise; blocks and checkwords: the 32 groups of one carousel period, 4 words per group "
        "(bit-exact); data_bits and tx_bits: their 3328 bits before and after the differential "
        "coding (bit-exact); baseband and mpx: samples 0..4095 at 240 kHz, baseband_late and "
        "mpx_late: samples late_start..late_start+2047 of the endless carousel (float64 model "
        "rounded once to float32, max_abs_err applies); rf: the cs16 output samples 0..8191 at "
        "480 kHz without noise (at most rf_max_lsb LSB apart, math.cos versus libm); pulse_peak "
        "is the grid peak of p before normalization",
    )


def emit_reference(path, seconds=8.0, snr_db=30.0, chunk=48000):
    """Write the long reference signal to ``path`` (raw cs16) and ``path + ".json"``.

    ``seconds`` of RF at 480 kHz with complex noise from ``default_rng(570)`` for ``snr_db``
    (``noise_sigma``), drawn as ``standard_normal((n, 2))`` per chunk of ``chunk`` multiplex
    samples (re, im per output sample). Not part of the registry: never checked or committed.
    """
    n_mpx = int(round(seconds * MPX_RATE_HZ))
    table = pulse_table(MPX_RATE_HZ)
    d = bits(CAROUSEL_PERIOD)
    n_bits = bits_needed(n_mpx, MPX_RATE_HZ)
    reps = -(-n_bits // len(d))
    a = amplitudes(differential(np.tile(d, reps)))
    rng = np.random.default_rng(SEED)
    sigma = noise_sigma(snr_db)
    fm = FmModulator()
    with open(path, "wb") as f:
        for m0 in range(0, n_mpx, chunk):
            count = min(chunk, n_mpx - m0)
            mpx = multiplex(m0, rds_baseband(a, m0, count, MPX_RATE_HZ, table))
            re, im = fm.run(mpx)
            noise = rng.standard_normal((2 * count, 2)) * sigma
            out = to_cs16(re + noise[:, 0], im + noise[:, 1])
            f.write(np.ascontiguousarray(out, dtype="<i2").tobytes())
    meta = {
        "format": "cs16",
        "rate_hz": RF_RATE_HZ,
        "mpx_rate_hz": MPX_RATE_HZ,
        "samples": 2 * n_mpx,
        "seconds": seconds,
        "snr_db": snr_db,
        "noise_sigma": sigma,
        "seed": SEED,
        "pi": PI_CODE,
        "pty": PTY,
        "ps": PS.decode("ascii"),
        "rt": RT_TEXT.decode("ascii"),
        "groups_per_period": CAROUSEL_PERIOD,
        "generator": "tests/golden/generate.py --emit rds_reference",
    }
    with open(path + ".json", "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    return meta
