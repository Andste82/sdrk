"""Cases for the gauss_noise kernel (M48 section 4): deterministic white Gaussian noise.

The model is written from the specification, independently of the Rust code:

- PCG32 (PCG-XSH-RR 64/32 of ``pcg_basic.c``) on Python integers masked to 64 bits, with the
  O(log n) jump of ``pcg_advance_lcg_64``. One draw per real value; a complex sample takes two
  draws (real part first), so sample ``n`` starts after ``n`` (real) or ``2 n`` (complex) draws.
- The inverse-CDF table of section 2.2, built here from ``scipy.special.ndtri`` (not from the
  AS241 transcription of ``build.rs``) and rounded in NumPy ``int64`` with its own rounding-tie
  guard: ``K[L][j] = -ndtri((32 + j) 2^(L - 38))``, ``B = floor(K 2^17 + 0.5)``, entry
  ``(B[L][j] << 12) | (B[L][j] - B[L][j + 1])``.
- The integer normal value ``zq`` of section 2.1 in Python integers (``clz`` from
  ``int.bit_length``), and the output formats of section 3: ``f32`` is
  ``float32(zq 2^-17 sigma)`` (the exact float64 product rounded once), ``Q15`` the integer
  formula from the 24-bit significand of ``sigma``, which the generator checks against
  ``q15.from_float`` of the exact value for every value.

No ``math`` function touches a sample value, so every sample case is ``bit_exact``. Each case
also stores the ideal value ``sigma sign (-ndtri(x / 2^33))`` (float64) with the contract bound
``max_abs_err_ideal``, the SNR against it and the regression guard ``snr_db_guard`` (the SNR
minus 1 dB, rounded down to 0.1 dB). The case ``icdf_table`` stores the table and the exact
distribution of the generator, enumerated over all 2^31 magnitude cells.

The case seed only names the case: the sequence is the PCG32 model with ``generator_seed`` and
``stream``; ``numpy.random.default_rng`` is not used.
"""

import math
import struct
from fractions import Fraction

import numpy as np
from scipy import special

from .. import q15
from ..registry import Array, case

MASK = (1 << 64) - 1
MULT = 6364136223846793005

# Table geometry (section 2.2).
OCTAVES = 32
SEGMENTS = 32
Q17 = 1 << 17
# The smallest allowed distance of K 2^17 from a rounding tie, in LSB (section 2.3).
TIE_GUARD_LSB = 1e-4
# The contract bound of the table error, in units of sigma (section 4).
MAX_TABLE_ERR = 5.7e-5
# Thresholds of the exact one-sided tail probabilities P(Z > t).
TAIL_T = (1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.3)


class Pcg32:
    """PCG-XSH-RR 64/32 with Python integers."""

    def __init__(self, seed, stream):
        self.inc = ((stream << 1) | 1) & MASK
        self.state = 0
        self._step()
        self.state = (self.state + seed) & MASK
        self._step()

    def _step(self):
        self.state = (self.state * MULT + self.inc) & MASK

    def next_u32(self):
        old = self.state
        self._step()
        x = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        r = old >> 59
        return ((x >> r) | (x << ((-r) & 31))) & 0xFFFFFFFF

    def advance(self, delta):
        """Skips ``delta`` draws (``pcg_advance_lcg_64``)."""
        cur_mult, cur_plus = MULT, self.inc
        acc_mult, acc_plus = 1, 0
        while delta > 0:
            if delta & 1:
                acc_mult = (acc_mult * cur_mult) & MASK
                acc_plus = (acc_plus * cur_mult + cur_plus) & MASK
            cur_plus = ((cur_mult + 1) * cur_plus) & MASK
            cur_mult = (cur_mult * cur_mult) & MASK
            delta >>= 1
        self.state = (acc_mult * self.state + acc_plus) & MASK


def self_test_pcg():
    """The known answer of pcg_basic (seed 42, stream 54) and the jump against stepping."""
    rng = Pcg32(42, 54)
    want = [0xA15C02B7, 0x7B47F409, 0xBA1D3330, 0x83D2F293, 0xBFA4784B, 0xCBED606E]
    got = [rng.next_u32() for _ in want]
    assert got == want, [hex(g) for g in got]
    a = Pcg32(5, 6)
    b = Pcg32(5, 6)
    for _ in range(777):
        a.next_u32()
    b.advance(777)
    assert a.state == b.state
    # The period is 2^64 draws.
    c = Pcg32(5, 6)
    c.advance(1 << 63)
    assert c.state != Pcg32(5, 6).state
    c.advance(1 << 63)
    assert c.state == Pcg32(5, 6).state


# ---------------------------------------------------------------------------------------------
# The table (section 2.2), from scipy.special.ndtri.


def knots():
    """The Q17 knots ``B[L][j]`` (int64, shape (32, 33)) and the smallest tie distance in LSB."""
    lj = np.arange(OCTAVES, dtype=np.int64)[:, None]
    jj = np.arange(SEGMENTS + 1, dtype=np.int64)[None, :]
    # q = (32 + j) 2^(L - 38): exact in float64 (a 6-bit integer times a power of two).
    q = np.ldexp((32 + jj).astype(np.float64), (lj - 38).astype(np.int32))
    k = -special.ndtri(q)
    # q = 1/2 exactly (L = 31, j = 32): the median.
    k[31, 32] = 0.0
    t = k * float(Q17)
    frac = t - np.floor(t)
    tie = float(np.min(np.abs(frac - 0.5)))
    assert tie >= TIE_GUARD_LSB, f"a knot lies within {tie} LSB of a rounding tie"
    b = np.floor(t + 0.5).astype(np.int64)
    return b, tie


def table():
    """The 1024 entries (Python ints) and the properties of section 2.2."""
    b, tie = knots()
    # Continuity across octaves: B[L][32] and B[L + 1][0] are the same quantile.
    assert np.array_equal(b[:-1, SEGMENTS], b[1:, 0])
    assert int(b[0, 0]) == 830729, int(b[0, 0])
    assert int(b[31, 32]) == 0
    base = b[:, :SEGMENTS]
    slope = b[:, :SEGMENTS] - b[:, 1:]
    assert int(slope.min()) >= 1 and int(slope.max()) <= 4095, (slope.min(), slope.max())
    assert int(base.max()) < (1 << 20)
    entries = ((base << 12) | slope).reshape(-1)
    assert len(entries) == 1024 and int(entries.max()) < (1 << 32)
    flat = base.reshape(-1)
    assert np.all(flat[:-1] >= flat[1:]), "the knots must be non-increasing"
    props = {
        "tie_distance_min_lsb": tie,
        "slope_min": int(slope.min()),
        "slope_max": int(slope.max()),
    }
    return [int(e) for e in entries], props


# ---------------------------------------------------------------------------------------------
# From a draw to an integer normal value (section 2.1).


def magnitude(entries, x):
    """``mag`` of the odd 32-bit ``x = 2 (r & 0x7FFFFFFF) + 1``."""
    lvl = x.bit_length() - 1
    n = (x << (31 - lvl)) & 0xFFFFFFFF
    sub = (n >> 26) & 31
    frac = (n >> 11) & 0x7FFF
    e = entries[(lvl << 5) | sub]
    return (e >> 12) - (((e & 0xFFF) * frac + (1 << 14)) >> 15)


def normal_q17(entries, r):
    """``zq`` of the draw ``r`` and the odd ``x`` of its cell."""
    x = ((r << 1) | 1) & 0xFFFFFFFF
    mag = magnitude(entries, x)
    return (-mag if r >> 31 else mag), x


def self_test_normal(entries):
    """Spot checks of section 2.1: the extremes, the median and the symmetry."""
    assert normal_q17(entries, 0)[0] == 830729
    assert normal_q17(entries, 0x80000000)[0] == -830729
    assert normal_q17(entries, 0x7FFFFFFF)[0] == 0
    assert normal_q17(entries, 0xFFFFFFFF)[0] == 0
    rng = Pcg32(1, 1)
    for _ in range(1000):
        r = rng.next_u32()
        assert normal_q17(entries, r ^ 0x80000000)[0] == -normal_q17(entries, r)[0]


# ---------------------------------------------------------------------------------------------
# The exact distribution (section 2.4), enumerated over all 2^31 magnitude cells.


def magnitudes_np(entries, lvl, sub, frac):
    """Vectorized ``mag`` for the octave ``lvl`` and arrays of ``sub`` and ``frac``."""
    e = np.asarray(entries, dtype=np.int64)[(lvl << 5) | sub]
    return (e >> 12) - (((e & 0xFFF) * frac + (1 << 14)) >> 15)


def cell_groups(entries):
    """Yields per octave ``(lvl, mag, count, x_first, x_last)``: the cells of equal ``(sub, frac)``
    as one group, with its magnitude, its number of odd ``x`` and its first and last odd ``x``."""
    for lvl in range(OCTAVES):
        if lvl <= 20:
            # Every odd x of the octave is its own group (x has at most 20 bits after the
            # leading one, so (sub, frac) is unique per x).
            x = np.arange((1 << lvl) | 1, 1 << (lvl + 1), 2, dtype=np.int64)
            n = x << (31 - lvl)
            sub = (n >> 26) & 31
            frac = (n >> 11) & 0x7FFF
            mag = magnitudes_np(entries, lvl, sub, frac)
            yield lvl, mag, np.ones_like(x), x, x
        else:
            # 2^(L - 20) x per (sub, frac), half of them odd.
            g = np.arange(1 << 20, dtype=np.int64)
            sub = g >> 15
            frac = g & 0x7FFF
            mag = magnitudes_np(entries, lvl, sub, frac)
            low = lvl - 20
            x0 = (np.int64(1) << lvl) | (g << low)
            cnt = np.full_like(g, 1 << (low - 1))
            yield lvl, mag, cnt, x0 + 1, x0 + (1 << low) - 1


def distribution(entries):
    """The exact distribution of ``|zq|``: ``counts[m]`` = number of the 2^31 magnitude cells with
    magnitude ``m``, and the largest table error against ndtri (at the first and the last cell of
    every group)."""
    counts = np.zeros(830730, dtype=np.int64)
    max_err = 0.0
    where = None
    for lvl, mag, cnt, x_first, x_last in cell_groups(entries):
        counts += np.bincount(mag, weights=cnt, minlength=830730).astype(np.int64)
        z = mag.astype(np.float64) / Q17
        for x in (x_first, x_last):
            ideal = -special.ndtri(np.ldexp(x.astype(np.float64), -33))
            err = np.abs(z - ideal)
            i = int(np.argmax(err))
            if float(err[i]) > max_err:
                max_err = float(err[i])
                where = float(ideal[i])
    assert int(counts.sum()) == 1 << 31
    return counts, max_err, where


def exact_numbers(entries):
    """The contract numbers of section 4 from the exact distribution."""
    counts, max_err, where = distribution(entries)
    assert max_err <= MAX_TABLE_ERR, max_err
    m = [int(v) for v in range(len(counts))]
    c = [int(v) for v in counts]
    # Moments of Z = +-m 2^-17 (each sign with half the weight of a magnitude cell; the mean is 0
    # by symmetry): E[Z^2] = sum c m^2 / 2^31 / 2^34, E[Z^4] = sum c m^4 / 2^31 / 2^68.
    s2 = sum(ci * mi * mi for ci, mi in zip(c, m))
    s4 = sum(ci * mi**4 for ci, mi in zip(c, m))
    var = Fraction(s2, 1 << (31 + 34))
    m4 = Fraction(s4, 1 << (31 + 68))
    kurt = m4 / (var * var)
    # The CDF of Z on the lattice -830729 .. 830729 (Q17); P(Z = +-m) = c[m] / 2^32 for m > 0.
    p = np.concatenate([counts[:0:-1], [2 * counts[0]], counts[1:]]).astype(np.int64)
    cum = np.cumsum(p)
    assert int(cum[-1]) == 1 << 32
    lattice = np.arange(-830729, 830730, dtype=np.float64) / Q17
    phi = special.ndtr(lattice)
    after = cum.astype(np.float64) / 2.0**32
    before = (cum - p).astype(np.float64) / 2.0**32
    ks = float(max(np.max(np.abs(after - phi)), np.max(np.abs(before - phi))))
    tails = []
    for t in TAIL_T:
        # P(Z > t): the magnitudes above t with a positive sign.
        above = sum(ci for ci, mi in zip(c, m) if mi > t * Q17)
        tails.append({"t": t, "p_gen": above / 2.0**32, "p_phi": float(special.ndtr(-t))})
    return {
        "max_abs_err_vs_ndtri": max_err,
        "max_abs_err_at_z": where,
        "zmax": 830729 / Q17,
        "variance": float(var),
        "kurtosis": float(kurt),
        "ks_vs_phi": ks,
        "tails": tails,
    }


# ---------------------------------------------------------------------------------------------
# Sample cases (section 3 output formats).


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def q15_scale(sigma32):
    """``(ms, sh)`` of section 3: ``sigma = ms 2^(2 - sh)`` with the 24-bit significand ``ms``;
    ``ms = 0`` for sigma 0 (or below 1e-30) and for ``sh >= 45``."""
    sigma64 = float(sigma32)
    if sigma64 < 1e-30:
        return 0, 21
    bits = f32_bits(sigma32)
    exp = (bits >> 23) & 0xFF
    assert exp != 0, "sigma above 1e-30 is a normal f32"
    ms = (bits & 0x7FFFFF) | 0x800000
    sh = 2 - (exp - 127 - 23)
    if sh >= 45:
        return 0, 21
    return ms, sh


def q15_value(zq, ms, sh):
    """The R7 value ``sat16(floor(zq 2^-17 sigma 32768 + 1/2))`` in integers."""
    a = abs(zq) * ms
    half = 1 << (sh - 1)
    v = (a + half) >> sh if zq >= 0 else -((a + half - 1) >> sh)
    return max(-32768, min(32767, v))


def draws(seed, stream, start, count):
    rng = Pcg32(seed, stream)
    rng.advance(start & MASK)
    return [rng.next_u32() for _ in range(count)]


def snr_db(ideal, got):
    sig = math.fsum(float(v) * float(v) for v in ideal)
    err = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(ideal, got))
    return math.inf if err == 0.0 else 10.0 * math.log10(sig / err)


def run(seed, stream, sigma, start, n, sample_type):
    self_test_pcg()
    entries, _ = table()
    self_test_normal(entries)
    complex_ = sample_type in ("c32", "cq15")
    d = 2 if complex_ else 1
    sigma32 = np.float32(sigma)
    sigma64 = float(sigma32)
    rs = draws(seed, stream, start * d, n * d)
    zq = []
    ideal = []
    for r in rs:
        z, x = normal_q17(entries, r)
        zq.append(z)
        mag = -float(special.ndtri(math.ldexp(float(x), -33)))
        ideal.append(sigma64 * (-mag if r >> 31 else mag))
    # The exact value: zq has at most 20 bits and sigma 24, so the float64 product is exact.
    v = [z * 2.0**-17 * sigma64 for z in zq]
    assert all(Fraction(a) == Fraction(z, Q17) * Fraction(sigma64) for a, z in zip(v, zq[:256]))
    params = {
        "generator_seed": seed,
        "stream": stream,
        "sigma": sigma64,
        "sigma_bits": f32_bits(sigma32),
        "start_index": start,
        "len": n,
        "sample_type": sample_type,
    }
    if sample_type in ("c32", "f32"):
        out = np.array(v, dtype=np.float64).astype(np.float32)
        got = [float(o) for o in out]
        bound = 5.7e-5 * sigma64 + 2.5e-7 * sigma64
        inside = list(range(len(v)))
    else:
        ms, sh = q15_scale(sigma32)
        raw = [q15_value(z, ms, sh) for z in zq]
        ref = q15.from_float(np.array(v, dtype=np.float64))
        assert [int(a) for a in ref] == raw, "the integer Q15 path differs from q15.from_float"
        got = [r / 32768.0 for r in raw]
        bound = (0.5 + 5.7e-5 * sigma64 * 32768.0) / 32768.0
        # Compared only where the ideal value lies inside full scale.
        inside = [i for i, x in enumerate(ideal) if -1.0 <= x <= 32767.0 / 32768.0]
        params["clipped"] = sum(1 for r in raw if r in (32767, -32768))
    err = max(abs(got[i] - ideal[i]) for i in inside)
    assert err <= bound, (sample_type, err, bound)
    snr = snr_db([ideal[i] for i in inside], [got[i] for i in inside])
    guard = math.floor((snr - 1.0) * 10.0) / 10.0
    params.update(
        {
            "max_abs_err_ideal": bound,
            "measured_max_abs_err_ideal": err,
            "snr_db_ideal": snr,
            "snr_db_guard": guard,
            "compared_ideal": len(inside),
        }
    )
    if sample_type == "c32":
        expected = Array("cf32", out[0::2] + 1j * out[1::2].astype(np.complex64))
        ideal_arr = Array("cf64", np.array(ideal[0::2]) + 1j * np.array(ideal[1::2]))
    elif sample_type == "f32":
        expected = Array("f32", out)
        ideal_arr = Array("f64", np.array(ideal, dtype=np.float64))
    elif sample_type == "cq15":
        expected = Array("ci16", q15.pack(raw[0::2], raw[1::2]))
        ideal_arr = Array("cf64", np.array(ideal[0::2]) + 1j * np.array(ideal[1::2]))
    else:
        expected = Array("i16", np.array(raw, dtype=np.int64))
        ideal_arr = Array("f64", np.array(ideal, dtype=np.float64))
    notes = (
        "PCG32, one draw per real value (complex: re then im); inverse CDF from the ndtri table "
        "of section 2.2 in integers; v = zq 2^-17 float64(float32(sigma)); f32 = v rounded once, "
        "Q15 = floor(v 32768 + 0.5) saturated (R7) in integers; ideal = sigma sign "
        "(-ndtri(x / 2^33)); the ideal error is compared where the ideal lies inside full scale"
    )
    return dict(
        params=params,
        arrays={"expected": expected, "ideal": ideal_arr},
        tolerance={"class": "bit_exact"},
        notes=notes,
    )


@case(kernel="gauss_noise", name="c32_unit_4096", seed=680)
def c32_unit(rng):
    return run(680, 1, 1.0, 0, 4096, "c32")


@case(kernel="gauss_noise", name="c32_seek_4096", seed=681)
def c32_seek(rng):
    return run(681, 7, 0.3, 1_000_000_007, 4096, "c32")


@case(kernel="gauss_noise", name="f32_unit_4096", seed=682)
def f32_unit(rng):
    return run(682, 1, 1.0, 0, 4096, "f32")


@case(kernel="gauss_noise", name="cq15_quarter_4096", seed=683)
def cq15_quarter(rng):
    return run(683, 2, 0.25, 0, 4096, "cq15")


@case(kernel="gauss_noise", name="q15_clip_4096", seed=684)
def q15_clip(rng):
    out = run(684, 3, 0.6, 0, 4096, "q15")
    # sigma 0.6 must saturate a noticeable share of the values (about 9.6 %).
    assert out["params"]["clipped"] > 200, out["params"]["clipped"]
    return out


@case(kernel="gauss_noise", name="icdf_table", seed=0)
def icdf_table(rng):
    self_test_pcg()
    entries, props = table()
    self_test_normal(entries)
    numbers = exact_numbers(entries)
    params = {"len": len(entries)}
    params.update(props)
    params.update(numbers)
    notes = (
        "GAUSS_ICDF of section 2.2 from scipy.special.ndtri (Q17 knots, round half up, own tie "
        "guard); entries are u32 stored with the bits of i32; the distribution numbers are exact "
        "over all 2^31 magnitude cells (tails: P(Z > t) of the generator and Phi(-t))"
    )
    raw = np.array(entries, dtype=np.uint32).view(np.int32).astype(np.int64)
    return dict(
        params=params,
        arrays={"entries": Array("i32", raw)},
        tolerance={"class": "bit_exact"},
        notes=notes,
    )
