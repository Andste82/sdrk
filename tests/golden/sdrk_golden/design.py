"""Filter design for the golden cases of the blocks: a port of ``sdrk_blocks::design`` that calls SciPy.

SciPy is the reference. The Rust design must equal these functions (tolerance 1e-12 on the taps,
exact on the tap count).
"""

import math

import numpy as np
import scipy.signal

# Number of phases of the tuned resampler (sdrk_blocks::TUNED_PHASES).
PHASES = 128


def kaiserord(ripple_db, transition_hz, fs_hz):
    """``scipy.signal.kaiserord(ripple, 2 * transition / fs)`` with the tap count forced odd."""
    n, beta = scipy.signal.kaiserord(ripple_db, 2.0 * transition_hz / fs_hz)
    n = int(n)
    if n % 2 == 0:
        n += 1
    return n, float(beta)


def lowpass(cutoff_hz, transition_hz, attenuation_db, fs_hz):
    """Taps of a block filter: ``kaiserord`` then ``firwin`` with a Kaiser window (float64)."""
    n, beta = kaiserord(attenuation_db, transition_hz, fs_hz)
    return scipy.signal.firwin(n, cutoff_hz, window=("kaiser", beta), fs=fs_hz)


def tuned_bank(cutoff_hz, transition_hz, attenuation_db, rate_in_hz):
    """Prototype of the tuned resampler: ``P * K`` taps at ``P * rate_in`` with gain ``P``.

    Returns (K, bank) where ``bank[p][i] = prototype[p + i * P]`` (float64, shape (P, K)).
    """
    n_hi, beta = kaiserord(attenuation_db, transition_hz, PHASES * rate_in_hz)
    k = -(-n_hi // PHASES)
    proto = scipy.signal.firwin(PHASES * k, cutoff_hz, window=("kaiser", beta), fs=PHASES * rate_in_hz) * PHASES
    bank = np.stack([proto[p::PHASES] for p in range(PHASES)])
    assert bank.shape == (PHASES, k)
    return k, bank


def tuned_step(rate_in_hz, rate_out_hz, ppm):
    """``round(2^32 * rate_in / rate_out / (1 + ppm * 1e-6))`` with the same operation order as Rust."""
    return int(math.floor(4294967296.0 * rate_in_hz / rate_out_hz / (1.0 + ppm * 1e-6) + 0.5))


def tuned_positions(count_in, step):
    """Positions of the outputs of the tuned resampler for ``count_in`` input samples.

    Returns a list of (n, p, mu, wrap) for every output whose window is complete; output ``j`` is at
    the 32.32 position ``j * step`` (input samples, the first output at input sample 0).
    """
    out = []
    j = 0
    while True:
        pos = j * step
        n = pos >> 32
        frac = pos & 0xFFFFFFFF
        p = frac >> 25
        mu = (frac >> 9) & 0xFFFF
        wrap = p + 1 == PHASES
        if n + 1 + (1 if wrap else 0) > count_in:
            break
        out.append((n, p, mu, wrap))
        j += 1
    return out
