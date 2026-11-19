"""Block cases of the self-synchronizing scrambler (M44.2, the maintainer's decision on I44.2-6):
``bit_scramble`` and ``bit_descramble`` against a one-bit-at-a-time model of the shift register.

The model is the textbook multiplicative scrambler of the polynomial ``1 + x^-b + x^-a`` (ITU-T
V.32 to V.34: ``1 + x^-18 + x^-23``), written independently of the Rust code: a register of the
last ``A = a m`` bits (``m`` the bits per symbol of the constellation, so the delays count
symbols), zero at the start. The scrambler sends ``s[n] = x[n] ^ s[n - A] ^ s[n - B]`` and keeps
the bits it sends; the descrambler gives ``y[n] = r[n] ^ r[n - A] ^ r[n - B]`` and keeps the bits
it receives. The bits are soft bits (``> 0`` is a 1): the output is the input with its sign bit
flipped where the parity of the register is 1, so the magnitudes stay; a 0 input counts as a 0
bit. No arithmetic is done on the values (a sign flip is exact), so the case is bit-exact.
"""

import numpy as np

from ..registry import Array, case

SIGN = np.uint32(0x8000_0000)


def flip(v, p):
    """``v`` (float32) with its sign bit flipped when ``p`` is 1."""
    bits = np.array([v], dtype=np.float32).view(np.uint32)
    if p:
        bits ^= SIGN
    return bits.view(np.float32)[0]


def run(x, a, b, descramble):
    """The scrambler (``descramble`` False) or the descrambler over the float32 soft bits ``x``
    with the delays ``a > b`` in bits; the register as a Python list of the bits so far."""
    reg = []
    out = np.empty_like(x)
    for n, v in enumerate(x):
        p = (reg[n - a] if n >= a else 0) ^ (reg[n - b] if n >= b else 0)
        y = flip(v, p)
        out[n] = y
        reg.append(int(v > 0) if descramble else int(y > 0))
    return out


def soft_bits(rng, n):
    """``n`` soft bits of random sign and magnitude 0.05 to 2, with a 0 and a negative 0."""
    mag = rng.uniform(0.05, 2.0, n)
    sign = np.where(rng.integers(0, 2, n) == 1, 1.0, -1.0)
    x = (mag * sign).astype(np.float32)
    x[100] = np.float32(0.0)
    x[1000] = np.float32(-0.0)
    return x


def make(rng, polynomial, taps, constellation, bits_per_symbol, n):
    a, b = taps[0] * bits_per_symbol, taps[1] * bits_per_symbol
    x = soft_bits(rng, n)
    scrambled = run(x, a, b, descramble=False)
    descrambled = run(x, a, b, descramble=True)
    # The round trip, and a run of ones becomes a pseudo-random stream.
    assert np.array_equal(run(scrambled, a, b, True)[np.abs(x) > 0], x[np.abs(x) > 0])
    ones = run(np.ones(4 * a + 2000, dtype=np.float32), a, b, False)[4 * a:]
    assert 0.4 < np.mean(ones > 0) < 0.6
    return dict(
        params={
            "polynomial": polynomial,
            "constellation": constellation,
            "delay_a": a,
            "delay_b": b,
            "bits": n,
        },
        arrays={
            "input": Array("f32", x),
            "scrambled": Array("f32", scrambled),
            "descrambled": Array("f32", descrambled),
        },
        tolerance={"class": "bit_exact"},
        model="float32",
        notes=f"bit_scramble and bit_descramble, polynomial 1 + x^-{taps[1]} + x^-{taps[0]} over "
        f"symbols of {bits_per_symbol} bits (delays {a} and {b} bits): input are {n} soft bits of "
        "random sign and magnitude 0.05 to 2 (bit 100 is 0, bit 1000 is -0), scrambled is "
        "bit_scramble of input (s[n] = x[n] ^ s[n - A] ^ s[n - B], the register holds the bits "
        "sent), descrambled is bit_descramble of input (y[n] = r[n] ^ r[n - A] ^ r[n - B], the "
        "register holds the bits received); the output is the input with its sign flipped where "
        "the parity is 1, a bit is 1 when the value is above 0; one-bit-at-a-time model of the "
        "register, zero at the start; exact",
    )


@case(kernel="blocks", name="bit_scramble_x23_x18_bpsk_f32", seed=880)
def bit_scramble_x23_x18_bpsk_f32(rng):
    return make(rng, "x23_x18", (23, 18), "bpsk", 1, 4096)


@case(kernel="blocks", name="bit_scramble_x23_x18_qam256_f32", seed=881)
def bit_scramble_x23_x18_qam256_f32(rng):
    return make(rng, "x23_x18", (23, 18), "qam256", 8, 8192)
