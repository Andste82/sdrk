"""Independent model of the PRBS generators of ``sdrk_proto::prbs`` (Fibonacci LFSR).

The register holds the last ``n`` output bits (bit ``k`` of the state is stage ``k + 1``,
``n`` the degree). Each step XORs the two tapped stages ``n`` and ``t``; the result is the
output bit and becomes the new stage 1. The seed is the initial register content (only the low
``n`` bits are used) and must be non-zero after masking.

Written from the description of the register in ``crates/proto/sdrk-proto/src/prbs.rs`` (M38 §4)
and the polynomials of M49 §6; it does not run any code of the repository.
"""

# name: (degree n, second tap t) of x^n + x^t + 1
POLYS = {
    "prbs5": (5, 3),
    "prbs7": (7, 6),
    "prbs9": (9, 5),
    "prbs10": (10, 7),
    "prbs11": (11, 9),
    "prbs15": (15, 14),
}


class Prbs:
    """A PRBS generator of one of the polynomials in ``POLYS``."""

    def __init__(self, name, seed):
        self.n, self.t = POLYS[name]
        self.mask = (1 << self.n) - 1
        self.state = seed & self.mask
        if self.state == 0:
            raise ValueError("prbs seed must be non-zero")

    def next_bit(self):
        bit = ((self.state >> (self.n - 1)) ^ (self.state >> (self.t - 1))) & 1
        self.state = ((self.state << 1) | bit) & self.mask
        return bit

    def bits(self, count):
        return [self.next_bit() for _ in range(count)]


def self_test():
    """Every polynomial has the full period ``2^n - 1`` (every non-zero state once) and
    ``2^(n-1)`` ones per period; the first bits of the M38 known answers."""
    for name, (n, _) in POLYS.items():
        p = Prbs(name, 1)
        seen = set()
        ones = 0
        for _ in range((1 << n) - 1):
            seen.add(p.state)
            ones += p.next_bit()
        assert len(seen) == (1 << n) - 1 and p.state == 1, name
        assert ones == 1 << (n - 1), name
    # sdrk-proto/tests/prbs.rs (M38): PRBS9 from 0x1ff starts 0 0 0 0 0 1 1 1 1 0 1 1.
    assert Prbs("prbs9", 0x1FF).bits(12) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]


self_test()
