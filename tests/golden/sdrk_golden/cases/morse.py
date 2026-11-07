"""Cases for Morse code (M47): the units of the encoder, the key envelope of the encoder block
and keyed envelopes with noise for the decoder block.

The reference is written from ITU-R M.1677-1 and the Farnsworth formula of the ARRL, without
looking at the Rust code: its own table (a dictionary of codes), its own unit expansion and its
own gap rounding. The units of a text are the key states (1 key down, 0 key up) one per unit;
the gap before a character is 3 units, before a character after a space 7 units; Farnsworth
timing stretches the gaps to ``3 ta / 19`` and ``7 ta / 19`` seconds of the ``ta`` seconds that
a word of 31 character units leaves at the effective speed, counted in units of the character
speed and rounded to whole units (at least 3 and 7).

Layout of the unit cases: ``data`` holds the texts back to back (``u8``), ``lengths`` their
lengths, ``units`` the units of all texts back to back (``u8``, 0 or 1) and ``unit_counts``
their numbers. The envelope case adds the key envelope ``envelope`` (``f32``) of the packets
in order, each followed by its word gap; unit ``k`` ends at the sample
``floor((k + 1) * spu + 0.5)`` with ``spu = fs * 1.2 / wpm``. The decoder case holds three
transmissions with noise (``envelope``, ``f32``), their start offsets in samples and the
words that were sent (``words``, ``word_lengths``).
"""

import math

import numpy as np

from ..registry import Array, case

ITU = (
    "A .- B -... C -.-. D -.. E . F ..-. G --. H .... I .. J .--- K -.- L .-.. "
    "M -- N -. O --- P .--. Q --.- R .-. S ... T - U ..- V ...- W .-- X -..- Y -.-- Z --.. "
    "0 ----- 1 .---- 2 ..--- 3 ...-- 4 ....- 5 ..... 6 -.... 7 --... 8 ---.. 9 ----. "
    ". .-.-.- , --..-- : ---... ? ..--.. ' .----. - -....- / -..-. ( -.--. ) -.--.- "
    '" .-..-. = -...- + .-.-. @ .--.-.'
)
SK_BYTE = 0x04
SK_CODE = "...-.-"


def _table():
    parts = ITU.split()
    assert len(parts) % 2 == 0
    table = {ord(parts[i]): parts[i + 1] for i in range(0, len(parts), 2)}
    table[SK_BYTE] = SK_CODE
    assert len(table) == 50 and len(set(table.values())) == 50
    return table


CODES = _table()
SPACES = (0x20, 0x09, 0x0D, 0x0A)
ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:?'-/()\"=+@abcdefghijklmnopqrstuvwxyz #%!\x04"


def code_of(byte):
    """The code of a byte (lower case folds to upper case), or None."""
    if ord("a") <= byte <= ord("z"):
        byte -= 32
    return CODES.get(byte)


def gaps(wpm, char_wpm):
    """Whole units of the gap between characters and between words."""
    if wpm == char_wpm:
        return 3, 7
    ta = (60.0 * char_wpm - 37.2 * wpm) / (wpm * char_wpm)  # seconds of gap in a word
    unit = 1.2 / char_wpm
    cg = int(3.0 * ta / 19.0 / unit + 0.5)
    wg = int(7.0 * ta / 19.0 / unit + 0.5)
    return max(3, cg), max(7, wg)


def units_of(text, char_gap=3, word_gap=7):
    """The units of one text: key states, no gap in front of the first character."""
    out = []
    sent = False
    word = False
    for b in text:
        if b in SPACES:
            word = True
            continue
        code = code_of(b)
        if code is None:
            continue
        if sent:
            out += [0] * (word_gap if word else char_gap)
        sent = True
        word = False
        for i, e in enumerate(code):
            if i:
                out.append(0)
            out += [1] * (3 if e == "-" else 1)
    return out


def _texts(rng, count):
    fixed = [
        b"PARIS",
        b"PARIS PARIS",
        bytes(ALPHABET[:26]),
        bytes(ALPHABET[26:36]),
        b".,:?'-/()\"=+@",
        b"The Quick Brown Fox",
        b"  leading   and  trailing  ",
        b"A#B%C!D",
        bytes([SK_BYTE]),
        b"E",
        b"T",
    ]
    texts = list(fixed)
    for n in rng.integers(1, 61, size=count).tolist():
        texts.append(bytes(rng.choice(np.frombuffer(ALPHABET, dtype=np.uint8), size=n).tolist()))
    return texts


def _pack(texts, units, extra=None):
    arrays = {
        "data": Array("u8", np.frombuffer(b"".join(texts), dtype=np.uint8)),
        "lengths": Array("i32", np.array([len(t) for t in texts], dtype=np.int32)),
        "units": Array("u8", np.array([u for us in units for u in us], dtype=np.uint8)),
        "unit_counts": Array("i32", np.array([len(us) for us in units], dtype=np.int32)),
    }
    arrays.update(extra or {})
    return arrays


@case("morse", "encode_units", 650)
def encode_units(rng):
    texts = _texts(rng, 29)
    units = [units_of(t) for t in texts]
    assert len(units_of(b"PARIS E")) == 43 + 7 + 1
    return dict(
        params={"texts": len(texts), "char_gap": 3, "word_gap": 7},
        arrays=_pack(texts, units),
        tolerance={"class": "bit_exact"},
        notes="40 texts (11 fixed, 29 random of 1 to 60 bytes from letters of both cases, digits, "
        "punctuation, spaces, the byte 0x04 (SK) and bytes without a code); units 1 key down, "
        "0 key up; no gap in front of the first character; ITU-R M.1677-1 written out in "
        "morse.py",
    )


@case("morse", "encode_units_farnsworth", 651)
def encode_units_farnsworth(rng):
    pairs = [(10.0, 18.0), (12.0, 25.0), (5.0, 20.0), (20.0, 40.0), (10.0, 13.0), (30.0, 30.0)]
    texts = _texts(rng, 0)[:5] + [bytes(rng.choice(np.frombuffer(ALPHABET, dtype=np.uint8), size=40).tolist())]
    units = []
    cg, wg = [], []
    for (w, c), t in zip(pairs, texts):
        g = gaps(w, c)
        cg.append(g[0])
        wg.append(g[1])
        units.append(units_of(t, *g))
    return dict(
        params={"pairs": len(pairs)},
        arrays=_pack(
            texts,
            units,
            {
                "wpm": Array("f32", np.array([p[0] for p in pairs], dtype=np.float32)),
                "char_wpm": Array("f32", np.array([p[1] for p in pairs], dtype=np.float32)),
                "char_gap": Array("i32", np.array(cg, dtype=np.int32)),
                "word_gap": Array("i32", np.array(wg, dtype=np.int32)),
            },
        ),
        tolerance={"class": "bit_exact"},
        notes="6 texts at (wpm, char_wpm) 10/18, 12/25, 5/20, 20/40, 10/13 and 30/30; the gaps "
        "from ta = (60 c - 37.2 s) / (s c) seconds, 3 ta / 19 and 7 ta / 19 in units of the "
        "character speed, rounded half up to whole units",
    )


FS_ENVELOPE = 1000
WPM_ENVELOPE = 17.0


@case("morse", "encode_envelope", 652)
def encode_envelope(rng):
    texts = [b"CQ CQ DE W1AW", b"73", b"The quick brown fox"]
    units = []
    for t in texts:
        u = units_of(t)
        # The block ends every packet with one word gap and sends the next packet without a
        # further gap in front.
        units.append(u + [0] * 7)
    flat = [x for us in units for x in us]
    spu = FS_ENVELOPE * 1.2 / WPM_ENVELOPE
    env = []
    for k, on in enumerate(flat):
        end = math.floor((k + 1) * spu + 0.5)
        env += [float(on)] * (end - len(env))
    return dict(
        params={"fs": FS_ENVELOPE, "wpm": WPM_ENVELOPE, "rise_ms": 0.0},
        arrays=_pack(texts, units, {"envelope": Array("f32", np.array(env, dtype=np.float32))}),
        tolerance={"class": "bit_exact"},
        notes="3 packets at 17 wpm and 1000 Hz (70.59 samples a unit), hard keying; each packet "
        "ends with its word gap of 7 units; unit k ends at floor((k + 1) spu + 0.5)",
    )


FS_DECODE = 1000
DECODE_RUNS = [
    (6.0, b"VVV CQ CQ DE W1AW K"),
    (12.0, b"VVV TEST 599 QSL"),
    (25.0, b"VVV THE QUICK BROWN FOX 73 SK"),
]


@case("morse_decode", "words", 653)
def decode_words(rng):
    sigma = 0.05
    parts = []
    starts = []
    words = []
    n = 0
    for wpm, text in DECODE_RUNS:
        spu = FS_DECODE * 1.2 / wpm
        lead = [0] * 6
        u = lead + units_of(text) + [0] * 14
        env = []
        for k, on in enumerate(u):
            end = math.floor((k + 1) * spu + 0.5)
            env += [float(on)] * (end - len(env))
        starts.append(n)
        # Silence of more than 20 word gaps between the transmissions (22): the decoder forgets
        # the speed.
        if (wpm, text) != DECODE_RUNS[-1]:
            env += [0.0] * int(22 * 7 * spu)
        n += len(env)
        parts.append(np.array(env, dtype=np.float32))
        words += text.split()
    env = np.concatenate(parts)
    env = env + rng.normal(0.0, sigma, size=env.shape).astype(np.float32)
    flat = b"".join(words)
    return dict(
        params={"fs": FS_DECODE, "sigma": sigma, "transmissions": len(DECODE_RUNS)},
        arrays={
            "envelope": Array("f32", env.astype(np.float32)),
            "starts": Array("i32", np.array(starts, dtype=np.int32)),
            "wpm": Array("f32", np.array([r[0] for r in DECODE_RUNS], dtype=np.float32)),
            "words": Array("u8", np.frombuffer(flat, dtype=np.uint8)),
            "word_lengths": Array("i32", np.array([len(w) for w in words], dtype=np.int32)),
        },
        tolerance={"class": "bit_exact"},
        notes="3 transmissions at 6, 12 and 25 wpm at 1000 Hz, hard keying plus Gaussian noise "
        "of 0.05 (26 dB below the key down level), each but the last followed by 22 word gaps of "
        "silence; the words that were sent",
    )
