"""Reference signals for the example flows of M39.1 (Tests 3 and 5).

``flows/aprs_bell202_20_frames`` (seed 542): a synthetic Bell 202 AFSK signal at 48 kHz with 20
AX.25 UI frames for the flow ``aprs_receiver``. It is made here, independently of the Rust
``fsk_mod`` and ``hdlc_encode``: an HDLC model of its own (flags 0x7E, bit stuffing after five
1s, bytes least significant bit first, the FCS CRC-16/X-25 of ``cases/proto.py`` low byte first),
NRZI (a 0 changes the tone, a 1 keeps it) and a continuous-phase tone generator, one sample at
a time with ``math.sin``. Every frame has its own imperfections, as different stations on one
channel have: 4 to 12 opening flags and 1 to 3 closing flags, a bit rate off by up to 200 ppm,
both tones off by up to 0.5 % (the same factor), a twist (space level over mark level) of -3 to
+3 dB, an amplitude of 0.25 to 0.6 and a random start phase. Between the frames are 20 to 60 ms
of silence. White Gaussian noise of sigma 0.01 (-40 dB of full scale) is on the whole signal,
which is then quantized to signed 8 bits (value / 128) to stay within 512 KiB (D695).

``flows/adsb_df17_200_frames`` (seed 544): 75 ms of a 2 MHz ``cu8`` baseband (150000 samples)
with 200 DF 17 extended squitters at random positions (a gap of at least 256 samples after each
frame, so that the median window of ``adsb_decode`` before a preamble holds no other frame),
random amplitudes 10 to 30 dB above the noise (uniform in dB; the SNR of a frame is the power of
a pulse sample over the complex noise power per sample, ``A^2 / (2 sigma^2)``), a random carrier
phase and a carrier offset of up to 100 kHz per frame, pulses of one sample at the preamble
samples ``p, p + 2, p + 7, p + 9`` and at ``p + 16 + 2 k`` (bit ``k`` = 1) or ``p + 17 + 2 k``
(bit ``k`` = 0), complex white Gaussian noise of sigma 0.018 per component and 40 noise bursts
(32 to 1024 samples, 0 to 12 dB above the noise) at random positions that overlap some frames.
Each frame records its effective SNR: its pulse power over the noise power with the strongest
burst that overlaps the window ``p - 256 .. p + 240`` (the median window and the frame); the
test requires every frame whose effective SNR is above 10 dB (D696). ``cu8`` is
``round(x 128 + 127.5)`` saturated to 0 ... 255, the inverse of the ``cu8`` mapping of
``docs/device.md``.

``flows/adsb_short_frames`` (seed 545): 24000 samples at 2 MHz, all frames at 20 to 30 dB: first
6 DF 17 extended squitters (their addresses fill the ICAO address cache of ``adsb_decode``) and
one DF 11 all-call reply with interrogator code 0 (its parity is the plain CRC-24, residue 0; its
address is cached too), then in random order 20 short frames (DF 4, DF 5 and DF 11, each with its
parity overlaid with one of these 7 addresses) and 4 short frames whose address was not seen
before. ``adsb_decode`` outputs the DF 17 and the checked DF 11 frame with ``CRC_OK``, the 20 known
short frames with flag bit 16 and drops the 4 unknown ones (D702, D704).

The frames are drawn with NumPy integers; the samples are computed one value at a time with
``math`` and plain Python arithmetic (NumPy only draws the uniform and normal values), so the
files are identical on every CPU.
"""

import math

import numpy as np

from ..registry import Array, case
from .proto import crc16_x25, crc24_adsb

# ---------------------------------------------------------------------------------------------
# Bell 202 AFSK with HDLC (AX.25)

AFSK_RATE = 48000
BAUD = 1200.0
MARK_HZ = 1200.0
SPACE_HZ = 2200.0
AFSK_NOISE_SIGMA = 0.01
AFSK_SCALE = 1.0 / 128.0
FLAG_BITS = [0, 1, 1, 1, 1, 1, 1, 0]


def ax25_address(call, ssid, ch, last):
    """The 7 bytes of an AX.25 address field (AX.25 2.2, section 3.12)."""
    padded = call.ljust(6)
    out = bytes((ord(c) << 1) & 0xFF for c in padded)
    return out + bytes([0x60 | (ssid << 1) | (0x80 if ch else 0) | (1 if last else 0)])


def random_call(rng):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = int(rng.integers(4, 7))
    s = [letters[int(rng.integers(0, 26))], letters[int(rng.integers(0, 26))], str(int(rng.integers(0, 10)))]
    while len(s) < n:
        s.append(letters[int(rng.integers(0, 26))])
    return "".join(s)


def random_info(rng):
    """APRS information fields: a position report, a status or a message."""
    kind = int(rng.integers(0, 3))
    if kind == 0:
        lat = int(rng.integers(0, 9000))
        lat_min = int(rng.integers(0, 6000))
        lon = int(rng.integers(0, 18000))
        lon_min = int(rng.integers(0, 6000))
        ns = "NS"[int(rng.integers(0, 2))]
        ew = "EW"[int(rng.integers(0, 2))]
        return f"!{lat // 100:02d}{lat_min // 100:02d}.{lat_min % 100:02d}{ns}/{lon // 100:03d}{lon_min // 100:02d}.{lon_min % 100:02d}{ew}-"
    text = "".join(chr(32 + int(rng.integers(0, 95))) for _ in range(int(rng.integers(3, 13))))
    if kind == 1:
        return ">" + text
    return ":" + random_call(rng).ljust(9) + ":" + text


def aprs_frame(rng):
    """One AX.25 UI frame without FCS: APRS, a random source, 0 to 2 digipeaters, 0x03 0xF0."""
    digis = [("WIDE1", 1), ("WIDE2", 1)][: int(rng.integers(0, 3))]
    addrs = [("APRS", 0, True), (random_call(rng), int(rng.integers(0, 16)), False)]
    addrs += [(c, s, False) for c, s in digis]
    out = b""
    for i, (c, s, ch) in enumerate(addrs):
        out += ax25_address(c, s, ch, i == len(addrs) - 1)
    return out + bytes([0x03, 0xF0]) + random_info(rng).encode("ascii")


def hdlc_bits(frame, preamble_flags, tail_flags):
    """Flags, the frame and its FCS (CRC-16/X-25, low byte first) with bit stuffing, flags."""
    fcs = crc16_x25(frame)
    data = frame + bytes([fcs & 0xFF, fcs >> 8])
    bits = FLAG_BITS * preamble_flags
    ones = 0
    for byte in data:
        for k in range(8):
            b = (byte >> k) & 1
            bits.append(b)
            ones = ones + 1 if b else 0
            if ones == 5:
                bits.append(0)
                ones = 0
    return bits + FLAG_BITS * tail_flags


def nrzi(bits, level):
    """NRZI: a 0 toggles the level (tone), a 1 keeps it."""
    out = []
    for b in bits:
        if b == 0:
            level ^= 1
        out.append(level)
    return out


@case("flows", "aprs_bell202_20_frames", 542)
def aprs_bell202_20_frames(rng):
    frames, starts, info = [], [], []
    audio = [0.0] * int(0.05 * AFSK_RATE)  # 50 ms of silence before the first frame
    for _ in range(20):
        frame = aprs_frame(rng)
        pre, tail = int(rng.integers(4, 13)), int(rng.integers(1, 4))
        baud = BAUD * (1.0 + float(rng.uniform(-200e-6, 200e-6)))
        tone = 1.0 + float(rng.uniform(-0.005, 0.005))
        twist_db = float(rng.uniform(-3.0, 3.0))
        amp = float(rng.uniform(0.25, 0.6))
        phase = float(rng.uniform(0.0, 1.0))
        frac = float(rng.uniform(0.0, 1.0))
        levels = nrzi(hdlc_bits(frame, pre, tail), int(rng.integers(0, 2)))
        gap = int(rng.integers(int(0.02 * AFSK_RATE), int(0.06 * AFSK_RATE) + 1))
        # Bit k covers the samples n with k <= (n + frac) baud / rate < k + 1; level 1 is mark.
        spb = AFSK_RATE / baud
        n_samples = int(math.ceil(len(levels) * spb - frac))
        amp_space = amp * 10.0 ** (twist_db / 20.0)
        starts.append(len(audio))
        for n in range(n_samples):
            k = min(int((n + frac) / spb), len(levels) - 1)
            mark = levels[k] == 1
            f = (MARK_HZ if mark else SPACE_HZ) * tone / AFSK_RATE
            audio.append((amp if mark else amp_space) * math.sin(2.0 * math.pi * phase))
            phase = math.fmod(phase + f, 1.0)
        audio.extend([0.0] * gap)
        frames.append(frame)
        info.append(
            {
                "preamble_flags": pre,
                "tail_flags": tail,
                "baud": round(baud, 4),
                "tone_factor": round(tone, 6),
                "twist_db": round(twist_db, 2),
                "amplitude": round(amp, 4),
            }
        )
    noise = rng.standard_normal(len(audio)).tolist()
    raw = []
    for v, z in zip(audio, noise):
        q = math.floor((v + AFSK_NOISE_SIGMA * z) * 128.0 + 0.5)
        raw.append(max(-128, min(127, q)))
    return dict(
        params={
            "rate_hz": AFSK_RATE,
            "scale": AFSK_SCALE,
            "noise_sigma": AFSK_NOISE_SIGMA,
            "frame_starts": starts,
            "frames": info,
        },
        arrays={
            "audio": Array("i8", np.array(raw, dtype=np.int8)),
            "frames": Array("u8", np.frombuffer(b"".join(frames), dtype=np.uint8)),
            "lengths": Array("i32", np.array([len(f) for f in frames], dtype=np.int32)),
        },
        tolerance={"class": "bit_exact"},
        notes="Bell 202 AFSK (mark 1200 Hz, space 2200 Hz, 1200 Bd, NRZI, HDLC with CRC-16/X-25) of 20 "
        "AX.25 UI frames at 48 kHz; per frame 4 to 12 opening and 1 to 3 closing flags, bit rate within "
        "200 ppm, tones within 0.5 %, twist -3 to +3 dB, amplitude 0.25 to 0.6, random start phase, "
        "20 to 60 ms of silence after it; noise sigma 0.01 on everything; audio is i8, value = raw * "
        "scale; frames holds the frames without FCS back to back, lengths their lengths; frame_starts "
        "is the first sample of each frame's flags; the receiver must decode all 20 frames",
    )


# ---------------------------------------------------------------------------------------------
# Mode S / ADS-B at 2 MHz

ADSB_RATE = 2000000
ADSB_NOISE_SIGMA = 0.018
LONG_SAMPLES = 16 + 2 * 112
SHORT_SAMPLES = 16 + 2 * 56
MEDIAN_WINDOW = 256
PREAMBLE = (0, 2, 7, 9)


def mode_s_frame(rng, df, nbytes, icao=None):
    """A Mode S downlink frame of ``nbytes`` bytes; DF 17 gets CA 5, a random address and ME and
    the CRC-24 parity, the short formats DF 4, 5 and 11 their parity overlaid with the address
    (``icao``, random if not given)."""
    if icao is None:
        icao = int(rng.integers(0, 1 << 24))
    if df == 17:
        head = bytes([(df << 3) | 5]) + icao.to_bytes(3, "big")
        body = head + rng.integers(0, 256, size=7, dtype=np.uint8).tobytes()
        return body + crc24_adsb(body).to_bytes(3, "big"), icao
    if df == 11:
        body = bytes([(df << 3) | 5]) + icao.to_bytes(3, "big")
    else:
        body = bytes([(df << 3) | int(rng.integers(0, 8))]) + rng.integers(0, 256, size=3, dtype=np.uint8).tobytes()
    assert len(body) == nbytes - 3
    return body + (crc24_adsb(body) ^ icao).to_bytes(3, "big"), icao


def frame_pulses(frame):
    """The sample offsets of the pulses: the preamble and one pulse per bit."""
    pulses = list(PREAMBLE)
    for k in range(8 * len(frame)):
        bit = (frame[k // 8] >> (7 - k % 8)) & 1
        pulses.append(16 + 2 * k + (0 if bit else 1))
    return pulses


def place(rng, n_total, count, length, lead):
    """Random frame starts: ``lead`` samples before the first, ``length + MEDIAN_WINDOW`` between
    the starts at least, the slack spread at random."""
    need = lead + count * (length + MEDIAN_WINDOW)
    slack = n_total - need
    assert slack > 0, slack
    cuts = sorted(int(v) for v in rng.integers(0, slack + 1, size=count))
    return [lead + c + i * (length + MEDIAN_WINDOW) for i, c in enumerate(cuts)]


def cu8(re_list, im_list):
    out = []
    for re, im in zip(re_list, im_list):
        for v in (re, im):
            q = math.floor(v * 128.0 + 127.5 + 0.5)
            out.append(max(0, min(255, q)))
    return np.array(out, dtype=np.uint8)


def adsb_signal(rng, n, specs, sigma, bursts):
    """Sum of the frames ``(start, frame, amplitude, phase, offset_hz)``, the noise and the noise
    bursts ``(start, length, sigma)``, quantized to ``cu8``."""
    re = [0.0] * n
    im = [0.0] * n
    for start, frame, amp, phase, offset in specs:
        for off in frame_pulses(frame):
            t = math.fmod(phase + offset * (start + off) / ADSB_RATE, 1.0)
            re[start + off] += amp * math.cos(2.0 * math.pi * t)
            im[start + off] += amp * math.sin(2.0 * math.pi * t)
    nre = rng.standard_normal(n).tolist()
    nim = rng.standard_normal(n).tolist()
    for i in range(n):
        re[i] += sigma * nre[i]
        im[i] += sigma * nim[i]
    for start, length, s in bursts:
        bre = rng.standard_normal(length).tolist()
        bim = rng.standard_normal(length).tolist()
        for j in range(length):
            if start + j < n:
                re[start + j] += s * bre[j]
                im[start + j] += s * bim[j]
    return cu8(re, im)


def random_carrier(rng):
    return float(rng.uniform(0.0, 1.0)), float(rng.uniform(-100e3, 100e3))


def amplitude_of(snr_db, sigma):
    """The pulse amplitude of ``snr_db`` = 10 log10(A^2 / (2 sigma^2))."""
    return math.sqrt(2.0 * sigma * sigma * 10.0 ** (snr_db / 10.0))


@case("flows", "adsb_df17_200_frames", 544)
def adsb_df17_200_frames(rng):
    n, count, sigma = 150000, 200, ADSB_NOISE_SIGMA
    starts = place(rng, n - LONG_SAMPLES, count, LONG_SAMPLES, MEDIAN_WINDOW)
    frames, specs = [], []
    for p in starts:
        frame, icao = mode_s_frame(rng, 17, 14)
        level_db = float(rng.uniform(10.0, 30.0))
        phase, offset = random_carrier(rng)
        specs.append((p, frame, amplitude_of(level_db, sigma), phase, offset))
        frames.append((p, frame, icao, level_db))
    bursts = []
    for _ in range(40):
        length = int(rng.integers(32, 1025))
        start = int(rng.integers(0, n - 32))
        burst_db = float(rng.uniform(0.0, 12.0))
        bursts.append((start, length, sigma * math.sqrt(10.0 ** (burst_db / 10.0)), burst_db))
    rf = adsb_signal(rng, n, specs, sigma, [(s, l, b) for s, l, b, _ in bursts])
    listed = []
    for p, frame, icao, level_db in frames:
        lo, hi = p - MEDIAN_WINDOW, p + LONG_SAMPLES
        worst = 0.0
        for s, l, b, _ in bursts:
            if s < hi and s + l > lo:
                worst = max(worst, b * b)
        snr = 10.0 * math.log10(10.0 ** (level_db / 10.0) * sigma * sigma / (sigma * sigma + worst))
        listed.append(
            {"index": p, "hex": frame.hex().upper(), "icao": icao, "level_db": round(level_db, 3), "snr_db": round(snr, 3)}
        )
    return dict(
        params={
            "rate_hz": ADSB_RATE,
            "noise_sigma": sigma,
            "frames": listed,
            "bursts": [{"start": s, "length": l, "level_db": round(d, 3)} for s, l, _, d in bursts],
            "min_snr_db": 10.0,
        },
        arrays={"rf": Array("u8", rf)},
        tolerance={"class": "bit_exact"},
        notes="200 DF 17 frames (CA 5, random address and ME, CRC-24 parity) at 2 MHz in 150000 cu8 "
        "samples; pulses of one sample at p, p + 2, p + 7, p + 9 and p + 16 + 2 k (bit 1) or p + 17 + 2 k "
        "(bit 0); amplitudes 10 to 30 dB above the noise (A^2 / (2 sigma^2), uniform in dB), random "
        "carrier phase, offset up to 100 kHz; noise sigma 0.018 per component, 40 bursts of 32 to 1024 "
        "samples 0 to 12 dB above it; frames lists index (= p), the 14 bytes, the address, level_db and "
        "snr_db (over the noise plus the strongest burst overlapping p - 256 .. p + 240); every frame with "
        "snr_db above min_snr_db must decode with CRC_OK at its index, and no other CRC_OK frame may appear",
    )


@case("flows", "adsb_short_frames", 545)
def adsb_short_frames(rng):
    n, sigma = 24000, ADSB_NOISE_SIGMA
    known_df17, short_known, short_unknown = 6, 20, 4
    count = known_df17 + 1 + short_known + short_unknown
    # A candidate is tested once a long frame's samples are in, so the last start leaves room.
    starts = place(rng, n - LONG_SAMPLES, count, LONG_SAMPLES, MEDIAN_WINDOW)
    addresses = []
    kinds = ["df17"] * known_df17 + ["df11_checked"]
    rest = ["known"] * short_known + ["unknown"] * short_unknown
    kinds += [rest[int(k)] for k in rng.permutation(len(rest))]
    specs, listed = [], []
    for p, kind in zip(starts, kinds):
        if kind == "df17":
            frame, icao = mode_s_frame(rng, 17, 14)
            addresses.append(icao)
            df, out = 17, "crc_ok"
        elif kind == "df11_checked":
            icao = int(rng.integers(0, 1 << 24))
            body = bytes([(11 << 3) | 5]) + icao.to_bytes(3, "big")
            frame = body + crc24_adsb(body).to_bytes(3, "big")
            addresses.append(icao)
            df, out = 11, "crc_ok"
        else:
            df = (4, 5, 11)[int(rng.integers(0, 3))]
            if kind == "known":
                icao = addresses[int(rng.integers(0, len(addresses)))]
                out = "address_parity"
            else:
                icao = int(rng.integers(0, 1 << 24))
                assert icao not in addresses
                out = "dropped"
            frame, icao = mode_s_frame(rng, df, 7, icao)
        level_db = float(rng.uniform(20.0, 30.0))
        phase, offset = random_carrier(rng)
        specs.append((p, frame, amplitude_of(level_db, sigma), phase, offset))
        listed.append(
            {"index": p, "hex": frame.hex().upper(), "df": df, "icao": icao, "out": out, "level_db": round(level_db, 3)}
        )
    rf = adsb_signal(rng, n, specs, sigma, [])
    return dict(
        params={"rate_hz": ADSB_RATE, "noise_sigma": sigma, "frames": listed},
        arrays={"rf": Array("u8", rf)},
        tolerance={"class": "bit_exact"},
        notes="Mode S frames at 2 MHz in 24000 cu8 samples, 20 to 30 dB above noise of sigma 0.018, pulses "
        "as in adsb_df17_200_frames: 6 DF 17 frames and one DF 11 frame with the plain CRC-24 parity "
        "(residue 0), then in random order 20 short frames (DF 4, DF 5, DF 11; parity = CRC-24 xor "
        "address) with one of these 7 addresses and 4 with an address not seen before; out is crc_ok "
        "(CRC_OK), address_parity (flag bit 16, 7 bytes) or dropped (no packet)",
    )
