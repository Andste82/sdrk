"""Cases for the sensor decoders of ``sdrk_proto::sensors`` (M46, test 2).

One case per decoder (and one more, ``bresser_3ch_recorded``, for the three recorded messages of
the sensor labelled AS2-WS-6B, which uses the Bresser 3CH protocol, D1119): 20 messages with random fields, each as the pulse train a detector would
report (pulse and gap widths in samples at 250 kHz, with timing error), and the readings the
decoder must return. The encoders are written from the public descriptions of the devices
(listed in the decisions log, D1106 to D1110) without looking at the Rust decoders; the CRC and
checksum models are bitwise and independent of the Rust code. The worked examples of the sources
are checked at import time (``_self_test``).

Layout of a case: ``pairs`` all pulse and gap widths back to back (``i32``, pulse first),
``pair_counts`` the number of pairs of each train (``i32``); the expected readings, one entry per
train, in the arrays ``ids``, ``channels`` (0 when the device has none), ``battery_ok`` (-1 when the
device has none), ``model_ids`` (the index into the parameter ``models``), ``temperature_c``, ``humidity``, ``wind_avg_km_h``, ``wind_max_km_h``,
``wind_dir_deg`` and ``rain_mm`` (``f64``, NaN when the device does not report the value).
Parameters: the rate, the model names (``models``), the number of copies of the message in a train
(``repeats``) and the number of readings the decoder returns for a train
(``readings_per_train``: one per copy, one for the device without a check, which needs two equal
copies), and the tolerance of the values.

The ``cu8`` bursts of the spec are not part of the cases: one burst of a WH2 message is 120 ms,
30 000 samples or 60 KB at 250 kHz, 20 of them exceed the 512 KiB of a case (R8.6); the whole
chain test renders its bursts in Rust from the same coding rules.
"""

import math
from fractions import Fraction

import numpy as np

from ..registry import Array, case

RATE = 250_000


# ---------------------------------------------------------------- bit helpers


def bits_of(value, n):
    """The ``n`` low bits of ``value``, most significant first."""
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def bytes_to_bits(data):
    out = []
    for b in data:
        out += bits_of(b, 8)
    return out


def nibble_bits(n):
    """A nibble as sent by the Oregon Scientific devices: least significant bit first."""
    return [(n >> j) & 1 for j in range(4)]


# ---------------------------------------------------------------- checksums


def crc8_poly31_msb(data, init=0):
    """CRC-8, polynomial x^8+x^5+x^4+1, MSB first, as a division of the message polynomial."""
    reg = init
    for byte in data:
        for bit in bits_of(byte, 8):
            top = (reg >> 7) & 1
            reg = (reg << 1) & 0xFF
            if top ^ bit:
                reg ^= 0x31
    return reg


def crc8_dallas(data):
    """The CRC-8 of the OneWire family (reflected polynomial 0x8C, init 0)."""
    crc = 0
    for byte in data:
        for _ in range(8):
            mix = (crc ^ byte) & 1
            crc >>= 1
            if mix:
                crc ^= 0x8C
            byte >>= 1
    return crc


# ---------------------------------------------------------------- pulse trains


def runs_pwm(bits, short, long_, gap):
    return [(short if b else long_, gap) for b in bits]


def runs_ppm(bits, pulse, short_gap, long_gap, end_gap):
    """Every bit a pulse and the gap that carries it; the frame ends with one more pulse."""
    pairs = [(pulse, long_gap if b else short_gap) for b in bits]
    pairs.append((pulse, end_gap))
    return pairs


def runs_manchester(bits, half):
    """Bit 1 is on-off, bit 0 off-on; the leading off half of a first 0 is not sent and a
    trailing off half is the end of the train."""
    levels = []
    for b in bits:
        levels += [1, 0] if b else [0, 1]
    while levels and levels[0] == 0:
        levels.pop(0)
    while levels and levels[-1] == 0:
        levels.pop()
    runs = []
    for v in levels:
        if runs and runs[-1][0] == v:
            runs[-1][1] += half
        else:
            runs.append([v, half])
    pairs = []
    for k in range(0, len(runs), 2):
        gap = runs[k + 1][1] if k + 1 < len(runs) else 0.0
        pairs.append((runs[k][1], gap))
    return pairs


def runs_nrz(bits, bit_us):
    """High runs are pulses, low runs gaps; the frame starts with a 1 and a trailing run of 0
    is not seen (the burst ends)."""
    bits = list(bits)
    while bits and bits[-1] == 0:
        bits.pop()
    assert bits and bits[0] == 1
    runs = []
    for b in bits:
        if runs and runs[-1][0] == b:
            runs[-1][1] += bit_us
        else:
            runs.append([b, bit_us])
    pairs = []
    for k in range(0, len(runs), 2):
        gap = runs[k + 1][1] if k + 1 < len(runs) else 0.0
        pairs.append((runs[k][1], gap))
    return pairs


def join_copies(copies, repeat_gap):
    """Several copies of a message in one train, separated by ``repeat_gap`` us."""
    pairs = []
    for c in copies:
        if pairs:
            pairs[-1] = (pairs[-1][0], repeat_gap)
        pairs += c
    pairs[-1] = (pairs[-1][0], 0.0)
    return pairs


def distort(rng, pairs, clock, jitter_rel=0.0, jitter_abs=0.0):
    """Timing error: one clock error for the train (uniform within +-``clock``) and an error of
    every width (relative and absolute in us), quantised to samples of ``RATE``; a gap of 0 stays 0
    and no width is shorter than one sample."""
    scale = 1.0 + rng.uniform(-clock, clock)
    out = []
    for p, g in pairs:
        row = []
        for w in (p, g):
            if w == 0:
                row.append(0)
                continue
            w = w * scale * (1.0 + rng.uniform(-jitter_rel, jitter_rel))
            w += rng.uniform(-jitter_abs, jitter_abs)
            row.append(max(1, int(round(w * 1e-6 * RATE))))
        out.append(tuple(row))
    return out


def quantise(pairs):
    """The widths in samples of ``RATE`` without any timing error; a gap of 0 stays 0."""
    return [tuple(0 if w == 0 else max(1, int(round(w * 1e-6 * RATE))) for w in p) for p in pairs]


# ---------------------------------------------------------------- devices

NAN = float("nan")


def blank(model):
    return dict(model=model, id=0, channel=0, battery_ok=-1, temperature_c=NAN, humidity=NAN,
                wind_avg_km_h=NAN, wind_max_km_h=NAN, wind_dir_deg=NAN, rain_mm=NAN)


def wh2_message(f, temp_tenths, hum, dev_id):
    """48 bits: eight 1 bits, 12 bit type and id (first bit 0), 12 bit temperature as sign and
    magnitude in tenths, 8 bit humidity, CRC-8 of the four payload bytes."""
    assert 0 <= dev_id < 0x800
    sign = 1 if temp_tenths < 0 else 0
    raw = (sign << 11) | abs(temp_tenths)
    payload = bits_of(dev_id, 12) + bits_of(raw, 12) + bits_of(hum, 8)
    data = [int("".join(map(str, payload[i:i + 8])), 2) for i in range(0, 32, 8)]
    return [1] * 8 + payload + bits_of(crc8_dallas(data), 8)


@case("sensors", "wh2", 637)
def wh2(rng):
    reps = 2
    msgs = []
    for _ in range(20):
        f = blank("wh2")
        f["id"] = int(rng.integers(0, 0x800))
        t = int(rng.integers(-300, 501))
        f["temperature_c"] = t / 10.0
        f["humidity"] = float(rng.integers(0, 101))
        bits = wh2_message(f, t, int(f["humidity"]), f["id"])
        copies = [runs_pwm(bits, 500.0, 1500.0, 1000.0)] * reps
        pairs = distort(rng, join_copies(copies, 8000.0), 0.08, 0.05)
        msgs.append((f, pairs))
    return finish(msgs, "wh2", reps, reps, 637,
                  "WH2 messages (PWM 500/1500 us pulses, 1000 us gaps, 48 bits with CRC-8 "
                  "0x31 reflected, init 0), two copies in a train 8 ms apart, clock error "
                  "+-8 % and width error +-5 %, 250 kHz")


def nexus_message(dev_id, battery_ok, channel, temp_tenths, hum):
    """36 bits: id, battery, 0, channel - 1, 12 bit two's complement temperature in tenths,
    1111, humidity."""
    return (bits_of(dev_id, 8) + [1 if battery_ok else 0, 0] + bits_of(channel - 1, 2)
            + bits_of(temp_tenths & 0xFFF, 12) + [1, 1, 1, 1] + bits_of(hum, 8))


def bits_to_hex(bits):
    pad = (-len(bits)) % 4
    bits = list(bits) + [0] * pad
    return "".join("%x" % int("".join(map(str, bits[i:i + 4])), 2) for i in range(0, len(bits), 4))


@case("sensors", "nexus", 638)
def nexus(rng):
    reps = 3
    msgs = []
    for _ in range(20):
        f = blank("nexus")
        f["id"] = int(rng.integers(0, 256))
        f["channel"] = int(rng.integers(1, 5))
        f["battery_ok"] = int(rng.integers(0, 2))
        t = int(rng.integers(-300, 501))
        f["temperature_c"] = t / 10.0
        f["humidity"] = float(rng.integers(1, 100))
        bits = nexus_message(f["id"], f["battery_ok"], f["channel"], t, int(f["humidity"]))
        copies = [runs_ppm(bits, 500.0, 1000.0, 2000.0, 4000.0)] * reps
        pairs = distort(rng, join_copies(copies, 4000.0), 0.08, 0.05)
        msgs.append((f, pairs))
    return finish(msgs, "nexus", reps, 1, 638,
                  "Nexus frames (PPM, 500 us pulses, gaps 1000/2000 us, 4000 us between frames, "
                  "36 bits, no check), three equal frames in a train, one reading per train "
                  "(a device without a check needs two equal frames), clock error +-8 % and "
                  "width error +-5 %, 250 kHz")


OREGON_PREAMBLE = [1] * 16
OREGON_SYNC = 0xA


def oregon_nibbles(dev, channel_nibble, rolling, flags, temp_tenths, hum=None, unknown=0):
    """The nibbles from the sensor id to the checksum (EC40: 14, 1D20: 17), as in the
    description: id (4 nibbles in the order sent), channel, rolling code (low nibble first),
    flags, temperature BCD (tenths first), sign, humidity (ones first, 1D20), unknown nibble,
    checksum (sum of the nibbles before it, low nibble first)."""
    n = [(dev >> s) & 0xF for s in (12, 8, 4, 0)]
    n += [channel_nibble, rolling & 0xF, rolling >> 4, flags]
    mag = abs(temp_tenths)
    n += [mag % 10, (mag // 10) % 10, mag // 100, 8 if temp_tenths < 0 else 0]
    if hum is not None:
        n += [hum % 10, hum // 10, unknown]
    s = sum(n) & 0xFF
    return n + [s & 0xF, s >> 4]


def oregon_bits(nibbles):
    """The data bits (preamble, sync, nibbles, post-amble of two nibbles) and then every data bit
    sent twice, the first copy inverted (version 2.1)."""
    data = list(OREGON_PREAMBLE) + nibble_bits(OREGON_SYNC)
    for n in nibbles + [0, 0]:
        data += nibble_bits(n)
    out = []
    for d in data:
        out += [1 - d, d]
    return out


@case("sensors", "oregon_v21", 639)
def oregon_v21(rng):
    reps = 2
    msgs = []
    half = 1e6 / 2048.0
    for _ in range(20):
        thgr = bool(rng.integers(0, 2))
        f = blank("oregon_thgr122nx" if thgr else "oregon_thn132n")
        rolling = int(rng.integers(0, 256))
        ch = int(rng.integers(1, 4))
        low = int(rng.integers(0, 2))
        flags = (0x4 if low else 0) | (0x8 if rng.integers(0, 2) else 0)
        t = int(rng.integers(-300, 501))
        f["id"] = rolling
        f["channel"] = ch
        f["battery_ok"] = 0 if low else 1
        f["temperature_c"] = t / 10.0
        if thgr:
            h = int(rng.integers(1, 100))
            f["humidity"] = float(h)
            nib = oregon_nibbles(0x1D20, 1 << (ch - 1), rolling, flags, t, h, int(rng.integers(0, 16)))
        else:
            nib = oregon_nibbles(0xEC40, 1 << (ch - 1), rolling, flags, t)
        copies = [runs_manchester(oregon_bits(nib), half)] * reps
        pairs = distort(rng, join_copies(copies, 8000.0), 0.08, 0.05)
        msgs.append((f, pairs))
    return finish(msgs, "oregon_v21", reps, reps, 639,
                  "Oregon Scientific v2.1 messages (Manchester at 1024 bit/s with every bit sent "
                  "twice, nibble-sum checksum), THN132N (id EC40) and THGR122NX (id 1D20) mixed, "
                  "two copies in a train 8 ms apart, clock error +-8 % and width error +-5 %, "
                  "250 kHz")


def wh1080_message(dev_id, temp_raw, hum, wind_avg, wind_max, rain_raw, status, direction):
    b = [0xFF, dev_id >> 4, ((dev_id & 0xF) << 4) | (temp_raw >> 8), temp_raw & 0xFF, hum,
         wind_avg, wind_max, rain_raw >> 8, rain_raw & 0xFF, (status << 4) | direction]
    b.append(crc8_poly31_msb(b[1:10]))
    return bytes_to_bits(b)


@case("sensors", "wh1080", 644)
def wh1080(rng):
    reps = 2
    msgs = []
    for _ in range(20):
        f = blank("wh1080")
        f["id"] = int(rng.integers(0, 0x1000))
        raw = int(rng.integers(100, 801))
        f["temperature_c"] = (raw - 400) / 10.0
        f["humidity"] = float(rng.integers(10, 100))
        avg = int(rng.integers(0, 100))
        gust = int(rng.integers(avg, 150))
        f["wind_avg_km_h"] = avg * 0.34 * 3.6
        f["wind_max_km_h"] = gust * 0.34 * 3.6
        d = int(rng.integers(0, 16))
        f["wind_dir_deg"] = d * 22.5
        rain = int(rng.integers(0, 4096))
        f["rain_mm"] = rain * 0.3
        bits = wh1080_message(f["id"], raw, int(f["humidity"]), avg, gust, rain,
                              int(rng.integers(0, 16)), d)
        copies = [runs_pwm(bits, 500.0, 1500.0, 1000.0)] * reps
        pairs = distort(rng, join_copies(copies, 8000.0), 0.08, 0.05)
        msgs.append((f, pairs))
    return finish(msgs, "wh1080", reps, reps, 644,
                  "WH1080 messages (88 bits, PWM timing of the WH2 assumed, CRC-8 0x31 init 0 "
                  "MSB first over B1 to B9; the status nibble of B9 is random), two copies in a "
                  "train 8 ms apart, clock error +-8 % and width error +-5 %, 250 kHz")


TX29_BIT_US = 58.0


def tx29_frame(dev_id, temp_tenths, hum, weak):
    digits = temp_tenths + 400
    assert 0 <= digits <= 999
    b = [0x90 | (dev_id >> 4), ((dev_id & 0xF) << 4) | (digits // 100),
         (((digits // 10) % 10) << 4) | (digits % 10), (weak << 7) | hum]
    b.append(crc8_poly31_msb(b))
    return b


def tx29_bits(frame):
    return bytes_to_bits([0xAA] * 4 + [0x2D, 0xD4] + frame)


@case("sensors", "lacrosse_tx29", 645)
def lacrosse_tx29(rng):
    reps = 2
    msgs = []
    for _ in range(20):
        f = blank("lacrosse_tx29")
        f["id"] = int(rng.integers(0, 256))
        t = int(rng.integers(-300, 451))
        f["temperature_c"] = t / 10.0
        no_hum = rng.integers(0, 5) == 0
        hum = 106 if no_hum else int(rng.integers(1, 100))
        if not no_hum:
            f["humidity"] = float(hum)
        weak = int(rng.integers(0, 2))
        f["battery_ok"] = 0 if weak else 1
        frame = tx29_frame(f["id"], t, hum, weak)
        copies = [runs_nrz(tx29_bits(frame), TX29_BIT_US)] * reps
        # a gap of 4 ms between the copies: the frame ends with the last bit, trailing zeros
        # are not seen, so the gap is the rest of the low run
        pairs = distort(rng, join_copies(copies, 4000.0), 0.01, 0.0, 0.03 * TX29_BIT_US)
        msgs.append((f, pairs))
    return finish(msgs, "lacrosse_tx29", reps, reps, 645,
                  "LaCrosse IT+ TX29 frames (FSK, NRZ at 17.241 kbit/s: four bytes 0xAA, the sync "
                  "word 0x2DD4 and five bytes with CRC-8 0x31 init 0 over the first four), two "
                  "copies in a train 4 ms apart, clock error +-1 % and width error +-0.03 bit; "
                  "trailing zero bits of a copy are not part of the train, 250 kHz")


BRESSER_PREAMBLE_US = 735.0


def bresser_frame(dev_id, battery_low, button, channel, raw_t, hum):
    """The five bytes of a frame of the Bresser 3CH family (the sensor labelled "AS2-WS-6B, type
    60109"): the id, byte 1 with battery low (bit 7), the test button flag (bit 6), the channel
    (bits 5 and 4) and the high four bits of the 12 bit temperature value, the low byte of the
    value, the humidity in percent as a binary number and the sum of the four bytes modulo 256."""
    assert 0 <= raw_t < 0x1000 and 0 <= channel < 4
    b1 = (battery_low << 7) | (button << 6) | (channel << 4) | (raw_t >> 8)
    b = [dev_id, b1, raw_t & 0xFF, hum]
    return b + [sum(b) & 0xFF]


def bresser_celsius(raw_t):
    """The temperature: tenths of a degree Fahrenheit plus 900 (F = (T - 900) / 10), so
    C = (T - 1220) / 18, rounded half away from zero to one decimal with exact fractions."""
    tenths = Fraction((raw_t - 1220) * 10, 18)
    mag = int(math.floor(abs(tenths) + Fraction(1, 2)))
    return (mag if tenths >= 0 else -mag) / 10.0


def bresser_frame_pairs(frame, preamble, one=(500.0, 250.0), zero=(250.0, 500.0)):
    """A frame: ``preamble`` pairs of pulse and gap of 735 us, then 40 bits MSB first with a fixed
    bit period of 750 us (1: pulse 500 us and gap 250 us, 0: pulse 250 us and gap 500 us); the gap
    of the last bit runs into the preamble of the next frame."""
    pairs = [(BRESSER_PREAMBLE_US, BRESSER_PREAMBLE_US)] * preamble
    return pairs + [one if bit else zero for bit in bytes_to_bits(frame)]


def bresser_burst(frame, preamble, frames, one=(500.0, 250.0), zero=(250.0, 500.0)):
    """``frames`` identical frames back to back; the last gap is the pause, written as 0."""
    pairs = bresser_frame_pairs(frame, preamble, one, zero) * frames
    pairs[-1] = (pairs[-1][0], 0.0)
    return pairs


@case("sensors", "bresser_3ch", 649)
def bresser_3ch(rng):
    msgs, frames_of, preambles, lost = [], [], [], []
    for _ in range(20):
        f = blank("Bresser-3CH")
        f["id"] = int(rng.integers(0, 256))
        f["channel"] = int(rng.integers(1, 4))
        low = int(rng.integers(0, 2))
        button = int(rng.integers(0, 2))
        f["battery_ok"] = 0 if low else 1
        raw_t = int(rng.integers(700, 2501))  # -20 F to 160 F, the limits of the device
        f["temperature_c"] = bresser_celsius(raw_t)
        f["humidity"] = float(rng.integers(0, 101))
        frame = bresser_frame(f["id"], low, button, f["channel"], raw_t, int(f["humidity"]))
        preamble = int(rng.integers(3, 6))
        frames = int(rng.integers(20, 25))
        drop_first = int(rng.integers(0, 4) == 0)
        pairs = bresser_burst(frame, preamble, frames)
        if drop_first:
            # the detector lost the first pulse of the burst: the train starts one pair later
            pairs = pairs[1:]
        msgs.append((f, distort(rng, pairs, 0.08, 0.05)))
        frames_of.append(frames)
        preambles.append(preamble)
        lost.append(drop_first)
    return finish(msgs, "bresser_3ch", 22, 1, 649,
                  "Bresser 3CH messages (OOK, 3 to 5 preamble pairs of 735 us, 40 bits with a bit "
                  "period of 750 us, 1 = 500/250 us, 0 = 250/500 us, sum of bytes 0 to 3 as "
                  "check; battery, button and channel in the high nibble of byte 1), 20 to 24 "
                  "identical frames back to back in a train (one reading with repeats = the "
                  "number of frames), the first pair dropped in about a quarter of the trains, "
                  "clock error +-8 % and width error +-5 %, 250 kHz",
                  extra_arrays={
                      "repeat_counts": Array("i32", np.array(frames_of, dtype=np.int32)),
                      "preamble_pairs": Array("i32", np.array(preambles, dtype=np.int32)),
                      "first_pair_lost": Array("i32", np.array(lost, dtype=np.int32)),
                  })


@case("sensors", "bresser_3ch_recorded", 649)
def bresser_3ch_recorded(rng):
    """The four messages recorded from the maintainer's sensor (cu8 at 2 MS/s, 2026-11-09). They
    are facts about the device, not generated data: no random value is drawn. The widths are the
    measured ones (bit 1: 496/238 us, bit 0: 252/480 us, preamble 735/735 us); the number of
    frames per train (22, 23, 24 and 20) and the preamble length (4 pairs) are illustrative."""
    recorded = [  # (bytes of the frame, the display of the sensor)
        ([0xD8, 0x65, 0xC7, 0x51, 0x55], None),  # recording 1: display not known
        ([0xD8, 0x66, 0x52, 0x40, 0xD0], (22.1, 64)),  # recording 2
        ([0xD8, 0x66, 0x59, 0x47, 0xDE], (22.5, 71)),  # recording 3
        ([0xD8, 0x65, 0xDB, 0x3C, 0x54], (15.5, 60)),  # recording 4, outdoors, 20 frames
    ]
    frame_counts = [22, 23, 24, 20]
    msgs, frames_of = [], []
    for k, (frame, display) in enumerate(recorded):
        raw_t = ((frame[1] & 0x0F) << 8) | frame[2]
        # the flags of byte 1 in all recordings: battery ok, button pressed, channel 2
        assert frame == bresser_frame(frame[0], 0, 1, 2, raw_t, frame[3])  # the sum checks
        f = blank("Bresser-3CH")
        f["id"] = frame[0]
        f["channel"] = 2
        f["battery_ok"] = 1
        f["temperature_c"] = bresser_celsius(raw_t)
        f["humidity"] = float(frame[3])
        if display is not None:
            assert (f["temperature_c"], f["humidity"]) == (display[0], float(display[1]))
        frames = frame_counts[k]
        pairs = bresser_burst(frame, 4, frames, one=(496.0, 238.0), zero=(252.0, 480.0))
        msgs.append((f, quantise(pairs)))
        frames_of.append(frames)
    return finish(msgs, "bresser_3ch", 22, 1, 649,
                  "The four recorded messages of the sensor labelled AS2-WS-6B, type 60109 (D8 "
                  "65 C7 51 55, D8 66 52 40 D0, D8 66 59 47 DE, D8 65 DB 3C 54; id 0xD8, "
                  "channel 2, battery ok, button pressed) with the measured widths, 4 preamble "
                  "pairs and 22, 23, 24 and 20 frames; the expected values are 14.4 C 81 % "
                  "(display not known), 22.1 C 64 %, 22.5 C 71 % and 15.5 C 60 % (the displays "
                  "of recordings 2 to 4); no random value, 250 kHz",
                  extra_arrays={
                      "repeat_counts": Array("i32", np.array(frames_of, dtype=np.int32)),
                  },
                  extra_params={"recorded": True})


def finish(msgs, model, reps, per_train, seed, notes, extra_arrays=None, extra_params=None):
    pairs, counts = [], []
    for _, p in msgs:
        counts.append(len(p))
        for a, b in p:
            pairs += [a, b]

    models = []
    for f, _ in msgs:
        if f["model"] not in models:
            models.append(f["model"])

    def col(key, dtype):
        return np.array([f[key] for f, _ in msgs], dtype=dtype)

    arrays = {
        "pairs": Array("i32", np.array(pairs, dtype=np.int32)),
        "pair_counts": Array("i32", np.array(counts, dtype=np.int32)),
        "ids": Array("i32", col("id", np.int32)),
        "channels": Array("i32", col("channel", np.int32)),
        "battery_ok": Array("i32", col("battery_ok", np.int32)),
        "model_ids": Array("i32", np.array([models.index(f["model"]) for f, _ in msgs],
                                           dtype=np.int32)),
    }
    for key in ("temperature_c", "humidity", "wind_avg_km_h", "wind_max_km_h", "wind_dir_deg",
                "rain_mm"):
        arrays[key] = Array("f64", col(key, np.float64))
    arrays.update(extra_arrays or {})
    return dict(
        params={
            **(extra_params or {}),
            "rate_hz": RATE,
            "models": models,
            "decoder": model,
            "trains": len(msgs),
            "repeats": reps,
            "readings_per_train": per_train,
            "max_abs_err": 1e-3,
            "seed": seed,
        },
        arrays=arrays,
        tolerance={"class": "max_abs_err", "max_abs_err": 1e-3, "skip": 0},
        notes=notes,
    )


# ---------------------------------------------------------------- examples of the sources


def _self_test():
    # Nexus: the example of the description, 5c910ff38 (id 0x5c, channel 2, 27.1 C, 56 %, battery 1)
    assert bits_to_hex(nexus_message(0x5C, 1, 2, 271, 56)) == "5c910ff38"
    # Oregon Scientific v2.1, two examples of the document (THGR122NX)
    ex1 = oregon_nibbles(0x1D20, 4, 0x58, 0xC, -84, 28, 8)
    assert "".join("%X" % n for n in ex1) == "1D20485C480882835"
    ex2 = oregon_nibbles(0x1D20, 1, 0xB6, 0x1, 190, 37, 10)
    assert "".join("%X" % n for n in ex2) == "1D2016B1091073A14"
    # LaCrosse IT+: the two examples of the article
    assert crc8_poly31_msb([0x92, 0x03, 0x94, 0x59]) == 0x3C
    assert crc8_poly31_msb([0x92, 0x86, 0x24, 0x28]) == 0xDE
    assert tx29_frame(0x20, -6, 89, 0) == [0x92, 0x03, 0x94, 0x59, 0x3C]
    assert tx29_frame(0x28, 224, 40, 0) == [0x92, 0x86, 0x24, 0x28, 0xDE]
    assert math.isclose(1e6 / 17241.0, TX29_BIT_US, abs_tol=0.01)
    # Bresser 3CH: the three recorded frames carry their own sums (battery ok, button, channel
    # 2), and the display values of two recordings follow from the temperature scale
    assert bresser_frame(0xD8, 0, 1, 2, 0x5C7, 0x51) == [0xD8, 0x65, 0xC7, 0x51, 0x55]
    assert bresser_frame(0xD8, 0, 1, 2, 0x652, 0x40) == [0xD8, 0x66, 0x52, 0x40, 0xD0]
    assert bresser_frame(0xD8, 0, 1, 2, 0x659, 0x47) == [0xD8, 0x66, 0x59, 0x47, 0xDE]
    assert bresser_frame(0xD8, 0, 1, 2, 0x5DB, 0x3C) == [0xD8, 0x65, 0xDB, 0x3C, 0x54]
    assert (bresser_celsius(0x652), bresser_celsius(0x659), bresser_celsius(0x5C7),
            bresser_celsius(0x5DB)) == (22.1, 22.5, 14.4, 15.5)

_self_test()
