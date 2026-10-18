# Device API (`sdrk-device`)

`sdrk-device` is the lowest crate of SDRk. It is `no_std`, heap-free and has no dependencies. External drivers depend only on this crate. It defines the shared error classification, raw sample formats, exact time types and the synchronous RX/TX stream traits. Control capabilities, async streams and the conformance suite follow in M12; sample conversion follows in M05.

## Errors

`ErrorKind` classifies every error in the workspace (`invalid_argument`, `length_mismatch`, ...). Each crate has its own `Error` that implements `SdrkError::kind`. The crate error is `Error::{InvalidFormat, InvalidRate, Overflow, BufferTooSmall}`.

## Value mapping

`SampleFormat` describes raw bytes. Amplitude = (raw - offset) / full_scale.

- offset is 127.5 for `U8`, otherwise 0.
- `Align::Lsb`: the effective value is the low `bits` bits, sign-extended. `Align::Msb`: `raw >> (container_bits - bits)` (arithmetic shift).
- `full_scale` is `2^(bits-1)` (`FullScale::PowerOfTwo`), an explicit integer (`Explicit(2048)` for bladeRF `SC16_Q11`), or 1.0 for `F32`.
- `swap_iq`: the first value of each pair is Q.
- `Planar`: all I values, then all Q values; both halves are equal.
- `Endian::Big` only affects the byte order of multi-byte containers in the input data. The host is always little-endian; big-endian targets fail to compile.

## Packing rules

Packed containers are LSB-first and little-endian, values in the order I0, Q0, I1, Q1, ...

`Packed12` (2 values in 3 bytes):

```
byte:   b0        b1        b2
bits:   7......0  7..4 3..0  7......0
        v0[7:0]   v1[3:0]   v1[11:4]
                  v0[11:8]
v0 = b0 | (b1 & 0x0F) << 8
v1 = (b1 >> 4) | b2 << 4        (both sign-extended from 12 bits)
```

`Packed10` (4 values in 5 bytes): the 40 bits are read LSB-first as four consecutive 10-bit values, each sign-extended from 10 bits. `Bits1`: 8 values per byte, bit 0 first.

`bytes_for_samples` returns `InvalidFormat` when the sample count does not fill whole groups (Packed12 real: even; Packed10 interleaved: even, real: multiple of 4; Bits1 interleaved: multiple of 4, real: multiple of 8) and `Overflow` when the byte count exceeds `usize`.

## Format constants (`formats`)

| Constant | Container, bits, layout | Device |
| --- | --- | --- |
| `CU8` | U8, 8, interleaved | RTL-SDR (zero at 127.5) |
| `CS8` | I8, 8, interleaved | HackRF |
| `CS16` | I16, 16, interleaved | Airspy HF+, LR2021 IQ16 |
| `CS12_LSB16` | I16, 12, interleaved | PlutoSDR / AD9361 |
| `SC16_Q11` | I16, 12, interleaved, full scale 2048 | bladeRF |
| `CS16_PLANAR` | I16, 16, planar | SDRplay API style |
| `PACKED12` | Packed12, 12, interleaved | LimeSDR-style |
| `PACKED10` | Packed10, 10, interleaved | ESP-SDR CAP20-style (bit order to be confirmed by the driver) |
| `I2S32_MSB` | I32, 24, MSB-aligned | SX1255 mode B over I2S |
| `CF32` | F32, interleaved | files, SoapySDR CF32 |
| `RF32` | F32, real | real-valued audio |
| `RS16` | I16, real | PCM audio |

## Time model

- `SampleIndex(u64)`: position of a sample in a stream.
- `Rate`: exact reduced fraction samples/second. `Ratio`: exact out/in factor of a rate-changing block. All conversions use `u128`/`i128` integers. Floats are never used for time: at MS/s rates an `f64` loses single samples after a few days.
- `TimeAnchor { index, unix_ns, source }` states that sample `index` was taken at `unix_ns`. `time_of` extrapolates to any other index, rounding toward negative infinity.

## Stream contract

`RxStream::read` fills a buffer with whole samples (and whole packing groups) and returns the bytes written plus a `BlockMeta` (start index, `BlockFlags`, optional anchor). `TxStream::write` accepts whole samples and a `BlockMeta` (optionally `tx_time_ns` for timed bursts); `poll_status` reports underruns, late bursts and burst completion. Both traits are synchronous; async variants follow in M12.
