# Device API (`sdrk-device`)

`sdrk-device` is the lowest crate of SDRk. It is `no_std`, heap-free and has no dependencies. External drivers depend only on this crate. It defines the shared error classification, raw sample formats, exact time types, the synchronous and asynchronous RX/TX stream traits and the optional control capabilities. The conformance checks that drivers run against these traits are in `sdrk-testkit` (see [Conformance checks](#conformance-checks)); sample conversion (`convert`, `view_cq15`, `view_c32`) is in `sdrk-core` since M05, see `docs/kernels.md`.

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
| `RF32` | F32, real | real-valued audio, real-sampling front ends |
| `RS16` | I16, real | PCM audio, real-sampling devices (Airspy R2 and Mini, direct-sampling front ends, sound-card IF receivers) |

**Real inputs.** A device that delivers real samples puts the whole band from 0 to `fs / 2` into one value per sample. The `convert` block and the automatic conversion of a raw pipeline input accept the formats with a real layout and produce `f32` or Q15 (`convert_real`, `ConvertRealBlock`); `Progress` counts bytes and real samples, a packing group is the smallest whole number of samples (`rs16`: one sample in 2 bytes). The block `real_to_complex` then shifts the signal by `fs / 4` and halves the rate; the output centre is the input frequency `fs / 4` and a station at the real frequency `f` is tuned with a `ddc` frequency of `f - fs / 4` (see [blocks.md](blocks.md) and the flow `real_input_fm_receiver`). Planar layouts cannot be streamed; real formats have no `convert_out` variant.

## Time model

- `SampleIndex(u64)`: position of a sample in a stream.
- `Rate`: exact reduced fraction samples/second. `Ratio`: exact out/in factor of a rate-changing block. All conversions use `u128`/`i128` integers. Floats are never used for time: at MS/s rates an `f64` loses single samples after a few days.
- `TimeAnchor { index, unix_ns, source }` states that sample `index` was taken at `unix_ns`. `time_of` extrapolates to any other index, rounding toward negative infinity.

## Stream contract

`RxStream::read` fills a buffer with whole samples (and whole packing groups) and returns the bytes written plus a `BlockMeta` (start index, `BlockFlags`, optional anchor). `TxStream::write` accepts whole samples and a `BlockMeta` (optionally `tx_time_ns` for timed bursts); `poll_status` reports underruns, late bursts and burst completion. Both traits are synchronous; the async variants are described below.

## Control capabilities

A driver implements only what its hardware can do; every trait is optional, `no_std` and allocation-free, and every setter returns the value that was actually applied (hardware rounds). Failures use the shared kinds (`Unsupported`, `OutOfRange`, `Device`, `Timeout`, `Disconnected`).

| Item | Content |
| --- | --- |
| `Range { min, max, step }` | A closed range; `step = 0` means continuous. `Range::contains` checks the bounds and the grid. |
| `Direction` | `Rx` or `Tx`. Channels are numbered per direction. |
| `Capabilities` | Built from `static` data and returned by reference: `rx_channels`, `tx_channels`, `full_duplex`, `formats`, `hw_timestamps`, `timed_commands`. |
| `Device` | `capabilities()`, `driver()` (for example `"rtlsdr"`), `serial()`. |
| `Tuner` | `set_frequency` (returns the actual frequency), `frequency`, `frequency_ranges`. |
| `RateControl` | `set_sample_rate(Rate)` (returns the actual `Rate`), `sample_rates`. |
| `GainControl` | `gain_stages` (name and range), `set_gain(stage, db)` (`None` = overall gain), `set_agc`. |
| `BandwidthControl` | `set_bandwidth`, `bandwidths`. |
| `HardwareTime` | `hardware_time()` (a `TimeAnchor`: sample index and device time) and `set_hardware_time(unix_ns)`. |
| `TimedCommands` | `schedule(CommandTime, Command)`; `CommandTime` is a sample index or unix nanoseconds; `Command` (`#[non_exhaustive]`) is `Frequency` or `Gain`. |

## Async streams

`AsyncRxStream` and `AsyncTxStream` (module `asyncio`) mirror `RxStream` and `TxStream` with `async fn read`, `write` and `poll_status`. They are for transports that cannot block: WebUSB, Web Serial and async USB stacks. The traits use `async fn` in traits (Rust 1.75) and carry `#[allow(async_fn_in_trait)]` on purpose: the futures have **no `Send` bound** because futures that wait on browser APIs are not `Send`, and a bound would exclude exactly the drivers these traits are for. The library ships no executor and no adapter from the async to the blocking traits. The tests drive the traits with a hand-written executor (`Waker::noop`).

## Conformance checks

`sdrk_testkit::conformance` (`no_std`, no heap) is the first driver conformance suite; drivers in other repositories call it from their own tests.

- `check_rx(stream, n_blocks, buf_size)` reads `n_blocks` blocks and checks: whole samples (whole packing groups) that fit the buffer; `start` equals the previous `start` plus its sample count, or lies strictly above with `GAP_BEFORE`; the time that the anchors give for the block start (`TimeAnchor::time_of`) never decreases; `format()` and `sample_rate()` stay the same unless the block carries `RETUNED`. A read that returns no bytes ends the run (end of a file); `Report::ended` says so.
- `check_tx(stream, n_blocks, buf_size)` writes zero samples in whole-sample blocks and checks: no more bytes accepted than offered, whole samples, progress on every call, stable format and rate, and that `poll_status` returns only blocks with flags.
- The internal buffer of both is `buf_size` bytes on the stack, at most `MAX_BUF` (2048 bytes); `check_rx_with` takes a caller-provided buffer of any size.
- `VirtualRx` and `VirtualTx` (records the bytes and the metadata of up to 64 writes, injects `UNDERRUN` and `LATE` statuses at configured sample indices) pass the checks, and so do the readers and writers of [`sdrk-io`](io.md).

## SoapySDR mapping

The device API was checked against the SoapySDR `Device.hpp` and `Constants.h` (master, retrieved 2026-10-19). SoapySDR names are in `code`; "gap" means that SDRk has no equivalent yet and that the item is in the [backlog](milestones/BACKLOG.md#phase-9--stabilization-and-extensions) ("Device API gaps") with the reason in the last column.

| SoapySDR group | SoapySDR API | SDRk | Remark |
| --- | --- | --- | --- |
| Identification | `getDriverKey` | `Device::driver` | |
| | `getHardwareKey`, `getHardwareInfo` | `Device::serial` (partly) | gap: free-form hardware key/value information; drivers expose what they need in their own API |
| | `enumerate`, `make`, `unmake` | none | device discovery and creation belong to the driver crates, not to the device traits |
| | `getNativeDeviceHandle` | none | not portable (WASM, MCU) |
| Channels | `getNumChannels(dir)` | `Capabilities::rx_channels`, `tx_channels` | |
| | `getFullDuplex` | `Capabilities::full_duplex` | per device, not per channel |
| | `getChannelInfo`, `setFrontendMapping`, `getFrontendMapping` | none | gap: rarely used, no portable meaning |
| | `listAntennas`, `setAntenna`, `getAntenna` | none | gap: antenna ports are board specific; add as an optional trait when a driver needs it |
| Stream API | `getStreamFormats`, `getNativeStreamFormat` (with full scale) | `Capabilities::formats`, `SampleFormat` (with `full_scale_value`) | SDRk formats describe layout, alignment, endianness and full scale explicitly |
| | `setupStream`, `closeStream`, `getStreamMTU` | the driver's constructor and `Drop`; block size is the buffer length of `read` | |
| | `activateStream`, `deactivateStream` (with `HAS_TIME`, `END_BURST` and a time) | driver specific (start and stop belong to the driver); a timed start is `TimedCommands::schedule` | gap: no generic start/stop trait yet |
| | `readStream` | `RxStream::read`, `AsyncRxStream::read` | returns bytes and a `BlockMeta` |
| | `writeStream` | `TxStream::write`, `AsyncTxStream::write` | |
| | `readStreamStatus` | `TxStream::poll_status`, `AsyncTxStream::poll_status` (`UNDERRUN`, `LATE`, `BURST_END`) | RX overflows arrive as `BlockFlags::OVERFLOW` on the next block |
| | flag `SOAPY_SDR_HAS_TIME` | `BlockMeta::anchor` (RX), `BlockMeta::tx_time_ns` (TX) | |
| | flag `SOAPY_SDR_END_BURST` | `BlockFlags::BURST_END` | `BURST_START` marks the first sample |
| | flag `SOAPY_SDR_ONE_PACKET` | none | a `read` returns what the driver has; the caller does not choose packet boundaries |
| | flag `SOAPY_SDR_END_ABRUPT` | none | gap: an aborted burst is reported as `LATE`/`UNDERRUN` status |
| | flags `SOAPY_SDR_MORE_FRAGMENTS`, `SOAPY_SDR_WAIT_TRIGGER` | none | gap: not needed for the supported devices |
| | `getNumDirectAccessBuffers`, `getDirectAccessBufferAddrs`, `acquireReadBuffer`, `releaseReadBuffer`, `acquireWriteBuffer`, `releaseWriteBuffer` | none | zero-copy access is a driver internal; the traits copy into the caller's buffer |
| Frequency | `setFrequency`, `getFrequency` (overall) | `Tuner::set_frequency`, `frequency` | returns the actual frequency |
| | `getFrequencyRange` | `Tuner::frequency_ranges` | |
| | components: `listFrequencies`, `setFrequency(name)`, `getFrequency(name)`, `getFrequencyRange(name)`, `getFrequencyArgsInfo` | none | gap: tuning components (RF, BB) are hardware specific; the overall frequency is what the pipeline needs |
| | `hasFrequencyCorrection`, `setFrequencyCorrection`, `getFrequencyCorrection` | none | gap: ppm correction is available in the pipeline (`resampler` `ppm`) |
| Gain | `setGain`, `getGain`, `getGainRange` (overall) | `GainControl::set_gain(None, db)`, `gain_stages` | |
| | elements: `listGains`, `setGain(name)`, `getGain(name)`, `getGainRange(name)` | `GainControl::gain_stages`, `set_gain(Some(name), db)` | there is no getter for one stage; `set_gain` returns the applied value |
| | `hasGainMode`, `setGainMode`, `getGainMode` | `GainControl::set_agc` | no getter |
| | `hasDCOffsetMode`, `setDCOffset*`, `hasIQBalance*`, `setIQBalance*` | none | gap: corrections are done in the DSP (`dc_block`; `iq_balance` is in the backlog) |
| Sample rate | `setSampleRate`, `getSampleRate`, `listSampleRates`, `getSampleRateRange` | `RateControl::set_sample_rate` (exact `Rate`), `sample_rates` (`Range`s) | SDRk rates are exact fractions |
| Bandwidth | `setBandwidth`, `getBandwidth`, `listBandwidths`, `getBandwidthRange` | `BandwidthControl::set_bandwidth`, `bandwidths` | |
| Clocking | `setMasterClockRate`, `getMasterClockRate`, `getMasterClockRates`, `setReferenceClockRate`, `getReferenceClockRate`, `getReferenceClockRates` | none | gap: needed for devices with selectable clocks only |
| | `listClockSources`, `setClockSource`, `getClockSource`, `listTimeSources`, `setTimeSource`, `getTimeSource` | none | gap: source selection (internal, external, GPS) is driver specific; `TimeSource` on an anchor says where a time came from |
| Time | `hasHardwareTime`, `getHardwareTime`, `setHardwareTime` | `Capabilities::hw_timestamps`, `HardwareTime::hardware_time`, `set_hardware_time` | SDRk reports the time together with a sample index |
| | `setCommandTime` | `TimedCommands::schedule(CommandTime, Command)` | explicit time per command instead of a stateful command time |
| Sensors | `listSensors`, `getSensorInfo`, `readSensor` (device and channel level) | none | gap: temperature, lock detect, RSSI; add as an optional trait |
| Registers, GPIO, I2C, SPI, UART | `listRegisterInterfaces`, `readRegister*`, `writeRegister*`, `listGPIOBanks`, `readGPIO*`, `writeGPIO*`, `readI2C`, `writeI2C`, `transactSPI`, `listUARTs`, `readUART`, `writeUART` | none | out of scope: low-level access belongs to the driver |
| Settings | `getSettingInfo`, `readSetting`, `writeSetting` (device and channel level) | none | gap: arbitrary key/value settings are not typed; a driver exposes its own typed API |
