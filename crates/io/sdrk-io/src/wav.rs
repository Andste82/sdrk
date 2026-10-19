//! WAV files: PCM 16-bit and IEEE float 32-bit, mono or stereo.
//!
//! Mono files are real signals (`RS16`, `RF32`). A stereo file is an IQ signal when the
//! caller says so (`iq: true`): the left channel is I, the right channel is Q (`CS16`,
//! `CF32`). Both layouts are byte-identical to the interleaved WAV frames, so samples pass
//! through unchanged.

use std::io::{Read, Seek, SeekFrom, Take, Write};

use sdrk_device::{BlockMeta, Rate, RxStream, SampleFormat, TxStream, formats};

use crate::error::{Error, Result};
use crate::raw::{FlagCounts, RawReader, check_whole, read_full};

const TAG_PCM: u16 = 1;
const TAG_FLOAT: u16 = 3;
const TAG_EXTENSIBLE: u16 = 0xFFFE;
const FILE: &str = "wav";
/// Size of the canonical header written by [`WavWriter`].
pub const WAV_HEADER_BYTES: usize = 44;

fn read_array<R: Read, const N: usize>(r: &mut R) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    if read_full(r, &mut b)? != N {
        return Err(Error::parse(FILE, "unexpected end of file in the header"));
    }
    Ok(b)
}

fn u16_at(b: &[u8], at: usize) -> Result<u16> {
    let s = b
        .get(at..at + 2)
        .ok_or_else(|| Error::parse(FILE, "fmt chunk too short"))?;
    Ok(u16::from_le_bytes([
        s.first().copied().unwrap_or(0),
        s.get(1).copied().unwrap_or(0),
    ]))
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    let s = b
        .get(at..at + 4)
        .ok_or_else(|| Error::parse(FILE, "fmt chunk too short"))?;
    let mut a = [0u8; 4];
    a.copy_from_slice(s);
    Ok(u32::from_le_bytes(a))
}

/// Skips `n` bytes of a reader that cannot seek.
fn skip<R: Read>(r: &mut R, mut n: u64) -> Result<()> {
    let mut sink = [0u8; 512];
    while n > 0 {
        let take = usize::try_from(n).unwrap_or(usize::MAX).min(sink.len());
        let part = sink
            .get_mut(..take)
            .ok_or_else(|| Error::parse(FILE, "internal skip buffer"))?;
        let got = read_full(r, part)?;
        if got == 0 {
            return Err(Error::parse(FILE, "unexpected end of file in a chunk"));
        }
        n -= got as u64;
    }
    Ok(())
}

/// Reads a WAV file as an RX stream.
///
/// Supported: PCM 16-bit and IEEE float 32-bit in a plain `fmt ` chunk or in
/// `WAVE_FORMAT_EXTENSIBLE`; one or two channels. Unknown chunks before `data` are skipped.
/// A data size of `0xFFFFFFFF` (streamed files) reads to the end of the input.
///
/// ```
/// use sdrk_device::{formats, RxStream};
/// use sdrk_io::{WavReader, WavWriter};
/// let mut w = WavWriter::new(std::io::Cursor::new(Vec::new()), formats::RS16, sdrk_device::Rate::hz(8000)?)?;
/// sdrk_device::TxStream::write(&mut w, &[1, 0, 2, 0], &Default::default())?;
/// let bytes = w.finish()?.into_inner();
/// let mut r = WavReader::new(&bytes[..], false, 1024)?;
/// assert_eq!(r.format(), formats::RS16);
/// let mut buf = [0u8; 8];
/// assert_eq!(r.read(&mut buf)?.0, 4);
/// # Ok::<(), sdrk_io::Error>(())
/// ```
#[derive(Debug)]
pub struct WavReader<R: Read> {
    raw: RawReader<Take<R>>,
}

impl<R: Read> WavReader<R> {
    /// Parses the header. `iq` selects the interpretation of a stereo file; `chunk_samples` is
    /// the largest number of samples per `read`.
    ///
    /// # Errors
    /// `Parse` for a malformed header, `Unsupported` for other codecs, bit depths or channel
    /// counts, `InvalidArgument` if `iq` does not fit the channel count (stereo needs
    /// `iq: true`, mono needs `iq: false`).
    pub fn new(mut inner: R, iq: bool, chunk_samples: usize) -> Result<Self> {
        let head: [u8; 12] = read_array(&mut inner)?;
        if head.get(..4) != Some(b"RIFF") || head.get(8..12) != Some(b"WAVE") {
            return Err(Error::parse(FILE, "not a RIFF/WAVE file"));
        }
        let mut fmt: Option<(u16, u16, u32, u16)> = None;
        loop {
            let ch: [u8; 8] = read_array(&mut inner)?;
            let id = ch.get(..4).unwrap_or(&[]);
            let size = u64::from(u32_at(&ch, 4)?);
            if id == b"fmt " {
                if !(16..=4096).contains(&size) {
                    return Err(Error::parse(FILE, "fmt chunk has an invalid size"));
                }
                let mut body = vec![0u8; size as usize];
                if read_full(&mut inner, &mut body)? != body.len() {
                    return Err(Error::parse(
                        FILE,
                        "unexpected end of file in the fmt chunk",
                    ));
                }
                if size % 2 == 1 {
                    skip(&mut inner, 1)?;
                }
                let mut tag = u16_at(&body, 0)?;
                let channels = u16_at(&body, 2)?;
                let rate = u32_at(&body, 4)?;
                let bits = u16_at(&body, 14)?;
                if tag == TAG_EXTENSIBLE {
                    if size < 40 || u16_at(&body, 16)? < 22 {
                        return Err(Error::parse(FILE, "extensible fmt chunk is too short"));
                    }
                    tag = u16_at(&body, 24)?;
                }
                fmt = Some((tag, channels, rate, bits));
            } else if id == b"data" {
                let (tag, channels, rate, bits) =
                    fmt.ok_or_else(|| Error::parse(FILE, "data chunk before the fmt chunk"))?;
                return Self::from_parts(inner, size, tag, channels, rate, bits, iq, chunk_samples);
            } else {
                skip(&mut inner, size + (size & 1))?;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        inner: R,
        data_size: u64,
        tag: u16,
        channels: u16,
        rate: u32,
        bits: u16,
        iq: bool,
        chunk_samples: usize,
    ) -> Result<Self> {
        let format = match (tag, bits, channels, iq) {
            (TAG_PCM, 16, 1, false) => formats::RS16,
            (TAG_PCM, 16, 2, true) => formats::CS16,
            (TAG_FLOAT, 32, 1, false) => formats::RF32,
            (TAG_FLOAT, 32, 2, true) => formats::CF32,
            (TAG_PCM | TAG_FLOAT, 16 | 32, 2, false) => {
                return Err(Error::invalid(
                    "a stereo wav file is an IQ signal only with `iq: true`",
                ));
            }
            (TAG_PCM | TAG_FLOAT, 16 | 32, 1, true) => {
                return Err(Error::invalid("`iq: true` needs a stereo wav file"));
            }
            (TAG_PCM | TAG_FLOAT, _, 1 | 2, _) => {
                return Err(Error::Unsupported(
                    "wav sample size (only 16-bit PCM and 32-bit float)",
                ));
            }
            (TAG_PCM | TAG_FLOAT, _, _, _) => {
                return Err(Error::Unsupported("wav channel count (only 1 or 2)"));
            }
            _ => return Err(Error::Unsupported("wav codec (only PCM and IEEE float)")),
        };
        if rate == 0 {
            return Err(Error::parse(FILE, "sample rate is zero"));
        }
        let rate = Rate::hz(u64::from(rate))?;
        let limit = if data_size == u64::from(u32::MAX) {
            u64::MAX
        } else {
            data_size
        };
        let raw = RawReader::new(
            inner.take(limit),
            format,
            rate,
            sdrk_device::SampleIndex(0),
            chunk_samples,
        )?;
        Ok(Self { raw })
    }
}

impl<R: Read> RxStream for WavReader<R> {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.raw.format()
    }

    fn sample_rate(&self) -> Rate {
        self.raw.sample_rate()
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta)> {
        self.raw.read(buf)
    }
}

/// Writes a WAV file. The sizes in the header are patched by [`WavWriter::finish`].
///
/// Supported formats: `RS16` and `RF32` (mono), `CS16` and `CF32` (stereo, I = left). The
/// header is the canonical 44-byte header (a 16-byte `fmt ` chunk, also for float data).
/// WAV stores the rate as a 32-bit integer, so the rate must be a whole number of hertz.
/// Timing is ignored; the flags of the blocks are counted.
///
/// ```
/// use sdrk_device::{formats, Rate, TxStream};
/// use sdrk_io::WavWriter;
/// let mut w = WavWriter::new(std::io::Cursor::new(Vec::new()), formats::CF32, Rate::hz(48_000)?)?;
/// w.write(&[0u8; 16], &Default::default())?;
/// let bytes = w.finish()?.into_inner();
/// assert_eq!(bytes.len(), 44 + 16);
/// assert_eq!(&bytes[..4], b"RIFF");
/// # Ok::<(), sdrk_io::Error>(())
/// ```
#[derive(Debug)]
pub struct WavWriter<W: Write + Seek> {
    inner: W,
    format: SampleFormat,
    rate: Rate,
    data_bytes: u64,
    counts: FlagCounts,
}

impl<W: Write + Seek> WavWriter<W> {
    /// Writes the provisional header.
    ///
    /// # Errors
    /// `Unsupported` for a format other than `RS16`, `RF32`, `CS16`, `CF32`;
    /// `InvalidArgument` if the rate is not a whole number of hertz that fits 32 bits.
    pub fn new(mut inner: W, format: SampleFormat, rate: Rate) -> Result<Self> {
        let (tag, channels, bits) = if format == formats::RS16 {
            (TAG_PCM, 1u16, 16u16)
        } else if format == formats::CS16 {
            (TAG_PCM, 2, 16)
        } else if format == formats::RF32 {
            (TAG_FLOAT, 1, 32)
        } else if format == formats::CF32 {
            (TAG_FLOAT, 2, 32)
        } else {
            return Err(Error::Unsupported(
                "wav writer formats (rs16, cs16, rf32, cf32)",
            ));
        };
        if rate.den() != 1 {
            return Err(Error::invalid("a wav file needs a whole number of hertz"));
        }
        let hz = u32::try_from(rate.num())
            .map_err(|_| Error::invalid("the sample rate does not fit 32 bits"))?;
        let block_align = channels * bits / 8;
        let mut h = Vec::with_capacity(WAV_HEADER_BYTES);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&36u32.to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&tag.to_le_bytes());
        h.extend_from_slice(&channels.to_le_bytes());
        h.extend_from_slice(&hz.to_le_bytes());
        h.extend_from_slice(&(hz * u32::from(block_align)).to_le_bytes());
        h.extend_from_slice(&block_align.to_le_bytes());
        h.extend_from_slice(&bits.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&0u32.to_le_bytes());
        inner.write_all(&h)?;
        Ok(Self {
            inner,
            format,
            rate,
            data_bytes: 0,
            counts: FlagCounts::default(),
        })
    }

    /// The flags seen so far, counted per flag.
    pub fn flag_counts(&self) -> FlagCounts {
        self.counts
    }

    /// Patches the sizes in the header, flushes and gives back the writer.
    ///
    /// # Errors
    /// `Unsupported` if the data exceeds the 4 GiB limit of the WAV format.
    pub fn finish(mut self) -> Result<W> {
        let data = u32::try_from(self.data_bytes)
            .ok()
            .filter(|d| d.checked_add(36).is_some())
            .ok_or(Error::Unsupported("wav data larger than 4 GiB"))?;
        self.inner.seek(SeekFrom::Start(4))?;
        self.inner.write_all(&(36 + data).to_le_bytes())?;
        self.inner.seek(SeekFrom::Start(40))?;
        self.inner.write_all(&data.to_le_bytes())?;
        self.inner.seek(SeekFrom::End(0))?;
        self.inner.flush()?;
        Ok(self.inner)
    }
}

impl<W: Write + Seek> TxStream for WavWriter<W> {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize> {
        check_whole(&self.format, buf.len())?;
        let total = self.data_bytes.saturating_add(buf.len() as u64);
        if total > u64::from(u32::MAX) - 36 {
            return Err(Error::Unsupported("wav data larger than 4 GiB"));
        }
        self.inner.write_all(buf)?;
        self.data_bytes = total;
        self.counts.record(meta.flags);
        Ok(buf.len())
    }

    fn poll_status(&mut self) -> Result<Option<BlockMeta>> {
        Ok(None)
    }
}
