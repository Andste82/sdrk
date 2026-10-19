//! A loopback channel simulator: what a transmitter writes arrives at the receiver after a
//! delay, an attenuation, a frequency offset and with additive noise.
//!
//! [`Loopback::new`] creates a transmit side ([`LoopbackTx`], a [`TxStream`]) and a receive
//! side ([`LoopbackRx`], an [`RxStream`]) that share one timeline: receive index 0 is the time
//! `t0_ns`. Both sides use the same `rate` and `format`.
//!
//! # Model
//!
//! The transmitted samples `x[m]` (decoded to `f64`, zero where nothing was transmitted) are
//! placed on the timeline: a write with `tx_time_ns` starts a burst at the receive index
//! `round((tx_time - t0) * rate / 1e9)`, other writes continue the current burst (or the
//! continuous transmission, which starts at index 0). The receiver sees, at index `n`,
//!
//! ```text
//! y[n] = g * x(n - d) * exp(j 2 pi f n / fs) + noise[n]
//! ```
//!
//! with the delay `d = delay_samples`, the amplitude gain `g = 10^(-attenuation_db / 20)`, the
//! frequency offset `f` and the sample rate `fs`. The integer part of `d` shifts the index; the
//! fractional part is applied with a 63-tap Kaiser(beta 8)-windowed-sinc interpolator computed
//! in `f64` (an exact identity for an integer delay). The phase `2 pi frac(f n / fs)` depends
//! only on the absolute index `n`. The noise has independent normal I and Q components of the
//! standard deviation `noise_sigma` (in units where 1.0 is full scale), drawn from a PCG32
//! generator positioned by the absolute index, so any chunking gives the same samples. The
//! result is encoded into the raw format with saturation.
//!
//! The receiver never ends. It produces what the transmitter has written *so far*: samples
//! written after the receiver passed their time are not seen (transmit first, then receive).
//! The first receive block carries the anchor `index 0 = t0_ns`.

use std::sync::{Arc, Mutex};
use std::vec;
use std::vec::Vec;

use num_complex::Complex64;
use sdrk_core::Error;
use sdrk_device::{
    BlockFlags, BlockMeta, Rate, RxStream, SampleFormat, SampleIndex, TimeAnchor, TimeSource,
    TxStream,
};

use crate::encode::{decode_reference, encode};
use crate::rng::Pcg32;

/// Taps of the fractional-delay interpolator.
pub const TAPS: usize = 63;
/// Kaiser window parameter of the interpolator.
pub const KAISER_BETA: f64 = 8.0;
/// Largest number of samples one receive block holds (the default of `max_block`).
pub const DEFAULT_MAX_BLOCK: usize = 4096;

const HALF: i64 = (TAPS as i64 - 1) / 2;
const NOISE_STREAM: u64 = 0x6c6f_6f70;
/// `next_u32` draws per complex noise sample (two normals of four draws).
const DRAWS_PER_SAMPLE: u64 = 8;

/// Parameters of the channel.
///
/// ```
/// use sdrk_device::{formats, Rate};
/// use sdrk_testkit::channel::LoopbackParams;
/// let p = LoopbackParams::new(Rate::hz(240_000)?, formats::CS8);
/// assert_eq!(p.delay_samples, 0.0);
/// # Ok::<(), sdrk_device::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopbackParams {
    /// Sample rate of both sides.
    pub rate: Rate,
    /// Raw format of both sides (one sample must be a whole number of bytes).
    pub format: SampleFormat,
    /// Time of receive index 0 in unix nanoseconds.
    pub t0_ns: i64,
    /// Delay in samples (not negative).
    pub delay_samples: f64,
    /// Attenuation in decibels (the amplitude is multiplied by `10^(-attenuation_db / 20)`).
    pub attenuation_db: f64,
    /// Frequency offset in hertz.
    pub freq_offset_hz: f64,
    /// Standard deviation of the I and of the Q noise component.
    pub noise_sigma: f64,
    /// Seed of the noise.
    pub seed: u64,
    /// Largest number of samples per receive block.
    pub max_block: usize,
}

impl LoopbackParams {
    /// A channel that delays, attenuates and distorts nothing.
    pub fn new(rate: Rate, format: SampleFormat) -> Self {
        Self {
            rate,
            format,
            t0_ns: 0,
            delay_samples: 0.0,
            attenuation_db: 0.0,
            freq_offset_hz: 0.0,
            noise_sigma: 0.0,
            seed: 1,
            max_block: DEFAULT_MAX_BLOCK,
        }
    }
}

/// A piece of the transmitted signal: raw bytes starting at a receive index.
#[derive(Debug)]
struct Segment {
    start: u64,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct Shared {
    params: LoopbackParams,
    sample_bytes: usize,
    taps: [f64; TAPS],
    gain: f64,
    segments: Vec<Segment>,
    /// The segment that the next write without a time extends, or `None` before the first
    /// write and after `BURST_END`.
    open: Option<usize>,
    /// Receive index where a continuous write without an open segment starts.
    cursor: u64,
    discarding: bool,
    status: std::collections::VecDeque<BlockMeta>,
}

/// The transmit side of a [`Loopback`].
#[derive(Debug, Clone)]
pub struct LoopbackTx {
    shared: Arc<Mutex<Shared>>,
    format: SampleFormat,
    rate: Rate,
}

/// The receive side of a [`Loopback`].
#[derive(Debug, Clone)]
pub struct LoopbackRx {
    shared: Arc<Mutex<Shared>>,
    format: SampleFormat,
    rate: Rate,
    next: u64,
    sent_anchor: bool,
}

/// A loopback channel; see the module documentation for the model.
///
/// ```
/// use sdrk_device::{formats, BlockMeta, Rate, RxStream, TxStream};
/// use sdrk_testkit::channel::{Loopback, LoopbackParams};
/// let mut p = LoopbackParams::new(Rate::hz(1000)?, formats::CS8);
/// p.delay_samples = 2.0;
/// let (mut tx, mut rx) = Loopback::new(p)?.split();
/// tx.write(&[64, 0, 32, 0], &BlockMeta::default())?; // two samples, real parts 0.5 and 0.25
/// let mut buf = [0u8; 12];
/// let (n, meta) = rx.read(&mut buf)?;
/// assert_eq!((n, meta.start.0), (12, 0));
/// assert_eq!(buf, [0, 0, 0, 0, 64, 0, 32, 0, 0, 0, 0, 0]); // delayed by two samples
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug)]
pub struct Loopback {
    tx: LoopbackTx,
    rx: LoopbackRx,
}

fn lock(shared: &Arc<Mutex<Shared>>) -> Result<std::sync::MutexGuard<'_, Shared>, Error> {
    shared
        .lock()
        .map_err(|_| Error::Internal("loopback state poisoned"))
}

/// Modified Bessel function of the first kind, order 0 (power series).
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    while term > 1e-18 * sum {
        term *= q / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let a = core::f64::consts::PI * x;
        libm::sin(a) / a
    }
}

/// Taps `g[j]`, `j = -31..=31`, of the interpolator for the fractional delay `frac`:
/// `g[j] = sinc(j - frac) * kaiser(j - frac)`.
fn fractional_taps(frac: f64) -> [f64; TAPS] {
    let half = (HALF + 1) as f64;
    let norm = bessel_i0(KAISER_BETA);
    let mut taps = [0.0; TAPS];
    for (k, t) in taps.iter_mut().enumerate() {
        let u = (k as i64 - HALF) as f64 - frac;
        let r = u / half;
        let w = if r.abs() < 1.0 {
            bessel_i0(KAISER_BETA * libm::sqrt(1.0 - r * r)) / norm
        } else {
            0.0
        };
        *t = sinc(u) * w;
    }
    taps
}

impl Loopback {
    /// Creates the channel.
    ///
    /// # Errors
    /// `InvalidParameter` for a format whose sample is not a whole number of bytes, a
    /// negative or non-finite delay, attenuation, offset or noise, or `max_block == 0`.
    pub fn new(params: LoopbackParams) -> Result<Self, Error> {
        let sample_bytes = params
            .format
            .bytes_for_samples(1)
            .map_err(|_| Error::InvalidParameter("a sample must be a whole number of bytes"))?;
        let ok = params.delay_samples.is_finite()
            && params.delay_samples >= 0.0
            && params.delay_samples < 1e15
            && params.attenuation_db.is_finite()
            && params.freq_offset_hz.is_finite()
            && params.noise_sigma.is_finite()
            && params.noise_sigma >= 0.0;
        if !ok {
            return Err(Error::InvalidParameter(
                "delay, attenuation, offset and noise must be finite (delay and noise not negative)",
            ));
        }
        if params.max_block == 0 {
            return Err(Error::InvalidParameter("max_block must be at least 1"));
        }
        let frac = params.delay_samples - libm::floor(params.delay_samples);
        let shared = Arc::new(Mutex::new(Shared {
            params,
            sample_bytes,
            taps: fractional_taps(frac),
            gain: libm::pow(10.0, -params.attenuation_db / 20.0),
            segments: Vec::new(),
            open: None,
            cursor: 0,
            discarding: false,
            status: std::collections::VecDeque::new(),
        }));
        Ok(Self {
            tx: LoopbackTx {
                shared: Arc::clone(&shared),
                format: params.format,
                rate: params.rate,
            },
            rx: LoopbackRx {
                shared,
                format: params.format,
                rate: params.rate,
                next: 0,
                sent_anchor: false,
            },
        })
    }

    /// Splits the channel into its two sides (they stay connected).
    pub fn split(self) -> (LoopbackTx, LoopbackRx) {
        (self.tx, self.rx)
    }

    /// The transmit side.
    pub fn tx(&mut self) -> &mut LoopbackTx {
        &mut self.tx
    }

    /// The receive side.
    pub fn rx(&mut self) -> &mut LoopbackRx {
        &mut self.rx
    }
}

impl TxStream for LoopbackTx {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize, Error> {
        let mut s = lock(&self.shared)?;
        if !buf.len().is_multiple_of(s.sample_bytes) {
            return Err(Error::InvalidParameter(
                "the data is not a whole number of samples",
            ));
        }
        if buf.is_empty() {
            return Ok(0);
        }
        if let Some(t) = meta.tx_time_ns {
            s.discarding = false;
            let rate = s.params.rate;
            let delta = i128::from(t) - i128::from(s.params.t0_ns);
            let den = i128::from(rate.den()) * 1_000_000_000;
            let num = delta * i128::from(rate.num());
            // round half up on the exact rational
            let index = (2 * num + den).div_euclid(2 * den);
            if index < 0 || u64::try_from(index).is_err() {
                s.discarding = true;
                s.status.push_back(BlockMeta {
                    start: SampleIndex(0),
                    flags: BlockFlags::LATE,
                    anchor: None,
                    tx_time_ns: Some(t),
                });
            } else {
                let start = u64::try_from(index).map_err(|_| Error::Internal("burst index"))?;
                s.segments.push(Segment {
                    start,
                    bytes: Vec::new(),
                });
                s.open = Some(s.segments.len() - 1);
            }
        } else if s.open.is_none() && !s.discarding {
            let start = s.cursor;
            s.segments.push(Segment {
                start,
                bytes: Vec::new(),
            });
            s.open = Some(s.segments.len() - 1);
        }
        if !s.discarding {
            let at = s.open.ok_or(Error::Internal("no open segment"))?;
            let seg = s
                .segments
                .get_mut(at)
                .ok_or(Error::Internal("segment index"))?;
            seg.bytes.extend_from_slice(buf);
            s.cursor = seg.start + (seg.bytes.len() / s.sample_bytes) as u64;
        }
        if meta.flags.contains(BlockFlags::BURST_END) {
            s.open = None;
            s.discarding = false;
        }
        Ok(buf.len())
    }

    fn poll_status(&mut self) -> Result<Option<BlockMeta>, Error> {
        Ok(lock(&self.shared)?.status.pop_front())
    }
}

impl LoopbackRx {
    /// Index of the next sample `read` will return.
    pub fn next_index(&self) -> u64 {
        self.next
    }
}

impl Shared {
    /// Decodes `x[m]` for `m` in `from..from + out.len()` (zero outside the segments); returns
    /// false if no segment overlaps the window (all zeros).
    fn input_window(&self, from: i64, out: &mut [Complex64]) -> Result<bool, Error> {
        out.fill(Complex64::new(0.0, 0.0));
        let mut any = false;
        let len = out.len() as i64;
        let sb = self.sample_bytes as i64;
        let mut tmp: Vec<Complex64> = Vec::new();
        for seg in &self.segments {
            let seg_start = i64::try_from(seg.start).unwrap_or(i64::MAX);
            let seg_len = (seg.bytes.len() as i64) / sb;
            let lo = from.max(seg_start);
            let hi = (from + len).min(seg_start.saturating_add(seg_len));
            if lo >= hi {
                continue;
            }
            any = true;
            let count = (hi - lo) as usize;
            let bytes = seg
                .bytes
                .get(((lo - seg_start) * sb) as usize..((hi - seg_start) * sb) as usize)
                .ok_or(Error::Internal("segment bytes"))?;
            tmp.clear();
            tmp.resize(count, Complex64::new(0.0, 0.0));
            decode_reference(&self.params.format, bytes, &mut tmp)
                .map_err(|_| Error::Internal("decoding transmitted samples"))?;
            let dst = out
                .get_mut((lo - from) as usize..(hi - from) as usize)
                .ok_or(Error::Internal("window slice"))?;
            for (d, x) in dst.iter_mut().zip(tmp.iter()) {
                *d += *x;
            }
        }
        Ok(any)
    }

    /// The channel output for the receive indices `start..start + out.len()`.
    fn channel(&self, start: u64, out: &mut [Complex64]) -> Result<(), Error> {
        let p = &self.params;
        let n = out.len();
        let d_int = libm::floor(p.delay_samples) as i64;
        let frac = p.delay_samples - libm::floor(p.delay_samples);
        let first = i64::try_from(start).map_err(|_| Error::Internal("start index"))?;
        // y[n] = sum_j g[j] x[n - d_int - j], j = -HALF..=HALF  (a single tap when frac == 0)
        let (lo_off, hi_off) = if frac == 0.0 { (0, 0) } else { (-HALF, HALF) };
        let win_from = first - d_int - hi_off;
        let win_len = n + (hi_off - lo_off) as usize;
        let mut x = vec![Complex64::new(0.0, 0.0); win_len];
        let any = self.input_window(win_from, &mut x)?;
        for (i, y) in out.iter_mut().enumerate() {
            *y = if !any {
                Complex64::new(0.0, 0.0)
            } else if frac == 0.0 {
                x.get(i).copied().unwrap_or_default()
            } else {
                let mut acc = Complex64::new(0.0, 0.0);
                for (k, g) in self.taps.iter().enumerate() {
                    let j = k as i64 - HALF;
                    // input index n - d_int - j, window position relative to win_from
                    let pos = (i as i64) + hi_off - j;
                    if let Some(v) = usize::try_from(pos).ok().and_then(|q| x.get(q)) {
                        acc += *v * *g;
                    }
                }
                acc
            };
        }
        let rate = p.rate.as_f64();
        let mut rng = Pcg32::new(p.seed, NOISE_STREAM);
        rng.advance(start.wrapping_mul(DRAWS_PER_SAMPLE));
        for (i, y) in out.iter_mut().enumerate() {
            let idx = start.wrapping_add(i as u64) as f64;
            let turns = p.freq_offset_hz * idx / rate;
            let phase = 2.0 * core::f64::consts::PI * (turns - libm::floor(turns));
            let rot = Complex64::new(libm::cos(phase), libm::sin(phase));
            let noise = Complex64::new(rng.normal() * p.noise_sigma, rng.normal() * p.noise_sigma);
            *y = *y * self.gain * rot + noise;
        }
        Ok(())
    }
}

impl RxStream for LoopbackRx {
    type Error = Error;

    fn format(&self) -> SampleFormat {
        self.format
    }

    fn sample_rate(&self) -> Rate {
        self.rate
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Error> {
        let s = lock(&self.shared)?;
        let n = (buf.len() / s.sample_bytes).min(s.params.max_block);
        if n == 0 {
            return Err(Error::BufferTooSmall {
                needed: s.sample_bytes,
                got: buf.len(),
            });
        }
        let mut y = vec![Complex64::new(0.0, 0.0); n];
        s.channel(self.next, &mut y)?;
        let written = encode(&s.params.format, &y, buf)
            .map_err(|_| Error::Internal("encoding the received samples"))?;
        let anchor = (!self.sent_anchor).then_some(TimeAnchor {
            index: SampleIndex(0),
            unix_ns: s.params.t0_ns,
            source: TimeSource::Host,
        });
        let meta = BlockMeta {
            start: SampleIndex(self.next),
            flags: BlockFlags::empty(),
            anchor,
            tx_time_ns: None,
        };
        self.sent_anchor = true;
        self.next += n as u64;
        Ok((written, meta))
    }
}
