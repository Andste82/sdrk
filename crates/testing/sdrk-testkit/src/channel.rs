//! Channel models: the [`Loopback`] channel simulator and the sample-stream impairments of
//! the digital-mode tests.
//!
//! # Impairments (M38)
//!
//! [`Awgn`], [`FrequencyOffset`], [`ClockDrift`] and [`Multipath`] (M43) work on `Complex64`
//! sample streams through the trait [`Impairment`]; [`ChannelChain`] applies a list of them in
//! order. Each keeps its state between calls, so a stream processed in chunks gives the same
//! samples as processed at once, and all randomness comes from a seeded [`Pcg32`]. A real
//! signal is passed as the real part.
//!
//! # Loopback
//!
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

use std::boxed::Box;
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

/// A sample-stream impairment of a test channel (M38).
pub trait Impairment: Send {
    /// Processes `input` and appends the resulting samples to `output` (the same number of
    /// samples, except for [`ClockDrift`], which resamples, and [`Multipath`], whose first
    /// [`lag`](Multipath::lag) inputs give no output). The state carries over to the next call.
    fn process(&mut self, input: &[Complex64], output: &mut Vec<Complex64>);
}

/// PCG32 stream of [`Awgn`].
const AWGN_STREAM: u64 = 0x6177_676e;

/// Additive white Gaussian noise: each sample gets independent normal I and Q components of
/// the standard deviation `sigma` (drawn in this order from a [`Pcg32`] with the seed).
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::channel::{Awgn, Impairment};
/// let mut a = Awgn::new(0.1, 498)?;
/// let mut out = Vec::new();
/// a.process(&[Complex64::new(1.0, 0.0); 4], &mut out);
/// assert_eq!(out.len(), 4);
/// assert!((out[0].re - 1.0).abs() < 1.0);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Awgn {
    sigma: f64,
    rng: Pcg32,
}

impl Awgn {
    /// Noise of the standard deviation `sigma` per component.
    ///
    /// # Errors
    /// `InvalidParameter` if `sigma` is negative or not finite.
    ///
    /// ```
    /// use sdrk_testkit::channel::Awgn;
    /// assert!(Awgn::new(-1.0, 1).is_err());
    /// ```
    pub fn new(sigma: f64, seed: u64) -> Result<Self, Error> {
        if !(sigma.is_finite() && sigma >= 0.0) {
            return Err(Error::InvalidParameter(
                "the noise sigma must be finite and not negative",
            ));
        }
        Ok(Self {
            sigma,
            rng: Pcg32::new(seed, AWGN_STREAM),
        })
    }

    /// Noise for the bit energy to noise density ratio `ebn0_db` of a signal with
    /// `bits_per_symbol` bits per symbol, `sps` samples per symbol and the mean power per
    /// sample `signal_power`: `sigma^2 = signal_power sps / (2 bits_per_symbol 10^(ebn0_db / 10))`
    /// per component (so `N0 = 2 sigma^2` is the complex noise power per sample).
    ///
    /// # Errors
    /// `InvalidParameter` for a non-finite `ebn0_db`, `bits_per_symbol == 0`, a non-positive
    /// or non-finite `sps`, or a negative or non-finite `signal_power`.
    ///
    /// ```
    /// use sdrk_testkit::channel::Awgn;
    /// let a = Awgn::from_ebn0(0.0, 1, 2.0, 1.0, 1)?;
    /// assert!((a.sigma() - 1.0).abs() < 1e-15);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn from_ebn0(
        ebn0_db: f64,
        bits_per_symbol: u32,
        sps: f64,
        signal_power: f64,
        seed: u64,
    ) -> Result<Self, Error> {
        let ok = ebn0_db.is_finite()
            && bits_per_symbol > 0
            && sps.is_finite()
            && sps > 0.0
            && signal_power.is_finite()
            && signal_power >= 0.0;
        if !ok {
            return Err(Error::InvalidParameter(
                "eb/n0 must be finite, bits per symbol and sps positive, the signal power not negative",
            ));
        }
        let ebn0 = libm::pow(10.0, ebn0_db / 10.0);
        let var = signal_power * sps / (2.0 * f64::from(bits_per_symbol) * ebn0);
        Self::new(libm::sqrt(var), seed)
    }

    /// The standard deviation per component.
    ///
    /// ```
    /// use sdrk_testkit::channel::Awgn;
    /// assert_eq!(Awgn::new(0.5, 1)?.sigma(), 0.5);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl Impairment for Awgn {
    fn process(&mut self, input: &[Complex64], output: &mut Vec<Complex64>) {
        output.reserve(input.len());
        for x in input {
            let re = self.rng.normal() * self.sigma;
            let im = self.rng.normal() * self.sigma;
            output.push(*x + Complex64::new(re, im));
        }
    }
}

/// A frequency offset: sample `n` (counted from the first sample processed) is multiplied by
/// `exp(j 2 pi frac(offset_hz n / fs))`, the phase computed in f64 from the index.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::channel::{FrequencyOffset, Impairment};
/// let mut f = FrequencyOffset::new(250.0, 1000.0)?; // a quarter turn per sample
/// let mut out = Vec::new();
/// f.process(&[Complex64::new(1.0, 0.0); 2], &mut out);
/// assert!((out[1] - Complex64::new(0.0, 1.0)).norm_sqr() < 1e-30);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct FrequencyOffset {
    offset_hz: f64,
    fs: f64,
    index: u64,
}

impl FrequencyOffset {
    /// An offset of `offset_hz` at the sample rate `fs` (both in hertz).
    ///
    /// # Errors
    /// `InvalidParameter` if `offset_hz` is not finite or `fs` is not finite and positive.
    ///
    /// ```
    /// use sdrk_testkit::channel::FrequencyOffset;
    /// assert!(FrequencyOffset::new(10.0, 0.0).is_err());
    /// ```
    pub fn new(offset_hz: f64, fs: f64) -> Result<Self, Error> {
        if !(offset_hz.is_finite() && fs.is_finite() && fs > 0.0) {
            return Err(Error::InvalidParameter(
                "the offset must be finite and the sample rate finite and positive",
            ));
        }
        Ok(Self {
            offset_hz,
            fs,
            index: 0,
        })
    }
}

impl Impairment for FrequencyOffset {
    fn process(&mut self, input: &[Complex64], output: &mut Vec<Complex64>) {
        output.reserve(input.len());
        for x in input {
            let turns = self.offset_hz * self.index as f64 / self.fs;
            let phase = 2.0 * core::f64::consts::PI * (turns - libm::floor(turns));
            output.push(*x * Complex64::new(libm::cos(phase), libm::sin(phase)));
            self.index = self.index.wrapping_add(1);
        }
    }
}

/// Taps per side of the [`ClockDrift`] interpolator.
pub const DRIFT_TAPS_PER_SIDE: usize = 32;
const DRIFT_TAPS: usize = 2 * DRIFT_TAPS_PER_SIDE;
/// Coefficients of the 4-term Blackman–Harris window of the [`ClockDrift`] interpolator.
const BLACKMAN_HARRIS: [f64; 4] = [0.35875, 0.48829, 0.14128, 0.01168];

/// A sample-clock drift: the output sample `m` is the input signal at the position
/// `p = m (1 + ppm 1e-6)` (input samples before index 0 are zero), interpolated in f64 with a
/// windowed sinc of 32 taps per side (input indices `floor(p) - 31 ..= floor(p) + 32`, window
/// 4-term Blackman–Harris over `|p - index| < 32`). A positive `ppm` compresses the signal in
/// time: a tone of `f` cycles per sample comes out at `f (1 + ppm 1e-6)`, and an event at the
/// input position `q` comes out at [`ClockDrift::output_position`]`(q) = q / (1 + ppm 1e-6)`.
/// Output `m` is produced once input `floor(p) + 32` has arrived, so the output lags by 32
/// input samples in processing (not in position).
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::channel::{ClockDrift, Impairment};
/// let mut d = ClockDrift::new(0.0)?;
/// let x: Vec<Complex64> = (0..100).map(|n| Complex64::new(n as f64, 0.0)).collect();
/// let mut y = Vec::new();
/// d.process(&x, &mut y);
/// assert_eq!(y.len(), 100 - 32);
/// assert_eq!(y[..], x[..68]); // zero drift is an exact identity
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ClockDrift {
    ratio: f64,
    /// Input samples from the absolute index `base` on.
    hist: Vec<Complex64>,
    base: u64,
    /// Next output index.
    next: u64,
    /// `cos` and `sin` of `q pi k / 32` for each tap `k = 0..64` and `q = 1..=3` (window
    /// recurrences).
    rot: Vec<[(f64, f64); 3]>,
}

impl ClockDrift {
    /// A drift of `ppm` parts per million (`|ppm| < 1e5`).
    ///
    /// # Errors
    /// `InvalidParameter` if `ppm` is not finite or `|ppm| >= 1e5`.
    ///
    /// ```
    /// use sdrk_testkit::channel::ClockDrift;
    /// assert!((ClockDrift::new(100.0)?.ratio() - 1.0001).abs() < 1e-15);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn new(ppm: f64) -> Result<Self, Error> {
        if !(ppm.is_finite() && libm::fabs(ppm) < 1e5) {
            return Err(Error::InvalidParameter(
                "the drift must be finite and below 1e5 ppm",
            ));
        }
        Ok(Self {
            ratio: 1.0 + ppm * 1e-6,
            hist: Vec::new(),
            base: 0,
            next: 0,
            rot: sinc_rotations(),
        })
    }

    /// The resampling ratio `1 + ppm 1e-6` (input samples per output sample).
    ///
    /// ```
    /// use sdrk_testkit::channel::ClockDrift;
    /// assert_eq!(ClockDrift::new(0.0)?.ratio(), 1.0);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// The output position of the input position `q` (`q / ratio`).
    ///
    /// ```
    /// use sdrk_testkit::channel::ClockDrift;
    /// assert!((ClockDrift::new(-50.0)?.output_position(1e6) - 1e6 / 0.99995).abs() < 1e-6);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn output_position(&self, q: f64) -> f64 {
        q / self.ratio
    }

    /// The 64 taps for the fractional position `mu` in `[0, 1)` (see [`sinc_taps`]).
    fn taps(&self, mu: f64) -> [f64; DRIFT_TAPS] {
        sinc_taps(&self.rot, mu)
    }
}

/// `cos` and `sin` of `q pi k / 32` for each tap `k = 0..64` and `q = 1..=3` (the window
/// recurrences of [`sinc_taps`]).
fn sinc_rotations() -> Vec<[(f64, f64); 3]> {
    let mut rot = vec![[(0.0, 0.0); 3]; DRIFT_TAPS];
    for (k, row) in rot.iter_mut().enumerate() {
        for (q, e) in row.iter_mut().enumerate() {
            let a = (q + 1) as f64 * core::f64::consts::PI * k as f64 / DRIFT_TAPS_PER_SIDE as f64;
            *e = (libm::cos(a), libm::sin(a));
        }
    }
    rot
}

/// The 64 taps of the windowed-sinc interpolator of M38 (shared by [`ClockDrift`] and
/// [`Multipath`]) for the fractional position `mu` in `[0, 1)`: tap `k` weights the input
/// `floor(p) - 31 + k`, at the distance `u = k - 31 - mu`; `rot` is [`sinc_rotations`].
fn sinc_taps(rot: &[[(f64, f64); 3]], mu: f64) -> [f64; DRIFT_TAPS] {
    let mut taps = [0.0; DRIFT_TAPS];
    let half = DRIFT_TAPS_PER_SIDE as f64;
    // sin(pi u) = (-1)^k sin(pi mu) = (-1)^k sin(pi (1 - mu)); the smaller argument keeps
    // the relative precision when mu is close to 1 (1 - mu is exact there)
    let s = libm::sin(core::f64::consts::PI * if mu > 0.5 { 1.0 - mu } else { mu });
    // window angle theta_k = pi u / 32 = theta_0 + pi k / 32
    let theta0 = core::f64::consts::PI * (-(half - 1.0) - mu) / half;
    let mut base = [(0.0, 0.0); 3];
    for (q, b) in base.iter_mut().enumerate() {
        let a = (q + 1) as f64 * theta0;
        *b = (libm::cos(a), libm::sin(a));
    }
    let [w0, w1, w2, w3] = BLACKMAN_HARRIS;
    for (k, (t, rk)) in taps.iter_mut().zip(rot.iter()).enumerate() {
        let u = k as f64 - (half - 1.0) - mu;
        if u == 0.0 {
            *t = 1.0;
            continue;
        }
        if libm::fabs(u) >= half {
            *t = 0.0;
            continue;
        }
        let mut w = w0;
        for (((c0, s0), (ck, sk)), wq) in base.iter().zip(rk).zip([w1, w2, w3]) {
            w += wq * (c0 * ck - s0 * sk);
        }
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        *t = sign * s / (core::f64::consts::PI * u) * w;
    }
    taps
}

impl Impairment for ClockDrift {
    fn process(&mut self, input: &[Complex64], output: &mut Vec<Complex64>) {
        self.hist.extend_from_slice(input);
        let end = self.base + self.hist.len() as u64; // one past the last input index
        let half = DRIFT_TAPS_PER_SIDE as i64;
        loop {
            let p = self.next as f64 * self.ratio;
            let i = libm::floor(p) as i64;
            let needed = i + half; // last input index of the window
            if needed < 0 || needed as u64 >= end {
                break;
            }
            let taps = self.taps(p - i as f64);
            let mut acc = Complex64::new(0.0, 0.0);
            for (k, g) in taps.iter().enumerate() {
                let idx = i - (half - 1) + k as i64;
                if idx < 0 || *g == 0.0 {
                    continue;
                }
                let rel = (idx as u64).wrapping_sub(self.base);
                if let Some(x) = usize::try_from(rel).ok().and_then(|r| self.hist.get(r)) {
                    acc += *x * *g;
                }
            }
            output.push(acc);
            self.next += 1;
        }
        // keep the inputs from the first index of the next window on
        let p = self.next as f64 * self.ratio;
        let keep_from = (libm::floor(p) as i64 - (half - 1)).max(0) as u64;
        if keep_from > self.base {
            let drop = usize::try_from(keep_from - self.base)
                .unwrap_or(usize::MAX)
                .min(self.hist.len());
            self.hist.drain(..drop);
            self.base += drop as u64;
        }
    }
}

/// `m e^(j ph)` with the `libm` cosine and sine.
fn polar(m: f64, ph: f64) -> Complex64 {
    Complex64::new(m * libm::cos(ph), m * libm::sin(ph))
}

/// Largest delay of a [`Multipath`] tap, in samples.
pub const MULTIPATH_MAX_DELAY: f64 = 65_536.0;

/// A static multipath channel (M43): the output is the sum of delayed and weighted copies of
/// the input, `y[n] = sum_t g_t x(n - d_t)` over the taps `(d_t, g_t)` (the delay in samples,
/// `0 <= d_t <=` [`MULTIPATH_MAX_DELAY`], the complex gain). Input samples before index 0 are
/// zero. An integer delay shifts the index exactly; a fractional delay is applied with the
/// windowed-sinc interpolator of [`ClockDrift`] (32 taps per side, 4-term Blackman–Harris
/// window), all in f64. The taps are combined into one impulse response at construction.
///
/// A fractional delay below 32 samples needs input samples after `n`, so output `n` is produced
/// once input `n + lag()` has arrived: the first [`lag`](Multipath::lag) inputs produce no output
/// (as for [`ClockDrift`], a lag in processing, not in position). Integer delays have lag 0.
///
/// The presets [`mild`](Multipath::mild) and [`severe`](Multipath::severe) take their delays in
/// symbols ([`Multipath::MILD`], [`Multipath::SEVERE`]) and scale them by the samples per
/// symbol `sps`.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::channel::{Impairment, Multipath};
/// // A direct path and an echo of half the amplitude two samples later.
/// let mut ch = Multipath::new(&[(0.0, Complex64::new(1.0, 0.0)), (2.0, Complex64::new(0.5, 0.0))])?;
/// assert_eq!(ch.lag(), 0);
/// let mut y = Vec::new();
/// ch.process(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)], &mut y);
/// assert_eq!(y, [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Multipath {
    /// The combined impulse response: `h[m]` weights the input `n + lag - m` of output `n`.
    h: Vec<Complex64>,
    lag: usize,
    /// The last `h.len()` inputs, oldest first (zeros before the first input).
    hist: std::collections::VecDeque<Complex64>,
    /// Inputs still to be taken before the first output.
    pending: usize,
}

impl Multipath {
    /// The preset `mild` of M43 section 4: `(delay in symbols, magnitude, phase in radians)` of
    /// each tap, `1`, `0.3 e^(j1)` after 1.3 symbols and `0.1 e^(-j2)` after 2.7 symbols.
    pub const MILD: [(f64, f64, f64); 3] = [(0.0, 1.0, 0.0), (1.3, 0.3, 1.0), (2.7, 0.1, -2.0)];

    /// The preset `severe` of M43 section 4: `1`, `0.6 e^(j0.5)` after 0.8 symbols,
    /// `0.4 e^(-j1.5)` after 2.1 symbols and `0.2` after 4 symbols.
    pub const SEVERE: [(f64, f64, f64); 4] = [
        (0.0, 1.0, 0.0),
        (0.8, 0.6, 0.5),
        (2.1, 0.4, -1.5),
        (4.0, 0.2, 0.0),
    ];

    /// A channel with the taps `(delay in samples, gain)`.
    ///
    /// # Errors
    /// `InvalidParameter` if there is no tap, a delay is not finite or outside
    /// `0 ..= MULTIPATH_MAX_DELAY`, or a gain is not finite.
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use sdrk_testkit::channel::Multipath;
    /// let one = Complex64::new(1.0, 0.0);
    /// assert_eq!(Multipath::new(&[(0.0, one), (2.5, one)])?.lag(), 29);
    /// assert!(Multipath::new(&[]).is_err());
    /// assert!(Multipath::new(&[(-1.0, one)]).is_err());
    /// assert!(Multipath::new(&[(1.0, Complex64::new(f64::NAN, 0.0))]).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn new(taps: &[(f64, Complex64)]) -> Result<Self, Error> {
        let valid = |(d, g): &(f64, Complex64)| {
            d.is_finite()
                && (0.0..=MULTIPATH_MAX_DELAY).contains(d)
                && g.re.is_finite()
                && g.im.is_finite()
        };
        if taps.is_empty() || !taps.iter().all(valid) {
            return Err(Error::InvalidParameter(
                "multipath needs at least one tap, delays from 0 to 65536 samples and finite gains",
            ));
        }
        // The contributions (lag, gain): lag j weights the input n - j of output n.
        let rot = sinc_rotations();
        let mut parts: Vec<(i64, Complex64)> = Vec::new();
        let half = DRIFT_TAPS_PER_SIDE as i64;
        for &(d, g) in taps {
            let whole = libm::floor(d);
            let frac = d - whole;
            let whole = whole as i64;
            if frac == 0.0 {
                parts.push((whole, g));
                continue;
            }
            // p = n - d: floor(p) = n - whole - 1 and mu = p - floor(p) = 1 - frac; tap k weights
            // the input floor(p) - 31 + k, which is n - (whole + 32 - k).
            for (k, c) in sinc_taps(&rot, 1.0 - frac).iter().enumerate() {
                if *c != 0.0 {
                    parts.push((whole + half - k as i64, g * *c));
                }
            }
        }
        let min = parts.iter().map(|p| p.0).min().unwrap_or(0);
        let max = parts.iter().map(|p| p.0).max().unwrap_or(0);
        let lag = usize::try_from(-min.min(0)).unwrap_or(0);
        let len = usize::try_from(max - min.min(0) + 1).unwrap_or(1);
        let mut h = vec![Complex64::new(0.0, 0.0); len];
        for (j, g) in parts {
            if let Some(v) = usize::try_from(j + lag as i64)
                .ok()
                .and_then(|m| h.get_mut(m))
            {
                *v += g;
            }
        }
        let hist = core::iter::repeat_n(Complex64::new(0.0, 0.0), len).collect();
        Ok(Self {
            h,
            lag,
            hist,
            pending: lag,
        })
    }

    /// The preset `taps` (delay in symbols, magnitude, phase) with the delays scaled by `sps`.
    fn preset(taps: &[(f64, f64, f64)], sps: f64) -> Result<Self, Error> {
        if !(sps.is_finite() && sps > 0.0) {
            return Err(Error::InvalidParameter(
                "the samples per symbol must be finite and positive",
            ));
        }
        let scaled: Vec<(f64, Complex64)> = taps
            .iter()
            .map(|&(d, m, ph)| (d * sps, polar(m, ph)))
            .collect();
        Self::new(&scaled)
    }

    /// The preset [`MILD`](Multipath::MILD) at `sps` samples per symbol (`sps` 1 for a channel on
    /// the symbol stream).
    ///
    /// # Errors
    /// `InvalidParameter` if `sps` is not finite and positive, or a scaled delay exceeds
    /// [`MULTIPATH_MAX_DELAY`].
    ///
    /// ```
    /// use sdrk_testkit::channel::Multipath;
    /// let ch = Multipath::mild(4.0)?; // delays 0, 5.2 and 10.8 samples
    /// assert_eq!(ch.lag(), 31 - 5);
    /// assert!(Multipath::mild(0.0).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn mild(sps: f64) -> Result<Self, Error> {
        Self::preset(&Self::MILD, sps)
    }

    /// The preset [`SEVERE`](Multipath::SEVERE) at `sps` samples per symbol.
    ///
    /// # Errors
    /// As [`mild`](Multipath::mild).
    ///
    /// ```
    /// use sdrk_testkit::channel::Multipath;
    /// let ch = Multipath::severe(1.0)?; // delays 0, 0.8, 2.1 and 4 symbols
    /// assert_eq!(ch.lag(), 31);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn severe(sps: f64) -> Result<Self, Error> {
        Self::preset(&Self::SEVERE, sps)
    }

    /// The processing lag: output `n` is produced once input `n + lag` has arrived (0 when all
    /// delays are whole samples, otherwise `31 - floor(d)` for the smallest fractional delay
    /// `d` below 32 samples).
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use sdrk_testkit::channel::Multipath;
    /// assert_eq!(Multipath::new(&[(40.5, Complex64::new(1.0, 0.0))])?.lag(), 0);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn lag(&self) -> usize {
        self.lag
    }

    /// The combined impulse response: element `m` weights the input `n + lag - m` of output `n`.
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use sdrk_testkit::channel::Multipath;
    /// let ch = Multipath::new(&[(1.0, Complex64::new(0.0, 2.0))])?;
    /// assert_eq!(ch.response(), [Complex64::new(0.0, 0.0), Complex64::new(0.0, 2.0)]);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn response(&self) -> &[Complex64] {
        &self.h
    }
}

impl Impairment for Multipath {
    fn process(&mut self, input: &[Complex64], output: &mut Vec<Complex64>) {
        output.reserve(input.len());
        for x in input {
            self.hist.pop_front();
            self.hist.push_back(*x);
            if self.pending > 0 {
                self.pending -= 1;
                continue;
            }
            let mut acc = Complex64::new(0.0, 0.0);
            for (g, v) in self.h.iter().zip(self.hist.iter().rev()) {
                acc += *g * *v;
            }
            output.push(acc);
        }
    }
}

/// A list of impairments applied in order (the output of one is the input of the next).
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::channel::{Awgn, ChannelChain, ClockDrift, FrequencyOffset, Impairment};
/// let mut ch = ChannelChain::new()
///     .with(ClockDrift::new(50.0)?)
///     .with(FrequencyOffset::new(10.0, 48_000.0)?)
///     .with(Awgn::from_ebn0(8.0, 1, 4.1, 1.0, 495)?);
/// assert_eq!(ch.len(), 3);
/// let mut out = Vec::new();
/// ch.process(&[Complex64::new(1.0, 0.0); 100], &mut out);
/// assert_eq!(out.len(), 100 - 32);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Default)]
pub struct ChannelChain {
    stages: Vec<Box<dyn Impairment>>,
    scratch: Vec<Complex64>,
}

impl core::fmt::Debug for ChannelChain {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ChannelChain")
            .field("stages", &self.stages.len())
            .finish_non_exhaustive()
    }
}

impl ChannelChain {
    /// An empty chain (passes the input through).
    ///
    /// ```
    /// use sdrk_testkit::channel::ChannelChain;
    /// assert!(ChannelChain::new().is_empty());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a stage (builder form).
    ///
    /// ```
    /// use sdrk_testkit::channel::{Awgn, ChannelChain};
    /// assert_eq!(ChannelChain::new().with(Awgn::new(0.1, 1)?).len(), 1);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    #[must_use]
    pub fn with(mut self, stage: impl Impairment + 'static) -> Self {
        self.push(stage);
        self
    }

    /// Appends a stage.
    ///
    /// ```
    /// use sdrk_testkit::channel::{ChannelChain, ClockDrift};
    /// let mut ch = ChannelChain::new();
    /// ch.push(ClockDrift::new(-20.0)?);
    /// assert_eq!(ch.len(), 1);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn push(&mut self, stage: impl Impairment + 'static) {
        self.stages.push(Box::new(stage));
    }

    /// Number of stages.
    ///
    /// ```
    /// use sdrk_testkit::channel::ChannelChain;
    /// assert_eq!(ChannelChain::new().len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// True if the chain has no stage.
    ///
    /// ```
    /// use sdrk_testkit::channel::ChannelChain;
    /// assert!(ChannelChain::new().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

impl Impairment for ChannelChain {
    fn process(&mut self, input: &[Complex64], output: &mut Vec<Complex64>) {
        let mut cur: Vec<Complex64> = input.to_vec();
        for stage in &mut self.stages {
            self.scratch.clear();
            stage.process(&cur, &mut self.scratch);
            core::mem::swap(&mut cur, &mut self.scratch);
        }
        output.extend_from_slice(&cur);
    }
}
