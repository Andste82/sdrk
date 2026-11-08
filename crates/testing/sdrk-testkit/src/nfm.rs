//! Narrowband FM channel scenes (M45): several NFM transmitters on a channel grid, rendered as
//! one complex baseband.
//!
//! A [`ChannelScene`] describes the band: the channel spacing, the sample rate `fs`, the level
//! of the complex Gaussian noise over the whole band and, per transmitter, its channel (an offset
//! index from the centre of the band), its audio ([`Audio::Tone`], [`Audio::Noise`] or
//! [`Audio::Silence`]), its peak deviation and its level. [`ChannelScene::renderer`] checks the
//! description and returns a [`SceneRenderer`] that writes the baseband in pieces of any length:
//! every transmitter keeps its phase from one sample to the next (the phase is accumulated, so
//! it is continuous also for noise modulation), and the noise comes from seeded PCG32 streams,
//! so the same scene and seed give the same samples in any chunking.
//!
//! The signal of a transmitter at the offset index `k` with the level `L` dBFS, the peak
//! deviation `D` Hz and the modulating signal `m[n]` is
//!
//! ```text
//! s[n]     = 10^(L / 20) exp(j 2 pi phi[n])
//! phi[n+1] = frac(phi[n] + (k spacing + D m[n]) / fs),   phi[0] = 0
//! ```
//!
//! with `m[n] = sin(2 pi f n / fs)` for a tone of `f` Hz, `m[n] = 0` for silence (a plain
//! carrier) and, for noise, Gaussian noise through a second-order Butterworth low-pass at
//! [`NOISE_AUDIO_HZ`], scaled to the mean power 1/2 of a unit sine (so a noise-modulated
//! transmitter has the RMS deviation of a tone at the same peak deviation, `D / sqrt(2)`). The
//! band noise is complex Gaussian with the total power `10^(noise_db / 10)` (half per
//! component); `f64::NEG_INFINITY` leaves it out.
//!
//! The random streams: the band noise uses `Pcg32::new(seed, 0)`, the noise audio of the
//! transmitter at position `i` of [`ChannelScene::channels`] `Pcg32::new(seed, 1 + i)`.
//!
//! ```
//! use num_complex::Complex64;
//! use sdrk_testkit::nfm::{Audio, ChannelScene, SceneChannel};
//! let channels = [SceneChannel { offset_index: -1, audio: Audio::Silence, deviation_hz: 2500.0, level_db: -6.0 }];
//! let scene = ChannelScene {
//!     spacing_hz: 12_500.0,
//!     channels: &channels,
//!     fs: 200_000.0,
//!     noise_db: f64::NEG_INFINITY,
//!     seed: 1,
//! };
//! let mut out = [Complex64::new(0.0, 0.0); 16];
//! scene.renderer()?.render(&mut out);
//! // An unmodulated carrier 12.5 kHz below the centre at -6 dBFS.
//! assert!((out[3].norm_sqr().sqrt() - 10f64.powf(-6.0 / 20.0)).abs() < 1e-12);
//! assert!((out[1].im.atan2(out[1].re) + 2.0 * core::f64::consts::PI * 12_500.0 / 200_000.0).abs() < 1e-12);
//! # Ok::<(), sdrk_testkit::nfm::SceneError>(())
//! ```

use core::f64::consts::PI;

use num_complex::Complex64;

use crate::rng::Pcg32;

/// The most transmitters a scene may hold (the renderer keeps their state without a heap).
pub const MAX_SCENE_CHANNELS: usize = 32;

/// Cutoff of the low-pass that shapes the noise audio of [`Audio::Noise`] in Hz: the top of the
/// speech band of two-way radio.
pub const NOISE_AUDIO_HZ: f64 = 3_000.0;

/// Samples of the impulse response of the noise low-pass summed for its noise gain.
const GAIN_SAMPLES: usize = 1 << 17;

/// The modulating signal of one transmitter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Audio {
    /// A sine of `freq_hz` (peak deviation `deviation_hz`).
    Tone {
        /// Audio frequency in Hz.
        freq_hz: f64,
    },
    /// Low-pass Gaussian noise (see the [module documentation](self)).
    Noise,
    /// No modulation: a plain carrier.
    Silence,
}

/// One transmitter of a [`ChannelScene`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneChannel {
    /// Channel number relative to the centre of the band: the carrier is at
    /// `offset_index * spacing_hz`.
    pub offset_index: i32,
    /// The modulating signal.
    pub audio: Audio,
    /// Peak frequency deviation in Hz.
    pub deviation_hz: f64,
    /// Level of the carrier in dBFS (amplitude `10^(level_db / 20)`).
    pub level_db: f64,
}

/// A band of NFM channels (see the [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelScene<'a> {
    /// Channel spacing in Hz.
    pub spacing_hz: f64,
    /// The transmitters, at most [`MAX_SCENE_CHANNELS`].
    pub channels: &'a [SceneChannel],
    /// Sample rate in Hz.
    pub fs: f64,
    /// Total power of the complex Gaussian band noise in dBFS; `f64::NEG_INFINITY` for none.
    pub noise_db: f64,
    /// Seed of the random streams.
    pub seed: u64,
}

/// A [`ChannelScene`] that cannot be rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SceneError {
    /// More than [`MAX_SCENE_CHANNELS`] transmitters.
    TooManyChannels,
    /// A value is out of its range; the name says which.
    InvalidParameter(&'static str),
}

impl core::fmt::Display for SceneError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooManyChannels => write!(
                f,
                "a channel scene holds at most {MAX_SCENE_CHANNELS} transmitters"
            ),
            Self::InvalidParameter(name) => write!(f, "invalid scene parameter `{name}`"),
        }
    }
}

impl core::error::Error for SceneError {}

/// A second-order Butterworth low-pass (bilinear transform, direct form 1, f64).
#[derive(Debug, Clone, Copy)]
struct Lowpass {
    b: [f64; 3],
    a: [f64; 2],
    x: [f64; 2],
    y: [f64; 2],
}

impl Lowpass {
    fn new(cutoff_hz: f64, fs: f64) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / fs;
        let (sin, cos) = (libm::sin(w0), libm::cos(w0));
        let alpha = sin / (2.0 * core::f64::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cos) / 2.0 / a0;
        Self {
            b: [b0, 2.0 * b0, b0],
            a: [-2.0 * cos / a0, (1.0 - alpha) / a0],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }

    fn step(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[0] * self.y[0]
            - self.a[1] * self.y[1];
        self.x = [x, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }

    /// The standard deviation of the output for unit-variance white input: the root of the
    /// energy of the impulse response.
    fn noise_gain(cutoff_hz: f64, fs: f64) -> f64 {
        let mut f = Self::new(cutoff_hz, fs);
        let mut energy = 0.0;
        for n in 0..GAIN_SAMPLES {
            let y = f.step(if n == 0 { 1.0 } else { 0.0 });
            energy += y * y;
        }
        libm::sqrt(energy)
    }
}

/// The running state of one transmitter.
#[derive(Debug, Clone)]
struct Transmitter {
    amplitude: f64,
    /// Carrier offset in cycles per sample.
    carrier: f64,
    /// Peak deviation in cycles per sample.
    deviation: f64,
    /// Tone frequency in cycles per sample (tones only).
    tone: f64,
    audio: Audio,
    /// Phase in cycles, in `[0, 1)`.
    phase: f64,
    rng: Pcg32,
    lowpass: Lowpass,
    /// Scale of the low-pass output to the mean power 1/2.
    noise_scale: f64,
}

/// Fractional part `x - floor(x)`.
fn frac(x: f64) -> f64 {
    x - libm::floor(x)
}

impl Transmitter {
    fn modulation(&mut self, n: u64) -> f64 {
        match self.audio {
            Audio::Tone { .. } => libm::sin(2.0 * PI * frac(self.tone * n as f64)),
            Audio::Noise => self.lowpass.step(self.rng.normal()) * self.noise_scale,
            Audio::Silence => 0.0,
        }
    }
}

/// Writes the baseband of a [`ChannelScene`] piece by piece (see the
/// [module documentation](self)).
#[derive(Debug, Clone)]
pub struct SceneRenderer {
    transmitters: heapless::Vec<Transmitter, MAX_SCENE_CHANNELS>,
    /// Standard deviation of the band noise per component (0 for none).
    noise_sigma: f64,
    noise: Pcg32,
    /// Index of the next sample.
    n: u64,
}

impl ChannelScene<'_> {
    /// Checks the scene and returns a renderer at sample 0.
    ///
    /// # Errors
    /// [`SceneError::TooManyChannels`] for more than [`MAX_SCENE_CHANNELS`] transmitters;
    /// [`SceneError::InvalidParameter`] for a sample rate that is not positive and finite, a
    /// spacing, deviation, level or tone frequency that is not finite, a carrier outside
    /// `(-fs/2, fs/2)` or a `noise_db` of NaN or `+inf`.
    pub fn renderer(&self) -> Result<SceneRenderer, SceneError> {
        if self.channels.len() > MAX_SCENE_CHANNELS {
            return Err(SceneError::TooManyChannels);
        }
        if !(self.fs.is_finite() && self.fs > 0.0) {
            return Err(SceneError::InvalidParameter("fs"));
        }
        if !self.spacing_hz.is_finite() {
            return Err(SceneError::InvalidParameter("spacing_hz"));
        }
        if self.noise_db.is_nan() || self.noise_db == f64::INFINITY {
            return Err(SceneError::InvalidParameter("noise_db"));
        }
        let gain = Lowpass::noise_gain(NOISE_AUDIO_HZ, self.fs);
        let mut transmitters = heapless::Vec::new();
        for (i, c) in self.channels.iter().enumerate() {
            let carrier_hz = f64::from(c.offset_index) * self.spacing_hz;
            if !(carrier_hz.abs() < self.fs / 2.0) {
                return Err(SceneError::InvalidParameter("offset_index"));
            }
            if !c.deviation_hz.is_finite() {
                return Err(SceneError::InvalidParameter("deviation_hz"));
            }
            if !c.level_db.is_finite() {
                return Err(SceneError::InvalidParameter("level_db"));
            }
            let tone = match c.audio {
                Audio::Tone { freq_hz } if !freq_hz.is_finite() => {
                    return Err(SceneError::InvalidParameter("freq_hz"));
                }
                Audio::Tone { freq_hz } => freq_hz / self.fs,
                Audio::Noise | Audio::Silence => 0.0,
            };
            let t = Transmitter {
                amplitude: libm::pow(10.0, c.level_db / 20.0),
                carrier: carrier_hz / self.fs,
                deviation: c.deviation_hz / self.fs,
                tone,
                audio: c.audio,
                phase: 0.0,
                rng: Pcg32::new(self.seed, 1 + i as u64),
                lowpass: Lowpass::new(NOISE_AUDIO_HZ, self.fs),
                noise_scale: core::f64::consts::FRAC_1_SQRT_2 / gain,
            };
            if transmitters.push(t).is_err() {
                return Err(SceneError::TooManyChannels);
            }
        }
        Ok(SceneRenderer {
            transmitters,
            noise_sigma: libm::sqrt(libm::pow(10.0, self.noise_db / 10.0) / 2.0),
            noise: Pcg32::new(self.seed, 0),
            n: 0,
        })
    }
}

impl SceneRenderer {
    /// Writes the next `out.len()` samples.
    pub fn render(&mut self, out: &mut [Complex64]) {
        for o in out.iter_mut() {
            let n = self.n;
            let mut sum = Complex64::new(0.0, 0.0);
            for t in &mut self.transmitters {
                let arg = 2.0 * PI * t.phase;
                sum += Complex64::new(libm::cos(arg), libm::sin(arg)) * t.amplitude;
                let m = t.modulation(n);
                t.phase = frac(t.phase + t.carrier + t.deviation * m);
            }
            if self.noise_sigma > 0.0 {
                sum += Complex64::new(self.noise.normal(), self.noise.normal()) * self.noise_sigma;
            }
            *o = sum;
            self.n = n.wrapping_add(1);
        }
    }

    /// Index of the next sample to render.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.n
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn zeros(n: usize) -> Vec<Complex64> {
        vec![Complex64::new(0.0, 0.0); n]
    }

    const FS: f64 = 200_000.0;

    fn scene(channels: &[SceneChannel], noise_db: f64) -> ChannelScene<'_> {
        ChannelScene {
            spacing_hz: 12_500.0,
            channels,
            fs: FS,
            noise_db,
            seed: 7,
        }
    }

    fn all() -> [SceneChannel; 3] {
        [
            SceneChannel {
                offset_index: -3,
                audio: Audio::Tone { freq_hz: 1000.0 },
                deviation_hz: 2500.0,
                level_db: -20.0,
            },
            SceneChannel {
                offset_index: 2,
                audio: Audio::Noise,
                deviation_hz: 2500.0,
                level_db: -10.0,
            },
            SceneChannel {
                offset_index: 0,
                audio: Audio::Silence,
                deviation_hz: 2500.0,
                level_db: -30.0,
            },
        ]
    }

    #[test]
    fn chunked_rendering_equals_one_shot() {
        let channels = all();
        let s = scene(&channels, -40.0);
        let mut whole = zeros(1000);
        s.renderer().unwrap().render(&mut whole);
        let mut chunked = zeros(1000);
        let mut r = s.renderer().unwrap();
        for piece in chunked.chunks_mut(37) {
            r.render(piece);
        }
        assert_eq!(whole, chunked);
        assert_eq!(r.position(), 1000);
    }

    #[test]
    fn a_tone_moves_the_instantaneous_frequency_by_the_deviation() {
        let channels = [all()[0]];
        let mut x = zeros(400);
        scene(&channels, f64::NEG_INFINITY)
            .renderer()
            .unwrap()
            .render(&mut x);
        for n in 1..x.len() {
            let d = x[n] * x[n - 1].conj();
            let f = libm::atan2(d.im, d.re) / (2.0 * PI) * FS;
            let m = libm::sin(2.0 * PI * 1000.0 * (n - 1) as f64 / FS);
            assert!((f - (-37_500.0 + 2500.0 * m)).abs() < 1e-6, "n = {n}: {f}");
            assert!((libm::sqrt(x[n].norm_sqr()) - 0.1).abs() < 1e-12);
        }
    }

    #[test]
    fn noise_audio_has_the_power_of_a_unit_sine_and_stays_in_the_speech_band() {
        let channels = [all()[1]];
        let mut r = scene(&channels, f64::NEG_INFINITY).renderer().unwrap();
        let t = &mut r.transmitters[0];
        let n = 400_000u64;
        let (mut power, mut diff) = (0.0, 0.0);
        let mut last = 0.0;
        for i in 0..n {
            let m = t.modulation(i);
            power += m * m;
            diff += (m - last) * (m - last);
            last = m;
        }
        let power = power / n as f64;
        assert!((power - 0.5).abs() < 0.02, "mean power {power}");
        // The RMS frequency of the noise audio, from the power of its difference: well below
        // 3 kHz times a small factor for a second-order skirt.
        let rms_hz = libm::sqrt(diff / n as f64 / power) / (2.0 * PI) * FS;
        assert!(rms_hz < 4_000.0, "RMS frequency {rms_hz} Hz");
    }

    #[test]
    fn the_band_noise_has_the_given_power() {
        let mut x = zeros(20_000);
        scene(&[], -20.0).renderer().unwrap().render(&mut x);
        let p = x.iter().map(Complex64::norm_sqr).sum::<f64>() / x.len() as f64;
        assert!((10.0 * libm::log10(p) + 20.0).abs() < 0.1, "{p}");
    }

    #[test]
    fn the_seed_selects_the_noise() {
        let channels = all();
        let mut a = zeros(64);
        let mut b = zeros(64);
        scene(&channels, -40.0).renderer().unwrap().render(&mut a);
        let mut other = scene(&channels, -40.0);
        other.seed = 8;
        other.renderer().unwrap().render(&mut b);
        assert_ne!(a, b);
    }

    #[test]
    fn invalid_scenes_are_rejected() {
        let mut c = all();
        assert_eq!(
            scene(&c, f64::NAN).renderer().err(),
            Some(SceneError::InvalidParameter("noise_db"))
        );
        c[0].offset_index = 8; // 100 kHz: half the sample rate
        assert_eq!(
            scene(&c, -40.0).renderer().err(),
            Some(SceneError::InvalidParameter("offset_index"))
        );
        c[0].offset_index = 0;
        c[0].audio = Audio::Tone { freq_hz: f64::NAN };
        assert_eq!(
            scene(&c, -40.0).renderer().err(),
            Some(SceneError::InvalidParameter("freq_hz"))
        );
        let mut s = scene(&[], -40.0);
        s.fs = 0.0;
        assert_eq!(s.renderer().err(), Some(SceneError::InvalidParameter("fs")));
        let many = vec![all()[0]; MAX_SCENE_CHANNELS + 1];
        assert_eq!(
            scene(&many, -40.0).renderer().err(),
            Some(SceneError::TooManyChannels)
        );
        assert!(scene(&many[..MAX_SCENE_CHANNELS], -40.0).renderer().is_ok());
    }
}
