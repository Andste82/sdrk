//! The band pulse detector of milestone M45.1: `band_pulse_detect_block`.
//!
//! | entry | measured | gate | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `band_pulse_detect_block` | `sdrk_blocks::BandPulseDetectBlock` (16 lanes at 250 kHz, the defaults: `auto`, 9 dB) | none (`info`) | one lane value (a time step of 16 lanes is 16 values) |
//!
//! The block calls the kernel `mag` once per piece of 64 steps over all lanes and runs the rest
//! in plain Rust: the envelope low-pass, the noise level and the trigger of every lane (a loop
//! over the 16 lanes of a step), and the detector of the lanes that are busy (scalar, one sample
//! after the other with the state of the sample before). No library has a band pulse detector
//! on the lanes of a channelizer: the natural opponent, `rtl_433`'s `pulse_detect_package`, is
//! GPL and is not used or linked (D962). So the entry is counted against our own code as
//! information (`Family::Own`), and `ours` and `fallback` run the same code.
//!
//! Workload: the lanes of the flow `ism433_pulses`, prepared once with `ChannelizeBlock` (16
//! channels, the defaults) from a band of 2 MHz with uniform noise and, in every period of
//! 40 ms, two PWM transmissions (24 bits, 200 / 400 us pulses, 200 us gaps) at -310 kHz and
//! +190 kHz and one FSK transmission (24 bits of 100 us, deviation 30 kHz) at +560 kHz, so two
//! or three of the 16 lanes are busy about a third of the time. The detector reads `n / 16`
//! steps per call cyclically from a buffer of whole periods.

use std::f64::consts::TAU;
use std::hint::black_box;

use sdrk::{C32, Rate};
use sdrk_blocks::{
    ArenaStorage, BandPulseDetectBlock, BandPulseDetectConfig, Block, ChannelizeBlock,
    ChannelizeConfig, LanesMut, LanesRef, Packet,
};
use sdrk_proto::pulses::{self, CONTINUED, Modulation};

use super::{Impl, Kernel, Layout, Pcg32};

/// Lanes of the channelizer.
const LANES: usize = 16;
/// Input rate of the band and the step rate of the lanes.
const FS: f64 = 2e6;
const STEP_HZ: u64 = 250_000;
/// Input samples of one period (40 ms).
const PERIOD_SAMPLES: usize = 80_000;
/// Steps of one period.
const PERIOD: usize = PERIOD_SAMPLES / (LANES / 2);

/// One transmission of the period: start (s), offset (Hz), FSK deviation (0: PWM).
struct Tx {
    start_s: f64,
    freq_hz: f64,
    deviation_hz: f64,
}

const TXS: [Tx; 3] = [
    Tx {
        start_s: 0.006,
        freq_hz: -310e3,
        deviation_hz: 0.0,
    },
    Tx {
        start_s: 0.012,
        freq_hz: 190e3,
        deviation_hz: 0.0,
    },
    Tx {
        start_s: 0.022,
        freq_hz: 560e3,
        deviation_hz: 30e3,
    },
];

/// The band of one period: noise of amplitude 0.01 and the three transmissions of amplitude
/// 0.1, with the bits of `rng`.
fn band(rng: &mut Pcg32) -> Vec<C32> {
    let mut x: Vec<C32> = (0..PERIOD_SAMPLES)
        .map(|_| C32::new(rng.unit() * 0.01, rng.unit() * 0.01))
        .collect();
    for tx in &TXS {
        let bits: Vec<bool> = (0..24).map(|_| rng.next_u32() & 1 == 1).collect();
        // The runs in µs: (on, frequency sign) per run.
        let mut runs: Vec<(f64, bool, f64)> = Vec::new();
        for &b in &bits {
            if tx.deviation_hz > 0.0 {
                runs.push((100.0, true, if b { 1.0 } else { -1.0 }));
            } else {
                runs.push((if b { 200.0 } else { 400.0 }, true, 0.0));
                runs.push((200.0, false, 0.0));
            }
        }
        let mut t_us = tx.start_s * 1e6;
        let mut phase = 0.0f64;
        for (us, on, sign) in runs {
            let n0 = (t_us * FS / 1e6) as usize;
            let n1 = ((t_us + us) * FS / 1e6) as usize;
            let f = tx.freq_hz + sign * tx.deviation_hz;
            for v in x.iter_mut().take(n1.min(PERIOD_SAMPLES)).skip(n0) {
                phase += TAU * f / FS;
                if on {
                    *v += C32::new((0.1 * phase.cos()) as f32, (0.1 * phase.sin()) as f32);
                }
            }
            t_us += us;
        }
    }
    x
}

fn config() -> BandPulseDetectConfig {
    BandPulseDetectConfig::default()
}

/// An arena of `bytes` (plus the alignment slack) that lives as long as the measurement binary.
fn leaked_arena(bytes: usize) -> ArenaStorage<'static> {
    let mem: &'static mut [u8] = Box::leak(vec![0u8; bytes + 16].into_boxed_slice());
    ArenaStorage::new(mem)
}

/// The block `band_pulse_detect` over `n` lane values (`n / 16` steps) per call.
pub(crate) struct BandPulseDetectBlockK {
    block: BandPulseDetectBlock<ArenaStorage<'static>>,
    /// Whole periods of lanes and then `n` more values.
    lanes: Vec<C32>,
    /// Steps per call.
    steps: usize,
    at: usize,
    got: Vec<Packet>,
    produced: usize,
}

impl BandPulseDetectBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let x = band(rng);
        let chan_cfg = ChannelizeConfig::default();
        let bytes = ChannelizeBlock::<ArenaStorage<'static>>::storage_bytes(&chan_cfg)
            .expect("channelize storage");
        let mut chan = ChannelizeBlock::new(&chan_cfg, &mut leaked_arena(bytes)).expect("chan");
        // Two periods through the channelizer; the second (settled) one is the period.
        let mut two = x.clone();
        two.extend_from_slice(&x);
        let mut all = vec![C32::default(); 2 * PERIOD * LANES];
        chan.process(&two, LanesMut::new(&mut all, LANES).expect("lanes"))
            .expect("channelize");
        let mut lanes = all[PERIOD * LANES..].to_vec();
        let steps = (layout.n / LANES).max(1);
        while lanes.len() < (PERIOD + steps) * LANES {
            lanes.extend_from_within(..PERIOD * LANES);
        }
        let cfg = config();
        let bytes =
            BandPulseDetectBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, LANES).expect("bytes");
        let block = BandPulseDetectBlock::new(
            &cfg,
            Rate::hz(STEP_HZ).expect("rate"),
            LANES,
            &mut leaked_arena(bytes),
        )
        .expect("band_pulse_detect");
        BandPulseDetectBlockK {
            block,
            lanes,
            steps,
            at: 0,
            got: vec![Packet::default(); 256],
            produced: 0,
        }
    }
}

#[inline(never)]
fn detect_call(
    block: &mut BandPulseDetectBlock<ArenaStorage<'static>>,
    lanes: &[C32],
    got: &mut [Packet],
) -> usize {
    match LanesRef::new(black_box(lanes), LANES) {
        Ok(view) => block.process(view, got).map_or(0, |p| p.produced),
        Err(_) => 0,
    }
}

impl Kernel for BandPulseDetectBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    let input = &self.lanes[self.at * LANES..(self.at + self.steps) * LANES];
                    self.produced = detect_call(&mut self.block, input, &mut self.got);
                    self.at = (self.at + self.steps) % PERIOD;
                }
                other => unreachable!("band_pulse_detect_block has no {other:?}"),
            }
            black_box(&self.got);
        }
    }

    /// 999 when three periods from a reset give the two OOK trains and the FSK frame of every
    /// period after the first (the first holds the warm-up of 4 ms and may lose a train).
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        self.at = 0;
        let (mut ook, mut fsk) = (0, 0);
        let calls = 3 * PERIOD / self.steps + 1;
        for _ in 0..calls {
            self.run(which, 1);
            for p in &self.got[..self.produced] {
                if p.flags.contains(CONTINUED) {
                    continue;
                }
                match pulses::decode(p.data()).map(|t| t.header().modulation) {
                    Ok(Modulation::Fsk) => fsk += 1,
                    Ok(_) => ook += 1,
                    Err(_) => return 0.0,
                }
            }
        }
        if (4..=6).contains(&ook) && (2..=3).contains(&fsk) {
            999.0
        } else {
            0.0
        }
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}
