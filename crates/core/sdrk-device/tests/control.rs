//! Control capabilities and async streams: a mock driver implements every trait.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unused_async_trait_impl,
    clippy::unnecessary_literal_bound
)]

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};

use sdrk_device::formats;
use sdrk_device::*;

fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}

/// Runs a future to completion; the mock futures never return `Pending`.
fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

/// A future that is pending exactly once, to check that the executor really polls.
struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: core::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

static FORMATS: [SampleFormat; 2] = [formats::CS8, formats::CS16];
static CAPS: Capabilities = Capabilities {
    rx_channels: 2,
    tx_channels: 1,
    full_duplex: true,
    formats: &FORMATS,
    hw_timestamps: true,
    timed_commands: true,
};
static FREQ: [Range; 1] = [Range {
    min: 1e6,
    max: 6e9,
    step: 0.0,
}];
static RATES: [Range; 2] = [
    Range {
        min: 1e6,
        max: 20e6,
        step: 0.0,
    },
    Range {
        min: 40e6,
        max: 40e6,
        step: 0.0,
    },
];
static GAINS: [(&str, Range); 2] = [
    (
        "lna",
        Range {
            min: 0.0,
            max: 40.0,
            step: 8.0,
        },
    ),
    (
        "vga",
        Range {
            min: 0.0,
            max: 62.0,
            step: 2.0,
        },
    ),
];
static BANDWIDTHS: [Range; 1] = [Range {
    min: 1.75e6,
    max: 28e6,
    step: 0.0,
}];

#[derive(Debug)]
struct Mock {
    freq: [f64; 2],
    rate: Rate,
    gain: f64,
    agc: bool,
    bw: f64,
    clock_ns: i64,
    scheduled: Vec<(CommandTime, Command)>,
    written: usize,
}

impl Mock {
    fn new() -> Self {
        Self {
            freq: [100e6, 0.0],
            rate: Rate::hz(2_000_000).unwrap(),
            gain: 0.0,
            agc: false,
            bw: 1.75e6,
            clock_ns: 0,
            scheduled: Vec::new(),
            written: 0,
        }
    }
}

impl Device for Mock {
    type Error = Error;
    fn capabilities(&self) -> &Capabilities {
        &CAPS
    }
    fn driver(&self) -> &str {
        "mock"
    }
    fn serial(&self) -> Option<&str> {
        Some("42")
    }
}

impl Tuner for Mock {
    type Error = Error;
    fn set_frequency(&mut self, dir: Direction, ch: u8, hz: f64) -> Result<f64, Error> {
        if !FREQ.iter().any(|r| r.contains(hz)) {
            return Err(Error::Overflow("frequency"));
        }
        let slot = usize::from(dir == Direction::Tx);
        if ch != 0 && slot == 1 {
            return Err(Error::Overflow("channel"));
        }
        // The mock tunes in 1 kHz steps.
        self.freq[slot] = (hz / 1000.0).round() * 1000.0;
        Ok(self.freq[slot])
    }
    fn frequency(&self, dir: Direction, _ch: u8) -> Result<f64, Error> {
        Ok(self.freq[usize::from(dir == Direction::Tx)])
    }
    fn frequency_ranges(&self, _dir: Direction, _ch: u8) -> &[Range] {
        &FREQ
    }
}

impl RateControl for Mock {
    type Error = Error;
    fn set_sample_rate(&mut self, _dir: Direction, _ch: u8, rate: Rate) -> Result<Rate, Error> {
        if !RATES.iter().any(|r| r.contains(rate.as_f64())) {
            return Err(Error::Overflow("rate"));
        }
        self.rate = rate;
        Ok(rate)
    }
    fn sample_rates(&self, _dir: Direction, _ch: u8) -> &[Range] {
        &RATES
    }
}

impl GainControl for Mock {
    type Error = Error;
    fn gain_stages(&self, _dir: Direction, _ch: u8) -> &[(&'static str, Range)] {
        &GAINS
    }
    fn set_gain(
        &mut self,
        _dir: Direction,
        _ch: u8,
        stage: Option<&str>,
        db: f64,
    ) -> Result<f64, Error> {
        if let Some(name) = stage
            && !GAINS.iter().any(|(n, _)| *n == name)
        {
            return Err(Error::InvalidFormat("unknown gain stage"));
        }
        self.gain = db.clamp(0.0, 102.0);
        Ok(self.gain)
    }
    fn set_agc(&mut self, _dir: Direction, _ch: u8, on: bool) -> Result<(), Error> {
        self.agc = on;
        Ok(())
    }
}

impl BandwidthControl for Mock {
    type Error = Error;
    fn set_bandwidth(&mut self, _dir: Direction, _ch: u8, hz: f64) -> Result<f64, Error> {
        self.bw = hz.clamp(1.75e6, 28e6);
        Ok(self.bw)
    }
    fn bandwidths(&self, _dir: Direction, _ch: u8) -> &[Range] {
        &BANDWIDTHS
    }
}

impl HardwareTime for Mock {
    type Error = Error;
    fn hardware_time(&self) -> Result<TimeAnchor, Error> {
        Ok(TimeAnchor {
            index: SampleIndex(0),
            unix_ns: self.clock_ns,
            source: TimeSource::Device,
        })
    }
    fn set_hardware_time(&mut self, unix_ns: i64) -> Result<(), Error> {
        self.clock_ns = unix_ns;
        Ok(())
    }
}

impl TimedCommands for Mock {
    type Error = Error;
    fn schedule(&mut self, at: CommandTime, cmd: Command) -> Result<(), Error> {
        if self.scheduled.len() >= 4 {
            return Err(Error::BufferTooSmall { needed: 5, got: 4 });
        }
        self.scheduled.push((at, cmd));
        Ok(())
    }
}

impl AsyncRxStream for Mock {
    type Error = Error;
    fn format(&self) -> SampleFormat {
        formats::CS8
    }
    fn sample_rate(&self) -> Rate {
        self.rate
    }
    async fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Error> {
        YieldOnce(false).await;
        let n = buf.len() & !1;
        buf.get_mut(..n).unwrap().fill(7);
        Ok((n, BlockMeta::default()))
    }
}

impl AsyncTxStream for Mock {
    type Error = Error;
    fn format(&self) -> SampleFormat {
        formats::CS8
    }
    fn sample_rate(&self) -> Rate {
        self.rate
    }
    async fn write(&mut self, buf: &[u8], _meta: &BlockMeta) -> Result<usize, Error> {
        YieldOnce(false).await;
        self.written += buf.len();
        Ok(buf.len())
    }
    async fn poll_status(&mut self) -> Result<Option<BlockMeta>, Error> {
        Ok(None)
    }
}

#[test]
fn range_contains_handles_edges_and_steps() {
    let cont = Range {
        min: 1.0,
        max: 2.0,
        step: 0.0,
    };
    assert!(cont.contains(1.0) && cont.contains(2.0) && cont.contains(1.5));
    assert!(!cont.contains(0.999) && !cont.contains(2.001));
    assert!(!cont.contains(f64::NAN));
    let stepped = Range {
        min: 0.0,
        max: 40.0,
        step: 8.0,
    };
    assert!(stepped.contains(0.0) && stepped.contains(40.0) && stepped.contains(24.0));
    assert!(!stepped.contains(4.0) && !stepped.contains(41.0));
    let offset = Range {
        min: 1.0,
        max: 5.0,
        step: 2.0,
    };
    assert!(offset.contains(3.0) && !offset.contains(2.0));
}

#[test]
fn mock_driver_implements_every_trait() {
    let mut m = Mock::new();
    assert_eq!(Device::driver(&m), "mock");
    assert_eq!(m.serial(), Some("42"));
    let caps = *m.capabilities();
    assert_eq!((caps.rx_channels, caps.tx_channels), (2, 1));
    assert!(caps.full_duplex && caps.hw_timestamps && caps.timed_commands);
    assert_eq!(caps.formats, &[formats::CS8, formats::CS16]);

    assert_eq!(
        m.set_frequency(Direction::Rx, 0, 100.0004e6).unwrap(),
        100e6
    );
    assert_eq!(m.frequency(Direction::Rx, 0).unwrap(), 100e6);
    assert_eq!(
        m.set_frequency(Direction::Rx, 0, 1.0).unwrap_err().kind(),
        ErrorKind::OutOfRange
    );
    assert_eq!(m.frequency_ranges(Direction::Rx, 0).len(), 1);

    let r = m
        .set_sample_rate(Direction::Rx, 0, Rate::hz(10_000_000).unwrap())
        .unwrap();
    assert_eq!(r, Rate::hz(10_000_000).unwrap());
    assert!(
        m.set_sample_rate(Direction::Rx, 0, Rate::hz(30_000_000).unwrap())
            .is_err()
    );
    assert_eq!(m.sample_rates(Direction::Rx, 0).len(), 2);

    assert_eq!(m.gain_stages(Direction::Rx, 0)[0].0, "lna");
    assert_eq!(
        m.set_gain(Direction::Rx, 0, Some("lna"), 16.0).unwrap(),
        16.0
    );
    assert_eq!(m.set_gain(Direction::Rx, 0, None, 500.0).unwrap(), 102.0);
    assert!(m.set_gain(Direction::Rx, 0, Some("nope"), 1.0).is_err());
    m.set_agc(Direction::Rx, 0, true).unwrap();
    assert!(m.agc);

    assert_eq!(m.set_bandwidth(Direction::Rx, 0, 1e3).unwrap(), 1.75e6);
    assert_eq!(m.bandwidths(Direction::Tx, 0).len(), 1);

    m.set_hardware_time(1_700_000_000_000_000_000).unwrap();
    let a = m.hardware_time().unwrap();
    assert_eq!(a.unix_ns, 1_700_000_000_000_000_000);
    assert_eq!(a.source, TimeSource::Device);

    for i in 0..4 {
        m.schedule(
            CommandTime::Index(SampleIndex(i)),
            Command::Frequency {
                dir: Direction::Rx,
                ch: 0,
                hz: 1e8,
            },
        )
        .unwrap();
    }
    let err = m
        .schedule(
            CommandTime::UnixNs(1),
            Command::Gain {
                dir: Direction::Tx,
                ch: 0,
                db: 3.0,
            },
        )
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::BufferTooSmall);
    assert_eq!(m.scheduled.len(), 4);
}

#[test]
fn async_streams_work_with_a_trivial_executor() {
    let mut m = Mock::new();
    let mut buf = [0u8; 9];
    let (n, meta) = block_on(AsyncRxStream::read(&mut m, &mut buf)).unwrap();
    assert_eq!((n, meta), (8, BlockMeta::default()));
    assert_eq!(&buf[..8], &[7; 8]);
    assert_eq!(buf[8], 0);

    let written = block_on(AsyncTxStream::write(
        &mut m,
        &[1, 2, 3, 4],
        &BlockMeta::default(),
    ));
    assert_eq!(written.unwrap(), 4);
    assert_eq!(m.written, 4);
    assert_eq!(block_on(AsyncTxStream::poll_status(&mut m)).unwrap(), None);
    assert_eq!(AsyncRxStream::sample_rate(&m), Rate::hz(2_000_000).unwrap());
    assert_eq!(AsyncTxStream::format(&m), formats::CS8);
}

#[test]
fn async_futures_need_no_send_bound() {
    // A driver with an `Rc` inside is not `Send`; the traits must still be implementable.
    struct Local(std::rc::Rc<u8>);
    impl AsyncRxStream for Local {
        type Error = Error;
        fn format(&self) -> SampleFormat {
            formats::CU8
        }
        fn sample_rate(&self) -> Rate {
            Rate::hz(1).unwrap()
        }
        async fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Error> {
            let keep = std::rc::Rc::clone(&self.0);
            YieldOnce(false).await;
            buf.fill(*keep);
            Ok((buf.len(), BlockMeta::default()))
        }
    }
    let mut l = Local(std::rc::Rc::new(9));
    let mut b = [0u8; 2];
    assert_eq!(block_on(l.read(&mut b)).unwrap().0, 2);
    assert_eq!(b, [9, 9]);
}

#[test]
fn public_types_are_send() {
    assert_send_sync::<Range>();
    assert_send_sync::<Direction>();
    assert_send_sync::<Capabilities>();
    assert_send_sync::<Command>();
    assert_send_sync::<CommandTime>();
    assert_send::<Mock>();
}
