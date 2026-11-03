//! Tests of the two-clock simulation (M28).
use sdrk_testkit::TwoClockSim;

#[test]
fn ppm_shifts_the_fill_by_the_integrated_rate_difference() {
    // Same seed, same jitter: the fill differs only by rate * ppm * time.
    let mut plain = TwoClockSim::new(48_000.0, 100.0, 960_000, 480, 462);
    let mut tuned = TwoClockSim::new(48_000.0, 100.0, 960_000, 480, 462);
    tuned.set_ppm(-100.0);
    for _ in 0..1000 {
        plain.step();
        tuned.step();
    }
    let expected = 48_000.0 * 100e-6 * plain.now_s();
    assert!((plain.fill() - tuned.fill() - expected).abs() < 1e-6);
    assert_eq!(tuned.ppm(), -100.0);
}

#[test]
fn overruns_and_underruns_are_counted() {
    let mut sim = TwoClockSim::new(48_000.0, 5000.0, 9600, 480, 1);
    for _ in 0..2000 {
        sim.step();
    }
    assert!(sim.overruns() > 0);
    assert_eq!(sim.underruns(), 0);
    let mut sim = TwoClockSim::new(48_000.0, -5000.0, 9600, 480, 1);
    for _ in 0..2000 {
        sim.step();
    }
    assert!(sim.underruns() > 0);
    assert_eq!(sim.overruns(), 0);
}

#[test]
fn deterministic_and_two_hours_are_fast() {
    let run = |seed| {
        let mut sim = TwoClockSim::new(48_000.0, 0.0, 9600, 480, seed);
        let mut last = (0.0, 0.0);
        while sim.now_s() < 7200.0 {
            last = sim.step();
        }
        last
    };
    let a = run(7);
    assert_eq!(a, run(7));
    assert_ne!(a, run(8));
    assert!(a.0 >= 7200.0);
}
