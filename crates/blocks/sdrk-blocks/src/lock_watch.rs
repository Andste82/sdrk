//! `LockWatch`: the lock state machine of a receiver built from building blocks (M44.1 §2.3,
//! §5): the states `Search`, `PullIn`, `Track` and `Verify`, and the named transitions between
//! them.

use core::marker::PhantomData;

/// The thresholds and time limits of a [`LockWatch`], fixed by the composite that holds it
/// (M44.1 §2.3; associated constants, so that the watch holds none of them).
pub trait LockPolicy {
    /// Symbols between two checks of the carrier loop (1 to 65535).
    const CHECK: u16;
    /// The largest distance in cycles per symbol of a clear verify estimate from the frequency of
    /// the loops, modulo a quarter cycle per symbol, that keeps the estimate.
    const TOLERANCE: f64;
}

/// The state of a [`LockWatch`] (M44.1 §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LockState {
    /// The acquisition searches (windows until one has a clear peak); the equalizer is blind and
    /// the loops run at frequency 0.
    Search,
    /// An estimate went to the loops; the equalizer is blind until its switch to the decision
    /// mode (with the hand-over of D1700 there), until the first check after the estimate.
    PullIn,
    /// Everything runs; a check every `check` symbols.
    Track,
    /// One verify window of the acquisition after a check without lock (D1619).
    Verify,
}

/// A transition of a [`LockWatch`]: every rule of the controller is one of them (M44.1 §5; the
/// table in `docs/blocks.md`, section `qam_demod`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum LockTransition {
    /// `Search` -> `Search`: a window without a clear peak; the next window follows at once
    /// (D1403).
    Miss,
    /// `Search` -> `PullIn`: a clear window, the first of the search; the estimate goes to the
    /// loops (D1403, D1404).
    Acquire {
        /// The estimate in cycles per symbol.
        frequency: f64,
    },
    /// `Search` -> `PullIn`: a clear window after windows without a clear peak; the equalizer,
    /// which adapted to the noise of those windows, starts again before the estimate goes to the
    /// loops (D1619).
    AcquireAfterMiss {
        /// The estimate in cycles per symbol.
        frequency: f64,
    },
    /// `PullIn` -> `Track`: the first check after the estimate, with lock; the pull-in of the
    /// equalizer's tracker ends (D1622).
    Settle,
    /// `PullIn` -> `Verify`: the first check after the estimate, without lock; the pull-in ends
    /// and the verify window follows (D1608, D1622).
    SettleUnlocked,
    /// `Track` -> `Track`: a check with lock keeps the estimate without a verify window (D1619).
    Hold,
    /// `Track` -> `Verify`: a check without lock (D1619).
    Doubt,
    /// `Verify` -> `Track`: the verify window shows no clear carrier away from the loops (no
    /// clear peak, or within the tolerance modulo a quarter cycle per symbol); the estimate is
    /// kept (D1605, D1608, D1618).
    Confirm,
    /// `Verify` -> `Search`: the verify window shows a clear carrier more than the tolerance away
    /// from the loops; the receiver starts again (the equalizer, the loops at frequency 0) and
    /// the acquisition searches (D1421, D1605, D1618, D1623).
    Restart,
}

/// The lock state machine of a receiver of building blocks (M44.1 §2.3): a heap-free supervisor
/// (not a block) that a composite feeds with the results of its acquisition windows and, every
/// `check` symbols, with the lock of its carrier loop, and that answers with the named
/// [`LockTransition`] to carry out. It counts the symbols to the next check itself; the composite
/// ends its runs where [`remaining`](LockWatch::remaining) says. Step 1 of M44.1 holds the states
/// and transitions of `qam_demod` before M44.1, one transition per rule; the recovery conditions
/// of §6 follow in step 2.
///
/// ```
/// use sdrk_blocks::{LockPolicy, LockState, LockTransition, LockWatch};
/// struct Policy;
/// impl LockPolicy for Policy {
///     const CHECK: u16 = 4096;
///     const TOLERANCE: f64 = 2e-4;
/// }
/// let mut w = LockWatch::<Policy>::new();
/// assert_eq!(w.window(None, 0.0), LockTransition::Miss);
/// assert_eq!(w.window(Some(0.001), 0.0), LockTransition::AcquireAfterMiss { frequency: 0.001 });
/// assert_eq!((w.state(), w.remaining()), (LockState::PullIn, 4096));
/// assert_eq!(w.advance(4096, true), Some(LockTransition::Settle));
/// assert_eq!(w.advance(4096, false), Some(LockTransition::Doubt));
/// assert_eq!(w.state(), LockState::Verify);
/// // a clear carrier 0.0005 cycles per symbol away from the loops
/// assert_eq!(w.window(Some(0.0015), 0.001), LockTransition::Restart);
/// assert_eq!((w.state(), w.restarts(), w.acquired()), (LockState::Search, 1, Some(0.001)));
/// ```
pub struct LockWatch<P: LockPolicy> {
    /// The last estimate that went to the loops, NaN before the first.
    acquired: f64,
    /// Restarts since the start or `reset` (saturating).
    restarts: u32,
    /// Symbols since the estimate or the last check (in `PullIn`, `Track` and `Verify`).
    since: u16,
    state: LockState,
    /// Whether a window of this search had no clear peak.
    missed: bool,
    _policy: PhantomData<fn() -> P>,
}

impl<P: LockPolicy> core::fmt::Debug for LockWatch<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LockWatch")
            .field("state", &self.state)
            .field("since", &self.since)
            .field("acquired", &self.acquired())
            .field("restarts", &self.restarts)
            .field("missed", &self.missed)
            .finish()
    }
}

impl<P: LockPolicy> Default for LockWatch<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: LockPolicy> LockWatch<P> {
    /// A watch in `Search` (a `CHECK` of 0 counts as 1).
    pub fn new() -> Self {
        Self {
            acquired: f64::NAN,
            restarts: 0,
            since: 0,
            state: LockState::Search,
            missed: false,
            _policy: PhantomData,
        }
    }

    /// Symbols between two checks.
    const fn check() -> u16 {
        if P::CHECK == 0 { 1 } else { P::CHECK }
    }

    /// The state.
    pub fn state(&self) -> LockState {
        self.state
    }

    /// The last estimate that went to the loops (kept over a restart); `None` before the first.
    pub fn acquired(&self) -> Option<f64> {
        (!self.acquired.is_nan()).then_some(self.acquired)
    }

    /// Restarts since the start or `reset`.
    pub fn restarts(&self) -> u32 {
        self.restarts
    }

    /// Symbols until the next check (`usize::MAX` in `Search`, which has no check).
    pub fn remaining(&self) -> usize {
        match self.state {
            LockState::Search => usize::MAX,
            _ => usize::from(Self::check().saturating_sub(self.since)),
        }
    }

    /// Back to `Search` as after `new` (the restarts at 0, no estimate).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// The end of an acquisition window with its estimate (`None` when it had no clear peak);
    /// `loops` is the frequency of the loops in cycles per symbol (read only in `Verify`). In
    /// `PullIn` and `Track` no window runs; the call keeps the state (`Confirm`).
    pub fn window(&mut self, estimate: Option<f64>, loops: f64) -> LockTransition {
        match (self.state, estimate) {
            (LockState::Search, None) => {
                self.missed = true;
                LockTransition::Miss
            }
            (LockState::Search, Some(f)) => {
                let fresh = !self.missed;
                self.missed = false;
                self.acquired = f;
                self.state = LockState::PullIn;
                self.since = 0;
                if fresh {
                    LockTransition::Acquire { frequency: f }
                } else {
                    LockTransition::AcquireAfterMiss { frequency: f }
                }
            }
            (LockState::Verify, Some(f)) if quarter_cycle_distance(f, loops) > P::TOLERANCE => {
                self.restarts = self.restarts.saturating_add(1);
                self.state = LockState::Search;
                self.missed = false;
                LockTransition::Restart
            }
            (LockState::Verify, _) => {
                self.state = LockState::Track;
                LockTransition::Confirm
            }
            _ => LockTransition::Confirm,
        }
    }

    /// Counts `n` symbols (at most [`remaining`](LockWatch::remaining)); at a check, with the
    /// lock `locked` of the carrier loop, the transition of the check.
    pub fn advance(&mut self, n: usize, locked: bool) -> Option<LockTransition> {
        if self.state == LockState::Search {
            return None;
        }
        // `n <= remaining() <= check`.
        self.since = self
            .since
            .saturating_add(u16::try_from(n).unwrap_or(u16::MAX));
        if self.since < Self::check() {
            return None;
        }
        self.since = 0;
        Some(match (self.state, locked) {
            (LockState::PullIn, true) => {
                self.state = LockState::Track;
                LockTransition::Settle
            }
            (LockState::PullIn, false) => {
                self.state = LockState::Verify;
                LockTransition::SettleUnlocked
            }
            (_, true) => LockTransition::Hold,
            (_, false) => {
                self.state = LockState::Verify;
                LockTransition::Doubt
            }
        })
    }
}

/// The distance in cycles per symbol of two carrier frequencies modulo the quarter cycle per
/// symbol that the fourth power of square QAM, and so the estimate, cannot resolve (D1608): an
/// estimate wrapped into `[-0.125, 0.125)` and loops just across the edge are 0.25 apart, the
/// same carrier for the window. In `[0, 0.125]`.
pub(crate) fn quarter_cycle_distance(a: f64, b: f64) -> f64 {
    let d = a - b;
    libm::fabs(d - 0.25 * libm::round(4.0 * d))
}

#[cfg(test)]
mod tests {
    use super::quarter_cycle_distance;

    /// The verify step compares the frequencies modulo a quarter cycle per symbol (D1608): the
    /// two sides of the edge of the estimate's range are one carrier.
    #[test]
    fn the_verify_distance_wraps_at_a_quarter_cycle() {
        let close = |a: f64, b: f64, want: f64| {
            let d = quarter_cycle_distance(a, b);
            assert!((d - want).abs() < 1e-12, "{a} {b}: {d} against {want}");
        };
        close(0.001, 0.001, 0.0);
        close(0.0012, 0.001, 2e-4);
        close(-0.12495, 0.12495, 1e-4);
        close(0.12495, -0.12495, 1e-4);
        close(-0.125, 0.125, 0.0);
        // loops that drifted past the edge of the range
        close(-0.1249, 0.1253, 2e-4);
        close(0.06, 0.001, 0.059);
        close(0.1, -0.1, 0.05);
        close(0.125, 0.0, 0.125);
        assert!(quarter_cycle_distance(0.0, f64::NAN).is_nan());
    }
}
