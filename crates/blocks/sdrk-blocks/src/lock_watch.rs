//! `LockWatch`: the lock state machine of a receiver built from building blocks (M44.1 §2.3,
//! §5, §6): the states `Search`, `PullIn`, `Track` and `Verify`, the named transitions between
//! them and the four actions a composite carries out.

use core::marker::PhantomData;

/// The thresholds and time limits of a [`LockWatch`], fixed by the composite that holds it
/// (M44.1 §2.3; associated constants, so that the watch holds none of them).
pub trait LockPolicy {
    /// Symbols between two checks of the carrier loop (1 to 65535).
    const CHECK: u16;
    /// The largest distance in cycles per symbol of a clear verify estimate from the frequency of
    /// the loops, modulo a quarter cycle per symbol, that keeps the estimate.
    const TOLERANCE: f64;
    /// The weight watch (M44.1 §6, condition 1): the low checks in a row in `Track` (a weight
    /// energy of the equalizer below the floor of the check, [`LockCheck::weight_floor`]) after
    /// which the receiver starts again (`Shrink`); 0 turns the weight watch off, at most 31.
    const WEIGHT_CHECKS: u8;
    /// The signal back after a gap (M44.1 §6, condition 2): a clear verify window after at least
    /// this many acquisition windows in a row without a clear peak starts the equalizer again
    /// with the new estimate (`Return`); 0 turns the condition off.
    const GAP_WINDOWS: u8;
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

/// What a composite does on a transition of a [`LockWatch`] (M44.1 §2.3): every
/// [`LockTransition`] maps to one of them ([`LockTransition::action`]); a transition that carries
/// an estimate also hands it to the loops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LockAction {
    /// Nothing starts again (an estimate that comes with the transition goes to the loops).
    Keep,
    /// Take one verify window of the acquisition.
    Verify,
    /// Start the equalizer again (its start weights and blind symbols), then hand the estimate
    /// of the transition to the loops.
    RestartEqualizer,
    /// Start the receiver again: the equalizer, the loops at frequency 0, and a new search of the
    /// acquisition.
    RestartAll,
}

/// A transition of a [`LockWatch`]: every rule of the controller is one of them (M44.1 §5, §6;
/// the table in `docs/blocks.md`, section `qam_demod`).
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
    /// `Track` -> `Search`: the weight watch (M44.1 §6, condition 1): the weight energy of the
    /// equalizer was below its floor ([`LockCheck::weight_floor`]) at
    /// [`LockPolicy::WEIGHT_CHECKS`] checks in a row (the weights of a false state shrink, I44-17); the receiver starts again as on
    /// `Restart`.
    Shrink,
    /// `Verify` -> `PullIn`: the signal back after a gap (M44.1 §6, condition 2): a clear verify
    /// window within the tolerance after at least [`LockPolicy::GAP_WINDOWS`] windows in a row
    /// without a clear peak; the equalizer, which adapted to the gap, starts again and the new
    /// estimate goes to the loops.
    Return {
        /// The estimate of the verify window in cycles per symbol.
        frequency: f64,
    },
}

impl LockTransition {
    /// The action of the transition (M44.1 §2.3): `AcquireAfterMiss` and `Return` start the
    /// equalizer again, `Restart` and `Shrink` the whole receiver, `SettleUnlocked` and `Doubt`
    /// take a verify window, every other transition keeps the receiver running.
    ///
    /// ```
    /// use sdrk_blocks::{LockAction, LockTransition};
    /// assert_eq!(LockTransition::Doubt.action(), LockAction::Verify);
    /// assert_eq!(LockTransition::Shrink.action(), LockAction::RestartAll);
    /// assert_eq!(LockTransition::Return { frequency: 0.001 }.action(), LockAction::RestartEqualizer);
    /// assert_eq!(LockTransition::Hold.action(), LockAction::Keep);
    /// ```
    pub fn action(&self) -> LockAction {
        match self {
            Self::AcquireAfterMiss { .. } | Self::Return { .. } => LockAction::RestartEqualizer,
            Self::Restart | Self::Shrink => LockAction::RestartAll,
            Self::SettleUnlocked | Self::Doubt => LockAction::Verify,
            Self::Miss | Self::Acquire { .. } | Self::Settle | Self::Hold | Self::Confirm => {
                LockAction::Keep
            }
        }
    }

    /// The estimate that goes to the loops with the transition (`Acquire`, `AcquireAfterMiss`,
    /// `Return`), in cycles per symbol.
    ///
    /// ```
    /// use sdrk_blocks::LockTransition;
    /// assert_eq!(LockTransition::Acquire { frequency: 0.002 }.frequency(), Some(0.002));
    /// assert_eq!(LockTransition::Restart.frequency(), None);
    /// ```
    pub fn frequency(&self) -> Option<f64> {
        match *self {
            Self::Acquire { frequency }
            | Self::AcquireAfterMiss { frequency }
            | Self::Return { frequency } => Some(frequency),
            _ => None,
        }
    }
}

/// The inputs of a check of a [`LockWatch`] (M44.1 §2.3): the lock of the carrier loop, and the
/// weight energy of the equalizer (`None` without one) with the floor below which it counts as
/// low (fixed by the composite per constellation: the energy of correct weights falls with the
/// signal-to-noise ratio, and each constellation has its own range).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LockCheck {
    /// Whether the carrier loop reports lock.
    pub locked: bool,
    /// The weight energy `sum |w|^2` of the equalizer; `None` without an equalizer.
    pub weight_energy: Option<f32>,
    /// The weight energy below which (or NaN) the check counts as low for the weight watch.
    pub weight_floor: f32,
}

/// The largest [`LockPolicy::WEIGHT_CHECKS`].
const MAX_WEIGHT_CHECKS: u8 = 31;

/// The lock state machine of a receiver of building blocks (M44.1 §2.3): a heap-free supervisor
/// (not a block) that a composite feeds with the results of its acquisition windows and, every
/// `CHECK` symbols, with the lock of its carrier loop and the weight energy of its equalizer and
/// its floor ([`LockCheck`]), and that answers with the named [`LockTransition`] to carry out (its
/// [`LockAction`] is one of `Keep`, `Verify`, `RestartEqualizer`, `RestartAll`). It counts the
/// symbols to the next check itself; the composite ends its runs where
/// [`remaining`](LockWatch::remaining) says. The transitions of §5 are the rules of `qam_demod`
/// before M44.1, one transition per rule; the recovery conditions of §6 add `Shrink` (the weight
/// watch) and `Return` (the signal back after a gap), with the thresholds of the [`LockPolicy`].
///
/// ```
/// use sdrk_blocks::{LockCheck, LockPolicy, LockState, LockTransition, LockWatch};
/// struct Policy;
/// impl LockPolicy for Policy {
///     const CHECK: u16 = 4096;
///     const TOLERANCE: f64 = 2e-4;
///     const WEIGHT_CHECKS: u8 = 2;
///     const GAP_WINDOWS: u8 = 3;
/// }
/// let check = |locked, energy| move || LockCheck {
///     locked,
///     weight_energy: Some(energy),
///     weight_floor: 0.7,
/// };
/// let mut w = LockWatch::<Policy>::new();
/// assert_eq!(w.window(None, 0.0), LockTransition::Miss);
/// assert_eq!(w.window(Some(0.001), 0.0), LockTransition::AcquireAfterMiss { frequency: 0.001 });
/// assert_eq!((w.state(), w.remaining()), (LockState::PullIn, 4096));
/// assert_eq!(w.advance(4096, check(true, 0.9)), Some(LockTransition::Settle));
/// assert_eq!(w.advance(4096, check(false, 0.95)), Some(LockTransition::Doubt));
/// assert_eq!(w.state(), LockState::Verify);
/// // a clear carrier 0.0005 cycles per symbol away from the loops
/// assert_eq!(w.window(Some(0.0015), 0.001), LockTransition::Restart);
/// assert_eq!((w.state(), w.restarts(), w.acquired()), (LockState::Search, 1, Some(0.001)));
/// // the weight watch: two low checks in a row in `Track`
/// assert_eq!(w.window(Some(0.001), 0.0), LockTransition::Acquire { frequency: 0.001 });
/// assert_eq!(w.advance(4096, check(true, 0.9)), Some(LockTransition::Settle));
/// assert_eq!(w.advance(4096, check(true, 0.5)), Some(LockTransition::Hold));
/// assert_eq!(w.low_checks(), 1);
/// assert_eq!(w.advance(4096, check(true, 0.5)), Some(LockTransition::Shrink));
/// assert_eq!((w.state(), w.restarts()), (LockState::Search, 2));
/// ```
pub struct LockWatch<P: LockPolicy> {
    /// The last estimate that went to the loops, NaN before the first.
    acquired: f64,
    /// Restarts since the start or `reset` (saturating).
    restarts: u32,
    /// Symbols since the estimate or the last check (in `PullIn`, `Track` and `Verify`).
    since: u16,
    state: LockState,
    /// The checks in a row in `Track` whose weight energy was low (the weight watch).
    low_checks: u8,
    /// Whether the conditions of step 2 are on.
    recovery: bool,
    /// Acquisition windows in a row without a clear peak (saturating).
    misses: u8,
    _policy: PhantomData<fn() -> P>,
}

impl<P: LockPolicy> core::fmt::Debug for LockWatch<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LockWatch")
            .field("state", &self.state())
            .field("since", &self.since)
            .field("acquired", &self.acquired())
            .field("restarts", &self.restarts)
            .field("misses", &self.misses)
            .field("low_checks", &self.low_checks())
            .field("recovery", &self.recovery())
            .finish_non_exhaustive()
    }
}

impl<P: LockPolicy> Default for LockWatch<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: LockPolicy> LockWatch<P> {
    /// A watch in `Search` with the conditions of step 2 on (a `CHECK` of 0 counts as 1; a
    /// `WEIGHT_CHECKS` above 31 counts as 31).
    pub fn new() -> Self {
        Self {
            acquired: f64::NAN,
            restarts: 0,
            since: 0,
            state: LockState::Search,
            low_checks: 0,
            recovery: true,
            misses: 0,
            _policy: PhantomData,
        }
    }

    /// Symbols between two checks.
    const fn check() -> u16 {
        if P::CHECK == 0 { 1 } else { P::CHECK }
    }

    /// The low checks in a row that start the receiver again (0 = off).
    const fn weight_checks() -> u8 {
        if P::WEIGHT_CHECKS > MAX_WEIGHT_CHECKS {
            MAX_WEIGHT_CHECKS
        } else {
            P::WEIGHT_CHECKS
        }
    }

    /// The state.
    pub fn state(&self) -> LockState {
        self.state
    }

    /// Sets the state; the low checks are kept in `Track` and `Verify` and cleared otherwise.
    fn set_state(&mut self, state: LockState) {
        if !matches!(state, LockState::Track | LockState::Verify) {
            self.low_checks = 0;
        }
        self.state = state;
    }

    /// The checks in a row in `Track` whose weight energy was low (the weight watch, M44.1 §6).
    pub fn low_checks(&self) -> u8 {
        self.low_checks
    }

    /// Whether the conditions of step 2 (M44.1 §6: `Shrink` and `Return`) are on (the default).
    pub fn recovery(&self) -> bool {
        self.recovery
    }

    /// Turns the conditions of step 2 (M44.1 §6) on or off; off, the watch takes the transitions
    /// of step 1 only (§5), which compares a receiver with the one before the recovery. The
    /// setting is kept over [`reset`](LockWatch::reset); the low checks are cleared.
    ///
    /// ```
    /// use sdrk_blocks::{LockCheck, LockPolicy, LockTransition, LockWatch};
    /// struct Policy;
    /// impl LockPolicy for Policy {
    ///     const CHECK: u16 = 4096;
    ///     const TOLERANCE: f64 = 2e-4;
    ///     const WEIGHT_CHECKS: u8 = 1;
    ///     const GAP_WINDOWS: u8 = 1;
    /// }
    /// let low = || LockCheck { locked: true, weight_energy: Some(0.1), weight_floor: 0.75 };
    /// let mut w = LockWatch::<Policy>::new();
    /// w.set_recovery(false);
    /// w.window(Some(0.001), 0.0);
    /// assert_eq!(w.advance(4096, low), Some(LockTransition::Settle));
    /// assert_eq!(w.advance(4096, low), Some(LockTransition::Hold));
    /// w.set_recovery(true);
    /// assert_eq!(w.advance(4096, low), Some(LockTransition::Shrink));
    /// ```
    pub fn set_recovery(&mut self, on: bool) {
        self.low_checks = 0;
        self.recovery = on;
    }

    /// Acquisition windows in a row without a clear peak (saturating at 255): in `Search` the
    /// windows of the search, in `Verify` the verify windows (the gap condition, M44.1 §6).
    pub fn misses(&self) -> u8 {
        self.misses
    }

    /// The last estimate that went to the loops (kept over a restart); `None` before the first.
    pub fn acquired(&self) -> Option<f64> {
        (!self.acquired.is_nan()).then_some(self.acquired)
    }

    /// Restarts since the start or `reset`: `Restart` and `Shrink` (the whole receiver) and
    /// `Return` (the equalizer after a gap).
    pub fn restarts(&self) -> u32 {
        self.restarts
    }

    /// Symbols until the next check (`usize::MAX` in `Search`, which has no check).
    pub fn remaining(&self) -> usize {
        match self.state() {
            LockState::Search => usize::MAX,
            _ => usize::from(Self::check().saturating_sub(self.since)),
        }
    }

    /// Back to `Search` as after `new` (the restarts at 0, no estimate); the switch of
    /// [`set_recovery`](LockWatch::set_recovery) is kept.
    pub fn reset(&mut self) {
        let on = self.recovery();
        *self = Self::new();
        self.set_recovery(on);
    }

    /// To `Search` after a restart of the whole receiver.
    fn enter_search(&mut self) {
        self.restarts = self.restarts.saturating_add(1);
        self.misses = 0;
        self.set_state(LockState::Search);
    }

    /// To `PullIn` with the estimate `f`.
    fn enter_pull_in(&mut self, f: f64) {
        self.misses = 0;
        self.acquired = f;
        self.since = 0;
        self.set_state(LockState::PullIn);
    }

    /// The end of an acquisition window with its estimate (`None` when it had no clear peak);
    /// `loops` is the frequency of the loops in cycles per symbol (read only in `Verify`). In
    /// `PullIn` and `Track` no window runs; the call keeps the state (`Confirm`).
    pub fn window(&mut self, estimate: Option<f64>, loops: f64) -> LockTransition {
        let state = self.state();
        if matches!(state, LockState::Search | LockState::Verify) && estimate.is_none() {
            self.misses = self.misses.saturating_add(1);
        }
        match (state, estimate) {
            (LockState::Search, None) => LockTransition::Miss,
            (LockState::Search, Some(f)) => {
                let fresh = self.misses == 0;
                self.enter_pull_in(f);
                if fresh {
                    LockTransition::Acquire { frequency: f }
                } else {
                    LockTransition::AcquireAfterMiss { frequency: f }
                }
            }
            (LockState::Verify, Some(f)) if quarter_cycle_distance(f, loops) > P::TOLERANCE => {
                self.enter_search();
                LockTransition::Restart
            }
            (LockState::Verify, Some(f))
                if self.recovery()
                    && P::GAP_WINDOWS != 0
                    && self.misses >= P::GAP_WINDOWS
                    && f.is_finite() =>
            {
                self.restarts = self.restarts.saturating_add(1);
                self.enter_pull_in(f);
                LockTransition::Return { frequency: f }
            }
            (LockState::Verify, estimate) => {
                if estimate.is_some() {
                    self.misses = 0;
                }
                self.set_state(LockState::Track);
                LockTransition::Confirm
            }
            _ => LockTransition::Confirm,
        }
    }

    /// Counts `n` symbols (at most [`remaining`](LockWatch::remaining)); at a check, with the
    /// inputs that `probe` reads then (the lock of the carrier loop and the weight energy of the
    /// equalizer; not called between checks), the transition of the check.
    pub fn advance(
        &mut self,
        n: usize,
        probe: impl FnOnce() -> LockCheck,
    ) -> Option<LockTransition> {
        let state = self.state();
        if state == LockState::Search {
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
        let input = probe();
        if state == LockState::PullIn {
            return Some(if input.locked {
                self.set_state(LockState::Track);
                LockTransition::Settle
            } else {
                self.set_state(LockState::Verify);
                LockTransition::SettleUnlocked
            });
        }
        // The weight watch counts the checks in `Track` (a NaN energy is low).
        let low = match input.weight_energy {
            Some(e) if self.recovery() && (e.is_nan() || e < input.weight_floor) => {
                self.low_checks().saturating_add(1)
            }
            _ => 0,
        };
        self.low_checks = low.min(MAX_WEIGHT_CHECKS);
        let limit = Self::weight_checks();
        if limit != 0 && low >= limit {
            self.enter_search();
            return Some(LockTransition::Shrink);
        }
        Some(if input.locked {
            LockTransition::Hold
        } else {
            self.set_state(LockState::Verify);
            LockTransition::Doubt
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
