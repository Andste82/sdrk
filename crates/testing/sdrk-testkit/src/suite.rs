//! The suite runner: executes registered test cases on any target.
//!
//! There is no `cargo test` and no heap on a microcontroller, so a case is a
//! plain function over a caller-provided [`Arena`]. Results go to a
//! [`Reporter`]; [`WriteReporter`] (and `StdReporter` with feature `std`)
//! print the line protocol that emulator runners parse:
//!
//! ```text
//! SDRK-SUITE START <n>
//! SDRK-CASE PASS <name>
//! SDRK-CASE FAIL <name> <msg> <detail>
//! SDRK-SUITE END passed=<p> failed=<f>
//! ```
//!
//! # Memory
//!
//! Every case works with an arena of 256 KiB, quick cases with 64 KiB. The
//! arena use of each registered case is documented in the `cases` module.

use core::fmt::{self, Write};

use crate::arena::Arena;

/// How much of the suite to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    /// Only quick cases (fit a 64 KiB arena, finish fast on a microcontroller).
    Quick,
    /// Every case.
    Full,
}

/// A failed check: which case, what failed and a short formatted detail.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Failure {
    /// Name of the case (filled in by [`run_suite`] when empty).
    pub case: &'static str,
    /// What went wrong, as a fixed message.
    pub msg: &'static str,
    /// Formatted details (values, indices); truncated to 160 bytes.
    pub detail: heapless::String<160>,
}

impl Failure {
    /// Creates a failure without detail.
    ///
    /// ```
    /// use sdrk_testkit::Failure;
    /// let f = Failure::new("case", "mismatch").detail(format_args!("index {}", 3));
    /// assert_eq!(f.detail.as_str(), "index 3");
    /// ```
    pub fn new(case: &'static str, msg: &'static str) -> Self {
        Self {
            case,
            msg,
            detail: heapless::String::new(),
        }
    }

    /// Sets the detail text; output beyond 160 bytes is cut off.
    #[must_use]
    pub fn detail(mut self, args: fmt::Arguments<'_>) -> Self {
        self.detail.clear();
        // A full buffer only truncates the detail; that is intended.
        let _ = Truncating(&mut self.detail).write_fmt(args);
        self
    }
}

/// Appends text to a fixed-capacity string and silently drops what does not
/// fit (whole characters only).
struct Truncating<'a>(&'a mut heapless::String<160>);

impl Write for Truncating<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for ch in s.chars() {
            if self.0.push(ch).is_err() {
                break;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.case, self.msg)?;
        if !self.detail.is_empty() {
            write!(f, " ({})", self.detail)?;
        }
        Ok(())
    }
}

impl core::error::Error for Failure {}

impl From<sdrk_core::Error> for Failure {
    fn from(e: sdrk_core::Error) -> Self {
        Failure::new("", "sdrk error").detail(format_args!("{e}"))
    }
}

impl From<sdrk_blocks::Error> for Failure {
    fn from(e: sdrk_blocks::Error) -> Self {
        Failure::new("", "block error").detail(format_args!("{e}"))
    }
}

impl From<sdrk_device::Error> for Failure {
    fn from(e: sdrk_device::Error) -> Self {
        Failure::new("", "device error").detail(format_args!("{e}"))
    }
}

/// A registered test case.
#[derive(Clone, Copy)]
pub struct TestCase {
    /// Unique name, used in the output protocol (no spaces).
    pub name: &'static str,
    /// Tier the case belongs to.
    pub tier: Tier,
    /// The test body; the arena is reset before each call.
    pub run: fn(&mut Arena<'_>) -> Result<(), Failure>,
}

impl fmt::Debug for TestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestCase")
            .field("name", &self.name)
            .field("tier", &self.tier)
            .finish_non_exhaustive()
    }
}

/// Counts of a finished run.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Summary {
    /// Cases that passed.
    pub passed: u32,
    /// Cases that failed.
    pub failed: u32,
    /// Cases not run because of the tier.
    pub skipped: u32,
}

/// Receives the progress of a suite run.
pub trait Reporter {
    /// Called once before the first case with the number of cases that will
    /// run. The default does nothing.
    fn suite_started(&mut self, _total: usize) {}
    /// A case is about to run.
    fn started(&mut self, name: &str);
    /// The case passed.
    fn passed(&mut self, name: &str);
    /// The case failed.
    fn failed(&mut self, f: &Failure);
    /// The run is over.
    fn finished(&mut self, s: &Summary);
}

/// Runs the cases of `tier`, resetting `arena` before each one.
///
/// ```
/// use sdrk_testkit::{all_cases, run_suite, Arena, Reporter, Failure, Summary, Tier};
/// struct Quiet;
/// impl Reporter for Quiet {
///     fn started(&mut self, _: &str) {}
///     fn passed(&mut self, _: &str) {}
///     fn failed(&mut self, _: &Failure) {}
///     fn finished(&mut self, _: &Summary) {}
/// }
/// let mut mem = [0u8; 64 * 1024];
/// let mut arena = Arena::new(&mut mem);
/// let s = run_suite(all_cases(), Tier::Quick, &mut arena, &mut Quiet);
/// assert_eq!(s.failed, 0);
/// ```
pub fn run_suite(
    cases: &[TestCase],
    tier: Tier,
    arena: &mut Arena<'_>,
    reporter: &mut impl Reporter,
) -> Summary {
    let selected = |c: &&TestCase| tier == Tier::Full || c.tier == Tier::Quick;
    let total = cases.iter().filter(selected).count();
    let mut summary = Summary::default();
    reporter.suite_started(total);
    for case in cases {
        if !selected(&case) {
            summary.skipped += 1;
            continue;
        }
        arena.reset();
        reporter.started(case.name);
        match (case.run)(arena) {
            Ok(()) => {
                summary.passed += 1;
                reporter.passed(case.name);
            }
            Err(mut f) => {
                if f.case.is_empty() {
                    f.case = case.name;
                }
                summary.failed += 1;
                reporter.failed(&f);
            }
        }
    }
    reporter.finished(&summary);
    summary
}

/// Every case registered by the testkit.
///
/// ```
/// assert!(sdrk_testkit::all_cases().iter().any(|c| c.name == "rng_known_answer"));
/// ```
pub fn all_cases() -> &'static [TestCase] {
    crate::cases::CASES.as_slice()
}

/// Writes `s` with line breaks replaced by spaces, so that every protocol
/// message stays on one line.
fn write_one_line(w: &mut impl Write, s: &str) -> fmt::Result {
    for ch in s.chars() {
        w.write_char(if ch == '\n' || ch == '\r' { ' ' } else { ch })?;
    }
    Ok(())
}

/// A [`Reporter`] that prints the protocol lines to any [`core::fmt::Write`],
/// for example a UART wrapper on a microcontroller.
///
/// ```
/// use sdrk_testkit::{Reporter, Summary, WriteReporter};
/// let mut out = String::new();
/// let mut r = WriteReporter::new(&mut out);
/// r.passed("x");
/// r.finished(&Summary { passed: 1, failed: 0, skipped: 0 });
/// assert_eq!(out, "SDRK-CASE PASS x\nSDRK-SUITE END passed=1 failed=0\n");
/// ```
#[derive(Debug)]
pub struct WriteReporter<W: Write> {
    out: W,
}

impl<W: Write> WriteReporter<W> {
    /// Wraps a writer.
    pub fn new(out: W) -> Self {
        Self { out }
    }

    /// Returns the writer.
    pub fn into_inner(self) -> W {
        self.out
    }
}

impl<W: Write> Reporter for WriteReporter<W> {
    fn suite_started(&mut self, total: usize) {
        // The writer is a best-effort sink; a failed write cannot be reported.
        let _ = writeln!(self.out, "SDRK-SUITE START {total}");
    }

    fn started(&mut self, _name: &str) {}

    fn passed(&mut self, name: &str) {
        let _ = writeln!(self.out, "SDRK-CASE PASS {name}");
    }

    fn failed(&mut self, f: &Failure) {
        let _ = write!(self.out, "SDRK-CASE FAIL {} ", f.case);
        let _ = write_one_line(&mut self.out, f.msg);
        let _ = self.out.write_char(' ');
        let _ = write_one_line(&mut self.out, &f.detail);
        let _ = self.out.write_char('\n');
    }

    fn finished(&mut self, s: &Summary) {
        let _ = writeln!(
            self.out,
            "SDRK-SUITE END passed={} failed={}",
            s.passed, s.failed
        );
    }
}

/// Prints the protocol to standard output (feature `std`).
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct StdReporter {
    inner: WriteReporter<StdoutWriter>,
}

#[cfg(feature = "std")]
#[derive(Debug, Default)]
struct StdoutWriter;

#[cfg(feature = "std")]
impl Write for StdoutWriter {
    #[allow(clippy::print_stdout)]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        std::print!("{s}");
        Ok(())
    }
}

#[cfg(feature = "std")]
impl Default for WriteReporter<StdoutWriter> {
    fn default() -> Self {
        Self::new(StdoutWriter)
    }
}

#[cfg(feature = "std")]
impl StdReporter {
    /// Creates a reporter that prints to stdout.
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "std")]
impl Reporter for StdReporter {
    fn suite_started(&mut self, total: usize) {
        self.inner.suite_started(total);
    }
    fn started(&mut self, name: &str) {
        self.inner.started(name);
    }
    fn passed(&mut self, name: &str) {
        self.inner.passed(name);
    }
    fn failed(&mut self, f: &Failure) {
        self.inner.failed(f);
    }
    fn finished(&mut self, s: &Summary) {
        self.inner.finished(s);
    }
}
