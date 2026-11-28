//! Branch mispredictions per sample (D1611 in `docs/decisions-log.md`).
//!
//! `llvm-mca` simulates a loop with a perfect branch predictor. For a data-dependent branch on
//! the hot path the cost of a form is `cycles + mispredictions × penalty`: the mispredictions
//! per sample are counted with Valgrind's `cachegrind --branch-sim=yes` on the benchmark
//! workload of the entry (the differential method of [`crate::bench`]: a run with the
//! measured calls minus a run without them), on the x86-64 build of each level, and the penalty
//! is fixed per core model ([`penalty`], `docs/performance.md`, "Branches (D1611)").
//!
//! Cachegrind's predictor is a simple model (a table of 2-bit counters indexed by the address
//! and a global history for the conditional branches, a table of the last target for the
//! indirect ones), not the predictor of any of the core models: on a branch whose direction
//! depends on random data every predictor fails about as often, which is what the count is for.

use anyhow::{Context, anyhow, bail};

/// The branch counts of one cachegrind run (or the difference of two).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Branches {
    /// Executed conditional branches (`Bc`).
    pub conditional: u64,
    /// Mispredicted conditional branches (`Bcm`).
    pub conditional_mispredicted: u64,
    /// Executed indirect branches (`Bi`).
    pub indirect: u64,
    /// Mispredicted indirect branches (`Bim`).
    pub indirect_mispredicted: u64,
}

impl Branches {
    /// The mispredictions, conditional and indirect (`Bcm + Bim`).
    #[must_use]
    pub fn mispredicted(&self) -> u64 {
        self.conditional_mispredicted + self.indirect_mispredicted
    }

    /// `self - empty`, field by field: the branches of the counted calls. The mispredictions
    /// saturate at zero (D2744): the predictor's tables are indexed by address and history, so
    /// the run without the counted calls can mispredict a few more of the shared code than the
    /// run with them (`fft_64` against its restore-only run in CI); the counted calls then add
    /// no measurable mispredictions. The executed branches are exact and must not decrease.
    ///
    /// # Errors
    /// An executed-branch field of `empty` is larger than the one of `self`.
    pub fn minus(&self, empty: &Branches) -> anyhow::Result<Branches> {
        let sub = |a: u64, b: u64, what: &str| {
            a.checked_sub(b)
                .ok_or_else(|| anyhow!("the run with the counted calls has fewer {what}"))
        };
        Ok(Branches {
            conditional: sub(self.conditional, empty.conditional, "conditional branches")?,
            conditional_mispredicted: self
                .conditional_mispredicted
                .saturating_sub(empty.conditional_mispredicted),
            indirect: sub(self.indirect, empty.indirect, "indirect branches")?,
            indirect_mispredicted: self
                .indirect_mispredicted
                .saturating_sub(empty.indirect_mispredicted),
        })
    }
}

/// Parses the `events:` and `summary:` lines of a cachegrind output file written with
/// `--cache-sim=no --branch-sim=yes`.
///
/// # Errors
/// A line is missing, or an event `Bc`, `Bcm`, `Bi` or `Bim` is missing or not a number.
pub fn parse_cachegrind(text: &str) -> anyhow::Result<Branches> {
    let events: Vec<&str> = text
        .lines()
        .find_map(|l| l.strip_prefix("events:"))
        .ok_or_else(|| anyhow!("no `events:` line in the cachegrind output"))?
        .split_whitespace()
        .collect();
    let summary: Vec<&str> = text
        .lines()
        .find_map(|l| l.strip_prefix("summary:"))
        .ok_or_else(|| anyhow!("no `summary:` line in the cachegrind output"))?
        .split_whitespace()
        .collect();
    if events.len() != summary.len() {
        bail!(
            "cachegrind: {} events but {} summary values",
            events.len(),
            summary.len()
        );
    }
    let get = |name: &str| -> anyhow::Result<u64> {
        let i = events
            .iter()
            .position(|e| *e == name)
            .ok_or_else(|| anyhow!("no event {name} (was --branch-sim=yes given?)"))?;
        summary
            .get(i)
            .ok_or_else(|| anyhow!("no summary value of {name}"))?
            .parse()
            .with_context(|| format!("the summary of {name}"))
    };
    Ok(Branches {
        conditional: get("Bc")?,
        conditional_mispredicted: get("Bcm")?,
        indirect: get("Bi")?,
        indirect_mispredicted: get("Bim")?,
    })
}

/// The misprediction penalty of a core model in cycles (D1612), the value the vendor's
/// optimisation guide gives as typical; `None` for a CPU without one.
///
/// - `skylake`, `alderlake`, `sapphirerapids`: 20, Intel 64 and IA-32 Architectures
///   Optimization Reference Manual (248966-050), section 22.5.6.1 "Branch Mispredicts" ("about
///   20 cycles", the factor of its `%BR.MISP.COST`); the manual gives no figure per core.
/// - `znver2`: 16, Software Optimization Guide for AMD Family 17h Models 30h and Greater
///   Processors (56305 Rev. 3.00), section 2.8 "Optimizing Branching" (12 to 18, "the common
///   case penalty is 16 cycles").
/// - `znver3`: 13, Software Optimization Guide for AMD Family 19h Processors (56665 Rev. 3.00),
///   section 2.8 (11 to 18, common case 13).
/// - `znver4`: 13, Software Optimization Guide for the AMD Zen4 Microarchitecture (57647
///   Rev. 1.00), section 2.8 (11 to 18, common case 13).
/// - `cortex-a72`: 20, measured on a Raspberry Pi 4 (D1628): 19.56 cycles per misprediction
///   from the PMU (`cpu_cycles`, `br_mis_pred`) on a data-dependent `tbz` with paths of equal
///   length, and a slope of 19.61 over a random share of 0 to 100 %. Arm's Cortex-A72 software
///   optimization guide (UAN 0016A) gives no figure.
/// - `cortex-a76`: 11, provisional: the Cortex-A76 software optimization guide could not be
///   obtained; the value is Arm's own microarchitecture disclosure, to be confirmed by a
///   measurement on a Raspberry Pi 5 (I44-15, parked in `docs/milestones/BACKLOG.md`).
#[must_use]
pub fn penalty(cpu: &str) -> Option<f64> {
    match cpu {
        // The same value from different sources: Intel's manual and the measurement of D1628.
        "skylake" | "alderlake" | "sapphirerapids" | "cortex-a72" => Some(20.0),
        "znver2" => Some(16.0),
        "znver3" | "znver4" => Some(13.0),
        "cortex-a76" => Some(11.0),
        _ => None,
    }
}

/// The cost of a form on a core model (D1611): `cycles + mispredictions × penalty`, all per
/// sample.
#[must_use]
pub fn cost(cycles: f64, mispredictions: f64, penalty: f64) -> f64 {
    mispredictions.mul_add(penalty, cycles)
}

/// Whether the x86-64 build of `level` runs under Valgrind on this host: Valgrind 3.26 has no
/// AVX-512, so `avx512` cannot be branch-simulated (open item I44-16); `neon` takes no branch
/// simulation (D1611).
#[must_use]
pub fn simulated_level(level: &str) -> bool {
    matches!(level, "sse2" | "sse4.2" | "avx2")
}

#[cfg(test)]
mod tests {
    use super::*;

    const OUT: &str = "# callgrind format
version: 1
creator: cachegrind 3.26.0
pid: 1
cmd: ./x run a b 16
events: Ir Bc Bcm Bi Bim
fl=???
fn=main
0 10 2 1 0 0
summary: 123456 7890 456 12 3
";

    #[test]
    fn the_summary_gives_the_branch_counts() {
        let b = parse_cachegrind(OUT).unwrap();
        assert_eq!(
            b,
            Branches {
                conditional: 7890,
                conditional_mispredicted: 456,
                indirect: 12,
                indirect_mispredicted: 3,
            }
        );
        assert_eq!(b.mispredicted(), 459);
        let empty = Branches {
            conditional: 890,
            conditional_mispredicted: 56,
            indirect: 2,
            indirect_mispredicted: 3,
        };
        assert_eq!(b.minus(&empty).unwrap().mispredicted(), 400);
        assert!(empty.minus(&b).is_err());
        // More mispredictions without the counted calls (the predictor's aliasing): zero.
        let noisy = Branches {
            conditional_mispredicted: 500,
            ..empty
        };
        let d = b.minus(&noisy).unwrap();
        assert_eq!((d.conditional, d.conditional_mispredicted), (7000, 0));
    }

    #[test]
    fn a_run_without_branch_simulation_is_rejected() {
        let text = "events: Ir\nsummary: 100\n";
        assert!(parse_cachegrind(text).is_err());
        assert!(parse_cachegrind("events: Ir Bc\nsummary: 1\n").is_err());
    }

    #[test]
    fn every_core_model_has_a_penalty() {
        for m in &crate::mca::MODELS {
            assert!(penalty(m.reference).is_some(), "{}", m.reference);
            assert!(penalty(m.counter).is_some(), "{}", m.counter);
        }
        assert!((cost(100.0, 0.5, 20.0) - 110.0).abs() < 1e-12);
    }
}
