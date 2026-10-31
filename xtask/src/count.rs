//! `cargo xtask count`: instruction counts of the SDRk kernels against established libraries
//! (VOLK, rustfft, FFTW, `KissFFT`, GNU Radio, KFR, CMSIS-DSP; Ne10 for information) and,
//! where none exists, our own fallback.
//!
//! The method is differential. The measurement binary of `benches/kernels` (`count run <kernel>
//! <impl> <k>`) performs one warm-up call and then `k` calls on the same buffers. It is started
//! twice under an instruction counter, with `k = 0` and with `k = K`; the instructions per
//! sample are `(count(K) - count(0)) / (K * samples)`. Everything outside the `K` calls (start
//! of the process, allocation, set-up, the warm-up call) cancels. The in-place FFTs restore
//! their input before every call; the count of the restore alone is subtracted from both sides.
//!
//! Counters per level:
//!
//! | Level | Counter |
//! | --- | --- |
//! | `sse2`, `sse4.2` | Intel SDE with a CPUID of a strict SSE2 or SSE4.2 CPU (`-cpuid_in`, `-chip_check_cpuid`), so that VOLK, rustfft and FFTW choose the code of that level themselves |
//! | `avx2`, `avx2-runtime` | Valgrind `callgrind` |
//! | `avx512` | Intel SDE (`-spr`) |
//! | `neon` | QEMU user mode with the `insn` plugin |
//! | `simd128` | wasmtime fuel (found by bisection) |
//!
//! Under SDE only the region between two SSC markers of the binary is counted, which makes the
//! counts exact; each is still the median of up to three runs. The C references are built by
//! `benches/refs/build.sh` (pinned tarballs) into `benches/target/refs/<platform>`. The limits
//! are in `benches/limits.json` and the results in `benches/results/<platform>.json`. See
//! `docs/performance.md`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, anyhow, bail};
use serde_json::{Value, json};

/// Calls per measurement in the differential method.
pub const K: u64 = 16;
/// Samples per call unless a kernel fixes its own size.
pub const N: u64 = 4096;
/// How much worse than the comparison the accuracy may be, in dB.
pub const SNR_MARGIN_DB: f64 = 1.0;
/// Runs per count under Intel SDE, of which the median is taken. The SSC markers make the
/// counts exact, so the third run is made only when the first two differ.
pub const SDE_RUNS: usize = 3;
/// Guest address space reserved by QEMU (`-R`): it fixes the memory layout, and with it the
/// instructions of the start-up code, so that identical runs give identical counts (without it
/// they differ by up to 50 000 instructions).
pub const QEMU_RESERVED_VA: &str = "0x200000000";
/// The SSC markers around the counted calls (`benches/kernels`, `region_start`/`region_stop`).
const SDE_REGION: [&str; 6] = [
    "-controller_default_start",
    "0",
    "-start_ssc_mark",
    "5d4b0001:repeat",
    "-stop_ssc_mark",
    "5d4b0002:repeat",
];
/// Sample counts of the informational small-block runs.
pub const SMALL_N: [u64; 2] = [64, 100];

/// The ratio limits. The defaults can be overridden by the object `_limits` in
/// `benches/limits.json` (`{"release": 1.10, ...}`), so that the gate is tightened together with
/// the entries it applies to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Caps {
    /// Hard limit of a `release` entry.
    pub release: f64,
    /// Limit of a `dev` entry.
    pub dev: f64,
    /// Up to this ratio a passing `dev` entry is reported as ready to be promoted; above it a
    /// passing entry gets a warning. With the defaults of D14 it equals the release limit, so it
    /// only marks `dev` entries.
    pub warn: f64,
    /// The goal; results record whether an entry reaches it.
    pub goal: f64,
    /// Limit of a level with run-time detection against our own numbers of the level it
    /// detects (the cost of the dispatch).
    pub runtime: f64,
}

impl Default for Caps {
    fn default() -> Self {
        Caps {
            release: 1.10,
            dev: 1.50,
            warn: 1.10,
            goal: 1.00,
            runtime: 1.02,
        }
    }
}

/// The release state of a kernel on one level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Must stay within [`Caps::release`].
    Release,
    /// Under development: within [`Caps::dev`].
    Dev,
    /// Reported, never judged: the comparison is not valid for this level (the `waiver` text
    /// says why).
    Info,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Status::Release => "release",
            Status::Dev => "dev",
            Status::Info => "info",
        }
    }
}

/// One entry of `benches/limits.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limit {
    /// The state.
    pub status: Status,
    /// If present, an instruction ratio above the limit is reported as waived: the text names
    /// the open item in `docs/open-items.md` that tracks it. A waiver never hides the numbers,
    /// and it never covers the accuracy.
    pub waiver: Option<String>,
    /// The same for an accuracy more than [`SNR_MARGIN_DB`] below the comparison.
    pub accuracy_waiver: Option<String>,
}

impl Limit {
    /// A `release` entry without waivers.
    #[must_use]
    pub const fn release() -> Self {
        Limit {
            status: Status::Release,
            waiver: None,
            accuracy_waiver: None,
        }
    }
}

/// The parsed `benches/limits.json`: the caps, and the entries by kernel, then level.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    /// The ratio limits.
    pub caps: Caps,
    entries: BTreeMap<String, BTreeMap<String, Limit>>,
}

impl Limits {
    /// Parses the file: `{ "_limits": { "release": 1.10, ... }, "<kernel>": { "<level>": {
    /// "status": "release" | "dev" | "info", "waiver": "<text>", "accuracy_waiver": "<text>" } } }`.
    /// Other keys starting with `_` are comments.
    ///
    /// # Errors
    /// Invalid JSON, an unknown status, or a cap that is not a positive number.
    pub fn parse(text: &str) -> anyhow::Result<Limits> {
        let value: Value = serde_json::from_str(text).context("limits.json is not valid JSON")?;
        let Some(kernels) = value.as_object() else {
            bail!("limits.json must be an object");
        };
        let mut caps = Caps::default();
        if let Some(over) = kernels.get("_limits") {
            let Some(over) = over.as_object() else {
                bail!("limits.json: `_limits` must be an object");
            };
            for (name, v) in over {
                let x = v
                    .as_f64()
                    .filter(|x| *x > 0.0)
                    .ok_or_else(|| anyhow!("limits.json: `_limits.{name}` must be > 0"))?;
                match name.as_str() {
                    "release" => caps.release = x,
                    "dev" => caps.dev = x,
                    "warn" => caps.warn = x,
                    "goal" => caps.goal = x,
                    "runtime" => caps.runtime = x,
                    _ => bail!("limits.json: unknown cap `_limits.{name}`"),
                }
            }
        }
        let mut entries = BTreeMap::new();
        for (kernel, levels) in kernels {
            if kernel.starts_with('_') {
                continue;
            }
            let Some(levels) = levels.as_object() else {
                bail!("limits.json: `{kernel}` must be an object");
            };
            let mut per_level = BTreeMap::new();
            for (level, entry) in levels {
                let status = match entry.get("status").and_then(Value::as_str) {
                    Some("release") => Status::Release,
                    Some("dev") => Status::Dev,
                    Some("info") => Status::Info,
                    other => bail!("limits.json: `{kernel}.{level}`: unknown status {other:?}"),
                };
                let text = |key: &str| entry.get(key).and_then(Value::as_str).map(str::to_string);
                per_level.insert(
                    level.clone(),
                    Limit {
                        status,
                        waiver: text("waiver"),
                        accuracy_waiver: text("accuracy_waiver"),
                    },
                );
            }
            entries.insert(kernel.clone(), per_level);
        }
        Ok(Limits { caps, entries })
    }

    /// The entry of a kernel on a level, `None` if it is not listed.
    #[must_use]
    pub fn get(&self, kernel: &str, level: &str) -> Option<&Limit> {
        self.entries.get(kernel).and_then(|l| l.get(level))
    }
}

/// An informational comparison next to the judged one: another library, the cheapest VOLK
/// implementation, or a variant of the buffers (misaligned, small blocks).
#[derive(Clone, Debug, PartialEq)]
pub struct Extra {
    /// What differs from the judged comparison (`misaligned`, `n = 64`, the library).
    pub label: String,
    /// Instructions per sample of our kernel in this variant.
    pub ours: f64,
    /// Instructions per sample of the comparison in this variant.
    pub other: f64,
    /// Name of the comparison.
    pub compared_with: String,
    /// SNR in dB of the comparison, where measured.
    pub other_snr_db: Option<f64>,
}

impl Extra {
    /// `ours / other`.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        ratio(self.ours, self.other)
    }
}

/// What the comparison of one kernel on one level measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// Kernel name.
    pub kernel: String,
    /// Level name (`sse2`, `avx2`, ...).
    pub level: String,
    /// Instructions per sample of our kernel.
    pub ours: f64,
    /// Instructions per sample of the comparison.
    pub other: f64,
    /// Name of the comparison.
    pub compared_with: String,
    /// SNR in dB of our kernel against the f64 reference.
    pub ours_snr_db: f64,
    /// SNR in dB of the comparison against the f64 reference.
    pub other_snr_db: f64,
    /// Informational comparisons, never judged.
    pub extras: Vec<Extra>,
}

impl Measurement {
    /// Instructions of ours divided by instructions of the comparison.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        ratio(self.ours, self.other)
    }
}

fn ratio(ours: f64, other: f64) -> f64 {
    if other > 0.0 {
        ours / other
    } else {
        f64::INFINITY
    }
}

/// The judgement of one measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Within the limit.
    Pass,
    /// Within the limit, with a remark: at or above [`Caps::warn`], a waiver that no longer
    /// covers anything, or a `dev` entry ready to become `release`.
    Warn(String),
    /// Above the limit or less accurate than the comparison, without a waiver for it, or not
    /// listed in `limits.json`.
    Fail(String),
    /// Above the limit or less accurate, with a waiver for each exceedance.
    Waived(String),
    /// Not judged (`info` entry); carries the reason from `limits.json`.
    Info(String),
}

impl Verdict {
    /// The word of the console line and of the results file.
    #[must_use]
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Warn(_) => "warn",
            Verdict::Fail(_) => "FAIL",
            Verdict::Waived(_) => "waived",
            Verdict::Info(_) => "info",
        }
    }

    /// The explanation, empty for [`Verdict::Pass`].
    #[must_use]
    pub fn detail(&self) -> &str {
        match self {
            Verdict::Pass => "",
            Verdict::Warn(d) | Verdict::Fail(d) | Verdict::Waived(d) | Verdict::Info(d) => d,
        }
    }
}

/// How a level is judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Against an established library (or our fallback): the entry of `limits.json`, which
    /// must exist.
    Reference,
    /// Against our own numbers of another level (run-time detection against the fixed level):
    /// [`Caps::runtime`]; without an entry in `limits.json` the entry is `release`.
    OwnLevel,
}

/// `ratio` (above `cap`) with as many decimals as it takes to print above the cap:
/// `1.0215x` rather than a `1.02x` that "exceeds" a limit of `1.02x`.
fn over(ratio: f64, cap: f64) -> String {
    let digits = (2..6)
        .find(|d| {
            format!("{ratio:.d$}", d = *d)
                .parse::<f64>()
                .is_ok_and(|r| r > cap)
        })
        .unwrap_or(6);
    format!("{ratio:.digits$}x")
}

/// Judges a measurement. The instruction ratio and the accuracy are checked independently, and
/// each exceedance needs its own waiver (`waiver`, `accuracy_waiver`).
#[must_use]
pub fn judge(caps: &Caps, rule: Rule, limit: Option<&Limit>, m: &Measurement) -> Verdict {
    let release = Limit::release();
    let limit = match (limit, rule) {
        (Some(l), _) => l,
        (None, Rule::OwnLevel) => &release,
        (None, Rule::Reference) => {
            return Verdict::Fail(format!(
                "no entry for `{}` at `{}` in benches/limits.json",
                m.kernel, m.level
            ));
        }
    };
    let cap = match (limit.status, rule) {
        (Status::Info, _) => return Verdict::Info(limit.waiver.clone().unwrap_or_default()),
        (Status::Release, Rule::Reference) => caps.release,
        (Status::Release, Rule::OwnLevel) => caps.runtime,
        (Status::Dev, _) => caps.dev,
    };
    let status = limit.status.name();
    let ratio = m.ratio();
    let slow = ratio > cap;
    let inaccurate = m.other_snr_db - m.ours_snr_db > SNR_MARGIN_DB;
    let mut failing = Vec::new();
    let mut waived = Vec::new();
    if slow {
        let text = format!(
            "{} instructions per sample against {} exceeds the {status} limit of {cap:.2}x",
            over(ratio, cap),
            m.compared_with
        );
        match &limit.waiver {
            Some(w) => waived.push(format!("{text} (waiver: {w})")),
            None => failing.push(text),
        }
    }
    if inaccurate {
        let text = format!(
            "accuracy {:.1} dB is more than {SNR_MARGIN_DB:.0} dB below the {:.1} dB of {}",
            m.ours_snr_db, m.other_snr_db, m.compared_with
        );
        match &limit.accuracy_waiver {
            Some(w) => waived.push(format!("{text} (accuracy waiver: {w})")),
            None => failing.push(text),
        }
    }
    if !failing.is_empty() {
        failing.extend(waived);
        return Verdict::Fail(failing.join("; "));
    }
    if !waived.is_empty() {
        return Verdict::Waived(waived.join("; "));
    }
    let mut notes = Vec::new();
    if let Some(w) = &limit.waiver {
        notes.push(format!(
            "{ratio:.2}x is within the {status} limit of {cap:.2}x: the waiver is no longer needed ({w})"
        ));
    }
    if let Some(w) = &limit.accuracy_waiver {
        notes.push(format!(
            "the accuracy is within {SNR_MARGIN_DB:.0} dB: the accuracy waiver is no longer needed ({w})"
        ));
    }
    if limit.status == Status::Dev && rule == Rule::Reference && ratio <= caps.warn {
        notes.push(format!(
            "{ratio:.2}x: ready to be promoted from dev to release (release limit {:.2}x)",
            caps.release
        ));
    } else if ratio > caps.warn && rule == Rule::Reference {
        notes.push(format!(
            "{ratio:.2}x is above {:.2}x ({status} limit {cap:.2}x)",
            caps.warn
        ));
    }
    if notes.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Warn(notes.join("; "))
    }
}

/// `--strict` (the release gate of open item M13-4): why an entry blocks a release, if it
/// does. Only a `release` entry without waivers that passes (with or without a warning)
/// does not.
#[must_use]
pub fn blocks_release(limit: Option<&Limit>, verdict: &Verdict) -> Option<String> {
    match verdict {
        Verdict::Fail(d) => return Some(d.clone()),
        Verdict::Waived(d) => return Some(format!("waived: {d}")),
        Verdict::Info(d) => return Some(format!("not judged (info): {d}")),
        Verdict::Pass | Verdict::Warn(_) => {}
    }
    let limit = limit?;
    match (limit.status, &limit.waiver, &limit.accuracy_waiver) {
        (Status::Dev, ..) => Some("status `dev`".into()),
        (_, Some(w), _) | (_, None, Some(w)) => Some(format!("waiver: {w}")),
        _ => None,
    }
}

/// The outcome of a whole level.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// One line per kernel for the console (extras indented below it).
    pub lines: Vec<String>,
    /// `false` if any measurement failed.
    pub ok: bool,
    /// Entries that block a release under `--strict`: `kernel level: reason`.
    pub blockers: Vec<String>,
    /// The per-kernel records for the results file.
    pub records: BTreeMap<String, Value>,
}

/// Judges all measurements of one level.
#[must_use]
pub fn evaluate(limits: &Limits, rule: Rule, level: &str, measurements: &[Measurement]) -> Report {
    let caps = &limits.caps;
    let mut report = Report {
        ok: true,
        ..Report::default()
    };
    for m in measurements {
        let limit = limits.get(&m.kernel, level);
        let verdict = judge(caps, rule, limit, m);
        if matches!(verdict, Verdict::Fail(_)) {
            report.ok = false;
        }
        if let Some(why) = blocks_release(limit, &verdict) {
            report.blockers.push(format!("{} {level}: {why}", m.kernel));
        }
        let status = match (limit, rule) {
            (Some(l), _) => l.status.name(),
            (None, Rule::OwnLevel) => "release",
            (None, Rule::Reference) => "unlisted",
        };
        let goal = m.ratio() <= caps.goal;
        let mut line = format!(
            "{:<22} {:<12} {:>8.2} vs {:>8.2} instr/sample  ratio {:>5.2}{}  [{status}]  {}",
            m.kernel,
            m.level,
            m.ours,
            m.other,
            m.ratio(),
            if goal { " goal" } else { "     " },
            verdict.word()
        );
        if !verdict.detail().is_empty() {
            let _ = write!(line, "  {}", verdict.detail());
        }
        report.lines.push(line);
        for e in &m.extras {
            report.lines.push(format!(
                "    {:<34} {:>8.2} vs {:>8.2} instr/sample  ratio {:>5.2}  ({})",
                e.label,
                e.ours,
                e.other,
                e.ratio(),
                e.compared_with
            ));
        }
        report
            .records
            .insert(m.kernel.clone(), record(m, status, &verdict, goal));
    }
    report
}

fn record(m: &Measurement, status: &str, verdict: &Verdict, goal: bool) -> Value {
    let extras: Vec<Value> = m
        .extras
        .iter()
        .map(|e| {
            json!({
                "label": e.label,
                "ours_instr_per_sample": round2(e.ours),
                "other_instr_per_sample": round2(e.other),
                "ratio": round2(e.ratio()),
                "compared_with": e.compared_with,
                "other_snr_db": e.other_snr_db.map(round2),
            })
        })
        .collect();
    let mut r = json!({
        "ours_instr_per_sample": round2(m.ours),
        "other_instr_per_sample": round2(m.other),
        "ratio": round2(m.ratio()),
        "compared_with": m.compared_with,
        "status": status,
        "verdict": verdict.word().to_lowercase(),
        "goal": goal,
        "snr_db": { "ours": round2(m.ours_snr_db), "other": round2(m.other_snr_db) },
    });
    if !extras.is_empty()
        && let Some(obj) = r.as_object_mut()
    {
        obj.insert("informational".into(), Value::Array(extras));
    }
    r
}

pub(crate) fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

// ---------------------------------------------------------------------------------------------
// The measurement plan

/// What a kernel is compared with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reference {
    /// VOLK (pinned per level, GNU Radio's call pattern for the FIR family).
    Volk,
    /// The cheapest VOLK implementation of the machine (informational).
    VolkCheapest,
    /// The loop of a GNU Radio block that is no VOLK kernel (`agc_cc`, `pwr_squelch_cc`),
    /// compiled with the generic flags of GNU Radio's own build.
    GnuRadio,
    /// rustfft, `FftPlanner`.
    RustFft,
    /// FFTW, single precision, out of place (as GNU Radio's `fft_vcc` uses it).
    Fftw,
    /// FFTW in place, which needs more passes than out of place.
    FftwInPlace,
    /// `KissFFT`, 16-bit fixed point.
    KissFft,
    /// Ne10, int16 NEON.
    Ne10,
    /// KFR's biquad cascade (`kfr::iir`), compiled for the level (D15).
    Kfr,
    /// CMSIS-DSP's biquad cascades in generic C, compiled for the level (D15, D16).
    Cmsis,
    /// liquid-dsp: the DC blocker (`iirfilt_*_create_dc_blocker`, portable C) and the halfband
    /// decimators and Hilbert transform decimator (`resamp2`, `firhilbf`) with the SIMD dot
    /// product of the level (the portable one for a fallback entry).
    Liquid,
    /// libairspy 1.0.10's real-to-complex converters (`iqconverter_float`, `iqconverter_int16`,
    /// conversion only).
    Libairspy,
    /// Our own fallback, where no established library exists.
    Fallback,
    /// Our own numbers of another level (run-time detection against that fixed level).
    OwnLevel(&'static str),
    /// Our own kernel on another workload of the table (`ours` of the kernel named): the
    /// work a specialised kernel saves against the general kernel it replaces (D63). It is
    /// information, never the judged comparison; the results name it `ours:<kernel>`.
    OwnKernel(&'static str),
}

impl Reference {
    /// The implementation name of the measurement binary.
    #[must_use]
    pub fn implementation(self) -> &'static str {
        match self {
            Reference::Volk | Reference::VolkCheapest => "volk",
            Reference::GnuRadio => "gnuradio",
            Reference::RustFft => "rustfft",
            Reference::Fftw => "fftw",
            Reference::FftwInPlace => "fftw-in-place",
            Reference::KissFft => "kissfft",
            Reference::Ne10 => "ne10",
            Reference::Kfr => "kfr",
            Reference::Cmsis => "cmsis",
            Reference::Liquid => "liquid",
            Reference::Libairspy => "airspy",
            Reference::Fallback => "fallback",
            Reference::OwnLevel(_) | Reference::OwnKernel(_) => "ours",
        }
    }
}

/// The kind of comparison a kernel has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// A VOLK kernel exists.
    Volk,
    /// A VOLK kernel exists, but not in the wasm32 build of VOLK (`benches/refs/volk-wasm.c`
    /// holds only the generic kernels of the entries of [`Family::Volk`]): there our fallback is
    /// the comparison, as information (D12).
    VolkNative,
    /// A VOLK kernel does the same work up to a constant offset (`convert_cu8`: VOLK converts
    /// signed bytes only): it is the gate, and our fallback is reported as information.
    VolkAndOwn,
    /// A loop of GNU Radio that is no VOLK kernel (`agc_cc`, `pwr_squelch_cc`,
    /// `frequency_modulator_fc`): it is the gate (where `limits.json` judges it), and our fallback
    /// is reported as information.
    GnuRadio,
    /// The C32 FFT.
    FftC32,
    /// The CQ15 FFT.
    FftCq15,
    /// The real-input FFT: per level (`LevelPlan::rfft`) FFTW's real-input transform or rustfft's
    /// real planner (`realfft`), our fallback as information.
    RealFft,
    /// No established library: our fallback.
    Own,
    /// A workload of an own-kernel comparison ([`Reference::OwnKernel`], D63): measured as
    /// [`Family::Own`], but not a kernel of the library, so it has no entry at a level that is
    /// judged against another build of ours (`avx2-runtime`: the workload runs a kernel whose
    /// run-time path is judged by its own entry).
    Workload,
    /// The f32 and C32 biquads (D15): KFR's SIMD cascade where the level has a build of it
    /// (`LevelPlan::kfr`), CMSIS-DSP's generic C as information; at a level without KFR,
    /// CMSIS-DSP is the gate.
    Kfr,
    /// The Q15 and CQ15 biquads (D15), `rotate` and `fm_mod` of `CQ15`: no SIMD library has
    /// them on our levels, so our fallback is the comparison; CMSIS-DSP's generic C (for
    /// `rotate` and `fm_mod` composed of its building blocks) as information.
    Q15Own,
    /// A fallback entry (D16, `<kernel>_fallback`): the measured side is our fallback, compiled
    /// for the level, and the gate is CMSIS-DSP's generic C. Not measured at a level with
    /// run-time detection, whose fallback is the baseline build.
    FallbackVsCmsis,
    /// A fallback entry as [`Family::FallbackVsCmsis`] whose gate is the scalar loop of a GNU
    /// Radio block (`rotate`: `gr::blocks::rotator`, `fm_mod`: `frequency_modulator_fc`,
    /// `deemphasis_fallback`: the IIR loop of `fm_deemph`, `agc`: `agc_cc`, `quad_demod`:
    /// `quadrature_demod_cf`), compiled with the generic flags of GNU Radio's own build;
    /// CMSIS-DSP's generic C as information where the entry has it.
    FallbackVsGnuRadio,
    /// A fallback entry as [`Family::FallbackVsCmsis`] whose gate is liquid-dsp's DC blocker
    /// (`dc_block_fallback`, `dc_block_real_fallback`) or its halfband decimator with the
    /// portable dot product (`halfband_decimate_fallback`, `halfband_decimate_real_fallback`).
    FallbackVsLiquid,
    /// The halfband decimators of the float types (`halfband_decimate`,
    /// `halfband_decimate_real`): liquid-dsp's `resamp2` with its SIMD dot product of the level
    /// (the portable loop at `simd128`, where liquid has no SIMD loop).
    HalfbandLiquid,
    /// `real_to_complex` of `C32`: liquid-dsp's `firhilbf` with its SIMD dot product; at `sse2`
    /// also libairspy's `iqconverter_float` (SSE2) as information.
    HilbertLiquid,
    /// `real_to_complex_cq15`: libairspy's `iqconverter_int16`, a lower accuracy class (truncating
    /// `>> 15`, a 32-bit accumulator, the odd samples halved). Its status is the maintainer's
    /// open decision (I23-1), `info` until then; our fallback as information.
    HilbertAirspyQ15,
    /// `real_to_complex_fallback`: liquid-dsp's `firhilbf` with the portable dot product; libairspy's
    /// generic C as information.
    FallbackHilbert,
    /// `real_to_complex_cq15_fallback`: libairspy's `iqconverter_int16` (generic C).
    FallbackVsAirspy,
    /// The fallback entry of a `C32` or real-input FFT (`fft_<n>_fallback`,
    /// `rfft_<n>_fallback`): as [`Family::FallbackVsCmsis`] against `arm_cfft_f32` or
    /// `arm_rfft_fast_f32`, with the input restore of the FFTs subtracted.
    FftFallbackVsCmsis,
    /// The fallback entry of a `CQ15` FFT (`fft_cq15_<n>..._fallback`): as
    /// [`Family::FftFallbackVsCmsis`] against `arm_cfft_q15` (fixed `1/N` scaling with
    /// truncating shifts), with `KissFFT` (scalar C as well) as information.
    FftCq15FallbackVsCmsis,
}

impl Family {
    /// The FFTs, whose measurements restore their input before every call (see `Impl::Restore`
    /// of the measurement binary) and whose sizes are part of the name.
    #[must_use]
    pub fn is_fft(self) -> bool {
        matches!(
            self,
            Family::FftC32
                | Family::FftCq15
                | Family::RealFft
                | Family::FftFallbackVsCmsis
                | Family::FftCq15FallbackVsCmsis
        )
    }

    /// A fallback entry (`<kernel>_fallback`), whose measured side is our fallback.
    #[must_use]
    pub fn is_fallback_entry(self) -> bool {
        matches!(
            self,
            Family::FallbackVsCmsis
                | Family::FallbackVsGnuRadio
                | Family::FallbackVsLiquid
                | Family::FallbackHilbert
                | Family::FallbackVsAirspy
                | Family::FftFallbackVsCmsis
                | Family::FftCq15FallbackVsCmsis
        )
    }

    /// The implementation that is measured: our fallback for a fallback entry, our kernel
    /// through the facade otherwise.
    #[must_use]
    pub fn measured(self) -> &'static str {
        if self.is_fallback_entry() {
            "fallback"
        } else {
            "ours"
        }
    }

    /// Whether the kernel has an entry at `plan`'s level.
    #[must_use]
    pub fn measured_at(self, plan: &LevelPlan) -> bool {
        !((self.is_fallback_entry() || self == Family::Workload) && plan.baseline.is_some())
    }
}

/// A kernel of the benchmark table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelSpec {
    /// Name (as the measurement binary knows it).
    pub name: &'static str,
    /// Its comparisons.
    pub family: Family,
    /// Samples per call.
    pub samples: u64,
    /// Also measured with the [`SMALL_N`] block sizes (informational).
    pub small_n: bool,
    /// Also measured with misaligned buffers (informational).
    pub misaligned: bool,
    /// Another kernel of the table whose `ours` is reported next to the comparisons as
    /// information ([`Reference::OwnKernel`]).
    pub own: Option<&'static str>,
}

const fn spec(name: &'static str, family: Family, small_n: bool) -> KernelSpec {
    KernelSpec {
        name,
        family,
        samples: N,
        small_n,
        misaligned: true,
        own: None,
    }
}

/// A kernel with an own-kernel comparison.
const fn spec_own(name: &'static str, family: Family, own: &'static str) -> KernelSpec {
    KernelSpec {
        name,
        family,
        samples: N,
        small_n: false,
        misaligned: true,
        own: Some(own),
    }
}

const fn fft(name: &'static str, family: Family, samples: u64) -> KernelSpec {
    KernelSpec {
        name,
        family,
        samples,
        small_n: false,
        misaligned: false,
        own: None,
    }
}

/// The kernels of the benchmark table, in report order.
pub static KERNELS: [KernelSpec; 139] = [
    spec("convert_cs8", Family::Volk, true),
    spec("convert_cs16", Family::Volk, true),
    spec("convert_cu8", Family::VolkAndOwn, true),
    spec("convert_real_rs16", Family::Volk, true),
    spec("convert_cq15_cs8", Family::Volk, true),
    spec("convert_cq15_cu8", Family::VolkAndOwn, true),
    spec("convert_out_cs16", Family::Volk, true),
    spec("convert_out_cs8", Family::Volk, true),
    spec("convert_out_cu8", Family::Volk, true),
    spec("convert_out_real_rs16", Family::Volk, true),
    spec("convert_out_real_rs8", Family::Volk, true),
    spec("convert_out_real_ru8", Family::Volk, true),
    spec("convert_cs8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_cs16_fallback", Family::FallbackVsCmsis, false),
    spec("convert_cu8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_real_rs16_fallback", Family::FallbackVsCmsis, false),
    spec("convert_cq15_cs8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_cq15_cu8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_out_cs16_fallback", Family::FallbackVsCmsis, false),
    spec("convert_out_cs8_fallback", Family::FallbackVsCmsis, false),
    spec("convert_out_cu8_fallback", Family::FallbackVsCmsis, false),
    spec(
        "convert_out_real_rs16_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "convert_out_real_rs8_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "convert_out_real_ru8_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("rotate", Family::Volk, false),
    spec("rotate_cq15", Family::Q15Own, false),
    spec("rotate_fallback", Family::FallbackVsGnuRadio, false),
    spec("rotate_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("fir", Family::Volk, false),
    spec("fir_decimate", Family::Volk, false),
    spec("fir_interpolate", Family::Volk, false),
    spec("resample", Family::Volk, false),
    spec("fir_interpolate_real", Family::Volk, false),
    spec("resample_real", Family::Volk, false),
    spec("fir_interpolate_real_q15", Family::Own, false),
    spec("resample_real_q15", Family::Own, false),
    spec("fir_cq15", Family::Own, false),
    spec("fir_decimate_cq15", Family::Own, false),
    spec("fir_interpolate_cq15", Family::Own, false),
    spec("resample_cq15", Family::Own, false),
    spec("mag_sq", Family::Volk, false),
    spec("mag_sq_q15", Family::Own, false),
    spec("quad_demod", Family::Volk, false),
    spec("quad_demod_cq15", Family::Own, false),
    spec("quad_demod_fallback", Family::FallbackVsGnuRadio, false),
    spec("quad_demod_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("fm_mod", Family::GnuRadio, false),
    spec("fm_mod_cq15", Family::Q15Own, false),
    spec("fm_mod_fallback", Family::FallbackVsGnuRadio, false),
    spec("fm_mod_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("squelch", Family::GnuRadio, false),
    spec("squelch_cq15", Family::Own, false),
    spec("mag", Family::Volk, false),
    spec("mag_q15", Family::Volk, false),
    spec("window", Family::Volk, false),
    spec("window_cq15", Family::Own, false),
    spec("power_db", Family::Volk, false),
    spec("power_db_q15", Family::Own, false),
    spec("mag_sq_fallback", Family::FallbackVsCmsis, false),
    spec("mag_sq_q15_fallback", Family::FallbackVsCmsis, false),
    spec("mag_fallback", Family::FallbackVsCmsis, false),
    spec("mag_q15_fallback", Family::FallbackVsCmsis, false),
    spec("window_fallback", Family::FallbackVsCmsis, false),
    spec("window_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("power_db_fallback", Family::FallbackVsCmsis, false),
    spec("power_db_q15_fallback", Family::FallbackVsCmsis, false),
    spec("agc", Family::GnuRadio, false),
    spec("agc_cq15", Family::Own, false),
    spec("agc_fallback", Family::FallbackVsGnuRadio, false),
    spec("dc_block_real", Family::Own, false),
    spec("dc_block_real_q15", Family::Own, false),
    spec("deemphasis", Family::Own, false),
    spec("deemphasis_q15", Family::Own, false),
    spec("dc_block", Family::Own, false),
    spec("dc_block_cq15", Family::Own, false),
    spec("dc_block_real_fallback", Family::FallbackVsLiquid, false),
    spec("deemphasis_fallback", Family::FallbackVsGnuRadio, false),
    spec("dc_block_fallback", Family::FallbackVsLiquid, false),
    spec("biquad", Family::Kfr, true),
    spec("biquad_cq15", Family::Q15Own, true),
    spec("biquad_real", Family::Kfr, true),
    spec("biquad_real_q15", Family::Q15Own, true),
    spec("biquad_fallback", Family::FallbackVsCmsis, false),
    spec("biquad_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("biquad_real_fallback", Family::FallbackVsCmsis, false),
    spec("biquad_real_q15_fallback", Family::FallbackVsCmsis, false),
    spec("fir_fallback", Family::FallbackVsCmsis, false),
    spec("fir_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("fir_decimate_fallback", Family::FallbackVsCmsis, false),
    spec("fir_decimate_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("fir_interpolate_fallback", Family::FallbackVsCmsis, false),
    spec(
        "fir_interpolate_cq15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("resample_fallback", Family::FallbackVsCmsis, false),
    spec("resample_cq15_fallback", Family::FallbackVsCmsis, false),
    spec(
        "fir_interpolate_real_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "fir_interpolate_real_q15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("resample_real_fallback", Family::FallbackVsCmsis, false),
    spec("resample_real_q15_fallback", Family::FallbackVsCmsis, false),
    spec("dot", Family::Volk, true),
    spec("dot_cq15", Family::Own, true),
    spec("dot_real", Family::VolkNative, true),
    spec("dot_real_q15", Family::Own, true),
    spec("scale", Family::VolkNative, false),
    spec("scale_cq15", Family::Own, false),
    spec("dot_fallback", Family::FallbackVsCmsis, false),
    spec("dot_cq15_fallback", Family::FallbackVsCmsis, false),
    spec("dot_real_fallback", Family::FallbackVsCmsis, false),
    spec("dot_real_q15_fallback", Family::FallbackVsCmsis, false),
    spec("scale_fallback", Family::FallbackVsCmsis, false),
    spec("scale_cq15_fallback", Family::FallbackVsCmsis, false),
    spec_own(
        "halfband_decimate",
        Family::HalfbandLiquid,
        "halfband_ref_fir_decimate",
    ),
    spec_own(
        "halfband_decimate_cq15",
        Family::Q15Own,
        "halfband_ref_fir_decimate_cq15",
    ),
    spec("halfband_decimate_real", Family::HalfbandLiquid, false),
    spec("halfband_decimate_real_q15", Family::Q15Own, false),
    spec_own(
        "real_to_complex",
        Family::HilbertLiquid,
        "halfband_ref_shifted",
    ),
    spec_own(
        "real_to_complex_cq15",
        Family::HilbertAirspyQ15,
        "halfband_ref_shifted_cq15",
    ),
    spec(
        "halfband_decimate_fallback",
        Family::FallbackVsLiquid,
        false,
    ),
    spec(
        "halfband_decimate_cq15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec(
        "halfband_decimate_real_fallback",
        Family::FallbackVsLiquid,
        false,
    ),
    spec(
        "halfband_decimate_real_q15_fallback",
        Family::FallbackVsCmsis,
        false,
    ),
    spec("real_to_complex_fallback", Family::FallbackHilbert, false),
    spec(
        "real_to_complex_cq15_fallback",
        Family::FallbackVsAirspy,
        false,
    ),
    spec("halfband_ref_fir_decimate", Family::Workload, false),
    spec("halfband_ref_fir_decimate_cq15", Family::Workload, false),
    spec("halfband_ref_shifted", Family::Workload, false),
    spec("halfband_ref_shifted_cq15", Family::Workload, false),
    fft("fft_64", Family::FftC32, 64),
    fft("fft_1024", Family::FftC32, 1024),
    fft("fft_4096", Family::FftC32, 4096),
    fft("fft_cq15_1024", Family::FftCq15, 1024),
    fft("fft_cq15_1024_default", Family::FftCq15, 1024),
    fft("fft_cq15_1024_full_scale", Family::FftCq15, 1024),
    fft("rfft_1024", Family::RealFft, 1024),
    fft("fft_64_fallback", Family::FftFallbackVsCmsis, 64),
    fft("fft_1024_fallback", Family::FftFallbackVsCmsis, 1024),
    fft("fft_4096_fallback", Family::FftFallbackVsCmsis, 4096),
    fft(
        "fft_cq15_1024_fallback",
        Family::FftCq15FallbackVsCmsis,
        1024,
    ),
    fft(
        "fft_cq15_1024_default_fallback",
        Family::FftCq15FallbackVsCmsis,
        1024,
    ),
    fft("rfft_1024_fallback", Family::FftFallbackVsCmsis, 1024),
];

/// The CPU Intel SDE emulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdeCpu {
    /// x86-64 with SSE2 and nothing above it (own CPUID definition).
    Sse2,
    /// x86-64-v2: up to SSE4.2 and POPCNT (own CPUID definition).
    Sse42,
    /// Sapphire Rapids (`-spr`).
    SapphireRapids,
}

/// How the measurement binary is run and counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    /// Valgrind callgrind.
    Callgrind,
    /// Intel SDE `-icount` between the SSC markers.
    Sde(SdeCpu),
    /// QEMU user mode with the `insn` plugin.
    Qemu,
    /// wasmtime with fuel.
    Wasmtime,
}

impl Counter {
    fn describe(self) -> &'static str {
        match self {
            Counter::Callgrind => "valgrind callgrind",
            Counter::Sde(SdeCpu::Sse2) => "Intel SDE, CPUID of a strict SSE2 x86-64 CPU",
            Counter::Sde(SdeCpu::Sse42) => "Intel SDE, CPUID of an x86-64-v2 (SSE4.2) CPU",
            Counter::Sde(SdeCpu::SapphireRapids) => "Intel SDE -spr",
            Counter::Qemu => "qemu-aarch64 insn plugin",
            Counter::Wasmtime => "wasmtime fuel",
        }
    }
}

/// What gives the execution counts per instruction for the simulated cycles of a level
/// (`cargo xtask measure`, D68): the counter of the level, in the same environment as its
/// count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profiler {
    /// callgrind with `--dump-instr=yes --collect-jumps=yes`.
    Callgrind,
    /// Intel SDE `-dcfg` on the CPU of the level; on a level counted with callgrind (the
    /// validation of the SDE path, `--cycle-counter sde`) on SDE's Haswell (`-hsw`, AVX2 and
    /// FMA).
    Sde,
    /// QEMU user mode with the plugin `xtask/plugins/bbcount.c`.
    Qemu,
}

impl Profiler {
    /// The profiler of the counter of `plan`, `None` for wasmtime (no core model, D68).
    #[must_use]
    pub fn of(plan: &LevelPlan) -> Option<Profiler> {
        match plan.counter {
            Counter::Callgrind => Some(Profiler::Callgrind),
            Counter::Sde(_) => Some(Profiler::Sde),
            Counter::Qemu => Some(Profiler::Qemu),
            Counter::Wasmtime => None,
        }
    }

    /// The name in the command line (`--cycle-counter`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Profiler::Callgrind => "callgrind",
            Profiler::Sde => "sde",
            Profiler::Qemu => "qemu",
        }
    }

    /// The profiler called `name`.
    #[must_use]
    pub fn parse(name: &str) -> Option<Profiler> {
        [Profiler::Callgrind, Profiler::Sde, Profiler::Qemu]
            .into_iter()
            .find(|p| p.name() == name)
    }

    /// What it is, for the report and the JSON (`block_counts`).
    #[must_use]
    pub fn describe(self, plan: &LevelPlan) -> String {
        match (self, plan.counter) {
            (Profiler::Callgrind, _) => {
                "valgrind callgrind, execution counts per instruction and taken jumps".to_string()
            }
            (Profiler::Sde, Counter::Sde(_)) => format!(
                "Intel SDE -dcfg, block and branch counts ({})",
                plan.counter.describe()
            ),
            (Profiler::Sde, _) => {
                "Intel SDE -dcfg -hsw, block and branch counts (validation of the SDE path)"
                    .to_string()
            }
            (Profiler::Qemu, _) => {
                "qemu-aarch64 with the bbcount plugin, translation-block and transfer counts"
                    .to_string()
            }
        }
    }
}

/// The `volk_config` of a level: which implementations VOLK's dispatcher takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolkPin {
    /// An empty `volk_config`: VOLK's default dispatch on the machine it selects.
    Default,
    /// The defaults of VOLK's `sse2_64_mmx` machine.
    Sse2,
    /// The defaults of VOLK's `sse4_2_64_mmx` machine.
    Sse42,
    /// Implementations that need no more than AVX2 and FMA (no AVX-512).
    Avx2,
    /// The defaults of VOLK's `neonv8` machine (D11).
    Neon,
}

impl VolkPin {
    /// The `volk_config` text.
    #[must_use]
    pub const fn config(self) -> &'static str {
        match self {
            Self::Default => "",
            Self::Sse2 => VOLK_SSE2_CONFIG,
            Self::Sse42 => VOLK_SSE42_CONFIG,
            Self::Avx2 => VOLK_AVX2_CONFIG,
            Self::Neon => VOLK_NEON_CONFIG,
        }
    }
}

/// A level of a platform: how to build the binary, how to count it and what it is compared
/// with.
#[derive(Clone, Copy, Debug)]
pub struct LevelPlan {
    /// Platform name given to `--platform`.
    pub platform: &'static str,
    /// Level name (key of `limits.json`).
    pub level: &'static str,
    /// `cargo --target`, if cross-compiling.
    pub target: Option<&'static str>,
    /// `RUSTFLAGS`.
    pub rustflags: &'static str,
    /// Feature `std` of the kernels crate (run-time detection of the level).
    pub runtime_detect: bool,
    /// The counter.
    pub counter: Counter,
    /// The platform of the C references (`benches/refs/build.sh`).
    pub refs: &'static str,
    /// The VOLK implementations the dispatcher may choose.
    pub volk_pin: VolkPin,
    /// The judged comparison of the C32 FFT and the informational ones.
    pub fft_c32: (Reference, &'static [Reference]),
    /// The judged comparison of the real-input FFT and the informational ones.
    pub rfft: (Reference, &'static [Reference]),
    /// Informational comparisons of the CQ15 FFT next to `KissFFT`.
    pub fft_cq15_extra: &'static [Reference],
    /// Also report the cheapest VOLK implementation of every VOLK kernel (D11).
    pub cheapest_volk: bool,
    /// KFR is built for this level and gates the f32 and C32 biquads (D15); without it
    /// CMSIS-DSP does.
    pub kfr: bool,
    /// Judged against our own numbers of this level instead of the libraries.
    pub baseline: Option<&'static str>,
}

const RUSTFFT: (Reference, &[Reference]) = (Reference::RustFft, &[]);
/// rustfft's real-input planner (`realfft`), our fallback as information.
const REALFFT: (Reference, &[Reference]) = (Reference::RustFft, &[Reference::Fallback]);

/// All levels, in the order they run.
pub const PLANS: [LevelPlan; 7] = [
    LevelPlan {
        platform: "x86_64",
        level: "sse2",
        target: None,
        rustflags: "",
        runtime_detect: false,
        counter: Counter::Sde(SdeCpu::Sse2),
        refs: "x86_64",
        volk_pin: VolkPin::Sse2,
        // rustfft has no planner below SSE4.1, so on this CPU it is scalar (D3). FFTW is
        // judged out of place, its usual use; in place it needs more passes.
        fft_c32: (
            Reference::Fftw,
            &[Reference::FftwInPlace, Reference::RustFft],
        ),
        // FFTW's real-input transform out of place, as the complex one (D3).
        rfft: (Reference::Fftw, &[Reference::RustFft, Reference::Fallback]),
        fft_cq15_extra: &[],
        cheapest_volk: false,
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "x86_64",
        level: "sse4.2",
        target: None,
        rustflags: "-C target-cpu=x86-64-v2",
        runtime_detect: false,
        counter: Counter::Sde(SdeCpu::Sse42),
        refs: "x86_64",
        volk_pin: VolkPin::Sse42,
        fft_c32: RUSTFFT,
        // FFTW has no codelets above SSE2, so it runs the code of `sse2` here, and its real-input
        // transform is the stronger reference at this level.
        rfft: (Reference::Fftw, &[Reference::RustFft, Reference::Fallback]),
        fft_cq15_extra: &[],
        cheapest_volk: false,
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "x86_64",
        level: "avx2",
        target: None,
        rustflags: "-C target-cpu=x86-64-v3",
        runtime_detect: false,
        counter: Counter::Callgrind,
        refs: "x86_64",
        volk_pin: VolkPin::Avx2,
        fft_c32: RUSTFFT,
        rfft: REALFFT,
        fft_cq15_extra: &[],
        cheapest_volk: false,
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "x86_64",
        level: "avx2-runtime",
        target: None,
        rustflags: "",
        runtime_detect: true,
        counter: Counter::Callgrind,
        refs: "x86_64",
        volk_pin: VolkPin::Avx2,
        fft_c32: RUSTFFT,
        rfft: REALFFT,
        fft_cq15_extra: &[],
        cheapest_volk: false,
        kfr: true,
        baseline: Some("avx2"),
    },
    LevelPlan {
        platform: "x86_64-avx512",
        level: "avx512",
        // An explicit target keeps `RUSTFLAGS` away from the build scripts, which run on the
        // host (without AVX-512) while the measurement binary runs under SDE.
        target: Some("x86_64-unknown-linux-gnu"),
        rustflags: "-C target-cpu=sapphirerapids",
        runtime_detect: false,
        counter: Counter::Sde(SdeCpu::SapphireRapids),
        refs: "x86_64",
        volk_pin: VolkPin::Default,
        fft_c32: RUSTFFT,
        rfft: REALFFT,
        fft_cq15_extra: &[],
        cheapest_volk: false,
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "aarch64",
        level: "neon",
        target: Some("aarch64-unknown-linux-gnu"),
        rustflags: "",
        runtime_detect: false,
        counter: Counter::Qemu,
        refs: "aarch64",
        volk_pin: VolkPin::Neon,
        fft_c32: (Reference::RustFft, &[Reference::Fftw]),
        rfft: (Reference::RustFft, &[Reference::Fftw, Reference::Fallback]),
        fft_cq15_extra: &[Reference::Ne10],
        cheapest_volk: true,
        kfr: true,
        baseline: None,
    },
    LevelPlan {
        platform: "wasm32",
        level: "simd128",
        target: Some("wasm32-wasip1"),
        rustflags: "-C target-feature=+simd128",
        runtime_detect: false,
        counter: Counter::Wasmtime,
        refs: "wasm32",
        volk_pin: VolkPin::Default,
        fft_c32: RUSTFFT,
        rfft: REALFFT,
        fft_cq15_extra: &[],
        cheapest_volk: false,
        // KFR in its Emscripten configuration (sse41 over SIMD128, benches/refs/build.sh, D29).
        kfr: true,
        baseline: None,
    },
];

/// The judged comparison of a kernel on a level and the informational ones.
#[must_use]
pub fn comparisons(plan: &LevelPlan, spec: &KernelSpec) -> (Reference, Vec<Reference>) {
    if let Some(level) = plan.baseline {
        return (Reference::OwnLevel(level), Vec::new());
    }
    let (gate, mut extras) = family_comparisons(plan, spec);
    if let Some(own) = spec.own {
        extras.push(Reference::OwnKernel(own));
    }
    (gate, extras)
}

fn family_comparisons(plan: &LevelPlan, spec: &KernelSpec) -> (Reference, Vec<Reference>) {
    match spec.family {
        Family::VolkNative if plan.refs == "wasm32" => (Reference::Fallback, Vec::new()),
        Family::Volk | Family::VolkAndOwn | Family::VolkNative => {
            let mut extras = Vec::new();
            if plan.cheapest_volk {
                extras.push(Reference::VolkCheapest);
            }
            if spec.family == Family::VolkAndOwn {
                extras.push(Reference::Fallback);
            }
            (Reference::Volk, extras)
        }
        Family::GnuRadio => (Reference::GnuRadio, vec![Reference::Fallback]),
        Family::FftC32 => (plan.fft_c32.0, plan.fft_c32.1.to_vec()),
        Family::FftCq15 => (Reference::KissFft, plan.fft_cq15_extra.to_vec()),
        Family::RealFft => (plan.rfft.0, plan.rfft.1.to_vec()),
        Family::Own | Family::Workload => (Reference::Fallback, Vec::new()),
        Family::Kfr if plan.kfr => (Reference::Kfr, vec![Reference::Cmsis]),
        // Without KFR at the level, CMSIS-DSP gates, as it does for a fallback entry.
        Family::Kfr | Family::FallbackVsCmsis | Family::FftFallbackVsCmsis => {
            (Reference::Cmsis, Vec::new())
        }
        Family::FftCq15FallbackVsCmsis => (Reference::Cmsis, vec![Reference::KissFft]),
        Family::Q15Own => (Reference::Fallback, vec![Reference::Cmsis]),
        Family::FallbackVsGnuRadio => (Reference::GnuRadio, vec![Reference::Cmsis]),
        // libairspy's SSE2 converter is the information at the level it is written for.
        Family::HilbertLiquid if plan.level == "sse2" => {
            (Reference::Liquid, vec![Reference::Libairspy])
        }
        Family::FallbackVsLiquid | Family::HalfbandLiquid | Family::HilbertLiquid => {
            (Reference::Liquid, Vec::new())
        }
        Family::HilbertAirspyQ15 => (Reference::Libairspy, vec![Reference::Fallback]),
        Family::FallbackHilbert => (Reference::Liquid, vec![Reference::Libairspy]),
        Family::FallbackVsAirspy => (Reference::Libairspy, Vec::new()),
    }
}

/// The defaults of VOLK 3.3's `sse2_64_mmx` machine for the level `sse2` (the strict SSE2 CPU
/// of SDE selects that machine). VOLK has no SSE2 implementation of `rotator2`, `atan2`,
/// `multiply_conjugate`, `8i_s32f_convert_32f` and `8i_convert_16i`: their portable `generic`
/// code remains. The compiler vectorizes that of the two conversions with SSE2, a fair reference; `rotator2` and
/// `atan2` have no SSE implementation at all and their `generic` code is scalar, so those entries
/// are `info`.
const VOLK_SSE2_CONFIG: &str = "\
volk_8i_s32f_convert_32f generic generic
volk_16i_s32f_convert_32f a_sse u_sse
volk_8i_convert_16i generic generic
volk_32f_s32f_convert_16i a_sse2 u_sse2
volk_32f_s32f_convert_8i a_sse2 u_sse2
volk_32f_s32f_x2_convert_8u a_sse2 u_sse2
volk_32fc_s32fc_x2_rotator2_32fc generic generic
volk_32fc_32f_dot_prod_32fc a_sse u_sse
volk_32fc_magnitude_squared_32f a_sse u_sse
volk_32fc_magnitude_32f a_sse u_sse
volk_16ic_magnitude_16i a_sse generic
volk_32fc_x2_multiply_conjugate_32fc generic generic
volk_32fc_s32f_atan2_32f generic generic
volk_32fc_32f_multiply_32fc a_sse generic
volk_32f_log2_32f generic generic
volk_32f_s32f_multiply_32f a_sse u_sse
volk_32f_s32f_add_32f a_sse u_sse
volk_32f_x2_dot_prod_32f a_sse u_sse
";

/// The defaults of VOLK 3.3's `sse4_2_64_mmx` machine for the level `sse4.2`.
const VOLK_SSE42_CONFIG: &str = "\
volk_8i_s32f_convert_32f a_sse4_1 u_sse4_1
volk_16i_s32f_convert_32f a_sse4_1 u_sse4_1
volk_8i_convert_16i a_sse4_1 u_sse4_1
volk_32f_s32f_convert_16i a_sse2 u_sse2
volk_32f_s32f_convert_8i a_sse2 u_sse2
volk_32f_s32f_x2_convert_8u a_sse2 u_sse2
volk_32fc_s32fc_x2_rotator2_32fc generic generic
volk_32fc_32f_dot_prod_32fc a_sse u_sse
volk_32fc_magnitude_squared_32f a_sse3 u_sse3
volk_32fc_magnitude_32f a_sse3 u_sse3
volk_16ic_magnitude_16i a_sse3 generic
volk_32fc_x2_multiply_conjugate_32fc a_sse3 u_sse3
volk_32fc_s32f_atan2_32f generic generic
volk_32fc_32f_multiply_32fc a_sse generic
volk_32f_log2_32f a_sse4_1 u_sse4_1
volk_32f_s32f_multiply_32f a_sse u_sse
volk_32f_s32f_add_32f a_sse u_sse
volk_32f_x2_dot_prod_32f a_sse4_1 u_sse4_1
";

/// The VOLK implementations that need no more than AVX2 and FMA, for the levels `avx2` and
/// `avx2-runtime`. Without this, a runner with AVX-512 makes VOLK dispatch to its AVX-512
/// implementations (`rotator2`, `atan2`, `dot_prod`, the conversions have them) and the AVX2
/// build of our kernels would be compared with a different instruction set than on a machine
/// without AVX-512.
const VOLK_AVX2_CONFIG: &str = "\
volk_8i_s32f_convert_32f a_avx2 u_avx2
volk_16i_s32f_convert_32f a_avx2 u_avx2
volk_8i_convert_16i a_avx2 u_avx2
volk_32f_s32f_convert_16i a_avx2 u_avx2
volk_32f_s32f_convert_8i a_avx2 u_avx2
volk_32f_s32f_x2_convert_8u a_avx2_fma u_avx2_fma
volk_32fc_s32fc_x2_rotator2_32fc a_avx u_avx
volk_32fc_32f_dot_prod_32fc a_avx2_fma u_avx2_fma
volk_32fc_magnitude_squared_32f a_avx u_avx
volk_32fc_magnitude_32f a_avx u_avx
volk_16ic_magnitude_16i a_avx2 u_avx2
volk_32fc_x2_multiply_conjugate_32fc a_avx u_avx
volk_32fc_s32f_atan2_32f a_avx2_fma u_avx2_fma
volk_32fc_32f_multiply_32fc a_avx generic
volk_32f_log2_32f a_avx2_fma u_avx2_fma
volk_32f_s32f_multiply_32f a_avx u_avx
volk_32f_s32f_add_32f a_avx u_avx
volk_32f_x2_dot_prod_32f a_avx2_fma u_avx2_fma
";

/// The default dispatch of VOLK 3.3's `neonv8` machine (what QEMU's CPU selects), pinned so
/// that a change of the machine or of VOLK's ranking shows up as a missing implementation
/// instead of a silently different reference (D11). The aligned dot product of the FIR family
/// is `a_neon`: `neonv8` needs no alignment, and the aligned entry point prefers aligned
/// implementations.
const VOLK_NEON_CONFIG: &str = "\
volk_8i_s32f_convert_32f neonv8 neonv8
volk_16i_s32f_convert_32f neonv8 neonv8
volk_8i_convert_16i neon neon
volk_32f_s32f_convert_16i neonv8 neonv8
volk_32f_s32f_convert_8i neonv8 neonv8
volk_32f_s32f_x2_convert_8u neonv8 neonv8
volk_32fc_s32fc_x2_rotator2_32fc neon neon
volk_32fc_32f_dot_prod_32fc a_neon neonv8
volk_32fc_magnitude_squared_32f neonv8 neonv8
volk_32fc_magnitude_32f neonv8 neonv8
volk_16ic_magnitude_16i neonv8 neonv8
volk_32fc_x2_multiply_conjugate_32fc neonv8 neonv8
volk_32fc_s32f_atan2_32f neonv8 neonv8
volk_32fc_32f_multiply_32fc neonv8 neonv8
volk_32f_log2_32f neonv8 neonv8
volk_32f_s32f_multiply_32f neonv8 neonv8
volk_32f_s32f_add_32f neonv8 neonv8
volk_32f_x2_dot_prod_32f neonv8 neonv8
";

/// The VOLK functions whose names VOLK reads as environment variables to override its
/// dispatch (`volk_rank_archs`); the measured processes run without them and without
/// `VOLK_GENERIC`.
const VOLK_FUNCTIONS: [&str; 18] = [
    "volk_8i_s32f_convert_32f",
    "volk_16i_s32f_convert_32f",
    "volk_8i_convert_16i",
    "volk_32f_s32f_convert_16i",
    "volk_32f_s32f_convert_8i",
    "volk_32f_s32f_x2_convert_8u",
    "volk_32fc_s32fc_x2_rotator2_32fc",
    "volk_32fc_32f_dot_prod_32fc",
    "volk_32fc_magnitude_squared_32f",
    "volk_32fc_magnitude_32f",
    "volk_16ic_magnitude_16i",
    "volk_32fc_x2_multiply_conjugate_32fc",
    "volk_32fc_s32f_atan2_32f",
    "volk_32fc_32f_multiply_32fc",
    "volk_32f_log2_32f",
    "volk_32f_s32f_multiply_32f",
    "volk_32f_s32f_add_32f",
    "volk_32f_x2_dot_prod_32f",
];

/// CPUID of SDE for the level `sse2`: `GenuineIntel`, family 15, SSE2 and nothing above it; XSAVE
/// with x87 and SSE state only. The XSAVE bits matter: on a CPU without OSXSAVE, the
/// `cpu_features` library inside VOLK falls back to the host's `/proc/cpuinfo` for the SSE
/// flags, and VOLK would select a machine the emulated CPU does not have (SDE's own `-p4p`
/// gives the `sse4_1` machine; `-p4` does not run 64-bit code).
const CPUID_SSE2: &str = "\
# CPUID_VERSION = 1.0
# x86-64 with SSE2 and nothing above it, for cargo xtask count.
00000000 ******** => 0000000d 756e6547 6c65746e 49656e69
00000001 ******** => 00000f68 00010800 0c000000 078bfbff
0000000d 00000000 => 00000003 00000240 00000240 00000000
0000000d 00000001 => 00000000 00000000 00000000 00000000
80000000 ******** => 80000001 00000000 00000000 00000000
80000001 ******** => 00000000 00000000 00000001 20100800
";

/// CPUID of SDE for the level `sse4.2`: x86-64-v2 (SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CX16,
/// LAHF), with XSAVE for x87 and SSE state as in [`CPUID_SSE2`].
const CPUID_SSE42: &str = "\
# CPUID_VERSION = 1.0
# x86-64-v2 (up to SSE4.2 and POPCNT), for cargo xtask count.
00000000 ******** => 0000000d 756e6547 6c65746e 49656e69
00000001 ******** => 000106a5 00010800 0c982201 078bfbff
0000000d 00000000 => 00000003 00000240 00000240 00000000
0000000d 00000001 => 00000000 00000000 00000000 00000000
80000000 ******** => 80000001 00000000 00000000 00000000
80000001 ******** => 00000000 00000000 00000001 20100800
";

/// Writes the CPUID definition of the level `sse2` or `sse4.2` for SDE's `-cpuid_in` to `path`
/// (`cargo xtask sde-cpuid`), so that CI runs the tests on the CPU the counts use.
///
/// # Errors
/// Fails on another level or if the file cannot be written.
pub fn write_cpuid(level: &str, path: &Path) -> anyhow::Result<bool> {
    let text = match level {
        "sse2" => CPUID_SSE2,
        "sse4.2" => CPUID_SSE42,
        other => bail!("no CPUID definition for the level `{other}` (sse2 or sse4.2)"),
    };
    std::fs::write(path, text).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(true)
}

// ---------------------------------------------------------------------------------------------
// Counters

/// Extracts the instruction count from the output of a counter.
pub mod parse {
    use anyhow::{anyhow, bail};

    /// The total of a callgrind run: the line `==pid== Collected : N ...` on standard error.
    ///
    /// # Errors
    /// No such line.
    pub fn callgrind(stderr: &str) -> anyhow::Result<u64> {
        for line in stderr.lines() {
            if let Some(rest) = line.split("Collected :").nth(1) {
                let first = rest
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| anyhow!("empty callgrind total"))?;
                return Ok(first.parse()?);
            }
        }
        bail!("no `Collected :` line in the callgrind output")
    }

    /// The total of Intel SDE's `-icount`: the sum of the lines `$$ TID: T ICOUNT: N` (one per
    /// thread) of its `-icount_file`.
    ///
    /// # Errors
    /// No such line.
    pub fn sde_icount(report: &str) -> anyhow::Result<u64> {
        let mut sum = 0u64;
        let mut seen = false;
        for line in report.lines() {
            if line.trim_start().starts_with("$$")
                && let Some(n) = line.split("ICOUNT:").nth(1)
            {
                sum += n.trim().parse::<u64>()?;
                seen = true;
            }
        }
        if seen {
            Ok(sum)
        } else {
            bail!("no `ICOUNT:` line in the SDE output")
        }
    }

    /// The total of the QEMU `insn` plugin: `total insns: N`, or the sum of the lines
    /// `cpu K insns: N`.
    ///
    /// # Errors
    /// Neither form is present.
    pub fn qemu_insn(stderr: &str) -> anyhow::Result<u64> {
        let mut sum = 0u64;
        let mut seen = false;
        for line in stderr.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("total insns:") {
                return Ok(rest.trim().parse()?);
            }
            if line.starts_with("cpu ")
                && let Some(n) = line.split("insns:").nth(1)
            {
                sum += n.trim().parse::<u64>()?;
                seen = true;
            }
        }
        if seen {
            Ok(sum)
        } else {
            bail!("no instruction count in the QEMU plugin output")
        }
    }
}

/// The median of the counts (the upper one of the two middle values for an even number), `None`
/// for no counts.
#[must_use]
pub fn median(mut counts: Vec<u64>) -> Option<u64> {
    counts.sort_unstable();
    counts.get(counts.len() / 2).copied()
}

/// The median of up to [`SDE_RUNS`] results of `run`: when the first two agree, they are the
/// median of any third, so it is not run.
///
/// # Errors
/// An error of `run`.
pub fn median_of_three(mut run: impl FnMut() -> anyhow::Result<u64>) -> anyhow::Result<u64> {
    let mut counts = vec![run()?, run()?];
    if counts.first() != counts.get(1) {
        while counts.len() < SDE_RUNS {
            counts.push(run()?);
        }
    }
    median(counts).ok_or_else(|| anyhow!("no count"))
}

/// The smallest fuel with which `run(fuel)` succeeds, found by bisection. `run` returns `true`
/// when the program completes.
///
/// # Errors
/// The program does not complete with the largest fuel that is tried.
pub fn least_fuel(mut run: impl FnMut(u64) -> anyhow::Result<bool>) -> anyhow::Result<u64> {
    // `lo` is known to fail (0 always does), `hi` to succeed.
    let (mut lo, mut hi) = (0u64, 1u64 << 20);
    while !run(hi)? {
        lo = hi;
        hi = hi
            .checked_mul(4)
            .filter(|h| *h < 1 << 50)
            .ok_or_else(|| anyhow!("the program needs more than 2^50 fuel"))?;
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if run(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}

/// Environment and tools of a counting session.
struct Session {
    plan: LevelPlan,
    binary: PathBuf,
    /// `VOLK_CONFIGPATH` with the pinned `volk_config` of the level.
    volk_config: PathBuf,
    /// The `lib` directory of the C references (VOLK is a shared library there).
    refs_lib: PathBuf,
    /// The CPUID definition of SDE, for the levels that need one.
    cpuid: Option<PathBuf>,
    /// Scratch directory of the level (SDE's count file, the configurations of the
    /// cheapest-VOLK search).
    scratch: PathBuf,
    qemu: Option<(PathBuf, PathBuf, PathBuf)>,
    /// `target/tools` of the work tree (the QEMU plugins).
    tools: PathBuf,
    /// The source of the block-count plugin of QEMU (`xtask/plugins/bbcount.c`).
    bbcount_source: PathBuf,
    /// SDE: the count of the region with `k = 0`, which holds only the markers and is the
    /// same for every kernel, so it is counted once.
    markers: std::cell::OnceCell<u64>,
}

/// The arguments of `count run` with `k` calls, written with as many digits as `calls`: both
/// runs of a measurement then have arguments of the same length. Their length moves the
/// start-up code of the process by a few instructions, and only identical start-ups cancel
/// exactly (otherwise the length of the environment decided the last digit of some counts).
fn run_args(kernel: &str, which: &str, k: u64, calls: u64, variant: &Variant) -> Vec<String> {
    let width = calls.to_string().len();
    let mut args: Vec<String> = ["run", kernel, which, &format!("{k:0width$}")]
        .iter()
        .map(ToString::to_string)
        .collect();
    args.extend(variant.options());
    args
}

/// One run of the measurement binary: the arguments and the `volk_config` directory.
struct Run<'a> {
    args: Vec<String>,
    volk_config: &'a Path,
}

impl Session {
    /// The command for `program` with the environment of the measured process: the pinned
    /// `volk_config`, the library path of the references, and without the variables that
    /// override VOLK's dispatch.
    fn command(&self, program: impl AsRef<std::ffi::OsStr>, run: &Run<'_>) -> Command {
        let mut cmd = Command::new(program);
        cmd.env("VOLK_CONFIGPATH", run.volk_config)
            .env_remove("VOLK_GENERIC");
        for f in VOLK_FUNCTIONS {
            cmd.env_remove(f);
        }
        if matches!(self.plan.counter, Counter::Callgrind | Counter::Sde(_)) {
            cmd.env("LD_LIBRARY_PATH", &self.refs_lib);
        }
        if matches!(self.plan.counter, Counter::Qemu) {
            // qemu-user reads every `QEMU_<OPTION>` variable as a command-line option
            // (`QEMU_VERSION` is `-version`, which prints the version and runs nothing), so
            // none may reach it from the caller's environment.
            for (name, _) in std::env::vars_os() {
                if is_qemu_option_variable(&name) {
                    cmd.env_remove(name);
                }
            }
        }
        cmd
    }

    /// The emulator in front of the binary: program and arguments.
    fn launcher(&self, counting: bool) -> anyhow::Result<(PathBuf, Vec<std::ffi::OsString>)> {
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        let program = match self.plan.counter {
            Counter::Callgrind => {
                if counting {
                    args.extend(
                        ["--tool=callgrind", "--callgrind-out-file=/dev/null"].map(Into::into),
                    );
                    PathBuf::from("valgrind")
                } else {
                    return Ok((self.binary.clone(), Vec::new()));
                }
            }
            Counter::Sde(_) => {
                args.extend(self.sde_cpu_args()?);
                if counting {
                    args.extend(SDE_REGION.map(Into::into));
                    args.extend(["-icount", "-icount_file"].map(Into::into));
                    args.push(self.scratch.join("sde-icount.txt").into());
                }
                args.push("--".into());
                sde()
            }
            Counter::Qemu => {
                let (qemu, plugin, sysroot) = self.qemu.clone().ok_or_else(|| anyhow!("qemu"))?;
                args.extend(["-R", QEMU_RESERVED_VA, "-L"].map(Into::into));
                args.push(sysroot.into());
                args.push("-E".into());
                args.push(format!("LD_LIBRARY_PATH={}", self.refs_lib.display()).into());
                if counting {
                    args.push("-plugin".into());
                    args.push(plugin.into());
                    args.extend(["-d", "plugin"].map(Into::into));
                }
                qemu
            }
            Counter::Wasmtime => {
                args.push("run".into());
                PathBuf::from("wasmtime")
            }
        };
        args.push(self.binary.clone().into());
        Ok((program, args))
    }

    /// The CPU options of SDE for this level: `-spr` for `avx512`, the CPUID definition of
    /// `sse2` and `sse4.2` with its check, and `-hsw` for a level counted with callgrind (the
    /// validation of the SDE profile).
    fn sde_cpu_args(&self) -> anyhow::Result<Vec<std::ffi::OsString>> {
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        match (self.plan.counter, &self.cpuid) {
            (Counter::Sde(SdeCpu::SapphireRapids), _) => args.push("-spr".into()),
            (Counter::Sde(_), Some(def)) => {
                args.push("-cpuid_in".into());
                args.push(def.into());
                // Any instruction above the CPUID stops the run: no library may use
                // code of a higher level than the one it is compared at.
                args.extend(["-chip_check_cpuid", "1"].map(Into::into));
            }
            (Counter::Sde(cpu), None) => bail!("no CPUID definition for {cpu:?}"),
            (Counter::Callgrind, _) => args.push("-hsw".into()),
            (counter, _) => bail!(
                "the level {} counts with {counter:?}, not on x86",
                self.plan.level
            ),
        }
        Ok(args)
    }

    /// Runs the measurement binary without a counter and returns its standard output.
    fn plain(&self, run: &Run<'_>) -> anyhow::Result<String> {
        let (program, args) = self.launcher(false)?;
        let out = self
            .command(program, run)
            .args(args)
            .args(&run.args)
            .stderr(Stdio::inherit())
            .output()
            .context("cannot run the measurement binary")?;
        if !out.status.success() {
            bail!("the measurement binary failed: {:?}", run.args);
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// Instructions of one run.
    fn count(&self, run: &Run<'_>) -> anyhow::Result<u64> {
        match self.plan.counter {
            Counter::Callgrind | Counter::Qemu => {
                let (program, args) = self.launcher(true)?;
                let out = self
                    .command(program, run)
                    .args(args)
                    .args(&run.args)
                    .output()
                    .context("cannot run the counter")?;
                let stderr = String::from_utf8_lossy(&out.stderr);
                if !out.status.success() {
                    bail!("the counter failed for {:?}: {stderr}", run.args);
                }
                if self.plan.counter == Counter::Qemu {
                    parse::qemu_insn(&stderr)
                } else {
                    parse::callgrind(&stderr)
                }
            }
            Counter::Sde(_) => {
                let (program, args) = self.launcher(true)?;
                let file = self.scratch.join("sde-icount.txt");
                median_of_three(|| {
                    // A run that writes no count must not leave the previous one behind.
                    match std::fs::remove_file(&file) {
                        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                            return Err(e).context("cannot remove SDE's count file");
                        }
                        _ => {}
                    }
                    let out = self
                        .command(&program, run)
                        .args(&args)
                        .args(&run.args)
                        .stdout(Stdio::null())
                        .output()
                        .context("cannot run sde64")?;
                    if !out.status.success() {
                        bail!(
                            "sde64 failed for {:?}: {}",
                            run.args,
                            String::from_utf8_lossy(&out.stderr)
                        );
                    }
                    parse::sde_icount(
                        &std::fs::read_to_string(&file).context("SDE wrote no count file")?,
                    )
                })
            }
            Counter::Wasmtime => least_fuel(|fuel| {
                let status = self
                    .command("wasmtime", run)
                    .arg("run")
                    .arg("-W")
                    .arg(format!("fuel={fuel}"))
                    .arg(&self.binary)
                    .args(&run.args)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .context("cannot run wasmtime")?;
                Ok(status.success())
            }),
        }
    }

    /// Instructions per sample of `which` on `kernel`: `(count(calls) - count(0)) / (calls *
    /// samples)`, with `calls` such that every measurement covers `K * N` samples. Under SDE
    /// `count(0)` is the markers alone and is counted once per level.
    fn per_sample(
        &self,
        kernel: &str,
        which: &str,
        variant: &Variant,
        volk_config: &Path,
    ) -> anyhow::Result<f64> {
        let calls = K * (N / variant.samples).max(1);
        let empty = || {
            self.count(&Run {
                args: run_args(kernel, which, 0, calls, variant),
                volk_config,
            })
        };
        let base = match (self.plan.counter, self.markers.get()) {
            (Counter::Sde(_), Some(b)) => *b,
            (Counter::Sde(_), None) => {
                let b = empty()?;
                let _ = self.markers.set(b);
                b
            }
            _ => empty()?,
        };
        let full = self.count(&Run {
            args: run_args(kernel, which, calls, calls, variant),
            volk_config,
        })?;
        let diff = full.checked_sub(base).ok_or_else(|| {
            anyhow!(
                "{kernel} {which}: the run with k = {calls} counted fewer instructions than k = 0"
            )
        })?;
        Ok(diff as f64 / (calls * variant.samples) as f64)
    }

    /// The SNR of one run of `which`, against the f64 reference.
    fn snr(&self, kernel: &str, which: &str, volk_config: &Path) -> anyhow::Result<f64> {
        let out = self.plain(&Run {
            args: ["accuracy", kernel, which]
                .map(ToString::to_string)
                .to_vec(),
            volk_config,
        })?;
        let v: Value = serde_json::from_str(out.trim()).context("accuracy output")?;
        v.get("snr_db")
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow!("no snr_db in {out}"))
    }

    /// The execution counts per instruction of `which` on `kernel` from `profiler`, for level 2
    /// of the measurement policy: the counts of a run with `calls` calls minus those of a run
    /// with none, so that only the counted calls remain (the differential method of
    /// [`Session::per_sample`], per instruction). The runs have the environment of the count
    /// of the level ([`Session::command`]: the pinned `volk_config`, the references; the CPU of
    /// SDE, the QEMU options).
    fn profile(
        &self,
        kernel: &str,
        which: &str,
        variant: &Variant,
        profiler: Profiler,
    ) -> anyhow::Result<crate::mca::Counts> {
        let calls = K * (N / variant.samples).max(1);
        match profiler {
            Profiler::Callgrind => self.callgrind_profile(kernel, which, variant),
            Profiler::Sde => {
                let full = self.sde_blocks(kernel, which, calls, calls, variant)?;
                let empty = self.sde_blocks(kernel, which, 0, calls, variant)?;
                crate::blockcount::expand(&crate::blockcount::difference(&full, &empty))
            }
            Profiler::Qemu => {
                let full = self.qemu_blocks(kernel, which, calls, calls, variant)?;
                let empty = self.qemu_blocks(kernel, which, 0, calls, variant)?;
                Ok(crate::mca::difference(&full, &empty))
            }
        }
    }

    /// The block counts of one run under SDE `-dcfg` (every image of the process, between the
    /// SSC markers of the counted calls).
    fn sde_blocks(
        &self,
        kernel: &str,
        which: &str,
        k: u64,
        calls: u64,
        variant: &Variant,
    ) -> anyhow::Result<crate::blockcount::BlockCounts> {
        let base = self.scratch.join(format!("dcfg-{kernel}-{which}-{k}"));
        let mut file = base.clone().into_os_string();
        file.push(".dcfg.json");
        let file = PathBuf::from(file);
        // SDE would read an existing DCFG of the same name as the basis of the new one.
        match std::fs::remove_file(&file) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(e).context("cannot remove an old DCFG");
            }
            _ => {}
        }
        let run = Run {
            args: run_args(kernel, which, k, calls, variant),
            volk_config: &self.volk_config,
        };
        let out = self
            .command(sde(), &run)
            .args(self.sde_cpu_args()?)
            // Only the region between the SSC markers, as in the count: outside it the
            // process does work that depends on the number of calls (the check of the
            // output), which the count leaves out.
            .args(SDE_REGION)
            .args(["-dcfg", "-dcfg:out_base_name"])
            .arg(&base)
            .args([
                "-dcfg:dcfg_file_suffix",
                ".dcfg.json",
                "-dcfg:read_dcfg_if_avail",
                "0",
                "-dcfg:analysis",
                "0",
                "-dcfg:source",
                "0",
                "-dcfg:symbols",
                "0",
                "--",
            ])
            .arg(&self.binary)
            .args(&run.args)
            .stdout(Stdio::null())
            .output()
            .context("cannot run sde64")?;
        if !out.status.success() {
            bail!(
                "sde64 -dcfg failed for {:?}: {}",
                run.args,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("SDE wrote no DCFG {}", file.display()))?;
        let blocks = crate::blockcount::parse_dcfg(&text, |object| {
            crate::blockcount::elf(object).map_or(0, |e| e.link_base())
        })?;
        let _ = std::fs::remove_file(&file);
        Ok(blocks)
    }

    /// The instruction counts of one run under QEMU with the plugin `bbcount`.
    fn qemu_blocks(
        &self,
        kernel: &str,
        which: &str,
        k: u64,
        calls: u64,
        variant: &Variant,
    ) -> anyhow::Result<crate::mca::Counts> {
        let (qemu, _, sysroot) = self
            .qemu
            .clone()
            .ok_or_else(|| anyhow!("the level {} is not counted under QEMU", self.plan.level))?;
        let plugin = bbcount_plugin(&self.tools, &self.bbcount_source)?;
        let log = self
            .scratch
            .join(format!("bbcount-{kernel}-{which}-{k}.txt"));
        let run = Run {
            args: run_args(kernel, which, k, calls, variant),
            volk_config: &self.volk_config,
        };
        let out = self
            .command(&qemu, &run)
            .args(["-R", QEMU_RESERVED_VA, "-L"])
            .arg(&sysroot)
            .arg("-E")
            .arg(format!("LD_LIBRARY_PATH={}", self.refs_lib.display()))
            .arg("-plugin")
            .arg(&plugin)
            .args(["-d", "plugin", "-D"])
            .arg(&log)
            .arg(&self.binary)
            .args(&run.args)
            .stdout(Stdio::null())
            .output()
            .context("cannot run qemu-aarch64")?;
        if !out.status.success() {
            bail!(
                "qemu-aarch64 with bbcount failed for {:?}: {}",
                run.args,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let text = std::fs::read_to_string(&log)
            .with_context(|| format!("the bbcount plugin wrote no {}", log.display()))?;
        let counts = crate::blockcount::parse_bbcount(&text, |object, offset| {
            crate::blockcount::elf(object)
                .ok()?
                .address_of_offset(offset)
        })?;
        let _ = std::fs::remove_file(&log);
        Ok(counts)
    }

    /// [`Session::profile`] under callgrind.
    fn callgrind_profile(
        &self,
        kernel: &str,
        which: &str,
        variant: &Variant,
    ) -> anyhow::Result<crate::mca::Counts> {
        let calls = K * (N / variant.samples).max(1);
        let mut runs = Vec::new();
        for k in [0, calls] {
            let file = self
                .scratch
                .join(format!("mca-{kernel}-{which}-{k}.callgrind"));
            let run = Run {
                args: run_args(kernel, which, k, calls, variant),
                volk_config: &self.volk_config,
            };
            let out = self
                .command("valgrind", &run)
                .args([
                    "-q",
                    "--tool=callgrind",
                    "--dump-instr=yes",
                    "--collect-jumps=yes",
                    "--compress-pos=no",
                    "--compress-strings=no",
                ])
                .arg(format!("--callgrind-out-file={}", file.display()))
                .arg(&self.binary)
                .args(&run.args)
                .stdout(Stdio::null())
                .output()
                .context("cannot run valgrind")?;
            if !out.status.success() {
                bail!(
                    "callgrind failed for {:?}: {}",
                    run.args,
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("cannot read {}", file.display()))?;
            runs.push(crate::mca::parse_callgrind(&text));
        }
        let [empty, full] = <[crate::mca::Counts; 2]>::try_from(runs)
            .map_err(|_| anyhow!("two callgrind runs expected"))?;
        Ok(crate::mca::difference(&full, &empty))
    }

    /// A JSON line of the measurement binary (`info`, `volk-info`, `volk-impls`).
    fn query(&self, args: &[&str]) -> anyhow::Result<Value> {
        let out = self.plain(&Run {
            args: args.iter().map(ToString::to_string).collect(),
            volk_config: &self.volk_config,
        })?;
        serde_json::from_str(out.trim()).with_context(|| format!("output of {args:?}: {out}"))
    }
}

/// The buffers of one measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Variant {
    samples: u64,
    misaligned: bool,
    /// Whether `--n` is passed (the FFT sizes are fixed by the kernel).
    explicit_n: bool,
}

impl Variant {
    fn standard(spec: &KernelSpec) -> Self {
        Variant {
            samples: spec.samples,
            misaligned: false,
            explicit_n: false,
        }
    }

    fn options(&self) -> Vec<String> {
        let mut o = Vec::new();
        if self.explicit_n {
            o.extend(["--n".to_string(), self.samples.to_string()]);
        }
        if self.misaligned {
            o.push("--misaligned".to_string());
        }
        o
    }

    fn label(&self) -> String {
        match (self.misaligned, self.explicit_n) {
            (true, _) => "misaligned by one sample".to_string(),
            (false, true) => format!("n = {}", self.samples),
            (false, false) => String::new(),
        }
    }
}

/// Whether `name` is a variable that qemu-user would read as one of its options.
fn is_qemu_option_variable(name: &std::ffi::OsStr) -> bool {
    name.to_str().is_some_and(|n| n.starts_with("QEMU_"))
}

fn sde() -> PathBuf {
    std::env::var_os("SDRK_SDE").map_or_else(|| PathBuf::from("sde64"), PathBuf::from)
}

/// The tools of the QEMU counter: the QEMU binary with plugins, the built `insn` plugin and the
/// sysroot of the target libc. See `docs/performance.md` for how to set them up.
fn qemu_tools(root: &Path) -> anyhow::Result<(PathBuf, PathBuf, PathBuf)> {
    let qemu = std::env::var_os("SDRK_QEMU_AARCH64")
        .map_or_else(|| PathBuf::from("qemu-aarch64"), PathBuf::from);
    let sysroot = std::env::var_os("SDRK_AARCH64_SYSROOT")
        .map_or_else(|| PathBuf::from("/usr/aarch64-linux-gnu"), PathBuf::from);
    let tools = root.join("target").join("tools");
    let plugin = tools.join("libinsn.so");
    if !plugin.exists() {
        build_insn_plugin(&tools, &plugin)?;
    }
    Ok((qemu, plugin, sysroot))
}

/// Builds the `insn` plugin of the QEMU tree into `target/tools/`. The source is taken from the
/// tree named by `SDRK_QEMU_SRC` (needs `include/qemu/qemu-plugin.h` and
/// `tests/tcg/plugins/insn.c`; the location has moved between QEMU releases) or downloaded
/// from the QEMU repository at the tag `SDRK_QEMU_TAG` (default `v10.2.1`). Needs a C compiler
/// and the glib development files (`libglib2.0-dev`, `pkg-config`).
pub(crate) fn build_insn_plugin(tools: &Path, plugin: &Path) -> anyhow::Result<()> {
    let src_dir = qemu_source(
        tools,
        &["tests/tcg/plugins/insn.c", "include/qemu/qemu-plugin.h"],
        true,
    )?;
    compile_plugin(&src_dir, &src_dir.join("tests/tcg/plugins/insn.c"), plugin)
}

/// The block-count plugin of QEMU (`xtask/plugins/bbcount.c`) built into `tools`, against the
/// `qemu-plugin.h` of the QEMU tree of [`build_insn_plugin`]. It is rebuilt when the source
/// changes (its SHA-256 is kept next to the library).
///
/// # Errors
/// The source cannot be read, the header cannot be fetched or the compiler fails.
pub(crate) fn bbcount_plugin(tools: &Path, source: &Path) -> anyhow::Result<PathBuf> {
    use sha2::Digest as _;
    let plugin = tools.join("libbbcount.so");
    let stamp = tools.join("libbbcount.so.sha256");
    let text =
        std::fs::read(source).with_context(|| format!("cannot read {}", source.display()))?;
    let want = format!("{:x}", sha2::Sha256::digest(&text));
    if plugin.exists() && std::fs::read_to_string(&stamp).is_ok_and(|s| s.trim() == want) {
        return Ok(plugin);
    }
    let src_dir = qemu_source(tools, &["include/qemu/qemu-plugin.h"], false)?;
    compile_plugin(&src_dir, source, &plugin)?;
    std::fs::write(&stamp, want)?;
    Ok(plugin)
}

/// The QEMU source tree with `files`: `SDRK_QEMU_SRC`, or the files downloaded into
/// `tools/qemu-src` from the tag `SDRK_QEMU_TAG` (default `v10.2.1`, the QEMU of the
/// development container); with `refresh` they are downloaded even if present.
fn qemu_source(tools: &Path, files: &[&str], refresh: bool) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(tools)?;
    if let Some(dir) = std::env::var_os("SDRK_QEMU_SRC") {
        Ok(PathBuf::from(dir))
    } else {
        let tag = std::env::var("SDRK_QEMU_TAG").unwrap_or_else(|_| "v10.2.1".to_string());
        let dir = tools.join("qemu-src");
        for file in files {
            let dest = dir.join(file);
            if !refresh && dest.exists() {
                continue;
            }
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let url = format!("https://raw.githubusercontent.com/qemu/qemu/{tag}/{file}");
            let status = Command::new("curl")
                .args(["-sfL", &url, "-o"])
                .arg(&dest)
                .status()
                .context("cannot run curl to fetch the QEMU plugin source")?;
            if !status.success() {
                bail!("cannot download {url}; set SDRK_QEMU_SRC to a QEMU source tree");
            }
        }
        Ok(dir)
    }
}

/// Compiles the QEMU plugin `source` into `plugin` against the headers of `src_dir`. Needs a C
/// compiler and the glib development files (`libglib2.0-dev`, `pkg-config`).
fn compile_plugin(src_dir: &Path, source: &Path, plugin: &Path) -> anyhow::Result<()> {
    let glib = Command::new("pkg-config")
        .args(["--cflags", "glib-2.0"])
        .output()
        .context("pkg-config is needed to build the QEMU plugin")?;
    if !glib.status.success() {
        bail!("pkg-config cannot find glib-2.0 (install libglib2.0-dev)");
    }
    let cflags = String::from_utf8_lossy(&glib.stdout).into_owned();
    let status = Command::new("gcc")
        .args(["-shared", "-fPIC", "-O2", "-o"])
        .arg(plugin)
        .arg(source)
        .arg(format!("-I{}", src_dir.join("include/qemu").display()))
        .arg(format!("-I{}", src_dir.join("include").display()))
        .args(cflags.split_whitespace())
        .status()
        .context("cannot run gcc")?;
    if !status.success() {
        bail!("building the QEMU plugin {} failed", source.display());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Driver

/// Builds the C references of `platform` (a no-op when they are up to date) and returns their
/// prefix.
pub(crate) fn references(root: &Path, platform: &str) -> anyhow::Result<PathBuf> {
    let prefix = root
        .join("benches")
        .join("target")
        .join("refs")
        .join(platform);
    let status = Command::new("bash")
        .arg(root.join("benches").join("refs").join("build.sh"))
        .arg(platform)
        .arg(&prefix)
        .status()
        .context("cannot run bash for benches/refs/build.sh")?;
    if !status.success() {
        bail!("building the C references for {platform} failed (benches/refs/build.sh)");
    }
    Ok(prefix)
}

/// The flags of a measurement build for `CARGO_ENCODED_RUSTFLAGS`: `flags` plus a remapping of
/// the paths of the work tree and of the Cargo home. Without it these paths end up in the
/// panic locations of the binary, and their length shifts its data and with it the counts (by
/// 0.01 instructions per sample, between two checkouts at different paths). The encoded form
/// keeps paths with spaces in one argument.
fn rustflags(flags: &str, root: &Path) -> std::ffi::OsString {
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".cargo")));
    encoded_rustflags(flags, root, cargo_home.as_deref())
}

/// [`rustflags`] with the Cargo home given.
fn encoded_rustflags(flags: &str, root: &Path, cargo_home: Option<&Path>) -> std::ffi::OsString {
    let mut args: Vec<std::ffi::OsString> = flags.split_whitespace().map(Into::into).collect();
    let remaps = [(Some(root), "/sdrk"), (cargo_home, "/cargo")];
    for (from, to) in remaps {
        if let Some(from) = from {
            let mut arg = std::ffi::OsString::from("--remap-path-prefix=");
            arg.push(from);
            arg.push("=");
            arg.push(to);
            args.push(arg);
        }
    }
    args.join(std::ffi::OsStr::new("\u{1f}"))
}

fn build(root: &Path, plan: &LevelPlan, refs: &Path) -> anyhow::Result<PathBuf> {
    let target_dir = root.join("benches").join("target").join(plan.level);
    let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cmd.current_dir(root.join("benches"))
        .args(["build", "--release", "-p", "kernels", "--target-dir"])
        .arg(&target_dir);
    if let Some(target) = plan.target {
        cmd.args(["--target", target]);
    }
    if plan.runtime_detect {
        cmd.args(["--features", "std"]);
    }
    // Only the references of benches/refs/build.sh, never a VOLK of the system: the
    // distribution's libvolk is built differently (-O2, LTO, ORC) and gives other counts.
    cmd.env_remove("RUSTFLAGS")
        .env("CARGO_ENCODED_RUSTFLAGS", rustflags(plan.rustflags, root))
        .env("SDRK_REFS_DIR", refs)
        .env("PKG_CONFIG_PATH", refs.join("lib").join("pkgconfig"))
        .env_remove("PKG_CONFIG_LIBDIR")
        .env_remove("PKG_CONFIG_SYSROOT_DIR");
    if plan.target.is_some() {
        cmd.env("PKG_CONFIG_ALLOW_CROSS", "1");
    }
    if plan.target == Some("aarch64-unknown-linux-gnu") {
        cmd.env(
            "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER",
            "aarch64-linux-gnu-gcc",
        );
    }
    println!(
        "==> building the measurement binary for {}/{}",
        plan.platform, plan.level
    );
    let status = cmd.status().context("cannot run cargo")?;
    if !status.success() {
        bail!("building benches/kernels failed");
    }
    let mut path = target_dir;
    if let Some(target) = plan.target {
        path.push(target);
    }
    path.push("release");
    path.push(if plan.target.is_some_and(|t| t.starts_with("wasm")) {
        "count.wasm"
    } else {
        "count"
    });
    Ok(path)
}

/// Writes `text` to `dir/volk/volk_config` and returns `dir` (the value of `VOLK_CONFIGPATH`).
fn volk_config(dir: PathBuf, text: &str) -> anyhow::Result<PathBuf> {
    let file = dir.join("volk").join("volk_config");
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&file, text)?;
    Ok(dir)
}

/// What one level measured, with the facts about its references for the results file.
struct LevelResult {
    measurements: Vec<Measurement>,
    volk: Value,
}

/// Our numbers of a level for the levels judged against it: kernel -> (instr/sample, SNR).
type OwnNumbers = BTreeMap<String, (f64, f64)>;

#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the Windows file system calls are large; xtask is a host tool"
)]
fn measure_level(
    root: &Path,
    plan: &LevelPlan,
    refs: &Path,
    only: &[String],
    variants: bool,
    own: &BTreeMap<&'static str, OwnNumbers>,
) -> anyhow::Result<LevelResult> {
    let session = session(root, plan, refs)?;
    let volk = session.query(&["volk-info"])?;
    let mut listed: Vec<String> = session
        .plain(&Run {
            args: vec!["list".to_string()],
            volk_config: &session.volk_config,
        })?
        .lines()
        .map(str::to_string)
        .collect();
    listed.sort_unstable();
    let mut known: Vec<&str> = KERNELS.iter().map(|k| k.name).collect();
    known.sort_unstable();
    if listed != known {
        bail!("the measurement binary lists {listed:?}, xtask knows {known:?}");
    }
    let mut out = Vec::new();
    for spec in &KERNELS {
        if !only.is_empty() && !only.iter().any(|k| k == spec.name) {
            continue;
        }
        if !spec.family.measured_at(plan) {
            continue;
        }
        let m = measure_kernel(&session, spec, variants, own)?;
        out.push(m);
    }
    Ok(LevelResult {
        measurements: out,
        volk,
    })
}

/// Builds the measurement binary of `plan` and sets up its counting session.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the Windows file system calls are large; xtask is a host tool"
)]
fn session(root: &Path, plan: &LevelPlan, refs: &Path) -> anyhow::Result<Session> {
    let binary = build(root, plan, refs)?;
    let scratch = root.join("benches").join("target").join(plan.level);
    std::fs::create_dir_all(&scratch)?;
    let cpuid = match plan.counter {
        Counter::Sde(SdeCpu::Sse2) => Some((CPUID_SSE2, "cpuid-sse2.def")),
        Counter::Sde(SdeCpu::Sse42) => Some((CPUID_SSE42, "cpuid-sse4.2.def")),
        _ => None,
    }
    .map(|(text, name)| -> anyhow::Result<PathBuf> {
        let path = scratch.join(name);
        std::fs::write(&path, text)?;
        Ok(path)
    })
    .transpose()?;
    let session = Session {
        plan: *plan,
        binary,
        volk_config: volk_config(scratch.join("volk-pinned"), plan.volk_pin.config())?,
        refs_lib: refs.join("lib"),
        cpuid,
        scratch: scratch.clone(),
        qemu: if plan.counter == Counter::Qemu {
            Some(qemu_tools(root)?)
        } else {
            None
        },
        tools: root.join("target").join("tools"),
        bbcount_source: root.join("xtask").join("plugins").join("bbcount.c"),
        markers: std::cell::OnceCell::new(),
    };
    Ok(session)
}

#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command is large; xtask is a host tool"
)]
fn measure_kernel(
    session: &Session,
    spec: &KernelSpec,
    variants: bool,
    own: &BTreeMap<&'static str, OwnNumbers>,
) -> anyhow::Result<Measurement> {
    let plan = &session.plan;
    let (gate, extras) = comparisons(plan, spec);
    let info = session.query(&["info", spec.name])?;
    let impls = info.get("impls").and_then(Value::as_object);
    let describe = |r: Reference| -> Option<String> {
        if let Reference::OwnLevel(level) = r {
            return Some(format!("ours at {level}"));
        }
        impls
            .and_then(|i| i.get(r.implementation()))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let label = describe(gate).ok_or_else(|| {
        anyhow!(
            "{} at {}: the comparison `{}` is not in the measurement binary (missing C reference?)",
            spec.name,
            plan.level,
            gate.implementation()
        )
    })?;
    println!("==> {} {} (ours vs {label})", plan.level, spec.name);
    let pinned = session.volk_config.as_path();
    let standard = Variant::standard(spec);
    // The in-place FFTs restore their input before every call, on both sides.
    let restore = if spec.family.is_fft() {
        session.per_sample(spec.name, "restore", &standard, pinned)?
    } else {
        0.0
    };
    let count = |which: &str, variant: &Variant, config: &Path| -> anyhow::Result<f64> {
        Ok(session.per_sample(spec.name, which, variant, config)? - restore)
    };
    let measured = spec.family.measured();
    let ours = count(measured, &standard, pinned)?;
    let ours_snr_db = session.snr(spec.name, measured, pinned)?;
    let (other, other_snr_db) = match gate {
        Reference::OwnLevel(level) => own
            .get(level)
            .and_then(|k| k.get(spec.name))
            .copied()
            .ok_or_else(|| anyhow!("{}: no numbers of the level {level}", spec.name))?,
        r => (
            count(r.implementation(), &standard, pinned)?,
            session.snr(spec.name, r.implementation(), pinned)?,
        ),
    };
    let mut m = Measurement {
        kernel: spec.name.to_string(),
        level: plan.level.to_string(),
        ours,
        other,
        compared_with: label,
        ours_snr_db,
        other_snr_db,
        extras: Vec::new(),
    };
    for r in extras {
        if r == Reference::VolkCheapest {
            m.extras.push(cheapest_volk(session, spec, ours, &count)?);
            continue;
        }
        if let Reference::OwnKernel(kernel) = r {
            m.extras
                .push(own_kernel_extra(session, kernel, ours, &standard, pinned)?);
            continue;
        }
        let Some(name) = describe(r) else {
            println!("    ({} is not in this build)", r.implementation());
            continue;
        };
        m.extras.push(Extra {
            label: format!("information: {}", r.implementation()),
            ours,
            other: count(r.implementation(), &standard, pinned)?,
            compared_with: name,
            other_snr_db: Some(session.snr(spec.name, r.implementation(), pinned)?),
        });
    }
    if variants && plan.baseline.is_none() {
        for v in buffer_variants(spec, &standard) {
            m.extras.push(Extra {
                label: v.label(),
                ours: count(measured, &v, pinned)?,
                other: count(gate.implementation(), &v, pinned)?,
                compared_with: m.compared_with.clone(),
                other_snr_db: None,
            });
        }
    }
    Ok(m)
}

/// The informational variants of the buffers of an entry: the small blocks and the misaligned
/// buffers, where the entry has them.
fn buffer_variants(spec: &KernelSpec, standard: &Variant) -> Vec<Variant> {
    let mut list = Vec::new();
    if spec.small_n {
        list.extend(SMALL_N.map(|n| Variant {
            samples: n,
            misaligned: false,
            explicit_n: true,
        }));
    }
    if spec.misaligned {
        list.push(Variant {
            misaligned: true,
            ..standard.clone()
        });
    }
    list
}

/// Our kernel on the workload of another entry of the table (`ours` of `kernel`, per sample of
/// its own input), as information next to `ours` of the entry (D63).
fn own_kernel_extra(
    session: &Session,
    kernel: &str,
    ours: f64,
    variant: &Variant,
    config: &Path,
) -> anyhow::Result<Extra> {
    Ok(Extra {
        label: format!("information: ours:{kernel}"),
        ours,
        other: session.per_sample(kernel, "ours", variant, config)?,
        compared_with: format!("ours:{kernel}"),
        other_snr_db: Some(session.snr(kernel, "ours", config)?),
    })
}

/// The cheapest combination of VOLK implementations for a kernel (D11: reported next to the
/// pinned default dispatch, and the goal). Every implementation of the machine is tried for
/// each VOLK function of the kernel, with a `volk_config` that pins it.
fn cheapest_volk(
    session: &Session,
    spec: &KernelSpec,
    ours: f64,
    count: &dyn Fn(&str, &Variant, &Path) -> anyhow::Result<f64>,
) -> anyhow::Result<Extra> {
    let found = session.query(&["volk-impls", spec.name])?;
    let functions: Vec<(String, Vec<String>)> = found
        .get("functions")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|f| {
                    let name = f.get("name")?.as_str()?.to_string();
                    let impls = f
                        .get("impls")?
                        .as_array()?
                        .iter()
                        .filter_map(|i| i.as_str().map(str::to_string))
                        .collect();
                    Some((name, impls))
                })
                .collect()
        })
        .unwrap_or_default();
    // Every combination: the product of the implementation lists (at most a few dozen).
    let mut combinations: Vec<Vec<(String, String)>> = vec![Vec::new()];
    for (function, impls) in &functions {
        combinations = combinations
            .into_iter()
            .flat_map(|c| {
                impls.iter().map(move |i| {
                    let mut next = c.clone();
                    next.push((function.clone(), i.clone()));
                    next
                })
            })
            .collect();
    }
    let dir = session.scratch.join("volk-cheapest");
    let standard = Variant::standard(spec);
    let mut best: Option<(f64, String, PathBuf)> = None;
    for combination in combinations {
        let mut text = String::new();
        for (function, implementation) in &combination {
            let _ = writeln!(text, "{function} {implementation} {implementation}");
        }
        let config = volk_config(dir.clone(), &text)?;
        let n = count("volk", &standard, &config)?;
        if best.as_ref().is_none_or(|(b, _, _)| n < *b) {
            let label = combination
                .iter()
                .map(|(f, i)| format!("{f}: {i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let keep = session.scratch.join("volk-cheapest-best");
            let _ = std::fs::remove_dir_all(&keep);
            best = Some((n, label, volk_config(keep, &text)?));
        }
    }
    let (other, label, config) =
        best.ok_or_else(|| anyhow!("{}: VOLK lists no implementation", spec.name))?;
    Ok(Extra {
        label: "goal: cheapest VOLK implementation".to_string(),
        ours,
        other,
        compared_with: label,
        other_snr_db: Some(session.snr(spec.name, "volk", &config)?),
    })
}

/// `cargo xtask count --pipeline`: instructions per input sample of `fm_receiver.json` and of
/// the SSB chain (`ddc`, `agc`, `ssb_demod`) at 2.4 MHz (callgrind, x86-64 with AVX2) and the
/// latency of the audio output, written to
/// `benches/results/pipeline.json`.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command is large; xtask is a host tool"
)]
fn pipeline(root: &Path, write: bool) -> anyhow::Result<bool> {
    let target_dir = root.join("benches").join("target").join("pipeline");
    println!("==> building the pipeline benchmark");
    let status = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(root.join("benches"))
        .args(["build", "--release", "-p", "pipeline-bench", "--target-dir"])
        .arg(&target_dir)
        .env_remove("RUSTFLAGS")
        .env(
            "CARGO_ENCODED_RUSTFLAGS",
            rustflags("-C target-cpu=x86-64-v3", root),
        )
        .status()
        .context("cannot run cargo")?;
    if !status.success() {
        bail!("building benches/pipeline failed");
    }
    let binary = target_dir.join("release").join("pipeline");
    let blocks = K;
    // One chain: the differential count and the latency line of the benchmark binary.
    let measure = |chain: Option<&str>| -> anyhow::Result<(f64, Value)> {
        let count = |k: u64| -> anyhow::Result<u64> {
            // As in `run_args`: both runs with arguments of the same length.
            let width = blocks.to_string().len();
            let out = Command::new("valgrind")
                .args(["--tool=callgrind", "--callgrind-out-file=/dev/null"])
                .arg(&binary)
                .args(["run", &format!("{k:0width$}")])
                .args(chain)
                .output()
                .context("cannot run valgrind")?;
            if !out.status.success() {
                bail!("valgrind failed: {}", String::from_utf8_lossy(&out.stderr));
            }
            parse::callgrind(&String::from_utf8_lossy(&out.stderr))
        };
        let diff = count(blocks)?.checked_sub(count(0)?).ok_or_else(|| {
            anyhow!("the pipeline run with more blocks counted fewer instructions")
        })?;
        let per_sample = diff as f64 / (blocks * N) as f64;
        let info = Command::new(&binary)
            .arg("info")
            .args(chain)
            .output()
            .context("cannot run the pipeline benchmark")?;
        let latency: Value = serde_json::from_slice(&info.stdout).context("latency output")?;
        Ok((per_sample, latency))
    };
    let (per_sample, latency) = measure(None)?;
    println!(
        "pipeline fm_receiver.json at 2.4 MHz: {per_sample:.1} instructions per input sample, latency {} samples = {} us",
        latency.get("latency_samples").unwrap_or(&Value::Null),
        latency.get("latency_us").unwrap_or(&Value::Null)
    );
    let (ssb_per_sample, ssb_latency) = measure(Some("ssb"))?;
    println!(
        "pipeline ssb chain (ddc, agc, ssb_demod) at 2.4 MHz: {ssb_per_sample:.1} instructions per input sample, latency {} samples = {} us",
        ssb_latency.get("latency_samples").unwrap_or(&Value::Null),
        ssb_latency.get("latency_us").unwrap_or(&Value::Null)
    );
    if write {
        let dir = root.join("benches").join("results");
        std::fs::create_dir_all(&dir)?;
        let doc = json!({
            "flow": "examples/flows/fm_receiver.json",
            "input": { "format": "cu8", "rate_hz": 2_400_000 },
            "platform": "x86_64 (AVX2)",
            "method": { "blocks": blocks, "block_samples": N, "counted": "(count(k blocks) - count(0 blocks)) / (k * block_samples)" },
            "instructions_per_input_sample": round2(per_sample),
            "latency": latency,
            "chains": {
                "ssb": {
                    "flow": "ddc (300 kHz, decimation 50, cutoff 4 kHz, transition 2 kHz) -> agc -> ssb_demod (usb), as examples/flows/ssb_receiver.json without the squelch",
                    "input": { "format": "cu8", "rate_hz": 2_400_000 },
                    "instructions_per_input_sample": round2(ssb_per_sample),
                    "latency": ssb_latency,
                },
            },
        });
        let mut text = serde_json::to_string_pretty(&doc)?;
        text.push('\n');
        std::fs::write(dir.join("pipeline.json"), text)?;
    }
    Ok(true)
}

/// The command line of `cargo xtask count`.
#[allow(
    clippy::struct_excessive_bools,
    reason = "one field per command-line flag"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// `--platform`.
    pub platform: Option<String>,
    /// `--level` (repeatable): only these levels of the platform.
    pub levels: Vec<String>,
    /// `--kernel` (repeatable): only these kernels.
    pub kernels: Vec<String>,
    /// Not `--no-write`.
    pub write: bool,
    /// `--pipeline`.
    pub pipeline: bool,
    /// `--strict`: every entry must be `release` without waivers and pass.
    pub strict: bool,
    /// Not `--no-variants`: also the informational misaligned and small-block runs.
    pub variants: bool,
    /// `--merge`: with `--kernel`, write the records of the measured kernels into the existing
    /// results file and keep everything else of it (a full run measures all kernels against
    /// all references, which a new kernel with no library to compare with does not need).
    pub merge: bool,
}

const USAGE: &str = "usage: cargo xtask count --platform <x86_64|x86_64-avx512|aarch64|wasm32> [--level L]... [--kernel NAME]... [--no-write] [--no-variants] [--merge] [--strict] | --pipeline [--no-write]";

impl Options {
    /// Parses the arguments after `count`.
    ///
    /// # Errors
    /// An unknown option or a missing value.
    pub fn parse(args: &[&str]) -> anyhow::Result<Options> {
        let mut o = Options {
            write: true,
            variants: true,
            ..Options::default()
        };
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            let mut value = || {
                it.next()
                    .map(|s| (*s).to_string())
                    .ok_or_else(|| anyhow!("`{arg}` needs a value; {USAGE}"))
            };
            match *arg {
                "--platform" => o.platform = Some(value()?),
                "--level" => o.levels.push(value()?),
                "--kernel" => o.kernels.push(value()?),
                "--no-write" => o.write = false,
                "--no-variants" => o.variants = false,
                "--merge" => o.merge = true,
                "--pipeline" => o.pipeline = true,
                "--strict" => o.strict = true,
                other => bail!("unknown option `{other}`; {USAGE}"),
            }
        }
        Ok(o)
    }
}

/// The plans to run for the options: the selected levels of the platform, plus the levels they
/// are judged against.
///
/// # Errors
/// An unknown platform or level.
pub fn select_plans(platform: &str, levels: &[String]) -> anyhow::Result<Vec<&'static LevelPlan>> {
    let all: Vec<&'static LevelPlan> = PLANS.iter().filter(|p| p.platform == platform).collect();
    if all.is_empty() {
        bail!("unknown platform `{platform}`");
    }
    for l in levels {
        if !all.iter().any(|p| p.level == l) {
            bail!("the platform `{platform}` has no level `{l}`");
        }
    }
    let selected = |level: &str| levels.is_empty() || levels.iter().any(|l| l == level);
    let baselines: Vec<&str> = all
        .iter()
        .filter(|p| selected(p.level))
        .filter_map(|p| p.baseline)
        .collect();
    Ok(all
        .into_iter()
        .filter(|p| selected(p.level) || baselines.contains(&p.level))
        .collect())
}

/// Measures the levels of `plans` and returns their results in the same order.
///
/// The counts do not depend on each other, so the levels run in parallel: first those judged
/// against the references, then those judged against our numbers of another level. The
/// counters are deterministic, and every level has its own target and scratch directory.
fn measure_levels(
    root: &Path,
    plans: &[&'static LevelPlan],
    options: &Options,
) -> anyhow::Result<Vec<LevelResult>> {
    // The C references are shared by the levels of a platform: build them before the threads.
    let mut refs: BTreeMap<&str, PathBuf> = BTreeMap::new();
    for plan in plans {
        if !refs.contains_key(plan.refs) {
            refs.insert(plan.refs, references(root, plan.refs)?);
        }
    }
    let mut own: BTreeMap<&'static str, OwnNumbers> = BTreeMap::new();
    let mut results: BTreeMap<&str, LevelResult> = BTreeMap::new();
    for judged_against_ours in [false, true] {
        let phase: Vec<&LevelPlan> = plans
            .iter()
            .copied()
            .filter(|p| p.baseline.is_some() == judged_against_ours)
            .collect();
        let measured = std::thread::scope(|scope| {
            let threads: Vec<_> = phase
                .iter()
                .map(|plan| {
                    let (refs, own) = (&refs, &own);
                    scope.spawn(move || {
                        let refs = refs
                            .get(plan.refs)
                            .ok_or_else(|| anyhow!("no references for {}", plan.refs))?;
                        measure_level(root, plan, refs, &options.kernels, options.variants, own)
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|t| {
                    t.join()
                        .map_err(|_| anyhow!("a measurement thread panicked"))?
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })?;
        for (plan, result) in phase.iter().zip(measured) {
            own.insert(
                plan.level,
                result
                    .measurements
                    .iter()
                    .map(|m| (m.kernel.clone(), (m.ours, m.ours_snr_db)))
                    .collect(),
            );
            results.insert(plan.level, result);
        }
    }
    plans
        .iter()
        .map(|plan| {
            results
                .remove(plan.level)
                .ok_or_else(|| anyhow!("no result for the level {}", plan.level))
        })
        .collect()
}

/// The judged records of `kernels` on the selected levels of `platform`, for `cargo xtask
/// measure`: one `(level, records)` pair per level, the records as in the results file
/// (instructions, ratio, comparison, status, verdict, SNR). Nothing is written, and the
/// informational variants are not measured.
///
/// # Errors
/// An unknown platform, level or kernel, or a tool that is missing or fails.
#[allow(
    clippy::large_stack_frames,
    reason = "the unoptimized build copies the table `KERNELS` onto the stack; xtask is a host tool"
)]
pub fn judged_records(
    root: &Path,
    platform: &str,
    levels: &[String],
    kernels: &[String],
) -> anyhow::Result<Vec<(&'static str, BTreeMap<String, Value>)>> {
    let plans = select_plans(platform, levels)?;
    for k in kernels {
        if !KERNELS.iter().any(|s| s.name == k) {
            bail!("unknown kernel `{k}`");
        }
    }
    let limits_path = root.join("benches").join("limits.json");
    let limits = Limits::parse(
        &std::fs::read_to_string(&limits_path)
            .with_context(|| format!("cannot read {}", limits_path.display()))?,
    )?;
    let options = Options {
        platform: Some(platform.to_string()),
        levels: levels.to_vec(),
        kernels: kernels.to_vec(),
        ..Options::default()
    };
    let results = measure_levels(root, &plans, &options)?;
    Ok(plans
        .iter()
        .zip(&results)
        .map(|(plan, result)| {
            let rule = if plan.baseline.is_some() {
                Rule::OwnLevel
            } else {
                Rule::Reference
            };
            let judged = evaluate(&limits, rule, plan.level, &result.measurements);
            (plan.level, judged.records)
        })
        .collect())
}

/// The instruction profiles of one count entry for level 2: ours and the judged comparison.
#[derive(Clone, Debug)]
pub struct EntryProfiles {
    /// The count entry.
    pub kernel: String,
    /// The comparison (as in the results file).
    pub compared_with: String,
    /// Samples processed by the counted calls (the divisor of the per-sample figures).
    pub samples: u64,
    /// Our side (the fallback for a fallback entry).
    pub ours: crate::mca::Counts,
    /// The judged comparison.
    pub other: crate::mca::Counts,
}

/// The instruction profiles of `kernels` at `level` of `platform` (level 2 of the measurement
/// policy): per entry, the execution counts per instruction of the counted calls for our side
/// and for the judged comparison, taken by `profiler` in the environment of the count of the
/// level (D68). Only levels judged against a reference can be profiled; the FFTs have the
/// restore of their input subtracted. An entry that cannot be profiled (not measured at the
/// level, a run that fails) is returned with its error, so that the report can name it.
///
/// # Errors
/// An unknown platform, level or kernel, a profiler that does not fit the counter of the level
/// (`sde` fits callgrind levels too, for its validation), or a tool that is missing or fails
/// before the first entry.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the plan table are large; xtask is a host tool"
)]
pub fn profiles(
    root: &Path,
    platform: &str,
    level: &str,
    kernels: &[String],
    profiler: Profiler,
) -> anyhow::Result<Vec<(String, anyhow::Result<EntryProfiles>)>> {
    let plan = PLANS
        .iter()
        .find(|p| p.platform == platform && p.level == level)
        .ok_or_else(|| anyhow!("the platform `{platform}` has no level `{level}`"))?;
    if plan.baseline.is_some() {
        bail!("the level {level} is judged against our own numbers, not against a reference");
    }
    match (profiler, plan.counter) {
        (Profiler::Callgrind, Counter::Callgrind)
        | (Profiler::Sde, Counter::Sde(_) | Counter::Callgrind)
        | (Profiler::Qemu, Counter::Qemu) => {}
        _ => bail!(
            "the level {level} counts with {}; its execution counts cannot come from {}",
            plan.counter.describe(),
            profiler.name()
        ),
    }
    let refs = references(root, plan.refs)?;
    let session = session(root, plan, &refs)?;
    let mut out = Vec::new();
    for spec in KERNELS
        .iter()
        .filter(|s| kernels.iter().any(|k| k == s.name))
    {
        if !spec.family.measured_at(plan) {
            out.push((
                spec.name.to_string(),
                Err(anyhow!("not measured at the level {level}")),
            ));
            continue;
        }
        let entry = || -> anyhow::Result<EntryProfiles> {
            let (gate, _) = comparisons(plan, spec);
            let info = session.query(&["info", spec.name])?;
            let compared_with = info
                .get("impls")
                .and_then(|i| i.get(gate.implementation()))
                .and_then(Value::as_str)
                .unwrap_or(gate.implementation())
                .to_string();
            println!(
                "==> {level} {} (instruction profiles from {}, ours vs {compared_with})",
                spec.name,
                profiler.name()
            );
            let standard = Variant::standard(spec);
            let restore = if spec.family.is_fft() {
                Some(session.profile(spec.name, "restore", &standard, profiler)?)
            } else {
                None
            };
            let profile = |which: &str| -> anyhow::Result<crate::mca::Counts> {
                let p = session.profile(spec.name, which, &standard, profiler)?;
                Ok(match &restore {
                    Some(r) => crate::mca::difference(&p, r),
                    None => p,
                })
            };
            Ok(EntryProfiles {
                kernel: spec.name.to_string(),
                compared_with,
                samples: K * (N / standard.samples).max(1) * standard.samples,
                ours: profile(spec.family.measured())?,
                other: profile(gate.implementation())?,
            })
        };
        out.push((spec.name.to_string(), entry()));
    }
    Ok(out)
}

/// Runs `cargo xtask count ...`.
///
/// ```text
/// cargo xtask count --platform <x86_64|x86_64-avx512|aarch64|wasm32|cortex-m4> [--level L]... [--kernel NAME]... [--no-write] [--no-variants] [--strict]
/// cargo xtask count --pipeline [--no-write]
/// ```
///
/// # Errors
/// A tool is missing or fails; the return value is `Ok(false)` if a measurement exceeds its
/// limit (or, with `--strict`, if any entry blocks a release).
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the plan table are large; xtask is a host tool"
)]
pub fn run(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let options = Options::parse(args)?;
    if options.pipeline {
        return pipeline(root, options.write);
    }
    if options.platform.as_deref() == Some("cortex-m4") {
        return crate::count_embedded::run(root, &options.kernels, options.write);
    }
    let Some(platform) = options.platform.as_deref() else {
        bail!("{USAGE}");
    };
    let plans = select_plans(platform, &options.levels)?;
    for k in &options.kernels {
        if !KERNELS.iter().any(|s| s.name == k) {
            bail!("unknown kernel `{k}`");
        }
    }
    let limits_path = root.join("benches").join("limits.json");
    let limits = Limits::parse(
        &std::fs::read_to_string(&limits_path)
            .with_context(|| format!("cannot read {}", limits_path.display()))?,
    )?;
    let mut all_ok = true;
    let mut blockers = Vec::new();
    let mut levels = serde_json::Map::new();
    let mut volk_flags = Value::Null;
    let results = measure_levels(root, &plans, &options)?;
    for (plan, result) in plans.iter().zip(&results) {
        let rule = if plan.baseline.is_some() {
            Rule::OwnLevel
        } else {
            Rule::Reference
        };
        let judged = evaluate(&limits, rule, plan.level, &result.measurements);
        for line in &judged.lines {
            println!("{line}");
        }
        all_ok &= judged.ok;
        blockers.extend(judged.blockers);
        levels.insert(
            plan.level.to_string(),
            json!({
                "counter": plan.counter.describe(),
                "judged_against": match plan.baseline {
                    Some(level) => format!("ours at {level}"),
                    None => "references".to_string(),
                },
                "volk": {
                    "machine": result.volk.get("machine"),
                    "alignment": result.volk.get("alignment"),
                    "pinned": plan.volk_pin.config().lines().collect::<Vec<_>>(),
                },
                "kernels": judged.records,
            }),
        );
        if let Some(flags) = result.volk.get("compiler_flags") {
            volk_flags = flags.clone();
        }
    }
    if options.write && options.kernels.is_empty() && options.levels.is_empty() {
        write_results(root, platform, &limits.caps, &levels, &volk_flags)?;
    } else if options.write && options.merge {
        if options.kernels.is_empty() {
            bail!("--merge needs --kernel");
        }
        merge_results(root, platform, &levels)?;
    }
    if options.strict && !blockers.is_empty() {
        eprintln!("--strict: {} entries block a release:", blockers.len());
        for b in &blockers {
            eprintln!("  {b}");
        }
        all_ok = false;
    }
    if all_ok {
        println!("count {platform}: ok");
    } else {
        eprintln!("count {platform}: limits exceeded");
    }
    Ok(all_ok)
}

/// The manifest of the C references of `platform` (`benches/target/refs/<refs>/refs.json`,
/// written by `benches/refs/build.sh`), or `null` when there is none.
fn reference_manifest(root: &Path, platform: &str) -> Value {
    let refs_platform = PLANS
        .iter()
        .find(|p| p.platform == platform)
        .map_or(platform, |p| p.refs);
    let manifest = root
        .join("benches")
        .join("target")
        .join("refs")
        .join(refs_platform)
        .join("refs.json");
    std::fs::read_to_string(&manifest)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null)
}

/// Adds to the `references` section of a results file what the manifest of the references has
/// and the section lacks: a library (`libraries.<name>`) or a top-level fact (a compiler such as
/// `kfr_cxx`). An entry the section already has is kept, because the records of the kernels
/// that were not measured again were made with it; the names of the entries that differ from
/// the manifest are returned, so that the caller can say that a full run is due.
fn merge_references(section: &mut Value, manifest: &Value) -> Vec<String> {
    let mut differ = Vec::new();
    let Some(manifest) = manifest.as_object() else {
        return differ;
    };
    if !section.is_object() {
        *section = json!({});
    }
    let Some(section) = section.as_object_mut() else {
        return differ;
    };
    for (key, value) in manifest {
        if key == "libraries" {
            let Some(libraries) = value.as_object() else {
                continue;
            };
            let target = section.entry("libraries").or_insert_with(|| json!({}));
            if !target.is_object() {
                *target = json!({});
            }
            let Some(target) = target.as_object_mut() else {
                continue;
            };
            for (name, library) in libraries {
                match target.get(name) {
                    None => {
                        target.insert(name.clone(), library.clone());
                    }
                    Some(old) if old != library => differ.push(format!("libraries.{name}")),
                    Some(_) => {}
                }
            }
        } else {
            match section.get(key) {
                None => {
                    section.insert(key.clone(), value.clone());
                }
                Some(old) if old != value => differ.push(key.clone()),
                Some(_) => {}
            }
        }
    }
    differ
}

/// Writes the records of the measured kernels into the results file of `platform`: for every
/// measured level, the records of the kernels in `measured` replace or add the entries of the
/// file, and the `references` section gets the libraries and compilers of the manifest of the
/// references that it lacks (a reference that is new since the last full run); every other
/// entry, and the rest of the file, stays as it is.
///
/// # Errors
/// The file cannot be read or written, or it has no entry for a measured level.
fn merge_results(
    root: &Path,
    platform: &str,
    measured: &serde_json::Map<String, Value>,
) -> anyhow::Result<()> {
    let path = root
        .join("benches")
        .join("results")
        .join(format!("{platform}.json"));
    let mut doc: Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )?;
    for (level, new) in measured {
        let Some(records) = new["kernels"].as_object() else {
            continue;
        };
        let kernels = doc
            .get_mut("levels")
            .and_then(|l| l.get_mut(level.as_str()))
            .and_then(|l| l.get_mut("kernels"))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| anyhow!("{} has no level {level}", path.display()))?;
        for (kernel, record) in records {
            kernels.insert(kernel.clone(), record.clone());
        }
    }
    let manifest = reference_manifest(root, platform);
    if let Some(object) = doc.as_object_mut() {
        let section = object.entry("references").or_insert(Value::Null);
        let differ = merge_references(section, &manifest);
        if !differ.is_empty() {
            println!(
                "note: {} of the references in {} differ from the manifest of the references and are kept until a full run: {}",
                differ.len(),
                path.display(),
                differ.join(", ")
            );
        }
    }
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    std::fs::write(&path, text)?;
    println!("merged the measured kernels into {}", path.display());
    Ok(())
}

#[allow(
    clippy::large_stack_frames,
    reason = "serde_json values and the file system calls are large; xtask is a host tool"
)]
fn write_results(
    root: &Path,
    platform: &str,
    caps: &Caps,
    levels: &serde_json::Map<String, Value>,
    volk_flags: &Value,
) -> anyhow::Result<()> {
    let references = reference_manifest(root, platform);
    let host_cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_default();
    let doc = json!({
        "platform": platform,
        "method": {
            "k": K,
            "samples": N,
            "counted": "(count(k = K) - count(k = 0)) / (K * samples), minus the restore of the input for the FFTs",
            "limits": { "release": caps.release, "dev": caps.dev, "warn": caps.warn, "goal": caps.goal, "runtime": caps.runtime },
        },
        "host_cpu": host_cpu,
        "references": references,
        "volk_compiler_flags": volk_flags,
        "levels": levels,
    });
    let dir = root.join("benches").join("results");
    std::fs::create_dir_all(&dir)?;
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    std::fs::write(dir.join(format!("{platform}.json")), text)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn merging_adds_the_new_references_and_keeps_the_measured_ones() {
        let mut section = json!({
            "cc": "gcc 15",
            "libraries": { "volk": { "version": "3.3.0" }, "fftw": { "version": "3.3.10" } },
        });
        let manifest = json!({
            "cc": "gcc 16",
            "kfr_cxx": "g++ 15",
            "libraries": {
                "volk": { "version": "3.3.0" },
                "fftw": { "version": "3.3.11" },
                "kfr": { "version": "7.2.0" },
            },
        });
        let differ = merge_references(&mut section, &manifest);
        assert_eq!(differ, vec!["cc".to_string(), "libraries.fftw".to_string()]);
        assert_eq!(section["cc"], "gcc 15");
        assert_eq!(section["kfr_cxx"], "g++ 15");
        assert_eq!(section["libraries"]["fftw"]["version"], "3.3.10");
        assert_eq!(section["libraries"]["kfr"]["version"], "7.2.0");
        assert_eq!(section["libraries"]["volk"]["version"], "3.3.0");
        let mut empty = Value::Null;
        assert_eq!(
            merge_references(&mut empty, &manifest),
            Vec::<String>::new()
        );
        assert_eq!(empty, manifest);
        let mut kept = json!({ "cc": "gcc 15" });
        assert_eq!(
            merge_references(&mut kept, &Value::Null),
            Vec::<String>::new()
        );
        assert_eq!(kept, json!({ "cc": "gcc 15" }));
    }

    #[test]
    fn qemu_option_variables_are_recognised() {
        for name in ["QEMU_VERSION", "QEMU_LD_PREFIX", "QEMU_"] {
            assert!(
                is_qemu_option_variable(std::ffi::OsStr::new(name)),
                "{name}"
            );
        }
        for name in [
            "SDRK_QEMU_AARCH64",
            "QEMU",
            "qemu_version",
            "LD_LIBRARY_PATH",
        ] {
            assert!(
                !is_qemu_option_variable(std::ffi::OsStr::new(name)),
                "{name}"
            );
        }
    }

    fn m(kernel: &str, ours: f64, other: f64) -> Measurement {
        Measurement {
            kernel: kernel.into(),
            level: "avx2".into(),
            ours,
            other,
            compared_with: "volk".into(),
            ours_snr_db: 130.0,
            other_snr_db: 130.0,
            extras: Vec::new(),
        }
    }

    fn limits(text: &str) -> Limits {
        Limits::parse(text).unwrap()
    }

    /// Caps with a warning band below the release limit, to test it.
    const CAPS: Caps = Caps {
        release: 1.15,
        dev: 1.50,
        warn: 1.10,
        goal: 1.00,
        runtime: 1.02,
    };

    fn limit(status: Status, waiver: Option<&str>, accuracy: Option<&str>) -> Limit {
        Limit {
            status,
            waiver: waiver.map(Into::into),
            accuracy_waiver: accuracy.map(Into::into),
        }
    }

    fn judged(l: &Limit, x: &Measurement) -> Verdict {
        judge(&CAPS, Rule::Reference, Some(l), x)
    }

    #[test]
    fn a_slow_release_kernel_fails_the_command() {
        let l = limits(r#"{ "dummy_slow": { "avx2": { "status": "release" } } }"#);
        let report = evaluate(&l, Rule::Reference, "avx2", &[m("dummy_slow", 20.0, 10.0)]);
        assert!(!report.ok);
        assert!(report.lines[0].contains("FAIL"), "{}", report.lines[0]);
        assert!(report.lines[0].contains("2.00x"));
    }

    #[test]
    fn the_default_caps_are_those_of_d14() {
        let caps = Caps::default();
        assert_eq!((caps.release, caps.dev, caps.goal), (1.10, 1.50, 1.00));
        assert_eq!(caps.warn, caps.release);
    }

    #[test]
    fn the_caps_set_the_release_and_dev_limits_and_the_warning_band() {
        let rel = Limit::release();
        let dev = limit(Status::Dev, None, None);
        assert_eq!(
            judged(&rel, &m("k", 11.4, 10.0)),
            Verdict::Warn("1.14x is above 1.10x (release limit 1.15x)".into())
        );
        assert_eq!(judged(&rel, &m("k", 10.9, 10.0)), Verdict::Pass);
        assert!(matches!(
            judged(&rel, &m("k", 11.6, 10.0)),
            Verdict::Fail(_)
        ));
        assert_eq!(
            judged(&dev, &m("k", 13.0, 10.0)),
            Verdict::Warn("1.30x is above 1.10x (dev limit 1.50x)".into())
        );
        assert!(matches!(
            judged(&dev, &m("k", 14.9, 10.0)),
            Verdict::Warn(_)
        ));
        assert!(matches!(
            judged(&dev, &m("k", 15.1, 10.0)),
            Verdict::Fail(_)
        ));
    }

    #[test]
    fn the_caps_come_from_limits_json() {
        let l = limits(
            r#"{ "_limits": { "release": 1.10 }, "k": { "avx2": { "status": "release" } } }"#,
        );
        assert_eq!(l.caps.release, 1.10);
        assert_eq!(l.caps.dev, 1.50);
        let report = evaluate(&l, Rule::Reference, "avx2", &[m("k", 11.2, 10.0)]);
        assert!(!report.ok, "{}", report.lines[0]);
        assert!(Limits::parse(r#"{ "_limits": { "release": 0 } }"#).is_err());
        assert!(Limits::parse(r#"{ "_limits": { "fast": 1.0 } }"#).is_err());
    }

    #[test]
    fn a_dev_entry_within_the_warning_threshold_is_ready_for_release() {
        let dev = limit(Status::Dev, None, None);
        assert!(
            matches!(judged(&dev, &m("k", 10.5, 10.0)), Verdict::Warn(w) if w.contains("promoted"))
        );
    }

    #[test]
    fn a_kernel_that_is_not_listed_fails() {
        let l = limits("{}");
        assert_eq!(l.get("anything", "avx2"), None);
        let report = evaluate(&l, Rule::Reference, "avx2", &[m("anything", 5.0, 10.0)]);
        assert!(!report.ok);
        assert!(report.lines[0].contains("no entry"), "{}", report.lines[0]);
    }

    #[test]
    fn a_failing_ratio_is_printed_above_the_limit() {
        let l = limit(Status::Release, None, None);
        let v = judge(&CAPS, Rule::OwnLevel, Some(&l), &m("rotate", 6.66, 6.52));
        assert!(
            v.detail().starts_with("1.021x instructions per sample"),
            "{}",
            v.detail()
        );
        assert!(v.detail().contains("limit of 1.02x"), "{}", v.detail());
        assert_eq!(over(1.3, 1.15), "1.30x");
    }

    #[test]
    fn lower_accuracy_than_the_comparison_fails() {
        let rel = Limit::release();
        let mut x = m("k", 10.0, 10.0);
        x.ours_snr_db = 128.5;
        assert!(matches!(judged(&rel, &x), Verdict::Fail(_)));
        x.ours_snr_db = 129.5;
        assert_eq!(judged(&rel, &x), Verdict::Pass);
        x.ours_snr_db = 140.0;
        assert_eq!(judged(&rel, &x), Verdict::Pass);
    }

    #[test]
    fn a_waiver_reports_but_does_not_fail() {
        let l = limits(
            r#"{ "fft_4096": { "avx2": { "status": "dev", "waiver": "open item M13-1" } } }"#,
        );
        let report = evaluate(&l, Rule::Reference, "avx2", &[m("fft_4096", 40.0, 10.0)]);
        assert!(report.ok);
        assert!(report.lines[0].contains("waived") && report.lines[0].contains("M13-1"));
        assert_eq!(report.records["fft_4096"]["verdict"], "waived");
    }

    #[test]
    fn a_ratio_waiver_does_not_hide_an_accuracy_regression() {
        let l = limit(Status::Dev, Some("M13-2"), None);
        let mut x = m("k", 40.0, 10.0);
        x.ours_snr_db = 120.0;
        match judged(&l, &x) {
            Verdict::Fail(d) => assert!(d.contains("accuracy") && d.contains("M13-2"), "{d}"),
            v => panic!("{v:?}"),
        }
        let both = limit(Status::Dev, Some("M13-2"), Some("M13-3"));
        assert!(matches!(judged(&both, &x), Verdict::Waived(_)));
        // An accuracy waiver alone does not cover the instruction count either.
        let acc = limit(Status::Dev, None, Some("M13-3"));
        assert!(matches!(judged(&acc, &x), Verdict::Fail(d) if d.contains("exceeds")));
    }

    #[test]
    fn a_waiver_that_covers_nothing_is_reported() {
        let l = limit(Status::Dev, Some("M13-2"), Some("M13-3"));
        match judged(&l, &m("k", 9.0, 10.0)) {
            Verdict::Warn(w) => {
                assert!(w.contains("waiver is no longer needed (M13-2)"), "{w}");
                assert!(
                    w.contains("accuracy waiver is no longer needed (M13-3)"),
                    "{w}"
                );
            }
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn strict_mode_blocks_everything_but_clean_release_entries() {
        let rel = Limit::release();
        assert_eq!(
            blocks_release(Some(&rel), &judged(&rel, &m("k", 9.0, 10.0))),
            None
        );
        assert_eq!(
            blocks_release(Some(&rel), &judged(&rel, &m("k", 11.2, 10.0))),
            None
        );
        let dev = limit(Status::Dev, None, None);
        assert!(blocks_release(Some(&dev), &judged(&dev, &m("k", 9.0, 10.0))).is_some());
        let stale = limit(Status::Release, Some("M13-1"), None);
        assert!(blocks_release(Some(&stale), &judged(&stale, &m("k", 9.0, 10.0))).is_some());
        let info = limit(Status::Info, Some("no SSE code in VOLK"), None);
        assert!(blocks_release(Some(&info), &judged(&info, &m("k", 9.0, 10.0))).is_some());
        assert!(
            blocks_release(
                None,
                &judge(&CAPS, Rule::Reference, None, &m("k", 1.0, 1.0))
            )
            .is_some()
        );
        let l = limits(
            r#"{ "k": { "avx2": { "status": "dev" } }, "j": { "avx2": { "status": "release" } } }"#,
        );
        let report = evaluate(
            &l,
            Rule::Reference,
            "avx2",
            &[m("k", 9.0, 10.0), m("j", 9.0, 10.0)],
        );
        assert!(report.ok);
        assert_eq!(report.blockers.len(), 1, "{:?}", report.blockers);
        assert!(report.blockers[0].starts_with("k avx2"));
    }

    #[test]
    fn run_time_detection_is_judged_against_our_fixed_level() {
        let mut x = m("k", 10.3, 10.0);
        x.level = "avx2-runtime".into();
        assert!(
            matches!(judge(&CAPS, Rule::OwnLevel, None, &x), Verdict::Fail(d) if d.contains("1.02x"))
        );
        x.ours = 10.1;
        assert_eq!(judge(&CAPS, Rule::OwnLevel, None, &x), Verdict::Pass);
        let waived = limit(Status::Release, Some("M14-9"), None);
        x.ours = 10.5;
        assert!(matches!(
            judge(&CAPS, Rule::OwnLevel, Some(&waived), &x),
            Verdict::Waived(_)
        ));
        let report = evaluate(
            &limits("{}"),
            Rule::OwnLevel,
            "avx2-runtime",
            &[m("k", 10.1, 10.0)],
        );
        assert!(report.ok);
        assert_eq!(report.records["k"]["status"], "release");
    }

    #[test]
    fn an_info_entry_is_reported_but_never_judged() {
        let l = limits(
            r#"{ "rotate": { "sse2": { "status": "info", "waiver": "VOLK has no SSE rotator" } } }"#,
        );
        let report = evaluate(&l, Rule::Reference, "sse2", &[m("rotate", 400.0, 10.0)]);
        assert!(report.ok);
        assert!(report.lines[0].contains("info") && report.lines[0].contains("SSE rotator"));
        assert_eq!(report.records["rotate"]["verdict"], "info");
        assert_eq!(report.records["rotate"]["status"], "info");
    }

    #[test]
    fn results_record_the_goal_and_the_informational_comparisons() {
        let l = limits(r#"{ "k": { "neon": { "status": "release" } } }"#);
        let mut x = m("k", 9.0, 10.0);
        x.level = "neon".into();
        x.extras.push(Extra {
            label: "goal: cheapest VOLK implementation".into(),
            ours: 9.0,
            other: 6.0,
            compared_with: "volk_x: neon".into(),
            other_snr_db: Some(120.0),
        });
        let report = evaluate(&l, Rule::Reference, "neon", &[x]);
        assert_eq!(report.records["k"]["goal"], true);
        assert_eq!(report.records["k"]["informational"][0]["ratio"], 1.5);
        assert!(report.lines[1].contains("cheapest"), "{:?}", report.lines);
    }

    #[test]
    fn limits_json_rejects_an_unknown_status() {
        assert!(Limits::parse(r#"{ "k": { "avx2": { "status": "fast" } } }"#).is_err());
        assert!(Limits::parse("[]").is_err());
        assert!(Limits::parse("{").is_err());
    }

    #[test]
    fn callgrind_output_is_parsed() {
        let text = "==123== Callgrind, a call-graph generating cache profiler\n==123== \n==123== Collected : 108110 2 3\n==123== I   refs: 108,110\n";
        assert_eq!(parse::callgrind(text).unwrap(), 108_110);
        assert!(parse::callgrind("nothing").is_err());
    }

    #[test]
    fn sde_and_qemu_output_is_parsed() {
        let icount = "$$ TID: 0 ICOUNT: 12345678\n$$ TID: 1 ICOUNT: 2\n";
        assert_eq!(parse::sde_icount(icount).unwrap(), 12_345_680);
        assert!(parse::sde_icount("no totals").is_err());
        assert_eq!(
            parse::qemu_insn("cpu 0 insns: 10\ncpu 1 insns: 5\n").unwrap(),
            15
        );
        assert_eq!(parse::qemu_insn("total insns: 99\n").unwrap(), 99);
        assert!(parse::qemu_insn("").is_err());
    }

    #[test]
    fn the_median_ignores_one_outlier() {
        assert_eq!(
            median(vec![1_985_000, 2_085_000, 1_981_625]),
            Some(1_985_000)
        );
        assert_eq!(median(vec![7]), Some(7));
        assert_eq!(median(Vec::new()), None);
    }

    #[test]
    fn the_third_sde_run_is_made_only_when_the_first_two_differ() {
        let mut runs = 0;
        let got = median_of_three(|| {
            runs += 1;
            Ok(5)
        })
        .unwrap();
        assert_eq!((got, runs), (5, 2));
        let mut values = vec![7u64, 100, 5].into_iter();
        assert_eq!(median_of_three(|| Ok(values.next().unwrap())).unwrap(), 7);
    }

    #[test]
    fn the_fuel_bisection_finds_the_exact_amount() {
        for need in [1u64, 2, 1000, 1_048_576, 1_048_577, 5_000_000_000] {
            let mut calls = 0;
            let got = least_fuel(|fuel| {
                calls += 1;
                Ok(fuel >= need)
            })
            .unwrap();
            assert_eq!(got, need);
            // The bisection starts from the last amount that failed, not from 0.
            assert!(calls <= 2 * 64_u32.saturating_sub(need.leading_zeros()) + 21);
        }
    }

    #[test]
    fn the_plan_covers_the_four_platforms_with_established_references() {
        for p in ["x86_64", "x86_64-avx512", "aarch64", "wasm32"] {
            assert!(PLANS.iter().any(|plan| plan.platform == p), "{p}");
        }
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        assert_eq!(
            comparisons(plan("sse2"), kernel("fft_1024")),
            (
                Reference::Fftw,
                vec![Reference::FftwInPlace, Reference::RustFft]
            )
        );
        assert_eq!(
            comparisons(plan("sse4.2"), kernel("fft_1024")).0,
            Reference::RustFft
        );
        assert_eq!(
            comparisons(plan("sse2"), kernel("rfft_1024")),
            (
                Reference::Fftw,
                vec![Reference::RustFft, Reference::Fallback]
            )
        );
        assert_eq!(
            comparisons(plan("avx2"), kernel("rfft_1024")),
            (Reference::RustFft, vec![Reference::Fallback])
        );
        assert_eq!(
            comparisons(plan("neon"), kernel("rotate")).0,
            Reference::Volk
        );
        assert_eq!(
            comparisons(plan("neon"), kernel("rotate")).1,
            vec![Reference::VolkCheapest]
        );
        assert_eq!(
            comparisons(plan("simd128"), kernel("mag_sq")).0,
            Reference::Volk
        );
        assert_eq!(
            comparisons(plan("sse2"), kernel("agc")),
            (Reference::GnuRadio, vec![Reference::Fallback])
        );
        assert_eq!(
            comparisons(plan("avx2-runtime"), kernel("agc")).0,
            Reference::OwnLevel("avx2")
        );
        assert_eq!(
            comparisons(plan("simd128"), kernel("agc_cq15")).0,
            Reference::Fallback
        );
        assert_eq!(
            comparisons(plan("neon"), kernel("squelch")),
            (Reference::GnuRadio, vec![Reference::Fallback])
        );
        assert_eq!(
            comparisons(plan("sse2"), kernel("squelch_cq15")).0,
            Reference::Fallback
        );
        assert_eq!(
            comparisons(plan("neon"), kernel("fft_cq15_1024")),
            (Reference::KissFft, vec![Reference::Ne10])
        );
        assert_eq!(
            comparisons(plan("avx2"), kernel("fm_mod")),
            (Reference::GnuRadio, vec![Reference::Fallback])
        );
        assert_eq!(
            comparisons(plan("avx2-runtime"), kernel("fir")).0,
            Reference::OwnLevel("avx2")
        );
        // VOLK's wasm32 build has no `volk_32f_x2_dot_prod_32f` and no `volk_32f_s32f_multiply_32f`.
        for k in ["dot_real", "scale"] {
            assert_eq!(comparisons(plan("avx2"), kernel(k)).0, Reference::Volk);
            assert_eq!(
                comparisons(plan("neon"), kernel(k)),
                (Reference::Volk, vec![Reference::VolkCheapest])
            );
            assert_eq!(
                comparisons(plan("simd128"), kernel(k)),
                (Reference::Fallback, Vec::new())
            );
        }
        assert_eq!(
            comparisons(plan("simd128"), kernel("dot")).0,
            Reference::Volk
        );
    }

    #[test]
    fn the_oscillators_have_scalar_references_for_their_fallbacks() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "neon", "simd128"] {
            for k in ["rotate_cq15", "fm_mod_cq15"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Fallback, vec![Reference::Cmsis]),
                    "{k} at {level}"
                );
                let fallback = kernel(&format!("{k}_fallback"));
                assert_eq!(fallback.family.measured(), "fallback");
                assert_eq!(
                    comparisons(plan(level), fallback),
                    (Reference::Cmsis, Vec::new())
                );
            }
            for k in ["rotate_fallback", "fm_mod_fallback"] {
                assert_eq!(kernel(k).family.measured(), "fallback");
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::GnuRadio, vec![Reference::Cmsis])
                );
            }
        }
        assert!(
            !kernel("rotate_fallback")
                .family
                .measured_at(plan("avx2-runtime"))
        );
    }

    #[test]
    fn the_biquads_are_judged_against_kfr_cmsis_and_their_fallback_entries() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            for k in ["biquad", "biquad_real"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Kfr, vec![Reference::Cmsis]),
                    "{k} at {level}"
                );
            }
        }
        assert_eq!(
            comparisons(plan("neon"), kernel("biquad_cq15")),
            (Reference::Fallback, vec![Reference::Cmsis])
        );
        let fallback = kernel("biquad_real_q15_fallback");
        assert_eq!(fallback.family.measured(), "fallback");
        assert_eq!(kernel("biquad_real").family.measured(), "ours");
        assert_eq!(
            comparisons(plan("avx512"), fallback),
            (Reference::Cmsis, Vec::new())
        );
        assert!(fallback.family.measured_at(plan("simd128")));
        assert!(!fallback.family.measured_at(plan("avx2-runtime")));
        assert!(
            kernel("biquad_real")
                .family
                .measured_at(plan("avx2-runtime"))
        );
    }

    #[test]
    fn the_halfband_entries_are_judged_against_external_references() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            // liquid-dsp's SIMD dot product is the gate of the float kernels.
            assert_eq!(
                comparisons(plan(level), kernel("halfband_decimate")),
                (
                    Reference::Liquid,
                    vec![Reference::OwnKernel("halfband_ref_fir_decimate")]
                ),
                "{level}"
            );
            assert_eq!(
                comparisons(plan(level), kernel("halfband_decimate_real")),
                (Reference::Liquid, Vec::new())
            );
            // libairspy's SSE2 converter is information at `sse2` only.
            let airspy = if level == "sse2" {
                vec![Reference::Libairspy]
            } else {
                Vec::new()
            };
            let mut extras = airspy;
            extras.push(Reference::OwnKernel("halfband_ref_shifted"));
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex")),
                (Reference::Liquid, extras)
            );
            // No library has a Q15 halfband decimator: our fallback gates, the CMSIS-DSP
            // composition is information; `real_to_complex_cq15` has libairspy (D64).
            assert_eq!(
                comparisons(plan(level), kernel("halfband_decimate_real_q15")),
                (Reference::Fallback, vec![Reference::Cmsis])
            );
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex_cq15")),
                (
                    Reference::Libairspy,
                    vec![
                        Reference::Fallback,
                        Reference::OwnKernel("halfband_ref_shifted_cq15")
                    ]
                )
            );
            // The fallback entries: liquid-dsp's portable dot product, the CMSIS-DSP
            // composition, libairspy's generic code.
            for k in [
                "halfband_decimate_fallback",
                "halfband_decimate_real_fallback",
            ] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Liquid, Vec::new())
                );
                assert_eq!(kernel(k).family.measured(), "fallback");
            }
            for k in [
                "halfband_decimate_cq15_fallback",
                "halfband_decimate_real_q15_fallback",
            ] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, Vec::new())
                );
            }
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex_fallback")),
                (Reference::Liquid, vec![Reference::Libairspy])
            );
            assert_eq!(
                comparisons(plan(level), kernel("real_to_complex_cq15_fallback")),
                (Reference::Libairspy, Vec::new())
            );
        }
        for k in [
            "halfband_decimate_fallback",
            "real_to_complex_cq15_fallback",
            "halfband_ref_fir_decimate",
            "halfband_ref_shifted_cq15",
        ] {
            assert!(!kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
            assert!(kernel(k).family.measured_at(plan("avx2")), "{k}");
        }
        // The own-kernel comparison runs `ours` of the other kernel, and every kernel it names
        // is in the table.
        assert_eq!(Reference::OwnKernel("x").implementation(), "ours");
        assert_eq!(Reference::Libairspy.implementation(), "airspy");
        for spec in &KERNELS {
            if let Some(own) = spec.own {
                assert!(KERNELS.iter().any(|k| k.name == own), "{own}");
            }
        }
    }

    #[test]
    fn the_recursion_fallbacks_are_judged_against_liquid_and_gnu_radio() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "avx2", "avx512", "neon", "simd128"] {
            for k in ["dc_block_fallback", "dc_block_real_fallback"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Liquid, Vec::new())
                );
            }
            assert_eq!(
                comparisons(plan(level), kernel("deemphasis_fallback")),
                (Reference::GnuRadio, vec![Reference::Cmsis])
            );
            assert_eq!(
                comparisons(plan(level), kernel("deemphasis")),
                (Reference::Fallback, Vec::new())
            );
        }
        assert_eq!(kernel("deemphasis_fallback").family.measured(), "fallback");
        assert!(
            !kernel("dc_block_fallback")
                .family
                .measured_at(plan("avx2-runtime"))
        );
    }

    #[test]
    fn the_fft_fallback_entries_are_judged_against_cmsis_with_the_restore_subtracted() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
            for k in [
                "fft_64_fallback",
                "fft_1024_fallback",
                "fft_4096_fallback",
                "rfft_1024_fallback",
            ] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, Vec::new()),
                    "{k} at {level}"
                );
            }
            for k in ["fft_cq15_1024_fallback", "fft_cq15_1024_default_fallback"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, vec![Reference::KissFft]),
                    "{k} at {level}"
                );
            }
        }
        for k in [
            "fft_1024_fallback",
            "fft_cq15_1024_fallback",
            "rfft_1024_fallback",
        ] {
            let family = kernel(k).family;
            assert!(family.is_fft(), "{k}");
            assert_eq!(family.measured(), "fallback", "{k}");
            assert!(!family.measured_at(plan("avx2-runtime")), "{k}");
        }
        assert_eq!(kernel("fft_1024").family.measured(), "ours");
    }

    #[test]
    fn the_dot_and_scale_fallbacks_are_judged_against_cmsis_dsp() {
        let plan = |l: &str| PLANS.iter().find(|p| p.level == l).unwrap();
        let kernel = |k: &str| KERNELS.iter().find(|s| s.name == k).unwrap();
        for k in [
            "dot_fallback",
            "dot_cq15_fallback",
            "dot_real_fallback",
            "dot_real_q15_fallback",
            "scale_fallback",
            "scale_cq15_fallback",
        ] {
            assert_eq!(kernel(k).family.measured(), "fallback", "{k}");
            for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
                assert_eq!(
                    comparisons(plan(level), kernel(k)),
                    (Reference::Cmsis, Vec::new()),
                    "{k} at {level}"
                );
            }
            assert!(!kernel(k).family.measured_at(plan("avx2-runtime")), "{k}");
        }
    }

    #[test]
    fn both_runs_of_a_measurement_have_arguments_of_the_same_length() {
        let variant = Variant {
            samples: 64,
            misaligned: true,
            explicit_n: true,
        };
        let empty = run_args("fir", "ours", 0, 1024, &variant);
        let full = run_args("fir", "ours", 1024, 1024, &variant);
        assert_eq!(
            empty,
            ["run", "fir", "ours", "0000", "--n", "64", "--misaligned"]
        );
        assert_eq!(
            full,
            ["run", "fir", "ours", "1024", "--n", "64", "--misaligned"]
        );
    }

    #[test]
    fn the_measurement_build_remaps_the_paths_of_the_checkout() {
        let flags = encoded_rustflags(
            "-C target-cpu=x86-64-v3",
            Path::new("/work/my tree"),
            Some(Path::new("/home/u/.cargo")),
        );
        let args: Vec<&str> = flags.to_str().unwrap_or_default().split('\u{1f}').collect();
        assert_eq!(
            args,
            [
                "-C",
                "target-cpu=x86-64-v3",
                "--remap-path-prefix=/work/my tree=/sdrk",
                "--remap-path-prefix=/home/u/.cargo=/cargo",
            ]
        );
        let without_flags = encoded_rustflags("", Path::new("/w"), None);
        assert_eq!(without_flags, "--remap-path-prefix=/w=/sdrk");
    }

    #[test]
    fn the_kernel_table_names_the_kernels_of_the_measurement_binary() {
        let source = include_str!("../../benches/kernels/src/lib.rs");
        let table = source
            .split("pub const KERNELS")
            .nth(1)
            .and_then(|t| t.split("];").next())
            .unwrap();
        let mut binary: Vec<&str> = table.split('"').skip(1).step_by(2).collect();
        binary.sort_unstable();
        let mut ours: Vec<&str> = KERNELS.iter().map(|k| k.name).collect();
        ours.sort_unstable();
        assert_eq!(binary, ours);
    }

    #[test]
    fn the_kernel_table_is_consistent() {
        let names: Vec<&str> = KERNELS.iter().map(|k| k.name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
        for spec in &KERNELS {
            let fallback = spec.family.is_fft() && spec.family.is_fallback_entry();
            let complex = matches!(spec.family, Family::FftC32 | Family::FftCq15)
                || (fallback && spec.name.starts_with("fft_"));
            let real =
                spec.family == Family::RealFft || (fallback && spec.name.starts_with("rfft_"));
            assert_eq!(
                fallback,
                spec.name.ends_with("_fallback") && (complex || real)
            );
            assert_eq!(complex, spec.name.starts_with("fft_"), "{}", spec.name);
            assert_eq!(real, spec.name.starts_with("rfft_"), "{}", spec.name);
            let fft = spec.family.is_fft();
            assert_eq!(fft, complex || real);
            // The FFT sizes are fixed by the name; their buffers are always aligned.
            assert!(!(fft && (spec.misaligned || spec.small_n)), "{}", spec.name);
        }
    }

    #[test]
    fn selecting_a_level_adds_the_level_it_is_judged_against() {
        let levels = |l: &[&str]| -> Vec<&str> {
            let l: Vec<String> = l.iter().map(ToString::to_string).collect();
            select_plans("x86_64", &l)
                .unwrap()
                .iter()
                .map(|p| p.level)
                .collect()
        };
        assert_eq!(levels(&["avx2-runtime"]), vec!["avx2", "avx2-runtime"]);
        assert_eq!(levels(&["sse2"]), vec!["sse2"]);
        assert_eq!(levels(&[]).len(), 4);
        assert!(select_plans("x86_64", &["neon".to_string()]).is_err());
        assert!(select_plans("riscv", &[]).is_err());
    }

    #[test]
    fn options_are_parsed() {
        let o = Options::parse(&[
            "--platform",
            "aarch64",
            "--kernel",
            "fir",
            "--strict",
            "--no-variants",
        ])
        .unwrap();
        assert_eq!(o.platform.as_deref(), Some("aarch64"));
        assert_eq!(o.kernels, vec!["fir".to_string()]);
        assert!(o.strict && !o.variants && o.write);
        assert!(Options::parse(&["--platform"]).is_err());
        assert!(Options::parse(&["--fast"]).is_err());
    }

    #[test]
    fn the_cpuid_definitions_hide_everything_above_their_level() {
        let leaf1_ecx = |text: &str| -> u32 {
            let line = text.lines().find(|l| l.starts_with("00000001")).unwrap();
            u32::from_str_radix(line.split_whitespace().nth(5).unwrap(), 16).unwrap()
        };
        // SSE3 (bit 0), SSSE3 (9), SSE4.1 (19), SSE4.2 (20), AVX (28): none at sse2.
        let sse2 = leaf1_ecx(CPUID_SSE2);
        assert_eq!(sse2 & (1 | 1 << 9 | 1 << 19 | 1 << 20 | 1 << 28), 0);
        // OSXSAVE (27) keeps cpu_features off the host's /proc/cpuinfo.
        assert_ne!(sse2 & 1 << 27, 0);
        let v2 = leaf1_ecx(CPUID_SSE42);
        assert_eq!(
            v2 & (1 | 1 << 9 | 1 << 19 | 1 << 20 | 1 << 23),
            1 | 1 << 9 | 1 << 19 | 1 << 20 | 1 << 23
        );
        assert_eq!(v2 & (1 << 28 | 1 << 12), 0, "no AVX, no FMA");
    }
}
