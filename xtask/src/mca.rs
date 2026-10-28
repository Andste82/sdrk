//! Simulated cycles per sample (level 2 of the measurement policy, D33 in
//! `docs/decisions-log.md`, `docs/performance.md#measurement-policy`).
//!
//! The input is the execution count of every instruction of the counted calls, taken by
//! callgrind with the differential method of `cargo xtask count` (a run with `K` calls minus a
//! run with none, [`difference`]). The instructions are grouped into basic blocks: runs of
//! consecutive instructions with the same count, cut after every control transfer
//! ([`blocks`]). The hot blocks (from the most executed instructions down, until
//! [`COVERAGE`] of the executed instructions are covered) are disassembled with `objdump` and
//! simulated by `llvm-mca` as a loop on a core model: the cycles per iteration and the port
//! bound (`Block RThroughput`). Weighted by their execution counts and scaled to all executed
//! instructions, they give the cycles per sample and the port bound per sample; the efficiency
//! is their ratio.
//!
//! Every block is simulated as if it repeated back to back with itself. That is the steady state
//! of a loop body; for a block executed once per call it is an approximation, which the coverage
//! by loop bodies keeps small for the counted kernels. The blind spots of the method (caches,
//! memory dependencies, branch prediction, gathers, divisions and square roots) are listed in
//! `docs/performance.md`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, anyhow, bail};

/// Execution counts per instruction: `(object file, address in the file) -> executions`.
pub type Counts = BTreeMap<(String, u64), u64>;

/// The share of the executed instructions that the simulated blocks must cover.
pub const COVERAGE: f64 = 0.99;
/// At most this many blocks are simulated per profile.
pub const MAX_BLOCKS: usize = 64;
/// Iterations of every `llvm-mca` run.
pub const ITERATIONS: u32 = 200;
/// Addresses of one object that are further apart than this are disassembled separately.
const CLUSTER_GAP: u64 = 4096;

/// The core models of an instruction-set level (D33): the reference model, on which a branch is
/// judged, and the counter-model of another vendor or core generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Models {
    /// The level (`sse2`, `avx2`, ...).
    pub level: &'static str,
    /// `-mtriple` of `llvm-mca`.
    pub triple: &'static str,
    /// The reference model (`-mcpu`).
    pub reference: &'static str,
    /// The counter-model (`-mcpu`).
    pub counter: &'static str,
}

/// The core models per level (D33). `avx2-runtime` uses the models of `avx2`.
pub const MODELS: [Models; 6] = [
    Models {
        level: "sse2",
        triple: "x86_64",
        reference: "skylake",
        counter: "znver2",
    },
    Models {
        level: "sse4.2",
        triple: "x86_64",
        reference: "skylake",
        counter: "znver2",
    },
    Models {
        level: "avx2",
        triple: "x86_64",
        reference: "znver3",
        counter: "alderlake",
    },
    Models {
        level: "avx2-runtime",
        triple: "x86_64",
        reference: "znver3",
        counter: "alderlake",
    },
    Models {
        level: "avx512",
        triple: "x86_64",
        reference: "znver4",
        counter: "sapphirerapids",
    },
    Models {
        level: "neon",
        triple: "aarch64",
        reference: "cortex-a72",
        counter: "cortex-a76",
    },
];

/// The models of `level`, if it has any (`simd128` cannot be simulated).
#[must_use]
pub fn models(level: &str) -> Option<&'static Models> {
    MODELS.iter().find(|m| m.level == level)
}

/// The `llvm-mca` to use: `SDRK_LLVM_MCA`, else the pinned build of the development container.
/// The `llvm-mca` of the distribution is another LLVM version and is never taken silently.
#[must_use]
pub fn llvm_mca() -> PathBuf {
    std::env::var_os("SDRK_LLVM_MCA").map_or_else(
        || PathBuf::from("/opt/llvm-mca-23.1.1/bin/llvm-mca"),
        PathBuf::from,
    )
}

/// Parses a callgrind output file written with `--dump-instr=yes --compress-pos=no
/// --compress-strings=no` into execution counts per instruction. The cost line after a
/// `calls=` line is the inclusive cost of the call and is skipped.
#[must_use]
pub fn parse_callgrind(text: &str) -> Counts {
    let mut counts = Counts::new();
    let mut object = String::new();
    let mut call_cost = false;
    for line in text.lines() {
        if let Some(ob) = line.strip_prefix("ob=") {
            object = ob.trim().to_string();
        } else if line.starts_with("calls=") {
            call_cost = true;
        } else if let Some(hex) = line.strip_prefix("0x") {
            if std::mem::take(&mut call_cost) {
                continue;
            }
            let mut parts = hex.split_whitespace();
            let address = parts.next().and_then(|a| u64::from_str_radix(a, 16).ok());
            let ir = parts.nth(1).and_then(|n| n.parse::<u64>().ok());
            if let (Some(address), Some(ir)) = (address, ir) {
                *counts.entry((object.clone(), address)).or_default() += ir;
            }
        }
    }
    counts
}

/// `full - empty` per instruction; instructions whose difference is not positive are left out.
#[must_use]
pub fn difference(full: &Counts, empty: &Counts) -> Counts {
    full.iter()
        .filter_map(|(key, &n)| {
            let d = n.saturating_sub(empty.get(key).copied().unwrap_or(0));
            (d > 0).then(|| (key.clone(), d))
        })
        .collect()
}

/// One instruction of a disassembly: address and the text for `llvm-mca`.
pub type Instruction = (u64, String);

/// Turns a line of `objdump -d --no-show-raw-insn` into an instruction for `llvm-mca`: the
/// comment and the symbol of a target are dropped, and an absolute branch target gets its `0x`.
#[must_use]
pub fn parse_objdump_line(line: &str) -> Option<Instruction> {
    let (address, text) = line.split_once(':')?;
    let address = u64::from_str_radix(address.trim(), 16).ok()?;
    let mut text = text.trim();
    if let Some(i) = text.find(" #") {
        text = text[..i].trim_end();
    }
    if let Some(i) = text.find(" <") {
        text = text[..i].trim_end();
    }
    if text.is_empty() || text.starts_with('(') {
        return None;
    }
    let mut words = text.splitn(2, char::is_whitespace);
    let mnemonic = words.next().unwrap_or_default();
    let operands = words.next().unwrap_or_default().trim();
    let is_target = !operands.is_empty() && operands.chars().all(|c| c.is_ascii_hexdigit());
    let text = if is_target && is_transfer(mnemonic) {
        format!("{mnemonic} 0x{operands}")
    } else {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    };
    Some((address, text))
}

/// Whether a mnemonic ends a basic block.
#[must_use]
pub fn is_transfer(mnemonic: &str) -> bool {
    // x86: `j*`, `call*`, `ret*`; AArch64: `b`, `b.<cond>`, `bl`, `br`, `blr`, `ret`, `cbz`,
    // `cbnz`, `tbz`, `tbnz`.
    let m = mnemonic.trim_start_matches("notrack").trim();
    m.starts_with('j')
        || m.starts_with("call")
        || m.starts_with("ret")
        || m.starts_with("b.")
        || matches!(
            m,
            "b" | "bl" | "br" | "blr" | "cbz" | "cbnz" | "tbz" | "tbnz"
        )
}

/// Disassembles `[lo, hi]` of `object` with `objdump`.
///
/// # Errors
/// `objdump` cannot run.
pub fn disassemble(object: &str, lo: u64, hi: u64) -> anyhow::Result<Vec<Instruction>> {
    let out = Command::new(std::env::var_os("SDRK_OBJDUMP").unwrap_or_else(|| "objdump".into()))
        .args(["-d", "--no-show-raw-insn"])
        .arg(format!("--start-address={lo:#x}"))
        .arg(format!("--stop-address={:#x}", hi + 16))
        .arg(object)
        .stderr(Stdio::null())
        .output()
        .context("cannot run objdump")?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(parse_objdump_line)
        .collect())
}

/// A basic block: its instructions and how often it ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The object file.
    pub object: String,
    /// Address of the first instruction.
    pub address: u64,
    /// Executions.
    pub count: u64,
    /// The instructions for `llvm-mca`.
    pub text: Vec<String>,
}

impl Block {
    /// Executed instructions of the block.
    #[must_use]
    pub fn executed(&self) -> u64 {
        self.count * self.text.len() as u64
    }
}

/// Cuts the disassembly of one object into basic blocks: runs of consecutive executed
/// instructions with the same count, cut after a control transfer.
#[must_use]
pub fn blocks(object: &str, code: &[Instruction], counts: &BTreeMap<u64, u64>) -> Vec<Block> {
    let mut out = Vec::new();
    let mut current: Option<Block> = None;
    for (address, text) in code {
        let count = counts.get(address).copied().unwrap_or(0);
        if let Some(b) = current.take() {
            if b.count == count {
                current = Some(b);
            } else {
                out.push(b);
            }
        }
        if count == 0 {
            continue;
        }
        let block = current.get_or_insert_with(|| Block {
            object: object.to_string(),
            address: *address,
            count,
            text: Vec::new(),
        });
        block.text.push(text.clone());
        let mnemonic = text.split_whitespace().next().unwrap_or_default();
        if is_transfer(mnemonic) {
            out.extend(current.take());
        }
    }
    out.extend(current);
    out
}

/// The basic blocks of a profile, the most executed instructions first.
///
/// # Errors
/// `objdump` cannot run.
pub fn profile_blocks(counts: &Counts) -> anyhow::Result<Vec<Block>> {
    let mut per_object: BTreeMap<&str, BTreeMap<u64, u64>> = BTreeMap::new();
    for ((object, address), &n) in counts {
        per_object
            .entry(object.as_str())
            .or_default()
            .insert(*address, n);
    }
    let mut out = Vec::new();
    for (object, addresses) in per_object {
        if object == "???" || !Path::new(object).exists() {
            continue;
        }
        let mut cluster: Option<(u64, u64)> = None;
        let mut ranges = Vec::new();
        for &a in addresses.keys() {
            cluster = match cluster {
                Some((lo, hi)) if a - hi <= CLUSTER_GAP => Some((lo, a)),
                Some(range) => {
                    ranges.push(range);
                    Some((a, a))
                }
                None => Some((a, a)),
            };
        }
        ranges.extend(cluster);
        for (lo, hi) in ranges {
            let code = disassemble(object, lo, hi)?;
            out.extend(blocks(object, &code, &addresses));
        }
    }
    out.sort_by(|a, b| {
        b.executed()
            .cmp(&a.executed())
            .then(a.address.cmp(&b.address))
    });
    Ok(out)
}

/// What `llvm-mca` says about one block on one model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockTiming {
    /// Simulated cycles per iteration.
    pub cycles: f64,
    /// The port bound per iteration (`Block RThroughput`).
    pub port_bound: f64,
}

/// Reads the cycles per iteration and the port bound from the summary of `llvm-mca`.
#[must_use]
pub fn parse_mca(summary: &str) -> Option<BlockTiming> {
    let value = |key: &str| -> Option<f64> {
        summary
            .lines()
            .find_map(|l| l.trim().strip_prefix(key))
            .and_then(|v| v.trim().parse::<f64>().ok())
    };
    let iterations = value("Iterations:")?;
    let cycles = value("Total Cycles:")?;
    let port_bound = value("Block RThroughput:")?;
    (iterations > 0.0).then(|| BlockTiming {
        cycles: cycles / iterations,
        port_bound,
    })
}

/// Simulates one block on `cpu`.
///
/// # Errors
/// `llvm-mca` cannot run or rejects the block.
pub fn simulate_block(text: &[String], triple: &str, cpu: &str) -> anyhow::Result<BlockTiming> {
    use std::io::Write as _;
    let mca = llvm_mca();
    let mut child = Command::new(&mca)
        .arg(format!("-mtriple={triple}"))
        .arg(format!("-mcpu={cpu}"))
        .arg(format!("-iterations={ITERATIONS}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("cannot run {} (set SDRK_LLVM_MCA)", mca.display()))?;
    let mut source = text.join("\n");
    source.push('\n');
    child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("no stdin"))?
        .write_all(source.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "llvm-mca rejected the block: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .next()
                .unwrap_or_default()
        );
    }
    parse_mca(&String::from_utf8_lossy(&out.stdout))
        .ok_or_else(|| anyhow!("no summary in the output of llvm-mca"))
}

/// The bottleneck class of a simulated profile (`docs/performance.md`).
#[must_use]
pub fn bottleneck(efficiency: f64) -> &'static str {
    if efficiency >= 0.8 {
        "throughput"
    } else {
        "latency"
    }
}

/// The simulation of one profile on one model.
#[derive(Clone, Debug, PartialEq)]
pub struct Simulation {
    /// The model (`-mcpu`).
    pub cpu: String,
    /// Executed instructions per sample (the count of `cargo xtask count`).
    pub instructions_per_sample: f64,
    /// Simulated cycles per sample.
    pub cycles_per_sample: f64,
    /// The port bound per sample.
    pub port_bound_per_sample: f64,
    /// `port bound / cycles`.
    pub efficiency: f64,
    /// Share of the executed instructions in the simulated blocks.
    pub coverage: f64,
    /// Number of simulated blocks.
    pub blocks: usize,
    /// Blocks `llvm-mca` rejected (address: reason); their instructions count as uncovered.
    pub rejected: Vec<String>,
}

/// Simulates the hot blocks of a profile on `cpu` and weights them by their executions.
///
/// # Errors
/// `llvm-mca` cannot run, or no block could be simulated.
pub fn simulate(
    blocks: &[Block],
    samples: u64,
    triple: &str,
    cpu: &str,
) -> anyhow::Result<Simulation> {
    let total: u64 = blocks.iter().map(Block::executed).sum();
    if total == 0 || samples == 0 {
        bail!("the profile is empty");
    }
    let (mut covered, mut seen, mut cycles, mut port, mut simulated) = (0u64, 0u64, 0.0, 0.0, 0);
    let mut rejected = Vec::new();
    for b in blocks.iter().take(MAX_BLOCKS) {
        if seen as f64 >= COVERAGE * total as f64 {
            break;
        }
        seen += b.executed();
        match simulate_block(&b.text, triple, cpu) {
            Ok(t) => {
                covered += b.executed();
                cycles += b.count as f64 * t.cycles;
                port += b.count as f64 * t.port_bound;
                simulated += 1;
            }
            Err(e) => rejected.push(format!("{:#x}: {e:#}", b.address)),
        }
    }
    if covered == 0 {
        bail!(
            "llvm-mca could not simulate any block: {}",
            rejected.join("; ")
        );
    }
    // The instructions outside the simulated blocks get the cycles per instruction of the
    // simulated ones.
    let scale = total as f64 / covered as f64;
    let samples = samples as f64;
    Ok(Simulation {
        cpu: cpu.to_string(),
        instructions_per_sample: total as f64 / samples,
        cycles_per_sample: cycles * scale / samples,
        port_bound_per_sample: port * scale / samples,
        efficiency: port / cycles,
        coverage: covered as f64 / total as f64,
        blocks: simulated,
        rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALLGRIND: &str = "\
positions: instr line
events: Ir
ob=/bin/a
fn=f
0x10 0 5
0x14 0 5
cfn=g
calls=5 0x40 0
0x14 0 999
0x18 0 5
ob=/lib/b.so
fn=h
0x20 0 3
";

    #[test]
    fn callgrind_counts_skip_the_inclusive_cost_of_calls() {
        let c = parse_callgrind(CALLGRIND);
        assert_eq!(c.get(&("/bin/a".to_string(), 0x14)), Some(&5));
        assert_eq!(c.get(&("/bin/a".to_string(), 0x18)), Some(&5));
        assert_eq!(c.get(&("/lib/b.so".to_string(), 0x20)), Some(&3));
        let mut empty = Counts::new();
        empty.insert(("/bin/a".to_string(), 0x10), 5);
        let d = difference(&c, &empty);
        assert!(!d.contains_key(&("/bin/a".to_string(), 0x10)));
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn objdump_lines_become_mca_input() {
        assert_eq!(
            parse_objdump_line("  38a560:\tvmovss 0x10(%rip),%xmm0        # 38b000 <c>"),
            Some((0x0038_a560, "vmovss 0x10(%rip),%xmm0".to_string()))
        );
        assert_eq!(
            parse_objdump_line("  38a570:\tjne    38a560 <f+0x10>"),
            Some((0x0038_a570, "jne 0x38a560".to_string()))
        );
        assert_eq!(parse_objdump_line("000000000038a560 <f>:"), None);
        assert!(is_transfer("ret") && is_transfer("b.ne") && !is_transfer("bic"));
    }

    #[test]
    fn blocks_follow_the_counts_and_end_at_transfers() {
        let code: Vec<Instruction> = [
            (0, "x0"),
            (1, "x1"),
            (2, "jne 0x0"),
            (3, "x3"),
            (4, "x4"),
            (5, "x5"),
        ]
        .iter()
        .map(|(a, t)| (*a, (*t).to_string()))
        .collect();
        let counts: BTreeMap<u64, u64> = [(0, 8), (1, 8), (2, 8), (3, 8), (4, 1)].into();
        let b = blocks("o", &code, &counts);
        assert_eq!(b.len(), 3);
        assert_eq!(b[0].text.len(), 3);
        assert_eq!((b[1].address, b[1].count), (3, 8));
        assert_eq!((b[2].address, b[2].count), (4, 1));
    }

    #[test]
    fn the_mca_summary_is_read() {
        let s = "Iterations:        200\nInstructions:      800\nTotal Cycles:      1003\nBlock RThroughput: 4.5\n";
        assert_eq!(
            parse_mca(s),
            Some(BlockTiming {
                cycles: 5.015,
                port_bound: 4.5
            })
        );
        assert_eq!(bottleneck(0.9), "throughput");
        assert_eq!(bottleneck(0.3), "latency");
        assert_eq!(models("avx2").map(|m| m.reference), Some("znver3"));
        assert!(models("simd128").is_none());
    }
}
