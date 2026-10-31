//! Simulated cycles per sample (level 2 of the measurement policy, D33 in
//! `docs/decisions-log.md`, `docs/performance.md#measurement-policy`).
//!
//! The input is the execution count of every instruction of the counted calls and the count of
//! every taken jump, taken by callgrind with the differential method of `cargo xtask count` (a
//! run with `K` calls minus a run with none, [`difference`]). The hot code is disassembled with
//! `objdump` and simulated by `llvm-mca` on a core model: the cycles per iteration and the port
//! bound (`Block RThroughput`) of every simulated unit. Weighted by their execution counts and
//! scaled to all executed instructions, they give the cycles per sample and the port bound per
//! sample; the efficiency is their ratio.
//!
//! Two modes cut the code into simulated units ([`CycleMode`]):
//!
//! - **Whole loop** (the default): the control-flow graph of the basic blocks is rebuilt from the
//!   instruction and jump counts ([`cfg()`]); its loops are found as strongly connected components
//!   (nested loops inside them, [`find_loops`]); every loop body is simulated as one region per
//!   frequent path from its header back to the header ([`loop_paths`]: the blocks of one
//!   iteration in execution order, each path weighted by the product of its branch frequencies).
//!   A dependency chain that runs through several blocks of one iteration is simulated as it
//!   runs. On `znver3`/`znver4`, whose `llvm-mca` 23.1.1 models mishandle eliminated register
//!   moves, a path is also simulated with its moves written as `lea` and the form as compiled is
//!   only trusted within the bound the `lea` form gives ([`simulate_region`]). The blocks outside
//!   every loop are simulated as in the per-block mode.
//! - **Per block** (the method of D35): every basic block, or run of blocks with the same count
//!   that only cold code separates ([`regions`]), is simulated as a loop of its own.
//!
//! In both modes the CET markers (`endbr64`) of code built with `-fcf-protection` are simulated
//! as the `nop` they execute as, not as the microcoded instruction of the `znver3`/`znver4`
//! models ([`cet_markers_as_nop`]).
//!
//! Every unit is simulated as if it repeated back to back with itself. That is the steady state
//! of a loop body; for a block executed once per call it is an approximation, which the coverage
//! by loop bodies keeps small for the counted kernels. The blind spots of the method (caches,
//! memory dependencies, branch prediction, gathers, divisions and square roots) are listed in
//! `docs/performance.md`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, anyhow, bail};

/// The execution counts of a profile.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// `(object file, address in the file) -> executions` of every instruction.
    pub instructions: BTreeMap<(String, u64), u64>,
    /// `(object file, address of the jump, address of the target) -> times taken`.
    pub jumps: BTreeMap<(String, u64, u64), u64>,
    /// Executions that a subtraction ([`difference`]) left negative: instructions that ran more
    /// often in the subtracted run (the harness code of the restore-only run of an FFT). The
    /// count subtracts them, so the total of the profile does too ([`profile`]).
    pub deducted: i64,
}

/// The share of the executed instructions that the simulated units must cover.
pub const COVERAGE: f64 = 0.99;
/// At most this many units are simulated per profile.
pub const MAX_BLOCKS: usize = 64;
/// Iterations of every `llvm-mca` run.
pub const ITERATIONS: u32 = 200;
/// Addresses of one object that are further apart than this are disassembled separately.
pub const CLUSTER_GAP: u64 = 4096;
/// Blocks with the same count are joined into one loop body when the code between them, which
/// never ran, is at most this many bytes (the cold paths of a loop body: a saturation, an
/// error branch).
pub const REGION_GAP: u64 = 256;
/// The paths of one loop body are enumerated until they cover this share of its iterations.
pub const PATH_COVERAGE: f64 = 0.99;
/// At most this many paths are simulated per loop body.
pub const MAX_PATHS: usize = 8;
/// Paths less frequent than this share of the iterations of their loop are not simulated.
pub const MIN_PATH_SHARE: f64 = 0.005;
/// A loop body whose simulated paths cover less than this share of its iterations is simulated
/// block by block instead.
pub const MIN_LOOP_COVERAGE: f64 = 0.5;
/// A bound on the work of the path search of one loop (partial paths taken from the queue).
const MAX_EXPANSIONS: usize = 100_000;

/// How the hot code is cut into the units that `llvm-mca` simulates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CycleMode {
    /// One region per frequent path through each loop body (the default).
    #[default]
    WholeLoop,
    /// One region per basic block or run of same-count blocks (the method of D35).
    PerBlock,
}

impl CycleMode {
    /// Both modes, the default first.
    pub const ALL: [CycleMode; 2] = [CycleMode::WholeLoop, CycleMode::PerBlock];

    /// The name in the command line and the JSON (`cycle_mode`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            CycleMode::WholeLoop => "whole-loop",
            CycleMode::PerBlock => "per-block",
        }
    }

    /// The mode called `name`.
    #[must_use]
    pub fn parse(name: &str) -> Option<CycleMode> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }
}

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

/// Parses a callgrind output file written with `--dump-instr=yes --collect-jumps=yes
/// --compress-pos=no --compress-strings=no` into execution counts per instruction and per
/// taken jump. The cost line after a `calls=` line is the inclusive cost of the call and is
/// skipped; the position line after a `jump=` (unconditional: `jump=<taken> <target>`) or
/// `jcnd=` line (conditional: `jcnd=<taken>/<executed> <target>`) is the address of the jump.
#[must_use]
pub fn parse_callgrind(text: &str) -> Counts {
    let mut counts = Counts::default();
    let mut object = String::new();
    let mut call_cost = false;
    let mut jump: Option<(u64, u64)> = None;
    let hex = |s: &str| {
        s.strip_prefix("0x")
            .and_then(|a| u64::from_str_radix(a, 16).ok())
    };
    for line in text.lines() {
        if let Some(ob) = line.strip_prefix("ob=") {
            object = ob.trim().to_string();
        } else if line.starts_with("calls=") {
            call_cost = true;
        } else if let Some(rest) = line
            .strip_prefix("jump=")
            .or_else(|| line.strip_prefix("jcnd="))
        {
            let mut parts = rest.split_whitespace();
            let taken = parts
                .next()
                .and_then(|n| n.split('/').next())
                .and_then(|n| n.parse::<u64>().ok());
            let target = parts.next().and_then(hex);
            jump = taken.zip(target);
        } else if line.starts_with("0x") {
            let mut parts = line.split_whitespace();
            let address = parts.next().and_then(hex);
            if let Some((taken, target)) = jump.take() {
                if let Some(address) = address
                    && taken > 0
                {
                    *counts
                        .jumps
                        .entry((object.clone(), address, target))
                        .or_default() += taken;
                }
                continue;
            }
            if std::mem::take(&mut call_cost) {
                continue;
            }
            let ir = parts.nth(1).and_then(|n| n.parse::<u64>().ok());
            if let (Some(address), Some(ir)) = (address, ir) {
                *counts
                    .instructions
                    .entry((object.clone(), address))
                    .or_default() += ir;
            }
        }
    }
    counts
}

/// `full - empty` per instruction and per jump; entries whose difference is not positive are
/// left out, and the executions they lack are kept in [`Counts::deducted`], so that the total
/// is the signed difference of the totals, as in the count.
#[must_use]
pub fn difference(full: &Counts, empty: &Counts) -> Counts {
    Counts {
        instructions: minus(&full.instructions, &empty.instructions),
        jumps: minus(&full.jumps, &empty.jumps),
        deducted: full.deducted - empty.deducted
            + shortfall(&full.instructions, &empty.instructions),
    }
}

/// `sum of max(0, b - a)` over the keys of `b`: what `a - b` lacks where it is negative.
#[must_use]
pub fn shortfall<K: Ord>(a: &BTreeMap<K, u64>, b: &BTreeMap<K, u64>) -> i64 {
    b.iter()
        .map(|(key, &n)| n.saturating_sub(a.get(key).copied().unwrap_or(0)))
        .sum::<u64>()
        .try_into()
        .unwrap_or(i64::MAX)
}

/// `a - b` per key; keys whose difference is not positive are left out.
#[must_use]
pub fn minus<K: Ord + Clone>(a: &BTreeMap<K, u64>, b: &BTreeMap<K, u64>) -> BTreeMap<K, u64> {
    a.iter()
        .filter_map(|(key, &n)| {
            let d = n.saturating_sub(b.get(key).copied().unwrap_or(0));
            (d > 0).then(|| (key.clone(), d))
        })
        .collect()
}

/// The taken jumps of one object: `(address of the jump, target) -> times taken`.
pub type Jumps = BTreeMap<(u64, u64), u64>;

/// One instruction of a disassembly: address and the text for `llvm-mca`.
pub type Instruction = (u64, String);

/// The instruction set of an object, for its disassembly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// x86-64 (AT&T syntax of `objdump`).
    X86,
    /// AArch64.
    Aarch64,
}

impl Arch {
    /// The instruction set of the object file `path` (x86-64 unless its ELF header says
    /// AArch64).
    #[must_use]
    pub fn of(path: &str) -> Arch {
        if crate::blockcount::machine(path) == Some(crate::blockcount::EM_AARCH64) {
            Arch::Aarch64
        } else {
            Arch::X86
        }
    }
}

/// The label that every AArch64 branch target and literal address is written as: `llvm-mca`
/// ignores the control flow, and the assembler of `llvm-mca` takes no absolute address there.
pub const AARCH64_TARGET: &str = ".Lt";

/// Turns a line of `objdump -d --no-show-raw-insn` into an instruction for `llvm-mca`: the
/// comment and the symbol of a target are dropped, and an absolute branch target gets its `0x`.
#[must_use]
pub fn parse_objdump_line(line: &str) -> Option<Instruction> {
    parse_objdump_line_for(line, Arch::X86)
}

/// [`parse_objdump_line`] for the instruction set `arch`. On AArch64 the comment starts with
/// `//` (`#` marks an immediate), and an address operand (a branch target, the page of an
/// `adrp`, a literal) becomes [`AARCH64_TARGET`].
#[must_use]
pub fn parse_objdump_line_for(line: &str, arch: Arch) -> Option<Instruction> {
    if arch == Arch::Aarch64 {
        return parse_aarch64_line(line);
    }
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

fn parse_aarch64_line(line: &str) -> Option<Instruction> {
    let (address, text) = line.split_once(':')?;
    let address = u64::from_str_radix(address.trim(), 16).ok()?;
    let mut text = text.trim();
    if let Some(i) = text.find("//") {
        text = text[..i].trim_end();
    }
    let symbol = text.find(" <").or_else(|| text.find("\t<"));
    if let Some(i) = symbol {
        text = text[..i].trim_end();
    }
    if text.is_empty() || text.starts_with('(') || text.starts_with('.') {
        return None;
    }
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mnemonic = mnemonic(&text).to_string();
    let (head, last) = match text.rsplit_once(',') {
        Some((head, last)) => (format!("{head}, "), last.trim()),
        None => match text.split_once(' ') {
            Some((m, operand)) => (format!("{m} "), operand.trim()),
            None => return Some((address, text)),
        },
    };
    let is_address = !last.is_empty() && last.chars().all(|c| c.is_ascii_hexdigit());
    if is_address && (symbol.is_some() || is_transfer(&mnemonic)) {
        return Some((address, format!("{head}{AARCH64_TARGET}")));
    }
    Some((address, text))
}

/// Whether a mnemonic is a call or a return: control leaves the function, so a transfer from
/// it is no edge of the control-flow graph of the function.
#[must_use]
pub fn is_call_or_return(mnemonic: &str) -> bool {
    let m = mnemonic.trim_start_matches("notrack").trim();
    m.starts_with("call")
        || m.starts_with("ret")
        || matches!(m, "bl" | "blr" | "blraa" | "blrab" | "blraaz" | "blrabz")
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

/// Whether a block ending in `mnemonic` never continues with the next instruction (an
/// unconditional jump or a return; a call returns to the next instruction).
#[must_use]
pub fn ends_flow(mnemonic: &str) -> bool {
    let m = mnemonic.trim_start_matches("notrack").trim();
    m.starts_with("jmp") || m.starts_with("ret") || matches!(m, "b" | "br" | "ud2")
}

fn mnemonic(text: &str) -> &str {
    text.split_whitespace().next().unwrap_or_default()
}

/// Disassembles `[lo, hi]` of `object` with `objdump`.
///
/// # Errors
/// `objdump` cannot run.
pub fn disassemble(object: &str, lo: u64, hi: u64) -> anyhow::Result<Vec<Instruction>> {
    let arch = Arch::of(object);
    let tool = match arch {
        Arch::X86 => std::env::var_os("SDRK_OBJDUMP").unwrap_or_else(|| "objdump".into()),
        Arch::Aarch64 => std::env::var_os("SDRK_AARCH64_OBJDUMP")
            .unwrap_or_else(|| "aarch64-linux-gnu-objdump".into()),
    };
    let out = Command::new(tool)
        .args(["-d", "--no-show-raw-insn"])
        .arg(format!("--start-address={lo:#x}"))
        .arg(format!("--stop-address={:#x}", hi + 16))
        .arg(object)
        .stderr(Stdio::null())
        .output()
        .context("cannot run objdump")?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| parse_objdump_line_for(l, arch))
        .collect())
}

/// A basic block: its instructions and how often it ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The object file.
    pub object: String,
    /// Address of the first instruction.
    pub address: u64,
    /// Address of the last instruction.
    pub end: u64,
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
            end: *address,
            count,
            text: Vec::new(),
        });
        block.end = *address;
        block.text.push(text.clone());
        if is_transfer(mnemonic(text)) {
            out.extend(current.take());
        }
    }
    out.extend(current);
    out
}

/// Joins the basic blocks of one object (in address order) into loop bodies: consecutive blocks
/// with the same count whose gap holds only code that never ran ([`REGION_GAP`]). A loop body
/// with a branch inside (a saturation compiled to a jump over a cold path) is cut into several
/// basic blocks; simulated one by one, each would be a loop of its own, and the dependency chain
/// that runs through all of them would be lost. Joined, `llvm-mca` sees the whole body, which is
/// the path the loop takes when the cold paths do not run.
#[must_use]
pub fn regions(blocks: Vec<Block>) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    for b in blocks {
        if let Some(last) = out.last_mut()
            && last.object == b.object
            && last.count == b.count
            && b.address > last.end
            && b.address - last.end <= REGION_GAP
        {
            last.end = b.end;
            last.text.extend(b.text);
            continue;
        }
        out.push(b);
    }
    out
}

/// A node of the control-flow graph: a basic block that ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    /// Address of the first instruction.
    pub address: u64,
    /// Address of the last instruction.
    pub end: u64,
    /// Executions.
    pub count: u64,
    /// The instructions for `llvm-mca`.
    pub text: Vec<String>,
}

impl Node {
    fn block(&self, object: &str) -> Block {
        Block {
            object: object.to_string(),
            address: self.address,
            end: self.end,
            count: self.count,
            text: self.text.clone(),
        }
    }
}

/// An edge of the control-flow graph and how often it was followed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    /// Index of the source node.
    pub from: usize,
    /// Index of the target node.
    pub to: usize,
    /// Times followed.
    pub count: u64,
}

/// The control-flow graph of the executed code of one object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cfg {
    /// The basic blocks in address order.
    pub nodes: Vec<Node>,
    /// The edges with a positive count.
    pub edges: Vec<Edge>,
}

/// The successors of every node with the counts of the edges.
type Successors = BTreeMap<usize, Vec<(usize, u64)>>;

/// The node of an index that is not in the graph.
static NO_NODE: Node = Node {
    address: 0,
    end: 0,
    count: 0,
    text: Vec::new(),
};

impl Cfg {
    fn successors(&self) -> Successors {
        let mut out = Successors::new();
        for e in &self.edges {
            out.entry(e.from).or_default().push((e.to, e.count));
        }
        out
    }

    /// Node `v` (an empty node if there is none).
    #[must_use]
    pub fn node(&self, v: usize) -> &Node {
        self.nodes.get(v).unwrap_or(&NO_NODE)
    }
}

/// The successors of `v`.
fn next_of(succ: &Successors, v: usize) -> &[(usize, u64)] {
    succ.get(&v).map_or(&[], Vec::as_slice)
}

/// Rebuilds the control-flow graph of the executed instructions of one object: `code` is its
/// disassembly in address order (clusters may be concatenated: an instruction that never ran
/// separates them), `counts` the executions per address and `jumps` the taken jumps
/// `(address of the jump, target) -> count`. A block starts at a jump target, after a control
/// transfer, after an instruction that never ran and where the count changes. A taken jump is an
/// edge to its target; the remaining executions of a block that can fall through (also over a
/// call, which returns there) are an edge to the next block. Calls are not edges: a called
/// function is a graph of its own, and a transfer from a call or a return (which an emulator
/// that records every transfer between blocks reports) is left out.
#[must_use]
pub fn cfg(code: &[Instruction], counts: &BTreeMap<u64, u64>, jumps: &Jumps) -> Cfg {
    let calls: BTreeSet<u64> = code
        .iter()
        .filter(|(_, t)| is_call_or_return(mnemonic(t)))
        .map(|&(a, _)| a)
        .collect();
    let kept: Jumps = jumps
        .iter()
        .filter(|((from, _), _)| !calls.contains(from))
        .map(|(&k, &c)| (k, c))
        .collect();
    let jumps = &kept;
    let targets: BTreeSet<u64> = jumps.keys().map(|&(_, to)| to).collect();
    let mut nodes: Vec<Node> = Vec::new();
    // For every node: whether the instruction after its last one in `code` starts the next node.
    let mut adjacent: Vec<bool> = Vec::new();
    // Whether the previous instruction of `code` ran, and whether it was a control transfer.
    let (mut previous_ran, mut previous_transfer) = (false, false);
    for (address, text) in code {
        let count = counts.get(address).copied().unwrap_or(0);
        if count == 0 {
            previous_ran = false;
            continue;
        }
        let continues = previous_ran && !previous_transfer && !targets.contains(address);
        if let Some(n) = nodes.last_mut().filter(|n| continues && n.count == count) {
            n.end = *address;
            n.text.push(text.clone());
        } else {
            if let Some(a) = adjacent.last_mut() {
                *a = previous_ran;
            }
            nodes.push(Node {
                address: *address,
                end: *address,
                count,
                text: vec![text.clone()],
            });
            adjacent.push(false);
        }
        previous_ran = true;
        previous_transfer = is_transfer(mnemonic(text));
    }
    let start: BTreeMap<u64, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.address, i))
        .collect();
    let mut edges = Vec::new();
    for (i, (n, &adjacent)) in nodes.iter().zip(&adjacent).enumerate() {
        let mut taken = 0;
        for (&(_, to), &c) in jumps.range((n.end, 0)..=(n.end, u64::MAX)) {
            taken += c;
            if let Some(&j) = start.get(&to) {
                edges.push(Edge {
                    from: i,
                    to: j,
                    count: c,
                });
            }
        }
        let last = n.text.last().map_or("", |t| mnemonic(t));
        let fall = n.count.saturating_sub(taken);
        if adjacent && !ends_flow(last) && fall > 0 {
            edges.push(Edge {
                from: i,
                to: i + 1,
                count: fall,
            });
        }
    }
    Cfg { nodes, edges }
}

/// A loop of the control-flow graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loop {
    /// The node where an iteration starts: the node entered most often from outside the loop.
    pub header: usize,
    /// All nodes of the loop, those of the inner loops included.
    pub nodes: BTreeSet<usize>,
    /// The loops nested inside (found after the edges back to the header are removed).
    pub inner: Vec<Loop>,
}

/// The strongly connected components of the subgraph of `succ` induced by `set` (Tarjan).
fn components(set: &BTreeSet<usize>, succ: &Successors) -> Vec<Vec<usize>> {
    struct State<'a> {
        set: &'a BTreeSet<usize>,
        succ: &'a Successors,
        index: BTreeMap<usize, usize>,
        low: BTreeMap<usize, usize>,
        stack: Vec<usize>,
        on_stack: BTreeSet<usize>,
        out: Vec<Vec<usize>>,
    }
    fn visit(s: &mut State<'_>, v: usize) {
        let i = s.index.len();
        s.index.insert(v, i);
        s.low.insert(v, i);
        s.stack.push(v);
        s.on_stack.insert(v);
        for &(w, _) in next_of(s.succ, v) {
            if !s.set.contains(&w) {
                continue;
            }
            let reached = if s.index.contains_key(&w) {
                if !s.on_stack.contains(&w) {
                    continue;
                }
                s.index.get(&w).copied()
            } else {
                visit(s, w);
                s.low.get(&w).copied()
            };
            if let (Some(r), Some(lv)) = (reached, s.low.get_mut(&v)) {
                *lv = (*lv).min(r);
            }
        }
        if s.low.get(&v) == s.index.get(&v) {
            let mut c = Vec::new();
            while let Some(w) = s.stack.pop() {
                s.on_stack.remove(&w);
                c.push(w);
                if w == v {
                    break;
                }
            }
            c.sort_unstable();
            s.out.push(c);
        }
    }
    let mut s = State {
        set,
        succ,
        index: BTreeMap::new(),
        low: BTreeMap::new(),
        stack: Vec::new(),
        on_stack: BTreeSet::new(),
        out: Vec::new(),
    };
    for &v in set {
        if !s.index.contains_key(&v) {
            visit(&mut s, v);
        }
    }
    s.out
}

/// The loops of the subgraph of `g` induced by `set`, nested loops inside them: every strongly
/// connected component with a cycle is a loop; its header is the node entered most often from
/// outside the component (its executions minus the edges from inside; then the most executed,
/// then the lowest address); the inner loops are
/// the loops of the component without its header.
#[must_use]
pub fn find_loops(g: &Cfg, set: &BTreeSet<usize>) -> Vec<Loop> {
    let succ = g.successors();
    loops_in(g, &succ, set)
}

fn loops_in(g: &Cfg, succ: &Successors, set: &BTreeSet<usize>) -> Vec<Loop> {
    let mut out = Vec::new();
    for c in components(set, succ) {
        let nodes: BTreeSet<usize> = c.iter().copied().collect();
        let self_loop = |&v: &usize| next_of(succ, v).iter().any(|&(w, _)| w == v);
        let cyclic = c.len() > 1 || c.first().is_some_and(self_loop);
        if !cyclic {
            continue;
        }
        // The entries of a node: its executions not explained by edges from inside the
        // component (jumps from outside, and calls, which are not edges: a loop at the start
        // of a function is entered by its calls).
        let entries = |v: usize| -> u64 {
            let inside: u64 = g
                .edges
                .iter()
                .filter(|e| e.to == v && nodes.contains(&e.from))
                .map(|e| e.count)
                .sum();
            g.node(v).count.saturating_sub(inside)
        };
        let Some(header) = c.iter().copied().max_by(|&a, &b| {
            entries(a)
                .cmp(&entries(b))
                .then(g.node(a).count.cmp(&g.node(b).count))
                .then(g.node(b).address.cmp(&g.node(a).address))
        }) else {
            continue;
        };
        let mut rest = nodes.clone();
        rest.remove(&header);
        let inner = loops_in(g, succ, &rest);
        out.push(Loop {
            header,
            nodes,
            inner,
        });
    }
    out.sort_by_key(|l| g.node(l.header).address);
    out
}

/// One path through a loop body: its own nodes in execution order (the nodes of inner loops are
/// left out; they are simulated with their own loop) and its share of the iterations.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopPath {
    /// The nodes in execution order, the header first.
    pub nodes: Vec<usize>,
    /// The share of the iterations (starts at the header) that take this path.
    pub share: f64,
}

/// A step of a path through a loop body: `Ok(node)` for a node of the body itself, `Err(k)` for
/// its inner loop `k`.
type Step = Result<usize, usize>;

/// A partial path of the search in [`loop_paths`], ordered by its share.
#[derive(Debug)]
struct Partial {
    share: f64,
    /// The next step; `None`: the path is complete.
    at: Option<Step>,
    nodes: Vec<usize>,
}

impl PartialEq for Partial {
    fn eq(&self, other: &Self) -> bool {
        self.share.total_cmp(&other.share).is_eq()
    }
}
impl Eq for Partial {}
impl PartialOrd for Partial {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Partial {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.share.total_cmp(&other.share)
    }
}

/// The most frequent paths of one iteration of `l`, most frequent first: from the header along
/// the edges of the loop until an edge back to the header or out of the loop. An inner loop is
/// passed as one step whose exits are weighted by their counts. The share of a path is the
/// product of the frequencies of its branches (each edge count over the counts of all edges
/// leaving its node), which reproduces the execution count of every node of the body when the
/// branches are independent. The search stops at [`PATH_COVERAGE`] of the iterations or after
/// [`MAX_PATHS`] paths; paths below [`MIN_PATH_SHARE`] are not returned.
#[must_use]
pub fn loop_paths(g: &Cfg, l: &Loop) -> Vec<LoopPath> {
    // The inner loop of every node that belongs to one (the outermost inner loop).
    let mut inner_of: BTreeMap<usize, usize> = BTreeMap::new();
    for (k, inner) in l.inner.iter().enumerate() {
        for &v in &inner.nodes {
            inner_of.insert(v, k);
        }
    }
    // The steps of a path: `Ok(node)` for an own node, `Err(k)` for the inner loop `k`; an
    // edge to `None` ends the path (back to the header or out of the loop).
    let step = |v: usize| -> Step { inner_of.get(&v).map_or(Ok(v), |&k| Err(k)) };
    let mut out_edges: BTreeMap<Step, BTreeMap<Option<Step>, u64>> = BTreeMap::new();
    for e in &g.edges {
        if !l.nodes.contains(&e.from) {
            continue;
        }
        let from = step(e.from);
        let to = (e.to != l.header && l.nodes.contains(&e.to)).then(|| step(e.to));
        if from.is_err() && to == Some(from) {
            // An edge inside an inner loop.
            continue;
        }
        *out_edges.entry(from).or_default().entry(to).or_default() += e.count;
    }
    let mut queue = BinaryHeap::new();
    queue.push(Partial {
        share: 1.0,
        at: Some(Ok(l.header)),
        nodes: Vec::new(),
    });
    let mut paths = Vec::new();
    let mut found = 0.0;
    let mut expansions = 0;
    while let Some(p) = queue.pop() {
        // The queue gives the most frequent partial path first: every later one is rarer.
        if p.share < MIN_PATH_SHARE || expansions > MAX_EXPANSIONS {
            break;
        }
        let Some(at) = p.at else {
            found += p.share;
            paths.push(LoopPath {
                nodes: p.nodes,
                share: p.share,
            });
            if found >= PATH_COVERAGE || paths.len() >= MAX_PATHS {
                break;
            }
            continue;
        };
        expansions += 1;
        let mut nodes = p.nodes;
        if let Ok(v) = at {
            nodes.push(v);
        }
        let Some(next) = out_edges.get(&at) else {
            // No edge leaves: the path ends here (a return, or a jump the profile lost).
            queue.push(Partial {
                share: p.share,
                at: None,
                nodes,
            });
            continue;
        };
        let sum: u64 = next.values().sum();
        for (&to, &c) in next {
            queue.push(Partial {
                share: p.share * c as f64 / sum as f64,
                at: to,
                nodes: nodes.clone(),
            });
        }
    }
    paths
}

/// Writes the branch-target markers of CET (`endbr64`, `endbr32`; every function entry and
/// indirect-branch target of code built with `-fcf-protection`) as `nop`. They are hint
/// instructions that execute as a `nop` (one decode and dispatch slot, no execution port), but
/// `llvm-mca` 23.1.1 has no scheduling information of its own for them and takes the generic
/// `WriteSystem` class: 100 cycles of latency on every x86 model, and on `znver3` and `znver4`
/// also the microcoded write of 100 micro-operations on the integer ALUs, about 51 (`znver3`)
/// and 34 (`znver4`) cycles per execution. The `notrack` prefix of indirect jumps is modelled
/// as the plain jump and is kept. Returns the text and the number of rewritten markers.
#[must_use]
pub fn cet_markers_as_nop(text: &[String]) -> (Vec<String>, usize) {
    let mut markers = 0;
    let out = text
        .iter()
        .map(|t| {
            if matches!(t.trim(), "endbr64" | "endbr32") {
                markers += 1;
                "nop".to_string()
            } else {
                t.clone()
            }
        })
        .collect();
    (out, markers)
}

/// The models whose register-move elimination is not trusted in loop regions: the `znver3`
/// and `znver4` models of `llvm-mca` 23.1.1 lose the dependency of later instructions on an
/// eliminated `mov r32, r32` (the release path of the `agc` `CQ15` fallback: 24.7 instead of
/// 79.0 cycles on `znver3`), and an eliminated 64-bit move on a loop-carried chain can also
/// lengthen it (the same path with only the 32-bit move rewritten: 89.0). See
/// [`simulate_region`].
pub const MOVE_CHECK_MODELS: [&str; 2] = ["znver3", "znver4"];

/// Rewrites the register-to-register moves of general registers (`mov %eax,%ecx`,
/// `mov %rax,%rcx`) as `lea (%rax),%ecx` / `lea (%rax),%rcx`: the same result, but no move that
/// the renamer eliminates, so the model keeps every dependency through it. Each `lea` costs one
/// cycle of latency and one ALU slot, which hardware with move elimination does not pay. Other
/// instructions are kept. Returns the text and the number of rewritten moves.
#[must_use]
pub fn keep_move_dependencies(text: &[String]) -> (Vec<String>, usize) {
    let mut moves = 0;
    let out = text
        .iter()
        .map(|t| {
            move_as_lea(t).map_or_else(
                || t.clone(),
                |lea| {
                    moves += 1;
                    lea
                },
            )
        })
        .collect();
    (out, moves)
}

/// `lea` for a move between two general registers of the same width (32 or 64 bits).
fn move_as_lea(text: &str) -> Option<String> {
    let (m, operands) = text.split_once(' ')?;
    if !matches!(m, "mov" | "movl" | "movq") {
        return None;
    }
    let (src, dst) = operands.split_once(',')?;
    let (src, dst) = (src.trim().strip_prefix('%')?, dst.trim().strip_prefix('%')?);
    let base = gpr64(src)?;
    gpr64(dst)?;
    (gpr32(src) == gpr32(dst)).then(|| format!("lea (%{base}),%{dst}"))
}

/// The 64-bit name of a 32- or 64-bit general register.
fn gpr64(r: &str) -> Option<String> {
    const LEGACY: [&str; 8] = ["ax", "bx", "cx", "dx", "si", "di", "bp", "sp"];
    if let Some(base) = r.strip_prefix('r').or_else(|| r.strip_prefix('e'))
        && LEGACY.contains(&base)
    {
        return Some(format!("r{base}"));
    }
    let n = r.strip_prefix('r')?;
    let digits = n.strip_suffix('d').unwrap_or(n);
    let v: u32 = digits.parse().ok()?;
    (8..=15).contains(&v).then(|| format!("r{v}"))
}

/// Whether a general register name (as accepted by [`gpr64`]) is a 32-bit one.
fn gpr32(r: &str) -> bool {
    r.starts_with('e') || (r.starts_with('r') && r.ends_with('d'))
}

/// A unit that `llvm-mca` simulates: a region of instructions and how often it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    /// The object file.
    pub object: String,
    /// Address of the first instruction.
    pub address: u64,
    /// Address of the last instruction.
    pub end: u64,
    /// How often the region runs (a path of a loop: the iterations that take it).
    pub weight: f64,
    /// The instructions for `llvm-mca`.
    pub text: Vec<String>,
    /// What the unit is (`block`, `loop 0x.. path 1/2 (73.0 %)`), for `SDRK_MCA_DEBUG`.
    pub label: String,
    /// Whether the unit is a path of a loop body (whole-loop mode), whose register moves are
    /// checked on [`MOVE_CHECK_MODELS`] ([`simulate_region`]).
    pub loop_path: bool,
}

impl Unit {
    /// Executed instructions of the unit.
    #[must_use]
    pub fn executed(&self) -> f64 {
        self.weight * self.text.len() as f64
    }

    fn of_block(b: Block) -> Unit {
        Unit {
            object: b.object,
            address: b.address,
            end: b.end,
            weight: b.count as f64,
            text: b.text,
            label: "block".to_string(),
            loop_path: false,
        }
    }
}

/// The units of a profile in one mode.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// The mode.
    pub mode: CycleMode,
    /// The units, the most executed instructions first.
    pub units: Vec<Unit>,
    /// All executed instructions of the profile (in the objects that exist).
    pub total: f64,
    /// Loop bodies simulated as regions (whole-loop mode; 0 in per-block mode).
    pub loops: usize,
}

/// The units of the loops of `g` (inner loops first) and of the blocks outside every loop.
fn loop_units(object: &str, g: &Cfg) -> (Vec<Unit>, usize) {
    fn walk(object: &str, g: &Cfg, l: &Loop, out: &mut Vec<Unit>, loops: &mut usize) {
        for inner in &l.inner {
            walk(object, g, inner, out, loops);
        }
        let paths = loop_paths(g, l);
        if paths.iter().map(|p| p.share).sum::<f64>() < MIN_LOOP_COVERAGE {
            // No few paths stand for the body (many independent branches): its own blocks
            // are simulated one by one, as in the per-block mode.
            let nested: BTreeSet<usize> = l
                .inner
                .iter()
                .flat_map(|i| i.nodes.iter().copied())
                .collect();
            let own: Vec<usize> = l
                .nodes
                .iter()
                .copied()
                .filter(|v| !nested.contains(v))
                .collect();
            out.extend(blocks_apart(object, g, &own));
            return;
        }
        *loops += 1;
        let n = paths.len();
        let iterations = g.node(l.header).count as f64;
        for (k, p) in paths.into_iter().enumerate() {
            let text: Vec<String> = p
                .nodes
                .iter()
                .flat_map(|&v| g.node(v).text.iter().cloned())
                .collect();
            let end = p.nodes.iter().map(|&v| g.node(v).end).max().unwrap_or(0);
            out.push(Unit {
                object: object.to_string(),
                address: g.node(l.header).address,
                end,
                weight: iterations * p.share,
                text,
                loop_path: true,
                label: format!(
                    "loop {:#x} path {}/{n} ({:.1} %, blocks {})",
                    g.node(l.header).address,
                    k + 1,
                    100.0 * p.share,
                    p.nodes
                        .iter()
                        .map(|&v| format!("{:#x}", g.node(v).address))
                        .collect::<Vec<_>>()
                        .join(" ")
                ),
            });
        }
    }
    let all: BTreeSet<usize> = (0..g.nodes.len()).collect();
    let loops = find_loops(g, &all);
    let mut out = Vec::new();
    let mut count = 0;
    for l in &loops {
        walk(object, g, l, &mut out, &mut count);
    }
    let in_loop: BTreeSet<usize> = loops.iter().flat_map(|l| l.nodes.iter().copied()).collect();
    let rest: Vec<usize> = (0..g.nodes.len())
        .filter(|i| !in_loop.contains(i))
        .collect();
    out.extend(blocks_apart(object, g, &rest));
    (out, count)
}

/// The nodes `nodes` (indices in address order) as units of the per-block mode: runs of
/// consecutive nodes are joined by [`regions`], but never across a node left out (a loop
/// between the code before and after it).
fn blocks_apart(object: &str, g: &Cfg, nodes: &[usize]) -> Vec<Unit> {
    let mut out = Vec::new();
    let mut run: Vec<Block> = Vec::new();
    let mut last: Option<usize> = None;
    for &v in nodes {
        if last.is_some_and(|l| l + 1 != v) {
            out.extend(
                regions(std::mem::take(&mut run))
                    .into_iter()
                    .map(Unit::of_block),
            );
        }
        run.push(g.node(v).block(object));
        last = Some(v);
    }
    out.extend(regions(run).into_iter().map(Unit::of_block));
    out
}

/// The units of a profile in `mode`, the most executed instructions first.
///
/// # Errors
/// `objdump` cannot run.
pub fn profile(counts: &Counts, mode: CycleMode) -> anyhow::Result<Profile> {
    let mut per_object: BTreeMap<&str, BTreeMap<u64, u64>> = BTreeMap::new();
    for ((object, address), &n) in &counts.instructions {
        per_object
            .entry(object.as_str())
            .or_default()
            .insert(*address, n);
    }
    let mut units = Vec::new();
    let mut total = 0.0;
    let mut loops = 0;
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
        let mut code = Vec::new();
        for (lo, hi) in ranges {
            let part = disassemble(object, lo, hi)?;
            total += blocks(object, &part, &addresses)
                .iter()
                .map(|b| b.executed() as f64)
                .sum::<f64>();
            match mode {
                CycleMode::PerBlock => units.extend(
                    regions(blocks(object, &part, &addresses))
                        .into_iter()
                        .map(Unit::of_block),
                ),
                CycleMode::WholeLoop => code.extend(part),
            }
        }
        if mode == CycleMode::WholeLoop {
            let jumps: Jumps = counts
                .jumps
                .range((object.to_string(), 0, 0)..=(object.to_string(), u64::MAX, u64::MAX))
                .map(|((_, from, to), &c)| ((*from, *to), c))
                .collect();
            let (u, l) = loop_units(object, &cfg(&code, &addresses, &jumps));
            units.extend(u);
            loops += l;
        }
    }
    units.sort_by(|a, b| {
        b.executed()
            .total_cmp(&a.executed())
            .then(a.address.cmp(&b.address))
    });
    // The executions a subtraction left negative belong to the counted total (with the mean
    // cycles per instruction of the simulated units, [`simulate`]).
    let total = (total - counts.deducted as f64).max(0.0);
    Ok(Profile {
        mode,
        units,
        total,
        loops,
    })
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

/// Simulates one block on `cpu`, its CET markers written as `nop` ([`cet_markers_as_nop`]).
///
/// # Errors
/// `llvm-mca` cannot run or rejects the block.
pub fn simulate_block(text: &[String], triple: &str, cpu: &str) -> anyhow::Result<BlockTiming> {
    use std::io::Write as _;
    let (text, _) = cet_markers_as_nop(text);
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
    let mut source = if triple.starts_with("aarch64") {
        format!("{AARCH64_TARGET}:\n")
    } else {
        String::new()
    };
    source.push_str(&text.join("\n"));
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

/// Simulates a loop path on `cpu`. On the models of [`MOVE_CHECK_MODELS`] a path with register
/// moves is simulated twice: as compiled and with the moves written as `lea`
/// ([`keep_move_dependencies`]). The `lea` form keeps every dependency and is at most one cycle
/// per rewritten move slower than real move elimination, so the true cycles lie in
/// `[lea - moves, lea]`. The form as compiled is taken when it lies in that range (the model
/// handled the moves); otherwise the model mishandled an eliminated move and the `lea` form is
/// taken. Returns the timing and whether the `lea` form was taken.
///
/// # Errors
/// `llvm-mca` cannot run or rejects the path.
pub fn simulate_region(
    text: &[String],
    triple: &str,
    cpu: &str,
) -> anyhow::Result<(BlockTiming, bool)> {
    let compiled = simulate_block(text, triple, cpu)?;
    if !MOVE_CHECK_MODELS.contains(&cpu) {
        return Ok((compiled, false));
    }
    let (rewritten, moves) = keep_move_dependencies(text);
    if moves == 0 {
        return Ok((compiled, false));
    }
    let lea = simulate_block(&rewritten, triple, cpu)?;
    Ok(choose_move_form(compiled, lea, moves))
}

/// The choice of [`simulate_region`]: the form as compiled if its cycles lie within
/// `[lea - moves, lea]` (a small tolerance for the rounding of the simulation), else the `lea`
/// form (`true`).
#[must_use]
pub fn choose_move_form(
    compiled: BlockTiming,
    lea: BlockTiming,
    moves: usize,
) -> (BlockTiming, bool) {
    let tolerance = 0.05;
    let low = lea.cycles - moves as f64 - tolerance;
    if compiled.cycles >= low && compiled.cycles <= lea.cycles + tolerance {
        (compiled, false)
    } else {
        (lea, true)
    }
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
    /// The mode the units were cut in.
    pub mode: CycleMode,
    /// Executed instructions per sample (the count of `cargo xtask count`).
    pub instructions_per_sample: f64,
    /// Simulated cycles per sample.
    pub cycles_per_sample: f64,
    /// The port bound per sample.
    pub port_bound_per_sample: f64,
    /// `port bound / cycles`.
    pub efficiency: f64,
    /// Share of the executed instructions in the simulated units.
    pub coverage: f64,
    /// Number of simulated units (blocks, or loop paths and blocks).
    pub blocks: usize,
    /// Loop bodies simulated as regions (whole-loop mode).
    pub loops: usize,
    /// Loop paths simulated with their register moves written as `lea` ([`simulate_region`]).
    pub moves_as_lea: usize,
    /// Units `llvm-mca` rejected (address: reason); their instructions count as uncovered.
    pub rejected: Vec<String>,
}

/// Simulates the hot units of a profile on `cpu` and weights them by their executions.
///
/// # Errors
/// `llvm-mca` cannot run, or no unit could be simulated.
pub fn simulate(
    profile: &Profile,
    samples: u64,
    triple: &str,
    cpu: &str,
) -> anyhow::Result<Simulation> {
    let total = profile.total;
    if total <= 0.0 || samples == 0 {
        bail!("the profile is empty");
    }
    let (mut covered, mut seen, mut cycles, mut port, mut simulated) = (0.0, 0.0, 0.0, 0.0, 0);
    let mut moves_as_lea = 0;
    let mut rejected = Vec::new();
    for u in profile.units.iter().take(MAX_BLOCKS) {
        if seen >= COVERAGE * total {
            break;
        }
        seen += u.executed();
        let timing = if u.loop_path {
            simulate_region(&u.text, triple, cpu)
        } else {
            simulate_block(&u.text, triple, cpu).map(|t| (t, false))
        };
        if std::env::var_os("SDRK_MCA_DEBUG").is_some() {
            eprintln!(
                "mca {cpu} {}: {:#x}..{:#x} {} ({} instructions x {:.1}) -> {:?}",
                profile.mode.name(),
                u.address,
                u.end,
                u.label,
                u.text.len(),
                u.weight,
                timing
                    .as_ref()
                    .map(|(t, lea)| format!(
                        "{:.2} cycles, port bound {:.2}{}",
                        t.cycles,
                        t.port_bound,
                        if *lea { ", moves as lea" } else { "" }
                    ))
                    .map_err(ToString::to_string)
            );
            if std::env::var_os("SDRK_MCA_DEBUG").is_some_and(|v| v == "2") {
                for t in &u.text {
                    eprintln!("    {t}");
                }
            }
        }
        match timing {
            Ok((t, lea)) => {
                moves_as_lea += usize::from(lea);
                covered += u.executed();
                cycles += u.weight * t.cycles;
                port += u.weight * t.port_bound;
                simulated += 1;
            }
            Err(e) => rejected.push(format!("{:#x}: {e:#}", u.address)),
        }
    }
    if covered <= 0.0 {
        bail!(
            "llvm-mca could not simulate any block: {}",
            rejected.join("; ")
        );
    }
    // The instructions outside the simulated units get the cycles per instruction of the
    // simulated ones.
    let scale = total / covered;
    let samples = samples as f64;
    Ok(Simulation {
        cpu: cpu.to_string(),
        mode: profile.mode,
        instructions_per_sample: total / samples,
        cycles_per_sample: cycles * scale / samples,
        port_bound_per_sample: port * scale / samples,
        efficiency: port / cycles,
        coverage: (covered / total).min(1.0),
        blocks: simulated,
        loops: profile.loops,
        moves_as_lea,
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
jcnd=3/5 0x10 0
0x18 0
jump=2 0x30 0
0x1c 0
0x1c 0 2
ob=/lib/b.so
fn=h
0x20 0 3
";

    #[test]
    fn callgrind_counts_skip_the_inclusive_cost_of_calls() {
        let c = parse_callgrind(CALLGRIND);
        let i = &c.instructions;
        assert_eq!(i.get(&("/bin/a".to_string(), 0x14)), Some(&5));
        assert_eq!(i.get(&("/bin/a".to_string(), 0x18)), Some(&5));
        assert_eq!(i.get(&("/bin/a".to_string(), 0x1c)), Some(&2));
        assert_eq!(i.get(&("/lib/b.so".to_string(), 0x20)), Some(&3));
        assert_eq!(c.jumps.get(&("/bin/a".to_string(), 0x18, 0x10)), Some(&3));
        assert_eq!(c.jumps.get(&("/bin/a".to_string(), 0x1c, 0x30)), Some(&2));
        let mut empty = Counts::default();
        empty.instructions.insert(("/bin/a".to_string(), 0x10), 5);
        empty.jumps.insert(("/bin/a".to_string(), 0x18, 0x10), 1);
        empty.instructions.insert(("/bin/a".to_string(), 0x40), 4);
        let d = difference(&c, &empty);
        assert!(!d.instructions.contains_key(&("/bin/a".to_string(), 0x10)));
        assert_eq!(d.instructions.len(), 4);
        // 0x40 ran only in the subtracted run: the total keeps it negative.
        assert_eq!(d.deducted, 4);
        assert_eq!(difference(&d, &d).deducted, 0);
        assert_eq!(d.jumps.get(&("/bin/a".to_string(), 0x18, 0x10)), Some(&2));
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
        assert!(ends_flow("jmp") && ends_flow("ret") && !ends_flow("jne") && !ends_flow("call"));
    }

    #[test]
    fn aarch64_lines_keep_immediates_and_label_their_addresses() {
        let a = |l: &str| parse_objdump_line_for(l, Arch::Aarch64).map(|(_, t)| t);
        assert_eq!(
            a("   40040:\tadd\tsp, sp, #0x3d0"),
            Some("add sp, sp, #0x3d0".into())
        );
        assert_eq!(
            a("   40054:\tldp\td15, d14, [sp], #80"),
            Some("ldp d15, d14, [sp], #80".into())
        );
        assert_eq!(
            a("    4a3c:\tb.ne\t4a20 <f+0x10>  // b.any"),
            Some("b.ne .Lt".into())
        );
        assert_eq!(
            a("    4a40:\tcbz\tw0, 4a20 <f+0x10>"),
            Some("cbz w0, .Lt".into())
        );
        assert_eq!(
            a("    4a44:\ttbnz\tw1, #3, 4a20 <f+0x10>"),
            Some("tbnz w1, #3, .Lt".into())
        );
        assert_eq!(a("    4a48:\tbl\t9000 <g>"), Some("bl .Lt".into()));
        assert_eq!(
            a("    4a4c:\tadrp\tx0, 2ae000 <h>"),
            Some("adrp x0, .Lt".into())
        );
        assert_eq!(a("    4a50:\tfmov\td0, d15"), Some("fmov d0, d15".into()));
        assert_eq!(a("    4a54:\tret"), Some("ret".into()));
        assert_eq!(a("0000000000040000 <f>:"), None);
        assert!(is_call_or_return("bl") && is_call_or_return("callq") && is_call_or_return("ret"));
        assert!(!is_call_or_return("b") && !is_call_or_return("blsr") && !is_call_or_return("jne"));
    }

    #[test]
    fn transfers_from_calls_and_returns_are_no_edges() {
        // A loop 0x10..0x18 whose body calls; an emulator reports the call as a transfer.
        let code: Vec<Instruction> = [
            (0x10, "add x0, x0, #1"),
            (0x14, "bl .Lt"),
            (0x18, "b.ne .Lt"),
            (0x1c, "ret"),
        ]
        .iter()
        .map(|&(a, t)| (a, t.to_string()))
        .collect();
        let counts: BTreeMap<u64, u64> = [(0x10, 10), (0x14, 10), (0x18, 10), (0x1c, 1)].into();
        let jumps: Jumps = [((0x14, 0x100), 10), ((0x18, 0x10), 9), ((0x1c, 0x500), 1)].into();
        let g = cfg(&code, &counts, &jumps);
        let all: BTreeSet<usize> = (0..g.nodes.len()).collect();
        let loops = find_loops(&g, &all);
        assert_eq!(loops.len(), 1);
        // The block of the call falls through to the branch back.
        assert!(g.edges.iter().any(|e| g.node(e.from).address == 0x10
            && g.node(e.to).address == 0x18
            && e.count == 10));
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
        // The first two blocks have the same count and are adjacent: one loop body.
        let r = regions(b);
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].address, r[0].end, r[0].text.len()), (0, 3, 4));
    }

    #[test]
    fn regions_do_not_join_distant_or_different_blocks() {
        let block = |address: u64, end: u64, count: u64| Block {
            object: "o".to_string(),
            address,
            end,
            count,
            text: vec!["x".to_string()],
        };
        let r = regions(vec![
            block(0, 10, 5),
            block(20, 30, 5),
            block(30 + REGION_GAP + 1, 400, 5),
            block(410, 420, 6),
        ]);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].text.len(), 2);
    }

    fn code(lines: &[(u64, &str)]) -> Vec<Instruction> {
        lines.iter().map(|(a, t)| (*a, (*t).to_string())).collect()
    }

    /// A rotated loop with a branch inside (the code of `gcc -O2` for
    /// `for (...) if (x & 1) s += 3 x; else s ^= x;`), 100 iterations, 48 odd values.
    fn diamond() -> (Vec<Instruction>, BTreeMap<u64, u64>, Jumps) {
        let c = code(&[
            (0x00, "test %esi,%esi"),
            (0x02, "jle 0x48"),
            (0x04, "xor %edx,%edx"),
            (0x06, "jmp 0x2e"),
            (0x08, "nop"),
            (0x20, "lea (%rax,%rax,2),%eax"),
            (0x23, "add $0x4,%rdi"),
            (0x27, "add %eax,%edx"),
            (0x29, "cmp %rcx,%rdi"),
            (0x2c, "je 0x3f"),
            (0x2e, "mov (%rdi),%eax"),
            (0x30, "test $0x1,%al"),
            (0x32, "jne 0x20"),
            (0x34, "add $0x4,%rdi"),
            (0x38, "xor %eax,%edx"),
            (0x3a, "cmp %rcx,%rdi"),
            (0x3d, "jne 0x2e"),
            (0x3f, "mov %edx,%eax"),
            (0x41, "ret"),
            (0x48, "xor %edx,%edx"),
        ]);
        let mut counts: BTreeMap<u64, u64> = BTreeMap::new();
        for (a, n) in [
            (0x00, 1),
            (0x02, 1),
            (0x04, 1),
            (0x06, 1),
            (0x20, 48),
            (0x23, 48),
            (0x27, 48),
            (0x29, 48),
            (0x2c, 48),
            (0x2e, 100),
            (0x30, 100),
            (0x32, 100),
            (0x34, 52),
            (0x38, 52),
            (0x3a, 52),
            (0x3d, 52),
            (0x3f, 1),
            (0x41, 1),
        ] {
            counts.insert(a, n);
        }
        let jumps: Jumps = [
            ((0x06, 0x2e), 1),
            ((0x2c, 0x3f), 1),
            ((0x32, 0x20), 48),
            ((0x3d, 0x2e), 52),
        ]
        .into();
        (c, counts, jumps)
    }

    #[test]
    fn the_control_flow_graph_follows_the_jumps() {
        let (c, counts, jumps) = diamond();
        let g = cfg(&c, &counts, &jumps);
        let starts: Vec<u64> = g.nodes.iter().map(|n| n.address).collect();
        assert_eq!(starts, [0x00, 0x04, 0x20, 0x2e, 0x34, 0x3f]);
        let edge = |from: u64, to: u64| {
            let f = starts.iter().position(|&a| a == from).unwrap();
            let t = starts.iter().position(|&a| a == to).unwrap();
            g.edges
                .iter()
                .find(|e| e.from == f && e.to == t)
                .map(|e| e.count)
        };
        assert_eq!(edge(0x00, 0x04), Some(1));
        assert_eq!(edge(0x04, 0x2e), Some(1));
        // The unconditional jump does not fall through, the never-taken `jle` does.
        assert_eq!(edge(0x04, 0x20), None);
        assert_eq!(edge(0x20, 0x2e), Some(47));
        assert_eq!(edge(0x20, 0x3f), Some(1));
        assert_eq!(edge(0x2e, 0x20), Some(48));
        assert_eq!(edge(0x2e, 0x34), Some(52));
        assert_eq!(edge(0x34, 0x2e), Some(52));
        assert_eq!(edge(0x34, 0x3f), None);
    }

    #[test]
    fn a_loop_with_a_branch_is_one_body_with_weighted_paths() {
        let (c, counts, jumps) = diamond();
        let g = cfg(&c, &counts, &jumps);
        let all: BTreeSet<usize> = (0..g.nodes.len()).collect();
        let loops = find_loops(&g, &all);
        assert_eq!(loops.len(), 1);
        let l = &loops[0];
        // The rotated loop is entered at 0x2e (the jump of the prologue), not at 0x20.
        assert_eq!(g.node(l.header).address, 0x2e);
        assert_eq!(l.nodes.len(), 3);
        assert_eq!(l.inner, []);
        let paths = loop_paths(&g, l);
        assert_eq!(paths.len(), 2);
        let addresses =
            |p: &LoopPath| -> Vec<u64> { p.nodes.iter().map(|&v| g.node(v).address).collect() };
        assert_eq!(addresses(&paths[0]), [0x2e, 0x34]);
        assert!((paths[0].share - 0.52).abs() < 1e-12);
        assert_eq!(addresses(&paths[1]), [0x2e, 0x20]);
        // 48 of 100 iterations take the odd side (one of them leaves the loop at its end: the
        // same instructions, so the same path).
        assert!((paths[1].share - 0.48).abs() < 1e-12);
        // Weighted by the iterations of the header the paths give the count of every block.
        let executed = |a: u64| -> f64 {
            paths
                .iter()
                .filter(|p| addresses(p).contains(&a))
                .map(|p| 100.0 * p.share)
                .sum()
        };
        assert!((executed(0x34) - 52.0).abs() < 1e-9);
        assert!((executed(0x20) - 48.0).abs() < 1e-9);
        let (units, n) = loop_units("o", &g);
        assert_eq!(n, 1);
        // Two paths, the prologue (two blocks with the same count joined as in the per-block
        // mode) and the epilogue (not joined to the prologue across the loop).
        assert_eq!(units.len(), 4);
        assert_eq!(
            units[0].text,
            [
                "mov (%rdi),%eax",
                "test $0x1,%al",
                "jne 0x20",
                "add $0x4,%rdi",
                "xor %eax,%edx",
                "cmp %rcx,%rdi",
                "jne 0x2e"
            ]
        );
        assert!((units[0].weight - 52.0).abs() < 1e-9);
    }

    #[test]
    fn nested_loops_are_simulated_inner_first_and_apart() {
        // for i in 0..10 { a; for j in 0..20 { b } c }, the inner loop a self-loop at 0x10.
        let c = code(&[
            (0x00, "a0"),
            (0x01, "a1"),
            (0x10, "b0"),
            (0x11, "jne 0x10"),
            (0x20, "c0"),
            (0x21, "jne 0x00"),
            (0x22, "ret"),
        ]);
        let counts: BTreeMap<u64, u64> = [
            (0x00, 10),
            (0x01, 10),
            (0x10, 200),
            (0x11, 200),
            (0x20, 10),
            (0x21, 10),
            (0x22, 1),
        ]
        .into();
        let jumps: Jumps = [((0x11, 0x10), 190), ((0x21, 0x00), 9)].into();
        let g = cfg(&c, &counts, &jumps);
        let all: BTreeSet<usize> = (0..g.nodes.len()).collect();
        let loops = find_loops(&g, &all);
        assert_eq!(loops.len(), 1);
        let outer = &loops[0];
        assert_eq!(g.node(outer.header).address, 0x00);
        assert_eq!(outer.inner.len(), 1);
        let inner = &outer.inner[0];
        assert_eq!(g.node(inner.header).address, 0x10);
        assert_eq!(inner.nodes.len(), 1);
        // The outer path passes the inner loop as one step: a, then c.
        let paths = loop_paths(&g, outer);
        assert_eq!(paths.len(), 1);
        let a: Vec<u64> = paths[0].nodes.iter().map(|&v| g.node(v).address).collect();
        assert_eq!(a, [0x00, 0x20]);
        assert!((paths[0].share - 1.0).abs() < 1e-12);
        let ip = loop_paths(&g, inner);
        assert_eq!(ip.len(), 1);
        let (units, n) = loop_units("o", &g);
        assert_eq!(n, 2);
        let weights: Vec<(u64, f64)> = units.iter().map(|u| (u.address, u.weight)).collect();
        assert_eq!(weights[0], (0x10, 200.0));
        assert_eq!(weights[1], (0x00, 10.0));
        assert_eq!(units[1].text, ["a0", "a1", "c0", "jne 0x00"]);
    }

    #[test]
    fn rare_paths_are_left_out_and_the_search_is_bounded() {
        // Ten independent branches taken in half the iterations: 1024 paths of 1/1024 each.
        let mut lines = Vec::new();
        let mut counts = BTreeMap::new();
        let mut jumps = BTreeMap::new();
        for k in 0..10u64 {
            let a = 0x100 * k;
            lines.push((a, format!("jne {:#x}", a + 0x10)));
            lines.push((a + 1, "x".to_string()));
            lines.push((a + 0x10, "y".to_string()));
            counts.insert(a, 64);
            counts.insert(a + 1, 32);
            counts.insert(a + 0x10, 64);
            jumps.insert((a, a + 0x10), 32);
        }
        lines.push((0xa00, "jne 0x0".to_string()));
        counts.insert(0xa00, 64);
        jumps.insert((0xa00, 0), 63);
        let c: Vec<Instruction> = lines.into_iter().collect();
        let g = cfg(&c, &counts, &jumps);
        let all: BTreeSet<usize> = (0..g.nodes.len()).collect();
        let loops = find_loops(&g, &all);
        assert_eq!(loops.len(), 1);
        let paths = loop_paths(&g, &loops[0]);
        assert_eq!(paths, [], "every path is below MIN_PATH_SHARE");
        // The body is simulated block by block then.
        let (units, n) = loop_units("o", &g);
        assert_eq!(n, 0);
        assert!(units.iter().all(|u| u.label == "block"));
        let executed: f64 = units.iter().map(Unit::executed).sum();
        assert!((executed - (10.0 * (64.0 + 32.0 + 64.0) + 64.0)).abs() < 1e-9);
    }

    #[test]
    fn register_moves_become_lea() {
        let t: Vec<String> = [
            "mov %eax,%ecx",
            "mov %r8,%rdx",
            "mov %r9d,%esi",
            "mov %eax,%r10d",
            "mov (%rdi),%eax",
            "mov %eax,%rcx",
            "movaps %xmm0,%xmm1",
            "mov $0x1,%eax",
        ]
        .map(ToString::to_string)
        .to_vec();
        let (rewritten, moves) = keep_move_dependencies(&t);
        assert_eq!(moves, 4);
        assert_eq!(
            rewritten,
            [
                "lea (%rax),%ecx",
                "lea (%r8),%rdx",
                "lea (%r9),%esi",
                "lea (%rax),%r10d",
                "mov (%rdi),%eax",
                "mov %eax,%rcx",
                "movaps %xmm0,%xmm1",
                "mov $0x1,%eax",
            ]
        );
    }

    #[test]
    fn cet_markers_become_nop() {
        let t: Vec<String> = [
            "endbr64",
            "mov %edi,%eax",
            "endbr32",
            "notrack jmp *%rax",
            "nop",
            "ret",
        ]
        .map(ToString::to_string)
        .to_vec();
        let (rewritten, markers) = cet_markers_as_nop(&t);
        assert_eq!(markers, 2);
        assert_eq!(
            rewritten,
            [
                "nop",
                "mov %edi,%eax",
                "nop",
                "notrack jmp *%rax",
                "nop",
                "ret"
            ]
        );
        // Code without markers is simulated as it is.
        let plain = &t[1..2];
        assert_eq!(cet_markers_as_nop(plain), (plain.to_vec(), 0));
    }

    #[test]
    fn a_cet_marker_costs_no_more_than_a_nop() -> anyhow::Result<()> {
        if !llvm_mca().exists() {
            return Ok(());
        }
        let with: Vec<String> = ["endbr64", "add %edi,%eax", "imul %eax,%ecx"]
            .map(ToString::to_string)
            .to_vec();
        let without = with[1..].to_vec();
        for cpu in ["znver3", "znver4", "alderlake"] {
            // As compiled, `endbr64` would take about 51 (`znver3`) and 34 (`znver4`) cycles.
            let a = simulate_block(&with, "x86_64", cpu)?;
            let b = simulate_block(&without, "x86_64", cpu)?;
            assert!((a.cycles - b.cycles).abs() < 0.1, "{cpu}: {a:?} {b:?}");
        }
        Ok(())
    }

    #[test]
    fn the_move_form_is_trusted_within_the_bound_of_the_lea_form() {
        let t = |cycles: f64| BlockTiming {
            cycles,
            port_bound: 1.0,
        };
        // A lost dependency: far below the lea form.
        assert_eq!(choose_move_form(t(24.7), t(79.0), 6), (t(79.0), true));
        // A lengthened chain: above the lea form.
        assert_eq!(choose_move_form(t(89.0), t(79.0), 6), (t(79.0), true));
        // Move elimination as on hardware: at most one cycle per move faster.
        assert_eq!(choose_move_form(t(11.1), t(12.1), 9), (t(11.1), false));
        assert_eq!(choose_move_form(t(12.1), t(12.1), 1), (t(12.1), false));
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
        assert_eq!(CycleMode::parse("per-block"), Some(CycleMode::PerBlock));
        assert_eq!(CycleMode::default().name(), "whole-loop");
    }
}
