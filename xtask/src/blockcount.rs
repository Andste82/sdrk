//! Execution counts per instruction from the emulators of `cargo xtask count`, for the
//! simulated cycles of the levels that callgrind cannot run (D68 in `docs/decisions-log.md`).
//!
//! - **Intel SDE** (`sse2`, `sse4.2`, `avx512`): the dynamic control-flow graph of `-dcfg`
//!   ([`parse_dcfg`]): the execution count of every basic block of every image of the process
//!   (the measurement binary, the static references inside it, VOLK's shared library, libc)
//!   and the count of every edge between blocks. The taken branches are the jumps of
//!   [`crate::mca::Counts`]. Blocks are given as offsets in their image and become addresses
//!   of the object file with its first load segment ([`Elf::link_base`]). The block counts of
//!   two runs are subtracted per block ([`difference`]) and only then expanded into
//!   instructions with the disassembly ([`expand`]).
//! - **QEMU** (`neon`): the plugin `xtask/plugins/bbcount.c` ([`parse_bbcount`]): the
//!   execution count of every translation block, the control transfers between blocks that
//!   are not a fall-through, and the file mappings of the QEMU process. The guest code is
//!   mapped from its files at `guest_base + address`, so the mappings give the object and the
//!   file offset of every guest address, in the binary and in the shared libraries.
//!
//! The result is a [`crate::mca::Counts`] as callgrind gives it: the rest of the method
//! ([`crate::mca::profile`], [`crate::mca::simulate`]) is the same for every level.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, anyhow, bail};
use serde_json::Value;

use crate::mca::Counts;

/// The ELF machine number of AArch64.
pub const EM_AARCH64: u16 = 183;
/// The ELF machine number of x86-64.
pub const EM_X86_64: u16 = 62;

/// A load segment of an ELF file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    /// Offset in the file.
    pub offset: u64,
    /// Virtual address in the file.
    pub vaddr: u64,
    /// Bytes in the file.
    pub filesz: u64,
}

/// What the cycle tools need of an ELF file: the machine and the load segments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Elf {
    /// `e_machine`.
    pub machine: u16,
    /// The `PT_LOAD` segments in file order.
    pub loads: Vec<Segment>,
}

impl Elf {
    /// Parses the header and the program headers of a 64-bit little-endian ELF file.
    ///
    /// # Errors
    /// Not such a file, or truncated.
    pub fn parse(bytes: &[u8]) -> anyhow::Result<Elf> {
        let u16_at = |at: usize| -> anyhow::Result<u16> {
            let b = bytes
                .get(at..at + 2)
                .ok_or_else(|| anyhow!("truncated ELF"))?;
            Ok(u16::from_le_bytes(b.try_into()?))
        };
        let u32_at = |at: usize| -> anyhow::Result<u32> {
            let b = bytes
                .get(at..at + 4)
                .ok_or_else(|| anyhow!("truncated ELF"))?;
            Ok(u32::from_le_bytes(b.try_into()?))
        };
        let u64_at = |at: usize| -> anyhow::Result<u64> {
            let b = bytes
                .get(at..at + 8)
                .ok_or_else(|| anyhow!("truncated ELF"))?;
            Ok(u64::from_le_bytes(b.try_into()?))
        };
        if bytes.get(..4) != Some(b"\x7fELF".as_slice())
            || bytes.get(4) != Some(&2)
            || bytes.get(5) != Some(&1)
        {
            bail!("not a 64-bit little-endian ELF file");
        }
        let machine = u16_at(18)?;
        let phoff = usize::try_from(u64_at(32)?)?;
        let phentsize = usize::from(u16_at(54)?);
        let phnum = usize::from(u16_at(56)?);
        let mut loads = Vec::new();
        for i in 0..phnum {
            let ph = phoff + i * phentsize;
            if u32_at(ph)? == 1 {
                loads.push(Segment {
                    offset: u64_at(ph + 8)?,
                    vaddr: u64_at(ph + 16)?,
                    filesz: u64_at(ph + 32)?,
                });
            }
        }
        Ok(Elf { machine, loads })
    }

    /// The address of the first load segment: the address in the file of offset 0 from the
    /// image start that SDE reports.
    #[must_use]
    pub fn link_base(&self) -> u64 {
        self.loads.iter().map(|s| s.vaddr).min().unwrap_or(0)
    }

    /// The address in the file of the byte at `offset` of the file, if a load segment holds it.
    #[must_use]
    pub fn address_of_offset(&self, offset: u64) -> Option<u64> {
        self.loads
            .iter()
            .find(|s| offset >= s.offset && offset - s.offset < s.filesz)
            .map(|s| s.vaddr + (offset - s.offset))
    }
}

/// The ELF facts of `path`, read once per process.
///
/// # Errors
/// The file cannot be read or is no 64-bit little-endian ELF file.
pub fn elf(path: &str) -> anyhow::Result<Elf> {
    static CACHE: OnceLock<Mutex<BTreeMap<String, Elf>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(c) = cache.lock()
        && let Some(e) = c.get(path)
    {
        return Ok(e.clone());
    }
    // The headers are at the start of the file; 64 KiB hold them for every object here.
    let mut bytes = Vec::new();
    {
        use std::io::Read as _;
        std::fs::File::open(path)
            .with_context(|| format!("cannot open {path}"))?
            .take(1 << 16)
            .read_to_end(&mut bytes)?;
    }
    let e = Elf::parse(&bytes).with_context(|| path.to_string())?;
    if let Ok(mut c) = cache.lock() {
        c.insert(path.to_string(), e.clone());
    }
    Ok(e)
}

/// The machine of the object `path`, `None` if it is no readable ELF file.
#[must_use]
pub fn machine(path: &str) -> Option<u16> {
    elf(path).ok().map(|e| e.machine)
}

/// Execution counts per basic block: `(object, address, bytes, instructions) -> executions`,
/// and the taken jumps as in [`Counts::jumps`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockCounts {
    /// The blocks.
    pub blocks: BTreeMap<(String, u64, u64, u64), u64>,
    /// `(object, address of the jump, target) -> times taken`.
    pub jumps: BTreeMap<(String, u64, u64), u64>,
    /// Instructions a subtraction left negative (see [`Counts::deducted`]).
    pub deducted: i64,
}

impl BlockCounts {
    /// Executed instructions.
    #[must_use]
    pub fn instructions(&self) -> u64 {
        self.blocks
            .iter()
            .map(|((_, _, _, n), c)| n * c)
            .sum::<u64>()
    }
}

/// `full - empty` per block and per jump (the blocks of two runs of one program have the same
/// boundaries: the runs differ in how often the counted calls run, not in the code).
#[must_use]
pub fn difference(full: &BlockCounts, empty: &BlockCounts) -> BlockCounts {
    let lacking: u64 = empty
        .blocks
        .iter()
        .map(|(key, &c)| key.3 * c.saturating_sub(full.blocks.get(key).copied().unwrap_or(0)))
        .sum();
    BlockCounts {
        blocks: crate::mca::minus(&full.blocks, &empty.blocks),
        jumps: crate::mca::minus(&full.jumps, &empty.jumps),
        deducted: full.deducted - empty.deducted + i64::try_from(lacking).unwrap_or(i64::MAX),
    }
}

/// The rows of a DCFG table (`[[header...], [row...], ...]`) as maps from the header names.
fn rows(table: &Value) -> Vec<BTreeMap<&str, &Value>> {
    let Some(all) = table.as_array() else {
        return Vec::new();
    };
    let Some((header, rest)) = all.split_first() else {
        return Vec::new();
    };
    let names: Vec<&str> = header
        .as_array()
        .map(|h| h.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    rest.iter()
        .filter_map(Value::as_array)
        .map(|r| names.iter().copied().zip(r.iter()).collect())
        .collect()
}

fn number(v: Option<&&Value>) -> Option<u64> {
    let v = v?;
    v.as_u64().or_else(|| {
        v.as_str()
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
    })
}

/// A count of a DCFG: a number, or one number per thread.
fn count(v: Option<&&Value>) -> u64 {
    match v {
        Some(Value::Array(per_thread)) => per_thread.iter().filter_map(Value::as_u64).sum(),
        Some(v) => v.as_u64().unwrap_or(0),
        None => 0,
    }
}

/// Whether an edge type of a DCFG is a taken branch (not a fall-through, call, return, `REP`
/// iteration or system call).
#[must_use]
pub fn is_branch_edge(edge_type: &str) -> bool {
    edge_type.ends_with("BRANCH")
}

/// Parses the DCFG of Intel SDE (`-dcfg`, JSON) into block and jump counts of every process.
/// `link_base(object)` is the address in the file of the image start (see [`Elf::link_base`]).
///
/// # Errors
/// Not a DCFG, or a block refers to an unknown file.
#[allow(
    clippy::large_stack_frames,
    reason = "serde_json values and the unoptimized maps are large; xtask is a host tool"
)]
pub fn parse_dcfg(text: &str, link_base: impl Fn(&str) -> u64) -> anyhow::Result<BlockCounts> {
    let d: Value = serde_json::from_str(text).context("the DCFG is no JSON")?;
    let field = |v: &Value, k: &str| v.get(k).cloned().unwrap_or(Value::Null);
    let files: BTreeMap<u64, String> = rows(&field(&d, "FILE_NAMES"))
        .into_iter()
        .filter_map(|r| {
            Some((
                number(r.get("FILE_NAME_ID"))?,
                r.get("FILE_NAME")?.as_str()?.to_string(),
            ))
        })
        .collect();
    let edge_types: BTreeMap<u64, String> = rows(&field(&d, "EDGE_TYPES"))
        .into_iter()
        .filter_map(|r| {
            Some((
                number(r.get("EDGE_TYPE_ID"))?,
                r.get("EDGE_TYPE")?.as_str()?.to_string(),
            ))
        })
        .collect();
    if files.is_empty() {
        bail!("the DCFG has no FILE_NAMES");
    }
    let mut out = BlockCounts::default();
    let processes = field(&d, "PROCESSES");
    let processes = processes.as_array().map(Vec::as_slice).unwrap_or_default();
    for process in processes.iter().skip(1) {
        let Some(data) = process.as_array().and_then(|p| p.get(1)) else {
            continue;
        };
        // node -> (object, first address, address of the last instruction)
        let mut nodes: BTreeMap<u64, (String, u64, u64)> = BTreeMap::new();
        for image in rows(&field(data, "IMAGES")) {
            let Some(image_data) = image.get("IMAGE_DATA") else {
                continue;
            };
            let id = number(Some(&&field(image_data, "FILE_NAME_ID")))
                .ok_or_else(|| anyhow!("an image without FILE_NAME_ID"))?;
            let object = files
                .get(&id)
                .ok_or_else(|| anyhow!("unknown FILE_NAME_ID {id}"))?;
            let base = link_base(object);
            for b in rows(&field(image_data, "BASIC_BLOCKS")) {
                let (Some(node), Some(offset), Some(size), Some(n)) = (
                    number(b.get("NODE_ID")),
                    number(b.get("ADDR_OFFSET")),
                    number(b.get("SIZE")),
                    number(b.get("NUM_INSTRS")),
                ) else {
                    continue;
                };
                let last = number(b.get("LAST_INSTR_OFFSET")).unwrap_or(0);
                let address = base + offset;
                nodes.insert(node, (object.clone(), address, address + last));
                let c = count(b.get("COUNT"));
                if c > 0 {
                    *out.blocks
                        .entry((object.clone(), address, size, n))
                        .or_default() += c;
                }
            }
        }
        for e in rows(&field(data, "EDGES")) {
            let ty = number(e.get("EDGE_TYPE_ID"))
                .and_then(|t| edge_types.get(&t))
                .map_or("", String::as_str);
            if !is_branch_edge(ty) {
                continue;
            }
            let c = count(e.get("COUNT_PER_THREAD"));
            let from = number(e.get("SOURCE_NODE_ID")).and_then(|n| nodes.get(&n));
            let to = number(e.get("TARGET_NODE_ID")).and_then(|n| nodes.get(&n));
            if let (Some((object, _, last)), Some((target_object, target, _))) = (from, to)
                && object == target_object
                && c > 0
            {
                *out.jumps
                    .entry((object.clone(), *last, *target))
                    .or_default() += c;
            }
        }
    }
    Ok(out)
}

/// Expands block counts into instruction counts with the disassembly of the blocks: every
/// instruction of a block gets its count (summed where blocks overlap).
///
/// # Errors
/// `objdump` cannot run, or the disassembly of a block does not have the number of instructions
/// the counter reported.
pub fn expand(blocks: &BlockCounts) -> anyhow::Result<Counts> {
    let mut per_object: BTreeMap<&str, Vec<(u64, u64, u64, u64)>> = BTreeMap::new();
    for ((object, address, size, n), &c) in &blocks.blocks {
        per_object
            .entry(object.as_str())
            .or_default()
            .push((*address, *size, *n, c));
    }
    let mut out = Counts {
        instructions: BTreeMap::new(),
        jumps: blocks.jumps.clone(),
        deducted: blocks.deducted,
    };
    let mut mismatched = Vec::new();
    for (object, list) in per_object {
        if !Path::new(object).exists() {
            continue;
        }
        // One disassembly per cluster of blocks.
        let mut clusters: Vec<(u64, u64)> = Vec::new();
        for &(a, size, _, _) in &list {
            let end = a + size;
            match clusters.last_mut() {
                Some((_, hi)) if a <= *hi + crate::mca::CLUSTER_GAP => *hi = (*hi).max(end),
                _ => clusters.push((a, end)),
            }
        }
        let mut addresses: Vec<u64> = Vec::new();
        for (lo, hi) in clusters {
            addresses.extend(
                crate::mca::disassemble(object, lo, hi)?
                    .into_iter()
                    .map(|(a, _)| a)
                    .filter(|&a| a < hi),
            );
        }
        addresses.sort_unstable();
        addresses.dedup();
        for (a, size, n, c) in list {
            let lo = addresses.partition_point(|&x| x < a);
            let hi = addresses.partition_point(|&x| x < a + size);
            let inside = addresses.get(lo..hi).unwrap_or_default();
            if inside.len() as u64 != n {
                mismatched.push(format!(
                    "{object} {a:#x}: {} instructions disassembled, {n} counted",
                    inside.len()
                ));
                continue;
            }
            for &i in inside {
                *out.instructions.entry((object.to_string(), i)).or_default() += c;
            }
        }
    }
    if !mismatched.is_empty() {
        bail!(
            "the disassembly does not match the counted blocks: {}",
            mismatched.join("; ")
        );
    }
    Ok(out)
}

/// One file mapping of the QEMU process.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Mapping {
    lo: u64,
    hi: u64,
    offset: u64,
    path: String,
}

/// Parses the output of the plugin `bbcount` into instruction counts per object, with the
/// addresses of the object files. `address_of(object, offset)` is the address in the file of
/// the byte at `offset` (see [`Elf::address_of_offset`]). Blocks outside every file mapping
/// (none for the code of an ordinary process) are left out. Every instruction is taken to be
/// `bytes / instructions` long, as on AArch64.
///
/// # Errors
/// No output of the plugin, or a malformed line.
pub fn parse_bbcount(
    text: &str,
    address_of: impl Fn(&str, u64) -> Option<u64>,
) -> anyhow::Result<Counts> {
    let hex = |s: Option<&str>| -> anyhow::Result<u64> {
        let s = s.ok_or_else(|| anyhow!("a short bbcount line"))?;
        u64::from_str_radix(s, 16).with_context(|| format!("not a hex number: {s}"))
    };
    let dec = |s: Option<&str>| -> anyhow::Result<u64> {
        let s = s.ok_or_else(|| anyhow!("a short bbcount line"))?;
        s.parse().with_context(|| format!("not a number: {s}"))
    };
    let mut seen = false;
    let mut guest_base = 0u64;
    let mut maps: Vec<Mapping> = Vec::new();
    let mut tbs: Vec<(u64, u64, u64, u64)> = Vec::new();
    let mut transfers: Vec<(u64, u64, u64)> = Vec::new();
    for line in text.lines() {
        let mut w = line.split_whitespace();
        match w.next() {
            Some("bbcount") => seen = true,
            Some("guest_base") => guest_base = hex(w.next())?,
            Some("map") => {
                let (lo, hi, offset) = (hex(w.next())?, hex(w.next())?, hex(w.next())?);
                let path = w.collect::<Vec<_>>().join(" ");
                maps.push(Mapping {
                    lo,
                    hi,
                    offset,
                    path,
                });
            }
            Some("tb") => tbs.push((
                hex(w.next())?,
                dec(w.next())?,
                dec(w.next())?,
                dec(w.next())?,
            )),
            Some("jump") => transfers.push((hex(w.next())?, hex(w.next())?, dec(w.next())?)),
            _ => {}
        }
    }
    if !seen {
        bail!("no output of the bbcount plugin");
    }
    maps.sort_by_key(|m| m.lo);
    let locate = |guest: u64| -> Option<(&str, u64)> {
        let host = guest.checked_add(guest_base)?;
        let i = maps.partition_point(|m| m.lo <= host).checked_sub(1)?;
        let m = maps.get(i)?;
        (host < m.hi)
            .then(|| address_of(&m.path, host - m.lo + m.offset).map(|a| (m.path.as_str(), a)))
            .flatten()
    };
    let mut out = Counts::default();
    for (vaddr, n, bytes, c) in tbs {
        if n == 0 || c == 0 {
            continue;
        }
        let Some((object, address)) = locate(vaddr) else {
            continue;
        };
        let size = bytes / n;
        for i in 0..n {
            *out.instructions
                .entry((object.to_string(), address + i * size))
                .or_default() += c;
        }
    }
    for (from, to, c) in transfers {
        if let (Some((object, a)), Some((target_object, b))) = (locate(from), locate(to))
            && object == target_object
        {
            *out.jumps.entry((object.to_string(), a, b)).or_default() += c;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elf_bytes(machine: u16, loads: &[(u64, u64, u64)]) -> Vec<u8> {
        let mut b = vec![0u8; 64];
        b[..4].copy_from_slice(b"\x7fELF");
        b[4] = 2;
        b[5] = 1;
        b[18..20].copy_from_slice(&machine.to_le_bytes());
        b[32..40].copy_from_slice(&64u64.to_le_bytes());
        b[54..56].copy_from_slice(&56u16.to_le_bytes());
        b[56..58].copy_from_slice(&(loads.len() as u16 + 1).to_le_bytes());
        // A PT_PHDR first, which is no load segment.
        let mut ph = vec![0u8; 56];
        ph[..4].copy_from_slice(&6u32.to_le_bytes());
        b.extend(ph);
        for &(offset, vaddr, filesz) in loads {
            let mut ph = vec![0u8; 56];
            ph[..4].copy_from_slice(&1u32.to_le_bytes());
            ph[8..16].copy_from_slice(&offset.to_le_bytes());
            ph[16..24].copy_from_slice(&vaddr.to_le_bytes());
            ph[32..40].copy_from_slice(&filesz.to_le_bytes());
            b.extend(ph);
        }
        b
    }

    #[test]
    fn elf_load_segments_map_offsets_to_addresses() {
        let e = Elf::parse(&elf_bytes(
            EM_AARCH64,
            &[(0, 0, 0x1000), (0x1000, 0x11000, 0x500)],
        ))
        .unwrap();
        assert_eq!(e.machine, EM_AARCH64);
        assert_eq!(e.loads.len(), 2);
        assert_eq!(e.link_base(), 0);
        assert_eq!(e.address_of_offset(0x40), Some(0x40));
        assert_eq!(e.address_of_offset(0x1010), Some(0x11010));
        assert_eq!(e.address_of_offset(0x2000), None);
        assert!(Elf::parse(b"not an elf").is_err());
        let fixed = Elf::parse(&elf_bytes(EM_X86_64, &[(0, 0x40_0000, 0x1000)])).unwrap();
        assert_eq!(fixed.link_base(), 0x40_0000);
    }

    const DCFG: &str = r#"{
      "MAJOR_VERSION": 1, "MINOR_VERSION": 0,
      "FILE_NAMES": [["FILE_NAME_ID", "FILE_NAME"], [1, "/bin/a"], [2, "/lib/v.so"]],
      "EDGE_TYPES": [["EDGE_TYPE_ID", "EDGE_TYPE"], [3, "CALL"], [6, "RETURN"],
                     [13, "DIRECT_CONDITIONAL_BRANCH"], [15, "DIRECT_UNCONDITIONAL_BRANCH"],
                     [18, "FALL_THROUGH"]],
      "SPECIAL_NODES": [["NODE_ID", "NODE_NAME"], [1, "START"], [2, "END"], [3, "UNKNOWN"]],
      "PROCESSES": [["PROCESS_ID", "PROCESS_DATA"], [42, {
        "INSTR_COUNT": 100,
        "IMAGES": [["IMAGE_ID", "LOAD_ADDR", "SIZE", "IMAGE_DATA"],
          [1, "0x5555000", 4096, {"FILE_NAME_ID": 1, "BASIC_BLOCKS": [
            ["NODE_ID", "ADDR_OFFSET", "SIZE", "NUM_INSTRS", "LAST_INSTR_OFFSET", "COUNT"],
            [10, "0x100", 12, 3, 10, 5], [11, "0x10c", 4, 2, 2, 7], [12, "0x200", 2, 1, 0, 0]]}],
          [2, "0x7000000", 4096, {"FILE_NAME_ID": 2, "BASIC_BLOCKS": [
            ["NODE_ID", "ADDR_OFFSET", "SIZE", "NUM_INSTRS", "LAST_INSTR_OFFSET", "COUNT"],
            [20, "0x40", 8, 2, 4, 9]]}]],
        "EDGES": [["EDGE_ID", "SOURCE_NODE_ID", "TARGET_NODE_ID", "EDGE_TYPE_ID", "COUNT_PER_THREAD"],
          [1, 10, 10, 13, [4]], [2, 10, 11, 18, [1]], [3, 11, 20, 3, [7]],
          [4, 11, 10, 15, [3, 2]], [5, 20, 11, 6, [9]]]
      }]]
    }"#;

    #[test]
    fn the_dcfg_gives_blocks_and_taken_branches() {
        let base = |o: &str| if o == "/bin/a" { 0x1000 } else { 0 };
        let c = parse_dcfg(DCFG, base).unwrap();
        assert_eq!(
            c.blocks.get(&("/bin/a".to_string(), 0x1100, 12, 3)),
            Some(&5)
        );
        assert_eq!(
            c.blocks.get(&("/bin/a".to_string(), 0x110c, 4, 2)),
            Some(&7)
        );
        assert_eq!(
            c.blocks.get(&("/lib/v.so".to_string(), 0x40, 8, 2)),
            Some(&9)
        );
        // A block that never ran is no entry.
        assert_eq!(c.blocks.len(), 3);
        assert_eq!(c.instructions(), 5 * 3 + 7 * 2 + 9 * 2);
        // The conditional branch at the last instruction of node 10 back to itself, and the
        // unconditional one of node 11 (two threads); calls, returns and fall-throughs are no
        // jumps.
        assert_eq!(
            c.jumps.get(&("/bin/a".to_string(), 0x110a, 0x1100)),
            Some(&4)
        );
        assert_eq!(
            c.jumps.get(&("/bin/a".to_string(), 0x110e, 0x1100)),
            Some(&5)
        );
        assert_eq!(c.jumps.len(), 2);
        let mut empty = BlockCounts::default();
        empty
            .blocks
            .insert(("/bin/a".to_string(), 0x1100, 12, 3), 1);
        empty.blocks.insert(("/bin/a".to_string(), 0x1200, 2, 1), 3);
        let d = difference(&c, &empty);
        assert_eq!(d.deducted, 3);
        assert_eq!(
            d.blocks.get(&("/bin/a".to_string(), 0x1100, 12, 3)),
            Some(&4)
        );
        assert!(parse_dcfg("{}", base).is_err());
        assert!(is_branch_edge("INDIRECT_BRANCH") && !is_branch_edge("FALL_THROUGH"));
    }

    const BBCOUNT: &str = "\
bbcount 1
guest_base 1000
map 1000 3000 0 /bin/a
map 3000 4000 2000 /bin/a
map 10000 12000 0 /lib/v.so
tb 100 3 12 5
tb 2100 2 8 4
tb 10010 1 4 9
tb 900000 1 4 9
jump 108 100 4
jump 2104 100 2
jump 104 10010 9
";

    #[test]
    fn the_plugin_output_gives_instructions_of_the_objects() {
        // /bin/a: offset 0x2000 is at 0x12000 in the file; /lib/v.so at its offsets.
        let address_of = |o: &str, off: u64| match o {
            "/bin/a" if off >= 0x2000 => Some(off + 0x10000),
            _ => Some(off),
        };
        let c = parse_bbcount(BBCOUNT, address_of).unwrap();
        let i = &c.instructions;
        let a = |x: u64| i.get(&("/bin/a".to_string(), x)).copied();
        assert_eq!((a(0x100), a(0x104), a(0x108)), (Some(5), Some(5), Some(5)));
        // Guest 0x2100 is host 0x3100 in the second mapping, offset 0x2100, file 0x12100.
        assert_eq!((a(0x12100), a(0x12104)), (Some(4), Some(4)));
        assert_eq!(i.get(&("/lib/v.so".to_string(), 0x1010)), Some(&9));
        // The block outside every mapping is left out.
        assert_eq!(i.len(), 6);
        assert_eq!(c.jumps.get(&("/bin/a".to_string(), 0x108, 0x100)), Some(&4));
        assert_eq!(
            c.jumps.get(&("/bin/a".to_string(), 0x12104, 0x100)),
            Some(&2)
        );
        // A transfer into another object (a call into a library) is no jump.
        assert_eq!(c.jumps.len(), 2);
        assert!(parse_bbcount("tb 1 1 4 1\n", address_of).is_err());
    }
}
