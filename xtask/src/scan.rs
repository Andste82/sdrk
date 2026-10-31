//! Source scanners for the heap and global-state checks.
//!
//! The scanners are deliberately simple line scanners (no parser): they blank
//! out comments and literals, track `#[cfg(test)]` and `#[cfg(feature = "std")]`
//! regions by brace counting and then look for forbidden tokens.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// One finding of a check: a message at `path:line`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// File the finding belongs to (may be a manifest for layer findings).
    pub path: PathBuf,
    /// 1-based line number, 0 if not applicable.
    pub line: usize,
    /// Human-readable description.
    pub message: String,
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(f, "{}: {}", self.path.display(), self.message)
        } else {
            write!(f, "{}:{}: {}", self.path.display(), self.line, self.message)
        }
    }
}

/// How `std` may be used in a heap-free crate inside `#[cfg(feature = "std")]` items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdPolicy {
    /// Never (device, facade, blocks).
    Forbidden,
    /// Anything (the testkit's `std` feature).
    Full,
    /// `extern crate std` and no path but `std::arch` (runtime CPU detection in backends, the
    /// inherent `f32::sqrt` of the core).
    ArchOnly,
}

/// Why a region of source is exempt from a check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exempt {
    /// Not exempt.
    No,
    /// Inside a `#[cfg(test)]` item.
    Test,
    /// Inside a `#[cfg(feature = "std")]` item where everything is allowed.
    StdFull,
    /// Inside a `#[cfg(feature = "std")]` item where only `std::arch` is allowed.
    StdArch,
}

impl Exempt {
    fn strength(self) -> u8 {
        match self {
            Exempt::No => 0,
            Exempt::StdArch => 1,
            Exempt::Test | Exempt::StdFull => 2,
        }
    }
}

fn blank(c: char) -> char {
    if c == '\n' { '\n' } else { ' ' }
}

fn raw_string_prefix(chars: &[char], i: usize) -> Option<(usize, usize)> {
    if chars.get(i) != Some(&'r') {
        return None;
    }
    let mut j = i + 1;
    let mut hashes = 0;
    while chars.get(j) == Some(&'#') {
        hashes += 1;
        j += 1;
    }
    (chars.get(j) == Some(&'"')).then_some((hashes, j - i + 1))
}

/// Replaces comments and the contents of string and char literals with spaces.
/// Newlines are kept, so line numbers stay valid.
#[must_use]
pub fn sanitize(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let at = |i: usize| chars.get(i).copied();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut block_depth = 0usize;
    while let Some(c) = at(i) {
        if block_depth > 0 {
            if c == '/' && at(i + 1) == Some('*') {
                block_depth += 1;
                out.push_str("  ");
                i += 2;
            } else if c == '*' && at(i + 1) == Some('/') {
                block_depth -= 1;
                out.push_str("  ");
                i += 2;
            } else {
                out.push(blank(c));
                i += 1;
            }
            continue;
        }
        if c == '/' && at(i + 1) == Some('/') {
            while let Some(d) = at(i) {
                if d == '\n' {
                    break;
                }
                out.push(' ');
                i += 1;
            }
        } else if c == '/' && at(i + 1) == Some('*') {
            block_depth = 1;
            out.push_str("  ");
            i += 2;
        } else if let Some((hashes, prefix)) = raw_string_prefix(&chars, i) {
            for _ in 0..prefix {
                out.push(' ');
            }
            i += prefix;
            while let Some(d) = at(i) {
                let closes = d == '"' && (1..=hashes).all(|k| at(i + k) == Some('#'));
                if closes {
                    for _ in 0..=hashes {
                        out.push(' ');
                    }
                    i += hashes + 1;
                    break;
                }
                out.push(blank(d));
                i += 1;
            }
        } else if c == '"' {
            out.push('"');
            i += 1;
            while let Some(d) = at(i) {
                if d == '\\' {
                    out.push(' ');
                    i += 1;
                    if let Some(e) = at(i) {
                        out.push(blank(e));
                        i += 1;
                    }
                } else if d == '"' {
                    out.push('"');
                    i += 1;
                    break;
                } else {
                    out.push(blank(d));
                    i += 1;
                }
            }
        } else if c == '\'' {
            if at(i + 1) == Some('\\') {
                // Escaped char literal such as '\n', '\'' or '\u{1F600}'.
                out.push_str("   ");
                i += 3;
                while let Some(d) = at(i) {
                    out.push(blank(d));
                    i += 1;
                    if d == '\'' {
                        break;
                    }
                }
            } else if at(i + 2) == Some('\'') {
                out.push_str("   ");
                i += 3;
            } else {
                // Lifetime or label.
                out.push('\'');
                i += 1;
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// One source line after sanitizing, with its exemption state.
#[derive(Debug, Clone)]
pub struct Line {
    /// 1-based line number.
    pub number: usize,
    /// Sanitized code of the line.
    pub code: String,
    /// Exemption state of the line.
    pub exempt: Exempt,
}

/// Result of annotating a source file.
#[derive(Debug, Clone, Default)]
pub struct Annotated {
    /// All lines.
    pub lines: Vec<Line>,
    /// Out-of-line module declarations (`mod x;`) with the exemption of the line they sit on.
    pub mods: Vec<(String, Exempt)>,
}

enum Region {
    None,
    Pending(Exempt),
    Active {
        kind: Exempt,
        depth: usize,
        opened: bool,
    },
}

/// The exemption a `cfg` attribute at the start of `raw_line` opens, and the byte offset after
/// it. `#[cfg(all(test, ...))]` is test code as well: it is compiled only into tests.
fn attr_kind(raw_line: &str, policy: StdPolicy) -> Option<(Exempt, usize)> {
    let t = raw_line.trim_start();
    let offset = raw_line.len() - t.len();
    let compact: String = t.chars().filter(|c| !c.is_whitespace()).collect();
    let kind = if compact.starts_with("#[cfg(test)]") || compact.starts_with("#[cfg(all(test,") {
        Exempt::Test
    } else if compact.starts_with("#[cfg(feature=\"std\")]") {
        match policy {
            StdPolicy::Forbidden => return None,
            StdPolicy::Full => Exempt::StdFull,
            StdPolicy::ArchOnly => Exempt::StdArch,
        }
    } else {
        return None;
    };
    let end = t.find(']')? + 1;
    Some((kind, offset + end))
}

/// Finds `mod <ident>;` in sanitized code.
fn find_mod_decl(code: &str) -> Option<String> {
    let mut rest = code;
    while let Some(pos) = rest.find("mod ") {
        let before_ok = rest[..pos]
            .chars()
            .next_back()
            .is_none_or(|c| !is_word(c) && c != ':');
        let after = &rest[pos + 4..];
        let ident: String = after
            .trim_start()
            .chars()
            .take_while(|c| is_word(*c))
            .collect();
        let tail = after
            .trim_start()
            .get(ident.len()..)
            .unwrap_or("")
            .trim_start();
        if before_ok && !ident.is_empty() && tail.starts_with(';') {
            return Some(ident);
        }
        rest = after;
    }
    None
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Sanitizes `source` and marks `#[cfg(test)]` / `#[cfg(feature = "std")]` regions.
#[must_use]
pub fn annotate(source: &str, policy: StdPolicy) -> Annotated {
    let clean = sanitize(source);
    let mut out = Annotated::default();
    let mut region = Region::None;
    for (idx, (raw, code)) in source.lines().zip(clean.lines()).enumerate() {
        let number = idx + 1;
        let mut exempt = Exempt::No;
        let mut rest = code;
        if matches!(region, Region::None | Region::Pending(_)) {
            if let Some((kind, end)) = attr_kind(raw, policy) {
                region = Region::Pending(kind);
                rest = code.get(end..).unwrap_or("");
                exempt = kind;
            } else if matches!(region, Region::Pending(_)) && code.trim_start().starts_with("#[") {
                // Another attribute between the cfg attribute and the item.
                exempt = Exempt::Test;
                if let Region::Pending(kind) = region {
                    exempt = kind;
                }
                rest = "";
            }
        }
        if let Region::Pending(kind) = region {
            if !rest.trim().is_empty() {
                region = Region::Active {
                    kind,
                    depth: 0,
                    opened: false,
                };
            }
            exempt = kind;
        }
        if let Region::Active {
            kind,
            mut depth,
            mut opened,
        } = region
        {
            exempt = kind;
            let mut ended = false;
            for ch in rest.chars() {
                match ch {
                    '{' => {
                        depth += 1;
                        opened = true;
                    }
                    '}' => depth = depth.saturating_sub(1),
                    ';' if depth == 0 && !opened => ended = true,
                    _ => {}
                }
                if ended || (opened && depth == 0) {
                    ended = true;
                    break;
                }
            }
            region = if ended {
                Region::None
            } else {
                Region::Active {
                    kind,
                    depth,
                    opened,
                }
            };
        }
        if let Some(name) = find_mod_decl(code) {
            out.mods.push((name, exempt));
        }
        out.lines.push(Line {
            number,
            code: code.to_string(),
            exempt,
        });
    }
    out
}

fn boundary_before(text: &str, pos: usize, extra: &[char]) -> bool {
    text.get(..pos)
        .and_then(|s| s.chars().next_back())
        .is_none_or(|c| !is_word(c) && !extra.contains(&c))
}

fn boundary_after(text: &str, end: usize) -> bool {
    text.get(end..)
        .and_then(|s| s.chars().next())
        .is_none_or(|c| !is_word(c))
}

/// Names of types imported from `heapless` (`use heapless::{Vec, String};`).
fn heapless_imports(clean_source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = clean_source;
    while let Some(pos) = rest.find("use heapless") {
        let tail = &rest[pos..];
        let end = tail.find(';').unwrap_or(tail.len());
        let stmt = &tail[..end];
        let mut word = String::new();
        for c in stmt.chars().chain(std::iter::once(' ')) {
            if is_word(c) {
                word.push(c);
            } else {
                if matches!(word.as_str(), "Vec" | "String" | "Box") {
                    names.push(std::mem::take(&mut word));
                }
                word.clear();
            }
        }
        rest = &tail[end..];
    }
    names
}

fn heap_tokens_in(code: &str, kind: Exempt, imported: &[String]) -> Vec<String> {
    let mut found = Vec::new();
    let normalized: String = code.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.contains("extern crate alloc") {
        found.push("`extern crate alloc`".to_string());
    }
    if normalized.contains("extern crate std") && kind != Exempt::StdArch {
        found.push("`extern crate std`".to_string());
    }
    for (pos, _) in code.match_indices("alloc::") {
        if boundary_before(code, pos, &[':', '.']) {
            found.push("`alloc::`".to_string());
        }
    }
    for (pos, _) in code.match_indices("std::") {
        if boundary_before(code, pos, &[':', '.']) {
            let after = code.get(pos + 5..).unwrap_or("");
            if kind == Exempt::StdArch && after.starts_with("arch") {
                continue;
            }
            found.push("`std::`".to_string());
        }
    }
    for name in ["Vec", "Box", "String"] {
        for (pos, _) in code.match_indices(name) {
            let before = code.get(..pos).unwrap_or("");
            if !boundary_after(code, pos + name.len()) {
                continue;
            }
            if before
                .chars()
                .next_back()
                .is_some_and(|c| is_word(c) || c == '.')
            {
                continue;
            }
            if before.ends_with("heapless::") {
                continue;
            }
            if !before.ends_with("::") && imported.iter().any(|n| n == name) {
                continue;
            }
            found.push(format!("`{name}`"));
        }
    }
    for mac in ["vec!", "format!"] {
        for (pos, _) in code.match_indices(mac) {
            if boundary_before(code, pos, &[':']) {
                found.push(format!("`{mac}`"));
            }
        }
    }
    found
}

/// Result of a scan of one file.
#[derive(Debug, Clone, Default)]
pub struct ScanResult {
    /// (line, message) pairs.
    pub findings: Vec<(usize, String)>,
    /// Out-of-line modules declared in the file with their exemption.
    pub mods: Vec<(String, Exempt)>,
}

/// Scans one file of a heap-free crate. `inherited` is the exemption the whole
/// file has because it is an out-of-line module of an exempt item.
#[must_use]
pub fn scan_heap(source: &str, policy: StdPolicy, inherited: Exempt) -> ScanResult {
    let ann = annotate(source, policy);
    let imported = heapless_imports(&sanitize(source));
    let mut result = ScanResult::default();
    for line in &ann.lines {
        let kind = if inherited.strength() > line.exempt.strength() {
            inherited
        } else {
            line.exempt
        };
        if matches!(kind, Exempt::Test | Exempt::StdFull) {
            continue;
        }
        for token in heap_tokens_in(&line.code, kind, &imported) {
            result.findings.push((
                line.number,
                format!("{token} is not allowed in a heap-free crate"),
            ));
        }
    }
    result.mods = ann
        .mods
        .into_iter()
        .map(|(name, kind)| {
            let eff = if inherited.strength() > kind.strength() {
                inherited
            } else {
                kind
            };
            (name, eff)
        })
        .collect();
    result
}

const GLOBAL_WORDS: [&str; 4] = ["OnceCell", "OnceLock", "LazyLock", "LazyCell"];
const GLOBAL_MACROS: [&str; 2] = ["thread_local!", "lazy_static!"];
const GLOBAL_TYPES: [&str; 4] = ["Cell", "Mutex", "RwLock", "Atomic"];

/// Scans one library file for global mutable state.
#[must_use]
pub fn scan_global_state(source: &str, inherited: Exempt) -> ScanResult {
    let ann = annotate(source, StdPolicy::Forbidden);
    let mut result = ScanResult::default();
    let lines: Vec<&Line> = ann
        .lines
        .iter()
        .filter(|l| l.exempt != Exempt::Test && inherited != Exempt::Test)
        .collect();
    for (k, line) in lines.iter().enumerate() {
        let code = &line.code;
        let compact = code.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut push = |msg: &str| {
            result
                .findings
                .push((line.number, format!("{msg} is not allowed (global state)")));
        };
        if compact.contains("static mut ") {
            push("`static mut`");
        }
        for mac in GLOBAL_MACROS {
            if compact.contains(mac) {
                push(&format!("`{mac}`"));
            }
        }
        for word in GLOBAL_WORDS {
            for (pos, _) in code.match_indices(word) {
                if boundary_before(code, pos, &[]) && boundary_after(code, pos + word.len()) {
                    push(&format!("`{word}`"));
                }
            }
        }
        let trimmed = compact.trim_start();
        let after_vis = trimmed
            .strip_prefix("pub(crate) ")
            .or_else(|| trimmed.strip_prefix("pub "))
            .unwrap_or(trimmed);
        if after_vis.starts_with("static ") && !after_vis.starts_with("static mut ") {
            // Join following lines until the initializer starts.
            let mut stmt = after_vis.to_string();
            let mut extra = 0;
            while !stmt.contains('=') && !stmt.contains(';') && extra < 8 {
                extra += 1;
                match lines.get(k + extra) {
                    Some(next) => {
                        stmt.push(' ');
                        stmt.push_str(next.code.trim());
                    }
                    None => break,
                }
            }
            let ty = stmt
                .split_once(':')
                .map_or("", |(_, t)| t.split('=').next().unwrap_or(""));
            if GLOBAL_TYPES.iter().any(|t| ty.contains(t)) {
                push("`static` with interior mutability");
            }
        }
    }
    result.mods = ann.mods;
    result
}

/// The functions of `sdrk-core` whose bodies may use `std` (decision D67): the square roots of
/// `num.rs`, with the feature `std` `f32::sqrt` / `f64::sqrt`.
const CORE_STD_FUNCTIONS: [&str; 2] = ["fn sqrt_f32(", "fn sqrt_f64("];

/// The line numbers of the bodies (signature to closing brace) of [`CORE_STD_FUNCTIONS`].
fn core_std_function_lines(lines: &[Line]) -> Vec<usize> {
    let mut allowed = Vec::new();
    let mut depth: Option<usize> = None;
    for line in lines {
        let code = &line.code;
        if depth.is_none()
            && CORE_STD_FUNCTIONS.iter().any(|f| {
                code.match_indices(f)
                    .any(|(pos, _)| boundary_before(code, pos, &[]))
            })
        {
            depth = Some(0);
        }
        let Some(mut d) = depth else { continue };
        allowed.push(line.number);
        let mut opened = d > 0;
        for ch in code.chars() {
            match ch {
                '{' => {
                    d += 1;
                    opened = true;
                }
                '}' => d = d.saturating_sub(1),
                _ => {}
            }
        }
        depth = if opened && d == 0 { None } else { Some(d) };
    }
    allowed
}

/// The uses of `std` in one non-test line of `sdrk-core`: a `std` path or crate, the feature
/// `std` in a `cfg`, and the inherent `sqrt` of `f32` / `f64`, which only `std` has.
fn core_std_tokens(raw: &str, code: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    for (pos, _) in code.match_indices("std") {
        if boundary_before(code, pos, &[]) && boundary_after(code, pos + 3) {
            found.push("`std`");
        }
    }
    let compact_raw: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    let compact_code: String = code.chars().filter(|c| !c.is_whitespace()).collect();
    if compact_code.contains("feature=") && compact_raw.contains("feature=\"std\"") {
        found.push("`feature = \"std\"`");
    }
    for (pos, _) in code.match_indices("sqrt(") {
        let before = code.get(..pos).unwrap_or("");
        if before.ends_with('.') || (before.ends_with("::") && !before.ends_with("libm::")) {
            found.push("`sqrt` of `std`");
        }
    }
    found
}

/// Scans one file of a core-layer crate for `std` (decision D67, a narrow exception). `file` is
/// the path relative to `src`. Allowed are only, in `sdrk-core` (`exception`), the bodies of
/// `num::sqrt_f32` / `num::sqrt_f64` and `extern crate std` under `#[cfg(feature = "std")]` in
/// `lib.rs`; tests are exempt.
#[must_use]
pub fn scan_core_std(
    file: &str,
    source: &str,
    inherited: Exempt,
    exception: bool,
) -> Vec<(usize, String)> {
    let ann = annotate(source, StdPolicy::Forbidden);
    let allowed = if exception && file == "num.rs" {
        core_std_function_lines(&ann.lines)
    } else {
        Vec::new()
    };
    let raw: Vec<&str> = source.lines().collect();
    let mut findings = Vec::new();
    for (k, line) in ann.lines.iter().enumerate() {
        if inherited == Exempt::Test || line.exempt == Exempt::Test {
            continue;
        }
        if allowed.contains(&line.number) {
            continue;
        }
        let raw_line = raw.get(k).copied().unwrap_or("");
        let compact = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        if exception && file == "lib.rs" {
            let next = raw.get(k + 1).map_or(String::new(), |s| compact(s));
            let prev = k
                .checked_sub(1)
                .and_then(|i| raw.get(i))
                .map_or(String::new(), |s| compact(s));
            let this = compact(raw_line);
            if (this == "#[cfg(feature=\"std\")]" && next == "externcratestd;")
                || (this == "externcratestd;" && prev == "#[cfg(feature=\"std\")]")
            {
                continue;
            }
        }
        for token in core_std_tokens(raw_line, &line.code) {
            findings.push((
                line.number,
                format!(
                    "{token} in the core layer outside `sdrk_core::num::sqrt_f32` / `sqrt_f64` \
                     (D67): a new std use in the core layer needs a maintainer decision"
                ),
            ));
        }
    }
    findings
}

/// [`scan_core_std`] over every `.rs` file under `src_dir` (the `src` of a core-layer crate;
/// `exception` for `sdrk-core`), with the test exemption of out-of-line modules.
///
/// # Errors
/// Fails if a file cannot be read.
pub fn scan_core_std_crate(src_dir: &Path, exception: bool) -> anyhow::Result<Vec<Finding>> {
    let mut findings = Vec::new();
    let mut test_files: Vec<PathBuf> = Vec::new();
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    for entry in WalkDir::new(src_dir).sort_by_file_name() {
        let entry = entry?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|e| e == "rs") {
            let path = entry.path().to_path_buf();
            let source = std::fs::read_to_string(&path)?;
            for (name, kind) in annotate(&source, StdPolicy::Forbidden).mods {
                if kind == Exempt::Test {
                    let dir = module_dir(&path);
                    test_files.push(dir.join(format!("{name}.rs")));
                    test_files.push(dir.join(&name).join("mod.rs"));
                }
            }
            files.push((path, source));
        }
    }
    for (path, source) in &files {
        let inherited = if test_files.iter().any(|t| t == path) {
            Exempt::Test
        } else {
            Exempt::No
        };
        let rel = path
            .strip_prefix(src_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        for (line, message) in scan_core_std(&rel, source, inherited, exception) {
            findings.push(Finding {
                path: path.clone(),
                line,
                message,
            });
        }
    }
    Ok(findings)
}

/// Kind of scan to run over a crate.
#[derive(Clone, Copy)]
pub enum Scan {
    /// Heap check with the given `std` policy.
    Heap(StdPolicy),
    /// Global-state check.
    GlobalState,
}

/// Scans every `.rs` file under `src_dir`, propagating exemptions of out-of-line modules.
///
/// # Errors
/// Fails if a file cannot be read.
pub fn scan_crate(src_dir: &Path, scan: Scan) -> anyhow::Result<Vec<Finding>> {
    let policy = match scan {
        Scan::Heap(p) => p,
        Scan::GlobalState => StdPolicy::Forbidden,
    };
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    for entry in WalkDir::new(src_dir).sort_by_file_name() {
        let entry = entry?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|e| e == "rs") {
            files.push((
                entry.path().to_path_buf(),
                std::fs::read_to_string(entry.path())?,
            ));
        }
    }
    let mut inherited: HashMap<PathBuf, Exempt> = HashMap::new();
    loop {
        let mut changed = false;
        for (path, source) in &files {
            let own = inherited.get(path).copied().unwrap_or(Exempt::No);
            let ann = annotate(source, policy);
            let dir = module_dir(path);
            for (name, kind) in ann.mods {
                let eff = if own.strength() > kind.strength() {
                    own
                } else {
                    kind
                };
                for candidate in [
                    dir.join(format!("{name}.rs")),
                    dir.join(&name).join("mod.rs"),
                ] {
                    let current = inherited.get(&candidate).copied().unwrap_or(Exempt::No);
                    if eff.strength() > current.strength() {
                        inherited.insert(candidate, eff);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut findings = Vec::new();
    for (path, source) in &files {
        let own = inherited.get(path).copied().unwrap_or(Exempt::No);
        let result = match scan {
            Scan::Heap(p) => scan_heap(source, p, own),
            Scan::GlobalState => scan_global_state(source, own),
        };
        for (line, message) in result.findings {
            findings.push(Finding {
                path: path.clone(),
                line,
                message,
            });
        }
    }
    Ok(findings)
}

fn module_dir(path: &Path) -> PathBuf {
    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
    match path.file_name().and_then(|n| n.to_str()) {
        Some("lib.rs" | "main.rs" | "mod.rs") => parent,
        _ => parent.join(path.file_stem().unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap(src: &str) -> Vec<usize> {
        scan_heap(src, StdPolicy::Forbidden, Exempt::No)
            .findings
            .into_iter()
            .map(|f| f.0)
            .collect()
    }

    #[test]
    fn clean_code_passes() {
        assert_eq!(
            heap("fn f(x: &[u8]) -> usize { x.len() }\n"),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn vec_is_found_with_line_number() {
        assert_eq!(
            heap("fn f() {\n    let v: Vec<u8> = Vec::new();\n}\n"),
            vec![2, 2]
        );
    }

    #[test]
    fn comments_and_strings_are_ignored() {
        let src = "// Vec<u8> here\n/* Box<u8>\n String */\nconst S: &str = \"Vec<u8> and std::fs\";\nconst R: &str = r#\"String\"#;\nfn f() {}\n";
        assert_eq!(heap(src), Vec::<usize>::new());
    }

    #[test]
    fn cfg_test_module_is_ignored() {
        let src = "fn f() {}\n#[cfg(test)]\nmod tests {\n    fn t() { let v: Vec<u8> = Vec::new(); }\n}\nfn g() { let _: String; }\n";
        assert_eq!(heap(src), vec![6]);
    }

    #[test]
    fn cfg_all_test_module_is_ignored() {
        let src = "#[cfg(all(test, feature = \"x\"))]\nmod tests {\n    extern crate std;\n    fn t() { let v: Vec<u8> = Vec::new(); }\n}\nfn g() { let _: String; }\n";
        assert_eq!(heap(src), vec![6]);
        // `any(test, ...)` is also compiled outside the tests.
        let src = "#[cfg(any(test, feature = \"x\"))]\nfn f() { let _: Vec<u8>; }\n";
        assert_eq!(heap(src), vec![2]);
    }

    #[test]
    fn alloc_std_and_macros_are_found() {
        assert_eq!(heap("extern crate alloc;\n").len(), 1);
        assert_eq!(heap("use std::fmt;\n").len(), 1);
        assert_eq!(
            heap("fn f() { let _ = format!(\"x\"); let _ = vec![1]; }\n").len(),
            2
        );
        assert_eq!(heap("use alloc::vec::Vec;\n").len(), 2);
    }

    #[test]
    fn method_alloc_and_heapless_are_fine() {
        assert_eq!(
            heap("fn f(s: &mut S) { s.alloc::<u8>(3); }\n"),
            Vec::<usize>::new()
        );
        assert_eq!(
            heap("fn f(a: heapless::Vec<u8, 4>, b: heapless::String<8>) {}\n"),
            Vec::<usize>::new()
        );
        assert_eq!(
            heap("use heapless::{String, Vec};\nfn f(a: Vec<u8, 4>, b: String<8>) {}\n"),
            Vec::<usize>::new()
        );
        assert_eq!(heap("fn f(a: HVec<u8>, b: MyString) {}\n").len(), 0);
    }

    #[test]
    fn std_cfg_regions_follow_policy() {
        let src = "#[cfg(feature = \"std\")]\nextern crate std;\n#[cfg(feature = \"std\")]\nmod host {\n    use std::fs;\n}\nuse std::io;\n";
        assert_eq!(
            scan_heap(src, StdPolicy::Full, Exempt::No).findings.len(),
            1
        );
        assert_eq!(
            scan_heap(src, StdPolicy::Forbidden, Exempt::No)
                .findings
                .len(),
            3
        );
        let arch = "#[cfg(feature = \"std\")]\nfn detect() -> bool { std::arch::is_x86_feature_detected!(\"avx2\") }\n#[cfg(feature = \"std\")]\nfn bad() { std::fs::read(\"x\"); }\n";
        let found = scan_heap(arch, StdPolicy::ArchOnly, Exempt::No).findings;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, 4);
    }

    #[test]
    fn out_of_line_mods_are_reported_with_exemption() {
        let src =
            "#[cfg(test)]\nmod tests;\nmod plain;\n#[cfg(feature = \"std\")]\npub mod host;\n";
        let mods = scan_heap(src, StdPolicy::Full, Exempt::No).mods;
        assert_eq!(
            mods,
            vec![
                ("tests".to_string(), Exempt::Test),
                ("plain".to_string(), Exempt::No),
                ("host".to_string(), Exempt::StdFull)
            ]
        );
    }

    fn global(src: &str) -> Vec<usize> {
        scan_global_state(src, Exempt::No)
            .findings
            .into_iter()
            .map(|f| f.0)
            .collect()
    }

    #[test]
    fn global_state_patterns_are_found() {
        assert_eq!(global("static mut X: u32 = 0;\n"), vec![1]);
        assert_eq!(
            global("static COUNTER: AtomicU32 = AtomicU32::new(0);\n"),
            vec![1]
        );
        assert_eq!(
            global("pub static M: Mutex<u8> = Mutex::new(0);\n"),
            vec![1]
        );
        assert_eq!(
            global("static C: core::cell::UnsafeCell<u8> =\n    UnsafeCell::new(0);\n"),
            vec![1]
        );
        assert_eq!(global("thread_local! { static X: u8 = 0; }\n"), vec![1]);
        assert_eq!(
            global("static L: OnceLock<u8> = OnceLock::new();\n").len(),
            2
        );
    }

    #[test]
    fn immutable_statics_and_tests_are_fine() {
        assert_eq!(
            global(
                "static TABLE: [u8; 4] = [1, 2, 3, 4];\nconst X: AtomicU32 = AtomicU32::new(0);\n"
            ),
            Vec::<usize>::new()
        );
        assert_eq!(
            global("#[cfg(test)]\nmod t {\n    static C: AtomicU32 = AtomicU32::new(0);\n}\n"),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn sanitize_keeps_lines_and_lifetimes() {
        let s = sanitize("fn f<'a>(x: &'a str) -> char { '\\'' } // c\n\"q\\\"r\"\n");
        assert_eq!(s.lines().count(), 2);
        assert!(s.contains("'a"));
        assert!(!s.contains("// c"));
    }

    const CORE_NUM: &str = "/// With the feature `std` it is `f32::sqrt`.\n#[inline]\npub fn sqrt_f32(x: f32) -> f32 {\n    #[cfg(feature = \"std\")]\n    {\n        x.sqrt()\n    }\n    #[cfg(not(feature = \"std\"))]\n    {\n        libm::sqrtf(x)\n    }\n}\n\npub fn sqrt_f64(x: f64) -> f64 {\n    #[cfg(feature = \"std\")]\n    {\n        f64::sqrt(x)\n    }\n    #[cfg(not(feature = \"std\"))]\n    {\n        libm::sqrt(x)\n    }\n}\n#[cfg(test)]\nmod tests {\n    use std::vec::Vec;\n    fn t() { let _ = 2f32.sqrt(); }\n}\n";
    const CORE_LIB: &str = "#![no_std]\n#[cfg(feature = \"std\")]\nextern crate std;\n#[cfg(all(test, not(feature = \"std\")))]\nextern crate std;\nmod num;\n";

    fn core_std(file: &str, src: &str) -> Vec<usize> {
        scan_core_std(file, src, Exempt::No, true)
            .into_iter()
            .map(|(l, _)| l)
            .collect()
    }

    #[test]
    fn core_std_allows_only_the_two_square_roots() {
        assert_eq!(core_std("num.rs", CORE_NUM), Vec::<usize>::new());
        assert_eq!(core_std("lib.rs", CORE_LIB), Vec::<usize>::new());
        // The same code anywhere else, or in another core crate, is a finding.
        assert_ne!(core_std("fallback/mag.rs", CORE_NUM), Vec::<usize>::new());
        assert_ne!(
            scan_core_std("num.rs", CORE_NUM, Exempt::No, false),
            Vec::new()
        );
        assert_ne!(
            scan_core_std("lib.rs", CORE_LIB, Exempt::No, false),
            Vec::new()
        );
        assert_eq!(
            scan_core_std("x.rs", CORE_NUM, Exempt::Test, false),
            Vec::new()
        );
    }

    #[test]
    fn core_std_rejects_a_second_use() {
        // A third function in num.rs with the inherent square root behind the feature.
        let third = format!(
            "{CORE_NUM}pub fn hypot(x: f32, y: f32) -> f32 {{\n    #[cfg(feature = \"std\")]\n    {{\n        (x * x + y * y).sqrt()\n    }}\n}}\n"
        );
        let n = CORE_NUM.lines().count();
        assert_eq!(core_std("num.rs", &third), vec![n + 2, n + 4]);
        let msg = &scan_core_std("num.rs", &third, Exempt::No, true)[0].1;
        assert!(msg.contains("needs a maintainer decision"), "{msg}");
        // A std path, a second `extern crate std` and a cfg on the feature elsewhere.
        assert_eq!(
            core_std("lib.rs", "#![no_std]\nextern crate std;\nmod num;\n"),
            vec![2]
        );
        assert_eq!(
            core_std("fir.rs", "fn f() -> u32 { std::process::id() }\n"),
            vec![1]
        );
        assert_eq!(
            core_std(
                "fir.rs",
                "#[cfg(any(feature = \"std\", feature = \"x\"))]\nfn f() {}\n"
            ),
            vec![1]
        );
        assert_eq!(
            core_std("agc.rs", "fn f(x: f32) -> f32 { f32::sqrt(x) }\n"),
            vec![1]
        );
        // Comments, strings, `no_std` and libm are not uses.
        assert_eq!(
            core_std(
                "agc.rs",
                "// std::fs and feature = \"std\"\nconst S: &str = \"std\";\nfn f(x: f32) -> f32 { libm::sqrtf(x) + libm::sqrt(1.0) as f32 }\n"
            ),
            Vec::<usize>::new()
        );
    }
}
