#!/usr/bin/env python3
"""Real core cycles per sample of our kernels against the reference libraries, on this machine.

Drives `count cycles` (PMU counters through perf_event_open: no emulator, no model, no wall
clock) over the kernels of benches/kernels and every implementation each supports (`ours`,
`fallback`, `volk`, `fftw`, `liquid`, ...). Needs a native build of the measurement binary:

    . ~/.cargo/env
    R=$PWD/benches/target/refs/aarch64        # or benches/refs/build.sh x86_64 <prefix>
    (cd benches && SDRK_REFS_DIR=$R PKG_CONFIG_PATH=$R/lib/pkgconfig CARGO_ENCODED_RUSTFLAGS= \
        cargo build --release -p kernels --target-dir target/native)
    python3 benches/cycles.py                  # all kernels, 5 repetitions
    python3 benches/cycles.py -k 'fir|fft_1024' --reps 9 --core 3

x86-64 level: pass the same -C target-cpu in CARGO_ENCODED_RUSTFLAGS as xtask does for the level
(x86-64-v2 for sse4.2, x86-64-v3 for avx2). Reports cycles per sample, instructions per sample
and IPC (median of the repetitions, with the spread), and for every other implementation the
ratio ours / other (below 1: ours is faster). For a `<kernel>_fallback` entry "ours" is the
implementation `fallback` (the portable code), see `ours_of`. Implementations of different kernels are
interleaved across the repetitions so that thermal drift hits all of them alike.
"""
import argparse, json, math, os, platform, re, statistics as st, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 4096                      # samples per call (kernels::N); the FFT entries are the exception
TARGET_CYCLES = 3e8           # one measurement covers about this many cycles (about 0.15 s)
SKIP_IMPLS = {"restore"}      # the baseline of xtask's differential method, not an entry


def fft_sizes():
    """The samples per call of the fft(...) entries of xtask's table."""
    src = (HERE.parent / "xtask" / "src" / "count.rs").read_text()
    return {m.group(1): int(m.group(3))
            for m in re.finditer(r'\bfft\(\s*"([^"]+)",\s*(?:Family::)?(\w+),\s*(\d+)\s*,?\s*\)', src)}


def ours_of(kernel):
    """The implementation that is "ours" in a kernel's row. A `<kernel>_fallback` entry measures
    the portable code, which on x86-64 and AArch64 is the implementation `fallback` (`ours` there
    is the SIMD backend, the fallback only runs by default without a SIMD level)."""
    return "fallback" if kernel.endswith("_fallback") else "ours"


def read(path, default="n/a"):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def env(core):
    f = f"/sys/devices/system/cpu/cpu{core}/cpufreq/"
    t = read("/sys/class/thermal/thermal_zone0/temp", "")
    return {
        "freq_khz": read(f + "scaling_cur_freq"),
        "governor": read(f + "scaling_governor"),
        "temp_c": round(int(t) / 1000, 1) if t.isdigit() else None,
        "load": read("/proc/loadavg").split()[0],
    }


class Runner:
    def __init__(self, binary, core):
        self.cmd = (["taskset", "-c", str(core)] if core is not None else []) + [str(binary)]

    def run(self, *args, timeout=600):
        p = subprocess.run(self.cmd + [str(a) for a in args], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr

    def cycles(self, kernel, impl, k):
        rc, out, err = self.run("cycles", kernel, impl, k)
        if rc != 0:
            raise RuntimeError(f"rc={rc}: {err.strip()[:200]}")
        return json.loads(out)


def calibrate(r, kernel, impl):
    """A k such that one measurement covers about TARGET_CYCLES."""
    c = r.cycles(kernel, impl, 2)["cycles"]
    return max(2, min(1 << 20, math.ceil(TARGET_CYCLES / max(c / 2, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=str(HERE / "target" / "neon-native" / "release" / "count"))
    ap.add_argument("--core", type=int, default=2, help="core to pin to (-1: do not pin)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("-k", "--kernels", default=".", help="regex on the kernel names")
    ap.add_argument("--impls", default=None, help="comma list; default: every supported one")
    ap.add_argument("--out", default=None, help="JSON result file (default benches/target/cycles-<arch>.json)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    r = Runner(a.bin, None if a.core < 0 else a.core)
    rc, out, _ = r.run("list")
    if rc != 0:
        sys.exit(f"cannot run {a.bin} (build it first, see --help)")
    kernels = [k for k in out.split() if re.search(a.kernels, k)]
    if a.list:
        print("\n".join(kernels)); return
    sizes = fft_sizes()
    want = set(a.impls.split(",")) if a.impls else None

    # A PMU check up front, with a readable message.
    try:
        r.cycles(kernels[0], "ours", 1)
    except Exception as e:
        sys.exit(f"PMU not usable ({e}). Needs root or perf_event_paranoid <= 2, and no VM/container hiding the PMU.")

    start_env = env(a.core)
    print(f"# {platform.machine()} core {a.core}  {start_env}", file=sys.stderr)
    results, t0 = {}, time.time()
    for idx, kernel in enumerate(kernels, 1):
        rc, out, _ = r.run("info", kernel)
        if rc != 0:
            results[kernel] = {"error": "info failed"}; continue
        info = json.loads(out)
        impls = [i for i in info["impls"] if i not in SKIP_IMPLS and (want is None or i in want)]
        samples = sizes.get(kernel, N)
        plan, runs = {}, {i: [] for i in impls}
        for i in impls:
            try:
                plan[i] = calibrate(r, kernel, i)
            except Exception as e:
                runs[i] = str(e)
        live = [i for i in impls if isinstance(runs[i], list)]
        for _ in range(a.reps):                      # interleaved across the implementations
            for i in live:
                try:
                    c = r.cycles(kernel, i, plan[i])
                    runs[i].append((c["cycles"], c["instructions"], c["k"]))
                except Exception as e:
                    runs[i] = str(e); live = [x for x in live if x != i]; break
        entry = {"samples_per_call": samples, "level": info.get("level"), "impls": {}}
        for i in impls:
            v = runs[i]
            if not isinstance(v, list) or not v:
                entry["impls"][i] = {"error": v if isinstance(v, str) else "no data"}; continue
            cps = [c / (k * samples) for c, _, k in v]
            ips = [n / (k * samples) for _, n, k in v]
            m = st.median(cps)
            entry["impls"][i] = {
                "cycles_per_sample": round(m, 3), "min": round(min(cps), 3), "max": round(max(cps), 3),
                "spread_pct": round(100 * (max(cps) - min(cps)) / m, 2) if m else 0,
                "instr_per_sample": round(st.median(ips), 3), "ipc": round(st.median(ips) / m, 3) if m else 0,
                "k": v[0][2], "reps": len(v),
            }
        results[kernel] = entry
        brief = "  ".join(f"{i}={d['cycles_per_sample']:.2f}" if "cycles_per_sample" in d else f"{i}=ERR" for i, d in entry["impls"].items())
        print(f"[{idx}/{len(kernels)}] {kernel}: {brief}", file=sys.stderr, flush=True)

    end_env = env(a.core)
    doc = {"machine": platform.machine(), "core": a.core, "reps": a.reps, "binary": a.bin,
           "env_start": start_env, "env_end": end_env, "seconds": round(time.time() - t0), "kernels": results}
    path = Path(a.out) if a.out else HERE / "target" / f"cycles-{platform.machine()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))

    # The table: cycles per sample, and ours / other for every other implementation.
    cols = sorted({i for k, e in results.items() for i in e.get("impls", {}) if i not in ("ours", ours_of(k))} | {"fallback"})
    print(f"\n{'kernel':34} {'ours':>9} {'ipc':>5}  " + "  ".join(f"{c:>18}" for c in cols))
    for kernel, e in results.items():
        impls = e.get("impls", {})
        me = ours_of(kernel)
        ours = impls.get(me, {}).get("cycles_per_sample")
        cells = []
        for c in cols:
            d = impls.get(c, {})
            if c == me or "cycles_per_sample" not in d:
                cells.append(f"{'':>18}"); continue
            ratio = f"{ours / d['cycles_per_sample']:.2f}" if ours and d["cycles_per_sample"] else "-"
            cells.append(f"{d['cycles_per_sample']:>9.2f} ({ratio:>5})")
        print(f"{kernel:34} {ours if ours is not None else float('nan'):>9.2f} {impls.get(me, {}).get('ipc', 0):>5.2f}  " + "  ".join(f"{c:>18}" for c in cells))
    print("\ncycles per sample; in brackets ours / other (below 1: ours needs fewer cycles)")
    print(f"env at start {start_env}\nenv at end   {end_env}\nsaved to {path}")


if __name__ == "__main__":
    main()
