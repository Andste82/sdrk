#!/usr/bin/env python3
"""Golden-vector generator for SDRk.

Usage: python tests/golden/generate.py [--all | --kernel NAME] [--check] [--list] [--out DIR]
"""

import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sdrk_golden import cases, fileio, q15, registry  # noqa: E402,F401


def render(selected):
    """Render all selected cases: {relative path: bytes}. Enforces the size limits."""
    out = {}
    total = 0
    for c in selected:
        files = fileio.render_case(c.kernel, c.name, c.seed, c.run())
        size = sum(len(v) for v in files.values())
        if size > registry.MAX_CASE_BYTES:
            raise SystemExit(f"error: case {c.kernel}/{c.name} is {size} bytes (limit {registry.MAX_CASE_BYTES})")
        total += size
        for fname, raw in files.items():
            out[f"{c.kernel}/{fname}"] = raw
    if total > registry.MAX_TOTAL_BYTES:
        raise SystemExit(f"error: vectors are {total} bytes (limit {registry.MAX_TOTAL_BYTES})")
    return out


def existing(root, kernels):
    """All committed files below ``root`` for the given kernels, as relative paths."""
    found = set()
    for kernel in kernels:
        d = os.path.join(root, kernel)
        if os.path.isdir(d):
            for fname in os.listdir(d):
                found.add(f"{kernel}/{fname}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="generate every case (default)")
    g.add_argument("--kernel", metavar="NAME", help="only cases of this kernel")
    ap.add_argument("--check", action="store_true", help="compare with the committed files")
    ap.add_argument("--list", action="store_true", help="print kernel/case lines")
    ap.add_argument("--out", default=os.path.join(HERE, "vectors"), help="output directory")
    args = ap.parse_args()

    selected = [c for c in registry.CASES if args.kernel is None or c.kernel == args.kernel]
    if args.kernel is not None and not selected:
        print(f"error: no cases for kernel {args.kernel}", file=sys.stderr)
        return 2
    if args.list:
        for c in selected:
            print(f"{c.kernel}/{c.name}")
        return 0

    q15.self_test()
    rendered = render(selected)
    kernels = sorted({c.kernel for c in selected})
    if args.kernel is None:
        # Whole-tree mode: also find stale kernel directories.
        root = args.out
        if os.path.isdir(root):
            kernels = sorted(set(kernels) | {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))})

    if args.check:
        root = args.out
        problems = []
        on_disk = existing(root, kernels)
        for rel, raw in sorted(rendered.items()):
            path = os.path.join(root, *rel.split("/"))
            if not os.path.isfile(path):
                problems.append(f"missing: {rel}")
            else:
                with open(path, "rb") as f:
                    if f.read() != raw:
                        problems.append(f"differs: {rel}")
        for rel in sorted(on_disk - set(rendered)):
            problems.append(f"stale: {rel}")
        if problems:
            print("\n".join(problems))
            print(f"golden check failed: {len(problems)} problem(s)")
            return 1
        print(f"golden check ok: {len(rendered)} files")
        return 0

    # Generate: write into a temporary sibling first only when needed is not required;
    # remove stale files of the touched kernels so the tree matches the registry.
    for rel in sorted(existing(args.out, kernels) - set(rendered)):
        os.remove(os.path.join(args.out, *rel.split("/")))
    for rel, raw in rendered.items():
        kernel, fname = rel.split("/")
        fileio.write_files(os.path.join(args.out, kernel), {fname: raw})
    print(f"wrote {len(rendered)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
