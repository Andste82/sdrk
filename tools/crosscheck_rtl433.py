#!/usr/bin/env python3
"""Cross-check of the SDRk sensor encoders and decoders against rtl_433.

rtl_433 is run as an external program on IQ files (cu8, 2 MS/s); its source code is never read
or copied. For every device of a directory written by the example `sensor_iq generate`:

  (a) the readings transmitted (the JSON sidecar, ground truth) are compared with what
      rtl_433 decodes from the IQ file: this tests our ENCODER and the transmit chain;
  (b) the output of our receiver chain (`sensor_iq decode`) on the same file is compared with
      the ground truth and with the output of rtl_433: this tests our DECODER.

Real recordings can be given with --recording FILE[:TEMP_C:HUMIDITY] (no ground truth, only
rtl_433 against our decoder and, when given, the values of the display).

How to run (rtl_433 is NOT a dependency of the repository and not part of the CI; it is run
as an external program on files, its code is never read or copied):

  cargo build --release -p sdrk-pipeline --example sensor_iq
  target/release/examples/sensor_iq generate /tmp/iq --count 5 --noise 2
  python3 tools/crosscheck_rtl433.py --iq-dir /tmp/iq --recording /path/dump2.cu8:22.1:64

Normal case only: glitches, truncation and timing tolerance are tested by the Rust tests.
Python 3 standard library only. Exit status 0 only when everything matches; 77 when rtl_433
is not installed (the check is skipped).
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OURS = os.path.join(
    os.environ.get("CARGO_TARGET_DIR", os.path.join(REPO, "target")),
    "release", "examples", "sensor_iq",
)

# Temperature tolerance (degrees C) per device = its resolution: 0.1 C for the devices that
# report tenths, one step of 0.1 F (0.0556 C) for the Bresser sensor.
TEMP_TOL = {"bresser_3ch": 0.06}
DEFAULT_TEMP_TOL = 0.1
# Tolerances of the other numeric fields (printed resolution of the JSON form: one decimal).
OTHER_TOL = {
    "wind_avg_km_h": 0.15,
    "wind_max_km_h": 0.15,
    # rtl_433 prints the 16 wind directions as whole degrees (68 for 67.5)
    "wind_dir_deg": 0.5,
    "rain_mm": 0.15,
    "pressure_hPa": 0.15,
}
# rtl_433 prints the raw channel nibble of the Oregon sensors (1, 2 and 4 for the switch
# positions 1 to 3); our readings carry the position. The mapping is applied to rtl_433's value.
RTL_CHANNEL_MAP = {"oregon_v21": {4: 3}}
# rtl_433 model names (lower case regular expressions) of the models of our devices.
RTL_MODELS = {
    "wh2": r"wh2|ws2",
    "nexus": r"nexus|prego|crx",
    "oregon_v21": r"oregon",
    "wh1080": r"wh1080|whx080|wh3080|wh\d+",
    "lacrosse_tx29": r"tx29|tx35|lacrosse",
    "bresser_3ch": r"bresser-3ch",
}
# Words of the -R help line that identify the protocol of a device.
RTL_HELP_WORDS = {
    "wh2": ["WH2,"],
    "nexus": ["Nexus"],
    "oregon_v21": ["Oregon Scientific Weather Sensor"],
    "wh1080": ["WH1080"],
    "lacrosse_tx29": ["TX29IT", "TX35DTH-IT"],
    "bresser_3ch": ["Sensor 3CH"],
}
FIELDS = [
    "id", "channel", "battery_ok", "temperature_C", "humidity",
    "wind_avg_km_h", "wind_max_km_h", "wind_dir_deg", "rain_mm", "pressure_hPa",
]


def tol_for(device, field):
    if field == "temperature_C":
        return TEMP_TOL.get(device, DEFAULT_TEMP_TOL)
    return OTHER_TOL.get(field, 0.0)


def normalize_rtl(rec):
    """The canonical fields of one rtl_433 JSON record (its model name is kept as _model)."""
    out = {"_model": rec.get("model", "")}
    t = rec.get("time", "")
    m = re.match(r"@([0-9.]+)s", str(t))
    out["_t"] = float(m.group(1)) if m else None
    if "id" in rec:
        out["id"] = int(rec["id"])
    if "channel" in rec:
        out["channel"] = int(rec["channel"])
    if "battery_ok" in rec:
        out["battery_ok"] = bool(int(rec["battery_ok"]))
    elif "battery" in rec:
        out["battery_ok"] = str(rec["battery"]).upper() == "OK"
    if "temperature_C" in rec:
        out["temperature_C"] = float(rec["temperature_C"])
    elif "temperature_F" in rec:
        out["temperature_C"] = (float(rec["temperature_F"]) - 32.0) / 1.8
    if "humidity" in rec:
        out["humidity"] = float(rec["humidity"])
    for ours, keys in (
        ("wind_avg_km_h", [("wind_avg_km_h", 1.0), ("wind_avg_m_s", 3.6)]),
        ("wind_max_km_h", [("wind_max_km_h", 1.0), ("wind_max_m_s", 3.6)]),
        ("wind_dir_deg", [("wind_dir_deg", 1.0)]),
        ("rain_mm", [("rain_mm", 1.0)]),
        ("pressure_hPa", [("pressure_hPa", 1.0)]),
    ):
        for key, k in keys:
            if key in rec:
                out[ours] = float(rec[key]) * k
                break
    return out


def normalize_ours(rec):
    out = {"_model": rec.get("model", ""), "_t": rec.get("time")}
    for f in FIELDS:
        if f in rec:
            out[f] = rec[f]
    if "battery_ok" in out:
        out["battery_ok"] = bool(out["battery_ok"])
    return out


def normalize_sent(rec):
    out = {"_model": rec.get("model", "")}
    for f in FIELDS:
        if f in rec:
            out[f] = rec[f]
    return out


def run_rtl433(path, freq, extra=()):
    cmd = ["rtl_433", "-r", path, "-s", "2000000", "-f", freq, "-F", "json", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    recs = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                recs.append(normalize_rtl(json.loads(line)))
            except ValueError:
                pass
    return recs, " ".join(cmd)


def rtl_protocols(device):
    """The -R numbers whose help line names the device."""
    p = subprocess.run(["rtl_433", "-R", "help"], capture_output=True, text=True)
    nums = []
    for line in (p.stdout + p.stderr).splitlines():
        m = re.match(r"\s*\[(\d+)\]\s+(.*)", line)
        if m and any(w in m.group(2) for w in RTL_HELP_WORDS.get(device, [])):
            nums.append(m.group(1))
    return nums


def rtl_decode(path, freq, device):
    """rtl_433 with its default decoders; when it finds nothing, with -R N, then with -G.
    Returns (records, how, command)."""
    recs, cmd = run_rtl433(path, freq)
    if recs:
        return recs, "default", cmd
    for n in rtl_protocols(device):
        recs, cmd = run_rtl433(path, freq, ("-R", n))
        if recs:
            return recs, "-R " + n, cmd
    recs, cmd = run_rtl433(path, freq, ("-G",))
    return recs, "-G" if recs else "none", cmd


def run_ours(binary, path, band):
    p = subprocess.run([binary, "decode", path, "--band", str(band)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("our decoder failed: " + p.stderr.strip())
    recs = []
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            recs.append(normalize_ours(json.loads(line)))
    return recs


def diff(device, a, b, an, bn, strict_sides=True):
    """Differences between two canonical readings a and b (names an, bn): a list of strings."""
    out = []
    for f in FIELDS:
        if f in a and f in b:
            x, y = a[f], b[f]
            if isinstance(x, bool) or isinstance(y, bool):
                if bool(x) != bool(y):
                    out.append("%s: %s=%s %s=%s" % (f, an, x, bn, y))
            elif f in ("id", "channel", "humidity"):
                if round(x) != round(y):
                    out.append("%s: %s=%s %s=%s" % (f, an, fmt(x), bn, fmt(y)))
            elif abs(x - y) > tol_for(device, f) + 1e-6:
                out.append("%s: %s=%s %s=%s (tol %s)" % (f, an, fmt(x), bn, fmt(y), tol_for(device, f)))
        elif strict_sides and (f in a) != (f in b):
            side, other = (an, bn) if f in a else (bn, an)
            out.append("%s: only in %s=%s" % (f, side, fmt((a if f in a else b)[f])))
    return out


def fmt(x):
    if isinstance(x, float):
        return ("%.3f" % x).rstrip("0").rstrip(".")
    return str(x)


def assign(sent_times, recs):
    """The records of every slot (a slot runs from its start to the start of the next one).
    Returns (list of lists, records outside of every slot)."""
    slots = [[] for _ in sent_times]
    stray = []
    for r in recs:
        t = r.get("_t")
        k = None
        if t is not None:
            for i, s in enumerate(sent_times):
                end = sent_times[i + 1] if i + 1 < len(sent_times) else math.inf
                if s - 0.01 <= t < end - 0.01:
                    k = i
                    break
        if k is None:
            stray.append(r)
        else:
            slots[k].append(r)
    return slots, stray


def check_device(side_path, ours_bin, directory):
    side = json.load(open(side_path))
    device = side["device"]
    path = os.path.join(directory, side["file"])
    band = 868 if side["modulation"] == "fsk" else 433
    freq = "868.3M" if band == 868 else "433.92M"
    sent = [normalize_sent(r["reading"]) for r in side["readings"]]
    times = [r["t_s"] for r in side["readings"]]
    res = {"device": device, "file": path, "sent": len(sent), "mismatches": []}
    rtl, how, cmd = rtl_decode(path, freq, device)
    res["rtl_how"], res["rtl_cmd"] = how, cmd
    ours = run_ours(ours_bin, path, band)
    res["rtl_decoded"], res["ours_decoded"] = len(rtl), len(ours)
    res["rtl_models"] = sorted({r["_model"] for r in rtl})
    cmap = RTL_CHANNEL_MAP.get(device, {})
    res["rtl_channel_mapped"] = 0
    for r in rtl:
        if r.get("channel") in cmap:
            r["channel"] = cmap[r["channel"]]
            res["rtl_channel_mapped"] += 1
    rslots, rstray = assign(times, rtl)
    oslots, ostray = assign(times, ours)
    rm = om = eq = 0
    for i, s in enumerate(sent):
        rr = rslots[i][0] if rslots[i] else None
        oo = oslots[i][0] if oslots[i] else None
        item = {"index": i, "sent": s}
        if rr is None:
            res["mismatches"].append(dict(item, what="rtl_433 decoded nothing"))
        else:
            d = diff(device, s, rr, "sent", "rtl_433", strict_sides=False)
            extra = [f for f in FIELDS if f in rr and f not in s]
            item["rtl_433"] = {k: v for k, v in rr.items() if not k.startswith("_")}
            item["rtl_model"] = rr["_model"]
            if d:
                res["mismatches"].append(dict(item, what="rtl_433 differs from sent", diffs=d))
            else:
                rm += 1
            if extra:
                item["rtl_extra_fields"] = extra
        if oo is None:
            res["mismatches"].append(dict(item, what="our decoder decoded nothing"))
        else:
            d = diff(device, s, oo, "sent", "ours", strict_sides=False)
            item["ours"] = {k: v for k, v in oo.items() if not k.startswith("_")}
            if d:
                res["mismatches"].append(dict(item, what="our decoder differs from sent", diffs=d))
            else:
                om += 1
        if rr is not None and oo is not None:
            d = diff(device, oo, rr, "ours", "rtl_433")
            if d:
                res["mismatches"].append(dict(item, what="our decoder differs from rtl_433", diffs=d))
            else:
                eq += 1
    res["rtl_matched"], res["ours_matched"], res["equal"] = rm, om, eq
    res["rtl_extra_records"] = sum(max(0, len(x) - 1) for x in rslots) + len(rstray)
    res["ours_extra_records"] = sum(max(0, len(x) - 1) for x in oslots) + len(ostray)
    return res


def check_recording(spec, ours_bin, band):
    parts = spec.split(":")
    path = parts[0]
    expect = None
    if len(parts) == 3:
        expect = (float(parts[1]), float(parts[2]))
    freq = "868.3M" if band == 868 else "433.92M"
    rtl, how, cmd = rtl_decode(path, freq, "bresser_3ch" if band == 433 else "lacrosse_tx29")
    ours = run_ours(ours_bin, path, band)
    res = {"recording": path, "rtl_how": how, "rtl_cmd": cmd, "rtl_decoded": len(rtl),
           "ours_decoded": len(ours), "mismatches": [], "pairs": 0, "equal": 0}
    # pair by time: the nearest record of the other side within 0.5 s
    used = set()
    for r in rtl:
        best, bd = None, 0.5
        for j, o in enumerate(ours):
            if j in used or r["_t"] is None or o["_t"] is None:
                continue
            if abs(o["_t"] - r["_t"]) < bd:
                best, bd = j, abs(o["_t"] - r["_t"])
        if best is None:
            res["mismatches"].append({"what": "only rtl_433 decoded it", "rtl_433": clean(r)})
            continue
        used.add(best)
        o = ours[best]
        res["pairs"] += 1
        d = diff("bresser_3ch", o, r, "ours", "rtl_433")
        if d:
            res["mismatches"].append({"what": "our decoder differs from rtl_433", "diffs": d,
                                      "rtl_433": clean(r), "ours": clean(o)})
        else:
            res["equal"] += 1
    for j, o in enumerate(ours):
        if j not in used:
            res["mismatches"].append({"what": "only our decoder decoded it", "ours": clean(o)})
    if expect is not None:
        res["expected"] = {"temperature_C": expect[0], "humidity": expect[1]}
        for name, recs in (("rtl_433", rtl), ("ours", ours)):
            good = [r for r in recs if abs(r.get("temperature_C", 1e9) - expect[0]) <= 0.1 + 1e-6
                    and abs(r.get("humidity", 1e9) - expect[1]) <= 0.5]
            res["display_" + name] = "%d of %d show %.1f C %d %%" % (len(good), len(recs), expect[0], expect[1])
            if len(good) != len(recs) or not recs:
                res["mismatches"].append({"what": name + " differs from the display", "values": [clean(r) for r in recs]})
    return res


def clean(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def print_device(res, out):
    out.append("== %s  (rtl_433: %s; models seen: %s)" % (res["device"], res["rtl_how"], ", ".join(res["rtl_models"]) or "-"))
    out.append("   sent %d | rtl_433 decoded %d (matched %d) | ours decoded %d (matched %d) | ours == rtl_433 for %d" % (
        res["sent"], res["rtl_decoded"], res["rtl_matched"], res["ours_decoded"], res["ours_matched"], res["equal"]))
    if res["rtl_channel_mapped"]:
        out.append("   note: %d rtl_433 channel values mapped (raw Oregon nibble 4 = switch position 3)" % res["rtl_channel_mapped"])
    if res["rtl_extra_records"] or res["ours_extra_records"]:
        out.append("   extra records (repeats or strays): rtl_433 %d, ours %d" % (res["rtl_extra_records"], res["ours_extra_records"]))
    for m in res["mismatches"]:
        out.append("   [%d] %s" % (m["index"], m["what"]))
        out.append("       sent    : %s" % json.dumps(m["sent"], sort_keys=True))
        if "rtl_433" in m:
            out.append("       rtl_433 : %s %s" % (m.get("rtl_model", ""), json.dumps(m["rtl_433"], sort_keys=True)))
        if "ours" in m:
            out.append("       ours    : %s" % json.dumps(m["ours"], sort_keys=True))
        for d in m.get("diffs", []):
            out.append("       diff    : " + d)


def print_recording(res, out):
    out.append("== %s  (rtl_433: %s)" % (res["recording"], res["rtl_how"]))
    out.append("   rtl_433 decoded %d | ours decoded %d | paired %d | equal %d" % (
        res["rtl_decoded"], res["ours_decoded"], res["pairs"], res["equal"]))
    for k in ("display_rtl_433", "display_ours"):
        if k in res:
            out.append("   %s: %s" % (k, res[k]))
    for m in res["mismatches"]:
        out.append("   %s" % m["what"])
        for k in ("rtl_433", "ours", "values"):
            if k in m:
                out.append("       %-8s: %s" % (k, json.dumps(m[k], sort_keys=True)))
        for d in m.get("diffs", []):
            out.append("       diff    : " + d)


def main():
    ap = argparse.ArgumentParser(
        description="Compare the SDRk sensor encoders and decoders with rtl_433 on IQ files "
                    "(see tools/README-crosscheck.md). Exit status 0 only if everything matches, "
                    "77 when rtl_433 is not installed.")
    ap.add_argument("--iq-dir", help="directory written by `sensor_iq generate` (DEVICE.cu8 and DEVICE.json)")
    ap.add_argument("--only", action="append", help="check only this device (repeatable)")
    ap.add_argument("--recording", action="append", default=[], metavar="FILE[:TEMP_C:HUMIDITY]",
                    help="a real recording (cu8, 2 MS/s): rtl_433 against our decoder, and against the display values when given")
    ap.add_argument("--band", type=int, default=433, choices=(433, 868), help="band of the recordings (default 433)")
    ap.add_argument("--ours", default=DEFAULT_OURS, help="the example binary sensor_iq (default: %(default)s)")
    args = ap.parse_args()
    if not args.iq_dir and not args.recording:
        ap.error("give --iq-dir and/or --recording")
    if shutil.which("rtl_433") is None:
        print("rtl_433 is not installed: the cross-check is skipped", file=sys.stderr)
        return 77
    if not os.path.exists(args.ours):
        print("%s not found: build it with `cargo build --release -p sdrk-pipeline --example sensor_iq`" % args.ours, file=sys.stderr)
        return 2
    results, recordings = [], []
    if args.iq_dir:
        for s in sorted(glob.glob(os.path.join(args.iq_dir, "*.json"))):
            if args.only and os.path.basename(s)[:-5] not in args.only:
                continue
            results.append(check_device(s, args.ours, args.iq_dir))
    for spec in args.recording:
        recordings.append(check_recording(spec, args.ours, args.band))
    ok = all(not r["mismatches"] and r["sent"] == r["rtl_matched"] == r["ours_matched"] == r["equal"] for r in results)
    ok = ok and all(not r["mismatches"] for r in recordings)
    out = []
    for r in results:
        print_device(r, out)
    for r in recordings:
        print_recording(r, out)
    out.append("RESULT: " + ("all match" if ok else "MISMATCHES"))
    print("\n".join(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
