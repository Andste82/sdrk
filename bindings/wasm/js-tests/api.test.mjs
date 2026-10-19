// The API of the package: validation, errors, events, timed parameters, pull mode, bursts and
// the TypeScript declarations.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { FLOWS, FM_RECEIVER, installedPackage } from "./helpers.mjs";

const flow = (name) => readFileSync(join(FLOWS, `${name}.json`), "utf8");

async function api() {
  const { module } = await installedPackage();
  return module.init();
}

test("version() and schema()", async () => {
  const { version, schema } = await api();
  const { module } = await installedPackage();
  const pkg = JSON.parse(readFileSync(join((await installedPackage()).dir, "node_modules", "sdrk-wasm", "package.json"), "utf8"));
  assert.equal(version(), pkg.version);
  const parsed = JSON.parse(schema());
  assert.match(parsed.$schema, /json-schema\.org/);
  assert.equal(typeof module.version, "function");
});

test("validate() is clean for every example and never throws", async () => {
  const { validate } = await api();
  for (const name of ["fm_receiver", "fm_receiver_grouped", "fm_transmitter", "resample_audio", "spectrum_only"]) {
    const report = validate(flow(name));
    assert.deepEqual(report.errors, [], name);
    assert.ok(report.memory_bytes > 0);
  }
  const bad = validate(FM_RECEIVER.replace('"decimation": 10', '"decimation": 0'));
  assert.equal(bad.errors.length, 1);
  assert.equal(bad.errors[0].code, "out_of_range");
  assert.equal(bad.errors[0].path, "nodes.tune.params.decimation");
  assert.equal(validate("{").errors[0].code, "syntax");
  assert.equal(validate("").errors[0].code, "syntax");
});

test("fromJson throws an SdrkError with the diagnostics", async () => {
  const { Pipeline, SdrkError } = await api();
  const bad = FM_RECEIVER.replace('"decimation": 10', '"decimation": 0').replace('"kind": "fm_demod"', '"kind": "nope"');
  try {
    Pipeline.fromJson(bad);
    assert.fail("should throw");
  } catch (error) {
    assert.ok(error instanceof SdrkError);
    assert.equal(error.kind, "Validation");
    assert.equal(error.diagnostics.length, 2);
    assert.equal(error.code, error.diagnostics[0].code);
    assert.equal(error.path, error.diagnostics[0].path);
    assert.ok(error.diagnostics.some((d) => d.code === "unknown_kind" && d.path === "nodes.demod.kind"));
  }
});

test("a typed array of the wrong type throws InvalidArgument", async () => {
  const { Pipeline, SdrkError } = await api();
  const pipeline = Pipeline.fromJson(FM_RECEIVER);
  for (const wrong of [new Float32Array(8), new Int16Array(8), [1, 2, 3], "abc", null]) {
    assert.throws(
      () => pipeline.push("rf", wrong),
      (error) => error instanceof SdrkError && error.kind === "InvalidArgument",
    );
  }
  assert.throws(() => pipeline.readStream("audio", new Int16Array(4)), { kind: "InvalidArgument" });
  assert.throws(() => pipeline.push("nope", new Uint8Array(2)), { kind: "InvalidArgument" });
  assert.throws(() => pipeline.push("rf", new Uint8Array(100_000)), { kind: "LengthMismatch" });
  pipeline.free();
});

test("events, timed parameters and gaps use bigint indices", async () => {
  const { Pipeline } = await api();
  const p = Pipeline.fromJson(FM_RECEIVER);
  assert.equal(p.pollEvent(), null);
  p.setParam("tune", "frequency_hz", 610_000, { input: "rf", index: 1000n });
  p.push("rf", new Uint8Array(4000).fill(128));
  const ack = p.pollEvent();
  assert.equal(ack.type, "paramApplied");
  assert.equal(ack.node, "tune");
  assert.equal(ack.param, "frequency_hz");
  assert.equal(ack.value, 610_000);
  assert.equal(ack.late, false);
  assert.equal(ack.output, "audio");
  assert.equal(ack.index, 20n);
  assert.equal(p.pollEvent(), null);

  p.push("rf", new Uint8Array(2000).fill(128), { start: 6000n });
  const gap = p.pollEvent();
  assert.equal(gap.type, "gap");
  assert.equal(gap.missing, 4000n);
  assert.throws(() => p.push("rf", new Uint8Array(2), { start: 1n }), { kind: "InvalidState" });

  // an anchor
  p.push("rf", new Uint8Array(200).fill(128), {
    anchor: { index: 8000n, unixNs: 1_700_000_000_000_000_000n, source: "gps" },
  });
  let anchor;
  for (let e; (e = p.pollEvent()); ) if (e.type === "anchor") anchor = e;
  assert.equal(anchor.anchor.unixNs, 1_700_000_000_000_000_000n);
  assert.equal(anchor.anchor.source, "gps");

  // timed by absolute time now works, a late request is flagged
  p.setParam("tune", "frequency_hz", 600_000, { unixNs: 1_700_000_000_000_000_000n });
  p.setParam("tune", "frequency_hz", 601_000, { input: "rf", index: 1n });
  assert.throws(() => p.setParam("tune", "frequency_hz", 1, { when: 1 }), { kind: "InvalidArgument" });
  assert.throws(() => p.setParam("tune", "decimation", 5), { kind: "InvalidState" });
  assert.throws(() => p.setParam("tune", "frequency_hz", Number.NaN), { kind: "InvalidArgument" });
  assert.throws(() => p.setParam("tune", "frequency_hz", "fast"), { kind: "InvalidArgument" });
  p.free();
});

test("setParam changes the configuration and reset clears the queues", async () => {
  const { Pipeline } = await api();
  const p = Pipeline.fromJson(FM_RECEIVER);
  p.setParam("tune", "frequency_hz", 620_000);
  assert.equal(JSON.parse(p.config()).nodes.tune.params.frequency_hz, 620_000);
  p.push("rf", new Uint8Array(8000).fill(128));
  assert.ok(p.available("audio") > 0);
  const stats = p.stats();
  assert.ok(stats.length >= 4);
  assert.equal(typeof stats[0].calls, "bigint");
  p.reset();
  assert.equal(p.available("audio"), 0);
  assert.equal(p.pollEvent(), null);
  const latency = p.latency("audio");
  assert.ok(latency.groupDelay > 0 && latency.maxBuffered > 0);
  assert.equal(p.dropped("audio"), 0n);
  assert.equal(p.chunkSize("audio"), 4096);
  p.free();
});

test("pull mode with a JS source and bursts", async () => {
  const { Pipeline } = await api();
  const flowText = flow("fm_transmitter").replace('"underrun": "zeros"', '"underrun": "short"');
  const p = Pipeline.fromJson(flowText);
  let remaining = 960;
  const asked = [];
  const source = (input, max) => {
    asked.push([input, max]);
    const n = Math.min(max, remaining, 480);
    if (n === 0) return null;
    remaining -= n;
    return Float32Array.from({ length: n }, (_, i) => 0.5 * Math.sin(i * 0.1));
  };
  const out = new Uint8Array(4000);
  p.beginBurst("rf", 1_700_000_000_000_000_000n);
  const first = p.pull("rf", out, source);
  assert.equal(first.samples, 4000);
  assert.equal(first.meta.start, 0n);
  assert.equal(first.meta.flags & 8, 8);
  assert.equal(first.meta.txTimeNs, 1_700_000_000_000_000_000n);
  assert.equal(asked[0][0], "audio");
  assert.ok(out.some((b) => b !== 0));
  let total = 4000;
  for (let n; (n = p.pull("rf", out, source).samples) > 0; ) total += n;
  p.endBurst("rf");
  for (let n; (n = p.pull("rf", out, source).samples) > 0; ) total += n;
  assert.ok(total > 90_000 && total < 110_000, `${total}`);
  const types = [];
  for (let e; (e = p.pollEvent()); ) types.push(e.type);
  assert.ok(types.includes("burstStart") && types.includes("burstEnd"), types.join());
  p.free();

  // Errors of the source, in continuous mode (no burst): the example flow fills with zeros.
  const q = Pipeline.fromJson(flow("fm_transmitter"));
  assert.throws(() => q.pull("rf", out, () => new Int16Array(4)), { kind: "InvalidArgument" });
  assert.throws(() => q.pull("rf", out, (_, max) => new Float32Array(max + 1)), { kind: "LengthMismatch" });
  assert.throws(
    () => q.pull("rf", out, () => { throw new RangeError("boom"); }),
    (error) => error instanceof RangeError && error.message === "boom",
  );
  assert.equal(q.pull("rf", out, () => null).samples, out.length); // zeros
  q.free();
});

test("a Q15 flow takes and gives Int16Array with interleaved complex samples", async () => {
  const { Pipeline } = await api();
  const p = Pipeline.fromJson(JSON.stringify({
    sdrk: 1, name: "q15",
    settings: { precision: "q15", max_chunk: 256 },
    inputs: { iq: { type: "cq15", rate_hz: 48000 } },
    nodes: { g: { kind: "scale", params: { gain_db: 0 } } },
    edges: [{ from: "iq", to: "g" }, { from: "g", to: "out" }],
    outputs: { out: { type: "stream" } },
  }));
  p.push("iq", Int16Array.of(1000, -2000, 3000, -4000));
  const out = new Int16Array(8);
  assert.equal(p.readStream("out", out), 2);
  assert.ok(Math.abs(out[0] - 1000) <= 1 && Math.abs(out[3] + 4000) <= 1);
  assert.throws(() => p.push("iq", new Int16Array(3)), { kind: "LengthMismatch" });
  p.free();
});

test("index.d.ts declares every export and every method", async () => {
  const { module, dir } = await installedPackage();
  const api = await module.init();
  const text = readFileSync(join(dir, "node_modules", "sdrk-wasm", "index.d.ts"), "utf8");
  for (const name of Object.keys(module)) {
    assert.match(text, new RegExp(`export (declare )?(class|function|const|type|interface) ${name}\\b|export \\{[^}]*\\b${name}\\b`), `export ${name}`);
  }
  for (const method of Object.getOwnPropertyNames(api.Pipeline.prototype)) {
    if (method === "constructor" || method === "__destroy_into_raw") continue;
    assert.match(text, new RegExp(`\\b${method}\\(`), `method ${method}`);
  }
  for (const name of ["Report", "Diagnostic", "SdrkEvent", "BlockMeta", "FrameMeta", "ParamTime", "PullResult", "NodeStats"]) {
    assert.match(text, new RegExp(`export (type|interface) ${name}\\b`), name);
  }
});
