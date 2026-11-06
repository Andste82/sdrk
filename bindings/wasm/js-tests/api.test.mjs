// The API of the package: validation, errors, events, timed parameters, pull mode, bursts,
// packets and the TypeScript declarations.

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
  for (const name of ["Report", "Diagnostic", "SdrkEvent", "BlockMeta", "FrameMeta", "ParamTime", "PullResult", "NodeStats", "Packet", "PacketIn"]) {
    assert.match(text, new RegExp(`export (type|interface) ${name}\\b`), name);
  }
});

/** The payload of a packet as lower-case hex. */
const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");

test("pushPackets and readPackets carry { index, flags, data } through frame_loop", async () => {
  const { Pipeline, validate } = await api();
  const { PacketFlags } = (await installedPackage()).module;
  assert.deepEqual(validate(flow("frame_loop")).errors, []);
  const p = Pipeline.fromJson(flow("frame_loop"));
  assert.deepEqual(p.readPackets("out"), []);
  const hello = hex(new TextEncoder().encode("hello"));
  const long = hex(Uint8Array.from({ length: 253 }, (_, i) => (i * 7) & 0xff));
  p.pushPackets("msg", [{ data: hello }, { index: 5n, flags: 0, data: "" }, { index: 9, data: long }]);
  assert.equal(p.available("out"), 3);
  const got = p.readPackets("out");
  assert.deepEqual(got.map((q) => q.data), [hello, "", long]);
  for (const q of got) {
    assert.equal(typeof q.index, "bigint");
    assert.equal(q.flags, PacketFlags.CrcOk);
  }
  assert.equal(got[0].index, 32n); // the first sync bit, after 4 preamble bytes
  assert.ok(got[1].index > got[0].index && got[2].index > got[1].index);
  assert.equal(p.available("out"), 0);
  assert.deepEqual(p.readPackets("out"), []);
  assert.equal(p.dropped("out"), 0n);

  // Errors: nothing is taken from a failing push.
  for (const bad of [
    [{ data: "ABCD" }],
    [{ data: "abc" }],
    [{ data: "zz" }],
    [{ data: "00".repeat(257) }],
    [{ flags: 0 }],
    [{ data: "00", index: -1 }],
    [{ data: "00", flags: 1.5 }],
    [42],
    "00",
    null,
  ]) {
    assert.throws(() => p.pushPackets("msg", bad), { kind: "InvalidArgument" }, JSON.stringify(bad));
  }
  assert.throws(() => p.pushPackets("msg", [{ data: hello }, { data: "0g" }]), { kind: "InvalidArgument" });
  assert.equal(p.available("out"), 0);
  assert.throws(() => p.push("msg", new Float32Array(4)), { kind: "InvalidArgument" });
  assert.throws(() => p.readStream("out", new Float32Array(4)), { kind: "InvalidArgument" });
  assert.throws(() => p.readPackets("nope"), { kind: "InvalidArgument" });
  assert.throws(() => p.pushPackets("msg", Array(4097).fill({ data: "" })), { kind: "LengthMismatch" });
  p.free();
});

test("a packet output with drop_oldest keeps the newest packets", async () => {
  const { Pipeline } = await api();
  const spec = JSON.parse(flow("frame_loop"));
  spec.outputs.out = { type: "packets", queue: 2, drop: "drop_oldest" };
  const p = Pipeline.fromJson(JSON.stringify(spec));
  p.pushPackets("msg", ["01", "02", "03", "04"].map((data) => ({ data })));
  assert.equal(p.dropped("out"), 2n);
  assert.deepEqual(p.readPackets("out").map((q) => q.data), ["03", "04"]);
  p.free();
});

test("pull mode with a packet source: null means idle, not the end", async () => {
  const { Pipeline } = await api();
  const tx = Pipeline.fromJson(JSON.stringify({
    sdrk: 1, name: "tx",
    inputs: { msg: { type: "packets", rate_hz: 48000 } },
    nodes: { enc: { kind: "frame_encode", params: { idle: "preamble" } } },
    edges: [{ from: "msg", to: "enc" }, { from: "enc", to: "tx" }],
    outputs: { tx: { type: "stream" } },
  }));
  const rx = Pipeline.fromJson(JSON.stringify({
    sdrk: 1, name: "rx",
    inputs: { bits: { type: "f32", rate_hz: 48000 } },
    nodes: { dec: { kind: "frame_decode" } },
    edges: [{ from: "bits", to: "dec" }, { from: "dec", to: "out" }],
    outputs: { out: { type: "packets", queue: 1024 } },
  }));
  const messages = ["c0ffee", "", "00ff00ff", "736472"];
  let calls = 0;
  let sent = 0;
  const source = (input, max) => {
    assert.equal(input, "msg");
    assert.ok(max >= 1);
    calls += 1;
    if (calls % 2 === 0 || sent >= messages.length) return null; // idle
    return [{ data: messages[sent++] }];
  };
  const out = new Float32Array(1024);
  const got = [];
  for (let i = 0; i < 40; i++) {
    const { samples } = tx.pull("tx", out, source);
    assert.equal(samples, out.length);
    assert.ok(out.every((v) => v === 1 || v === -1)); // the idle signal is the preamble
    rx.push("bits", out);
    got.push(...rx.readPackets("out"));
  }
  assert.equal(sent, messages.length);
  assert.deepEqual(got.map((q) => q.data), messages);
  assert.throws(() => tx.pull("tx", out, () => [{ data: "00" }, { data: "00" }].concat(Array(5000).fill({ data: "" }))), { kind: "LengthMismatch" });
  assert.throws(() => tx.pull("tx", out, () => new Float32Array(4)), { kind: "InvalidArgument" });
  tx.free();
  rx.free();
});

test("RDS: the decoder over rds_decode packets and the text helpers of sdrk_proto::rds", async () => {
  const { RdsDecoder, rdsDecodeText, rdsEncodeText } = await api();
  // The RDS table of IEC 62106: 0x99 is ü, 0x97 ö, 0x8D ß, 0xAB $; 0x24 is ¤.
  assert.equal(rdsDecodeText(new Uint8Array([0x57, 0x99, 0x72]), "rds"), "Wür");
  assert.equal(rdsDecodeText(new Uint8Array([0x67, 0x72, 0x97, 0x8d, 0x74, 0x65, 0x6e])), "größten");
  assert.equal(rdsDecodeText(new Uint8Array([0x24, 0xab])), "¤$");
  // UTF-8 from a station that does not follow the standard.
  const utf8 = new TextEncoder().encode("größten");
  assert.equal(rdsDecodeText(utf8, "auto"), "größten");
  assert.notEqual(rdsDecodeText(utf8, "rds"), "größten");
  assert.throws(() => rdsDecodeText(new Uint8Array([0x41]), "latin1"), /charset/);
  assert.deepEqual(Array.from(rdsEncodeText("Wü$")), [0x57, 0x99, 0xab]);
  assert.throws(() => rdsEncodeText("~"));

  // Four 0A groups give the name, two 2A groups the text; as hex packets like readPackets.
  const hex = (w) => w.map((v) => v.toString(16).padStart(4, "0")).join("");
  const ok = 1 | (0b1111 << 16);
  const name = rdsEncodeText("SWR1 BW ");
  const packets = [0, 1, 2, 3].map((seg) => ({
    index: 0n,
    flags: ok,
    data: hex([0xd3c2, 0x0548 | seg, 0xe0e0, (name[2 * seg] << 8) | name[2 * seg + 1]]),
  }));
  const text = [...rdsEncodeText("Württemberg"), 0x0d, 0x20, 0x20, 0x20, 0x20];
  for (let seg = 0; seg < 3; seg++) {
    const c = (text[4 * seg] << 8) | text[4 * seg + 1];
    const d = (text[4 * seg + 2] << 8) | text[4 * seg + 3];
    packets.push({ index: 0n, flags: ok, data: hex([0xd3c2, 0x2540 | seg, c, d]) });
  }
  const rds = new RdsDecoder();
  try {
    assert.equal(rds.ps(), undefined);
    assert.equal(rds.push(packets), 7);
    assert.equal(rds.pi(), 0xd3c2);
    assert.equal(rds.pty(), 10);
    assert.equal(rds.tp(), true);
    assert.equal(rds.ps("rds"), "SWR1 BW ");
    assert.equal(rds.rt("rds"), "Württemberg");
    assert.equal(rds.groups(), 7);
    assert.equal(rds.okGroups(), 7);
    rds.reset();
    assert.equal(rds.pi(), undefined);
    assert.equal(rds.groups(), 0);
  } finally {
    rds.free();
  }
});
