// The FlowBuilder produces the model of the JSON files.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { FLOWS, FM_RECEIVER, installedPackage } from "./helpers.mjs";

async function builderFor() {
  const { module } = await installedPackage();
  return new module.FlowBuilder("fm-receiver", {
    description: "Receive one FM broadcast station and output 48 kHz audio.",
  })
    .settings({ precision: "f32", max_chunk: 4096 })
    .input("rf", { format: "cu8", rate_hz: 2400000, description: "RTL-SDR IQ" })
    .node("tune", "ddc", { frequency_hz: 600000, decimation: 10, cutoff_hz: 100000 })
    .node("demod", "fm_demod", { deviation_hz: 75000, audio_decimation: 5, deemphasis_us: 50 })
    .node("fft", "spectrum", { fft_size: 1024, averaging: 4 })
    .edge("rf", "tune")
    .edge("rf", "fft")
    .edge("tune", "demod")
    .edge("demod", "audio")
    .edge("fft", "spectrum")
    .output("audio", { type: "stream", capacity: 16384, drop: "error" })
    .output("spectrum", { type: "frames", queue: 4, drop: "drop_oldest" });
}

test("the builder gives the same model as examples/flows/fm_receiver.json", async () => {
  const builder = await builderFor();
  assert.deepStrictEqual(JSON.parse(builder.toJson()), JSON.parse(FM_RECEIVER));
  assert.deepStrictEqual(builder.toObject(), JSON.parse(FM_RECEIVER));
});

test("the builder writes the canonical text of the JSON files", async () => {
  const builder = await builderFor();
  assert.equal(builder.toJson(), FM_RECEIVER);
});

test("the grouped example is built with groups and backend preferences", async () => {
  const { module } = await installedPackage();
  const file = JSON.parse(readFileSync(join(FLOWS, "fm_receiver_grouped.json"), "utf8"));
  const builder = new module.FlowBuilder(file.name, { description: file.description });
  builder.settings(file.settings);
  for (const [name, spec] of Object.entries(file.inputs)) builder.input(name, spec);
  for (const [name, spec] of Object.entries(file.groups ?? {})) builder.group(name, spec);
  for (const [name, node] of Object.entries(file.nodes)) {
    const { kind, params, ...options } = node;
    builder.node(name, kind, params, options);
  }
  for (const edge of file.edges) builder.edge(edge.from, edge.to);
  for (const [name, spec] of Object.entries(file.outputs)) builder.output(name, spec);
  assert.deepStrictEqual(JSON.parse(builder.toJson()), file);
});

test("a built flow validates and builds", async () => {
  const { module } = await installedPackage();
  const api = await module.init();
  const builder = await builderFor();
  const report = api.validate(builder.toJson());
  assert.deepEqual(report.errors, []);
  const pipeline = api.Pipeline.fromJson(builder.toJson());
  assert.equal(JSON.parse(pipeline.config()).name, "fm-receiver");
  pipeline.free();
});

test("the builder does not share state with the objects passed in", async () => {
  const { module } = await installedPackage();
  const params = { frequency_hz: 1 };
  const builder = new module.FlowBuilder("t").node("a", "ddc", params);
  params.frequency_hz = 2;
  assert.equal(builder.toObject().nodes.a.params.frequency_hz, 1);
  assert.throws(() => new module.FlowBuilder(5), TypeError);
});
