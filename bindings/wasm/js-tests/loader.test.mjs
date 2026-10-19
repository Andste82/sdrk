// The loader: variants, automatic selection and the reproducibility rule.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  FIXTURE, FM_RECEIVER, SIGNAL_SAMPLES, audioSha256, audioSinad, fmReceiveSignal, hasFile,
  installedPackage, receive, sha256,
} from "./helpers.mjs";

test("the receive signal of the JS tests is the one of the Rust tests", () => {
  // Both generators use the same fdlibm algorithms for sin and cos; a different engine would
  // make the checksum test below meaningless, so it is checked first.
  const cu8 = fmReceiveSignal(SIGNAL_SAMPLES);
  assert.equal(sha256(cu8), FIXTURE.input_sha256);
});

test("the scalar variant reproduces the checksum of the fixture bit for bit", async () => {
  const { module } = await installedPackage();
  const api = await module.init({ variant: "scalar" });
  assert.equal(api.variant, "scalar");
  const pipeline = api.Pipeline.fromJson(FM_RECEIVER);
  const audio = receive(pipeline, fmReceiveSignal(SIGNAL_SAMPLES));
  assert.equal(audio.length, FIXTURE.audio_samples);
  assert.equal(audioSha256(audio), FIXTURE.audio_sha256);
  assert.ok(Math.abs(audioSinad(audio) - FIXTURE.sinad_db) < 0.01, `${audioSinad(audio)}`);
  pipeline.free();
});

test("the simd variant meets the SINAD rule of the fixture", async (t) => {
  const { module } = await installedPackage();
  if (!module.simdSupported()) return t.skip("this engine has no SIMD128");
  const api = await module.init({ variant: "simd" });
  assert.equal(api.variant, "simd");
  const pipeline = api.Pipeline.fromJson(FM_RECEIVER);
  const audio = receive(pipeline, fmReceiveSignal(SIGNAL_SAMPLES));
  const sinad = audioSinad(audio);
  assert.ok(sinad >= FIXTURE.sinad_db - 0.5, `SINAD ${sinad} dB, fixture ${FIXTURE.sinad_db} dB`);
  assert.ok(sinad >= 40);
  pipeline.free();
});

test("the relaxed variant meets the same rule when it is in the package", async (t) => {
  if (!(await hasFile("sdrk_relaxed.wasm"))) return t.skip("the package was built without --relaxed");
  const { module } = await installedPackage();
  const api = await module.init({ variant: "relaxed" });
  assert.equal(api.variant, "relaxed");
  const pipeline = api.Pipeline.fromJson(FM_RECEIVER);
  const audio = receive(pipeline, fmReceiveSignal(SIGNAL_SAMPLES));
  assert.ok(audioSinad(audio) >= FIXTURE.sinad_db - 0.5);
  pipeline.free();
});

test("automatic selection picks simd where SIMD128 is available", async () => {
  const { module } = await installedPackage();
  const expected = module.simdSupported() ? "simd" : "scalar";
  assert.equal(module.defaultVariant(), expected);
  const api = await module.init();
  assert.equal(api.variant, expected);
  const auto = await module.init({ variant: "auto" });
  assert.equal(auto.variant, expected);
  assert.equal(module.loadedVariant() !== null, true);
});

test("backends() tells the variants apart", async () => {
  const { module } = await installedPackage();
  const scalar = await module.init({ variant: "scalar" });
  for (const backend of Object.values(scalar.backends())) {
    assert.ok(["fallback", "chained"].includes(backend), backend);
  }
  if (module.simdSupported()) {
    const simd = await module.init({ variant: "simd" });
    assert.equal(simd.backends().rotate, "simd");
    assert.equal(simd.backends().ddc, "chained");
  }
});

test("both variants can be loaded side by side and have their own error class", async () => {
  const { module } = await installedPackage();
  const scalar = await module.init({ variant: "scalar" });
  const other = await module.init({ variant: module.simdSupported() ? "simd" : "scalar" });
  assert.equal(typeof scalar.SdrkError, "function");
  try {
    scalar.Pipeline.fromJson("{");
    assert.fail("should throw");
  } catch (error) {
    assert.ok(error instanceof scalar.SdrkError);
    assert.ok(error instanceof Error);
    assert.equal(error.name, "SdrkError");
    assert.equal(error.kind, "Validation");
    assert.equal(error.code, "syntax");
  }
  assert.equal(scalar.version(), other.version());
});

test("an unknown variant is rejected and the wasm file can be given explicitly", async () => {
  const { module, dir } = await installedPackage();
  await assert.rejects(() => module.init({ variant: "fast" }), TypeError);
  const { readFile } = await import("node:fs/promises");
  const { join } = await import("node:path");
  const bytes = await readFile(join(dir, "node_modules", "sdrk-wasm", "sdrk_scalar.wasm"));
  const api = await module.init({ variant: "scalar", wasm: bytes });
  assert.equal(api.variant, "scalar");
});
