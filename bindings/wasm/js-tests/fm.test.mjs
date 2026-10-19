// The FM receive flow through the packed package, with a small JS meter.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  FIXTURE, FM_RECEIVER, SIGNAL_SAMPLES, audioSinad, fmReceiveSignal, installedPackage, receive,
} from "./helpers.mjs";

test("the installed package receives FM with SINAD of at least 40 dB", async () => {
  const { module } = await installedPackage();
  const api = await module.init();
  const pipeline = api.Pipeline.fromJson(FM_RECEIVER);
  try {
    const audio = receive(pipeline, fmReceiveSignal(SIGNAL_SAMPLES));
    assert.ok(audio.length >= FIXTURE.audio_samples - 100, `${audio.length} audio samples`);
    const sinad = audioSinad(audio);
    assert.ok(sinad >= 40, `SINAD ${sinad} dB`);
    assert.ok(sinad >= FIXTURE.sinad_db - 0.5, `SINAD ${sinad} dB, fixture ${FIXTURE.sinad_db} dB`);
  } finally {
    pipeline.free();
  }
});

test("the spectrum output delivers frames", async () => {
  const { module } = await installedPackage();
  const api = await module.init();
  const pipeline = api.Pipeline.fromJson(FM_RECEIVER);
  try {
    const cu8 = fmReceiveSignal(8192);
    pipeline.push("rf", cu8.subarray(0, 8192));
    pipeline.push("rf", cu8.subarray(8192));
    assert.ok(pipeline.available("spectrum") >= 1);
    const frame = new Float32Array(1024);
    const meta = pipeline.readFrame("spectrum", frame);
    assert.equal(typeof meta.firstSample, "bigint");
    assert.equal(meta.averaged, 4);
    assert.ok(frame.some((v) => v !== 0));
    assert.ok(Number.isFinite(frame[0]));
  } finally {
    pipeline.free();
  }
});
