// Shared helpers of the JavaScript tests: installing the packed tarball, the receive signal,
// a small SINAD meter and the fixtures.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..", "..");

export const FLOWS = join(root, "examples", "flows");
export const FM_RECEIVER_PATH = join(FLOWS, "fm_receiver.json");
export const FM_RECEIVER = readFileSync(FM_RECEIVER_PATH, "utf8");
export const FIXTURE = JSON.parse(
  readFileSync(join(here, "..", "tests", "fixtures", "fm_receiver.json"), "utf8"),
);

/** The tarball made by `cargo xtask wasm-pkg` (or the one named by `SDRK_WASM_TGZ`). */
export function tarballPath() {
  if (process.env.SDRK_WASM_TGZ) return resolve(process.env.SDRK_WASM_TGZ);
  const dir = join(root, "target", "wasm-pkg-tgz");
  const files = existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".tgz")) : [];
  if (files.length === 0) {
    throw new Error(`no tarball in ${dir}; run \`cargo xtask wasm-pkg\` first`);
  }
  return join(dir, files.sort().at(-1));
}

let installed;

/**
 * Installs the tarball into a fresh temporary project (`npm install <tgz>`) and returns the
 * module of the installed package, imported from there.
 */
export async function installedPackage() {
  if (!installed) {
    installed = (async () => {
      const dir = mkdtempSync(join(tmpdir(), "sdrk-wasm-test-"));
      writeFileSync(
        join(dir, "package.json"),
        JSON.stringify({ name: "sdrk-wasm-consumer", private: true, type: "module" }),
      );
      const npm = process.platform === "win32" ? "npm.cmd" : "npm";
      execFileSync(
        npm,
        ["install", "--no-audit", "--no-fund", "--no-progress", tarballPath()],
        { cwd: dir, stdio: "pipe", shell: process.platform === "win32" },
      );
      const entry = join(dir, "node_modules", "sdrk-wasm", "index.js");
      return { dir, entry, module: await import(pathToFileURL(entry).href) };
    })();
  }
  return installed;
}

/** True if the installed package contains the file (for example `sdrk_relaxed.wasm`). */
export async function hasFile(name) {
  const { dir } = await installedPackage();
  return existsSync(join(dir, "node_modules", "sdrk-wasm", name));
}

export function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

// --------------------------------------------------------------------------------------
// The E2E-RX signal: an FM broadcast at +600 kHz with a 1 kHz tone at 2.4 MS/s, as CU8. A port
// of `sdrk_testkit::fm::fm_baseband` and of the CU8 encoder; `Math.sin`/`Math.cos` of V8 are
// the same fdlibm algorithms as the `libm` crate, so the bytes are identical (checked against
// `input_sha256` of the fixture).

const TAU = 2 * Math.PI;
const frac = (x) => x - Math.floor(x);

export const IF_RATE = 2_400_000;
export const CARRIER_OFFSET = 600_000;
export const DEVIATION = 75_000;
export const AUDIO_HZ = 1_000;
export const AMPLITUDE = 0.9;

/** The CU8 bytes of `samples` samples of the receive signal. */
export function fmReceiveSignal(samples) {
  const bytes = new Uint8Array(2 * samples);
  const beta = (DEVIATION * 1.0) / AUDIO_HZ;
  for (let n = 0; n < samples; n++) {
    const carrier = TAU * frac((CARRIER_OFFSET * n) / IF_RATE);
    const audio = TAU * frac((AUDIO_HZ * n) / IF_RATE);
    const phase = carrier + beta * (1 - Math.cos(audio));
    const re = Math.cos(phase) * AMPLITUDE;
    const im = Math.sin(phase) * AMPLITUDE;
    bytes[2 * n] = encodeU8(re);
    bytes[2 * n + 1] = encodeU8(im);
  }
  return bytes;
}

function encodeU8(x) {
  const v = Math.floor(x * 128 + 127.5 + 0.5);
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** Samples of the receive signal used by the tests: 0.25 s, like the Rust tests. */
export const SIGNAL_SAMPLES = IF_RATE / 4;
export const PUSH_SAMPLES = 4096;
export const SINAD_SKIP = 1200;

/** Pushes the signal through a pipeline and returns all audio as one Float32Array. */
export function receive(pipeline, cu8) {
  const blocks = [];
  let total = 0;
  const out = new Float32Array(8192);
  for (let at = 0; at < cu8.length; at += 2 * PUSH_SAMPLES) {
    pipeline.push("rf", cu8.subarray(at, at + 2 * PUSH_SAMPLES));
    for (;;) {
      const n = pipeline.readStream("audio", out);
      if (n === 0) break;
      blocks.push(out.slice(0, n));
      total += n;
    }
  }
  const audio = new Float32Array(total);
  let at = 0;
  for (const block of blocks) {
    audio.set(block, at);
    at += block.length;
  }
  return audio;
}

/**
 * SINAD in dB of a 1 kHz tone: least-squares fit of cos, sin and an offset at the known
 * frequency; the rest is noise and distortion.
 */
export function sinadDb(signal, rate, freq) {
  const n = signal.length;
  let scc = 0, sss = 0, scs = 0, sc = 0, ss = 0, xc = 0, xs = 0, x1 = 0;
  for (let i = 0; i < n; i++) {
    const a = (TAU * freq * i) / rate;
    const c = Math.cos(a);
    const s = Math.sin(a);
    const x = signal[i];
    scc += c * c; sss += s * s; scs += c * s; sc += c; ss += s;
    xc += x * c; xs += x * s; x1 += x;
  }
  const m = [[scc, scs, sc], [scs, sss, ss], [sc, ss, n]];
  const rhs = [xc, xs, x1];
  const det3 = (a) =>
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  const det = det3(m);
  const column = (k) => det3(m.map((row, r) => row.map((v, c) => (c === k ? rhs[r] : v)))) / det;
  const [a, b, dc] = [column(0), column(1), column(2)];
  let residual = 0, total = 0;
  for (let i = 0; i < n; i++) {
    const t = (TAU * freq * i) / rate;
    const r = signal[i] - a * Math.cos(t) - b * Math.sin(t) - dc;
    residual += r * r;
    total += (signal[i] - dc) * (signal[i] - dc);
  }
  return residual <= 0 ? Infinity : 10 * Math.log10(total / residual);
}

/** SINAD of the audio of the receive test (the first samples are skipped). */
export function audioSinad(audio) {
  return sinadDb(audio.subarray(SINAD_SKIP), 48_000, 1_000);
}

export function audioSha256(audio) {
  return sha256(new Uint8Array(audio.buffer, audio.byteOffset, audio.byteLength));
}
