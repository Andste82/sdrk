// Loader of the SDRk WebAssembly package.
//
//   import { init, Pipeline, FlowBuilder } from "sdrk-wasm";
//   await init();                       // picks the SIMD128 build where the engine has it
//   const pipeline = Pipeline.fromJson(flowJson);
//
// `init` loads one of up to three builds of the same code: `simd` (WebAssembly SIMD128),
// `scalar` (no SIMD) and, only when asked for with `init({ variant: "relaxed" })`, `relaxed`
// (SIMD128 plus relaxed SIMD). Works in browsers (fetch and instantiateStreaming) and in Node 20
// or later (the file is read with `fs`).

import { FlowBuilder } from "./builder.js";

export { FlowBuilder };

/** Flags of `BlockMeta.flags` (bit values of the Rust `BlockFlags`). */
export const BlockFlags = Object.freeze({
  GapBefore: 1,
  Overflow: 2,
  Underrun: 4,
  BurstStart: 8,
  BurstEnd: 16,
  Late: 32,
  Retuned: 64,
});

/** A module with one function that uses a v128 instruction: valid only with SIMD128. */
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253,
  15, 253, 98, 11,
]);

/** True if this engine runs WebAssembly SIMD128. */
export function simdSupported() {
  try {
    return typeof WebAssembly === "object" && WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
}

const VARIANTS = ["simd", "scalar", "relaxed"];

/** Which variant `init()` picks without an explicit `variant`. */
export function defaultVariant() {
  return simdSupported() ? "simd" : "scalar";
}

function isNode() {
  return (
    typeof process !== "undefined" &&
    process.versions != null &&
    process.versions.node != null &&
    typeof window === "undefined"
  );
}

async function importVariant(variant) {
  // Literal specifiers, so that bundlers see every file.
  switch (variant) {
    case "simd":
      return import("./sdrk_simd.js");
    case "scalar":
      return import("./sdrk_scalar.js");
    case "relaxed":
      return import("./sdrk_relaxed.js");
    default:
      throw new TypeError(`unknown variant "${variant}"; use ${VARIANTS.join(", ")}`);
  }
}

async function wasmSource(variant, override) {
  if (override !== undefined) return override;
  const url = new URL(`./sdrk_${variant}.wasm`, import.meta.url);
  if (isNode()) {
    const name = "node:fs/promises";
    const { readFile } = await import(/* webpackIgnore: true */ /* @vite-ignore */ name);
    return readFile(url);
  }
  return url;
}

const loaded = new Map();
let current = null;

function describe(variant, mod) {
  return Object.freeze({
    variant,
    Pipeline: mod.Pipeline,
    validate: mod.validate,
    schema: mod.schema,
    backends: mod.backends,
    version: mod.version,
    SdrkError: mod.errorClass(),
  });
}

/**
 * Loads a build and returns its API: `{ variant, Pipeline, validate, schema, backends, version,
 * SdrkError }`. The first successful call also binds the module-level exports of this package.
 *
 * @param {{ variant?: "auto" | "simd" | "scalar" | "relaxed", wasm?: BufferSource | WebAssembly.Module | URL | Response | Promise<Response> }} [options]
 *   `variant`: "auto" (the default) picks `simd` where SIMD128 is available and `scalar`
 *   otherwise; "relaxed" is chosen explicitly only. `wasm`: the module bytes or URL, if the
 *   files are served from somewhere else.
 */
export async function init(options = {}) {
  const requested = options.variant ?? "auto";
  const variant = requested === "auto" ? defaultVariant() : requested;
  if (!VARIANTS.includes(variant)) {
    throw new TypeError(`unknown variant "${requested}"; use auto, ${VARIANTS.join(", ")}`);
  }
  let promise = loaded.get(variant);
  if (promise === undefined || options.wasm !== undefined) {
    promise = (async () => {
      const mod = await importVariant(variant);
      await mod.default({ module_or_path: await wasmSource(variant, options.wasm) });
      return describe(variant, mod);
    })();
    loaded.set(variant, promise);
    promise.catch(() => loaded.delete(variant));
  }
  const api = await promise;
  if (current === null) {
    current = api;
    Pipeline = api.Pipeline;
    validate = api.validate;
    schema = api.schema;
    backends = api.backends;
    version = api.version;
    SdrkError = api.SdrkError;
  }
  return api;
}

/** The name of the variant behind the module-level exports, or `null` before `init`. */
export function loadedVariant() {
  return current === null ? null : current.variant;
}

// Bound by the first `init()`; ES module live bindings make them visible to importers.
export let Pipeline;
export let validate;
export let schema;
export let backends;
export let version;
export let SdrkError;
