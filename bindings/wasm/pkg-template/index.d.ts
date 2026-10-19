// Types of the SDRk WebAssembly package.

export { FlowBuilder } from "./builder.js";
export type {
  FlowObject,
  FlowSettings,
  FormatSpec,
  InputSpec,
  NodeOptions,
  OutputSpec,
  ParamValue,
} from "./builder.js";

/** The builds of the library. */
export type Variant = "simd" | "scalar" | "relaxed";

/** Options of {@link init}. */
export interface InitOptions {
  /**
   * "auto" (the default) picks "simd" where the engine has WebAssembly SIMD128 and "scalar"
   * otherwise. "relaxed" (SIMD128 plus relaxed SIMD) is never chosen automatically.
   */
  variant?: "auto" | Variant;
  /** The bytes, the compiled module, the URL or the response of the `.wasm` file. */
  wasm?: BufferSource | WebAssembly.Module | URL | string | Response | Promise<Response>;
}

/** The API of one loaded build. */
export interface Api {
  readonly variant: Variant;
  readonly Pipeline: typeof Pipeline;
  readonly validate: typeof validate;
  readonly schema: typeof schema;
  readonly backends: typeof backends;
  readonly version: typeof version;
  readonly SdrkError: typeof SdrkError;
}

/**
 * Loads a build and returns its API. The first call also binds the exports `Pipeline`,
 * `validate`, `schema`, `backends`, `version` and `SdrkError` of this module.
 */
export function init(options?: InitOptions): Promise<Api>;
/** True if this engine runs WebAssembly SIMD128. */
export function simdSupported(): boolean;
/** The variant `init()` picks without an explicit `variant`. */
export function defaultVariant(): "simd" | "scalar";
/** The variant behind the module-level exports, or `null` before `init`. */
export function loadedVariant(): Variant | null;

/** The kinds of errors, named like the Rust `ErrorKind`. */
export type ErrorKind =
  | "InvalidArgument"
  | "LengthMismatch"
  | "BufferTooSmall"
  | "OutOfRange"
  | "Unsupported"
  | "CapacityExceeded"
  | "InvalidState"
  | "ResourceLimit"
  | "Parse"
  | "Validation"
  | "Io"
  | "Device"
  | "Timeout"
  | "Disconnected"
  | "Internal";

/** The codes of {@link Diagnostic}. */
export type DiagCode =
  | "syntax"
  | "schema_version"
  | "unknown_field"
  | "invalid_name"
  | "duplicate_name"
  | "unknown_kind"
  | "unknown_param"
  | "wrong_type"
  | "out_of_range"
  | "missing_param"
  | "invalid_value"
  | "unknown_endpoint"
  | "multiple_inputs"
  | "missing_input"
  | "dangling_output"
  | "cycle"
  | "type_mismatch"
  | "rate_unsupported"
  | "limit_exceeded"
  | "unknown_group"
  | "backend_unavailable";

/** One problem of a pipeline description. */
export interface Diagnostic {
  /** Location, for example `nodes.tune.params.frequency_hz`. */
  path: string;
  code: DiagCode;
  message: string;
}

/** What validation found out about one node. */
export interface NodeReport {
  name: string;
  kind: string;
  /** Sample rate at the input: an integer, or "num/den". */
  rate_in: number | string;
  rate_out: number | string;
  /** Group delay in input samples, "num/den". */
  delay_samples_in: string;
  /** True for nodes that the pipeline inserted (conversions). */
  auto: boolean;
}

/** Latency figures of one output. */
export interface OutputReport {
  name: string;
  /** Group delay in samples of the output, "num/den". */
  group_delay: string;
  max_buffered: number;
}

/** The result of validation. The field names are those of the Rust report. */
export interface Report {
  errors: Diagnostic[];
  warnings: Diagnostic[];
  memory_bytes: number;
  cost_per_input_sample: number;
  nodes: NodeReport[];
  outputs: OutputReport[];
  group_edges?: { from: string; to: string; from_group: string; to_group: string }[];
}

/** The error that every failing call throws. */
export class SdrkError extends Error {
  readonly name: "SdrkError";
  readonly kind: ErrorKind;
  /** Diagnostic code of the first problem of a failed build. */
  readonly code?: DiagCode;
  /** Location of the first problem of a failed build. */
  readonly path?: string;
  /** All problems of a failed build. */
  readonly diagnostics: Diagnostic[];
}

/** Typed array that carries the samples of a stream. */
export type Samples = Uint8Array | Int16Array | Float32Array;

/** Flags of {@link BlockMeta.flags}: OR them together. */
export const BlockFlags: {
  readonly GapBefore: 1;
  readonly Overflow: 2;
  readonly Underrun: 4;
  readonly BurstStart: 8;
  readonly BurstEnd: 16;
  readonly Late: 32;
  readonly Retuned: 64;
};

/** An absolute time for one sample. */
export interface TimeAnchor {
  index: bigint;
  unixNs: bigint;
  source: "device" | "host" | "gps" | "unknown";
}

/** Metadata of one block of samples. */
export interface BlockMeta {
  /** Index of the first sample (default: right after the previous block). */
  start?: bigint | number;
  /** OR of {@link BlockFlags}. */
  flags?: number;
  anchor?: { index: bigint | number; unixNs: bigint | number; source?: TimeAnchor["source"] };
  /** Transmit: requested start time of a burst in unix nanoseconds. */
  txTimeNs?: bigint | number;
}

/** Result of {@link Pipeline.pull}. */
export interface PullResult {
  /** Samples written to `out` (bytes for a raw output). */
  samples: number;
  /** Metadata of the first sample. */
  meta: { start: bigint; flags: number; anchor?: TimeAnchor; txTimeNs?: bigint };
}

/** Metadata of a frame of a frames output. */
export interface FrameMeta {
  /** Input index of the first sample of the frame. */
  firstSample: bigint;
  /** Number of windows averaged into the frame. */
  averaged: number;
  exponent: number;
}

/** When a timed parameter change takes effect. */
export type ParamTime = { input: string; index: bigint | number } | { unixNs: bigint | number };

/** Statistics of one node. */
export interface NodeStats {
  name: string;
  calls: bigint;
  samplesIn: bigint;
  samplesOut: bigint;
  frames: bigint;
}

interface EventBase {
  /** The output the event belongs to. */
  output: string;
  /** Sample index on that output. */
  index: bigint;
}

/** Something that happened at an output. */
export type SdrkEvent =
  | (EventBase & { type: "gap"; missing: bigint })
  | (EventBase & { type: "overflow" })
  | (EventBase & { type: "underrun" })
  | (EventBase & { type: "anchor"; anchor: TimeAnchor })
  | (EventBase & { type: "retuned" })
  | (EventBase & { type: "burstStart"; txTimeNs?: bigint })
  | (EventBase & { type: "burstEnd" })
  | (EventBase & { type: "late" })
  | (EventBase & {
      type: "paramApplied";
      node: string;
      param: string | null;
      value: number | boolean | string;
      late: boolean;
    })
  | (EventBase & { type: "eventsLost"; count: bigint })
  | (EventBase & { type: "unknown" });

/** The supplier of a pull call: at most `max` samples for `input`, or null when exhausted. */
export type PullSource = (input: string, max: number) => Samples | null | undefined;

/** A built pipeline. Not shared between workers: one pipeline per worker. */
export class Pipeline {
  private constructor();
  /** Builds a pipeline from JSON text. Throws {@link SdrkError} with the diagnostics. */
  static fromJson(flowJson: string): Pipeline;
  /**
   * Pushes samples into an input. The array type must match the input: `Uint8Array` for raw
   * bytes, `Float32Array` for `c32` (interleaved re, im) and `f32`, `Int16Array` for `cq15`
   * and `q15`.
   */
  push(input: string, data: Samples, meta?: BlockMeta): void;
  /**
   * Produces data for a transmit output (pull mode); `source` supplies the input data. At most
   * {@link Pipeline.chunkSize} samples per call.
   */
  pull(output: string, out: Samples, source: PullSource): PullResult;
  /** Reads samples of a stream output (complex ones interleaved); returns the sample count. */
  readStream(output: string, out: Samples): number;
  /** Reads the oldest frame of a frames output; `null` if there is none. */
  readFrame(output: string, frame: Float32Array): FrameMeta | null;
  /** Samples (bytes for a raw output, frames for a frames output) waiting in an output. */
  available(output: string): number;
  /** Samples one `readStream` or `pull` call moves at most for an output. */
  chunkSize(output: string): number;
  /** Samples (frames) the drop policy of an output has discarded. */
  dropped(output: string): bigint;
  /** Changes a run-time parameter, immediately or at a time. */
  setParam(node: string, name: string, value: number | boolean | string, at?: ParamTime): void;
  /** The oldest event, or `null`. */
  pollEvent(): SdrkEvent | null;
  beginBurst(output: string, txTimeNs?: bigint): void;
  endBurst(output: string): void;
  /** Group delay and worst-case buffering of an output, in samples of the output. */
  latency(output: string): { groupDelay: number; maxBuffered: number };
  /** The configuration as JSON text (run-time parameters at their current values). */
  config(): string;
  report(): Report;
  stats(): NodeStats[];
  /** Resets the state of all nodes and discards queued data and events. */
  reset(): void;
  /** Releases the memory of the pipeline. */
  free(): void;
}

/** The version of the library. */
export function version(): string;
/** The JSON Schema of the pipeline description, as text. */
export function schema(): string;
/** Validates a description; never throws on an invalid one. */
export function validate(flowJson: string): Report;
/** The backend of every kernel in this build ("simd", "fallback", "chained"). */
export function backends(): Record<string, string>;
