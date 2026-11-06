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
  readonly RdsDecoder: typeof RdsDecoder;
  readonly rdsDecodeText: typeof rdsDecodeText;
  readonly rdsEncodeText: typeof rdsEncodeText;
}

/**
 * Loads a build and returns its API. The first call also binds the exports `Pipeline`,
 * `validate`, `schema`, `backends`, `version`, `SdrkError`, `RdsDecoder`, `rdsDecodeText` and
 * `rdsEncodeText` of this module.
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
  | "backend_unavailable"
  | "unknown_port"
  | "rate_mismatch"
  | "clock_domain"
  | "misaligned_inputs";

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
  /** True for nodes that the pipeline inserted (conversions and delays that align two paths). */
  auto: boolean;
  /** The pipeline input that the data of the node derives from. */
  clock_domain: string;
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

/** Flags of {@link Packet.flags}: OR them together. */
export const PacketFlags: {
  readonly CrcOk: 1;
  readonly CrcFail: 2;
  readonly Corrected: 4;
  readonly Truncated: 8;
};

/**
 * A packet: a byte message with a sample index and flags, as in the JSON Lines packet files of
 * `sdrk-io`.
 */
export interface Packet {
  /** Sample index where the packet starts (decoders); a transmitting encoder ignores it. */
  index: bigint;
  /** OR of {@link PacketFlags}. */
  flags: number;
  /** The payload as lower-case hex without spaces, at most 256 bytes (512 digits). */
  data: string;
}

/** A packet to push: `index` (a `bigint` or a safe integer) and `flags` default to 0. */
export interface PacketIn {
  index?: bigint | number;
  flags?: number;
  data: string;
}

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

/**
 * The supplier of a pull call: at most `max` samples for `input`, or null when exhausted. For
 * a packet input: at most `max` packets; null or an empty array means "nothing to send now"
 * (the encoder sends its idle signal), never the end.
 */
export type PullSource = (
  input: string,
  max: number,
) => Samples | PacketIn[] | null | undefined;

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
  /**
   * Pushes packets into a packet input (`type: "packets"`); all are taken, or the call throws
   * before any is taken.
   */
  pushPackets(input: string, packets: PacketIn[]): void;
  /** Reads every packet waiting in a packet output, oldest first. */
  readPackets(output: string): Packet[];
  /** Reads samples of a stream output (complex ones interleaved); returns the sample count. */
  readStream(output: string, out: Samples): number;
  /** Reads the oldest frame of a frames output; `null` if there is none. */
  readFrame(output: string, frame: Float32Array): FrameMeta | null;
  /**
   * Samples (bytes for a raw output, frames for a frames output, packets for a packet output)
   * waiting in an output.
   */
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

/**
 * The character set of an RDS text: "rds" (the RDS table of IEC 62106 Annex E, the default),
 * "utf8", or "auto" (UTF-8 when the bytes are valid UTF-8 with a multi-byte sequence, which some
 * stations send against the standard; else the RDS table).
 */
export type RdsCharset = "rds" | "utf8" | "auto";
/** RDS bytes as text. Throws {@link SdrkError} for an unknown charset. */
export function rdsDecodeText(bytes: Uint8Array, charset?: RdsCharset): string;
/** Text as RDS bytes. Throws {@link SdrkError} for a character without an RDS code. */
export function rdsEncodeText(text: string): Uint8Array;

/**
 * Station data from the RDS groups of an `rds_decode` output (`sdrk_proto::rds::RdsState`): PI,
 * PTY, TP, and the programme service name and radiotext once complete.
 */
export class RdsDecoder {
  constructor();
  /** Takes packets as `readPackets` returns them; others than RDS groups are ignored. */
  push(packets: PacketIn[]): number;
  /** Forgets the station and the counts (after tuning to another station). */
  reset(): void;
  pi(): number | undefined;
  pty(): number | undefined;
  tp(): boolean | undefined;
  /** The programme service name (8 characters) once all segments arrived. */
  ps(charset?: RdsCharset): string | undefined;
  /** The radiotext once complete (up to the carriage return). */
  rt(charset?: RdsCharset): string | undefined;
  /** The name as far as it arrived, spaces for missing segments (a display that fills). */
  psPartial(charset?: RdsCharset): string | undefined;
  /** The radiotext as far as it arrived, spaces for missing segments. */
  rtPartial(charset?: RdsCharset): string | undefined;
  /** Groups taken, and groups taken with CRC_OK. */
  groups(): number;
  okGroups(): number;
  /** Releases the memory of the decoder. */
  free(): void;
}
