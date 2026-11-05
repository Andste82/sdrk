/** A raw sample format name ("cu8", "cs16", ...) or a custom format object. */
export type FormatSpec = string | Record<string, unknown>;

/** Settings of a flow. */
export interface FlowSettings {
  precision?: "f32" | "q15";
  max_chunk?: number;
  event_queue?: number;
  pending_params?: number;
}

/**
 * An input: raw bytes in a `format`, converted samples of a `type`, or packets
 * (`type: "packets"`).
 */
export interface InputSpec {
  format?: FormatSpec;
  type?: "c32" | "cq15" | "f32" | "q15" | "packets";
  /**
   * Samples per second: an integer, or an exact fraction written as "num/den". Required for
   * samples; for a packet input the rate of the stream that a pure encoder produces from it
   * (required when such an encoder reads the input, else optional).
   */
  rate_hz?: number | string;
  description?: string;
}

/** An output of a flow. */
export interface OutputSpec {
  type: "stream" | "frames" | "packets";
  format?: FormatSpec;
  capacity?: number;
  queue?: number;
  drop?: "error" | "drop_oldest" | "drop_newest";
  underrun?: "zeros" | "short";
  description?: string;
}

/** A parameter value. */
export type ParamValue = number | boolean | string | Record<string, unknown>;

/** Optional attributes of a node. */
export interface NodeOptions {
  group?: string;
  backend?: { prefer?: string; require?: string };
  /** Alignment of the inputs of a node with two inputs: "auto" (default) or "none". */
  align?: "auto" | "none";
  description?: string;
}

/** A pipeline description in the model of the JSON files. */
export interface FlowObject {
  sdrk: 1;
  name: string;
  description?: string;
  settings?: FlowSettings;
  inputs: Record<string, InputSpec>;
  groups?: Record<string, { description?: string }>;
  nodes: Record<string, { kind: string; params?: Record<string, ParamValue>; group?: string; backend?: object; align?: "auto" | "none"; description?: string }>;
  edges: { from: string; to: string; port?: "a" | "b" }[];
  outputs: Record<string, OutputSpec>;
}

/** Builds a pipeline description in code; `toJson()` gives the same text as the JSON files. */
export class FlowBuilder {
  constructor(name: string, options?: { description?: string });
  description(text: string): this;
  settings(settings: FlowSettings): this;
  input(name: string, spec: InputSpec): this;
  node(name: string, kind: string, params?: Record<string, ParamValue>, options?: NodeOptions): this;
  group(name: string, spec?: { description?: string }): this;
  /** `port` is the input port of a node with two inputs: "a" (the default) or "b". */
  edge(from: string, to: string, port?: "a" | "b"): this;
  output(name: string, spec: OutputSpec): this;
  /** The description as a plain object in canonical key order. */
  toObject(): FlowObject;
  /** The description as canonical JSON text. */
  toJson(): string;
}
