// FlowBuilder: builds a pipeline description in code. The result is the same model as the JSON
// files (`toJson()` writes the canonical form: keys in the order of the documentation, maps
// sorted by key).

const TOP_ORDER = [
  "sdrk",
  "name",
  "description",
  "settings",
  "inputs",
  "groups",
  "nodes",
  "edges",
  "outputs",
];
const SETTINGS_ORDER = ["precision", "max_chunk", "event_queue", "pending_params"];
const INPUT_ORDER = ["format", "type", "rate_hz", "description"];
const NODE_ORDER = ["kind", "params", "group", "backend", "description"];
const OUTPUT_ORDER = [
  "type",
  "format",
  "capacity",
  "queue",
  "drop",
  "underrun",
  "description",
];
const EDGE_ORDER = ["from", "to"];

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Copies `object` with the keys in `order` first (when present), the others sorted after. */
function ordered(object, order) {
  const out = {};
  for (const key of order) {
    if (object[key] !== undefined) out[key] = object[key];
  }
  for (const key of Object.keys(object).sort()) {
    if (!(key in out) && object[key] !== undefined) out[key] = object[key];
  }
  return out;
}

/** Copies a map with its keys sorted and every value passed through `fn`. */
function sortedMap(map, fn) {
  const out = {};
  for (const key of Object.keys(map).sort()) out[key] = fn(map[key]);
  return out;
}

function clone(value) {
  if (Array.isArray(value)) return value.map(clone);
  if (isPlainObject(value)) {
    const out = {};
    for (const key of Object.keys(value)) out[key] = clone(value[key]);
    return out;
  }
  return value;
}

export class FlowBuilder {
  #name;
  #description;
  #settings;
  #inputs = {};
  #groups = {};
  #nodes = {};
  #edges = [];
  #outputs = {};

  /**
   * @param {string} name name of the pipeline (free text)
   * @param {{ description?: string }} [options]
   */
  constructor(name, options = {}) {
    if (typeof name !== "string") throw new TypeError("the name of a flow must be a string");
    this.#name = name;
    this.#description = options.description;
  }

  /** Sets the description of the whole flow. */
  description(text) {
    this.#description = text;
    return this;
  }

  /** Merges `settings` (`precision`, `max_chunk`, `event_queue`, `pending_params`). */
  settings(settings) {
    this.#settings = { ...(this.#settings ?? {}), ...settings };
    return this;
  }

  /** Adds an input: `{ format | type, rate_hz, description? }`. */
  input(name, spec) {
    this.#inputs[name] = clone(spec);
    return this;
  }

  /**
   * Adds a processing node.
   * @param {string} name
   * @param {string} kind block kind, for example "ddc"
   * @param {Record<string, number | boolean | string | object>} [params]
   * @param {{ group?: string, backend?: object, description?: string }} [options]
   */
  node(name, kind, params = {}, options = {}) {
    const node = { kind };
    if (params && Object.keys(params).length > 0) node.params = clone(params);
    if (options.group !== undefined) node.group = options.group;
    if (options.backend !== undefined) node.backend = clone(options.backend);
    if (options.description !== undefined) node.description = options.description;
    this.#nodes[name] = node;
    return this;
  }

  /** Declares a node group: `{ description? }`. */
  group(name, spec = {}) {
    this.#groups[name] = clone(spec);
    return this;
  }

  /** Connects `from` (an input or a node) to `to` (a node or an output). */
  edge(from, to) {
    this.#edges.push({ from, to });
    return this;
  }

  /** Adds an output: `{ type: "stream" | "frames", format?, capacity?, queue?, drop?, ... }`. */
  output(name, spec) {
    this.#outputs[name] = clone(spec);
    return this;
  }

  /** The description as a plain object in canonical key order. */
  toObject() {
    const flow = { sdrk: 1, name: this.#name };
    if (this.#description !== undefined) flow.description = this.#description;
    if (this.#settings !== undefined) flow.settings = ordered(this.#settings, SETTINGS_ORDER);
    flow.inputs = sortedMap(this.#inputs, (spec) => ordered(spec, INPUT_ORDER));
    if (Object.keys(this.#groups).length > 0) {
      flow.groups = sortedMap(this.#groups, (spec) => ordered(spec, ["description"]));
    }
    flow.nodes = sortedMap(this.#nodes, (spec) => {
      const node = ordered(spec, NODE_ORDER);
      if (node.params) node.params = sortedMap(node.params, (value) => value);
      return node;
    });
    flow.edges = this.#edges.map((edge) => ordered(edge, EDGE_ORDER));
    flow.outputs = sortedMap(this.#outputs, (spec) => ordered(spec, OUTPUT_ORDER));
    return ordered(flow, TOP_ORDER);
  }

  /** The description as canonical JSON text (two-space indent, trailing newline). */
  toJson() {
    return `${JSON.stringify(this.toObject(), null, 2)}\n`;
  }
}
