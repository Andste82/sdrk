// The JavaScript error class of the bindings. `SdrkError` is what every failing call throws.

/**
 * An error of the SDRk bindings.
 *
 * `kind` is the name of the Rust `ErrorKind` (for example `"InvalidArgument"`). A failed build
 * also has `code` (the diagnostic code of the first problem), `path` (its location) and
 * `diagnostics` (all problems).
 */
export class SdrkError extends Error {
  constructor(message, kind, code, path, diagnostics) {
    super(message);
    this.name = "SdrkError";
    this.kind = kind;
    this.code = code;
    this.path = path;
    this.diagnostics = diagnostics;
  }
}

export function makeError(message, kind, code, path, diagnostics) {
  return new SdrkError(message, kind, code, path, diagnostics);
}

export function errorClass() {
  return SdrkError;
}
