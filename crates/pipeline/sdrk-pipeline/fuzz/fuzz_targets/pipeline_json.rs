//! Fuzz target: `from_json`, `validate_json` and, if the description is valid, `build` and a
//! push of a few bytes. None of them may panic or hang.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sdrk_pipeline::{BuildOptions, InputData, PipelineSpec, Registry};

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let registry = Registry::standard();
    // Keep a build of a huge but valid description from exhausting the fuzzer's memory.
    let mut opts = BuildOptions::default();
    opts.limits.max_memory_bytes = 64 * 1024 * 1024;
    let _ = PipelineSpec::from_json(text);
    let report = registry.validate_json(text, &opts);
    let _ = report.to_json();
    if report.is_ok() {
        if let Ok(mut pipeline) = registry.build_json(text, &opts) {
            let names: Vec<String> = pipeline.config().inputs.keys().cloned().collect();
            for name in names {
                let _ = pipeline.push(&name, InputData::Raw(&[0, 1, 2, 3, 4, 5, 6, 7]));
                let _ = pipeline.push(&name, InputData::C32(&[sdrk::C32::new(0.5, -0.5); 8]));
            }
        }
    }
});
