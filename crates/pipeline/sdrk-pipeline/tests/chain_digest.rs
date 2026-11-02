//! The pipeline run behind the digest of the suite case `chain_fm_receiver` (M27): the
//! `fm_receiver` flow in `Q15` precision over 0.05 s of the FM test signal must give the audio
//! whose digest the case on the target compares with.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

mod common;

use sdrk::Q15;
use sdrk_pipeline::InputData;
use sdrk_testkit::{
    CHAIN_FM_RECEIVER_AUDIO, CHAIN_FM_RECEIVER_DIGEST, CHAIN_FM_RECEIVER_SAMPLES, DIGEST_START,
    digest_q15,
};
use serde_json::{Value, json};

use common::{build, cu8_recording, flow};

/// The `fm_receiver` flow in `Q15` precision without the spectrum branch.
fn q15_flow() -> String {
    let mut doc: Value = serde_json::from_str(&flow("fm_receiver")).unwrap();
    doc["settings"]["precision"] = json!("q15");
    doc["settings"]["max_chunk"] = json!(256);
    doc["nodes"].as_object_mut().unwrap().remove("fft");
    doc["outputs"].as_object_mut().unwrap().remove("spectrum");
    doc["edges"] = json!([
        { "from": "rf", "to": "tune" },
        { "from": "tune", "to": "demod" },
        { "from": "demod", "to": "audio" }
    ]);
    doc.to_string()
}

fn digest_of(piece: usize) -> (usize, u64) {
    let bytes = cu8_recording(CHAIN_FM_RECEIVER_SAMPLES);
    let mut pipeline = build(&q15_flow());
    let mut buf = vec![Q15::default(); 4096];
    let (mut count, mut hash) = (0, DIGEST_START);
    for part in bytes.chunks(piece) {
        pipeline.push("rf", InputData::Raw(part)).unwrap();
        loop {
            let got = pipeline.read_stream("audio", &mut buf).unwrap();
            if got == 0 {
                break;
            }
            hash = digest_q15(hash, &buf[..got]);
            count += got;
        }
    }
    (count, hash)
}

#[test]
fn the_pipeline_audio_has_the_digest_of_the_suite_case() {
    let (count, hash) = digest_of(512);
    assert_eq!(count, CHAIN_FM_RECEIVER_AUDIO);
    assert_eq!(
        hash, CHAIN_FM_RECEIVER_DIGEST,
        "digest {hash:#018x}; update CHAIN_FM_RECEIVER_DIGEST in the testkit"
    );
}

#[test]
fn the_digest_does_not_depend_on_the_push_size() {
    assert_eq!(digest_of(200), digest_of(512));
}
