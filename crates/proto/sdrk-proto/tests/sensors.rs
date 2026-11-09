//! `sdrk_proto::sensors` framework (M46 §2): the reading, its JSON form, the sink, the
//! result ranking and the registry.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use core::fmt::Write;

use sdrk_proto::pulses::{Header, Modulation};
use sdrk_proto::sensors::{self, DecodeResult, JsonBuf, MAX_READINGS, Mic, Reading, ReadingSink};

fn header() -> Header {
    Header {
        modulation: Modulation::Ook,
        channel: 2,
        freq_offset_hz: 123_456,
        level_db10: -37,
        noise_db10: -500,
        rate_hz: 250_000,
    }
}

fn json(r: &Reading) -> String {
    let mut s = String::new();
    r.to_json(&mut s).unwrap();
    s
}

#[test]
fn the_json_has_a_fixed_field_order_and_only_present_fields() {
    let mut r = Reading::new("demo", 4711, Mic::Checksum, &header());
    assert_eq!(
        json(&r),
        r#"{"model":"demo","id":4711,"mic":"checksum","freq_MHz":0.1235,"rssi_dB":-3.7,"repeats":1}"#
    );
    r.time_s = Some(12.3456);
    r.channel = Some(2);
    r.battery_ok = Some(false);
    r.temperature_c = Some(-0.4);
    r.humidity = Some(56.0);
    r.wind_avg_km_h = Some(3.6);
    r.wind_max_km_h = Some(10.8);
    r.wind_dir_deg = Some(337.5);
    r.rain_mm = Some(0.3);
    r.pressure_hpa = Some(1013.2);
    r.center_hz = 433_920_000;
    r.repeats = 3;
    assert_eq!(
        json(&r),
        concat!(
            r#"{"time":12.346,"model":"demo","id":4711,"channel":2,"battery_ok":false,"#,
            r#""temperature_C":-0.4,"humidity":56,"wind_avg_km_h":3.6,"wind_max_km_h":10.8,"#,
            r#""wind_dir_deg":337.5,"rain_mm":0.3,"pressure_hPa":1013.2,"mic":"checksum","#,
            r#""freq_MHz":434.0435,"rssi_dB":-3.7,"repeats":3}"#
        )
    );
    // negative zero is written as zero
    r.temperature_c = Some(-0.0);
    assert!(json(&r).contains(r#""temperature_C":0.0,"#));
}

#[test]
fn non_finite_values_and_small_buffers_are_errors_not_panics() {
    let mut r = Reading::new("demo", 1, Mic::Crc, &header());
    r.temperature_c = Some(f32::NAN);
    let mut s = String::new();
    assert!(r.to_json(&mut s).is_err());
    r.temperature_c = None;
    r.time_s = Some(f64::INFINITY);
    assert!(r.to_json(&mut s).is_err());

    let r = Reading::new("demo", 1, Mic::Crc, &header());
    let mut b = JsonBuf::new();
    r.to_json(&mut b).unwrap();
    assert_eq!(b.as_bytes(), json(&r).as_bytes());
    b.clear();
    assert_eq!(b.as_bytes().len(), 0);
    let long = "x".repeat(JsonBuf::CAPACITY);
    assert!(b.write_str(&long).is_ok());
    assert!(b.write_str("y").is_err());
    assert_eq!(b.as_bytes().len(), JsonBuf::CAPACITY);
}

#[test]
fn the_model_is_escaped() {
    let r = Reading::new("a\"b\\c\n", 1, Mic::Crc, &header());
    assert!(json(&r).starts_with(r#"{"model":"a\"b\\c\u000a","id":1"#));
}

#[test]
fn the_content_comparison_ignores_radio_fields_time_and_repeats() {
    let mut a = Reading::new("demo", 1, Mic::Crc, &header());
    a.temperature_c = Some(20.0);
    let mut b = a;
    b.rssi_db = -50.0;
    b.freq_offset_hz = 7;
    b.time_s = Some(1.0);
    b.repeats = 2;
    assert!(a.same_content(&b));
    b.temperature_c = Some(20.1);
    assert!(!a.same_content(&b));
    b = a;
    b.id = 2;
    assert!(!a.same_content(&b));
}

#[test]
fn the_sink_holds_at_most_max_readings() {
    let mut sink = ReadingSink::new();
    assert!(sink.is_empty());
    for i in 0..MAX_READINGS {
        assert!(sink.push(Reading::new("demo", i as u32, Mic::Repeat, &header())));
    }
    assert!(!sink.push(Reading::new("demo", 99, Mic::Repeat, &header())));
    assert_eq!(sink.len(), MAX_READINGS);
    assert_eq!(sink.get(3).unwrap().id, 3);
    assert!(sink.get(MAX_READINGS).is_none());
    assert_eq!(sink.iter().map(|r| r.id).collect::<Vec<_>>(), [0, 1, 2, 3]);
    sink.clear();
    assert!(sink.iter().next().is_none());
}

#[test]
fn results_are_ranked_and_decoded_counts_add_up() {
    use DecodeResult::{BadIntegrity, Decoded, NoMatch, OutOfRange, Truncated};
    assert_eq!(NoMatch.best(Truncated), Truncated);
    assert_eq!(Truncated.best(OutOfRange), OutOfRange);
    assert_eq!(OutOfRange.best(BadIntegrity), BadIntegrity);
    assert_eq!(BadIntegrity.best(Decoded(1)), Decoded(1));
    assert_eq!(Decoded(1).best(BadIntegrity), Decoded(1));
    assert_eq!(Decoded(1).best(Decoded(2)), Decoded(3));
    assert_eq!(Decoded(2).count(), 2);
    assert_eq!(BadIntegrity.count(), 0);
}

#[test]
fn the_registry_is_consistent() {
    for d in sensors::DECODERS {
        assert_eq!(sensors::by_name(d.name()).unwrap().name(), d.name());
        assert_eq!(d.name(), d.name().to_lowercase());
    }
    assert!(sensors::by_name("no such decoder").is_none());
}

fn assert_send<T: Send>() {}

#[test]
fn public_types_are_send() {
    assert_send::<Reading>();
    assert_send::<ReadingSink>();
    assert_send::<JsonBuf>();
    assert_send::<DecodeResult>();
}
