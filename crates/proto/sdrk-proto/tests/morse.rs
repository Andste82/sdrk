//! Morse code of `sdrk-proto` (M47 Tests 1, 2, 4, 6 and 7): the table against an independent
//! copy of ITU-R M.1677-1, the exact units of the encoder (PARIS is 50 units, Farnsworth), and
//! the decoder on keyed samples at 5 to 40 words per minute, with speed drift, sloppy keying,
//! a change of speed after an idle gap and the cases it must count (unknown, overflow).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use sdrk_proto::Error;
use sdrk_proto::morse::{self, Decoder, Encoder, Push, SK};

/// The table of ITU-R M.1677-1 as `character code` pairs, written separately from the module.
const ITU: &str = "A .- B -... C -.-. D -.. E . F ..-. G --. H .... I .. J .--- K -.- L .-.. \
    M -- N -. O --- P .--. Q --.- R .-. S ... T - U ..- V ...- W .-- X -..- Y -.-- Z --.. \
    0 ----- 1 .---- 2 ..--- 3 ...-- 4 ....- 5 ..... 6 -.... 7 --... 8 ---.. 9 ----. \
    . .-.-.- , --..-- : ---... ? ..--.. ' .----. - -....- / -..-. ( -.--. ) -.--.- \
    \" .-..-. = -...- + .-.-. @ .--.-.";

/// The units of `text` as key states (`true` is key down), through the encoder.
fn units_of(enc: &mut Encoder, text: &[u8]) -> Vec<bool> {
    let mut out = Vec::new();
    for &c in text {
        let _ = enc.push(c).unwrap();
        while let Some(k) = enc.next_unit() {
            out.push(k);
        }
    }
    out
}

/// `#` for key down and `.` for key up.
fn pattern(units: &[bool]) -> String {
    units.iter().map(|&k| if k { '#' } else { '.' }).collect()
}

/// A small deterministic generator (xorshift64*), so that the tests need no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `[-a, a)`.
    fn spread(&mut self, a: f64) -> f64 {
        (2.0 * self.unit() - 1.0) * a
    }
}

/// Key states per sample: every unit lasts `unit(k)` samples (`k` counts the units; the
/// fractional lengths are carried so that the total stays exact).
fn expand(units: &[bool], mut unit: impl FnMut(usize) -> f64) -> Vec<bool> {
    let mut out = Vec::new();
    let mut t = 0.0f64;
    for (k, &key) in units.iter().enumerate() {
        t += unit(k);
        let end = (t + 0.5) as usize;
        while out.len() < end {
            out.push(key);
        }
    }
    out
}

/// Feeds `samples` and a long silence to a decoder and returns the text.
fn decode(dec: &mut Decoder, samples: &[bool]) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, &on) in samples.iter().enumerate() {
        if let Some(e) = dec.push(on, i as u64) {
            out.push(e.byte);
        }
    }
    out
}

/// The text sent after a lead-in of silence, decoded with the hint `hint`.
fn round_trip(text: &[u8], wpm: f32, spu: f64, hint: f32) -> Vec<u8> {
    let mut enc = Encoder::new(wpm, wpm).unwrap();
    let mut units = vec![false; 4];
    units.extend(units_of(&mut enc, text));
    units.extend(std::iter::repeat_n(false, 12));
    let samples = expand(&units, |_| spu);
    let mut dec = Decoder::new(hint).unwrap();
    decode(&mut dec, &samples)
}

#[test]
fn the_table_is_that_of_the_standard() {
    let mut it = ITU.split_whitespace();
    let mut n = 0;
    let mut codes = Vec::new();
    while let (Some(c), Some(code)) = (it.next(), it.next()) {
        let b = c.as_bytes()[0];
        assert_eq!(morse::code(b), Some(code), "code of {c}");
        let dashes: Vec<bool> = code.bytes().map(|e| e == b'-').collect();
        assert_eq!(morse::character(&dashes), Some(b), "character of {code}");
        codes.push(code.to_string());
        n += 1;
    }
    assert_eq!(n, 49);
    // The prosign SK has its own byte; every code is used once.
    assert_eq!(morse::code(SK), Some("...-.-"));
    codes.push("...-.-".to_string());
    codes.sort();
    codes.dedup();
    assert_eq!(codes.len(), 50);
    // Lower case folds to upper case; bytes without a code have none.
    assert_eq!(morse::code(b'q'), morse::code(b'Q'));
    assert_eq!(morse::code(b'#'), None);
    assert_eq!(morse::code(0), None);
    assert_eq!(morse::character(&[]), None);
    assert_eq!(morse::character(&[false; 7]), None);
}

#[test]
fn paris_is_fifty_units() {
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    let u = units_of(&mut enc, b"PARIS E");
    // 43 units for PARIS, 7 for the word gap, then the dot of E.
    assert_eq!(u.len(), 43 + 7 + 1);
    assert!(u[50]);
    assert!(u[43..50].iter().all(|&k| !k));
    assert_eq!(
        pattern(&u[..43]),
        "#.###.###.#...#.###...#.###.#...#.#...#.#.#"
    );
    assert_eq!(morse::units(b"PARIS E", 3, 7), 51);
}

#[test]
fn the_encoder_sends_exact_units() {
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    assert_eq!(enc.char_gap(), 3);
    assert_eq!(enc.word_gap(), 7);
    // SOS: ... --- ...
    let u = units_of(&mut enc, b"SOS");
    assert_eq!(pattern(&u), "#.#.#...###.###.###...#.#.#");
    // The same bytes at a new start, in lower case, give the same units.
    enc.reset();
    assert_eq!(units_of(&mut enc, b"sos"), u);
    // A space is a word gap; several spaces and leading spaces do not add anything.
    enc.reset();
    let u = units_of(&mut enc, b"  E   \t\n E");
    assert_eq!(pattern(&u), "#.......#");
    // The prosigns: AR is +, BT is =, KN is (, SK is its own byte.
    enc.reset();
    assert_eq!(pattern(&units_of(&mut enc, b"+")), "#.###.#.###.#");
    enc.reset();
    assert_eq!(pattern(&units_of(&mut enc, &[SK])), "#.#.#.###.#.###");
    // A byte without a code is skipped and takes no gap.
    enc.reset();
    assert_eq!(enc.push(b'#').unwrap(), Push::Skipped);
    assert_eq!(enc.push(b'E').unwrap(), Push::Queued);
    assert_eq!(enc.next_unit(), Some(true));
    assert_eq!(enc.next_unit(), None);
    assert_eq!(enc.push(b'#').unwrap(), Push::Skipped);
    assert_eq!(enc.push(b'E').unwrap(), Push::Queued);
    let rest: Vec<bool> = std::iter::from_fn(|| enc.next_unit()).collect();
    assert_eq!(pattern(&rest), "...#");
    // A packet boundary is a word gap.
    enc.reset();
    let mut u = units_of(&mut enc, b"E");
    enc.word_break();
    u.extend(units_of(&mut enc, b"E"));
    assert_eq!(pattern(&u), "#.......#");
}

#[test]
fn the_encoder_is_busy_until_drained() {
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    assert_eq!(enc.push(b'A').unwrap(), Push::Queued);
    assert!(!enc.is_idle());
    assert!(matches!(enc.push(b'B'), Err(Error::InvalidArgument(_))));
    while enc.next_unit().is_some() {}
    assert!(enc.is_idle());
    assert!(enc.push(b'B').is_ok());
    for bad in [
        (0.5, 0.5),
        (20.0, 10.0),
        (20.0, 101.0),
        (f32::NAN, 20.0),
        (20.0, f32::NAN),
    ] {
        assert!(Encoder::new(bad.0, bad.1).is_err(), "{bad:?}");
    }
}

#[test]
fn farnsworth_stretches_only_the_gaps() {
    // 18 words per minute for the characters, 10 for the text.
    let mut enc = Encoder::new(10.0, 18.0).unwrap();
    assert_eq!((enc.char_gap(), enc.word_gap()), (9, 22));
    let u = units_of(&mut enc, b"EE");
    assert_eq!(pattern(&u), "#.........#");
    // PARIS and its word gap take 31 + 4 * 9 + 22 = 89 units of 1/18 of 1.2 s: 10.1 words per
    // minute.
    let wpm: f64 = 60.0 / (89.0 * 1.2 / 18.0);
    assert!((wpm - 10.0).abs() < 0.2, "{wpm}");
    // Equal speeds give the plain gaps.
    let enc = Encoder::new(25.0, 25.0).unwrap();
    assert_eq!((enc.char_gap(), enc.word_gap()), (3, 7));
}

#[test]
fn a_clean_round_trip_at_every_speed() {
    let text: &[u8] = b"VVV CQ CQ DE DL1ABC K 0123456789 .,:?'-/()\"=+@ \x04 TEST";
    let want = text.to_vec();
    for spu in [6.0f64, 7.0, 12.0, 24.0, 60.0, 240.0] {
        let got = round_trip(text, 20.0, spu, spu as f32);
        // The decoder returns a space after the last word too.
        assert_eq!(
            String::from_utf8(got).unwrap(),
            format!("{} ", String::from_utf8(want.clone()).unwrap()),
            "spu {spu}"
        );
    }
}

#[test]
fn the_decoder_finds_the_speed_from_a_preamble() {
    // The hint is 4 times too large and 4 times too small; the preamble settles the estimate.
    let text: &[u8] = b"VVV VVV CQ DE W1AW K";
    for (spu, hint) in [(12.0, 48.0), (48.0, 12.0), (30.0, 6.0)] {
        let got = round_trip(text, 20.0, spu, hint);
        // The characters after the first word are exact.
        let s = String::from_utf8(got).unwrap();
        assert!(
            s.ends_with("VVV CQ DE W1AW K "),
            "spu {spu} hint {hint}: {s:?}"
        );
    }
}

#[test]
fn the_decoder_follows_a_drift_of_twenty_percent() {
    let text: &[u8] = b"VVV THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890 K";
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    let mut units = vec![false; 4];
    units.extend(units_of(&mut enc, text));
    units.extend(std::iter::repeat_n(false, 12));
    let n = units.len() as f64;
    // The unit swings by -20 % to +20 % once over the text.
    let samples = expand(&units, |k| {
        let phase = std::f64::consts::TAU * k as f64 / n;
        12.0 * (1.0 + 0.2 * phase.sin())
    });
    let mut dec = Decoder::new(12.0).unwrap();
    let got = decode(&mut dec, &samples);
    assert_eq!(
        String::from_utf8(got).unwrap(),
        format!("{} ", String::from_utf8(text.to_vec()).unwrap())
    );
    assert_eq!((dec.unknown(), dec.overflow()), (0, 0));
}

#[test]
fn sloppy_keying_is_decoded() {
    // Dots, dashes and gaps with random lengths: key-down units +-15 %, key-up units +-25 %.
    let text: &[u8] = b"VVV CQ CQ DE DL1ABC PSE K THE QUICK BROWN FOX 73";
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    let mut units = vec![false; 4];
    units.extend(units_of(&mut enc, text));
    units.extend(std::iter::repeat_n(false, 12));
    let mut rng = Rng(0x5eed_0001);
    let samples = expand(&units, |k| {
        let a = if units[k] { 0.15 } else { 0.25 };
        24.0 * (1.0 + rng.spread(a))
    });
    let mut dec = Decoder::new(24.0).unwrap();
    let got = String::from_utf8(decode(&mut dec, &samples)).unwrap();
    assert_eq!(
        got,
        format!("{} ", String::from_utf8(text.to_vec()).unwrap())
    );
}

#[test]
fn a_new_speed_after_an_idle_gap_is_learned() {
    let first: &[u8] = b"VVV CQ DE W1AW K";
    let second: &[u8] = b"VVV QRZ DE DL1ABC K";
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    let mut units = vec![false; 4];
    units.extend(units_of(&mut enc, first));
    // More than 20 word gaps of silence, then the second sender, 30 % slower.
    units.extend(std::iter::repeat_n(false, 200));
    let mid = units.len();
    enc.reset();
    units.extend(units_of(&mut enc, second));
    units.extend(std::iter::repeat_n(false, 12));
    let samples = expand(&units, |k| if k < mid { 12.0 } else { 12.0 * 1.3 });
    let mut dec = Decoder::new(12.0).unwrap();
    let got = String::from_utf8(decode(&mut dec, &samples)).unwrap();
    assert_eq!(got, "VVV CQ DE W1AW K VVV QRZ DE DL1ABC K ");
}

#[test]
fn unknown_and_overlong_characters_are_counted() {
    let spu = 8.0;
    let mut dec = Decoder::new(8.0).unwrap();
    // Seven dots are no character: "?" and one unknown. Nine dots are discarded.
    let mut units = vec![false; 4];
    for dots in [7, 9] {
        for k in 0..dots {
            if k > 0 {
                units.push(false);
            }
            units.push(true);
        }
        units.extend([false, false, false, false]);
    }
    // A plain E afterwards shows that the decoder went on.
    units.push(true);
    units.extend(std::iter::repeat_n(false, 12));
    let got = decode(&mut dec, &expand(&units, |_| spu));
    assert_eq!(got, b"?E ");
    assert_eq!((dec.unknown(), dec.overflow()), (1, 1));
}

#[test]
fn flush_and_reset() {
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    let mut units = vec![false; 4];
    units.extend(units_of(&mut enc, b"AB"));
    let samples = expand(&units, |_| 8.0);
    let mut dec = Decoder::new(8.0).unwrap();
    let got = decode(&mut dec, &samples);
    // Nothing is complete yet: the gap after B is missing.
    assert_eq!(got, b"A");
    assert!(dec.pending());
    let (a, b) = dec.flush();
    assert_eq!(a.map(|e| e.byte), Some(b'B'));
    assert_eq!(b.map(|e| e.byte), Some(b' '));
    assert_eq!(dec.flush(), (None, None));
    dec.reset();
    assert_eq!((dec.unknown(), dec.overflow(), dec.uncertain()), (0, 0, 0));
    assert!(Decoder::new(0.5).is_err());
    assert!(Decoder::new(f32::NAN).is_err());
}

#[test]
fn characters_carry_the_index_of_their_first_sample() {
    let mut enc = Encoder::new(20.0, 20.0).unwrap();
    let mut units = vec![false; 4];
    units.extend(units_of(&mut enc, b"EE"));
    units.extend(std::iter::repeat_n(false, 12));
    let samples = expand(&units, |_| 10.0);
    let mut dec = Decoder::new(10.0).unwrap();
    let mut starts = Vec::new();
    for (i, &on) in samples.iter().enumerate() {
        if let Some(e) = dec.push(on, 1000 + i as u64)
            && e.byte != b' '
        {
            starts.push(e.start);
        }
    }
    // The first E starts after 4 units of silence, the second 4 units later.
    assert_eq!(starts, [1040, 1080]);
}
