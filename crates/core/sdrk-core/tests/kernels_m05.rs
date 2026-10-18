//! Tests of the `convert`, `rotate` and `fir` kernels of `sdrk-core` (M05).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::identity_op,
    clippy::useless_vec,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::items_after_statements,
    clippy::similar_names
)]

use sdrk_core::kernels::convert::{Convert, ConvertReal};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::timing::{Delay, RatioKind, Timing};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15, Rate, Sample};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat, SdrkError, formats};

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

// ---------------------------------------------------------------- convert

fn conv_c32(f: &SampleFormat, bytes: &[u8], n: usize) -> Vec<C32> {
    let mut out = vec![C32::ZERO; n];
    assert_eq!(Fallback.convert(f, bytes, &mut out).unwrap(), n);
    out
}

fn conv_cq15(f: &SampleFormat, bytes: &[u8], n: usize) -> Vec<CQ15> {
    let mut out = vec![CQ15::ZERO; n];
    assert_eq!(Fallback.convert(f, bytes, &mut out).unwrap(), n);
    out
}

fn q(re: i16, im: i16) -> CQ15 {
    CQ15 {
        re: Q15(re),
        im: Q15(im),
    }
}

#[test]
fn convert_spot_values() {
    // CU8: zero is at 127.5.
    assert_eq!(
        conv_c32(&formats::CU8, &[0, 255, 128, 127], 2),
        [
            C32::new(-127.5 / 128.0, 127.5 / 128.0),
            C32::new(0.5 / 128.0, -0.5 / 128.0)
        ]
    );
    assert_eq!(
        conv_cq15(&formats::CU8, &[0, 255, 128, 127], 2),
        [q(-32640, 32640), q(128, -128)]
    );
    // CS8
    assert_eq!(
        conv_c32(&formats::CS8, &[0x80, 0x7F], 1),
        [C32::new(-1.0, 127.0 / 128.0)]
    );
    assert_eq!(
        conv_cq15(&formats::CS8, &[0x80, 0x7F], 1),
        [q(-32768, 127 * 256)]
    );
    // CS16 and its swapped / big-endian / planar variants.
    let cs16 = [0x00, 0x40, 0x00, 0xC0];
    assert_eq!(conv_cq15(&formats::CS16, &cs16, 1), [q(16384, -16384)]);
    assert_eq!(
        conv_cq15(&formats::CS16.with_swap_iq(true), &cs16, 1),
        [q(-16384, 16384)]
    );
    let big = SampleFormat::new(
        Container::I16,
        16,
        true,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Big,
    )
    .unwrap();
    assert_eq!(
        conv_cq15(&big, &[0x40, 0x00, 0xC0, 0x00], 1),
        [q(16384, -16384)]
    );
    assert_eq!(
        conv_cq15(&formats::CS16_PLANAR, &[1, 0, 3, 0, 2, 0, 4, 0], 2),
        [q(1, 2), q(3, 4)]
    );
    // 12 bits in 16: sign-extended low bits; upper container bits are ignored.
    assert_eq!(
        conv_cq15(&formats::CS12_LSB16, &[0xFF, 0xF7, 0x00, 0x08], 1),
        [q(2047 * 16, -2048 * 16)]
    );
    assert_eq!(
        conv_c32(&formats::CS12_LSB16, &[0x00, 0x04, 0x00, 0xFC], 1),
        [C32::new(0.5, -0.5)]
    );
    // bladeRF: full scale 2048 means e * 16 in Q15.
    assert_eq!(
        conv_cq15(&formats::SC16_Q11, &[0x01, 0x00, 0xFF, 0xFF], 1),
        [q(16, -16)]
    );
    // Packed 12: v0 = b0 | (b1 & 0xF) << 8, v1 = b1 >> 4 | b2 << 4.
    assert_eq!(
        conv_cq15(&formats::PACKED12, &[0x01, 0xF0, 0xFF], 1),
        [q(16, -16)]
    );
    assert_eq!(
        conv_cq15(&formats::PACKED12, &[0x00, 0x08, 0x80], 1),
        [q(-2048 * 16, -2048 * 16)]
    );
    // Packed 10: four consecutive 10-bit values LSB first.
    let (v0, v1, v2, v3) = (1u64, 0x3FFu64, 0x200u64, 0x1FFu64);
    let packed = v0 | v1 << 10 | v2 << 20 | v3 << 30;
    let bytes: Vec<u8> = (0..5).map(|i| (packed >> (8 * i)) as u8).collect();
    assert_eq!(
        conv_cq15(&formats::PACKED10, &bytes, 2),
        [q(1 * 64, -64), q(-512 * 64, 511 * 64)]
    );
    // 24 bits MSB-aligned in 32.
    let msb = (-3i32 << 8).to_le_bytes();
    let msb2 = (5i32 << 8).to_le_bytes();
    let bytes = [msb, msb2].concat();
    assert_eq!(conv_cq15(&formats::I2S32_MSB, &bytes, 1), [q(0, 0)]); // -3 and 5 are below one Q15 LSB (shift 8)
    assert_eq!(
        conv_c32(&formats::I2S32_MSB, &bytes, 1),
        [C32::new(-3.0 / 8388608.0, 5.0 / 8388608.0)]
    );
    // F32 containers pass through, including NaN and infinity; Q15 uses the f64 rule.
    let vals = [f32::NAN, f32::INFINITY, -0.0, 0.5, f32::NEG_INFINITY, 2.0];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = conv_c32(&formats::CF32, &bytes, 3);
    assert!(out[0].re.is_nan());
    assert_eq!(
        (out[0].im, out[1].re.to_bits(), out[1].im),
        (f32::INFINITY, (-0.0f32).to_bits(), 0.5)
    );
    assert_eq!(
        conv_cq15(&formats::CF32, &bytes, 3),
        [q(0, 32767), q(0, 16384), q(-32768, 32767)]
    );
}

#[test]
fn convert_real_formats() {
    let mut f = [0f32; 2];
    let mut qv = [Q15(0); 2];
    let rs16 = [0x00, 0x40, 0x00, 0xC0];
    assert_eq!(
        Fallback
            .convert_real(&formats::RS16, &rs16, &mut f)
            .unwrap(),
        2
    );
    assert_eq!(f, [0.5, -0.5]);
    assert_eq!(
        Fallback
            .convert_real(&formats::RS16, &rs16, &mut qv)
            .unwrap(),
        2
    );
    assert_eq!(qv, [Q15(16384), Q15(-16384)]);
    let rf32: Vec<u8> = [1.5f32, -0.25]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        Fallback
            .convert_real(&formats::RF32, &rf32, &mut f)
            .unwrap(),
        2
    );
    assert_eq!(f, [1.5, -0.25]);
    assert_eq!(
        Fallback
            .convert_real(&formats::RF32, &rf32, &mut qv)
            .unwrap(),
        2
    );
    assert_eq!(qv, [Q15(32767), Q15(-8192)]);
}

#[test]
fn convert_explicit_full_scale_rounds_half_up() {
    // Full scale 3: e * 32768 / 3, round half up.
    let f = formats::CS16
        .with_full_scale(FullScale::Explicit(3))
        .unwrap();
    let bytes: Vec<u8> = [1i16, -1, 2, -2]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        conv_cq15(&f, &bytes, 2),
        [q(10923, -10923), q(21845, -21845)]
    );
    // A tie: e = 1, full scale 65536 gives exactly 0.5 -> 1 (half up); e = -1 gives -0.5 -> 0.
    let f = formats::CS16
        .with_full_scale(FullScale::Explicit(65536))
        .unwrap();
    let bytes: Vec<u8> = [1i16, -1].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(conv_cq15(&f, &bytes, 1), [q(1, 0)]);
    // U8 with an explicit full scale.
    let f = formats::CU8
        .with_full_scale(FullScale::Explicit(255))
        .unwrap();
    assert_eq!(conv_cq15(&f, &[255, 0], 1), [q(16384, -16384)]);
    let f = formats::CU8
        .with_full_scale(FullScale::Explicit(200))
        .unwrap();
    // (255 - 127.5) / 200 * 32768 = 20889.6 -> 20890; the negative value -20889.6 -> -20890.
    assert_eq!(conv_cq15(&f, &[255, 0], 1), [q(20890, -20890)]);
}

#[test]
fn convert_32_bit_containers_use_f64_beyond_24_bits() {
    let f = SampleFormat::new(
        Container::I32,
        32,
        true,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Little,
    )
    .unwrap();
    let bytes: Vec<u8> = [i32::MAX, i32::MIN]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = conv_c32(&f, &bytes, 1);
    // 2147483647 / 2^31 rounds to 1.0 in f32.
    assert_eq!(out, [C32::new(1.0, -1.0)]);
    assert_eq!(conv_cq15(&f, &bytes, 1), [q(32767, -32768)]);
}

#[test]
fn convert_error_cases() {
    let mut out = [C32::ZERO; 4];
    // Partial samples and partial packing groups.
    let e = Fallback
        .convert(&formats::CS16, &[0; 5], &mut out)
        .unwrap_err();
    assert_eq!(e, Error::InvalidParameter("partial sample"));
    assert!(matches!(
        Fallback.convert(&formats::PACKED12, &[0; 4], &mut out),
        Err(Error::InvalidParameter("partial sample"))
    ));
    assert!(matches!(
        Fallback.convert(&formats::PACKED10, &[0; 7], &mut out),
        Err(Error::InvalidParameter("partial sample"))
    ));
    assert!(matches!(
        Fallback.convert(&formats::CS16_PLANAR, &[0; 6], &mut out),
        Err(Error::InvalidParameter("partial sample"))
    ));
    // Output too small.
    let e = Fallback
        .convert(&formats::CS16, &[0; 12], &mut out[..2])
        .unwrap_err();
    assert_eq!(e, Error::BufferTooSmall { needed: 3, got: 2 });
    // Bits1 is not supported.
    let bits1 = SampleFormat::new(
        Container::Bits1,
        1,
        true,
        Layout::Interleaved,
        Align::Lsb,
        Endian::Little,
    )
    .unwrap();
    assert!(matches!(
        Fallback.convert(&bits1, &[0; 4], &mut out),
        Err(Error::Unsupported(_))
    ));
    // Wrong layout for the trait.
    assert!(matches!(
        Fallback.convert(&formats::RS16, &[0; 4], &mut out),
        Err(Error::InvalidParameter(_))
    ));
    let mut real = [0f32; 4];
    assert!(matches!(
        Fallback.convert_real(&formats::CS16, &[0; 4], &mut real),
        Err(Error::InvalidParameter(_))
    ));
    // Error classification.
    assert_eq!(e.kind(), sdrk_device::ErrorKind::BufferTooSmall);
    // Empty input is fine and a larger output only gets the decoded prefix.
    assert_eq!(Fallback.convert(&formats::CS16, &[], &mut out).unwrap(), 0);
    out = [C32::new(9.0, 9.0); 4];
    assert_eq!(
        Fallback
            .convert(&formats::CS16, &[0, 0, 0, 0], &mut out)
            .unwrap(),
        1
    );
    assert_eq!(out[1], C32::new(9.0, 9.0));
}

// -------------------------------------------------------------------- nco

#[test]
fn nco_increments() {
    let rate = Rate::hz(1_000_000).unwrap();
    assert_eq!(Nco::new(250_000.0, rate).unwrap().increment(), 1 << 30);
    assert_eq!(Nco::new(-250_000.0, rate).unwrap().increment(), 3 << 30);
    assert_eq!(Nco::new(0.0, rate).unwrap().increment(), 0);
    assert_eq!(Nco::new(500_000.0, rate).unwrap().increment(), 1 << 31);
    // Whole turns vanish; frequencies above the rate alias.
    assert_eq!(Nco::new(1_250_000.0, rate).unwrap().increment(), 1 << 30);
    assert_eq!(Nco::new(1_000_000.0, rate).unwrap().increment(), 0);
    // Rates that are not whole hertz.
    let half = Rate::new(1, 2).unwrap();
    assert_eq!(Nco::new(0.125, half).unwrap().increment(), 1 << 30);
    // Round to nearest: 0.1 * 2^32 = 429496729.6.
    assert_eq!(
        Nco::new(0.1, Rate::hz(1).unwrap()).unwrap().increment(),
        429496730
    );
    // Rounding up to 2^32 wraps to 0.
    assert_eq!(
        Nco::new(1.0 - 2f64.powi(-40), Rate::hz(1).unwrap())
            .unwrap()
            .increment(),
        0
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            Nco::new(bad, rate),
            Err(Error::InvalidParameter(_))
        ));
    }
    // A finite frequency over a tiny rate that overflows the quotient is rejected too.
    assert!(Nco::new(f64::MAX, Rate::new(1, u64::MAX).unwrap()).is_err());
}

#[test]
fn nco_state_handling() {
    let rate = Rate::hz(1000).unwrap();
    let mut nco = Nco::new(100.0, rate).unwrap();
    assert_eq!((nco.phase(), nco.increment()), (0, 429496730));
    nco.set_phase(0xDEAD_BEEF);
    nco.set_frequency(-250.0, rate).unwrap();
    assert_eq!((nco.phase(), nco.increment()), (0xDEAD_BEEF, 3 << 30));
    assert!(nco.set_frequency(f64::NAN, rate).is_err());
    assert_eq!(nco.increment(), 3 << 30, "a failed retune changes nothing");
    nco.reset();
    assert_eq!((nco.phase(), nco.increment()), (0, 3 << 30));
    assert_eq!(Nco::from_increment(7).increment(), 7);
    assert_eq!(Nco::default(), Nco::from_increment(0));
    assert_eq!(nco.ratio(), sdrk_device::Ratio::ONE);
    assert_eq!(nco.ratio_kind(), RatioKind::Constant);
    assert_eq!(nco.group_delay(), Delay::ZERO);
}

#[test]
fn phase_is_continuous_across_calls_and_retunes() {
    let rate = Rate::hz(1000).unwrap();
    let input = [C32::new(1.0, 0.0); 100];
    let mut one = [C32::ZERO; 100];
    let mut nco = Nco::new(123.0, rate).unwrap();
    Fallback.rotate(&mut nco, &input, &mut one).unwrap();
    assert_eq!(nco.phase(), nco.increment().wrapping_mul(100));

    // Two calls equal one call.
    let mut two = [C32::ZERO; 100];
    let mut nco2 = Nco::new(123.0, rate).unwrap();
    let (a, b) = two.split_at_mut(37);
    Fallback.rotate(&mut nco2, &input[..37], a).unwrap();
    Fallback.rotate(&mut nco2, &input[37..], b).unwrap();
    assert_eq!(one, two);
    assert_eq!(nco, nco2);

    // A retune in the middle continues from the current phase: the first sample after
    // the retune sits at the phase reached so far.
    let mut nco3 = Nco::new(123.0, rate).unwrap();
    let mut out = [C32::ZERO; 50];
    Fallback.rotate(&mut nco3, &input[..50], &mut out).unwrap();
    let phase = nco3.phase();
    nco3.set_frequency(-77.0, rate).unwrap();
    assert_eq!(nco3.phase(), phase);
    let mut next = [C32::ZERO; 1];
    Fallback.rotate(&mut nco3, &input[..1], &mut next).unwrap();
    let mut expect = [C32::ZERO; 1];
    let mut probe = Nco::from_increment(0);
    probe.set_phase(phase);
    Fallback
        .rotate(&mut probe, &input[..1], &mut expect)
        .unwrap();
    assert_eq!(next, expect);
}

#[test]
fn rotate_by_known_angles() {
    // Quarter turns are exact in CQ15 (cos = 0, sin = +-32767/32768 up to table rounding).
    let mut nco = Nco::from_increment(1 << 30);
    let input = [q(16384, 0); 4];
    let mut out = [CQ15::ZERO; 4];
    Fallback.rotate(&mut nco, &input, &mut out).unwrap();
    assert_eq!(out[0], q(16384, 0));
    assert_eq!(out[1], q(0, 16384));
    assert_eq!(out[2], q(-16384, 0));
    assert_eq!(out[3], q(0, -16384));
    assert_eq!(nco.phase(), 0, "four quarter turns wrap to zero");

    // C32: positive frequency moves the spectrum up, i.e. multiplies by e^{+j...}.
    let mut nco = Nco::from_increment(1 << 30);
    let input = [C32::new(1.0, 0.0); 4];
    let mut out = [C32::ZERO; 4];
    Fallback.rotate(&mut nco, &input, &mut out).unwrap();
    for (o, (re, im)) in out
        .iter()
        .zip([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)])
    {
        assert!(
            (o.re - re).abs() < 1e-6 && (o.im - im).abs() < 1e-6,
            "{o:?}"
        );
    }
}

#[test]
fn rotate_with_zero_frequency_is_the_identity() {
    let mut s = 5u64;
    let c32: Vec<C32> = (0..1000)
        .map(|_| C32::new(unit(&mut s), unit(&mut s)))
        .collect();
    let mut out = vec![C32::ZERO; c32.len()];
    let mut nco = Nco::from_increment(0);
    Fallback.rotate(&mut nco, &c32, &mut out).unwrap();
    assert_eq!(out, c32);
    assert_eq!(nco.phase(), 0);

    // CQ15: cos(0) is 32767/32768, so the identity is exact for |x| <= 16384 and
    // within one LSB below full scale.
    let small: Vec<CQ15> = (0..1000)
        .map(|_| {
            q(
                (lcg(&mut s) % 32769) as i16 - 16384,
                (lcg(&mut s) % 32769) as i16 - 16384,
            )
        })
        .collect();
    let mut out = vec![CQ15::ZERO; small.len()];
    Fallback
        .rotate(&mut Nco::from_increment(0), &small, &mut out)
        .unwrap();
    assert_eq!(out, small);
    let full: Vec<CQ15> = (0..1000)
        .map(|_| q(lcg(&mut s) as i16, lcg(&mut s) as i16))
        .collect();
    Fallback
        .rotate(&mut Nco::from_increment(0), &full, &mut out)
        .unwrap();
    for (o, i) in out.iter().zip(&full) {
        assert!((i32::from(o.re.0) - i32::from(i.re.0)).abs() <= 1);
        assert!((i32::from(o.im.0) - i32::from(i.im.0)).abs() <= 1);
    }
}

#[test]
fn rotate_argument_checks_and_empty_input() {
    let mut nco = Nco::from_increment(99);
    let mut out = [C32::ZERO; 2];
    let e = Fallback
        .rotate(&mut nco, &[C32::ZERO; 3], &mut out)
        .unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 3,
            got: 2
        }
    );
    assert_eq!(nco.phase(), 0, "a failed call leaves the oscillator alone");
    Rotate::<C32>::rotate(&Fallback, &mut nco, &[], &mut []).unwrap();
    Rotate::<CQ15>::rotate(&Fallback, &mut nco, &[], &mut []).unwrap();
    assert_eq!(nco.phase(), 0);
}

#[test]
fn rotate_handles_extreme_inputs() {
    // f32: NaN propagates, infinity never panics, silence stays silent.
    let nan = [
        C32::new(f32::NAN, 0.0),
        C32::new(0.0, f32::INFINITY),
        C32::new(f32::MAX, f32::MIN),
    ];
    let mut out = [C32::ZERO; 3];
    Fallback
        .rotate(&mut Nco::from_increment(12345), &nan, &mut out)
        .unwrap();
    assert!(out[0].re.is_nan() || out[0].im.is_nan());
    let mut out = [C32::new(1.0, 1.0); 3];
    Fallback
        .rotate(&mut Nco::from_increment(12345), &[C32::ZERO; 3], &mut out)
        .unwrap();
    assert_eq!(out, [C32::ZERO; 3]);
    // Q15: full scale rotated by 45 degrees saturates instead of wrapping.
    let full = [q(-32768, -32768); 3];
    let mut out = [CQ15::ZERO; 3];
    let mut nco = Nco::from_increment(1 << 29);
    nco.set_phase(1 << 29);
    Fallback.rotate(&mut nco, &full, &mut out).unwrap();
    // 45 degrees: (-1 - j) * (0.7071 + 0.7071 j) = -1.414 j, so the imaginary part saturates.
    assert_eq!(out[0], q(0, -32768));
    // 90 degrees: (-1 - j) * j = 1 - j; sin(90 degrees) is 32767/32768 in the table.
    assert_eq!(out[1], q(32767, -32767));
}

// -------------------------------------------------------------------- fir

#[test]
fn fir_state_validation() {
    // Wrong history lengths and empty taps.
    assert!(matches!(
        FirState::<C32, _, _>::new([1.0f32, 2.0, 3.0], [C32::ZERO; 3]),
        Err(Error::LengthMismatch {
            expected: 2,
            got: 3
        })
    ));
    assert!(matches!(
        FirState::<C32, _, _>::new([1.0f32, 2.0, 3.0], [C32::ZERO; 1]),
        Err(Error::LengthMismatch {
            expected: 2,
            got: 1
        })
    ));
    assert!(matches!(
        FirState::<CQ15, _, _>::new([] as [Q15; 0], [] as [CQ15; 0]),
        Err(Error::InvalidParameter(_))
    ));
    // One tap needs no history; borrowed slices and Vecs work as storage.
    let s = FirState::<C32, _, _>::new([0.5f32], [] as [C32; 0]).unwrap();
    assert_eq!(s.taps(), &[0.5]);
    let mut hist = vec![C32::new(9.0, 9.0); 2];
    let taps = [1.0f32, 0.0, 0.0];
    let s = FirState::<C32, _, _>::new(&taps[..], &mut hist[..]).unwrap();
    assert_eq!(s.history(), &[C32::ZERO; 2], "the history is zeroed");
}

#[test]
fn fir_timing() {
    let s = FirState::<C32, _, _>::new([0.0f32; 64], [C32::ZERO; 63]).unwrap();
    assert_eq!(s.ratio(), sdrk_device::Ratio::ONE);
    assert_eq!(s.ratio_kind(), RatioKind::Constant);
    assert_eq!(s.group_delay(), Delay::new(63, 2).unwrap());
    assert_eq!(s.group_delay().as_f64(), 31.5);
    let s = FirState::<CQ15, _, _>::new([Q15(1)], [] as [CQ15; 0]).unwrap();
    assert_eq!(s.group_delay().reduced(), Delay::ZERO);
}

#[test]
fn fir_impulse_response_equals_the_taps() {
    let h = [0.25f32, -0.5, 0.125, 0.75, -1.0, 0.5];
    let mut st = FirState::<C32, _, _>::new(h, [C32::ZERO; 5]).unwrap();
    let mut x = [C32::ZERO; 16];
    x[0] = C32::new(1.0, -2.0);
    let mut y = [C32::ZERO; 16];
    Fallback.fir(&mut st, &x, &mut y).unwrap();
    for (k, o) in y.iter().enumerate() {
        let t = h.get(k).copied().unwrap_or(0.0);
        assert_eq!(*o, C32::new(t, -2.0 * t), "tap {k}");
    }
    // CQ15: a unit impulse of 0.99997 reproduces taps for in-range values up to rounding;
    // for an exact copy use the impulse 1/2 with taps that are even.
    let hq = [8192i16, -16384, 4096, 24576, -32768, 16384].map(Q15);
    let mut st = FirState::<CQ15, _, _>::new(hq, [CQ15::ZERO; 5]).unwrap();
    let mut x = [CQ15::ZERO; 16];
    x[0] = q(32767, 0);
    let mut y = [CQ15::ZERO; 16];
    Fallback.fir(&mut st, &x, &mut y).unwrap();
    for (k, o) in y.iter().enumerate() {
        let t = i32::from(hq.get(k).map_or(0, |v| v.0));
        assert!((i32::from(o.re.0) - t).abs() <= 1, "tap {k}: {o:?}");
        assert_eq!(o.im, Q15(0));
    }
    // The taps survive the call and the history holds the last N - 1 inputs.
    assert_eq!(st.taps(), &hq);
}

#[test]
fn fir_history_and_reset() {
    let mut st = FirState::<C32, _, _>::new([1.0f32, 1.0, 1.0, 1.0], [C32::ZERO; 3]).unwrap();
    let input: Vec<C32> = (1..=5).map(|i| C32::new(i as f32, 0.0)).collect();
    let mut out = [C32::ZERO; 5];
    Fallback.fir(&mut st, &input, &mut out).unwrap();
    assert_eq!(out.map(|c| c.re), [1.0, 3.0, 6.0, 10.0, 14.0]);
    assert_eq!(
        st.history().iter().map(|c| c.re).collect::<Vec<_>>(),
        [3.0, 4.0, 5.0]
    );
    // A short block shifts the history.
    let mut one = [C32::ZERO; 1];
    Fallback
        .fir(&mut st, &[C32::new(6.0, 0.0)], &mut one)
        .unwrap();
    assert_eq!(one[0].re, 6.0 + 5.0 + 4.0 + 3.0);
    assert_eq!(
        st.history().iter().map(|c| c.re).collect::<Vec<_>>(),
        [4.0, 5.0, 6.0]
    );
    // Empty blocks change nothing.
    Fallback.fir(&mut st, &[], &mut []).unwrap();
    assert_eq!(
        st.history().iter().map(|c| c.re).collect::<Vec<_>>(),
        [4.0, 5.0, 6.0]
    );
    st.reset();
    assert_eq!(st.history(), &[C32::ZERO; 3]);
    Fallback
        .fir(&mut st, &[C32::new(1.0, 0.0)], &mut one)
        .unwrap();
    assert_eq!(one[0].re, 1.0);
}

#[test]
fn fir_rejects_mismatched_lengths() {
    let mut st = FirState::<CQ15, _, _>::new([Q15(100)], [] as [CQ15; 0]).unwrap();
    let mut out = [CQ15::ZERO; 1];
    let e = Fallback
        .fir(&mut st, &[CQ15::ZERO; 2], &mut out)
        .unwrap_err();
    assert_eq!(
        e,
        Error::LengthMismatch {
            expected: 2,
            got: 1
        }
    );
}

#[test]
fn fir_extremes() {
    // NaN propagates through the f32 filter and infinity does not panic.
    let mut st = FirState::<C32, _, _>::new([0.5f32, 0.5], [C32::ZERO; 1]).unwrap();
    let x = [
        C32::new(f32::NAN, 1.0),
        C32::new(f32::INFINITY, 1.0),
        C32::new(1.0, f32::NEG_INFINITY),
    ];
    let mut y = [C32::ZERO; 3];
    Fallback.fir(&mut st, &x, &mut y).unwrap();
    assert!(y[0].re.is_nan() && y[1].re.is_nan());
    // Q15: eight taps of 0.25 on DC at full scale saturate instead of wrapping.
    let mut st = FirState::<CQ15, _, _>::new([Q15(8192); 8], [CQ15::ZERO; 7]).unwrap();
    let x = [q(32767, -32768); 32];
    let mut y = [CQ15::ZERO; 32];
    Fallback.fir(&mut st, &x, &mut y).unwrap();
    assert_eq!(y[31], q(32767, -32768));
    assert_eq!(y[0], q(8192, -8192));
    // Silence in, silence out.
    let mut st = FirState::<CQ15, _, _>::new([Q15(-32768); 3], [CQ15::ZERO; 2]).unwrap();
    Fallback
        .fir(&mut st, &[CQ15::ZERO; 8], &mut y[..8])
        .unwrap();
    assert!(y[..8].iter().all(|s| *s == CQ15::ZERO));
}

// ------------------------------------------------------------------ misc

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<Nco>();
    assert_sync::<Nco>();
    assert_send::<FirState<C32, Vec<f32>, Vec<C32>>>();
    assert_sync::<FirState<CQ15, [Q15; 4], [CQ15; 3]>>();
    assert_send::<FirState<CQ15, &'static [Q15], &'static mut [CQ15]>>();
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..len).map(|_| (lcg(&mut s) >> 24) as u8).collect()
    }

    #[test]
    fn convert_agrees_bit_for_bit_on_random_bytes() {
        let mut all = vec![
            formats::CU8,
            formats::CS8,
            formats::CS16,
            formats::CS12_LSB16,
            formats::SC16_Q11,
            formats::CS16_PLANAR,
            formats::PACKED12,
            formats::PACKED10,
            formats::I2S32_MSB,
            formats::CF32,
            formats::RF32,
            formats::RS16,
        ];
        all.push(formats::CU8.with_swap_iq(true));
        all.push(
            formats::CU8
                .with_full_scale(FullScale::Explicit(200))
                .unwrap(),
        );
        all.push(
            formats::CS16
                .with_full_scale(FullScale::Explicit(30000))
                .unwrap()
                .with_swap_iq(true),
        );
        all.push(
            SampleFormat::new(
                Container::I16,
                16,
                true,
                Layout::Interleaved,
                Align::Lsb,
                Endian::Big,
            )
            .unwrap(),
        );
        all.push(
            SampleFormat::new(
                Container::I16,
                10,
                true,
                Layout::Planar,
                Align::Msb,
                Endian::Big,
            )
            .unwrap(),
        );
        all.push(
            SampleFormat::new(
                Container::I32,
                32,
                true,
                Layout::Interleaved,
                Align::Lsb,
                Endian::Big,
            )
            .unwrap(),
        );
        all.push(
            SampleFormat::new(
                Container::I32,
                20,
                true,
                Layout::Real,
                Align::Lsb,
                Endian::Little,
            )
            .unwrap(),
        );
        all.push(
            SampleFormat::new(
                Container::F32,
                32,
                true,
                Layout::Interleaved,
                Align::Lsb,
                Endian::Big,
            )
            .unwrap()
            .with_swap_iq(true),
        );
        all.push(
            SampleFormat::new(
                Container::I8,
                6,
                true,
                Layout::Interleaved,
                Align::Msb,
                Endian::Little,
            )
            .unwrap(),
        );
        for (i, f) in all.iter().enumerate() {
            let n = 64;
            let bytes = random_bytes(f.bytes_for_samples(n).unwrap(), i as u64 + 1);
            if f.layout() == Layout::Real {
                let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
                Fallback.convert_real(f, &bytes, &mut a).unwrap();
                Reference.convert_real(f, &bytes, &mut b).unwrap();
                assert!(
                    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "format {i}"
                );
                let (mut a, mut b) = (vec![Q15(0); n], vec![Q15(0); n]);
                Fallback.convert_real(f, &bytes, &mut a).unwrap();
                Reference.convert_real(f, &bytes, &mut b).unwrap();
                assert_eq!(a, b, "format {i}");
            } else {
                let (mut a, mut b) = (vec![C32::ZERO; n], vec![C32::ZERO; n]);
                Fallback.convert(f, &bytes, &mut a).unwrap();
                Reference.convert(f, &bytes, &mut b).unwrap();
                let bits = |v: &[C32]| {
                    v.iter()
                        .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&a), bits(&b), "format {i}");
                let (mut a, mut b) = (vec![CQ15::ZERO; n], vec![CQ15::ZERO; n]);
                Fallback.convert(f, &bytes, &mut a).unwrap();
                Reference.convert(f, &bytes, &mut b).unwrap();
                assert_eq!(a, b, "format {i}");
            }
        }
    }

    /// Every raw value of the 8-bit and 16-bit formats, checked against closed forms and the reference.
    #[test]
    fn convert_is_exhaustive_for_small_formats() {
        // All 65536 pairs of CU8 and CS8 bytes.
        let pairs: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        for (f, name) in [(formats::CU8, "cu8"), (formats::CS8, "cs8")] {
            let n = pairs.len() / 2;
            let (mut a, mut b) = (vec![CQ15::ZERO; n], vec![CQ15::ZERO; n]);
            Fallback.convert(&f, &pairs, &mut a).unwrap();
            Reference.convert(&f, &pairs, &mut b).unwrap();
            assert_eq!(a, b, "{name}");
            for (o, w) in a.iter().zip(pairs.chunks_exact(2)) {
                let want = |v: u8| {
                    if name == "cu8" {
                        (2 * i32::from(v) - 255) << 7
                    } else {
                        i32::from(v as i8) << 8
                    }
                };
                assert_eq!(
                    (i32::from(o.re.0), i32::from(o.im.0)),
                    (want(w[0]), want(w[1])),
                    "{name}"
                );
            }
            let (mut fa, mut fb) = (vec![C32::ZERO; n], vec![C32::ZERO; n]);
            Fallback.convert(&f, &pairs, &mut fa).unwrap();
            Reference.convert(&f, &pairs, &mut fb).unwrap();
            assert_eq!(fa, fb, "{name}");
        }
        // Every 16-bit word as I, with Q cycling, for the 16-bit formats.
        let words: Vec<u8> = (0..=u16::MAX)
            .flat_map(|w| [w, w.wrapping_mul(40503)])
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut all16 = vec![formats::CS16, formats::CS12_LSB16, formats::SC16_Q11];
        all16.push(
            SampleFormat::new(
                Container::I16,
                12,
                true,
                Layout::Interleaved,
                Align::Msb,
                Endian::Big,
            )
            .unwrap(),
        );
        all16.push(
            SampleFormat::new(
                Container::I16,
                15,
                true,
                Layout::Interleaved,
                Align::Lsb,
                Endian::Little,
            )
            .unwrap(),
        );
        for (i, f) in all16.iter().enumerate() {
            let n = words.len() / 4;
            let (mut a, mut b) = (vec![CQ15::ZERO; n], vec![CQ15::ZERO; n]);
            Fallback.convert(f, &words, &mut a).unwrap();
            Reference.convert(f, &words, &mut b).unwrap();
            assert_eq!(a, b, "format {i}");
            let (mut fa, mut fb) = (vec![C32::ZERO; n], vec![C32::ZERO; n]);
            Fallback.convert(f, &words, &mut fa).unwrap();
            Reference.convert(f, &words, &mut fb).unwrap();
            assert_eq!(fa, fb, "format {i}");
        }
        // Packed 12: every first value with a strided second value; packed 10 likewise.
        let mut p12 = Vec::new();
        for v0 in 0..4096u32 {
            for v1 in (0..4096u32).step_by(61) {
                let all = v0 | v1 << 12;
                p12.extend_from_slice(&all.to_le_bytes()[..3]);
            }
        }
        let n = p12.len() / 3;
        let (mut a, mut b) = (vec![CQ15::ZERO; n], vec![CQ15::ZERO; n]);
        Fallback.convert(&formats::PACKED12, &p12, &mut a).unwrap();
        Reference.convert(&formats::PACKED12, &p12, &mut b).unwrap();
        assert_eq!(a, b);
        let mut p10 = Vec::new();
        for v0 in 0..1024u64 {
            for v1 in (0..1024u64).step_by(37) {
                let all = v0 | v1 << 10 | (v0 ^ 0x155) << 20 | (v1 ^ 0x2AA) << 30;
                p10.extend_from_slice(&all.to_le_bytes()[..5]);
            }
        }
        let n = p10.len() / 5 * 2;
        let (mut a, mut b) = (vec![CQ15::ZERO; n], vec![CQ15::ZERO; n]);
        Fallback.convert(&formats::PACKED10, &p10, &mut a).unwrap();
        Reference.convert(&formats::PACKED10, &p10, &mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn convert_errors_agree() {
        let mut out = [C32::ZERO; 2];
        for bytes in [&[0u8; 5][..], &[0u8; 16][..], &[][..]] {
            let a = Fallback.convert(&formats::CS16, bytes, &mut out);
            let b = Reference.convert(&formats::CS16, bytes, &mut out);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn rotate_and_fir_stay_close_to_the_reference() {
        let mut s = 77u64;
        let x: Vec<C32> = (0..2048)
            .map(|_| C32::new(unit(&mut s) * 0.5, unit(&mut s) * 0.5))
            .collect();
        let (mut a, mut b) = (vec![C32::ZERO; x.len()], vec![C32::ZERO; x.len()]);
        let (mut na, mut nb) = (
            Nco::from_increment(0x1357_9BDF),
            Nco::from_increment(0x1357_9BDF),
        );
        Fallback.rotate(&mut na, &x, &mut a).unwrap();
        Reference.rotate(&mut nb, &x, &mut b).unwrap();
        assert_eq!(na, nb);
        let err: f64 = a
            .iter()
            .zip(&b)
            .map(|(p, r)| f64::from((p.re - r.re).powi(2) + (p.im - r.im).powi(2)))
            .sum();
        let sig: f64 = b.iter().map(|r| f64::from(r.re * r.re + r.im * r.im)).sum();
        assert!(10.0 * (sig / err).log10() > 120.0);

        let taps: Vec<f32> = (0..33).map(|_| unit(&mut s) * 0.1).collect();
        let mk = || FirState::<C32, _, _>::new(taps.clone(), vec![C32::ZERO; 32]).unwrap();
        let (mut sa, mut sb) = (mk(), mk());
        Fallback.fir(&mut sa, &x, &mut a).unwrap();
        Reference.fir(&mut sb, &x, &mut b).unwrap();
        assert_eq!(sa.history(), sb.history());
        let err: f64 = a
            .iter()
            .zip(&b)
            .map(|(p, r)| f64::from((p.re - r.re).powi(2) + (p.im - r.im).powi(2)))
            .sum();
        let sig: f64 = b.iter().map(|r| f64::from(r.re * r.re + r.im * r.im)).sum();
        assert!(10.0 * (sig / err).log10() > 110.0);
    }

    #[test]
    fn q15_fir_is_bit_exact() {
        let mut s = 78u64;
        let x: Vec<CQ15> = (0..2048)
            .map(|_| q(lcg(&mut s) as i16, lcg(&mut s) as i16))
            .collect();
        for n_taps in [1usize, 2, 9, 64] {
            let taps: Vec<Q15> = (0..n_taps)
                .map(|_| Q15((lcg(&mut s) >> 17) as i16 - 8192))
                .collect();
            let mk =
                || FirState::<CQ15, _, _>::new(taps.clone(), vec![CQ15::ZERO; n_taps - 1]).unwrap();
            let (mut sa, mut sb) = (mk(), mk());
            let (mut a, mut b) = (vec![CQ15::ZERO; x.len()], vec![CQ15::ZERO; x.len()]);
            Fallback.fir(&mut sa, &x, &mut a).unwrap();
            Reference.fir(&mut sb, &x, &mut b).unwrap();
            assert_eq!(a, b, "{n_taps} taps");
            assert_eq!(sa.history(), sb.history());
        }
    }

    #[test]
    fn q15_rotate_is_close_to_the_reference() {
        let mut s = 79u64;
        let x: Vec<CQ15> = (0..2048)
            .map(|_| {
                q(
                    (lcg(&mut s) >> 17) as i16 - 8192,
                    (lcg(&mut s) >> 17) as i16 - 8192,
                )
            })
            .collect();
        let (mut a, mut b) = (vec![CQ15::ZERO; x.len()], vec![CQ15::ZERO; x.len()]);
        Fallback
            .rotate(&mut Nco::from_increment(0x0F0F_0F0F), &x, &mut a)
            .unwrap();
        Reference
            .rotate(&mut Nco::from_increment(0x0F0F_0F0F), &x, &mut b)
            .unwrap();
        let err: f64 = a
            .iter()
            .zip(&b)
            .map(|(p, r)| {
                f64::from(
                    (i32::from(p.re.0) - i32::from(r.re.0)).pow(2)
                        + (i32::from(p.im.0) - i32::from(r.im.0)).pow(2),
                )
            })
            .sum();
        let sig: f64 = b
            .iter()
            .map(|r| f64::from(i32::from(r.re.0).pow(2) + i32::from(r.im.0).pow(2)))
            .sum();
        assert!(10.0 * (sig / err).log10() > 80.0);
    }
}
