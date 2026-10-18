//! Integration tests of `sdrk-core` (tests 1-6 and 8 of M02, plus `Scale` against the reference).

#![allow(
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::items_after_statements
)]
use sdrk_core::num::{
    Acc64, cq15_mul, cq15_mul_conj, flush_denormal, flush_denormal_f64, q15_add_sat, q15_from_f32,
    q15_mul, q15_sub_sat, round_shift, sat16,
};
use sdrk_core::timing::{Delay, Progress, RatioKind, Timing};
use sdrk_core::{C32, CQ15, ComplexSample, Error, Q15, RealSample, Result, Sample, SampleType};
use sdrk_device::{ErrorKind, Ratio, SdrkError};

struct XorShift64(u64);
impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn q15(&mut self) -> Q15 {
        Q15((self.next() >> 24) as u16 as i16)
    }
    fn unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn sat_ref(x: i64) -> i16 {
    x.clamp(-32768, 32767) as i16
}

#[test]
fn round_shift_table() {
    assert_eq!(round_shift(3, 1), 2);
    assert_eq!(round_shift(2, 1), 1);
    assert_eq!(round_shift(-3, 1), -1);
    assert_eq!(round_shift(-2, 1), -1);
    assert_eq!(round_shift(-1, 1), 0);
    for x in [-5i64, 0, 7, i64::MAX, i64::MIN] {
        assert_eq!(round_shift(x, 0), x);
    }
    // No overflow near the top of the range.
    assert_eq!(round_shift(i64::MAX, 15), i64::MAX >> 15);
}

#[test]
fn sat16_bounds() {
    assert_eq!(sat16(32768), 32767);
    assert_eq!(sat16(-32769), -32768);
    assert_eq!(sat16(i64::MAX), 32767);
    assert_eq!(sat16(i64::MIN), -32768);
    assert_eq!(sat16(-5), -5);
}

#[test]
fn q15_mul_table_and_random() {
    let m = |a: i16, b: i16| q15_mul(Q15(a), Q15(b)).0;
    assert_eq!(m(-32768, -32768), 32767);
    assert_eq!(m(16384, 16384), 8192);
    assert_eq!(m(-16384, 16384), -8192);
    assert_eq!(m(1, 1), 0);
    assert_eq!(m(32767, 32767), 32766);
    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    for _ in 0..1_000_000 {
        let (a, b) = (rng.q15(), rng.q15());
        let expect = sat_ref((i64::from(a.0) * i64::from(b.0) + (1 << 14)) >> 15);
        assert_eq!(q15_mul(a, b).0, expect);
    }
}

#[test]
fn q15_add_sub_saturate() {
    assert_eq!(q15_add_sat(Q15::MAX, Q15(1)), Q15::MAX);
    assert_eq!(q15_add_sat(Q15::MIN, Q15(-1)), Q15::MIN);
    assert_eq!(q15_sub_sat(Q15::MIN, Q15(1)), Q15::MIN);
    assert_eq!(q15_sub_sat(Q15::MAX, Q15(-1)), Q15::MAX);
    assert_eq!(q15_add_sat(Q15(5), Q15(-7)), Q15(-2));
    assert_eq!(q15_sub_sat(Q15(5), Q15(7)), Q15(-2));
}

#[test]
fn q15_from_f32_rules() {
    assert_eq!(q15_from_f32(1.0), Q15(32767));
    assert_eq!(q15_from_f32(-1.0), Q15(-32768));
    assert_eq!(q15_from_f32(0.5), Q15(16384));
    assert_eq!(q15_from_f32(f32::NAN), Q15(0));
    assert_eq!(q15_from_f32(f32::INFINITY), Q15(32767));
    assert_eq!(q15_from_f32(f32::NEG_INFINITY), Q15(-32768));
    assert_eq!(q15_from_f32(1.0 / 65536.0), Q15(1));
    assert_eq!(q15_from_f32(-1.0 / 65536.0), Q15(0));
    assert_eq!(q15_from_f32(1e30), Q15(32767));
    assert_eq!(Q15::from_f32(0.25), Q15(8192));
}

#[test]
fn q15_f32_round_trip_exhaustive() {
    for r in i16::MIN..=i16::MAX {
        assert_eq!(q15_from_f32(Q15(r).to_f32()), Q15(r));
    }
}

#[test]
fn real_sample_conversions() {
    assert_eq!(<f32 as RealSample>::from_f64_round(0.1), 0.1f32);
    assert_eq!(<f32 as RealSample>::to_f64(0.5), 0.5);
    assert_eq!(Q15(-32768).to_f64(), -1.0);
    assert_eq!(Q15::from_f64_round(0.5), Q15(16384));
    assert_eq!(Q15::from_f64_round(f64::NAN), Q15(0));
    assert_eq!(Q15::from_f64_round(2.0), Q15::MAX);
    assert_eq!(Q15::from_f64_round(-2.0), Q15::MIN);
}

#[test]
fn complex_sample_conversions() {
    let c = C32::new(0.25, -0.5);
    assert_eq!(<C32 as ComplexSample>::re(c), 0.25);
    assert_eq!(<C32 as ComplexSample>::im(c), -0.5);
    assert_eq!(<C32 as ComplexSample>::new(1.0, 2.0), C32::new(1.0, 2.0));
    assert_eq!(C32::from_c64_round(c.to_c64()), c);
    let q = CQ15::from_c32(c);
    assert_eq!(
        q,
        CQ15 {
            re: Q15(8192),
            im: Q15(-16384)
        }
    );
    assert_eq!(q.to_c32(), c);
    assert_eq!(<CQ15 as ComplexSample>::re(q), Q15(8192));
    assert_eq!(<CQ15 as ComplexSample>::im(q), Q15(-16384));
    assert_eq!(
        <CQ15 as ComplexSample>::new(Q15(1), Q15(2)),
        CQ15 {
            re: Q15(1),
            im: Q15(2)
        }
    );
    assert_eq!(CQ15::from_c64_round(q.to_c64()), q);
}

fn rand_cq15(rng: &mut XorShift64) -> CQ15 {
    CQ15 {
        re: rng.q15(),
        im: rng.q15(),
    }
}

#[test]
fn cq15_mul_identities() {
    let one = CQ15 {
        re: Q15(32767),
        im: Q15(0),
    };
    let j = CQ15 {
        re: Q15(0),
        im: Q15(32767),
    };
    let mut rng = XorShift64(12345);
    for _ in 0..10_000 {
        let x = rand_cq15(&mut rng);
        let y = cq15_mul(x, one);
        assert!((i32::from(y.re.0) - i32::from(x.re.0)).abs() <= 1);
        assert!((i32::from(y.im.0) - i32::from(x.im.0)).abs() <= 1);
        let r = cq15_mul(x, j);
        // x * j = -im + j re; -(-32768) saturates, so allow the +1 of the saturated value.
        assert!((i32::from(r.re.0) - (-i32::from(x.im.0)).min(32767)).abs() <= 1);
        assert!((i32::from(r.im.0) - i32::from(x.re.0)).abs() <= 1);
    }
}

#[test]
fn cq15_mul_matches_formula() {
    let mut rng = XorShift64(777);
    for _ in 0..100_000 {
        let (a, b) = (rand_cq15(&mut rng), rand_cq15(&mut rng));
        let (ar, ai, br, bi) = (
            i64::from(a.re.0),
            i64::from(a.im.0),
            i64::from(b.re.0),
            i64::from(b.im.0),
        );
        let f = |x: i64| sat_ref((x + (1 << 14)) >> 15);
        let p = cq15_mul(a, b);
        assert_eq!(
            (p.re.0, p.im.0),
            (f(ar * br - ai * bi), f(ar * bi + ai * br))
        );
        let c = cq15_mul_conj(a, b);
        assert_eq!(
            (c.re.0, c.im.0),
            (f(ar * br + ai * bi), f(ai * br - ar * bi))
        );
    }
}

#[test]
fn acc64_behaviour() {
    let mut acc = Acc64::default();
    for _ in 0..1000 {
        acc.mac(Q15::MIN, Q15::MIN);
    }
    assert_eq!(acc.0, 1000 * (1i64 << 30));
    assert_eq!(acc.finish(), Q15::MAX);
    let mut neg = Acc64::default();
    for _ in 0..1000 {
        neg.mac(Q15::MIN, Q15::MAX);
    }
    assert_eq!(neg.finish(), Q15::MIN);
    // A single rounding at the end: 3 * 0.5 LSB-products must not round individually.
    let mut a = Acc64::default();
    a.mac(Q15(181), Q15(181)); // 32761 -> rounds to 1 on its own
    a.mac(Q15(181), Q15(-181));
    assert_eq!(a.finish(), Q15(0));
}

#[test]
fn flush_denormal_cases() {
    assert_eq!(flush_denormal(1e-31), 0.0);
    assert_eq!(flush_denormal(-1e-31), 0.0);
    assert_eq!(flush_denormal(1e-29), 1e-29);
    assert!(flush_denormal(f32::NAN).is_nan());
    assert_eq!(flush_denormal_f64(1e-31), 0.0);
    assert_eq!(flush_denormal_f64(-1e-31), 0.0);
    assert_eq!(flush_denormal_f64(1e-29), 1e-29);
    assert!(flush_denormal_f64(f64::NAN).is_nan());
}

#[test]
fn delay_sanity() -> Result<()> {
    assert_eq!(Delay::ZERO.as_f64(), 0.0);
    let d = Delay::new(63, 2)?;
    assert_eq!((d.num(), d.den()), (63, 2));
    assert_eq!(d.as_f64(), 31.5);
    assert_eq!(Delay::new(126, 4)?.reduced(), d);
    assert_eq!(Delay::new(0, 7)?.reduced(), Delay::ZERO);
    assert_eq!(Delay::new(31, 1)?.add(Delay::new(1, 2)?)?, d);
    assert_eq!(Delay::new(1, 3)?.add(Delay::new(1, 6)?)?, Delay::new(1, 2)?);
    assert!(matches!(Delay::new(1, 0), Err(Error::InvalidParameter(_))));
    assert!(Delay::new(u64::MAX, 1)?.add(Delay::new(1, 1)?).is_err());
    assert!(
        Delay::new(1, u64::MAX)?
            .add(Delay::new(1, u64::MAX - 1)?)
            .is_err()
    );
    Ok(())
}

#[test]
fn progress_and_ratio_kind() {
    assert_eq!(
        Progress::default(),
        Progress {
            consumed: 0,
            produced: 0
        }
    );
    assert_ne!(RatioKind::Constant, RatioKind::Variable);

    struct T;
    impl Timing for T {
        fn ratio(&self) -> Ratio {
            Ratio::ONE
        }
        fn group_delay(&self) -> Delay {
            Delay::ZERO
        }
    }
    assert_eq!(T.ratio_kind(), RatioKind::Constant);
    assert_eq!(T.ratio(), Ratio::ONE);
    assert_eq!(T.group_delay(), Delay::ZERO);
}

#[test]
fn sample_type_tags() {
    assert_eq!(SampleType::of::<C32>(), SampleType::C32);
    assert_eq!(SampleType::of::<CQ15>(), SampleType::CQ15);
    assert_eq!(SampleType::of::<f32>(), SampleType::F32);
    assert_eq!(SampleType::of::<Q15>(), SampleType::Q15);
    assert_eq!(<C32 as Sample>::ZERO, C32::new(0.0, 0.0));
    assert_eq!(<CQ15 as Sample>::ZERO, CQ15::default());
    assert_eq!(<Q15 as Sample>::ZERO, Q15(0));
    assert_eq!(<f32 as Sample>::ZERO, 0.0);
}

#[test]
fn memory_layout_matches_interleaved_formats() {
    let x = [CQ15 {
        re: Q15(0x0102),
        im: Q15(-2),
    }];
    assert_eq!(
        bytemuck::cast_slice::<CQ15, u8>(&x),
        &[0x02, 0x01, 0xFE, 0xFF]
    );
    assert_eq!(core::mem::size_of::<CQ15>(), 4);
    assert_eq!(core::mem::size_of::<C32>(), 8);
    let y = [C32::new(1.0, 2.0)];
    let mut expect = [0u8; 8];
    expect[..4].copy_from_slice(&1.0f32.to_le_bytes());
    expect[4..].copy_from_slice(&2.0f32.to_le_bytes());
    assert_eq!(bytemuck::cast_slice::<C32, u8>(&y), &expect);
}

#[test]
fn error_kinds_and_display() {
    let cases: [(Error, ErrorKind, &str); 6] = [
        (
            Error::LengthMismatch {
                expected: 2,
                got: 3,
            },
            ErrorKind::LengthMismatch,
            "length mismatch: expected 2, got 3",
        ),
        (
            Error::BufferTooSmall { needed: 4, got: 1 },
            ErrorKind::BufferTooSmall,
            "buffer too small: needed 4, got 1",
        ),
        (
            Error::InvalidParameter("x"),
            ErrorKind::InvalidArgument,
            "invalid parameter: x",
        ),
        (
            Error::Unsupported("y"),
            ErrorKind::Unsupported,
            "unsupported: y",
        ),
        (
            Error::Internal("z"),
            ErrorKind::Internal,
            "internal error: z",
        ),
        (
            Error::Device(sdrk_device::Error::Overflow("o")),
            ErrorKind::OutOfRange,
            "device error: arithmetic overflow: o",
        ),
    ];
    for (e, k, s) in cases {
        assert_eq!(e.kind(), k);
        assert_eq!(format!("{e}"), s);
    }
    let wrapped: Error = sdrk_device::Error::Overflow("o").into();
    assert!(core::error::Error::source(&wrapped).is_some());
    assert!(core::error::Error::source(&Error::Internal("z")).is_none());
}

#[test]
fn public_types_are_send_sync() {
    fn assert_ss<T: Send + Sync>() {}
    assert_ss::<Q15>();
    assert_ss::<CQ15>();
    assert_ss::<C32>();
    assert_ss::<SampleType>();
    assert_ss::<Error>();
    assert_ss::<Delay>();
    assert_ss::<RatioKind>();
    assert_ss::<Progress>();
    assert_ss::<Acc64>();
    assert_ss::<sdrk_core::Fallback>();
    #[cfg(feature = "reference")]
    assert_ss::<sdrk_core::Reference>();
}

// ---- Scale ----

mod scale {
    use super::*;
    use sdrk_core::Fallback;
    use sdrk_core::kernels::basic::Scale;

    fn snr_db(reference: &[C32], out: &[C32]) -> f64 {
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for (r, o) in reference.iter().zip(out) {
            let (rr, ri) = (f64::from(r.re), f64::from(r.im));
            let (dr, di) = (rr - f64::from(o.re), ri - f64::from(o.im));
            sig += rr * rr + ri * ri;
            err += dr * dr + di * di;
        }
        if err == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (sig / err).log10()
        }
    }

    #[test]
    fn fallback_c32_values() -> Result<()> {
        let input = [C32::new(1.0, -2.0), C32::new(0.0, 4.0)];
        let mut out = [C32::default(); 2];
        Fallback.scale(&input, &mut out, -0.5)?;
        assert_eq!(out, [C32::new(-0.5, 1.0), C32::new(-0.0, -2.0)]);
        Ok(())
    }

    #[test]
    fn fallback_cq15_values() -> Result<()> {
        let input = [
            CQ15 {
                re: Q15(16384),
                im: Q15(-16384),
            },
            CQ15 {
                re: Q15::MIN,
                im: Q15::MIN,
            },
        ];
        let mut out = [CQ15::default(); 2];
        Fallback.scale(&input, &mut out, Q15(16384))?;
        assert_eq!(
            out[0],
            CQ15 {
                re: Q15(8192),
                im: Q15(-8192)
            }
        );
        // -1 * -1 saturates to the maximum.
        Fallback.scale(&input, &mut out, Q15::MIN)?;
        assert_eq!(
            out[1],
            CQ15 {
                re: Q15::MAX,
                im: Q15::MAX
            }
        );
        Ok(())
    }

    #[test]
    fn f32_edge_values() -> Result<()> {
        let input = [C32::new(f32::NAN, f32::INFINITY), C32::new(f32::MAX, 0.0)];
        let mut out = [C32::default(); 2];
        Fallback.scale(&input, &mut out, 2.0)?;
        assert!(out[0].re.is_nan());
        assert_eq!(out[0].im, f32::INFINITY);
        assert_eq!(out[1].re, f32::INFINITY);
        let zeros = [C32::default(); 3];
        let mut o3 = [C32::new(1.0, 1.0); 3];
        Fallback.scale(&zeros, &mut o3, 123.0)?;
        assert_eq!(o3, zeros);
        Ok(())
    }

    #[test]
    fn empty_input_is_ok() -> Result<()> {
        Fallback.scale(&[] as &[C32], &mut [], 1.0)?;
        Ok(())
    }

    #[test]
    fn length_mismatch() {
        let mut out = [C32::default(); 3];
        let e = Fallback.scale(&[C32::default(); 2], &mut out, 1.0);
        assert_eq!(
            e,
            Err(Error::LengthMismatch {
                expected: 2,
                got: 3
            })
        );
        let mut outq = [CQ15::default(); 1];
        let e = Fallback.scale(&[CQ15::default(); 2], &mut outq, Q15(1));
        assert_eq!(
            e,
            Err(Error::LengthMismatch {
                expected: 2,
                got: 1
            })
        );
    }

    #[cfg(feature = "reference")]
    mod against_reference {
        use super::*;
        use sdrk_core::Reference;

        #[test]
        fn c32_snr_at_least_140_db() -> Result<()> {
            let mut rng = XorShift64(0xC0FFEE);
            let input: Vec<C32> = (0..4096)
                .map(|_| C32::new(rng.unit_f32(), rng.unit_f32()))
                .collect();
            let gain = 0.37f32;
            let mut a = vec![C32::default(); 4096];
            let mut b = vec![C32::default(); 4096];
            Fallback.scale(&input, &mut a, gain)?;
            Reference.scale(&input, &mut b, gain)?;
            assert!(snr_db(&b, &a) >= 140.0);
            Ok(())
        }

        #[test]
        fn cq15_bit_exact() -> Result<()> {
            let mut rng = XorShift64(0xBEEF);
            let input: Vec<CQ15> = (0..4096).map(|_| rand_cq15(&mut rng)).collect();
            for gain in [Q15(12345), Q15::MIN, Q15::MAX, Q15(0), Q15(-1)] {
                let mut a = vec![CQ15::default(); 4096];
                let mut b = vec![CQ15::default(); 4096];
                Fallback.scale(&input, &mut a, gain)?;
                Reference.scale(&input, &mut b, gain)?;
                assert_eq!(a, b);
            }
            Ok(())
        }

        #[test]
        fn reference_length_mismatch() {
            let mut out = [C32::default(); 1];
            assert_eq!(
                Reference.scale(&[C32::default(); 2], &mut out, 1.0),
                Err(Error::LengthMismatch {
                    expected: 2,
                    got: 1
                })
            );
            let mut outq = [CQ15::default(); 1];
            assert_eq!(
                Reference.scale(&[CQ15::default(); 2], &mut outq, Q15(1)),
                Err(Error::LengthMismatch {
                    expected: 2,
                    got: 1
                })
            );
        }
    }
}
