//! The direct loops of the fallback's `convert` and `convert_real` (one value per byte or
//! 16-bit word, full container, power-of-two scale, interleaved without I/Q swap or real) against
//! the executable specification, bit for bit: every input value, every length up to past the
//! vector widths the compiler may use, and the output beyond the samples untouched.
#![cfg(feature = "reference")]
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use sdrk_core::kernels::convert::{Convert, ConvertReal};
use sdrk_core::{C32, CQ15, Fallback, Q15, Reference};
use sdrk_device::{Align, Container, Endian, Layout, SampleFormat, formats};

/// Written beyond the converted samples.
const SENTINEL: f32 = 1234.5;
const Q15_SENTINEL: i16 = 0x1235;

fn format(
    container: Container,
    bits: u8,
    signed: bool,
    layout: Layout,
    endian: Endian,
) -> SampleFormat {
    SampleFormat::new(container, bits, signed, layout, Align::Lsb, endian).unwrap()
}

/// The formats of the direct loops and their bytes per value.
fn direct_formats() -> Vec<(SampleFormat, usize)> {
    let mut list = Vec::new();
    for layout in [Layout::Interleaved, Layout::Real] {
        for endian in [Endian::Little, Endian::Big] {
            list.push((format(Container::U8, 8, false, layout, endian), 1));
            list.push((format(Container::I8, 8, true, layout, endian), 1));
            list.push((format(Container::I16, 16, true, layout, endian), 2));
        }
    }
    assert!(list.contains(&(formats::CS16, 2)) && list.contains(&(formats::RS16, 2)));
    list
}

/// Converts `input` with the fallback and the specification and compares all outputs.
fn check(f: &SampleFormat, input: &[u8]) {
    let samples = f.samples_in_bytes(input.len());
    let what = format!("{f:?}, {} bytes", input.len());
    if f.layout() == Layout::Real {
        let mut want = vec![SENTINEL; samples + 3];
        let mut got = want.clone();
        assert_eq!(
            Fallback.convert_real(f, input, &mut got),
            Reference.convert_real(f, input, &mut want),
            "{what}"
        );
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&want), bits(&got), "{what}");
        let mut want = vec![Q15(Q15_SENTINEL); samples + 3];
        let mut got = want.clone();
        assert_eq!(
            Fallback.convert_real(f, input, &mut got),
            Reference.convert_real(f, input, &mut want),
            "{what}"
        );
        assert_eq!(want, got, "{what}");
    } else {
        let mut want = vec![C32::new(SENTINEL, SENTINEL); samples + 3];
        let mut got = want.clone();
        assert_eq!(
            Fallback.convert(f, input, &mut got),
            Reference.convert(f, input, &mut want),
            "{what}"
        );
        let bits = |x: &[C32]| {
            x.iter()
                .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&want), bits(&got), "{what}");
        let s = Q15(Q15_SENTINEL);
        let mut want = vec![CQ15 { re: s, im: s }; samples + 3];
        let mut got = want.clone();
        assert_eq!(
            Fallback.convert(f, input, &mut got),
            Reference.convert(f, input, &mut want),
            "{what}"
        );
        assert_eq!(want, got, "{what}");
    }
}

#[test]
fn every_value_bit_exact() {
    for (f, width) in direct_formats() {
        let input: Vec<u8> = (0..1usize << (8 * width))
            .flat_map(|v| v.to_le_bytes()[..width].to_vec())
            .collect();
        check(&f, &input);
    }
}

#[test]
fn every_length_bit_exact() {
    let input: Vec<u8> = (0..2 * 2 * 160u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    for (f, width) in direct_formats() {
        let per_sample = if f.layout() == Layout::Real {
            width
        } else {
            2 * width
        };
        for samples in 0..=160 {
            check(&f, &input[..samples * per_sample]);
        }
        // Partial samples and short outputs give the errors of the general path.
        for len in [1, per_sample + 1, 3 * per_sample - 1] {
            if len % per_sample != 0 {
                check(&f, &input[..len]);
            }
        }
        let mut short = vec![C32::ZERO; 3];
        let mut short_q = vec![Q15(0); 3];
        if f.layout() == Layout::Real {
            assert_eq!(
                Fallback.convert_real(&f, &input[..4 * per_sample], &mut short_q),
                Reference.convert_real(&f, &input[..4 * per_sample], &mut short_q.clone())
            );
        } else {
            assert_eq!(
                Fallback.convert(&f, &input[..4 * per_sample], &mut short),
                Reference.convert(&f, &input[..4 * per_sample], &mut short.clone())
            );
        }
    }
}
