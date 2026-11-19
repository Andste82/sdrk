//! Golden tests of the blocks `bit_scramble` and `bit_descramble` of M44.2 (the maintainer's
//! decision on I44.2-6): `blocks/bit_scramble_x23_x18_bpsk_f32` (seed 880, the plain bit
//! scrambler `1 + x^-18 + x^-23` of ITU-T V.32 to V.34) and `blocks/bit_scramble_x23_x18_qam256_f32`
//! (seed 881, the same polynomial over symbols of 8 bits), against the one-bit-at-a-time model of
//! the register of the Python generator, bit for bit, in one call, one bit per call and in place.
#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_possible_truncation)]

use sdrk::Rate;
use sdrk_blocks::{
    BitDescrambleBlock, BitPolynomial, BitScrambleBlock, BitScrambleConfig, Block,
    SymbolMapConstellation,
};
use sdrk_testkit::golden::{self, Array, Tolerance};
use sdrk_testkit::{VecStorage, bit_exact};

fn f32s<'a>(case: &'a golden::GoldenCase, name: &str) -> &'a [f32] {
    let Array::F32(v) = case.array(name).unwrap() else {
        panic!("{name} must be f32");
    };
    v
}

/// Runs `block` over `input` in one call, one bit per call (after a reset) and returns both.
fn run<B: Block<In = f32, Out = f32>>(block: &mut B, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut whole = vec![0.0f32; input.len()];
    let p = block.process(input, &mut whole).unwrap();
    assert_eq!((p.consumed, p.produced), (input.len(), input.len()));
    block.reset();
    let mut one = vec![0.0f32; input.len()];
    for (x, y) in input.iter().zip(one.iter_mut()) {
        let p = block
            .process(core::slice::from_ref(x), core::slice::from_mut(y))
            .unwrap();
        assert_eq!((p.consumed, p.produced), (1, 1));
    }
    (whole, one)
}

#[test]
fn bit_scramble_and_bit_descramble_match_the_register_model() {
    for (name, constellation, m) in [
        (
            "bit_scramble_x23_x18_bpsk_f32",
            SymbolMapConstellation::Bpsk,
            1,
        ),
        (
            "bit_scramble_x23_x18_qam256_f32",
            SymbolMapConstellation::Qam256,
            8,
        ),
    ] {
        let case = golden::load("blocks", name).unwrap();
        assert_eq!(case.tolerance, Tolerance::BitExact);
        assert_eq!(case.param_str("polynomial").unwrap(), "x23_x18");
        let cfg = BitScrambleConfig {
            polynomial: BitPolynomial::X23X18,
            constellation,
        };
        assert_eq!(
            (
                case.param_u64("delay_a").unwrap() as usize,
                case.param_u64("delay_b").unwrap() as usize
            ),
            cfg.delays()
        );
        assert_eq!(cfg.delays(), (23 * m, 18 * m));
        let input = f32s(&case, "input");
        let rate = Rate::hz(9600).unwrap();
        let mut s = BitScrambleBlock::new(&cfg, rate, &mut VecStorage).unwrap();
        let mut d = BitDescrambleBlock::new(&cfg, rate, &mut VecStorage).unwrap();
        for (what, want, (whole, one), place) in [
            ("scrambled", f32s(&case, "scrambled"), run(&mut s, input), {
                let mut v = input.to_vec();
                let mut b = BitScrambleBlock::new(&cfg, rate, &mut VecStorage).unwrap();
                b.process_in_place(&mut v);
                v
            }),
            (
                "descrambled",
                f32s(&case, "descrambled"),
                run(&mut d, input),
                {
                    let mut v = input.to_vec();
                    let mut b = BitDescrambleBlock::new(&cfg, rate, &mut VecStorage).unwrap();
                    b.process_in_place(&mut v);
                    v
                },
            ),
        ] {
            assert_eq!(bit_exact(&whole, want), None, "{name} {what}: whole");
            assert_eq!(
                bit_exact(&one, want),
                None,
                "{name} {what}: one bit per call"
            );
            assert_eq!(bit_exact(&place, want), None, "{name} {what}: in place");
        }
    }
}
