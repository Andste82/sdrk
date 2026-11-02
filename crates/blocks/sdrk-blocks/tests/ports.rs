//! Port types: a stream block, a pair block and a frame block defined against the new `Block`
//! trait, the lockstep rule of `Two` and the blanket implementation for the stream elements.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use sdrk::{C32, CQ15, Delay, Progress, Q15, Ratio, RatioKind};
use sdrk_blocks::{
    Block, BlockDescription, BlockTiming, Error, Frame, FrameMeta, FrameOut, ParamValue, Ports,
    Result, StreamElem, Two,
};
use sdrk_device::{ErrorKind, SdrkError};

static DESC: BlockDescription = BlockDescription {
    kind: "test",
    summary: "test block",
    details: "test block",
    inputs: &[],
    outputs: &[],
    params: &[],
    required: &[],
};

fn unit() -> BlockTiming {
    BlockTiming {
        ratio: Ratio::ONE,
        kind: RatioKind::Constant,
        delay: Delay::ZERO,
    }
}

/// Implements the parts of `Block` that no test block needs.
macro_rules! plain {
    () => {
        fn description(&self) -> &'static BlockDescription {
            &DESC
        }
        fn timing(&self) -> BlockTiming {
            unit()
        }
        fn reset(&mut self) {}
        fn set_param(&mut self, _: &str, _: ParamValue) -> Result<()> {
            Err(Error::UnknownParameter)
        }
        fn param(&self, _: &str) -> Result<ParamValue> {
            Err(Error::UnknownParameter)
        }
        fn min_input(&self) -> usize {
            1
        }
    };
}

/// Stream block with the signature the real blocks keep.
struct Double;

impl Block for Double {
    type In = C32;
    type Out = C32;
    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        for (o, i) in output.iter_mut().zip(input) {
            *o = C32::new(i.re * 2.0, i.im * 2.0);
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    plain!();
}

/// Two real streams in, one complex stream out.
struct Interleave;

impl Block for Interleave {
    type In = Two<f32, f32>;
    type Out = C32;
    fn process(&mut self, input: (&[f32], &[f32]), output: &mut [C32]) -> Result<Progress> {
        let (re, im) = input;
        if re.len() != im.len() {
            return Err(Error::Core(sdrk_core::Error::LengthMismatch {
                expected: re.len(),
                got: im.len(),
            }));
        }
        let n = re.len().min(output.len());
        for ((o, r), i) in output.iter_mut().zip(re).zip(im) {
            *o = C32::new(*r, *i);
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    plain!();
}

/// One complex stream in, two real streams out.
struct Split;

impl Block for Split {
    type In = C32;
    type Out = Two<f32, f32>;
    fn process(&mut self, input: &[C32], output: (&mut [f32], &mut [f32])) -> Result<Progress> {
        let (re, im) = output;
        if re.len() != im.len() {
            return Err(Error::Core(sdrk_core::Error::LengthMismatch {
                expected: re.len(),
                got: im.len(),
            }));
        }
        let n = input.len().min(re.len());
        for ((x, r), i) in input.iter().zip(re.iter_mut()).zip(im.iter_mut()) {
            *r = x.re;
            *i = x.im;
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }
    plain!();
}

/// Frame block: one frame of the sums of `frame_len` consecutive samples.
struct Summer {
    len: usize,
    fill: usize,
    sum: f32,
}

impl Block for Summer {
    type In = f32;
    type Out = Frame;
    fn process(&mut self, input: &[f32], output: FrameOut<'_>) -> Result<Progress> {
        if output.values.len() != self.len {
            return Err(Error::Core(sdrk_core::Error::LengthMismatch {
                expected: self.len,
                got: output.values.len(),
            }));
        }
        let mut consumed = 0;
        for x in input {
            self.sum += *x;
            self.fill += 1;
            consumed += 1;
            if self.fill == self.len {
                output.values.fill(self.sum);
                *output.meta = Some(FrameMeta {
                    first_sample: 0,
                    averaged: 1,
                    exponent: 0,
                });
                self.fill = 0;
                self.sum = 0.0;
                return Ok(Progress {
                    consumed,
                    produced: 1,
                });
            }
        }
        Ok(Progress {
            consumed,
            produced: 0,
        })
    }
    fn frame_len(&self) -> usize {
        self.len
    }
    plain!();
}

#[test]
fn stream_block_keeps_its_signature() {
    let mut b = Double;
    let mut out = [C32::default(); 3];
    let p = b.process(&[C32::new(1.0, -1.0); 3], &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (3, 3));
    assert_eq!(out[2], C32::new(2.0, -2.0));
    assert_eq!(b.frame_len(), 0);
}

#[test]
fn pair_block_interleaves_and_counts_pairs() {
    let mut b = Interleave;
    let mut out = [C32::default(); 8];
    let p = b
        .process((&[1.0, 2.0, 3.0], &[-1.0, -2.0, -3.0]), &mut out)
        .unwrap();
    assert_eq!((p.consumed, p.produced), (3, 3));
    assert_eq!(out[1], C32::new(2.0, -2.0));
    // Limited by the output room: pairs, not samples.
    let mut small = [C32::default(); 2];
    let p = b
        .process((&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), &mut small)
        .unwrap();
    assert_eq!((p.consumed, p.produced), (2, 2));
}

#[test]
fn pair_input_of_unequal_length_is_rejected() {
    let mut b = Interleave;
    let mut out = [C32::new(9.0, 9.0); 4];
    let e = b
        .process((&[1.0, 2.0, 3.0], &[1.0, 2.0]), &mut out)
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
    assert!(out.iter().all(|o| *o == C32::new(9.0, 9.0)));
}

#[test]
fn pair_output_of_unequal_length_is_rejected() {
    let mut b = Split;
    let (mut re, mut im) = ([7.0f32; 4], [7.0f32; 3]);
    let e = b
        .process(&[C32::new(1.0, 2.0); 4], (&mut re, &mut im))
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
    assert!(re.iter().chain(&im).all(|v| *v == 7.0));
    let mut im = [7.0f32; 4];
    let p = b
        .process(&[C32::new(1.0, 2.0); 3], (&mut re, &mut im))
        .unwrap();
    assert_eq!((p.consumed, p.produced), (3, 3));
    assert_eq!((re[0], im[0]), (1.0, 2.0));
}

#[test]
fn frame_block_completes_frames() {
    let mut b = Summer {
        len: 4,
        fill: 0,
        sum: 0.0,
    };
    assert_eq!(b.frame_len(), 4);
    let mut values = [0.0f32; 4];
    let mut meta = None;
    let p = b
        .process(
            &[1.0; 3],
            FrameOut {
                values: &mut values,
                meta: &mut meta,
            },
        )
        .unwrap();
    assert_eq!((p.consumed, p.produced), (3, 0));
    assert!(meta.is_none());
    let p = b
        .process(
            &[1.0; 5],
            FrameOut {
                values: &mut values,
                meta: &mut meta,
            },
        )
        .unwrap();
    assert_eq!((p.consumed, p.produced), (1, 1));
    assert_eq!(values, [4.0; 4]);
    assert_eq!(meta.map(|m| m.averaged), Some(1));
    let mut short = [0.0f32; 3];
    let e = b
        .process(
            &[1.0],
            FrameOut {
                values: &mut short,
                meta: &mut meta,
            },
        )
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
}

#[test]
fn blanket_impl_covers_every_stream_element() {
    fn count<T: StreamElem>() -> usize {
        <T as Ports>::COUNT
    }
    assert_eq!(count::<C32>(), 1);
    assert_eq!(count::<CQ15>(), 1);
    assert_eq!(count::<f32>(), 1);
    assert_eq!(count::<Q15>(), 1);
    assert_eq!(count::<u8>(), 1);
    assert_eq!(<Two<f32, f32> as Ports>::COUNT, 2);
    assert_eq!(<Two<C32, u8> as Ports>::COUNT, 2);
    assert_eq!(<Frame as Ports>::COUNT, 1);
}

#[test]
fn stream_elements_map_to_port_types() {
    use sdrk_blocks::PortType;
    assert_eq!(<C32 as StreamElem>::PORT_TYPE, PortType::C32);
    assert_eq!(<CQ15 as StreamElem>::PORT_TYPE, PortType::CQ15);
    assert_eq!(<f32 as StreamElem>::PORT_TYPE, PortType::F32);
    assert_eq!(<Q15 as StreamElem>::PORT_TYPE, PortType::Q15);
    assert_eq!(<u8 as StreamElem>::PORT_TYPE, PortType::Raw);
}

#[test]
fn port_types_and_blocks_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Two<f32, f32>>();
    assert_send::<Frame>();
    assert_send::<FrameOut<'static>>();
    assert_send::<Double>();
    assert_send::<Interleave>();
    assert_send::<Split>();
    assert_send::<Summer>();
}

fn views_roundtrip<P: Ports>(
    input: P::Ref<'_>,
    mut output: P::Mut<'_>,
    n: usize,
) -> (usize, usize) {
    let rest_in = P::ref_skip(input, n);
    let n_before = P::mut_len(&output);
    {
        let re = P::mut_reborrow(&mut output);
        assert_eq!(P::mut_len(&re), n_before);
    }
    let rest_out = P::mut_skip(output, n);
    (P::ref_len(&rest_in), P::mut_len(&rest_out))
}

#[test]
fn stream_helpers() {
    let data = [1u8, 2, 3, 4];
    let mut out = [0u8; 4];
    assert_eq!(<u8 as Ports>::ref_len(&&data[..]), 4);
    assert_eq!(views_roundtrip::<u8>(&data, &mut out, 1), (3, 3));
    assert_eq!(views_roundtrip::<u8>(&data, &mut out, 4), (0, 0));
    assert_eq!(views_roundtrip::<u8>(&data, &mut out, 9), (0, 0));
    let rest = <u8 as Ports>::ref_skip(&data, 3);
    assert_eq!(rest, &[4]);
    let mut view: &mut [u8] = &mut out;
    <u8 as Ports>::mut_reborrow(&mut view)[0] = 7;
    let tail = <u8 as Ports>::mut_skip(view, 2);
    tail[0] = 9;
    assert_eq!(out, [7, 0, 9, 0]);
}

#[test]
fn two_helpers_skip_in_lockstep() {
    type P = Two<f32, C32>;
    let a = [1.0f32, 2.0, 3.0];
    let b = [C32::new(1.0, 0.0), C32::new(2.0, 0.0), C32::new(3.0, 0.0)];
    let mut oa = [0.0f32; 3];
    let mut ob = [C32::new(0.0, 0.0); 3];
    assert_eq!(<P as Ports>::ref_len(&(&a[..], &b[..])), 3);
    let rest = <P as Ports>::ref_skip((&a, &b), 2);
    assert_eq!((rest.0, rest.1), (&a[2..], &b[2..]));
    assert_eq!(
        views_roundtrip::<P>((&a, &b), (&mut oa, &mut ob), 1),
        (2, 2)
    );
    assert_eq!(
        views_roundtrip::<P>((&a, &b), (&mut oa, &mut ob), 7),
        (0, 0)
    );
    let mut view = (&mut oa[..], &mut ob[..]);
    let (x, y) = <P as Ports>::mut_reborrow(&mut view);
    x[1] = 5.0;
    y[1] = C32::new(5.0, 0.0);
    let (x, y) = <P as Ports>::mut_skip(view, 2);
    x[0] = 6.0;
    y[0] = C32::new(6.0, 0.0);
    assert_eq!(oa, [0.0, 5.0, 6.0]);
    assert_eq!(ob[2], C32::new(6.0, 0.0));
}

#[test]
fn frame_helpers_have_capacity_one() {
    let mut values = [0.0f32; 4];
    let mut meta = None;
    let mut out = FrameOut {
        values: &mut values,
        meta: &mut meta,
    };
    assert_eq!(<Frame as Ports>::mut_len(&out), 1);
    assert_eq!(<Frame as Ports>::ref_len(&&[1.0f32, 2.0][..]), 2);
    <Frame as Ports>::mut_reborrow(&mut out).values[0] = 3.0;
    assert_eq!(<Frame as Ports>::mut_len(&out), 1);
    let same = <Frame as Ports>::mut_skip(out, 0);
    assert_eq!(<Frame as Ports>::mut_len(&same), 1);
    let none = <Frame as Ports>::mut_skip(same, 1);
    assert_eq!(<Frame as Ports>::mut_len(&none), 0);
    assert_eq!(none.values.len(), 0);
    let r = <Frame as Ports>::ref_skip(&[1.0f32, 2.0, 3.0], 2);
    assert_eq!(r, &[3.0]);
    assert_eq!(values[0], 3.0);
}

#[test]
fn stream_elements_are_pod_and_default() {
    fn check<T: StreamElem>() -> usize {
        assert_eq!(
            bytemuck::bytes_of(&T::default()).len(),
            core::mem::size_of::<T>()
        );
        core::mem::size_of::<T>()
    }
    assert_eq!(check::<C32>(), 8);
    assert_eq!(check::<CQ15>(), 4);
    assert_eq!(check::<f32>(), 4);
    assert_eq!(check::<Q15>(), 2);
    assert_eq!(check::<u8>(), 1);
}
