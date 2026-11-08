//! Spec test 3 of M45: the lane port type `Lanes<T>` and its views `LanesRef` and `LanesMut`
//! (count, construction errors, steps, `step(i)`, skips in whole steps) and a test-only block
//! `Lanes<C32>` -> `C32` that sums the lanes of every step. That a lane view is no stream
//! element is a `compile_fail` example in the rustdoc of `Lanes`.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]

use sdrk::{C32, Delay, Progress, Ratio, RatioKind};
use sdrk_blocks::{
    Block, BlockDescription, BlockTiming, Error, LaneElem, Lanes, LanesMut, LanesRef, ParamValue,
    PortDesc, PortType, Ports,
};

type L = Lanes<C32>;

fn values(n: usize) -> Vec<C32> {
    (0..n).map(|i| C32::new(i as f32, -(i as f32))).collect()
}

static SUM_DESCRIPTION: BlockDescription = BlockDescription {
    kind: "lane_sum",
    summary: "Test block: the sum of the lanes of every step.",
    details: "Reads a lane stream and writes the sum of the lanes of each time step.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Lanes {
            elem: LaneElem::C32,
            width_param: "",
        }],
        description: "The lanes.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The sums.",
    }],
    params: &[],
    required: &[],
};

/// Sums the lanes of every step: one output sample per step.
struct LaneSum;

impl Block for LaneSum {
    type In = Lanes<C32>;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &SUM_DESCRIPTION
    }

    fn process(
        &mut self,
        input: LanesRef<'_, C32>,
        output: &mut [C32],
    ) -> sdrk_blocks::Result<Progress> {
        let n = input.steps().min(output.len());
        for (i, o) in output[..n].iter_mut().enumerate() {
            *o = input.step(i).unwrap().iter().sum();
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: Ratio::ONE,
            kind: RatioKind::Constant,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {}

    fn set_param(&mut self, _name: &str, _value: ParamValue) -> sdrk_blocks::Result<()> {
        Err(Error::UnknownParameter)
    }

    fn param(&self, _name: &str) -> sdrk_blocks::Result<ParamValue> {
        Err(Error::UnknownParameter)
    }

    fn min_input(&self) -> usize {
        1
    }
}

#[test]
fn a_lane_port_is_one_port() {
    assert_eq!(<Lanes<C32> as Ports>::COUNT, 1);
    assert_eq!(<Lanes<f32> as Ports>::COUNT, 1);
}

#[test]
fn the_views_reject_a_width_of_zero_and_a_partial_step() {
    let data = values(12);
    assert!(matches!(
        LanesRef::new(&data, 0),
        Err(Error::InvalidParameter { name: "width", .. })
    ));
    assert_eq!(
        LanesRef::new(&data[..11], 4).unwrap_err(),
        Error::Core(sdrk_core::Error::LengthMismatch {
            expected: 12,
            got: 11
        })
    );
    let mut room = values(12);
    assert!(matches!(
        LanesMut::new(&mut room, 0),
        Err(Error::InvalidParameter { name: "width", .. })
    ));
    assert!(matches!(
        LanesMut::new(&mut room[..10], 4),
        Err(Error::Core(sdrk_core::Error::LengthMismatch { .. }))
    ));
    // Empty data is zero steps of any width; width 1 is a plain stream.
    assert_eq!(LanesRef::new(&data[..0], 16).unwrap().steps(), 0);
    assert_eq!(LanesRef::new(&data, 1).unwrap().steps(), 12);
}

#[test]
fn steps_and_step_i_are_whole_steps() {
    let data = values(12);
    let view = LanesRef::new(&data, 3).unwrap();
    assert_eq!((view.width(), view.steps()), (3, 4));
    assert_eq!(view.data(), &data[..]);
    for i in 0..4 {
        assert_eq!(view.step(i).unwrap(), &data[3 * i..3 * i + 3]);
    }
    assert!(view.step(4).is_none());
    assert!(view.step(usize::MAX).is_none());

    let mut room = vec![C32::default(); 12];
    let mut out = LanesMut::new(&mut room, 3).unwrap();
    assert_eq!((out.width(), out.steps()), (3, 4));
    for i in 0..4 {
        out.step_mut(i)
            .unwrap()
            .copy_from_slice(&data[3 * i..3 * i + 3]);
    }
    assert!(out.step_mut(4).is_none());
    assert_eq!(out.step(2).unwrap(), &data[6..9]);
    assert_eq!(out.data(), &data[..]);
    out.data_mut()[11] = C32::new(7.0, 7.0);
    assert_eq!(room[11], C32::new(7.0, 7.0));
}

#[test]
fn skip_and_len_count_whole_steps() {
    let data = values(16);
    let view = LanesRef::new(&data, 4).unwrap();
    assert_eq!(L::ref_len(&view), 4);
    let rest = L::ref_skip(view, 1);
    assert_eq!((rest.width(), rest.steps()), (4, 3));
    assert_eq!(rest.data(), &data[4..]);
    assert_eq!(rest.step(0).unwrap(), &data[4..8]);
    // Beyond the end: an empty view of the same width.
    let none = L::ref_skip(view, 9);
    assert_eq!((none.width(), none.steps(), none.data().len()), (4, 0, 0));
    assert_eq!(L::ref_skip(view, usize::MAX).steps(), 0);

    let mut room = vec![C32::default(); 16];
    let mut out = LanesMut::new(&mut room, 4).unwrap();
    assert_eq!(L::mut_len(&out), 4);
    {
        let mut short = L::mut_reborrow(&mut out);
        short.step_mut(0).unwrap()[0] = C32::new(1.0, 0.0);
    }
    let mut tail = L::mut_skip(out, 3);
    assert_eq!((tail.width(), L::mut_len(&tail)), (4, 1));
    tail.step_mut(0).unwrap()[3] = C32::new(2.0, 0.0);
    assert_eq!(room[0], C32::new(1.0, 0.0));
    assert_eq!(room[15], C32::new(2.0, 0.0));
}

#[test]
fn a_block_that_reads_lanes_sums_every_step_in_any_chunking() {
    const WIDTH: usize = 8;
    const STEPS: usize = 37;
    let data = values(WIDTH * STEPS);
    let expected: Vec<C32> = data
        .chunks_exact(WIDTH)
        .map(|step| step.iter().sum())
        .collect();
    assert_eq!(LaneSum.description().kind, "lane_sum");
    // Output room of 1 to 9 samples per call: the block consumes as many steps as it writes,
    // and the caller resubmits the rest with `ref_skip`, which skips whole steps.
    for room in 1..10 {
        let mut block = LaneSum;
        let mut view = LanesRef::new(&data, WIDTH).unwrap();
        let mut got = Vec::new();
        while view.steps() > 0 {
            let mut out = vec![C32::default(); room];
            let p = block.process(view, &mut out).unwrap();
            assert_eq!(p.consumed, p.produced);
            assert!(p.produced > 0);
            got.extend_from_slice(&out[..p.produced]);
            view = L::ref_skip(view, p.consumed);
        }
        assert_eq!(got, expected, "room {room}");
    }
}

#[test]
fn the_lane_port_type_names_its_element_and_width_parameter() {
    let t = PortType::Lanes {
        elem: LaneElem::C32,
        width_param: "channels",
    };
    assert!(t.is_lanes());
    assert_ne!(
        t,
        PortType::Lanes {
            elem: LaneElem::C32,
            width_param: ""
        }
    );
    assert_eq!(LaneElem::C32.port_type(), PortType::C32);
    assert_eq!(LaneElem::of(PortType::C32), Some(LaneElem::C32));
    assert_eq!(LaneElem::of(PortType::CQ15), None);
    assert_eq!(LaneElem::C32.name(), "c32");
    assert!(SUM_DESCRIPTION.inputs[0].types[0].is_lanes());
}

#[test]
fn the_lane_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Lanes<C32>>();
    assert_send_sync::<LanesRef<'static, C32>>();
    assert_send_sync::<LanesMut<'static, C32>>();
    assert_send_sync::<LaneElem>();
}
