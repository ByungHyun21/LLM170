
use crate::matmul::cpu::{matmul, matmul_batch};
use crate::matmul::weight::Weight;

pub fn mm_group(xs: &[Vec<f32>], ws: &[Weight], outs: &mut [Vec<Vec<f32>>]) {
    for (w, out) in ws.iter().zip(outs.iter_mut()) {
        matmul_batch(xs, w, out);
    }
}

pub fn mm_batch(xs: &[Vec<f32>], w: &Weight, outs: &mut [Vec<f32>]) {
    matmul_batch(xs, w, outs);
}

pub fn mm_group_ref(xs: &[&[f32]], ws: &[Weight], outs: &mut [Vec<Vec<f32>>]) {
    for (w, out) in ws.iter().zip(outs.iter_mut()) {
        crate::matmul::cpu::matmul_batch_ref(xs, w, out);
    }
}

pub fn mm(x: &[f32], w: &Weight, out: &mut [f32]) {
    matmul(x, w, out);
}
