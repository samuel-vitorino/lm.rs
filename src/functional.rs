use crate::quantization::{MutableQuantizedTensor, QuantizedTensor};

use rayon::prelude::*;
use std::convert::TryInto;
use std::ops::Deref;
use wide::{f32x8, i32x8};

/// Allocs to use either a `Vec` or a slice in the same place
pub enum SliceOrVec<'a, T> {
    Slice(&'a [T]),
    Vec(Vec<T>),
}

impl<'a, T> Deref for SliceOrVec<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            &Self::Slice(slice) => slice,
            &Self::Vec(ref vec) => &vec[..],
        }
    }
}

// Some helper functions

pub fn slice_to_u32(slice: &[u8]) -> u32 {
    assert!(slice.len() == 4, "Slice must be exactly 4 bytes long");
    u32::from_ne_bytes(slice.try_into().expect("Slice with incorrect length"))
}

pub fn slice_to_f32(slice: &[u8]) -> f32 {
    assert!(slice.len() == 4, "Slice must be exactly 4 bytes long");
    f32::from_ne_bytes(slice.try_into().expect("Slice with incorrect length"))
}

pub fn u8_to_f32_slice(data: &[u8]) -> &[f32] {
    let (prefix, f32data, suffix) = unsafe { data.align_to::<f32>() };
    assert!(prefix.is_empty(), "Data was not aligned correctly");
    assert!(suffix.is_empty(), "Data was not aligned correctly");
    f32data
}

pub fn u8_to_i8_slice(data: &[u8]) -> &[i8] {
    let (prefix, i8data, suffix) = unsafe { data.align_to::<i8>() };
    assert!(prefix.is_empty(), "Data was not aligned correctly");
    assert!(suffix.is_empty(), "Data was not aligned correctly");
    i8data
}

pub fn random_u32(mut state: u64) -> u32 {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;

    ((state * 0x2545F4914F6CDD1Du64) >> 32) as u32
}

pub fn random_f32(state: u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16777216.0f32
}

// Functions used in NNs

pub fn rmsnorm(
    o: &mut [f32],
    x: &[f32],
    weight: &[f32],
    size: usize,
    eps: f32,
    add_unit_offset: bool,
) {
    let n_simd = size / 8;

    let mut ss_sim = f32x8::ZERO;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        ss_sim += x_vec * x_vec;
    }

    let mut ss = ss_sim.reduce_add();

    ss /= size as f32;
    ss += eps;
    ss = 1.0 / ss.sqrt();

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        let w_vec = f32x8::from(&weight[j * 8..j * 8 + 8]);

        let r = if add_unit_offset {
            ((1.0 + w_vec) * (ss * x_vec)).to_array()
        } else {
            (w_vec * (ss * x_vec)).to_array()
        };

        for k in 0..8 {
            o[(j * 8) + k] = r[k];
        }
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut sum: f32 = 0.0;
    let mut max_val: f32 = x[0];

    for i in x.iter() {
        if *i > max_val {
            max_val = *i;
        }
    }

    for i in x.iter_mut() {
        *i = (*i - max_val).exp();
        sum += *i;
    }

    for i in x.iter_mut() {
        *i /= sum;
    }
}

pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    let n = x.len();
    let n_simd = n / 8;

    xout.par_iter_mut().enumerate().for_each(|(i, val)| {
        let mut sum = f32x8::ZERO;
        let w_slice = &w[i * n..i * n + n];

        for j in 0..n_simd {
            let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
            let w_vec = f32x8::from(&w_slice[j * 8..j * 8 + 8]);
            sum += w_vec * x_vec;
        }

        *val = sum.reduce_add();
    });
}

pub fn matmul_q8(
    xout: &mut [f32],
    x: &MutableQuantizedTensor,
    w: &QuantizedTensor,
    n: usize,
    gs: usize,
) {
    let n_simd = gs / 8;

    xout.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
        let ni: usize = i * n;

        *xout_elem = (0..=(n - gs))
            .step_by(gs)
            .map(|j| {
                let mut ival = i32x8::ZERO;

                for k in 0..n_simd {
                    let x_vec = i32x8::from(&x.q[j + k * 8..j + k * 8 + 8]);
                    let w_vec = i32x8::from(&w.q[ni + j + k * 8..ni + j + k * 8 + 8]);

                    ival += x_vec * w_vec;
                }

                (ival.reduce_add() as f32) * w.s[(ni + j) / gs] * x.s[j / gs]
            })
            .sum();
    });
}

pub fn matmul_q4(
    xout: &mut [f32],
    x: &MutableQuantizedTensor,
    w: &QuantizedTensor,
    n: usize,
    gs: usize,
) {
    let group_size = gs / 2;
    let n_simd = group_size / 8;

    let mask_a = i32x8::new([0x0F; 8]);
    let mask_b = i32x8::new([0xF0; 8]);

    xout.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
        let ni: usize = i * n / 2;

        *xout_elem = (0..=(n / 2 - group_size))
            .step_by(group_size)
            .map(|j| {
                let mut ival = i32x8::ZERO;

                for k in 0..n_simd {
                    let x_vec = i32x8::from(&x.q[j + k * 8..j + k * 8 + 8]);
                    let w_vec = i32x8::from(&w.q[ni + j + k * 8..ni + j + k * 8 + 8]);

                    let x_a = (x_vec & mask_a) - 8;
                    let w_a = (w_vec & mask_a) - 8;

                    let x_b = (mask_a & ((x_vec & mask_b) >> 4)) - 8;
                    let w_b = (mask_a & ((w_vec & mask_b) >> 4)) - 8;

                    ival += x_a * w_a;
                    ival += x_b * w_b;
                }

                (ival.reduce_add() as f32) * w.s[(ni + j) / group_size] * x.s[j / group_size]
            })
            .sum();
    });
}
