
use crate::quantization::{QuantizedTensor, MutableQuantizedTensor};

use std::{convert::TryInto};
use rayon::prelude::*;
use wide::{f32x8, i32x8};

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

pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize, eps: f32, add_unit_offset: bool) {
    let n_simd = size/8;

    let mut ss_sim = f32x8::ZERO;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j*8..j*8+8]); 
        ss_sim += x_vec * x_vec;
    } 

    let mut ss = ss_sim.reduce_add();

    ss /= size as f32;
    ss += eps;
    ss = 1.0 / ss.sqrt();

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j*8..j*8+8]);
        let w_vec = f32x8::from(&weight[j*8..j*8+8]);
        
        let r = if add_unit_offset {
            ((1.0 + w_vec) * (ss * x_vec)).to_array()
        } else {
            (w_vec * (ss * x_vec)).to_array()
        };

        for k in 0..8 {
            o[(j*8) + k] = r[k];
        } 
    } 
}

pub fn layernorm(o: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], size: usize, eps: f32) {
    let n_simd = size / 8;

    let mut mean_sim = f32x8::ZERO;
    let mut var_sim = f32x8::ZERO;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        mean_sim += x_vec;
    }

    let mean = mean_sim.reduce_add() / size as f32;

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        let diff = x_vec - f32x8::splat(mean);
        var_sim += diff * diff;
    }

    let variance = var_sim.reduce_add() / size as f32 + eps;
    let inv_std = 1.0 / variance.sqrt();

    for j in 0..n_simd {
        let x_vec = f32x8::from(&x[j * 8..j * 8 + 8]);
        let w_vec = f32x8::from(&weight[j * 8..j * 8 + 8]);
        let b_vec = f32x8::from(&bias[j * 8..j * 8 + 8]);

        let normalized = (x_vec - f32x8::splat(mean)) * f32x8::splat(inv_std);
        let r = (normalized * w_vec + b_vec).to_array(); 

        for k in 0..8 {
            o[(j * 8) + k] = r[k];
        }
    }
}

pub fn tanh_f32x8(input: f32x8) -> f32x8 {
    let two = f32x8::splat(2.0);
    let exp_2x = (input * two).exp();
    (exp_2x - f32x8::splat(1.0)) / (exp_2x + f32x8::splat(1.0))
}

pub fn softmax(x: &mut [f32]){
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

pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, o: usize) {
    let n_simd = n / 8;

    xout.par_chunks_exact_mut(o).enumerate().for_each(|(j, elem)| {
        let xi = j*n;

        elem.par_chunks_exact_mut(4).enumerate().for_each(|(i, xout_elem)| {
            let new_i = i*4;
            let ni0: usize = new_i * n;
            let ni1: usize = (new_i + 1) * n;
            let ni2: usize = (new_i + 2) * n;
            let ni3: usize = (new_i + 3) * n;
            
            xout_elem.iter_mut().for_each(|m| *m = 0.0);

            for j in 0..n_simd {
                let x_vec = f32x8::from(&x[xi+j*8..xi+j*8+8]);
                let w_vec0 = f32x8::from(&w[ni0+j*8..ni0+j*8+8]);
                let w_vec1 = f32x8::from(&w[ni1+j*8..ni1+j*8+8]);
                let w_vec2 = f32x8::from(&w[ni2+j*8..ni2+j*8+8]);
                let w_vec3 = f32x8::from(&w[ni3+j*8..ni3+j*8+8]);
                
                xout_elem[0] += (x_vec * w_vec0).reduce_add();
                xout_elem[1] += (x_vec * w_vec1).reduce_add();
                xout_elem[2] += (x_vec * w_vec2).reduce_add();
                xout_elem[3] += (x_vec * w_vec3).reduce_add();
            }
        });
    });
}

pub fn matmul_q8(xout: &mut [f32], x: &MutableQuantizedTensor, w: &QuantizedTensor, n: usize, o: usize, gs: usize) {
    let n_simd = gs / 8;
    
    xout.par_chunks_exact_mut(o).enumerate().for_each(|(j, elem)| {
        let xi = j*n;

        elem.par_chunks_exact_mut(4).enumerate().for_each(|(i, xout_elem)| { 
            let new_i = i*4;
            let ni0: usize = new_i * n;
            let ni1: usize = (new_i + 1) * n;
            let ni2: usize = (new_i + 2) * n;
            let ni3: usize = (new_i + 3) * n;

            xout_elem.iter_mut().for_each(|m| *m = 0.0);

            for j in (0..=(n - gs)).step_by(gs) {
                let mut ival0 = i32x8::ZERO;
                let mut ival1 = i32x8::ZERO;
                let mut ival2 = i32x8::ZERO;
                let mut ival3 = i32x8::ZERO;

                for k in 0..n_simd {
                    let x_vec = i32x8::from(&x.q[xi+j+k*8..xi+j+k*8+8]);
                    let w_vec0 = i32x8::from(&w.q[ni0+j+k*8..ni0+j+k*8+8]);
                    let w_vec1 = i32x8::from(&w.q[ni1+j+k*8..ni1+j+k*8+8]);
                    let w_vec2 = i32x8::from(&w.q[ni2+j+k*8..ni2+j+k*8+8]);
                    let w_vec3 = i32x8::from(&w.q[ni3+j+k*8..ni3+j+k*8+8]);

                    ival0 += x_vec * w_vec0;
                    ival1 += x_vec * w_vec1;
                    ival2 += x_vec * w_vec2;
                    ival3 += x_vec * w_vec3;
                }

                xout_elem[0] += (ival0.reduce_add() as f32) * w.s[(ni0 + j) / gs] * x.s[(xi + j) / gs];
                xout_elem[1] += (ival1.reduce_add() as f32) * w.s[(ni1 + j) / gs] * x.s[(xi + j) / gs];
                xout_elem[2] += (ival2.reduce_add() as f32) * w.s[(ni2 + j) / gs] * x.s[(xi + j) / gs];
                xout_elem[3] += (ival3.reduce_add() as f32) * w.s[(ni3 + j) / gs] * x.s[(xi + j) / gs];
            }
        });
    });
}

pub fn matmul_q4(xout: &mut [f32], x: &MutableQuantizedTensor, w: &QuantizedTensor, n: usize, o: usize, gs: usize) {
    let group_size = gs / 2;
    let n_simd = group_size / 8;

    let mask_a = i32x8::new([0x0F; 8]);
    let mask_b = i32x8::new([0xF0; 8]);
    
    xout.par_chunks_exact_mut(o).enumerate().for_each(|(j, elem)| {
        let xi = j*n;
        
        elem.par_iter_mut().enumerate().for_each(|(i, xout_elem)| {
            let ni: usize = i * n / 2;

            *xout_elem = (0..=(n/2 - group_size)).step_by(group_size).map(|j| {
                let mut ival = i32x8::ZERO;

                for k in 0..n_simd {
                    let x_vec = i32x8::from(&x.q[xi+j+k*8..xi+j+k*8+8]);
                    let w_vec = i32x8::from(&w.q[ni+j+k*8..ni+j+k*8+8]);

                    let x_a = (x_vec & mask_a) - 8;
                    let w_a = (w_vec & mask_a) - 8;
                    
                    let x_b = (mask_a & ((x_vec & mask_b) >> 4)) - 8;
                    let w_b = (mask_a & ((w_vec & mask_b) >> 4)) - 8;

                    ival += x_a * w_a;
                    ival += x_b * w_b;
                }

                (ival.reduce_add() as f32) * w.s[(ni + j) / group_size] * x.s[(xi + j) / group_size] 
            }).sum();
        });
    });
}

pub fn matmul_rest(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, o: usize) {
    let n_simd = n / 8;
    
    let rest = n_simd * 8;

    xout.par_chunks_exact_mut(o).enumerate().for_each(|(j, elem)| {
        let xi = j*n;

        elem.par_iter_mut().enumerate().for_each(|(i, val)| {
            let mut sum = f32x8::ZERO;
            let mut final_sum: f32 = 0.0;
            let w_slice = &w[i * n..i * n + n];

            for j in 0..n_simd {
                let x_vec = f32x8::from(&x[xi+j*8..xi+j*8+8]);
                let w_vec = f32x8::from(&w_slice[j*8..j*8+8]);
                sum += w_vec * x_vec;
            }

            final_sum += sum.reduce_add();
            
            for r in rest..n {
                final_sum += w_slice[r] * x[r];
            }

            *val = final_sum;
        });
    });
}

pub fn concat<T: Clone>(arr0: &[T], arr1: &[T]) -> Vec<T> {
    let mut concat_arr: Vec<T> = Vec::new();

    concat_arr.extend_from_slice(arr0);
    concat_arr.extend_from_slice(arr1);

    concat_arr
}