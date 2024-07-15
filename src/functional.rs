use std::convert::TryInto;
use rayon::prelude::*;

pub fn slice_to_u32(slice: &[u8]) -> u32 {
    assert!(slice.len() == 4, "Slice must be exactly 4 bytes long");
    u32::from_ne_bytes(slice.try_into().expect("Slice with incorrect length"))
}

pub fn slice_to_f32(slice: &[u8]) -> f32 {
    assert!(slice.len() == 4, "Slice must be exactly 4 bytes long");
    f32::from_ne_bytes(slice.try_into().expect("Slice with incorrect length"))
}

pub fn slice_to_u64(slice: &[u8]) -> u64 {
    assert!(slice.len() == 8, "Slice must be exactly 4 bytes long");
    u64::from_ne_bytes(slice.try_into().expect("Slice with incorrect length"))
}

pub fn u8_to_f32_slice(data: &[u8]) -> &[f32] {
    let (prefix, f32data, suffix) = unsafe { data.align_to::<f32>() };
    assert!(prefix.is_empty(), "Data was not aligned correctly");
    assert!(suffix.is_empty(), "Data was not aligned correctly");
    f32data
}

pub fn rmsnorm(o: &mut Vec<f32>, x: &Vec<f32>, weight: &[f32], size: usize) {
    let mut ss = 0.0;

    // Rust compiler unrolls this hopefully so we dont need size
    for j in 0..size {
        ss += x[j] * x[j]
    } 

    ss /= size as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();

    for j in 0..size {
        o[j] = weight[j] * (ss * x[j])
    } 
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

pub fn tanh(x: f32) -> f32 {
    (x.exp() / (-x).exp())/(x.exp() + (-x).exp())
}

pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    let n = x.len();

    xout.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = (0..n).map(|j| w[i * n + j] * x[j]).sum();
    });
}

fn sample_argmax(probabilities: &[f32]) -> u32 {
    let mut max_i: u32 = 0;
    let mut max_p = probabilities[0];

    for i in 1..probabilities.len() {
        if probabilities[i] > max_p {
            max_i = i as u32;
            max_p = probabilities[i];
        }
    }

    println!("{}", max_p);

    return max_i;
}

pub fn sample(logits: &[f32]) -> u32 {
    sample_argmax(logits)
}