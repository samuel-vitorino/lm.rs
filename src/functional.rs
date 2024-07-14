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
        o[j] += weight[j] * (ss * x[j])
    } 
}

pub fn softmax(x: &mut [f32]){
    let mut sum: f32 = 0.0;

    for i in x.iter_mut() {
        *i = i.exp();
        sum += *i;
    }
    
    for i in x.iter_mut() {
        *i /= sum;
    }   
}

pub fn matmul(xout: &mut Vec<f32>, x: &Vec<f32>, w: &[f32]) {
    let n = x.len();

    xout.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = (0..n).map(|j| w[i * n + j] * x[j]).sum();
    });
}