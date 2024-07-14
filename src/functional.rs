use std::convert::TryInto;

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

pub fn u8_to_f32_slice(u8_slice: &[u8]) -> &[f32] {
    assert!(u8_slice.len() % 4 == 0, "Slice length must be a multiple of 4");

    unsafe {
        std::slice::from_raw_parts(u8_slice.as_ptr() as *const f32, u8_slice.len() / 4)
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