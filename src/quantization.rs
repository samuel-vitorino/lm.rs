#[derive(Debug, Copy, Clone, PartialEq)]
pub enum QuantType {
    None,
    Q8_0,
}

pub struct QuantizedTensor<'a>{
    pub q: &'a [i8],
    pub s: &'a [f32],
}
pub struct MutableQuantizedTensor<'a>{
    pub q: &'a mut [i8],
    pub s: &'a mut [f32],
}

pub fn dequantize(qx: &QuantizedTensor, x: &mut [f32], n: usize, gs: u32) {
    for i in 0..n {
        x[i] = qx.q[i] as f32 * qx.s[(i as u32 / gs) as usize];
    }
}

pub fn quantize(qx: &mut MutableQuantizedTensor, x: & [f32], n: usize, gs: u32) {
    let num_groups: u32 = n as u32 / gs;
    let q_max: f32 = 127.0f32;

    for group in 0..num_groups {
        let mut wmax: f32 = 0.0;
        for i in 0..gs {
            let val: f32 = x[(group * gs + i) as usize].abs();
            if val > wmax {
                wmax = val;
            }
        }

        let scale = wmax / q_max;
        
        qx.s[group as usize] = scale;

        for i in 0..gs {
            let quant_value = x[(group * gs + i) as usize] / scale;
            let quantized: i8 = quant_value.round() as i8;
            qx.q[(group * gs + i) as usize] = quantized;
        }
    }
}