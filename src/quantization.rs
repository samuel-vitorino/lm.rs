#[derive(Debug, Copy, Clone, PartialEq)]
pub enum QuantType {
    None = 0,
    Q8_0 = 1,
    Q4_0 = 2,
}

pub enum Tensor<'a> {
    FloatSlice(&'a [f32]),
    FloatVec(Vec<f32>),
    Quantized(Vec<QuantizedTensor<'a>>),
}

impl<'a> Tensor<'a> {
    pub fn as_float(&'a self) -> &'a [f32] {
        match self {
            Tensor::FloatSlice(data) => data,
            Tensor::FloatVec(data) => data,
            Tensor::Quantized(_) => panic!("Tried to access float tensor that is quantized"),
        }
    }

    pub fn as_quantized(&'a self) -> &'a [QuantizedTensor<'a>] {
        match self {
            Tensor::Quantized(data) => data,
            _ => panic!("Tried to access quantized tensor that is not quantized"),
        }
    }
}

pub struct QuantizedTensor<'a>{
    pub q: &'a [i8],
    pub s: &'a [f32],
}
pub struct MutableQuantizedTensor {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
}

fn unpack(value: i8) -> (i8, i8) {
    let a: i8 = (value & 0x0F) - 8;

    let b: i8 = (0x0F & ((value & 0xF0u8 as i8) >> 4)) - 8;

    (a, b)
}

pub fn dequantize(qx: &QuantizedTensor, x: &mut  [f32], n: usize, gs: u32, q_type: QuantType) {
    match q_type {
        QuantType::Q8_0 => { 
            for (i, value) in x.iter_mut().enumerate().take(n) {
                *value = qx.q[i] as f32 * qx.s[(i as u32 / gs) as usize];
            }
        },
        QuantType::Q4_0 => { 
            for i in 0..(n/2) {
                let (a, b) = unpack(qx.q[i]);
                let scale = qx.s[((i*2) as u32 / gs) as usize];
                x[i*2] = a as f32 * scale;
                x[i*2+1] = b as f32 * scale;
            }
        },
        _ => (),
    };
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

pub fn quantize_q4(qx: &mut MutableQuantizedTensor, x: & [f32], n: usize, gs: u32) {
    let num_groups: u32 = n as u32 / gs;
    let q_max: f32 = -8.0f32;

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

        for i in 0..(gs/2) {
            let quant_value_a = x[(group * gs + i*2) as usize] / scale;
            let quant_value_b = x[(group * gs + i*2 + 1) as usize] / scale;
            let quantized_a: i8 = ((quant_value_a + 8.0).round() as u8).clamp(0, 15) as i8;
            let quantized_b: i8 = ((quant_value_b + 8.0).round() as u8).clamp(0, 15) as i8;
        
            qx.q[(group * gs / 2 + i) as usize] = quantized_a | (quantized_b << 4);
        }
    }
}