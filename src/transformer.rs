use crate::functional::slice_to_u32;
use crate::functional::u8_to_f32_slice;
use crate::functional::u8_to_i8_slice;
use crate::functional::rmsnorm;
use crate::functional::matmul;
use crate::functional::qmatmul;
use crate::functional::softmax;

use crate::quantization::*;

use memmap2::Mmap;
use rayon::prelude::*;
use std::alloc::{dealloc, Layout};
use std::mem::{MaybeUninit, size_of};

fn init_param<'a>(data: &'a [u8], offset: &mut usize, n: u32, size_each: u32) -> &'a [f32]{
    let ptr: &[f32]  = u8_to_f32_slice(&data[*offset..(*offset + ((n * size_each) as usize * size_of::<f32>()))]);

    *offset += (n * size_each) as usize * size_of::<f32>();

    return ptr;
}

fn init_param_quant<'a>(data: &'a [u8], offset: &mut usize, n: u32, size_each: u32, gs: u32) -> &'a [QuantizedTensor<'a>]{
    let mut res: Vec<QuantizedTensor> = Vec::with_capacity(n as usize);

    for _ in 0..n {
        let mut qt = QuantizedTensor { q: &mut [], s: &mut [] };

        qt.q = u8_to_i8_slice(&data[*offset..(*offset + (size_each as usize * size_of::<i8>()))]);
        
        *offset += size_each as usize * size_of::<i8>() ;

        qt.s = u8_to_f32_slice(&data[*offset..(*offset + ((size_each / gs) as usize * size_of::<f32>()))]);
        
        *offset += (size_each / gs) as usize * size_of::<f32>();

        res.push(qt);
    }

    return Box::leak(res.into_boxed_slice());
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct TranformerArgs {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    head_size: u32,
    n_kv_heads: u32,
    pub vocab_size: u32,
    seq_len: u32,
    q_type: QuantType,
    group_size: u32,
}

pub struct TransformerWeights<'a> {
    token_embedding_table: &'a [f32],

    // Attention

    wq: MaybeUninit<&'a [f32]>,
    wk: MaybeUninit<&'a [f32]>,
    wv: MaybeUninit<&'a [f32]>,
    wo: MaybeUninit<&'a [f32]>,
    
    wq_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    wk_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    wv_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    wo_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,

    w_rms_att: &'a [f32],

    // FFN
    w1: MaybeUninit<&'a [f32]>,
    w2: MaybeUninit<&'a [f32]>,
    w3: MaybeUninit<&'a [f32]>,

    w1_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    w2_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    w3_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,

    w_rms_post_att: &'a [f32],
    w_rms_pre_ffn: &'a [f32],
    w_rms_post_ffn: &'a [f32],

    w_rms_final: &'a [f32],

    w_cls: MaybeUninit<&'a [f32]>,
    w_cls_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
}

pub struct TransformerState<'a>
{
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>, 
    xb3: Vec<f32>, 
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    xq: MaybeUninit<MutableQuantizedTensor<'a>>,
    xq1: MaybeUninit<MutableQuantizedTensor<'a>>,
    hq: MaybeUninit<MutableQuantizedTensor<'a>>,
    logits: Vec<f32>, 

    // kv cache
    key_cache: Vec<f32>,
    value_cache: Vec<f32>, 
}

pub struct Transformer<'a> {
    pub args: TranformerArgs,
    weights: TransformerWeights<'a>,
    state: TransformerState<'a>,
}

impl<'a> Transformer<'a> {
    pub fn new(data: &'a Mmap) -> Transformer<'a> {
        assert_eq!(data[0..4], [0x6c, 0x6d, 0x72, 0x73], "Model not in llm.rs format.");

        let lmrs_version = slice_to_u32(&data[4..8]);

        println!("LMRS version: {}", lmrs_version);
        
        let (head, body, _) = unsafe { data[8..45].align_to::<TranformerArgs>() };

        assert!(head.is_empty(), "Data was not aligned");
        
        let cfg = &body[0];

        let head_size = cfg.head_size;
        
        let mut offset: usize = 48;

        let quantized = cfg.q_type != QuantType::None;
        
        if cfg.q_type == QuantType::Q8_0 { println!("Using Q8_0 quantization.\n") };

        let kv_dim = cfg.head_size * cfg.n_kv_heads;

        if !quantized {
            let emb_tab = init_param(&data, &mut offset, 1, cfg.vocab_size * cfg.dim);
            let rms_att = init_param(&data, &mut offset, cfg.n_layers, cfg.dim);
            let wq = init_param(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size);
            let wk = init_param(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size);
            let wv = init_param(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size);
            let wo = init_param(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size);
            let rms_post_att = init_param(&data, &mut offset, cfg.n_layers, cfg.dim);
            let rms_pre_ffn = init_param(&data, &mut offset, cfg.n_layers, cfg.dim);
            let w1 = init_param(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
            let w2 = init_param(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
            let w3 = init_param(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
            let rms_post_ffn = init_param(&data, &mut offset, cfg.n_layers, cfg.dim);
            let rms_final = init_param(&data, &mut offset, 1, cfg.dim);
            
            let weights = TransformerWeights {
                token_embedding_table: emb_tab,
                wq: MaybeUninit::new(wq),
                wk: MaybeUninit::new(wk),
                wv: MaybeUninit::new(wv),
                wo: MaybeUninit::new(wo),
                wq_quant: MaybeUninit::uninit(),
                wk_quant: MaybeUninit::uninit(),
                wv_quant: MaybeUninit::uninit(),
                wo_quant: MaybeUninit::uninit(),
                w_rms_att: rms_att,
                w1: MaybeUninit::new(w1),
                w2: MaybeUninit::new(w2),
                w3: MaybeUninit::new(w3),
                w1_quant: MaybeUninit::uninit(),
                w2_quant: MaybeUninit::uninit(),
                w3_quant: MaybeUninit::uninit(),
                w_rms_post_att: rms_post_att,
                w_rms_pre_ffn: rms_pre_ffn,
                w_rms_post_ffn: rms_post_ffn,
                w_rms_final: rms_final,
                w_cls: MaybeUninit::new(emb_tab),
                w_cls_quant: MaybeUninit::uninit(),
            };

            let state = TransformerState {
                x: vec![0.0; cfg.dim as usize],
                xb: vec![0.0; cfg.dim as usize],
                xb2: vec![0.0; cfg.dim as usize],
                xb3: vec![0.0; (cfg.head_size*cfg.n_heads) as usize],
                hb: vec![0.0; cfg.hidden_dim as usize],
                hb2: vec![0.0; cfg.hidden_dim as usize],
                q: vec![0.0; (cfg.head_size*cfg.n_heads) as usize],
                xq: MaybeUninit::uninit(),
                xq1: MaybeUninit::uninit(),
                hq: MaybeUninit::uninit(),
                key_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
                value_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
                logits: vec![0.0; cfg.vocab_size as usize],
            };
            
            return Transformer {
                args: *cfg,
                weights,
                state,
            }
        } 

        println!("Loading weights...");

        let emb_tab_quant = init_param_quant(&data, &mut offset, 1, cfg.vocab_size * cfg.dim, cfg.group_size);

        let mut emb_tab: Vec<f32> = vec![0.0; (cfg.vocab_size * cfg.dim) as usize];

        dequantize(&emb_tab_quant[0], &mut emb_tab, (cfg.vocab_size * cfg.dim) as usize, cfg.group_size);

        let rms_att = init_param(&data, &mut offset, cfg.n_layers, cfg.dim);
        let wq_quant = init_param_quant(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size);
        let wk_quant = init_param_quant(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size, cfg.group_size);
        let wv_quant = init_param_quant(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size, cfg.group_size);
        let wo_quant = init_param_quant(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size);
        let rms_post_att = init_param(&data, &mut offset, cfg.n_layers, cfg.dim);
        let rms_pre_ffn = init_param(&data, &mut offset, cfg.n_layers, cfg.dim);
        let w1_quant = init_param_quant(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size);
        let w2_quant = init_param_quant(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size);
        let w3_quant = init_param_quant(&data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size);
        let rms_post_ffn = init_param(&data, &mut offset, cfg.n_layers, cfg.dim);
        let rms_final = init_param(&data, &mut offset, 1, cfg.dim); 
        
        let weights = TransformerWeights {
            token_embedding_table: Box::leak(emb_tab.into_boxed_slice()),
            wq: MaybeUninit::uninit(),
            wk: MaybeUninit::uninit(),
            wv: MaybeUninit::uninit(),
            wo: MaybeUninit::uninit(),
            wq_quant: MaybeUninit::new(wq_quant),
            wk_quant: MaybeUninit::new(wk_quant),
            wv_quant: MaybeUninit::new(wv_quant),
            wo_quant: MaybeUninit::new(wo_quant),
            w_rms_att: rms_att,
            w1: MaybeUninit::uninit(),
            w2: MaybeUninit::uninit(),
            w3: MaybeUninit::uninit(),
            w1_quant: MaybeUninit::new(w1_quant),
            w2_quant: MaybeUninit::new(w2_quant),
            w3_quant: MaybeUninit::new(w3_quant),
            w_rms_post_att: rms_post_att,
            w_rms_pre_ffn: rms_pre_ffn,
            w_rms_post_ffn: rms_post_ffn,
            w_rms_final: rms_final,
            w_cls: MaybeUninit::uninit(),
            w_cls_quant: MaybeUninit::new(emb_tab_quant),
        };

        let state = TransformerState {
            x: vec![0.0; cfg.dim as usize],
            xb: vec![0.0; cfg.dim as usize],
            xb2: vec![0.0; cfg.dim as usize],
            xb3: vec![0.0; (cfg.head_size*cfg.n_heads) as usize],
            hb: vec![0.0; cfg.hidden_dim as usize],
            hb2: vec![0.0; cfg.hidden_dim as usize],
            q: vec![0.0; (cfg.head_size*cfg.n_heads) as usize],
            xq: MaybeUninit::new(MutableQuantizedTensor { q: Box::leak(vec![0; (cfg.dim) as usize].into_boxed_slice()), s: Box::leak(vec![0.0; (cfg.dim) as usize].into_boxed_slice())}),
            xq1: MaybeUninit::new(MutableQuantizedTensor { q: Box::leak(vec![0; (cfg.head_size*cfg.n_heads) as usize].into_boxed_slice()), s: Box::leak(vec![0.0; (cfg.head_size*cfg.n_heads) as usize].into_boxed_slice())}),
            hq: MaybeUninit::new(MutableQuantizedTensor { q: Box::leak(vec![0; (cfg.hidden_dim) as usize].into_boxed_slice()), s: Box::leak(vec![0.0; (cfg.hidden_dim) as usize].into_boxed_slice())}),
            key_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            value_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            logits: vec![0.0; cfg.vocab_size as usize],
        };
        
        println!("Done.\n");
        
        return Transformer {
            args: *cfg,
            weights,
            state,
        }
    }

    pub fn forward(&mut self, token: u32, pos: u32) -> &mut [f32] {
        let p = self.args;
        let w = &self.weights;
        let s = &mut self.state;
        let x = &mut s.x;
        let dim = p.dim;
        let head_size = p.head_size;
        let att_dim = p.n_heads * head_size;
        let kv_dim = head_size * p.n_kv_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let hidden_dim = p.hidden_dim;
        let gs = p.group_size;

        let quantized = p.q_type != QuantType::None;

        x.copy_from_slice(&w.token_embedding_table[(token * dim) as usize..(token * dim + dim) as usize]);

        // Gemma normalizes the token embeddings by sqrt(dim)
        let normalizer = (dim as f32).sqrt();
        for i in x.iter_mut() {
            *i *= normalizer;
        }

        for l in 0..p.n_layers {
            rmsnorm(&mut s.xb, x, &w.w_rms_att[(l*dim) as usize..(l*dim + dim) as usize], dim as usize);

            let loff = l * p.seq_len * kv_dim; 
            let mut k = &mut s.key_cache[(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim) as usize];
            let mut v = &mut s.value_cache[(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim) as usize];
            
            unsafe {
                if !quantized {
                    matmul(&mut s.q, &s.xb, &w.wq.assume_init()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize]);
                    matmul(k, &s.xb, &w.wk.assume_init()[(l*dim*kv_dim) as usize..(l*dim*kv_dim + dim*kv_dim) as usize]);
                    matmul(v, &s.xb, &w.wv.assume_init()[(l*dim*kv_dim) as usize..(l*dim*kv_dim + dim*kv_dim) as usize]);
                } else {
                    let sxq = &mut *s.xq.as_mut_ptr();

                    quantize(sxq, &s.xb, dim as usize, gs);
                    qmatmul(&mut s.q, sxq, &w.wq_quant.assume_init()[l as usize], dim as usize, gs as usize);
                    qmatmul(&mut k, sxq, &w.wk_quant.assume_init()[l as usize], dim as usize, gs as usize);
                    qmatmul(&mut v, sxq, &w.wv_quant.assume_init()[l as usize], dim as usize, gs as usize);
                }
            }

            // In gemma they chunk and stack the input matrix to use as complex numbers so actually for calculation, vec[i+1] is vec[i+head_size/2]
            for i in 0..p.n_heads {
                for j in 0..(head_size/2) {
                    let head_dim: u32 = j * 2;
                    let freq: f32 = 1.0 / 10000.0f32.powf(head_dim as f32/head_size as f32);
                    let val: f32 = pos as f32 * freq;
                    let fcr = val.cos();
                    let fci = val.sin();
                    let rotn: u32 = if (i*head_size) + j + head_size/2 < kv_dim {2} else {1};

                    for v in 0..rotn{
                        let vec: &mut [f32] = if v == 0 {&mut s.q} else {k};
                        let v0: f32 = vec[((i*head_size) + j) as usize];
                        let v1: f32 = vec[(((i*head_size) + j)+(head_size/2)) as usize];
                        
                        vec[((i*head_size) + j) as usize] = v0 * fcr - v1 * fci;
                        vec[(((i*head_size) + j)+(head_size/2)) as usize]= v0 * fci + v1 * fcr;
                    }
                }
            }

            s.xb3.par_chunks_mut(head_size as usize).enumerate().for_each( |(h, xb)| {
                let q = &s.q[(h as u32 * head_size) as usize..(h as u32 * head_size + head_size) as usize];

                let att = &mut vec![0.0; p.seq_len as usize];

                for t in 0..pos+1 {
                    let k = &s.key_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size) as usize..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size) as usize];
                    
                    let mut score: f32 = 0.0;

                    for i in 0..head_size {
                        score += q[i as usize] * k[i as usize];
                    }
                    
                    score /= (256.0f32).sqrt();

                    // Softcapping
                    score /= 50.0f32;
                    score = (score as f64).tanh() as f32;
                    score *= 50.0f32;
                    
                    // Local attention
                    score += if pos - t <= 4096 {0.0} else {-2.3819763e38};
 
                    att[t as usize] = score;
                }

                softmax(&mut att[..(pos+1) as usize]);

                xb.fill(0.0);

                for t in 0..pos+1 {
                    let v = &s.value_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size) as usize..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size) as usize];
                    let a = att[t as usize];

                    for i in 0..head_size {
                        xb[i as usize] += a * v[i as usize];
                    }
                }
            });

            unsafe {
                if !quantized {
                    matmul(&mut s.xb2, &s.xb3, &w.wo.assume_init()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize]);
                } else {
                    let sxq1 = &mut *s.xq1.as_mut_ptr();
                    
                    quantize(sxq1, &s.xb3, att_dim as usize, gs);
                    qmatmul(&mut s.xb2, sxq1, &w.wo_quant.assume_init()[l as usize], att_dim as usize, gs as usize)
                }
            }
            
            rmsnorm(&mut s.xb, &s.xb2, &w.w_rms_post_att[(l*dim) as usize..(l*dim + dim) as usize], dim as usize);
        
            for i in 0..dim {
                x[i as usize] += s.xb[i as usize];
            }
            
            rmsnorm(&mut s.xb, &x, &w.w_rms_pre_ffn[(l*dim) as usize..(l*dim + dim) as usize], dim as usize);

            // GeGLU is w2(GELU(w1(x)) * w3(x)) 
            // w1 -> gate_proj weights
            // w2 -> down_proj weights
            // w3 -> up_proj weights
            // GELU using tanh as the approximation

            unsafe {
                if !quantized {
                    matmul(&mut s.hb, &s.xb, &w.w1.assume_init()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize]);
                    matmul(&mut s.hb2, &s.xb, &w.w3.assume_init()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize]);
                } else {
                    let sxq = &mut *s.xq.as_mut_ptr();
                    
                    quantize(sxq, &s.xb, dim as usize, gs);
                    qmatmul(&mut s.hb, sxq, &w.w1_quant.assume_init()[l as usize], dim as usize, gs as usize);
                    qmatmul(&mut s.hb2, sxq, &w.w3_quant.assume_init()[l as usize], dim as usize, gs as usize);
                }
            }
            
            for i in 0..hidden_dim {
                let mut val = s.hb[i as usize];
                
                val *= 0.5 * (1.0 + ((0.7978845608028654 * (val + 0.044715 * val * val * val) as f64).tanh()) as f32);
                

                val *= s.hb2[i as usize];
                
                s.hb[i as usize] = val;
            }

            unsafe {
                if !quantized {
                    matmul(&mut s.xb, &s.hb, &w.w2.assume_init()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize]);
                } else {
                    let shq = &mut *s.hq.as_mut_ptr();

                    quantize(shq, &s.hb, hidden_dim as usize, gs);
                    qmatmul(&mut s.xb, shq, &w.w2_quant.assume_init()[l as usize], hidden_dim as usize, gs as usize);
                }
            }

            rmsnorm(&mut s.xb2, &s.xb, &w.w_rms_post_ffn[(l*dim) as usize..(l*dim + dim) as usize], dim as usize);
            
            for i in 0..dim {
                x[i as usize] += s.xb2[i as usize];
            }    
        }

        s.xb.copy_from_slice(x);

        rmsnorm(x, &s.xb, &w.w_rms_final, dim as usize);
        
        unsafe {
            if !quantized {
                matmul(&mut s.logits, &x, &w.w_cls.assume_init());
            } else {
                let sxq = &mut *s.xq.as_mut_ptr();
                
                quantize(sxq, &x, dim as usize, gs);
                qmatmul(&mut s.logits, sxq, &w.w_cls_quant.assume_init()[0], dim as usize, gs as usize);
            }
        }

        for d in 0..dim {
            s.logits[d as usize] /= 30.0;
            s.logits[d as usize] = (s.logits[d as usize] as f64).tanh() as f32;
            s.logits[d as usize] *= 30.0;
        }
        
        return &mut s.logits;
    }
}

// Deallocate fields created with Box::leak
impl<'a> Drop for Transformer<'a> {
    fn drop(&mut self) {
        if self.args.q_type != QuantType::None {
            unsafe {
                // Weights
                dealloc(self.weights.token_embedding_table.as_ptr() as *mut u8, Layout::array::<f32>(self.weights.token_embedding_table.len()).unwrap());
                
                let layer_weights_layout = Layout::array::<QuantizedTensor>(self.args.n_layers as usize).unwrap();
                dealloc(self.weights.wq_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.wk_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.wv_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.wo_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.w1_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.w2_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.w3_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.w_cls_quant.assume_init().as_ptr() as *mut u8, Layout::array::<QuantizedTensor>(self.weights.w_cls_quant.assume_init().len()).unwrap());

                // State
                let sxq = &mut *self.state.xq.as_mut_ptr();
                dealloc(sxq.q.as_ptr() as *mut u8, Layout::array::<i8>(sxq.q.len()).unwrap());
                dealloc(sxq.s.as_ptr() as *mut u8, Layout::array::<f32>(sxq.s.len()).unwrap());
                
                let sxq1 = &mut *self.state.xq1.as_mut_ptr();
                dealloc(sxq1.q.as_ptr() as *mut u8, Layout::array::<i8>(sxq1.q.len()).unwrap());
                dealloc(sxq1.s.as_ptr() as *mut u8, Layout::array::<f32>(sxq1.s.len()).unwrap());
                
                let shq = &mut *self.state.hq.as_mut_ptr();
                dealloc(shq.q.as_ptr() as *mut u8, Layout::array::<i8>(shq.q.len()).unwrap());
                dealloc(shq.s.as_ptr() as *mut u8, Layout::array::<f32>(shq.s.len()).unwrap());
            }
        }
    }
}
