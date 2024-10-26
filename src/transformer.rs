use crate::functional::slice_to_u32;
use crate::functional::u8_to_f32_slice;
use crate::functional::u8_to_i8_slice;
use crate::functional::rmsnorm;
use crate::functional::matmul;
use crate::functional::matmul_q4;
use crate::functional::matmul_q8;
use crate::functional::softmax;
use crate::quantization::*;

use memmap2::Mmap;
use rayon::prelude::*;
use std::mem::size_of;

pub fn init_param<'a>(data: &'a [u8], offset: &mut usize, n: u32, size_each: u32) -> Tensor<'a> {
    let ptr: &[f32]  = u8_to_f32_slice(&data[*offset..(*offset + ((n * size_each) as usize * size_of::<f32>()))]);

    *offset += (n * size_each) as usize * size_of::<f32>();

    Tensor::FloatSlice(ptr)
}

pub fn init_param_quant<'a>(data: &'a [u8], offset: &mut usize, n: u32, size_each: u32, gs: u32, q_type: QuantType) -> Tensor<'a> {
    let mut res: Vec<QuantizedTensor> = Vec::with_capacity(n as usize);
    let groups = (size_each / gs) as usize;
    let mut size = size_each;
    
    if q_type == QuantType::Q4_0 {
        size /= 2;
    }

    for _ in 0..n {
        let mut qt = QuantizedTensor { q: &mut [], s: &mut [] };

        qt.q = u8_to_i8_slice(&data[*offset..(*offset + (size as usize * size_of::<i8>()))]);
        
        *offset += size as usize * size_of::<i8>() ;

        qt.s = u8_to_f32_slice(&data[*offset..(*offset + (groups * size_of::<f32>()))]);
        
        *offset += groups * size_of::<f32>();

        res.push(qt);
    }

    Tensor::Quantized(res)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ModelType {
    GEMMA,
    LLAMA,
    PHI
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct TransformerArgs {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    head_size: u32,
    n_kv_heads: u32,
    pub vocab_size: u32,
    seq_len: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    q_type: QuantType,
    pub model_type: ModelType,
    group_size: u32,
    pub multimodal: bool,
}

pub struct TransformerWeights<'a> {
    token_embedding: Tensor<'a>,
    token_embedding_quant: Option<Tensor<'a>>,

    // Attention

    wq: Tensor<'a>,
    wk: Tensor<'a>,
    wv: Tensor<'a>,
    wo: Tensor<'a>,

    w_rms_att: Tensor<'a>,

    // FFN
    w1: Tensor<'a>,
    w2: Tensor<'a>,
    w3: Tensor<'a>,

    w_rms_post_att: Tensor<'a>,

    w_rms_pre_ffn: Option<Tensor<'a>>,
    w_rms_post_ffn: Option<Tensor<'a>>,

    w_rms_final: Tensor<'a>,

    lm_head: Option<Tensor<'a>>
}

pub struct TransformerState
{
    xb: Vec<f32>,
    xq: Option<MutableQuantizedTensor>,
    logits: Vec<f32>, 

    // kv cache
    key_cache: Vec<f32>,
    value_cache: Vec<f32>, 
}

pub struct Transformer<'a> {
    pub args: TransformerArgs,
    weights: TransformerWeights<'a>,
    state: TransformerState,
}

impl<'a> Transformer<'a> {
    pub fn new(data: &'a Mmap) -> (Transformer<'a>, usize) {
        assert_eq!(data[0..4], [0x6c, 0x6d, 0x72, 0x73], "Model not in lm.rs format.");

        let lmrs_version = slice_to_u32(&data[4..8]);

        println!("LMRS version: {}", lmrs_version);
        
        let (head, body, _) = unsafe { data[8..55].align_to::<TransformerArgs>() };

        assert!(head.is_empty(), "Data was not aligned");
        
        let mut cfg = body[0];

        println!("Model type: {:?}\n", cfg.model_type);

        let head_size = cfg.head_size;
        
        let mut offset: usize = 256;

        let quantized = cfg.q_type != QuantType::None;
        
        if quantized { println!("Using {:?} quantization.", cfg.q_type) };
        
        // For now this will do so we don't run out of memory
        if cfg.seq_len > 8192 {
            cfg.seq_len = 8192;
        }

        let kv_dim = cfg.head_size * cfg.n_kv_heads;

        let mut rms_pre_ffn = None;
        let mut rms_post_ffn = None;
        let mut lm_head = None;

        if !quantized {
            let token_embedding = init_param(data, &mut offset, 1, cfg.vocab_size * cfg.dim);
            let rms_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            let wq = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size);
            let wk = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size);
            let wv = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size);
            let wo = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size);
            let rms_post_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);

            if cfg.model_type == ModelType::GEMMA {
                rms_pre_ffn = Some(init_param(data, &mut offset, cfg.n_layers, cfg.dim));
            }

            let w1 = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
            let w2 = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
            let w3 = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
            
            if cfg.model_type == ModelType::GEMMA {
                rms_post_ffn = Some(init_param(data, &mut offset, cfg.n_layers, cfg.dim));
            }

            let rms_final = init_param(data, &mut offset, 1, cfg.dim);

            if cfg.model_type == ModelType::PHI {
                lm_head = Some(init_param(data, &mut offset, 1, cfg.dim * cfg.vocab_size));
            }
            
            let weights = TransformerWeights {
                token_embedding,
                token_embedding_quant: None,
                wq,
                wk,
                wv,
                wo,
                w_rms_att: rms_att,
                w1,
                w2,
                w3,
                w_rms_post_att: rms_post_att,
                w_rms_pre_ffn: rms_pre_ffn,
                w_rms_post_ffn: rms_post_ffn,
                w_rms_final: rms_final,
                lm_head,
            };

            let state = TransformerState {
                xb: vec![0.0; cfg.dim as usize],
                xq: None,
                key_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
                value_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
                logits: vec![0.0; cfg.vocab_size as usize],
            };
            
            return (Transformer {
                args: cfg,
                weights,
                state,
            }, offset)
        } 

        println!("Loading weights...");

        let token_embedding_quant = init_param_quant(data, &mut offset, 1, cfg.vocab_size * cfg.dim, cfg.group_size, cfg.q_type);

        let mut emb_tab: Vec<f32> = vec![0.0; (cfg.vocab_size * cfg.dim) as usize];

        dequantize(&token_embedding_quant.as_quantized()[0], &mut emb_tab, (cfg.vocab_size * cfg.dim) as usize, cfg.group_size, cfg.q_type);

        let rms_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let wq = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wk = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size, cfg.group_size, cfg.q_type);
        let wv = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_kv_heads * head_size, cfg.group_size, cfg.q_type);
        let wo = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let rms_post_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);

        if cfg.model_type == ModelType::GEMMA {
            rms_pre_ffn = Some(init_param(data, &mut offset, cfg.n_layers, cfg.dim));
        }

        let w1 = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size, cfg.q_type);
        let w2 = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size, cfg.q_type);
        let w3 = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size, cfg.q_type);

        if cfg.model_type == ModelType::GEMMA {
            rms_post_ffn = Some(init_param(data, &mut offset, cfg.n_layers, cfg.dim));
        }

        let rms_final = init_param(data, &mut offset, 1, cfg.dim); 
        
        if cfg.model_type == ModelType::PHI {
            lm_head = Some(init_param_quant(data, &mut offset, 1, cfg.dim * cfg.vocab_size, cfg.group_size, cfg.q_type));
        }
        
        let weights = TransformerWeights {
            token_embedding: Tensor::FloatVec(emb_tab),
            token_embedding_quant: Some(token_embedding_quant),
            wq,
            wk,
            wv,
            wo,
            w_rms_att: rms_att,
            w1,
            w2,
            w3,
            w_rms_post_att: rms_post_att,
            w_rms_pre_ffn: rms_pre_ffn,
            w_rms_post_ffn: rms_post_ffn,
            w_rms_final: rms_final,
            lm_head,
        };

        let state = TransformerState {
            xb: vec![0.0; cfg.dim as usize],
            xq: Some(MutableQuantizedTensor { q: vec![0; (cfg.dim) as usize], s: vec![0.0; (cfg.dim) as usize]}),
            key_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            value_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            logits: vec![0.0; cfg.vocab_size as usize],
        };
        
        println!("Done.\n");
        
        (Transformer {
            args: cfg,
            weights,
            state,
        }, offset)
    }

    pub fn forward(&mut self, token: u32, pos: u32) -> &mut [f32] {
        let p = self.args;
        let x = &mut vec![0.0; (p.dim) as usize];
        let dim = p.dim;
        let gs = p.group_size;

        let quantized = p.q_type != QuantType::None;

        x.copy_from_slice(&self.weights.token_embedding.as_float()[(token * dim) as usize..(token * dim + dim) as usize]);
        
        // Gemma normalizes the token embeddings by sqrt(dim)
        if p.model_type == ModelType::GEMMA {
            let normalizer = (dim as f32).sqrt();
            for i in x.iter_mut() {
                *i *= normalizer;
            }
        }

        for l in 0..p.n_layers {
            self.forward_layer(x, 1, l, pos);
        }

        let s = &mut self.state;
        let w = &self.weights;
        
        s.xb.copy_from_slice(x);

        rmsnorm(x, &s.xb, w.w_rms_final.as_float(), dim as usize, p.rms_norm_eps, p.model_type == ModelType::GEMMA);

        if !quantized {
            if p.model_type != ModelType::PHI {
                matmul(&mut s.logits, x, w.token_embedding.as_float(), dim as usize, p.vocab_size as usize);
            } else {
                matmul(&mut s.logits, x, w.lm_head.as_ref().unwrap().as_float(), dim as usize, p.vocab_size as usize);
            }
        } else {
            let sxq = &mut s.xq.as_mut().unwrap();
            
            if p.q_type == QuantType::Q8_0 {
                quantize(sxq, x, dim as usize, gs);
                
                if p.model_type != ModelType::PHI {
                    matmul_q8(&mut s.logits, sxq, &w.token_embedding_quant.as_ref().unwrap().as_quantized()[0], dim as usize, p.vocab_size as usize, gs as usize);
                } else {
                    matmul_q8(&mut s.logits, sxq, &w.lm_head.as_ref().unwrap().as_quantized()[0], dim as usize, p.vocab_size as usize, gs as usize);
                }
            } else if p.q_type == QuantType::Q4_0 {
                quantize_q4(sxq, x, dim as usize, gs);

                if p.model_type != ModelType::PHI {
                    matmul_q4(&mut s.logits, sxq, &w.token_embedding_quant.as_ref().unwrap().as_quantized()[0], dim as usize, p.vocab_size as usize, gs as usize);
                } else {
                    matmul_q4(&mut s.logits, sxq, &w.lm_head.as_ref().unwrap().as_quantized()[0], dim as usize, p.vocab_size as usize, gs as usize);
                }
            }
        }

        if p.model_type == ModelType::GEMMA {
            for d in 0..dim {
                s.logits[d as usize] /= 30.0;
                s.logits[d as usize] = (s.logits[d as usize] as f64).tanh() as f32;
                s.logits[d as usize] *= 30.0;
            }
        }
        
        &mut s.logits
    }

    // x -> (T, D)
    // sl -> T
    fn forward_layer(&mut self, x: &mut [f32], sl: u32, l: u32, pos: u32) {
        let p = self.args;
        let w = &self.weights;
        let s = &mut self.state;
        let dim = p.dim;
        let head_size = p.head_size;
        let att_dim = p.n_heads * head_size;
        let kv_dim = head_size * p.n_kv_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let hidden_dim = p.hidden_dim;
        let gs = p.group_size;

        let quantized = p.q_type != QuantType::None;
        let total_shape: usize = (sl*p.dim) as usize;
        let total_shape_hidden: usize = (sl*p.hidden_dim) as usize;

        let mut embeddings = x.to_vec();
        let mut temp_embeddings = vec![0.0; total_shape];
        let mut hidden_embeddings = vec![0.0; total_shape_hidden];
        let mut temp_hidden_embeddings = vec![0.0; total_shape_hidden];

        embeddings.par_chunks_mut(dim as usize).enumerate().for_each( |(i, xb)| {
            rmsnorm(xb, &x[i*dim as usize..i*dim as usize + dim as usize], &w.w_rms_att.as_float()[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.rms_norm_eps, p.model_type == ModelType::GEMMA);
        });
        
        let loff = l * p.seq_len * kv_dim; 
        let mut sq = vec![0.0; (att_dim * sl) as usize];
        let k = &mut s.key_cache[(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim*sl) as usize];
        let v = &mut s.value_cache[(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim*sl) as usize];


        if !quantized {
            matmul(&mut sq, &embeddings, &w.wq.as_float()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize], dim as usize, att_dim as usize);
            matmul(k, &embeddings, &w.wk.as_float()[(l*dim*kv_dim) as usize..(l*dim*kv_dim + dim*kv_dim) as usize], dim as usize, kv_dim as usize);
            matmul(v, &embeddings, &w.wv.as_float()[(l*dim*kv_dim) as usize..(l*dim*kv_dim + dim*kv_dim) as usize], dim as usize, kv_dim as usize);
        } else {
            let sxq = &mut MutableQuantizedTensor { q: vec![0; total_shape], s: vec![0.0; total_shape]};
            
            if p.q_type == QuantType::Q8_0 {
                quantize(sxq, &embeddings, total_shape, gs);
                
                matmul_q8(&mut sq, sxq, &w.wq.as_quantized()[l as usize], dim as usize, att_dim as usize, gs as usize);
                matmul_q8(k, sxq, &w.wk.as_quantized()[l as usize], dim as usize, kv_dim as usize, gs as usize);
                matmul_q8(v, sxq, &w.wv.as_quantized()[l as usize], dim as usize, kv_dim as usize, gs as usize);
            } else if p.q_type == QuantType::Q4_0 {
                quantize_q4(sxq, &embeddings, total_shape, gs);
                
                matmul_q4(&mut sq, sxq, &w.wq.as_quantized()[l as usize], dim as usize, att_dim as usize, gs as usize);
                matmul_q4(k, sxq, &w.wk.as_quantized()[l as usize], dim as usize, kv_dim as usize, gs as usize);
                matmul_q4(v, sxq, &w.wv.as_quantized()[l as usize], dim as usize, kv_dim as usize, gs as usize);
            }
        }
         
        // RoPE
        k.par_chunks_mut(kv_dim as usize).zip(sq.par_chunks_mut(att_dim as usize)).enumerate().for_each( |(idx, (tk, tq))| {
            for i in 0..p.n_heads {
                for j in 0..(head_size/2) {
                    let head_dim: u32 = j * 2;
                    let mut freq: f32 = 1.0 / p.rope_theta.powf(head_dim as f32/head_size as f32);

                    let mut scaling_factor = 1.0;

                    if p.model_type == ModelType::LLAMA {
                        let wavelen = (2.0 * std::f32::consts::PI) / freq;
                        
                        // Should be on args
                        let factor = 32.0;
                        let low_freq_factor = 1.0;
                        let high_freq_factor = 4.0;
                        let old_context_len = 8192.0;

                        let low_freq_wavelen = old_context_len / low_freq_factor;
                        let high_freq_wavelen = old_context_len / high_freq_factor;

                        if wavelen > low_freq_wavelen {
                            freq /= factor;
                        } else if wavelen <= low_freq_wavelen && wavelen >= high_freq_wavelen {
                            let smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                            
                            freq = (1.0 - smooth_factor) * freq / factor + smooth_factor * freq
                        }
                    }

                    if p.model_type == ModelType::PHI {
                        let short_factor: [f64; 48] = [1.08, 1.1, 1.1300000000000001, 1.2800000000000002, 1.3100000000000003, 1.4500000000000004, 1.4500000000000004, 1.9500000000000008, 2.030000000000001, 2.4299999999999926, 2.5699999999999896, 2.9499999999999815, 3.729999999999965, 3.869999999999962, 4.189999999999955, 4.43999999999995, 4.6399999999999455, 4.979999999999938, 5.159999999999934, 5.279999999999932, 5.759999999999922, 5.889999999999919, 5.889999999999919, 5.969999999999917, 6.089999999999915, 6.2799999999999105, 6.7699999999999, 6.8899999999998975, 7.109999999999893, 7.129999999999892, 7.179999999999891, 7.289999999999889, 7.339999999999888, 7.559999999999883, 7.619999999999882, 7.69999999999988, 7.879999999999876, 7.879999999999876, 7.879999999999876, 7.939999999999875, 7.949999999999875, 7.979999999999874, 8.19999999999987, 8.439999999999864, 8.469999999999864, 8.589999999999861, 8.809999999999857, 8.999999999999853];

                        freq *= (1.0/short_factor[j as usize]) as f32;
                        let scale = 131072f32/4096f32;
                        scaling_factor = (1.0 + scale.ln() / 4096f32.ln()).sqrt();
                    }

                    let val: f32 = (pos + idx as u32) as f32 * freq;
                    let fcr = val.cos() * scaling_factor;
                    let fci = val.sin() * scaling_factor;
                    let rotn: u32 = if (i*head_size) + j + head_size/2 < kv_dim {2} else {1};

                    for v in 0..rotn{
                        let vec: &mut [f32] = if v == 0 {tq} else {tk};
                        let v0: f32 = vec[((i*head_size) + j) as usize];
                        let v1: f32 = vec[(((i*head_size) + j)+(head_size/2)) as usize];
                        
                        vec[((i*head_size) + j) as usize] = v0 * fcr - v1 * fci;
                        vec[(((i*head_size) + j)+(head_size/2)) as usize]= v0 * fci + v1 * fcr;
                    }
                }
            }
        });

        if att_dim > dim {
            embeddings.extend(vec![0.0; ((att_dim - dim)*sl) as usize]);
        }
        
        embeddings.par_chunks_exact_mut(att_dim as usize).enumerate().for_each( |(i, elem)| {
            elem.par_chunks_mut(head_size as usize).enumerate().for_each( |(h, xb)| {
                let q = &sq[(i as u32 * att_dim + h as u32 * head_size) as usize..(i as u32 * att_dim + h as u32 * head_size + head_size) as usize];

                let att = &mut vec![0.0; p.seq_len as usize];

                for t in 0..pos + i as u32 + 1 {
                    let k = &s.key_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size) as usize..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size) as usize];
                    
                    let mut score: f32 = 0.0;

                    for idx in 0..head_size {
                        score += q[idx as usize] * k[idx as usize];
                    }
                    
                    score /= (head_size as f32).sqrt();
                    
                    if p.model_type == ModelType::GEMMA {
                        // Softcapping
                        score /= 50.0f32;
                        score = (score as f64).tanh() as f32;
                        score *= 50.0f32;
                        
                        // Local attention
                        score += if pos - t <= 4096 {0.0} else {-2.3819763e38};
                    }

                    att[t as usize] = score;
                }

                softmax(&mut att[..(pos + i as u32 + 1) as usize]);

                xb.fill(0.0);

                for t in 0..pos + i as u32 + 1 {
                    let v = &s.value_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size) as usize..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size) as usize];
                    let a = att[t as usize];

                    for idx in 0..head_size {
                        xb[idx as usize] += a * v[idx as usize];
                    }
                }
            });
        });
        
        if !quantized {
            matmul(&mut temp_embeddings, &embeddings, &w.wo.as_float()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize], att_dim as usize, dim as usize);
        } else {
            let sxq1 = &mut MutableQuantizedTensor { q: vec![0; (att_dim * sl) as usize], s: vec![0.0; (att_dim * sl) as usize]};
            
            if p.q_type == QuantType::Q8_0 {
                quantize(sxq1, &embeddings, (att_dim*sl) as usize, gs);
                matmul_q8(&mut temp_embeddings, sxq1, &w.wo.as_quantized()[l as usize], att_dim as usize, dim as usize, gs as usize)
            } else {
                quantize_q4(sxq1, &embeddings, (att_dim*sl) as usize, gs);
                matmul_q4(&mut temp_embeddings, sxq1, &w.wo.as_quantized()[l as usize], att_dim as usize, dim as usize, gs as usize)
            }
        }
        
        x.par_chunks_exact_mut(dim as usize).zip(embeddings.par_chunks_exact_mut(dim as usize)).zip(temp_embeddings.par_chunks_exact(dim as usize)).for_each( |((xelem, emb), temb)| {
            if p.model_type == ModelType::GEMMA {
                rmsnorm(emb, temb, &w.w_rms_post_att.as_float()[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.rms_norm_eps, p.model_type == ModelType::GEMMA);
            
                for i in 0..dim {
                    xelem[i as usize] += emb[i as usize];
                }
                
                rmsnorm(emb, xelem, &w.w_rms_pre_ffn.as_ref().unwrap().as_float()[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.rms_norm_eps, true);
            } else {
                for i in 0..dim {
                    xelem[i as usize] += temb[i as usize];
                }
                
                rmsnorm(emb, xelem, &w.w_rms_post_att.as_float()[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.rms_norm_eps, p.model_type == ModelType::GEMMA);
            }
        });
         
        // GeGLU is w2(GELU(w1(x)) * w3(x)) 
        // w1 -> gate_proj weights
        // w2 -> down_proj weights
        // w3 -> up_proj weights
        // GELU using tanh as the approximation

        if !quantized {
            matmul(&mut hidden_embeddings, &embeddings, &w.w1.as_float()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize], dim as usize, hidden_dim as usize);
            matmul(&mut temp_hidden_embeddings, &embeddings, &w.w3.as_float()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize], dim as usize, hidden_dim as usize);
        } else {
            let sxq = &mut MutableQuantizedTensor { q: vec![0; (dim * sl) as usize], s: vec![0.0; (dim * sl) as usize]};
            
            if p.q_type == QuantType::Q8_0 {
                quantize(sxq, &embeddings, total_shape, gs);
                matmul_q8(&mut hidden_embeddings, sxq, &w.w1.as_quantized()[l as usize], dim as usize, hidden_dim as usize, gs as usize);
                matmul_q8(&mut temp_hidden_embeddings, sxq, &w.w3.as_quantized()[l as usize], dim as usize, hidden_dim as usize, gs as usize);
            } else if p.q_type == QuantType::Q4_0{
                quantize_q4(sxq, &embeddings, total_shape, gs);
                matmul_q4(&mut hidden_embeddings, sxq, &w.w1.as_quantized()[l as usize], dim as usize, hidden_dim as usize, gs as usize);
                matmul_q4(&mut temp_hidden_embeddings, sxq, &w.w3.as_quantized()[l as usize], dim as usize, hidden_dim as usize, gs as usize);
            }
        }

        hidden_embeddings.par_chunks_exact_mut(hidden_dim as usize).zip(temp_hidden_embeddings.par_chunks_exact(hidden_dim as usize)).for_each( |(hb, hb2)| {
            for i in 0..hidden_dim {
                let mut val = hb[i as usize];

                // Best case we would have the activation in the args, but for now this will do 
                if p.model_type == ModelType::GEMMA {
                    // GELU
                    val *= 0.5 * (1.0 + ((0.7978845608028654 * (val + 0.044715 * val * val * val) as f64).tanh()) as f32);   
                } else {
                    // SiLU
                    val *= 1.0 / (1.0 + (-val).exp());
                }

                val *= hb2[i as usize];
                
                hb[i as usize] = val;
            }
        });

        if !quantized {
            matmul(&mut embeddings, &hidden_embeddings, &w.w2.as_float()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize], hidden_dim as usize, dim as usize);
        } else {
            let shq = &mut MutableQuantizedTensor { q: vec![0; (hidden_dim * sl) as usize], s: vec![0.0; (hidden_dim * sl) as usize]};

            if p.q_type == QuantType::Q8_0 {
                quantize(shq, &hidden_embeddings, total_shape_hidden, gs);
                matmul_q8(&mut embeddings, shq, &w.w2.as_quantized()[l as usize], hidden_dim as usize, dim as usize, gs as usize);
            } else if p.q_type == QuantType::Q4_0 {
                quantize_q4(shq, &hidden_embeddings, total_shape_hidden, gs);
                matmul_q4(&mut embeddings, shq, &w.w2.as_quantized()[l as usize], hidden_dim as usize, dim as usize, gs as usize);
            }
        }

        x.par_chunks_exact_mut(dim as usize).zip(embeddings.par_chunks_exact(dim as usize)).zip(temp_embeddings.par_chunks_exact_mut(dim as usize)).for_each(| ((xelem, emb), temb) | {
            if p.model_type == ModelType::GEMMA {
                rmsnorm(temb, emb, &w.w_rms_post_ffn.as_ref().unwrap().as_float()[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.rms_norm_eps, true);
                
                for i in 0..dim {
                    xelem[i as usize] += temb[i as usize];
                }
            } else {
                for i in 0..dim {
                    xelem[i as usize] += emb[i as usize];
                }
            }
        });
    }

    pub fn get_embeddings(&self, tokens: &[u32]) -> Vec<f32> {
        let n_tokens = tokens.len();
        let dim = self.args.dim;
        let mut out_embeddings: Vec<f32> = Vec::with_capacity(dim as usize * n_tokens);

        for t in tokens.iter().take(n_tokens) {
            out_embeddings.extend(&self.weights.token_embedding.as_float()[(t * dim) as usize..(t * dim + dim) as usize]);
        }

        out_embeddings
    }

    pub fn fill_kv_cache(&mut self, embeddings: &mut [f32], curr_pos: u32) -> u32 {
        let p = self.args;
        let num_embeddings = embeddings.len() as u32/self.args.dim;    
        let mut pos = curr_pos;

        for l in 0..p.n_layers {
            self.forward_layer(embeddings, num_embeddings, l, pos)
        }

        pos += num_embeddings;

        pos
    }
}