use crate::functional::matmul;
use crate::functional::matmul_q4;
use crate::functional::matmul_q8;
use crate::functional::rmsnorm;
use crate::functional::slice_to_u32;
use crate::functional::softmax;
use crate::functional::u8_to_f32_slice;
use crate::functional::u8_to_i8_slice;

use crate::quantization::*;

use memmap2::Mmap;
use rayon::prelude::*;
use std::alloc::{dealloc, Layout};
use std::mem::size_of;

fn init_param<'a>(data: &'a [u8], offset: &mut usize, n: u32, size_each: u32) -> &'a [f32] {
    let ptr: &[f32] =
        u8_to_f32_slice(&data[*offset..(*offset + ((n * size_each) as usize * size_of::<f32>()))]);

    *offset += (n * size_each) as usize * size_of::<f32>();

    ptr
}

fn init_param_quant<'a>(
    data: &'a [u8],
    offset: &mut usize,
    n: u32,
    size_each: u32,
    gs: u32,
    q_type: QuantType,
) -> &'a [QuantizedTensor<'a>] {
    let mut res: Vec<QuantizedTensor> = Vec::with_capacity(n as usize);
    let groups = (size_each / gs) as usize;
    let mut size = size_each;

    if q_type == QuantType::Q4_0 {
        size /= 2;
    }

    for _ in 0..n {
        let mut qt = QuantizedTensor {
            q: &mut [],
            s: &mut [],
        };

        qt.q = u8_to_i8_slice(&data[*offset..(*offset + (size as usize * size_of::<i8>()))]);

        *offset += size as usize * size_of::<i8>();

        qt.s = u8_to_f32_slice(&data[*offset..(*offset + (groups * size_of::<f32>()))]);

        *offset += groups * size_of::<f32>();

        res.push(qt);
    }

    Box::leak(res.into_boxed_slice())
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ModelType {
    GEMMA,
    LLAMA,
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
}

pub struct TransformerWeights<'a> {
    token_embedding_table: &'a [f32],

    // Attention
    wq: Option<&'a [f32]>,
    wk: Option<&'a [f32]>,
    wv: Option<&'a [f32]>,
    wo: Option<&'a [f32]>,

    wq_quant: Option<&'a [QuantizedTensor<'a>]>,
    wk_quant: Option<&'a [QuantizedTensor<'a>]>,
    wv_quant: Option<&'a [QuantizedTensor<'a>]>,
    wo_quant: Option<&'a [QuantizedTensor<'a>]>,

    w_rms_att: &'a [f32],

    // FFN
    w1: Option<&'a [f32]>,
    w2: Option<&'a [f32]>,
    w3: Option<&'a [f32]>,

    w1_quant: Option<&'a [QuantizedTensor<'a>]>,
    w2_quant: Option<&'a [QuantizedTensor<'a>]>,
    w3_quant: Option<&'a [QuantizedTensor<'a>]>,

    w_rms_post_att: &'a [f32],

    w_rms_pre_ffn: Option<&'a [f32]>,
    w_rms_post_ffn: Option<&'a [f32]>,

    w_rms_final: &'a [f32],

    w_cls: Option<&'a [f32]>,
    w_cls_quant: Option<&'a [QuantizedTensor<'a>]>,
}

pub struct TransformerState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    xb3: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    xq: Option<MutableQuantizedTensor>,
    xq1: Option<MutableQuantizedTensor>,
    hq: Option<MutableQuantizedTensor>,
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
    pub fn new(data: &'a Mmap) -> Transformer<'a> {
        assert_eq!(
            data[0..4],
            [0x6c, 0x6d, 0x72, 0x73],
            "Model not in lm.rs format."
        );

        let lmrs_version = slice_to_u32(&data[4..8]);

        println!("LMRS version: {}", lmrs_version);

        let (head, body, _) = unsafe { data[8..54].align_to::<TransformerArgs>() };

        assert!(head.is_empty(), "Data was not aligned");

        let cfg = &body[0];

        println!("Model type: {:?}\n", cfg.model_type);

        let head_size = cfg.head_size;

        let mut offset: usize = 256;

        let quantized = cfg.q_type != QuantType::None;

        if quantized {
            println!("Using {:?} quantization.", cfg.q_type)
        };

        let kv_dim = cfg.head_size * cfg.n_kv_heads;

        let mut rms_pre_ffn = None;
        let mut rms_post_ffn = None;

        if !quantized {
            let emb_tab = init_param(data, &mut offset, 1, cfg.vocab_size * cfg.dim);
            let rms_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            let wq = init_param(
                data,
                &mut offset,
                cfg.n_layers,
                cfg.dim * cfg.n_heads * head_size,
            );
            let wk = init_param(
                data,
                &mut offset,
                cfg.n_layers,
                cfg.dim * cfg.n_kv_heads * head_size,
            );
            let wv = init_param(
                data,
                &mut offset,
                cfg.n_layers,
                cfg.dim * cfg.n_kv_heads * head_size,
            );
            let wo = init_param(
                data,
                &mut offset,
                cfg.n_layers,
                cfg.dim * cfg.n_heads * head_size,
            );
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

            let weights = TransformerWeights {
                token_embedding_table: emb_tab,
                wq: Some(wq),
                wk: Some(wk),
                wv: Some(wv),
                wo: Some(wo),
                wq_quant: None,
                wk_quant: None,
                wv_quant: None,
                wo_quant: None,
                w_rms_att: rms_att,
                w1: Some(w1),
                w2: Some(w2),
                w3: Some(w3),
                w1_quant: None,
                w2_quant: None,
                w3_quant: None,
                w_rms_post_att: rms_post_att,
                w_rms_pre_ffn: rms_pre_ffn,
                w_rms_post_ffn: rms_post_ffn,
                w_rms_final: rms_final,
                w_cls: Some(emb_tab),
                w_cls_quant: None,
            };

            let state = TransformerState {
                x: vec![0.0; cfg.dim as usize],
                xb: vec![0.0; cfg.dim as usize],
                xb2: vec![0.0; cfg.dim as usize],
                xb3: vec![0.0; (cfg.head_size * cfg.n_heads) as usize],
                hb: vec![0.0; cfg.hidden_dim as usize],
                hb2: vec![0.0; cfg.hidden_dim as usize],
                q: vec![0.0; (cfg.head_size * cfg.n_heads) as usize],
                xq: None,
                xq1: None,
                hq: None,
                key_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
                value_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
                logits: vec![0.0; cfg.vocab_size as usize],
            };

            return Transformer {
                args: *cfg,
                weights,
                state,
            };
        }

        println!("Loading weights...");

        let emb_tab_quant = init_param_quant(
            data,
            &mut offset,
            1,
            cfg.vocab_size * cfg.dim,
            cfg.group_size,
            cfg.q_type,
        );

        let mut emb_tab: Vec<f32> = vec![0.0; (cfg.vocab_size * cfg.dim) as usize];

        dequantize(
            &emb_tab_quant[0],
            &mut emb_tab,
            (cfg.vocab_size * cfg.dim) as usize,
            cfg.group_size,
            cfg.q_type,
        );

        let rms_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let wq_quant = init_param_quant(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_heads * head_size,
            cfg.group_size,
            cfg.q_type,
        );
        let wk_quant = init_param_quant(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_kv_heads * head_size,
            cfg.group_size,
            cfg.q_type,
        );
        let wv_quant = init_param_quant(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_kv_heads * head_size,
            cfg.group_size,
            cfg.q_type,
        );
        let wo_quant = init_param_quant(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.n_heads * head_size,
            cfg.group_size,
            cfg.q_type,
        );
        let rms_post_att = init_param(data, &mut offset, cfg.n_layers, cfg.dim);

        if cfg.model_type == ModelType::GEMMA {
            rms_pre_ffn = Some(init_param(data, &mut offset, cfg.n_layers, cfg.dim));
        }

        let w1_quant = init_param_quant(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.hidden_dim,
            cfg.group_size,
            cfg.q_type,
        );
        let w2_quant = init_param_quant(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.hidden_dim,
            cfg.group_size,
            cfg.q_type,
        );
        let w3_quant = init_param_quant(
            data,
            &mut offset,
            cfg.n_layers,
            cfg.dim * cfg.hidden_dim,
            cfg.group_size,
            cfg.q_type,
        );

        if cfg.model_type == ModelType::GEMMA {
            rms_post_ffn = Some(init_param(data, &mut offset, cfg.n_layers, cfg.dim));
        }

        let rms_final = init_param(data, &mut offset, 1, cfg.dim);

        let weights = TransformerWeights {
            token_embedding_table: Box::leak(emb_tab.into_boxed_slice()),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            wq_quant: Some(wq_quant),
            wk_quant: Some(wk_quant),
            wv_quant: Some(wv_quant),
            wo_quant: Some(wo_quant),
            w_rms_att: rms_att,
            w1: None,
            w2: None,
            w3: None,
            w1_quant: Some(w1_quant),
            w2_quant: Some(w2_quant),
            w3_quant: Some(w3_quant),
            w_rms_post_att: rms_post_att,
            w_rms_pre_ffn: rms_pre_ffn,
            w_rms_post_ffn: rms_post_ffn,
            w_rms_final: rms_final,
            w_cls: None,
            w_cls_quant: Some(emb_tab_quant),
        };

        let state = TransformerState {
            x: vec![0.0; cfg.dim as usize],
            xb: vec![0.0; cfg.dim as usize],
            xb2: vec![0.0; cfg.dim as usize],
            xb3: vec![0.0; (cfg.head_size * cfg.n_heads) as usize],
            hb: vec![0.0; cfg.hidden_dim as usize],
            hb2: vec![0.0; cfg.hidden_dim as usize],
            q: vec![0.0; (cfg.head_size * cfg.n_heads) as usize],
            xq: Some(MutableQuantizedTensor {
                q: vec![0; (cfg.dim) as usize],
                s: vec![0.0; (cfg.dim) as usize],
            }),
            xq1: Some(MutableQuantizedTensor {
                q: vec![0; (cfg.head_size * cfg.n_heads) as usize],
                s: vec![0.0; (cfg.head_size * cfg.n_heads) as usize],
            }),
            hq: Some(MutableQuantizedTensor {
                q: vec![0; (cfg.hidden_dim) as usize],
                s: vec![0.0; (cfg.hidden_dim) as usize],
            }),
            key_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            value_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            logits: vec![0.0; cfg.vocab_size as usize],
        };

        println!("Done.\n");

        Transformer {
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

        x.copy_from_slice(
            &w.token_embedding_table[(token * dim) as usize..(token * dim + dim) as usize],
        );

        // Gemma normalizes the token embeddings by sqrt(dim)
        if p.model_type == ModelType::GEMMA {
            let normalizer = (dim as f32).sqrt();
            for i in x.iter_mut() {
                *i *= normalizer;
            }
        }

        for l in 0..p.n_layers {
            rmsnorm(
                &mut s.xb,
                x,
                &w.w_rms_att[(l * dim) as usize..(l * dim + dim) as usize],
                dim as usize,
                p.rms_norm_eps,
                p.model_type == ModelType::GEMMA,
            );

            let loff = l * p.seq_len * kv_dim;
            let k = &mut s.key_cache
                [(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim) as usize];
            let v = &mut s.value_cache
                [(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim) as usize];

            {
                if !quantized {
                    matmul(
                        &mut s.q,
                        &s.xb,
                        &w.wq.expect("Field not initialized")[(l * dim * att_dim) as usize
                            ..(l * dim * att_dim + dim * att_dim) as usize],
                    );
                    matmul(
                        k,
                        &s.xb,
                        &w.wk.expect("Field not initialized")[(l * dim * kv_dim) as usize
                            ..(l * dim * kv_dim + dim * kv_dim) as usize],
                    );
                    matmul(
                        v,
                        &s.xb,
                        &w.wv.expect("Field not initialized")[(l * dim * kv_dim) as usize
                            ..(l * dim * kv_dim + dim * kv_dim) as usize],
                    );
                } else {
                    let sxq = s.xq.as_mut().expect("Field not initialized");

                    if p.q_type == QuantType::Q8_0 {
                        quantize(sxq, &s.xb, dim as usize, gs);

                        matmul_q8(
                            &mut s.q,
                            sxq,
                            &w.wq_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                        matmul_q8(
                            k,
                            sxq,
                            &w.wk_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                        matmul_q8(
                            v,
                            sxq,
                            &w.wv_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                    } else if p.q_type == QuantType::Q4_0 {
                        quantize_q4(sxq, &s.xb, dim as usize, gs);

                        matmul_q4(
                            &mut s.q,
                            sxq,
                            &w.wq_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                        matmul_q4(
                            k,
                            sxq,
                            &w.wk_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                        matmul_q4(
                            v,
                            sxq,
                            &w.wv_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                    }
                }
            }

            for i in 0..p.n_heads {
                for j in 0..(head_size / 2) {
                    let head_dim: u32 = j * 2;
                    let mut freq: f32 = 1.0 / p.rope_theta.powf(head_dim as f32 / head_size as f32);

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
                            let smooth_factor = (old_context_len / wavelen - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);

                            freq = (1.0 - smooth_factor) * freq / factor + smooth_factor * freq
                        }
                    }

                    let val: f32 = pos as f32 * freq;
                    let fcr = val.cos();
                    let fci = val.sin();
                    let rotn: u32 = if (i * head_size) + j + head_size / 2 < kv_dim {
                        2
                    } else {
                        1
                    };

                    for v in 0..rotn {
                        let vec: &mut [f32] = if v == 0 { &mut s.q } else { k };
                        let v0: f32 = vec[((i * head_size) + j) as usize];
                        let v1: f32 = vec[(((i * head_size) + j) + (head_size / 2)) as usize];

                        vec[((i * head_size) + j) as usize] = v0 * fcr - v1 * fci;
                        vec[(((i * head_size) + j) + (head_size / 2)) as usize] =
                            v0 * fci + v1 * fcr;
                    }
                }
            }

            s.xb3
                .par_chunks_mut(head_size as usize)
                .enumerate()
                .for_each(|(h, xb)| {
                    let q = &s.q[(h as u32 * head_size) as usize
                        ..(h as u32 * head_size + head_size) as usize];

                    let att = &mut vec![0.0; p.seq_len as usize];

                    for t in 0..pos + 1 {
                        let k = &s.key_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size)
                            as usize
                            ..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size)
                                as usize];

                        let mut score: f32 = 0.0;

                        for i in 0..head_size {
                            score += q[i as usize] * k[i as usize];
                        }

                        score /= (head_size as f32).sqrt();

                        if p.model_type == ModelType::GEMMA {
                            // Softcapping
                            score /= 50.0f32;
                            score = (score as f64).tanh() as f32;
                            score *= 50.0f32;

                            // Local attention
                            score += if pos - t <= 4096 { 0.0 } else { -2.3819763e38 };
                        }

                        att[t as usize] = score;
                    }

                    softmax(&mut att[..(pos + 1) as usize]);

                    xb.fill(0.0);

                    for t in 0..pos + 1 {
                        let v = &s.value_cache[(loff + t * kv_dim + (h as u32 / kv_mul) * head_size)
                            as usize
                            ..(loff + t * kv_dim + (h as u32 / kv_mul) * head_size + head_size)
                                as usize];
                        let a = att[t as usize];

                        for i in 0..head_size {
                            xb[i as usize] += a * v[i as usize];
                        }
                    }
                });

            {
                if !quantized {
                    matmul(
                        &mut s.xb2,
                        &s.xb3,
                        &w.wo.expect("Field not initialized")[(l * dim * att_dim) as usize
                            ..(l * dim * att_dim + dim * att_dim) as usize],
                    );
                } else {
                    let sxq1 = s.xq1.as_mut().expect("Field not initialized");

                    if p.q_type == QuantType::Q8_0 {
                        quantize(sxq1, &s.xb3, att_dim as usize, gs);
                        matmul_q8(
                            &mut s.xb2,
                            sxq1,
                            &w.wo_quant.expect("Field not initialized")[l as usize],
                            att_dim as usize,
                            gs as usize,
                        )
                    } else {
                        quantize_q4(sxq1, &s.xb3, att_dim as usize, gs);
                        matmul_q4(
                            &mut s.xb2,
                            sxq1,
                            &w.wo_quant.expect("Field not initialized")[l as usize],
                            att_dim as usize,
                            gs as usize,
                        )
                    }
                }
            }

            if p.model_type == ModelType::GEMMA {
                rmsnorm(
                    &mut s.xb,
                    &s.xb2,
                    &w.w_rms_post_att[(l * dim) as usize..(l * dim + dim) as usize],
                    dim as usize,
                    p.rms_norm_eps,
                    p.model_type == ModelType::GEMMA,
                );

                for i in 0..dim {
                    x[i as usize] += s.xb[i as usize];
                }

                {
                    rmsnorm(
                        &mut s.xb,
                        x,
                        &w.w_rms_pre_ffn.expect("Field not initialized")
                            [(l * dim) as usize..(l * dim + dim) as usize],
                        dim as usize,
                        p.rms_norm_eps,
                        true,
                    );
                }
            } else {
                for i in 0..dim {
                    x[i as usize] += s.xb2[i as usize];
                }

                rmsnorm(
                    &mut s.xb,
                    x,
                    &w.w_rms_post_att[(l * dim) as usize..(l * dim + dim) as usize],
                    dim as usize,
                    p.rms_norm_eps,
                    p.model_type == ModelType::GEMMA,
                );
            }

            // GeGLU is w2(GELU(w1(x)) * w3(x))
            // w1 -> gate_proj weights
            // w2 -> down_proj weights
            // w3 -> up_proj weights
            // GELU using tanh as the approximation

            {
                if !quantized {
                    matmul(
                        &mut s.hb,
                        &s.xb,
                        &w.w1.expect("Field not initialized")[(l * dim * hidden_dim) as usize
                            ..(l * dim * hidden_dim + dim * hidden_dim) as usize],
                    );
                    matmul(
                        &mut s.hb2,
                        &s.xb,
                        &w.w3.expect("Field not initialized")[(l * dim * hidden_dim) as usize
                            ..(l * dim * hidden_dim + dim * hidden_dim) as usize],
                    );
                } else {
                    let sxq = s.xq.as_mut().expect("Field not initialized");

                    if p.q_type == QuantType::Q8_0 {
                        quantize(sxq, &s.xb, dim as usize, gs);
                        matmul_q8(
                            &mut s.hb,
                            sxq,
                            &w.w1_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                        matmul_q8(
                            &mut s.hb2,
                            sxq,
                            &w.w3_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                    } else if p.q_type == QuantType::Q4_0 {
                        quantize_q4(sxq, &s.xb, dim as usize, gs);
                        matmul_q4(
                            &mut s.hb,
                            sxq,
                            &w.w1_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                        matmul_q4(
                            &mut s.hb2,
                            sxq,
                            &w.w3_quant.expect("Field not initialized")[l as usize],
                            dim as usize,
                            gs as usize,
                        );
                    }
                }
            }

            for i in 0..hidden_dim {
                let mut val = s.hb[i as usize];

                // Best case we would have the activation in the args, but for now this will do
                if p.model_type == ModelType::GEMMA {
                    // GELU
                    val *= 0.5
                        * (1.0
                            + ((0.7978845608028654 * (val + 0.044715 * val * val * val) as f64)
                                .tanh()) as f32);
                } else {
                    // SiLU
                    val *= 1.0 / (1.0 + (-val).exp());
                }

                val *= s.hb2[i as usize];

                s.hb[i as usize] = val;
            }

            {
                if !quantized {
                    matmul(
                        &mut s.xb,
                        &s.hb,
                        &w.w2.expect("Field not initialized")[(l * dim * hidden_dim) as usize
                            ..(l * dim * hidden_dim + dim * hidden_dim) as usize],
                    );
                } else {
                    let shq = s.hq.as_mut().expect("Field not initialized");

                    if p.q_type == QuantType::Q8_0 {
                        quantize(shq, &s.hb, hidden_dim as usize, gs);
                        matmul_q8(
                            &mut s.xb,
                            shq,
                            &w.w2_quant.expect("Field not initialized")[l as usize],
                            hidden_dim as usize,
                            gs as usize,
                        );
                    } else if p.q_type == QuantType::Q4_0 {
                        quantize_q4(shq, &s.hb, hidden_dim as usize, gs);
                        matmul_q4(
                            &mut s.xb,
                            shq,
                            &w.w2_quant.expect("Field not initialized")[l as usize],
                            hidden_dim as usize,
                            gs as usize,
                        );
                    }
                }
            }

            if p.model_type == ModelType::GEMMA {
                {
                    rmsnorm(
                        &mut s.xb2,
                        &s.xb,
                        &w.w_rms_post_ffn.expect("Field not initialized")
                            [(l * dim) as usize..(l * dim + dim) as usize],
                        dim as usize,
                        p.rms_norm_eps,
                        true,
                    );
                }

                for i in 0..dim {
                    x[i as usize] += s.xb2[i as usize];
                }
            } else {
                for i in 0..dim {
                    x[i as usize] += s.xb[i as usize];
                }
            }
        }

        s.xb.copy_from_slice(x);

        rmsnorm(
            x,
            &s.xb,
            w.w_rms_final,
            dim as usize,
            p.rms_norm_eps,
            p.model_type == ModelType::GEMMA,
        );

        {
            if !quantized {
                matmul(&mut s.logits, x, w.w_cls.expect("Field not initialized"));
            } else {
                let sxq = s.xq.as_mut().expect("Field not initialized");

                if p.q_type == QuantType::Q8_0 {
                    quantize(sxq, x, dim as usize, gs);
                    matmul_q8(
                        &mut s.logits,
                        sxq,
                        &w.w_cls_quant.expect("Field not initialized")[0],
                        dim as usize,
                        gs as usize,
                    );
                } else if p.q_type == QuantType::Q4_0 {
                    quantize_q4(sxq, x, dim as usize, gs);
                    matmul_q4(
                        &mut s.logits,
                        sxq,
                        &w.w_cls_quant.expect("Field not initialized")[0],
                        dim as usize,
                        gs as usize,
                    );
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
}

// Deallocate fields created with Box::leak
impl<'a> Drop for Transformer<'a> {
    fn drop(&mut self) {
        if self.args.q_type != QuantType::None {
            unsafe {
                // Weights
                dealloc(
                    self.weights.token_embedding_table.as_ptr() as *mut u8,
                    Layout::array::<f32>(self.weights.token_embedding_table.len()).unwrap(),
                );

                let layer_weights_layout =
                    Layout::array::<QuantizedTensor>(self.args.n_layers as usize).unwrap();
                dealloc(
                    self.weights
                        .wq_quant
                        .expect("Field not initialized")
                        .as_ptr() as *mut u8,
                    layer_weights_layout,
                );
                dealloc(
                    self.weights
                        .wk_quant
                        .expect("Field not initialized")
                        .as_ptr() as *mut u8,
                    layer_weights_layout,
                );
                dealloc(
                    self.weights
                        .wv_quant
                        .expect("Field not initialized")
                        .as_ptr() as *mut u8,
                    layer_weights_layout,
                );
                dealloc(
                    self.weights
                        .wo_quant
                        .expect("Field not initialized")
                        .as_ptr() as *mut u8,
                    layer_weights_layout,
                );
                dealloc(
                    self.weights
                        .w1_quant
                        .expect("Field not initialized")
                        .as_ptr() as *mut u8,
                    layer_weights_layout,
                );
                dealloc(
                    self.weights
                        .w2_quant
                        .expect("Field not initialized")
                        .as_ptr() as *mut u8,
                    layer_weights_layout,
                );
                dealloc(
                    self.weights
                        .w3_quant
                        .expect("Field not initialized")
                        .as_ptr() as *mut u8,
                    layer_weights_layout,
                );
                dealloc(
                    self.weights
                        .w_cls_quant
                        .expect("Field not initialized")
                        .as_ptr() as *mut u8,
                    Layout::array::<QuantizedTensor>(
                        self.weights
                            .w_cls_quant
                            .expect("Field not initialized")
                            .len(),
                    )
                    .unwrap(),
                );
            }
        }
    }
}
