use crate::quantization::{Tensor, MutableQuantizedTensor, QuantType, quantize, quantize_q4};
use crate::transformer::{init_param, init_param_quant};
use crate::functional::{matmul, matmul_rest, matmul_q8, matmul_q4, concat, layernorm, softmax};

use rayon::prelude::*;
use wide::f32x8;

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct VisionTransformerArgs {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    head_size: u32,
    layernorm_eps: f32,
    pub patch_size: u32,
    pub image_size: u32,
    q_type: QuantType,
    group_size: u32,
}

pub struct VisionTransformerWeights<'a> {
    class_embedding: Tensor<'a>,

    patch_embedding: Tensor<'a>,
    
    position_embedding: Tensor<'a>,

    // Attention

    wq: Tensor<'a>,
    wq_bias: Tensor<'a>,
    wk: Tensor<'a>,
    wk_bias: Tensor<'a>,
    wv: Tensor<'a>,
    wv_bias: Tensor<'a>,
    wo: Tensor<'a>,
    wo_bias: Tensor<'a>,
    
    layer_norm1: Tensor<'a>,
    layer_norm2: Tensor<'a>,
    layer_norm1_bias: Tensor<'a>,
    layer_norm2_bias: Tensor<'a>,

    // FFN

    w1: Tensor<'a>,
    w1_bias: Tensor<'a>,
    w2: Tensor<'a>,
    w2_bias: Tensor<'a>,

    pre_layer_norm: Tensor<'a>,
    pre_layer_norm_bias: Tensor<'a>,
}

pub struct VisionTransformer<'a> {
    weights: VisionTransformerWeights<'a>,
    pub args: VisionTransformerArgs
}

pub fn qkv_split(qkv: &[f32], dim: u32, num_crops: u32, n_heads: u32, n_patches: u32, out_shape: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>){
    let mut q: Vec<f32> = Vec::with_capacity((num_crops*out_shape) as usize);
    let mut k: Vec<f32> = Vec::with_capacity((num_crops*out_shape) as usize);
    let mut v: Vec<f32> = Vec::with_capacity((num_crops*out_shape) as usize);

    let head_size = dim / n_heads;

    for i in 0..num_crops {
        for h in 0..n_heads {
            for t in 0..n_patches {
                q.extend(&qkv[((t*dim*3) + h * head_size + (i*3*out_shape)) as usize..((t*dim*3) + h * head_size + (i*3*out_shape) + head_size) as usize]);
                k.extend(&qkv[((t*dim*3 + dim) + h * head_size + (i*3*out_shape)) as usize..((t*dim*3 + dim) + h * head_size + (i*3*out_shape) + head_size) as usize]);
            }
            
            for j in 0..head_size {
                for t in 0..n_patches {
                    v.push(qkv[((t*dim*3 + 2*dim) + j + h*head_size + (i*3*out_shape)) as usize]);
                }
            }
        }
    }

    (q, k, v)
}

impl<'a> VisionTransformer<'a> {
    pub fn new(data: &'a [u8]) -> (VisionTransformer<'a>, usize) {
        let (head, body, _) = unsafe { data[..37].align_to::<VisionTransformerArgs>() };

        assert!(head.is_empty(), "Data was not aligned");
        
        let cfg = body[0];

        let head_size = cfg.head_size;
        
        let mut offset: usize = 128;

        let quantized = cfg.q_type != QuantType::None;
        
        let class_embedding = init_param(data, &mut offset, 1, cfg.dim);
        let patch_embedding = init_param(data, &mut offset, 1, cfg.dim*3*cfg.patch_size*cfg.patch_size);
        
        if !quantized {
            let position_embedding = init_param(data, &mut offset, 1, cfg.dim*577);

            let layer_norm1 = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            let layer_norm1_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            let layer_norm2 = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            let layer_norm2_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);

            let wq = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size);
            let wq_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            let wk = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size);
            let wk_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            let wv = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size);
            let wv_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            
            let wo = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size);
            let wo_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            
            let w1 = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
            let w1_bias = init_param(data, &mut offset, cfg.n_layers, cfg.hidden_dim);
            
            let w2 = init_param(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim);
            let w2_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
            
            let pre_layer_norm = init_param(data, &mut offset, 1, cfg.dim);
            let pre_layer_norm_bias = init_param(data, &mut offset, 1, cfg.dim);
            
            let weights = VisionTransformerWeights {
                class_embedding,
                patch_embedding,
                position_embedding,
                layer_norm1,
                layer_norm1_bias,
                layer_norm2,
                layer_norm2_bias,
                wq,
                wk,
                wv,
                wo,
                wq_bias,
                wk_bias,
                wv_bias,
                wo_bias,
                w1,
                w2,
                w1_bias,
                w2_bias,
                pre_layer_norm,
                pre_layer_norm_bias,
            };

            return (VisionTransformer {
                args: cfg,
                weights,
            }, offset)
        } 

        println!("Loading vision encoder weights...");

        let position_embedding = init_param(data, &mut offset, 1, cfg.dim*577);

        let layer_norm1 = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let layer_norm1_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let layer_norm2 = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let layer_norm2_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);

        let wq = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wq_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let wk = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wk_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let wv = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wv_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        
        let wo = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wo_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        
        let w1 = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size, cfg.q_type);
        let w1_bias = init_param(data, &mut offset, cfg.n_layers, cfg.hidden_dim);
        
        let w2 = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size, cfg.q_type);
        let w2_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        
        let pre_layer_norm = init_param(data, &mut offset, 1, cfg.dim);
        let pre_layer_norm_bias = init_param(data, &mut offset, 1, cfg.dim);
        
        let weights = VisionTransformerWeights {
            class_embedding,
            patch_embedding,
            position_embedding,
            layer_norm1,
            layer_norm1_bias,
            layer_norm2,
            layer_norm2_bias,
            wq_bias,
            wk_bias,
            wv_bias,
            wo_bias,
            wq,
            wk,
            wv,
            wo,
            w1_bias,
            w2_bias,
            w1,
            w2,
            pre_layer_norm,
            pre_layer_norm_bias,
        };
        
        println!("Done.\n");
        
        (VisionTransformer {
            args: cfg,
            weights,
        }, offset)
    }

    pub fn forward(&mut self, pixel_values: &[f32], num_crops: u32) -> (Vec<f32>, u32) {
        let p = self.args;
        let w = &self.weights;
        let dim = p.dim;
        let head_size = p.head_size;
        let att_dim = p.n_heads * head_size;
        let hidden_dim = p.hidden_dim;
        let gs = p.group_size;

        let quantized = p.q_type != QuantType::None;
        
        let patches_per_row = p.image_size/p.patch_size; 
        let mut n_patches = patches_per_row*patches_per_row;
        let mut out_shape = p.dim*n_patches;
        let img_pixels = 3*p.image_size*p.image_size;
        let mut patch_embeds: Vec<f32> = vec![0.0; (num_crops*out_shape) as usize];
        let patch_shape = p.patch_size*p.patch_size;
        
        // A conv2d is just a matmul between the kernel and image (matmul_rest because the kernel size is not divisible by 8)
        for b in 0..num_crops {
            matmul_rest(&mut patch_embeds[(b*out_shape) as usize..(b*out_shape + out_shape) as usize], &w.patch_embedding.as_float(), &pixel_values[(b*img_pixels) as usize..(b*img_pixels + img_pixels) as usize], (patch_shape*3) as usize, n_patches as usize);
        }

        // Cat class embedding
        let mut embeddings = Vec::with_capacity((num_crops*out_shape + num_crops*dim) as usize);

        for i in 0..num_crops {
            let mut transposed: Vec<f32> = Vec::with_capacity((n_patches*dim) as usize);

            for k in 0..n_patches {
                for c in (0..dim*n_patches).step_by(n_patches as usize) {
                    transposed.push(patch_embeds[(i*(out_shape) + c + k) as usize]);
                }
            }

            embeddings.extend(concat::<f32>(w.class_embedding.as_float(), &transposed));
        }

        n_patches += 1;
        out_shape += p.dim;

        // Add position embeddings
        for i in 0..embeddings.len() {
            let p_idx = i % out_shape as usize;

            embeddings[i as usize] += w.position_embedding.as_float()[p_idx as usize];
        }

        // Input layernorm
        let mut norm_embeddings = vec![0.0; embeddings.len()];
        
        for i in 0..num_crops {
            norm_embeddings[(i*out_shape) as usize..(i*out_shape + out_shape) as usize].par_chunks_mut(dim as usize).enumerate().for_each( |(k, nemb)| {
                layernorm(nemb, &embeddings[(i*out_shape+k as u32*dim) as usize..(i*out_shape+k as u32*dim + p.dim) as usize], w.pre_layer_norm.as_float(), w.pre_layer_norm_bias.as_float(), dim as usize, p.layernorm_eps);
            });
        }
        
        let mut qkv: Vec<f32> = vec![0.0; (norm_embeddings.len() * 3) as usize];

        // In PHI they use the penultimate layer output (?)
        for l in 0..p.n_layers-1 {
            let mut x = norm_embeddings.clone();
            
            for i in 0..num_crops {
                embeddings[(i*out_shape) as usize..(i*out_shape + out_shape) as usize].par_chunks_mut(dim as usize).enumerate().for_each( |(k, emb)| {
                    layernorm(emb, &norm_embeddings[(i*out_shape+k as u32*dim) as usize..(i*out_shape+k as u32*dim + p.dim) as usize], &w.layer_norm1.as_float()[(l*dim) as usize..(l*dim + dim) as usize], &w.layer_norm1_bias.as_float()[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.layernorm_eps);
                });
            }

            for i in 0..num_crops {
                qkv[(i*out_shape*3) as usize..(i*out_shape*3 + out_shape*3) as usize].par_chunks_mut((dim*3) as usize).enumerate().for_each( |(h, xb)| {
                    if !quantized {
                        matmul(&mut xb[..dim as usize], &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.wq.as_float()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize], dim as usize, dim as usize);
                        matmul(&mut xb[dim as usize..(dim*2) as usize], &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.wk.as_float()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize], dim as usize, dim as usize);
                        matmul(&mut xb[(dim*2) as usize..(dim*3) as usize], &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.wv.as_float()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize], dim as usize, dim as usize);
                    } else {
                        let mut sxq = MutableQuantizedTensor { q: vec![0; (dim) as usize], s: vec![0.0; dim as usize]};

                        if p.q_type == QuantType::Q8_0 {
                            quantize(&mut sxq, &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                            
                            matmul_q8(&mut xb[..dim as usize], &sxq, &w.wq.as_quantized()[l as usize], dim as usize, dim as usize, gs as usize);
                            matmul_q8(&mut xb[dim as usize..(dim*2) as usize], &sxq, &w.wk.as_quantized()[l as usize], dim as usize, dim as usize, gs as usize);
                            matmul_q8(&mut xb[(dim*2) as usize..(dim*3) as usize], &sxq, &w.wv.as_quantized()[l as usize], dim as usize, dim as usize, gs as usize);
                        } else if p.q_type == QuantType::Q4_0 {
                            quantize_q4(&mut sxq, &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                            
                            matmul_q4(&mut xb[..dim as usize], &sxq, &w.wq.as_quantized()[l as usize], dim as usize, dim as usize, gs as usize);
                            matmul_q4(&mut xb[dim as usize..(dim*2) as usize], &sxq, &w.wk.as_quantized()[l as usize], dim as usize, dim as usize, gs as usize);
                            matmul_q4(&mut xb[(dim*2) as usize..(dim*3) as usize], &sxq, &w.wv.as_quantized()[l as usize], dim as usize, dim as usize, gs as usize);
                        }
                    }
                    
                    // Add bias
                    let n_simd = dim/8;
                    let scale = f32x8::splat((head_size as f32).sqrt());
                    
                    for k in 0..n_simd {
                        let wq_bias_vec = f32x8::from(&w.wq_bias.as_float()[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);
                        let wk_bias_vec = f32x8::from(&w.wk_bias.as_float()[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);
                        let wv_bias_vec = f32x8::from(&w.wv_bias.as_float()[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);

                        let mut xq_vec = f32x8::from(&xb[(k*8) as usize..(k*8+8) as usize]);
                        let mut xk_vec = f32x8::from(&xb[(dim+k*8) as usize..(dim+k*8+8) as usize]);
                        let mut xv_vec = f32x8::from(&xb[(dim*2+k*8) as usize..(dim*2+k*8+8) as usize]);

                        xq_vec += wq_bias_vec;
                        xk_vec += wk_bias_vec;
                        xv_vec += wv_bias_vec;
                        
                        // Apply scale here for convenience
                        let xq = (xq_vec/scale).to_array();
                        let xk = xk_vec.to_array();
                        let xv = xv_vec.to_array();

                        for j in 0..8 {
                            xb[(k*8 + j) as usize] = xq[j as usize];
                            xb[(k*8 + dim + j) as usize] = xk[j as usize];
                            xb[(k*8 + 2*dim + j) as usize] = xv[j as usize];
                        }
                    }
                })
            }
            
            // Split into q k v, and reshape so all the heads are consequent

            let (q, k, v) = qkv_split(&qkv, dim, num_crops, p.n_heads, n_patches, out_shape);

            let att_size = p.n_heads*n_patches*n_patches;
            let mut att: Vec<f32> = vec![0.0; (att_size*num_crops) as usize];
            
            // Q * K
            // Shape - ((c*heads)*T*head_size)
             
            for i in 0..num_crops {
                att[(i*att_size) as usize..(i*att_size + att_size) as usize].par_chunks_mut((n_patches) as usize).enumerate().for_each( |(h, xb)| {
                    let curr_head = h as u32 / n_patches;
                    matmul_rest(xb, &q[(i*out_shape + (h as u32 * head_size)) as usize..(i*out_shape + (h as u32 * head_size) + head_size) as usize], &k[(i*out_shape + (curr_head*head_size*n_patches)) as usize..((i*out_shape) + (curr_head*head_size*n_patches) + head_size*n_patches) as usize], head_size as usize, n_patches as usize);
                })
            }

            // Softmax

            for i in 0..(num_crops*p.n_heads) {
                for k in 0..n_patches {
                    softmax(&mut att[(i*n_patches*n_patches + k*n_patches) as usize..(i*n_patches*n_patches + k*n_patches + n_patches) as usize])
                }
            }

            // Attention weights * v

            for i in 0..num_crops {
                embeddings[(i*out_shape) as usize..(i*out_shape + out_shape) as usize].par_chunks_mut((head_size) as usize).enumerate().for_each( |(h, xb)| {
                    let curr_head = h as u32 / n_patches;
                    matmul_rest(xb, &att[(i*att_size + (h as u32 * n_patches)) as usize..(i*att_size + (h as u32 * n_patches) + n_patches) as usize], &v[(i*out_shape + curr_head*n_patches*head_size) as usize..((i*out_shape + curr_head*n_patches*head_size) + n_patches*head_size) as usize], n_patches as usize, head_size as usize);
                })
            }
            
            // Transpose V from (t1h1,t2h1,t3h1...) -> (t1h1t1h2t1h3t2hh1t2h2...)
            
            for i in 0..num_crops {
                for k in 0..n_patches {
                    for j in 0..p.n_heads {
                        norm_embeddings[(i*out_shape + k*dim + j*head_size) as usize..(i*out_shape + k*dim + j*head_size + head_size) as usize].copy_from_slice(&embeddings[(i*out_shape + j*head_size*n_patches + k*head_size) as usize..(i*out_shape + j*head_size*n_patches + k*head_size + head_size) as usize]);
                    }
                }
            }
            
            // Out linear projection
            
            for i in 0..num_crops {
                embeddings[(i*out_shape) as usize..(i*out_shape + out_shape) as usize].par_chunks_mut((dim) as usize).enumerate().for_each( |(h, xb)| {
                    if !quantized {
                        matmul(xb, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.wo.as_float()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize], dim as usize, dim as usize);
                    } else {
                        let mut sxq = MutableQuantizedTensor { q: vec![0; (dim) as usize], s: vec![0.0; dim as usize]};

                        if p.q_type == QuantType::Q8_0 {
                            quantize(&mut sxq, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                            
                            matmul_q8(xb, &sxq, &w.wo.as_quantized()[l as usize], dim as usize, dim as usize, gs as usize);
                        } else if p.q_type == QuantType::Q4_0 {
                            quantize_q4(&mut sxq, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                            
                            matmul_q4(xb, &sxq, &w.wo.as_quantized()[l as usize], dim as usize, dim as usize, gs as usize);
                        }
                    }
                    
                    // Add bias
                    let n_simd = dim/8;
                    
                    for k in 0..n_simd {
                        let wo_bias_vec = f32x8::from(&w.wo_bias.as_float()[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);

                        let mut xo_vec = f32x8::from(&xb[(k*8) as usize..(k*8+8) as usize]);

                        xo_vec += wo_bias_vec;
                        
                        let xo = xo_vec.to_array();

                        for j in 0..8 {
                            xb[(k*8 + j) as usize] = xo[j as usize];
                        }
                    }
                })
            }
            
            // Add residual 

            for i in 0..num_crops {
                for t in 0..n_patches {
                    for d in 0..dim {
                        embeddings[(i*out_shape + t * dim + d) as usize] += x[(i*out_shape + t * dim + d) as usize];
                    }
                }
            }

            x.copy_from_slice(&embeddings);
            
            for i in 0..num_crops {
                norm_embeddings[(i*out_shape) as usize..(i*out_shape + out_shape) as usize].par_chunks_mut(dim as usize).enumerate().for_each( |(k, nemb)| {
                    layernorm(nemb, &embeddings[(i*out_shape+k as u32*dim) as usize..(i*out_shape+k as u32*dim + p.dim) as usize], &w.layer_norm2.as_float()[(l*dim) as usize..(l*dim + dim) as usize], &w.layer_norm2_bias.as_float()[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.layernorm_eps);
                });
            }
            
            // MLP with QuickGELU activation w2(QuickGELU(w1(x)))

            for i in 0..num_crops {
                embeddings[(i*out_shape) as usize..(i*out_shape + out_shape) as usize].par_chunks_mut((dim) as usize).enumerate().for_each( |(h, xb)| {
                    let mut hidden_emb = vec![0.0; hidden_dim as usize];

                    if !quantized {
                        matmul(&mut hidden_emb, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.w1.as_float()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize], dim as usize, hidden_dim as usize);
                    } else {
                        let mut sxq = MutableQuantizedTensor { q: vec![0; (dim) as usize], s: vec![0.0; dim as usize]};

                        if p.q_type == QuantType::Q8_0 {
                            quantize(&mut sxq, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                            
                            matmul_q8(&mut hidden_emb, &sxq, &w.w1.as_quantized()[l as usize], dim as usize, hidden_dim as usize, gs as usize);
                        } else if p.q_type == QuantType::Q4_0 {
                            quantize_q4(&mut sxq, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                            
                            matmul_q4(&mut hidden_emb, &sxq, &w.w1.as_quantized()[l as usize], dim as usize, hidden_dim as usize, gs as usize);
                        }
                    }
                    
                    // Add bias
                    let mut n_simd = hidden_dim/8;
                    
                    for k in 0..n_simd {
                        let w1_bias_vec = f32x8::from(&w.w1_bias.as_float()[(l*hidden_dim+k*8) as usize..(l*hidden_dim+k*8+8) as usize]);

                        let mut x1_vec = f32x8::from(&hidden_emb[(k*8) as usize..(k*8+8) as usize]);

                        x1_vec += w1_bias_vec;
                        
                        let x1 = x1_vec.to_array();

                        for j in 0..8 {
                            hidden_emb[(k*8 + j) as usize] = x1[j as usize];
                            
                            // QuickGELU
                            hidden_emb[(k*8 + j) as usize] *= 1.0 / (1.0 + (-(1.702*hidden_emb[(k*8 + j) as usize])).exp());
                        }
                    }
                    
                    if !quantized {
                        matmul(xb, &hidden_emb, &w.w2.as_float()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize], hidden_dim as usize, dim as usize);
                    } else {
                        let mut sxq = MutableQuantizedTensor { q: vec![0; (hidden_dim) as usize], s: vec![0.0; hidden_dim as usize]};

                        if p.q_type == QuantType::Q8_0 {
                            quantize(&mut sxq, &hidden_emb, hidden_dim as usize, gs);
                            
                            matmul_q8(xb, &sxq, &w.w2.as_quantized()[l as usize], hidden_dim as usize, dim as usize, gs as usize);
                        } else if p.q_type == QuantType::Q4_0 {
                            quantize_q4(&mut sxq, &hidden_emb, hidden_dim as usize, gs);
                            
                            matmul_q4(xb, &sxq, &w.w2.as_quantized()[l as usize], hidden_dim as usize, dim as usize, gs as usize);
                        }
                    }

                    n_simd = dim/8;
                    
                    for k in 0..n_simd {
                        let w2_bias_vec = f32x8::from(&w.w2_bias.as_float()[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);

                        let mut x2_vec = f32x8::from(&xb[(k*8) as usize..(k*8+8) as usize]);

                        x2_vec += w2_bias_vec;
                        
                        let x2 = x2_vec.to_array();

                        for j in 0..8 {
                            xb[(k*8 + j) as usize] = x2[j as usize];
                        }
                    }
                })
            }
            
            // Add residual 

            for i in 0..num_crops {
                for t in 0..n_patches {
                    for d in 0..dim {
                        embeddings[(i*out_shape + t * dim + d) as usize] += x[(i*out_shape + t * dim + d) as usize];
                    }
                }
            }

            norm_embeddings.copy_from_slice(&embeddings);
        }

        // Remove CLS embedding
        let new_shape = dim*(n_patches - 1);
        let mut out_patches = vec![0.0; (num_crops*new_shape) as usize];
        
        for i in 0..num_crops {
            for p in 1..n_patches {
                out_patches[(i*new_shape + (p-1)*dim) as usize..(i*new_shape + (p-1)*dim + dim) as usize].copy_from_slice(&norm_embeddings[(i*out_shape + p*dim) as usize..(i*out_shape + p*dim + dim) as usize]);
            }
        }
         
        (out_patches, new_shape)
    }
}