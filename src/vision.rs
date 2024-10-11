use crate::quantization::{QuantizedTensor, MutableQuantizedTensor, QuantType, quantize};
use crate::transformer::{init_param, init_param_quant};
use crate::functional::{matmul, matmul_q8, matmul_conv_q8, matmul_conv, concat, layernorm, softmax};

use rayon::prelude::*;
use wide::f32x8;
use std::mem::MaybeUninit;
use std::alloc::dealloc;
use std::alloc::Layout;

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
    class_embedding: &'a [f32],

    patch_embedding: MaybeUninit<&'a [f32]>,
    patch_embedding_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    
    position_embedding: &'a [f32],

    // Attention

    wq: MaybeUninit<&'a [f32]>,
    wq_bias: &'a [f32],
    wk: MaybeUninit<&'a [f32]>,
    wk_bias: &'a [f32],
    wv: MaybeUninit<&'a [f32]>,
    wv_bias: &'a [f32],
    wo: MaybeUninit<&'a [f32]>,
    wo_bias: &'a [f32],
    
    wq_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    wk_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    wv_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    wo_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,

    layer_norm1: &'a [f32],
    layer_norm2: &'a [f32],
    layer_norm1_bias: &'a [f32],
    layer_norm2_bias: &'a [f32],

    // FFN

    w1: MaybeUninit<&'a [f32]>,
    w1_bias: &'a [f32],
    w2: MaybeUninit<&'a [f32]>,
    w2_bias: &'a [f32],

    w1_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,
    w2_quant: MaybeUninit<&'a [QuantizedTensor<'a>]>,

    pre_layer_norm: &'a [f32],
    pre_layer_norm_bias: &'a [f32],
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
        
        let cfg = &body[0];

        let head_size = cfg.head_size;
        
        let mut offset: usize = 128;

        let quantized = cfg.q_type != QuantType::None;
        
        let class_embedding = init_param(data, &mut offset, 1, cfg.dim);
        
        if !quantized {
            let patch_embedding = init_param(data, &mut offset, 1, cfg.dim*3*cfg.patch_size*cfg.patch_size);
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
                patch_embedding: MaybeUninit::new(patch_embedding),
                patch_embedding_quant: MaybeUninit::uninit(),
                position_embedding,
                layer_norm1,
                layer_norm1_bias,
                layer_norm2,
                layer_norm2_bias,
                wq: MaybeUninit::new(wq),
                wk: MaybeUninit::new(wk),
                wv: MaybeUninit::new(wv),
                wo: MaybeUninit::new(wo),
                wq_bias,
                wk_bias,
                wv_bias,
                wo_bias,
                wq_quant: MaybeUninit::uninit(),
                wk_quant: MaybeUninit::uninit(),
                wv_quant: MaybeUninit::uninit(),
                wo_quant: MaybeUninit::uninit(),
                w1: MaybeUninit::new(w1),
                w2: MaybeUninit::new(w2),
                w1_bias,
                w2_bias,
                w1_quant: MaybeUninit::uninit(),
                w2_quant: MaybeUninit::uninit(),
                pre_layer_norm,
                pre_layer_norm_bias,
            };

            return (VisionTransformer {
                args: *cfg,
                weights,
            }, offset)
        } 

        println!("Loading vision encoder weights...");

        let patch_embedding_quant = init_param_quant(data, &mut offset, 1, cfg.dim*3*cfg.patch_size*cfg.patch_size, cfg.patch_size*cfg.patch_size, cfg.q_type);
        let position_embedding = init_param(data, &mut offset, 1, cfg.dim*577);

        let layer_norm1 = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let layer_norm1_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let layer_norm2 = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let layer_norm2_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);

        let wq_quant = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wq_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let wk_quant = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wk_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        let wv_quant = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wv_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        
        let wo_quant = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.n_heads * head_size, cfg.group_size, cfg.q_type);
        let wo_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        
        let w1_quant = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size, cfg.q_type);
        let w1_bias = init_param(data, &mut offset, cfg.n_layers, cfg.hidden_dim);
        
        let w2_quant = init_param_quant(data, &mut offset, cfg.n_layers, cfg.dim * cfg.hidden_dim, cfg.group_size, cfg.q_type);
        let w2_bias = init_param(data, &mut offset, cfg.n_layers, cfg.dim);
        
        let pre_layer_norm = init_param(data, &mut offset, 1, cfg.dim);
        let pre_layer_norm_bias = init_param(data, &mut offset, 1, cfg.dim);
        
        let weights = VisionTransformerWeights {
            class_embedding,
            patch_embedding: MaybeUninit::uninit(),
            patch_embedding_quant: MaybeUninit::new(patch_embedding_quant),
            position_embedding,
            layer_norm1,
            layer_norm1_bias,
            layer_norm2,
            layer_norm2_bias,
            wq: MaybeUninit::uninit(),
            wk: MaybeUninit::uninit(),
            wv: MaybeUninit::uninit(),
            wo: MaybeUninit::uninit(),
            wq_bias,
            wk_bias,
            wv_bias,
            wo_bias,
            wq_quant: MaybeUninit::new(wq_quant),
            wk_quant: MaybeUninit::new(wk_quant),
            wv_quant: MaybeUninit::new(wv_quant),
            wo_quant: MaybeUninit::new(wo_quant),
            w1: MaybeUninit::uninit(),
            w2: MaybeUninit::uninit(),
            w1_bias,
            w2_bias,
            w1_quant: MaybeUninit::new(w1_quant),
            w2_quant: MaybeUninit::new(w2_quant),
            pre_layer_norm,
            pre_layer_norm_bias,
        };
        
        println!("Done.\n");
        
        (VisionTransformer {
            args: *cfg,
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
        
        for b in 0..num_crops {
            unsafe {
                if !quantized {
                    matmul_conv(&mut patch_embeds[(b*out_shape) as usize..(b*out_shape + out_shape) as usize], &pixel_values[(b*img_pixels) as usize..(b*img_pixels + img_pixels) as usize], &w.patch_embedding.assume_init(), 196*3, patches_per_row);
                } else {
                    let imgq = &mut MutableQuantizedTensor { q: &mut vec![0; img_pixels as usize], s: &mut vec![0.0; img_pixels as usize]};
                    
                    quantize(imgq, &pixel_values[(b*img_pixels) as usize..(b*img_pixels + img_pixels) as usize], img_pixels as usize, patch_shape);
                    matmul_conv_q8(&mut patch_embeds[(b*out_shape) as usize..(b*out_shape + out_shape) as usize], imgq, &w.patch_embedding_quant.assume_init()[0], (patch_shape*3) as usize, patch_shape as usize, patches_per_row);
                }
            }
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

            embeddings.extend(concat::<f32>(w.class_embedding, &transposed));
        }

        n_patches += 1;
        out_shape += p.dim;

        // Add position embeddings
        for i in 0..embeddings.len() {
            let p_idx = i % out_shape as usize;

            embeddings[i as usize] += w.position_embedding[p_idx as usize];
        }

        // Input layernorm
        let mut norm_embeddings = vec![0.0; embeddings.len()];
        
        for i in 0..num_crops {
            for k in 0..n_patches {
                layernorm(&mut norm_embeddings[(i*out_shape+k*dim) as usize..(i*out_shape+k*dim + p.dim) as usize], 
                    &embeddings[(i*out_shape+k*dim) as usize..(i*out_shape+k*dim + p.dim) as usize], w.pre_layer_norm, w.pre_layer_norm_bias, dim as usize, p.layernorm_eps);
            }
        }
        
        let mut qkv: Vec<f32> = vec![0.0; (norm_embeddings.len() * 3) as usize];

        // In PHI they use the penultimate layer output (?)
        for l in 0..p.n_layers-1 {
            let mut x = norm_embeddings.clone();
            
            for i in 0..num_crops {
                for k in 0..n_patches {
                    layernorm(&mut embeddings[(i*out_shape+k*dim) as usize..(i*out_shape+k*dim + p.dim) as usize], 
                        &norm_embeddings[(i*out_shape+k*dim) as usize..(i*out_shape+k*dim + p.dim) as usize], &w.layer_norm1[(l*dim) as usize..(l*dim + dim) as usize], &w.layer_norm1_bias[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.layernorm_eps);
                }
            }

            for i in 0..num_crops {
                qkv[(i*out_shape*3) as usize..(i*out_shape*3 + out_shape*3) as usize].par_chunks_mut((dim*3) as usize).enumerate().for_each( |(h, xb)| {
                    unsafe {
                        if !quantized {
                            matmul(&mut xb[..dim as usize], &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.wq.assume_init()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize]);
                            matmul(&mut xb[dim as usize..(dim*2) as usize], &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.wk.assume_init()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize]);
                            matmul(&mut xb[(dim*2) as usize..(dim*3) as usize], &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.wv.assume_init()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize]);
                        } else {
                            let mut sxq = MutableQuantizedTensor { q: &mut vec![0; (dim) as usize], s: &mut vec![0.0; dim as usize]};

                            if p.q_type == QuantType::Q8_0 {
                                quantize(&mut sxq, &embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                                
                                matmul_q8(&mut xb[..dim as usize], &sxq, &w.wq_quant.assume_init()[l as usize], dim as usize, gs as usize);
                                matmul_q8(&mut xb[dim as usize..(dim*2) as usize], &sxq, &w.wk_quant.assume_init()[l as usize], dim as usize, gs as usize);
                                matmul_q8(&mut xb[(dim*2) as usize..(dim*3) as usize], &sxq, &w.wv_quant.assume_init()[l as usize], dim as usize, gs as usize);
                            } 
                        }
                    }
                    
                    // Add bias
                    let n_simd = dim/8;
                    let scale = f32x8::splat((head_size as f32).sqrt());
                    
                    for k in 0..n_simd {
                        let wq_bias_vec = f32x8::from(&w.wq_bias[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);
                        let wk_bias_vec = f32x8::from(&w.wk_bias[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);
                        let wv_bias_vec = f32x8::from(&w.wv_bias[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);

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
                    matmul(xb, &q[(i*out_shape + (h as u32 * head_size)) as usize..(i*out_shape + (h as u32 * head_size) + head_size) as usize], &k[(i*out_shape + (curr_head*head_size*n_patches)) as usize..((i*out_shape) + (curr_head*head_size*n_patches) + head_size*n_patches) as usize]);
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
                    matmul(xb, &att[(i*att_size + (h as u32 * n_patches)) as usize..(i*att_size + (h as u32 * n_patches) + n_patches) as usize], &v[(i*out_shape + curr_head*n_patches*head_size) as usize..((i*out_shape + curr_head*n_patches*head_size) + n_patches*head_size) as usize]);
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
                    unsafe {
                        if !quantized {
                            matmul(xb, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.wo.assume_init()[(l*dim*att_dim) as usize..(l*dim*att_dim + dim*att_dim) as usize]);
                        } else {
                            let mut sxq = MutableQuantizedTensor { q: &mut vec![0; (dim) as usize], s: &mut vec![0.0; dim as usize]};

                            if p.q_type == QuantType::Q8_0 {
                                quantize(&mut sxq, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                                
                                matmul_q8(xb, &sxq, &w.wo_quant.assume_init()[l as usize], dim as usize, gs as usize);
                            } 
                        }
                    }
                    
                    // Add bias
                    let n_simd = dim/8;
                    
                    for k in 0..n_simd {
                        let wo_bias_vec = f32x8::from(&w.wo_bias[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);

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
                for k in 0..n_patches {
                    layernorm(&mut norm_embeddings[(i*out_shape+k*dim) as usize..(i*out_shape+k*dim + p.dim) as usize], 
                        &embeddings[(i*out_shape+k*dim) as usize..(i*out_shape+k*dim + p.dim) as usize], &w.layer_norm2[(l*dim) as usize..(l*dim + dim) as usize], &w.layer_norm2_bias[(l*dim) as usize..(l*dim + dim) as usize], dim as usize, p.layernorm_eps);
                }
            }
            
            // MLP with QuickGELU activation w2(QuickGELU(w1(x)))

            for i in 0..num_crops {
                embeddings[(i*out_shape) as usize..(i*out_shape + out_shape) as usize].par_chunks_mut((dim) as usize).enumerate().for_each( |(h, xb)| {
                    let mut hidden_emb = vec![0.0; hidden_dim as usize];

                    unsafe {
                        if !quantized {
                            matmul(&mut hidden_emb, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], &w.w1.assume_init()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize]);
                        } else {
                            let mut sxq = MutableQuantizedTensor { q: &mut vec![0; (dim) as usize], s: &mut vec![0.0; dim as usize]};

                            if p.q_type == QuantType::Q8_0 {
                                quantize(&mut sxq, &norm_embeddings[(i*out_shape+(h as u32*dim)) as usize..(i*out_shape+(h as u32*dim) + dim) as usize], dim as usize, gs);
                                
                                matmul_q8(&mut hidden_emb, &sxq, &w.w1_quant.assume_init()[l as usize], dim as usize, gs as usize);
                            } 
                        }
                    }
                    
                    // Add bias
                    let mut n_simd = hidden_dim/8;
                    
                    for k in 0..n_simd {
                        let w1_bias_vec = f32x8::from(&w.w1_bias[(l*hidden_dim+k*8) as usize..(l*hidden_dim+k*8+8) as usize]);

                        let mut x1_vec = f32x8::from(&hidden_emb[(k*8) as usize..(k*8+8) as usize]);

                        x1_vec += w1_bias_vec;
                        
                        let x1 = x1_vec.to_array();

                        for j in 0..8 {
                            hidden_emb[(k*8 + j) as usize] = x1[j as usize];
                            
                            // QuickGELU
                            hidden_emb[(k*8 + j) as usize] *= 1.0 / (1.0 + (-(1.702*hidden_emb[(k*8 + j) as usize])).exp());
                        }
                    }
                    
                    unsafe {
                        if !quantized {
                            matmul(xb, &hidden_emb, &w.w2.assume_init()[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize]);
                        } else {
                            let mut sxq = MutableQuantizedTensor { q: &mut vec![0; (hidden_dim) as usize], s: &mut vec![0.0; hidden_dim as usize]};

                            if p.q_type == QuantType::Q8_0 {
                                quantize(&mut sxq, &hidden_emb, hidden_dim as usize, gs);
                                
                                matmul_q8(xb, &sxq, &w.w2_quant.assume_init()[l as usize], hidden_dim as usize, gs as usize);
                            } 
                        }
                    }

                    n_simd = dim/8;
                    
                    for k in 0..n_simd {
                        let w2_bias_vec = f32x8::from(&w.w2_bias[(l*dim+k*8) as usize..(l*dim+k*8+8) as usize]);

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

// Deallocate fields created with Box::leak
impl<'a> Drop for VisionTransformer<'a> {
    fn drop(&mut self) {
        if self.args.q_type != QuantType::None {
            unsafe {
                let patch_weights_layout = Layout::array::<QuantizedTensor>(1).unwrap();
                dealloc(self.weights.patch_embedding_quant.assume_init().as_ptr() as *mut u8, patch_weights_layout);
                
                let layer_weights_layout = Layout::array::<QuantizedTensor>(self.args.n_layers as usize).unwrap();
                dealloc(self.weights.wq_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.wk_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.wv_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.wo_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.w1_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
                dealloc(self.weights.w2_quant.assume_init().as_ptr() as *mut u8, layer_weights_layout);
            }
        }
    }
}