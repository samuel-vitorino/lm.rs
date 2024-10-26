use crate::quantization::{Tensor, QuantType, MutableQuantizedTensor, quantize, quantize_q4};
use crate::transformer::{init_param, init_param_quant};
use crate::functional::{matmul, matmul_q8, matmul_q4, concat};

use image::imageops::resize;
use image::{ImageBuffer, Rgb};
use wide::f32x8;
use rayon::prelude::*;

fn transpose_img(image: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut transposed_image = Vec::new();

    for x in 0..width {
        for y in 0..height {
            transposed_image.push(image[((x*3) + ((y*3) * width)) as usize]);
            transposed_image.push(image[((x*3) + 1 + ((y*3) * width)) as usize]);
            transposed_image.push(image[((x*3) + 2 + ((y*3) * width)) as usize]);
        }
    }

    return transposed_image;
}

fn normalize_img(image: &[u8], mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let mut normalized_image = Vec::new();

    for x in (0..image.len()).step_by(3) {
        let r = image[x];
        let g = image[x+1];
        let b = image[x+2];

        let norm_r = (((r as f32)/255.0) as f32 - mean[0])/std[0];
        let norm_g = ((g as f32)/255.0 - mean[1])/std[1];
        let norm_b = ((b as f32)/255.0 - mean[2])/std[2];

        normalized_image.push(norm_r);
        normalized_image.push(norm_g);
        normalized_image.push(norm_b);
    }

    return normalized_image;
}

fn pad_img(image: &[u8], width: u32, height: u32, pad_left: u32, pad_top: u32, pad_right: u32, pad_bottom: u32, value: [u8; 3]) -> Vec<u8> {
    let mut padded_image = Vec::new();
    
    for _p in 0..pad_top {
        for _x in 0..width {
            padded_image.push(value[0]);
            padded_image.push(value[1]);
            padded_image.push(value[2]);
        }
    }

    for y in 0..height {
        for _p in 0..pad_left {
            padded_image.push(value[0]);
            padded_image.push(value[1]);
            padded_image.push(value[2]);
        }

        for x in 0..width {
            padded_image.push(image[((x*3) + ((y*3) * width)) as usize]);
            padded_image.push(image[((x*3) + 1 + ((y*3) * width)) as usize]);
            padded_image.push(image[((x*3) + 2 + ((y*3) * width)) as usize]);
        }
        
        for _p in 0..pad_right {
            padded_image.push(value[0]);
            padded_image.push(value[1]);
            padded_image.push(value[2]);
        }
    }
    
    for _p in 0..pad_bottom {
        for _x in 0..width {
            padded_image.push(value[0]);
            padded_image.push(value[1]);
            padded_image.push(value[2]);
        }
    }

    padded_image
}

fn crop_img(img: &[u8], size: (u32, u32), num_crops: u32) -> Vec<u8> {
    let mut cropped_img: Vec<u8> = Vec::new();
    let crop_size = 336;
    let crops_per_side_x = size.0/crop_size;

    println!("{:?}", size);
    println!("{}", num_crops);

    for c in 0..num_crops {
        let grid_y = c/crops_per_side_x;
        let grid_x = c%crops_per_side_x;

        for y in 0..crop_size {
            for x in (0..crop_size*3).step_by(3) {
                cropped_img.push(img[((grid_x*3)*crop_size + x + (y*3)*size.0 + (grid_y*3)*crop_size*size.0) as usize]);
                cropped_img.push(img[((grid_x*3)*crop_size + x + 1 + (y*3)*size.0 + (grid_y*3)*crop_size*size.0) as usize]);
                cropped_img.push(img[((grid_x*3)*crop_size + x + 2 + (y*3)*size.0 + (grid_y*3)*crop_size*size.0) as usize]);
            }
        }
    }

    cropped_img
}

fn view_as_patches(img: &[f32], size: u32, patch_size: u32, num_crops: u32) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();
    let patches_per_row = size/patch_size;

    for c in 0..num_crops {
        for y in 0..patches_per_row {
            for x in 0..patches_per_row {
                let mut r: Vec<f32> = Vec::new();
                let mut g: Vec<f32> = Vec::new();
                let mut b: Vec<f32> = Vec::new();
                for py in 0..patch_size {
                    for px in (0..patch_size*3).step_by(3) {
                        r.push(img[(c*3*size*size) as usize + ((x*3)*patch_size + px) as usize + ((y*3)*size*patch_size + (py*3)*size) as usize]);
                        g.push(img[(c*3*size*size) as usize + ((x*3)*patch_size + px + 1) as usize + ((y*3)*size*patch_size + (py*3)*size) as usize]);
                        b.push(img[(c*3*size*size) as usize + ((x*3)*patch_size + px + 2) as usize + ((y*3)*size*patch_size + (py*3)*size) as usize]);
                    }
                }
                out.extend(r);
                out.extend(g);
                out.extend(b);
            }
        }
    }

    out
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
struct ProcessorArgs {
    hidden_dim: u32,
    text_dim: u32,
    q_type: QuantType,
    group_size: u32,
}

struct ProcessorWeights<'a> {
    glb_gn: Tensor<'a>,
    sub_gn: Tensor<'a>,
    
    img_projection0: Tensor<'a>,
    img_projection0_bias: Tensor<'a>,
    img_projection1: Tensor<'a>,
    img_projection1_bias: Tensor<'a>,
}

pub struct PHI3VProcessor<'a> {
    weights: ProcessorWeights<'a>,
    args: ProcessorArgs
}

impl<'a> PHI3VProcessor<'a> {
    pub fn new(data: &'a [u8]) -> PHI3VProcessor<'a> { 
        let (head, body, _) = unsafe { data[..13].align_to::<ProcessorArgs>() };

        assert!(head.is_empty(), "Data was not aligned");
        
        let cfg = body[0];

        let mut offset: usize = 128;

        let hidden_dim = cfg.hidden_dim;
        let text_dim = cfg.text_dim;
        let quantized = cfg.q_type != QuantType::None;

        let glb_gn = init_param(data, &mut offset, 1, hidden_dim);
        let sub_gn = init_param(data, &mut offset, 1, hidden_dim);

        if !quantized {
            let img_projection0 = init_param(data, &mut offset, 1, hidden_dim * text_dim);
            let img_projection1 = init_param(data, &mut offset, 1, text_dim * text_dim);
            let img_projection0_bias = init_param(data, &mut offset, 1, text_dim);
            let img_projection1_bias = init_param(data, &mut offset, 1, text_dim);

            let weights = ProcessorWeights {
                glb_gn,
                sub_gn,
                img_projection0,
                img_projection1,
                img_projection0_bias,
                img_projection1_bias
            };

            return PHI3VProcessor {
                weights,
                args: cfg
            }
        }

        println!("Loading processor weights...");

        let img_projection0 = init_param_quant(data, &mut offset, 1, text_dim * hidden_dim, cfg.group_size, cfg.q_type);
        let img_projection1 = init_param_quant(data, &mut offset, 1, text_dim * text_dim, cfg.group_size, cfg.q_type);
        let img_projection0_bias = init_param(data, &mut offset, 1, text_dim);
        let img_projection1_bias = init_param(data, &mut offset, 1, text_dim);
        
        let weights = ProcessorWeights {
            glb_gn,
            sub_gn,
            img_projection0,
            img_projection1,
            img_projection0_bias,
            img_projection1_bias
        };

        println!("Done.\n");

        return PHI3VProcessor {
            weights,
            args: cfg
        }
    }
    
    pub fn forward(&self, out_patches: &[f32], new_shape: u32, patch_side: u32, w_crop: u32, h_crop: u32) -> Vec<f32> {
        let p = self.args;
        let w = &self.weights;
        let hidden_dim = p.hidden_dim;

        let quantized = p.q_type != QuantType::None;

        let mut global_features = PHI3VProcessor::reshape_hd_patches_2x2merge(&out_patches[..new_shape as usize], 1, 1);
        PHI3VProcessor::add_image_newline(&mut global_features, w.sub_gn.as_float(), patch_side as usize, patch_side as usize, hidden_dim as usize);
        
        let mut sub_image_features = PHI3VProcessor::reshape_hd_patches_2x2merge(&out_patches[new_shape as usize..], h_crop as usize, w_crop as usize);
        PHI3VProcessor::add_image_newline(&mut sub_image_features, w.sub_gn.as_float(), (h_crop*patch_side) as usize, (w_crop*patch_side) as usize, hidden_dim as usize);

        let sub_len = sub_image_features.len();
        let glb_len = global_features.len();

        let mut out_embeddings = Vec::with_capacity(sub_len + glb_len + hidden_dim as usize);

        out_embeddings.extend(sub_image_features);
        out_embeddings.extend(w.glb_gn.as_float());
        out_embeddings.extend(global_features);
        
        let num_embeds = (h_crop * patch_side) * ((w_crop * patch_side + 1)) + (patch_side*(patch_side + 1)) + 1;
        
        let out_shape = num_embeds * p.text_dim;

        let mut out_features = vec![0.0; out_shape as usize];

        // Img projection mlp with GELU activation
        out_features.par_chunks_mut(p.text_dim as usize).enumerate().for_each( |(h, xb)| {
            let mut hidden_emb = vec![0.0; p.text_dim as usize];

            if !quantized {
                matmul(&mut hidden_emb, &out_embeddings[((h as u32*hidden_dim)) as usize..((h as u32*hidden_dim) + hidden_dim) as usize], &w.img_projection0.as_float(), hidden_dim as usize, p.text_dim as usize);
            } else {
                let mut sxq = MutableQuantizedTensor { q: vec![0; (hidden_dim) as usize], s: vec![0.0; hidden_dim as usize]};

                if p.q_type == QuantType::Q8_0 {
                    quantize(&mut sxq, &out_embeddings[((h as u32*hidden_dim)) as usize..((h as u32*hidden_dim) + hidden_dim) as usize], hidden_dim as usize, p.group_size);
                    
                    matmul_q8(&mut hidden_emb, &sxq, &w.img_projection0.as_quantized()[0], hidden_dim as usize, p.text_dim as usize, p.group_size as usize);
                } else if p.q_type == QuantType::Q4_0 {
                    quantize_q4(&mut sxq, &out_embeddings[((h as u32*hidden_dim)) as usize..((h as u32*hidden_dim) + hidden_dim) as usize], hidden_dim as usize, p.group_size);
                    
                    matmul_q4(&mut hidden_emb, &sxq, &w.img_projection0.as_quantized()[0], hidden_dim as usize, p.text_dim as usize, p.group_size as usize);
                }
            }
            
            // Add bias
            let mut n_simd = p.text_dim/8;
            
            for k in 0..n_simd {
                let w1_bias_vec = f32x8::from(&w.img_projection0_bias.as_float()[(k*8) as usize..(k*8+8) as usize]);

                let mut x1_vec = f32x8::from(&hidden_emb[(k*8) as usize..(k*8+8) as usize]);

                x1_vec += w1_bias_vec;
                
                let x1 = x1_vec.to_array();

                for j in 0..8 {
                    let idx = (k*8 + j) as usize;

                    hidden_emb[idx] = x1[j as usize];
                    
                    // GELU
                    hidden_emb[idx] *= 0.5 * (1.0 + ((0.7978845608028654 * (hidden_emb[idx] + 0.044715 * hidden_emb[idx] * hidden_emb[idx] * hidden_emb[idx]) as f64).tanh()) as f32);   
                }
            }
            
            if !quantized {
                matmul(xb, &hidden_emb, &w.img_projection1.as_float(), p.text_dim as usize, p.text_dim as usize);
            } else {
                let mut sxq = MutableQuantizedTensor { q: vec![0; (p.text_dim) as usize], s: vec![0.0; p.text_dim as usize]};

                if p.q_type == QuantType::Q8_0 {
                    quantize(&mut sxq, &hidden_emb, p.text_dim as usize, p.group_size);
                    
                    matmul_q8(xb, &sxq, &w.img_projection1.as_quantized()[0], p.text_dim as usize, p.text_dim as usize, p.group_size as usize);
                } else if p.q_type == QuantType::Q4_0 {
                    quantize_q4(&mut sxq, &hidden_emb, p.text_dim as usize, p.group_size);
                    
                    matmul_q4(xb, &sxq, &w.img_projection1.as_quantized()[0], p.text_dim as usize, p.text_dim as usize, p.group_size as usize);
                }
            }

            n_simd = p.text_dim/8;
            
            for k in 0..n_simd {
                let w2_bias_vec = f32x8::from(&w.img_projection1_bias.as_float()[(k*8) as usize..(k*8+8) as usize]);

                let mut x2_vec = f32x8::from(&xb[(k*8) as usize..(k*8+8) as usize]);

                x2_vec += w2_bias_vec;
                
                let x2 = x2_vec.to_array();

                for j in 0..8 {
                    xb[(k*8 + j) as usize] = x2[j as usize];
                }
            }
        });

        out_features
    }

    pub fn process(&self, pixels: &[u8], width: u32, height: u32, patch_size: u32, mut num_crops: u32) -> (Vec<f32>, u32, u32, u32) {
        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std = [0.26862954, 0.26130258, 0.27577711];

        let (transposed_image, new_w, new_h) = PHI3VProcessor::hd_transform(pixels, width, height, num_crops);
        
        let resized_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(new_w, new_h, transposed_image.clone()).expect("Failed to create image");

        let global_img = resize(&resized_img, 336, 336, image::imageops::FilterType::Triangle);

        let cropped_img;

        let w_crop = new_w/336;
        let h_crop = new_h/336;

        num_crops = w_crop*h_crop;

        if num_crops > 2 {
            cropped_img = crop_img(&transposed_image, (new_w, new_h), num_crops);
        } else {
            cropped_img = transposed_image;
        }

        num_crops += 1;

        let final_img = concat::<u8>(&global_img, &cropped_img);
        let normalized_img = normalize_img(&final_img, mean, std);
        
        let patches = view_as_patches(&normalized_img, 336, patch_size, num_crops);

        (patches, w_crop, h_crop, num_crops)
    }
    
    fn reshape_hd_patches_2x2merge(image_features: &[f32], h_crop: usize, w_crop: usize) -> Vec<f32> {
        const C: usize = 1024;
        const H: usize = 24;
        const L: usize = H * H;
        
        let n = image_features.len() / (L * C);
        let num_images = n / (h_crop * w_crop);
        let out_h = h_crop * H / 2;
        let out_w = w_crop * H / 2;
        let out_c = 4 * C;
        
        let mut result = vec![0.0; num_images * out_h * out_w * out_c];
        
        for img_idx in 0..num_images {
            for hc in 0..h_crop {
                for wc in 0..w_crop {
                    let patch_idx = img_idx * h_crop * w_crop + hc * w_crop + wc;
                    
                    for i in 0..H/2 {
                        for j in 0..H/2 {
                            let mut merged_features = Vec::with_capacity(4 * C);
                            for di in 0..2 {
                                for dj in 0..2 {
                                    let old_y = i * 2 + di;
                                    let old_x = j * 2 + dj;
                                    let old_start = patch_idx * L * C + (old_y * H + old_x) * C;
                                    merged_features.extend_from_slice(&image_features[old_start..old_start + C]);
                                }
                            }
                            
                            let new_h = hc * H/2 + i;
                            let new_w = wc * H/2 + j;
                            let new_idx = ((img_idx * out_h + new_h) * out_w + new_w) * out_c;
                            result[new_idx..new_idx + out_c].copy_from_slice(&merged_features);
                        }
                    }
                }
            }
        }
        
        result
    }
    
    fn padding_336(image: &[u8], width: u32, height: u32) -> (Vec<u8>, u32){
        let tar = ((height as f32 / 336.0).ceil() * 336.0) as u32;
        let top_padding = (tar - height)/2;
        let bottom_padding = tar - height - top_padding;
        
        let b = pad_img(image, width, height, 0, top_padding, 0, bottom_padding, [255, 255, 255]);

        (b, height + top_padding + bottom_padding)
    }

    fn hd_transform(img: &[u8], width: u32, height: u32, hd_num: u32) -> (Vec<u8>, u32, u32) {
        let mut trans = false;
        let mut new_img = Vec::from(img);
        let mut new_width = width;
        let mut new_height = height;

        if width < height {
            new_img = transpose_img(img, width, height);
            trans = true;
            new_width = height;
            new_height = width;
        }

        let ratio: f32 = new_width as f32 / new_height as f32;
        let mut scale: f32 = 1.0;

        while scale*(scale/ratio).ceil() <= hd_num as f32 {
            scale += 1.0;
        }

        scale -= 1.0;

        let mut new_w = (scale * 336.0) as u32;
        let mut new_h = (new_w as f32 / ratio) as u32;

        let img_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(new_width, new_height, new_img).expect("Failed to create image");
        let resized_img = resize(&img_buffer, new_w, new_h, image::imageops::FilterType::Triangle);

        (new_img, new_h) = PHI3VProcessor::padding_336(resized_img.as_raw() as &[u8], new_w, new_h);

        if trans {
            new_img = transpose_img(&new_img, new_w, new_h);
            let temp_w = new_w;
            new_w = new_h;
            new_h = temp_w;
        }

        (new_img, new_w, new_h)
    }

    fn insert_slice_at_position(vec: &mut Vec<f32>, index: usize, slice: &[f32]) {
        vec.reserve(slice.len());

        let tail = vec.split_off(index);

        vec.extend(slice);

        vec.extend(tail);
    }

    fn add_image_newline(img: &mut Vec<f32>, separator: &[f32], h: usize, w: usize, dim: usize) {
        for i in 0..h {
            PHI3VProcessor::insert_slice_at_position(img, i*w*dim + i*dim + w*dim, separator);
        }
    }
}