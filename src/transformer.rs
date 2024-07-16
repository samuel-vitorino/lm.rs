use crate::functional::slice_to_u32;
use crate::functional::u8_to_f32_slice;
use crate::functional::rmsnorm;
use crate::functional::matmul;
use crate::functional::softmax;
use memmap2::Mmap;

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct TranformerArgs {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    seq_len: u32,
}

pub struct TransformerWeights<'a> {
    token_embedding_table: &'a [f32],

    //Attention
    wq: &'a [f32],
    wk: &'a [f32],
    wv: &'a [f32],
    wo: &'a [f32],
    w_rms_att: &'a [f32],

    //FFN
    w1: &'a [f32],
    w2: &'a [f32],
    w3: &'a [f32],
    w_rms_ffn: &'a [f32],

    w_rms_final: &'a [f32],

    w_cls: &'a [f32],
}

pub struct TransformerState{
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>, 
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>, 
    logits: Vec<f32>, 

    // kv cache
    key_cache: Vec<f32>,
    value_cache: Vec<f32>, 
}

pub struct Transformer<'a> {
    args: TranformerArgs,
    weights: TransformerWeights<'a>,
    state: TransformerState,
}

impl<'a> Transformer<'a> {
    pub fn new(data: &'a Mmap) -> Transformer {
        assert_eq!(data[0..4], [0x6c, 0x6d, 0x72, 0x73], "Model not in llm.rs format.");

        let lmrs_version = slice_to_u32(&data[4..8]);

        println!("LMRS version: {}\n", lmrs_version);
        
        let (head, body, _) = unsafe { data[8..36].align_to::<TranformerArgs>() };

        assert!(head.is_empty(), "Data was not aligned");
        
        let cfg = &body[0];

        let head_size = cfg.dim/cfg.n_heads;

        let emb_tab = &data[36..(36 + (cfg.vocab_size* cfg.dim * 4)) as usize];

        let mut offset: usize = (36 + (cfg.vocab_size * cfg.dim * 4)) as usize;

        // Attention weights
        let rms_att = &data[offset..offset + (cfg.n_layers * cfg.dim * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.dim * 4) as usize;
        
        let wq = &data[offset..offset + (cfg.n_layers * cfg.dim * (cfg.n_heads * head_size) * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.dim * (cfg.n_heads * head_size) * 4) as usize;

        let wk = &data[offset..offset + (cfg.n_layers * cfg.dim * (cfg.n_kv_heads * head_size) * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.dim * (cfg.n_kv_heads * head_size) * 4) as usize;

        let wv = &data[offset..offset + (cfg.n_layers * cfg.dim * (cfg.n_kv_heads * head_size) * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.dim * (cfg.n_kv_heads * head_size) * 4) as usize;

        let wo = &data[offset..offset + (cfg.n_layers * cfg.dim * (cfg.n_heads * head_size) * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.dim * (cfg.n_heads * head_size) * 4) as usize;

        // FFN weights
        let rms_ffn = &data[offset..offset + (cfg.n_layers * cfg.dim * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.dim * 4) as usize;

        let w1 = &data[offset..offset + (cfg.n_layers * cfg.dim * cfg.hidden_dim * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.dim * cfg.hidden_dim * 4) as usize;

        let w2 = &data[offset..offset + (cfg.n_layers * cfg.hidden_dim * cfg.dim * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.hidden_dim * cfg.dim * 4) as usize;

        let w3 = &data[offset..offset + (cfg.n_layers * cfg.dim * cfg.hidden_dim * 4) as usize];

        offset = offset + (cfg.n_layers * cfg.dim * cfg.hidden_dim * 4) as usize;

        // Final rms and cls weights
        let rms_final = &data[offset..offset + (cfg.dim*4) as usize];

        let w_cls = emb_tab;

        let weights = TransformerWeights {
            token_embedding_table: u8_to_f32_slice(&emb_tab),
            wq: u8_to_f32_slice(&wq),
            wk: u8_to_f32_slice(&wk),
            wv: u8_to_f32_slice(&wv),
            wo: u8_to_f32_slice(&wo),
            w_rms_att: u8_to_f32_slice(&rms_att),
            w1: u8_to_f32_slice(&w1),
            w2: u8_to_f32_slice(&w2),
            w3: u8_to_f32_slice(&w3),
            w_rms_ffn: u8_to_f32_slice(&rms_ffn),
            w_rms_final: u8_to_f32_slice(&rms_final),
            w_cls: u8_to_f32_slice(&w_cls),
        };
        
        let kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
        let state = TransformerState {
            x: vec![0.0; cfg.dim as usize],
            xb: vec![0.0; cfg.dim as usize],
            xb2: vec![0.0; cfg.dim as usize],
            hb: vec![0.0; cfg.hidden_dim as usize],
            hb2: vec![0.0; cfg.hidden_dim as usize],
            q: vec![0.0; cfg.dim as usize],
            key_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            value_cache: vec![0.0; (cfg.n_layers * cfg.seq_len * kv_dim) as usize],
            att: vec![0.0; (cfg.n_heads * cfg.seq_len) as usize],
            logits: vec![0.0; cfg.vocab_size as usize],
        };

        Transformer {
            args: *cfg,
            weights,
            state,
        }
    }

    pub fn forward(&mut self, token: u32, pos: u32) -> &[f32] {
        let p = self.args;
        let w = &self.weights;
        let s = &mut self.state;
        let x = &mut s.x;
        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads;
        let hidden_dim = p.hidden_dim;
        let head_size = dim / p.n_heads;

        x.copy_from_slice(&w.token_embedding_table[(token * dim) as usize..(token * dim + dim) as usize]);
        let normalizer = (dim as f32).sqrt();

        for i in x.iter_mut() {
            *i *= normalizer;
        }

        for l in 0..p.n_layers {
            rmsnorm(&mut s.xb, x, &w.w_rms_att[(l*dim) as usize..(l*dim + dim) as usize], dim as usize);

            let loff = l * p.seq_len * kv_dim; 
            let mut k = &mut s.key_cache[(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim) as usize];
            let mut v = &mut s.value_cache[(loff + pos * kv_dim) as usize..(loff + pos * kv_dim + kv_dim) as usize];
            
            matmul(&mut s.q, &s.xb, &w.wq[(l*dim*dim) as usize..(l*dim*dim + dim*dim) as usize]);
            matmul(k, &s.xb, &w.wk[(l*dim*kv_dim) as usize..(l*dim*kv_dim + dim*kv_dim) as usize]);
            matmul(v, &s.xb, &w.wv[(l*dim*kv_dim) as usize..(l*dim*kv_dim + dim*kv_dim) as usize]);
            
            for i in (0..dim).step_by(2) {
                let head_dim: u32 = i % head_size;
                let freq: f32 = 1.0 / 10000.0f32.powf(head_dim as f32/head_size as f32);
                let val: f32 = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                let rotn: u32 = if i < kv_dim {2} else {1};

                for v in 0..rotn{
                    let vec: &mut [f32] = if v == 0 {&mut s.q} else {k};
                    let v0: f32 = vec[i as usize];
                    let v1: f32 = vec[(i+1) as usize];

                    vec[i as usize] = v0 * fcr - v1 * fci;
                    vec[(i+1) as usize] = v0 * fci + v1 * fcr;
                }
            }
            
            for h in 0..p.n_heads {
                let q = &mut s.q[(h*head_size) as usize..(h*head_size + head_size) as usize];

                let att = &mut s.att[(h*p.seq_len) as usize..(h*p.seq_len + p.seq_len) as usize];

                for t in 0..pos+1 {
                    k = &mut s.key_cache[(loff + t * kv_dim + (h / kv_mul) * head_size) as usize..(loff + t * kv_dim + (h / kv_mul) * head_size + head_size) as usize];
                    
                    let mut score: f32 = 0.0;

                    for i in 0..head_size {
                        score += q[i as usize] * k[i as usize];
                    }

                    score /= (head_size as f32).sqrt();
 
                    att[t as usize] = score;
                }
            
                softmax(&mut att[..(pos+1) as usize]);

                let xb = &mut s.xb[(h * head_size) as usize..(h * head_size + head_size) as usize];

                xb.fill(0.0);

                for t in 0..pos+1 {
                    v = &mut s.value_cache[(loff + t * kv_dim + (h / kv_mul) * head_size) as usize..(loff + t * kv_dim + (h / kv_mul) * head_size + head_size) as usize];
                    let a = att[t as usize];

                    for i in 0..head_size {
                        xb[i as usize] += a * v[i as usize];
                    }
                }
            }
            
            matmul(&mut s.xb2, &s.xb, &w.wo[(l*dim*dim) as usize..(l*dim*dim + dim*dim) as usize]);
            
            for i in 0..dim {
                x[i as usize] += s.xb2[i as usize];
            }

            rmsnorm(&mut s.xb, x, &w.w_rms_ffn[(l*dim) as usize..(l*dim + dim) as usize], dim as usize);

            //GeGLU is w2(GELU(w1(x)) * w3(x)) 
            matmul(&mut s.hb, &s.xb, &w.w1[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize]);
            matmul(&mut s.hb2, &s.xb, &w.w3[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize]);
            
            for i in 0..hidden_dim {
                let mut val = s.hb[i as usize];
                
                val *= 0.5 * (1.0 + ((0.7978845608028654 * (val + 0.044715 * val * val * val) as f64).tanh()) as f32);
                

                val *= s.hb2[i as usize];
                
                s.hb[i as usize] = val;
            }
            
            matmul(&mut s.xb, &s.hb, &w.w2[(l*dim*hidden_dim) as usize..(l*dim*hidden_dim + dim*hidden_dim) as usize]);
            
            for i in 0..dim {
                x[i as usize] += s.xb[i as usize];
            }
        }

        s.xb.copy_from_slice(x);

        rmsnorm(x, &s.xb, &w.w_rms_final, dim as usize);
        
        matmul(&mut s.logits, &x, &w.w_cls);
        
        return &s.logits;
    }
}