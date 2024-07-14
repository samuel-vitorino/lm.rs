use crate::functional::slice_to_u32;
use crate::functional::u8_to_f32_slice;
use crate::functional::rmsnorm;
use crate::functional::matmul;
use std::fs::File;
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
    data: &'a Mmap,
}

impl<'a> Transformer<'a> {
    pub fn new(data: &'a Mmap) -> Transformer {
        assert_eq!(data[0..4], [0x6c, 0x6d, 0x72, 0x73], "Model not in llm.rs format.");

        let lmrs_version = slice_to_u32(&data[4..8]);

        println!("LMRS version: {}", lmrs_version);

        let (head, body, _) = unsafe { data[8..64].align_to::<TranformerArgs>() };

        assert!(head.is_empty(), "Data was not aligned");
        
        let cfg = &body[0];

        println!("{:?}", cfg);
        
        let head_size = cfg.dim/cfg.n_heads;

        let emb_tab = &data[64..(64 + (cfg.vocab_size* cfg.dim * 4)) as usize];

        let mut offset: usize = (64 + cfg.vocab_size * cfg.dim) as usize;

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
            weights: weights,
            state: state,
            data: data,
        }
    }

    pub fn forward(&mut self, token: u32, pos: u32) {
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

        for l in 0..p.n_layers {
            rmsnorm(&mut s.xb, x, &w.w_rms_att[(l*dim) as usize..(l*dim + dim) as usize], dim as usize);

            let loff = l * p.seq_len * kv_dim;
            let k = s.key_cache[(loff + pos * kv_dim) as usize];
            let v = s.value_cache[(loff + pos * kv_dim) as usize];
            
            matmul(&mut s.q, &s.xb, &w.wq[(l*dim*dim) as usize..(l*dim*dim + dim*dim) as usize]);
        }
    }
}