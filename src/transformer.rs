use std::fs;
use functional::slice_to_u32;

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
}

pub struct Transformer<'a> {
    args: TranformerArgs,
    weights: TransformerWeights<'a>,
}

//Parse model from file
pub fn parse_model(path: &str) {
    let data: Vec<u8> = fs::read(path).expect("REASON");

    assert_eq!(data[0..4], [0x6c, 0x6d, 0x72, 0x73], "Model not in llm.rs format.");

    let lmrs_version = slice_to_u32(&data[4..8]);

    println!("LMRS version: {}", lmrs_version);

    let (head, body, _) = unsafe { data[8..64].align_to::<TranformerArgs>() };

    assert!(head.is_empty(), "Data was not aligned");
    
    let cfg = &body[0];

    println!("{:?}", cfg);

    
}
