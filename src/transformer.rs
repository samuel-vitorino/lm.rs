struct TranformerArgs {
    dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    multiple_of: u32,
    ffn_dim_multiplier: f32,
    norm_eps: f32,
    rope_theta: f32,
}

struct TransformerWeights {
    token_embedding_table: [f32],

    //Attention
    wq: [f32],
    wk: [f32],
    wv: [f32],
    wo: [f32],
    w_rms_att: [f32],

    //FFN
    w1: [f32],
    w2: [f32],
    w3: [f32],
    w_rms_ffn: [f32],

    w_rms_final: [f32],

    w_cls: [f32],
}

struct Transformer {
    args: TranformerArgs,
    weights: TransformerWeights,
}
