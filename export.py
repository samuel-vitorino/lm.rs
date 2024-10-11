# Export safetensor model to lm.rs format

import torch
import struct
import argparse
import json
import re
import gc
import numpy as np

from contextlib import ExitStack
from safetensors import safe_open

from utils.io import write_tensors_by_group

if __name__ == "__main__":
    model_types = ["GEMMA", "LLAMA", "PHI"]

    parser = argparse.ArgumentParser(description="Export safetensors model to lm.rs format.")
    parser.add_argument('--files', type=str, nargs='+', required=True, help='a list of safetensor file paths')
    parser.add_argument('--config', type=str, required=True, help='path of the config file of the model')
    parser.add_argument('--save-path', type=str, required=True, help='path of the output model file')
    parser.add_argument('--quantize', action='store_true', default=False, help='use quantization')
    parser.add_argument('--quantize-type', type=int, default=1, help='type of quantization - 1 for Q8_0, 2 for Q4_0')
    parser.add_argument('--group-size', type=int, default=128, help='groups to use in quantization')
    parser.add_argument('--type', type=str, required=True, choices=model_types, help='model type')
    parser.add_argument('--vision-config', type=str, required=False, help='path to the vision model config')

    version = 4

    args = parser.parse_args()

    if args.vision_config and args.type != "PHI":
        print("Error: --multimodal can only be used when --type is PHI.")
        sys.exit(1)

    if args.quantize:
        assert args.quantize_type == 1 or args.quantize_type == 2
    
    quantize_type = args.quantize_type if args.quantize else 0 # 0 for no quantization, 1 for Q8_0, 2 for Q4_0
    group_size = args.group_size

    ew = []

    model_type = model_types.index(args.type)
    multimodal = 1 if args.vision_config else 0

    with open(args.config, 'r') as file:
        cfg = json.load(file)

    out_file = open(f"{args.save_path}.lmrs", 'wb', buffering=False)

    #lmrs
    out_file.write(struct.pack('I', 0x73726d6c))
    out_file.write(struct.pack('I', version))

    head_dim = cfg["head_dim"] if "head_dim" in cfg else cfg["hidden_size"] // cfg["num_attention_heads"]

    header = struct.pack('IIIIIIIIff', cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], head_dim,
                                    cfg["num_key_value_heads"], cfg["vocab_size"], cfg["max_position_embeddings"], cfg["rms_norm_eps"], cfg["rope_theta"])

    out_file.write(header)

    types = struct.pack('BB', quantize_type, model_type) 

    out_file.write(types)

    with ExitStack() as stack:
        files = [stack.enter_context(safe_open(file_path, framework="pt", device="cpu")) for file_path in args.files]

        if args.quantize:
            dim = files[0].get_tensor("model.embed_tokens.weight").shape[1]

            while dim % group_size != 0:
                group_size //= 2
                print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
        
        out_file.write(struct.pack('I', group_size))
        
        out_file.write(struct.pack('B', multimodal))

        pad = 256 - out_file.tell()
        assert pad >= 0
        out_file.write(b'\0' * pad)
        
        # Embedding table / cls layer weights
        ew.extend(write_tensors_by_group(files, "model.embed_tokens.weight", out_file, m_type="", quantize_type=quantize_type))

        # Attention weights
        write_tensors_by_group(files, "input_layernorm", out_file)

        if args.type == "PHI":
            ew.extend(write_tensors_by_group(files, "self_attn.qkv_proj", out_file, quantize_type=quantize_type, splits=3, split_idx=0))
            ew.extend(write_tensors_by_group(files, "self_attn.qkv_proj", out_file, quantize_type=quantize_type, splits=3, split_idx=1))
            ew.extend(write_tensors_by_group(files, "self_attn.qkv_proj", out_file, quantize_type=quantize_type, splits=3, split_idx=2))
        else:
            ew.extend(write_tensors_by_group(files, "self_attn.q_proj", out_file, quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "self_attn.k_proj", out_file, quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "self_attn.v_proj", out_file, quantize_type=quantize_type))
        
        ew.extend(write_tensors_by_group(files, "self_attn.o_proj", out_file, quantize_type=quantize_type))
        
        # FFN weights
        write_tensors_by_group(files, "post_attention_layernorm", out_file)

        if args.type == "GEMMA":
            write_tensors_by_group(files, "pre_feedforward_layernorm", out_file)

        if args.type == "PHI":
            ew.extend(write_tensors_by_group(files, "mlp.gate_up_proj", out_file, quantize_type=quantize_type, splits=2, split_idx=0))
            ew.extend(write_tensors_by_group(files, "mlp.down_proj", out_file, quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "mlp.gate_up_proj", out_file, quantize_type=quantize_type, splits=2, split_idx=1))
        else:
            ew.extend(write_tensors_by_group(files, "mlp.gate_proj", out_file, quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "mlp.down_proj", out_file, quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "mlp.up_proj", out_file, quantize_type=quantize_type))
        
        if args.type == "GEMMA":
            write_tensors_by_group(files, "post_feedforward_layernorm", out_file)
         
        # Final norm weights
        write_tensors_by_group(files, "model.norm.weight", out_file, m_type="")

        if args.type == "PHI":
            ew.extend(write_tensors_by_group(files, "lm_head.weight", out_file, m_type="", quantize_type=quantize_type))

        if args.vision_config:
            with open(args.vision_config, 'r') as file:
                vision_cfg = json.load(file)
            
            prev_pos = out_file.tell()

            vision_head_dim = vision_cfg["vision_config"]["hidden_size"] // vision_cfg["vision_config"]["num_attention_heads"]
            vision_header = struct.pack('IIIIIfII', vision_cfg["vision_config"]["hidden_size"], vision_cfg["vision_config"]["intermediate_size"], 
                            vision_cfg["vision_config"]["num_hidden_layers"], vision_cfg["vision_config"]["num_attention_heads"], vision_head_dim, vision_cfg["vision_config"]["layer_norm_eps"], 
                            vision_cfg["vision_config"]["patch_size"], vision_cfg["vision_config"]["image_size"])
    
            out_file.write(vision_header)

            out_file.write(struct.pack('B', quantize_type))
            
            out_file.write(struct.pack('I', group_size))
            
            pad = 128 - (out_file.tell() - prev_pos)
            assert pad >= 0
            out_file.write(b'\0' * pad)

            # CLIP encoder

            ew.extend(write_tensors_by_group(files, "class_embedding", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "patch_embedding.weight", out_file, m_type="model.vision_embed_tokens", quantize_type=quantize_type, group_size=196))
            ew.extend(write_tensors_by_group(files, "position_embedding.weight", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "layer_norm1.weight", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "layer_norm1.bias", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "layer_norm2.weight", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "layer_norm2.bias", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "self_attn.q_proj.weight", out_file, m_type="model.vision_embed_tokens", quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "self_attn.q_proj.bias", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "self_attn.k_proj.weight", out_file, m_type="model.vision_embed_tokens", quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "self_attn.k_proj.bias", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "self_attn.v_proj.weight", out_file, m_type="model.vision_embed_tokens", quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "self_attn.v_proj.bias", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "self_attn.out_proj.weight", out_file, m_type="model.vision_embed_tokens", quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "self_attn.out_proj.bias", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "mlp.fc1.weight", out_file, m_type="model.vision_embed_tokens", quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "mlp.fc1.bias", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "mlp.fc2.weight", out_file, m_type="model.vision_embed_tokens", quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "mlp.fc2.bias", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "pre_layrnorm.weight", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "pre_layrnorm.bias", out_file, m_type="model.vision_embed_tokens"))
            
            # Processor 
            prev_pos = out_file.tell()

            processor_header = struct.pack('II', vision_cfg["vision_config"]["intermediate_size"], cfg["hidden_size"])
    
            out_file.write(processor_header)

            out_file.write(struct.pack('B', quantize_type))
            
            out_file.write(struct.pack('I', group_size))

            pad = 128 - (out_file.tell() - prev_pos)
            assert pad >= 0
            out_file.write(b'\0' * pad)

            ew.extend(write_tensors_by_group(files, "glb_GN", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "sub_GN", out_file, m_type="model.vision_embed_tokens"))
            ew.extend(write_tensors_by_group(files, "img_projection", out_file, m_type="weight", quantize_type=quantize_type))
            ew.extend(write_tensors_by_group(files, "img_projection", out_file, m_type="bias"))

    
    if args.quantize:
        ew.sort(reverse=True)
        print(f"Max quantization group error across all weights: {ew[0]}. Mean: {sum(ew)/len(ew)}.")

    print(f"Successfully converted {args.type} model to lmrs format.")

    out_file.close()
