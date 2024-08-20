#Export safetensor model to llm.rs format
import torch
import struct
import argparse
import json
import re
import gc
import numpy as np

from contextlib import ExitStack
from safetensors import safe_open

def extract_layer_number(key):
    parts = key.split('.')
    for i, part in enumerate(parts):
        if part == 'layers':
            return int(parts[i + 1])
    return 0

def write_tensors_by_group(files, layer_pattern, out_file, quantize=False):
    ew = []

    for f in files:
        layers_keys = [key for key in f.keys() if layer_pattern in key]

        for layer in sorted(layers_keys, key=extract_layer_number):
            print(f"Writing: {layer}")
            
            w = f.get_tensor(layer)

            if not quantize:
                serialize_fp32(out_file, w)
            else:
                q, s, err = quantize_q80(w, group_size)
                
                serialize_int8(out_file, q)
                serialize_fp32(out_file, s)

                ew.append((err, w.shape))
                print(f"{layer} quantized {tuple(w.shape)} to Q8_0 with max error {err}")

    return ew

def serialize_fp32(file, tensor, chunk_size=1024*1024):
    tensor = tensor.detach().cpu().to(torch.float32).numpy().ravel()

    for i in range(0, len(tensor), chunk_size):
        chunk = tensor[i:i + chunk_size]
        file.write(chunk.tobytes())

def serialize_int8(file, tensor, chunk_size=1024*1024):
    tensor = tensor.detach().cpu().numpy().astype(np.int8).ravel()
    
    for i in range(0, len(tensor), chunk_size):
        chunk = tensor[i:i + chunk_size]
        file.write(chunk.tobytes())

# https://github.com/karpathy/llama2.c/blob/master/export.py
def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export safetensors model to llm.rs format.")
    parser.add_argument('--files', type=str, nargs='+', required=True, help='a list of safetensor file paths')
    parser.add_argument('--config', type=str, required=True, help='path of the config file of the model')
    parser.add_argument('--save-path', type=str, required=True, help='path of the output model file')
    parser.add_argument('--quantize', action='store_true', default=False, help='use Q8_0 quantization')
    parser.add_argument('--group-size', type=int, default=256, help='groups to use in quantization')

    version = 3

    args = parser.parse_args()
    
    quantize_type = 1 if args.quantize else 0 # 0 for no quantization, 1 for Q8_0
    group_size = args.group_size

    ew = []

    with open(args.config, 'r') as file:
        cfg = json.load(file)

    out_file = open(f"{args.save_path}.lmrs", 'wb', buffering=False)

    #lmrs
    out_file.write(struct.pack('I', 0x73726d6c))
    out_file.write(struct.pack('I', version))

    header = struct.pack('IIIIIIIIB', cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["head_dim"],
                                    cfg["num_key_value_heads"], cfg["vocab_size"], cfg["max_position_embeddings"], quantize_type)

    out_file.write(header)

    with ExitStack() as stack:
        files = [stack.enter_context(safe_open(file_path, framework="pt", device="cpu")) for file_path in args.files]

        if args.quantize:
            dim = files[0].get_tensor("model.embed_tokens.weight").shape[1]

            while dim % group_size != 0:
                group_size //= 2
                print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
        
        out_file.write(struct.pack('I', group_size))

        pad = 48 - out_file.tell()
        assert pad >= 0
        out_file.write(b'\0' * pad)
        
        # Embedding table / cls layer weights
        ew.extend(write_tensors_by_group(files, "model.embed_tokens.weight", out_file, quantize=args.quantize))

        # Attention weights
        write_tensors_by_group(files, "input_layernorm", out_file)
        ew.extend(write_tensors_by_group(files, "self_attn.q_proj", out_file, quantize=args.quantize))
        ew.extend(write_tensors_by_group(files, "self_attn.k_proj", out_file, quantize=args.quantize))
        ew.extend(write_tensors_by_group(files, "self_attn.v_proj", out_file, quantize=args.quantize))
        ew.extend(write_tensors_by_group(files, "self_attn.o_proj", out_file, quantize=args.quantize))
        
        # FFN weights
        write_tensors_by_group(files, "post_attention_layernorm", out_file)
        write_tensors_by_group(files, "pre_feedforward_layernorm", out_file)
        ew.extend(write_tensors_by_group(files, "mlp.gate_proj", out_file, quantize=args.quantize))
        ew.extend(write_tensors_by_group(files, "mlp.down_proj", out_file, quantize=args.quantize))
        ew.extend(write_tensors_by_group(files, "mlp.up_proj", out_file, quantize=args.quantize))
        write_tensors_by_group(files, "post_feedforward_layernorm", out_file)
         
        # Final norm weights
        write_tensors_by_group(files, "model.norm.weight", out_file)
    
    if args.quantize:
        ew.sort(reverse=True)
        print(f"Max quantization group error across all weights: {ew[0][0]}")

    print("Successfully converted gemma model to lmrs format.")

    out_file.close()
