#Export safetensor model to llm.rs format
import torch
import struct
import argparse
import json
import re
import gc

from contextlib import ExitStack
from safetensors import safe_open
from safetensors.torch import save_file

def write_tensors_by_group(files, layer_pattern, out_file):
    for f in files:
        layers_keys = [key for key in f.keys() if layer_pattern in key]

        for layer in sorted(layers_keys):
            serialize_fp32(out_file, f.get_tensor(layer))

def serialize_fp32(file, tensor, chunk_size=1024*1024):
    tensor = tensor.to(torch.float32)
    num_elements = tensor.numel()
    tensor = tensor.view(-1)

    for i in range(0, num_elements, chunk_size):
        chunk = tensor[i:i + chunk_size].numpy()
        b = struct.pack(f'{len(chunk)}f', *chunk)
        file.write(b)
        del chunk, b
        gc.collect()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export safetensors model to llm.rs format.")
    parser.add_argument('--files', type=str, nargs='+', required=True, help='a list of safetensor file paths')
    parser.add_argument('--config', type=str, required=True, help='path of the config file of the model')
    parser.add_argument('--save_path', type=str, required=True, help='path of the output model file')

    version = 1

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        cfg = json.load(file)

    out_file = open(f"{args.save_path}.lmrs", 'wb', buffering=False)

    #lmrs
    out_file.write(struct.pack('I', 0x73726d6c))
    out_file.write(struct.pack('I', version))

    header = struct.pack('IIIIIII', cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"],
                                    cfg["num_key_value_heads"], cfg["vocab_size"], cfg["max_position_embeddings"])

    out_file.write(header)

    with ExitStack() as stack:
        files = [stack.enter_context(safe_open(file_path, framework="pt", device="cpu")) for file_path in args.files]
        
        # Embedding table / cls layer weights
        serialize_fp32(out_file, files[0].get_tensor("model.embed_tokens.weight"))

        # Attention weights
        write_tensors_by_group(files, "input_layernorm", out_file)
        write_tensors_by_group(files, "self_attn.q_proj", out_file)
        write_tensors_by_group(files, "self_attn.k_proj", out_file)
        write_tensors_by_group(files, "self_attn.v_proj", out_file)
        write_tensors_by_group(files, "self_attn.o_proj", out_file)
        
        # FFN weights
        write_tensors_by_group(files, "post_attention_layernorm", out_file)
        write_tensors_by_group(files, "mlp.down_proj", out_file)
        write_tensors_by_group(files, "mlp.gate_proj", out_file)
        write_tensors_by_group(files, "mlp.up_proj", out_file)
         
        # Final norm weights
        serialize_fp32(out_file, files[-1].get_tensor("model.norm.weight"))

    print("Successfully converted gemma model to lmrs format.")

    out_file.close()
