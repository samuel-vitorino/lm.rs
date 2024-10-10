import torch
import numpy as np

def quantize_q40(w, group_size):
    """
    takes a tensor and returns the Q4_0 quantized version
    i.e. quantization into int4, range [0,15]
    """

    assert w.numel() % group_size == 0
    assert group_size % 2 == 0

    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    
    wmax = torch.abs(w).max(dim=1).values
    
    scale = wmax / -7.5
    
    quant = w / scale[:,None]

    uint8val = (quant + 8).round().to(torch.uint8).clamp(0, 15)
    
    fp32val = ((uint8val.float() - 8) * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    
    err = torch.abs(fp32valr - w).max(dim=1).values
    
    maxerr = err.max().item()

    # Pack two int4 values into one byte (int8)
    
    uint8val = uint8val.view(quant.shape[0], quant.shape[1]//2, 2)

    packed_quant = torch.zeros(quant.shape[0], quant.shape[1]//2, dtype=torch.uint8)

    packed_quant = uint8val[..., 0] | (uint8val[..., 1] << 4)
    
    return packed_quant, scale, maxerr 

# https://github.com/karpathy/llama2.c/blob/master/export.py
def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    
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