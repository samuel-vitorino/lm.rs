import os
import struct
from pathlib import Path

from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_id):
        self.model = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        self.n_words = len(self.model)
        
        self.bos_id = self.model.bos_token_id
        self.eos_id = self.model.eos_token_id

    def export(self):
        tokens, scores = [], []
        for i in range(self.n_words):
            t = self.model.decode([i])
            
            # just for easier compatibility with sentencepiece tokenizers
            s = 1.0
            
            b = t.encode('utf-8') 

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        with open("tokenizer.bin", 'wb') as f:
            f.write(struct.pack("IIII", self.n_words, max_token_length, self.bos_id, self.eos_id))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)