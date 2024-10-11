import os
import struct
from transformers import AutoTokenizer
from typing import List

class Tokenizer:
    def __init__(self, model_id=None):
        self.model_id = model_id

        if "vision" in model_id:
            self.special_tokens = AutoTokenizer.from_pretrained(model_id).added_tokens_decoder
            model_id = "microsoft/Phi-3.5-mini-instruct"
        else:
            self.special_tokens = AutoTokenizer.from_pretrained(model_id).added_tokens_decoder

        self.model = AutoTokenizer.from_pretrained(model_id, use_fast=False).sp_model
        self.n_words = self.model.vocab_size()

        self.n_words = self.model.vocab_size()
        
        self.bos_id = self.model.bos_id()
        self.eos_id = 32007

    def export(self):
        tokens, scores = [], []
        for i in range(self.n_words):
            t = self.model.id_to_piece(i)

            s = self.model.get_score(i)

            t = t.replace('▁', ' ')
            b = t.encode('utf-8')

            tokens.append(b)
            scores.append(s)

        for i in self.special_tokens.keys():
            if i <= self.n_words:
                continue

            t = self.special_tokens[i].content
            s = 0
            
            t = t.replace('▁', ' ')
            b = t.encode('utf-8')

            tokens.append(b)
            scores.append(s)

            self.n_words += 1

        # Temporary fix
        if self.model_id == "microsoft/Phi-3.5-mini-instruct":
            t = "<|placeholder7|>"
            b = t.encode('utf-8')

            tokens.append(b)
            scores.append(s)

            self.n_words += 1

        max_token_length = max(len(t) for t in tokens)

        with open("tokenizer.bin", 'wb') as f:
            f.write(struct.pack("IIII", self.n_words, max_token_length, self.bos_id, self.eos_id))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)