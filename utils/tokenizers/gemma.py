import os
import struct
from transformers import AutoTokenizer
from typing import List

# Convert sentencepiece tokenizer model into lmrs format (slight modification from karpathy's code)

class Tokenizer:
    def __init__(self, model_id):
        self.sp_model = AutoTokenizer.from_pretrained(model_id, use_fast=False).sp_model

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def export(self):

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        with open("tokenizer.bin", 'wb') as f:
            f.write(struct.pack("IIII", self.n_words, max_token_length, self.bos_id, self.eos_id))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)