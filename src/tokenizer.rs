use crate::functional::slice_to_f32;
use crate::functional::slice_to_u32;
use crate::transformer::ModelType;
use std::fs;

#[derive(Debug, Clone)]
struct TokenIndex {
    text: String,
    id: u32,
}

pub struct Tokenizer {
    vocab_size: u32,
    vocab: Vec<String>,
    pub bos: u32,
    pub eos: u32,
    vocab_scores: Vec<f32>,
    sorted_vocab: Vec<TokenIndex>,
    //For now i don't use this, only allow seqs of max this size, future work
    //max_token_len: u32,
}

impl Tokenizer {
    pub fn new(path: &str) -> Tokenizer {
        let data: Vec<u8> = fs::read(path).expect("Error reading tokenizer file.");

        let vocab_size = slice_to_u32(&data[0..4]);
        //let max_token_len = slice_to_u32(&data[4..8]);
        let bos = slice_to_u32(&data[8..12]);
        let eos = slice_to_u32(&data[12..16]);
        let mut vocab: Vec<String> = vec![];
        let mut vocab_scores: Vec<f32> = vec![];
        let sorted_vocab: Vec<TokenIndex> = vec![];

        let mut offset: usize = 16;

        for _ in 0..vocab_size {
            let score = slice_to_f32(&data[offset..offset + 4]);

            vocab_scores.push(score);

            offset += 4;

            let str_len = slice_to_u32(&data[offset..offset + 4]);

            offset += 4;

            let token_str = String::from_utf8(data[offset..offset + str_len as usize].to_vec())
                .expect("Error reading token string");

            vocab.push(token_str);

            offset += str_len as usize;
        }

        Tokenizer {
            vocab_size,
            //max_token_len,
            vocab,
            bos,
            eos,
            vocab_scores,
            sorted_vocab,
        }
    }

    pub fn encode(
        &mut self,
        text: &str,
        bos: bool,
        eos: bool,
        chat_format: bool,
        model_type: ModelType,
    ) -> Vec<u32> {
        assert!(!text.is_empty(), "Text to encode should not be empty");

        if self.sorted_vocab.is_empty() {
            for i in 0..self.vocab_size as usize {
                self.sorted_vocab.push(
                    // Using clone, should point to the vocabs string, but cant deal with rust rn
                    TokenIndex {
                        text: self.vocab[i].clone(),
                        id: i as u32,
                    },
                )
            }
            self.sorted_vocab.sort_by(|a, b| a.text.cmp(&b.text));
        }

        let mut tokens: Vec<u32> = Vec::new();

        if bos {
            tokens.push(self.bos)
        }

        if chat_format {
            if model_type == ModelType::GEMMA {
                tokens.extend([self.bos, 106, 1645, 108]);
            } else if model_type == ModelType::LLAMA {
                tokens.extend([128006, 882, 128007, 271]);
            }
        }

        for c in text.chars() {
            let c_str = c.to_string();
            match self
                .sorted_vocab
                .binary_search_by(|token| token.text.cmp(&c_str))
            {
                Ok(index) => tokens.push(self.sorted_vocab[index].id),
                Err(_) => {
                    for b in c_str.into_bytes().iter() {
                        tokens.push(*b as u32 + 3)
                    }
                }
            }
        }

        loop {
            let mut best_score: f32 = -1e10;
            let mut best_id: u32 = 0;
            let mut best_idx: i32 = -1;

            for idx in 0..tokens.len() - 1 {
                let new_t = self.vocab[tokens[idx] as usize].clone()
                    + &self.vocab[tokens[idx + 1] as usize];

                if let Ok(index) = self
                    .sorted_vocab
                    .binary_search_by(|token| token.text.cmp(&new_t))
                {
                    let temp_t = &self.sorted_vocab[index];
                    if self.vocab_scores[temp_t.id as usize] > best_score {
                        best_score = self.vocab_scores[temp_t.id as usize];
                        best_id = temp_t.id;
                        best_idx = idx as i32;
                    }
                }
            }

            if best_idx == -1 {
                break;
            }

            tokens[best_idx as usize] = best_id;
            tokens.remove((best_idx + 1) as usize);
        }

        if chat_format {
            if model_type == ModelType::GEMMA {
                tokens.extend([107, 108, 106, 2516, 108]);
            } else if model_type == ModelType::LLAMA {
                tokens.extend([128009, 128006, 78191, 128007, 271]);
            }
        }

        if eos {
            tokens.push(self.eos)
        }

        tokens
    }

    pub fn decode(&self, token: u32) -> String {
        let piece = self.vocab[token as usize].to_string();

        if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
            if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
                return char::from(byte_val).to_string();
            }
        }
        piece
    }
}
