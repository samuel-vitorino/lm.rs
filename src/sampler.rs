use crate::functional::random_f32;
use crate::functional::softmax;

#[derive(Debug, Copy, Clone)]
struct ProbIndex {
    prob: f32,
    index: u32,
}

pub struct Sampler {
    vocab_size: u32,
    probindex: Vec<ProbIndex>,
    temperature: f32,
    top_p: f32,
    seed: u64,
}

impl Sampler {
    pub fn new(vocab_size: u32, temperature: f32, top_p: f32, seed: u64) -> Sampler {
        Sampler {
            vocab_size,
            probindex: vec![
                ProbIndex {
                    prob: 0.0,
                    index: 0
                };
                vocab_size as usize
            ],
            temperature,
            top_p,
            seed,
        }
    }

    fn sample_argmax(probabilities: &[f32]) -> u32 {
        let mut max_i: u32 = 0;
        let mut max_p = probabilities[0];

        for (i, p) in probabilities.iter().enumerate().skip(1) {
            if *p > max_p {
                max_i = i as u32;
                max_p = *p;
            }
        }

        max_i
    }

    fn sample_mult(probabilities: &[f32], rand: f32) -> u32 {
        let mut cdf: f32 = 0.0;
        let n = probabilities.len();

        for (i, p) in probabilities.iter().enumerate() {
            cdf += *p;
            if rand < cdf {
                return i as u32;
            }
        }

        (n - 1) as u32
    }

    fn compare(a: &ProbIndex, b: &ProbIndex) -> std::cmp::Ordering {
        if a.prob > b.prob {
            std::cmp::Ordering::Less
        } else if a.prob < b.prob {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }

    fn sample_topp(&mut self, probabilities: &[f32], top_p: f32, rand: f32) -> u32 {
        let n = probabilities.len();
        let mut n0 = 0;

        let cutoff: f32 = (1.0f32 - top_p) / (n - 1) as f32;

        for (i, p) in probabilities.iter().enumerate() {
            if *p >= cutoff {
                self.probindex[n0].index = i as u32;
                self.probindex[n0].prob = *p;
                n0 += 1;
            }
        }

        self.probindex.sort_by(Sampler::compare);

        let mut cumulative_prob: f32 = 0.0;

        let mut last_idx = n0 - 1;

        for i in 0..n0 {
            cumulative_prob += self.probindex[i].prob;
            if cumulative_prob > top_p {
                last_idx = i;
                break;
            }
        }

        let r = rand * cumulative_prob;
        let mut cdf: f32 = 0.0;

        for i in 0..last_idx + 1 {
            cdf += self.probindex[i].prob;
            if r < cdf {
                return self.probindex[i].index;
            }
        }

        self.probindex[last_idx].index
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let next: u32;

        if self.temperature == 0.0f32 {
            next = Sampler::sample_argmax(logits);
        } else {
            for q in 0..self.vocab_size {
                logits[q as usize] /= self.temperature;
            }

            softmax(logits);

            let rand: f32 = random_f32(self.seed);

            if self.top_p <= 0.0 || self.top_p >= 1.0 {
                next = Sampler::sample_mult(logits, rand);
            } else {
                next = self.sample_topp(logits, self.top_p, rand);
            }
        }

        next
    }
}
