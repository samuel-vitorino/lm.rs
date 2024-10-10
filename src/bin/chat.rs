use lmrs::sampler::Sampler;
use lmrs::tokenizer::Tokenizer;
use lmrs::transformer::ModelType;
use lmrs::transformer::Transformer;

use chrono::Local;
use clap::Parser;
use memmap2::Mmap;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long, default_value_t = String::from("tokenizer.bin"))]
    tokenizer: String,
    #[arg(long, default_value_t = 1.0f32)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9f32)]
    top_p: f32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = false)]
    show_metrics: bool,
}

fn main() {
    let logo = r#"
    L      M     M  RRRR    ssss
    L      MM   MM  R   R  s
    L      M M M M  RRRR    sss
    L      M  M  M  R  R       s
    LLLL   M     M  R   R  sssss
    "#;

    println!("{}", logo);

    let args = Args::parse();
    let model_path: &str = args.model.as_str();
    let tokenizer_path: &str = args.tokenizer.as_str();

    assert!(
        fs::metadata(tokenizer_path).is_ok(),
        "Tokenizer file not found: {}",
        tokenizer_path
    );
    assert!(
        fs::metadata(model_path).is_ok(),
        "Model file not found: {}",
        model_path
    );

    let mut tokenizer = Tokenizer::new(args.tokenizer.as_str());

    let file = File::open(model_path).expect("Error opening model file");
    let data = unsafe { Mmap::map(&file).expect("MMap failed") };

    let mut model = Transformer::new(&data);

    let seed: u64 = match args.seed {
        Some(seed_value) => seed_value,
        None => {
            let start = SystemTime::now();
            let since_epoch = start
                .duration_since(UNIX_EPOCH)
                .expect("Error getting time since epoch");

            since_epoch.as_millis() as u64
        }
    };

    let mut sampler = Sampler::new(model.args.vocab_size, args.temperature, args.top_p, seed);

    let mut user_turn = true;
    let mut user_idx: usize = 0;
    let mut pos = 0;
    let mut token: u32;
    let mut next: u32 = 0;
    let mut num_prompt_tokens = 0;
    let mut total_tokens: f32 = 0.0;
    let mut total_duration: f32 = 0.0;

    let mut prompt_tokens: Vec<u32> = Vec::new();

    let mut user_prompt: String;

    loop {
        if user_turn {
            user_prompt = String::from("");

            print!("You: ");
            io::stdout().flush().unwrap();

            io::stdin()
                .read_line(&mut user_prompt)
                .expect("Failed to read line");

            // System prompt
            if model.args.model_type == ModelType::LLAMA && pos == 0 {
                // First part of chat template with initial tags and cut off date
                prompt_tokens.extend([
                    128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220,
                    2366, 18, 198, 15724, 2696, 25, 220,
                ]);

                let today = Local::now().date_naive();
                let formatted_date = today.format("%d %b %Y").to_string();
                prompt_tokens.extend(tokenizer.encode(
                    &formatted_date,
                    false,
                    false,
                    false,
                    model.args.model_type,
                ));

                prompt_tokens.extend([271, 128009])
            }

            prompt_tokens.extend(tokenizer.encode(
                user_prompt.trim(),
                false,
                false,
                true,
                model.args.model_type,
            ));
            num_prompt_tokens = prompt_tokens.len();

            user_turn = false;
            user_idx = 0;

            println!("Assistant:");
        }

        if user_idx < num_prompt_tokens {
            token = prompt_tokens[user_idx];
            user_idx += 1;
        } else {
            token = next;
        }

        if token == tokenizer.eos && user_idx >= num_prompt_tokens {
            user_turn = true;
            println!();
            prompt_tokens = Vec::new();

            if args.show_metrics {
                let toks = total_tokens / (total_duration / 1000.0);

                println!("Speed: {:.2} tok/s", toks);

                total_duration = 0.0;
                total_tokens = 0.0;
            }

            continue;
        }

        let processing_start = Instant::now();

        let logits: &mut [f32] = model.forward(token, pos);
        next = sampler.sample(logits);
        pos += 1;

        if user_idx >= num_prompt_tokens
            && next != tokenizer.eos
            && !(model.args.model_type == ModelType::GEMMA && next == 107)
        {
            let piece = tokenizer.decode(next);
            print!("{}", piece);
            io::stdout().flush().unwrap();
        }

        let duration = processing_start.elapsed();
        total_duration += duration.as_millis() as f32;
        total_tokens += 1.0;
    }
}
