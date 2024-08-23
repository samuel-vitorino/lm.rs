use lmrs::transformer::Transformer;
use lmrs::tokenizer::Tokenizer;
use lmrs::sampler::Sampler;

use std::fs;
use std::io;
use std::io::Write;
use std::fs::File;
use clap::Parser;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use memmap2::Mmap;

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

    assert!(fs::metadata(tokenizer_path).is_ok(), "Tokenizer file not found: {}", tokenizer_path);
    assert!(fs::metadata(model_path).is_ok(), "Model file not found: {}", model_path);

    let mut tokenizer = Tokenizer::new(args.tokenizer.as_str());

    let file = File::open(model_path).expect("Error opening model file");
    let data = unsafe { Mmap::map(&file).expect("MMap failed")  };

    let mut model = Transformer::new(&data);

    let seed: u64;

    match args.seed {
        Some(seed_value) => {
            seed = seed_value;
        }
        None => {
            let start = SystemTime::now();
            let since_epoch = start.duration_since(UNIX_EPOCH).expect("Error getting time since epoch");
            seed = since_epoch.as_millis() as u64;
        }
    }

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

            io::stdin().read_line(&mut user_prompt).expect("Failed to read line");

            // Even when using gemma 2b-it you can do chat_format = false and use text completion
            prompt_tokens = tokenizer.encode(user_prompt.trim(), true, false, true);
            num_prompt_tokens = prompt_tokens.len();

            user_turn = false; 
            user_idx = 0;
            
            print!("Assistant: \n");
            io::stdout().flush().unwrap();
        }

        if user_idx < num_prompt_tokens {
            token = prompt_tokens[user_idx];
            user_idx += 1;
        } else {
            token = next;
        }

        if token == 1 { 
            user_turn = true; 
            println!("");
            if args.show_metrics {
                let toks = total_tokens/(total_duration/1000.0);
                
                println!("Speed: {:.2} tok/s", toks);

                total_duration = 0.0;
                total_tokens = 0.0;
            } 
        }
        
        let processing_start = Instant::now();

        let logits: &mut [f32] = model.forward(token, pos);
        next = sampler.sample(logits);
        pos += 1;

        if user_idx >= num_prompt_tokens && next != 1 && token != 107 {
            let piece = tokenizer.decode(token);
            print!("{}", piece);
            io::stdout().flush().unwrap();
        }   

        let duration = processing_start.elapsed();
        total_duration += duration.as_millis() as f32;
        total_tokens += 1 as f32;
    }
}