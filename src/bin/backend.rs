use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::Result;

use lmrs::sampler::Sampler;
use lmrs::tokenizer::Tokenizer;
use lmrs::transformer::ModelType;
use lmrs::transformer::Transformer;

use chrono::Local;
use memmap2::Mmap;
use std::fs;
use std::fs::File;
use std::time::{SystemTime, UNIX_EPOCH};

/// Simple WebSocket server
#[derive(Parser)]
#[command(name = "lmrs-api")]
#[command(version = "1.0")]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1")]
    ip: String,
    #[arg(short, long, default_value = "8080")]
    port: u16,
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let addr = format!("{}:{}", args.ip, args.port);
    let listener = TcpListener::bind(&addr).await.expect("Failed to bind");

    println!("Listening on: {}", addr);

    while let Ok((stream, _)) = listener.accept().await {
        tokio::spawn(async move {
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

            let file = File::open(&model_path).expect("Error opening model file");
            let data = unsafe { Mmap::map(&file).expect("MMap failed") };

            let ws_stream = accept_async(stream).await.expect("Failed to accept");
            let (mut write, mut read) = ws_stream.split();

            let mut tokenizer = Tokenizer::new(tokenizer_path);

            let mut model = Transformer::new(&data);

            let seed: u64;

            match args.seed {
                Some(seed_value) => {
                    seed = seed_value;
                }
                None => {
                    let start = SystemTime::now();
                    let since_epoch = start
                        .duration_since(UNIX_EPOCH)
                        .expect("Error getting time since epoch");
                    seed = since_epoch.as_millis() as u64;
                }
            }

            let mut sampler =
                Sampler::new(model.args.vocab_size, args.temperature, args.top_p, seed);
            let mut pos = 0;

            while let Some(msg) = read.next().await {
                let message_text: String;

                match msg {
                    Ok(Message::Text(text)) => {
                        message_text = text;
                    }
                    Ok(Message::Close(reason)) => {
                        println!("Connection closed: {:?}", reason);
                        break;
                    }
                    Err(e) => {
                        eprintln!("Error receiving message: {}", e);
                        break;
                    }
                    _ => {
                        continue;
                    }
                }

                let mut token: u32;
                let mut next: u32 = 0;
                let num_prompt_tokens;
                let mut user_idx = 0;

                let mut prompt_tokens: Vec<u32> = Vec::new();

                println!("Processing prompt: {}", message_text);

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
                    message_text.trim(),
                    false,
                    false,
                    true,
                    model.args.model_type,
                ));
                num_prompt_tokens = prompt_tokens.len();

                while next != tokenizer.eos || user_idx < num_prompt_tokens {
                    if user_idx < num_prompt_tokens {
                        token = prompt_tokens[user_idx];
                        user_idx += 1;
                    } else {
                        token = next;
                    }

                    if token == tokenizer.eos && user_idx >= num_prompt_tokens {
                        break;
                    }

                    let logits: &mut [f32] = model.forward(token, pos);
                    next = sampler.sample(logits);
                    pos += 1;

                    if user_idx >= num_prompt_tokens
                        && next != tokenizer.eos
                        && !(model.args.model_type == ModelType::GEMMA && next == 107)
                    {
                        let piece = tokenizer.decode(next);
                        if write.send(piece.into()).await.is_err() {
                            break;
                        }
                    }
                }

                if write.send("<eos>".into()).await.is_err() {
                    break;
                }

                println!("Done!\n");
            }
        });
    }

    Ok(())
}
