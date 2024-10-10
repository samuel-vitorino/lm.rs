use lmrs::transformer::ModelType;
use lmrs::transformer::Transformer;
use lmrs::tokenizer::Tokenizer;
use lmrs::sampler::Sampler;

#[cfg(feature = "multimodal")]
use lmrs::vision::VisionTransformer;
#[cfg(feature = "multimodal")]
use lmrs::processor::PHI3VProcessor;
#[cfg(feature = "multimodal")]
use image::open;

use std::fs;
use std::io;
use std::io::Write;
use std::fs::File;
use clap::Parser;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use memmap2::Mmap;
use chrono::Local;

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
    #[arg(long)]
    image: Option<String>,
    #[arg(long, default_value_t = 2)]
    num_crops: u32,
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

    let (mut model, _offset_transformer) = Transformer::new(&data);
    
    let mut pos = 0;

    #[cfg(feature = "multimodal")]
    let mut image_path: String = String::new();
    #[cfg(feature = "multimodal")]
    let mut image_pos = 0;
    
    #[cfg(feature = "multimodal")]
    match args.image {
        Some(image_value) => {
            image_path = image_value;
        }
        None => {
        }
    };
    
    #[cfg(feature = "multimodal")]
    if !image_path.is_empty() {
        if !model.args.multimodal {
            eprintln!("Cannot use images in a non-multimodal model.");
            std::process::exit(1);
        }

        let (mut vision_model, offset_vision) = VisionTransformer::new(&data[_offset_transformer..]);
        let processor = PHI3VProcessor::new(&data[offset_vision + _offset_transformer..]);

        let img = open(image_path.clone()).expect("Image file not found!");
        let rgb_image = img.to_rgb8();

        let (width, height) = rgb_image.dimensions();

        let pixels: &[u8] = rgb_image.as_raw();

        println!("Preprocessing the image...");

        let (patches, w_crop, h_crop, num_crops_processed) = processor.process(pixels, width, height, vision_model.args.patch_size, args.num_crops);

        println!("Encoding the image...");

        let (patch_embeddings, patch_emb_shape) = vision_model.forward(&patches, num_crops_processed);

        let image_features = processor.forward(&patch_embeddings, patch_emb_shape, vision_model.args.image_size/vision_model.args.patch_size/2, w_crop, h_crop);

        let mut prefix = model.get_embeddings(&[1, 32010, 29871, 13]);

        let suffix = model.get_embeddings(&[1, 29871, 13]);

        prefix.extend(image_features);
        prefix.extend(suffix);

        println!("Filling KV Cache...\n");

        pos = model.fill_kv_cache(&mut prefix, 0);
        image_pos = pos;
    }

    let seed: u64 = match args.seed {
        Some(seed_value) => {
            seed_value
        }
        None => {
            let start = SystemTime::now();
            let since_epoch = start.duration_since(UNIX_EPOCH).expect("Error getting time since epoch");
            
            since_epoch.as_millis() as u64
        }
    };

    let mut sampler = Sampler::new(model.args.vocab_size, args.temperature, args.top_p, seed);

    let mut user_turn = true;
    let mut user_idx: usize = 0;
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
            
            // System prompt
            if model.args.model_type == ModelType::LLAMA && pos == 0 {
                // First part of chat template with initial tags and cut off date
                prompt_tokens.extend([128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220]);
                
                let today = Local::now().date_naive();
                let formatted_date = today.format("%d %b %Y").to_string();
                prompt_tokens.extend(tokenizer.encode(&formatted_date, false, false, false, model.args.model_type));

                prompt_tokens.extend([271, 128009])
            }
            
            #[cfg(feature = "multimodal")]
            if !image_path.is_empty() && pos == image_pos {
                prompt_tokens.extend(tokenizer.encode(user_prompt.trim(), false, false, false, model.args.model_type));
                prompt_tokens.extend([32007, 29871, 13, 32001, 29871, 13]);
            } else {
                prompt_tokens.extend(tokenizer.encode(user_prompt.trim(), false, false, true, model.args.model_type));
            }

            #[cfg(not(feature = "multimodal"))]
            prompt_tokens.extend(tokenizer.encode(user_prompt.trim(), false, false, true, model.args.model_type));
            
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
                let toks = total_tokens/(total_duration/1000.0);
                
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

        if user_idx >= num_prompt_tokens && next != tokenizer.eos && !(model.args.model_type == ModelType::GEMMA && next == 107) {
            let piece = tokenizer.decode(next);
            print!("{}", piece);
            io::stdout().flush().unwrap();
        }   

        let duration = processing_start.elapsed();
        total_duration += duration.as_millis() as f32;
        total_tokens += 1.0;
    }
}