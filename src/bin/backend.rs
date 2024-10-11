use clap::Parser;
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use futures_util::{StreamExt, SinkExt};
use tokio_tungstenite::tungstenite::Result;
use tokio_tungstenite::tungstenite::Message;
#[cfg(feature = "backend-multimodal")]
use image::ImageReader;
use serde::{Deserialize, Serialize};
use base64::prelude::*;

use lmrs::transformer::Transformer;
use lmrs::transformer::ModelType;
use lmrs::tokenizer::Tokenizer;
use lmrs::sampler::Sampler;

#[cfg(feature = "backend-multimodal")]
use lmrs::vision::VisionTransformer;
#[cfg(feature = "backend-multimodal")]
use lmrs::processor::PHI3VProcessor;

use std::time::{SystemTime, UNIX_EPOCH};
use std::fs::File;
use memmap2::Mmap;
use chrono::Local;
use std::fs;
#[cfg(feature = "backend-multimodal")]
use std::io::Cursor;

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
    #[arg(long, default_value_t = false)]
    multimodal: bool,
    #[arg(long, default_value_t = 1)]
    num_crops: u32,
}

#[derive(Deserialize)]
struct ChatMessage {
    image: Option<String>, // Base64-encoded image, optional
    text: String,          // Text prompt
}

#[derive(Serialize)]
struct ResponseMessage {
    category: MessageCategory,
    text: String,
}

#[derive(Serialize)]
enum MessageCategory {
    STATUS,
    OUTPUT,
    FEATURE
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
            
            assert!(fs::metadata(tokenizer_path).is_ok(), "Tokenizer file not found: {}", tokenizer_path);
            assert!(fs::metadata(model_path).is_ok(), "Model file not found: {}", model_path);
            
            let file = File::open(&model_path).expect("Error opening model file");
            let data = unsafe { Mmap::map(&file).expect("MMap failed")  };

            let ws_stream = accept_async(stream).await.expect("Failed to accept");
            let (mut write, mut read) = ws_stream.split();

            let mut response = ResponseMessage {
                category: MessageCategory::STATUS,
                text: String::new(),
            };

            let mut tokenizer = Tokenizer::new(tokenizer_path);

            let (mut model, _offset_transformer) = Transformer::new(&data);

            #[cfg(feature = "backend-multimodal")]
            let mut vision_model: Option<VisionTransformer> = None;
            #[cfg(feature = "backend-multimodal")]
            let mut processor: Option<PHI3VProcessor> = None;
            #[cfg(feature = "backend-multimodal")]
            let mut image_pos = 0;
             
            #[cfg(feature = "backend-multimodal")]
            if !model.args.multimodal && args.multimodal {
                eprintln!("Current model doesn't support multimodality.");
                std::process::exit(1);
            } 
            
            #[cfg(feature = "backend-multimodal")]
            if args.multimodal {
                let vision_result = VisionTransformer::new(&data[_offset_transformer..]);
                vision_model = Some(vision_result.0);
                let processor_result = PHI3VProcessor::new(&data[vision_result.1 + _offset_transformer..]);
                processor = Some(processor_result);
                
                response.category = MessageCategory::FEATURE;
                response.text = String::from("multimodal");

                if write.send(serde_json::to_string(&response).unwrap().into()).await.is_err() {
                    return;
                }
            }

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
            let mut pos = 0;
 
            while let Some(msg) = read.next().await {
                let mut message_text: String = String::from("");

                match msg {
                    Ok(Message::Text(text)) => {
                        match serde_json::from_str::<ChatMessage>(&text) {
                            Ok(parsed_message) => {
                                message_text = parsed_message.text; 

                                #[cfg(feature = "backend-multimodal")]
                                if let Some(vision_model) = &mut vision_model {
                                    if let Some(processor) = &mut processor {
                                        if let Some(image_data) = parsed_message.image {
                                            let image_data = BASE64_STANDARD.decode(image_data).expect("Image must be in base64");

                                            let cursor = Cursor::new(image_data);

                                            let img = ImageReader::new(cursor).with_guessed_format().expect("Image format not supported.").decode().expect("Error decoding image.");

                                            let rgb_image = img.to_rgb8();

                                            let (width, height) = rgb_image.dimensions();

                                            let pixels: &[u8] = rgb_image.as_raw();

                                            response.category = MessageCategory::STATUS;
                                            response.text = String::from("Preprocessing the image");

                                            if write.send(serde_json::to_string(&response).unwrap().into()).await.is_err() {
                                                break;
                                            }

                                            let (patches, w_crop, h_crop, num_crops_processed) = processor.process(pixels, width, height, vision_model.args.patch_size, args.num_crops);

                                            response.category = MessageCategory::STATUS;
                                            response.text = String::from("Encoding the image");

                                            if write.send(serde_json::to_string(&response).unwrap().into()).await.is_err() {
                                                break;
                                            }

                                            let (patch_embeddings, patch_emb_shape) = vision_model.forward(&patches, num_crops_processed);

                                            let image_features = processor.forward(&patch_embeddings, patch_emb_shape, vision_model.args.image_size/vision_model.args.patch_size/2, w_crop, h_crop);

                                            let mut prefix = model.get_embeddings(&[1, 32010, 29871, 13]);

                                            let suffix = model.get_embeddings(&[1, 29871, 13]);

                                            prefix.extend(image_features);
                                            prefix.extend(suffix);

                                            response.category = MessageCategory::STATUS;
                                            response.text = String::from("Filling KV cache");

                                            if write.send(serde_json::to_string(&response).unwrap().into()).await.is_err() {
                                                break;
                                            }

                                            pos += model.fill_kv_cache(&mut prefix, pos);
                                            image_pos += pos;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                println!("Error parsing message: {:?}", e);
                            }
                        }
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
                    prompt_tokens.extend([128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220]);
                    
                    let today = Local::now().date_naive();
                    let formatted_date = today.format("%d %b %Y").to_string();
                    prompt_tokens.extend(tokenizer.encode(&formatted_date, false, false, false, model.args.model_type));

                    prompt_tokens.extend([271, 128009])
                }
                
                #[cfg(feature = "backend-multimodal")]
                if args.multimodal && pos == image_pos {
                    prompt_tokens.extend(tokenizer.encode(message_text.trim(), false, false, false, model.args.model_type));
                    prompt_tokens.extend([32007, 29871, 13, 32001, 29871, 13]);
                } else {
                    prompt_tokens.extend(tokenizer.encode(message_text.trim(), false, false, true, model.args.model_type));
                }
                
                #[cfg(not(feature = "backend-multimodal"))]
                prompt_tokens.extend(tokenizer.encode(message_text.trim(), false, false, true, model.args.model_type));
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

                    if user_idx >= num_prompt_tokens && next != tokenizer.eos && !(model.args.model_type == ModelType::GEMMA && next == 107) {
                        let piece = tokenizer.decode(next);
                        
                        response.category = MessageCategory::OUTPUT;
                        response.text = piece;

                        if write.send(serde_json::to_string(&response).unwrap().into()).await.is_err() {
                            break;
                        }
                    }   
                } 
                
                response.category = MessageCategory::OUTPUT;
                response.text = String::from("<eos>");

                if write.send(serde_json::to_string(&response).unwrap().into()).await.is_err() {
                    break;
                }
                
                println!("Done!\n");
            }
        });
    }
    
    Ok(())
}
