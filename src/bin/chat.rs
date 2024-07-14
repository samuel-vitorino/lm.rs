use llmrs::transformer::Transformer;
use llmrs::tokenizer::Tokenizer;

use std::env;
use std::time::Instant;
use std::fs::File;
use memmap2::Mmap;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path: &str = &args[1];

    let mut tokenizer = Tokenizer::new("tokenizer.bin");
        
    let file = File::open(model_path).expect("Model file required");
    let data = unsafe { Mmap::map(&file).expect("MMap failed")  };

    let mut model = Transformer::new(&data);

    let tokens = tokenizer.encode("This message is tokenized! さあ行こう", true, false);

    println!("{:?}", tokens);

    for i in 1..tokens.len() {
        print!("{}", tokenizer.decode(tokens[i-1], tokens[i]));
    }

    println!("");


    for i in 0..1 {
        let start = Instant::now();

        model.forward(tokens[0], 0);

        let end = Instant::now();

        let duration = end.duration_since(start);

        let in_seconds = duration.as_secs_f64();

        println!("Elapsed time in seconds: {:.3}", in_seconds);
    }
}