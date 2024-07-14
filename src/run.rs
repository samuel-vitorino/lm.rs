mod transformer;
mod tokenizer;
mod functional;

use tokenizer::Tokenizer;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path: &str = &args[1];

    println!("{}", model_path);

    //io::parse_model(model_path);

    //Load tokenizer;
    let mut t = Tokenizer::new("tokenizer.bin");

    let tokens = t.encode("This message is tokenized! さあ行こう", true, false);

    println!("{:?}", tokens);

    for i in 1..tokens.len() {
        print!("{}", t.decode(tokens[i-1], tokens[i]));
    }

    println!("");
}