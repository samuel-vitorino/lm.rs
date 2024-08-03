# Implementation of Gemma 2 compatible inference code in Rust

![Gemma greeting the user](repo_cover.png)

**Now supporting Gemma 2 2b! For Gemma 1 2b refer to the gemma1 branch.**

Inspired by Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and [llm.c](https://github.com/karpathy/llm.c) I decided to create the most minimal code that can perform full inference on Google's Gemma models on the CPU (only tested the 2B-it).

Disclaimer: most of the code could be optimized and improved (runs at 3 tok/s on my 8-core laptop). This is just an excuse for me to write Rust for the first time. Isn't it incredible that in a few years, we could have AGI running with just **~600** lines of poorly written Rust code?

Some things to do in the future:

- [X] Add other sampling methods.
- [ ] Test the 9B and 27B models (would probably have to change the code).
- [ ] Parallelize the multi head attention loop.
- [X] Add performance metrics.
- [ ] Ability to give a system prompt
- [ ] Rust language optimizations.

## Instructions

Install additional python dependencies (assuming you already have pytorch installed) used in export.py and tokenizer.py:

```properties
pip install -r requirements.txt
```

Download the .safetensors, config.json and tokenizer.model files from [huggingface](https://huggingface.co/google/gemma-2-2b-it) (So we don't have to clone the pytorch repo).

Use the export.py script to convert the model bfloat16 weights into float32 LMRS format:

```properties
python export.py --files [ordered .safetensor files] --config [model config.json] --save_path [name and path to save]
```

Use the tokenizer.py script to convert the tokenizer.model sentencepiece tokenizer into the LMRS tokenizer format:

```properties
python tokenizer.py
```

Finally compile the rust code with cargo:

```properties
cargo build --release
```

And you are good to go:

```properties
./target/release/chat --model [model weights file]
```

Other arguments include tokenizer, temperature, top-p, show-metrics etc. To check available arguments run with --help.

## License

MIT





