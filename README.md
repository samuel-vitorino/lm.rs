# Implementation of Gemma 2 compatible inference code in Rust

![Gemma greeting the user](repo_cover.png)

**Now supporting Gemma 2 2B and quantization! For Gemma 1 2B refer to the gemma1 branch. [WebUI](https://github.com/samuel-vitorino/lm.rs-webui) also now available. [Terminal Chat Video Demo](https://www.youtube.com/watch?v=3HHl2KSPAc8).** 

Inspired by Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and [llm.c](https://github.com/karpathy/llm.c) I decided to create the most minimal code that can perform full inference on Google's Gemma models on the CPU (only tested the 2B-it).

Disclaimer: most of the code could be optimized and improved (now running at 10.56 tok/s on my 8-core laptop at Q8_0 quantization, and at 20 tok/s on a 16-core AMD EPYC). This is just an excuse for me to write Rust for the first time. Isn't it incredible that in a few years, we could have AGI running in a few lines of poorly written Rust code?

Some things to do in the future:

- [X] Add other sampling methods.
- [ ] Test the 9B and 27B models (would probably have to change the code).
- [X] Parallelize the multi head attention loop.
- [X] Add performance metrics.
- [ ] Ability to give a system prompt
- [X] Quantization support (int8).

## Instructions

You can download the prepared gemma2-2b int8 quantized model and tokenizer model files in the lmrs format from [huggingface](https://huggingface.co/samuel-vitorino/gemma2-2b-it-q8_0-LMRS). If you'd prefer to convert the model published by Google on [huggingface](https://huggingface.co/google/gemma-2-2b-it) yourself, please refer to the following section. Otherwise, you can skip ahead to the build section.

### Model Conversion

Install additional python dependencies (assuming you already have pytorch installed) used in export.py and tokenizer.py:

```properties
pip install -r requirements.txt
```

Download the .safetensors, config.json and tokenizer.model files from [huggingface](https://huggingface.co/google/gemma-2-2b-it) (So we don't have to clone the pytorch repo).

Use the export.py script to convert the model bfloat16 weights into the LMRS format:

```properties
python export.py --files [ordered .safetensor files] --config [model config.json] --save-path [name and path to save]
```

To export the int8 quantized version use the --quantize flag. The model size should be 4X smaller (from ~9.8G to ~2.5G, depending on the group size).

Use the tokenizer.py script to convert the tokenizer.model sentencepiece tokenizer into the LMRS tokenizer format:

```properties
python tokenizer.py
```

### Build

Compile the rust code with cargo (make sure to pass the target-cpu flag):

```properties
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin chat
```

And you are good to go:

```properties
./target/release/chat --model [model weights file]
```

Other arguments include tokenizer, temperature, top-p, show-metrics etc. To check available arguments run with --help.

---

To run the backend for the [WebUI](https://github.com/samuel-vitorino/lm.rs-webui), first compile:

```properties
RUSTFLAGS="-C target-cpu=native" cargo build --release --features backend --bin backend
```

Then run:

```properties
./target/release/backend --model [model weights file]
```

You can change the ip and port with --ip and --port. Other flags such as temperature, etc. are also available. You can now connect via the web interface.

## License

MIT





