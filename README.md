<div align="center">

<picture>
    <img alt="lmrs logo" src="repo_cover.svg">
</picture>

lm.rs: run inference on Language Models locally on the CPU with Rust

<h3>

[WebUI](https://github.com/samuel-vitorino/lm.rs-webui) | [Hugging Face](https://huggingface.co/collections/samuel-vitorino/lmrs-66c7da8a50ce52b61bee70b7) | [Video Demo](https://www.youtube.com/watch?v=FAIN5Jxc0nE) 

</h3>

</div>

---

**Now supporting LLama3.2 1B and 3B models! [WebUI](https://github.com/samuel-vitorino/lm.rs-webui) also now available.**

Inspired by Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and [llm.c](https://github.com/karpathy/llm.c) I decided to create the most minimal code (not so minimal atm) that can perform full inference on Language Models on the CPU. Previously only Google's Gemma 2 models were supported, but I decided to add support for the new Llama 3.2 models.

Disclaimer: some of the code could be optimized and improved. This is just an excuse for me to write Rust for the first time. Isn't it incredible that in a few years, we could have AGI running in a few lines of poorly written Rust code?

## Prepared models

Some benchmarks and download links for the models and tokenizers. I recommend using Q8_0, Q4_0 quantization still being improved. Speed measured on a 16-core AMD Epyc.

|        Model       | Size | Speed |
| ------------------ | ------------- | ------------- |
| [Gemma 2 2B IT Q4_0](https://huggingface.co/samuel-vitorino/gemma2-2b-it-q4_0-LMRS) | 1.39G          | 20 tok/s |
| [Gemma 2 2B IT Q8_0](https://huggingface.co/samuel-vitorino/gemma2-2b-it-q8_0-LMRS) | 2.66GB  | 18 tok/s |
| [Gemma 2 9B IT Q4_0](https://huggingface.co/samuel-vitorino/gemma2-9b-it-q4_0-LMRS) | 4.91GB  | 7 tok/s  | 
| [Gemma 2 9B IT Q8_0](https://huggingface.co/samuel-vitorino/gemma2-9b-it-q8_0-LMRS) | 9.53GB | 8 tok/s  |
| [Llama 3.2 1B IT](https://huggingface.co/samuel-vitorino/Llama-3.2-1B-Instruct-LMRS) | 4.94GB  | 20 tok/s  | 
| [Llama 3.2 1B IT Q8_0](https://huggingface.co/samuel-vitorino/Llama-3.2-1B-Instruct-Q8_0-LMRS) | 1.27GB | 35 tok/s  |
| [Llama 3.2 3B IT Q4_0](https://huggingface.co/samuel-vitorino/Llama-3.2-3B-Instruct-Q4_0-LMRS) | 1.71GB  | 17 tok/s  | 
| [Llama 3.2 3B IT Q8_0](https://huggingface.co/samuel-vitorino/Llama-3.2-3B-Instruct-Q8_0-LMRS) | 3.31GB | 16 tok/s  |

## Instructions

You can download the prepared quantized model and tokenizer model files in the lmrs format from Hugging Face. If you'd prefer to convert the models published by Google/Meta on Hugging Face yourself, please refer to the following section. Otherwise, you can skip ahead to the build section.

### Model Conversion

Install additional python dependencies (assuming you already have pytorch installed) used in export.py and tokenizer.py:

```properties
pip install -r requirements.txt
```

Download the **.safetensors**, **config.json** and **tokenizer.model** files from the original model's page on Hugging Face (So we don't have to clone the pytorch repo). On llama's repo, the tokenizer.model is inside the **original** folder.

Use the export.py script to convert the model bfloat16 weights into the LMRS format:

```properties
python export.py --files [ordered .safetensor files] --config [model config.json] --save-path [name and path to save] --type [model type (GEMMA/LLAMA)]
```

To export the quantized version use the --quantize and --quantize-type flags. The int8 quantized model size should be 4X smaller (from ~9.8G to ~2.5G, depending on the group size).

Use the tokenizer.py script to convert the tokenizer.model sentencepiece tokenizer into the LMRS tokenizer format:

```properties
python tokenizer.py --tokenizer-model [path to the tokenizer.model file] --tokenizer-type [type of the tokenizer (GEMMA/LLAMA)]
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

## TODOs

Some things to do in the future:

- [X] Add other sampling methods.
- [X] Test the 9B and 27B models (tested the 9B, 27B would be too slow).
- [X] Parallelize the multi head attention loop.
- [X] Add performance metrics.
- [ ] Ability to give a system prompt
- [X] Quantization support (int8, int4).

## License

MIT





