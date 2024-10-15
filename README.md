<div align="center">

<picture>
    <img alt="lmrs logo" src="repo_cover.svg">
</picture>

lm.rs: run inference on Language Models locally on the CPU with Rust

<h3>

[WebUI](https://github.com/samuel-vitorino/lm.rs-webui) | [Hugging Face](https://huggingface.co/collections/samuel-vitorino/lmrs-66c7da8a50ce52b61bee70b7) | [Text Demo](https://www.youtube.com/watch?v=PRptDEBzd4I) | [Vision Demo](https://www.youtube.com/watch?v=bz3f0PqG9Nk)

</h3>

</div>

---

**🌃 Now supporting multimodality with PHI-3.5-vision model! PHI-3.5-mini text-only model also now supported.**

Inspired by Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and [llm.c](https://github.com/karpathy/llm.c) I decided to create the most minimal code (not so minimal atm) that can perform full inference on Language Models on the CPU without ML libraries. Previously only Google's Gemma 2 models were supported, but I decided to add support for the new Llama 3.2 models, and more recently the option to use images with PHI-3.5.

**News:** Implemented batch processing, boosting the image encoding speed by up to ~3x. Llama 3.2 1B now runs at 50 tok/s on my 16-core machine.

**Disclaimer:** Some of the code could be optimized and improved. This is just an excuse for me to write Rust for the first time. Isn't it incredible that in a few years, we could have AGI running in a few lines of poorly written Rust code?

## Prepared models

Some benchmarks and download links for the models and tokenizers. I recommend using Q8_0, Q4_0 quantization still being improved. Speed measured on a 16-core AMD Epyc.

|        Model       | Size | Speed |
| ------------------ | ------------- | ------------- |
| [Gemma 2 2B IT Q4_0](https://huggingface.co/samuel-vitorino/gemma2-2b-it-q4_0-LMRS) | 1.39G          | 20 tok/s |
| [Gemma 2 2B IT Q8_0](https://huggingface.co/samuel-vitorino/gemma2-2b-it-q8_0-LMRS) | 2.66GB  | 24 tok/s |
| [Gemma 2 9B IT Q4_0](https://huggingface.co/samuel-vitorino/gemma2-9b-it-q4_0-LMRS) | 4.91GB  | 7 tok/s  | 
| [Gemma 2 9B IT Q8_0](https://huggingface.co/samuel-vitorino/gemma2-9b-it-q8_0-LMRS) | 9.53GB | 8 tok/s  |
| [Llama 3.2 1B IT](https://huggingface.co/samuel-vitorino/Llama-3.2-1B-Instruct-LMRS) | 4.94GB  | 21 tok/s  | 
| [Llama 3.2 1B IT Q8_0](https://huggingface.co/samuel-vitorino/Llama-3.2-1B-Instruct-Q8_0-LMRS) | 1.27GB | 50 tok/s  |
| [Llama 3.2 3B IT Q4_0](https://huggingface.co/samuel-vitorino/Llama-3.2-3B-Instruct-Q4_0-LMRS) | 1.71GB  | 17 tok/s  | 
| [Llama 3.2 3B IT Q8_0](https://huggingface.co/samuel-vitorino/Llama-3.2-3B-Instruct-Q8_0-LMRS) | 3.31GB | 19 tok/s  |
| [PHI 3.5 IT Vision Q8_0](https://huggingface.co/samuel-vitorino/Phi-3.5-vision-instruct-q8_0-LMRS) | 4.28GB | 17 tok/s  |
| [PHI 3.5 IT Mini Q8_0](https://huggingface.co/samuel-vitorino/Phi-3.5-mini-instruct-q8_0-LMRS) | 3.94GB | 18 tok/s  |

## Instructions

You can download the prepared quantized model and tokenizer model files in the lmrs format from huggingface. If you'd prefer to convert the models published by Google/Meta on huggingface yourself, please refer to the following section. Otherwise, you can skip ahead to the build section.

### Model Conversion

Install additional python dependencies (assuming you already have pytorch installed) used in export.py and tokenizer.py:

```properties
pip install -r requirements.txt
```

Download the **.safetensors** and **config.json** files from the original model's page on huggingface (So we don't have to clone the pytorch repo). For multimodal models (PHI3.5 Vision), we also need the CLIP **.config** [file](https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/config.json).

Use the export.py script to convert the model bfloat16 weights into the LMRS format:

```properties
python export.py --files [ordered .safetensor files] --config [model config.json] --save-path [name and path to save] --type [model type (GEMMA/LLAMA/PHI)]
```

To export the quantized version use the **--quantize** and **--quantize-type** flags. The int8 quantized model size should be 4X smaller (from ~9.8G to ~2.5G, depending on the group size). For multimodal models include the **--vision-config** argument.

Use the tokenizer.py script to convert the tokenizer model into the LMRS tokenizer format:

```properties
python tokenizer.py --model-id [huggingface model_id] --tokenizer-type [type of the tokenizer (GEMMA/LLAMA/PHI)]
```

### Build

Compile the rust code with cargo (make sure to pass the target-cpu flag):

```properties
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin chat
```

To enable multimodality, include the multimodal feature by passing the **--features multimodal** argument.

And you are good to go:

```properties
./target/release/chat --model [model weights file]
```

Other arguments include tokenizer, temperature, top-p, show-metrics etc. To check available arguments run with --help. For multimodal models use the **--image** argument with the image path. When using PHI3.5-vision I recommend using a temperature of 0.

---

Run the desktop app

```properties
RUSTFLAGS="-C target-cpu=native" cargo build --release --features desktop --bin desktop
```

Then run:

```properties
./target/release/backend --model [model weights file]
```

---

To run the backend for the [WebUI](https://github.com/samuel-vitorino/lm.rs-webui), first compile:

```properties
RUSTFLAGS="-C target-cpu=native" cargo build --release --features backend --bin backend
```

For multimodality enable the **backend-multimodal** feature.

Then run:

```properties
./target/release/backend --model [model weights file]
```

You can change the ip and port with --ip and --port. Other flags such as temperature, etc. are also available. For multimodal compatibility use the **--multimodal** flag. You can now connect via the web interface.

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





