# squirrel-rs

An LLM inference engine for GGUF models, written from scratch in Rust. Runs
Llama-3.2 on the CPU with `Q4_0`, `Q8_0` or `BF16` weights.

## Prerequisites

- Rust (stable), via [rustup](https://rustup.rs)
- Apple Silicon or another aarch64 machine for the hand-written NEON kernels.
  It builds and runs elsewhere, just on the scalar fallbacks.

## Getting the weights

The engine needs a `.gguf` file and a `tokenizer.json`, both in `assets/`:

```sh
mkdir -p assets && cd assets

# weights: Q8_0 (1.3 GB) or BF16 (2.5 GB)
curl -LO https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
curl -LO https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-BF16.gguf

# tokenizer
curl -LO https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json
```

## Running

```sh
cargo run --release -- \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --prompt "The capital of France is" \
  --max-tokens 50
```

Options: `--model` (default `Llama-3.2-1B-Instruct-Q8_0.gguf`), `--prompt`,
`--prompt-file`, `--max-tokens` (default 512), `--assets` (default `assets/`).
Per-token timings go to stderr, generated text to stdout.

## Tests and benchmarks

```sh
cargo test --release
scripts/bench_threads.py
```
