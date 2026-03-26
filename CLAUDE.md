# fast-whisper-rs

Rust reimplementation of [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) using candle as the inference backend.

## Build & Test

```bash
cargo check                      # Type check
cargo clippy -- -D warnings      # Lint
cargo test                       # Run tests
cargo test -- --ignored          # Run integration tests (requires GPU + model)
cargo build --release            # Release build
```

## Architecture

- `src/main.rs` — Entry point, CLI dispatch
- `src/cli.rs` — Clap argument definitions and validation
- `src/audio.rs` — Audio loading (file/URL), decoding (symphonia), resampling, chunking
- `src/model.rs` — HuggingFace Hub download, candle model initialization
- `src/inference.rs` — Mel spectrogram, batched greedy decoding, timestamp extraction
- `src/output.rs` — JSON output building and serialization
- `src/convert.rs` — SRT/VTT/TXT format conversion

## Key Dependencies

- **candle-core, candle-nn, candle-transformers** — Inference backend (whisper model in `candle_transformers::models::whisper`)
- **hf-hub** — Model download from HuggingFace
- **symphonia** — Audio decoding (mp3, wav, flac, ogg, m4a)
- **clap** — CLI argument parsing (derive API)
- **serde, serde_json** — JSON serialization
- **reqwest** — HTTP downloads for URL audio input
- **indicatif** — Progress bars
- **tokenizers** — HuggingFace tokenizer for decoding
- **byteorder** — Reading mel filter binary data
- **anyhow** — Error handling

## Feature Flags

- `cuda` — Enable CUDA GPU support
- `metal` — Enable Metal (macOS) GPU support
- `flash-attn` — Enable Flash Attention 2

## Conventions

- All CLI flags match insanely-fast-whisper exactly (same names, same defaults)
- Output JSON format matches insanely-fast-whisper exactly
- Tests that require a GPU/model use `#[ignore]` so `cargo test` works without hardware
