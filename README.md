# fast-whisper-rs

A Rust reimplementation of [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) using [candle](https://github.com/huggingface/candle) as the inference backend. Transcribe audio files locally with OpenAI's Whisper models — no Python runtime required.

## Why?

[insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) is a fantastic CLI for fast local transcription, but it requires Python, PyTorch, and a stack of dependencies. fast-whisper-rs aims to provide the same experience as a single static binary: same CLI flags, same output format, zero runtime dependencies.

## Features

- Transcribe and translate audio files (mp3, wav, flac, ogg, m4a)
- Audio input from local files or URLs
- Batched inference for fast transcription
- Chunk-level and word-level timestamps
- Speaker diarization via pyannote ONNX models
- Flash Attention 2 support
- CUDA and Metal GPU acceleration
- Convert transcripts to SRT, VTT, or TXT
- Output JSON format matches insanely-fast-whisper exactly

## Install

### Install script (macOS and Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/christopher-kapic/fast-whisper-rs/master/scripts/install.sh | bash
```

Install a specific version:

```bash
curl -fsSL https://raw.githubusercontent.com/christopher-kapic/fast-whisper-rs/master/scripts/install.sh | bash -s v0.1.0
```

Install to a custom directory:

```bash
curl -fsSL https://raw.githubusercontent.com/christopher-kapic/fast-whisper-rs/master/scripts/install.sh | INSTALL_DIR=~/.local/bin bash
```

### Cargo

```bash
cargo install --git https://github.com/christopher-kapic/fast-whisper-rs
```

With GPU support:

```bash
# CUDA
cargo install --git https://github.com/christopher-kapic/fast-whisper-rs --features cuda

# Metal (macOS)
cargo install --git https://github.com/christopher-kapic/fast-whisper-rs --features metal
```

## Update

To update, just re-run the same install command you used originally. The install script is idempotent — it downloads the latest release and replaces the existing binary in place.

```bash
# Install script — fetches the latest release
curl -fsSL https://raw.githubusercontent.com/christopher-kapic/fast-whisper-rs/master/scripts/install.sh | bash

# Cargo
cargo install --git https://github.com/christopher-kapic/fast-whisper-rs --force
```

## Usage

```bash
# Basic transcription
fast-whisper-rs --file-name audio.mp3

# Translate to English
fast-whisper-rs --file-name audio.mp3 --task translate

# Use a specific model and language
fast-whisper-rs --file-name audio.mp3 --model-name openai/whisper-medium --language fr

# Word-level timestamps
fast-whisper-rs --file-name audio.mp3 --timestamp word

# Transcribe from a URL
fast-whisper-rs --file-name https://example.com/audio.mp3

# Custom output path
fast-whisper-rs --file-name audio.mp3 --transcript-path result.json

# Speaker diarization
fast-whisper-rs --file-name audio.mp3 --num-speakers 2
fast-whisper-rs --file-name audio.mp3 --min-speakers 2 --max-speakers 5
```

### Convert output format

```bash
fast-whisper-rs convert output.json -f srt
fast-whisper-rs convert output.json -f vtt
fast-whisper-rs convert output.json -f txt -o ./subtitles/
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--file-name` | | Path or URL to the audio file |
| `--model-name` | `openai/whisper-large-v3` | HuggingFace model name |
| `--task` | `transcribe` | `transcribe` or `translate` |
| `--language` | `None` (auto-detect) | Language code (e.g. `en`, `fr`, `es`) |
| `--batch-size` | `24` | Batch size for inference |
| `--flash` | `False` | Enable Flash Attention 2 (`True`/`False`) |
| `--timestamp` | `chunk` | `chunk` or `word` level timestamps |
| `--device-id` | `0` | GPU device ID or `mps` for Metal |
| `--transcript-path` | `output.json` | Output JSON path |
| `--hf-token` | `no_token` | HuggingFace API token |
| `--num-speakers` | | Exact number of speakers (diarization) |
| `--min-speakers` | | Minimum speakers (diarization) |
| `--max-speakers` | | Maximum speakers (diarization) |
| `--diarization_model` | `pyannote/speaker-diarization-3.1` | Diarization model |

All flags match [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) for drop-in compatibility.

## Building from source

```bash
git clone https://github.com/christopher-kapic/fast-whisper-rs
cd fast-whisper-rs
cargo build --release
```

With GPU features:

```bash
cargo build --release --features cuda
cargo build --release --features metal
cargo build --release --features flash-attn
```

## License

MIT
