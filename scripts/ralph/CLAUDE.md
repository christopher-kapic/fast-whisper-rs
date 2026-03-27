# Ralph Agent Instructions

You are an autonomous coding agent building `fast-whisper-rs` — a Rust reimplementation of insanely-fast-whisper using candle as the inference backend.

## Your Task

1. Read the PRD at `scripts/ralph/prd.json`
2. Read the progress log at `scripts/ralph/progress.txt` (check Codebase Patterns section first)
3. Check you're on the correct branch from PRD `branchName`. If not, check it out or create from main.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story
6. Run quality checks: `cargo check && cargo clippy -- -D warnings && cargo test`
7. Update CLAUDE.md files if you discover reusable patterns (see below)
8. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
9. Update the PRD to set `passes: true` for the completed story
10. Append your progress to `scripts/ralph/progress.txt`

## Project Context

This is a Rust drop-in replacement for the Python `insanely-fast-whisper` (../insanely-fast-whisper, or ../../../insanely-fast-whisper/ relative to this CLAUDE.md file) CLI. Key points:

- **Binary name:** `fast-whisper-rs`
- **Inference backend:** candle (`candle-core`, `candle-nn`, `candle-transformers`) — HuggingFace's Rust ML framework
- **candle already has a Whisper example** in `candle-transformers` with mel spectrogram, model loading, and greedy decoding. Build on that, don't reinvent it.
- **Key optimizations to match:** batched inference, Flash Attention 2 / SDPA, FP16
- **Audio decoding:** symphonia crate for decoding audio to PCM f32 mono 16kHz
- **Model download:** hf-hub crate for downloading from HuggingFace Hub
- **CLI must match** insanely-fast-whisper's arguments exactly (same flag names, same defaults)
- **Output JSON format must match** insanely-fast-whisper exactly:
  ```json
  {"speakers": [], "chunks": [{"text": "...", "timestamp": [start, end]}], "text": "full text"}
  ```

## Quality Checks (Rust)

Before committing, run:
```bash
cargo check
cargo clippy -- -D warnings
cargo test
```

All three must pass. Do NOT commit broken code.

## CRITICAL: Parity Rules

These are real pitfalls from CLI reimplementations. Follow them exactly:

### Flag Semantics (match behavior, not just names)

| Flag | Type | Default | Parity Notes |
|------|------|---------|-------------|
| `--file-name` | String | REQUIRED | Path or URL |
| `--device-id` | String | `"0"` | GPU device number or literal `"mps"` |
| `--transcript-path` | String | `"output.json"` | |
| `--model-name` | String | `"openai/whisper-large-v3"` | |
| `--task` | String | `"transcribe"` | Choices: `transcribe`, `translate` |
| `--language` | String | `"None"` | **DEFAULT IS THE LITERAL STRING `"None"`, NOT Option::None.** Python code checks `args.language == "None"` to mean auto-detect. Match this: default to string `"None"`, treat `"None"` as auto-detect. |
| `--batch-size` | int | `24` | |
| `--flash` | bool | `False` | **TAKES A VALUE: `--flash True` / `--flash False`.** Python uses `type=bool` in argparse, meaning it requires a value argument. It is NOT a bare flag. In clap, do NOT use `action = SetTrue`. Use a string/bool value arg. |
| `--timestamp` | String | `"chunk"` | Choices: `chunk`, `word` |
| `--hf-token` | String | `"no_token"` | |
| `--diarization_model` | String | `"pyannote/speaker-diarization-3.1"` | **UNDERSCORE, not hyphen.** Python uses `--diarization_model`. Clap auto-converts `_` to `-` by default — you MUST override this with `#[arg(long = "diarization_model")]` to keep the underscore. |
| `--num-speakers` | Option\<int\> | `None` | Mutually exclusive with min/max |
| `--min-speakers` | Option\<int\> | `None` | Must be >= 1, <= max-speakers |
| `--max-speakers` | Option\<int\> | `None` | Must be >= 1, >= min-speakers |

### .en Model Detection
Python: `args.model_name.split(".")[-1] == "en"` — splits on `.`, checks last segment equals `"en"`. This only removes `--task` from generate_kwargs. It does NOT affect `--language`.

### Output JSON
- Python uses `json.dump(result, fp, ensure_ascii=False)` — Unicode preserved, not escaped
- Key order in Python dict: `speakers`, `chunks`, `text` (use `#[derive(Serialize)]` field order to match)
- `timestamp` values are `[f64, f64]` or `[f64, null]` — the end timestamp of the final chunk can be null
- `speakers` is always `[]` when diarization is disabled

### Exit Codes
- Python argparse exits with code **2** for argument validation errors
- Runtime errors exit with code **1** (unhandled exception)
- Match these exactly. Use `std::process::exit(2)` for arg validation failures.

### Stdout Messages
- Success message goes to **stdout** (not stderr): `"Voila!✨ Your file has been transcribed go check it out over here 👉 {path}"`
- With diarization: `"Voila!✨ Your file has been transcribed & speaker segmented go check it out over here 👉 {path}"`
- Match these messages exactly (including emoji).

### Validation Rules
- `--num-speakers` mutually exclusive with `--min-speakers` / `--max-speakers`
- All speaker counts must be >= 1
- `--min-speakers` <= `--max-speakers`
- `.en` model suffix: strip `--task` (English-only models don't support translate)

## CRITICAL: Diarization Architecture

Speaker diarization uses ONNX Runtime (ort crate) to run pyannote models natively in Rust. The pipeline has 4 stages:

### Pipeline Stages

1. **Segmentation** — PyanNet model via ort. Sliding window (10s chunks, 90% overlap) over audio. Output: frame-level speaker activity (num_frames, num_speakers). Binarize with threshold.
2. **Embedding** — WeSpeaker ResNet34 model via ort. For each active speaker region, compute 80-bin fbank features then extract embedding vector. Output: (num_chunks, num_speakers, embedding_dim).
3. **Clustering** — Pure Rust math with ndarray. L2-normalize embeddings, PLDA transform (LDA dim reduction to 128), agglomerative hierarchical clustering with cosine distance. Output: cluster assignments mapping each (chunk, speaker) to a global speaker ID.
4. **Post-processing** — Reconstruct speaker timeline, merge consecutive same-speaker segments, align with ASR transcript chunks via argmin on end timestamps. This must match insanely-fast-whisper/src/insanely_fast_whisper/utils/diarize.py exactly.

### Model Distribution

ONNX models are NOT committed to the repo (too large ~100MB total). Instead:
- A GitHub Actions workflow exports models and attaches them as release assets
- At runtime, fast-whisper-rs downloads models from the GitHub release and caches them at ~/.cache/fast-whisper-rs/diarization/
- Models: segmentation.onnx (~3-5MB), embedding.onnx (~85-95MB), plda_xvec_transform.npz + plda.npz (~2-4MB)

### Diarization Trigger

Diarization runs when `--hf-token` is not `"no_token"` (matching Python: `if args.hf_token != "no_token"`). Note: the hf-token is NOT used for downloading diarization models (those come from GitHub releases), but its presence signals that the user wants diarization.

### Fbank Features (for embedding model)

WeSpeaker expects 80-bin fbank features, NOT raw waveform:
- Frame: 400 samples (25ms at 16kHz), hop: 160 samples (10ms)
- Pre-emphasis: 0.97, Hamming window, FFT, 80 mel-scale triangular filters, log energy
- Matches torchaudio.compliance.kaldi.fbank

### Pyannote Reference Code

The Python diarization implementation to match is at:
- `insanely-fast-whisper/src/insanely_fast_whisper/utils/diarize.py` — segment merging + ASR alignment
- `insanely-fast-whisper/src/insanely_fast_whisper/utils/diarization_pipeline.py` — pipeline orchestration
- pyannote-audio source at `pyannote-audio/src/pyannote/audio/pipelines/speaker_diarization.py` — internal pipeline stages

### Output JSON with Diarization

When diarization is active, the `speakers` field changes from `[]` to a list of per-chunk entries:
```json
{"speakers": [{"speaker": "SPEAKER_00", "text": "...", "timestamp": [start, end]}, ...], "chunks": [...], "text": "..."}
```
Each ASR chunk gets a speaker label. group_by_speaker is always false (matching Python).

## Reference: convert subcommand

The `convert` subcommand converts JSON output to SRT/VTT/TXT:
```
fast-whisper-rs convert <input_file> -f <format> -o <output_dir> --verbose
```
- Formats: srt, vtt, txt
- SRT timestamps: `HH:MM:SS,mmm --> HH:MM:SS,mmm`
- VTT timestamps: `HH:MM:SS.mmm --> HH:MM:SS.mmm` with `WEBVTT` header

## Progress Report Format

APPEND to scripts/ralph/progress.txt (never replace, always append):
```
## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - Useful context
---
```

## Consolidate Patterns

If you discover a **reusable pattern**, add it to the `## Codebase Patterns` section at the TOP of progress.txt:

```
## Codebase Patterns
- candle device selection: use candle_core::Device::cuda_if_available(device_id)
- Whisper model types are in candle_transformers::models::whisper
```

## Update CLAUDE.md Files

Add valuable learnings to CLAUDE.md files in relevant directories for future iterations.

## Stop Condition

After completing a user story, check if ALL stories have `passes: true`.

If ALL stories are complete, reply with:
<promise>COMPLETE</promise>

If there are still stories with `passes: false`, end your response normally.

## Important

- Work on ONE story per iteration
- Commit frequently
- Keep CI green
- Read the Codebase Patterns section in progress.txt before starting
- Use `cargo` features flags for optional CUDA/Metal support where appropriate
