use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::whisper as m;
use indicatif::{ProgressBar, ProgressStyle};
use tokenizers::Tokenizer;

use crate::audio::chunk_audio;
use crate::model::WhisperModel;

/// A decoded segment with text and timestamps.
#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub start: f64,
    pub end: Option<f64>,
}

/// Get token ID for a special token string from the tokenizer.
fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow!("Token '{}' not found in tokenizer", token))
}

/// Decode token IDs back to text using the tokenizer.
fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String> {
    tokenizer
        .decode(tokens, true)
        .map_err(|e| anyhow!("Failed to decode tokens: {}", e))
}

/// Compute mel spectrogram from PCM f32 samples and return as a tensor.
pub fn pcm_to_mel_tensor(
    config: &m::Config,
    mel_filters: &[f32],
    samples: &[f32],
    device: &Device,
) -> Result<Tensor> {
    let mel = m::audio::pcm_to_mel(config, samples, mel_filters);
    let n_mel = config.num_mel_bins;
    let n_frames = mel.len() / n_mel;
    let mel_tensor = Tensor::from_vec(mel, (1, n_mel, n_frames), device)?;
    // Truncate to N_FRAMES (3000) so the encoder conv layers produce exactly
    // max_source_positions (1500) features, matching the positional embeddings.
    let mel_tensor = if n_frames > m::N_FRAMES {
        mel_tensor.narrow(2, 0, m::N_FRAMES)?
    } else {
        mel_tensor
    };
    if !device.is_cpu() {
        Ok(mel_tensor.to_dtype(DType::F16)?)
    } else {
        Ok(mel_tensor)
    }
}

/// Transcribe audio samples using batched inference.
///
/// Chunks audio into 30-second segments, batches mel spectrograms through the encoder
/// (up to `batch_size` at a time), then decodes each result sequentially.
/// Shows a progress bar during transcription.
pub fn transcribe(
    whisper: &mut WhisperModel,
    samples: &[f32],
    task: &str,
    language: &str,
    batch_size: usize,
) -> Result<Vec<Segment>> {
    let chunks = chunk_audio(samples);
    let total_chunks = chunks.len();
    let mut all_segments: Vec<Segment> = Vec::new();

    let pb = ProgressBar::new(total_chunks as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40}] {pos}/{len} chunks ({eta})",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_message("Transcribing");

    let batch_size = batch_size.max(1);

    for batch_start in (0..total_chunks).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(total_chunks);
        let batch_chunks = &chunks[batch_start..batch_end];

        // Compute mel spectrograms for each chunk in the batch
        let mels: Vec<Tensor> = batch_chunks
            .iter()
            .map(|chunk| {
                pcm_to_mel_tensor(
                    &whisper.config,
                    &whisper.mel_filters,
                    chunk,
                    &whisper.device,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Stack into batch tensor: (batch_size, n_mel, n_frames)
        let batch_mel = Tensor::cat(&mels, 0)?;

        // Encode entire batch at once
        let batch_features = whisper.model.encoder.forward(&batch_mel, true)?;

        // Decode each item in the batch sequentially (decoder uses KV cache per-sequence)
        for (i, _chunk) in batch_chunks.iter().enumerate() {
            let chunk_idx = batch_start + i;
            let chunk_offset = chunk_idx as f64 * 30.0;

            // Extract this item's encoder features: (1, seq_len, d_model)
            let item_features = batch_features.i(i..i + 1)?;

            let segments =
                decode_with_features(whisper, &item_features, task, language, chunk_offset)?;
            all_segments.extend(segments);

            pb.inc(1);
        }
    }

    pb.finish_with_message("Transcription complete");
    Ok(all_segments)
}

/// Decode a single 30-second audio chunk using greedy decoding with timestamps.
///
/// `chunk_offset` is the time offset (in seconds) of this chunk within the full audio.
pub fn decode_chunk(
    whisper: &mut WhisperModel,
    samples: &[f32],
    task: &str,
    language: &str,
    chunk_offset: f64,
) -> Result<Vec<Segment>> {
    let mel = pcm_to_mel_tensor(&whisper.config, &whisper.mel_filters, samples, &whisper.device)?;
    let audio_features = whisper.model.encoder.forward(&mel, true)?;
    decode_with_features(whisper, &audio_features, task, language, chunk_offset)
}

/// Decode from pre-computed encoder features using greedy decoding with timestamps.
///
/// `audio_features` shape: (1, seq_len, d_model).
/// `chunk_offset` is the time offset (in seconds) of this chunk within the full audio.
fn decode_with_features(
    whisper: &mut WhisperModel,
    audio_features: &Tensor,
    task: &str,
    language: &str,
    chunk_offset: f64,
) -> Result<Vec<Segment>> {
    let device = whisper.device.clone();

    // Resolve special token IDs
    let sot = token_id(&whisper.tokenizer, m::SOT_TOKEN)?;
    let eot = token_id(&whisper.tokenizer, m::EOT_TOKEN)?;
    let transcribe = token_id(&whisper.tokenizer, m::TRANSCRIBE_TOKEN)?;
    let translate = token_id(&whisper.tokenizer, m::TRANSLATE_TOKEN)?;
    let no_timestamps = token_id(&whisper.tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

    let task_token = if task == "translate" {
        translate
    } else {
        transcribe
    };
    let timestamp_begin = no_timestamps + 1;

    // Build initial prompt: [SOT, language?, task]
    let mut prompt: Vec<u32> = vec![sot];

    if language != "None" {
        let lang_str = format!("<|{}|>", language);
        if let Some(lang_id) = whisper.tokenizer.token_to_id(&lang_str) {
            prompt.push(lang_id);
        } else {
            eprintln!(
                "Warning: language '{}' not found in tokenizer, using auto-detect",
                language
            );
        }
    }

    prompt.push(task_token);
    // No NO_TIMESTAMPS token — we want timestamp predictions

    // Greedy decode with timestamps
    let max_tokens = whisper.config.max_target_positions / 2;
    let mut segments: Vec<Segment> = Vec::new();
    let mut text_tokens: Vec<u32> = Vec::new();
    let mut last_timestamp: Option<f64> = None;

    whisper.model.decoder.reset_kv_cache();

    let mut tokens = prompt;

    for i in 0..max_tokens {
        let token_t = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        let ys = whisper
            .model
            .decoder
            .forward(&token_t, audio_features, i == 0)?;
        let logits = whisper.model.decoder.final_linear(&ys)?;

        // Get logits for the last position: (batch, seq_len, vocab) -> (vocab,)
        let (_, seq_dim, _) = logits.dims3()?;
        let logits = logits.i((0, seq_dim - 1))?.to_dtype(DType::F32)?;

        let logits_vec = logits.to_vec1::<f32>()?;
        let next_token = suppress_and_argmax(&logits_vec, &whisper.config.suppress_tokens);

        if next_token == eot {
            // Flush remaining text tokens as final segment
            if !text_tokens.is_empty() {
                let text = decode_tokens(&whisper.tokenizer, &text_tokens)?;
                let start = last_timestamp.unwrap_or(0.0) + chunk_offset;
                segments.push(Segment {
                    text,
                    start,
                    end: None,
                });
            }
            break;
        }

        if next_token >= timestamp_begin {
            let time = (next_token - timestamp_begin) as f64 * 0.02;

            if !text_tokens.is_empty() {
                let text = decode_tokens(&whisper.tokenizer, &text_tokens)?;
                let start = last_timestamp.unwrap_or(0.0) + chunk_offset;
                segments.push(Segment {
                    text,
                    start,
                    end: Some(time + chunk_offset),
                });
                text_tokens.clear();
            }

            last_timestamp = Some(time);
        } else {
            text_tokens.push(next_token);
        }

        // Next iteration: only feed the new token (KV cache handles history)
        tokens = vec![next_token];
    }

    // Handle case where decoding produced text but no segments
    if segments.is_empty() && !text_tokens.is_empty() {
        let text = decode_tokens(&whisper.tokenizer, &text_tokens)?;
        segments.push(Segment {
            text,
            start: chunk_offset,
            end: None,
        });
    }

    Ok(segments)
}

/// Suppress specified tokens and return the argmax token ID.
fn suppress_and_argmax(logits: &[f32], suppress_tokens: &[u32]) -> u32 {
    let mut best_token = 0u32;
    let mut best_logit = f32::NEG_INFINITY;

    for (i, &logit) in logits.iter().enumerate() {
        let token = i as u32;
        if suppress_tokens.contains(&token) {
            continue;
        }
        if logit > best_logit {
            best_logit = logit;
            best_token = token;
        }
    }

    best_token
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::CHUNK_SAMPLES;

    #[test]
    fn test_suppress_and_argmax_no_suppression() {
        let logits = vec![0.1, 0.5, 0.3, 0.2];
        let result = suppress_and_argmax(&logits, &[]);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_suppress_and_argmax_with_suppression() {
        let logits = vec![0.1, 0.5, 0.3, 0.2];
        let result = suppress_and_argmax(&logits, &[1]);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_suppress_and_argmax_suppress_multiple() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let result = suppress_and_argmax(&logits, &[1, 3]);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_suppress_and_argmax_all_suppressed() {
        let logits = vec![0.1, 0.5];
        // All tokens suppressed — returns 0 with NEG_INFINITY (edge case)
        let result = suppress_and_argmax(&logits, &[0, 1]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_timestamp_token_to_time() {
        // Each timestamp token increment = 0.02 seconds
        let timestamp_begin: u32 = 50365;
        let token = timestamp_begin + 50; // 50 * 0.02 = 1.0 second
        let time = (token - timestamp_begin) as f64 * 0.02;
        assert!((time - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_timestamp_token_zero() {
        let timestamp_begin: u32 = 50365;
        let token = timestamp_begin;
        let time = (token - timestamp_begin) as f64 * 0.02;
        assert!((time - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_timestamp_token_thirty_seconds() {
        // 1500 * 0.02 = 30.0 seconds (full chunk)
        let timestamp_begin: u32 = 50365;
        let token = timestamp_begin + 1500;
        let time = (token - timestamp_begin) as f64 * 0.02;
        assert!((time - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_segment_offset_applied() {
        let chunk_offset = 60.0;
        let timestamp_begin: u32 = 50365;
        let token = timestamp_begin + 500; // 10.0 seconds within chunk
        let time = (token - timestamp_begin) as f64 * 0.02;
        let absolute_time = time + chunk_offset;
        assert!((absolute_time - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_pcm_to_mel_tensor_shape() {
        let config = m::Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 8,
            decoder_layers: 6,
            suppress_tokens: vec![],
        };

        // mel filters: num_mel_bins * (N_FFT/2 + 1) = 80 * 201
        let mel_filters = vec![0.0f32; config.num_mel_bins * (m::N_FFT / 2 + 1)];

        // 30 seconds of silence at 16kHz
        let samples = vec![0.0f32; m::N_SAMPLES];

        let tensor = pcm_to_mel_tensor(&config, &mel_filters, &samples, &Device::Cpu).unwrap();
        let dims = tensor.dims();
        assert_eq!(dims[0], 1); // batch
        assert_eq!(dims[1], 80); // mel bins
        assert!(dims[2] > 0); // has frames
    }

    #[test]
    fn test_batch_iteration_exact() {
        // 6 chunks with batch_size=3 => 2 batches, no remainder
        let total_chunks = 6;
        let batch_size = 3;
        let mut batches: Vec<(usize, usize)> = Vec::new();
        for batch_start in (0..total_chunks).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_chunks);
            batches.push((batch_start, batch_end));
        }
        assert_eq!(batches, vec![(0, 3), (3, 6)]);
    }

    #[test]
    fn test_batch_iteration_remainder() {
        // 7 chunks with batch_size=3 => 3 batches, last has 1 chunk
        let total_chunks = 7;
        let batch_size = 3;
        let mut batches: Vec<(usize, usize)> = Vec::new();
        for batch_start in (0..total_chunks).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_chunks);
            batches.push((batch_start, batch_end));
        }
        assert_eq!(batches, vec![(0, 3), (3, 6), (6, 7)]);
    }

    #[test]
    fn test_batch_iteration_single_chunk() {
        // 1 chunk with batch_size=24 => 1 batch
        let total_chunks = 1;
        let batch_size = 24;
        let mut batches: Vec<(usize, usize)> = Vec::new();
        for batch_start in (0..total_chunks).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_chunks);
            batches.push((batch_start, batch_end));
        }
        assert_eq!(batches, vec![(0, 1)]);
    }

    #[test]
    fn test_batch_iteration_batch_larger_than_chunks() {
        // 3 chunks with batch_size=10 => 1 batch with all 3
        let total_chunks = 3;
        let batch_size = 10;
        let mut batches: Vec<(usize, usize)> = Vec::new();
        for batch_start in (0..total_chunks).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_chunks);
            batches.push((batch_start, batch_end));
        }
        assert_eq!(batches, vec![(0, 3)]);
    }

    #[test]
    fn test_chunk_offset_calculation() {
        // Verify chunk offsets are correct: chunk_idx * 30.0 seconds
        let batch_start = 3;
        let offsets: Vec<f64> = (0..3)
            .map(|i| (batch_start + i) as f64 * 30.0)
            .collect();
        assert!((offsets[0] - 90.0).abs() < 1e-10);
        assert!((offsets[1] - 120.0).abs() < 1e-10);
        assert!((offsets[2] - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_mel_batch_stacking() {
        // Verify that multiple mel tensors can be concatenated along batch dim
        let config = m::Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 8,
            decoder_layers: 6,
            suppress_tokens: vec![],
        };
        let mel_filters = vec![0.0f32; config.num_mel_bins * (m::N_FFT / 2 + 1)];
        let samples = vec![0.0f32; CHUNK_SAMPLES];

        let mel1 = pcm_to_mel_tensor(&config, &mel_filters, &samples, &Device::Cpu).unwrap();
        let mel2 = pcm_to_mel_tensor(&config, &mel_filters, &samples, &Device::Cpu).unwrap();
        let mel3 = pcm_to_mel_tensor(&config, &mel_filters, &samples, &Device::Cpu).unwrap();

        let batch = Tensor::cat(&[mel1, mel2, mel3], 0).unwrap();
        let dims = batch.dims();
        assert_eq!(dims[0], 3); // batch size
        assert_eq!(dims[1], 80); // mel bins
        assert!(dims[2] > 0); // frames
    }

    #[test]
    fn test_pcm_to_mel_tensor_128_bins() {
        let config = m::Config {
            num_mel_bins: 128,
            max_source_positions: 1500,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            vocab_size: 51866,
            max_target_positions: 448,
            decoder_attention_heads: 20,
            decoder_layers: 32,
            suppress_tokens: vec![],
        };

        let mel_filters = vec![0.0f32; config.num_mel_bins * (m::N_FFT / 2 + 1)];
        let samples = vec![0.0f32; m::N_SAMPLES];

        let tensor = pcm_to_mel_tensor(&config, &mel_filters, &samples, &Device::Cpu).unwrap();
        let dims = tensor.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 128);
        assert!(dims[2] > 0);
    }
}
