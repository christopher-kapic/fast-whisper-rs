use anyhow::{anyhow, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Whisper expects 16kHz mono audio.
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Each chunk is 30 seconds of audio at 16kHz = 480000 samples.
pub const CHUNK_SAMPLES: usize = 30 * WHISPER_SAMPLE_RATE as usize;

/// Returns true if the input looks like an HTTP(S) URL.
/// Matches Python's exact check: `inputs.startswith('http://') or inputs.startswith('https://')`.
pub fn is_url(input: &str) -> bool {
    input.starts_with("http://") || input.starts_with("https://")
}

/// Load audio from either a local file or URL, returning PCM f32 samples at 16kHz mono.
pub fn load_audio(input: &str) -> Result<Vec<f32>> {
    if is_url(input) {
        let bytes = download_audio(input)?;
        // Try to extract extension from URL path for format hint
        let ext = url_extension(input);
        load_audio_from_bytes(bytes, ext.as_deref())
    } else {
        load_audio_from_file(input)
    }
}

/// Extract file extension from a URL path (ignoring query params and fragments).
fn url_extension(url: &str) -> Option<String> {
    // Strip query string and fragment
    let path = url.split('?').next().unwrap_or(url);
    let path = path.split('#').next().unwrap_or(path);
    // Get the last path segment
    let filename = path.rsplit('/').next()?;
    let ext = filename.rsplit('.').next()?;
    if ext == filename {
        // No dot found
        None
    } else {
        Some(ext.to_lowercase())
    }
}

/// Download audio from a URL into memory with a progress bar.
fn download_audio(url: &str) -> Result<Vec<u8>> {
    let client = Client::new();
    let response = client
        .get(url)
        .send()
        .map_err(|e| anyhow!("Failed to download audio from '{}': {}", url, e))?
        .error_for_status()
        .map_err(|e| anyhow!("HTTP error downloading '{}': {}", url, e))?;

    let total_size = response.content_length();

    let pb = if let Some(size) = total_size {
        let pb = ProgressBar::new(size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );
        pb.set_message("Downloading audio");
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{msg} {spinner} {bytes}")
                .unwrap(),
        );
        pb.set_message("Downloading audio");
        pb
    };

    let mut bytes: Vec<u8> = Vec::with_capacity(total_size.unwrap_or(0) as usize);
    let mut reader = response;
    let mut buf = [0u8; 8192];
    loop {
        use std::io::Read;
        let n = reader
            .read(&mut buf)
            .map_err(|e| anyhow!("Error reading download stream: {}", e))?;
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        pb.set_position(bytes.len() as u64);
    }

    pb.finish_with_message("Download complete");
    Ok(bytes)
}

/// Load audio from a local file path and return PCM f32 samples at 16kHz mono.
pub fn load_audio_from_file(path: &str) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow!("Failed to open audio file '{}': {}", path, e))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    decode_audio(mss, Some(path))
}

/// Load audio from an in-memory buffer and return PCM f32 samples at 16kHz mono.
pub fn load_audio_from_bytes(bytes: Vec<u8>, hint_ext: Option<&str>) -> Result<Vec<f32>> {
    let cursor = std::io::Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = hint_ext {
        hint.with_extension(ext);
    }
    decode_audio_with_hint(mss, hint)
}

/// Decode audio from a MediaSourceStream to PCM f32 mono at 16kHz.
fn decode_audio(mss: MediaSourceStream, path: Option<&str>) -> Result<Vec<f32>> {
    let mut hint = Hint::new();
    if let Some(path) = path {
        if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
    }
    decode_audio_with_hint(mss, hint)
}

/// Core decoding logic with a format hint.
fn decode_audio_with_hint(mss: MediaSourceStream, hint: Hint) -> Result<Vec<f32>> {
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| anyhow!("Failed to probe audio format: {}", e))?;

    let mut format = probed.format;

    // Find the first audio track.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No audio track found"))?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let source_sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("Unknown sample rate"))?;
    let channels = codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &decoder_opts)
        .map_err(|e| anyhow!("Failed to create audio decoder: {}", e))?;

    let mut raw_samples: Vec<f32> = Vec::new();

    // Decode all packets.
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(anyhow!("Error reading audio packet: {}", e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow!("Decode error: {}", e)),
        };

        let spec = *decoded.spec();
        let num_frames = decoded.frames();
        if num_frames == 0 {
            continue;
        }

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        raw_samples.extend_from_slice(sample_buf.samples());
    }

    if raw_samples.is_empty() {
        return Ok(Vec::new());
    }

    // Convert to mono by averaging channels.
    let mono = if channels > 1 {
        to_mono(&raw_samples, channels)
    } else {
        raw_samples
    };

    // Resample to 16kHz if needed.
    if source_sample_rate != WHISPER_SAMPLE_RATE {
        Ok(resample(&mono, source_sample_rate, WHISPER_SAMPLE_RATE))
    } else {
        Ok(mono)
    }
}

/// Convert interleaved multi-channel audio to mono by averaging channels.
fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linear interpolation resampling.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let frac = (src_idx - idx0 as f64) as f32;

        let s0 = samples[idx0.min(samples.len() - 1)];
        let s1 = samples[(idx0 + 1).min(samples.len() - 1)];
        output.push(s0 + frac * (s1 - s0));
    }

    output
}

/// Split audio samples into 30-second chunks at 16kHz (480000 samples each).
/// The last chunk is zero-padded to CHUNK_SAMPLES if shorter.
pub fn chunk_audio(samples: &[f32]) -> Vec<Vec<f32>> {
    if samples.is_empty() {
        return vec![vec![0.0f32; CHUNK_SAMPLES]];
    }

    let mut chunks = Vec::new();
    for chunk in samples.chunks(CHUNK_SAMPLES) {
        let mut padded = chunk.to_vec();
        if padded.len() < CHUNK_SAMPLES {
            padded.resize(CHUNK_SAMPLES, 0.0);
        }
        chunks.push(padded);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_exact_30s() {
        // Exactly 30 seconds = 1 chunk, no padding needed.
        let samples = vec![1.0f32; CHUNK_SAMPLES];
        let chunks = chunk_audio(&samples);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), CHUNK_SAMPLES);
        assert!(chunks[0].iter().all(|&s| s == 1.0));
    }

    #[test]
    fn test_chunk_partial_last_chunk() {
        // 1.5 chunks worth of audio = 2 chunks, second is zero-padded.
        let n = CHUNK_SAMPLES + CHUNK_SAMPLES / 2;
        let samples = vec![0.5f32; n];
        let chunks = chunk_audio(&samples);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), CHUNK_SAMPLES);
        assert_eq!(chunks[1].len(), CHUNK_SAMPLES);
        // First chunk is all 0.5
        assert!(chunks[0].iter().all(|&s| s == 0.5));
        // Second chunk: first half is 0.5, second half is 0.0 (zero-padded)
        let half = CHUNK_SAMPLES / 2;
        assert!(chunks[1][..half].iter().all(|&s| s == 0.5));
        assert!(chunks[1][half..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_chunk_single_short() {
        // Less than 30 seconds = 1 chunk, zero-padded.
        let samples = vec![0.3f32; 1000];
        let chunks = chunk_audio(&samples);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), CHUNK_SAMPLES);
        assert!(chunks[0][..1000].iter().all(|&s| s == 0.3));
        assert!(chunks[0][1000..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_chunk_empty_audio() {
        // Empty audio produces one zero-padded chunk.
        let chunks = chunk_audio(&[]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), CHUNK_SAMPLES);
        assert!(chunks[0].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_chunk_multiple_exact() {
        // Exactly 3 chunks.
        let samples = vec![1.0f32; CHUNK_SAMPLES * 3];
        let chunks = chunk_audio(&samples);
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.len(), CHUNK_SAMPLES);
            assert!(chunk.iter().all(|&s| s == 1.0));
        }
    }

    #[test]
    fn test_to_mono_stereo() {
        let stereo = vec![1.0f32, 0.0, 0.5, 0.5, 0.0, 1.0];
        let mono = to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
        assert!((mono[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_resample_identity() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(&samples, 16000, 16000);
        assert_eq!(result.len(), samples.len());
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_downsample() {
        // 32kHz to 16kHz should roughly halve the number of samples.
        let samples: Vec<f32> = (0..3200).map(|i| i as f32).collect();
        let result = resample(&samples, 32000, 16000);
        assert_eq!(result.len(), 1600);
    }

    #[test]
    fn test_resample_upsample() {
        // 8kHz to 16kHz should roughly double the number of samples.
        let samples: Vec<f32> = (0..800).map(|i| i as f32).collect();
        let result = resample(&samples, 8000, 16000);
        assert_eq!(result.len(), 1600);
    }

    #[test]
    fn test_is_url_http() {
        assert!(is_url("http://example.com/audio.wav"));
        assert!(is_url("http://example.com"));
    }

    #[test]
    fn test_is_url_https() {
        assert!(is_url("https://example.com/audio.mp3"));
        assert!(is_url("https://huggingface.co/datasets/audio/file.flac"));
    }

    #[test]
    fn test_is_url_local_paths() {
        assert!(!is_url("audio.wav"));
        assert!(!is_url("/home/user/audio.mp3"));
        assert!(!is_url("./relative/path.wav"));
        assert!(!is_url("../parent/audio.flac"));
    }

    #[test]
    fn test_is_url_paths_containing_http() {
        // Local files with "http" in the name should NOT be treated as URLs
        assert!(!is_url("http_huggingface_co.png"));
        assert!(!is_url("https_file.wav"));
        assert!(!is_url("/data/http_download.mp3"));
    }

    #[test]
    fn test_url_extension() {
        assert_eq!(url_extension("https://example.com/audio.wav"), Some("wav".to_string()));
        assert_eq!(url_extension("https://example.com/audio.MP3"), Some("mp3".to_string()));
        assert_eq!(url_extension("https://example.com/audio.wav?token=abc"), Some("wav".to_string()));
        assert_eq!(url_extension("https://example.com/audio.flac#section"), Some("flac".to_string()));
        assert_eq!(url_extension("https://example.com/noext"), None);
        assert_eq!(url_extension("https://example.com/"), None);
    }
}
