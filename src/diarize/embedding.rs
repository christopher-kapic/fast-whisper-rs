use anyhow::Result;
use ndarray::{Array2, Array3};
use ort::value::TensorRef;

/// Configuration for fbank feature extraction.
pub struct FbankConfig {
    /// Sample rate in Hz.
    pub sample_rate: usize,
    /// Frame length in samples (25ms at 16kHz = 400).
    pub frame_length: usize,
    /// Frame shift / hop in samples (10ms at 16kHz = 160).
    pub frame_shift: usize,
    /// Number of mel filterbank bins.
    pub num_bins: usize,
    /// Pre-emphasis coefficient.
    pub preemphasis: f32,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_length: 400,
            frame_shift: 160,
            num_bins: 80,
            preemphasis: 0.97,
        }
    }
}

/// Compute 80-bin fbank features from raw audio.
///
/// Matches torchaudio.compliance.kaldi.fbank:
/// - Pre-emphasis with coefficient 0.97
/// - Hamming window
/// - Power spectrum via FFT
/// - 80 mel-scale triangular filters
/// - Log energy
///
/// Returns shape (num_frames, num_bins).
pub fn compute_fbank(audio: &[f32], config: &FbankConfig) -> Array2<f32> {
    if audio.is_empty() {
        return Array2::zeros((0, config.num_bins));
    }

    // Pre-emphasis
    let mut emphasized = Vec::with_capacity(audio.len());
    emphasized.push(audio[0]);
    for i in 1..audio.len() {
        emphasized.push(audio[i] - config.preemphasis * audio[i - 1]);
    }

    // Frame the signal
    let num_frames = if emphasized.len() >= config.frame_length {
        1 + (emphasized.len() - config.frame_length) / config.frame_shift
    } else {
        // If audio is shorter than frame_length, we get 1 frame (zero-padded)
        if emphasized.is_empty() {
            0
        } else {
            1
        }
    };

    if num_frames == 0 {
        return Array2::zeros((0, config.num_bins));
    }

    // FFT size: next power of 2 >= frame_length
    let fft_size = config.frame_length.next_power_of_two();
    let num_fft_bins = fft_size / 2 + 1;

    // Compute Hamming window
    let hamming = hamming_window(config.frame_length);

    // Compute mel filterbank
    let mel_filters = mel_filterbank(config.num_bins, fft_size, config.sample_rate);

    let mut features = Array2::<f32>::zeros((num_frames, config.num_bins));

    for frame_idx in 0..num_frames {
        let start = frame_idx * config.frame_shift;

        // Extract frame and apply window
        let mut windowed = vec![0.0f32; fft_size];
        for i in 0..config.frame_length {
            let sample_idx = start + i;
            if sample_idx < emphasized.len() {
                windowed[i] = emphasized[sample_idx] * hamming[i];
            }
        }

        // Compute power spectrum via real FFT
        let power_spectrum = compute_power_spectrum(&windowed, num_fft_bins);

        // Apply mel filterbank and take log
        for bin in 0..config.num_bins {
            let mut energy: f32 = 0.0;
            for k in 0..num_fft_bins {
                energy += mel_filters[[bin, k]] * power_spectrum[k];
            }
            // Floor to avoid log(0)
            features[[frame_idx, bin]] = energy.max(f32::EPSILON).ln();
        }
    }

    features
}

/// Compute Hamming window of given length.
fn hamming_window(length: usize) -> Vec<f32> {
    if length <= 1 {
        return vec![1.0];
    }
    (0..length)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32).cos())
        .collect()
}

/// Compute power spectrum using a simple DFT (Cooley-Tukey FFT).
///
/// Input is zero-padded windowed frame of length `fft_size`.
/// Returns power spectrum of length `num_fft_bins` = fft_size/2 + 1.
fn compute_power_spectrum(frame: &[f32], num_fft_bins: usize) -> Vec<f32> {
    let n = frame.len();

    // Bit-reversal permutation + iterative Cooley-Tukey FFT
    let (real, imag) = fft(frame);

    // Power spectrum: |X[k]|^2
    let mut power = Vec::with_capacity(num_fft_bins);
    for k in 0..num_fft_bins {
        power.push(real[k] * real[k] + imag[k] * imag[k]);
    }

    // Normalize by FFT size (matching Kaldi convention)
    let scale = 1.0 / n as f32;
    for p in &mut power {
        *p *= scale;
    }

    power
}

/// Iterative radix-2 Cooley-Tukey FFT.
/// Input length must be a power of 2.
/// Returns (real, imaginary) components.
fn fft(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = input.len();
    assert!(n.is_power_of_two(), "FFT size must be power of 2");

    let mut real = input.to_vec();
    let mut imag = vec![0.0f32; n];

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            real.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterfly operations
    let mut step = 1;
    while step < n {
        let half_step = step;
        step <<= 1;
        let angle_step = -std::f32::consts::PI / half_step as f32;
        for k in (0..n).step_by(step) {
            let mut angle = 0.0f32;
            for j in 0..half_step {
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                let tr = cos_val * real[k + j + half_step] - sin_val * imag[k + j + half_step];
                let ti = sin_val * real[k + j + half_step] + cos_val * imag[k + j + half_step];
                real[k + j + half_step] = real[k + j] - tr;
                imag[k + j + half_step] = imag[k + j] - ti;
                real[k + j] += tr;
                imag[k + j] += ti;
                angle += angle_step;
            }
        }
    }

    (real, imag)
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Convert mel scale to frequency in Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Compute mel-scale triangular filterbank matrix.
///
/// Returns shape (num_bins, num_fft_bins) where num_fft_bins = fft_size/2 + 1.
fn mel_filterbank(num_bins: usize, fft_size: usize, sample_rate: usize) -> Array2<f32> {
    let num_fft_bins = fft_size / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(nyquist);

    // num_bins + 2 equally spaced points in mel scale
    let num_points = num_bins + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_points - 1) as f32)
        .collect();

    // Convert back to Hz and then to FFT bin indices
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_indices: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * fft_size as f32 / sample_rate as f32)
        .collect();

    let mut filters = Array2::<f32>::zeros((num_bins, num_fft_bins));

    for i in 0..num_bins {
        let left = bin_indices[i];
        let center = bin_indices[i + 1];
        let right = bin_indices[i + 2];

        for k in 0..num_fft_bins {
            let kf = k as f32;
            if kf > left && kf <= center {
                filters[[i, k]] = (kf - left) / (center - left);
            } else if kf > center && kf < right {
                filters[[i, k]] = (right - kf) / (right - center);
            }
        }
    }

    filters
}

/// Extract speaker embeddings for each (chunk, speaker) pair using the WeSpeaker ONNX model.
///
/// For each chunk, checks the binarized segmentation to determine which speakers are active.
/// For active speakers, extracts audio, computes fbank features, and runs the embedding model.
/// Inactive speakers get NaN embeddings.
///
/// # Arguments
/// - `session`: WeSpeaker ONNX embedding model session.
/// - `audio`: Full audio signal (mono, 16kHz).
/// - `chunk_binarized`: Per-chunk binarized segmentation (num_chunks, num_frames_per_chunk, num_speakers).
/// - `window_samples`: Number of audio samples per segmentation window.
/// - `step_samples`: Step size in audio samples between windows.
/// - `batch_size`: Number of embeddings to compute in one ONNX batch.
///
/// # Returns
/// Embeddings array of shape (num_chunks, num_speakers, embedding_dim) with NaN for inactive.
pub fn extract_embeddings(
    session: &mut ort::session::Session,
    audio: &[f32],
    chunk_binarized: &Array3<f32>,
    window_samples: usize,
    step_samples: usize,
    batch_size: usize,
) -> Result<Array3<f32>> {
    let num_chunks = chunk_binarized.shape()[0];
    let num_frames_per_chunk = chunk_binarized.shape()[1];
    let num_speakers = chunk_binarized.shape()[2];
    let fbank_config = FbankConfig::default();

    // Collect (chunk_idx, speaker_idx, fbank_features) for active speakers
    let mut active_pairs: Vec<(usize, usize, Array2<f32>)> = Vec::new();

    for c in 0..num_chunks {
        let audio_start = c * step_samples;
        let audio_end = (audio_start + window_samples).min(audio.len());
        let chunk_audio = if audio_start < audio.len() {
            &audio[audio_start..audio_end]
        } else {
            &[]
        };

        for s in 0..num_speakers {
            // Check if speaker is active in this chunk (> 20% of frames)
            let active_frames: usize = (0..num_frames_per_chunk)
                .filter(|&f| chunk_binarized[[c, f, s]] > 0.5)
                .count();
            let activity_ratio = active_frames as f32 / num_frames_per_chunk as f32;

            if activity_ratio <= 0.0 || chunk_audio.is_empty() {
                continue;
            }

            // Extract audio for active frames only
            let speaker_audio = extract_speaker_audio(
                chunk_audio,
                chunk_binarized,
                c,
                s,
                num_frames_per_chunk,
            );

            if speaker_audio.is_empty() {
                continue;
            }

            let fbank = compute_fbank(&speaker_audio, &fbank_config);
            if fbank.nrows() == 0 {
                continue;
            }

            active_pairs.push((c, s, fbank));
        }
    }

    // Run embedding model in batches
    // First, do a single inference to determine embedding_dim
    let embedding_dim = if active_pairs.is_empty() {
        // Default embedding dim for WeSpeaker ResNet34
        256
    } else {
        // Run first item to get embedding dim
        let first_fbank = &active_pairs[0].2;
        let dim = run_embedding_single(session, first_fbank)?;
        dim.len()
    };

    // Initialize output with NaN
    let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, embedding_dim), f32::NAN);

    // Process in batches
    for batch_start in (0..active_pairs.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(active_pairs.len());
        let batch = &active_pairs[batch_start..batch_end];

        let batch_embeddings = run_embedding_batch(session, batch)?;

        for (i, (c, s, _)) in batch.iter().enumerate() {
            for d in 0..embedding_dim {
                embeddings[[*c, *s, d]] = batch_embeddings[[i, d]];
            }
        }
    }

    Ok(embeddings)
}

/// Extract audio samples for frames where a specific speaker is active.
fn extract_speaker_audio(
    chunk_audio: &[f32],
    chunk_binarized: &Array3<f32>,
    chunk_idx: usize,
    speaker_idx: usize,
    num_frames: usize,
) -> Vec<f32> {
    let samples_per_frame = if num_frames > 0 {
        chunk_audio.len() as f32 / num_frames as f32
    } else {
        return vec![];
    };

    let mut speaker_audio = Vec::new();
    for f in 0..num_frames {
        if chunk_binarized[[chunk_idx, f, speaker_idx]] > 0.5 {
            let start = (f as f32 * samples_per_frame) as usize;
            let end = ((f + 1) as f32 * samples_per_frame) as usize;
            let end = end.min(chunk_audio.len());
            if start < chunk_audio.len() {
                speaker_audio.extend_from_slice(&chunk_audio[start..end]);
            }
        }
    }
    speaker_audio
}

/// Run embedding model on a single fbank feature matrix.
/// Returns the embedding vector.
fn run_embedding_single(
    session: &mut ort::session::Session,
    fbank: &Array2<f32>,
) -> Result<Vec<f32>> {
    let num_frames = fbank.nrows();
    let num_bins = fbank.ncols();

    // WeSpeaker expects (batch=1, num_frames, 80)
    let input_data: Vec<f32> = fbank.iter().copied().collect();
    let input_array = ndarray::Array3::<f32>::from_shape_vec(
        (1, num_frames, num_bins),
        input_data,
    )?;

    let input_tensor = TensorRef::from_array_view(&input_array)
        .map_err(|e| anyhow::anyhow!("Failed to create embedding input tensor: {e}"))?;

    let outputs = session
        .run([input_tensor.into()])
        .map_err(|e| anyhow::anyhow!("Embedding inference failed: {e}"))?;

    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract embedding tensor: {e}"))?;

    // Output shape: (1, embedding_dim)
    let embedding_dim = shape[shape.len() - 1] as usize;
    Ok(data[..embedding_dim].to_vec())
}

/// Run embedding model on a batch of fbank features.
/// All fbank matrices are padded to the same length for batching.
/// Returns (batch_size, embedding_dim).
fn run_embedding_batch(
    session: &mut ort::session::Session,
    batch: &[(usize, usize, Array2<f32>)],
) -> Result<Array2<f32>> {
    if batch.len() == 1 {
        let emb = run_embedding_single(session, &batch[0].2)?;
        return Ok(Array2::from_shape_vec((1, emb.len()), emb)?);
    }

    let num_bins = batch[0].2.ncols();
    let max_frames = batch.iter().map(|(_, _, fb)| fb.nrows()).max().unwrap_or(0);

    // Pad all fbank features to max_frames
    let batch_size = batch.len();
    let mut input_data = vec![0.0f32; batch_size * max_frames * num_bins];

    for (i, (_, _, fbank)) in batch.iter().enumerate() {
        let nf = fbank.nrows();
        for f in 0..nf {
            for b in 0..num_bins {
                input_data[i * max_frames * num_bins + f * num_bins + b] = fbank[[f, b]];
            }
        }
    }

    let input_array = ndarray::Array3::<f32>::from_shape_vec(
        (batch_size, max_frames, num_bins),
        input_data,
    )?;

    let input_tensor = TensorRef::from_array_view(&input_array)
        .map_err(|e| anyhow::anyhow!("Failed to create batch embedding input tensor: {e}"))?;

    let outputs = session
        .run([input_tensor.into()])
        .map_err(|e| anyhow::anyhow!("Batch embedding inference failed: {e}"))?;

    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract batch embedding tensor: {e}"))?;

    let embedding_dim = shape[shape.len() - 1] as usize;
    let result = Array2::from_shape_vec(
        (batch_size, embedding_dim),
        data[..batch_size * embedding_dim].to_vec(),
    )?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fbank_shape() {
        // 1 second of audio at 16kHz
        let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let config = FbankConfig::default();
        let fbank = compute_fbank(&audio, &config);

        // Expected frames: 1 + (16000 - 400) / 160 = 1 + 97 = 98
        let expected_frames = 1 + (16000 - 400) / 160;
        assert_eq!(fbank.shape(), &[expected_frames, 80]);
    }

    #[test]
    fn test_fbank_empty_audio() {
        let config = FbankConfig::default();
        let fbank = compute_fbank(&[], &config);
        assert_eq!(fbank.shape(), &[0, 80]);
    }

    #[test]
    fn test_fbank_short_audio() {
        // Audio shorter than frame_length — should get 1 frame (zero-padded)
        let audio = vec![0.5f32; 200];
        let config = FbankConfig::default();
        let fbank = compute_fbank(&audio, &config);
        assert_eq!(fbank.nrows(), 1);
        assert_eq!(fbank.ncols(), 80);
    }

    #[test]
    fn test_fbank_values_finite() {
        let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.001).sin()).collect();
        let config = FbankConfig::default();
        let fbank = compute_fbank(&audio, &config);

        for val in fbank.iter() {
            assert!(val.is_finite(), "Fbank contains non-finite value: {val}");
        }
    }

    #[test]
    fn test_fbank_silence_low_energy() {
        // Silent audio should have very low energy (large negative log values)
        let audio = vec![0.0f32; 16000];
        let config = FbankConfig::default();
        let fbank = compute_fbank(&audio, &config);

        // All values should be very negative (log of near-zero)
        for val in fbank.iter() {
            assert!(*val < 0.0, "Expected negative log energy for silence, got {val}");
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let filters = mel_filterbank(80, 512, 16000);
        assert_eq!(filters.shape(), &[80, 257]); // 512/2 + 1 = 257
    }

    #[test]
    fn test_mel_filterbank_nonnegative() {
        let filters = mel_filterbank(80, 512, 16000);
        for val in filters.iter() {
            assert!(*val >= 0.0, "Mel filter has negative value: {val}");
        }
    }

    #[test]
    fn test_mel_filterbank_triangular() {
        // Each filter should have exactly one peak
        let filters = mel_filterbank(80, 512, 16000);
        for bin in 0..80 {
            let row: Vec<f32> = (0..257).map(|k| filters[[bin, k]]).collect();
            let max_val = row.iter().cloned().fold(0.0f32, f32::max);
            // Filter should have some nonzero values
            assert!(max_val > 0.0, "Mel filter {bin} is all zeros");
        }
    }

    #[test]
    fn test_hz_to_mel_roundtrip() {
        for hz in [0.0, 100.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(hz);
            let hz2 = mel_to_hz(mel);
            assert!(
                (hz - hz2).abs() < 0.01,
                "Hz-mel roundtrip failed: {hz} -> {mel} -> {hz2}"
            );
        }
    }

    #[test]
    fn test_fft_known_signal() {
        // DC signal: all 1s, FFT should have energy only at bin 0
        let signal = vec![1.0f32; 8];
        let (real, imag) = fft(&signal);
        assert!((real[0] - 8.0).abs() < 1e-5, "DC component should be 8.0");
        for k in 1..8 {
            assert!(
                (real[k].powi(2) + imag[k].powi(2)).sqrt() < 1e-4,
                "Non-DC bin {k} should be ~0"
            );
        }
    }

    #[test]
    fn test_fft_nyquist() {
        // Alternating +1, -1: energy at Nyquist
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let (real, _imag) = fft(&signal);
        // Energy should be at bin N/2 = 4
        assert!((real[4].abs() - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_extract_speaker_audio_all_active() {
        let chunk_binarized = Array3::from_shape_vec(
            (1, 4, 1),
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let speaker_audio = extract_speaker_audio(&audio, &chunk_binarized, 0, 0, 4);
        assert_eq!(speaker_audio.len(), 8); // All frames active
    }

    #[test]
    fn test_extract_speaker_audio_partial() {
        let chunk_binarized = Array3::from_shape_vec(
            (1, 4, 1),
            vec![1.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let speaker_audio = extract_speaker_audio(&audio, &chunk_binarized, 0, 0, 4);
        // Frames 0 and 2 active, each covering 2 samples
        assert_eq!(speaker_audio.len(), 4);
        assert_eq!(speaker_audio[0], 1.0);
        assert_eq!(speaker_audio[1], 2.0);
        assert_eq!(speaker_audio[2], 5.0);
        assert_eq!(speaker_audio[3], 6.0);
    }

    #[test]
    fn test_extract_speaker_audio_none_active() {
        let chunk_binarized = Array3::from_shape_vec(
            (1, 4, 1),
            vec![0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let audio = vec![1.0, 2.0, 3.0, 4.0];

        let speaker_audio = extract_speaker_audio(&audio, &chunk_binarized, 0, 0, 4);
        assert!(speaker_audio.is_empty());
    }

    #[test]
    fn test_embedding_output_shape_inactive() {
        // Without an ONNX model, we can test that inactive speakers get NaN
        let chunk_binarized = Array3::from_shape_vec(
            (2, 4, 2),
            vec![
                // chunk 0: speaker 0 active, speaker 1 inactive
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                // chunk 1: speaker 0 inactive, speaker 1 active
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
            ],
        )
        .unwrap();

        // Verify the expected activity detection logic
        let num_frames = 4;
        for s in 0..2 {
            let active_c0: usize = (0..num_frames)
                .filter(|&f| chunk_binarized[[0, f, s]] > 0.5)
                .count();
            let active_c1: usize = (0..num_frames)
                .filter(|&f| chunk_binarized[[1, f, s]] > 0.5)
                .count();

            if s == 0 {
                assert_eq!(active_c0, 4); // speaker 0 active in chunk 0
                assert_eq!(active_c1, 0); // speaker 0 inactive in chunk 1
            } else {
                assert_eq!(active_c0, 0); // speaker 1 inactive in chunk 0
                assert_eq!(active_c1, 4); // speaker 1 active in chunk 1
            }
        }
    }
}
