use anyhow::{Context, Result};
use ndarray::{Array2, Array3};
use ort::value::TensorRef;

/// Configuration for the sliding window segmentation.
pub struct SegmentationConfig {
    /// Window duration in seconds (default: 10.0).
    pub window_duration: f64,
    /// Step ratio — fraction of window duration (default: 0.1, giving 90% overlap).
    pub step_ratio: f64,
    /// Sample rate in Hz (default: 16000).
    pub sample_rate: usize,
    /// Binarization onset threshold (default: 0.5).
    pub threshold: f32,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            window_duration: 10.0,
            step_ratio: 0.1,
            sample_rate: 16000,
            threshold: 0.5,
        }
    }
}

impl SegmentationConfig {
    /// Number of samples per window.
    pub fn window_samples(&self) -> usize {
        (self.window_duration * self.sample_rate as f64) as usize
    }

    /// Step size in samples.
    pub fn step_samples(&self) -> usize {
        (self.window_duration * self.step_ratio * self.sample_rate as f64) as usize
    }
}

/// Result of the segmentation stage.
pub struct SegmentationOutput {
    /// Aggregated binary speaker activity: (num_frames, num_speakers), values 0.0 or 1.0.
    pub activity: Array2<f32>,
    /// Per-chunk raw segmentation outputs: (num_chunks, num_frames_per_chunk, num_speakers).
    pub chunk_segmentations: Array3<f32>,
    /// Per-chunk binarized segmentations: (num_chunks, num_frames_per_chunk, num_speakers).
    pub chunk_binarized: Array3<f32>,
    /// Duration of each output frame in seconds (determined by model stride).
    pub frame_duration: f64,
    /// Total audio duration in seconds.
    pub total_duration: f64,
    /// Sliding window parameters used.
    pub window_samples: usize,
    /// Step size in samples.
    pub step_samples: usize,
}

/// Generate sliding window start indices over audio of `num_samples` length.
///
/// Returns a vec of (start_sample, chunk_length) pairs.
/// The last window is zero-padded if it extends beyond the audio.
pub fn sliding_window_indices(
    num_samples: usize,
    window_samples: usize,
    step_samples: usize,
) -> Vec<(usize, usize)> {
    if num_samples == 0 {
        return vec![];
    }
    let mut indices = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + window_samples).min(num_samples);
        let chunk_len = end - start;
        indices.push((start, chunk_len));
        if start + window_samples >= num_samples {
            break;
        }
        start += step_samples;
    }
    indices
}

/// Extract a chunk from audio, zero-padding if shorter than `window_samples`.
fn extract_chunk(audio: &[f32], start: usize, chunk_len: usize, window_samples: usize) -> Vec<f32> {
    let mut chunk = vec![0.0f32; window_samples];
    chunk[..chunk_len].copy_from_slice(&audio[start..start + chunk_len]);
    chunk
}

/// Run the segmentation ONNX model on sliding windows over the audio.
///
/// The segmentation model expects input shape (batch, 1, num_samples) and
/// produces output shape (batch, num_frames, num_speakers).
pub fn run_segmentation(
    session: &mut ort::session::Session,
    audio: &[f32],
    config: &SegmentationConfig,
) -> Result<SegmentationOutput> {
    let num_samples = audio.len();
    let total_duration = num_samples as f64 / config.sample_rate as f64;
    let window_samples = config.window_samples();
    let step_samples = config.step_samples();

    let indices = sliding_window_indices(num_samples, window_samples, step_samples);
    let num_chunks = indices.len();

    if num_chunks == 0 {
        anyhow::bail!("No audio data to segment");
    }

    // Run model on each chunk, collecting outputs
    let mut chunk_outputs: Option<Array3<f32>> = None;
    let mut num_frames_per_chunk = 0usize;
    let mut num_speakers = 0usize;

    for (i, &(start, chunk_len)) in indices.iter().enumerate() {
        let chunk_data = extract_chunk(audio, start, chunk_len, window_samples);
        let input_array = ndarray::Array3::<f32>::from_shape_vec(
            (1, 1, window_samples),
            chunk_data,
        )?;
        let input_tensor = TensorRef::from_array_view(&input_array)
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;
        let outputs = session
            .run([input_tensor.into()])
            .map_err(|e| anyhow::anyhow!("Segmentation inference failed: {e}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output tensor: {e}"))?;

        // Initialize output array on first chunk
        if chunk_outputs.is_none() {
            num_frames_per_chunk = shape[1] as usize;
            num_speakers = shape[2] as usize;
            chunk_outputs =
                Some(Array3::<f32>::zeros((num_chunks, num_frames_per_chunk, num_speakers)));
        }

        let out = chunk_outputs.as_mut().unwrap();
        for f in 0..num_frames_per_chunk {
            for s in 0..num_speakers {
                out[[i, f, s]] = data[f * num_speakers + s];
            }
        }
    }

    let chunk_outputs = chunk_outputs.context("No chunks processed")?;

    // Frame duration: window covers window_duration seconds, model outputs num_frames_per_chunk frames
    let frame_duration = config.window_duration / num_frames_per_chunk as f64;

    // Binarize per-chunk (hysteresis thresholding, matching pyannote)
    let chunk_binarized = binarize_chunks(&chunk_outputs, config.threshold);

    // Aggregate overlapping windows with Hamming windowing
    let activity = aggregate_with_hamming(
        &chunk_outputs,
        num_frames_per_chunk,
        num_speakers,
        step_samples,
        window_samples,
        frame_duration,
        total_duration,
        config.threshold,
    );

    Ok(SegmentationOutput {
        activity,
        chunk_segmentations: chunk_outputs,
        chunk_binarized,
        frame_duration,
        total_duration,
        window_samples,
        step_samples,
    })
}

/// Aggregate overlapping chunk outputs using overlap-add with Hamming windowing.
///
/// Matches pyannote's `Inference.aggregate()` with `hamming=True`.
/// After aggregation, the result is binarized with the given threshold.
#[allow(clippy::too_many_arguments)]
fn aggregate_with_hamming(
    chunk_outputs: &Array3<f32>,
    num_frames_per_chunk: usize,
    num_speakers: usize,
    step_samples: usize,
    window_samples: usize,
    frame_duration: f64,
    total_duration: f64,
    threshold: f32,
) -> Array2<f32> {
    let num_chunks = chunk_outputs.shape()[0];

    // Compute total number of output frames
    let step_frames =
        (step_samples as f64 / window_samples as f64 * num_frames_per_chunk as f64).round()
            as usize;
    let num_frames = if num_chunks == 0 {
        0
    } else {
        num_frames_per_chunk + (num_chunks - 1) * step_frames
    };
    // Clamp to actual audio duration
    let max_frames = (total_duration / frame_duration).ceil() as usize;
    let num_frames = num_frames.min(max_frames);

    let mut aggregated = Array2::<f32>::zeros((num_frames, num_speakers));
    let mut weight_sum = Array2::<f32>::zeros((num_frames, num_speakers));

    // Hamming window
    let hamming = hamming_window(num_frames_per_chunk);

    for chunk_idx in 0..num_chunks {
        let start_frame = chunk_idx * step_frames;
        let end_frame = (start_frame + num_frames_per_chunk).min(num_frames);
        let usable_frames = end_frame - start_frame;

        for f in 0..usable_frames {
            let w = hamming[f];
            for s in 0..num_speakers {
                let val = chunk_outputs[[chunk_idx, f, s]];
                if !val.is_nan() {
                    aggregated[[start_frame + f, s]] += val * w;
                    weight_sum[[start_frame + f, s]] += w;
                }
            }
        }
    }

    // Average
    let epsilon = 1e-12f32;
    for f in 0..num_frames {
        for s in 0..num_speakers {
            let w = weight_sum[[f, s]];
            if w > epsilon {
                aggregated[[f, s]] /= w;
            }
        }
    }

    // Binarize the aggregated output
    binarize_aggregated(&aggregated, threshold)
}

/// Compute a Hamming window of length `n`.
pub fn hamming_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
        .collect()
}

/// Hysteresis binarization on per-chunk segmentation outputs.
///
/// Matches pyannote's `binarize_ndarray` from `signal.py`.
/// Input shape: (num_chunks, num_frames_per_chunk, num_speakers).
/// Applies per-speaker (column) hysteresis with onset=offset=threshold, initial_state=false.
pub fn binarize_chunks(scores: &Array3<f32>, threshold: f32) -> Array3<f32> {
    let (num_chunks, num_frames, num_speakers) = (
        scores.shape()[0],
        scores.shape()[1],
        scores.shape()[2],
    );
    let mut result = Array3::<f32>::zeros((num_chunks, num_frames, num_speakers));

    // pyannote binarize operates on (batch_size, num_frames) where batch_size = num_speakers
    // with onset = offset = threshold, initial_state = False
    for c in 0..num_chunks {
        for s in 0..num_speakers {
            let mut active = false; // initial_state = False
            for f in 0..num_frames {
                let val = scores[[c, f, s]];
                let val = if val.is_nan() { 0.0 } else { val };
                if val > threshold {
                    active = true;
                } else if val < threshold {
                    // offset == onset, so < threshold turns off
                    active = false;
                }
                // if val == threshold, state doesn't change (hysteresis)
                result[[c, f, s]] = if active { 1.0 } else { 0.0 };
            }
        }
    }

    result
}

/// Simple threshold binarization on aggregated (num_frames, num_speakers) output.
fn binarize_aggregated(scores: &Array2<f32>, threshold: f32) -> Array2<f32> {
    let (num_frames, num_speakers) = (scores.shape()[0], scores.shape()[1]);
    let mut result = Array2::<f32>::zeros((num_frames, num_speakers));

    for f in 0..num_frames {
        for s in 0..num_speakers {
            let val = scores[[f, s]];
            let val = if val.is_nan() { 0.0 } else { val };
            if val > threshold {
                result[[f, s]] = 1.0;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_indices_basic() {
        // 3 seconds of audio at 16kHz, 10s window, 1s step
        // Only one window since audio < window
        let indices = sliding_window_indices(48000, 160000, 16000);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], (0, 48000));
    }

    #[test]
    fn test_sliding_window_indices_exact() {
        // Exactly 10s of audio = 160000 samples
        let indices = sliding_window_indices(160000, 160000, 16000);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], (0, 160000));
    }

    #[test]
    fn test_sliding_window_indices_overlap() {
        // 20s of audio = 320000 samples, 10s window, 1s step
        // Windows: 0..160000, 16000..176000, ..., up to covering 320000
        let indices = sliding_window_indices(320000, 160000, 16000);

        // First window starts at 0
        assert_eq!(indices[0].0, 0);
        assert_eq!(indices[0].1, 160000);

        // Second window starts at 16000
        assert_eq!(indices[1].0, 16000);
        assert_eq!(indices[1].1, 160000);

        // Verify overlap: consecutive windows overlap by 90%
        let overlap = indices[0].1 as f64 - (indices[1].0 - indices[0].0) as f64;
        let overlap_ratio = overlap / 160000.0;
        assert!((overlap_ratio - 0.9).abs() < 0.01);

        // Last window should cover the end
        let last = indices.last().unwrap();
        assert!(last.0 + last.1 >= 320000 || last.0 + 160000 >= 320000);

        // Verify full coverage: every sample is covered by at least one window
        for sample in (0..320000).step_by(1000) {
            let covered = indices
                .iter()
                .any(|&(start, len)| sample >= start && sample < start + len);
            assert!(covered, "Sample {sample} not covered");
        }
    }

    #[test]
    fn test_sliding_window_indices_empty() {
        let indices = sliding_window_indices(0, 160000, 16000);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_sliding_window_indices_short_audio() {
        // Audio shorter than window — single zero-padded window
        let indices = sliding_window_indices(8000, 160000, 16000);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], (0, 8000));
    }

    #[test]
    fn test_hamming_window_length() {
        let w = hamming_window(10);
        assert_eq!(w.len(), 10);
    }

    #[test]
    fn test_hamming_window_symmetry() {
        let w = hamming_window(100);
        for i in 0..50 {
            assert!(
                (w[i] - w[99 - i]).abs() < 1e-6,
                "Hamming window not symmetric at {i}: {} vs {}",
                w[i],
                w[99 - i]
            );
        }
    }

    #[test]
    fn test_hamming_window_endpoints() {
        let w = hamming_window(100);
        // Hamming window endpoints are 0.08 (= 0.54 - 0.46)
        assert!((w[0] - 0.08).abs() < 1e-5);
        assert!((w[99] - 0.08).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_window_peak() {
        let w = hamming_window(101);
        // Peak at center should be 1.0
        assert!((w[50] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_window_single() {
        let w = hamming_window(1);
        assert_eq!(w, vec![1.0]);
    }

    #[test]
    fn test_binarize_chunks_basic() {
        // (1 chunk, 5 frames, 2 speakers)
        let data = Array3::from_shape_vec(
            (1, 5, 2),
            vec![
                0.1, 0.8, // frame 0: spk0 off, spk1 on
                0.6, 0.3, // frame 1: spk0 on, spk1 off
                0.7, 0.2, // frame 2: spk0 on, spk1 off
                0.4, 0.9, // frame 3: spk0 off, spk1 on
                0.2, 0.6, // frame 4: spk0 off, spk1 on
            ],
        )
        .unwrap();

        let bin = binarize_chunks(&data, 0.5);
        assert_eq!(bin[[0, 0, 0]], 0.0); // 0.1 < 0.5
        assert_eq!(bin[[0, 0, 1]], 1.0); // 0.8 > 0.5
        assert_eq!(bin[[0, 1, 0]], 1.0); // 0.6 > 0.5
        assert_eq!(bin[[0, 1, 1]], 0.0); // 0.3 < 0.5
        assert_eq!(bin[[0, 3, 0]], 0.0); // 0.4 < 0.5
        assert_eq!(bin[[0, 3, 1]], 1.0); // 0.9 > 0.5
    }

    #[test]
    fn test_binarize_chunks_hysteresis() {
        // Test that exact threshold value preserves state (hysteresis behavior)
        let data = Array3::from_shape_vec(
            (1, 4, 1),
            vec![
                0.6, // > 0.5 → active
                0.5, // == 0.5 → stays active (hysteresis)
                0.4, // < 0.5 → inactive
                0.5, // == 0.5 → stays inactive (hysteresis)
            ],
        )
        .unwrap();

        let bin = binarize_chunks(&data, 0.5);
        assert_eq!(bin[[0, 0, 0]], 1.0);
        assert_eq!(bin[[0, 1, 0]], 1.0); // hysteresis: stays active
        assert_eq!(bin[[0, 2, 0]], 0.0);
        assert_eq!(bin[[0, 3, 0]], 0.0); // hysteresis: stays inactive
    }

    #[test]
    fn test_binarize_chunks_nan_handling() {
        let data = Array3::from_shape_vec(
            (1, 3, 1),
            vec![f32::NAN, 0.8, f32::NAN],
        )
        .unwrap();

        let bin = binarize_chunks(&data, 0.5);
        assert_eq!(bin[[0, 0, 0]], 0.0); // NaN → 0.0 < threshold
        assert_eq!(bin[[0, 1, 0]], 1.0);
        assert_eq!(bin[[0, 2, 0]], 0.0); // NaN → 0.0 < threshold
    }

    #[test]
    fn test_binarize_aggregated() {
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![0.3, 0.7, 0.6, 0.4, 0.1, 0.9],
        )
        .unwrap();

        let bin = binarize_aggregated(&data, 0.5);
        assert_eq!(bin[[0, 0]], 0.0);
        assert_eq!(bin[[0, 1]], 1.0);
        assert_eq!(bin[[1, 0]], 1.0);
        assert_eq!(bin[[1, 1]], 0.0);
        assert_eq!(bin[[2, 0]], 0.0);
        assert_eq!(bin[[2, 1]], 1.0);
    }

    #[test]
    fn test_aggregate_single_chunk() {
        // Single chunk — aggregation should just be hamming-weighted average (= identity for 1 chunk)
        let num_frames = 10;
        let num_speakers = 2;
        let mut data = Array3::<f32>::zeros((1, num_frames, num_speakers));
        // Set all speaker 0 to 0.8, speaker 1 to 0.3
        for f in 0..num_frames {
            data[[0, f, 0]] = 0.8;
            data[[0, f, 1]] = 0.3;
        }

        let activity = aggregate_with_hamming(
            &data,
            num_frames,
            num_speakers,
            2,    // step_frames won't matter for 1 chunk
            10,   // window_samples
            0.01, // frame_duration (10 frames over 0.1s)
            0.1,  // total_duration
            0.5,
        );

        // With single chunk, output should be binarized version
        for f in 0..activity.shape()[0] {
            assert_eq!(activity[[f, 0]], 1.0, "Speaker 0 should be active at frame {f}");
            assert_eq!(activity[[f, 1]], 0.0, "Speaker 1 should be inactive at frame {f}");
        }
    }

    #[test]
    fn test_aggregate_overlap_blending() {
        // Two overlapping chunks with different values
        // Chunk 0: 4 frames, all [0.8, 0.2]
        // Chunk 1: 4 frames, all [0.2, 0.8]
        // Step = 2 frames, so frames 2-3 overlap
        let num_frames_per_chunk = 4;
        let num_speakers = 2;
        let mut data = Array3::<f32>::zeros((2, num_frames_per_chunk, num_speakers));

        for f in 0..num_frames_per_chunk {
            data[[0, f, 0]] = 0.8;
            data[[0, f, 1]] = 0.2;
            data[[1, f, 0]] = 0.2;
            data[[1, f, 1]] = 0.8;
        }

        let activity = aggregate_with_hamming(
            &data,
            num_frames_per_chunk,
            num_speakers,
            2,    // step_samples (maps to 2 step_frames)
            4,    // window_samples
            0.01, // frame_duration
            0.06, // total_duration = 6 frames * 0.01
            0.5,
        );

        // Frames 0-1: only chunk 0 → 0.8/0.2 → binary 1.0/0.0
        assert_eq!(activity[[0, 0]], 1.0);
        assert_eq!(activity[[0, 1]], 0.0);

        // Frames 4-5: only chunk 1 → 0.2/0.8 → binary 0.0/1.0
        assert_eq!(activity[[4, 0]], 0.0);
        assert_eq!(activity[[4, 1]], 1.0);

        // Frames 2-3: overlap of both chunks — blended values
        // The exact values depend on Hamming weights, but should be around 0.5
        // After binarization with threshold 0.5, result depends on exact blend
    }

    #[test]
    fn test_config_defaults() {
        let config = SegmentationConfig::default();
        assert_eq!(config.window_samples(), 160000);
        assert_eq!(config.step_samples(), 16000);
        assert!((config.threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extract_chunk_zero_padding() {
        let audio = vec![1.0f32, 2.0, 3.0];
        let chunk = extract_chunk(&audio, 0, 3, 5);
        assert_eq!(chunk, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_extract_chunk_full() {
        let audio = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let chunk = extract_chunk(&audio, 1, 4, 4);
        assert_eq!(chunk, vec![2.0, 3.0, 4.0, 5.0]);
    }
}
