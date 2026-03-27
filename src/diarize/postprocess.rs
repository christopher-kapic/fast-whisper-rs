use crate::inference::Segment;

/// A diarization segment with start/end times and speaker label.
#[derive(Debug, Clone)]
pub struct DiarizationSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

/// A transcript chunk with an assigned speaker label.
#[derive(Debug, Clone)]
pub struct SpeakerChunk {
    pub speaker: String,
    pub text: String,
    pub start: f64,
    pub end: Option<f64>,
}

/// Reconstruct a speaker timeline from per-chunk binarized segmentation and cluster assignments.
///
/// For each chunk, extracts contiguous active regions per speaker, maps local speaker indices
/// to global speaker IDs via cluster assignments, then merges overlapping same-speaker segments.
pub fn reconstruct_timeline(
    chunk_binarized: &ndarray::Array3<f32>,
    assignments: &std::collections::HashMap<(usize, usize), usize>,
    step_duration: f64,
    frame_duration: f64,
) -> Vec<DiarizationSegment> {
    let num_chunks = chunk_binarized.shape()[0];
    let num_frames = chunk_binarized.shape()[1];
    let num_speakers = chunk_binarized.shape()[2];

    let mut segments = Vec::new();

    for c in 0..num_chunks {
        let chunk_start = c as f64 * step_duration;

        for s in 0..num_speakers {
            let global_id = match assignments.get(&(c, s)) {
                Some(&id) => id,
                None => continue,
            };

            let speaker = format!("SPEAKER_{global_id:02}");

            // Find contiguous active frame runs for this speaker in this chunk
            let mut region_start: Option<usize> = None;
            for f in 0..num_frames {
                let active = chunk_binarized[[c, f, s]] > 0.5;
                match (active, region_start) {
                    (true, None) => region_start = Some(f),
                    (false, Some(start)) => {
                        segments.push(DiarizationSegment {
                            start: chunk_start + start as f64 * frame_duration,
                            end: chunk_start + f as f64 * frame_duration,
                            speaker: speaker.clone(),
                        });
                        region_start = None;
                    }
                    _ => {}
                }
            }
            if let Some(start) = region_start {
                segments.push(DiarizationSegment {
                    start: chunk_start + start as f64 * frame_duration,
                    end: chunk_start + num_frames as f64 * frame_duration,
                    speaker: speaker.clone(),
                });
            }
        }
    }

    // Merge overlapping segments per speaker, then sort by start time
    merge_overlapping_per_speaker(&mut segments);
    segments
}

/// Merge overlapping or adjacent segments that belong to the same speaker.
fn merge_overlapping_per_speaker(segments: &mut Vec<DiarizationSegment>) {
    if segments.is_empty() {
        return;
    }

    // Sort by speaker, then start time
    segments.sort_by(|a, b| {
        a.speaker
            .cmp(&b.speaker)
            .then(a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for seg in segments.iter().skip(1) {
        if seg.speaker == current.speaker && seg.start <= current.end {
            current.end = current.end.max(seg.end);
        } else {
            merged.push(current);
            current = seg.clone();
        }
    }
    merged.push(current);

    // Sort by start time
    merged.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    *segments = merged;
}

/// Merge consecutive segments from the same speaker into super-segments.
///
/// Matches diarize.py lines 81-112 exactly:
/// - When speaker changes, the super-segment end = next segment's start
/// - Last segment keeps its actual end time
pub fn merge_segments(segments: &[DiarizationSegment]) -> Vec<DiarizationSegment> {
    if segments.is_empty() {
        return vec![];
    }

    let mut new_segments = Vec::new();
    let mut prev = &segments[0];

    for cur in segments.iter().skip(1) {
        if cur.speaker != prev.speaker {
            new_segments.push(DiarizationSegment {
                start: prev.start,
                end: cur.start, // KEY: end = next segment's start
                speaker: prev.speaker.clone(),
            });
            prev = cur;
        }
    }

    // Add the last segment(s)
    let last = segments.last().unwrap();
    new_segments.push(DiarizationSegment {
        start: prev.start,
        end: last.end,
        speaker: prev.speaker.clone(),
    });

    new_segments
}

/// Align diarization segments with ASR transcript chunks using argmin on end timestamps.
///
/// Matches diarize.py post_process_segments_and_transcripts (lines 115-152) with
/// group_by_speaker=False: each ASR chunk gets a speaker label individually.
pub fn align_with_transcript(
    diarization_segments: &[DiarizationSegment],
    transcript: &[Segment],
) -> Vec<SpeakerChunk> {
    if diarization_segments.is_empty() || transcript.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();
    let mut offset = 0usize; // tracks how much of transcript has been consumed

    for segment in diarization_segments {
        let remaining = &transcript[offset..];
        if remaining.is_empty() {
            break;
        }

        let end_time = segment.end;

        // Build end_timestamps for remaining transcript chunks
        // None end timestamps are treated as f64::MAX (matching Python sys.float_info.max)
        let end_timestamps: Vec<f64> = remaining
            .iter()
            .map(|chunk| chunk.end.unwrap_or(f64::MAX))
            .collect();

        // Find argmin of |end_timestamps - end_time|
        let upto_idx = end_timestamps
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (**a - end_time).abs();
                let db = (**b - end_time).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Assign speaker to all chunks up to and including upto_idx
        for chunk in remaining.iter().take(upto_idx + 1) {
            result.push(SpeakerChunk {
                speaker: segment.speaker.clone(),
                text: chunk.text.clone(),
                start: chunk.start,
                end: chunk.end,
            });
        }

        offset += upto_idx + 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- merge_segments tests ---

    #[test]
    fn test_merge_consecutive_same_speaker() {
        let segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 1.0,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 1.0,
                end: 2.0,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 2.0,
                end: 3.0,
                speaker: "SPEAKER_00".to_string(),
            },
        ];

        let merged = merge_segments(&segments);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].speaker, "SPEAKER_00");
        assert!((merged[0].start - 0.0).abs() < 1e-9);
        assert!((merged[0].end - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_merge_alternating_speakers() {
        let segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 1.5,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 1.5,
                end: 3.0,
                speaker: "SPEAKER_01".to_string(),
            },
            DiarizationSegment {
                start: 3.0,
                end: 5.0,
                speaker: "SPEAKER_00".to_string(),
            },
        ];

        let merged = merge_segments(&segments);
        assert_eq!(merged.len(), 3);

        // First: SPEAKER_00 0.0 -> 1.5 (end = next segment's start)
        assert_eq!(merged[0].speaker, "SPEAKER_00");
        assert!((merged[0].start - 0.0).abs() < 1e-9);
        assert!((merged[0].end - 1.5).abs() < 1e-9);

        // Second: SPEAKER_01 1.5 -> 3.0 (end = next segment's start)
        assert_eq!(merged[1].speaker, "SPEAKER_01");
        assert!((merged[1].start - 1.5).abs() < 1e-9);
        assert!((merged[1].end - 3.0).abs() < 1e-9);

        // Third: SPEAKER_00 3.0 -> 5.0 (last segment keeps actual end)
        assert_eq!(merged[2].speaker, "SPEAKER_00");
        assert!((merged[2].start - 3.0).abs() < 1e-9);
        assert!((merged[2].end - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_merge_single_segment() {
        let segments = vec![DiarizationSegment {
            start: 0.0,
            end: 5.0,
            speaker: "SPEAKER_00".to_string(),
        }];

        let merged = merge_segments(&segments);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].speaker, "SPEAKER_00");
        assert!((merged[0].start - 0.0).abs() < 1e-9);
        assert!((merged[0].end - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_merge_empty() {
        let segments: Vec<DiarizationSegment> = vec![];
        let merged = merge_segments(&segments);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_speaker_change_end_uses_next_start() {
        // Key behavior: when speaker changes, super-segment end = next segment's start
        let segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 2.0,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 1.5,
                end: 4.0,
                speaker: "SPEAKER_01".to_string(),
            },
        ];

        let merged = merge_segments(&segments);
        assert_eq!(merged.len(), 2);
        // First segment end should be 1.5 (next segment's start), not 2.0
        assert!((merged[0].end - 1.5).abs() < 1e-9);
        // Last segment keeps actual end
        assert!((merged[1].end - 4.0).abs() < 1e-9);
    }

    // --- align_with_transcript tests ---

    #[test]
    fn test_align_exact_match() {
        let diar_segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 2.5,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 2.5,
                end: 5.0,
                speaker: "SPEAKER_01".to_string(),
            },
        ];

        let transcript = vec![
            Segment {
                text: " Hello.".to_string(),
                start: 0.0,
                end: Some(2.5),
            },
            Segment {
                text: " World.".to_string(),
                start: 2.5,
                end: Some(5.0),
            },
        ];

        let result = align_with_transcript(&diar_segments, &transcript);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].speaker, "SPEAKER_00");
        assert_eq!(result[0].text, " Hello.");
        assert_eq!(result[1].speaker, "SPEAKER_01");
        assert_eq!(result[1].text, " World.");
    }

    #[test]
    fn test_align_closest_match() {
        // Diar end=3.0 is closest to chunk end 2.5 (not 5.0), so takes 1 chunk
        // Diar end=7.0 covers remaining 2 chunks
        let diar_segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 3.0,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 3.0,
                end: 7.0,
                speaker: "SPEAKER_01".to_string(),
            },
        ];

        let transcript = vec![
            Segment {
                text: " First.".to_string(),
                start: 0.0,
                end: Some(2.5),
            },
            Segment {
                text: " Second.".to_string(),
                start: 2.5,
                end: Some(5.0),
            },
            Segment {
                text: " Third.".to_string(),
                start: 5.0,
                end: Some(7.0),
            },
        ];

        let result = align_with_transcript(&diar_segments, &transcript);
        assert_eq!(result.len(), 3);
        // First diar segment (end=3.0) closest to chunk end 2.5
        assert_eq!(result[0].speaker, "SPEAKER_00");
        assert_eq!(result[0].text, " First.");
        // Remaining chunks get SPEAKER_01 (end=7.0 matches chunk end 7.0)
        assert_eq!(result[1].speaker, "SPEAKER_01");
        assert_eq!(result[2].speaker, "SPEAKER_01");
    }

    #[test]
    fn test_align_null_end_timestamps() {
        // Diar end=5.0: |2.5-5.0|=2.5, |MAX-5.0|≈MAX → argmin picks chunk 0
        // Only 1 chunk assigned per diar segment (upto_idx=0).
        // Need a second diar segment to cover chunk 1.
        let diar_segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 5.0,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 5.0,
                end: 10.0,
                speaker: "SPEAKER_00".to_string(),
            },
        ];

        let transcript = vec![
            Segment {
                text: " Hello.".to_string(),
                start: 0.0,
                end: Some(2.5),
            },
            Segment {
                text: " Final.".to_string(),
                start: 2.5,
                end: None, // null end → f64::MAX
            },
        ];

        let result = align_with_transcript(&diar_segments, &transcript);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].speaker, "SPEAKER_00");
        assert_eq!(result[0].text, " Hello.");
        // Second chunk with null end: |MAX-10.0|≈MAX, picked as closest remaining
        assert_eq!(result[1].speaker, "SPEAKER_00");
        assert_eq!(result[1].text, " Final.");
        assert_eq!(result[1].end, None);
    }

    #[test]
    fn test_align_empty_diarization() {
        let transcript = vec![Segment {
            text: " Hello.".to_string(),
            start: 0.0,
            end: Some(2.5),
        }];

        let result = align_with_transcript(&[], &transcript);
        assert!(result.is_empty());
    }

    #[test]
    fn test_align_empty_transcript() {
        let diar_segments = vec![DiarizationSegment {
            start: 0.0,
            end: 5.0,
            speaker: "SPEAKER_00".to_string(),
        }];

        let result = align_with_transcript(&diar_segments, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_align_multiple_chunks_per_speaker() {
        // One diarization segment covers multiple ASR chunks
        let diar_segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 5.0,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 5.0,
                end: 8.0,
                speaker: "SPEAKER_01".to_string(),
            },
        ];

        let transcript = vec![
            Segment {
                text: " A.".to_string(),
                start: 0.0,
                end: Some(2.0),
            },
            Segment {
                text: " B.".to_string(),
                start: 2.0,
                end: Some(4.0),
            },
            Segment {
                text: " C.".to_string(),
                start: 4.0,
                end: Some(6.0),
            },
            Segment {
                text: " D.".to_string(),
                start: 6.0,
                end: Some(8.0),
            },
        ];

        let result = align_with_transcript(&diar_segments, &transcript);
        assert_eq!(result.len(), 4);
        // Diar end=5.0: closest to chunk end 4.0 (|4-5|=1) vs 6.0 (|6-5|=1) — tie goes to first (argmin)
        // Actually argmin with equal distances picks first occurrence
        assert_eq!(result[0].speaker, "SPEAKER_00");
        assert_eq!(result[1].speaker, "SPEAKER_00");
        // Remaining chunks get SPEAKER_01
        assert_eq!(result[2].speaker, "SPEAKER_01");
        assert_eq!(result[3].speaker, "SPEAKER_01");
    }

    // --- reconstruct_timeline tests ---

    #[test]
    fn test_reconstruct_timeline_basic() {
        use ndarray::Array3;
        use std::collections::HashMap;

        // 2 chunks, 4 frames each, 2 speakers
        let mut binarized = Array3::<f32>::zeros((2, 4, 2));
        // Chunk 0: speaker 0 active in frames 0-3
        for f in 0..4 {
            binarized[[0, f, 0]] = 1.0;
        }
        // Chunk 1: speaker 1 active in frames 0-3
        for f in 0..4 {
            binarized[[1, f, 1]] = 1.0;
        }

        let mut assignments = HashMap::new();
        assignments.insert((0, 0), 0); // chunk 0, speaker 0 → global 0
        assignments.insert((1, 1), 1); // chunk 1, speaker 1 → global 1

        let segments = reconstruct_timeline(&binarized, &assignments, 1.0, 0.1);

        assert_eq!(segments.len(), 2);
        // First segment: SPEAKER_00 at chunk 0
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert!((segments[0].start - 0.0).abs() < 1e-9);
        // Second segment: SPEAKER_01 at chunk 1
        assert_eq!(segments[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn test_reconstruct_timeline_empty_assignments() {
        use ndarray::Array3;
        use std::collections::HashMap;

        let binarized = Array3::<f32>::zeros((1, 4, 2));
        let assignments = HashMap::new();

        let segments = reconstruct_timeline(&binarized, &assignments, 1.0, 0.1);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_reconstruct_overlapping_chunks_merge() {
        use ndarray::Array3;
        use std::collections::HashMap;

        // 2 overlapping chunks, same speaker active in both
        let mut binarized = Array3::<f32>::zeros((2, 4, 1));
        for f in 0..4 {
            binarized[[0, f, 0]] = 1.0;
            binarized[[1, f, 0]] = 1.0;
        }

        let mut assignments = HashMap::new();
        assignments.insert((0, 0), 0);
        assignments.insert((1, 0), 0);

        // step_duration=0.2, frame_duration=0.1
        // chunk 0: 0.0..0.4, chunk 1: 0.2..0.6
        let segments = reconstruct_timeline(&binarized, &assignments, 0.2, 0.1);

        // Should merge into single segment 0.0..0.6
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert!((segments[0].start - 0.0).abs() < 1e-9);
        assert!((segments[0].end - 0.6).abs() < 1e-9);
    }

    // --- Integration test: known segments + transcript ---

    #[test]
    fn test_integration_diarization_alignment() {
        // Simulate a 2-speaker conversation
        let diar_segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 3.0,
                speaker: "SPEAKER_00".to_string(),
            },
            DiarizationSegment {
                start: 3.0,
                end: 6.0,
                speaker: "SPEAKER_01".to_string(),
            },
            DiarizationSegment {
                start: 6.0,
                end: 10.0,
                speaker: "SPEAKER_00".to_string(),
            },
        ];

        let merged = merge_segments(&diar_segments);

        let transcript = vec![
            Segment {
                text: " Hi there.".to_string(),
                start: 0.0,
                end: Some(2.8),
            },
            Segment {
                text: " How are you?".to_string(),
                start: 2.8,
                end: Some(5.5),
            },
            Segment {
                text: " I'm fine thanks.".to_string(),
                start: 5.5,
                end: Some(9.0),
            },
        ];

        let result = align_with_transcript(&merged, &transcript);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].speaker, "SPEAKER_00");
        assert_eq!(result[0].text, " Hi there.");
        assert_eq!(result[1].speaker, "SPEAKER_01");
        assert_eq!(result[1].text, " How are you?");
        assert_eq!(result[2].speaker, "SPEAKER_00");
        assert_eq!(result[2].text, " I'm fine thanks.");
    }
}
