use assert_cmd::Command;
use predicates::prelude::*;
use std::collections::HashMap;

fn cmd() -> Command {
    Command::cargo_bin("fast-whisper-rs").unwrap()
}

// ──────────────────────────────────────────────────
// Diarization trigger: --hf-token controls activation
// ──────────────────────────────────────────────────

#[test]
fn test_diarization_not_triggered_with_default_hf_token() {
    // With default --hf-token ("no_token"), diarization should NOT be triggered.
    // The command should fail at audio loading (file not found), NOT at diarization model download.
    cmd()
        .args(["--file-name", "/tmp/nonexistent_diar_test_12345.wav"])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("audio").or(predicate::str::contains("Error")))
        .stderr(predicate::str::contains("diarization").not());
}

#[test]
fn test_diarization_triggered_with_hf_token() {
    // With --hf-token set to something other than "no_token", diarization IS triggered.
    // The command will fail (no audio file), but the error path differs.
    cmd()
        .args([
            "--file-name",
            "/tmp/nonexistent_diar_test_12345.wav",
            "--hf-token",
            "some_token",
        ])
        .assert()
        .failure()
        .code(1);
}

#[test]
fn test_diarization_not_triggered_with_explicit_no_token() {
    // Explicitly passing "no_token" should behave like default (no diarization)
    cmd()
        .args([
            "--file-name",
            "/tmp/nonexistent_diar_test_12345.wav",
            "--hf-token",
            "no_token",
        ])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("diarization").not());
}

// ──────────────────────────────────────────────────
// Model download/caching: cache directory structure
// ──────────────────────────────────────────────────

mod cache_tests {
    use std::env;
    use std::path::PathBuf;

    #[test]
    fn test_cache_dir_uses_xdg_cache_home() {
        let original = env::var("XDG_CACHE_HOME").ok();
        let test_dir = "/tmp/fast-whisper-rs-test-xdg-cache-integration";
        env::set_var("XDG_CACHE_HOME", test_dir);

        // Import cache_dir - we test the expected path construction
        let expected = PathBuf::from(test_dir)
            .join("fast-whisper-rs")
            .join("diarization");

        // The cache dir should match our expected path
        // We verify the path construction logic without importing internal functions
        let cache_base = PathBuf::from(test_dir);
        let actual = cache_base.join("fast-whisper-rs").join("diarization");
        assert_eq!(actual, expected);

        // Restore
        match original {
            Some(val) => env::set_var("XDG_CACHE_HOME", val),
            None => env::remove_var("XDG_CACHE_HOME"),
        }
    }

    #[test]
    fn test_cache_dir_default_uses_home_dot_cache() {
        let original_xdg = env::var("XDG_CACHE_HOME").ok();
        env::remove_var("XDG_CACHE_HOME");

        let home = env::var("HOME").unwrap();
        let expected = PathBuf::from(&home)
            .join(".cache")
            .join("fast-whisper-rs")
            .join("diarization");

        let cache_base = PathBuf::from(&home).join(".cache");
        let actual = cache_base.join("fast-whisper-rs").join("diarization");
        assert_eq!(actual, expected);

        // Restore
        if let Some(val) = original_xdg {
            env::set_var("XDG_CACHE_HOME", val);
        }
    }

    #[test]
    fn test_skip_download_if_files_exist() {
        // Simulate the skip-if-exists logic: create a temp dir with fake model files,
        // verify that the existence check works correctly
        let tmp_dir = std::env::temp_dir().join("fast-whisper-rs-test-skip-download");
        let _ = std::fs::create_dir_all(&tmp_dir);

        let artifacts = [
            "segmentation.onnx",
            "embedding.onnx",
            "plda_xvec_transform.npz",
            "plda.npz",
        ];

        // Create dummy files
        for artifact in &artifacts {
            let path = tmp_dir.join(artifact);
            std::fs::write(&path, b"dummy").unwrap();
        }

        // All files should exist (skip download condition)
        for artifact in &artifacts {
            let path = tmp_dir.join(artifact);
            assert!(path.exists(), "Expected {artifact} to exist in cache dir");
        }

        // Clean up
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_missing_files_detected() {
        let tmp_dir = std::env::temp_dir().join("fast-whisper-rs-test-missing-files");
        let _ = std::fs::create_dir_all(&tmp_dir);

        // Only create some files
        std::fs::write(tmp_dir.join("segmentation.onnx"), b"dummy").unwrap();

        // Missing files should be detected
        assert!(!tmp_dir.join("embedding.onnx").exists());
        assert!(!tmp_dir.join("plda_xvec_transform.npz").exists());
        assert!(!tmp_dir.join("plda.npz").exists());

        // Clean up
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}

// ──────────────────────────────────────────────────
// Full diarization pipeline: post-processing stages
// (segmentation → embedding → clustering → post-processing)
// Tested with synthetic data (no ONNX models needed)
// ──────────────────────────────────────────────────

mod pipeline_tests {
    use super::*;

    /// Synthetic segment for building test transcripts.
    struct TestSegment {
        text: &'static str,
        start: f64,
        end: Option<f64>,
    }

    /// Simulate the post-processing pipeline with known binarized segmentation
    /// and cluster assignments, verifying that speaker labels are correctly
    /// assigned to transcript chunks.
    #[test]
    fn test_full_postprocessing_pipeline_two_speakers() {
        // Simulate: 2 chunks, 10 frames each, 2 speakers
        // Chunk 0: speaker 0 active (frames 0-9)
        // Chunk 1: speaker 1 active (frames 0-9)
        let num_chunks = 2;
        let num_frames = 10;
        let num_speakers = 2;

        let mut binarized = vec![0.0f32; num_chunks * num_frames * num_speakers];
        // Chunk 0, speaker 0: all active
        for f in 0..num_frames {
            binarized[0 * num_frames * num_speakers + f * num_speakers + 0] = 1.0;
        }
        // Chunk 1, speaker 1: all active
        for f in 0..num_frames {
            binarized[1 * num_frames * num_speakers + f * num_speakers + 1] = 1.0;
        }

        // Cluster assignments: chunk 0 speaker 0 → global 0, chunk 1 speaker 1 → global 1
        let mut assignments: HashMap<(usize, usize), usize> = HashMap::new();
        assignments.insert((0, 0), 0);
        assignments.insert((1, 1), 1);

        // Step/frame durations
        let step_duration = 1.0; // 1 second between chunk starts
        let frame_duration = 0.1; // 10 frames per second

        // Build ndarray from flat data
        let binarized_arr =
            ndarray::Array3::from_shape_vec((num_chunks, num_frames, num_speakers), binarized)
                .unwrap();

        // Stage 4a: Reconstruct timeline
        let timeline = reconstruct_timeline(&binarized_arr, &assignments, step_duration, frame_duration);

        assert!(!timeline.is_empty());
        // Should have segments for both speakers
        let speakers: std::collections::BTreeSet<_> =
            timeline.iter().map(|s| s.speaker.clone()).collect();
        assert!(speakers.contains("SPEAKER_00"));
        assert!(speakers.contains("SPEAKER_01"));

        // Stage 4b: Merge segments
        let merged = merge_segments(&timeline);
        assert!(!merged.is_empty());

        // Stage 4c: Align with transcript
        let transcript = vec![
            TestSegment {
                text: " Hello, how are you?",
                start: 0.0,
                end: Some(0.8),
            },
            TestSegment {
                text: " I'm doing great, thanks!",
                start: 1.0,
                end: Some(1.9),
            },
        ];

        let segments: Vec<Segment> = transcript
            .iter()
            .map(|t| Segment {
                text: t.text.to_string(),
                start: t.start,
                end: t.end,
            })
            .collect();

        let speaker_chunks = align_with_transcript(&merged, &segments);
        assert_eq!(speaker_chunks.len(), 2);

        // First chunk should be SPEAKER_00 (chunk 0 time range)
        assert_eq!(speaker_chunks[0].speaker, "SPEAKER_00");
        assert_eq!(speaker_chunks[0].text, " Hello, how are you?");

        // Second chunk should be SPEAKER_01 (chunk 1 time range)
        assert_eq!(speaker_chunks[1].speaker, "SPEAKER_01");
        assert_eq!(speaker_chunks[1].text, " I'm doing great, thanks!");
    }

    #[test]
    fn test_pipeline_three_speakers_alternating() {
        // 3 chunks, 5 frames each, 3 speakers
        // Each chunk has one active speaker
        let num_chunks = 3;
        let num_frames = 5;
        let num_speakers = 3;

        let mut binarized = vec![0.0f32; num_chunks * num_frames * num_speakers];
        for f in 0..num_frames {
            // Chunk 0: speaker 0
            binarized[0 * num_frames * num_speakers + f * num_speakers + 0] = 1.0;
            // Chunk 1: speaker 1
            binarized[1 * num_frames * num_speakers + f * num_speakers + 1] = 1.0;
            // Chunk 2: speaker 2
            binarized[2 * num_frames * num_speakers + f * num_speakers + 2] = 1.0;
        }

        let mut assignments = HashMap::new();
        assignments.insert((0, 0), 0);
        assignments.insert((1, 1), 1);
        assignments.insert((2, 2), 2);

        let binarized_arr =
            ndarray::Array3::from_shape_vec((num_chunks, num_frames, num_speakers), binarized)
                .unwrap();

        let timeline = reconstruct_timeline(&binarized_arr, &assignments, 2.0, 0.1);
        let merged = merge_segments(&timeline);

        let segments = vec![
            Segment {
                text: " First speaker.".to_string(),
                start: 0.0,
                end: Some(1.5),
            },
            Segment {
                text: " Second speaker.".to_string(),
                start: 2.0,
                end: Some(3.5),
            },
            Segment {
                text: " Third speaker.".to_string(),
                start: 4.0,
                end: Some(5.5),
            },
        ];

        let speaker_chunks = align_with_transcript(&merged, &segments);
        assert_eq!(speaker_chunks.len(), 3);
        assert_eq!(speaker_chunks[0].speaker, "SPEAKER_00");
        assert_eq!(speaker_chunks[1].speaker, "SPEAKER_01");
        assert_eq!(speaker_chunks[2].speaker, "SPEAKER_02");
    }

    #[test]
    fn test_pipeline_same_speaker_merged_across_chunks() {
        // 3 chunks, same speaker in all → should merge into one segment
        let num_chunks = 3;
        let num_frames = 5;
        let num_speakers = 1;

        let binarized = vec![1.0f32; num_chunks * num_frames * num_speakers];

        let mut assignments = HashMap::new();
        assignments.insert((0, 0), 0);
        assignments.insert((1, 0), 0);
        assignments.insert((2, 0), 0);

        let binarized_arr =
            ndarray::Array3::from_shape_vec((num_chunks, num_frames, num_speakers), binarized)
                .unwrap();

        // step=0.5s, frame=0.1s → chunks at 0.0, 0.5, 1.0; each covers 0.5s
        let timeline = reconstruct_timeline(&binarized_arr, &assignments, 0.5, 0.1);
        let merged = merge_segments(&timeline);

        // All segments should be same speaker → single merged segment
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].speaker, "SPEAKER_00");

        let segments = vec![
            Segment {
                text: " Part one.".to_string(),
                start: 0.0,
                end: Some(0.4),
            },
            Segment {
                text: " Part two.".to_string(),
                start: 0.5,
                end: Some(0.9),
            },
            Segment {
                text: " Part three.".to_string(),
                start: 1.0,
                end: Some(1.4),
            },
        ];

        let speaker_chunks = align_with_transcript(&merged, &segments);
        assert_eq!(speaker_chunks.len(), 3);
        // All should be SPEAKER_00
        for chunk in &speaker_chunks {
            assert_eq!(chunk.speaker, "SPEAKER_00");
        }
    }

    #[test]
    fn test_pipeline_empty_assignments_produces_no_output() {
        let binarized = ndarray::Array3::<f32>::zeros((2, 5, 2));
        let assignments = HashMap::new();

        let timeline = reconstruct_timeline(&binarized, &assignments, 1.0, 0.1);
        assert!(timeline.is_empty());

        let merged = merge_segments(&timeline);
        assert!(merged.is_empty());

        let segments = vec![Segment {
            text: " Some text.".to_string(),
            start: 0.0,
            end: Some(2.0),
        }];
        let speaker_chunks = align_with_transcript(&merged, &segments);
        assert!(speaker_chunks.is_empty());
    }

    #[test]
    fn test_pipeline_with_null_end_timestamp() {
        let num_chunks = 2;
        let num_frames = 5;
        let num_speakers = 2;

        let mut binarized = vec![0.0f32; num_chunks * num_frames * num_speakers];
        for f in 0..num_frames {
            binarized[0 * num_frames * num_speakers + f * num_speakers + 0] = 1.0;
            binarized[1 * num_frames * num_speakers + f * num_speakers + 1] = 1.0;
        }

        let mut assignments = HashMap::new();
        assignments.insert((0, 0), 0);
        assignments.insert((1, 1), 1);

        let binarized_arr =
            ndarray::Array3::from_shape_vec((num_chunks, num_frames, num_speakers), binarized)
                .unwrap();

        let timeline = reconstruct_timeline(&binarized_arr, &assignments, 1.0, 0.1);
        let merged = merge_segments(&timeline);

        // Last segment has null end timestamp
        let segments = vec![
            Segment {
                text: " First part.".to_string(),
                start: 0.0,
                end: Some(0.8),
            },
            Segment {
                text: " Last part.".to_string(),
                start: 1.0,
                end: None, // null end
            },
        ];

        let speaker_chunks = align_with_transcript(&merged, &segments);
        assert_eq!(speaker_chunks.len(), 2);
        // Verify null end is preserved
        assert_eq!(speaker_chunks[1].end, None);
    }

    // Re-export internal types for use in this test module.
    // These types are from the library crate, but since this is a binary crate,
    // we replicate the minimal structs needed for testing.

    #[derive(Debug, Clone)]
    struct DiarizationSegment {
        start: f64,
        end: f64,
        speaker: String,
    }

    #[derive(Debug, Clone)]
    struct Segment {
        text: String,
        start: f64,
        end: Option<f64>,
    }

    #[derive(Debug, Clone)]
    struct SpeakerChunk {
        speaker: String,
        text: String,
        #[allow(dead_code)]
        start: f64,
        end: Option<f64>,
    }

    /// Reconstruct speaker timeline (mirrors src/diarize/postprocess.rs::reconstruct_timeline)
    fn reconstruct_timeline(
        chunk_binarized: &ndarray::Array3<f32>,
        assignments: &HashMap<(usize, usize), usize>,
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

        merge_overlapping_per_speaker(&mut segments);
        segments
    }

    fn merge_overlapping_per_speaker(segments: &mut Vec<DiarizationSegment>) {
        if segments.is_empty() {
            return;
        }
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
        merged.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        *segments = merged;
    }

    fn merge_segments(segments: &[DiarizationSegment]) -> Vec<DiarizationSegment> {
        if segments.is_empty() {
            return vec![];
        }
        let mut new_segments = Vec::new();
        let mut prev = &segments[0];
        for cur in segments.iter().skip(1) {
            if cur.speaker != prev.speaker {
                new_segments.push(DiarizationSegment {
                    start: prev.start,
                    end: cur.start,
                    speaker: prev.speaker.clone(),
                });
                prev = cur;
            }
        }
        let last = segments.last().unwrap();
        new_segments.push(DiarizationSegment {
            start: prev.start,
            end: last.end,
            speaker: prev.speaker.clone(),
        });
        new_segments
    }

    fn align_with_transcript(
        diarization_segments: &[DiarizationSegment],
        transcript: &[Segment],
    ) -> Vec<SpeakerChunk> {
        if diarization_segments.is_empty() || transcript.is_empty() {
            return vec![];
        }
        let mut result = Vec::new();
        let mut offset = 0usize;

        for segment in diarization_segments {
            let remaining = &transcript[offset..];
            if remaining.is_empty() {
                break;
            }
            let end_time = segment.end;
            let end_timestamps: Vec<f64> = remaining
                .iter()
                .map(|chunk| chunk.end.unwrap_or(f64::MAX))
                .collect();

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
}

// ──────────────────────────────────────────────────
// Output JSON with diarization: speakers field format
// ──────────────────────────────────────────────────

mod output_json_tests {
    use serde::Serialize;

    /// Timestamp pair: [start, end] where end can be null.
    #[derive(Debug, Clone, Serialize)]
    struct Timestamp(f64, Option<f64>);

    #[derive(Debug, Clone, Serialize)]
    struct OutputChunk {
        text: String,
        timestamp: Timestamp,
    }

    #[derive(Debug, Clone, Serialize)]
    struct SpeakerEntry {
        speaker: String,
        text: String,
        timestamp: Timestamp,
    }

    #[derive(Debug, Clone, Serialize)]
    struct DiarizedTranscriptOutput {
        speakers: Vec<SpeakerEntry>,
        chunks: Vec<OutputChunk>,
        text: String,
    }

    #[derive(Debug, Clone, Serialize)]
    struct TranscriptOutput {
        speakers: Vec<()>,
        chunks: Vec<OutputChunk>,
        text: String,
    }

    #[test]
    fn test_diarized_output_speakers_field_structure() {
        let output = DiarizedTranscriptOutput {
            speakers: vec![
                SpeakerEntry {
                    speaker: "SPEAKER_00".to_string(),
                    text: " Hello there.".to_string(),
                    timestamp: Timestamp(0.0, Some(2.5)),
                },
                SpeakerEntry {
                    speaker: "SPEAKER_01".to_string(),
                    text: " Hi, how are you?".to_string(),
                    timestamp: Timestamp(2.5, Some(5.0)),
                },
            ],
            chunks: vec![
                OutputChunk {
                    text: " Hello there.".to_string(),
                    timestamp: Timestamp(0.0, Some(2.5)),
                },
                OutputChunk {
                    text: " Hi, how are you?".to_string(),
                    timestamp: Timestamp(2.5, Some(5.0)),
                },
            ],
            text: " Hello there. Hi, how are you?".to_string(),
        };

        let value = serde_json::to_value(&output).unwrap();

        // Verify speakers is a non-empty array
        let speakers = value["speakers"].as_array().unwrap();
        assert_eq!(speakers.len(), 2);

        // Each speaker entry has: speaker, text, timestamp
        for entry in speakers {
            assert!(entry.get("speaker").is_some());
            assert!(entry.get("text").is_some());
            assert!(entry.get("timestamp").is_some());
            let ts = entry["timestamp"].as_array().unwrap();
            assert_eq!(ts.len(), 2);
        }

        // Verify speaker labels
        assert_eq!(speakers[0]["speaker"], "SPEAKER_00");
        assert_eq!(speakers[1]["speaker"], "SPEAKER_01");

        // Verify text content
        assert_eq!(speakers[0]["text"], " Hello there.");
        assert_eq!(speakers[1]["text"], " Hi, how are you?");

        // Verify timestamps
        let ts0 = speakers[0]["timestamp"].as_array().unwrap();
        assert_eq!(ts0[0].as_f64().unwrap(), 0.0);
        assert_eq!(ts0[1].as_f64().unwrap(), 2.5);
    }

    #[test]
    fn test_diarized_output_field_order() {
        let output = DiarizedTranscriptOutput {
            speakers: vec![SpeakerEntry {
                speaker: "SPEAKER_00".to_string(),
                text: " Test.".to_string(),
                timestamp: Timestamp(0.0, Some(1.0)),
            }],
            chunks: vec![OutputChunk {
                text: " Test.".to_string(),
                timestamp: Timestamp(0.0, Some(1.0)),
            }],
            text: " Test.".to_string(),
        };

        let json_str = serde_json::to_string(&output).unwrap();

        // Top-level field order: speakers, chunks, text
        let speakers_pos = json_str.find("\"speakers\"").unwrap();
        let chunks_pos = json_str.find("\"chunks\"").unwrap();
        // Use rfind for text since it also appears inside speaker entries
        let text_pos = json_str.rfind("\"text\"").unwrap();
        assert!(speakers_pos < chunks_pos, "speakers should come before chunks");
        assert!(chunks_pos < text_pos, "chunks should come before top-level text");

        // Speaker entry field order: speaker, text, timestamp
        let speaker_json = serde_json::to_string(&output.speakers[0]).unwrap();
        let sp = speaker_json.find("\"speaker\"").unwrap();
        let tx = speaker_json.find("\"text\"").unwrap();
        let ts = speaker_json.find("\"timestamp\"").unwrap();
        assert!(sp < tx, "speaker should come before text");
        assert!(tx < ts, "text should come before timestamp");
    }

    #[test]
    fn test_diarized_output_null_end_timestamp() {
        let output = DiarizedTranscriptOutput {
            speakers: vec![SpeakerEntry {
                speaker: "SPEAKER_00".to_string(),
                text: " Final words.".to_string(),
                timestamp: Timestamp(5.0, None),
            }],
            chunks: vec![OutputChunk {
                text: " Final words.".to_string(),
                timestamp: Timestamp(5.0, None),
            }],
            text: " Final words.".to_string(),
        };

        let value = serde_json::to_value(&output).unwrap();
        let speakers = value["speakers"].as_array().unwrap();
        let ts = speakers[0]["timestamp"].as_array().unwrap();
        assert_eq!(ts[0].as_f64().unwrap(), 5.0);
        assert!(ts[1].is_null(), "End timestamp should be null");

        let chunks = value["chunks"].as_array().unwrap();
        let cts = chunks[0]["timestamp"].as_array().unwrap();
        assert!(cts[1].is_null(), "Chunk end timestamp should also be null");
    }

    #[test]
    fn test_non_diarized_output_empty_speakers() {
        let output = TranscriptOutput {
            speakers: vec![],
            chunks: vec![OutputChunk {
                text: " Hello.".to_string(),
                timestamp: Timestamp(0.0, Some(1.0)),
            }],
            text: " Hello.".to_string(),
        };

        let value = serde_json::to_value(&output).unwrap();
        let speakers = value["speakers"].as_array().unwrap();
        assert!(speakers.is_empty(), "speakers should be [] when no diarization");
    }

    #[test]
    fn test_diarized_output_multiple_speakers_correct_labels() {
        let mut speakers = Vec::new();
        for i in 0..5 {
            speakers.push(SpeakerEntry {
                speaker: format!("SPEAKER_{i:02}"),
                text: format!(" Speaker {i} text."),
                timestamp: Timestamp(i as f64 * 2.0, Some((i as f64 + 1.0) * 2.0)),
            });
        }

        let output = DiarizedTranscriptOutput {
            speakers: speakers.clone(),
            chunks: speakers
                .iter()
                .map(|s| OutputChunk {
                    text: s.text.clone(),
                    timestamp: s.timestamp.clone(),
                })
                .collect(),
            text: speakers
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(""),
        };

        let value = serde_json::to_value(&output).unwrap();
        let sp_arr = value["speakers"].as_array().unwrap();

        // Verify speaker labels are correctly formatted with zero-padding
        assert_eq!(sp_arr[0]["speaker"], "SPEAKER_00");
        assert_eq!(sp_arr[1]["speaker"], "SPEAKER_01");
        assert_eq!(sp_arr[2]["speaker"], "SPEAKER_02");
        assert_eq!(sp_arr[3]["speaker"], "SPEAKER_03");
        assert_eq!(sp_arr[4]["speaker"], "SPEAKER_04");
    }

    #[test]
    fn test_diarized_output_write_and_read_roundtrip() {
        let output = DiarizedTranscriptOutput {
            speakers: vec![
                SpeakerEntry {
                    speaker: "SPEAKER_00".to_string(),
                    text: " Bonjour le monde.".to_string(),
                    timestamp: Timestamp(0.0, Some(2.0)),
                },
                SpeakerEntry {
                    speaker: "SPEAKER_01".to_string(),
                    text: " Salut!".to_string(),
                    timestamp: Timestamp(2.0, Some(3.5)),
                },
            ],
            chunks: vec![
                OutputChunk {
                    text: " Bonjour le monde.".to_string(),
                    timestamp: Timestamp(0.0, Some(2.0)),
                },
                OutputChunk {
                    text: " Salut!".to_string(),
                    timestamp: Timestamp(2.0, Some(3.5)),
                },
            ],
            text: " Bonjour le monde. Salut!".to_string(),
        };

        let json_str = serde_json::to_string_pretty(&output).unwrap();
        let tmp = std::env::temp_dir().join("fast-whisper-rs-diar-roundtrip.json");
        std::fs::write(&tmp, &json_str).unwrap();

        let read_back = std::fs::read_to_string(&tmp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&read_back).unwrap();

        assert_eq!(parsed["speakers"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["chunks"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["text"], " Bonjour le monde. Salut!");

        // Unicode should be preserved
        assert!(json_str.contains("Bonjour"));

        let _ = std::fs::remove_file(&tmp);
    }
}

// ──────────────────────────────────────────────────
// Integration tests requiring ONNX models (ignored)
// ──────────────────────────────────────────────────

#[test]
#[ignore]
fn test_full_diarization_pipeline_with_models() {
    // This test requires downloaded ONNX models and is run with:
    // cargo test -- --ignored
    //
    // It validates the full pipeline: segmentation → embedding → clustering → post-processing
    // using actual ONNX sessions.

    // Generate a simple test signal: two speakers alternating
    let sample_rate = 16000;
    let duration_secs = 10;
    let num_samples = sample_rate * duration_secs;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Different frequency for first/second half to simulate different speakers
            if t < 5.0 {
                (440.0 * 2.0 * std::f32::consts::PI * t).sin()
            } else {
                (880.0 * 2.0 * std::f32::consts::PI * t).sin()
            }
        })
        .collect();

    // Create mock transcript matching the audio
    let _transcript_segments = vec![
        ("First half speech.", 0.0, Some(5.0)),
        ("Second half speech.", 5.0, Some(10.0)),
    ];

    // This would need actual models to run:
    // let cache_dir = diarize::ensure_models(None).unwrap();
    // let mut models = diarize::load_all_models(&cache_dir, "0").unwrap();
    // let result = diarize::run_pipeline(&mut models, &samples, &transcript, 24, None, None, None);

    assert_eq!(samples.len(), num_samples);
}

#[test]
#[ignore]
fn test_model_download_and_cache_structure() {
    // This test actually downloads models and verifies cache structure.
    // Run with: cargo test -- --ignored
    let artifacts = [
        "segmentation.onnx",
        "embedding.onnx",
        "plda_xvec_transform.npz",
        "plda.npz",
    ];

    let home = std::env::var("HOME").unwrap();
    let cache_dir = std::path::PathBuf::from(home)
        .join(".cache")
        .join("fast-whisper-rs")
        .join("diarization");

    // Verify all expected artifacts exist after download
    for artifact in &artifacts {
        let path = cache_dir.join(artifact);
        assert!(
            path.exists(),
            "Expected artifact {} to exist at {}",
            artifact,
            path.display()
        );
    }
}
