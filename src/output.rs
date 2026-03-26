use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::inference::Segment;

/// A timestamp pair: [start, end] where end can be null.
#[derive(Debug, Clone, Serialize)]
struct Timestamp(f64, Option<f64>);

/// A single chunk in the output JSON.
#[derive(Debug, Clone, Serialize)]
struct OutputChunk {
    text: String,
    timestamp: Timestamp,
}

/// The top-level output JSON structure.
/// Field order matches Python: speakers, chunks, text.
#[derive(Debug, Clone, Serialize)]
struct TranscriptOutput {
    speakers: Vec<()>,
    chunks: Vec<OutputChunk>,
    text: String,
}

/// Build the output JSON structure from decoded segments.
pub fn build_output(segments: &[Segment]) -> serde_json::Value {
    let chunks: Vec<OutputChunk> = segments
        .iter()
        .map(|seg| OutputChunk {
            text: seg.text.clone(),
            timestamp: Timestamp(seg.start, seg.end),
        })
        .collect();

    let full_text: String = segments
        .iter()
        .map(|seg| seg.text.as_str())
        .collect::<Vec<&str>>()
        .join("");

    let output = TranscriptOutput {
        speakers: vec![],
        chunks,
        text: full_text,
    };

    serde_json::to_value(output).expect("Failed to serialize output")
}

/// Write the transcript JSON to a file at the given path.
pub fn write_output(segments: &[Segment], transcript_path: &str) -> Result<()> {
    let value = build_output(segments);
    let json_string = serde_json::to_string_pretty(&value)?;

    let path = Path::new(transcript_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;
    file.write_all(json_string.as_bytes())?;
    file.write_all(b"\n")?;

    Ok(())
}

/// Print the success message to stdout (matching Python exactly).
pub fn print_success_message(transcript_path: &str, has_diarization: bool) {
    if has_diarization {
        println!("Voila!✨ Your file has been transcribed & speaker segmented go check it out over here 👉 {transcript_path}");
    } else {
        println!("Voila!✨ Your file has been transcribed go check it out over here 👉 {transcript_path}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_segments() -> Vec<Segment> {
        vec![
            Segment {
                text: " Hello world.".to_string(),
                start: 0.0,
                end: Some(2.5),
            },
            Segment {
                text: " How are you?".to_string(),
                start: 2.5,
                end: Some(5.0),
            },
        ]
    }

    #[test]
    fn test_output_json_structure() {
        let segments = sample_segments();
        let value = build_output(&segments);

        // Verify top-level keys exist
        assert!(value.get("speakers").is_some());
        assert!(value.get("chunks").is_some());
        assert!(value.get("text").is_some());

        // speakers is empty array
        assert_eq!(value["speakers"], serde_json::json!([]));
    }

    #[test]
    fn test_output_json_field_order() {
        let segments = sample_segments();
        let value = build_output(&segments);
        let json_str = serde_json::to_string(&value).unwrap();

        // Verify field order: speakers before chunks before text
        let speakers_pos = json_str.find("\"speakers\"").unwrap();
        let chunks_pos = json_str.find("\"chunks\"").unwrap();
        let text_pos = json_str.find("\"text\"").unwrap();
        assert!(speakers_pos < chunks_pos);
        assert!(chunks_pos < text_pos);
    }

    #[test]
    fn test_output_chunks_structure() {
        let segments = sample_segments();
        let value = build_output(&segments);
        let chunks = value["chunks"].as_array().unwrap();

        assert_eq!(chunks.len(), 2);

        // First chunk
        assert_eq!(chunks[0]["text"], " Hello world.");
        let ts0 = chunks[0]["timestamp"].as_array().unwrap();
        assert_eq!(ts0[0].as_f64().unwrap(), 0.0);
        assert_eq!(ts0[1].as_f64().unwrap(), 2.5);

        // Second chunk
        assert_eq!(chunks[1]["text"], " How are you?");
        let ts1 = chunks[1]["timestamp"].as_array().unwrap();
        assert_eq!(ts1[0].as_f64().unwrap(), 2.5);
        assert_eq!(ts1[1].as_f64().unwrap(), 5.0);
    }

    #[test]
    fn test_output_null_end_timestamp() {
        let segments = vec![
            Segment {
                text: " Hello.".to_string(),
                start: 0.0,
                end: Some(2.0),
            },
            Segment {
                text: " Final chunk.".to_string(),
                start: 2.0,
                end: None,
            },
        ];
        let value = build_output(&segments);
        let chunks = value["chunks"].as_array().unwrap();

        // Last chunk has null end timestamp
        let ts = chunks[1]["timestamp"].as_array().unwrap();
        assert_eq!(ts[0].as_f64().unwrap(), 2.0);
        assert!(ts[1].is_null());
    }

    #[test]
    fn test_output_full_text_concatenation() {
        let segments = sample_segments();
        let value = build_output(&segments);
        assert_eq!(value["text"], " Hello world. How are you?");
    }

    #[test]
    fn test_output_unicode_preserved() {
        let segments = vec![Segment {
            text: " こんにちは世界 café résumé".to_string(),
            start: 0.0,
            end: Some(3.0),
        }];
        let value = build_output(&segments);
        let json_str = serde_json::to_string(&value).unwrap();

        // Unicode should be preserved, not escaped (serde_json default)
        assert!(json_str.contains("こんにちは世界"));
        assert!(json_str.contains("café"));
        assert!(json_str.contains("résumé"));
    }

    #[test]
    fn test_output_empty_segments() {
        let segments: Vec<Segment> = vec![];
        let value = build_output(&segments);

        assert_eq!(value["speakers"], serde_json::json!([]));
        assert_eq!(value["chunks"], serde_json::json!([]));
        assert_eq!(value["text"], "");
    }

    #[test]
    fn test_output_single_segment() {
        let segments = vec![Segment {
            text: " Just one segment.".to_string(),
            start: 0.0,
            end: None,
        }];
        let value = build_output(&segments);

        let chunks = value["chunks"].as_array().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(value["text"], " Just one segment.");
    }

    #[test]
    fn test_write_and_read_output() {
        let segments = sample_segments();
        let dir = std::env::temp_dir().join("fast-whisper-rs-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_output.json");
        let path_str = path.to_str().unwrap();

        write_output(&segments, path_str).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();

        assert_eq!(parsed["speakers"], serde_json::json!([]));
        assert_eq!(parsed["chunks"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["text"], " Hello world. How are you?");

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_timestamp_serialization_format() {
        // Verify timestamps serialize as [f64, f64] or [f64, null]
        let ts_with_end = Timestamp(1.5, Some(3.0));
        let json = serde_json::to_string(&ts_with_end).unwrap();
        assert_eq!(json, "[1.5,3.0]");

        let ts_null_end = Timestamp(1.5, None);
        let json = serde_json::to_string(&ts_null_end).unwrap();
        assert_eq!(json, "[1.5,null]");
    }

    #[test]
    fn test_success_message_no_diarization() {
        // Just verify the function doesn't panic - actual output goes to stdout
        print_success_message("output.json", false);
    }

    #[test]
    fn test_success_message_with_diarization() {
        print_success_message("output.json", true);
    }
}
