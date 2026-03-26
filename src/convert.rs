use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::cli::OutputFormat;

/// A chunk from the transcript JSON.
#[derive(Debug, Deserialize)]
struct Chunk {
    text: String,
    timestamp: (f64, Option<f64>),
}

/// The transcript JSON structure (only fields we need for conversion).
#[derive(Debug, Deserialize)]
struct Transcript {
    chunks: Vec<Chunk>,
}

/// Format seconds as HH:MM:SS,mmm (SRT format — comma separator).
fn format_srt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Format seconds as HH:MM:SS.mmm (VTT format — dot separator).
fn format_vtt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

/// Convert transcript chunks to SRT format.
fn to_srt(chunks: &[Chunk]) -> String {
    let mut out = String::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let start = format_srt_timestamp(chunk.timestamp.0);
        let end = format_srt_timestamp(chunk.timestamp.1.unwrap_or(chunk.timestamp.0));
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{start} --> {end}\n"));
        out.push_str(chunk.text.trim());
        out.push('\n');
        out.push('\n');
    }
    out
}

/// Convert transcript chunks to VTT format.
fn to_vtt(chunks: &[Chunk]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (i, chunk) in chunks.iter().enumerate() {
        let start = format_vtt_timestamp(chunk.timestamp.0);
        let end = format_vtt_timestamp(chunk.timestamp.1.unwrap_or(chunk.timestamp.0));
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{start} --> {end}\n"));
        out.push_str(chunk.text.trim());
        out.push('\n');
        out.push('\n');
    }
    out
}

/// Convert transcript chunks to plain text format (one chunk per line).
fn to_txt(chunks: &[Chunk]) -> String {
    let mut out = String::new();
    for chunk in chunks {
        out.push_str(chunk.text.trim());
        out.push('\n');
    }
    out
}

/// Print each entry to stdout (verbose mode).
fn print_verbose(chunks: &[Chunk], format: &OutputFormat) {
    for (i, chunk) in chunks.iter().enumerate() {
        match format {
            OutputFormat::Srt => {
                let start = format_srt_timestamp(chunk.timestamp.0);
                let end =
                    format_srt_timestamp(chunk.timestamp.1.unwrap_or(chunk.timestamp.0));
                println!("{}", i + 1);
                println!("{start} --> {end}");
                println!("{}", chunk.text.trim());
                println!();
            }
            OutputFormat::Vtt => {
                let start = format_vtt_timestamp(chunk.timestamp.0);
                let end =
                    format_vtt_timestamp(chunk.timestamp.1.unwrap_or(chunk.timestamp.0));
                println!("{}", i + 1);
                println!("{start} --> {end}");
                println!("{}", chunk.text.trim());
                println!();
            }
            OutputFormat::Txt => {
                println!("{}", chunk.text.trim());
            }
        }
    }
}

/// Run the convert subcommand: read JSON, convert to the target format, write output.
pub fn run_convert(
    input_file: &str,
    output_format: &OutputFormat,
    output_dir: &str,
    verbose: bool,
) -> Result<()> {
    // Read and parse the input JSON
    let json_str = fs::read_to_string(input_file)
        .with_context(|| format!("Failed to read input file: {input_file}"))?;
    let transcript: Transcript =
        serde_json::from_str(&json_str).context("Failed to parse transcript JSON")?;

    // Convert to the target format
    let (content, extension) = match output_format {
        OutputFormat::Srt => (to_srt(&transcript.chunks), "srt"),
        OutputFormat::Vtt => (to_vtt(&transcript.chunks), "vtt"),
        OutputFormat::Txt => (to_txt(&transcript.chunks), "txt"),
    };

    // Verbose: print each entry
    if verbose {
        print_verbose(&transcript.chunks, output_format);
    }

    // Write output file: output.<format> in the output directory
    let output_path = Path::new(output_dir).join(format!("output.{extension}"));
    fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {output_dir}"))?;
    fs::write(&output_path, &content)
        .with_context(|| format!("Failed to write output file: {}", output_path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                text: " Hello world.".to_string(),
                timestamp: (0.0, Some(2.5)),
            },
            Chunk {
                text: " How are you?".to_string(),
                timestamp: (2.5, Some(5.0)),
            },
            Chunk {
                text: " I am fine.".to_string(),
                timestamp: (5.0, Some(61.123)),
            },
        ]
    }

    #[test]
    fn test_srt_timestamp_format() {
        assert_eq!(format_srt_timestamp(0.0), "00:00:00,000");
        assert_eq!(format_srt_timestamp(2.5), "00:00:02,500");
        assert_eq!(format_srt_timestamp(61.123), "00:01:01,123");
        assert_eq!(format_srt_timestamp(3661.999), "01:01:01,999");
        assert_eq!(format_srt_timestamp(0.001), "00:00:00,001");
    }

    #[test]
    fn test_vtt_timestamp_format() {
        assert_eq!(format_vtt_timestamp(0.0), "00:00:00.000");
        assert_eq!(format_vtt_timestamp(2.5), "00:00:02.500");
        assert_eq!(format_vtt_timestamp(61.123), "00:01:01.123");
        assert_eq!(format_vtt_timestamp(3661.999), "01:01:01.999");
    }

    #[test]
    fn test_srt_conversion() {
        let chunks = sample_chunks();
        let srt = to_srt(&chunks);

        assert!(srt.contains("1\n00:00:00,000 --> 00:00:02,500\nHello world.\n"));
        assert!(srt.contains("2\n00:00:02,500 --> 00:00:05,000\nHow are you?\n"));
        assert!(srt.contains("3\n00:00:05,000 --> 00:01:01,123\nI am fine.\n"));
    }

    #[test]
    fn test_vtt_conversion() {
        let chunks = sample_chunks();
        let vtt = to_vtt(&chunks);

        assert!(vtt.starts_with("WEBVTT\n\n"));
        assert!(vtt.contains("1\n00:00:00.000 --> 00:00:02.500\nHello world.\n"));
        assert!(vtt.contains("2\n00:00:02.500 --> 00:00:05.000\nHow are you?\n"));
        assert!(vtt.contains("3\n00:00:05.000 --> 00:01:01.123\nI am fine.\n"));
    }

    #[test]
    fn test_txt_conversion() {
        let chunks = sample_chunks();
        let txt = to_txt(&chunks);

        let lines: Vec<&str> = txt.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "Hello world.");
        assert_eq!(lines[1], "How are you?");
        assert_eq!(lines[2], "I am fine.");
    }

    #[test]
    fn test_srt_index_starts_at_one() {
        let chunks = vec![Chunk {
            text: " Test.".to_string(),
            timestamp: (0.0, Some(1.0)),
        }];
        let srt = to_srt(&chunks);
        assert!(srt.starts_with("1\n"));
    }

    #[test]
    fn test_vtt_header() {
        let chunks = vec![Chunk {
            text: " Test.".to_string(),
            timestamp: (0.0, Some(1.0)),
        }];
        let vtt = to_vtt(&chunks);
        assert!(vtt.starts_with("WEBVTT\n\n1\n"));
    }

    #[test]
    fn test_srt_comma_separator() {
        // SRT uses comma in timestamps
        let ts = format_srt_timestamp(1.5);
        assert!(ts.contains(','));
        assert!(!ts.contains('.'));
    }

    #[test]
    fn test_vtt_dot_separator() {
        // VTT uses dot in timestamps
        let ts = format_vtt_timestamp(1.5);
        assert!(ts.contains('.'));
        assert!(!ts.contains(','));
    }

    #[test]
    fn test_null_end_timestamp_fallback() {
        let chunks = vec![Chunk {
            text: " Final.".to_string(),
            timestamp: (10.0, None),
        }];
        let srt = to_srt(&chunks);
        // When end is null, use start as fallback
        assert!(srt.contains("00:00:10,000 --> 00:00:10,000"));

        let vtt = to_vtt(&chunks);
        assert!(vtt.contains("00:00:10.000 --> 00:00:10.000"));
    }

    #[test]
    fn test_empty_chunks() {
        let chunks: Vec<Chunk> = vec![];
        assert_eq!(to_srt(&chunks), "");
        assert_eq!(to_vtt(&chunks), "WEBVTT\n\n");
        assert_eq!(to_txt(&chunks), "");
    }

    #[test]
    fn test_text_trimming() {
        // Whisper outputs often have leading spaces
        let chunks = vec![Chunk {
            text: "  padded text  ".to_string(),
            timestamp: (0.0, Some(1.0)),
        }];
        let txt = to_txt(&chunks);
        assert_eq!(txt, "padded text\n");
    }

    #[test]
    fn test_run_convert_srt() {
        let dir = std::env::temp_dir().join("fast-whisper-rs-convert-test");
        let _ = std::fs::create_dir_all(&dir);

        // Create sample JSON
        let json = r#"{"speakers":[],"chunks":[{"text":" Hello.","timestamp":[0.0,2.0]},{"text":" World.","timestamp":[2.0,4.0]}],"text":" Hello. World."}"#;
        let input = dir.join("input.json");
        std::fs::write(&input, json).unwrap();

        let output_dir = dir.join("out_srt");
        run_convert(
            input.to_str().unwrap(),
            &OutputFormat::Srt,
            output_dir.to_str().unwrap(),
            false,
        )
        .unwrap();

        let output = std::fs::read_to_string(output_dir.join("output.srt")).unwrap();
        assert!(output.contains("1\n00:00:00,000 --> 00:00:02,000\nHello.\n"));
        assert!(output.contains("2\n00:00:02,000 --> 00:00:04,000\nWorld.\n"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_convert_vtt() {
        let dir = std::env::temp_dir().join("fast-whisper-rs-convert-vtt-test");
        let _ = std::fs::create_dir_all(&dir);

        let json = r#"{"speakers":[],"chunks":[{"text":" Hello.","timestamp":[0.0,2.0]}],"text":" Hello."}"#;
        let input = dir.join("input.json");
        std::fs::write(&input, json).unwrap();

        let output_dir = dir.join("out_vtt");
        run_convert(
            input.to_str().unwrap(),
            &OutputFormat::Vtt,
            output_dir.to_str().unwrap(),
            false,
        )
        .unwrap();

        let output = std::fs::read_to_string(output_dir.join("output.vtt")).unwrap();
        assert!(output.starts_with("WEBVTT\n\n"));
        assert!(output.contains("00:00:00.000 --> 00:00:02.000"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_run_convert_txt() {
        let dir = std::env::temp_dir().join("fast-whisper-rs-convert-txt-test");
        let _ = std::fs::create_dir_all(&dir);

        let json = r#"{"speakers":[],"chunks":[{"text":" Hello.","timestamp":[0.0,2.0]},{"text":" World.","timestamp":[2.0,4.0]}],"text":" Hello. World."}"#;
        let input = dir.join("input.json");
        std::fs::write(&input, json).unwrap();

        let output_dir = dir.join("out_txt");
        run_convert(
            input.to_str().unwrap(),
            &OutputFormat::Txt,
            output_dir.to_str().unwrap(),
            false,
        )
        .unwrap();

        let output = std::fs::read_to_string(output_dir.join("output.txt")).unwrap();
        assert_eq!(output, "Hello.\nWorld.\n");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
