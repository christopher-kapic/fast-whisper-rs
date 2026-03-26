use assert_cmd::Command;
use predicates::prelude::*;

const SAMPLE_JSON: &str = r#"{
  "speakers": [],
  "chunks": [
    {
      "text": " Hello world.",
      "timestamp": [0.0, 2.5]
    },
    {
      "text": " How are you?",
      "timestamp": [2.5, 5.0]
    },
    {
      "text": " I am fine.",
      "timestamp": [5.0, null]
    }
  ],
  "text": " Hello world. How are you? I am fine."
}"#;

fn cmd() -> Command {
    Command::cargo_bin("fast-whisper-rs").unwrap()
}

fn setup_test_dir(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("fast-whisper-rs-convert-integ-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.json");
    std::fs::write(&input, SAMPLE_JSON).unwrap();
    (dir, input)
}

// ──────────────────────────────────────────────────
// SRT conversion
// ──────────────────────────────────────────────────

#[test]
fn test_convert_to_srt() {
    let (dir, input) = setup_test_dir("srt");
    let output_dir = dir.join("out");

    cmd()
        .args([
            "convert",
            input.to_str().unwrap(),
            "-f",
            "srt",
            "-o",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let output = std::fs::read_to_string(output_dir.join("output.srt")).unwrap();

    // Verify exact SRT format
    let expected = "\
1
00:00:00,000 --> 00:00:02,500
Hello world.

2
00:00:02,500 --> 00:00:05,000
How are you?

3
00:00:05,000 --> 00:00:05,000
I am fine.

";
    assert_eq!(output, expected);

    let _ = std::fs::remove_dir_all(&dir);
}

// ──────────────────────────────────────────────────
// VTT conversion
// ──────────────────────────────────────────────────

#[test]
fn test_convert_to_vtt() {
    let (dir, input) = setup_test_dir("vtt");
    let output_dir = dir.join("out");

    cmd()
        .args([
            "convert",
            input.to_str().unwrap(),
            "-f",
            "vtt",
            "-o",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let output = std::fs::read_to_string(output_dir.join("output.vtt")).unwrap();

    let expected = "\
WEBVTT

1
00:00:00.000 --> 00:00:02.500
Hello world.

2
00:00:02.500 --> 00:00:05.000
How are you?

3
00:00:05.000 --> 00:00:05.000
I am fine.

";
    assert_eq!(output, expected);

    let _ = std::fs::remove_dir_all(&dir);
}

// ──────────────────────────────────────────────────
// TXT conversion
// ──────────────────────────────────────────────────

#[test]
fn test_convert_to_txt() {
    let (dir, input) = setup_test_dir("txt");
    let output_dir = dir.join("out");

    cmd()
        .args([
            "convert",
            input.to_str().unwrap(),
            "-f",
            "txt",
            "-o",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let output = std::fs::read_to_string(output_dir.join("output.txt")).unwrap();

    let expected = "Hello world.\nHow are you?\nI am fine.\n";
    assert_eq!(output, expected);

    let _ = std::fs::remove_dir_all(&dir);
}

// ──────────────────────────────────────────────────
// Verbose output
// ──────────────────────────────────────────────────

#[test]
fn test_convert_verbose_srt() {
    let (dir, input) = setup_test_dir("verbose-srt");
    let output_dir = dir.join("out");

    cmd()
        .args([
            "convert",
            input.to_str().unwrap(),
            "-f",
            "srt",
            "-o",
            output_dir.to_str().unwrap(),
            "--verbose",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("00:00:00,000 --> 00:00:02,500"))
        .stdout(predicate::str::contains("Hello world."));

    let _ = std::fs::remove_dir_all(&dir);
}

// ──────────────────────────────────────────────────
// Output filename is always output.<format>
// ──────────────────────────────────────────────────

#[test]
fn test_convert_output_filename() {
    let (dir, input) = setup_test_dir("filename");

    for (fmt, ext) in [("srt", "srt"), ("vtt", "vtt"), ("txt", "txt")] {
        let output_dir = dir.join(format!("out_{fmt}"));
        cmd()
            .args([
                "convert",
                input.to_str().unwrap(),
                "-f",
                fmt,
                "-o",
                output_dir.to_str().unwrap(),
            ])
            .assert()
            .success();

        let expected_path = output_dir.join(format!("output.{ext}"));
        assert!(
            expected_path.exists(),
            "Expected output file at {}, but it doesn't exist",
            expected_path.display()
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
