use assert_cmd::Command;
use predicates::prelude::*;

fn cmd() -> Command {
    Command::cargo_bin("fast-whisper-rs").unwrap()
}

// ──────────────────────────────────────────────────
// Valid argument combinations
// ──────────────────────────────────────────────────

#[test]
fn test_default_args_require_file_name() {
    // Running with no arguments but no --file-name should exit 2
    cmd().assert().failure().code(2);
}

#[test]
fn test_flash_true_accepted() {
    // --flash True is valid (will fail at runtime because file doesn't exist, but parsing succeeds)
    cmd()
        .args(["--file-name", "nonexistent.wav", "--flash", "True"])
        .assert()
        .failure()
        .code(1) // runtime error (file not found), not parse error
        .stderr(predicate::str::contains("Error"));
}

#[test]
fn test_flash_false_accepted() {
    cmd()
        .args(["--file-name", "nonexistent.wav", "--flash", "False"])
        .assert()
        .failure()
        .code(1);
}

#[test]
fn test_flash_invalid_value_rejected() {
    cmd()
        .args(["--file-name", "test.wav", "--flash", "yes"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn test_diarization_model_underscore_flag() {
    // --diarization_model (underscore) must be accepted
    cmd()
        .args([
            "--file-name",
            "nonexistent.wav",
            "--diarization_model",
            "my/model",
        ])
        .assert()
        .failure()
        .code(1); // runtime error, not parse error
}

// ──────────────────────────────────────────────────
// Speaker count validation (exit code 2)
// ──────────────────────────────────────────────────

#[test]
fn test_num_speakers_with_min_speakers_exits_2() {
    cmd()
        .args([
            "--file-name",
            "test.wav",
            "--num-speakers",
            "3",
            "--min-speakers",
            "1",
        ])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("mutually exclusive"));
}

#[test]
fn test_num_speakers_with_max_speakers_exits_2() {
    cmd()
        .args([
            "--file-name",
            "test.wav",
            "--num-speakers",
            "3",
            "--max-speakers",
            "5",
        ])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("mutually exclusive"));
}

#[test]
fn test_min_greater_than_max_exits_2() {
    cmd()
        .args([
            "--file-name",
            "test.wav",
            "--min-speakers",
            "5",
            "--max-speakers",
            "2",
        ])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("min-speakers"));
}

#[test]
fn test_num_speakers_zero_exits_2() {
    cmd()
        .args(["--file-name", "test.wav", "--num-speakers", "0"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("must be >= 1"));
}

// ──────────────────────────────────────────────────
// Task and timestamp validation
// ──────────────────────────────────────────────────

#[test]
fn test_invalid_task_exits_2() {
    cmd()
        .args(["--file-name", "test.wav", "--task", "summarize"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn test_invalid_timestamp_exits_2() {
    cmd()
        .args(["--file-name", "test.wav", "--timestamp", "sentence"])
        .assert()
        .failure()
        .code(2);
}

// ──────────────────────────────────────────────────
// Runtime errors exit with code 1
// ──────────────────────────────────────────────────

#[test]
fn test_nonexistent_file_exits_1() {
    cmd()
        .args(["--file-name", "/tmp/this_file_does_not_exist_12345.wav"])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("Error"));
}

// ──────────────────────────────────────────────────
// Convert subcommand validation
// ──────────────────────────────────────────────────

#[test]
fn test_convert_invalid_format_exits_2() {
    cmd()
        .args(["convert", "input.json", "-f", "mp4"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn test_convert_missing_input_exits_2() {
    cmd()
        .args(["convert", "-f", "srt"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn test_convert_nonexistent_input_exits_1() {
    cmd()
        .args(["convert", "/tmp/nonexistent_input_12345.json", "-f", "srt"])
        .assert()
        .failure()
        .code(1);
}
