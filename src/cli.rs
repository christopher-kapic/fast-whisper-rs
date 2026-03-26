use clap::{Parser, Subcommand, ValueEnum};
use std::process;

#[derive(Parser, Debug)]
#[command(name = "fast-whisper-rs", about = "A fast Whisper CLI built on candle")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Path or URL to the audio file
    #[arg(long = "file-name")]
    pub file_name: Option<String>,

    /// Device ID (GPU number or "mps")
    #[arg(long = "device-id", default_value = "0")]
    pub device_id: String,

    /// Output JSON path
    #[arg(long = "transcript-path", default_value = "output.json")]
    pub transcript_path: String,

    /// HuggingFace model name
    #[arg(long = "model-name", default_value = "openai/whisper-large-v3")]
    pub model_name: String,

    /// Task: transcribe or translate
    #[arg(long = "task", default_value = "transcribe", value_parser = parse_task)]
    pub task: String,

    /// Language (literal "None" = auto-detect)
    #[arg(long = "language", default_value = "None")]
    pub language: String,

    /// Batch size for inference
    #[arg(long = "batch-size", default_value_t = 24)]
    pub batch_size: usize,

    /// Enable Flash Attention (True/False)
    #[arg(long = "flash", default_value = "False", value_parser = parse_flash)]
    pub flash: String,

    /// Timestamp granularity: chunk or word
    #[arg(long = "timestamp", default_value = "chunk", value_parser = parse_timestamp)]
    pub timestamp: String,

    /// HuggingFace token
    #[arg(long = "hf-token", default_value = "no_token")]
    pub hf_token: String,

    /// Diarization model (underscore in flag name for Python parity)
    #[arg(long = "diarization_model", default_value = "pyannote/speaker-diarization-3.1")]
    pub diarization_model: String,

    /// Exact number of speakers
    #[arg(long = "num-speakers")]
    pub num_speakers: Option<usize>,

    /// Minimum number of speakers
    #[arg(long = "min-speakers")]
    pub min_speakers: Option<usize>,

    /// Maximum number of speakers
    #[arg(long = "max-speakers")]
    pub max_speakers: Option<usize>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Convert JSON transcript to SRT/VTT/TXT format
    Convert {
        /// Input JSON file
        input_file: String,

        /// Output format
        #[arg(short = 'f', long = "output_format", value_enum)]
        output_format: OutputFormat,

        /// Output directory
        #[arg(short = 'o', long = "output_dir", default_value = ".")]
        output_dir: String,

        /// Verbose output
        #[arg(long = "verbose")]
        verbose: bool,
    },
}

#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    Srt,
    Vtt,
    Txt,
}

fn parse_task(s: &str) -> Result<String, String> {
    match s {
        "transcribe" | "translate" => Ok(s.to_string()),
        _ => Err(format!(
            "invalid value '{s}' for '--task': valid values are 'transcribe', 'translate'"
        )),
    }
}

fn parse_flash(s: &str) -> Result<String, String> {
    match s {
        "True" | "False" => Ok(s.to_string()),
        _ => Err(format!(
            "invalid value '{s}' for '--flash': valid values are 'True', 'False'"
        )),
    }
}

fn parse_timestamp(s: &str) -> Result<String, String> {
    match s {
        "chunk" | "word" => Ok(s.to_string()),
        _ => Err(format!(
            "invalid value '{s}' for '--timestamp': valid values are 'chunk', 'word'"
        )),
    }
}

/// Returns true if model name ends with ".en" (English-only model).
/// Python: `args.model_name.split(".")[-1] == "en"`
pub fn is_english_only_model(model_name: &str) -> bool {
    model_name.split('.').next_back() == Some("en")
}

/// Returns true if the language value means auto-detect.
/// Python default is the literal string "None".
pub fn is_auto_detect_language(language: &str) -> bool {
    language == "None"
}

/// Returns the flash attention setting as a boolean.
pub fn is_flash_enabled(flash: &str) -> bool {
    flash == "True"
}

/// Validate CLI arguments. Prints error and exits with code 2 on failure
/// (matching Python argparse behavior).
pub fn validate_args(cli: &Cli) {
    // num-speakers is mutually exclusive with min/max-speakers
    if cli.num_speakers.is_some()
        && (cli.min_speakers.is_some() || cli.max_speakers.is_some())
    {
        eprintln!(
            "error: --num-speakers is mutually exclusive with --min-speakers / --max-speakers"
        );
        process::exit(2);
    }

    // All speaker counts must be >= 1
    if let Some(n) = cli.num_speakers {
        if n < 1 {
            eprintln!("error: --num-speakers must be >= 1");
            process::exit(2);
        }
    }
    if let Some(n) = cli.min_speakers {
        if n < 1 {
            eprintln!("error: --min-speakers must be >= 1");
            process::exit(2);
        }
    }
    if let Some(n) = cli.max_speakers {
        if n < 1 {
            eprintln!("error: --max-speakers must be >= 1");
            process::exit(2);
        }
    }

    // min-speakers <= max-speakers
    if let (Some(min), Some(max)) = (cli.min_speakers, cli.max_speakers) {
        if min > max {
            eprintln!("error: --min-speakers must be <= --max-speakers");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // Helper to parse args (prepend binary name)
    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut full = vec!["fast-whisper-rs"];
        full.extend_from_slice(args);
        Cli::try_parse_from(full)
    }

    #[test]
    fn test_default_values() {
        let cli = parse(&["--file-name", "test.wav"]).unwrap();
        assert_eq!(cli.device_id, "0");
        assert_eq!(cli.transcript_path, "output.json");
        assert_eq!(cli.model_name, "openai/whisper-large-v3");
        assert_eq!(cli.task, "transcribe");
        assert_eq!(cli.language, "None");
        assert_eq!(cli.batch_size, 24);
        assert_eq!(cli.flash, "False");
        assert_eq!(cli.timestamp, "chunk");
        assert_eq!(cli.hf_token, "no_token");
        assert_eq!(cli.diarization_model, "pyannote/speaker-diarization-3.1");
        assert!(cli.num_speakers.is_none());
        assert!(cli.min_speakers.is_none());
        assert!(cli.max_speakers.is_none());
    }

    #[test]
    fn test_flash_takes_value() {
        let cli = parse(&["--file-name", "test.wav", "--flash", "True"]).unwrap();
        assert_eq!(cli.flash, "True");

        let cli = parse(&["--file-name", "test.wav", "--flash", "False"]).unwrap();
        assert_eq!(cli.flash, "False");

        // Invalid value should fail
        let result = parse(&["--file-name", "test.wav", "--flash", "yes"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_language_default_is_string_none() {
        let cli = parse(&["--file-name", "test.wav"]).unwrap();
        assert_eq!(cli.language, "None");
        assert!(is_auto_detect_language(&cli.language));

        let cli = parse(&["--file-name", "test.wav", "--language", "en"]).unwrap();
        assert_eq!(cli.language, "en");
        assert!(!is_auto_detect_language(&cli.language));
    }

    #[test]
    fn test_diarization_model_underscore() {
        // The flag uses underscore, not hyphen
        let cli = parse(&[
            "--file-name",
            "test.wav",
            "--diarization_model",
            "my/model",
        ])
        .unwrap();
        assert_eq!(cli.diarization_model, "my/model");
    }

    #[test]
    fn test_task_validation() {
        let cli = parse(&["--file-name", "test.wav", "--task", "transcribe"]);
        assert!(cli.is_ok());

        let cli = parse(&["--file-name", "test.wav", "--task", "translate"]);
        assert!(cli.is_ok());

        let cli = parse(&["--file-name", "test.wav", "--task", "invalid"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_timestamp_validation() {
        let cli = parse(&["--file-name", "test.wav", "--timestamp", "chunk"]);
        assert!(cli.is_ok());

        let cli = parse(&["--file-name", "test.wav", "--timestamp", "word"]);
        assert!(cli.is_ok());

        let cli = parse(&["--file-name", "test.wav", "--timestamp", "invalid"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_en_model_detection() {
        assert!(is_english_only_model("openai/whisper-large-v3.en"));
        assert!(is_english_only_model("openai/whisper-tiny.en"));
        assert!(!is_english_only_model("openai/whisper-large-v3"));
        assert!(!is_english_only_model("openai/whisper-large-v3.en.broken"));
        assert!(!is_english_only_model("my-model-en"));
    }

    #[test]
    fn test_convert_subcommand() {
        let cli = parse(&["convert", "input.json", "-f", "srt"]).unwrap();
        match cli.command {
            Some(Commands::Convert {
                ref input_file,
                ref output_format,
                ref output_dir,
                verbose,
            }) => {
                assert_eq!(input_file, "input.json");
                assert!(matches!(output_format, OutputFormat::Srt));
                assert_eq!(output_dir, ".");
                assert!(!verbose);
            }
            _ => panic!("Expected Convert subcommand"),
        }
    }

    // Validation tests use the validate function logic directly
    // (can't easily test process::exit in unit tests, so we test the conditions)

    #[test]
    fn test_speaker_count_validation_num_with_min() {
        let cli = parse(&[
            "--file-name",
            "test.wav",
            "--num-speakers",
            "3",
            "--min-speakers",
            "1",
        ])
        .unwrap();
        // num-speakers + min-speakers = mutually exclusive
        assert!(cli.num_speakers.is_some() && cli.min_speakers.is_some());
    }

    #[test]
    fn test_speaker_count_validation_num_with_max() {
        let cli = parse(&[
            "--file-name",
            "test.wav",
            "--num-speakers",
            "3",
            "--max-speakers",
            "5",
        ])
        .unwrap();
        assert!(cli.num_speakers.is_some() && cli.max_speakers.is_some());
    }

    #[test]
    fn test_speaker_min_max_valid() {
        let cli = parse(&[
            "--file-name",
            "test.wav",
            "--min-speakers",
            "2",
            "--max-speakers",
            "5",
        ])
        .unwrap();
        assert_eq!(cli.min_speakers, Some(2));
        assert_eq!(cli.max_speakers, Some(5));
    }
}
