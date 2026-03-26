#[allow(dead_code)]
mod audio;
#[allow(dead_code)]
mod cli;
#[allow(dead_code)]
mod convert;
#[allow(dead_code)]
mod inference;
#[allow(dead_code)]
mod model;
#[allow(dead_code)]
mod output;

use clap::Parser;
use cli::{Cli, Commands};

fn main() {
    let cli = Cli::parse();

    // Validate arguments (exits with code 2 on failure)
    cli::validate_args(&cli);

    match cli.command {
        Some(Commands::Convert {
            input_file,
            output_format,
            output_dir,
            verbose,
        }) => {
            if let Err(e) = convert::run_convert(&input_file, &output_format, &output_dir, verbose)
            {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        None => {
            // Main transcription command
            let file_name = match cli.file_name {
                Some(ref f) => f.clone(),
                None => {
                    eprintln!("error: --file-name is required for transcription");
                    std::process::exit(2);
                }
            };

            // Determine effective task (strip for .en models)
            let task = if cli::is_english_only_model(&cli.model_name) {
                "transcribe".to_string()
            } else {
                cli.task.clone()
            };

            // Check flash attention availability
            let flash = cli::is_flash_enabled(&cli.flash);
            if flash {
                #[cfg(not(feature = "flash-attn"))]
                {
                    eprintln!("Warning: --flash True requires compilation with 'flash-attn' feature. Using SDPA instead.");
                }
            }

            // Load audio
            let samples = match audio::load_audio(&file_name) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error loading audio: {e}");
                    std::process::exit(1);
                }
            };

            // Load model
            let mut whisper =
                match model::load_whisper_model(&cli.model_name, &cli.hf_token, &cli.device_id) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Error loading model: {e}");
                        std::process::exit(1);
                    }
                };

            // Run batched transcription
            let segments = match inference::transcribe(
                &mut whisper,
                &samples,
                &task,
                &cli.language,
                cli.batch_size,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error during transcription: {e}");
                    std::process::exit(1);
                }
            };

            // Write output JSON
            if let Err(e) = output::write_output(&segments, &cli.transcript_path) {
                eprintln!("Error writing output: {e}");
                std::process::exit(1);
            }

            // Print success message to stdout (matching Python exactly)
            output::print_success_message(&cli.transcript_path, false);
        }
    }
}
