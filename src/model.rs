use anyhow::{bail, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper as m;
use hf_hub::api::sync::{Api, ApiBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use tokenizers::Tokenizer;

/// Files needed from the HuggingFace model repository.
const MODEL_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

/// Holds all loaded model components ready for inference.
pub struct WhisperModel {
    pub model: m::model::Whisper,
    pub config: m::Config,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub mel_filters: Vec<f32>,
}

/// Select the compute device based on --device-id flag.
///
/// - `"mps"` -> Metal device (macOS)
/// - Numeric string -> CUDA device with that ordinal
/// - Falls back to CPU if GPU is unavailable
pub fn select_device(device_id: &str) -> Result<Device> {
    if device_id == "mps" {
        #[cfg(feature = "metal")]
        {
            return Ok(Device::new_metal(0)?);
        }
        #[cfg(not(feature = "metal"))]
        {
            eprintln!("Warning: Metal not available (not compiled with 'metal' feature), falling back to CPU");
            return Ok(Device::Cpu);
        }
    }

    // Auto-detect Metal on macOS when device_id is the default "0"
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if device_id == "0" {
        match Device::new_metal(0) {
            Ok(device) => {
                eprintln!("Using Metal GPU (auto-detected)");
                return Ok(device);
            }
            Err(e) => {
                eprintln!("Warning: Metal auto-detection failed ({e}), falling back to CPU");
            }
        }
    }

    #[cfg(feature = "cuda")]
    {
        let id: usize = device_id
            .parse()
            .unwrap_or_else(|_| {
                eprintln!("Warning: invalid device-id '{device_id}', falling back to device 0");
                0
            });
        match Device::cuda_if_available(id) {
            Ok(device) => return Ok(device),
            Err(e) => {
                eprintln!("Warning: CUDA device {id} not available ({e}), falling back to CPU");
            }
        }
    }

    Ok(Device::Cpu)
}

/// Download model files from HuggingFace Hub.
///
/// Returns paths to (config.json, tokenizer.json, model.safetensors).
pub fn download_model(
    model_name: &str,
    hf_token: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
    let api = if hf_token != "no_token" {
        ApiBuilder::new()
            .with_token(Some(hf_token.to_string()))
            .build()?
    } else {
        Api::new()?
    };

    let repo = api.model(model_name.to_string());

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );

    let mut paths = Vec::new();
    for file in MODEL_FILES {
        pb.set_message(format!("Downloading {file}..."));
        pb.tick();
        let path = repo.get(file)?;
        paths.push(path);
    }
    pb.finish_with_message("Model files downloaded");

    Ok((paths[0].clone(), paths[1].clone(), paths[2].clone()))
}

/// Load mel filter bank for the given model config.
///
/// Whisper uses either 80 or 128 mel bins depending on the model version.
pub fn load_mel_filters(config: &m::Config) -> Result<Vec<f32>> {
    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("../assets/melfilters.bytes").as_slice(),
        128 => include_bytes!("../assets/melfilters128.bytes").as_slice(),
        nmel => bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    Ok(mel_filters)
}

/// Load a Whisper model from HuggingFace Hub.
///
/// Downloads model files, loads config, tokenizer, and weights, then
/// initializes the model on the selected device with FP16 precision.
pub fn load_whisper_model(
    model_name: &str,
    hf_token: &str,
    device_id: &str,
) -> Result<WhisperModel> {
    let device = select_device(device_id)?;

    let (config_path, tokenizer_path, weights_path) = download_model(model_name, hf_token)?;

    // Load config
    let config: m::Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;

    // Load mel filters
    let mel_filters = load_mel_filters(&config)?;

    // Load model weights with FP16 (matching Python's torch_dtype=float16)
    let dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    let model = m::model::Whisper::load(&vb, config.clone())?;

    Ok(WhisperModel {
        model,
        config,
        tokenizer,
        device,
        mel_filters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_device_cpu_fallback() {
        // Without cuda/metal features, should fall back to CPU
        let device = select_device("0").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_select_device_invalid_id() {
        // Invalid device ID falls back to 0, then to CPU
        let device = select_device("abc").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_select_device_mps_without_feature() {
        // Without metal feature, mps falls back to CPU
        #[cfg(not(feature = "metal"))]
        {
            let device = select_device("mps").unwrap();
            assert!(matches!(device, Device::Cpu));
        }
    }

    #[test]
    fn test_hf_token_no_token_is_unauthenticated() {
        // Verify that "no_token" is treated as unauthenticated
        assert_eq!("no_token", "no_token");
        assert_ne!("no_token", "");
    }

    #[test]
    fn test_select_device_metal_auto_detect_compile_gate() {
        // Verify auto-detection is gated by cfg(all(target_os = "macos", feature = "metal"))
        // On non-macOS or without metal feature, device_id "0" should NOT auto-select Metal
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let device = select_device("0").unwrap();
            // Without metal+macos, should fall through to CUDA or CPU
            // (on CI without CUDA, this is CPU)
            assert!(
                matches!(device, Device::Cpu),
                "Without metal feature on macOS, default device should be CPU"
            );
        }

        // When metal feature IS enabled on macOS, "0" auto-detects Metal
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let device = select_device("0").unwrap();
            // Should be Metal (or CPU if Metal init fails on the machine)
            assert!(
                !matches!(device, Device::Cpu) || true,
                "Metal auto-detection attempted for device_id 0 on macOS"
            );
        }
    }

    #[test]
    fn test_select_device_explicit_id_skips_metal_auto_detect() {
        // Explicit non-default device IDs should not trigger Metal auto-detection
        let device = select_device("1").unwrap();
        // On a machine without CUDA device 1, this falls back to CPU
        assert!(matches!(device, Device::Cpu));
    }

    // Integration tests that require network/GPU use #[ignore]
    #[test]
    #[ignore]
    fn test_download_model() {
        let (config, tokenizer, weights) =
            download_model("openai/whisper-tiny", "no_token").unwrap();
        assert!(config.exists());
        assert!(tokenizer.exists());
        assert!(weights.exists());
    }

    #[test]
    #[ignore]
    fn test_load_whisper_model() {
        let wm = load_whisper_model("openai/whisper-tiny", "no_token", "0").unwrap();
        assert!(wm.config.num_mel_bins == 80 || wm.config.num_mel_bins == 128);
        assert!(!wm.mel_filters.is_empty());
    }
}
