pub mod segmentation;

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Default GitHub release URL base for diarization model artifacts.
const DEFAULT_RELEASE_URL: &str =
    "https://github.com/chriswmann/fast-whisper-rs/releases/download/diarization-models-v1";

/// Model artifact filenames.
const SEGMENTATION_MODEL: &str = "segmentation.onnx";
const EMBEDDING_MODEL: &str = "embedding.onnx";
const PLDA_XVEC_TRANSFORM: &str = "plda_xvec_transform.npz";
const PLDA_FILE: &str = "plda.npz";

const ALL_ARTIFACTS: &[&str] = &[
    SEGMENTATION_MODEL,
    EMBEDDING_MODEL,
    PLDA_XVEC_TRANSFORM,
    PLDA_FILE,
];

/// PLDA transform parameters loaded from xvec_transform.npz.
pub struct PldaTransform {
    /// Mean vector for centering embeddings.
    pub mean: Array2<f32>,
    /// LDA transform matrix.
    pub transform: Array2<f32>,
}

/// PLDA model parameters loaded from plda.npz.
pub struct PldaModel {
    /// Within-class covariance (Phi).
    pub phi: Array2<f32>,
    /// Between-class covariance (Sigma).
    pub sigma: Array2<f32>,
}

/// All loaded diarization models and parameters.
pub struct DiarizationModels {
    pub segmentation: ort::session::Session,
    pub embedding: ort::session::Session,
    pub plda_transform: PldaTransform,
    pub plda_model: PldaModel,
}

/// Returns the cache directory for diarization models.
///
/// Respects `XDG_CACHE_HOME` if set, otherwise uses `~/.cache`.
/// Final path: `{cache_base}/fast-whisper-rs/diarization/`
pub fn cache_dir() -> Result<PathBuf> {
    let cache_base = if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg)
    } else {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("Could not determine home directory")?;
        PathBuf::from(home).join(".cache")
    };
    Ok(cache_base.join("fast-whisper-rs").join("diarization"))
}

/// Downloads a single file from `url` to `dest`, showing progress.
fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::new();
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to request {url}"))?;

    if !response.status().is_success() {
        bail!(
            "Failed to download {url}: HTTP {}",
            response.status()
        );
    }

    let total_size = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("█▓░"),
    );

    let file_name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    pb.set_message(file_name);

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    let mut file = std::fs::File::create(dest)
        .with_context(|| format!("Failed to create file {}", dest.display()))?;

    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }
    pb.finish_with_message("done");

    Ok(())
}

/// Ensures all diarization model artifacts are present in the cache directory.
///
/// Downloads any missing files from the GitHub release URL.
/// Skips files that already exist (no content hash check).
pub fn ensure_models(release_url: Option<&str>) -> Result<PathBuf> {
    let dir = cache_dir()?;
    let base_url = release_url.unwrap_or(DEFAULT_RELEASE_URL);

    for artifact in ALL_ARTIFACTS {
        let dest = dir.join(artifact);
        if dest.exists() {
            eprintln!("Cached: {artifact}");
            continue;
        }
        let url = format!("{base_url}/{artifact}");
        eprintln!("Downloading: {artifact}");
        download_file(&url, &dest)?;
    }

    Ok(dir)
}

/// Load an ONNX model as an ort Session with appropriate execution provider.
///
/// Tries CUDA (if device_id is numeric) or CoreML (if device_id is "mps"),
/// falling back to CPU.
pub fn load_onnx_session(model_path: &Path, device_id: &str) -> Result<ort::session::Session> {
    let builder = ort::session::Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?;

    // Configure execution providers based on device_id
    let mut builder = if device_id == "mps" {
        builder
            .with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("Failed to set CoreML execution provider: {e}"))?
    } else if device_id.parse::<usize>().is_ok() {
        builder
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("Failed to set CUDA execution provider: {e}"))?
    } else {
        builder
    };

    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to load ONNX model from {}: {e}",
                model_path.display()
            )
        })?;

    Ok(session)
}

/// Parse a .npy file from raw bytes into an Array2<f32>.
fn parse_npy_to_array2(data: &[u8]) -> Result<Array2<f32>> {
    use npyz::NpyFile;

    let npy = NpyFile::new(data)?;
    let shape = npy.shape().to_vec();
    let flat: Vec<f32> = npy.into_vec()?;

    match shape.len() {
        1 => {
            // Treat 1D as (1, N)
            Ok(Array2::from_shape_vec((1, shape[0] as usize), flat)?)
        }
        2 => Ok(Array2::from_shape_vec(
            (shape[0] as usize, shape[1] as usize),
            flat,
        )?),
        _ => bail!("Expected 1D or 2D array in .npy, got {}D", shape.len()),
    }
}

/// Load a .npz file (zip of .npy files) and extract named arrays.
fn load_npz_arrays(path: &Path) -> Result<std::collections::HashMap<String, Array2<f32>>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)?;

    let mut arrays = std::collections::HashMap::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry
            .name()
            .trim_end_matches(".npy")
            .to_string();

        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        let arr = parse_npy_to_array2(&buf)
            .with_context(|| format!("Failed to parse array '{name}' from {}", path.display()))?;
        arrays.insert(name, arr);
    }

    Ok(arrays)
}

/// Load PLDA transform parameters from xvec_transform.npz.
///
/// Expected arrays: 'mean' and 'transform'.
pub fn load_plda_transform(path: &Path) -> Result<PldaTransform> {
    let arrays = load_npz_arrays(path)?;

    let mean = arrays
        .get("mean")
        .context("xvec_transform.npz missing 'mean' array")?
        .clone();
    let transform = arrays
        .get("transform")
        .context("xvec_transform.npz missing 'transform' array")?
        .clone();

    Ok(PldaTransform { mean, transform })
}

/// Load PLDA model parameters from plda.npz.
///
/// Expected arrays: 'Phi' and 'Sigma' (note capitalization matching Python source).
pub fn load_plda_model(path: &Path) -> Result<PldaModel> {
    let arrays = load_npz_arrays(path)?;

    let phi = arrays
        .get("Phi")
        .context("plda.npz missing 'Phi' array")?
        .clone();
    let sigma = arrays
        .get("Sigma")
        .context("plda.npz missing 'Sigma' array")?
        .clone();

    Ok(PldaModel { phi, sigma })
}

/// Load all diarization models from the cache directory.
pub fn load_all_models(cache_dir: &Path, device_id: &str) -> Result<DiarizationModels> {
    let segmentation = load_onnx_session(&cache_dir.join(SEGMENTATION_MODEL), device_id)?;
    let embedding = load_onnx_session(&cache_dir.join(EMBEDDING_MODEL), device_id)?;
    let plda_transform = load_plda_transform(&cache_dir.join(PLDA_XVEC_TRANSFORM))?;
    let plda_model = load_plda_model(&cache_dir.join(PLDA_FILE))?;

    Ok(DiarizationModels {
        segmentation,
        embedding,
        plda_transform,
        plda_model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_cache_dir_default() {
        // Temporarily unset XDG_CACHE_HOME to test default behavior
        let original = env::var("XDG_CACHE_HOME").ok();
        env::remove_var("XDG_CACHE_HOME");

        let dir = cache_dir().unwrap();
        let home = env::var("HOME").unwrap();
        assert_eq!(
            dir,
            PathBuf::from(home)
                .join(".cache")
                .join("fast-whisper-rs")
                .join("diarization")
        );

        // Restore
        if let Some(val) = original {
            env::set_var("XDG_CACHE_HOME", val);
        }
    }

    #[test]
    fn test_cache_dir_with_xdg() {
        let original = env::var("XDG_CACHE_HOME").ok();
        env::set_var("XDG_CACHE_HOME", "/tmp/test-xdg-cache");

        let dir = cache_dir().unwrap();
        assert_eq!(
            dir,
            PathBuf::from("/tmp/test-xdg-cache")
                .join("fast-whisper-rs")
                .join("diarization")
        );

        // Restore
        match original {
            Some(val) => env::set_var("XDG_CACHE_HOME", val),
            None => env::remove_var("XDG_CACHE_HOME"),
        }
    }

    #[test]
    fn test_file_existence_check() {
        // Existing file should be detected
        let existing = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        assert!(existing.exists());

        // Non-existing file should not
        let missing = PathBuf::from("/tmp/fast-whisper-rs-test-nonexistent-12345.onnx");
        assert!(!missing.exists());
    }

    #[test]
    fn test_all_artifacts_defined() {
        assert_eq!(ALL_ARTIFACTS.len(), 4);
        assert!(ALL_ARTIFACTS.contains(&"segmentation.onnx"));
        assert!(ALL_ARTIFACTS.contains(&"embedding.onnx"));
        assert!(ALL_ARTIFACTS.contains(&"plda_xvec_transform.npz"));
        assert!(ALL_ARTIFACTS.contains(&"plda.npz"));
    }

    #[test]
    fn test_parse_npy_to_array2() {
        // Create a minimal valid .npy file for a 2x3 f32 array
        let mut buf = Vec::new();
        // Magic: \x93NUMPY
        buf.extend_from_slice(b"\x93NUMPY");
        // Version 1.0
        buf.push(1);
        buf.push(0);
        // Header: needs to describe dtype and shape
        let header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        // Pad header to align to 64 bytes total (magic=6 + version=2 + header_len=2 + header)
        let preamble_len = 6 + 2 + 2; // magic + version + header_len field
        let total_header = header.len();
        let padding_needed = 64 - ((preamble_len + total_header) % 64);
        let padded_len = total_header + padding_needed;

        buf.extend_from_slice(&(padded_len as u16).to_le_bytes());
        buf.extend_from_slice(header);
        // Pad with spaces, end with \n
        for _ in 0..padding_needed - 1 {
            buf.push(b' ');
        }
        buf.push(b'\n');

        // Data: 6 f32 values
        for i in 0..6u32 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }

        let arr = parse_npy_to_array2(&buf).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 0.0);
        assert_eq!(arr[[1, 2]], 5.0);
    }

    // Integration tests that require model files use #[ignore]
    #[test]
    #[ignore]
    fn test_ensure_models_downloads() {
        let dir = ensure_models(None).unwrap();
        for artifact in ALL_ARTIFACTS {
            assert!(dir.join(artifact).exists(), "Missing: {artifact}");
        }
    }

    #[test]
    #[ignore]
    fn test_load_onnx_session() {
        let dir = ensure_models(None).unwrap();
        let session = load_onnx_session(&dir.join(SEGMENTATION_MODEL), "0");
        assert!(session.is_ok());
    }

    #[test]
    #[ignore]
    fn test_load_plda_files() {
        let dir = ensure_models(None).unwrap();
        let transform = load_plda_transform(&dir.join(PLDA_XVEC_TRANSFORM));
        assert!(transform.is_ok());
        let model = load_plda_model(&dir.join(PLDA_FILE));
        assert!(model.is_ok());
    }
}
