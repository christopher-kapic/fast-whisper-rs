#!/usr/bin/env python3
"""Export pyannote diarization models to ONNX and extract PLDA artifacts.

This script:
1. Loads the pyannote/speaker-diarization-3.1 segmentation model and exports it to ONNX
2. Downloads the WeSpeaker ResNet34 ONNX embedding model from HuggingFace
3. Extracts PLDA .npz files from the pipeline
4. Validates exported models by running onnxruntime inference on dummy inputs

Requirements:
    pip install pyannote.audio torch onnx onnxruntime numpy huggingface_hub

Usage:
    python scripts/export_diarization_models.py --output-dir ./diarization_models
    # Or with a custom HuggingFace token:
    python scripts/export_diarization_models.py --output-dir ./diarization_models --hf-token hf_xxxxx

To trigger manually via GitHub Actions:
    gh workflow run export-models.yml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def export_segmentation_model(pipeline, output_dir: Path, opset: int = 17) -> Path:
    """Export the PyanNet segmentation model to ONNX."""
    print("Exporting segmentation model to ONNX...")

    seg_model = pipeline._segmentation.model_
    seg_model.eval()

    # Input: (batch, 1, num_samples) at 16kHz, 10-second window = 160000 samples
    dummy_input = torch.randn(1, 1, 160000, dtype=torch.float32)

    output_path = output_dir / "segmentation.onnx"

    torch.onnx.export(
        seg_model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "num_samples"},
            "output": {0: "batch", 1: "num_frames"},
        },
    )

    print(f"  Saved segmentation model to {output_path}")
    return output_path


def download_embedding_model(output_dir: Path) -> Path:
    """Download the WeSpeaker ResNet34 ONNX embedding model from HuggingFace."""
    print("Downloading WeSpeaker embedding model...")

    from huggingface_hub import hf_hub_download

    # The ONNX model is available directly on HuggingFace
    local_path = hf_hub_download(
        repo_id="hbredin/wespeaker-voxceleb-resnet34-LM",
        filename="speaker-embedding.onnx",
    )

    output_path = output_dir / "embedding.onnx"

    # Copy to output directory
    import shutil
    shutil.copy2(local_path, output_path)

    print(f"  Saved embedding model to {output_path}")
    return output_path


def extract_plda_files(pipeline, output_dir: Path) -> tuple[Path, Path]:
    """Extract PLDA and xvector transform .npz files from the pipeline."""
    print("Extracting PLDA artifacts...")

    clustering = pipeline.klustering

    # Extract xvector transform (mean + LDA matrix)
    xvec_path = output_dir / "plda_xvec_transform.npz"
    np.savez(
        xvec_path,
        mean=np.array(clustering.preprocessing_transform_params_["mean"], dtype=np.float32),
        transform=np.array(clustering.preprocessing_transform_params_["transform"], dtype=np.float32),
    )
    print(f"  Saved xvector transform to {xvec_path}")

    # Extract PLDA model parameters
    plda_path = output_dir / "plda.npz"
    plda_params = clustering.plda_params_
    np.savez(
        plda_path,
        phi=np.array(plda_params["Phi"], dtype=np.float32),
        sigma=np.array(plda_params["Sigma"], dtype=np.float32),
    )
    print(f"  Saved PLDA parameters to {plda_path}")

    return xvec_path, plda_path


def validate_segmentation(model_path: Path) -> None:
    """Validate segmentation ONNX model with dummy input."""
    print("Validating segmentation model...")

    session = ort.InferenceSession(str(model_path))

    # 10-second input at 16kHz
    dummy = np.random.randn(1, 1, 160000).astype(np.float32)
    outputs = session.run(None, {"input": dummy})

    output = outputs[0]
    assert output.ndim == 3, f"Expected 3D output, got {output.ndim}D"
    assert output.shape[0] == 1, f"Expected batch=1, got {output.shape[0]}"
    # num_frames depends on model stride (~587 for 10s), num_speakers is typically 3-7
    print(f"  Segmentation output shape: {output.shape} (batch, num_frames, num_speakers)")
    print("  Segmentation model validated successfully.")


def validate_embedding(model_path: Path) -> None:
    """Validate WeSpeaker ONNX embedding model with dummy input."""
    print("Validating embedding model...")

    session = ort.InferenceSession(str(model_path))

    # Check input names to determine expected format
    input_info = session.get_inputs()
    print(f"  Embedding model inputs: {[(i.name, i.shape) for i in input_info]}")

    output_info = session.get_outputs()
    print(f"  Embedding model outputs: {[(o.name, o.shape) for o in output_info]}")

    # WeSpeaker expects fbank features: (batch, num_frames, 80)
    dummy = np.random.randn(1, 100, 80).astype(np.float32)
    input_name = input_info[0].name
    outputs = session.run(None, {input_name: dummy})

    output = outputs[0]
    assert output.ndim == 2, f"Expected 2D output, got {output.ndim}D"
    assert output.shape[0] == 1, f"Expected batch=1, got {output.shape[0]}"
    print(f"  Embedding output shape: {output.shape} (batch, embedding_dim)")
    print("  Embedding model validated successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Export pyannote diarization models to ONNX and PLDA artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./diarization_models",
        help="Directory to save exported models (default: ./diarization_models)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for accessing pyannote models",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "Error: HuggingFace token required. Provide via --hf-token or HF_TOKEN env var.",
            file=sys.stderr,
        )
        print(
            "You need access to pyannote/speaker-diarization-3.1 on HuggingFace.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load the full pipeline to access sub-models
    print("Loading pyannote pipeline (this may take a minute)...")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # Export segmentation model
    seg_path = export_segmentation_model(pipeline, output_dir)

    # Download embedding model
    emb_path = download_embedding_model(output_dir)

    # Extract PLDA files
    xvec_path, plda_path = extract_plda_files(pipeline, output_dir)

    # Validate exports
    print("\nValidating exported models...")
    validate_segmentation(seg_path)
    validate_embedding(emb_path)

    # Summary
    print("\nExport complete! Artifacts:")
    for f in [seg_path, emb_path, xvec_path, plda_path]:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    print(f"\nAll artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
