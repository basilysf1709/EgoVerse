"""Download the PaliGemma backbone into the egoverse-training-outputs volume.

pi0.5 (openpi PyTorch) builds PaliGemma from a random config; to initialize the
VLM backbone from pretrained weights we need the HF snapshot available on a
volume so the training container loads it from a local path (no per-container
re-download). This fetches it once using the egoverse-hf HF_TOKEN secret (which
has accepted the gated google/paligemma license).

Run:
  MODAL_ENVIRONMENT=robotics modal run \
    egomimic/scripts/data_download/download_paligemma_modal.py -- \
    --repo google/paligemma-3b-pt-224

Result:
  egoverse-training-outputs:/paligemma_weights/<repo basename>/  (full HF snapshot)
which mounts at /root/EgoVerse/logs/paligemma_weights/<basename> in training.
"""

from __future__ import annotations

import modal

OUTPUT_VOLUME_NAME = "egoverse-training-outputs"
OUTPUT_MOUNT = "/out"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["huggingface_hub>=0.24", "hf_transfer>=0.1.6"])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("egomimic-paligemma-download", image=image)
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={OUTPUT_MOUNT: output_vol},
    secrets=[modal.Secret.from_name("egoverse-hf")],
    timeout=3600,
)
def download(
    repo: str = "google/paligemma-3b-pt-224", subdir: str = "paligemma_weights"
) -> dict:
    import os
    from pathlib import Path

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    dest = Path(OUTPUT_MOUNT) / subdir / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    # Skip the .gguf quantized dumps; we only need config + safetensors + tokenizer.
    path = snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        token=token,
        ignore_patterns=["*.gguf"],
    )
    files = sorted(p.name for p in Path(path).rglob("*") if p.is_file())
    total_gb = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9
    output_vol.commit()
    rel = str(dest.relative_to(OUTPUT_MOUNT))
    return {
        "volume_rel_path": rel,
        "n_files": len(files),
        "total_gb": round(total_gb, 2),
        "files": files[:40],
    }


@app.local_entrypoint()
def main(repo: str = "google/paligemma-3b-pt-224") -> None:
    info = download.remote(repo=repo)
    print(f"Downloaded {repo}")
    print(
        f"  volume-relative path: {info['volume_rel_path']}  (→ /root/EgoVerse/logs/{info['volume_rel_path']})"
    )
    print(f"  files: {info['n_files']}   size: {info['total_gb']} GB")
    for f in info["files"]:
        print("   -", f)
