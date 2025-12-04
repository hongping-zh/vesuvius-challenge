"""DynUNet inference on AutoDL for Vesuvius Challenge - 320^3 Real Test Data.

This version is optimized for the real Kaggle test volume (320x320x320).

Key differences from the original:
- Uses test_volume_320.npy (real Kaggle test data)
- Optimized sliding window parameters for 320^3 volume
- More detailed progress tracking

Run on AutoDL from project root as:

    conda activate torch_env
    python run_inference_autodl_320.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


from models.dynunet import VesuviusDynUNet


# -----------------------------------------------------------------------------
# Fixed paths for AutoDL (320^3 version)
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "autodl_dynunet_realdata_8epoch.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints_dynunet_realdata_8epoch" / "best_model.pth"

# Use real 320^3 test data from Kaggle
TEST_VOLUME_PATH = PROJECT_ROOT / "data" / "processed" / "test" / "test_volume_320.npy"

OUTPUT_NPY_PATH = PROJECT_ROOT / "predictions_dynunet_320.npy"
SUBMISSION_PATH = PROJECT_ROOT / "prediction.tif"


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    if model_cfg.get("type", "dynunet") != "dynunet":
        raise ValueError(
            f"Expected model.type='dynunet' in config, got {model_cfg.get('type')}"
        )

    in_channels = model_cfg["in_channels"]
    base_num_features = model_cfg.get("base_num_features", 64)
    out_channels = model_cfg["out_channels"]
    deep_supervision = model_cfg.get("deep_supervision", True)

    model = VesuviusDynUNet(
        in_channels=in_channels,
        base_num_features=base_num_features,
        num_classes=out_channels,
        deep_supervision=deep_supervision,
    )
    model.to(device)
    return model


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading weights: {path}")
    ckpt = torch.load(str(path), map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing {len(missing)} weight keys (ignored with strict=False)")
    if unexpected:
        print(f"Found {len(unexpected)} unused weight keys (ignored with strict=False)")


def load_volume(path: Path) -> np.ndarray:
    """Load and normalize test volume.
    
    Handles both uint8 (Kaggle format) and float32 (preprocessed format).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Test volume not found: {path}")

    print(f"Loading test volume: {path}")
    vol = np.load(str(path))
    
    print(f"  Original dtype: {vol.dtype}")
    print(f"  Shape: {vol.shape}")
    print(f"  Range: [{vol.min():.4f}, {vol.max():.4f}]")
    
    # Convert to float32 if needed
    vol = vol.astype(np.float32)
    
    # Normalize
    mean = vol.mean()
    std = vol.std()
    vol = (vol - mean) / (std + 1e-8)
    
    print(f"  After normalization: [{vol.min():.4f}, {vol.max():.4f}]")
    
    return vol


def sliding_window_inference(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size=(128, 128, 128),
    overlap: float = 0.5,
    batch_size: int = 2,
    device: str = "cuda",
    in_channels: int = 1,
) -> np.ndarray:
    """Sliding window inference with progress tracking.
    
    For 320^3 volume with 128^3 patches and 50% overlap:
    - Stride: 64
    - Patches per dimension: ceil((320-128)/64) + 1 = 4
    - Total patches: 4^3 = 64
    
    With 75% overlap:
    - Stride: 32
    - Patches per dimension: ceil((320-128)/32) + 1 = 7
    - Total patches: 7^3 = 343
    """
    model.eval()
    model.to(device)

    D, H, W = volume.shape
    pd, ph, pw = patch_size

    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    print(f"\nSliding window setup:")
    print(f"  Volume shape: {volume.shape}")
    print(f"  Patch size: {patch_size}")
    print(f"  Overlap: {overlap}")
    print(f"  Stride: ({stride_d}, {stride_h}, {stride_w})")

    patches = []
    for d in range(0, max(1, D - pd + 1), stride_d):
        for h in range(0, max(1, H - ph + 1), stride_h):
            for w in range(0, max(1, W - pw + 1), stride_w):
                patches.append((d, h, w))

    if not patches:
        raise ValueError("No patches generated for given volume/patch_size.")

    print(f"  Total patches: {len(patches)}")
    print(f"  Estimated time: {len(patches) * 2 / 60:.1f} min (assuming 2 sec/patch)")

    output = np.zeros((D, H, W), dtype=np.float32)
    counts = np.zeros((D, H, W), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Inference"):
            batch_coords = patches[i : i + batch_size]
            batch_data = []
            for d, h, w in batch_coords:
                patch = volume[d : d + pd, h : h + ph, w : w + pw]
                pad_d = pd - patch.shape[0]
                pad_h = ph - patch.shape[1]
                pad_w = pw - patch.shape[2]
                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    patch = np.pad(
                        patch,
                        ((0, max(0, pad_d)), (0, max(0, pad_h)), (0, max(0, pad_w))),
                        mode="constant",
                    )
                batch_data.append(patch)

            # (B, 1, D, H, W)
            batch_tensor = torch.from_numpy(np.stack(batch_data)).float().unsqueeze(1)

            # Tile single-channel volume to match model's expected input channels
            if in_channels > 1:
                batch_tensor = batch_tensor.repeat(1, in_channels, 1, 1, 1)

            batch_tensor = batch_tensor.to(device)

            preds = model(batch_tensor)
            preds = torch.sigmoid(preds)
            preds = preds.cpu().numpy()

            for j, (d, h, w) in enumerate(batch_coords):
                d_end = min(d + pd, D)
                h_end = min(h + ph, H)
                w_end = min(w + pw, W)
                pd_eff = d_end - d
                ph_eff = h_end - h
                pw_eff = w_end - w

                output[d:d_end, h:h_end, w:w_end] += preds[j, 0, :pd_eff, :ph_eff, :pw_eff]
                counts[d:d_end, h:h_end, w:w_end] += 1

    output = output / (counts + 1e-8)
    
    print(f"\nInference statistics:")
    print(f"  Output shape: {output.shape}")
    print(f"  Coverage (min counts): {counts.min():.0f}")
    print(f"  Coverage (max counts): {counts.max():.0f}")
    
    return output


def simple_postprocess(preds: np.ndarray, threshold: float = 0.3, min_size: int = 1) -> np.ndarray:
    """Simple threshold-based post-processing.
    
    Parameters
    ----------
    preds : np.ndarray
        Probability map (D, H, W) with values in [0, 1].
    threshold : float
        Threshold for binarization (default 0.3).
    min_size : int
        Minimum connected component size (default 1, no filtering).
    
    Returns
    -------
    np.ndarray
        Binary mask (D, H, W) with values 0 or 1.
    """
    mask = (preds > threshold).astype(np.uint8)
    
    print(f"\nPost-processing:")
    print(f"  Threshold: {threshold}")
    print(f"  Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Positive ratio after threshold: {mask.mean():.6f}")
    
    # Skip connected component filtering if min_size <= 1
    if min_size <= 1:
        return mask
    
    from scipy import ndimage
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    out = np.isin(labeled, keep_labels).astype(np.uint8)
    
    print(f"  Components kept: {len(keep_labels)} / {num}")
    return out


def save_tif_submission(mask: np.ndarray, path: Path) -> None:
    """Save 3D mask as a TIFF file for submission.
    
    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (D, H, W) with values 0 or 1.
    path : Path
        Output path for the TIFF file.
    """
    import tifffile as tiff
    
    # Ensure mask is uint8 and binary
    mask_uint8 = mask.astype(np.uint8)
    
    # Save as TIFF with compression
    tiff.imwrite(str(path), mask_uint8, compression='deflate')
    
    file_size_mb = path.stat().st_size / 1e6
    
    print(f"\n{'='*60}")
    print(f"Submission TIF generated: {path}")
    print(f"  Shape: {mask_uint8.shape}")
    print(f"  Dtype: {mask_uint8.dtype}")
    print(f"  Unique values: {np.unique(mask_uint8)}")
    print(f"  Positive ratio: {mask_uint8.mean():.6f}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"{'='*60}")


def main() -> None:
    print("=" * 60)
    print("AutoDL DynUNet Inference - 320^3 Real Test Data")
    print("=" * 60)
    print()

    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config = load_config(CONFIG_PATH)
    patch_size = tuple(config.get("data", {}).get("patch_size", [128, 128, 128]))
    print(f"\nPatch size from config: {patch_size}")

    model = build_model(config, device)
    load_checkpoint(model, CHECKPOINT_PATH, device)

    volume = load_volume(TEST_VOLUME_PATH)
    
    # Verify correct test volume size
    if volume.shape != (320, 320, 320):
        print(f"\n⚠️  WARNING: Expected volume shape (320, 320, 320), got {volume.shape}")
        print(f"    Make sure you are using the correct test_volume_320.npy file!")
    else:
        print(f"\n✓ Correct volume size: {volume.shape}")

    preds = sliding_window_inference(
        model=model,
        volume=volume,
        patch_size=patch_size,
        overlap=0.5,  # Adjust to 0.75 for higher quality, 0.25 for faster inference
        batch_size=2,  # Reduce to 1 if GPU memory is insufficient
        device=str(device),
        in_channels=config["model"].get("in_channels", 1),
    )

    print(f"\nInference completed!")
    print(f"  Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
    
    np.save(OUTPUT_NPY_PATH, preds.astype(np.float32))
    print(f"  Probability map saved to: {OUTPUT_NPY_PATH}")

    # Post-process: use lower threshold to avoid all-zero submission
    mask = simple_postprocess(preds, threshold=0.3, min_size=1)
    save_tif_submission(mask, SUBMISSION_PATH)

    elapsed = time.time() - start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    
    print("\n" + "=" * 60)
    print("✓ Inference completed successfully!")
    print("=" * 60)
    print(f"Total time: {h}h {m}m {s}s")
    print(f"\nNext steps:")
    print(f"  1. Download {SUBMISSION_PATH} to your local machine")
    print(f"  2. Upload to Kaggle Dataset (replace old version)")
    print(f"  3. Submit to competition")
    print("=" * 60)


if __name__ == "__main__":
    main()
