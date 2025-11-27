"""One-click DynUNet inference on AutoDL for Vesuvius Challenge.

- Uses configs/autodl_dynunet_realdata_8epoch.yaml
- Loads models/checkpoints_dynunet_realdata_8epoch/best_model.pth
- Runs sliding-window inference on data/processed/test/volume.npy
- Applies simple threshold + connected component post-processing
- Writes submission_dynunet_8epoch.csv in a simple flat format.

Run on AutoDL from project root as:

    python run_inference_autodl.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from models.dynunet import VesuviusDynUNet


# -----------------------------------------------------------------------------
# Fixed paths for AutoDL
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "autodl_dynunet_realdata_8epoch.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints_dynunet_realdata_8epoch" / "best_model.pth"
TEST_VOLUME_PATH = PROJECT_ROOT / "data" / "processed" / "test" / "volume.npy"
OUTPUT_NPY_PATH = PROJECT_ROOT / "predictions_dynunet.npy"
SUBMISSION_PATH = PROJECT_ROOT / "submission_dynunet_8epoch.csv"


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
    if not path.is_file():
        raise FileNotFoundError(f"Test volume not found: {path}")

    print(f"Loading test volume: {path}")
    vol = np.load(str(path)).astype(np.float32)
    print(f"  Shape: {vol.shape}")
    print(f"  Range: [{vol.min():.4f}, {vol.max():.4f}]")

    mean = vol.mean()
    std = vol.std()
    vol = (vol - mean) / (std + 1e-8)
    return vol


def sliding_window_inference(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size=(128, 128, 128),
    overlap: float = 0.5,
    batch_size: int = 2,
    device: str = "cuda",
) -> np.ndarray:
    model.eval()
    model.to(device)

    D, H, W = volume.shape
    pd, ph, pw = patch_size

    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    patches = []
    for d in range(0, max(1, D - pd + 1), stride_d):
        for h in range(0, max(1, H - ph + 1), stride_h):
            for w in range(0, max(1, W - pw + 1), stride_w):
                patches.append((d, h, w))

    if not patches:
        raise ValueError("No patches generated for given volume/patch_size.")

    print(f"Total patches: {len(patches)}")

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

            batch_tensor = torch.from_numpy(np.stack(batch_data)).float().unsqueeze(1)
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
    return output


def simple_postprocess(preds: np.ndarray, threshold: float = 0.5, min_size: int = 100) -> np.ndarray:
    from scipy import ndimage

    mask = (preds > threshold).astype(np.uint8)
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    out = np.isin(labeled, keep_labels).astype(np.uint8)
    return out


def save_submission(mask: np.ndarray, path: Path) -> None:
    flat = mask.astype(np.uint8).reshape(-1)
    prediction_str = "".join(str(int(v)) for v in flat)

    df = pd.DataFrame(
        {
            "id": ["vesuvius_test"],
            "prediction": [prediction_str],
        }
    )
    df.to_csv(path, index=False)
    print(f"Submission file generated: {path}")


def main() -> None:
    print("=" * 60)
    print("AutoDL DynUNet One-click Inference")
    print("=" * 60)
    print()

    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(CONFIG_PATH)
    patch_size = tuple(config.get("data", {}).get("patch_size", [128, 128, 128]))
    print(f"Patch size: {patch_size}")

    model = build_model(config, device)
    load_checkpoint(model, CHECKPOINT_PATH, device)

    volume = load_volume(TEST_VOLUME_PATH)

    preds = sliding_window_inference(
        model=model,
        volume=volume,
        patch_size=patch_size,
        overlap=0.5,
        batch_size=2,
        device=str(device),
    )

    print(f"\nInference completed, prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
    np.save(OUTPUT_NPY_PATH, preds.astype(np.float32))
    print(f"Probability map saved to: {OUTPUT_NPY_PATH}")

    mask = simple_postprocess(preds, threshold=0.5, min_size=100)
    save_submission(mask, SUBMISSION_PATH)

    elapsed = time.time() - start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    print("\n" + "=" * 60)
    print("One-click inference completed!")
    print("=" * 60)
    print(f"Total time: {h}h {m}m")


if __name__ == "__main__":
    main()
