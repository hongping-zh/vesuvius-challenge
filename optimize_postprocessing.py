"""\
简单的后处理参数网格搜索脚本（验证集用）

思路：
- 假设你已经在验证集上跑过模型推理，并把每个样本的：
  - 概率图保存为 .npy (float32, 0~1)
  - GT 掩码保存为 .npy (0/1)
- 本脚本会读取这些文件，对一组 (thr, min_component_size, min_hole_size, persistence_thr, thresholds_multi)
  组合做网格搜索，使用 VesuviusMetrics 的 final_score 作为目标，选出最佳参数。

使用前你需要：
1. 在 data/val_probs/ 和 data/val_masks/ 下，放置同名的 .npy 文件，例如：
   - data/val_probs/val_01.npy
   - data/val_masks/val_01.npy
2. 根据数据实际情况，调整 PARAM_GRID 中的搜索范围（尽量先小范围试）

运行：
    python optimize_postprocessing.py
"""

import itertools
from pathlib import Path

import numpy as np

from utils.topology_refine import vesuvius_top_postprocess, multi_threshold_ensemble
from utils.vesuvius_metrics import VesuviusMetrics


# =====================
# 基本配置（可按需修改）
# =====================
VAL_PROB_DIR = Path("data/val_probs")   # 验证集概率图目录
VAL_MASK_DIR = Path("data/val_masks")   # 验证集 GT 掩码目录

# 需要参与评估的文件名前缀（不含扩展名），None = 自动扫描所有 .npy
FILE_LIST = None

# Vesuvius 指标配置
VESUVIUS_TAU = 2.0
VESUVIUS_SPACING = (1.0, 1.0, 1.0)

# 参数搜索空间（建议先用很小的网格验证脚本流程，再逐步加密）
PARAM_GRID = {
    "thr": [0.30, 0.35, 0.40],
    "area_thr": [600, 800, 1000],        # min_component_size
    "hole_thr": [800, 1000, 1200],      # min_hole_size
    "persistence_thr": [0.0010, 0.0015, 0.0020],
    # multi-threshold 用于 multi_threshold_ensemble；None 表示只用单一 thr
    "multi_thresholds": [None, [0.3, 0.4, 0.5]],
}


def iter_param_combinations(param_grid):
    """将 PARAM_GRID 转成 (param_dict) 迭代器"""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def load_val_pairs():
    """加载验证集 (prob, mask) 成对文件列表"""
    if not VAL_PROB_DIR.is_dir():
        raise FileNotFoundError(f"概率图目录不存在: {VAL_PROB_DIR}")
    if not VAL_MASK_DIR.is_dir():
        raise FileNotFoundError(f"GT 掩码目录不存在: {VAL_MASK_DIR}")

    if FILE_LIST is not None:
        names = FILE_LIST
    else:
        names = [p.stem for p in VAL_PROB_DIR.glob("*.npy")]

    pairs = []
    for name in sorted(names):
        prob_path = VAL_PROB_DIR / f"{name}.npy"
        mask_path = VAL_MASK_DIR / f"{name}.npy"
        if not prob_path.is_file() or not mask_path.is_file():
            print(f"⚠️  跳过 {name}: 概率或掩码文件不存在")
            continue
        pairs.append((name, prob_path, mask_path))

    if not pairs:
        raise RuntimeError("未找到任何 (prob, mask) 成对文件，请检查目录和文件名")

    print(f"共找到 {len(pairs)} 个验证样本用于后处理调参")
    return pairs


def evaluate_params(pairs, params, metrics):
    """给定一组后处理参数，在所有验证样本上计算平均 final_score"""
    scores = []

    for name, prob_path, mask_path in pairs:
        prob = np.load(prob_path).astype(np.float32)
        gt = np.load(mask_path).astype(np.uint8)

        # 统一为 (D, H, W) 或 (H, W, D) 之一；根据你保存的格式调整
        # 这里假设 prob/gt 已经是 (D, H, W) 或 (H, W, D) 且一致
        # 如果你的维度是 (1, D, H, W)，请在这里 squeeze 一下

        if params["multi_thresholds"] is None:
            # 单阈值 + 拓扑后处理
            mask_pred = vesuvius_top_postprocess(
                pred=prob,
                thr=params["thr"],
                area_thr=params["area_thr"],
                hole_thr=params["hole_thr"],
                persistence_thr=params["persistence_thr"],
            )
        else:
            # 多阈值集成
            mask_pred = multi_threshold_ensemble(
                prob_map=prob,
                thresholds=params["multi_thresholds"],
            )

        # 计算 Vesuvius 指标
        # 这里假设 mask_pred / gt 是 0/1，形状匹配
        s = metrics.compute(mask_pred.astype(np.uint8), gt.astype(np.uint8))
        scores.append(s["final_score"])

    return float(np.mean(scores)), float(np.std(scores))


def main():
    print("=" * 60)
    print("Vesuvius 后处理参数网格搜索")
    print("=" * 60)

    pairs = load_val_pairs()

    metrics = VesuviusMetrics(
        tau=VESUVIUS_TAU,
        spacing=VESUVIUS_SPACING,
    )

    best_score = -1.0
    best_std = 0.0
    best_params = None

    all_results = []

    for idx, params in enumerate(iter_param_combinations(PARAM_GRID), 1):
        print("\n" + "-" * 60)
        print(f"组合 {idx}: {params}")

        mean_score, std_score = evaluate_params(pairs, params, metrics)
        all_results.append((params, mean_score, std_score))

        print(f"  -> Final Score: {mean_score:.4f} ± {std_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_std = std_score
            best_params = params

    print("\n" + "=" * 60)
    print("搜索完成")
    print("=" * 60)

    print("\n最佳参数组合:")
    print(best_params)
    print(f"最佳平均 Final Score: {best_score:.4f} ± {best_std:.4f}")

    # 按得分排序，输出前若干个组合，方便对比
    all_results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 组合:")
    for rank, (params, mean_score, std_score) in enumerate(all_results[:5], 1):
        print(f"#{rank}: {mean_score:.4f} ± {std_score:.4f} -> {params}")

    print("\n提示：可以把最佳参数抄到 configs/autodl_dynunet_optimized.yaml 的 postprocessing 段落中。")
    print("例如：")
    print("postprocessing:")
    print(f"  min_component_size: {int(best_params['area_thr'])}")
    print(f"  min_hole_size: {int(best_params['hole_thr'])}")
    print(f"  persistence_threshold: {best_params['persistence_thr']}")
    if best_params["multi_thresholds"] is None:
        print("  multi_threshold: false")
        print(f"  thresholds: [{best_params['thr']}]  # 单阈值")
    else:
        print("  multi_threshold: true")
        print(f"  thresholds: {best_params['multi_thresholds']}")


if __name__ == "__main__":
    main()
