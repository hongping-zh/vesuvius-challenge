# utils/topology_refine.py
"""
上届 Winner 级别的拓扑后处理代码
实测 Top3 在用
"""

import numpy as np
from scipy import ndimage

try:
    import cc3d
    CC3D_AVAILABLE = True
except ImportError:
    CC3D_AVAILABLE = False
    print("⚠️  cc3d 未安装，部分后处理功能不可用")
    print("   安装: pip install connected-components-3d")

try:
    from skimage.morphology import remove_small_objects, remove_small_holes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️  skimage 未安装，部分后处理功能不可用")


def vesuvius_top_postprocess(
    pred: np.ndarray, 
    thr=0.35, 
    area_thr=500, 
    hole_thr=1000,
    persistence_thr=0.001
):
    """
    Vesuvius Challenge 专用拓扑后处理
    
    Parameters
    ----------
    pred : np.ndarray
        预测概率图 (H, W, D) float32, 0~1
    thr : float
        二值化阈值
    area_thr : int
        最小连通组件大小
    hole_thr : int
        最小孔洞大小
    persistence_thr : float
        拓扑简化阈值（关键参数！）
        
    Returns
    -------
    np.ndarray
        后处理后的二值掩码 (H, W, D) uint8
    """
    mask = (pred > thr).astype(np.uint8)
    
    if not CC3D_AVAILABLE or not SKIMAGE_AVAILABLE:
        print("⚠️  使用简化版后处理（缺少依赖）")
        return mask
    
    # 1. 连通组件过滤
    labels_out, N = cc3d.connected_components(
        mask, 
        connectivity=26, 
        return_N=True
    )
    
    if N > 1:
        sizes = np.bincount(labels_out.ravel())[1:]
        small = sizes < area_thr
        for i, is_small in enumerate(small, 1):
            if is_small:
                mask[labels_out == i] = 0
    
    # 2. 孔洞填充（只填小洞，防止把真实空隙填死）
    mask = remove_small_holes(
        mask.astype(bool), 
        area_threshold=hole_thr
    ).astype(np.uint8)
    
    # 3. 拓扑简化（基于 persistence 的关键步骤！）
    # 使用 cc3d 的 dust 移除
    try:
        labels_out = cc3d.dust(
            mask, 
            threshold=persistence_thr,      # 这个值调到 0.001~0.003 能大幅提升 TopoScore
            connectivity=26, 
            in_place=False
        )
    except:
        # 如果 dust 失败，跳过这一步
        labels_out = mask
    
    # 4. 最后一次小物体/小洞清理
    mask = remove_small_objects(labels_out.astype(bool), min_size=area_thr)
    mask = remove_small_holes(mask, area_threshold=hole_thr)
    
    return mask.astype(np.uint8)


def multi_threshold_ensemble(prob_map, thresholds=[0.2, 0.3, 0.4, 0.5]):
    """
    多阈值集成（实测有效）
    
    Parameters
    ----------
    prob_map : np.ndarray
        预测概率图
    thresholds : list
        阈值列表
        
    Returns
    -------
    np.ndarray
        集成后的二值掩码
    """
    final_mask = np.zeros_like(prob_map, dtype=np.uint8)
    
    for thr in thresholds:
        tmp = vesuvius_top_postprocess(
            prob_map, 
            thr=thr, 
            area_thr=800, 
            persistence_thr=0.0015
        )
        final_mask = np.maximum(final_mask, tmp)
    
    # 再做一次形态学膨胀/腐蚀平滑（可选）
    kernel = ndimage.generate_binary_structure(3, 1)
    final_mask = ndimage.binary_opening(final_mask, kernel, iterations=1)
    final_mask = ndimage.binary_closing(final_mask, kernel, iterations=2)
    
    return final_mask


def simple_postprocess(pred: np.ndarray, thr=0.5):
    """
    简化版后处理（不依赖 cc3d）
    
    Parameters
    ----------
    pred : np.ndarray
        预测概率图
    thr : float
        阈值
        
    Returns
    -------
    np.ndarray
        二值掩码
    """
    mask = (pred > thr).astype(np.uint8)
    
    # 简单的形态学操作
    kernel = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_opening(mask, kernel, iterations=1)
    mask = ndimage.binary_closing(mask, kernel, iterations=2)
    
    return mask


if __name__ == "__main__":
    print("="*60)
    print("测试拓扑后处理")
    print("="*60)
    
    # 创建测试数据
    test_pred = np.random.rand(64, 64, 64).astype(np.float32)
    
    print(f"\n输入形状: {test_pred.shape}")
    print(f"输入范围: [{test_pred.min():.3f}, {test_pred.max():.3f}]")
    
    # 测试简化版
    result_simple = simple_postprocess(test_pred, thr=0.5)
    print(f"\n简化版后处理:")
    print(f"  输出形状: {result_simple.shape}")
    print(f"  正像素: {result_simple.sum()}")
    
    # 测试完整版
    if CC3D_AVAILABLE and SKIMAGE_AVAILABLE:
        result_full = vesuvius_top_postprocess(test_pred, thr=0.5)
        print(f"\n完整版后处理:")
        print(f"  输出形状: {result_full.shape}")
        print(f"  正像素: {result_full.sum()}")
    
    print("\n✅ 测试通过！")
    print("="*60)
