"""
Vesuvius Challenge 评估指标

实现三大核心指标:
1. SurfaceDice@τ (35%)
2. VOI_score (35%)
3. TopoScore (30%)
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label
import torch


class SurfaceDiceMetric:
    """
    Surface Dice @ τ
    
    表面距离容忍的 Dice 系数
    """
    
    def __init__(self, tau=2.0, spacing=(1.0, 1.0, 1.0)):
        """
        Parameters
        ----------
        tau : float
            距离容忍度（物理单位）
        spacing : tuple
            体素间距 (sz, sy, sx)
        """
        self.tau = tau
        self.spacing = spacing
    
    def extract_surface(self, mask):
        """
        提取表面点
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码
        
        Returns
        -------
        np.ndarray
            表面点坐标 (N, 3)
        """
        # 腐蚀以找到边界
        eroded = ndimage.binary_erosion(mask)
        surface = mask & ~eroded
        
        # 获取表面点坐标
        coords = np.argwhere(surface)
        
        # 应用间距
        coords_physical = coords * np.array(self.spacing)
        
        return coords_physical
    
    def compute(self, pred, target):
        """
        计算 SurfaceDice@τ
        
        Parameters
        ----------
        pred : np.ndarray
            预测掩码
        target : np.ndarray
            真实掩码
        
        Returns
        -------
        float
            SurfaceDice 分数 [0, 1]
        """
        # 边界情况
        pred_empty = not pred.any()
        target_empty = not target.any()
        
        if pred_empty and target_empty:
            return 1.0
        if pred_empty or target_empty:
            return 0.0
        
        # 提取表面
        pred_surface = self.extract_surface(pred)
        target_surface = self.extract_surface(target)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return 0.0
        
        # 计算最近距离
        from scipy.spatial import cKDTree
        
        # Pred → Target
        tree_target = cKDTree(target_surface)
        distances_p2t, _ = tree_target.query(pred_surface)
        matches_p2t = (distances_p2t <= self.tau).sum()
        
        # Target → Pred
        tree_pred = cKDTree(pred_surface)
        distances_t2p, _ = tree_pred.query(target_surface)
        matches_t2p = (distances_t2p <= self.tau).sum()
        
        # 双向平均
        recall = matches_t2p / len(target_surface)
        precision = matches_p2t / len(pred_surface)
        
        # F1 score
        if recall + precision == 0:
            return 0.0
        
        surface_dice = 2 * recall * precision / (recall + precision)
        
        return surface_dice


class VOIMetric:
    """
    Variation of Information (VOI) Score
    
    评估实例分割的分裂和合并
    """
    
    def __init__(self, alpha=0.3, connectivity=26):
        """
        Parameters
        ----------
        alpha : float
            VOI 缩放因子
        connectivity : int
            连通性 (6, 18, 或 26)
        """
        self.alpha = alpha
        
        # 转换连通性为结构元素
        if connectivity == 6:
            self.structure = ndimage.generate_binary_structure(3, 1)
        elif connectivity == 18:
            self.structure = ndimage.generate_binary_structure(3, 2)
        else:  # 26
            self.structure = ndimage.generate_binary_structure(3, 3)
    
    def compute_voi(self, pred_labels, target_labels):
        """
        计算 VOI
        
        Parameters
        ----------
        pred_labels : np.ndarray
            预测的实例标签
        target_labels : np.ndarray
            真实的实例标签
        
        Returns
        -------
        tuple
            (voi_split, voi_merge, voi_total)
        """
        # 只考虑前景
        mask = (pred_labels > 0) | (target_labels > 0)
        pred_labels = pred_labels[mask]
        target_labels = target_labels[mask]
        
        if len(pred_labels) == 0:
            return 0.0, 0.0, 0.0
        
        # 计算联合分布
        pred_ids = np.unique(pred_labels)
        target_ids = np.unique(target_labels)
        
        n = len(pred_labels)
        
        # P(pred, target)
        joint_hist = np.zeros((len(pred_ids), len(target_ids)))
        for i, p_id in enumerate(pred_ids):
            for j, t_id in enumerate(target_ids):
                joint_hist[i, j] = ((pred_labels == p_id) & (target_labels == t_id)).sum()
        
        joint_prob = joint_hist / n
        
        # P(pred), P(target)
        pred_prob = joint_prob.sum(axis=1)
        target_prob = joint_prob.sum(axis=0)
        
        # H(pred | target) - split
        voi_split = 0
        for j in range(len(target_ids)):
            if target_prob[j] > 0:
                for i in range(len(pred_ids)):
                    if joint_prob[i, j] > 0:
                        voi_split -= joint_prob[i, j] * np.log2(joint_prob[i, j] / target_prob[j])
        
        # H(target | pred) - merge
        voi_merge = 0
        for i in range(len(pred_ids)):
            if pred_prob[i] > 0:
                for j in range(len(target_ids)):
                    if joint_prob[i, j] > 0:
                        voi_merge -= joint_prob[i, j] * np.log2(joint_prob[i, j] / pred_prob[i])
        
        voi_total = voi_split + voi_merge
        
        return voi_split, voi_merge, voi_total
    
    def compute(self, pred, target):
        """
        计算 VOI_score
        
        Parameters
        ----------
        pred : np.ndarray
            预测掩码
        target : np.ndarray
            真实掩码
        
        Returns
        -------
        float
            VOI_score [0, 1]
        """
        # 边界情况
        pred_empty = not pred.any()
        target_empty = not target.any()
        
        if pred_empty and target_empty:
            return 1.0
        if pred_empty or target_empty:
            return 0.0
        
        # 连通组件标记
        pred_labels, _ = ndimage.label(pred, structure=self.structure)
        target_labels, _ = ndimage.label(target, structure=self.structure)
        
        # 计算 VOI
        _, _, voi_total = self.compute_voi(pred_labels, target_labels)
        
        # 转换为分数
        voi_score = 1 / (1 + self.alpha * voi_total)
        
        return voi_score


class TopoScoreMetric:
    """
    Topological Score
    
    基于 Betti 数匹配的拓扑特征评估
    """
    
    def __init__(self, weights=(0.34, 0.33, 0.33)):
        """
        Parameters
        ----------
        weights : tuple
            (w0, w1, w2) for k=0,1,2
        """
        self.weights = weights
    
    def compute_betti_numbers(self, mask):
        """
        计算 Betti 数
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码
        
        Returns
        -------
        tuple
            (b0, b1, b2)
        """
        # b0: 连通组件数
        labeled, b0 = ndimage.label(mask)
        
        # b1 和 b2 需要更复杂的计算（持久同调）
        # 这里使用简化版本
        
        # b1: 估计隧道/把手数（使用欧拉特征）
        # χ = b0 - b1 + b2
        # 对于简单情况，假设 b2 = 0
        
        # 计算欧拉特征（简化版）
        # 使用边界信息
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        
        # 粗略估计 b1
        b1 = max(0, int(boundary.sum() / mask.sum() * b0) - b0)
        
        # b2: 空腔数（简化版）
        # 检查内部空洞
        filled = ndimage.binary_fill_holes(mask)
        holes = filled & ~mask
        _, b2 = ndimage.label(holes)
        
        return b0, b1, b2
    
    def compute_topo_f1(self, pred_betti, target_betti):
        """
        计算拓扑 F1 分数
        
        Parameters
        ----------
        pred_betti : int
            预测的 Betti 数
        target_betti : int
            真实的 Betti 数
        
        Returns
        -------
        float
            F1 分数
        """
        if pred_betti == 0 and target_betti == 0:
            return 1.0
        
        # 匹配的特征数（最小值）
        matched = min(pred_betti, target_betti)
        
        # Precision and Recall
        precision = matched / pred_betti if pred_betti > 0 else 0
        recall = matched / target_betti if target_betti > 0 else 0
        
        # F1
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def compute(self, pred, target):
        """
        计算 TopoScore
        
        Parameters
        ----------
        pred : np.ndarray
            预测掩码
        target : np.ndarray
            真实掩码
        
        Returns
        -------
        float
            TopoScore [0, 1]
        """
        # 边界情况
        pred_empty = not pred.any()
        target_empty = not target.any()
        
        if pred_empty and target_empty:
            return 1.0
        if pred_empty or target_empty:
            return 0.0
        
        # 计算 Betti 数
        pred_betti = self.compute_betti_numbers(pred)
        target_betti = self.compute_betti_numbers(target)
        
        # 计算每个维度的 F1
        f1_scores = []
        active_weights = []
        
        for k in range(3):
            # 如果两者都没有特征，跳过这个维度
            if pred_betti[k] == 0 and target_betti[k] == 0:
                continue
            
            f1 = self.compute_topo_f1(pred_betti[k], target_betti[k])
            f1_scores.append(f1)
            active_weights.append(self.weights[k])
        
        # 加权平均
        if len(f1_scores) == 0:
            return 1.0
        
        active_weights = np.array(active_weights)
        active_weights = active_weights / active_weights.sum()  # 归一化
        
        topo_score = np.dot(f1_scores, active_weights)
        
        return topo_score


class VesuviusMetrics:
    """
    Vesuvius Challenge 完整评估指标
    """
    
    def __init__(self, tau=2.0, spacing=(1.0, 1.0, 1.0)):
        """
        Parameters
        ----------
        tau : float
            SurfaceDice 的距离容忍度
        spacing : tuple
            体素间距
        """
        self.surface_dice = SurfaceDiceMetric(tau=tau, spacing=spacing)
        self.voi = VOIMetric()
        self.topo_score = TopoScoreMetric()
    
    def compute(self, pred, target):
        """
        计算完整的 Vesuvius 分数
        
        Parameters
        ----------
        pred : np.ndarray or torch.Tensor
            预测掩码
        target : np.ndarray or torch.Tensor
            真实掩码
        
        Returns
        -------
        dict
            包含所有指标的字典
        """
        # 转换为 numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # 二值化
        pred_binary = (pred > 0.5).astype(bool)
        target_binary = (target > 0.5).astype(bool)
        
        # 计算各个指标
        surface_dice_score = self.surface_dice.compute(pred_binary, target_binary)
        voi_score = self.voi.compute(pred_binary, target_binary)
        topo_score_value = self.topo_score.compute(pred_binary, target_binary)
        
        # 最终分数
        final_score = (
            0.30 * topo_score_value +
            0.35 * surface_dice_score +
            0.35 * voi_score
        )
        
        return {
            'surface_dice': surface_dice_score,
            'voi_score': voi_score,
            'topo_score': topo_score_value,
            'final_score': final_score
        }


def test_metrics():
    """测试评估指标"""
    print("测试 Vesuvius Challenge 评估指标...")
    
    # 创建测试数据
    pred = np.random.rand(64, 64, 64) > 0.5
    target = np.random.rand(64, 64, 64) > 0.5
    
    # 测试各个指标
    print("\n1. SurfaceDice@τ")
    metric = SurfaceDiceMetric(tau=2.0)
    score = metric.compute(pred, target)
    print(f"   Score: {score:.4f}")
    
    print("\n2. VOI_score")
    metric = VOIMetric()
    score = metric.compute(pred, target)
    print(f"   Score: {score:.4f}")
    
    print("\n3. TopoScore")
    metric = TopoScoreMetric()
    score = metric.compute(pred, target)
    print(f"   Score: {score:.4f}")
    
    print("\n4. Complete Vesuvius Metrics")
    metrics = VesuviusMetrics()
    scores = metrics.compute(pred, target)
    print(f"   SurfaceDice: {scores['surface_dice']:.4f}")
    print(f"   VOI_score: {scores['voi_score']:.4f}")
    print(f"   TopoScore: {scores['topo_score']:.4f}")
    print(f"   Final Score: {scores['final_score']:.4f}")
    
    print("\n✓ 所有评估指标测试通过")


if __name__ == "__main__":
    test_metrics()
