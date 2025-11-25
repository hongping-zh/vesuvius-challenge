"""
拓扑感知损失函数

针对 Vesuvius Challenge 的特殊评估指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import numpy as np


class SurfaceDistanceLoss(nn.Module):
    """
    表面距离损失
    
    关注预测和真实表面之间的距离
    对应评估指标: SurfaceDice@τ
    """
    
    def __init__(self, tau=2.0):
        super().__init__()
        self.tau = tau
    
    def compute_surface_distances(self, mask):
        """
        计算到表面的距离
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码 (D, H, W)
        
        Returns
        -------
        np.ndarray
            距离图
        """
        # 确保是布尔类型
        mask = mask.astype(bool)
        
        # 提取表面（边界）
        eroded = ndimage.binary_erosion(mask)
        surface = mask & ~eroded
        
        # 计算距离变换
        distance = ndimage.distance_transform_edt(~surface)
        
        return distance
    
    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : torch.Tensor
            预测 (B, C, D, H, W)
        target : torch.Tensor
            目标 (B, C, D, H, W)
        """
        pred = torch.sigmoid(pred)
        pred_binary = (pred > 0.5).cpu().numpy()
        target_binary = target.cpu().numpy()
        
        batch_size = pred.shape[0]
        total_loss = 0
        
        for b in range(batch_size):
            pred_mask = pred_binary[b, 0]
            target_mask = target_binary[b, 0]
            
            # 计算表面距离
            pred_dist = self.compute_surface_distances(pred_mask)
            target_dist = self.compute_surface_distances(target_mask)
            
            # 转换为 torch tensor
            pred_dist_t = torch.from_numpy(pred_dist).float().to(pred.device)
            target_dist_t = torch.from_numpy(target_dist).float().to(pred.device)
            
            # 计算损失（距离应该小）
            loss = F.smooth_l1_loss(pred_dist_t, target_dist_t)
            total_loss += loss
        
        return total_loss / batch_size


class CenterlineDiceLoss(nn.Module):
    """
    中心线 Dice 损失 (clDice)
    
    保持拓扑连通性
    参考: https://arxiv.org/abs/2003.07311
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def soft_skeletonize(self, x, iterations=10):
        """
        软骨架化（可微分）
        
        Parameters
        ----------
        x : torch.Tensor
            输入 (B, C, D, H, W)
        iterations : int
            迭代次数
        """
        # 使用形态学操作的软版本
        for _ in range(iterations):
            # 软腐蚀
            min_pool = F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
            x = torch.min(x, -min_pool)
        
        return x
    
    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : torch.Tensor
            预测 (B, C, D, H, W)
        target : torch.Tensor
            目标 (B, C, D, H, W)
        """
        pred = torch.sigmoid(pred)
        
        # 计算骨架
        pred_skel = self.soft_skeletonize(pred)
        target_skel = self.soft_skeletonize(target)
        
        # 计算 tprec (骨架被预测覆盖)
        tprec_num = (pred_skel * target).sum()
        tprec_den = target_skel.sum()
        tprec = (tprec_num + self.smooth) / (tprec_den + self.smooth)
        
        # 计算 tsens (预测骨架被目标覆盖)
        tsens_num = (target_skel * pred).sum()
        tsens_den = pred_skel.sum()
        tsens = (tsens_num + self.smooth) / (tsens_den + self.smooth)
        
        # clDice
        cl_dice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
        
        return 1 - cl_dice


class TopologyPreservingLoss(nn.Module):
    """
    拓扑保持损失
    
    惩罚拓扑错误（额外的连通组件、孔洞等）
    """
    
    def __init__(self, weight_components=1.0, weight_holes=1.0):
        super().__init__()
        self.weight_components = weight_components
        self.weight_holes = weight_holes
    
    def count_components(self, mask):
        """
        计算连通组件数（Betti-0）
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码
        
        Returns
        -------
        int
            连通组件数
        """
        labeled, num_components = ndimage.label(mask)
        return num_components
    
    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : torch.Tensor
            预测 (B, C, D, H, W)
        target : torch.Tensor
            目标 (B, C, D, H, W)
        """
        pred = torch.sigmoid(pred)
        pred_binary = (pred > 0.5).cpu().numpy()
        target_binary = target.cpu().numpy()
        
        batch_size = pred.shape[0]
        total_loss = 0
        
        for b in range(batch_size):
            pred_mask = pred_binary[b, 0]
            target_mask = target_binary[b, 0]
            
            # 计算连通组件数差异
            pred_components = self.count_components(pred_mask)
            target_components = self.count_components(target_mask)
            
            component_diff = abs(pred_components - target_components)
            
            # 归一化损失
            loss = self.weight_components * component_diff / max(target_components, 1)
            total_loss += loss
        
        return torch.tensor(total_loss / batch_size, device=pred.device)


class VesuviusCompositeLoss(nn.Module):
    """
    Vesuvius Challenge 组合损失
    
    结合多个损失以匹配评估指标
    """
    
    def __init__(
        self,
        dice_weight=0.3,
        bce_weight=0.2,
        surface_weight=0.25,
        centerline_weight=0.15,
        topology_weight=0.1
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.surface_weight = surface_weight
        self.centerline_weight = centerline_weight
        self.topology_weight = topology_weight
        
        # 基础损失
        try:
            from .losses import DiceLoss
        except ImportError:
            # 如果相对导入失败，使用绝对导入
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from losses import DiceLoss
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 拓扑感知损失
        self.surface_loss = SurfaceDistanceLoss(tau=2.0)
        self.centerline_loss = CenterlineDiceLoss()
        self.topology_loss = TopologyPreservingLoss()
    
    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : torch.Tensor
            预测 (B, C, D, H, W)
        target : torch.Tensor
            目标 (B, C, D, H, W)
        """
        # 基础损失
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        # 拓扑感知损失
        surface = self.surface_loss(pred, target)
        centerline = self.centerline_loss(pred, target)
        topology = self.topology_loss(pred, target)
        
        # 组合
        total_loss = (
            self.dice_weight * dice +
            self.bce_weight * bce +
            self.surface_weight * surface +
            self.centerline_weight * centerline +
            self.topology_weight * topology
        )
        
        return total_loss, {
            'dice': dice.item(),
            'bce': bce.item(),
            'surface': surface.item(),
            'centerline': centerline.item(),
            'topology': topology.item()
        }


def test_losses():
    """测试损失函数"""
    print("测试拓扑感知损失函数...")
    
    # 创建测试数据
    batch_size = 2
    pred = torch.randn(batch_size, 1, 32, 32, 32)
    target = torch.randint(0, 2, (batch_size, 1, 32, 32, 32)).float()
    
    # 测试各个损失
    print("\n1. Surface Distance Loss")
    surface_loss = SurfaceDistanceLoss()
    loss = surface_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n2. Centerline Dice Loss")
    cl_loss = CenterlineDiceLoss()
    loss = cl_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n3. Topology Preserving Loss")
    topo_loss = TopologyPreservingLoss()
    loss = topo_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n4. Composite Loss")
    composite_loss = VesuviusCompositeLoss()
    loss, components = composite_loss(pred, target)
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Components: {components}")
    
    print("\n✓ 所有损失函数测试通过")


if __name__ == "__main__":
    test_losses()
