"""
评估指标
"""

import torch
import numpy as np


def dice_coefficient(pred, target, threshold=0.5, smooth=1.0):
    """
    计算 Dice 系数
    
    Parameters
    ----------
    pred : torch.Tensor
        预测值 (B, C, D, H, W)
    target : torch.Tensor
        目标值 (B, C, D, H, W)
    threshold : float
        二值化阈值
    smooth : float
        平滑项
    
    Returns
    -------
    float
        Dice 系数
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """
    计算 IoU (Intersection over Union)
    
    Parameters
    ----------
    pred : torch.Tensor
        预测值
    target : torch.Tensor
        目标值
    threshold : float
        二值化阈值
    smooth : float
        平滑项
    
    Returns
    -------
    float
        IoU 分数
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def precision_recall(pred, target, threshold=0.5):
    """
    计算精确率和召回率
    
    Parameters
    ----------
    pred : torch.Tensor
        预测值
    target : torch.Tensor
        目标值
    threshold : float
        二值化阈值
    
    Returns
    -------
    tuple
        (precision, recall)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    TP = ((pred == 1) & (target == 1)).sum().float()
    FP = ((pred == 1) & (target == 0)).sum().float()
    FN = ((pred == 0) & (target == 1)).sum().float()
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    
    return precision, recall


def f1_score(pred, target, threshold=0.5):
    """
    计算 F1 分数
    
    Parameters
    ----------
    pred : torch.Tensor
        预测值
    target : torch.Tensor
        目标值
    threshold : float
        二值化阈值
    
    Returns
    -------
    float
        F1 分数
    """
    precision, recall = precision_recall(pred, target, threshold)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1


class MetricTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置"""
        self.dice_scores = []
        self.iou_scores = []
        self.losses = []
    
    def update(self, pred, target, loss=None):
        """更新指标"""
        dice = dice_coefficient(pred, target)
        iou = iou_score(pred, target)
        
        self.dice_scores.append(dice.item())
        self.iou_scores.append(iou.item())
        
        if loss is not None:
            self.losses.append(loss.item())
    
    def get_average(self):
        """获取平均值"""
        return {
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0,
            'loss': np.mean(self.losses) if self.losses else 0
        }
