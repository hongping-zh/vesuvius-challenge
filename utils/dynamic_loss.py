# utils/dynamic_loss.py
"""
动态 Loss 权重调度
前 20-50 epoch 只学定位，后 30 epoch 再开拓扑约束
"""

import numpy as np


class DynamicLossScheduler:
    """
    动态损失权重调度器
    
    策略：
    - 前期（Epoch 1-20）：只学习基础分割（Dice + BCE/Focal）
    - 后期（Epoch 21+）：逐渐加入拓扑约束（Surface + Topology）
    """
    
    def __init__(
        self,
        total_epochs=50,
        warmup_epochs=20,
        strategy='two_stage'
    ):
        """
        Parameters
        ----------
        total_epochs : int
            总训练轮数
        warmup_epochs : int
            预热轮数（只学基础分割）
        strategy : str
            调度策略
            - 'two_stage': 两阶段（推荐）
            - 'linear': 线性增长
            - 'cosine': 余弦增长
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy
        
        print(f"📊 动态 Loss 调度:")
        print(f"   总轮数: {total_epochs}")
        print(f"   预热轮数: {warmup_epochs}")
        print(f"   策略: {strategy}")
    
    def get_weights(self, epoch):
        """
        获取当前 epoch 的损失权重
        
        Parameters
        ----------
        epoch : int
            当前 epoch (从 0 开始)
            
        Returns
        -------
        weights : dict
            损失权重字典
        """
        if self.strategy == 'two_stage':
            return self._two_stage_weights(epoch)
        elif self.strategy == 'linear':
            return self._linear_weights(epoch)
        elif self.strategy == 'cosine':
            return self._cosine_weights(epoch)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _two_stage_weights(self, epoch):
        """
        两阶段策略（推荐）
        
        阶段 1 (Epoch 0-warmup): 只学基础分割
        阶段 2 (Epoch warmup+): 加入拓扑约束
        """
        if epoch < self.warmup_epochs:
            # 阶段 1: 只学定位
            return {
                'dice': 0.5,
                'bce': 0.5,
                'focal': 0.0,      # 可选：用 Focal 替代 BCE
                'surface': 0.0,
                'centerline': 0.0,
                'topology': 0.0
            }
        else:
            # 阶段 2: 加入拓扑约束
            return {
                'dice': 0.4,
                'bce': 0.2,
                'focal': 0.0,
                'surface': 0.2,
                'centerline': 0.0,  # 墨迹检测不需要
                'topology': 0.2
            }
    
    def _linear_weights(self, epoch):
        """线性增长策略"""
        if epoch < self.warmup_epochs:
            # 预热阶段
            return {
                'dice': 0.5,
                'bce': 0.5,
                'focal': 0.0,
                'surface': 0.0,
                'centerline': 0.0,
                'topology': 0.0
            }
        else:
            # 线性增长
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            
            surface_weight = 0.2 * progress
            topology_weight = 0.2 * progress
            
            return {
                'dice': 0.5 - 0.1 * progress,
                'bce': 0.5 - 0.1 * progress,
                'focal': 0.0,
                'surface': surface_weight,
                'centerline': 0.0,
                'topology': topology_weight
            }
    
    def _cosine_weights(self, epoch):
        """余弦增长策略"""
        if epoch < self.warmup_epochs:
            # 预热阶段
            return {
                'dice': 0.5,
                'bce': 0.5,
                'focal': 0.0,
                'surface': 0.0,
                'centerline': 0.0,
                'topology': 0.0
            }
        else:
            # 余弦增长
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            
            # 余弦调度：0 -> 1
            cosine_progress = (1 - np.cos(progress * np.pi)) / 2
            
            surface_weight = 0.2 * cosine_progress
            topology_weight = 0.2 * cosine_progress
            
            return {
                'dice': 0.5 - 0.1 * cosine_progress,
                'bce': 0.5 - 0.1 * cosine_progress,
                'focal': 0.0,
                'surface': surface_weight,
                'centerline': 0.0,
                'topology': topology_weight
            }
    
    def print_schedule(self):
        """打印完整调度表"""
        print("\n" + "="*60)
        print("Loss 权重调度表")
        print("="*60)
        
        # 打印关键 epoch
        key_epochs = [0, self.warmup_epochs//2, self.warmup_epochs-1, 
                      self.warmup_epochs, (self.warmup_epochs + self.total_epochs)//2, 
                      self.total_epochs-1]
        
        print(f"\n{'Epoch':<8} {'Dice':<8} {'BCE':<8} {'Surface':<10} {'Topology':<10}")
        print("-"*60)
        
        for epoch in key_epochs:
            if epoch >= self.total_epochs:
                continue
            weights = self.get_weights(epoch)
            print(f"{epoch:<8} {weights['dice']:<8.3f} {weights['bce']:<8.3f} "
                  f"{weights['surface']:<10.3f} {weights['topology']:<10.3f}")
        
        print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("测试动态 Loss 调度")
    print("="*60)
    
    # 测试两阶段策略
    scheduler = DynamicLossScheduler(
        total_epochs=50,
        warmup_epochs=20,
        strategy='two_stage'
    )
    
    scheduler.print_schedule()
    
    # 测试线性策略
    print("\n")
    scheduler_linear = DynamicLossScheduler(
        total_epochs=50,
        warmup_epochs=20,
        strategy='linear'
    )
    
    scheduler_linear.print_schedule()
    
    print("\n✅ 测试通过！")
    print("="*60)
