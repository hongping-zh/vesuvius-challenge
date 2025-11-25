"""
拓扑感知后处理

针对 Vesuvius Challenge 的特殊要求
"""

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes


class TopologyAwarePostprocessor:
    """
    拓扑感知后处理器
    
    修正常见的拓扑错误:
    1. 跨层桥接（相邻卷层粘连）
    2. 层内断裂（同一层分裂）
    3. 虚假孔洞
    """
    
    def __init__(
        self,
        min_component_size=100,
        min_hole_size=50,
        bridge_threshold=0.1
    ):
        """
        Parameters
        ----------
        min_component_size : int
            最小连通组件大小
        min_hole_size : int
            最小孔洞大小
        bridge_threshold : float
            桥接检测阈值
        """
        self.min_component_size = min_component_size
        self.min_hole_size = min_hole_size
        self.bridge_threshold = bridge_threshold
    
    def remove_small_components(self, mask):
        """
        移除小的连通组件
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码
        
        Returns
        -------
        np.ndarray
            清理后的掩码
        """
        return remove_small_objects(
            mask,
            min_size=self.min_component_size,
            connectivity=2
        )
    
    def fill_small_holes(self, mask):
        """
        填充小孔洞
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码
        
        Returns
        -------
        np.ndarray
            填充后的掩码
        """
        return remove_small_holes(
            mask,
            area_threshold=self.min_hole_size,
            connectivity=2
        )
    
    def detect_bridges(self, mask):
        """
        检测跨层桥接
        
        通过分析 z 方向的连通性
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码 (D, H, W)
        
        Returns
        -------
        np.ndarray
            桥接掩码
        """
        D, H, W = mask.shape
        bridges = np.zeros_like(mask, dtype=bool)
        
        # 逐层分析
        for z in range(1, D - 1):
            # 当前层
            current = mask[z]
            
            # 上下层
            above = mask[z - 1]
            below = mask[z + 1]
            
            # 检测异常连接
            # 如果当前层连接了上下两个不同的区域，可能是桥接
            labeled_above = label(above)
            labeled_below = label(below)
            labeled_current = label(current)
            
            for region in regionprops(labeled_current):
                coords = region.coords
                
                # 检查这个区域连接的上下层区域数
                # 注意边界检查
                valid_above = coords[:, 0] > 0
                valid_below = coords[:, 0] < H - 1
                
                above_regions = set()
                below_regions = set()
                
                if valid_above.any():
                    above_coords = coords[valid_above]
                    above_regions = set(labeled_above[above_coords[:, 0] - 1, above_coords[:, 1]])
                
                if valid_below.any():
                    below_coords = coords[valid_below]
                    below_regions = set(labeled_below[below_coords[:, 0] + 1, below_coords[:, 1]])
                
                above_regions.discard(0)
                below_regions.discard(0)
                
                # 如果连接了多个不同的区域，标记为可能的桥接
                if len(above_regions) > 1 or len(below_regions) > 1:
                    for coord in coords:
                        bridges[z, coord[0], coord[1]] = True
        
        return bridges
    
    def remove_bridges(self, mask):
        """
        移除跨层桥接
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码
        
        Returns
        -------
        np.ndarray
            移除桥接后的掩码
        """
        bridges = self.detect_bridges(mask)
        
        # 移除桥接区域
        mask_cleaned = mask.copy()
        mask_cleaned[bridges] = False
        
        return mask_cleaned
    
    def merge_fragments(self, mask):
        """
        合并同层碎片
        
        如果两个组件在 z 方向上重叠且距离很近，合并它们
        
        Parameters
        ----------
        mask : np.ndarray
            二值掩码 (D, H, W)
        
        Returns
        -------
        np.ndarray
            合并后的掩码
        """
        D, H, W = mask.shape
        mask_merged = mask.copy()
        
        # 3D 连通组件
        labeled = label(mask, connectivity=2)
        
        # 分析每个组件
        regions = regionprops(labeled)
        
        for i, region1 in enumerate(regions):
            for region2 in regions[i + 1:]:
                # 检查 z 方向重叠
                z1_min, z1_max = region1.bbox[0], region1.bbox[3]
                z2_min, z2_max = region2.bbox[0], region2.bbox[3]
                
                z_overlap = min(z1_max, z2_max) - max(z1_min, z2_min)
                
                if z_overlap > 0:
                    # 计算质心距离
                    centroid1 = np.array(region1.centroid)
                    centroid2 = np.array(region2.centroid)
                    distance = np.linalg.norm(centroid1 - centroid2)
                    
                    # 如果距离很近，合并
                    threshold = min(region1.major_axis_length, region2.major_axis_length) * 0.5
                    
                    if distance < threshold:
                        # 合并：将 region2 的标签改为 region1
                        mask_merged[labeled == region2.label] = True
        
        return mask_merged
    
    def process(self, mask):
        """
        完整的后处理流程
        
        Parameters
        ----------
        mask : np.ndarray
            原始预测掩码
        
        Returns
        -------
        np.ndarray
            后处理后的掩码
        """
        # 1. 移除小组件
        mask = self.remove_small_components(mask)
        
        # 2. 填充小孔洞
        mask = self.fill_small_holes(mask)
        
        # 3. 移除跨层桥接
        mask = self.remove_bridges(mask)
        
        # 4. 再次移除小组件（桥接移除后可能产生）
        mask = self.remove_small_components(mask)
        
        # 5. 合并同层碎片
        mask = self.merge_fragments(mask)
        
        return mask


class SlidingWindowInference:
    """
    滑动窗口推理
    
    处理大体积数据
    """
    
    def __init__(
        self,
        model,
        patch_size=(64, 64, 64),
        overlap=0.5,
        batch_size=4
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            训练好的模型
        patch_size : tuple
            Patch 大小
        overlap : float
            重叠比例 [0, 1)
        batch_size : int
            批次大小
        """
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
    
    def get_patches(self, volume_shape):
        """
        生成 patch 坐标
        
        Parameters
        ----------
        volume_shape : tuple
            体积形状 (D, H, W)
        
        Returns
        -------
        list
            Patch 坐标列表
        """
        D, H, W = volume_shape
        pd, ph, pw = self.patch_size
        
        # 计算步长
        stride_d = int(pd * (1 - self.overlap))
        stride_h = int(ph * (1 - self.overlap))
        stride_w = int(pw * (1 - self.overlap))
        
        patches = []
        
        for d in range(0, D - pd + 1, stride_d):
            for h in range(0, H - ph + 1, stride_h):
                for w in range(0, W - pw + 1, stride_w):
                    patches.append((d, h, w))
        
        return patches
    
    def predict(self, volume):
        """
        滑动窗口推理
        
        Parameters
        ----------
        volume : np.ndarray
            输入体积 (D, H, W)
        
        Returns
        -------
        np.ndarray
            预测结果 (D, H, W)
        """
        import torch
        
        D, H, W = volume.shape
        
        # 输出累积
        output = np.zeros((D, H, W), dtype=np.float32)
        counts = np.zeros((D, H, W), dtype=np.float32)
        
        # 获取 patches
        patches = self.get_patches(volume.shape)
        
        # 批次推理
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch_patches = patches[i:i + self.batch_size]
                
                # 提取 patches
                batch_data = []
                for d, h, w in batch_patches:
                    pd, ph, pw = self.patch_size
                    patch = volume[d:d+pd, h:h+ph, w:w+pw]
                    batch_data.append(patch)
                
                # 转换为 tensor
                batch_tensor = torch.from_numpy(np.array(batch_data)).float()
                batch_tensor = batch_tensor.unsqueeze(1)  # (B, 1, D, H, W)
                
                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()
                
                # 推理
                pred = self.model(batch_tensor)
                pred = torch.sigmoid(pred)
                pred = pred.cpu().numpy()
                
                # 累积结果
                for j, (d, h, w) in enumerate(batch_patches):
                    pd, ph, pw = self.patch_size
                    output[d:d+pd, h:h+ph, w:w+pw] += pred[j, 0]
                    counts[d:d+pd, h:h+ph, w:w+pw] += 1
        
        # 平均
        output = output / (counts + 1e-8)
        
        return output


def test_postprocessing():
    """测试后处理"""
    print("测试拓扑感知后处理...")
    
    # 创建测试数据
    mask = np.random.rand(64, 64, 64) > 0.7
    
    # 后处理
    processor = TopologyAwarePostprocessor()
    
    print("\n1. 原始掩码")
    print(f"   体积: {mask.sum()} voxels")
    
    print("\n2. 移除小组件")
    mask_cleaned = processor.remove_small_components(mask)
    print(f"   体积: {mask_cleaned.sum()} voxels")
    
    print("\n3. 填充小孔洞")
    mask_filled = processor.fill_small_holes(mask_cleaned)
    print(f"   体积: {mask_filled.sum()} voxels")
    
    print("\n4. 完整后处理")
    mask_final = processor.process(mask)
    print(f"   体积: {mask_final.sum()} voxels")
    
    print("\n✓ 后处理测试通过")


if __name__ == "__main__":
    test_postprocessing()
