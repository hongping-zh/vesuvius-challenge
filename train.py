"""
训练脚本

优化用于 AutoDL RTX 5090
- 混合精度训练
- 梯度累积
- 检查点保存
- WandB 监控
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse

from models.unet3d import UNet3D, UNet3DLite
from utils.dataset import VesuviusDataset
from utils.losses import DiceBCELoss
from utils.metrics import dice_coefficient

# 拓扑感知损失和指标
from utils.topology_losses import VesuviusCompositeLoss
from utils.vesuvius_metrics import VesuviusMetrics
from utils.postprocessing import TopologyAwarePostprocessor


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps=4, use_composite_loss=False):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    loss_components = {'dice': 0, 'bce': 0, 'surface': 0, 'centerline': 0, 'topology': 0}
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # 混合精度训练
        with autocast():
            outputs = model(images)
            
            # 处理 Deep Supervision（多输出）
            if isinstance(outputs, list):
                # DynUNet 的 deep supervision 输出
                main_output = outputs[0]
                if use_composite_loss:
                    # 主输出的损失
                    loss, components = criterion(main_output, masks)
                    # 辅助输出的损失（权重递减）
                    for idx, aux_output in enumerate(outputs[1:], 1):
                        aux_loss, _ = criterion(aux_output, masks)
                        loss += aux_loss * (0.5 ** idx)  # 权重递减
                    # 累积损失组件
                    for key in components:
                        loss_components[key] += components[key]
                else:
                    # 主输出的损失
                    loss = criterion(main_output, masks)
                    # 辅助输出的损失
                    for idx, aux_output in enumerate(outputs[1:], 1):
                        loss += criterion(aux_output, masks) * (0.5 ** idx)
                # 用主输出计算指标
                outputs = main_output
            else:
                # 单输出模型
                if use_composite_loss:
                    loss, components = criterion(outputs, masks)
                    # 累积损失组件
                    for key in components:
                        loss_components[key] += components[key]
                else:
                    loss = criterion(outputs, masks)
            
            loss = loss / accumulation_steps  # 梯度累积
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 计算指标
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
        
        total_loss += loss.item() * accumulation_steps
        total_dice += dice.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'dice': f'{dice.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    # 平均损失组件
    if use_composite_loss:
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        return avg_loss, avg_dice, loss_components
    
    return avg_loss, avg_dice, None


def validate(model, dataloader, criterion, device, vesuvius_metrics=None, postprocessor=None):
    """验证"""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    # Vesuvius 指标
    vesuvius_scores = {'surface_dice': 0, 'voi_score': 0, 'topo_score': 0, 'final_score': 0}
    vesuvius_count = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            if isinstance(criterion, VesuviusCompositeLoss):
                loss, _ = criterion(outputs, masks)
            else:
                loss = criterion(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            # 计算 Vesuvius 指标
            if vesuvius_metrics is not None:
                pred = torch.sigmoid(outputs)
                if postprocessor is not None:
                    # 后处理（在 CPU 上）
                    pred_np = (pred > 0.5).cpu().numpy()
                    for b in range(pred_np.shape[0]):
                        pred_np[b, 0] = postprocessor.process(pred_np[b, 0])
                    pred = torch.from_numpy(pred_np).float()
                
                # 计算指标
                for b in range(pred.shape[0]):
                    scores = vesuvius_metrics.compute(pred[b, 0], masks[b, 0])
                    for key in vesuvius_scores:
                        vesuvius_scores[key] += scores[key]
                    vesuvius_count += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    # 平均 Vesuvius 指标
    if vesuvius_count > 0:
        for key in vesuvius_scores:
            vesuvius_scores[key] /= vesuvius_count
    
    return avg_loss, avg_dice, vesuvius_scores if vesuvius_count > 0 else None


def train(config):
    """主训练函数"""
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建模型
    model_type = config['model']['type']
    
    if model_type == 'dynunet':
        # DynUNet 模型
        from models.dynunet import VesuviusDynUNet
        model = VesuviusDynUNet(
            in_channels=config['model']['in_channels'],
            base_num_features=config['model'].get('base_num_features', 64),
            num_classes=config['model']['out_channels'],
            deep_supervision=config['model'].get('deep_supervision', True)
        )
    elif model_type == 'unet3d':
        model = UNet3D(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels']
        )
    else:
        model = UNet3DLite(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels']
        )
    
    model = model.to(device)

    # 可选：加载预训练权重
    pretrained_ckpt = config['model'].get('pretrained_checkpoint')
    if pretrained_ckpt:
        ckpt_path = Path(pretrained_ckpt)
        if ckpt_path.is_file():
            print(f"加载预训练权重: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            # 兼容直接 state_dict 或 包含在字典中的情况
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            elif isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"⚠️  预训练权重缺少 {len(missing)} 个键（已忽略）")
            if unexpected:
                print(f"⚠️  预训练权重包含 {len(unexpected)} 个未使用键（已忽略）")
        else:
            print(f"⚠️  未找到预训练权重文件: {ckpt_path}")
    
    # 数据加载
    dataset_type = config['data'].get('dataset_type', 'standard')
    
    if dataset_type == 'ink_aware':
        # 使用 Ink-only Sampling
        from utils.ink_sampling import InkAwareVesuviusDataset
        from utils.multi_channel import MultiChannelVesuviusDataset
        
        print("使用 Ink-only Sampling 数据集")
        
        # 创建基础数据集
        train_base = InkAwareVesuviusDataset(
            data_dir=config['data']['train_dir'],
            patch_size=config['data']['patch_size'],
            positive_ratio=config['data'].get('positive_ratio', 0.7),
            min_ink_pixels=config['data'].get('min_ink_pixels', 100),
            num_samples_per_epoch=config['data'].get('num_samples_per_epoch', 500),
            augment=True
        )
        
        val_base = InkAwareVesuviusDataset(
            data_dir=config['data']['val_dir'],
            patch_size=config['data']['patch_size'],
            positive_ratio=config['data'].get('positive_ratio', 0.7),
            min_ink_pixels=config['data'].get('min_ink_pixels', 100),
            num_samples_per_epoch=config['data'].get('num_samples_per_epoch', 100),
            augment=False
        )
        
        # 如果使用多通道
        if 'channels' in config['data']:
            print(f"使用多通道输入: {config['data']['channels']}")
            train_dataset = MultiChannelVesuviusDataset(train_base, config['data']['channels'])
            val_dataset = MultiChannelVesuviusDataset(val_base, config['data']['channels'])
        else:
            train_dataset = train_base
            val_dataset = val_base
    else:
        # 标准数据集
        train_dataset = VesuviusDataset(
            data_dir=config['data']['train_dir'],
            patch_size=config['data']['patch_size'],
            augment=True
        )
        
        val_dataset = VesuviusDataset(
            data_dir=config['data']['val_dir'],
            patch_size=config['data']['patch_size'],
            augment=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # 损失函数和优化器
    # 根据配置选择损失函数
    loss_type = config.get('loss', {}).get('type', 'dice_bce')
    
    # 动态 Loss 调度器
    use_dynamic_loss = config['training'].get('use_dynamic_loss', False)
    loss_scheduler = None
    
    if use_dynamic_loss:
        from utils.dynamic_loss import DynamicLossScheduler
        loss_schedule_strategy = config['training'].get('loss_schedule', 'two_stage')
        loss_scheduler = DynamicLossScheduler(
            total_epochs=config['training']['epochs'],
            warmup_epochs=config['training'].get('warmup_epochs', 20),
            strategy=loss_schedule_strategy
        )
        print(f"使用动态 Loss 权重调度: {loss_schedule_strategy}")
    
    if loss_type == 'vesuvius_composite':
        print("使用 Vesuvius 组合损失函数")
        criterion = VesuviusCompositeLoss(
            dice_weight=config['loss'].get('dice_weight', 0.3),
            bce_weight=config['loss'].get('bce_weight', 0.2),
            surface_weight=config['loss'].get('surface_weight', 0.25),
            centerline_weight=config['loss'].get('centerline_weight', 0.15),
            topology_weight=config['loss'].get('topology_weight', 0.1)
        )
        use_composite_loss = True
    else:
        print("使用标准 DiceBCE 损失函数")
        criterion = DiceBCELoss()
        use_composite_loss = False
    
    # Vesuvius 评估指标
    use_vesuvius_metrics = config.get('evaluation', {}).get('use_vesuvius_metrics', False)
    vesuvius_metrics = None
    postprocessor = None
    
    if use_vesuvius_metrics:
        print("使用 Vesuvius 评估指标")
        vesuvius_metrics = VesuviusMetrics(
            tau=config['evaluation'].get('surface_dice_tau', 2.0),
            spacing=tuple(config['evaluation'].get('spacing', [1.0, 1.0, 1.0]))
        )
        
        # 后处理
        if config.get('postprocessing', {}).get('enabled', False):
            print("启用拓扑感知后处理")
            postprocessor = TopologyAwarePostprocessor(
                min_component_size=config['postprocessing'].get('min_component_size', 100),
                min_hole_size=config['postprocessing'].get('min_hole_size', 50),
                bridge_threshold=config['postprocessing'].get('bridge_threshold', 0.1)
            )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # 混合精度
    scaler = GradScaler()
    
    # WandB
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project'],
            config=config
        )
        wandb.watch(model)
    
    # 训练循环
    best_dice = 0
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 60)
        
        # 更新动态 Loss 权重
        if loss_scheduler is not None and use_composite_loss:
            weights = loss_scheduler.get_weights(epoch)
            criterion.dice_weight = weights['dice']
            criterion.bce_weight = weights['bce']
            criterion.surface_weight = weights['surface']
            criterion.centerline_weight = weights['centerline']
            criterion.topology_weight = weights['topology']
            print(f"📊 Loss 权重: Dice={weights['dice']:.2f}, BCE={weights['bce']:.2f}, "
                  f"Surface={weights['surface']:.2f}, Topology={weights['topology']:.2f}")
        
        # 训练
        train_loss, train_dice, loss_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=config['training']['accumulation_steps'],
            use_composite_loss=use_composite_loss
        )
        
        # 验证
        val_loss, val_dice, vesuvius_scores = validate(
            model, val_loader, criterion, device,
            vesuvius_metrics=vesuvius_metrics,
            postprocessor=postprocessor
        )
        
        # 学习率调度
        scheduler.step()
        
        # 打印结果
        print(f"\nTrain Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        if loss_components:
            print(f"  Loss Components: Dice={loss_components['dice']:.4f}, BCE={loss_components['bce']:.4f}, "
                  f"Surface={loss_components['surface']:.4f}, Centerline={loss_components['centerline']:.4f}, "
                  f"Topology={loss_components['topology']:.4f}")
        
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        if vesuvius_scores:
            print(f"  Vesuvius Metrics: SurfaceDice={vesuvius_scores['surface_dice']:.4f}, "
                  f"VOI={vesuvius_scores['voi_score']:.4f}, Topo={vesuvius_scores['topo_score']:.4f}, "
                  f"Final={vesuvius_scores['final_score']:.4f}")
        
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # WandB 记录
        if config['logging']['use_wandb']:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            # 添加损失组件
            if loss_components:
                for key, value in loss_components.items():
                    log_dict[f'train_{key}_loss'] = value
            
            # 添加 Vesuvius 指标
            if vesuvius_scores:
                for key, value in vesuvius_scores.items():
                    log_dict[f'val_{key}'] = value
            
            wandb.log(log_dict)
        
        # 保存检查点（使用 Vesuvius Final Score 如果可用）
        current_score = vesuvius_scores['final_score'] if vesuvius_scores else val_dice
        
        if current_score > best_dice:
            best_dice = current_score
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'config': config
            }, checkpoint_path)
            print(f"✅ 保存最佳模型 (Dice: {best_dice:.4f})")
        
        # 定期保存
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_dice,
                'config': config
            }, checkpoint_path)
    
    print("\n" + "=" * 60)
    print(f"训练完成！最佳 Dice: {best_dice:.4f}")
    print("=" * 60)
    
    if config['logging']['use_wandb']:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='训练 Vesuvius Challenge 模型')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                        help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 开始训练
    train(config)


if __name__ == "__main__":
    main()
