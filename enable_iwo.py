# enable_iwo.py
"""
IWO (Incremental Weight Optimization) 启用与融合工具

设计思路：
- 不修改原始 MBRConv 实现（保持与 Stage-1 完全一致，方便加载 22.7dB 的 best.pkl）
- 在 Stage-2 中，仅对 MBRConv 的 conv_out(1×1) 包装成 IWOConv1x1：
    W_final = W_pre (冻结) + W_learn (可训练)
- fuse_iwo() 时再把 IWOConv1x1 融合回普通 Conv2d，供 slim()/ONNX/INT8 使用
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

from model.utils_IWO import IWOConv1x1


def enable_iwo_for_model(model: nn.Module, config: Dict[str, Any] = None):
    """
    扫描模型，找到 MBRConv* 中的 conv_out (1×1 卷积)，
    将其替换为 IWOConv1x1(W_pre + W_learn)。

    Args:
        model: 已训练好的 baseline 模型（MobileIELLENet）
        config: IWO 配置字典，包含：
            - init_std: W_learn 的初始化标准差（默认 0，即全零初始化）
            - target_layers: 要启用 IWO 的层名列表（默认为所有 MBRConv 的 conv_out）

    Returns:
        model: 启用了 IWO 的模型
        iwo_params: IWO 可训练参数列表（用于单独 optimizer param group）
    """
    if config is None:
        config = {}

    init_std = float(config.get('init_std', 0.0))
    target_layers = config.get('target_layers', None)  # None 表示所有 MBRConv
    if target_layers is not None:
        target_layers = set(target_layers)

    iwo_params: List[nn.Parameter] = []
    iwo_count = 0

    print("[IWO] 开始启用 IWO...")

    for name, module in model.named_modules():
        module_type = type(module).__name__

        # 仅处理 MBRConv1 / MBRConv3 / MBRConv5
        if module_type not in ("MBRConv1", "MBRConv3", "MBRConv5"):
            continue

        if target_layers is not None and name not in target_layers:
            continue

        # 必须有 conv_out 子模块
        conv_out = getattr(module, "conv_out", None)
        if conv_out is None:
            continue

        # 已经是 IWOConv1x1 的情况（防止重复包装）
        if isinstance(conv_out, IWOConv1x1):
            iwo_params.append(conv_out.weight_learn)
            iwo_count += 1
            print(f"[IWO] {name}.conv_out 已经是 IWOConv1x1，跳过包装")
            continue

        if not isinstance(conv_out, nn.Conv2d):
            # 不是 Conv2d，无法包装 IWO，直接跳过
            continue

        # 用当前 conv_out 权重构造 IWOConv1x1
        iwo_conv = IWOConv1x1(conv_out, init_std=init_std)

        # 替换模块中的 conv_out
        setattr(module, "conv_out", iwo_conv)

        # IWO 可训练参数（增量核）
        iwo_params.append(iwo_conv.weight_learn)
        iwo_count += 1

        print(f"[IWO] {name}.conv_out → IWOConv1x1 (in={iwo_conv.in_channels}, "
              f"out={iwo_conv.out_channels})")

    print(f"[IWO] 完成！共启用 {iwo_count} 个 IWO 层")
    print(f"[IWO] IWO 可训练参数总数: {sum(p.numel() for p in iwo_params)}")

    return model, iwo_params


def fuse_iwo(model: nn.Module) -> nn.Module:
    """
    将模型中的 IWOConv1x1(W_pre + W_learn) 融合回单一 Conv2d，
    方便导出 ONNX / INT8 / slim。

    Args:
        model: 启用了 IWO 的模型

    Returns:
        model: 权重已融合的模型（conv_out 恢复为普通 Conv2d，且已含 W_pre+W_learn）
    """
    print("[IWO FUSE] 开始融合 IWO 权重...")
    fuse_count = 0

    for name, module in model.named_modules():
        conv_out = getattr(module, "conv_out", None)
        if isinstance(conv_out, IWOConv1x1):
            # 使用 IWOConv1x1 的 to_plain_conv() 得到融合后的 Conv2d
            plain_conv = conv_out.to_plain_conv()
            setattr(module, "conv_out", plain_conv)
            fuse_count += 1
            print(f"[IWO FUSE] {name}.conv_out → nn.Conv2d (已融合 W_pre + W_learn)")

    print(f"[IWO FUSE] 完成！共融合 {fuse_count} 个 IWO 层")
    return model


def get_iwo_param_groups(
    model: nn.Module,
    base_lr: float,
    iwo_lr_scale: float = 1.0,
    gate_lr_scale: float = 0.3,
):
    """
    构建优化器参数组：
    1. backbone 参数（非 IWO、非 gate）
    2. IWO 参数（IWOConv1x1.weight_learn）
    3. gate 参数（FST 中 .u. / .v. 等，可选）

    Args:
        model: 模型
        base_lr: 基础学习率
        iwo_lr_scale: IWO 学习率倍数（相对 base_lr）
        gate_lr_scale: Gate 学习率倍数（相对 base_lr）

    Returns:
        param_groups: 优化器参数组列表
    """
    backbone_params = []
    iwo_params = []
    gate_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # IWO：只认 IWOConv1x1 的 weight_learn
        if "conv_out.weight_learn" in name:
            iwo_params.append(param)
        # Gate：FST 中的 u/v（如果你现在的 FST 不是 u/v，就不会命中，问题不大）
        elif ".u." in name or ".v." in name:
            gate_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": base_lr, "name": "backbone"},
        {"params": iwo_params, "lr": base_lr * iwo_lr_scale, "name": "iwo"},
        {"params": gate_params, "lr": base_lr * gate_lr_scale, "name": "gate"},
    ]

    print(f"[PARAM GROUPS] backbone: {len(backbone_params)} tensors")
    print(f"[PARAM GROUPS] IWO: {len(iwo_params)} tensors (lr_scale={iwo_lr_scale})")
    print(f"[PARAM GROUPS] gate: {len(gate_params)} tensors (lr_scale={gate_lr_scale})")

    return param_groups


def save_iwo_checkpoint(model, save_path, epoch, val_psnr, optimizer=None):
    """
    保存 IWO 训练 checkpoint（包含 IWOConv1x1 的状态）
    """
    checkpoint = {
        "epoch": epoch,
        "val_psnr": val_psnr,
        "state_dict": model.state_dict(),
        "iwo_enabled": True,
    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    print(f"[IWO SAVE] Checkpoint saved: {save_path}")


def load_iwo_checkpoint(model, checkpoint_path, device="cuda", load_optimizer=None):
    """
    加载 IWO checkpoint（一般不在本阶段用到，留作扩展）

    Args:
        model: 已经启用 IWO 的模型
        checkpoint_path: checkpoint 路径
        device: 设备字符串
        load_optimizer: 若传入 optimizer，将一并恢复其 state_dict

    Returns:
        epoch: 已训练的 epoch
        val_psnr: 对应的验证 PSNR
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    epoch = checkpoint.get("epoch", 0)
    val_psnr = checkpoint.get("val_psnr", 0.0)

    if load_optimizer is not None and "optimizer" in checkpoint:
        load_optimizer.load_state_dict(checkpoint["optimizer"])
        print("[IWO LOAD] Optimizer state restored")

    print(f"[IWO LOAD] Checkpoint loaded: epoch={epoch}, val_psnr={val_psnr:.3f}")
    return epoch, val_psnr


if __name__ == "__main__":
    print("IWO 工具模块 - 使用示例：")
    print("1. enable_iwo_for_model(model, config) - 启用 IWO")
    print("2. fuse_iwo(model) - 融合 IWO 权重")
    print("3. get_iwo_param_groups(model, base_lr) - 获取优化器参数组")
    print("4. save_iwo_checkpoint(...) - 保存 IWO checkpoint")
    print("5. load_iwo_checkpoint(...) - 加载 IWO checkpoint")
