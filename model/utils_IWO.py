# model/utils_IWO.py
"""
IWO 相关的基础层实现：IWOConv1x1

用途：
- 用于包装原始 MBRConv* 里的 conv_out(1×1) 卷积，实现
    W_final = W_pre + W_learn
- 不改变 Stage-1 训练用的 MBRConv 定义，避免与 best.pkl 不兼容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IWOConv1x1(nn.Module):
    """
    1×1 卷积的 IWO 包装版本：

    - weight_pre: 从预训练 conv_out.weight 拷贝来的固定权重（冻结）
    - weight_learn: 增量权重（可训练）
    - bias: 从原始 conv_out.bias 继承，可训练

    forward 时使用：
        y = Conv2d(x, weight_pre + weight_learn, bias)
    """

    def __init__(self, conv: nn.Conv2d, init_std: float = 0.0):
        super(IWOConv1x1, self).__init__()

        assert isinstance(conv, nn.Conv2d), "IWOConv1x1 只能包装 nn.Conv2d"

        # 保存卷积的结构超参（用于 to_plain_conv）
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        # 冻结的预训练权重 W_pre（直接从 baseline conv_out 拷贝）
        self.weight_pre = nn.Parameter(conv.weight.data.clone(), requires_grad=False)

        # 可训练的增量核 W_learn
        self.weight_learn = nn.Parameter(torch.zeros_like(conv.weight))
        if init_std is not None and init_std > 0:
            nn.init.normal_(self.weight_learn, mean=0.0, std=float(init_std))

        # bias 直接拷贝（保持可训练）
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        """
        兼容原先代码中对 conv_out.weight 的访问：
        MBRConv.slim() 里如果拿 self.conv_out.weight，会直接看到 W_pre + W_learn。
        """
        return self.weight_pre + self.weight_learn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_pre + self.weight_learn
        return F.conv2d(
            x,
            w,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def to_plain_conv(self) -> nn.Conv2d:
        """
        返回一个等价的普通 Conv2d，权重为 W_pre + W_learn，用于 fuse_iwo()。

        ★ 关键修复：
        - 新建的 Conv2d 默认在 CPU，我们需要把它移动到与 weight_pre 相同的 device，
          否则后续 slim() 中会出现 CPU/GPU 混用导致 matmul 报错。
        """
        device = self.weight_pre.device  # 原模块所在设备 (通常是 cuda:0)

        conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
        )

        with torch.no_grad():
            conv.weight.copy_(self.weight_pre + self.weight_learn)
            if self.bias is not None:
                conv.bias.copy_(self.bias)

        # ★ 移到与原权重相同的设备
        conv = conv.to(device)
        return conv
