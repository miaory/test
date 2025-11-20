# model/lle.py
import torch
import torch.nn as nn

from .utils import (
    MBRConv5,
    MBRConv3,
    MBRConv1,
    DropBlock,
    FST,
    FSTS,
    SCA,        # 线性通道注意
)


class MobileIELLENet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(MobileIELLENet, self).__init__()
        self.channels = channels

        # 与训练时一致：head 的 block1 是 Sequential
        self.head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )
        # ★ 与训练时一致，body 的 block1 不是 Sequential（避免 .0 前缀）
        self.body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels
        )

        # 原 HDPA/att1 → SCA（线性、恒等初始化，稳定）
        self.att = SCA(channels)

        # Tail 保持 MBRConv3（导出可折叠）
        self.tail = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.tail_warm = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.drop = DropBlock(3)

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        return self.tail(x2)

    def forward_warm(self, x):
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail_warm(x)

    def slim(self):
        """
        将当前（MBR + 门控 FST）模型压缩为轻量版（MobileIELLENetS）：
        - MBRConv* → 单一 Conv2d（通过各自 slim() 折叠）
        - FST（门控版：含 u/v 逐通道 1×1）→ FSTS（同门控结构）参数直拷
        - PReLU/SCA 参数直拷（★ 必须同步 SCA，否则数值会漂）
        """
        net_slim = MobileIELLENetS(self.channels)
        slim_sd = net_slim.state_dict()

        for name, mod in self.named_modules():
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"

            # 1) MBRConvK：折叠为单层 Conv2d
            if isinstance(mod, (MBRConv3, MBRConv5, MBRConv1)):
                if w_key in slim_sd:
                    w, b = mod.slim()
                    slim_sd[w_key] = w
                    if b_key in slim_sd:
                        slim_sd[b_key] = b

            # 2) FST → FSTS：拷贝门控 u/v 及 bias
            elif isinstance(mod, FST):
                if f"{name}.u.weight" in slim_sd:
                    slim_sd[f"{name}.u.weight"] = mod.u.weight.data.clone()
                    slim_sd[f"{name}.u.bias"] = mod.u.bias.data.clone()
                if f"{name}.v.weight" in slim_sd:
                    slim_sd[f"{name}.v.weight"] = mod.v.weight.data.clone()
                    slim_sd[f"{name}.v.bias"] = mod.v.bias.data.clone()
                if f"{name}.bias" in slim_sd:
                    slim_sd[f"{name}.bias"] = mod.bias.data.clone()

            # 3) PReLU：参数直拷
            elif isinstance(mod, nn.PReLU):
                if w_key in slim_sd:
                    slim_sd[w_key] = mod.weight.data.clone()

            # 4) SCA：同步 fc 权重与 gate_scale
            elif isinstance(mod, SCA):
                if hasattr(net_slim, "att") and isinstance(net_slim.att, SCA):
                    net_slim.att.fc.weight.data.copy_(mod.fc.weight.data)
                    net_slim.att.fc.bias.data.copy_(mod.fc.bias.data)
                    net_slim.att.gate_scale = float(mod.gate_scale)

        missing, unexpected = net_slim.load_state_dict(slim_sd, strict=False)
        try:
            miss_n, unexp_n = len(missing), len(unexpected)
        except Exception:
            miss_n = unexp_n = -1
        if miss_n > 0 or unexp_n > 0:
            print(f"[SLIM WARN] state_dict not fully matched: missing={miss_n}, unexpected={unexp_n}")

        return net_slim


class MobileIELLENetS(nn.Module):
    def __init__(self, channels):
        super(MobileIELLENetS, self).__init__()
        # 轻量版：MBR 由单层 Conv2d 近似；FSTS 采用与 FST 相同的逐通道门控实现
        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2, bias=True),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            ),
            channels,
        )
        self.body = FSTS(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            channels,
        )

        # 部署侧也使用同样的 SCA（无激活）
        self.att = SCA(channels)

        self.tail = nn.Conv2d(channels, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        return self.tail(x2)
