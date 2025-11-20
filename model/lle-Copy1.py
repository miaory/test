# import torch.nn as nn
# import torch

# from .utils import (
#     MBRConv5,
#     MBRConv3,
#     MBRConv1,
#     DropBlock,
#     FST,
#     FSTS,
#     SCA,        # 新增：引入线性通道注意
# )

# class MobileIELLENet(nn.Module):
#     def __init__(self, channels, rep_scale=4):
#         super(MobileIELLENet, self).__init__()
#         self.channels = channels

#         self.head = FST(
#             nn.Sequential(
#                 MBRConv5(3, channels, rep_scale=rep_scale),
#                 nn.PRELU(channels) if hasattr(nn, "PRELU") else nn.PReLU(channels),
#                 MBRConv3(channels, channels, rep_scale=rep_scale)
#             ),
#             channels
#         )
#         self.body = FST(
#             MBRConv3(channels, channels, rep_scale=rep_scale),
#             channels
#         )

#         # 原 HDPA/att1 → SCA（线性、无激活；极轻）
#         self.att = SCA(channels)

#         # Tail 保持你的 MBRConv3（可在 slim 时折叠）
#         self.tail = MBRConv3(channels, 3, rep_scale=rep_scale)
#         self.tail_warm = MBRConv3(channels, 3, rep_scale=rep_scale)
#         self.drop = DropBlock(3)
        
#     def forward(self, x):
#         x0 = self.head(x)
#         x1 = self.body(x0)
#         x2 = self.att(x1)
#         return self.tail(x2)

#     def forward_warm(self, x):
#         x = self.drop(x)
#         x = self.head(x)
#         x = self.body(x)
#         return self.tail(x), self.tail_warm(x)

#     def slim(self):
#         """
#         将当前（MBR + 门控FST）模型压缩为轻量版（MobileIELLENetS）：
#         - MBRConv* → 单一 Conv2d（通过各自 slim() 折叠）
#         - FST（门控版：含 u/v 逐通道1×1）→ FSTS（同门控结构）参数直拷
#         - PReLU 参数直拷
#         """
#         net_slim = MobileIELLENetS(self.channels)
#         weight_slim = net_slim.state_dict()
#         for name, mod in self.named_modules():
#             if isinstance(mod, (MBRConv3, MBRConv5, MBRConv1)):
#                 if f'{name}.weight' in weight_slim:
#                     w, b = mod.slim()
#                     weight_slim[f'{name}.weight'] = w
#                     weight_slim[f'{name}.bias'] = b
#             elif isinstance(mod, FST):
#                 # FST 是逐通道门控：包含 u、v 两个 1×1_dw 卷积与 bias
#                 if f'{name}.u.weight' in weight_slim:
#                     weight_slim[f'{name}.u.weight'] = mod.u.weight
#                     weight_slim[f'{name}.u.bias']   = mod.u.bias
#                 if f'{name}.v.weight' in weight_slim:
#                     weight_slim[f'{name}.v.weight'] = mod.v.weight
#                     weight_slim[f'{name}.v.bias']   = mod.v.bias
#                 if f'{name}.bias' in weight_slim:
#                     weight_slim[f'{name}.bias'] = mod.bias
#             elif isinstance(mod, nn.PReLU):
#                 if f'{name}.weight' in weight_slim:
#                     weight_slim[f'{name}.weight'] = mod.weight
#         net_slim.load_state_dict(weight_slim, strict=False)
#         return net_slim


# class MobileIELLENetS(nn.Module):
#     def __init__(self, channels):
#         super(MobileIELLENetS, self).__init__()
#         # 轻量版：MBR 由单层 Conv2d 近似；FSTS 采用与 FST 相同的“逐通道门控”实现
#         self.head = FSTS(
#             nn.Sequential(
#                 nn.Conv2d(3, channels, 5, 1, 2),
#                 nn.PReLU(channels),
#                 nn.Conv2d(channels, channels, 3, 1, 1)
#             ),
#             channels
#         )
#         self.body = FSTS(
#             nn.Conv2d(channels, channels, 3, 1, 1),
#             channels
#         )

#         # 部署侧也用相同的 SCA（线性、无激活）
#         self.att = SCA(channels)

#         self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        
#     def forward(self, x):
#         x0 = self.head(x)
#         x1 = self.body(x0)
#         x2 = self.att(x1)
#         return self.tail(x2)
import torch.nn as nn
import torch

from .utils import (
    MBRConv5,
    MBRConv3,
    MBRConv1,
    DropBlock,
    FST,
    FSTS,
    SCA,        # 新增：线性通道注意
)

class MobileIELLENet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(MobileIELLENet, self).__init__()
        self.channels = channels

        self.head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )
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
        将当前（MBR + 门控FST）模型压缩为轻量版（MobileIELLENetS）：
        - MBRConv* → 单一 Conv2d（通过各自 slim() 折叠）
        - FST（门控版：含 u/v 逐通道1×1）→ FSTS（同门控结构）参数直拷
        - PReLU 参数直拷
        """
        net_slim = MobileIELLENetS(self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, (MBRConv3, MBRConv5, MBRConv1)):
                if f'{name}.weight' in weight_slim:
                    w, b = mod.slim()
                    weight_slim[f'{name}.weight'] = w
                    weight_slim[f'{name}.bias'] = b
            elif isinstance(mod, FST):
                # FST 是逐通道门控：包含 u、v 两个 1×1_dw 卷积与 bias
                if f'{name}.u.weight' in weight_slim:
                    weight_slim[f'{name}.u.weight'] = mod.u.weight
                    weight_slim[f'{name}.u.bias']   = mod.u.bias
                if f'{name}.v.weight' in weight_slim:
                    weight_slim[f'{name}.v.weight'] = mod.v.weight
                    weight_slim[f'{name}.v.bias']   = mod.v.bias
                if f'{name}.bias' in weight_slim:
                    weight_slim[f'{name}.bias'] = mod.bias
            elif isinstance(mod, nn.PReLU):
                if f'{name}.weight' in weight_slim:
                    weight_slim[f'{name}.weight'] = mod.weight
        net_slim.load_state_dict(weight_slim, strict=False)
        return net_slim


class MobileIELLENetS(nn.Module):
    def __init__(self, channels):
        super(MobileIELLENetS, self).__init__()
        # 轻量版：MBR 由单层 Conv2d 近似；FSTS 采用与 FST 相同的“逐通道门控”实现
        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            channels
        )
        self.body = FSTS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            channels
        )

        # 部署侧也使用同样的 SCA（无激活）
        self.att = SCA(channels)

        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        return self.tail(x2)
