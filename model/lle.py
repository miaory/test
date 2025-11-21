import torch
import torch.nn as nn

from .utils import (
    MBRConv5,
    MBRConv3,
    MBRConv1,
    DropBlock,
    FST,
    FSTS,
    SCA,        # 使用 utils 中修改后的 Contrast-Aware SCA
)

class MobileIELLENet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(MobileIELLENet, self).__init__()
        self.channels = channels

        # Head: MBRConv5 + PReLU + MBRConv3 (Wrapped in FST)
        self.head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )
        
        # Body: MBRConv3 (Wrapped in FST)
        self.body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels
        )

        # Attention: Contrast-Aware SCA
        self.att = SCA(channels)

        # Tail: MBRConv3 (负责重建)
        self.tail = MBRConv3(channels, 3, rep_scale=rep_scale)
        
        # Warmup 分支 (仅训练用)
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
        
        # ★★★ 关键修复：必须包含 SCA，否则输入到 tail 的特征分布不一致 ★★★
        x = self.att(x)
        
        return self.tail(x), self.tail_warm(x)

    def slim(self):
        """
        将训练模型压缩为推理模型 (MobileIELLENetS)
        """
        net_slim = MobileIELLENetS(self.channels)
        weight_slim = net_slim.state_dict()
        
        for name, mod in self.named_modules():
            # 1) MBRConv 折叠
            if isinstance(mod, (MBRConv3, MBRConv5, MBRConv1)):
                if f'{name}.weight' in weight_slim:
                    w, b = mod.slim()
                    weight_slim[f'{name}.weight'] = w
                    weight_slim[f'{name}.bias'] = b
            
            # 2) FST 参数拷贝 (u, v, bias)
            elif isinstance(mod, FST):
                if f'{name}.u.weight' in weight_slim:
                    weight_slim[f'{name}.u.weight'] = mod.u.weight.data.clone()
                    weight_slim[f'{name}.u.bias']   = mod.u.bias.data.clone()
                if f'{name}.v.weight' in weight_slim:
                    weight_slim[f'{name}.v.weight'] = mod.v.weight.data.clone()
                    weight_slim[f'{name}.v.bias']   = mod.v.bias.data.clone()
                if f'{name}.bias' in weight_slim:
                    weight_slim[f'{name}.bias'] = mod.bias.data.clone()
            
            # 3) PReLU 参数拷贝
            elif isinstance(mod, nn.PReLU):
                if f'{name}.weight' in weight_slim:
                    weight_slim[f'{name}.weight'] = mod.weight.data.clone()
            
            # 4) SCA 参数拷贝 (适配新名称 linear)
            elif isinstance(mod, SCA):
                if hasattr(net_slim, "att") and isinstance(net_slim.att, SCA):
                    # 拷贝权重 (fc -> linear)
                    weight_slim[f'{name}.linear.weight'] = mod.linear.weight.data.clone()
                    weight_slim[f'{name}.linear.bias']   = mod.linear.bias.data.clone()
                    
                    # 拷贝缩放因子
                    if hasattr(net_slim.att, 'gate_scale'):
                        net_slim.att.gate_scale = float(mod.gate_scale)

        net_slim.load_state_dict(weight_slim, strict=False)
        return net_slim


class MobileIELLENetS(nn.Module):
    def __init__(self, channels):
        super(MobileIELLENetS, self).__init__()
        # 部署版结构
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

        self.att = SCA(channels)

        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        return self.tail(x2)
