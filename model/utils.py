import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# MBRConv5: 5x5 多分支重参数化卷积 (保持原版逻辑)
# ==============================================================================
class MBRConv5(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv2_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)

    def forward(self, inp):
        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)
        x = torch.cat(
            [x1, x2, x3, x4, x5,
             self.conv_bn(x1), self.conv1_bn(x2), self.conv2_bn(x3),
             self.conv_crossh_bn(x4), self.conv_crossv_bn(x5)], 1)
        return self.conv_out(x)

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = nn.functional.pad(self.conv1.weight, (2, 2, 2, 2))
        conv2_weight = nn.functional.pad(self.conv2.weight, (1, 1, 1, 1))
        conv_crossv_weight = nn.functional.pad(self.conv_crossv.weight, (1, 1, 2, 2))
        conv_crossh_weight = nn.functional.pad(self.conv_crossh.weight, (2, 2, 1, 1))

        def fuse_bn(conv_w, conv_b, bn):
            k = 1 / (bn.running_var + bn.eps).sqrt()
            b = -bn.running_mean * k
            w = conv_w * k.view(-1, 1, 1, 1) * bn.weight.view(-1, 1, 1, 1)
            bias = (conv_b * k + b) * bn.weight + bn.bias
            return w, bias

        # Fuse branches
        w_conv, b_conv = fuse_bn(self.conv.weight, self.conv.bias, self.conv_bn[0])
        w_c1, b_c1 = fuse_bn(conv1_weight, self.conv1.bias, self.conv1_bn[0])
        w_c2, b_c2 = fuse_bn(conv2_weight, self.conv2.bias, self.conv2_bn[0])
        w_cv, b_cv = fuse_bn(conv_crossv_weight, self.conv_crossv.bias, self.conv_crossv_bn[0])
        w_ch, b_ch = fuse_bn(conv_crossh_weight, self.conv_crossh.bias, self.conv_crossh_bn[0])

        # Raw weights for concatenation (paper logic implies features are cat then conv_out)
        # Note: The original code concats raw features AND bn features. 
        # Assuming standard RepVGG style fusion for simplicity or adhering to original implementation:
        # The original code cats: [x1...x5, bn(x1)...bn(x5)]. 
        # This implies we need weights for the raw convs AND the fused BN convs.
        
        # Raw conv weights (padded)
        raw_w_conv = self.conv.weight
        raw_w_c1 = conv1_weight
        raw_w_c2 = conv2_weight
        raw_w_ch = conv_crossh_weight
        raw_w_cv = conv_crossv_weight
        
        # Concatenate all 10 branches weights
        weight = torch.cat([
            raw_w_conv, raw_w_c1, raw_w_c2, raw_w_ch, raw_w_cv,
            w_conv, w_c1, w_c2, w_ch, w_cv
        ], dim=0)
        
        # Concatenate all 10 branches biases
        # Raw convs have bias, BN fused convs have bias
        bias_cat = torch.cat([
            self.conv.bias, self.conv1.bias, self.conv2.bias, self.conv_crossh.bias, self.conv_crossv.bias,
            b_conv, b_c1, b_c2, b_ch, b_cv
        ], dim=0)

        # Merge with conv_out
        weight_compress = self.conv_out.weight.squeeze()
        # Reshape weight for matmul: (Out_All, In, K, K) -> (Out_All, In*K*K)
        # But simpler: 1x1 conv_out is just a linear combination of input channels.
        # Final Weight = conv_out.weight * concatenated_weights
        # Shape: (Out_Final, In_All) * (In_All, In_Raw, K, K) -> (Out_Final, In_Raw, K, K)
        
        weight = torch.matmul(weight_compress, weight.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        bias = torch.matmul(weight_compress, bias_cat)
        
        if self.conv_out.bias is not None:
            bias = bias + self.conv_out.bias
            
        return weight, bias


# ==============================================================================
# MBRConv3: 3x3 多分支重参数化卷积
# ==============================================================================
class MBRConv3(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

    def forward(self, inp):    
        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)
        x = torch.cat(
        [    x0,x1,x2,x3,
             self.conv_bn(x0), self.conv1_bn(x1),
             self.conv_crossh_bn(x2), self.conv_crossv_bn(x3)], 1)    
        return self.conv_out(x)

    def slim(self):
        conv_weight = self.conv.weight
        conv1_weight = nn.functional.pad(self.conv1.weight, (1, 1, 1, 1))
        conv_crossh_weight = nn.functional.pad(self.conv_crossh.weight, (1, 1, 0, 0))
        conv_crossv_weight = nn.functional.pad(self.conv_crossv.weight, (0, 0, 1, 1))

        def fuse_bn(conv_w, conv_b, bn):
            k = 1 / (bn.running_var + bn.eps).sqrt()
            b = -bn.running_mean * k
            w = conv_w * k.view(-1, 1, 1, 1) * bn.weight.view(-1, 1, 1, 1)
            bias = (conv_b * k + b) * bn.weight + bn.bias
            return w, bias

        w_conv, b_conv = fuse_bn(self.conv.weight, self.conv.bias, self.conv_bn[0])
        w_c1, b_c1 = fuse_bn(conv1_weight, self.conv1.bias, self.conv1_bn[0])
        w_ch, b_ch = fuse_bn(conv_crossh_weight, self.conv_crossh.bias, self.conv_crossh_bn[0])
        w_cv, b_cv = fuse_bn(conv_crossv_weight, self.conv_crossv.bias, self.conv_crossv_bn[0])

        weight = torch.cat([
            self.conv.weight, conv1_weight, conv_crossh_weight, conv_crossv_weight,
            w_conv, w_c1, w_ch, w_cv
        ], dim=0)

        bias_cat = torch.cat([
            self.conv.bias, self.conv1.bias, self.conv_crossh.bias, self.conv_crossv.bias,
            b_conv, b_c1, b_ch, b_cv
        ], dim=0)

        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1)).view(self.conv_out.out_channels, self.in_channels, 3, 3)
        bias = torch.matmul(weight_compress, bias_cat.unsqueeze(-1)).squeeze(-1)
        
        if self.conv_out.bias is not None:
            bias = bias + self.conv_out.bias

        return weight, bias


# ==============================================================================
# MBRConv1: 1x1 多分支重参数化卷积
# ==============================================================================
class MBRConv1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp): 
        x0 = self.conv(inp)  
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        return self.conv_out(x) 

    def slim(self):
        def fuse_bn(conv_w, conv_b, bn):
            k = 1 / (bn.running_var + bn.eps).sqrt()
            b = -bn.running_mean * k
            w = conv_w * k.view(-1, 1, 1, 1) * bn.weight.view(-1, 1, 1, 1)
            bias = (conv_b * k + b) * bn.weight + bn.bias
            return w, bias

        w_conv, b_conv = fuse_bn(self.conv.weight, self.conv.bias, self.conv_bn[0])
        
        weight = torch.cat([self.conv.weight, w_conv], dim=0)
        bias_cat = torch.cat([self.conv.bias, b_conv], dim=0)

        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        bias = torch.matmul(weight_compress, bias_cat)

        if self.conv_out.bias is not None:
            bias = bias + self.conv_out.bias
        return weight, bias


# ==============================================================================
# FST: 逐通道门控版本 (Channel-Gated FST)
# ==============================================================================
class FST(nn.Module):
    def __init__(self, block1, channels):
        """
        y = block1(x)
        out = (DW1x1_u(y)) * (DW1x1_v(y)) + bias
        """
        super(FST, self).__init__()
        self.block1 = block1
        self.u = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        
        with torch.no_grad():
            nn.init.ones_(self.u.weight); nn.init.zeros_(self.u.bias)
            nn.init.ones_(self.v.weight); nn.init.zeros_(self.v.bias)

    def forward(self, x):
        y = self.block1(x)
        return self.u(y) * self.v(y) + self.bias

class FSTS(nn.Module):
    def __init__(self, block1, channels):
        super(FSTS, self).__init__()
        self.block1 = block1
        self.u = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        
        with torch.no_grad():
            nn.init.ones_(self.u.weight); nn.init.zeros_(self.u.bias)
            nn.init.ones_(self.v.weight); nn.init.zeros_(self.v.bias)
        
    def forward(self, x):
        y = self.block1(x)
        return self.u(y) * self.v(y) + self.bias


# ==============================================================================
# DropBlock
# ==============================================================================
class DropBlock(nn.Module):
    def __init__(self, block_size, p=0.5):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p / block_size / block_size

    def forward(self, x):
        mask = 1 - (torch.rand_like(x[:, :1]) >= self.p).float()
        mask = nn.functional.max_pool2d(mask, self.block_size, 1, self.block_size // 2)
        return x * (1 - mask)


# ==============================================================================
# SCA: Contrast-Aware (Mean + Std)
# 核心修正：linear 初始化为全 0，确保初始恒等映射
# ==============================================================================
# model/utils.py 中的 SCA 类

class SCA(nn.Module):
    """
    Contrast-Aware SCA (Calibrated Sigmoid):
    y = x * Sigmoid(Linear([Mean, Std]))
    
    关键修正：
    Bias 初始化为 -1.1，使得初始 Sigmoid 输出约为 0.25。
    这模拟了原版 HDPA (0.5 * 0.5 = 0.25) 的特征衰减幅度，
    解决 PSNR 卡在 15dB 的问题。
    """
    def __init__(self, channels: int, gate_scale: float = 0.1):
        super().__init__()
        # 输入维度 = 2 * channels (Mean + Std)
        self.linear = nn.Linear(channels * 2, channels, bias=True)
        
        with torch.no_grad():
            # 权重为 0，不看输入
            nn.init.zeros_(self.linear.weight)
            
            # ★★★ 核心：偏置设为 -1.1 ★★★
            # Sigmoid(-1.1) ≈ 0.25
            # 完美对齐原版双路注意力的数值分布
            nn.init.constant_(self.linear.bias, -1.1)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Mean + Std
        mu = torch.mean(x, dim=(2, 3)) 
        std = torch.std(x, dim=(2, 3), unbiased=False) 
        stats = torch.cat([mu, std], dim=1)
        
        # 2. Sigmoid 生成权重
        w_att = torch.sigmoid(self.linear(stats))
        
        return x * w_att.view(b, c, 1, 1)
