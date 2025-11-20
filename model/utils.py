# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MBRConv5(nn.Module):
#     def __init__(self, in_channels, out_channels, rep_scale=4):
#         super(MBRConv5, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
#         self.conv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
#         self.conv1_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
#         self.conv2_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
#         self.conv_crossh_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
#         self.conv_crossv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)
        
#     def forward(self, inp):
#         x1 = self.conv(inp)
#         x2 = self.conv1(inp)
#         x3 = self.conv2(inp)
#         x4 = self.conv_crossh(inp)
#         x5 = self.conv_crossv(inp)
#         x = torch.cat(
#             [x1, x2, x3, x4, x5,
#              self.conv_bn(x1),
#              self.conv1_bn(x2),
#              self.conv2_bn(x3),
#              self.conv_crossh_bn(x4),
#              self.conv_crossv_bn(x5)],
#             1
#         )
#         out = self.conv_out(x)
#         return out 

#     def slim(self):
#         conv_weight = self.conv.weight
#         conv_bias = self.conv.bias

#         conv1_weight = self.conv1.weight
#         conv1_bias = self.conv1.bias
#         conv1_weight = nn.functional.pad(conv1_weight, (2, 2, 2, 2))

#         conv2_weight = self.conv2.weight
#         conv2_weight = nn.functional.pad(conv2_weight, (1, 1, 1, 1))
#         conv2_bias = self.conv2.bias

#         conv_crossv_weight = self.conv_crossv.weight
#         conv_crossv_weight = nn.functional.pad(conv_crossv_weight, (1, 1, 2, 2))
#         conv_crossv_bias = self.conv_crossv.bias

#         conv_crossh_weight = self.conv_crossh.weight
#         conv_crossh_weight = nn.functional.pad(conv_crossh_weight, (2, 2, 1, 1))
#         conv_crossh_bias = self.conv_crossh.bias

#         conv1_bn_weight = self.conv1.weight
#         conv1_bn_weight = nn.functional.pad(conv1_bn_weight, (2, 2, 2, 2))

#         conv2_bn_weight = self.conv2.weight
#         conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))

#         conv_crossv_bn_weight = self.conv_crossv.weight
#         conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2))

#         conv_crossh_bn_weight = self.conv_crossh.weight
#         conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))

#         bn = self.conv_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5

#         conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_bias = self.conv.bias * k + b
#         conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

#         bn = self.conv1_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv1_bn_bias = self.conv1.bias * k + b
#         conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

#         bn = self.conv2_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv2_bn_weight = self.conv2.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv2_bn_bias = self.conv2.bias * k + b
#         conv2_bn_bias = conv2_bn_bias * bn.weight + bn.bias

#         conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))
#         bn = self.conv_crossv_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossv_bn_bias = self.conv_crossv.bias * k + b
#         conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

#         conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2))
#         bn = self.conv_crossh_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossh_bn_bias = self.conv_crossh.bias * k + b
#         conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

#         conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))
#         weight = torch.cat(
#             [conv_weight, conv1_weight, conv2_weight,
#              conv_crossh_weight, conv_crossv_weight,
#              conv_bn_weight, conv1_bn_weight, conv2_bn_weight,
#              conv_crossh_bn_weight, conv_crossv_bn_weight],
#             0
#         )
#         weight_compress = self.conv_out.weight.squeeze()
#         weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])
#         bias_ = torch.cat(
#             [conv_bias, conv1_bias, conv2_bias,
#              conv_crossh_bias, conv_crossv_bias,
#              conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
#              conv_crossh_bn_bias, conv_crossv_bn_bias],
#             0
#         )
#         bias = torch.matmul(weight_compress, bias_)
#         if isinstance(self.conv_out.bias, torch.Tensor):
#             bias = bias + self.conv_out.bias
#         return weight, bias


# ##############################################################################################################
# class MBRConv3(nn.Module):
#     def __init__(self, in_channels, out_channels, rep_scale=4):
#         super(MBRConv3, self).__init__()
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.rep_scale = rep_scale
        
#         self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
#         self.conv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
#         self.conv1_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
#         self.conv_crossh_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
#         self.conv_crossv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

#     def forward(self, inp):
#         x0 = self.conv(inp)
#         x1 = self.conv1(inp)
#         x2 = self.conv_crossh(inp)
#         x3 = self.conv_crossv(inp)
#         x = torch.cat(
#             [x0, x1, x2, x3,
#              self.conv_bn(x0),
#              self.conv1_bn(x1),
#              self.conv_crossh_bn(x2),
#              self.conv_crossv_bn(x3)],
#             1
#         )
#         out = self.conv_out(x)
#         return out

#     def slim(self):
#         conv_weight = self.conv.weight
#         conv_bias = self.conv.bias

#         conv1_weight = self.conv1.weight
#         conv1_bias = self.conv1.bias
#         conv1_weight = F.pad(conv1_weight, (1, 1, 1, 1))

#         conv_crossh_weight = self.conv_crossh.weight
#         conv_crossh_bias = self.conv_crossh.bias
#         conv_crossh_weight = F.pad(conv_crossh_weight, (1, 1, 0, 0))

#         conv_crossv_weight = self.conv_crossv.weight
#         conv_crossv_bias = self.conv_crossv.bias
#         conv_crossv_weight = F.pad(conv_crossv_weight, (0, 0, 1, 1))

#         # conv_bn
#         bn = self.conv_bn[0]
#         k = 1 / torch.sqrt(bn.running_var + bn.eps)
#         conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_bias = self.conv.bias * k + (-bn.running_mean * k)
#         conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

#         # conv1_bn
#         bn = self.conv1_bn[0]
#         k = 1 / torch.sqrt(bn.running_var + bn.eps)
#         conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv1_bn_weight = F.pad(conv1_bn_weight, (1, 1, 1, 1))
#         conv1_bn_bias = self.conv1.bias * k + (-bn.running_mean * k)
#         conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

#         # conv_crossh_bn
#         bn = self.conv_crossh_bn[0]
#         k = 1 / torch.sqrt(bn.running_var + bn.eps)
#         conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (1, 1, 0, 0))
#         conv_crossh_bn_bias = self.conv_crossh.bias * k + (-bn.running_mean * k)
#         conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

#         # conv_crossv_bn
#         bn = self.conv_crossv_bn[0]
#         k = 1 / torch.sqrt(bn.running_var + bn.eps)
#         conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (0, 0, 1, 1))
#         conv_crossv_bn_bias = self.conv_crossv.bias * k + (-bn.running_mean * k)
#         conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

#         weight = torch.cat([
#             conv_weight,
#             conv1_weight,
#             conv_crossh_weight,
#             conv_crossv_weight,
#             conv_bn_weight,
#             conv1_bn_weight,
#             conv_crossh_bn_weight,
#             conv_crossv_bn_weight
#         ], dim=0)

#         bias = torch.cat([
#             conv_bias,
#             conv1_bias,
#             conv_crossh_bias,
#             conv_crossv_bias,
#             conv_bn_bias,
#             conv1_bn_bias,
#             conv_crossh_bn_bias,
#             conv_crossv_bn_bias
#         ], dim=0)

#         weight_compress = self.conv_out.weight.squeeze()
#         weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
#         weight = weight.view(self.conv_out.out_channels, self.in_channels, 3, 3)

#         bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)
#         if self.conv_out.bias is not None:
#             bias += self.conv_out.bias

#         return weight, bias

# ######################################################################################################
# class MBRConv1(nn.Module):
#     def __init__(self, in_channels, out_channels, rep_scale=4):
#         super(MBRConv1, self).__init__()
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.rep_scale = rep_scale
        
#         self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
#         self.conv_bn = nn.Sequential(
#             nn.BatchNorm2d(out_channels * rep_scale)
#         )
#         self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

#     def forward(self, inp):
#         x0 = self.conv(inp)
#         x = torch.cat([x0, self.conv_bn(x0)], 1)
#         out = self.conv_out(x)
#         return out 

#     def slim(self):
#         conv_weight = self.conv.weight
#         conv_bias = self.conv.bias

#         bn = self.conv_bn[0]
#         k = 1 / (bn.running_var + bn.eps) ** .5
#         b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
#         conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         conv_bn_bias = self.conv.bias * k + b
#         conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

#         weight = torch.cat([conv_weight, conv_bn_weight], 0)
#         weight_compress = self.conv_out.weight.squeeze()
#         weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

#         bias = torch.cat([conv_bias, conv_bn_bias], 0)
#         bias = torch.matmul(weight_compress, bias)

#         if isinstance(self.conv_out.bias, torch.Tensor):
#             bias = bias + self.conv_out.bias
#         return weight, bias

# # =========================
# # FST：逐通道乘法门控（GatedUnit-Lite）
# # 接口保持一致：FST(block1, channels) / FSTS(block1, channels)
# # =========================
# class FST(nn.Module):
#     def __init__(self, block1, channels):
#         """
#         y = block1(x)
#         out = (DW1x1_u(y)) * (DW1x1_v(y)) + bias
#         逐通道 1x1（groups=channels）≈ 极轻量逐通道仿射，再逐点乘。
#         """
#         super(FST, self).__init__()
#         self.block1 = block1
#         self.u = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
#         self.v = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
#         self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))
#         with torch.no_grad():
#             nn.init.ones_(self.u.weight); nn.init.zeros_(self.u.bias)
#             nn.init.ones_(self.v.weight); nn.init.zeros_(self.v.bias)

#     def forward(self, x):
#         y = self.block1(x)
#         return self.u(y) * self.v(y) + self.bias

# class FSTS(nn.Module):
#     def __init__(self, block1, channels):
#         super(FSTS, self).__init__()
#         self.block1 = block1
#         self.u = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
#         self.v = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
#         self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))
#         with torch.no_grad():
#             nn.init.ones_(self.u.weight); nn.init.zeros_(self.u.bias)
#             nn.init.ones_(self.v.weight); nn.init.zeros_(self.v.bias)

#     def forward(self, x):
#         y = self.block1(x)
#         return self.u(y) * self.v(y) + self.bias

# ##################################################################################
# class DropBlock(nn.Module):
#     def __init__(self, block_size, p=0.5):
#         super(DropBlock, self).__init__()
#         self.block_size = block_size
#         self.p = p / block_size / block_size

#     def forward(self, x):
#         mask = 1 - (torch.rand_like(x[:, :1]) >= self.p).float()
#         mask = nn.functional.max_pool2d(mask, self.block_size, 1, self.block_size // 2)
#         return x * (1 - mask)

# # =========================
# # 新增：SCA（Simple Channel Attention，线性、无激活）
# # =========================
# class SCA(nn.Module):
#     """Simple Channel Attention (linear, activation-free).
#     w = W · GAP(x); y = x ⊙ w
#     """
#     def __init__(self, channels: int):
#         super().__init__()
#         self.fc = nn.Linear(channels, channels, bias=True)
#         with torch.no_grad():
#             if self.fc.weight.shape[0] == self.fc.weight.shape[1]:
#                 nn.init.eye_(self.fc.weight)
#             else:
#                 nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
#             nn.init.zeros_(self.fc.bias)

#     def forward(self, x):
#         b, c, _, _ = x.shape
#         g = torch.mean(x, dim=(2, 3))        # B, C
#         w = self.fc(g).view(b, c, 1, 1)      # B, C, 1, 1
#         return x * w
import torch
import torch.nn as nn
import torch.nn.functional as F

class MBRConv5(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv2_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        ) 
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)
        
    def forward(self, inp):   
        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)
        x = torch.cat(
            [x1, x2, x3, x4, x5,
             self.conv_bn(x1),
             self.conv1_bn(x2),
             self.conv2_bn(x3),
             self.conv_crossh_bn(x4),
             self.conv_crossv_bn(x5)],
            1
        )
        out = self.conv_out(x)
        return out 

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias = self.conv1.bias
        conv1_weight = nn.functional.pad(conv1_weight, (2, 2, 2, 2))

        conv2_weight = self.conv2.weight
        conv2_weight = nn.functional.pad(conv2_weight, (1, 1, 1, 1))
        conv2_bias = self.conv2.bias

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_weight = nn.functional.pad(conv_crossv_weight, (1, 1, 2, 2))
        conv_crossv_bias = self.conv_crossv.bias

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_weight = nn.functional.pad(conv_crossh_weight, (2, 2, 1, 1))
        conv_crossh_bias = self.conv_crossh.bias

        conv1_bn_weight = self.conv1.weight
        conv1_bn_weight = nn.functional.pad(conv1_bn_weight, (2, 2, 2, 2))

        conv2_bn_weight = self.conv2.weight
        conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))

        conv_crossv_bn_weight = self.conv_crossv.weight
        conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2))

        conv_crossh_bn_weight = self.conv_crossh.weight
        conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5

        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        bn = self.conv1_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_bias = self.conv1.bias * k + b
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        bn = self.conv2_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv2_bn_weight = self.conv2.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_bias = self.conv2.bias * k + b
        conv2_bn_bias = conv2_bn_bias * bn.weight + bn.bias

        conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))
        bn = self.conv_crossv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_bias = self.conv_crossv.bias * k + b
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2))
        bn = self.conv_crossh_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_bias = self.conv_crossh.bias * k + b
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))
        weight = torch.cat(
            [conv_weight, conv1_weight, conv2_weight,
             conv_crossh_weight, conv_crossv_weight,
             conv_bn_weight, conv1_bn_weight, conv2_bn_weight,
             conv_crossh_bn_weight, conv_crossv_bn_weight],
            0
        )
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])
        bias_ = torch.cat(
            [conv_bias, conv1_bias, conv2_bias,
             conv_crossh_bias, conv_crossv_bias,
             conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
             conv_crossh_bn_bias, conv_crossv_bn_bias],
            0
        )
        bias = torch.matmul(weight_compress, bias_)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias


##############################################################################################################
class MBRConv3(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv3, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

    def forward(self, inp):    
        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)
        x = torch.cat(
        [    x0,x1,x2,x3,
             self.conv_bn(x0),
             self.conv1_bn(x1),
             self.conv_crossh_bn(x2),
             self.conv_crossv_bn(x3)],
            1
        )    
        out = self.conv_out(x)
        return out

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = self.conv1.weight
        conv1_bias = self.conv1.bias
        conv1_weight = F.pad(conv1_weight, (1, 1, 1, 1))

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_bias = self.conv_crossh.bias
        conv_crossh_weight = F.pad(conv_crossh_weight, (1, 1, 0, 0))

        conv_crossv_weight = self.conv_crossv.weight
        conv_crossv_bias = self.conv_crossv.bias
        conv_crossv_weight = F.pad(conv_crossv_weight, (0, 0, 1, 1))

        # conv_bn
        bn = self.conv_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + (-bn.running_mean * k)
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        # conv1_bn
        bn = self.conv1_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = F.pad(conv1_bn_weight, (1, 1, 1, 1))
        conv1_bn_bias = self.conv1.bias * k + (-bn.running_mean * k)
        conv1_bn_bias = self.conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        # conv_crossh_bn
        bn = self.conv_crossh_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (1, 1, 0, 0))
        conv_crossh_bn_bias = self.conv_crossh.bias * k + (-bn.running_mean * k)
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        # conv_crossv_bn
        bn = self.conv_crossv_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (0, 0, 1, 1))
        conv_crossv_bn_bias = self.conv_crossv.bias * k + (-bn.running_mean * k)
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([
            conv_weight,
            conv1_weight,
            conv_crossh_weight,
            conv_crossv_weight,
            conv_bn_weight,
            conv1_bn_weight,
            conv_crossh_bn_weight,
            conv_crossv_bn_weight
        ], dim=0)

        bias = torch.cat([
            conv_bias,
            conv1_bias,
            conv_crossh_bias,
            conv_crossv_bias,
            conv_bn_bias,
            conv1_bn_bias,
            conv_crossh_bn_bias,
            conv_crossv_bn_bias
        ], dim=0)

        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
        weight = weight.view(self.conv_out.out_channels, self.in_channels, 3, 3)

        bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)
        if self.conv_out.bias is not None:
            bias += self.conv_out.bias

        return weight, bias
    
######################################################################################################
class MBRConv1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp): 
        x0 = self.conv(inp)  
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        out = self.conv_out(x)
        return out 

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([conv_weight, conv_bn_weight], 0)
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

        bias = torch.cat([conv_bias, conv_bn_bias], 0)
        bias = torch.matmul(weight_compress, bias)

        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias
    
# =========================
# FST 替换为逐通道乘法门控（GatedUnit-Lite）
# 接口保持一致：FST(block1, channels) / FSTS(block1, channels)
# =========================
class FST(nn.Module):
    def __init__(self, block1, channels):
        """
        Drop-in 替代原 FST 的乘法门控：
          y = block1(x)
          out = (DW1x1_u(y)) * (DW1x1_v(y)) + bias
        逐通道 1x1（groups=channels）≈ 极轻量逐通道仿射，再逐点乘。
        """
        super(FST, self).__init__()
        self.block1 = block1
        self.u = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        # 初始化为“近似恒等”，替换后数值平稳
        with torch.no_grad():
            nn.init.ones_(self.u.weight); nn.init.zeros_(self.u.bias)
            nn.init.ones_(self.v.weight); nn.init.zeros_(self.v.bias)

    def forward(self, x):
        y = self.block1(x)
        return self.u(y) * self.v(y) + self.bias
        
class FSTS(nn.Module):
    def __init__(self, block1, channels):
        """
        与 FST 同实现，保留类名以兼容现有代码（例如 slim 版本或 S 模型）
        """
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

##################################################################################
class DropBlock(nn.Module):
    def __init__(self, block_size, p=0.5):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p / block_size / block_size

    def forward(self, x):
        mask = 1 - (torch.rand_like(x[:, :1]) >= self.p).float()
        mask = nn.functional.max_pool2d(mask, self.block_size, 1, self.block_size // 2)
        return x * (1 - mask)

# =========================
# 新增：SCA（Simple Channel Attention，线性、恒等初始化）
# =========================
class SCA(nn.Module):
    """Simple Channel Attention: y = x * (1 + gate_scale * Linear(GAP(x)))"""
    def __init__(self, channels: int, gate_scale: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(channels, channels, bias=True)
        self.gate_scale = gate_scale
        with torch.no_grad():
            nn.init.zeros_(self.fc.weight)  # 初始不改变幅度
            nn.init.ones_(self.fc.bias)     # w 初始为 1
    def forward(self, x):
        b, c, _, _ = x.shape
        g = torch.mean(x, dim=(2, 3))           # B, C
        w = 1 + self.gate_scale * self.fc(g)    # 近似恒等
        return x * w.view(b, c, 1, 1)
