# # import torch
# # import torch.nn as nn

# # class MobileIENetS(nn.Module):
# #     def __init__(self, channels):
# #         super(MobileIENetS, self).__init__()
# #         self.head = FST(
# #             nn.Sequential(
# #                 nn.Conv2d(3, channels, 5, 1, 2),
# #                 nn.PReLU(channels),
# #                 nn.Conv2d(channels, channels, 3, 1, 1)
# #             ),
# #             channels
# #         )
# #         self.body = FST(
# #             nn.Conv2d(channels, channels, 3, 1, 1),
# #             channels
# #         )
# #         self.att = nn.Sequential(
# #             nn.AdaptiveAvgPool2d(1),
# #             nn.Conv2d(channels, channels, 1),
# #             nn.Sigmoid()
# #         )
# #         self.att1 = nn.Sequential( 
# #             nn.Conv2d(1, channels, 1, 1),
# #             nn.Sigmoid()
# #         )
# #         self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        
# #     def forward(self, x):
# #         x0 = self.head(x)
# #         x1 = self.body(x0)
# #         x2 = self.att(x1)
# #         max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)   
# #         x3 = self.att1(max_out)
# #         x4 = torch.mul(x3, x2) * x1
# #         return self.tail(x4)

# # class FST(nn.Module):
# #     def __init__(self, block1, channels):
# #         super(FST, self).__init__()
# #         self.block1 = block1
# #         self.weight1 = nn.Parameter(torch.randn(1)) 
# #         self.weight2 = nn.Parameter(torch.randn(1)) 
# #         self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))
        
# #     def forward(self, x):
# #         x1 = self.block1(x)
# #         weighted_block1 = self.weight1 * x1
# #         weighted_block2 = self.weight2 * x1
# #         return weighted_block1 * weighted_block2 + self.bias

# # def export_onnx(pretrained_model_path):
# #     model = MobileIENetS(12)  
    
# #     checkpoint = torch.load(pretrained_model_path)
# #     model.load_state_dict(checkpoint)
# #     model.eval()  
    
# #     dummy_input = torch.randn(1, 3, 400, 600)
    
# #     torch.onnx.export(
# #         model,                          
# #         dummy_input,                   
# #         "LLE.onnx",        
# #         opset_version=12,              
# #         export_params=True,            
# #         do_constant_folding=True,       
# #         input_names=['input'],          
# #         output_names=['output'],     
# #         dynamic_axes=None
# #     )
# #     print("ONNX Success.")

# # if __name__ == "__main__":
# #     pretrained_model_path = r'./pretrain/lolv1_best_slim.pkl'
# #     export_onnx(pretrained_model_path)


# # 替换fst
# import torch
# import torch.nn as nn

# class FST(nn.Module):
#     """
#     门控版 FST（GatedUnit-Lite）：
#       y = block1(x)
#       out = (DW1x1_u(y)) * (DW1x1_v(y)) + bias
#     - 两条逐通道 1x1 卷积产生 u、v，然后逐点乘；
#     - 初始化为“近似恒等”（权重=1，偏置=0），替换后数值更稳；
#     - 对 INT8 per-channel 量化友好；仅包含 Conv/Mul/Add 运算，ONNX 友好。
#     """
#     def __init__(self, block1, channels):
#         super(FST, self).__init__()
#         self.block1 = block1
#         self.u = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
#         self.v = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
#         self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))
#         # 初始化为恒等映射
#         with torch.no_grad():
#             nn.init.ones_(self.u.weight); nn.init.zeros_(self.u.bias)
#             nn.init.ones_(self.v.weight); nn.init.zeros_(self.v.bias)

#     def forward(self, x):
#         y = self.block1(x)
#         return self.u(y) * self.v(y) + self.bias


# class MobileIENetS(nn.Module):
#     def __init__(self, channels):
#         super(MobileIENetS, self).__init__()
#         self.head = FST(
#             nn.Sequential(
#                 nn.Conv2d(3, channels, 5, 1, 2),
#                 nn.PReLU(channels),
#                 nn.Conv2d(channels, channels, 3, 1, 1)
#             ),
#             channels
#         )
#         self.body = FST(
#             nn.Conv2d(channels, channels, 3, 1, 1),
#             channels
#         )
#         self.att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels, 1),
#             nn.Sigmoid()
#         )
#         self.att1 = nn.Sequential( 
#             nn.Conv2d(1, channels, 1, 1),
#             nn.Sigmoid()
#         )
#         self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        
#     def forward(self, x):
#         x0 = self.head(x)
#         x1 = self.body(x0)
#         x2 = self.att(x1)
#         max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)   
#         x3 = self.att1(max_out)
#         x4 = torch.mul(x3, x2) * x1
#         return self.tail(x4)


# def _migrate_square_fst_to_gate(ckpt_sd: dict, model_sd: dict) -> dict:
#     """
#     将历史平方版 FST 的权重 (weight1/weight2/bias) 映射为门控版的 (u/v/bias)。
#     - weight1 -> u.weight（按形状展开为逐通道 1x1 的卷积核）
#     - weight2 -> v.weight
#     - bias    -> bias（原样）
#     - 同时将 u.bias / v.bias 置零（若存在）
#     """
#     if ckpt_sd is None:
#         return ckpt_sd

#     sd = dict(ckpt_sd)  # 复制，避免原地修改副作用
#     keys = list(sd.keys())

#     for k in keys:
#         if k.endswith('.weight1'):
#             base = k[:-8]  # 去掉 ".weight1"
#             uk = base + '.u.weight'
#             ub = base + '.u.bias'
#             if uk in model_sd:
#                 w1 = sd.pop(k)
#                 # 兼容张量/标量：统一成 [1,1,1,1] 再广播到目标形状
#                 if not torch.is_tensor(w1):
#                     w1 = torch.tensor(w1)
#                 w1 = w1.view(1,1,1,1)
#                 sd[uk] = torch.ones_like(model_sd[uk]) * w1
#                 if ub in model_sd:
#                     sd[ub] = torch.zeros_like(model_sd[ub])

#         elif k.endswith('.weight2'):
#             base = k[:-8]
#             vk = base + '.v.weight'
#             vb = base + '.v.bias'
#             if vk in model_sd:
#                 w2 = sd.pop(k)
#                 if not torch.is_tensor(w2):
#                     w2 = torch.tensor(w2)
#                 w2 = w2.view(1,1,1,1)
#                 sd[vk] = torch.ones_like(model_sd[vk]) * w2
#                 if vb in model_sd:
#                     sd[vb] = torch.zeros_like(model_sd[vb])

#         # 对 *.bias：与门控版的 bias 名称相同，可直接沿用，无需改键名
#         # 这里不做处理，保持原样。如果旧权重里没有 bias，门控版会用默认 0。

#     return sd


# def export_onnx(pretrained_model_path: str, channels: int = 12, onnx_path: str = "LLE.onnx"):
#     device = torch.device("cpu")
#     model = MobileIENetS(channels).to(device).eval()

#     # 读入 checkpoint（兼容 {state_dict: ...} 或直接是 state dict）
#     ckpt = torch.load(pretrained_model_path, map_location=device)
#     if isinstance(ckpt, dict) and 'state_dict' in ckpt:
#         sd = ckpt['state_dict']
#     else:
#         sd = ckpt

#     # 进行 FST 权重迁移（若是旧平方版权重）
#     sd = _migrate_square_fst_to_gate(sd, model.state_dict())

#     # 加载（允许略微不匹配，避免无关键报错）
#     missing, unexpected = model.load_state_dict(sd, strict=False)
#     if missing:
#         print("[Info] Missing keys:", missing)
#     if unexpected:
#         print("[Info] Unexpected keys:", unexpected)

#     # 导出 ONNX
#     dummy_input = torch.randn(1, 3, 400, 600, device=device)
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         opset_version=12,
#         export_params=True,
#         do_constant_folding=True,
#         input_names=['input'],
#         output_names=['output'],
#         dynamic_axes=None
#     )
#     print(f"ONNX Success -> {onnx_path}")


# if __name__ == "__main__":
#     pretrained_model_path = r'/home/featurize/work/test9/experiments/2025-11-15 03-56-38 train_lle/models/best_slim_reexport.pkl'
#     export_onnx(pretrained_model_path, channels=12, onnx_path="LLE.onnx")
# -*- coding: utf-8 -*-
"""
把 LLE 的 slim.pkl（state_dict）导出为 ONNX。

使用方法：
    python torch_to_onnx.py

注意：
1. 修改下面的 CKPT_PATH 指向你的 best_slim.pkl / slim.pkl。
2. 修改 CHANNELS = 你的 YAML 里 model.channels 的值（比如 32 / 48 等）。
3. 根据需要修改 INPUT_H/INPUT_W 为你实际部署时的输入尺寸。
"""

import os
import torch
import torch.nn as nn

# 从你的项目里导入 slim 结构（MobileIELLENetS）
from model.lle import MobileIELLENetS

# ================== 手动配置区域 ==================

# ① slim 权重路径（你的 slim.pkl 或 best_slim.pkl）
CKPT_PATH = "/home/featurize/work/test9/experiments/2025-11-15 03-56-38 train_lle/models/best_slim_reexport.pkl"

# ② 通道数：一定要和 YAML 里 model.channels 一致！！！
#    去你当前实验用的 YAML 里看：config['model']['channels']
CHANNELS = 12  # ← 比如 32 / 48 / 64，根据你实际配置改

# ③ 输入分辨率（导出用的 dummy input 尺寸）
INPUT_H = 256
INPUT_W = 256

# ④ ONNX 输出路径
ONNX_PATH = "MobileIE_LLE_slim.onnx"

# ⑤ ONNX opset 版本（17 一般够用）
OPSET = 17

# ==================================================


def _load_slim_state_dict(ckpt_path, device):
    """加载 slim.pkl 里的 state_dict，并做一些兼容处理。"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 slim 权重文件: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=device)

    # 可能是直接保存的 state_dict，也可能包了一层 {'state_dict': ...}
    if isinstance(sd, dict) and ("state_dict" in sd or "model" in sd):
        sd = sd.get("state_dict", sd.get("model"))

    if not isinstance(sd, dict):
        raise RuntimeError(
            f"slim 权重文件里不是 state_dict（dict），实际类型: {type(sd)}。\n"
            f"当前脚本是按 state_dict 方式写的，请确认训练保存逻辑。"
        )

    # 兼容 DataParallel: 去掉 'module.' 前缀
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    return sd


def build_slim_model_from_state_dict(ckpt_path, channels, device):
    """
    1. 构建 MobileIELLENetS(channels)
    2. 加载 slim.pkl 的 state_dict
    """
    print(f"[Info] 构建 MobileIELLENetS(channels={channels}) ...")
    model = MobileIELLENetS(channels=channels).to(device)

    print(f"[Info] 从 slim 权重加载 state_dict: {ckpt_path}")
    sd = _load_slim_state_dict(ckpt_path, device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Info] load_state_dict 完成: missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  missing keys 示例:", missing[:5])
    if len(unexpected) > 0:
        print("  unexpected keys 示例:", unexpected[:5])

    model.eval()
    return model


def export_onnx_from_slim(ckpt_path, onnx_path,
                           channels=32,
                           height=256, width=256,
                           opset=17):
    """主导出函数：从 slim state_dict → MobileIELLENetS → ONNX"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    # 1) 构建 slim 模型并加载权重
    model = build_slim_model_from_state_dict(ckpt_path, channels, device)

    # 2) 构造 dummy 输入（注意尺寸要和你部署时一致）
    dummy_input = torch.randn(1, 3, height, width, device=device)
    print(f"[Info] dummy input shape: {tuple(dummy_input.shape)}")

    # 3) 导出 ONNX
    print(f"[Info] 开始导出 ONNX -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"[OK] ONNX 导出完成: {onnx_path}")


if __name__ == "__main__":
    export_onnx_from_slim(
        ckpt_path=CKPT_PATH,
        onnx_path=ONNX_PATH,
        channels=CHANNELS,
        height=INPUT_H,
        width=INPUT_W,
        opset=OPSET,
    )
