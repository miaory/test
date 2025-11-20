"""
SSIM 测试脚本
计算模型输出与 Ground Truth 之间的 SSIM 指标

使用方法:
    python test_ssim.py -task test -config config/lle.yaml
"""
import torch
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from PIL import Image

from option import get_option
from model import import_model
from data import import_loader


def calculate_ssim(img1, img2):
    """
    计算两张图片的 SSIM
    
    Args:
        img1: 第一张图片 (H, W, C) numpy array, uint8
        img2: 第二张图片 (H, W, C) numpy array, uint8
    
    Returns:
        ssim_value: SSIM 值
    """
    # 转换为灰度图或使用多通道 SSIM
    if img1.shape[2] == 3:
        # 使用 multichannel SSIM
        ssim_value = ssim(img1, img2, multichannel=True, channel_axis=2, data_range=255)
    else:
        ssim_value = ssim(img1, img2, data_range=255)
    
    return ssim_value


def tensor_to_uint8(tensor):
    """
    将 PyTorch tensor 转换为 uint8 numpy array
    
    Args:
        tensor: PyTorch tensor (C, H, W) in range [0, 1]
    
    Returns:
        numpy array (H, W, C) in range [0, 255]
    """
    arr = tensor.detach().clamp(0, 1).cpu().numpy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.rint(arr * 255.0).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def test_ssim():
    """测试模型的 SSIM"""
    opt = get_option()
    
    # 加载数据
    test_loader = import_loader(opt)
    if test_loader is None:
        raise RuntimeError("Failed to load test data")
    
    # 加载模型
    model = import_model(opt)
    
    # 加载 checkpoint
    ckpt = opt.config.get("test", {}).get("ckpt", "")
    if ckpt and os.path.isfile(ckpt):
        print(f"Loading checkpoint: {ckpt}")
        state_dict = torch.load(ckpt, map_location=opt.device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.to(opt.device)
    
    ssim_list = []
    psnr_list = []
    
    print("\nCalculating SSIM...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # 解包数据
            if len(batch) == 3:
                inp, gt, name = batch
            else:
                inp, gt = batch
                name = None
            
            inp = inp.to(opt.device)
            if gt is not None:
                gt = gt.to(opt.device)
            else:
                print("Warning: No ground truth available")
                continue
            
            # 推理
            pred = model(inp).clamp(0, 1)
            
            # 转换为 uint8
            pred_u8 = tensor_to_uint8(pred[0])
            gt_u8 = tensor_to_uint8(gt[0])
            
            # 计算 SSIM
            ssim_value = calculate_ssim(pred_u8, gt_u8)
            ssim_list.append(ssim_value)
            
            # 同时计算 PSNR
            diff = pred_u8.astype(np.float64) - gt_u8.astype(np.float64)
            mse = np.mean(diff * diff)
            if mse > 1e-12:
                psnr = 10.0 * np.log10((255.0 ** 2) / mse)
                psnr_list.append(psnr)
    
    # 输出结果
    if ssim_list:
        mean_ssim = float(np.mean(ssim_list))
        std_ssim = float(np.std(ssim_list))
        print(f"\nAverage SSIM: {mean_ssim:.4f} (±{std_ssim:.4f})")
        print(f"Min SSIM: {np.min(ssim_list):.4f}")
        print(f"Max SSIM: {np.max(ssim_list):.4f}")
    
    if psnr_list:
        mean_psnr = float(np.mean(psnr_list))
        std_psnr = float(np.std(psnr_list))
        print(f"\nAverage PSNR: {mean_psnr:.4f} dB (±{std_psnr:.4f})")
        print(f"Min PSNR: {np.min(psnr_list):.4f} dB")
        print(f"Max PSNR: {np.max(psnr_list):.4f} dB")
    
    print(f"\nTotal samples: {len(ssim_list)}")


if __name__ == "__main__":
    test_ssim()
