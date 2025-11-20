# import torch
# import numpy as np
# import os
# from PIL import Image
# import random

# IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


# def _list_images(root):
#     return sorted([
#         f for f in os.listdir(root)
#         if f.lower().endswith(IMG_EXTS) and not f.startswith('.')
#     ])


# def _ensure_min_size(img_pil, min_h, min_w):
#     """若尺寸小于给定最小值，则等比例放大到不小于该值。"""
#     w, h = img_pil.size
#     if h >= min_h and w >= min_w:
#         return img_pil
#     scale = max(min_h / h, min_w / w)
#     new_w = int(round(w * scale))
#     new_h = int(round(h * scale))
#     return img_pil.resize((new_w, new_h), resample=Image.BICUBIC)


# def _random_crop_pair(inp_pil, gt_pil, patch):
#     """在两张 PIL 图上做相同的随机裁剪（patch, patch）"""
#     w, h = inp_pil.size
#     assert gt_pil.size == (w, h), "inp/gt 尺寸不一致，请检查数据对齐"
#     if h == patch and w == patch:
#         return inp_pil, gt_pil
#     if h < patch or w < patch:
#         # 理论上上游已放大，这里兜底
#         inp_pil = _ensure_min_size(inp_pil, patch, patch)
#         gt_pil = _ensure_min_size(gt_pil, patch, patch)
#         w, h = inp_pil.size
#     top = random.randint(0, h - patch)
#     left = random.randint(0, w - patch)
#     box = (left, top, left + patch, top + patch)
#     return inp_pil.crop(box), gt_pil.crop(box)


# class LLEData(torch.utils.data.Dataset):
#     def __init__(self, opt, inp_path, gt_path=None, filenames=None, phase='train', patch_size=None):
#         """
#         filenames: 可选的文件名列表（只含文件名，不含路径）。若提供则按该列表读取；否则自动从inp_path读取全部图片文件。
#         phase: 'train' | 'valid' | 'test' | 'demo'
#         patch_size: 仅在 phase='train' 时使用；将样本裁剪为固定大小 patch（int）。
#         """
#         super(LLEData, self).__init__()
#         self.inp_path = inp_path
#         self.gt_path = gt_path
#         self.phase = phase
#         self.patch_size = int(patch_size) if (patch_size is not None) else None

#         if filenames is not None:
#             self.img_li = list(filenames)
#         else:
#             self.img_li = _list_images(inp_path)

#     def __getitem__(self, index):
#         fname = self.img_li[index]
#         inp_pil = Image.open(os.path.join(self.inp_path, fname)).convert('RGB')

#         if self.gt_path:
#             gt_pil = Image.open(os.path.join(self.gt_path, fname)).convert('RGB')
#         else:
#             gt_pil = None

#         # 训练：随机裁剪固定 patch
#         if self.phase == 'train' and self.gt_path is not None and self.patch_size is not None:
#             # 先确保尺寸不小于 patch，再随机裁剪
#             inp_pil = _ensure_min_size(inp_pil, self.patch_size, self.patch_size)
#             gt_pil = _ensure_min_size(gt_pil, self.patch_size, self.patch_size)
#             inp_pil, gt_pil = _random_crop_pair(inp_pil, gt_pil, self.patch_size)

#         # 其他阶段：不裁剪，原尺寸
#         inp = np.array(inp_pil).transpose([2, 0, 1]).astype(np.float32) / 255.0
#         inp = torch.tensor(inp, dtype=torch.float32)  # 留在 CPU

#         if gt_pil is not None:
#             gt = np.array(gt_pil).transpose([2, 0, 1]).astype(np.float32) / 255.0
#             gt = torch.tensor(gt, dtype=torch.float32)
#             return inp, gt, os.path.splitext(fname)[0]

#         return inp, os.path.splitext(fname)[0]

#     def __len__(self):
#         return len(self.img_li)

import torch
import numpy as np
import os
from PIL import Image
import random

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _list_images(root):
    return sorted([
        f for f in os.listdir(root)
        if f.lower().endswith(IMG_EXTS) and not f.startswith('.')
    ])


def _ensure_min_size(img_pil, min_h, min_w):
    """若尺寸小于给定最小值，则等比例放大到不小于该值。"""
    w, h = img_pil.size
    if h >= min_h and w >= min_w:
        return img_pil
    scale = max(min_h / h, min_w / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # 统一用 BICUBIC 放大
    Resampling = getattr(Image, "Resampling", Image)
    return img_pil.resize((new_w, new_h), resample=Resampling.BICUBIC)


def _random_crop_pair(inp_pil, gt_pil, patch):
    """在两张 PIL 图上做相同的随机裁剪（patch, patch）"""
    w, h = inp_pil.size
    assert gt_pil.size == (w, h), "inp/gt 尺寸不一致，请检查数据对齐"
    if h == patch and w == patch:
        return inp_pil, gt_pil
    if h < patch or w < patch:
        # 理论上上游已放大，这里兜底
        inp_pil = _ensure_min_size(inp_pil, patch, patch)
        gt_pil = _ensure_min_size(gt_pil, patch, patch)
        w, h = inp_pil.size
    top = random.randint(0, h - patch)
    left = random.randint(0, w - patch)
    box = (left, top, left + patch, top + patch)
    return inp_pil.crop(box), gt_pil.crop(box)


# === 多尺度随机缩放 + 抗混叠 ===
def _rescale_aa(img_pil, scale):
    """
    抗混叠缩放：下采样用 LANCZOS，上采样用 BICUBIC。
    Pillow 新版用 Image.Resampling.*，旧版直接用 Image.*，这里做了兼容。
    """
    w, h = img_pil.size
    new_w = max(32, int(round(w * scale)))
    new_h = max(32, int(round(h * scale)))
    Resampling = getattr(Image, "Resampling", Image)
    resample = Resampling.LANCZOS if scale < 1.0 else Resampling.BICUBIC
    return img_pil.resize((new_w, new_h), resample=resample)


class LLEData(torch.utils.data.Dataset):
    def __init__(self, opt, inp_path, gt_path=None, filenames=None, phase='train', patch_size=None):
        """
        filenames: 可选的文件名列表（只含文件名，不含路径）。若提供则按该列表读取；否则自动从inp_path读取全部图片文件。
        phase: 'train' | 'valid' | 'test' | 'demo'
        patch_size: 仅在 phase='train' 时使用；将样本裁剪为固定大小 patch（int）。
        """
        super(LLEData, self).__init__()
        self.inp_path = inp_path
        self.gt_path = gt_path
        self.phase = phase
        self.patch_size = int(patch_size) if (patch_size is not None) else None

        # 随机缩放范围（可从 opt.config 里读取，可选）
        # 默认 0.6~1.4（足够覆盖 512→4K 的尺度随机性），仅在训练裁剪前使用
        try:
            cfg = getattr(opt, "config", {})
            self.scale_min = float(cfg.get("data", {}).get("scale_min", 0.6))
            self.scale_max = float(cfg.get("data", {}).get("scale_max", 1.4))
        except Exception:
            self.scale_min, self.scale_max = 0.6, 1.4

        if filenames is not None:
            self.img_li = list(filenames)
        else:
            self.img_li = _list_images(inp_path)

    def __getitem__(self, index):
        fname = self.img_li[index]
        inp_pil = Image.open(os.path.join(self.inp_path, fname)).convert('RGB')

        if self.gt_path:
            gt_pil = Image.open(os.path.join(self.gt_path, fname)).convert('RGB')
        else:
            gt_pil = None

        # 训练：多尺度随机缩放（抗混叠）→ 兜底放大至 >= patch → 随机裁剪固定 patch
        if self.phase == 'train' and self.gt_path is not None and self.patch_size is not None:
            # 1) 多尺度随机缩放（inp/gt 同步）
            s = random.uniform(self.scale_min, self.scale_max)
            inp_pil = _rescale_aa(inp_pil, s)
            gt_pil = _rescale_aa(gt_pil, s)

            # 2) 兜底：确保尺寸不小于 patch
            inp_pil = _ensure_min_size(inp_pil, self.patch_size, self.patch_size)
            gt_pil = _ensure_min_size(gt_pil, self.patch_size, self.patch_size)

            # 3) 再随机裁剪
            inp_pil, gt_pil = _random_crop_pair(inp_pil, gt_pil, self.patch_size)

        # 其他阶段：不裁剪，原尺寸
        inp = np.array(inp_pil).transpose([2, 0, 1]).astype(np.float32) / 255.0
        inp = torch.tensor(inp, dtype=torch.float32)  # 留在 CPU

        if gt_pil is not None:
            gt = np.array(gt_pil).transpose([2, 0, 1]).astype(np.float32) / 255.0
            gt = torch.tensor(gt, dtype=torch.float32)
            return inp, gt, os.path.splitext(fname)[0]

        return inp, os.path.splitext(fname)[0]

    def __len__(self):
        return len(self.img_li)
