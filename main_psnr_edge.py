# -*- coding: utf-8 -*-
"""
第二阶段：PSNR + 难区域加权微调脚本（不改模型结构、不用 IWO）

用法（注意：不带 -config）：
    python main_psnr_edge.py -task train -model_task lle_psnr -device cuda

YAML：
  - 使用你原来给 model_task=lle_psnr 的 yaml（已经改成 lle_psnr_edge 那份）
  - 从 Stage1 或上一轮 PSNR best 加载 ckpt
  - loss.weights 里只开 charb + 极小的 msssim
  - loss.edge_dark.enable: true 开启难区域加权
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import time
import json
import traceback
from copy import deepcopy

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import autocast, GradScaler

# 你项目里的模块（与 main2.py 保持一致）
from option import get_option          # 解析命令行 + 读取 yaml，返回 Namespace
from data import import_loader         # 构建 dataloader（签名：import_loader(opt)）
from model import import_model         # 构建模型（签名：import_model(opt)）

# 如果有颜色一致性模块，可以按需打开；PSNR 专项阶段一般不用
try:
    from loss_color import ColorConsistencyLoss
    HAS_COLOR_LOSS = True
except ImportError:
    ColorConsistencyLoss = None
    HAS_COLOR_LOSS = False


# -------------------- 日志/进度工具 -------------------- #
def _log_warn(logger, msg: str):
    if logger is None:
        print("[WARN]", msg)
        return
    if hasattr(logger, "warning"):
        logger.warning(msg)
    elif hasattr(logger, "warn"):
        logger.warn(msg)
    else:
        logger.info("[WARN] " + msg)


def _tqdm(iterable, **kwargs):
    disable_env = os.environ.get("TQDM_DISABLE", "")
    kwargs.setdefault("disable", (not sys.stdout.isatty()) or bool(disable_env))
    kwargs.setdefault("leave", False)
    return tqdm(iterable, **kwargs)


# -------------------- 随机性控制 -------------------- #
def set_seed(seed: int = 2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -------------------- Charbonnier & SSIM 损失 -------------------- #
class CharbonnierLoss(nn.Module):
    """标准 Charbonnier（平滑 L1）"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y, weight=None):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


class SSIMLoss(nn.Module):
    """
    简单版 SSIM（窗口 11x11，高斯权重），输出 1 - SSIM（越小越好）
    """
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", self._create_window(window_size, channel))

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1d_window = self._gaussian(window_size, 1.5).unsqueeze(1)  # [W,1]
        # ★ 这里修正：用 1d @ 1d.T 得到 2d 窗口
        _2d_window = _1d_window @ _1d_window.t()                    # [W,W]
        _2d_window = _2d_window.float().unsqueeze(0).unsqueeze(0)   # [1,1,W,W]
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window

        if channel != self.channel or window.dtype != img1.dtype:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        ssim = ssim_map.mean()
        return 1.0 - ssim


# -------------------- 边缘 + 暗区加权器 -------------------- #
class EdgeDarkWeighter(nn.Module):
    """
    根据“边缘强度 + 暗区”生成像素权重 w:
        w = 1 + λ_edge * edge + λ_dark * mask_dark
    然后会在 Charbonnier 里作为 weight 使用。
    """
    def __init__(
        self,
        edge_lambda: float = 1.0,
        dark_lambda: float = 0.5,
        dark_thresh: float = 0.25,
        use_input_for_edge: bool = True
    ):
        super().__init__()
        self.edge_lambda = edge_lambda
        self.dark_lambda = dark_lambda
        self.dark_thresh = dark_thresh
        self.use_input_for_edge = use_input_for_edge

        # Sobel kernel （3x3）
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, inp: torch.Tensor, gt: torch.Tensor):
        """
        inp, gt: [B, C, H, W], 范围 [0,1]
        返回: w [B, 1, H, W]
        """
        # 选一个做灰度图：默认为输入图像
        x = inp if self.use_input_for_edge else gt

        # 灰度 [B,1,H,W]
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x[:, 0:1]

        # Sobel 边缘
        ex = F.conv2d(gray, self.kx, padding=1)
        ey = F.conv2d(gray, self.ky, padding=1)
        edge_mag = torch.sqrt(ex * ex + ey * ey + 1e-6)

        # 归一化 edge 到 [0,1]
        B = edge_mag.size(0)
        edge_norm = torch.zeros_like(edge_mag)
        for i in range(B):
            e = edge_mag[i:i+1]
            emin = e.amin(dim=(1, 2, 3), keepdim=True)
            emax = e.amax(dim=(1, 2, 3), keepdim=True)
            edge_norm[i:i+1] = (e - emin) / (emax - emin + 1e-6)

        # 暗区 mask（用 GT 的灰度更稳）
        if gt.size(1) == 3:
            gray_gt = 0.299 * gt[:, 0:1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]
        else:
            gray_gt = gt[:, 0:1]
        mask_dark = (gray_gt < self.dark_thresh).float()

        w = 1.0 + self.edge_lambda * edge_norm + self.dark_lambda * mask_dark
        return w


# -------------------- EMA 模型 -------------------- #
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1.0 - d))


# -------------------- PSNR -------------------- #
def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, gt, reduction="mean").item()
    if mse == 0:
        return 99.0
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return psnr


# -------------------- dataloader batch 解析 -------------------- #
def parse_batch(batch, device):
    if isinstance(batch, dict):
        inp = None
        gt = None
        for k in ["inp", "input", "lq", "LQ"]:
            if k in batch:
                inp = batch[k]
                break
        for k in ["gt", "GT", "target"]:
            if k in batch:
                gt = batch[k]
                break
        assert inp is not None and gt is not None, "无法从 batch dict 中找到 inp / gt，请根据你的 keys 修改 parse_batch()"
    else:
        assert len(batch) >= 2, "batch 长度 < 2，无法解析 inp / gt"
        inp, gt = batch[0], batch[1]

    inp = inp.to(device, non_blocking=True).float()
    gt = gt.to(device, non_blocking=True).float()
    if inp.max() > 1.5:
        inp = inp / 255.0
    if gt.max() > 1.5:
        gt = gt / 255.0
    return inp, gt


# -------------------- 构建模型 + 优化器 -------------------- #
def build_model_and_optimizer(opt, cfg, device):
    # 与原工程保持一致，只传 opt
    model = import_model(opt)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params} / Trainable: {trainable_params}")

    train_cfg = cfg.get("train", {})
    ckpt_path = train_cfg.get("ckpt", "")
    if ckpt_path:
        print(f"[Model] 从 checkpoint 加载权重: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Model] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    lr = float(train_cfg.get("lr", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optim = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    max_epoch = int(train_cfg.get("epoch", 200))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epoch)

    return model, optim, sched


# -------------------- 构建损失：Charb + 少量 SSIM + Edge/Dark 加权 -------------------- #
def build_loss(cfg, device):
    loss_cfg = cfg.get("loss", {})
    weights_cfg = loss_cfg.get("weights", {})
    w_charb = float(weights_cfg.get("charb", 1.0))
    w_msssim = float(weights_cfg.get("msssim", 0.0))

    # Edge/Dark 配置
    edge_dark_cfg = loss_cfg.get("edge_dark", {})
    ed_enable = bool(edge_dark_cfg.get("enable", False))
    ed_edge_lambda = float(edge_dark_cfg.get("edge_lambda", 1.0))
    ed_dark_lambda = float(edge_dark_cfg.get("dark_lambda", 0.5))
    ed_dark_thresh = float(edge_dark_cfg.get("dark_thresh", 0.25))
    ed_use_input = bool(edge_dark_cfg.get("use_input_for_edge", True))

    if ed_enable:
        edge_weighter = EdgeDarkWeighter(
            edge_lambda=ed_edge_lambda,
            dark_lambda=ed_dark_lambda,
            dark_thresh=ed_dark_thresh,
            use_input_for_edge=ed_use_input
        ).to(device)
    else:
        edge_weighter = None

    print(f"[Loss] charb={w_charb}, msssim={w_msssim}, edge_dark.enable={ed_enable}, "
          f"edge_lambda={ed_edge_lambda}, dark_lambda={ed_dark_lambda}, dark_thresh={ed_dark_thresh}")

    charbonnier = CharbonnierLoss()
    ssim_loss = SSIMLoss()

    def criterion(inp, pred, gt):
        loss_total = 0.0
        loss_dict = {}

        weight_map = None
        if edge_weighter is not None:
            weight_map = edge_weighter(inp, gt)  # [B,1,H,W]
            if weight_map.shape[1] == 1 and pred.shape[1] > 1:
                weight_map_b = weight_map.expand(-1, pred.shape[1], -1, -1)
            else:
                weight_map_b = weight_map
        else:
            weight_map_b = None

        if w_charb > 0:
            lc = charbonnier(pred, gt, weight=weight_map_b)
            loss_total = loss_total + w_charb * lc
            loss_dict["charb"] = lc.item()

        if w_msssim > 0:
            ls = ssim_loss(pred, gt)
            loss_total = loss_total + w_msssim * ls
            loss_dict["msssim"] = ls.item()

        loss_dict["total"] = float(loss_total.item())
        return loss_total, loss_dict

    return criterion


# -------------------- 训练 & 验证 -------------------- #
def train_one_epoch(
    epoch,
    model,
    ema: ModelEMA,
    optimizer,
    scaler: GradScaler,
    train_loader,
    criterion,
    device,
    use_amp: bool = True
):
    model.train()
    if ema is not None:
        ema.ema.eval()

    pbar = _tqdm(train_loader, desc=f"[Train] Epoch {epoch}")
    loss_meter = []

    for i, batch in enumerate(pbar):
        inp, gt = parse_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            pred = model(inp)
            loss, loss_dict = criterion(inp, pred, gt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        loss_meter.append(loss.item())
        if (i + 1) % 10 == 0:
            pbar.set_postfix({"loss": f"{np.mean(loss_meter):.4f}"})

    return float(np.mean(loss_meter)) if loss_meter else 0.0


@torch.no_grad()
def validate(model, valid_loader, device):
    model.eval()
    psnr_list = []

    for batch in _tqdm(valid_loader, desc="[Valid]", leave=False):
        inp, gt = parse_batch(batch, device)
        pred = model(inp)
        pred = torch.clamp(pred, 0.0, 1.0)
        psnr_val = calculate_psnr(pred, gt)
        psnr_list.append(psnr_val)

    return float(np.mean(psnr_list)) if psnr_list else 0.0


# -------------------- 主流程 -------------------- #
def main():
    opt = get_option()
    cfg = opt.config if hasattr(opt, "config") else {}
    seed = cfg.get("data", {}).get("seed", 2025)
    set_seed(seed)

    device = torch.device("cuda" if opt.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    print("[Config]")
    try:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
    except Exception:
        print(cfg)

    # 与原工程一致：import_loader(opt)
    loaders = import_loader(opt)
    if isinstance(loaders, (list, tuple)):
        if len(loaders) == 3:
            train_loader, valid_loader, test_loader = loaders
        elif len(loaders) == 2:
            train_loader, valid_loader = loaders
            test_loader = None
        else:
            raise RuntimeError("import_loader 返回长度不对，请检查 data/import_loader.py")
    else:
        raise RuntimeError("import_loader 返回值类型不支持")

    model, optim, sched = build_model_and_optimizer(opt, cfg, device)
    criterion = build_loss(cfg, device)

    train_cfg = cfg.get("train", {})
    use_ema = bool(train_cfg.get("ema", True))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    if use_ema:
        print(f"[EMA] 启用, decay={ema_decay}")
    else:
        print("[EMA] 未启用")

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    task = opt.task
    max_epoch = int(train_cfg.get("epoch", 200))
    best_psnr = -1.0
    best_psnr_ema = -1.0

    exp_name = cfg.get("exp_name", "lle_psnr_edge")
    root = getattr(opt, "root", os.getcwd())
    save_dir = os.path.join(root, "experiments", exp_name)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)

    if task == "train":
        for epoch in range(1, max_epoch + 1):
            t0 = time.time()
            avg_loss = train_one_epoch(
                epoch,
                model,
                ema,
                optim,
                scaler,
                train_loader,
                criterion,
                device,
                use_amp=use_amp
            )
            sched.step()

            val_psnr = validate(model, valid_loader, device)
            if ema is not None:
                val_psnr_ema = validate(ema.ema, valid_loader, device)
            else:
                val_psnr_ema = val_psnr

            is_best = val_psnr > best_psnr
            is_best_ema = val_psnr_ema > best_psnr_ema

            if is_best:
                best_psnr = val_psnr
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, "models", "best_model.pkl")
                )
            if is_best_ema:
                best_psnr_ema = val_psnr_ema
                if ema is not None:
                    torch.save(
                        ema.ema.state_dict(),
                        os.path.join(save_dir, "models", "best_model_ema.pkl")
                    )

            t1 = time.time()
            print(
                f"[Epoch {epoch:03d}/{max_epoch}] "
                f"loss={avg_loss:.4f}, "
                f"PSNR={val_psnr:.3f}, "
                f"PSNR_EMA={val_psnr_ema:.3f}, "
                f"best={best_psnr:.3f}, "
                f"best_ema={best_psnr_ema:.3f}, "
                f"time/epoch={t1 - t0:.1f}s"
            )

        print(f"[Train Done] best PSNR={best_psnr:.3f}, best PSNR(EMA)={best_psnr_ema:.3f}")
    elif task == "test":
        model_path_ema = os.path.join(save_dir, "models", "best_model_ema.pkl")
        model_path = os.path.join(save_dir, "models", "best_model.pkl")
        if os.path.isfile(model_path_ema):
            print(f"[Test] 加载 EMA 模型: {model_path_ema}")
            sd = torch.load(model_path_ema, map_location=device)
            model.load_state_dict(sd, strict=False)
        elif os.path.isfile(model_path):
            print(f"[Test] 加载模型: {model_path}")
            sd = torch.load(model_path, map_location=device)
            model.load_state_dict(sd, strict=False)
        else:
            print("[Test] 未找到 best_model_ema.pkl 或 best_model.pkl")
            return

        if test_loader is None:
            psnr_test = validate(model, valid_loader, device)
        else:
            psnr_test = validate(model, test_loader, device)
        print(f"[Test] PSNR = {psnr_test:.3f}")
    else:
        raise ValueError(f"未知 task: {task}, 仅支持 train / test")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL] 运行异常:", e)
        traceback.print_exc()
