import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import sys
import time
import json
import math
import traceback
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler

from logger import Logger
from option import get_option
from data import import_loader
from loss import import_loss
from model import import_model

# （如项目中用到颜色一致性损失，这里保留导入；未用也不影响）
try:
    from loss_color import ColorConsistencyLoss, rgb_to_lab
except Exception:
    ColorConsistencyLoss, rgb_to_lab = None, None


# -------------------- 实用函数 --------------------
def _log_warn(logger, msg: str):
    if hasattr(logger, "warning"):
        logger.warning(msg)
    elif hasattr(logger, "warn"):
        logger.warn(msg)
    else:
        logger.info(f"[WARN] {msg}")

def _tqdm(iterable, **kwargs):
    disable_env = os.environ.get("TQDM_DISABLE", "")
    kwargs.setdefault("disable", (not sys.stdout.isatty()) or bool(disable_env))
    return tqdm(iterable, **kwargs)

def _as_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def psnr_torch(pred, gt, eps=1e-10):
    # pred/gt: B,C,H,W in [0,1]
    mse = torch.mean((pred - gt) ** 2)
    mse = torch.clamp(mse, min=eps)
    return 10.0 * torch.log10(1.0 / mse)

def _device_from_opt(opt):
    dev = getattr(opt, "device", "cuda")
    return torch.device(dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu")


# -------------------- 数据加载封装（兼容不同仓库写法） --------------------
def build_loaders(opt, logger):
    """期待 import_loader 返回 dict 或具名对象；尽量兼容。"""
    loaders = import_loader(opt)
    train_loader = None
    val_loader = None
    test_loader = None

    if isinstance(loaders, dict):
        train_loader = loaders.get("train", None)
        val_loader = loaders.get("valid", loaders.get("val", None))
        test_loader = loaders.get("test", None)
    else:
        # 兜底：尝试属性访问
        train_loader = getattr(loaders, "train", None)
        val_loader = getattr(loaders, "valid", getattr(loaders, "val", None))
        test_loader = getattr(loaders, "test", None)

    if train_loader is None and (getattr(opt, "task", "train") == "train"):
        _log_warn(logger, "[Data] 未找到 train_loader，请检查 data.import_loader 的返回。")
    if val_loader is None:
        _log_warn(logger, "[Data] 未找到 valid/val loader，验证将被跳过。")

    return train_loader, val_loader, test_loader


# -------------------- 模型/损失构建 --------------------
def build_model(opt, logger):
    # 你的仓库通常以 -model_task 控制：['isp','lle','sr']
    model = import_model(opt)
    return model


def build_criterion(opt, logger):
    crit = import_loss(opt)
    return crit


# -------------------- IWO：载入基底权重、冻结、参数分组 --------------------
def load_base_ckpt_for_iwo(model, ckpt_path, device, logger):
    if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
        raise RuntimeError(f"[IWO] 指定的基底权重不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容可能的多种保存格式
    cand = None
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "net", "ema", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                cand = ckpt[k]
                break
    base_sd = cand if cand is not None else ckpt

    # 去掉可能的 "module." 前缀
    if any(k.startswith("module.") for k in base_sd.keys()):
        base_sd = {k[len("module."):]: v for k, v in base_sd.items()}

    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    logger.info(f"[IWO] 加载基底权重完成: missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        _log_warn(logger, f"[IWO] missing keys 示例: {missing[:6]}")
    if len(unexpected) > 0:
        _log_warn(logger, f"[IWO] unexpected keys 示例: {unexpected[:6]}")

    # 将所有增量权重（名字含 'weight1'）置零（通常是新引入的 Param）
    with torch.no_grad():
        n_zeros = 0
        for n, p in model.named_parameters():
            if "weight1" in n:
                p.zero_()
                n_zeros += 1
        logger.info(f"[IWO] 已将 {n_zeros} 个增量权重（*weight1）零初始化。")


def build_iwo_optimizer(model, tr_cfg, logger):
    """只收集并训练名字包含 'weight1' 的增量参数；门控 .u./.v. 可选降低 LR。"""
    lr_main = _as_float(tr_cfg.get("lr", 8e-5), 8e-5)
    gate_lr_scale = _as_float(tr_cfg.get("gate_lr_scale", 0.3), 0.3)
    iwo_only = bool(tr_cfg.get("iwo_only", True))

    main_params, gate_params = [], []
    for n, p in model.named_parameters():
        if iwo_only:
            # 先设置 requires_grad，仅放开增量和可选 gate
            if ("weight1" in n) or (".u." in n) or (".v." in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

        if not p.requires_grad:
            continue

        if "weight1" in n:
            main_params.append(p)
        elif (".u." in n) or (".v." in n):
            gate_params.append(p)

    if len(main_params) == 0 and len(gate_params) == 0:
        raise RuntimeError("[IWO] 没有可训练的参数：请确认模型非 slim（训练图），以及 *weight1 是否存在。")

    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": lr_main})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": lr_main * gate_lr_scale})

    logger.info(f"[IWO] 参数分组：weight1={len(main_params)}，gate={len(gate_params)}，lr={lr_main}, gate_lr_scale={gate_lr_scale}")
    return Adam(param_groups, lr=lr_main, weight_decay=0.0), (main_params, gate_params)


# -------------------- 评估 --------------------
@torch.no_grad()
def evaluate_psnr(model, loader, device, logger, shave=0):
    if loader is None:
        return None
    model.eval()
    vals = []
    for batch in loader:
        # 兼容不同的 batch 结构：期望返回 inp, gt, meta/idx
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                inp, gt = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            else:
                continue
        elif isinstance(batch, dict):
            inp = batch.get("inp", batch.get("input")).to(device, non_blocking=True)
            gt  = batch.get("gt", batch.get("target")).to(device, non_blocking=True)
        else:
            continue

        pred = model(inp).clamp(0, 1)
        if shave and min(pred.shape[-2:]) > 2 * shave and min(gt.shape[-2:]) > 2 * shave:
            pred = pred[..., shave:-shave, shave:-shave]
            gt   = gt[...,   shave:-shave, shave:-shave]

        vals.append(psnr_torch(pred, gt).item())

    m = float(np.mean(vals)) if len(vals) else 0.0
    logger.info(f"[Eval] PSNR={m:.4f} dB  (N={len(vals)})")
    return m


# -------------------- 训练主流程 --------------------
def train(opt):
    device = _device_from_opt(opt)
    logger = Logger(opt)

    cfg = getattr(opt, "config", {})
    tr_cfg = cfg.get("train", {})
    te_cfg = cfg.get("test", {})

    logger.info(f"CUDA 可用，将在 {'GPU' if device.type=='cuda' else 'CPU'} 上运行。")

    # 1) 数据
    train_loader, val_loader, _ = build_loaders(opt, logger)

    # 2) 模型 & 损失
    model = build_model(opt, logger).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params} / Trainable parameters: {trainable_params}")

    criterion = build_criterion(opt, logger)

    # 3) IWO：在已有常规权重上继续训练
    iwo_enable = bool(tr_cfg.get("iwo_enable", False))
    if iwo_enable:
        iwo_ckpt = tr_cfg.get("iwo_from_ckpt", "")
        load_base_ckpt_for_iwo(model, iwo_ckpt, device, logger)
        optimizer, (iwo_params, gate_params) = build_iwo_optimizer(model, tr_cfg, logger)
    else:
        # 原始训练：不启用 IWO
        base_lr = _as_float(tr_cfg.get("lr", 2e-4), 2e-4)
        optimizer = Adam(model.parameters(), lr=base_lr, weight_decay=0.0)

    # 4) 训练配置
    epochs = int(tr_cfg.get("epoch", 2000))
    save_every = int(tr_cfg.get("save_every", 20))
    warmup = bool(tr_cfg.get("warmup", True))
    warmup_epoch = int(tr_cfg.get("warmup_epoch", 1))
    lr_warmup = _as_float(tr_cfg.get("lr_warmup", 1e-6), 1e-6)

    # 输出目录
    exp_name = cfg.get("exp_name", "lle")
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    exp_dir = _ensure_dir(os.path.join("experiments", f"{stamp} train_{exp_name}"))
    model_dir = _ensure_dir(os.path.join(exp_dir, "models"))
    cfg_out = os.path.join(exp_dir, "config.yaml")

    # 保存一份运行时配置
    try:
        from option import save_yaml
        save_yaml(cfg_out, cfg)
    except Exception:
        pass

    # AMP & Grad clip
    scaler = GradScaler("cuda" in device.type)
    max_norm = float(tr_cfg.get("grad_clip", 1.0))

    # Warmup：仅学习率
    if warmup and warmup_epoch > 0:
        logger.info(f"start warming-up ({warmup_epoch} epochs, lr={lr_warmup})")
        for pg in optimizer.param_groups:
            pg["initial_lr"] = pg.get("lr", lr_warmup)
            pg["lr"] = lr_warmup

    best_psnr = -1e9
    best_path = None
    shave = int(te_cfg.get("shave", 0))

    # 预构建 IWO 正则的参数列表（避免每步遍历命名）
    iwo_reg = _as_float(tr_cfg.get("iwo_reg", 1e-4), 1e-4)
    iwo_reg_epochs = int(tr_cfg.get("iwo_reg_epochs", 80))
    iwo_reg_params = [p for n, p in model.named_parameters() if ("weight1" in n)]

    # 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        tbar = _tqdm(train_loader, total=len(train_loader), ncols=100, desc=f"[{epoch}/{epochs}]")
        loss_meter = []

        for batch in tbar:
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    inp, gt = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                else:
                    continue
            elif isinstance(batch, dict):
                inp = batch.get("inp", batch.get("input")).to(device, non_blocking=True)
                gt  = batch.get("gt",  batch.get("target")).to(device, non_blocking=True)
            else:
                continue

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
                pred = model(inp)
                rec_loss = criterion(pred, gt) if callable(criterion) else torch.nn.functional.l1_loss(pred, gt)

                loss = rec_loss

                # IWO：前若干 epoch 对增量权重做 L2 正则
                if iwo_enable and (epoch <= iwo_reg_epochs) and (iwo_reg > 0.0) and len(iwo_reg_params) > 0:
                    reg_val = torch.zeros((), device=pred.device)
                    for p in iwo_reg_params:
                        if p.requires_grad and p.dtype in (torch.float16, torch.float32, torch.bfloat16):
                            reg_val = reg_val + (p.float() ** 2).sum()
                    loss = loss + iwo_reg * reg_val

            scaler.scale(loss).backward()
            if max_norm is not None and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()

            loss_meter.append(loss.detach().float().item())
            tbar.set_postfix(loss=np.mean(loss_meter))

        # Warmup 结束，恢复到初始 LR
        if warmup and epoch == warmup_epoch:
            for pg in optimizer.param_groups:
                base_lr = pg.get("initial_lr", pg["lr"])
                pg["lr"] = base_lr
            logger.info("warmup 结束，恢复学习率。")

        # 验证 & 保存
        do_validate = (val_loader is not None) and ((epoch % save_every == 0) or (epoch == epochs))
        if do_validate:
            val_psnr = evaluate_psnr(model, val_loader, device, logger, shave=shave)
            if val_psnr is not None and val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(model_dir, "best.pkl")
                torch.save({"model": model.state_dict(), "epoch": epoch, "psnr": best_psnr}, best_path)
                logger.info(f"[Save] best @ epoch {epoch}: PSNR={best_psnr:.4f} → {best_path}")

        # 周期性保存
        if epoch % save_every == 0:
            ckpt_path = os.path.join(model_dir, f"epoch_{epoch:04d}.pkl")
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
            logger.info(f"[Save] {ckpt_path}")

    logger.info(f"[Done] best_psnr={best_psnr:.4f}, best_path={best_path}")
    return best_psnr, best_path, exp_dir


# -------------------- 测试主流程 --------------------
@torch.no_grad()
def test(opt):
    device = _device_from_opt(opt)
    logger = Logger(opt)

    cfg = getattr(opt, "config", {})
    te_cfg = cfg.get("test", {})

    logger.info(f"CUDA 可用，将在 {'GPU' if device.type=='cuda' else 'CPU'} 上运行。")

    # 数据（只要 val/test 任一）
    _, val_loader, test_loader = build_loaders(opt, logger)
    loader = test_loader if test_loader is not None else val_loader
    if loader is None:
        raise RuntimeError("[Test] 未找到可用的数据加载器（val/test）。")

    # 模型
    model = build_model(opt, logger).to(device)

    # 选择 ckpt
    ckpt_path = te_cfg.get("ckpt", "")
    if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
        _log_warn(logger, f"[Model] 未提供 test.ckpt 或文件不存在：{ckpt_path}")
        # 兼容性：尝试从 experiments 下自动找一个最新的 best.pkl
        latest = None
        base = "experiments"
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for f in files:
                    if f == "best.pkl":
                        p = os.path.join(root, f)
                        if (latest is None) or (os.path.getmtime(p) > os.path.getmtime(latest)):
                            latest = p
        if latest is None:
            raise RuntimeError("[Model] 找不到可用的权重文件，请设置 test.ckpt。")
        ckpt_path = latest

    logger.info(f"[Model] Load checkpoint for TEST from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = None
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "net", "ema", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]; break
    if sd is None:
        sd = ckpt
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        _log_warn(logger, f"[Model] Missing keys: {missing[:6]}")
    if len(unexpected) > 0:
        _log_warn(logger, f"[Model] Unexpected keys: {unexpected[:6]}")

    shave = int(te_cfg.get("shave", 0))
    psnr = evaluate_psnr(model, loader, device, logger, shave=shave)
    return psnr


# -------------------- 入口 --------------------
def main():
    opt = get_option()
    task = getattr(opt, "task", "train")
    if task == "train":
        train(opt)
    elif task == "test":
        test(opt)
    elif task == "demo":
        # 这里保留与原流程兼容的占位
        print("[Demo] 请在你的仓库里实现 demo 逻辑或改为 test/train。")
    else:
        print(f"[Error] 未知 task: {task}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        sys.exit(1)
