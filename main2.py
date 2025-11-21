import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import sys
import time
import json
import traceback
import numpy as np
from tqdm import tqdm
from datetime import datetime
import math

import torch
from torch.optim import Adam
from torch.amp import autocast, GradScaler
import torch.nn as nn

import cv2  # 用于保存预测图像

from logger import Logger
from option import get_option
from data import import_loader
from loss import import_loss
from model import import_model

from loss_color import ColorConsistencyLoss, rgb_to_lab  # 颜色一致性

# -------------------- 日志兼容封装 --------------------
def _log_warn(logger, msg: str):
    if hasattr(logger, "warning"):
        logger.warning(msg)
    elif hasattr(logger, "warn"):
        logger.warn(msg)
    else:
        logger.info(f"[WARN] {msg}")

# -------------------- 进度条包装 --------------------
def _tqdm(iterable, **kwargs):
    disable_env = os.environ.get("TQDM_DISABLE", "")
    kwargs.setdefault("disable", (not sys.stdout.isatty()) or bool(disable_env))
    kwargs.setdefault("leave", False)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(iterable, **kwargs)


def load_checkpoint(model, checkpoint_path, device, logger):
    """加载指定路径的 checkpoint 权重 (自动剔除形状不匹配的参数)"""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"加载 checkpoint 权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取 checkpoint 中的 state_dict
        if 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint
            
        # 获取当前模型的 state_dict 信息
        model_state = model.state_dict()
        
        # ★★★ 智能过滤：构建一个新的 state_dict，剔除形状不匹配的参数 ★★★
        filtered_sd = {}
        mismatched_keys = []
        
        for k, v in sd.items():
            if k in model_state:
                # 检查形状是否一致
                if v.shape == model_state[k].shape:
                    filtered_sd[k] = v
                else:
                    # 形状不匹配（例如 att.fc.weight），跳过加载
                    mismatched_keys.append(f"{k} ({v.shape} -> {model_state[k].shape})")
            else:
                # 多余的键，保留给 load_state_dict 报 unexpected（或者直接忽略）
                filtered_sd[k] = v
        
        if len(mismatched_keys) > 0:
            logger.info(f"[Load Skip] 因形状不匹配跳过加载的参数: {mismatched_keys}")
        
        # 使用过滤后的字典加载，strict=False 允许缺失
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        
        # 打印日志
        if len(missing) > 0:
            # 这里的 missing 应该包含 att.fc.weight 等被我们剔除的键
            logger.info(f"[Load Warn] 缺失键值 (将使用代码默认初始化): {missing}")
        # if len(unexpected) > 0:
        #     logger.info(f"[Load Warn] 多余键值: {unexpected}")
            
        logger.info("权重加载成功 (已自动处理形状冲突)")
    else:
        logger.warning(f"未找到指定的 checkpoint 权重文件: {checkpoint_path}")

# -------------------- 小工具 --------------------
def _maybe_set_cudnn(opt, logger):
    device = getattr(opt, "device", "cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA 可用，将在 GPU 上运行。")
    else:
        _log_warn(logger, "CUDA 不可用或 device 不是 'cuda'，当前在 CPU 上运行。")

def _as_float(x, default_val):
    try:
        return float(x)
    except Exception:
        return float(default_val)

def _as_int(x, default_val):
    try:
        return int(x)
    except Exception:
        return int(default_val)

def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def _to_device_maybe(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x

def _looks_like_names(x):
    if isinstance(x, str): return True
    if isinstance(x, (list, tuple)) and (len(x) == 0 or isinstance(x[0], (str, bytes))):
        return True
    return False

def _unpack_batch(batch, device):
    inp = gt = name = None
    if isinstance(batch, dict):
        inp  = batch.get("inp") or batch.get("lr")
        gt   = batch.get("gt")  or batch.get("hr")
        name = batch.get("name", None)
        if gt is not None and not isinstance(gt, torch.Tensor) and _looks_like_names(gt):
            name, gt = gt, None
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            inp, gt, name = batch
        elif len(batch) == 2:
            inp, x = batch
            if isinstance(x, torch.Tensor) or hasattr(x, "shape"):
                gt = x
            elif _looks_like_names(x):
                name = x
            else:
                name = x
        elif len(batch) == 1:
            inp = batch[0]
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
    else:
        inp = batch
    inp = _to_device_maybe(inp, device)
    if isinstance(gt, torch.Tensor):
        gt = gt.to(device, non_blocking=True)
    return inp, gt, name

def _split_loaders(loaders, task: str):
    """返回: (train_loader, val_loader, test_loader)"""
    tr = vl = te = None
    if isinstance(loaders, dict):
        tr = loaders.get('train') or loaders.get('tr')
        vl = loaders.get('val') or loaders.get('valid')
        te = loaders.get('test') or loaders.get('demo')
        if task in ('test', 'demo') and te is None:
            te = vl
    elif isinstance(loaders, (list, tuple)):
        if len(loaders) == 3:
            tr, vl, te = loaders
        elif len(loaders) == 2:
            tr, vl = loaders
            if task in ('test', 'demo'):
                te = vl
        elif len(loaders) == 1:
            if task in ('test', 'demo'):
                te = loaders[0]
            else:
                tr = loaders[0]
        else:
            raise ValueError(f"Unexpected number of loaders: {len(loaders)}")
    else:
        if task in ('test', 'demo'):
            te = loaders
        else:
            tr = loaders
    return tr, vl, te

# === 8-bit PSNR（clamp→×255→round→uint8） ===
def _to_uint8_rgb(t: torch.Tensor):
    # CHW [0,1] -> HWC uint8
    arr = t.detach().clamp(0, 1).cpu().numpy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.rint(arr * 255.0).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))

def _psnr_uint8(pred_u8: np.ndarray, gt_u8: np.ndarray):
    diff = pred_u8.astype(np.float64) - gt_u8.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse <= 1e-12: return 99.0
    return 10.0 * math.log10((255.0 ** 2) / mse)

def _save_predictions(pred: torch.Tensor, names, save_dir: str, prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    b = pred.size(0)
    for i in range(b):
        if isinstance(names, (list, tuple)) and i < len(names) and names[i] is not None:
            base = str(names[i]); base = os.path.splitext(base)[0]
        elif isinstance(names, str):
            base = os.path.splitext(names)[0]
        else:
            base = f"img_{i:06d}"
        out_path = os.path.join(save_dir, f"{prefix}{base}.png")
        img = _to_uint8_rgb(pred[i])[:, :, ::-1]  # RGB->BGR
        cv2.imwrite(out_path, img)

# -------------------- 权重一致性自检 --------------------
def _state_looks_original(sd_keys):
    keys = list(sd_keys)
    return any(('running_mean' in k or 'running_var' in k or 'bn' in k or 'branch' in k) for k in keys)

def _safe_load_ckpt(ckpt_path, device, expect_reparam, logger, strict=False):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict', sd.get('model'))
    if expect_reparam and _state_looks_original(sd.keys()):
        raise RuntimeError(
            f"检测到你在 re-parameterized 架构下加载的是 original 权重：{ckpt_path}\n"
            f"请改用通过 slim() 导出的 *_slim.pkl（例如 best_slim.pkl）。"
        )
    if (not expect_reparam) and (not _state_looks_original(sd.keys())):
        _log_warn(logger, f"你似乎在 original 架构下加载了 slim 权重：{ckpt_path}（一般不推荐，可能会有缺键）")
    return sd, strict

# -------------------- 训练增强：冻结BN / 门控恒等正则 / 参数分组 --------------------
def freeze_bn(m):
    """验证/推理期冻结 BN 统计"""
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

def gate_identity_reg(model: nn.Module):
    """
    对门控FST/FSTS的 u/v 做“靠近恒等”的小正则
    假设门控层具备 .u / .v 两个 1x1_dw 卷积
    """
    reg = 0.0
    for mod in model.modules():
        # 检查是否是 FST/FSTS 模块 (拥有 u 和 v 属性)
        if hasattr(mod, 'u') and hasattr(mod, 'v') and hasattr(mod.u, 'weight') and hasattr(mod.v, 'weight'):
            # 目标：让 u(x) 和 v(x) 的权重接近 1，偏置接近 0
            # 这样 FST 初始行为接近 y * y (二次项)，保持数值稳定
            reg += (mod.u.weight - 1).abs().mean() + (mod.v.weight - 1).abs().mean()
            if hasattr(mod.u, 'bias') and mod.u.bias is not None:
                reg += mod.u.bias.abs().mean()
            if hasattr(mod.v, 'bias') and mod.v.bias is not None:
                reg += mod.v.bias.abs().mean()
    return reg

# -------------------- EMA --------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert n in self.shadow
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[n] = p.data.clone()
            p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[n])
        self.backup = {}

# -------------------- 历史归档 --------------------
def _ensure_history(best_dir: str, logger):
    # 获取 log_path，并找到上一级目录
    log_path = logger.log_path  # 获取与日志文件相同的目录
    parent_dir = os.path.dirname(log_path)  # 获取上一级目录
    history_dir = os.path.join(parent_dir, "history")  # 将 history 放在上一级目录下
    os.makedirs(history_dir, exist_ok=True)
    meta_path = os.path.join(history_dir, "best_history.jsonl")
    return history_dir, meta_path

def _archive_stub(epoch, psnr):
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"best_E{epoch}_PSNR{psnr:.3f}_{ts}"

def _append_best_meta(meta_path, record: dict, logger):
    try:
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        _log_warn(logger, f"[HISTORY] write meta failed: {e}")

# -------------------- 学习率调度器 --------------------
def _get_scheduler(optimizer, tr_cfg, logger):
    sch = (tr_cfg.get("scheduler") or "cosine").lower()
    if sch not in ("cosine", "cosine_wr", "plateau", "step"):
        _log_warn(logger, f"未知 scheduler={sch}，回退到 cosine。")
        sch = "cosine"

    if sch == "cosine":
        T = _as_int(tr_cfg.get("cosine_T", 800), 800)            # 总周期（近似），无重启
        eta_min = _as_float(tr_cfg.get("eta_min", 1e-7), 1e-7)
        logger.info(f"[LR] scheduler=cosine(T={T}, eta_min={eta_min})")
        return sch, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T, eta_min=eta_min)

    if sch == "cosine_wr":
        T0 = _as_int(tr_cfg.get("cosine_T0", 700), 700)          # 首次重启
        Tmult = _as_int(tr_cfg.get("cosine_Tmult", 2), 2)
        eta_min = _as_float(tr_cfg.get("eta_min", 1e-7), 1e-7)
        logger.info(f"[LR] scheduler=cosine_wr(T0={T0}, Tmult={Tmult}, eta_min={eta_min})")
        return sch, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tmult, eta_min=eta_min)

    if sch == "plateau":
        patience = _as_int(tr_cfg.get("plateau_patience", 20), 20)
        factor   = _as_float(tr_cfg.get("plateau_factor", 0.5), 0.5)
        min_lr   = _as_float(tr_cfg.get("eta_min", 1e-7), 1e-7)
        logger.info(f"[LR] scheduler=plateau(patience={patience}, factor={factor}, min_lr={min_lr})")
        return sch, torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr, verbose=True)

    # step
    step_size = _as_int(tr_cfg.get("step_size", 200), 200)
    gamma     = _as_float(tr_cfg.get("gamma", 0.5), 0.5)
    logger.info(f"[LR] scheduler=step(step_size={step_size}, gamma={gamma})")
    return sch, torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def _get_lr(optimizer):
    for g in optimizer.param_groups:
        return g.get("lr", None)

# -------------------- 验证 --------------------
def validate(model, val_loader, device, logger, color_loss=None, max_eval: int = 0):
    model.eval()
    model.apply(freeze_bn)

    psnr_list, last_pred = [], None
    with torch.no_grad():
        n = 0
        for batch in _tqdm(val_loader, desc="[VAL]"):
            inp, gt, _ = _unpack_batch(batch, device)
            pred = model(inp).clamp(0, 1)
            last_pred = pred

            pu8 = _to_uint8_rgb(pred[0])
            gu8 = _to_uint8_rgb(gt[0])
            psnr_list.append(_psnr_uint8(pu8, gu8))

            n += 1
            if max_eval and n >= max_eval:
                break

        if color_loss is not None and last_pred is not None:
            _, a_, b_ = rgb_to_lab(last_pred)
            a_mean_cv = (a_ + 128.0).mean().item()
            b_mean_cv = (b_ + 128.0).mean().item()
            logger.info(f"[VAL] ā={a_mean_cv:.1f}, b̄={b_mean_cv:.1f}")

    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    logger.info(f"[VAL] PSNR: {mean_psnr:.3f} dB")
    return mean_psnr

# -------------------- 推理 --------------------
@torch.no_grad()
def infer_and_maybe_save(model, data_loader, device, logger, save_images=False, save_dir=None, color_loss=None, prefix=""):
    model.eval()
    model.apply(freeze_bn)
    psnr_list = []
    last_pred = None
    for batch in _tqdm(data_loader, desc="[INFER]"):
        inp, gt, name = _unpack_batch(batch, device)
        pred = model(inp).clamp(0, 1)
        last_pred = pred
        if save_images and save_dir:
            names = name if isinstance(name, (list, tuple, str)) else [None] * pred.size(0)
            _save_predictions(pred, names, save_dir, prefix=prefix)
        if gt is not None:
            pu8 = _to_uint8_rgb(pred[0])
            gu8 = _to_uint8_rgb(gt[0])
            psnr_list.append(_psnr_uint8(pu8, gu8))
    if color_loss is not None and last_pred is not None:
        _, a_, b_ = rgb_to_lab(last_pred)
        a_mean_cv = (a_ + 128.0).mean().item()
        b_mean_cv = (b_ + 128.0).mean().item()
        logger.info(f"[INFER] ā={a_mean_cv:.1f}, b̄={b_mean_cv:.1f}")
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    if psnr_list:
        logger.info(f"[INFER] Mean PSNR: {mean_psnr:.3f} dB")
    return mean_psnr

# -------------------- 训练 --------------------
def train():
    opt = get_option()
    logger = Logger(opt)
    _maybe_set_cudnn(opt, logger)

    train_loader, val_loader = import_loader(opt)
    if train_loader is None:
        raise RuntimeError("import_loader 未返回 train_loader，无法训练。")

    model = import_model(opt)
   # 从 YAML 配置文件中获取 checkpoint 路径
    ckpt = opt.config.get("train", {}).get("ckpt", "")
    load_checkpoint(model, ckpt, opt.device, logger)  # 加载 checkpoint 权重
    # 参数量
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    logger.info(f"Total number of parameters: {total} / Trainable parameters: {trainable}")

    tr = opt.config.get("train", {})
    use_warmup   = bool(tr.get("warmup", False))
    warm_epochs  = _as_int(tr.get("warmup_epoch", 0), 0)
    lr_warm      = _as_float(tr.get("lr_warmup", 1e-6), 1e-6)

    # 训练控制项
    val_every    = max(1, _as_int(tr.get("val_every", 5), 5))
    use_ema      = bool(tr.get("ema", False))
    ema_decay    = _as_float(tr.get("ema_decay", 0.999), 0.999)

    # CPU 快速验证
    fast_eval_when_cpu = bool(tr.get("fast_eval_when_cpu", False))
    fast_eval_max      = _as_int(tr.get("fast_eval_max", 0), 0)

    # ★ 持久化的 GradScaler（warmup 与主训练共用）
    scaler = GradScaler(enabled=opt.device.startswith("cuda"))

    # ====== Warmup（若模型实现了 forward_warm）======
    if use_warmup and warm_epochs > 0 and hasattr(model, "forward_warm"):
        logger.info(f"start warming-up ({warm_epochs} epochs, lr={lr_warm:g})")
        optim_warm = Adam(model.parameters(), lr=lr_warm, weight_decay=0.0)
        warm_crit  = import_loss('warmup').to(opt.device)
        for e in range(1, warm_epochs + 1):
            model.train()
            # ★★★ 强制冻结BN（Warmup阶段）★★★
            model.apply(freeze_bn)
            
            running = []
            for batch in _tqdm(train_loader, desc=f"[WARM {e}/{warm_epochs}]"):
                inp, gt, _ = _unpack_batch(batch, opt.device)
                optim_warm.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", dtype=torch.float16, enabled=opt.device.startswith("cuda")):
                    out1, out2 = model.forward_warm(inp)
                    loss = warm_crit(inp, gt, out1, out2)
                if not torch.isfinite(loss):
                    _log_warn(logger, "[skip step][warmup] non-finite loss")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optim_warm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim_warm)
                scaler.update()
                running.append(loss.item())
            logger.info(f"[WARM {e}] train_loss={np.mean(running):.6f}")
        logger.info("warming-up phase done")
    elif use_warmup and not hasattr(model, "forward_warm"):
        _log_warn(logger, "YAML 启用了 warmup，但模型无 forward_warm，已跳过 warmup。")

    # ====== 主训练（UIEDP 仍走你的 loss.py；这里叠加颜色一致性）======
    lr_main = _as_float(tr.get("lr", 2e-4), 2e-4)

    # 参数分组：门控 .u. / .v. 使用更小 LR
    gate_lr_scale   = _as_float(tr.get("gate_lr_scale", 0.3), 0.3)
    gate_reg_base   = _as_float(tr.get("gate_reg", 1e-4), 1e-4)
    gate_reg_epochs = _as_int(tr.get("gate_reg_epochs", 80), 80)
    gate_reg_tail   = _as_int(tr.get("gate_reg_tail", 80), 80)      # ★ 新增：尾部平滑衰减长度
    gate_reg_floor  = _as_float(tr.get("gate_reg_floor", 1e-6), 1e-6)  # ★ 新增：最低保持值

    gate_params, other_params = [], []
    for n, p in model.named_parameters():
        if ('.u.' in n) or ('.v.' in n):
            gate_params.append(p)
        else:
            other_params.append(p)

    optimizer = Adam([
        {'params': other_params, 'lr': lr_main},
        {'params': gate_params,  'lr': max(lr_main * gate_lr_scale, 1e-7)},
    ], weight_decay=0.0)

    # 调度器（可选）
    sch_name, lr_sch = _get_scheduler(optimizer, tr, logger)

    criterion_main = import_loss(opt.model_task).to(opt.device)

    # 颜色一致性（窗口门控 + warm-up + 总系数 scale）
    lc = opt.config.get("loss", {}).get("color", {})
    use_color = bool(lc.get("enable", True))
    color_warm_epochs = _as_int(lc.get("warmup_epoch", 10), 10)
    color_scale = _as_float(lc.get("scale", 1.0), 1.0)
    color_loss = ColorConsistencyLoss(
        a_window=tuple(lc.get("a_window", [126.0, 134.0])),
        b_window=tuple(lc.get("b_window", [128.0, 140.0])),
        var_band=tuple(lc.get("var_band", [40.0, 1200.0])),
        w_lab_mean=_as_float(lc.get("w_lab_mean", 0.08), 0.08),
        w_lab_var=_as_float(lc.get("w_lab_var", 0.02), 0.02),
        w_rgb_stat=_as_float(lc.get("w_rgb_stat", 0.03), 0.03),
        mu_gap=_as_float(lc.get("mu_gap", 0.08), 0.08),
        rho_min=_as_float(lc.get("rho_min", 0.10), 0.10),
    ).to(opt.device)

    # EMA
    use_ema_flag = use_ema
    ema = EMA(model, decay=ema_decay) if use_ema_flag else None

    epochs = _as_int(tr.get("epoch", 2000), 2000)
    save_every = _as_int(tr.get("save_every", 20), 20)

    # 保存目录（保持原风格）
    save_root = os.path.join(os.getcwd(), "experiments", f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')} train_{opt.model_task}")
    os.makedirs(save_root, exist_ok=True)
    best_dir = os.path.join(save_root, "models")
    os.makedirs(best_dir, exist_ok=True)
    best_path = os.path.join(best_dir, "best.pkl")
    history_dir, meta_path = _ensure_history(best_dir, logger)

    best_psnr = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        
        # ★★★ 关键修改：在微调阶段强制冻结 BN 统计量 ★★★
        # 防止小 BatchSize 破坏预训练模型的统计分布
        model.apply(freeze_bn)
        
        running = []
        pbar = _tqdm(train_loader, desc=f"[{epoch}/{epochs}]")
        for it, batch in enumerate(pbar, start=1):
            inp, gt, _ = _unpack_batch(batch, opt.device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16, enabled=opt.device.startswith("cuda")):
                pred = model(inp).clamp(0, 1)
                rec_loss = criterion_main(pred, gt)

                # 颜色项
                if use_color:
                    cw = 1.0 if epoch > color_warm_epochs else float(epoch) / max(1, color_warm_epochs)
                    col_total, _ = color_loss(pred)
                    loss = rec_loss + color_scale * cw * col_total
                else:
                    loss = rec_loss

                # 门控“恒等”正则 —— 平滑衰减
                if gate_reg_base > 0:
                    if epoch <= gate_reg_epochs:
                        lam = gate_reg_base
                    else:
                        if gate_reg_tail > 0:
                            t = min(epoch - gate_reg_epochs, gate_reg_tail) / float(gate_reg_tail)
                            lam = gate_reg_base * (1.0 - t) + gate_reg_floor
                        else:
                            lam = gate_reg_floor
                    loss = loss + lam * gate_identity_reg(model)

            if not torch.isfinite(loss):
                _log_warn(logger, "[skip step] loss 非有限")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if use_ema_flag and ema is not None:
                ema.update(model)

            running.append(loss.item())

        # ===== 验证（按频率 / CPU 快速评测） =====
        do_validate = (epoch == 1) or (epoch % val_every == 0) or (epoch == epochs)
        val_psnr = float('-inf')
        # ★ 用于保存验证时的完整模型状态
        validation_state_dict = None
        
        if val_loader is not None and do_validate:
            max_eval = fast_eval_max if (fast_eval_when_cpu and not opt.device.startswith("cuda")) else 0

            if use_ema_flag and ema is not None:
                ema.apply_to(model)
                val_psnr = validate(model, val_loader, opt.device, logger,
                                    color_loss if use_color else None,
                                    max_eval=max_eval)
                # ★ 保存验证时的完整状态（EMA参数 + 此时的BN统计）
                validation_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                ema.restore(model)
            else:
                val_psnr = validate(model, val_loader, opt.device, logger,
                                    color_loss if use_color else None,
                                    max_eval=max_eval)
                validation_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

            logger.info(f"[E{epoch}] lr={_get_lr(optimizer):.6g}, train_loss={np.mean(running):.4f}, val_psnr={val_psnr:.3f}")
        elif val_loader is None:
            logger.info(f"[E{epoch}] lr={_get_lr(optimizer):.6g}, train_loss={np.mean(running):.4f}, no val_loader")

        # ===== 学习率调度步进 =====
        try:
            if sch_name == "plateau":
                # PSNR 越大越好
                if val_psnr != float('-inf'):
                    lr_sch.step(val_psnr)
            else:
                lr_sch.step()
        except Exception:
            pass

        # ===== 保存最好（best + history + slim + JSONL） =====
        if val_psnr != float('-inf') and val_psnr > best_psnr:
            best_psnr = val_psnr

            # ★ 使用验证时保存的完整状态（包括EMA参数和对应的BN统计）
            if validation_state_dict is not None:
                sd_to_save = validation_state_dict
            else:
                logger.warning("[SAVE] validation_state_dict is None, using current model state")
                sd_to_save = model.state_dict()

            # 保存最新 best（覆盖）
            torch.save(sd_to_save, best_path)
            logger.info(f"[BEST] {best_path}  PSNR={best_psnr:.3f}")

            # 历史归档（不覆盖）
            stub = _archive_stub(epoch, val_psnr)
            hist_path = os.path.join(history_dir, f"{stub}.pkl")
            torch.save(sd_to_save, hist_path)
            logger.info(f"[BEST] history archived: {hist_path}")

            # 按 YAML 开关导出 slim（含历史）
            slim_latest = None
            slim_hist = None
            try:
                if opt.config.get("train", {}).get("save_slim", False) and hasattr(model, "slim"):
                    # 如果使用了EMA且当前不在EMA状态下（在apply_to之外），需要临时切到EMA状态导出slim
                    if use_ema_flag and ema is not None:
                        ema.apply_to(model)
                        slim_obj = model.slim()
                        ema.restore(model)
                    else:
                        slim_obj = model.slim()
                    slim_sd = slim_obj if isinstance(slim_obj, dict) else slim_obj.state_dict()

                    slim_latest = os.path.join(best_dir, "best_slim.pkl")
                    torch.save(slim_sd, slim_latest)
                    logger.info(f"[BEST] 导出 slim: {slim_latest}")

                    slim_hist = os.path.join(history_dir, f"{stub}_slim.pkl")
                    torch.save(slim_sd, slim_hist)
                    logger.info(f"[BEST] slim history archived: {slim_hist}")
            except Exception as e:
                _log_warn(logger, f"[SAVE slim] failed: {e}")
                slim_latest = None
                slim_hist = None

            # 记录元数据（JSONL）
            _append_best_meta(
                meta_path,
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": int(epoch),
                    "val_psnr": float(val_psnr),
                    "train_loss_mean": float(np.mean(running)) if running else None,
                    "latest": os.path.basename(best_path),
                    "history": os.path.basename(hist_path),
                    "latest_slim": os.path.basename(slim_latest) if slim_latest else None,
                    "history_slim": os.path.basename(slim_hist) if slim_hist else None,
                },
                logger
            )

        # 周期快照
        if save_every and (epoch % save_every == 0):
            ep_path = os.path.join(best_dir, f"epoch_{epoch:04d}.pkl")
            torch.save(model.state_dict(), ep_path)
            logger.info(f"[SAVE] {ep_path}")

    logger.info(f"训练完成。最佳 PSNR: {best_psnr:.3f} | 权重: {best_path}")

# -------------------- 测试 --------------------
def test():
    opt = get_option()
    logger = Logger(opt)
    _maybe_set_cudnn(opt, logger)

    test_loader = import_loader(opt)
    if test_loader is None:
        raise RuntimeError("import_loader 未提供 test_loader。")

    model = import_model(opt)

    ckpt = opt.config.get("test", {}).get("ckpt", "")
    if ckpt and os.path.isfile(ckpt):
        logger.info(f"[TEST] load checkpoint: {ckpt}")
        sd, strict = _safe_load_ckpt(
            ckpt, opt.device,
            expect_reparam=(opt.config.get('model', {}).get('type') == 're-parameterized'),
            logger=logger, strict=False
        )
        model.load_state_dict(sd, strict=strict)

    model.eval()
    model.apply(freeze_bn)
    psnr_list = []
    
    # 根据任务类型读取 save 配置和输出目录
    task = opt.task.lower()
    timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    
    if task == 'demo':
        save_flag = bool(opt.config.get('demo', {}).get('save', True))  # demo 默认保存
        out_dir = os.path.join(os.getcwd(), "experiments", f"{timestamp} demo_{opt.model_task}")
    else:
        save_flag = bool(opt.config.get('test', {}).get('save', False))
        out_dir = os.path.join(os.getcwd(), "experiments", f"{timestamp} test_{opt.model_task}")
    
    if save_flag: os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for batch in _tqdm(test_loader, desc="[TEST]"):
            inp, gt, name = _unpack_batch(batch, opt.device)
            pred = model(inp).clamp(0, 1)

            # 保存图像（可选）
            if save_flag:
                names = name if isinstance(name, (list, tuple, str)) else [None] * pred.size(0)
                _save_predictions(pred, names, out_dir, prefix="")

            # 8-bit PSNR
            if gt is not None:
                pu8 = _to_uint8_rgb(pred[0])
                gu8 = _to_uint8_rgb(gt[0])
                psnr_list.append(_psnr_uint8(pu8, gu8))

    if psnr_list:
        mean_psnr = float(np.mean(psnr_list))
        logger.info(f"[TEST] 平均 PSNR: {mean_psnr:.4f} dB；输出目录: {out_dir}")
        print(f"[TEST] 平均 PSNR: {mean_psnr:.4f} dB；输出目录: {out_dir}")
    else:
        if save_flag:
            logger.info(f"[DEMO] 推理完成，输出图像已保存至: {out_dir}")
            print(f"[DEMO] 推理完成，输出图像已保存至: {out_dir}")
        else:
            logger.info("[TEST] 无 GT，未计算 PSNR。")

# -------------------- 入口 --------------------
if __name__ == "__main__":
    try:
        opt = get_option()
        task = opt.task.lower()
        if task == "train":
            train()
        elif task in ("test", "val", "eval"):
            test()
        elif task == "demo":
            # 复用 test 的推理保存与 8-bit 评测流程
            test()
        else:
            raise ValueError(f"Unknown task: {opt.task}")
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
