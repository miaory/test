# """
# IWO (Incremental Weight Optimization) 训练脚本
# 阶段二训练：在已训练好的模型基础上，启用IWO进行增量优化
# 目标：PSNR 从 22.x → 23+
# """

# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# import sys
# import time
# import json
# import traceback
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import math

# import torch
# from torch.optim import Adam
# from torch.amp import autocast, GradScaler
# import torch.nn as nn

# import cv2

# from logger import Logger
# from option import get_option
# from data import import_loader
# from loss import import_loss
# from model import import_model
# from loss_color import ColorConsistencyLoss, rgb_to_lab

# # IWO工具
# from enable_iwo import (
#     enable_iwo_for_model,
#     fuse_iwo,
#     get_iwo_param_groups,
#     save_iwo_checkpoint,
#     load_iwo_checkpoint
# )

# # 复用main2.py中的工具函数
# from main2 import (
#     _log_warn,
#     _tqdm,
#     _maybe_set_cudnn,
#     _as_float,
#     _as_int,
#     count_parameters,
#     _to_device_maybe,
#     _looks_like_names,
#     _unpack_batch,
#     _split_loaders,
#     _to_uint8_rgb,
#     _psnr_uint8,
#     _save_predictions,
#     freeze_bn,
#     gate_identity_reg,
#     EMA,
#     _ensure_history,
#     _archive_stub,
#     _append_best_meta,
#     _get_scheduler,
#     _get_lr,
#     validate,
#     infer_and_maybe_save,
# )


# def train_iwo():
#     """IWO阶段训练主函数"""
#     opt = get_option()
    
#     # ★ 强制加载 IWO 配置文件
#     import yaml
#     iwo_config_path = os.path.join(opt.root, 'config', f'{opt.model_task}_iwo.yaml')
#     if os.path.isfile(iwo_config_path):
#         with open(iwo_config_path, 'r', encoding='utf-8') as f:
#             opt.config = yaml.load(f, Loader=yaml.FullLoader)
#         print(f"[IWO] 已加载 IWO 配置: {iwo_config_path}")
#     else:
#         print(f"[IWO] 警告: 未找到 {iwo_config_path}，使用默认配置")
    
#     logger = Logger(opt)
#     _maybe_set_cudnn(opt, logger)
    
#     logger.info("="*60)
#     logger.info("IWO (Incremental Weight Optimization) 训练阶段")
#     logger.info("="*60)
    
#     # ========== 1. 加载数据 ==========
#     train_loader, val_loader = import_loader(opt)
#     if train_loader is None:
#         raise RuntimeError("import_loader 未返回 train_loader，无法训练。")
    
#     # ========== 2. 加载已训练好的模型 ==========
#     model = import_model(opt)
    
#     # 从配置中获取预训练权重路径
#     tr_cfg = opt.config.get("train", {})
#     pretrained_ckpt = tr_cfg.get("pretrained_ckpt", "")
    
#     if not pretrained_ckpt or not os.path.isfile(pretrained_ckpt):
#         raise FileNotFoundError(
#             f"IWO训练需要预训练权重！请在YAML中设置 train.pretrained_ckpt\n"
#             f"当前路径: {pretrained_ckpt}"
#         )
    
#     logger.info(f"[IWO] 加载预训练权重: {pretrained_ckpt}")
#     checkpoint = torch.load(pretrained_ckpt, map_location=opt.device)
#     if 'state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['state_dict'])
#     else:
#         model.load_state_dict(checkpoint)
#     logger.info("[IWO] 预训练权重加载成功")
    
#     # ========== 3. 启用IWO ==========
#     iwo_cfg = tr_cfg.get("iwo", {})
#     iwo_config = {
#         'init_std': _as_float(iwo_cfg.get('init_std', 0.0), 0.0),
#         'target_layers': iwo_cfg.get('target_layers', None),  # None=所有MBRConv
#     }
    
#     model, iwo_params = enable_iwo_for_model(model, iwo_config)
#     model = model.to(opt.device)
    
#     # 参数统计
#     trainable = count_parameters(model, trainable_only=True)
#     total = count_parameters(model, trainable_only=False)
#     iwo_param_count = sum(p.numel() for p in iwo_params)
#     logger.info(f"Total parameters: {total} / Trainable: {trainable}")
#     logger.info(f"IWO parameters: {iwo_param_count} ({iwo_param_count/trainable*100:.2f}%)")
    
#     # ========== 4. 构建优化器（参数分组）==========
#     # ★★★ Epoch 3达到峰值后崩溃，使用极低学习率 ★★★
#     lr_main = 1e-5  # 从 3e-5 进一步降到 1e-5（极低）
#     iwo_lr_scale = 0.1  # 从 0.2 降到 0.1（IWO参数LR = 1e-6）
#     gate_lr_scale = 0.1  # 从 0.2 降到 0.1
    
#     # 调试：打印配置
#     logger.info(f"[HARDCODED v2] lr_main={lr_main}, iwo_lr_scale={iwo_lr_scale}, gate_lr_scale={gate_lr_scale}")
#     logger.info(f"[EFFECTIVE LR] IWO={lr_main*iwo_lr_scale:.2e}, Gate={lr_main*gate_lr_scale:.2e}, Backbone={lr_main:.2e}")
#     logger.info(f"[CONFIG READ] lr_iwo={tr_cfg.get('lr_iwo')}, iwo.lr_scale={iwo_cfg.get('lr_scale')}")
    
#     # 参数分组：backbone / IWO / gate
#     param_groups = get_iwo_param_groups(
#         model, 
#         base_lr=lr_main,
#         iwo_lr_scale=iwo_lr_scale,
#         gate_lr_scale=gate_lr_scale
#     )
    
#     optimizer = Adam(param_groups, weight_decay=0.0)
    
#     # 学习率调度器
#     sch_name, lr_sch = _get_scheduler(optimizer, tr_cfg, logger)
    
#     # ========== 5. 损失函数 ==========
#     criterion_main = import_loss(opt.model_task).to(opt.device)
    
#     # 颜色一致性
#     lc = opt.config.get("loss", {}).get("color", {})
#     use_color = bool(lc.get("enable", True))
#     color_warm_epochs = _as_int(lc.get("warmup_epoch", 5), 5)  # IWO阶段可以更短
#     color_scale = _as_float(lc.get("scale", 1.0), 1.0)
#     color_loss = ColorConsistencyLoss(
#         a_window=tuple(lc.get("a_window", [126.0, 134.0])),
#         b_window=tuple(lc.get("b_window", [128.0, 140.0])),
#         var_band=tuple(lc.get("var_band", [40.0, 1200.0])),
#         w_lab_mean=_as_float(lc.get("w_lab_mean", 0.08), 0.08),
#         w_lab_var=_as_float(lc.get("w_lab_var", 0.02), 0.02),
#         w_rgb_stat=_as_float(lc.get("w_rgb_stat", 0.03), 0.03),
#         mu_gap=_as_float(lc.get("mu_gap", 0.08), 0.08),
#         rho_min=_as_float(lc.get("rho_min", 0.10), 0.10),
#     ).to(opt.device)
    
#     # ========== 6. 训练配置 ==========
#     epochs = _as_int(iwo_cfg.get("epoch", 80), 80)  # IWO阶段epoch数
#     val_every = max(1, _as_int(tr_cfg.get("val_every", 5), 5))
#     save_every = _as_int(tr_cfg.get("save_every", 20), 20)
    
#     # EMA
#     use_ema = bool(tr_cfg.get("ema", False))
#     ema_decay = _as_float(tr_cfg.get("ema_decay", 0.999), 0.999)
#     ema = EMA(model, decay=ema_decay) if use_ema else None
    
#     # Gate正则
#     gate_reg_base = _as_float(tr_cfg.get("gate_reg", 1e-4), 1e-4)
#     gate_reg_epochs = _as_int(tr_cfg.get("gate_reg_epochs", 40), 40)
#     gate_reg_tail = _as_int(tr_cfg.get("gate_reg_tail", 40), 40)
#     gate_reg_floor = _as_float(tr_cfg.get("gate_reg_floor", 1e-6), 1e-6)
    
#     # GradScaler
#     scaler = GradScaler(enabled=opt.device.startswith("cuda"))
    
#     # ========== 7. 保存目录 ==========
#     save_root = os.path.join(
#         os.getcwd(), 
#         "experiments", 
#         f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')} iwo_{opt.model_task}"
#     )
#     os.makedirs(save_root, exist_ok=True)
#     best_dir = os.path.join(save_root, "models")
#     os.makedirs(best_dir, exist_ok=True)
    
#     best_path = os.path.join(best_dir, "best_iwo.pkl")
#     best_fused_path = os.path.join(best_dir, "best_iwo_fused.pkl")
#     best_slim_path = os.path.join(best_dir, "best_iwo_slim.pkl")
    
#     history_dir, meta_path = _ensure_history(best_dir, logger)
    
#     best_psnr = -1e9
    
#     # ========== 8. 训练循环 ==========
#     logger.info(f"[IWO] 开始训练 {epochs} epochs...")
    
#     for epoch in range(1, epochs + 1):
#         model.train()
#         running = []
#         pbar = _tqdm(train_loader, desc=f"[IWO {epoch}/{epochs}]")
        
#         for it, batch in enumerate(pbar, start=1):
#             inp, gt, _ = _unpack_batch(batch, opt.device)
            
#             optimizer.zero_grad(set_to_none=True)
            
#             with autocast(device_type="cuda", dtype=torch.float16, enabled=opt.device.startswith("cuda")):
#                 pred = model(inp).clamp(0, 1)
#                 rec_loss = criterion_main(pred, gt)
                
#                 # 颜色一致性
#                 if use_color:
#                     cw = 1.0 if epoch > color_warm_epochs else float(epoch) / max(1, color_warm_epochs)
#                     col_total, _ = color_loss(pred)
#                     loss = rec_loss + color_scale * cw * col_total
#                 else:
#                     loss = rec_loss
                
#                 # Gate正则（平滑衰减）
#                 if gate_reg_base > 0:
#                     if epoch <= gate_reg_epochs:
#                         lam = gate_reg_base
#                     else:
#                         if gate_reg_tail > 0:
#                             t = min(epoch - gate_reg_epochs, gate_reg_tail) / float(gate_reg_tail)
#                             lam = gate_reg_base * (1.0 - t) + gate_reg_floor
#                         else:
#                             lam = gate_reg_floor
#                     loss = loss + lam * gate_identity_reg(model)
            
#             # 检查loss有效性
#             if not torch.isfinite(loss):
#                 _log_warn(logger, "[skip step] loss 非有限")
#                 continue
            
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optimizer)
#             scaler.update()
            
#             if use_ema and ema is not None:
#                 ema.update(model)
            
#             running.append(loss.item())
        
#         # ========== 9. 验证 ==========
#         do_validate = (epoch == 1) or (epoch % val_every == 0) or (epoch == epochs)
#         val_psnr = float('-inf')
#         # ★ 用于保存验证时的完整模型状态（EMA参数+BN统计）
#         validation_state_dict = None
        
#         if val_loader is not None and do_validate:
#             if use_ema and ema is not None:
#                 ema.apply_to(model)
#                 val_psnr = validate(model, val_loader, opt.device, logger,
#                                   color_loss if use_color else None, max_eval=0)
#                 # ★ 保存验证时的完整状态（EMA参数 + 此时的BN统计）
#                 validation_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
#                 ema.restore(model)
#             else:
#                 val_psnr = validate(model, val_loader, opt.device, logger,
#                                   color_loss if use_color else None, max_eval=0)
#                 validation_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            
#             logger.info(f"[IWO E{epoch}] lr={_get_lr(optimizer):.6g}, "
#                        f"train_loss={np.mean(running):.4f}, val_psnr={val_psnr:.3f}")
#         elif val_loader is None:
#             logger.info(f"[IWO E{epoch}] lr={_get_lr(optimizer):.6g}, "
#                        f"train_loss={np.mean(running):.4f}, no val_loader")
        
#         # ========== 10. 学习率调度 ==========
#         try:
#             if sch_name == "plateau":
#                 if val_psnr != float('-inf'):
#                     lr_sch.step(val_psnr)
#             else:
#                 lr_sch.step()
#         except Exception:
#             pass
        
#         # ========== 11. 保存最佳模型 ==========
#         if val_psnr != float('-inf') and val_psnr > best_psnr:
#             best_psnr = val_psnr
            
#             # ★ 关键修复：融合时必须使用EMA权重
#             # 1) 保存IWO原始形式（含weight1）- 使用EMA权重
#             if use_ema and ema is not None:
#                 ema.apply_to(model)
#                 sd_to_save = model.state_dict()
#                 save_iwo_checkpoint(model, best_path, epoch, val_psnr, optimizer)
#                 logger.info(f"[IWO BEST] {best_path}  PSNR={best_psnr:.3f} (EMA)")
#             else:
#                 sd_to_save = model.state_dict()
#                 save_iwo_checkpoint(model, best_path, epoch, val_psnr, optimizer)
#                 logger.info(f"[IWO BEST] {best_path}  PSNR={best_psnr:.3f}")
            
#             # 2) 融合权重并保存（W_fused = W_pre + W_learn）
#             # ★ 此时model仍处于EMA状态，确保融合的是EMA权重
#             try:
#                 # 创建模型副本用于融合
#                 import copy
#                 model_fused = copy.deepcopy(model)  # ★ 复制EMA状态的模型
#                 model_fused = fuse_iwo(model_fused)
#                 torch.save(model_fused.state_dict(), best_fused_path)
#                 logger.info(f"[IWO BEST] 融合权重: {best_fused_path} (EMA已融合)")
                
#                 # 3) 导出slim版本（用于ONNX/INT8）
#                 if hasattr(model_fused, 'slim'):
#                     slim_obj = model_fused.slim()
#                     slim_sd = slim_obj if isinstance(slim_obj, dict) else slim_obj.state_dict()
#                     torch.save(slim_sd, best_slim_path)
#                     logger.info(f"[IWO BEST] Slim版本: {best_slim_path} (EMA已融合)")
                
#                 del model_fused  # 释放内存
#             except Exception as e:
#                 _log_warn(logger, f"[IWO FUSE] 融合失败: {e}")
            
#             # ★ 最后才恢复模型到非EMA状态（用于继续训练）
#             if use_ema and ema is not None:
#                 ema.restore(model)
            
#             # 历史归档
#             stub = _archive_stub(epoch, val_psnr)
#             hist_path = os.path.join(history_dir, f"{stub}_iwo.pkl")
#             torch.save(sd_to_save, hist_path)
#             logger.info(f"[IWO BEST] history archived: {hist_path}")
            
#             # 记录元数据
#             _append_best_meta(
#                 meta_path,
#                 {
#                     "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "epoch": int(epoch),
#                     "val_psnr": float(val_psnr),
#                     "train_loss_mean": float(np.mean(running)) if running else None,
#                     "iwo_enabled": True,
#                     "latest": os.path.basename(best_path),
#                     "fused": os.path.basename(best_fused_path),
#                     "slim": os.path.basename(best_slim_path),
#                     "history": os.path.basename(hist_path),
#                 },
#                 logger
#             )
        
#         # 周期快照
#         if save_every and (epoch % save_every == 0):
#             ep_path = os.path.join(best_dir, f"iwo_epoch_{epoch:04d}.pkl")
#             torch.save(model.state_dict(), ep_path)
#             logger.info(f"[IWO SAVE] {ep_path}")
    
#     logger.info("="*60)
#     logger.info(f"[IWO] 训练完成！最佳 PSNR: {best_psnr:.3f}")
#     logger.info(f"[IWO] 原始权重: {best_path}")
#     logger.info(f"[IWO] 融合权重: {best_fused_path}")
#     logger.info(f"[IWO] Slim权重: {best_slim_path}")
#     logger.info("="*60)


# if __name__ == "__main__":
#     try:
#         train_iwo()
#     except Exception as e:
#         print("Fatal error:", e)
#         traceback.print_exc()
"""
IWO (Incremental Weight Optimization) 训练脚本
阶段二训练：在已训练好的模型基础上，启用IWO进行增量优化
目标：PSNR 从 22.x → 23+
"""

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

import cv2

from logger import Logger
from option import get_option
from data import import_loader
from loss import import_loss
from model import import_model
from loss_color import ColorConsistencyLoss, rgb_to_lab

# IWO工具
from enable_iwo import (
    enable_iwo_for_model,
    fuse_iwo,
    get_iwo_param_groups,
    save_iwo_checkpoint,
    load_iwo_checkpoint
)

# 复用main2.py中的工具函数
from main2 import (
    _log_warn,
    _tqdm,
    _maybe_set_cudnn,
    _as_float,
    _as_int,
    count_parameters,
    _to_device_maybe,
    _looks_like_names,
    _unpack_batch,
    _split_loaders,
    _to_uint8_rgb,
    _psnr_uint8,
    _save_predictions,
    freeze_bn,
    gate_identity_reg,
    EMA,
    _ensure_history,
    _archive_stub,
    _append_best_meta,
    _get_scheduler,
    _get_lr,
    validate,
    infer_and_maybe_save,
)


def train_iwo():
    """IWO阶段训练主函数"""
    opt = get_option()
    
    # ★ 强制加载 IWO 配置文件
    import yaml
    iwo_config_path = os.path.join(opt.root, 'config', f'{opt.model_task}_iwo.yaml')
    if os.path.isfile(iwo_config_path):
        with open(iwo_config_path, 'r', encoding='utf-8') as f:
            opt.config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"[IWO] 已加载 IWO 配置: {iwo_config_path}")
    else:
        print(f"[IWO] 警告: 未找到 {iwo_config_path}，使用默认配置")
    
    logger = Logger(opt)
    _maybe_set_cudnn(opt, logger)
    
    logger.info("="*60)
    logger.info("IWO (Incremental Weight Optimization) 训练阶段")
    logger.info("="*60)
    
    # ========== 1. 加载数据 ==========
    train_loader, val_loader = import_loader(opt)
    if train_loader is None:
        raise RuntimeError("import_loader 未返回 train_loader，无法训练。")
    
    # ========== 2. 加载已训练好的模型 ==========
    model = import_model(opt)
    
    # 从配置中获取预训练权重路径
    tr_cfg = opt.config.get("train", {})
    pretrained_ckpt = tr_cfg.get("pretrained_ckpt", "")
    
    if not pretrained_ckpt or not os.path.isfile(pretrained_ckpt):
        raise FileNotFoundError(
            f"IWO训练需要预训练权重！请在YAML中设置 train.pretrained_ckpt\n"
            f"当前路径: {pretrained_ckpt}"
        )
    
    logger.info(f"[IWO] 加载预训练权重: {pretrained_ckpt}")
    checkpoint = torch.load(pretrained_ckpt, map_location=opt.device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info("[IWO] 预训练权重加载成功")
    
    # ========== 3. 启用IWO ==========
    iwo_cfg = tr_cfg.get("iwo", {})
    iwo_config = {
        'init_std': _as_float(iwo_cfg.get('init_std', 0.0), 0.0),
        'target_layers': iwo_cfg.get('target_layers', None),  # None=所有MBRConv
    }
    
    model, iwo_params = enable_iwo_for_model(model, iwo_config)
    model = model.to(opt.device)
    
    # 参数统计
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    iwo_param_count = sum(p.numel() for p in iwo_params)
    logger.info(f"Total parameters: {total} / Trainable: {trainable}")
    logger.info(f"IWO parameters: {iwo_param_count} ({iwo_param_count/trainable*100:.2f}%)")
    
    # ========== 4. 构建优化器（参数分组）==========
    # ★★★ Epoch 3达到峰值后崩溃，使用极低学习率 ★★★
    lr_main = 1e-5  # 从 3e-5 进一步降到 1e-5（极低）
    iwo_lr_scale = 0.1  # 从 0.2 降到 0.1（IWO参数LR = 1e-6）
    gate_lr_scale = 0.1  # 从 0.2 降到 0.1
    
    # 调试：打印配置
    logger.info(f"[HARDCODED v2] lr_main={lr_main}, iwo_lr_scale={iwo_lr_scale}, gate_lr_scale={gate_lr_scale}")
    logger.info(f"[EFFECTIVE LR] IWO={lr_main*iwo_lr_scale:.2e}, Gate={lr_main*gate_lr_scale:.2e}, Backbone={lr_main:.2e}")
    logger.info(f"[CONFIG READ] lr_iwo={tr_cfg.get('lr_iwo')}, iwo.lr_scale={iwo_cfg.get('lr_scale')}")
    
    # 参数分组：backbone / IWO / gate
    param_groups = get_iwo_param_groups(
        model, 
        base_lr=lr_main,
        iwo_lr_scale=iwo_lr_scale,
        gate_lr_scale=gate_lr_scale
    )
    
    optimizer = Adam(param_groups, weight_decay=0.0)
    
    # 学习率调度器
    sch_name, lr_sch = _get_scheduler(optimizer, tr_cfg, logger)
    
    # ========== 5. 损失函数 ==========
    criterion_main = import_loss(opt.model_task).to(opt.device)
    
    # 颜色一致性
    lc = opt.config.get("loss", {}).get("color", {})
    use_color = bool(lc.get("enable", True))
    color_warm_epochs = _as_int(lc.get("warmup_epoch", 5), 5)  # IWO阶段可以更短
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
    
    # ========== 6. 训练配置 ==========
    epochs = _as_int(iwo_cfg.get("epoch", 80), 80)  # IWO阶段epoch数
    val_every = max(1, _as_int(tr_cfg.get("val_every", 5), 5))
    save_every = _as_int(tr_cfg.get("save_every", 20), 20)
    
    # EMA
    use_ema = bool(tr_cfg.get("ema", False))
    ema_decay = _as_float(tr_cfg.get("ema_decay", 0.999), 0.999)
    ema = EMA(model, decay=ema_decay) if use_ema else None
    
    # Gate正则
    gate_reg_base = _as_float(tr_cfg.get("gate_reg", 1e-4), 1e-4)
    gate_reg_epochs = _as_int(tr_cfg.get("gate_reg_epochs", 40), 40)
    gate_reg_tail = _as_int(tr_cfg.get("gate_reg_tail", 40), 40)
    gate_reg_floor = _as_float(tr_cfg.get("gate_reg_floor", 1e-6), 1e-6)
    
    # GradScaler
    scaler = GradScaler(enabled=opt.device.startswith("cuda"))
    
    # ========== 7. 保存目录 ==========
    save_root = os.path.join(
        os.getcwd(), 
        "experiments", 
        f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')} iwo_{opt.model_task}"
    )
    os.makedirs(save_root, exist_ok=True)
    best_dir = os.path.join(save_root, "models")
    os.makedirs(best_dir, exist_ok=True)
    
    best_path = os.path.join(best_dir, "best_iwo.pkl")
    best_fused_path = os.path.join(best_dir, "best_iwo_fused.pkl")
    best_slim_path = os.path.join(best_dir, "best_iwo_slim.pkl")
    
    history_dir, meta_path = _ensure_history(best_dir, logger)
    
    best_psnr = -1e9
    
    # ========== 8. 训练循环 ==========
    logger.info(f"[IWO] 开始训练 {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        pbar = _tqdm(train_loader, desc=f"[IWO {epoch}/{epochs}]")
        
        for it, batch in enumerate(pbar, start=1):
            inp, gt, _ = _unpack_batch(batch, opt.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type="cuda", dtype=torch.float16, enabled=opt.device.startswith("cuda")):
                pred = model(inp).clamp(0, 1)
                rec_loss = criterion_main(pred, gt)
                
                # 颜色一致性
                if use_color:
                    cw = 1.0 if epoch > color_warm_epochs else float(epoch) / max(1, color_warm_epochs)
                    col_total, _ = color_loss(pred)
                    loss = rec_loss + color_scale * cw * col_total
                else:
                    loss = rec_loss
                
                # Gate正则（平滑衰减）
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
            
            # 检查loss有效性
            if not torch.isfinite(loss):
                _log_warn(logger, "[skip step] loss 非有限")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if use_ema and ema is not None:
                ema.update(model)
            
            running.append(loss.item())
        
        # ========== 9. 验证 ==========
        do_validate = (epoch == 1) or (epoch % val_every == 0) or (epoch == epochs)
        val_psnr = float('-inf')
        # ★ 用于保存验证时的完整模型状态（EMA参数+BN统计）
        validation_state_dict = None
        
        if val_loader is not None and do_validate:
            if use_ema and ema is not None:
                ema.apply_to(model)
                val_psnr = validate(model, val_loader, opt.device, logger,
                                  color_loss if use_color else None, max_eval=0)
                # ★ 保存验证时的完整状态（EMA参数 + 此时的BN统计）
                validation_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                ema.restore(model)
            else:
                val_psnr = validate(model, val_loader, opt.device, logger,
                                  color_loss if use_color else None, max_eval=0)
                validation_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            
            logger.info(f"[IWO E{epoch}] lr={_get_lr(optimizer):.6g}, "
                       f"train_loss={np.mean(running):.4f}, val_psnr={val_psnr:.3f}")
        elif val_loader is None:
            logger.info(f"[IWO E{epoch}] lr={_get_lr(optimizer):.6g}, "
                       f"train_loss={np.mean(running):.4f}, no val_loader")
        
        # ========== 10. 学习率调度 ==========
        try:
            if sch_name == "plateau":
                if val_psnr != float('-inf'):
                    lr_sch.step(val_psnr)
            else:
                lr_sch.step()
        except Exception:
            pass
        
        # ========== 11. 保存最佳模型 ==========
        if val_psnr != float('-inf') and val_psnr > best_psnr:
            best_psnr = val_psnr
            
            # ★★★ 关键修复：使用validation_state_dict（包含验证时的EMA参数+BN统计）
            if validation_state_dict is None:
                logger.warning("[SAVE] validation_state_dict is None! Fallback to current state")
                validation_state_dict = model.state_dict()
            
            # 1) 保存IWO原始形式（含weight1）- 直接保存验证时的状态
            checkpoint = {
                'epoch': epoch,
                'val_psnr': val_psnr,
                'state_dict': validation_state_dict,
                'iwo_enabled': True,
            }
            if optimizer is not None:
                checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, best_path)
            logger.info(f"[IWO BEST] {best_path}  PSNR={best_psnr:.3f} (EMA+BN)")
            
            # 2) 融合权重并保存（W_fused = W_pre + W_learn）
            # ★ 加载validation_state_dict到模型，然后融合
            try:
                import copy
                # 临时创建一个干净的模型副本
                model_fused = copy.deepcopy(model)
                # 加载验证时的完整状态（EMA参数+BN统计）
                model_fused.load_state_dict(validation_state_dict)
                # 融合IWO权重
                model_fused = fuse_iwo(model_fused)
                torch.save(model_fused.state_dict(), best_fused_path)
                logger.info(f"[IWO BEST] 融合权重: {best_fused_path} (EMA+BN已融合)")
                
                # 3) 导出slim版本（用于ONNX/INT8）
                if hasattr(model_fused, 'slim'):
                    slim_obj = model_fused.slim()
                    slim_sd = slim_obj if isinstance(slim_obj, dict) else slim_obj.state_dict()
                    torch.save(slim_sd, best_slim_path)
                    logger.info(f"[IWO BEST] Slim版本: {best_slim_path} (EMA+BN已融合)")
                
                del model_fused  # 释放内存
            except Exception as e:
                _log_warn(logger, f"[IWO FUSE] 融合失败: {e}")
            
            # 历史归档
            stub = _archive_stub(epoch, val_psnr)
            hist_path = os.path.join(history_dir, f"{stub}_iwo.pkl")
            torch.save(validation_state_dict, hist_path)
            logger.info(f"[IWO BEST] history archived: {hist_path}")
            
            # 记录元数据
            _append_best_meta(
                meta_path,
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": int(epoch),
                    "val_psnr": float(val_psnr),
                    "train_loss_mean": float(np.mean(running)) if running else None,
                    "iwo_enabled": True,
                    "latest": os.path.basename(best_path),
                    "fused": os.path.basename(best_fused_path),
                    "slim": os.path.basename(best_slim_path),
                    "history": os.path.basename(hist_path),
                },
                logger
            )
        
        # 周期快照
        if save_every and (epoch % save_every == 0):
            ep_path = os.path.join(best_dir, f"iwo_epoch_{epoch:04d}.pkl")
            torch.save(model.state_dict(), ep_path)
            logger.info(f"[IWO SAVE] {ep_path}")
    
    logger.info("="*60)
    logger.info(f"[IWO] 训练完成！最佳 PSNR: {best_psnr:.3f}")
    logger.info(f"[IWO] 原始权重: {best_path}")
    logger.info(f"[IWO] 融合权重: {best_fused_path}")
    logger.info(f"[IWO] Slim权重: {best_slim_path}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        train_iwo()
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
