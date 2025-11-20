import torch
from importlib import import_module
from .lle import MobileIELLENet, MobileIELLENetS
from .isp import MobileIEISPNet, MobileIEISPNetS
import os

__all__ = {
    'MobileIELLENet',
    'MobileIELLENetS',
    'MobileIEISPNet',
    'MobileIEISPNetS',
    'import_model'
}


def import_model(opt):
    # ★ 处理特殊的 model_task（如 lle_psnr 也用 LLE 模型）
    task_mapping = {
        'lle_psnr': 'lle',  # Stage2 PSNR微调也用LLE模型
    }
    actual_task = task_mapping.get(opt.model_task, opt.model_task)
    
    model_name = 'MobileIE' + actual_task.upper()
    kwargs = {'channels': opt.config['model']['channels']}

    # ---------------- 构建模型 ----------------
    if opt.config['model']['type'] == 're-parameterized':
        model_name += 'NetS'
    elif opt.config['model']['type'] == 'original':
        model_name += 'Net'
        kwargs['rep_scale'] = opt.config['model']['rep_scale']
    else:
        raise ValueError('unknown model type, please choose from [original, re-parameterized]')

    model = getattr(import_module('model'), model_name)(**kwargs)
    model = model.to(opt.device)

    # ---------------- 加载预训练权重（可选） ----------------
    if opt.config['model']['pretrained']:
        ckpt_path = opt.config['model']['pretrained']
        if os.path.exists(ckpt_path):
            print(f"[Model] Load pretrained weights from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=opt.device)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[Warning] Pretrained path not found: {ckpt_path}")

    # ---------------- 加载训练好的 checkpoint（测试/推理阶段） ----------------
    if opt.task in ['test', 'demo']:
        # 优先读取 YAML 中 test.ckpt；否则尝试默认路径
        ckpt_path = opt.config.get('test', {}).get('ckpt', None)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(opt.save_model_dir, 'best_slim.pkl')
        if os.path.exists(ckpt_path):
            print(f"[Model] Load checkpoint for {opt.task.upper()} from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=opt.device)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[Warning] Checkpoint not found: {ckpt_path}")

    # ---------------- Slim 模型（按原逻辑） ----------------
    if opt.config['model']['type'] == 'original' and opt.config['model']['need_slim'] is True:
        model = model.slim().to(opt.device)

    return model
