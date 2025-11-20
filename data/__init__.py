# -*- coding: utf-8 -*-
from torch.utils import data
from importlib import import_module
import os

from .lledata import LLEData
from .ispdata import ISPData  # 如未使用可保留

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

__all__ = {
    'ISPData',
    'LLEData',
    'import_loader'
}

def _list_images(root):
    return sorted([
        f for f in os.listdir(root)
        if f.lower().endswith(IMG_EXTS) and not f.startswith('.')
    ])

def _common_filenames(inp_dir, gt_dir):
    inp_files = set(_list_images(inp_dir))
    gt_files  = set(_list_images(gt_dir))
    common = sorted(list(inp_files & gt_files))
    if len(common) == 0:
        raise FileNotFoundError(
            f"No matched pairs found between:\n  inp: {inp_dir}\n  gt : {gt_dir}\n"
            "Check that file names (including extensions) are identical."
        )
    return common

def _is_empty_path(p):
    return (p is None) or (str(p).strip() == '')

def _read_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [x.strip() for x in f if x.strip()]

def _maybe_splits():
    """若存在 splits/ 列表就返回其路径，否则返回 None。"""
    base = os.path.join(os.getcwd(), 'splits')
    tr = os.path.join(base, 'train_list.txt')
    va = os.path.join(base, 'valid_list.txt')
    te = os.path.join(base, 'test_90.txt')
    if os.path.isfile(tr) and os.path.isfile(va) and os.path.isfile(te):
        return tr, va, te
    return None

def import_loader(opt):
    # ★ 处理特殊的 model_task（如 lle_psnr 也用 LLEData）
    task_mapping = {
        'lle_psnr': 'lle',  # Stage2 PSNR微调也用LLE数据加载器
    }
    actual_task = task_mapping.get(opt.model_task, opt.model_task)
    
    dataset_name = actual_task.upper() + 'Data'
    dataset = getattr(import_module('data'), dataset_name)

    TRAIN_COUNT = 800  # 旧逻辑：前800做训练；剩余作为valid/test
    patch_size = int(opt.config.get('train', {}).get('patch_size', 256))  # 训练时裁剪，val/test 不裁剪

    train_inp_path = opt.config['train']['train_inp']
    train_gt_path  = opt.config['train']['train_gt']

    all_files = _common_filenames(train_inp_path, train_gt_path)
    splits = _maybe_splits()

    if opt.task == 'train':
        if splits is not None:
            tr_txt, va_txt, _ = splits
            train_files = _read_list(tr_txt)
            valid_files = _read_list(va_txt)
        else:
            # 回退到旧逻辑（不建议长期使用）
            if len(all_files) < TRAIN_COUNT + 1:
                raise ValueError(f"Found only {len(all_files)} matched pairs, "
                                 f"need at least {TRAIN_COUNT+1} for 800/90 split.")
            train_files = all_files[:TRAIN_COUNT]
            valid_files = all_files[TRAIN_COUNT:]

        train_data = dataset(
            opt, train_inp_path, train_gt_path,
            filenames=train_files, phase='train', patch_size=patch_size
        )
        valid_data = dataset(
            opt, train_inp_path, train_gt_path,
            filenames=valid_files, phase='valid', patch_size=None  # ★ 验证不裁剪/不增强
        )

        train_loader = data.DataLoader(
            train_data,
            batch_size=opt.config['train']['batch_size'],
            shuffle=True,
            num_workers=opt.config['train']['num_workers'],
            drop_last=True,
            pin_memory=True
        )
        valid_loader = data.DataLoader(
            valid_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['train']['num_workers'],
            drop_last=False,
            pin_memory=True
        )
        return train_loader, valid_loader

    elif opt.task == 'test':
        test_cfg = opt.config.get('test', {}) or {}
        inp_test_path = test_cfg.get('test_inp', None)
        gt_test_path  = test_cfg.get('test_gt', None)

        # 情况 A：用户给了独立测试目录 → 全量对齐同名评测
        if not _is_empty_path(inp_test_path) and not _is_empty_path(gt_test_path):
            common = _common_filenames(inp_test_path, gt_test_path)
            test_data = dataset(
                opt, inp_test_path, gt_test_path,
                filenames=common, phase='test', patch_size=None
            )
        else:
            # 情况 B：YAML.data.test_list 优先；其次 splits/test_90.txt；最后旧逻辑“后90”
            yaml_test_list = (opt.config.get('data', {}) or {}).get('test_list', '')
            if yaml_test_list and os.path.isfile(yaml_test_list):
                test_files = _read_list(yaml_test_list)
            elif splits is not None:
                _, _, te_txt = splits
                test_files = _read_list(te_txt)
            else:
                if len(all_files) < TRAIN_COUNT + 1:
                    raise ValueError(f"Found only {len(all_files)} matched pairs, "
                                     f"need at least {TRAIN_COUNT+1} for 800/90 split.")
                test_files = all_files[TRAIN_COUNT:]

            test_data = dataset(
                opt, train_inp_path, train_gt_path,
                filenames=test_files, phase='test', patch_size=None  # ★ 测试不裁剪/不增强
            )

        test_loader = data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=test_cfg.get('num_workers', 0),
            drop_last=False,
            pin_memory=True
        )
        return test_loader

    elif opt.task == 'demo':
        inp_demo_path = opt.config['demo']['demo_inp']
        demo_data = dataset(opt, inp_demo_path, gt_path=None, filenames=None, phase='demo', patch_size=None)
        demo_loader = data.DataLoader(
            demo_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['demo']['num_workers'],
            drop_last=False,
            pin_memory=True
        )
        return demo_loader

    else:
        raise ValueError('unknown task, please choose from [train, test, demo]')
