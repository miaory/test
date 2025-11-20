# -*- coding: utf-8 -*-
"""
Export a slim (re-parameterized) checkpoint from an original checkpoint.

用法:
  python export_slim.py \
    --cfg config/lle.yaml \
    --in_ckpt "experiments/2025-11-09 16-07-19 train_lle/models/best.pkl" \
    --out    "experiments/2025-11-09 16-07-19 train_lle/models/best_slim_reexport.pkl" \
    --model_task lle \
    --check
"""

import os
import sys
import argparse
import yaml
import torch


def _find_project_root():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [os.getcwd(), here, os.path.dirname(here)]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "model")):
            return c
    return os.getcwd()


_PROJECT_ROOT = _find_project_root()
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from model import import_model
except Exception as e:
    print(f"[ERR] 无法从 'model' 导入 import_model。请在项目根目录运行本脚本。root={_PROJECT_ROOT}\n{e}")
    sys.exit(1)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _exp_config_from_ckpt(ckpt_path: str) -> str:
    models_dir = os.path.dirname(ckpt_path)
    exp_dir = os.path.dirname(models_dir)
    cfg_path = os.path.join(exp_dir, "config.yaml")
    return cfg_path if os.path.isfile(cfg_path) else ""


def _merge_model_args(base_cfg: dict, exp_cfg_path: str):
    if not exp_cfg_path:
        print("[Info] 未找到相邻的实验 config.yaml，将仅使用 --cfg 的参数。")
        return base_cfg

    try:
        exp_cfg = load_yaml(exp_cfg_path)
        print(f"[Info] 从实验配置同步模型参数: {exp_cfg_path}")
    except Exception as e:
        print(f"[WARN] 读取实验 config.yaml 失败：{e}")
        return base_cfg

    base_cfg.setdefault("model", {})
    base_cfg["model"].setdefault("args", {})

    src_args = {}
    if isinstance(exp_cfg.get("model"), dict):
        if isinstance(exp_cfg["model"].get("args"), dict):
            src_args.update(exp_cfg["model"]["args"])
        for k in ["channels", "rep_scale", "pretrained"]:
            if k in exp_cfg["model"]:
                src_args[k] = exp_cfg["model"][k]

    for k, v in src_args.items():
        base_cfg["model"]["args"][k] = v

    base_cfg["model"]["type"] = "original"
    base_cfg["model"]["need_slim"] = False
    return base_cfg


def _build_opt(cfg: dict, model_task: str, device: str, in_ckpt: str):
    """构造 import_model 需要的 opt；补齐 opt.task 避免 AttributeError。"""
    class Opt:
        pass
    opt = Opt()
    opt.config = cfg
    opt.model_task = model_task
    opt.device = device
    opt.task = "test"  # ★ 关键补齐：import_model 代码会访问这个字段
    # （可选）给一些常见字段，某些项目里会被用到但不影响本脚本
    opt.root = _PROJECT_ROOT
    models_dir = os.path.dirname(in_ckpt)
    opt.experiments = os.path.dirname(models_dir)
    opt.save_model_dir = models_dir
    return opt


def _load_state_strict(model, ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)
    try:
        model.load_state_dict(state, strict=True)
        print("[OK] 严格加载 original 权重成功。")
    except Exception as e:
        info = model.load_state_dict(state, strict=False)
        missing = getattr(info, "missing_keys", [])
        unexpected = getattr(info, "unexpected_keys", [])
        raise RuntimeError(
            f"Strict 加载失败：missing={len(missing)}, unexpected={len(unexpected)}\n"
            f"missing 示例: {missing[:5]}\n"
            f"unexpected 示例: {unexpected[:5]}\n"
            f"Raw error: {e}"
        )


def _call_slim(model):
    if not hasattr(model, "slim"):
        raise RuntimeError("当前模型未实现 slim()。")
    res = model.slim()
    return res if res is not None else model


def _check_equivalence(m1, m2, device="cpu", h=64, w=64, tol_max=1e-5, tol_mse=5e-11):
    torch.manual_seed(0)
    x = torch.rand(1, 3, h, w, device=device)
    m1 = m1.to(device).eval()
    m2 = m2.to(device).eval()
    with torch.no_grad():
        y1 = m1(x)
        y2 = m2(x)
    diff = (y1 - y2).abs()
    max_abs = float(diff.max().item())
    mse = float(((y1 - y2) ** 2).mean().item())
    print(f"[SLIM CHECK] max|orig-slim|={max_abs:.3e}, mse={mse:.3e}")
    if max_abs > tol_max or mse > tol_mse:
        print("[WARN] slim 输出与 original 偏差较大，请检查 BN 融合/多分支合并/门控 1×1 融合实现。")
    else:
        print("[OK] slim 等价性在容差内。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/lle.yaml", help="基础配置文件（会被实验 config 同步关键参数）")
    ap.add_argument("--in_ckpt", required=True, help="original 权重路径（如 best.pkl）")
    ap.add_argument("--out", required=True, help="导出的 slim 权重保存路径")
    ap.add_argument("--model_task", default="lle", choices=["lle", "isp", "sr"], help="模型任务")
    ap.add_argument("--device", default="cpu", help="构建/检查所用设备（建议 CPU 检查更稳定）")
    ap.add_argument("--check", action="store_true", help="是否进行 original/slim 数值等价性检查")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    exp_cfg = _exp_config_from_ckpt(args.in_ckpt)
    cfg = _merge_model_args(cfg, exp_cfg)

    cfg.setdefault("model", {})
    cfg["model"]["type"] = "original"
    cfg["model"]["need_slim"] = False
    cfg["model"].setdefault("args", {})

    print("[Info] 最终使用的模型参数:", cfg["model"]["args"])

    opt = _build_opt(cfg, args.model_task, args.device, args.in_ckpt)
    model_org = import_model(opt)
    model_org.eval()
    print(f"[Info] 严格加载 original 权重: {args.in_ckpt}")
    _load_state_strict(model_org, args.in_ckpt)

    print("[Info] 正在进行 slim 重参数化 ...")
    model_slim = _call_slim(model_org)
    model_slim.eval()

    if args.check:
        _check_equivalence(model_org, model_slim, device="cpu")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": model_slim.state_dict()}, args.out)
    print(f"[OK] slim 权重已保存: {args.out}")


if __name__ == "__main__":
    main()
