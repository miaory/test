# avg_ckpt.py
import os
import re
import glob
import argparse

import torch


def parse_psnr_from_name(name: str, default: float = 0.0) -> float:
    """
    从文件名里解析 PSNR，比如：
    best_E188_PSNR22.721_20251115-xxxx.pkl
    """
    m = re.search(r"PSNR([0-9.]+)", name)
    if not m:
        return default
    try:
        return float(m.group(1))
    except Exception:
        return default


def load_state_dict(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        type=str,
        required=True,
        help="history 目录的通配符，例如 '.../history/best_E*_PSNR*.pkl'"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出的平均权重路径，例如 best_avg5.pkl"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="选 PSNR 最高的前 K 个 checkpoint 做平均"
    )
    args = parser.parse_args()

    files = glob.glob(args.glob)
    if not files:
        raise RuntimeError(f"没有匹配到任何文件: {args.glob}")

    # 解析 PSNR 并排序
    items = []
    for f in files:
        psnr = parse_psnr_from_name(os.path.basename(f))
        items.append((psnr, f))

    # 按 PSNR 从大到小排
    items.sort(key=lambda x: x[0], reverse=True)
    topk = items[: args.topk]

    print("将要平均的 checkpoint：")
    for psnr, f in topk:
        print(f"  {psnr:.3f} dB  <-  {f}")

    # 开始做逐参数平均
    avg_state = {}
    n = len(topk)

    for idx, (psnr, f) in enumerate(topk, start=1):
        print(f"[{idx}/{n}] 加载 {f}")
        sd = load_state_dict(f, map_location="cpu")

        if not avg_state:
            # 第一个：复制一份 float32
            for k, v in sd.items():
                if not torch.is_floating_point(v):
                    avg_state[k] = v.clone()
                else:
                    avg_state[k] = v.detach().clone().float()
        else:
            # 后续：累加
            for k, v in sd.items():
                if not torch.is_floating_point(v):
                    # 非浮点（例如 BN 的 running_*）直接覆盖或保持第一个即可
                    continue
                avg_state[k] += v.detach().clone().float()

    # 取平均
    for k, v in avg_state.items():
        if torch.is_floating_point(v):
            avg_state[k] = v / float(n)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": avg_state}, args.out)
    print(f"\n已保存平均权重到: {args.out}")


if __name__ == "__main__":
    main()

