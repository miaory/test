import os
import argparse
import random

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def list_images(root):
    return sorted([f for f in os.listdir(root)
                   if f.lower().endswith(IMG_EXTS) and not f.startswith('.')])

def common_pairs(inp_dir, gt_dir):
    xs = set(list_images(inp_dir))
    ys = set(list_images(gt_dir))
    com = sorted(xs & ys)
    if not com:
        raise FileNotFoundError(f"No matched pairs between:\n  {inp_dir}\n  {gt_dir}\n"
                                "Check identical file names (incl. extensions).")
    return com

def write_list(path, names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for n in names: f.write(n + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True)
    ap.add_argument('--gt',  required=True)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--train', type=int, default=800)  # 训练集800对
    ap.add_argument('--test',  type=int, default=90)   # 测试集90对
    ap.add_argument('--outdir', default='splits')
    args = ap.parse_args()

    all_files = common_pairs(args.inp, args.gt)
    if len(all_files) < args.train + args.test:
        raise ValueError(f"Need at least {args.train + args.test} pairs, got {len(all_files)}.")

    rng = random.Random(args.seed)
    files = list(all_files)
    rng.shuffle(files)

    test = files[:args.test]
    train = files[args.test:args.test + args.train]

    assert len(set(train) & set(test)) == 0  # 确保没有重复的文件

    write_list(os.path.join(args.outdir, 'train_list.txt'), train)
    write_list(os.path.join(args.outdir, 'test_90.txt'), test)

    print(f"Done. Wrote to '{args.outdir}':")
    print(f"  train_list.txt : {len(train)}")
    print(f"  test_90.txt    : {len(test)}")

if __name__ == '__main__':
    main()
