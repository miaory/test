# tools/make_splits_890.py
# 用法：
  # python tools/make_splits_890.py \
  #   --inp /home/featurize/data/raw-890/raw-890 \
  #   --gt  /home/featurize/data/reference-890/reference-890 \
  #   --seed 2025 --train 710 --val 90 --test 90
import os, argparse, random

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
    ap.add_argument('--train', type=int, default=710)
    ap.add_argument('--val',   type=int, default=90)
    ap.add_argument('--test',  type=int, default=90)
    ap.add_argument('--outdir', default='splits')
    args = ap.parse_args()

    all_files = common_pairs(args.inp, args.gt)
    if len(all_files) < args.train + args.val + args.test:
        raise ValueError(f"Need at least {args.train+args.val+args.test} pairs, got {len(all_files)}.")

    rng = random.Random(args.seed)
    files = list(all_files)
    rng.shuffle(files)

    test = files[:args.test]
    remain = files[args.test:]
    train = remain[:args.train]
    val   = remain[args.train: args.train + args.val]

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val)   & set(test)) == 0

    write_list(os.path.join(args.outdir, 'train_list.txt'), train)
    write_list(os.path.join(args.outdir, 'valid_list.txt'), val)
    write_list(os.path.join(args.outdir, 'test_90.txt'),    test)

    print(f"Done. Wrote to '{args.outdir}':")
    print(f"  train_list.txt : {len(train)}")
    print(f"  valid_list.txt : {len(val)}")
    print(f"  test_90.txt    : {len(test)}")

if __name__ == '__main__':
    main()
