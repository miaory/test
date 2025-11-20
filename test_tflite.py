"""TFLite 模型测试脚本
与 main2.py 的测试流程保持一致，使用相同的数据加载和 PSNR 计算方式
支持动态输入尺寸
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from datetime import datetime

# 导入 main2.py 中的配置和数据加载
from option import get_option
from data import import_loader

def load_tflite_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"TFLite 模型文件不存在: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("\n" + "="*60)
    print("TFLite 模型信息")
    print("="*60)
    print(f"模型路径: {model_path}")
    print(f"\n输入张量:")
    for inp in input_details:
        print(f"  名称: {inp['name']}")
        print(f"  形状: {inp['shape']}")
        print(f"  类型: {inp['dtype']}")
    print(f"\n输出张量:")
    for out in output_details:
        print(f"  名称: {out['name']}")
        print(f"  形状: {out['shape']}")
        print(f"  类型: {out['dtype']}")
    print("="*60 + "\n")
    return interpreter, input_details, output_details

def preprocess_for_tflite(img_tensor):
    img_np = img_tensor.cpu().numpy()
    img_np = np.transpose(img_np, (0, 2, 3, 1))
    img_np = img_np.astype(np.float32)
    img_np = np.clip(img_np, 0.0, 1.0)
    return img_np

def postprocess_from_tflite(output_np):
    output_np = np.transpose(output_np[0], (2, 0, 1))
    output_np = np.clip(output_np, 0.0, 1.0)
    return output_np

def to_uint8_rgb(arr):
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.rint(arr * 255.0).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))

def psnr_uint8(pred_u8, gt_u8):
    diff = pred_u8.astype(np.float64) - gt_u8.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 ** 2) / mse)

def save_image(img_arr, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = Image.fromarray(img_arr)
    img.save(save_path)

def resize_input_tensor(interpreter, input_details, new_shape):
    interpreter.resize_tensor_input(input_details[0]['index'], new_shape)
    interpreter.allocate_tensors()

def test_tflite(model_path, opt, save_outputs=False):
    interpreter, input_details, output_details = load_tflite_model(model_path)
    print("加载测试数据...")
    test_loader = import_loader(opt)
    if test_loader is None:
        raise RuntimeError("无法加载测试数据")
    print(f"测试样本数: {len(test_loader.dataset)}\n")
    output_dir = None
    if save_outputs:
        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        output_dir = os.path.join(os.getcwd(), "experiments", f"{timestamp} tflite_test_{opt.model_task}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}\n")
    psnr_list = []
    current_shape = None
    print("开始测试...")
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="[TFLite TEST]")):
        if isinstance(batch, dict):
            inp = batch.get("inp") or batch.get("lr")
            gt = batch.get("gt") or batch.get("hr")
            name = batch.get("name", None)
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                inp, gt, name = batch
            elif len(batch) == 2:
                inp, gt = batch
                name = None
            else:
                inp = batch[0]
                gt = None
                name = None
        else:
            inp = batch
            gt = None
            name = None
        inp_np = preprocess_for_tflite(inp)
        if current_shape is None or current_shape != inp_np.shape:
            current_shape = inp_np.shape
            resize_input_tensor(interpreter, input_details, current_shape)
        interpreter.set_tensor(input_details[0]['index'], inp_np)
        interpreter.invoke()
        output_np = interpreter.get_tensor(output_details[0]['index'])
        pred_chw = postprocess_from_tflite(output_np)
        if gt is not None:
            gt_chw = gt[0].cpu().numpy()
            pred_u8 = to_uint8_rgb(pred_chw)
            gt_u8 = to_uint8_rgb(gt_chw)
            psnr_val = psnr_uint8(pred_u8, gt_u8)
            psnr_list.append(psnr_val)
        if save_outputs and output_dir is not None:
            pred_u8 = to_uint8_rgb(pred_chw)
            if name is not None:
                if isinstance(name, (list, tuple)):
                    img_name = str(name[0]) if len(name) > 0 else f"img_{batch_idx:06d}.png"
                else:
                    img_name = str(name)
            else:
                img_name = f"img_{batch_idx:06d}.png"
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_name = os.path.splitext(img_name)[0] + '.png'
            save_path = os.path.join(output_dir, img_name)
            save_image(pred_u8, save_path)
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    if psnr_list:
        mean_psnr = float(np.mean(psnr_list))
        std_psnr = float(np.std(psnr_list))
        min_psnr = float(np.min(psnr_list))
        max_psnr = float(np.max(psnr_list))
        print(f"平均 PSNR: {mean_psnr:.4f} dB")
        print(f"标准差:    {std_psnr:.4f} dB")
        print(f"最小值:    {min_psnr:.4f} dB")
        print(f"最大值:    {max_psnr:.4f} dB")
        print(f"测试样本:  {len(psnr_list)}")
    else:
        print("无 Ground Truth，未计算 PSNR")
    if save_outputs and output_dir is not None:
        print(f"\n输出图像已保存至: {output_dir}")
    print("="*60 + "\n")
    return psnr_list

def main():
    parser = argparse.ArgumentParser(description='TFLite 模型测试脚本')
    parser.add_argument('--model', type=str, default='./LLE.tflite', help='TFLite 模型路径')
    parser.add_argument('--save', action='store_true', help='是否保存输出图像')
    parser.add_argument('-model_task', default='lle', type=str, choices=['isp', 'lle', 'sr'], help='模型任务类型')
    args = parser.parse_args()
    sys.argv = ['test_tflite.py', '-task', 'test', '-model_task', args.model_task]
    opt = get_option()
    test_tflite(args.model, opt, save_outputs=args.save)

if __name__ == "__main__":
    main()
