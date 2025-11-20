import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 禁用GPU,避免CUDA错误

import onnx
from onnx import shape_inference
import numpy as np

# 步骤1: 修复ONNX模型的动态输入
print("正在加载ONNX模型...")
onnx_path = "/home/featurize/work/test9/MobileIE_LLE_slim.onnx"
model = onnx.load(onnx_path)

# 查看原始输入信息
print("\n原始输入信息:")
for input_tensor in model.graph.input:
    print(f"  名称: {input_tensor.name}")
    shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"  形状: {shape}")

# 设置固定输入尺寸 (batch_size, channels, height, width)
# 根据错误信息,尝试使用更大的输入尺寸
fixed_input_shape = [1, 3, 256, 256]  # 或者根据你的实际需求调整

print(f"\n设置固定输入形状为: {fixed_input_shape}")

# 修改输入维度
for input_tensor in model.graph.input:
    # 清除现有维度
    while len(input_tensor.type.tensor_type.shape.dim) > 0:
        input_tensor.type.tensor_type.shape.dim.pop()
    
    # 添加新的固定维度
    for dim_value in fixed_input_shape:
        dim = input_tensor.type.tensor_type.shape.dim.add()
        dim.dim_value = dim_value

# 运行形状推断
print("\n运行形状推断...")
try:
    model = shape_inference.infer_shapes(model)
    print("形状推断成功!")
except Exception as e:
    print(f"形状推断警告: {e}")

# 保存修复后的模型
fixed_onnx_path = "/home/featurize/work/MobileIE_LLE_slim_fixed.onnx"
onnx.save(model, fixed_onnx_path)
print(f"\n已保存修复后的ONNX模型到: {fixed_onnx_path}")

# 验证修复后的模型
print("\n验证修复后的模型:")
fixed_model = onnx.load(fixed_onnx_path)
for input_tensor in fixed_model.graph.input:
    print(f"  名称: {input_tensor.name}")
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"  形状: {shape}")

# 步骤2: 转换为TFLite
print("\n" + "="*50)
print("开始转换ONNX到TFLite...")
print("="*50)

from onnx2tflite.converter import onnx_converter

try:
    onnx_converter(
        onnx_model_path=fixed_onnx_path,
        need_simplify=True,
        output_path="/home/featurize/work/test9",
        target_formats=['tflite'],
        weight_quant=False,
        int8_model=False,
        int8_mean=None,
        int8_std=None,
        image_root=None
    )
    print("\n转换成功!")
except Exception as e:
    print(f"\n转换失败: {e}")
    import traceback
    traceback.print_exc()
