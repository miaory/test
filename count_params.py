"""
模型参数量统计工具
支持查看 .pkl / .pth / .onnx 模型的参数量和详细信息
"""
import os
import sys
import argparse
import torch
import onnx


def count_pytorch_params(model_path):
    """统计PyTorch模型参数量"""
    print(f"\n{'='*60}")
    print(f"PyTorch模型参数统计: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 提取state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"✅ Checkpoint类型: dict (含state_dict)")
                if 'epoch' in checkpoint:
                    print(f"   训练epoch: {checkpoint['epoch']}")
                if 'val_psnr' in checkpoint:
                    print(f"   验证PSNR: {checkpoint['val_psnr']:.3f}")
            else:
                state_dict = checkpoint
                print(f"✅ Checkpoint类型: state_dict")
        else:
            state_dict = checkpoint
            print(f"✅ Checkpoint类型: OrderedDict/其他")
        
        # 统计参数
        total_params = 0
        trainable_params = 0
        layer_info = []
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                num_params = param.numel()
                total_params += num_params
                
                # 假设所有参数都是可训练的（state_dict中没有requires_grad信息）
                trainable_params += num_params
                
                layer_info.append({
                    'name': name,
                    'shape': list(param.shape),
                    'params': num_params,
                    'dtype': str(param.dtype)
                })
        
        # 打印总体统计
        print(f"\n{'='*60}")
        print(f"📊 参数统计")
        print(f"{'='*60}")
        print(f"总参数量:        {total_params:>15,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数量:    {trainable_params:>15,} ({trainable_params/1e6:.2f}M)")
        print(f"层数:            {len(layer_info):>15,}")
        
        # 检查是否包含IWO参数
        has_iwo = any('weight1' in info['name'] for info in layer_info)
        if has_iwo:
            iwo_params = sum(info['params'] for info in layer_info if 'weight1' in info['name'])
            print(f"\n⚠️  检测到IWO参数:")
            print(f"   IWO参数量:    {iwo_params:>15,} ({iwo_params/1e6:.2f}M)")
            print(f"   占比:         {iwo_params/total_params*100:>14.2f}%")
        
        # 按参数量排序，显示Top 10
        print(f"\n{'='*60}")
        print(f"📋 参数量最多的10层")
        print(f"{'='*60}")
        layer_info_sorted = sorted(layer_info, key=lambda x: x['params'], reverse=True)
        
        print(f"{'层名':<50} {'形状':<25} {'参数量':>15}")
        print(f"{'-'*60}")
        for i, info in enumerate(layer_info_sorted[:10], 1):
            shape_str = str(info['shape'])
            print(f"{i:2d}. {info['name']:<47} {shape_str:<25} {info['params']:>12,}")
        
        # 可选：显示所有层
        show_all = input("\n是否显示所有层的详细信息? (y/n): ").strip().lower()
        if show_all == 'y':
            print(f"\n{'='*60}")
            print(f"📋 所有层详细信息")
            print(f"{'='*60}")
            print(f"{'层名':<50} {'形状':<25} {'参数量':>15}")
            print(f"{'-'*60}")
            for i, info in enumerate(layer_info, 1):
                shape_str = str(info['shape'])
                print(f"{i:3d}. {info['name']:<47} {shape_str:<25} {info['params']:>12,}")
        
        return total_params
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def count_onnx_params(model_path):
    """统计ONNX模型参数量"""
    print(f"\n{'='*60}")
    print(f"ONNX模型参数统计: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    try:
        model = onnx.load(model_path)
        
        # 统计参数
        total_params = 0
        initializer_info = []
        
        for initializer in model.graph.initializer:
            # 计算参数量
            shape = [dim for dim in initializer.dims]
            num_params = 1
            for dim in shape:
                num_params *= dim
            
            total_params += num_params
            initializer_info.append({
                'name': initializer.name,
                'shape': shape,
                'params': num_params,
                'dtype': onnx.TensorProto.DataType.Name(initializer.data_type)
            })
        
        # 打印总体统计
        print(f"\n{'='*60}")
        print(f"📊 参数统计")
        print(f"{'='*60}")
        print(f"总参数量:        {total_params:>15,} ({total_params/1e6:.2f}M)")
        print(f"初始化器数量:    {len(initializer_info):>15,}")
        
        # 输入输出信息
        print(f"\n{'='*60}")
        print(f"📥 输入信息")
        print(f"{'='*60}")
        for inp in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in inp.type.tensor_type.shape.dim]
            print(f"   {inp.name}: {shape}")
        
        print(f"\n{'='*60}")
        print(f"📤 输出信息")
        print(f"{'='*60}")
        for out in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                    for dim in out.type.tensor_type.shape.dim]
            print(f"   {out.name}: {shape}")
        
        # 按参数量排序，显示Top 10
        print(f"\n{'='*60}")
        print(f"📋 参数量最多的10个初始化器")
        print(f"{'='*60}")
        initializer_info_sorted = sorted(initializer_info, key=lambda x: x['params'], reverse=True)
        
        print(f"{'名称':<50} {'形状':<25} {'参数量':>15}")
        print(f"{'-'*60}")
        for i, info in enumerate(initializer_info_sorted[:10], 1):
            shape_str = str(info['shape'])
            print(f"{i:2d}. {info['name']:<47} {shape_str:<25} {info['params']:>12,}")
        
        return total_params
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(model_paths):
    """比较多个模型的参数量"""
    print(f"\n{'='*60}")
    print(f"模型参数量对比")
    print(f"{'='*60}")
    
    results = []
    for path in model_paths:
        if not os.path.isfile(path):
            print(f"⚠️  文件不存在: {path}")
            continue
        
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.pkl', '.pth', '.pt']:
            params = count_pytorch_params(path)
        elif ext == '.onnx':
            params = count_onnx_params(path)
        else:
            print(f"⚠️  不支持的文件格式: {ext}")
            continue
        
        if params is not None:
            results.append({
                'name': os.path.basename(path),
                'path': path,
                'params': params
            })
    
    # 打印对比表
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"📊 对比总结")
        print(f"{'='*60}")
        print(f"{'模型':<40} {'参数量':>15} {'相对比例':>10}")
        print(f"{'-'*60}")
        
        base_params = results[0]['params']
        for i, result in enumerate(results, 1):
            ratio = result['params'] / base_params * 100
            print(f"{i}. {result['name']:<37} {result['params']:>12,} {ratio:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description='模型参数量统计工具')
    parser.add_argument('model_path', nargs='+', help='模型文件路径（支持.pkl/.pth/.onnx）')
    parser.add_argument('--compare', action='store_true', help='对比多个模型')
    
    args = parser.parse_args()
    
    if args.compare or len(args.model_path) > 1:
        compare_models(args.model_path)
    else:
        model_path = args.model_path[0]
        
        if not os.path.isfile(model_path):
            print(f"❌ 文件不存在: {model_path}")
            return
        
        ext = os.path.splitext(model_path)[1].lower()
        
        if ext in ['.pkl', '.pth', '.pt']:
            count_pytorch_params(model_path)
        elif ext == '.onnx':
            count_onnx_params(model_path)
        else:
            print(f"❌ 不支持的文件格式: {ext}")
            print(f"   支持的格式: .pkl, .pth, .pt, .onnx")


if __name__ == "__main__":
    main()
