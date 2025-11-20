# import torch
# import time
# from thop import profile, clever_format
# from model import lle
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# width = 640
# height = 480


# def compute_FLOPs_and_model_size(model, width, height):
#     input = torch.randn(1, 3, width, height).cuda()
#     macs, params = profile(model, inputs=(input,), verbose=False)
#     return macs, params

# @torch.no_grad()
# def compute_fps_and_inference_time(model, shape, epoch=100, warmup=10, device=None):
#     total_time = 0.0

#     if not device:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
    
#     model.eval()  # Switch to evaluation mode

#     # Warm-up iterations
#     for _ in range(warmup):
#         data = torch.randn(shape).to(device)
#         model(data)
    
#     # Actual timing iterations
#     for _ in range(epoch):
#         data = torch.randn(shape).to(device)

#         start = time.time()
#         outputs = model(data)
#         torch.cuda.synchronize()  # Ensure CUDA has finished all tasks
#         end = time.time()

#         total_time += (end - start)

#     avg_inference_time = total_time / epoch
#     fps = epoch / total_time

#     return fps, avg_inference_time

# def test_model_flops(width, height):
#     model = lle.MobileIELLENetS(channels=12) 
#     model.cuda()

#     FLOPs, params = compute_FLOPs_and_model_size(model, width, height)

#     model_size = params * 4.0 / 1024 / 1024
#     flops, params = clever_format([FLOPs, params], "%.3f")

#     print('Number of parameters: {}'.format(params))
#     print('Size of model: {:.2f} MB'.format(model_size))
#     print('Computational complexity: {} FLOPs'.format(flops))

# def test_fps_and_inference_time(width, height):
#     model = lle.MobileIELLENetS(channels=12)  
#     model.cuda()

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     fps, avg_inference_time = compute_fps_and_inference_time(model, (1, 3, width, height), device=device)
#     print('device: {} - fps: {:.3f}, average inference time per frame: {:.6f} seconds'.format(device.type, fps, avg_inference_time))

# if __name__ == '__main__':
#     test_model_flops(width, height)
#     test_fps_and_inference_time(width, height)import torch
import torch
import time
from thop import profile, clever_format
from model import lle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

width = 640
height = 480


def compute_FLOPs_and_model_size(model, width, height):
    input = torch.randn(1, 3, width, height).cuda()
    macs, params = profile(model, inputs=(input,), verbose=False)
    return macs, params

@torch.no_grad()
def compute_fps_and_inference_time(model, shape, epoch=100, warmup=10, device=None):
    total_time = 0.0

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()  # Switch to evaluation mode

    # Warm-up iterations
    for _ in range(warmup):
        data = torch.randn(shape).to(device)
        model(data)
    
    # Actual timing iterations
    for _ in range(epoch):
        data = torch.randn(shape).to(device)

        start = time.time()
        outputs = model(data)
        torch.cuda.synchronize()  # Ensure CUDA has finished all tasks
        end = time.time()

        total_time += (end - start)

    avg_inference_time = total_time / epoch
    fps = epoch / total_time

    return fps, avg_inference_time

def test_model_flops(width, height):
    model = lle.MobileIELLENetS(channels=12) 
    model.cuda()

    FLOPs, params = compute_FLOPs_and_model_size(model, width, height)

    model_size = params * 4.0 / 1024 / 1024
    flops, params = clever_format([FLOPs, params], "%.3f")

    print('Number of parameters: {}'.format(params))
    print('Size of model: {:.2f} MB'.format(model_size))
    print('Computational complexity: {} FLOPs'.format(flops))

def test_fps_and_inference_time(width, height, num_tests=50):
    model = lle.MobileIELLENetS(channels=12)  
    model.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f'\nRunning {num_tests} FPS tests...')
    fps_results = []
    time_results = []
    
    for i in range(num_tests):
        fps, avg_inference_time = compute_fps_and_inference_time(model, (1, 3, width, height), device=device)
        fps_results.append(fps)
        time_results.append(avg_inference_time)
        if (i + 1) % 10 == 0:
            print(f'  Progress: {i + 1}/{num_tests} tests completed')
    
    # 计算统计信息
    import numpy as np
    fps_mean = np.mean(fps_results)
    fps_std = np.std(fps_results)
    fps_min = np.min(fps_results)
    fps_max = np.max(fps_results)
    
    time_mean = np.mean(time_results)
    time_std = np.std(time_results)
    time_min = np.min(time_results)
    time_max = np.max(time_results)
    
    print('\n' + '='*60)
    print('FPS Statistics (over {} tests):'.format(num_tests))
    print('='*60)
    print('device: {}'.format(device.type))
    print('FPS        - mean: {:.3f}, std: {:.3f}, min: {:.3f}, max: {:.3f}'.format(fps_mean, fps_std, fps_min, fps_max))
    print('Infer time - mean: {:.6f}s, std: {:.6f}s, min: {:.6f}s, max: {:.6f}s'.format(time_mean, time_std, time_min, time_max))
    print('='*60)

if __name__ == '__main__':
    test_model_flops(width, height)
    test_fps_and_inference_time(width, height, num_tests=50)