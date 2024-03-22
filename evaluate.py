import time, torch
from thop import profile
from torchvision.models.resnet import resnet101
from torchvision.models.resnet import resnet50


def test_model_speed(imgw=480, imgh=320, iterations=200):
    
    
    device = torch.device('cuda')
    # 启用 PyTorch 中 CuDNN（CUDA深度神经网络库）加速
    # torch.backends.cudnn.enabled = True
    # 启用自动优化，二者同时开启效果最好
    # torch.backends.cudnn.benchmark = True

    model = resnet50()
    model.eval()
    model.to(device)
    print('\n=========Speed Testing=========')
    # print(f'Model: {model}')
    print(f'Size (W, H): {imgw}, {imgh}')
    
    input = torch.randn(1, 3, imgh, imgw).cuda()
    with torch.no_grad():
        # 预热模型
        for _ in range(10):
            model(input)
        # 迭代次数，相当于处理图片数量    
        iterations = iterations
        

        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000

    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(f'FPS: {FPS}\n')

def test_model_parameters(imgw=224, imgh=224):
    
    model = resnet50()
    input = torch.randn(4, 3, imgh, imgw)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


if __name__ == '__main__':
    test_model_parameters()