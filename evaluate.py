import time, torch
from torchvision.models.resnet import resnet101
from torchvision.models.resnet import resnet50
from toolbox.models.enet import ENet
from toolbox.models.fcn import FCN  
from toolbox.models.bisenet import BiSeNet
from toolbox.models.bisenetv1 import BiSeNetv1  
from toolbox.models.bisenetv2 import BiSeNetv2
from toolbox.models.dfanet import DFANet 
from toolbox.models.segnet import SegNet 

def test_model_speed(model, imgh, imgw, iterations=200):
    
    
    device = torch.device('cuda')
    # 启用 PyTorch 中 CuDNN（CUDA深度神经网络库）加速
    # torch.backends.cudnn.enabled = True
    # 启用自动优化，二者同时开启效果最好
    # torch.backends.cudnn.benchmark = True

    model.eval()
    model.to(device)
    print('\n=========Speed Testing=========')
    # print(f'Model: {model}')
    print(f'Size (H, W): {imgh}, {imgw}')
    
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


def cal_model_params(model, imgh, imgw):
    
    
    with torch.cuda.device(0):
        model = model
        try:
            from ptflops import get_model_complexity_info
            model.eval()
            macs, params = get_model_complexity_info(model, (3, imgh, imgw), as_strings=True, print_per_layer_stat=False, verbose=False)
            print('{:<20} {:<8}'.format('MACs:', macs))
            print('{:<20} {:<8}'.format('Parameters:', params))
        except:
            import numpy as np
            params = np.sum([p.numel() for p in model.parameters()])
            print(f'Number of parameters: {params / 1e6:.2f}M\n')


if __name__ == '__main__':
    model = SegNet(12)
    imgh = 1024
    imgw = 2048
    test_model_speed(model=model, imgh=imgh, imgw=imgw, iterations=100)
    cal_model_params(model=model, imgh=imgh, imgw=imgw)