import gc
import numpy as np

from utils import *

with use_local_python():
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.transforms import Compose, ToTensor
    from torchvision.models.resnet import Bottleneck, ResNet

name = "[AI] Depth Estimation"
version = "0.8.1"
desc = """Torch pipeline for depth estimation.
Based on "Depth-Estimation-PyTorch" from:
https://github.com/wolverinn/Depth-Estimation-PyTorch"""

settings = {
    "Scale": build_variable(1.0, 0.5, 2.0, 0.1, """Scale image for AI processing.
Do note that depth estimation uses 1/4 of the resolution.""")
}

resnet_path = resolve_path('ai_depth_estimation\\resnet101-63fe2227.pth')
model_path = resolve_path('ai_depth_estimation\\fyn_model.pt')

# Following code is using:
# Depth-Estimation-PyTorch
# https://github.com/wolverinn/Depth-Estimation-PyTorch

def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

class I2D(nn.Module):
    def __init__(self):
        super(I2D, self).__init__()

        resnet = ResNet(Bottleneck, [3, 4, 23, 3])
        resnet.load_state_dict(torch.load(resnet_path))

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1) # 256
        self.layer2 = nn.Sequential(resnet.layer2) # 512
        self.layer3 = nn.Sequential(resnet.layer3) # 1024
        self.layer4 = nn.Sequential(resnet.layer4) # 2048

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Depth prediction
        self.predict1 = smooth(256, 64)
        self.predict2 = predict(64, 1)
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        _,_,H,W = x.size() # batchsize N,channel,height,width
        
        # Bottom-up
        c1 = self.layer0(x) 
        c2 = self.layer1(c1) # 256 channels, 1/4 size
        c3 = self.layer2(c2) # 512 channels, 1/8 size
        c4 = self.layer3(c3) # 1024 channels, 1/16 size
        c5 = self.layer4(c4) # 2048 channels, 1/32 size

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # 256 channels, 1/16 size
        p4 = self.smooth1(p4) 
        p3 = self._upsample_add(p4, self.latlayer2(c3)) # 256 channels, 1/8 size
        p3 = self.smooth2(p3) # 256 channels, 1/8 size
        p2 = self._upsample_add(p3, self.latlayer3(c2)) # 256, 1/4 size
        p2 = self.smooth3(p2) # 256 channels, 1/4 size

        return self.predict2( self.predict1(p2) )     # depth; 1/4 size, mode = "L"

device = None
rgb_transform = None
i2d = None

def after_change_settings(key: str, value: float) -> None:
    if key == "Scale":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def on_load() -> None:
    global device, rgb_transform, i2d
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rgb_transform = Compose([ToTensor()])
    with torch.no_grad():
        i2d = I2D().to(device)
        i2d.load_state_dict(torch.load(model_path, map_location='cpu'))
        i2d.eval()
    print(f'Pipeline="{name}" was loaded with {"CUDA" if device.type == "cuda" else "CPU"} support.')

def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    global device, rgb_transform, i2d
    scale = read_value(settings, "Scale")
    with torch.no_grad():
        if scale != 1:
            frame = cv2.resize(frame, (int(scale * width), int(scale * height)))
        frame = rgb_transform(frame).float().unsqueeze(0)
        frame = frame.to(device)
        frame = i2d(frame)
        frame = frame.int().squeeze(0)
        frame = frame.cpu().numpy()
        
        img = np.zeros((frame.shape[1], frame.shape[2], 3)).astype(np.uint8)
        img[:,:,0] = (frame / 65535 * 255).astype(np.uint8)
        img[:,:,1] = img[:,:,0]
        img[:,:,2] = img[:,:,0]

        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        return img

def on_unload() -> None:
    global device, rgb_transform, i2d
    device = None
    rgb_transform = None
    i2d = None
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    print(f'Pipeline="{name}" was unloaded.')