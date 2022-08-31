import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from utils import build_variable, read_value, resolve_path

name = "AI Fast style transfer"
version = "0.0.1"
desc = "Torch pipeline for fast style transfer."

settings = {
    "Scale": build_variable(0.75, 0.25, 1.0, 0.05, "Scale image for AI processing.")
}

# Following code is using:
# Fast neural style with MobileNetV2 bottleneck blocks
# https://github.com/mmalotin/pytorch-fast-neural-style-mobilenetV2

class Bottleneck(nn.Module):
    def __init__(self, inp_c, out_c, kernel_size, stride, t=1):
        assert stride in [1, 2], 'stride must be either 1 or 2'
        super().__init__()
        self.residual = stride == 1 and inp_c == out_c
        pad = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv1 = nn.Conv2d(inp_c, t*inp_c, 1, 1, bias=False)
        self.in1 = nn.InstanceNorm2d(t*inp_c, affine=True)
        self.conv2 = nn.Conv2d(t*inp_c, t*inp_c, kernel_size, stride,
                               groups=t*inp_c, bias=False)
        self.in2 = nn.InstanceNorm2d(t*inp_c, affine=True)
        self.conv3 = nn.Conv2d(t*inp_c, out_c, 1, 1, bias=False)
        self.in3 = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        out = F.relu6(self.in1(self.conv1(x)))
        out = self.reflection_pad(out)
        out = F.relu6(self.in2(self.conv2(out)))
        out = self.in3(self.conv3(out))
        if self.residual:
            out = x + out
        return out

class UpsampleConv(nn.Module):
    def __init__(self, inp_c, out_c, kernel_size, stride, upsample=2):
        super().__init__()
        if upsample:
            self.upsample = nn.Upsample(mode='nearest', scale_factor=upsample)
        else:
            self.upsample = None
        self.conv1 = Bottleneck(inp_c, out_c, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample is not None:
            x_in = self.upsample(x_in)
        out = F.relu(self.conv1(x_in))
        return out

class TransformerMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv Layers
        self.reflection_pad = nn.ReflectionPad2d(9//2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, bias=False)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = Bottleneck(32, 64, kernel_size=3, stride=2)
        self.conv3 = Bottleneck(64, 128, kernel_size=3, stride=2)
        # Residual Layers
        self.res1 = Bottleneck(128, 128,  3, 1)
        self.res2 = Bottleneck(128, 128,  3, 1)
        self.res3 = Bottleneck(128, 128,  3, 1)
        self.res4 = Bottleneck(128, 128,  3, 1)
        self.res5 = Bottleneck(128, 128,  3, 1)
        # Upsampling Layers
        self.upconv1 = UpsampleConv(128, 64, kernel_size=3, stride=1)
        self.upconv2 = UpsampleConv(64, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = F.relu(self.in1(self.conv1(out)))
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.upconv1(out)
        out = self.upconv2(out)
        out = self.conv4(self.reflection_pad(out))
        return out

device = None
transform = None
net = None

def on_load() -> None:
    global device, transform, net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255).unsqueeze(0).to(device))
    ])
    with torch.no_grad():
        net = TransformerMobileNet()
        model_path = resolve_path('ai_test/mosaic.pth')
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
    print(f'Pipeline="{name}" was loaded.')

def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    global transform, net
    scale = read_value(settings, "Scale")
    with torch.no_grad():
        if scale != 1:
            frame = cv2.resize(frame, (int(scale * width), int(scale * height)))
        t = transform(frame)
        res = net(t)[0]
        img = res.permute(1, 2, 0).clamp(0, 255)
        img = img.cpu().numpy()
        img = img.astype("uint8")
        if scale != 1:
            img = cv2.resize(img, (width, height))
        return img

def on_unload() -> None:
    global device, transform, net
    del device
    del transform
    del net
    device = None
    transform = None
    net = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'Pipeline="{name}" was unloaded.')