import gc
import numpy as np

from pipeline_utils import *

with use_local_python():
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms

name = "[AI] Cartoon-GAN"
version = "1.1.2"
desc = """Torch pipeline for AI cartoon stylization.
Based on "Cartoon-GAN" from:
https://github.com/FilipAndersson245/cartoon-gan"""

settings = {
    "Scale": build_variable(0.75, 0.25, 1.0, 0.05, "Scale image for AI processing.")
}

# Following code is using:
# Cartoon-GAN
# https://github.com/FilipAndersson245/cartoon-gan

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(inplace=True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=0, bias=False)
    )

def add_resblocks(channel_num, nr_blocks):
    return nn.Sequential(*[ResBlock(channel_num) for i in range(nr_blocks)])

class UpBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=2, add_blur=False):
        super(UpBlock, self).__init__()

        self.shuffle = nn.ConvTranspose2d(
            in_f, out_f, kernel_size=3, stride=stride, padding=0)
        self.has_blur = add_blur
        if self.has_blur:
            self.blur = nn.AvgPool2d(2, 1)

    def forward(self, x):
        x = self.shuffle(x)
        if self.has_blur:
            x = self.blur(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1,
                      padding=7//2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            conv3x3(64, 128, stride=2),
            conv3x3(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 256, stride=2),
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.res = nn.Sequential(add_resblocks(256, 8))

        self.up = nn.Sequential(
            UpBlock(256, 128, stride=2, add_blur=True),
            conv3x3(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            UpBlock(128, 64, stride=2, add_blur=True),
            conv3x3(64, 64, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=7//2)
        )

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        return x

def inv_normalize(img):
    # Adding 0.1 to all normalization values since the model is trained (erroneously) without correct de-normalization
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img

device = None
transform = None
net = None

def after_change_settings(key: str, value: float) -> None:
    if key == "Scale":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def on_load() -> None:
    global device, transform, net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        net = Generator()
        model_path = resolve_path('ai_cartoon_gan/trained_netG.pth')
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
    print(f'Pipeline="{name}" was loaded with {"CUDA" if device.type == "cuda" else "CPU"} support.')

def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    global transform, net
    scale = read_value(settings, "Scale")
    with torch.no_grad():
        if scale != 1:
            frame = cv2.resize(frame, (int(scale * width), int(scale * height)))
        t = torch.from_numpy(np.array([transform(frame).numpy()])).to(device)
        img = inv_normalize(net(t)[0])[0].cpu().mul(255).byte()
        img = np.transpose(img.cpu().numpy(), (1, 2, 0))
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        return img

def on_unload() -> None:
    global device, transform, net
    device = None
    transform = None
    net = None
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    print(f'Pipeline="{name}" was unloaded.')