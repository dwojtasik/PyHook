import gc
import numpy as np

from pipeline_utils import *

with use_local_python():
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable

name = "[AI] Multi Style Transfer"
version = "0.8.1"
desc = """Torch pipeline for fast multi style transfer.
Based on "PyTorch-Style-Transfer" from:
https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer"""

settings = {
    "Scale": build_variable(0.75, 0.25, 1.0, 0.05, "Scale image for AI processing."),
    "Style rotation": build_variable(False, None, None, None, "Rotate though each style."),
    "Frame rotation": build_variable(30, 10, 300, 10, "Frame count before style is changed."),
    "Style": build_variable(
        0, 0, 20, 1, "%COMBO[Candy,Composition,Delaunay,Escher,Feathers,Frida,La muse,Mosaic duck,Mosaic,Pencil,Picasso,Rain princess,Scream,Seated nude,Shipwreck,Starry night,Stars,Strip,Udnie,Wave,Woman]Style image."
    ),
}

style_dir = "ai_multi_style_transfer"
paths = [
    resolve_path(f"{style_dir}\\candy.jpg"),
    resolve_path(f"{style_dir}\\composition_vii.jpg"),
    resolve_path(f"{style_dir}\\Robert_Delaunay,_1906,_Portrait.jpg"),
    resolve_path(f"{style_dir}\\escher_sphere.jpg"),
    resolve_path(f"{style_dir}\\feathers.jpg"),
    resolve_path(f"{style_dir}\\frida_kahlo.jpg"),
    resolve_path(f"{style_dir}\\la_muse.jpg"),
    resolve_path(f"{style_dir}\\mosaic_ducks_massimo.jpg"),
    resolve_path(f"{style_dir}\\mosaic.jpg"),
    resolve_path(f"{style_dir}\\pencil.jpg"),
    resolve_path(f"{style_dir}\\picasso_selfport1907.jpg"),
    resolve_path(f"{style_dir}\\rain_princess.jpg"),
    resolve_path(f"{style_dir}\\the_scream.jpg"),
    resolve_path(f"{style_dir}\\seated-nude.jpg"),
    resolve_path(f"{style_dir}\\shipwreck.jpg"),
    resolve_path(f"{style_dir}\\starry_night.jpg"),
    resolve_path(f"{style_dir}\\stars2.jpg"),
    resolve_path(f"{style_dir}\\strip.jpg"),
    resolve_path(f"{style_dir}\\udnie.jpg"),
    resolve_path(f"{style_dir}\\wave.jpg"),
    resolve_path(f"{style_dir}\\woman-with-hat-matisse.jpg"),
]

# Following code is using:
# PyTorch-Style-Transfer
# https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = cv2.resize(img, (size, size2))
        else:
            img = cv2.resize(img, (size, size))

    elif scale is not None:
        img = cv2.resize(img, (int(img.size[0] / scale), int(img.size[1] / scale)))
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


class StyleLoader():
    def __init__(self, style_size, cuda=True):
        self.style_size = style_size
        self.files = paths
        self.cuda = cuda
    
    def get(self, i):
        idx = i%len(self.files)
        filepath = self.files[idx]
        style = tensor_load_rgbimage(filepath, self.style_size)    
        style = style.unsqueeze(0)
        style = preprocess_batch(style)
        if self.cuda:
            style = style.cuda()
        style_v = Variable(style, requires_grad=False)
        return style_v

    def size(self):
        return len(self.files)


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def var(x, dim=0):
    '''
    Calculates variance.
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)

class MultConst(nn.Module):
    def forward(self, input):
        return 255*input


class GramMatrix(nn.Module):
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram
    

class Basicblock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Basicblock, self).__init__()
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes,
                                                        kernel_size=1, stride=stride)
        conv_block=[]
        conv_block+=[norm_layer(inplanes),
                                nn.ReLU(inplace=True),
                                ConvLayer(inplanes, planes, kernel_size=3, stride=stride),
                                norm_layer(planes),
                                nn.ReLU(inplace=True),
                                ConvLayer(planes, planes, kernel_size=3, stride=1),
                                norm_layer(planes)]
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, input):
        if self.downsample is not None:
            residual = self.residual_layer(input)
        else:
            residual = input
        return residual + self.conv_block(input)
            

class UpBasicblock(nn.Module):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBasicblock, self).__init__()
        self.residual_layer = UpsampleConvLayer(inplanes, planes,
                                                      kernel_size=1, stride=1, upsample=stride)
        conv_block=[]
        conv_block+=[norm_layer(inplanes),
                                nn.ReLU(inplace=True),
                                UpsampleConvLayer(inplanes, planes, kernel_size=3, stride=1, upsample=stride),
                                norm_layer(planes),
                                nn.ReLU(inplace=True),
                                ConvLayer(planes, planes, kernel_size=3, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, input):
        return self.residual_layer(input) + self.conv_block(input)


class Bottleneck(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
                                                        kernel_size=1, stride=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes),
                                    nn.ReLU(inplace=True),
                                    ConvLayer(planes, planes, kernel_size=3, stride=stride)]
        conv_block += [norm_layer(planes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class UpBottleneck(nn.Module):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                      kernel_size=1, stride=1, upsample=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes),
                                    nn.ReLU(inplace=True),
                                    UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)]
        conv_block += [norm_layer(planes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return  self.residual_layer(x) + self.conv_block(x)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class Inspiration(nn.Module):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1,C,C), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.Tensor(B,C,C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        # input X is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(self.G),self.G)
        return torch.bmm(self.P.transpose(1,2).expand(X.size(0), self.C, self.C), X.view(X.size(0),X.size(1),-1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]


class Net(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6, gpu_ids=[]):
        super(Net, self).__init__()
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        model1 = []
        model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                            norm_layer(64),
                            nn.ReLU(inplace=True),
                            block(64, 32, 2, 1, norm_layer),
                            block(32*expansion, ngf, 2, 1, norm_layer)]
        self.model1 = nn.Sequential(*model1)

        model = []
        self.ins = Inspiration(ngf*expansion)
        model += [self.model1]
        model += [self.ins]    

        for i in range(n_blocks):
            model += [block(ngf*expansion, ngf, 1, None, norm_layer)]
        
        model += [upblock(ngf*expansion, 32, 2, norm_layer),
                            upblock(32*expansion, 16, 2, norm_layer),
                            norm_layer(16*expansion),
                            nn.ReLU(inplace=True),
                            ConvLayer(16*expansion, output_nc, kernel_size=7, stride=1)]

        self.model = nn.Sequential(*model)

    def setTarget(self, Xs):
        F = self.model1(Xs)
        G = self.gram(F)
        self.ins.setTarget(G)

    def forward(self, input):
        return self.model(input)


style_model = None
style_loader = None
cuda = False


def after_change_settings(key: str, value: float) -> None:
    global style_model, style_loader
    if key == "Scale":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if key == "Style rotation":
        if value == 0: # Reset to selected style
            with torch.no_grad():
                style = read_value(settings, "Style")
                style_v = style_loader.get(style)
                style_v = Variable(style_v.data)
                style_model.setTarget(style_v)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if key == "Style":
        rotate_style = read_value(settings, "Style rotation")
        if not rotate_style:
            with torch.no_grad():
                style_v = style_loader.get(int(value))
                style_v = Variable(style_v.data)
                style_model.setTarget(style_v)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def on_load() -> None:
    global style_model, style_loader, cuda
    cuda = torch.cuda.is_available()
    with torch.no_grad():
        style_model = Net(ngf=128)
        model_dict = torch.load(resolve_path(f"{style_dir}\\21styles.model"))
        for key in list(model_dict.keys()):
            if key.endswith(('running_mean', 'running_var')):
                del model_dict[key]
        style_model.load_state_dict(model_dict, False)
        style_model.eval()
        if cuda:
            style_loader = StyleLoader(512)
            style_model.cuda()
        else:
            style_loader = StyleLoader(512, False)
        style = read_value(settings, "Style")
        style_v = style_loader.get(style)
        style_v = Variable(style_v.data)
        style_model.setTarget(style_v)
    print(f'Pipeline="{name}" was loaded with {"CUDA" if cuda else "CPU"} support.')


def on_frame_process(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    global style_model, style_loader, cuda
    scale = read_value(settings, "Scale")
    rotate_style = read_value(settings, "Style rotation")
    with torch.no_grad():
        if scale != 1:
            frame = cv2.resize(frame, (int(scale * width), int(scale * height)))

        if rotate_style:
            rotate_frame = read_value(settings, "Frame rotation")
            if frame_num % rotate_frame == 0:
                style_v = style_loader.get(frame_num // rotate_frame)
                style_v = Variable(style_v.data)
                style_model.setTarget(style_v)

        img = np.array(frame).transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float()
        if cuda:
            img=img.cuda()
        img = Variable(img)
        img = style_model(img)
        if cuda:
            img = img.cpu().clamp(0, 255).data[0].numpy()
        else:
            img = img.clamp(0, 255).data[0].numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        return img


def on_unload() -> None:
    global style_model, style_loader, cuda
    style_model = None
    style_loader = None
    cuda = False
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    print(f'Pipeline="{name}" was unloaded.')
