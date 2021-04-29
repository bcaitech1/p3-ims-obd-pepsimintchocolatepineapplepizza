import torch
import torch.nn as nn
from torchvision import models


class CBR(nn.Module):
    """Convolution + BN + Relu"""

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class TCBR(nn.Module):
    """transposed convolution + BN + ReLU"""

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv_t(x)
        x = self.bn(x)
        return self.relu(x)


def is_maxpool(module):
    return module.__class__.__name__ == "MaxPool2d"


def is_max_unpool(module):
    return module.__class__.__name__ == "MaxUnpool2d"


def is_relu(module):
    return module.__class__.__name__ == "ReLU"


def is_conv2d(module):
    return module.__class__.__name__ == "Conv2d"


def is_TCBR(module):
    return module.__class__.__name__ == "TCBR"


class Encoder4VGG16(nn.Module):
    def __init__(self, backbone, return_indices=True):
        super().__init__()
        self.backbone = backbone
        self.return_indices = return_indices
        self.set_maxpool_return_indices()

    def forward(self, x):
        if not self.return_indices:
            return self.backbone(x)
        indices_list = []
        for module in self.backbone.children():
            if is_maxpool(module):
                x, indices = module(x)
                indices_list.append(indices)
            else:
                x = module(x)
        return x, indices_list

    def set_maxpool_return_indices(self):
        for module in self.backbone.children():
            if is_maxpool(module):
                module.return_indices = True


class Decoder4VGG16(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = self._make_module(backbone)

    def _make_module(self, backbone):
        modules = []
        for module in backbone.children():
            if is_maxpool(module):
                modules.append(nn.MaxUnpool2d(module.kernel_size))
            elif is_conv2d(module):
                modules.append(
                    TCBR(module.out_channels,
                         module.in_channels,
                         module.kernel_size,
                         module.stride,
                         module.padding)
                )
            elif is_relu(module):
                continue
        modules.pop(0)
        return nn.Sequential(*modules[::-1])

    def forward(self, x, indices: list):
        for module in self.features.children():
            if is_max_unpool(module):
                x = module(x, indices.pop())
            elif is_TCBR(module):
                x = module(x)
        return x


class DeconvNet4VGG16(nn.Module):
    def __init__(self, classes=12):
        super().__init__()
        vgg16 = models.vgg16_bn(True)
        self.encoder = Encoder4VGG16(vgg16.features)
        self.fcn = nn.Sequential(
            CBR(512, 1024, 7, 1, 0),
            nn.Dropout(0.5),
            CBR(1024, 1024, 1, 1, 0),
            nn.Dropout(0.5),
            TCBR(1024, 512, 7, 1, 0)
        )
        self.decoder = Decoder4VGG16(vgg16.features)
        self.classifier = nn.Conv2d(64, classes, 1)

    def forward(self, x):
        x, indices = self.encoder(x)
        x = self.fcn(x)
        x = self.decoder(x, indices)
        return self.classifier(x)


def get_model(name):
    if name == "DeconvNet4VGG16":
        return DeconvNet4VGG16()
    raise ValueError("incorrect model name: %s" % name)


if __name__ == '__main__':
    model = get_model("DeconvNet4VGG16")
    print(model, "\n")
    inputs = torch.rand(1, 3, 512, 512)
    outputs = model(inputs)
    print(outputs.shape)
