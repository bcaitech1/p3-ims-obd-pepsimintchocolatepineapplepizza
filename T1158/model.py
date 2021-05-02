import torch
import torch.nn as nn
from torchvision import models


class CBR(nn.Module):
    """Convolution + BN + Relu"""

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding,
                              dilation=dilation)
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


def is_transposed_conv2d(module):
    return module.__class__.__name__ == "ConvTranspose2d"


def is_TCBR(module):
    return module.__class__.__name__ == "TCBR"


class DeconvNetEncoder4VGG16(nn.Module):
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


class DeconvNetDecoder4VGG16(nn.Module):
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
        self.encoder = DeconvNetEncoder4VGG16(vgg16.features)
        self.fcn = nn.Sequential(
            CBR(512, 1024, 7, 1, 0),
            nn.Dropout(0.5),
            CBR(1024, 1024, 1, 1, 0),
            nn.Dropout(0.5),
            TCBR(1024, 512, 7, 1, 0)
        )
        self.decoder = DeconvNetDecoder4VGG16(vgg16.features)
        self.classifier = nn.Conv2d(64, classes, 1)

    def forward(self, x):
        x, indices = self.encoder(x)
        x = self.fcn(x)
        x = self.decoder(x, indices)
        return self.classifier(x)


class UNetEncoder4VGG16(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = self._make_module(backbone)

    def _make_module(self, backbone):
        modules = []
        _buffer = []
        for idx, module in enumerate(backbone.children(), 1):
            if is_maxpool(module):
                modules.extend([nn.Sequential(*_buffer),
                                nn.MaxPool2d(module.kernel_size)])
                _buffer = []
            else:
                _buffer.append(module)
        return nn.Sequential(*modules)

    def forward(self, x):
        intermediate_outputs = []
        for module in self.features.children():
            x = module(x)
            if is_maxpool(module):
                continue
            intermediate_outputs.append(x)
        return x, intermediate_outputs


class UNetDecoder4VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 2, 2, 0),
            nn.Sequential(
                CBR(1024, 512, 3, 1, 1),
                CBR(512, 512, 3, 1, 1),
                CBR(512, 512, 3, 1, 1)
            ),
            nn.ConvTranspose2d(512, 512, 2, 2, 0),
            nn.Sequential(
                CBR(1024, 512, 3, 1, 1),
                CBR(512, 512, 3, 1, 1),
                CBR(512, 256, 3, 1, 1)
            ),
            nn.ConvTranspose2d(256, 256, 2, 2, 0),
            nn.Sequential(
                CBR(512, 256, 3, 1, 1),
                CBR(256, 256, 3, 1, 1),
                CBR(256, 128, 3, 1, 1)
            ),
            nn.ConvTranspose2d(128, 128, 2, 2, 0),
            nn.Sequential(
                CBR(256, 128, 3, 1, 1),
                CBR(128, 64, 3, 1, 1)
            ),
            nn.ConvTranspose2d(64, 64, 2, 2, 0),
            nn.Sequential(
                CBR(128, 64, 3, 1, 1)
            )
        )

    def forward(self, x, intermediate_outputs):
        for module in self.features.children():
            if is_transposed_conv2d(module):
                x = module(x)
            else:
                x = torch.cat((x, intermediate_outputs.pop()), 1)
                x = module(x)
        return x


class _UNet(nn.Module):
    def __init__(self, encoder, fcn, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.fcn = fcn
        self.decoder = decoder
        self.classifier = classifier

    def forward(self, x):
        outputs, intermediate_outputs = self.encoder(x)
        outputs = self.fcn(outputs)
        outputs = self.decoder(outputs, intermediate_outputs)
        return self.classifier(outputs)


class UNet4VGG16(_UNet):
    def __init__(self, classes=12):
        vgg16 = models.vgg16_bn(True)
        encoder = UNetEncoder4VGG16(vgg16.features)
        fcn = nn.Sequential(
            CBR(512, 1024, 3, 1, 1),
            nn.Dropout(0.5),
            CBR(1024, 1024, 1, 1, 0),
            nn.Dropout(0.5),
            CBR(1024, 512, 3, 1, 1),
            nn.Dropout(0.5)
        )
        decoder = UNetDecoder4VGG16()
        classifier = nn.Conv2d(64, classes, 1, 1, 0)
        super(UNet4VGG16, self).__init__(encoder, fcn, decoder, classifier)


class DeepLabV1VGG16(nn.Module):
    def __init__(self, classes=12):
        super(DeepLabV1VGG16, self).__init__()
        vgg16 = models.vgg16_bn(True)
        self.modify_last_conv(vgg16.features)
        self.modify_max_pool(vgg16.features)
        self.encoder = vgg16.features
        self.encoder.add_module("avg_pool", nn.AvgPool2d(3, 1, 1))
        self.fcn = nn.Sequential(
            CBR(512, 1024, 3, 1, 12, 12),
            nn.Dropout(0.5),
            CBR(1024, 1024, 1, 1, 0),
            nn.Dropout(0.5),
            CBR(1024, classes, 1, 1, 0)
        )
        self.classifier = nn.UpsamplingBilinear2d(256)

    def modify_last_conv(self, encoder):
        """last 3 convolution layers modify padding, dilation"""
        n_last_conv = 3
        modules = [*encoder.children()][::-1]
        for module in modules:
            if n_last_conv > 0 and is_conv2d(module):
                n_last_conv -= 1
                module.padding = (2, 2)
                module.dilation = (2, 2)
        return

    def modify_max_pool(self, encoder):
        """max pool kernel_size 2 -> 3, stride 2 -> 1, add padding=1"""
        # config = [(kernel_size, stride, padding),...]
        config = [(3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 1, 1), (3, 1, 1)]
        idx = 0
        for module in encoder.children():
            if is_maxpool(module):
                kernel_size, stride, padding = config[idx]
                module.kernel_size = (kernel_size, kernel_size)
                module.stride = (stride, stride)
                module.padding = (padding, padding)
                idx += 1
        return

    def forward(self, x):
        x = self.encoder(x)
        x = self.fcn(x)
        return self.classifier(x)


def get_model(name):
    if name == "DeconvNet4VGG16":
        return DeconvNet4VGG16()
    if name == "UNet4VGG16":
        return UNet4VGG16()
    if name == "DeepLabV1VGG16":
        return DeepLabV1VGG16()
    raise ValueError("incorrect model name: %s" % name)


if __name__ == '__main__':
    model = get_model("DeepLabV1VGG16")
    print(model)
    # model.cuda()
    arr = torch.rand(1, 3, 256, 256)
    output = model(arr)
    print(output.shape)
