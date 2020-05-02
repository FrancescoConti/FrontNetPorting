import torch.nn as nn
import torch
from ConvBlock import ConvBlock, BypassBlock
from ConvBlock import printRes
import numpy as np
import logging
from nemo.quant.pact import PACT_IntegerAdd, PACT_Identity

np.set_printoptions(threshold=np.inf)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class KeepLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(KeepLinear, self).__init__(*args, **kwargs)

class KeepConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(KeepConv2d, self).__init__(*args, **kwargs)

class KeepBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(KeepBatchNorm2d, self).__init__(*args, **kwargs)

class KeepReLU(nn.ReLU):
    def __init__(self, *args, **kwargs):
        super(KeepReLU, self).__init__(*args, **kwargs)

class HannaNet(nn.Module):
    def __init__(self, block, layers, isGray=False, enable_prefc=False, keep_linear=False, residuals=False):
        super(HannaNet, self).__init__()

        if isGray ==True:
            self.name = "HannaNetGray"
        else:
            self.name = "HannaNetRGB"
        self.inplanes = 32
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        self.groups = 1
        self.base_width = 64
        self.enable_prefc = enable_prefc
        self.residuals = residuals

        if isGray == True:
            self.conv = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        else:
            self.conv = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU()

        self.layer1  = ConvBlock(32, 32, stride=2)
        if self.residuals:
            self.bypass1 = BypassBlock(32, 32, stride=2)
            self.add1    = PACT_IntegerAdd()
        self.layer2  = ConvBlock(32, 64, stride=2)
        if self.residuals:
            self.bypass2 = BypassBlock(32, 64, stride=2)
            self.add2    = PACT_IntegerAdd()
        self.layer3  = ConvBlock(64, 128, stride=2)
        if self.residuals:
            self.bypass3 = BypassBlock(64, 128, stride=2)
            self.add3    = PACT_IntegerAdd()

        if self.enable_prefc:
            self.prefc      = nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=False)
            self.prefc_bn   = nn.BatchNorm2d(128)
            self.prefc_relu = nn.ReLU()

        self.dropout = PACT_Identity() #nn.Dropout()

        fcSize = 1920
        if keep_linear:
            self.fc = KeepLinear(fcSize, 4)
        else:
            self.fc = nn.Linear(fcSize, 4)

    def forward(self, x):

        conv5x5 = self.conv(x)
        btn = self.bn(conv5x5)
        relu1 = self.relu1(btn)
        max_pool = self.maxpool(relu1)

        l1 = self.layer1(max_pool)
        if self.residuals: 
            b1 = self.bypass1(max_pool)
            a1 = self.add1(l1, b1)
        else:
            a1 = l1
        l2 = self.layer2(a1)
        if self.residuals: 
            b2 = self.bypass2(a1)
            a2 = self.add2(l2, b2)
        else:
            a2 = l2
        l3 = self.layer3(a2)
        if self.residuals: 
            b3 = self.bypass3(a2)
            a3 = self.add3(l3, b3)
        else:
            a3 = l3

        if self.enable_prefc:
            prefc = self.prefc_relu(self.prefc_bn(self.prefc(a3)))
        else:
            prefc = a3

        out = prefc.flatten(1)

        out = self.dropout(out)
        out = self.fc(out)

        x = out[:, 0]
        y = out[:, 1]
        z = out[:, 2]
        phi = out[:, 3]
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)
        phi = phi.unsqueeze(1)

        return [x, y, z, phi]

def PrintRelu(layer, name):
    logger = logging.getLogger('')
    enable = logger.isEnabledFor(logging.INFO)
    if (enable == True):
        tmp = layer.reshape(-1)
        logging.info("{}={}".format(name, list(tmp.numpy())))

def PrintFC(layer, name):
    logger = logging.getLogger('')
    enable = logger.isEnabledFor(logging.INFO)
    if (enable == True):
        logging.info("{}={}".format(name, layer.numpy()))
