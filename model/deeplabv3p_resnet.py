import sys
# from .cc_attention import CrissCrossAttention
sys.path.insert(0, '..')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
from model.criss_cross import CrissCrossAttention


def get_sobel(in_chan):
    filter_x = np.array([[
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ] for _ in range(in_chan)]).astype(np.float32)
    filter_y = np.array([[
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ] for _ in range(in_chan)]).astype(np.float32)
    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x.unsqueeze(0), requires_grad=False)
    filter_y = nn.Parameter(filter_y.unsqueeze(0), requires_grad=False)
    conv_x = nn.Conv2d(in_chan, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    return conv_x, conv_y

def run_sobel(conv_x, conv_y,input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g_x = torch.sigmoid(g_x)
    g_y = torch.sigmoid(g_y)
    g = torch.sqrt(torch.pow(g_x,2)+ torch.pow(g_y,2))
    return g


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False, sobel=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError
        self.sobel = sobel
        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])
        if self.sobel:
            print("------------\nuse sobel\n-----------")
            self.conv_x, self.conv_y = get_sobel(512)

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        if self.sobel:
            weight = run_sobel(self.conv_x,self.conv_y,x)
            x = x * weight
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict and 'conv1' not in k:
                model_dict[k] = v
        state_dict.update(model_dict)
        # import pdb
        # pdb.set_trace()
        try:
            self.load_state_dict(state_dict,strict=False)
        except:
            model_dict = {}
            for k, v in pretrain_dict.items():
                if k in state_dict and 'conv1' not in k:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.load_state_dict(state_dict, strict=False)


def ResNet101(nInputChannels=3, os=16, pretrained=False,sobel=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained,sobel=sobel)
    return model


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ZPool2D(nn.Module):
    def __init__(self):
        super(ZPool2D, self).__init__()

    def forward(self, X, mu):
        # mu = self.avgpool(X)
        # sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)).sum() / (X.shape[-2] * X.shape[-1]))
        D = X - mu
        return D
        # return D / sigma


class DeepLabv3_plus_res101(nn.Module):
    def __init__(self, nInputChannels=3, out_channels=1, os=16, pretrained=False, _print=True,
                 ela=0, cc=False, zpool=False, hp=False, nocat=False, sobel=False):

        super(DeepLabv3_plus_res101, self).__init__()
        self.ela = ela
        if self.ela:
            self.filters = self.SRMLayer()
            nInputChannels = 6

        self.hp = hp
        if self.hp:
            self.hp_filters = self.HPfilters()
            nInputChannels = 9

        self.cc = cc
        if self.cc:
            self.criss_cross = CrissCrossAttention(2048)
            self.cc_down = nn.Sequential(
                nn.Conv2d(2048, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        self.zpool = zpool
        if self.zpool:
            self.Zp = ZPool2D()
        self.nocat = nocat

        # Atrous Conv
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained,sobel=sobel)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if self.zpool:
            self.aspp1 = ASPP_module(2048, 2048, rate=rates[0])
            self.aspp2 = ASPP_module(2048, 2048, rate=rates[1])
            self.aspp3 = ASPP_module(2048, 2048, rate=rates[2])
            self.aspp4 = ASPP_module(2048, 2048, rate=rates[3])

            self.down_chan1 = nn.Sequential(
                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU())
            self.down_chan2 = nn.Sequential(
                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU())
            self.down_chan3 = nn.Sequential(
                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU())
            self.down_chan4 = nn.Sequential(
                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU())
            self.down_chan5 = nn.Sequential(
                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU())

        else:
            self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
            self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
            self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
            self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()
        if self.zpool:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                                 nn.BatchNorm2d(256),
                                                 nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        if self.nocat:
            self.last_inchal = 256
        else:
            self.last_inchal = 304
        self.last_conv = nn.Sequential(nn.Conv2d(self.last_inchal, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, out_channels, kernel_size=1, stride=1))
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(out_channels))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))

    def SRMLayer(self):
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = np.asarray(
            [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])  # shape=(3,3,5,5)
        filters = torch.from_numpy(filters.astype(np.float32))
        filters = nn.Parameter(filters, requires_grad=False)
        return filters

    def HPfilters(self):
        filter1 = [[0, 0, 0],
                   [0, -1, 0],
                   [0, 1, 0]]
        filter2 = [[0, 0, 0],
                   [0, -1, 1],
                   [0, 0, 0]]
        filter3 = [[0, 0, 0],
                   [0, -1, 0],
                   [0, 0, 1]]
        filters = np.asarray([[filter1], [filter2], [filter3]])  # 3*1*3*3
        filters = torch.from_numpy(filters.astype(np.float32))
        filters = nn.Parameter(filters, requires_grad=True)
        return filters

    def forward(self, input):
        if self.ela:
            srm = F.conv2d(input, self.filters, bias=None, stride=1, padding=2, dilation=1, groups=1)
            input = torch.cat((input, srm), dim=1)
        elif self.hp:
            in1, in2, in3 = input.split(1, 1)
            # print(in1.size())
            hp_f0 = F.conv2d(in1, self.hp_filters, bias=None, stride=1, padding=1, dilation=1, groups=1)
            hp_f1 = F.conv2d(in2, self.hp_filters, bias=None, stride=1, padding=1, dilation=1, groups=1)
            hp_f2 = F.conv2d(in3, self.hp_filters, bias=None, stride=1, padding=1, dilation=1, groups=1)
            input = torch.cat((hp_f0, hp_f1, hp_f2), dim=1)
            # print(hp_f0.size(), input.size())

        x, low_level_features = self.resnet_features(input)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        # import pdb
        # pdb.set_trace()
        if self.cc:
            x5 = self.criss_cross(x)
            x5 = self.criss_cross(x5)  # 2048
            # x5 = self.cc_down(x5)
        else:
            x5 = self.global_avg_pool(x)
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        if self.zpool:
            x1 = self.down_chan1(self.Zp(x, x1))
            x2 = self.down_chan2(self.Zp(x, x2))
            x3 = self.down_chan3(self.Zp(x, x3))
            x4 = self.down_chan4(self.Zp(x, x4))
            x5 = self.down_chan5(self.Zp(x, x5))
        else:
            if self.cc:
                x5 = self.cc_down(x5)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                   int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
        if not self.nocat:
            low_level_features = self.conv2(low_level_features)
            low_level_features = self.bn2(low_level_features)
            low_level_features = self.relu(low_level_features)

            x = torch.cat((x, low_level_features), dim=1)

        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus_res101(nInputChannels=3, out_channels=1, os=16, pretrained=False, _print=True, nocat=True)
    print("nocat")
    model.eval()
    image = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(image)
    print(output.size())
