"""Model definition."""

from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision
import torch

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class model_unrolled_IC_resnet50(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet50'):
        super(model_unrolled_IC_resnet50, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self._input_size = 224

        print(("""
        Initializing model:
        base model:         {}.
        input_representation:     {}.
        num_class:          {}.
        num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

####
        norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        # if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
        replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)

        # IC1
        self.IC1_features = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=7, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(256))
        self.IC_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC1_classifier = nn.Linear(256 * 4 *4, num_class)

        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        
        # IC2
        self.IC2_features = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=7, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(512))
        self.IC_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC2_classifier = nn.Linear(512 * 4 *4, num_class)

        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        # IC3
        self.IC3_features = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(1024))
        # self.IC2_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC3_classifier = nn.Linear(1024 * 4 * 4, num_class)


        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_class)


        
    def forward(self, input):
        input = input.view((-1, ) + input.size()[-3:])

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # IC1
        out_IC1 = self.IC1_features(x)
        out_IC1 = self.IC_adaptive_avg_pool(out_IC1)
        out_IC1 = torch.flatten(out_IC1, 1)
        out_IC1 = self.IC1_classifier(out_IC1)

        x = self.layer2(x)

        # IC2
        out_IC2 = self.IC2_features(x)
        out_IC2 = self.IC_adaptive_avg_pool(out_IC2)
        out_IC2 = torch.flatten(out_IC2, 1)
        out_IC2 = self.IC2_classifier(out_IC2)

        x = self.layer3(x)

        # IC3
        out_IC3 = self.IC3_features(x)
        out_IC3 = self.IC_adaptive_avg_pool(out_IC3)
        out_IC3 = torch.flatten(out_IC3, 1)
        out_IC3 = self.IC3_classifier(out_IC3)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return out_IC1, out_IC2, out_IC3, x
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))])
