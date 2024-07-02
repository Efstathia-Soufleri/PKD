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

class Model_unrolled(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet152'):
        super(Model_unrolled, self).__init__()
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

        if self._representation == 'mv':
            self.data_bn = nn.BatchNorm2d(2)
            self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.data_bn = nn.BatchNorm2d(3)
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        # Basic Block 0
        # layer -> block -> layer module 
        self.conv1_0_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_0_1 = nn.BatchNorm2d(64)
        self.relu1_0_1 = nn.ReLU(inplace=True)

        self.conv1_0_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_0_2 = nn.BatchNorm2d(64)

        # Basic Block 1
        self.conv1_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1_1 = nn.BatchNorm2d(64)
        self.relu1_1_1 = nn.ReLU(inplace=True)

        self.conv1_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1_2 = nn.BatchNorm2d(64)

        # Layer 2
        # Basic Block 0
        self.conv2_0_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_0_1 = nn.BatchNorm2d(128)
        self.relu2_0_1 = nn.ReLU(inplace=True)

        self.conv2_0_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_0_2 = nn.BatchNorm2d(128)

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        # Basic Block 1
        self.conv2_1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1_1 = nn.BatchNorm2d(128)
        self.relu2_1_1 = nn.ReLU(inplace=True)

        self.conv2_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1_2 = nn.BatchNorm2d(128)

        # Layer 3
        # Basic Block 0
        self.conv3_0_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_0_1 = nn.BatchNorm2d(256)
        self.relu3_0_1 = nn.ReLU(inplace=True)

        self.conv3_0_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_0_2 = nn.BatchNorm2d(256)

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
       
        # Basic Block 1
        self.conv3_1_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1_1 = nn.BatchNorm2d(256)
        self.relu3_1_1 = nn.ReLU(inplace=True)

        self.conv3_1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1_2 = nn.BatchNorm2d(256)

        # Layer 4
        # Basic Block 0
        self.conv4_0_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_0_1 = nn.BatchNorm2d(512)
        self.relu4_0_1 = nn.ReLU(inplace=True)

        self.conv4_0_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_0_2 = nn.BatchNorm2d(512)

        self.shortcut4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
       
        # Basic Block 1
        self.conv4_1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1_1 = nn.BatchNorm2d(512)
        self.relu4_1_1 = nn.ReLU(inplace=True)

        self.conv4_1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1_2 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_class)

        

    def forward(self, input):
        x = input.view((-1, ) + input.size()[-3:])
        x = self.data_bn(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1
        # Basic Block 1
        identity = x
        x = self.conv1_0_1(x) 
        x = self.bn1_0_1(x) 
        x = self.relu1_0_1(x) 

        x = self.conv1_0_2(x) 
        x = self.bn1_0_2(x) 
        x += identity
        x = self.relu(x)

        # Basic Block 1
        identity = x
        x = self.conv1_1_1(x) 
        x = self.bn1_1_1(x) 
        x = self.relu1_1_1(x) 

        x = self.conv1_1_2(x) 
        x = self.bn1_1_2(x) 
        x += identity
        x = self.relu(x)

        # Layer 2
        # Basic Block 0
        identity = x
        x = self.conv2_0_1(x) 
        x = self.bn2_0_1(x) 
        x = self.relu2_0_1(x) 

        x = self.conv2_0_2(x) 
        x = self.bn2_0_2(x) 

        x += self.shortcut2(identity)
        x = self.relu(x)

        # Basic Block 1
        identity = x
        x = self.conv2_1_1(x)
        x = self.bn2_1_1(x) 
        x = self.relu2_1_1(x)

        x = self.conv2_1_2(x)
        x = self.bn2_1_2(x)
        x += identity
        x = self.relu(x)

        # Layer 3
        # Basic Block 0
        identity = x
        x = self.conv3_0_1(x)
        x = self.bn3_0_1(x) 
        x = self.relu3_0_1(x)

        x = self.conv3_0_2(x) 
        x = self.bn3_0_2(x) 

        x += self.shortcut3(identity)
        x = self.relu(x)
       
        # Basic Block 1
        identity = x
        x = self.conv3_1_1(x) 
        x = self.bn3_1_1(x)
        x = self.relu3_1_1(x) 

        x = self.conv3_1_2(x) 
        x = self.bn3_1_2(x) 
        x += identity
        x = self.relu(x)

        # Layer 4
        # Basic Block 0
        identity = x
        x = self.conv4_0_1(x) 
        x = self.bn4_0_1(x) 
        x = self.relu4_0_1(x) 

        x = self.conv4_0_2(x) 
        x = self.bn4_0_2(x) 

        x += self.shortcut4(identity)
        x = self.relu(x)
       
        # Basic Block 1
        identity = x
        x = self.conv4_1_1(x)
        x = self.bn4_1_1(x) 
        x = self.relu4_1_1(x) 

        x = self.conv4_1_2(x) 
        x = self.bn4_1_2(x) 
        x += identity
        x = self.relu(x)

        x = self.avgpool(x) 

        x = torch.flatten(x, 1)

        x = self.fc(x) 

        return x

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
