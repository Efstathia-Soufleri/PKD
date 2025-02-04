"""Model definition."""

from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.num_class = num_class

        print(("""
        Initializing model:
        base model:         {}.
        input_representation:     {}.
        num_class:          {}.
        num_segments:       {}.
            """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):

        if self._representation == 'mv':
            setattr(self.base_model, 'conv1',
                    nn.Conv2d(2, 64, 
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)


    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            # print(self.base_model)
            feature_dim = getattr(self.base_model, 'fc').in_features
            setattr(self.base_model, 'fc', nn.Linear(feature_dim, self.num_class))
            self._input_size = 224

        elif 'mobilenet_v2' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1]=nn.Linear(feature_dim, self.num_class)
            self._input_size = 224
        
        elif 'shufflenet_v2_x1_0' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            # print(self.base_model)
            feature_dim = getattr(self.base_model, 'fc').in_features
            setattr(self.base_model, 'fc', nn.Linear(feature_dim, self.num_class))
            self._input_size = 224

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        # print('FORWARD PASS !!!!')
        input = input.view((-1, ) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        base_out = self.base_model(input)
        return base_out

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
