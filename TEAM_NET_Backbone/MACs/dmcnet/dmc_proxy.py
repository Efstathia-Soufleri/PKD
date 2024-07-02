
"""Model definition."""

import torch
from torch import nn
import transforms_custom
import torchvision
import torch.nn.functional as F
from dmcnet.dmc_net_gen import ContextNetwork

class DMCNet(nn.Module):
    def __init__(self, num_class, 
                 is_shift=False, shift_div=8, shift_place='blockres',
                 temporal_pool=False,
                 base_model='resnet152',
                 num_p_segments=25,
                 num_i_segments=3,
                 dropout = 0.5):
        super(DMCNet, self).__init__()
        self._enable_pbn = False
        self._num_p_segments = num_p_segments
        self._num_i_segments = num_i_segments
        self.I_model = TSNResNet(num_class, self._num_i_segments, representation='iframe',
                              base_model=base_model, is_shift=is_shift, shift_div=8,
                              dropout = dropout)
        self.MV_model = TSNResNet(num_class, self._num_p_segments, representation='mv',
                              base_model='resnet18', is_shift=is_shift, shift_div=8,
                              dropout = dropout)
        self.R_model = TSNResNet(num_class, self._num_p_segments, representation='residual',
                              base_model='resnet18', is_shift=is_shift, shift_div=8,
                              dropout = dropout)
        self.OF_model = TSNResNet(num_class, self._num_p_segments, representation='mv',
                            base_model='resnet18', is_shift=is_shift, shift_div=8,
                            dropout = dropout)
    
        self.gen_flow_model = ContextNetwork(5, True, 0)


    def forward(self, i_x, mv_x, r_x):
        i_x = i_x.view((-1, ) + i_x.size()[-3:])

        mv_x = mv_x.view((-1, ) + mv_x.size()[-3:])
        
        r_x = r_x.view((-1, ) + r_x.size()[-3:])

        of_x = self.gen_flow_model(torch.cat((mv_x, r_x), 1))

        of_x = of_x.view((-1, ) + of_x.size()[-3:])
        
        mv_x = self.MV_model.data_bn(mv_x)
        of_x = self.OF_model.data_bn(of_x)

        
        r_x = self.R_model.data_bn(r_x)

        # layer0
        i_x = self.I_model.base_model.conv1(i_x)
        i_x = self.I_model.base_model.bn1(i_x)
        i_x = self.I_model.base_model.relu(i_x)
        i_x = self.I_model.base_model.maxpool(i_x)

        mv_x = self.MV_model.base_model.conv1(mv_x)
        mv_x = self.MV_model.base_model.bn1(mv_x)
        mv_x = self.MV_model.base_model.relu(mv_x)
        mv_x = self.MV_model.base_model.maxpool(mv_x)

        of_x = self.OF_model.base_model.conv1(of_x)
        of_x = self.OF_model.base_model.bn1(of_x)
        of_x = self.OF_model.base_model.relu(of_x)
        of_x = self.OF_model.base_model.maxpool(of_x)

        r_x = self.R_model.base_model.conv1(r_x)
        r_x = self.R_model.base_model.bn1(r_x)
        r_x = self.R_model.base_model.relu(r_x)
        r_x = self.R_model.base_model.maxpool(r_x)


        # layer1
        i_x_res1 = self.I_model.base_model.layer1(i_x)
        mv_x_res1 = self.MV_model.base_model.layer1(mv_x)
        of_x_res1 = self.OF_model.base_model.layer1(of_x)
        r_x_res1 = self.R_model.base_model.layer1(r_x)


        # layer2
        i_x_res2 = self.I_model.base_model.layer2(i_x_res1)
        mv_x_res2 = self.MV_model.base_model.layer2(mv_x_res1)
        r_x_res2 = self.R_model.base_model.layer2(r_x_res1)
        of_x_res2 = self.OF_model.base_model.layer2(of_x_res1)

        # layer3
        i_x_res3 = self.I_model.base_model.layer3(i_x_res2)
        mv_x_res3 = self.MV_model.base_model.layer3(mv_x_res2)
        r_x_res3 = self.R_model.base_model.layer3(r_x_res2)     
        of_x_res3 = self.OF_model.base_model.layer3(of_x_res2)

        # layer4
        i_x_res4 = self.I_model.base_model.layer4(i_x_res3)
        mv_x_res4 = self.MV_model.base_model.layer4(mv_x_res3)
        r_x_res4 = self.R_model.base_model.layer4(r_x_res3)
        of_x_res4 = self.OF_model.base_model.layer4(of_x_res3)



        i_x_pool = self.I_model.base_model.avgpool(i_x_res4)
        mv_x_pool = self.MV_model.base_model.avgpool(mv_x_res4)
        of_x_pool = self.OF_model.base_model.avgpool(of_x_res4)
        r_x_pool = self.R_model.base_model.avgpool(r_x_res4)


        i_x_pool = i_x_pool.squeeze()
        mv_x_pool = mv_x_pool.squeeze()
        r_x_pool = r_x_pool.squeeze()
        of_x_pool = of_x_pool.squeeze()

        # pdb.set_trace()
        i_x_pool = self.I_model.base_model.fc(i_x_pool)
        mv_x_pool = self.MV_model.base_model.fc(mv_x_pool)
        r_x_pool = self.R_model.base_model.fc(r_x_pool)
        of_x_pool = self.OF_model.base_model.fc(of_x_pool)

        i_x_pred = self.I_model.new_fc(i_x_pool)
        mv_x_pred = self.MV_model.new_fc(mv_x_pool)
        r_x_pred = self.R_model.new_fc(r_x_pool)
        of_x_pool = self.OF_model.new_fc(of_x_pool)


        i_x_pred = i_x_pred.view((-1,self._num_i_segments)+i_x_pred.size()[1:]).mean(1)
        mv_x_pred = mv_x_pred.view((-1,self._num_p_segments)+mv_x_pred.size()[1:]).mean(1)
        of_x_pred = of_x_pool.view((-1,self._num_p_segments)+of_x_pool.size()[1:]).mean(1)
        r_x_pred = r_x_pred.view((-1,self._num_p_segments)+r_x_pred.size()[1:]).mean(1)
        out = i_x_pred + mv_x_pred + r_x_pred + of_x_pred
        return out

    def get_optim_policies(self):
        params_dict = dict(self.named_parameters())
        params = []
        for key, value in params_dict.items():
            decay_mult = 0.0 if 'bias' or 'bn' in key else 1.0
            if 'new_fc.weight' in key:
                lr_mult = 5
            elif 'new_fc.bias' in key:
                lr_mult = 10
            else:
                lr_mult = 1

            params += [{'params': value, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]
        return params


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class TSNResNet(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 is_shift=False, shift_div=8, shift_place='blockres',
                 temporal_pool=False, dropout=0.5,
                 base_model='resnet152'):
        super(TSNResNet, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.dropout = dropout

    #     print(("""
    # Initializing model:
    # base model:         {}.
    # input_representation:     {}.
    # num_class:          {}.
    # num_segments:       {}.
    #     """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, num_class)

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
            self.base_model = getattr(torchvision.models, base_model)()
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            self.base_model.last_layer_name = 'fc'
            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        input = input.view((-1, ) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        base_out = self.base_model(input)
        return self.new_fc(base_out)

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224


    def get_augmentation(self, data_name):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)

        return torchvision.transforms.Compose(
            [transforms.GroupMultiScaleCrop(self._input_size, scales),
            transforms.GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))
            ])

if __name__ == '__main__':
    model = TSNResNet(101, 3, 'iframe',
                  base_model='resnet50', is_shift=False, shift_div=8) 
    pdb.set_trace()