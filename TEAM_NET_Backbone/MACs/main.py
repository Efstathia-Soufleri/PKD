from teamnet.team_net_model import TEAM_Net
from coviar_proxy import Coviar
from dmcnet.dmc_proxy import DMCNet
from c3d.cv_c3d import CVC3D
import torch
from ptflops import get_model_complexity_info
import os
import pandas as pd
from mfnet.mfnet import MFNET_3D
from mimo.mimo_resnet import mimo_resnet18
from torchvision import models

class TeamNetWrapper(torch.nn.Module):
    def __init__(self, num_segments, num_classes=101):
        super(TeamNetWrapper, self).__init__()
        self.model = TEAM_Net(
            num_class=num_classes, 
            num_segments=num_segments,
            base_model='resnet50') 
    
    def forward(self, inputs):
        input_i, input_mv, input_r = inputs[:, :, 0:3], inputs[:, :, 3:5], inputs[:, :, 5:]
        output = self.model(input_i, input_mv, input_r)
        return output       

class CoviarWrapper(torch.nn.Module):
    def __init__(self, num_i_segments=3, num_p_segments=11, i_base='resnet50', num_classes=101):
        super(CoviarWrapper, self).__init__()
        self.model = Coviar(
            num_i_segments=3, 
            num_p_segments=11,
            num_class=num_classes,
            base_model=i_base) 
    
    def forward(self, inputs):
        input_i, input_mv, input_r = inputs[:, :3, 0:3], inputs[:, :, 3:5], inputs[:, :, 5:]
        output = self.model(input_i, input_mv, input_r)
        return output         

class DMCWrapper(torch.nn.Module):
    def __init__(self, num_p_segments=11, num_i_segments=3, i_base='resnet152', num_classes=101):
        super(DMCWrapper, self).__init__()
        self.model = DMCNet(
            num_class=num_classes,
            base_model=i_base,
            num_p_segments=num_p_segments,
            num_i_segments=num_i_segments) 
    
    def forward(self, inputs):
        input_i, input_mv, input_r = inputs[:, :3, 0:3], inputs[:, :, 3:5], inputs[:, :, 5:]
        output = self.model(input_i, input_mv, input_r)
        return output

class CVC3DWrapper(torch.nn.Module):
    def __init__(self, num_segments=16, num_classes=101):
        super(CVC3DWrapper, self).__init__()
        self.model = CVC3D(
            num_classes=num_classes,
            num_segments=num_segments) 
    
    def forward(self, inputs):
        input_i, input_mv, input_r = inputs[:, :, 0:3], inputs[:, :, 3:5], inputs[:, :, 5:]
        output = self.model(input_i, input_mv, input_r)
        return output                                  
                             

if __name__ == '__main__':
    datasets = [(101, 'ucf101'), (51, 'hmdb51')]
    methods = []
    gmacs = []
    lengths = []
    all_datasets = []
    for num_classes, dataset in datasets:
        total_length = 0
        save_path = 'set path'
        for split in [1, 2, 3]:
            save_file = f"val_logits_{dataset}_IC2_seed_1_split{split}.pt"
            val_logits = torch.load(os.path.join(save_path, save_file))
            target = val_logits['true_labels']
            total_length += len(target)
            print(f"Split {split}: {len(target)}")

        with torch.cuda.device(0):
            num_segments = 8
            num_crops = 1
            model = models.resnet18()
            setattr(model, 'conv1', torch.nn.Conv2d(9, 64,
                                        kernel_size=(7, 7),
                                        stride=(2, 2),
                                        padding=(3, 3),
                                        bias=False))

            setattr(model, 'fc', torch.nn.Linear(512, num_classes))

            # Calculate FLOPs and count parameters
            macs, params = get_model_complexity_info(
                model, 
                (9,224,224),
                as_strings=False,
                print_per_layer_stat=False)       

            print(macs)   

            # gmac = num_segments * num_crops * macs * total_length / (1024 ** 3)
            gmac = 15.4 * total_length # from paper
            methods.append('MIMO')
            gmacs.append(gmac)
            all_datasets.append(dataset)

        with torch.cuda.device(0):
            num_segments = 12
            num_crops = 15
            net = MFNET_3D(num_classes=num_classes, pretrained=False)
            macs, params = get_model_complexity_info(
                net, 
                (3, num_segments, 224, 224), 
                as_strings=False,
                print_per_layer_stat=False, 
                verbose=False)
            
            '''
            Claim from paper 'Our method samples 15
            clips from each video, then our total amount of GFLOPS
            is: 8.53 x 15 = 128.'
            Remark we find each forward pass is roughly 7.9 GMACs 
            maybe they divided by 1000 instead of 1024 
            '''
            gmac = num_crops * macs * total_length / (1024 ** 3)
            methods.append('MFCD-NET')
            gmacs.append(gmac)
            all_datasets.append(dataset)
        
        with torch.cuda.device(0):
            '''
            Following C3D [27], all videos were resized to 128x171
            resolution. Then, we uniformly sample 16 frames from each
            video to feed the CV-C3D network. During testing phase, the
            action category is predicted by passing only a single center
            crop with size 112x112 through the network
            '''
            num_segments = 16
            net = CVC3DWrapper(num_segments, num_classes)
            macs, params = get_model_complexity_info(
                net, 
                (num_segments, 8, 112, 112), 
                as_strings=False,
                print_per_layer_stat=False, 
                verbose=False)

            gmac = macs * total_length / (1024 ** 3)
            methods.append('CV-C3D')
            gmacs.append(gmac)
            all_datasets.append(dataset)

        with torch.cuda.device(0):
            num_segments = 8
            net = TeamNetWrapper(num_segments, num_classes)
            macs, params = get_model_complexity_info(
                net, 
                (num_segments, 8, 224, 224), 
                as_strings=False,
                print_per_layer_stat=False, 
                verbose=False)

            gmac = macs * total_length / (1024 ** 3)
            methods.append('TEAM-NET')
            gmacs.append(gmac)
            all_datasets.append(dataset)

        with torch.cuda.device(0):
            num_i_segments = 3
            num_p_segments = 11
            net = CoviarWrapper(
                num_i_segments=num_i_segments,
                num_p_segments=num_p_segments,
                i_base='resnet50', 
                num_classes=num_classes)

            macs, params = get_model_complexity_info(
                net, 
                (11, 8, 224, 224), 
                as_strings=False,
                print_per_layer_stat=False, 
                verbose=False)
            gmac = macs * total_length / (1024 ** 3)
            methods.append('Coviar')
            gmacs.append(gmac)
            all_datasets.append(dataset)

        with torch.cuda.device(0):
            num_crops = 5 * 2
            net = CoviarWrapper(i_base='resnet18', num_classes=num_classes)
            macs, params = get_model_complexity_info(
                net, 
                (11, 8, 224, 224), 
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False)

            gmac = num_crops * macs * total_length / (1024 ** 3)
            methods.append('Wu et al.')
            gmacs.append(gmac)
            all_datasets.append(dataset)

        with torch.cuda.device(0):
            '''
            As in paper:
            For I, MV, and R, we follow the exactly same set-
            ting as in CoViAR [55]: 25 frames are uniformly sampled
            for each video; each sampled frame has 5 crops augmented
            with flipping; ; all 250 (25x2x5) score predictions are av-
            eraged to obtain one video-level prediction.
            '''
            num_i_segments = 3
            num_p_segments = 11
            num_crops = 5 * 2
            net = DMCWrapper(
                i_base='resnet152', 
                num_i_segments = 3,
                num_p_segments = 11,
                num_classes=num_classes)

            macs, params = get_model_complexity_info(
                net, 
                (11, 8, 224, 224), 
                as_strings=False,
                print_per_layer_stat=False, 
                verbose=False)

            gmac = (macs * num_crops/ (1024 ** 3))
            gmac = total_length * gmac
            methods.append('DMC Net')
            gmacs.append(gmac)
            all_datasets.append(dataset)

df = pd.DataFrame(data={'Dataset': all_datasets, 'GMACs': gmacs, 'Method': methods})

pd.options.display.float_format = '{:,.2f}'.format

# Note mean of 1 value is the value we use mean for the pretty print with grouping
result_df = df.groupby(["Dataset", "Method"]).mean()
result_df.to_csv('./mac_results.csv')
print(result_df)