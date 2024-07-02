"""Run testing given a trained model."""

import argparse
import time
import os
import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision
from model import Model
import torch.nn as nn
from dataset import CoviarDataSet
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale
from model_unrolled import Model_unrolled

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--data-name', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('--representation', type=str, choices=['iframe', 'residual', 'mv'])
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--save-scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test-crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--learning_type', type=str)
parser.add_argument('--IC', type=int, default=1)
parser.add_argument('--split', type=str)

args = parser.parse_args()

if args.data_name == 'ucf101':
    num_class = 101
elif args.data_name == 'hmdb51':
    num_class = 51
else:
    raise ValueError('Unknown dataset '+args.data_name)

def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        global activation
        gpu = f'cuda:{output.get_device()}'
        activation[gpu][name] = output.detach().cpu()
    return hook

class IC_Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, arch) -> None:
        super().__init__()
        self.net = Model(num_class, num_segments, representation, base_model=arch)
        
        # IC1
        self.IC1_features = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64))
        self.IC_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC1_classifier = nn.Linear(64 * 4 *4, num_class)
        
        # IC2
        self.IC2_features = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128))
        self.IC_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC2_classifier = nn.Linear(128 * 4 *4, num_class)
        
        # IC3
        self.IC3_features = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256))
        self.IC_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC3_classifier = nn.Linear(256 * 4 *4, num_class)
 

        layer1 = self.net.base_model.layer1[-1]
        layer2 = self.net.base_model.layer2[-1]
        layer3 = self.net.base_model.layer3[-1]
        layer4 = self.net.base_model.layer4[-1]

        layer1.register_forward_hook(getActivation('layer1'))
        layer2.register_forward_hook(getActivation('layer2'))
        layer3.register_forward_hook(getActivation('layer3'))
        layer4.register_forward_hook(getActivation('layer4'))

    def forward(self, input_frame):
        out = self.net(input_frame)

        # print(f"gpu {self.IC1_features[0].weight.get_device()}")
        gpu = f"cuda:{self.IC1_features[0].weight.get_device()}"
        # print(f"activation shape {activation[gpu]['layer1'].shape}")
        out_IC1 = self.IC1_features(activation[gpu]['layer1'].to(gpu))
        out_IC1 = self.IC_adaptive_avg_pool(out_IC1)
        out_IC1 = torch.flatten(out_IC1, 1)
        out_IC1 = self.IC1_classifier(out_IC1)

        gpu = f"cuda:{self.IC2_features[0].weight.get_device()}"
        out_IC2 = self.IC2_features(activation[gpu]['layer2'].to(gpu))
        out_IC2 = self.IC_adaptive_avg_pool(out_IC2)
        out_IC2 = torch.flatten(out_IC2, 1)
        out_IC2 = self.IC2_classifier(out_IC2)

        gpu = f"cuda:{self.IC3_features[0].weight.get_device()}"
        out_IC3 = self.IC3_features(activation[gpu]['layer3'].to(gpu))
        out_IC3 = self.IC_adaptive_avg_pool(out_IC3)
        out_IC3 = torch.flatten(out_IC3, 1)
        out_IC3 = self.IC3_classifier(out_IC3)

        return out_IC1, out_IC2, out_IC3, out

class IC_Model_Resnet50(nn.Module):
    def __init__(self, num_class, num_segments, representation, arch) -> None:
        super().__init__()
        self.net = Model(num_class, num_segments, representation, base_model=arch)
        
        # IC1
        self.IC1_features = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64))
        self.IC_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC1_classifier = nn.Linear(64 * 4 *4, num_class)
        
        # IC2
        self.IC2_features = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256))
        self.IC_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC2_classifier = nn.Linear(256 * 4 *4, num_class)
        
        # IC3
        self.IC3_features = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024))
        self.IC_adaptive_avg_pool = nn.AdaptiveAvgPool2d(4)
        self.IC3_classifier = nn.Linear(1024 * 4 *4, num_class)
 

        layer1 = self.net.base_model.layer1[-1]
        layer2 = self.net.base_model.layer2[-1]
        layer3 = self.net.base_model.layer3[-1]
        layer4 = self.net.base_model.layer4[-1]

        layer1.register_forward_hook(getActivation('layer1'))
        layer2.register_forward_hook(getActivation('layer2'))
        layer3.register_forward_hook(getActivation('layer3'))
        layer4.register_forward_hook(getActivation('layer4'))

    def forward(self, input_frame):
        out = self.net(input_frame)

        # print(f"gpu {self.IC1_features[0].weight.get_device()}")
        gpu = f"cuda:{self.IC1_features[0].weight.get_device()}"
        # print(f"activation shape {activation[gpu]['layer1'].shape}")
        out_IC1 = self.IC1_features(activation[gpu]['layer1'].to(gpu))
        out_IC1 = self.IC_adaptive_avg_pool(out_IC1)
        out_IC1 = torch.flatten(out_IC1, 1)
        out_IC1 = self.IC1_classifier(out_IC1)

        gpu = f"cuda:{self.IC2_features[0].weight.get_device()}"
        out_IC2 = self.IC2_features(activation[gpu]['layer2'].to(gpu))
        out_IC2 = self.IC_adaptive_avg_pool(out_IC2)
        out_IC2 = torch.flatten(out_IC2, 1)
        out_IC2 = self.IC2_classifier(out_IC2)

        gpu = f"cuda:{self.IC3_features[0].weight.get_device()}"
        out_IC3 = self.IC3_features(activation[gpu]['layer3'].to(gpu))
        out_IC3 = self.IC_adaptive_avg_pool(out_IC3)
        out_IC3 = torch.flatten(out_IC3, 1)
        out_IC3 = self.IC3_classifier(out_IC3)

        return out_IC1, out_IC2, out_IC3, out
 
def main():
    if args.representation == 'iframe':
        net = IC_Model_Resnet50(num_class, args.test_segments, args.representation, args.arch)
    else:
        net = IC_Model(num_class, args.test_segments, args.representation, args.arch)
    
    # print(net)
 
    checkpoint = torch.load(args.weights)
    print("Loaded the model.")
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    data_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.test_segments,
            representation=args.representation,
            transform=torchvision.transforms.Compose([
                    GroupScale(int(224* 256 // 224)),
                    GroupCenterCrop(224),
                    ]),
            is_train=False,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    global activation
    activation = {}

    gpus = [0]

    for gpu in gpus:
        activation[f"cuda:{gpu}"] = {}

    net = torch.nn.DataParallel(net, device_ids = gpus).cuda()

    net.eval()

    total_num = len(data_loader.dataset)

    output1 = []
    output2 = []
    output3 = []
    output4 = []

    def forward_video(data):
        out1, out2, out3, out4 = net(data)

        out1 = out1.view((-1, args.test_segments * args.test_crops) + out1.size()[1:])
        out1 = torch.mean(out1, dim=1)

        out2 = out2.view((-1, args.test_segments * args.test_crops) + out2.size()[1:])
        out2 = torch.mean(out2, dim=1)

        out3 = out3.view((-1, args.test_segments * args.test_crops) + out3.size()[1:])
        out3 = torch.mean(out3, dim=1)

        out4 = out4.view((-1, args.test_segments * args.test_crops) + out4.size()[1:])
        out4 = torch.mean(out4, dim=1)

        return out1.data.cpu(), out2.data.cpu(), out3.data.cpu(), out4.data.cpu()


    proc_start_time = time.time()


    for i, (data, label) in enumerate(data_loader):

        out1, out2, out3, out4 = forward_video(data)
        output1.append((out1, label[0]))
        output2.append((out2, label[0]))
        output3.append((out3, label[0]))
        output4.append((out4, label[0]))

        cnt_time = time.time() - proc_start_time
        if (i + 1) % 100 == 0:
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                            total_num,
                                                                            float(cnt_time) / (i+1)))

    video_pred_output1 = [np.argmax(x[0]) for x in output1]
    video_pred_output2 = [np.argmax(x[0]) for x in output2]
    video_pred_output3 = [np.argmax(x[0]) for x in output3]
    video_pred_output4 = [np.argmax(x[0]) for x in output4]

    video_labels = [x[1] for x in output1]
    
    print('IC1 Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred_output1) == np.array(video_labels))) / len(video_pred_output1) * 100.0,
        len(video_pred_output1)))
    
    print('IC2 Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred_output2) == np.array(video_labels))) / len(video_pred_output2) * 100.0,
        len(video_pred_output2)))

    print('IC3 Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred_output3) == np.array(video_labels))) / len(video_pred_output3) * 100.0,
        len(video_pred_output3)))

    # Specify the folder path you want to check and create
    if 'train' in args.test_list:
        folder_path = './logits/'+str(args.data_name)+'_kinetics_pretrained_trainsets/'
    else:
        folder_path = './logits/'+str(args.data_name)+'_kinetics_pretrained/'
        
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


    # create the folders
    if args.IC == 1:
        torch.save(output1, folder_path +'/IC1_'+str(args.representation)+'_frames_'+str(args.test_segments)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    elif args.IC == 2:
        torch.save(output2, folder_path +'/IC2_'+str(args.representation)+'_frames_'+str(args.test_segments)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    elif args.IC == 3:
        torch.save(output3, folder_path +'/IC3_'+str(args.representation)+'_frames_'+str(args.test_segments)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

if __name__ == '__main__':
    main()
