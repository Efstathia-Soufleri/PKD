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


def main():

    net = Model(num_class, args.test_segments, args.representation, args.arch)
    
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

    # if args.gpus is not None:
    #     devices = [args.gpus[i] for i in range(args.workers)]
    # else:
    #     devices = list(range(args.workers))

    # net = torch.nn.DataParallel(net, device_ids = [0]).cuda()
    # net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    total_num = len(data_loader.dataset)

    output = []

    def forward_video(data):
        out = net(data)

        out = out.view((-1, args.test_segments * args.test_crops) + out.size()[1:])
        out = torch.mean(out, dim=1)

        return out.data.cpu()


    proc_start_time = time.time()


    for i, (data, label) in enumerate(data_loader):
        
        out = forward_video(data)
        # output_final.append((video_scores_final, label[0]))
        output.append((out, label[0]))

        cnt_time = time.time() - proc_start_time
        if (i + 1) % 100 == 0:
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                            total_num,
                                                                            float(cnt_time) / (i+1)))

    video_pred_output1 = [np.argmax(x[0]) for x in output]

    video_labels = [x[1] for x in output]

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
    
    print('IC4 Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred_output1) == np.array(video_labels))) / len(video_pred_output1) * 100.0,
        len(video_pred_output1)))
    

    torch.save(output, folder_path +'/IC4_'+str(args.representation)+'_frames_'+str(args.test_segments)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')


if __name__ == '__main__':
    main()
