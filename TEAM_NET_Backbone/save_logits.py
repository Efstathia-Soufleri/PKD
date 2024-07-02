import time
import numpy as np
import os
import torch
import torch.nn.parallel
import torchvision
import argparse
from dataset_team.dataset_fuse import CoviarDataSet
from models import team_net_ic, transforms
from tqdm import tqdm

SAVE_FREQ = 40
PRINT_FREQ = 20

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    parser = argparse.ArgumentParser(description="Standard video-level testing")
    parser.add_argument('--data-name', type=str, default='ucf101')
    parser.add_argument('--no-accumulation', action='store_true', help='disable accumulation of motion vectors and residuals.')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--train-list', type=str)
    parser.add_argument('--test-list', type=str)
    parser.add_argument('--model-root', type=str, default='./checkpoints/')
    parser.add_argument('--save-root', type=str, default='./logits/')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_clips', type=int, default=10)
    parser.add_argument('--seed', type=int, default=736)
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of workers for data loader.')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size.')
    parser.add_argument('--split', type=str, default='split1', help='split.')
    parser.add_argument('--is_shift', action='store_true', help='enable TSM')
    parser.add_argument('--ic', type=str, default='IC2', help='IC To Optimize')

    args = parser.parse_args()

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51

    model = team_net_ic.TEAM_Net(
        num_class, 
        args.num_segments,
        base_model=args.arch, 
        shift_div=8,
        is_shift=args.is_shift)
    
    model = torch.nn.DataParallel(model.cuda(), device_ids=[0,1,2])

    model_name = f'resnet50_8f_checkpoint_KD_progressive_{args.ic}_seed_{args.seed}_split_{args.split}.pth.tar'
    checkpoint = torch.load(os.path.join(args.model_root, args.data_name.lower(), model_name))
    print("Model {}, epoch {} best prec@1: {}".format(args.ic, checkpoint['epoch'], checkpoint['best_prec1']))
    model.load_state_dict(checkpoint['state_dict'])

    train_set = CoviarDataSet(
        args.data_root,
        args.data_name,
        video_list=args.train_list,
        num_segments=args.num_segments,
        frame_transform=torchvision.transforms.Compose([
            transforms.GroupScale(int(model.module.I_model.scale_size)),
            transforms.GroupCenterCrop(model.module.I_model.crop_size),
        ]),
        motion_transform=torchvision.transforms.Compose([
            transforms.GroupScale(int(model.module.MV_model.scale_size)),
            transforms.GroupCenterCrop(model.module.MV_model.crop_size),
        ]),
        residual_transform=torchvision.transforms.Compose([
            transforms.GroupScale(int(model.module.R_model.scale_size)),
            transforms.GroupCenterCrop(model.module.R_model.crop_size),
        ]),
        is_train=True,
        accumulate=(not args.no_accumulation)
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=False,
        num_workers=args.workers)
    
    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.num_segments,
            frame_transform=torchvision.transforms.Compose([
                transforms.GroupScale(int(model.module.I_model.scale_size)),
                transforms.GroupCenterCrop(model.module.I_model.crop_size),
                ]),
            motion_transform=torchvision.transforms.Compose([
                transforms.GroupScale(int(model.module.MV_model.scale_size)),
                transforms.GroupCenterCrop(model.module.MV_model.crop_size),
                ]),
            residual_transform=torchvision.transforms.Compose([
                transforms.GroupScale(int(model.module.R_model.scale_size)),
                transforms.GroupCenterCrop(model.module.R_model.crop_size),
                ]),
            is_train=False,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)  
    
    model.eval()

    train_logits = {
        'ic1': None, 
        'ic2': None,
        'ic3': None,
        'out': None,
        'r': None,
        'mv': None,
        'i': None,
        'true_labels': None,
    }

    val_logits = {
        'ic1': None, 
        'ic2': None,
        'ic3': None,
        'out': None,
        'r': None,
        'mv': None,
        'i': None,
        'true_labels': None, 
    }

    def update_logits(logit_dict, key, tensor):
        if logit_dict[key] is None:
            logit_dict[key] = tensor.cpu()
        else:
            logit_dict[key] = torch.cat([logit_dict[key], tensor.cpu()], axis=0)

    print("Starting save")
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda(non_blocking=True)

            input_i, input_mv, input_r = input[0].cuda(non_blocking=True), \
                                        input[1].cuda(non_blocking=True), \
                                        input[2].cuda(non_blocking=True)


            out_IC1, out_IC2, out_IC3, out, teacher_out_i, teacher_out_mv, teacher_out_r = model(input_i, input_mv, input_r)
            
            update_logits(train_logits, 'ic1', out_IC1)
            update_logits(train_logits, 'ic2', out_IC2)
            update_logits(train_logits, 'ic3', out_IC3)
            update_logits(train_logits, 'out', out)
            update_logits(train_logits, 'r', teacher_out_r)
            update_logits(train_logits, 'mv', teacher_out_mv)
            update_logits(train_logits, 'i', teacher_out_i)
            update_logits(train_logits, 'true_labels', target)

            if i % PRINT_FREQ == 0:
                print(f"Train: {i} / {len(train_loader)}")
    
    save_path = args.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_filename = os.path.join(save_path, f"train_logits_{args.data_name}_{args.ic}_seed_{args.seed}_{args.split}.pt")
    torch.save(train_logits, full_filename)

    print("Starting validation save")
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            input_i, input_mv, input_r = input[0].cuda(non_blocking=True), \
                                        input[1].cuda(non_blocking=True), \
                                        input[2].cuda(non_blocking=True)


            out_IC1, out_IC2, out_IC3, out, teacher_out_i, teacher_out_mv, teacher_out_r = model(input_i, input_mv, input_r)
            
            update_logits(val_logits, 'ic1', out_IC1)
            update_logits(val_logits, 'ic2', out_IC2)
            update_logits(val_logits, 'ic3', out_IC3)
            update_logits(val_logits, 'out', out)
            update_logits(val_logits, 'r', teacher_out_r)
            update_logits(val_logits, 'mv', teacher_out_mv)
            update_logits(val_logits, 'i', teacher_out_i)
            update_logits(val_logits, 'true_labels', target)

            if i % PRINT_FREQ == 0:
                print(f"Val: {i} / {len(val_loader)}")
    
    save_path = args.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_filename = os.path.join(save_path, f"val_logits_{args.data_name}_{args.ic}_seed_{args.seed}_{args.split}.pt")
    torch.save(val_logits, full_filename)   


if __name__ == '__main__':
    main()
