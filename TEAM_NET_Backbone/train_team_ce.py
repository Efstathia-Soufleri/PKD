import shutil
import time
import numpy as np
import os
import random
import torch
import torch.nn.parallel
import torchvision
import datetime
from dataset_team import dataset_fuse as video_dataset
from models import team_net_ic, transforms
from train_options import parser
from tqdm import trange

SAVE_FREQ = 40
PRINT_FREQ = 20
args = parser.parse_args()

local_args = dict()
local_args['recover_from_checkpoint'] = None

# Since we start we pretrained model make sure to place they are placed
# according to this path
local_args['pretrained'] = os.path.join(
    args.checkpoint_dir,
    args.data_name,
    f"resnet50_8f_checkpoint_TEAM_NET_{args.split}_.pth.tar")

local_args['logdir'] = os.path.join(
    'log',
    args.data_name,
    f"{args.arch}_{args.num_segments}")

def main():
    seed = 1

    # if you are using multi-GPU.
    torch.cuda.manual_seed_all(seed)  

    global best_prec1
    global best_prec5
    best_prec1 = 0
    best_prec5 = 0

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    elif args.data_name == 'kinetic400':
        num_class = 400

    model = team_net_ic.TEAM_Net(
        num_class, 
        args.num_segments,
        base_model=args.arch, 
        is_shift=args.is_shift, 
        shift_div=8, 
        dropout=args.dropout)

    random_seed = args.random_seed 
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    if args.is_train:
        # continue training from the last checkpoint
        if local_args['recover_from_checkpoint'] is not None:
            print('loading last trained checkpoint')
            model_checkpoint = torch.load(local_args['recover_from_checkpoint'], map_location='cpu')
            model_dict = model.state_dict()
            for k, v in model_checkpoint['state_dict'].items():
                if 'module' in k:
                    k = k.replace('module.', '')
                    model_dict.update({k: v})
                else:
                    model_dict.update({k: v})
            model.load_state_dict(model_dict)
            print("model epoch {} best prec@1: {}".format(model_checkpoint['epoch'], model_checkpoint['best_prec1']))

        # load pretrained model
        if local_args['pretrained'] is not None:
            checkpoint = torch.load(local_args['pretrained'])
            model_dict = model.state_dict()
            for k, v in model_dict.items():
                if '_classifier' in k:
                    continue
                else:
                    model_dict.update({k: checkpoint['state_dict'][k]})
            model.load_state_dict(model_dict)
        
        policies = model.get_optim_policies()
        lr_mult = -1
        decay_mult = -1
        for param_group in policies:
            lr_mult = param_group['lr_mult']
            decay_mult = param_group['decay_mult']
            param_group['lr'] = args.lr * param_group['lr_mult']
            param_group['weight_decay'] = args.weight_decay * param_group['decay_mult']
        
        for param in model.parameters():
            param.requires_grad = False

        for param in model.IC1_classifier.parameters():
            param.requires_grad = True

        # IC2
        for param in model.IC2_classifier.parameters():
            param.requires_grad = True

        # IC3
        for param in model.IC3_classifier.parameters():
            param.requires_grad = True


        params3 = torch.nn.ParameterList([
            model.IC1_classifier.weight,
            model.IC1_classifier.bias,
            model.IC2_classifier.weight,
            model.IC2_classifier.bias,
            model.IC3_classifier.weight,
            model.IC3_classifier.bias])
        
        optimizer_IC3 = torch.optim.Adam(
            [
                {
                    'params': params3, 
                    'lr': args.lr, 
                    'lr_mult': lr_mult, 
                    'decay_mult': decay_mult
                }
            ], 
            weight_decay=args.weight_decay, 
            eps=0.001)
        
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizers = [optimizer_IC3] #[optimizer_IC1, optimizer_IC2, optimizer_IC3]


        train_loader = torch.utils.data.DataLoader(
                video_dataset.CoviarDataSet(
                    args.data_root,
                    args.data_name,
                    video_list=args.train_list,
                    num_segments=args.num_segments,
                    frame_transform=model.I_model.get_augmentation(args.data_name),
                    motion_transform=model.MV_model.get_augmentation(args.data_name),
                    residual_transform=model.R_model.get_augmentation(args.data_name),
                    is_train=True,
                    accumulate=(not args.no_accumulation)
                    ),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            video_dataset.CoviarDataSet(
                args.data_root,
                args.data_name,
                video_list=args.test_list,
                num_segments=args.num_segments,
                frame_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.I_model.scale_size)),
                    transforms.GroupCenterCrop(model.I_model.crop_size),
                    ]),
                motion_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.MV_model.scale_size)),
                    transforms.GroupCenterCrop(model.MV_model.crop_size),
                    ]),
                residual_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.R_model.scale_size)),
                    transforms.GroupCenterCrop(model.R_model.crop_size),
                    ]),
                is_train=False,
                accumulate=(not args.no_accumulation),
                ),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        # create log file for tensorboard
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        logdir = os.path.join(local_args['logdir'], cur_time)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        writer = None
        best_prec1_IC1 = 0
        best_prec1_IC2 = 0
        best_prec1_IC3 = 0

        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        for epoch in trange(args.epochs):
            cur_lr = adjust_learning_rate(optimizers, epoch, args.lr_steps, args.lr_decay)
            cur_lr = 0

            train(train_loader, model, criterion, optimizers, epoch, cur_lr, writer)

            if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
                prec1_IC1, prec1_IC2, prec1_IC3 = validate(val_loader, model, criterion)

                is_best_IC1 = prec1_IC1 > best_prec1_IC1
                best_prec1_IC1 = max(prec1_IC1, best_prec1_IC1)
                if is_best_IC1:
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_prec1_IC1,
                        },
                        is_best_IC1,
                        filename='checkpoint_CE_IC1_seed_'+str(args.random_seed)+'_split_'+str(args.split)+'.pth.tar')
                
                is_best_IC2 = prec1_IC2 > best_prec1_IC2
                best_prec1_IC2 = max(prec1_IC2, best_prec1_IC2)
                if is_best_IC2:
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_prec1_IC2,
                        },
                        is_best_IC2,
                        filename='checkpoint_CE_IC2_seed_'+str(args.random_seed)+'_split_'+str(args.split)+'.pth.tar')
                    
                is_best_IC3 = prec1_IC3 > best_prec1_IC3
                best_prec1_IC3 = max(prec1_IC3, best_prec1_IC3)
                if is_best_IC3:
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_prec1_IC3,
                        },
                        is_best_IC3,
                        filename='checkpoint_CE_IC3_seed_'+str(args.random_seed)+'_split_'+str(args.split)+'.pth.tar')
                    
                print('Best Inference Precision - IC1 : ', best_prec1_IC1)
                print('Best Inference Precision - IC2 : ', best_prec1_IC2)
                print('Best Inference Precision - IC3 : ', best_prec1_IC3)
    else:      
        criterion = torch.nn.CrossEntropyLoss().cuda()
        args.teacher_weight = f"./checkpoints/{args.data_name}/resnet50_8f_checkpoint_TEAM_NET_{args.split}_.pth.tar"
        checkpoint = torch.load(args.teacher_weight)
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        model.load_state_dict(checkpoint['state_dict'])
        val_loader = torch.utils.data.DataLoader(
            video_dataset.CoviarDataSet(
                args.data_root,
                args.data_name,
                video_list=args.test_list,
                num_segments=args.num_segments,
                frame_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.I_model.scale_size)),
                    transforms.GroupCenterCrop(model.I_model.crop_size),
                    ]),
                motion_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.MV_model.scale_size)),
                    transforms.GroupCenterCrop(model.MV_model.crop_size),
                    ]),
                residual_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.R_model.scale_size)),
                    transforms.GroupCenterCrop(model.R_model.crop_size),
                    ]),
                is_train=False,
                accumulate=(not args.no_accumulation),
                ),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)  
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
        prec1 = validate(val_loader, model, criterion)

def cross_entropy(prob1, prob2):
    return torch.mean(-torch.sum(prob2 * torch.log(prob1), 1))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train(train_loader, model, criterion, optimizers, epoch, cur_lr, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_IC1 = AverageMeter()
    top1_IC1 = AverageMeter()
    top5_IC1 = AverageMeter()
    top1_acc_mv = AverageMeter()
    top5_acc_mv = AverageMeter()

    losses_IC2 = AverageMeter()
    top1_IC2 = AverageMeter()
    top5_IC2 = AverageMeter()

    losses_IC3 = AverageMeter()
    top1_IC3 = AverageMeter()
    top5_IC3 = AverageMeter()

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_IC3 = optimizers[0]

    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        input_i, input_mv, input_r = input[0].cuda(non_blocking=True), \
                                    input[1].cuda(non_blocking=True), \
                                    input[2].cuda(non_blocking=True)


        out_IC1, out_IC2, out_IC3, out_test, teacher_out_i, teacher_out_mv, teacher_out_residual = model(input_i, input_mv, input_r)


        loss1 = criterion(out_IC1, target)
        loss2 = criterion(out_IC2, target)
        loss3 = criterion(out_IC3, target)

        prec1, prec5 = accuracy(out_IC1.data, target, topk=(1, 5))
        losses_IC1.update(loss1.item(), input_i.size(0))
        top1_IC1.update(prec1.item(), input_i.size(0))
        top5_IC1.update(prec5.item(), input_i.size(0))

        prec1, prec5 = accuracy(out_IC2.data, target, topk=(1, 5))
        losses_IC2.update(loss2.item(), input_i.size(0))
        top1_IC2.update(prec1.item(), input_i.size(0))
        top5_IC2.update(prec5.item(), input_i.size(0))

        prec1, prec5 = accuracy(out_IC3.data, target, topk=(1, 5))
        losses_IC3.update(loss3.item(), input_i.size(0))
        top1_IC3.update(prec1.item(), input_i.size(0))
        top5_IC3.update(prec5.item(), input_i.size(0))

        optimizer_IC3.zero_grad()
        (loss1 + loss2 + loss3).backward()
        optimizer_IC3.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % PRINT_FREQ == 0:
            cur_lr3 = get_lr(optimizer_IC3)

            now = datetime.datetime.now()
            print(now.strftime("[%Y-%m-%d %H:%M:%S]"))
            print(('IC1: Epoch: [{0}/{1}], lr: {lr:.7f}\t'
                    'Loss {loss.val:.4f}\t'
                    'Prec@1 {top1.val:.3f}\t'
                    'Prec@5 {top5.val:.3f}'.format(
                        epoch, args.epochs,
                        loss=losses_IC1,
                        top1=top1_IC1,
                        top5=top5_IC1,
                        lr=cur_lr3)))
                    
            print(('IC2: Epoch: [{0}/{1}], lr: {lr:.7f}\t'
                    'Loss {loss.val:.4f}\t'
                    'Prec@1 {top1.val:.3f}\t'
                    'Prec@5 {top5.val:.3f}'.format(
                        epoch, args.epochs,
                        loss=losses_IC2,
                        top1=top1_IC2,
                        top5=top5_IC2,
                        lr=cur_lr3)))
                    
            print(('IC3: Epoch: [{0}/{1}], lr: {lr:.7f}\t'
                    'Loss {loss.val:.4f}\t'
                    'Prec@1 {top1.val:.3f}\t'
                    'Prec@5 {top5.val:.3f}'.format(
                        epoch, args.epochs,
                        loss=losses_IC3,
                        top1=top1_IC3,
                        top5=top5_IC3,
                        lr=cur_lr3)))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_IC1 = AverageMeter()
    top1_IC1 = AverageMeter()
    top5_IC1 = AverageMeter()

    losses_IC2 = AverageMeter()
    top1_IC2 = AverageMeter()
    top5_IC2 = AverageMeter()

    losses_IC3 = AverageMeter()
    top1_IC3 = AverageMeter()
    top5_IC3 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_i, input_mv, input_r = input[0].cuda(non_blocking=True), \
                                        input[1].cuda(non_blocking=True), \
                                        input[2].cuda(non_blocking=True)

            out_IC1, out_IC2, out_IC3, output, teacher_out_i, teacher_out_mv, teacher_out_residual = model(input_i, input_mv, input_r)

            loss1 = criterion(out_IC1, target)
            loss2 = criterion(out_IC2, target)
            loss3 = criterion(out_IC3, target)

            prec1, prec5 = accuracy(out_IC1.data, target, topk=(1, 5))
            losses_IC1.update(loss1.item(), input_i.size(0))
            top1_IC1.update(prec1.item(), input_i.size(0))
            top5_IC1.update(prec5.item(), input_i.size(0))

            prec1, prec5 = accuracy(out_IC2.data, target, topk=(1, 5))
            losses_IC2.update(loss2.item(), input_i.size(0))
            top1_IC2.update(prec1.item(), input_i.size(0))
            top5_IC2.update(prec5.item(), input_i.size(0))

            prec1, prec5 = accuracy(out_IC3.data, target, topk=(1, 5))
            losses_IC3.update(loss3.item(), input_i.size(0))
            top1_IC3.update(prec1.item(), input_i.size(0))
            top5_IC3.update(prec5.item(), input_i.size(0))

            loss = criterion(output, target) 

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input_i.size(0))
            top1.update(prec1.item(), input_i.size(0))
            top5.update(prec5.item(), input_i.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  \t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))
    
    print(('IC1: Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
        .format(top1=top1_IC1, top5=top5_IC1, loss=losses_IC1)))
    
    print(('IC2: Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1_IC2, top5=top5_IC2, loss=losses_IC2)))
    
    print(('IC3: Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1_IC3, top5=top5_IC3, loss=losses_IC3)))

    return top1_IC1.avg, top1_IC2.avg, top1_IC3.avg


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.arch, '{}f'.format(args.num_segments), filename))
    folder = os.path.join(args.checkpoint_dir, args.data_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder , filename)
    torch.save(state, filename)
    if is_best:
        best_name = f'{args.arch}_tsn_{args.num_segments}_TEAM_NET_{args.split}_seed_{args.random_seed}_CE_IC.pth.tar'
        best_name = os.path.join(folder, best_name)
        shutil.copyfile(filename, best_name)

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


def adjust_learning_rate(optimizers, epoch, lr_steps, lr_decay):
    for optimizer in optimizers:
        decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        wd = args.weight_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = wd * param_group['decay_mult']

    return lr


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


if __name__ == '__main__':
    main()