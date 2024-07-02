"""Run training."""

import shutil
import time
import numpy as np
import datetime
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
from dataset_fuse import CoviarDataSet
from model_unrolled import Model_unrolled
from train_options import parser
from transforms import GroupCenterCrop
from transforms import GroupScale
from model_unrolled_all_IC import model_unrolled_all_IC
from model import Model
import random 
import torch.nn.functional as F
from model_unrolled_IC_resnet152 import model_unrolled_IC_resnet152
from model_unrolled_resnet152 import model_unrolled_resnet152

SAVE_FREQ = 40
PRINT_FREQ = 20
best_prec1 = 0


global activation

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
    global args
    global best_prec1_IC1
    global best_prec1_IC2
    global best_prec1_IC3
    global best_prec1_IC4

    args = parser.parse_args()

    random_seed = args.random_seed 
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    else:
        raise ValueError('Unknown dataset '+ args.data_name)
    
    global activation
    activation = {}

    gpus = [0, 1, 2]

    # model with ICs
    if args.representation == 'mv' or args.representation == 'residual':
        model_IC = IC_Model(num_class, args.num_segments, args.representation, args.arch)
        checkpoint = torch.load('kinetics_pretrained_'+str(args.representation)+'_'+str(args.data_name)+'_'+str(args.representation)+'_rolled_model_resnet18_seed_4711_'+str(args.split)+'_v2_'+str(args.representation)+'_model_best.pth.tar')
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        model_IC.net.load_state_dict(base_dict)
    elif args.representation == 'iframe':
        model_IC = IC_Model_Resnet50(num_class, args.num_segments, args.representation, args.arch)
        checkpoint = torch.load('./kinetics_pretrained_'+str(args.data_name)+'_iframe_rolled_model_resnet50_seed_4711_'+str(args.split)+'_v3_iframe_model_best.pth.tar')
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        model_IC.net.load_state_dict(base_dict)
    
    # teacher model mv
    teacher_model_mv = Model(num_class, args.num_segments, 'mv', base_model='resnet18')
    teacher_model_mv = torch.nn.DataParallel(teacher_model_mv, device_ids = gpus).cuda()
    teacher_model_mv.load_state_dict(torch.load('./kinetics_pretrained_mv_'+str(args.data_name)+'_mv_rolled_model_resnet18_seed_4711_'+str(args.split)+'_v2_mv_model_best.pth.tar')['state_dict'])
    teacher_model_mv.eval()

    # teacher model residual
    teacher_model_residual = Model(num_class, args.num_segments, 'residual', base_model='resnet18')
    teacher_model_residual = torch.nn.DataParallel(teacher_model_residual, device_ids = gpus).cuda()
    teacher_model_residual.load_state_dict(torch.load('./kinetics_pretrained_residual_'+str(args.data_name)+'_residual_rolled_model_resnet18_seed_4711_'+str(args.split)+'_v2_residual_model_best.pth.tar')['state_dict'])
    teacher_model_residual.eval()

    # teacher model iframe
    teacher_model_iframe = Model(num_class, args.num_segments, 'iframe', base_model='resnet50')
    teacher_model_iframe = torch.nn.DataParallel(teacher_model_iframe, device_ids = gpus).cuda()
    teacher_model_iframe.load_state_dict(torch.load('./kinetics_pretrained_'+str(args.data_name)+'_iframe_rolled_model_resnet50_seed_4711_'+str(args.split)+'_v3_iframe_model_best.pth.tar')['state_dict'])
    teacher_model_iframe.eval()

    for gpu in gpus:
        activation[f"cuda:{gpu}"] = {}

    model_IC = torch.nn.DataParallel(model_IC, device_ids = gpus).cuda()

    # Freeze all the parameters of the model
    for param in model_IC.module.parameters():
        param.requires_grad = False

    start = time.time()

    train_loader = torch.utils.data.DataLoader(
                # video_dataset.
                CoviarDataSet(
                    args.data_root,
                    args.data_name,
                    video_list=args.train_list,
                    num_segments=args.num_segments,
                    frame_transform=torchvision.transforms.Compose(
                    [GroupMultiScaleCrop(224* 256 // 224, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_mv=('iframe' == 'mv'))
                    ]),
                    motion_transform=torchvision.transforms.Compose(
                    [GroupMultiScaleCrop(224* 256 // 224, [1, .875, .75]),
                    GroupRandomHorizontalFlip(is_mv=('mv' == 'mv'))
                    ]),
                    residual_transform=torchvision.transforms.Compose(
                    [GroupMultiScaleCrop(224* 256 // 224, [1, .875, .75]),
                    GroupRandomHorizontalFlip(is_mv=('residual' == 'mv'))
                    ]),
                    is_train=True,
                    accumulate=(not args.no_accumulation),
                    # dense_sample=False
                    ),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

    end = time.time()
    print("Time elapsed for loading training data: {:.2f} mins".format((end-start)/60))
    
    start = time.time()

    val_loader = torch.utils.data.DataLoader(
            # video_dataset.
            CoviarDataSet(
                args.data_root,
                args.data_name,
                video_list=args.test_list,
                num_segments=args.num_segments,
                frame_transform=torchvision.transforms.Compose([
                    GroupScale(int(224* 256 // 224)),
                    GroupCenterCrop(224),
                    ]),
                motion_transform=torchvision.transforms.Compose([
                    GroupScale(int(224* 256 // 224)),
                    GroupCenterCrop(224),
                    ]),
                residual_transform=torchvision.transforms.Compose([
                    GroupScale(int(224* 256 // 224)),
                    GroupCenterCrop(224),
                    ]),
                is_train=False,
                accumulate=(not args.no_accumulation),
                # dense_sample=False
                ),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    end = time.time()
    print("Time elapsed for loading testing data: {:.2f} mins".format((end-start)/60))

    # model_IC = torch.nn.DataParallel(model_IC, device_ids = [0, 1, 2, 3]).cuda()

    params_dict = dict(model_IC.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        if ('module.conv1.' in key
                or 'module.bn1.' in key
                or 'data_bn' in key) and args.representation in ['mv', 'residual']:
            lr_mult = 0.1
            # print(key)
        elif ('module.fc.' in key or 'classifier.' in key):
            lr_mult = 1.0
            # print(key)
        else:
            lr_mult = 0.01

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]

    # IC1
    for param in model_IC.module.IC1_features.parameters():
        param.requires_grad = True

    for param in model_IC.module.IC1_classifier.parameters():
        param.requires_grad = True

    # IC2
    for param in model_IC.module.IC2_features.parameters():
        param.requires_grad = True

    for param in model_IC.module.IC2_classifier.parameters():
        param.requires_grad = True

    # IC3
    for param in model_IC.module.IC3_features.parameters():
        param.requires_grad = True

    for param in model_IC.module.IC3_classifier.parameters():
        param.requires_grad = True


    params1 = torch.nn.ParameterList([  model_IC.module.IC1_features[0].weight, 
                                        model_IC.module.IC1_features[1].weight, 
                                        model_IC.module.IC1_features[1].bias, 
                                        model_IC.module.IC1_classifier.weight])
    
    params2 = torch.nn.ParameterList([  model_IC.module.IC2_features[0].weight, 
                                        model_IC.module.IC2_features[1].weight, 
                                        model_IC.module.IC2_features[1].bias, 
                                        model_IC.module.IC2_classifier.weight])
    
    params3 = torch.nn.ParameterList([  model_IC.module.IC3_features[0].weight, 
                                        model_IC.module.IC3_features[1].weight, 
                                        model_IC.module.IC3_features[1].bias, 
                                        model_IC.module.IC3_classifier.weight])
    

    optimizer_IC1 = torch.optim.Adam([{'params': params1, 'lr': args.lr, 
                                       'lr_mult': lr_mult, 
                                       'decay_mult': decay_mult}], weight_decay=args.weight_decay, eps=0.001)
    
    optimizer_IC2 = torch.optim.Adam([{'params': params2, 'lr': args.lr, 
                                       'lr_mult': lr_mult, 
                                       'decay_mult': decay_mult}], weight_decay=args.weight_decay, eps=0.001)
    
    optimizer_IC3 = torch.optim.Adam([{'params': params3, 'lr': args.lr, 
                                       'lr_mult': lr_mult, 
                                       'decay_mult': decay_mult}], weight_decay=args.weight_decay, eps=0.001)
    
    criterion = torch.nn.CrossEntropyLoss().cuda()

    best_prec1_IC1 = 0
    best_prec1_IC2 = 0
    best_prec1_IC3 = 0

    for epoch in range(args.epochs):
        cur_lr1 = adjust_learning_rate(optimizer_IC1, epoch, args.lr_steps, args.lr_decay)
        cur_lr2 = adjust_learning_rate(optimizer_IC2, epoch, args.lr_steps, args.lr_decay)
        cur_lr3 = adjust_learning_rate(optimizer_IC3, epoch, args.lr_steps, args.lr_decay)

        start = time.time()

        train(train_loader, teacher_model_residual, teacher_model_mv, model_IC, criterion, optimizer_IC1, optimizer_IC2,
              optimizer_IC3, epoch, cur_lr1, cur_lr2, cur_lr3)
        
        end = time.time()
        print("Time elapsed for training one epoch: {:.2f} ".format((end-start)/60))

        prec1_IC1, prec1_IC2, prec1_IC3 = validate(val_loader, model_IC, criterion)

        is_best_IC1 = prec1_IC1 > best_prec1_IC1
        best_prec1_IC1 = max(prec1_IC1, best_prec1_IC1)
        if is_best_IC1:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_IC.state_dict(),
                    'best_prec1': best_prec1_IC1,
                },
                is_best_IC1,
                filename='checkpoint_KD_progressive_IC1_seed_'+str(args.random_seed)+'.pth.tar')
        
        is_best_IC2 = prec1_IC2 > best_prec1_IC2
        best_prec1_IC2 = max(prec1_IC2, best_prec1_IC2)
        if is_best_IC2:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_IC.state_dict(),
                    'best_prec1': best_prec1_IC2,
                },
                is_best_IC2,
                filename='checkpoint_KD_progressive_IC2_seed_'+str(args.random_seed)+'.pth.tar')
            
        is_best_IC3 = prec1_IC3 > best_prec1_IC3
        best_prec1_IC3 = max(prec1_IC3, best_prec1_IC3)
        if is_best_IC3:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_IC.state_dict(),
                    'best_prec1': best_prec1_IC3,
                },
                is_best_IC3,
                filename='checkpoint_KD_progressive_IC3_seed_'+str(args.random_seed)+'.pth.tar')
            
        print('Best Inference Precision - IC1 : ', best_prec1_IC1)
        print('Best Inference Precision - IC2 : ', best_prec1_IC2)
        print('Best Inference Precision - IC3 : ', best_prec1_IC3)

def cross_entropy(prob1, prob2):
    return torch.mean(-torch.sum(prob2 * torch.log(prob1), 1))

def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

def train(train_loader, teacher_model_residual, teacher_model_mv, model_IC, criterion, 
          optimizer_IC1, optimizer_IC2, optimizer_IC3, epoch, cur_lr1, cur_lr2, cur_lr3):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_IC1 = AverageMeter()
    top1_IC1 = AverageMeter()
    top5_IC1 = AverageMeter()

    losses_IC2 = AverageMeter()
    top1_IC2 = AverageMeter()
    top5_IC2 = AverageMeter()

    losses_IC3 = AverageMeter()
    top1_IC3 = AverageMeter()
    top5_IC3 = AverageMeter()

    model_IC.train()
    teacher_model_residual.eval()
    teacher_model_mv.eval()

    softmax1 = torch.nn.Softmax(dim=1)
    softmax0 = torch.nn.Softmax(dim=0)

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        target = target.cuda()
        input_i, input_mv, input_r = input[0].cuda(non_blocking=True), \
                                    input[1].cuda(non_blocking=True), \
                                    input[2].cuda(non_blocking=True)
        
        target_var = torch.autograd.Variable(target)

        if args.representation == 'mv':
            out_IC1, out_IC2, out_IC3, out_IC4 = model_IC(input_mv)
        elif args.representation == 'residual':
            out_IC1, out_IC2, out_IC3, out_IC4 = model_IC(input_r)
        elif args.representation == 'iframe':
            out_IC1, out_IC2, out_IC3, out_IC4 = model_IC(input_i)

        out_IC1 = out_IC1.view((-1, args.num_segments) + out_IC1.size()[1:])
        out_IC1 = torch.mean(out_IC1, dim=1)

        out_IC2 = out_IC2.view((-1, args.num_segments) + out_IC2.size()[1:])
        out_IC2 = torch.mean(out_IC2, dim=1)

        out_IC3 = out_IC3.view((-1, args.num_segments) + out_IC3.size()[1:])
        out_IC3 = torch.mean(out_IC3, dim=1)

        out_IC4 = out_IC4.view((-1, args.num_segments) + out_IC4.size()[1:])
        out_IC4 = torch.mean(out_IC4, dim=1)

        teacher_out_residual = teacher_model_residual(input_r)
        teacher_out_residual = teacher_out_residual.view((-1, args.num_segments) + teacher_out_residual.size()[1:])
        teacher_out_residual = torch.mean(teacher_out_residual, dim=1)

        teacher_out_mv = teacher_model_mv(input_mv)
        teacher_out_mv = teacher_out_mv.view((-1, args.num_segments) + teacher_out_mv.size()[1:])
        teacher_out_mv = torch.mean(teacher_out_mv, dim=1)

        if epoch < 50:
            loss1 = cross_entropy(softmax1(out_IC1/args.T), softmax1((teacher_out_mv/args.T).detach()))
            loss2 = cross_entropy(softmax1(out_IC2/args.T), softmax1((teacher_out_mv/args.T).detach()))
            loss3 = cross_entropy(softmax1(out_IC3/args.T), softmax1((teacher_out_mv/args.T).detach()))
        elif epoch >= 50 and epoch < 100:
            loss1 = cross_entropy(softmax1(out_IC1/args.T), softmax1((teacher_out_residual/args.T).detach()))
            loss2 = cross_entropy(softmax1(out_IC2/args.T), softmax1((teacher_out_residual/args.T).detach()))
            loss3 = cross_entropy(softmax1(out_IC3/args.T), softmax1((teacher_out_residual/args.T).detach()))
        else:
            loss1 = cross_entropy(softmax1(out_IC1/args.T), softmax1((out_IC4/args.T).detach()))
            loss2 = cross_entropy(softmax1(out_IC2/args.T), softmax1((out_IC4/args.T).detach()))
            loss3 = cross_entropy(softmax1(out_IC3/args.T), softmax1((out_IC4/args.T).detach()))

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

        optimizer_IC1.zero_grad()
        loss1.backward()
        optimizer_IC1.step()

        optimizer_IC2.zero_grad()
        loss2.backward()
        optimizer_IC2.step()

        optimizer_IC3.zero_grad()
        loss3.backward()
        optimizer_IC3.step()

    now = datetime.datetime.now()
    print(now.strftime("[%Y-%m-%d %H:%M:%S]"))

    print((' IC1: Epoch: [{0}/{1}], lr: {lr:.7f}\t'
            'Loss {loss.val:.4f}\t'
            'Prec@1 {top1.val:.3f}\t'
            'Prec@5 {top5.val:.3f}'.format(
                epoch, args.epochs,
                loss=losses_IC1,
                top1=top1_IC1,
                top5=top5_IC1,
                lr=cur_lr1)))
            
    print(('IC2: Epoch: [{0}/{1}], lr: {lr:.7f}\t'
            'Loss {loss.val:.4f}\t'
            'Prec@1 {top1.val:.3f}\t'
            'Prec@5 {top5.val:.3f}'.format(
                epoch, args.epochs,
                loss=losses_IC2,
                top1=top1_IC2,
                top5=top5_IC2,
                lr=cur_lr2)))
            
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
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_i, input_mv, input_r = input[0].cuda(non_blocking=True), \
                                        input[1].cuda(non_blocking=True), \
                                        input[2].cuda(non_blocking=True)
            
            if args.representation == 'mv':
                out_IC1, out_IC2, out_IC3, out_IC4 = model(input_mv)
            elif args.representation == 'residual':
                out_IC1, out_IC2, out_IC3, out_IC4 = model(input_r)
            elif args.representation == 'iframe':
                out_IC1, out_IC2, out_IC3, out_IC4 = model(input_i)

            out_IC1 = out_IC1.view((-1, args.num_segments) + out_IC1.size()[1:])
            out_IC1 = torch.mean(out_IC1, dim=1)

            out_IC2 = out_IC2.view((-1, args.num_segments) + out_IC2.size()[1:])
            out_IC2 = torch.mean(out_IC2, dim=1)

            out_IC3 = out_IC3.view((-1, args.num_segments) + out_IC3.size()[1:])
            out_IC3 = torch.mean(out_IC3, dim=1)

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


    print(('IC1: Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1_IC1, top5=top5_IC1, loss=losses_IC1)))
    
    print(('IC2: Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1_IC2, top5=top5_IC2, loss=losses_IC2)))
    
    print(('IC3: Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1_IC3, top5=top5_IC3, loss=losses_IC3)))

    return top1_IC1.avg, top1_IC2.avg, top1_IC3.avg

def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.model_prefix, args.representation.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.model_prefix, args.representation.lower(), 'model_best_multiple_gpus.pth.tar'))
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

def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
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
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
