import time
import os
import torch
import argparse

SAVE_FREQ = 40
PRINT_FREQ = 5

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

def validate(val_loader, model, criterion, beta2, beta3, beta4):
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

            out_IC1, out_IC2, out_IC3, output, teacher_out_i, teacher_out_mv, teacher_out_r = model(input_i, input_mv, input_r)

            out_IC2 = beta2[0] * out_IC1 + beta2[1] * out_IC2
            out_IC3 = beta3[0] * out_IC1 + beta3[1] * out_IC2 + beta3[2] * out_IC3
            output = beta4[0] * out_IC1 + beta4[1] * out_IC2 + beta4[2] * out_IC3  + beta4[3] * teacher_out_i + beta4[4] * teacher_out_mv + beta4[5] * teacher_out_r

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

def main():
    parser = argparse.ArgumentParser(description="Standard video-level testing")
    parser.add_argument('--data-name', type=str, default='ucf101')
    parser.add_argument('--no-accumulation', action='store_true', help='disable accumulation of motion vectors and residuals.')
    parser.add_argument('--data-root', type=str, default='./set-data-path')
    parser.add_argument('--train-list', type=str)
    parser.add_argument('--test-list', type=str)
    parser.add_argument('--model-root', type=str, default='./checkpoints/')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_clips', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N', help='number of workers for data loader.')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size.')
    parser.add_argument('--split', type=str, default='split1', help='split.')
    parser.add_argument('--is_shift', action='store_true', help='enable TSM')
    parser.add_argument('--ic', type=str, default='IC2', help='IC To Optimize')
    parser.add_argument('--save-root', type=str, default='./logits/')
    parser.add_argument('--save-wise-path', type=str, default='./data/')

    args = parser.parse_args()

    beta2 = torch.nn.Parameter(torch.Tensor([0, 1]))
    beta3 = torch.nn.Parameter(torch.Tensor([0, 0, 1]))
    beta4 = torch.nn.Parameter(torch.Tensor([0, 0, 0, 1, 1.2, 1.3]))
    params = torch.nn.ParameterList([beta2, beta3, beta4])
    optimizer = torch.optim.SGD(params, lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.01
    iterations = 500
    save_path = args.save_root
    save_file = f"train_logits_{args.data_name}_{args.ic}_seed_{args.seed}_{args.split}.pt"
    train_logits = torch.load(os.path.join(save_path, save_file))
    out_IC1 = train_logits['ic1']
    out_IC2 = train_logits['ic2']
    out_IC3 = train_logits['ic3']
    out = train_logits['out']
    target = train_logits['true_labels']
    teacher_out_i = train_logits['i']
    teacher_out_mv = train_logits['mv']
    teacher_out_r = train_logits['r']
    for epoch in range(iterations):
            
        optimizer.zero_grad()
        
        out2 = beta2[0] * out_IC1 + beta2[1] * out_IC2
        out3 = beta3[0] * out_IC1 + beta3[1] * out_IC2 + beta3[2] * out_IC3
        out4 = beta4[0] * out_IC1 + beta4[1] * out_IC2 + beta4[2] * out_IC3 + beta4[3] * teacher_out_i + beta4[4] * teacher_out_mv + beta4[5] * teacher_out_r

        loss2 = criterion(out2, target) + alpha * torch.norm(beta2, p=1)
        loss3 = criterion(out3, target) + alpha * torch.norm(beta3, p=1)
        loss4 = criterion(out4, target) + alpha * torch.norm(beta4, p=1)

        loss2.backward()
        loss3.backward()
        loss4.backward()

        optimizer.step()

    beta2.requires_grad = False
    beta3.requires_grad = False
    beta4.requires_grad = False

    beta2[beta2 < 0 ] = 0
    beta3[beta3 < 0 ] = 0
    beta4[beta4 < 0 ] = 0

    param_dict = {
        'beta2': beta2.data,
        'beta3': beta3.data,
        'beta4': beta4.data
    }

    print(f"Epoch {epoch}: \n{beta2.data}, \n{beta3.data}, \n{beta4.data}")

    save_path_for_params = os.path.join(
        args.save_wise_path, 
        f"wise_{args.data_name}_{args.ic}_seed_{args.seed}_{args.split}.pt")
    torch.save(param_dict, save_path_for_params)

    save_file = f"val_logits_{args.data_name}_{args.ic}_seed_{args.seed}_{args.split}.pt"
    val_logits = torch.load(os.path.join(save_path, save_file))
    out_IC1 = val_logits['ic1']
    out_IC2 = val_logits['ic2']
    out_IC3 = val_logits['ic3']
    out = val_logits['out']
    target = val_logits['true_labels']
    teacher_out_i = val_logits['i']
    teacher_out_mv = val_logits['mv']
    teacher_out_r = val_logits['r']

    out2 = beta2[0] * out_IC1 + beta2[1] * out_IC2
    out3 = beta3[0] * out_IC1 + beta3[1] * out_IC2 + beta3[2] * out_IC3
    out4 = beta4[0] * out_IC1 + beta4[1] * out_IC2 + beta4[2] * out_IC3 + beta4[3] * teacher_out_i + beta4[4] * teacher_out_mv + beta4[5] * teacher_out_r

    out2_fixed = out_IC1 + out_IC2
    out3_fixed = out_IC1 + out_IC2 + out_IC3
    out4_fixed = out_IC1 + out_IC2 + out_IC3 + teacher_out_i + teacher_out_mv + teacher_out_r

    acc_out2 = (torch.argmax(torch.nn.functional.softmax(out2, 1), 1) == target).sum() / len(target)
    acc_out3 = (torch.argmax(torch.nn.functional.softmax(out3, 1), 1) == target).sum() / len(target)
    acc_out4 = (torch.argmax(torch.nn.functional.softmax(out4, 1), 1) == target).sum() / len(target)

    acc_out_ic2 = (torch.argmax(torch.nn.functional.softmax(out_IC2, 1), 1) == target).sum() / len(target)
    acc_out_ic3 = (torch.argmax(torch.nn.functional.softmax(out_IC3, 1), 1) == target).sum() / len(target)
    acc_out_ic4 = (torch.argmax(torch.nn.functional.softmax(out, 1), 1) == target).sum() / len(target)

    acc_fixed_ic2 = (torch.argmax(torch.nn.functional.softmax(out2_fixed, 1), 1) == target).sum() / len(target)
    acc_fixed_ic3 = (torch.argmax(torch.nn.functional.softmax(out3_fixed, 1), 1) == target).sum() / len(target)
    acc_fixed_ic4 = (torch.argmax(torch.nn.functional.softmax(out4_fixed, 1), 1) == target).sum() / len(target)

    print(f"Constant Scaling IC2 {acc_fixed_ic2:.4f} IC3 {acc_fixed_ic3:.4f} IC4 {acc_fixed_ic4:.4f}")
    print(f"With WISE IC2 {acc_out2:.4f} IC3 {acc_out3:.4f} IC4 {acc_out4:.4f}")
    print(f"Vanilla IC2 {acc_out_ic2:.4f} IC3 {acc_out_ic3:.4f} IC4 {acc_out_ic4:.4f}")

if __name__ == '__main__':
    main()
