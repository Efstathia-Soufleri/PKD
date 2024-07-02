"""Run testing given a trained model."""

import argparse
import time
import numpy as np
import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision
import matplotlib.pyplot as plt
from dataset import CoviarDataSet
# from model_unrolled_all_IC import Model_unrolled_IC
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale
from model_unrolled_all_IC import model_unrolled_all_IC
import os

parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('--data-name', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--save-scores', type=str, default=None)
parser.add_argument('--test_segments_mv', type=int, default=25)
parser.add_argument('--test_segments_r', type=int, default=25)
parser.add_argument('--test_segments_i', type=int, default=25)
parser.add_argument('--test-crops', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.999999)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--learning_type', type=str)
parser.add_argument('--split', type=str)
args = parser.parse_args()


if args.data_name == 'ucf101':
    num_class = 101
elif args.data_name == 'hmdb51':
    num_class = 51
else:
    raise ValueError('Unknown dataset '+args.data_name)


def Average(lst): 
    return sum(lst) / len(lst) 

def optimize_scaling_coefficients():
    output1_mv = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC1_mv_frames_'+str(args.test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_mv = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC2_mv_frames_'+str(args.test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_mv = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC3_mv_frames_'+str(args.test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_mv = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC4_mv_frames_'+str(args.test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_r = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC1_residual_frames_'+str(args.test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_r = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC2_residual_frames_'+str(args.test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_r = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC3_residual_frames_'+str(args.test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_r = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC4_residual_frames_'+str(args.test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_iframe = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC1_iframe_frames_'+str(args.test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_iframe = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC2_iframe_frames_'+str(args.test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_iframe = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC3_iframe_frames_'+str(args.test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_iframe = torch.load('./logits/ucf101_kinetics_pretrained_trainsets/IC4_iframe_frames_'+str(args.test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')


    num_samples = len(output1_r)
    print('Number of Samples: ', num_samples)

    pred1_mv = torch.zeros((num_class, num_samples))
    pred2_mv = torch.zeros((num_class, num_samples))
    pred3_mv = torch.zeros((num_class, num_samples))
    pred4_mv = torch.zeros((num_class, num_samples))

    pred1_r = torch.zeros((num_class, num_samples))
    pred2_r = torch.zeros((num_class, num_samples))
    pred3_r = torch.zeros((num_class, num_samples))
    pred4_r = torch.zeros((num_class, num_samples))

    pred1_iframe = torch.zeros((num_class, num_samples))
    pred2_iframe = torch.zeros((num_class, num_samples))
    pred3_iframe = torch.zeros((num_class, num_samples))
    pred4_iframe = torch.zeros((num_class, num_samples))

    labels = torch.zeros((num_samples, 1))

    for ii in range(0, num_samples):
        pred4_mv[:,ii] = output4_mv[ii][0]
        pred4_r[:,ii] = output4_r[ii][0]
        pred4_iframe[:,ii] = output4_iframe[ii][0]

        pred1_mv[:,ii] = output1_mv[ii][0]
        pred1_r[:,ii] = output1_r[ii][0]
        pred1_iframe[:,ii] = output1_iframe[ii][0]

        pred2_mv[:,ii] = output2_mv[ii][0]
        pred2_r[:,ii] = output2_r[ii][0]
        pred2_iframe[:,ii] = output2_iframe[ii][0]

        pred3_mv[:,ii] = output3_mv[ii][0]
        pred3_r[:,ii] = output3_r[ii][0]
        pred3_iframe[:,ii] = output3_iframe[ii][0]

        labels[ii] = output1_mv[ii][1]


    labels = labels.squeeze()
    labels = labels.long()

    iterations = 500
    ########################
    num_classifiers = 2
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    w_r1_r2 = torch.Tensor([1.0, 1.0])
    w_r1_r2 = torch.nn.Parameter(w_r1_r2)
    optimizer = torch.optim.SGD([w_r1_r2], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.0

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2)

    # w_r1_r2 = [0.2498, 0.7958]

    ################################
    num_classifiers = 3
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    w_r1_r2_r3 = torch.Tensor([1.0, 1.0, 1.0])
    w_r1_r2_r3 = torch.nn.Parameter(w_r1_r2_r3)
    optimizer = torch.optim.SGD([w_r1_r2_r3], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.0

    for epoch in range(iterations): 
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3)

    # w_r1_r2_r3 = [0.1281, 0.1388, 0.8283]

    ################################
    num_classifiers = 4
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    w_r1_r2_r3_r4 = torch.Tensor([1.0, 1.0, 1.0, 1.0])
    w_r1_r2_r3_r4 = torch.nn.Parameter(w_r1_r2_r3_r4)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.0

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4)

    # w_r1_r2_r3_r4 = [0.0105, 0.1209, 0.4798, 2.0171]

    #############################################
    num_classifiers = 5
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    logits[4][:][:] = pred1_mv
    w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
    w_r1_r2_r3_r4_mv1 = torch.nn.Parameter(w_r1_r2_r3_r4_mv1)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4_mv1], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.0

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4_mv1[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4_mv1, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4_mv1)

    # w_r1_r2_r3_r4_mv1 = 0.0124, 0.0550, 0.1325, 1.1510, 0.7957 # [0.0641, 0.0785, 0.2338, 1.2906, 0.6676]

    #############################################
    num_classifiers = 6
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    logits[4][:][:] = pred1_mv
    logits[5][:][:] = pred2_mv
    w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
    w_r1_r2_r3_r4_mv1_mv2 = torch.nn.Parameter(w_r1_r2_r3_r4_mv1_mv2)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4_mv1_mv2], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.1

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4_mv1_mv2[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4_mv1_mv2, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4_mv1_mv2)

    # w_r1_r2_r3_r4_mv1_mv2 = [ 0.0046, -0.0011,  0.0774,  0.5375,  0.0156,  0.0504] # [0.0047, 0.0017, 0.0763, 0.5367, 0.0097, 0.0496]

    #############################################
    num_classifiers = 7
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    logits[4][:][:] = pred1_mv
    logits[5][:][:] = pred2_mv
    logits[6][:][:] = pred3_mv
    w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
    w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.nn.Parameter(w_r1_r2_r3_r4_mv1_mv2_mv3)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4_mv1_mv2_mv3], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.1

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4_mv1_mv2_mv3[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4_mv1_mv2_mv3, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4_mv1_mv2_mv3)

    # w_r1_r2_r3_r4_mv1_mv2_mv3 = [-0.0041, -0.0023,  0.0561,  0.5151,  0.0125,  0.0119,  0.0874] # [0.0047, 0.0017, 0.0763, 0.5367, 0.0097, 0.0496]

    #############################################
    num_classifiers = 8
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    logits[4][:][:] = pred1_mv
    logits[5][:][:] = pred2_mv
    logits[6][:][:] = pred3_mv
    logits[7][:][:] = pred4_mv
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.nn.Parameter(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4_mv1_mv2_mv3_mv4], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.1

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4)

    # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = [-0.0036,  0.0005,  0.0100,  0.4305,  0.0084,  0.0126,  0.0109,  0.1461]

    #############################################
    num_classifiers = 9
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    logits[4][:][:] = pred1_mv
    logits[5][:][:] = pred2_mv
    logits[6][:][:] = pred3_mv
    logits[7][:][:] = pred4_mv
    logits[8][:][:] = pred1_iframe
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.nn.Parameter(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.1

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1)

    # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = [ 0.0055,  0.0043,  0.0116,  0.4057, -0.0004,  0.0124, -0.0040,  0.1355, 0.0391]

    #############################################
    num_classifiers = 10
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    logits[4][:][:] = pred1_mv
    logits[5][:][:] = pred2_mv
    logits[6][:][:] = pred3_mv
    logits[7][:][:] = pred4_mv
    logits[8][:][:] = pred1_iframe
    logits[9][:][:] = pred2_iframe
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.nn.Parameter(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.1

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2)

    # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = [ 0.0005,  0.0134,  0.0101,  0.3615,  0.0005,  0.0116, -0.0011,  0.1194, 0.0058,  0.1005]

##############################################

    num_classifiers = 11
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    logits[4][:][:] = pred1_mv
    logits[5][:][:] = pred2_mv
    logits[6][:][:] = pred3_mv
    logits[7][:][:] = pred4_mv
    logits[8][:][:] = pred1_iframe
    logits[9][:][:] = pred2_iframe
    logits[10][:][:] = pred3_iframe
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.nn.Parameter(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.1

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3)

    # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = [-0.0029,  0.0091,  0.0069,  0.2899,  0.0066,  0.0109,  0.0028,  0.0881, 0.0071,  0.0134,  0.1810]

    ##############################################

    num_classifiers = 12
    logits = torch.zeros((num_classifiers, num_class, num_samples))
    logits[0][:][:] = pred1_r
    logits[1][:][:] = pred2_r
    logits[2][:][:] = pred3_r
    logits[3][:][:] = pred4_r
    logits[4][:][:] = pred1_mv
    logits[5][:][:] = pred2_mv
    logits[6][:][:] = pred3_mv
    logits[7][:][:] = pred4_mv
    logits[8][:][:] = pred1_iframe
    logits[9][:][:] = pred2_iframe
    logits[10][:][:] = pred3_iframe
    logits[11][:][:] = pred4_iframe
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.nn.Parameter(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
    optimizer = torch.optim.SGD([w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.1

    for epoch in range(iterations):
        optimizer.zero_grad()
        weighted_res = torch.zeros_like(logits[0])
        for i in range(num_classifiers):
            weighted_res += w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[i] * logits[i]
        loss = criterion(weighted_res.T, labels) + alpha * torch.norm(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4, p=1)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)

    # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = [ 0.0006, -0.0029,  0.0044,  0.1664,  0.0052,  0.0134, -0.0031,  0.0433, 0.0038,  0.0119,  0.0100,  0.2247]

    ##############################################

    return w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4
    
def main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4):
    softmax = torch.nn.Softmax(dim=1)

    # evaluation
    output1_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC1_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC2_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC3_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC4_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_r = torch.load('./logits/ucf101_kinetics_pretrained/IC1_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_r = torch.load('./logits/ucf101_kinetics_pretrained/IC2_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_r = torch.load('./logits/ucf101_kinetics_pretrained/IC3_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_r = torch.load('./logits/ucf101_kinetics_pretrained/IC4_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC1_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC2_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC3_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC4_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    num_samples_test = len(output1_mv)
    print('Number of Samples: ', num_samples_test)

    pred1_mv = []
    pred2_mv = []
    pred3_mv = []
    pred4_mv = []

    pred1_r = []
    pred2_r = []
    pred3_r = []
    pred4_r = []

    pred1_iframe = []
    pred2_iframe = []
    pred3_iframe = []
    pred4_iframe = []

    labels = []
    for ii in range(0, num_samples_test):
        pred1_mv.append(output1_mv[ii][0])
        pred2_mv.append(output2_mv[ii][0])
        pred3_mv.append(output3_mv[ii][0])
        pred4_mv.append(output4_mv[ii][0])

        pred1_r.append(output1_r[ii][0])
        pred2_r.append(output2_r[ii][0])
        pred3_r.append(output3_r[ii][0])
        pred4_r.append(output4_r[ii][0])

        pred1_iframe.append(output1_iframe[ii][0])
        pred2_iframe.append(output2_iframe[ii][0])
        pred3_iframe.append(output3_iframe[ii][0])
        pred4_iframe.append(output4_iframe[ii][0])

        labels.append(output1_mv[ii][1])


    w_r1_r2 = w_r1_r2.detach().numpy()
    w_r1_r2_r3 = w_r1_r2_r3.detach().numpy()
    w_r1_r2_r3_r4 = w_r1_r2_r3_r4.detach().numpy()
    w_r1_r2_r3_r4_mv1 = w_r1_r2_r3_r4_mv1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2 = w_r1_r2_r3_r4_mv1_mv2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3 = w_r1_r2_r3_r4_mv1_mv2_mv3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4.detach().numpy()

    counter_IC1_predictions_mv = 0
    correct_IC1_predictions_mv = 0

    counter_IC2_predictions_mv = 0
    correct_IC2_predictions_mv = 0

    counter_IC3_predictions_mv = 0
    correct_IC3_predictions_mv = 0

    counter_IC4_predictions_mv = 0
    correct_IC4_predictions_mv = 0

    counter_IC1_predictions_r = 0
    correct_IC1_predictions_r = 0

    counter_IC2_predictions_r = 0
    correct_IC2_predictions_r = 0

    counter_IC3_predictions_r = 0
    correct_IC3_predictions_r = 0

    counter_IC4_predictions_r = 0
    correct_IC4_predictions_r = 0

    counter_IC1_predictions_iframe = 0
    correct_IC1_predictions_iframe = 0

    counter_IC2_predictions_iframe = 0
    correct_IC2_predictions_iframe = 0

    counter_IC3_predictions_iframe = 0
    correct_IC3_predictions_iframe = 0

    counter_IC4_predictions_iframe = 0
    correct_IC4_predictions_iframe = 0

    correct = 0

    for ii in range(0, num_samples_test):
        # residual network
        if softmax(pred1_r[ii]).max() > threshold:
            prediction = np.argmax(pred1_r[ii])
            counter_IC1_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii])/2)).max() > threshold:
            prediction = np.argmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii])/2)
            counter_IC2_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii])/3)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii])/3)
            counter_IC3_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii])/4)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii])/4)
            counter_IC4_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_r += 1
                correct += 1

        # MV network
        elif softmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii])/5).max() > threshold:
            # print(w_r1_r2_r3_r4_mv1)
            prediction = np.argmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii])/5)
            counter_IC1_predictions_mv += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii])/6)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii])/6)
            counter_IC2_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC2_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii])/7)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii])/7)
            counter_IC3_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC3_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii])/8)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii])/8)
            counter_IC4_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC4_predictions_mv += 1
                correct += 1

        # iframe network
        elif softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii])/9).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii])/9)
            counter_IC1_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii])/10)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii])/10)
            counter_IC2_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii])/11)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii])/11)
            counter_IC3_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_iframe += 1
                correct += 1
        else:
            if args.split == 'split1':
                prediction = np.argmax((1.8532*pred4_iframe[ii]+1.3720*pred4_r[ii]+0.5565*pred4_mv[ii])/3)
            elif args.split == 'split2':
                prediction = np.argmax((1.8532*pred4_iframe[ii]+1.3720*pred4_r[ii]+0.5565*pred4_mv[ii])/3)
            elif args.split == 'split3':
                prediction = np.argmax((1.5995*pred4_iframe[ii]+1.4348*pred4_r[ii]+0.9558*pred4_mv[ii])/3)
            counter_IC4_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_iframe += 1
                correct += 1

    cost_IC1_r_flops = 1.09
    cost_IC2_r_flops = 1.09+0.88
    cost_IC3_r_flops = 1.09+0.88+0.52
    cost_IC4_r_flops = 1.09+0.88+0.52+0.41

    cost_IC1_mv_flops = 1.13
    cost_IC2_mv_flops = 1.13+0.87
    cost_IC3_mv_flops = 1.13+0.87+0.53
    cost_IC4_mv_flops = 1.13+0.87+0.53+0.41
    cost_r = 2.9
    cost_mv = 2.94

    cost_IC1_i_flops = 2.97 - 1.71
    cost_IC2_i_flops = 2.97 + 2.88 - 1.71 - 0.92
    cost_IC3_i_flops = 2.97 + 2.88 + 3.32 - 1.71 - 0.92
    cost_IC4_i_flops = 2.97 + 2.88 + 3.32 + 0.81 - 1.71 - 0.92

    cost_flops = (cost_IC1_r_flops *counter_IC1_predictions_r + \
                    cost_IC2_r_flops *counter_IC2_predictions_r + \
                    cost_IC3_r_flops *counter_IC3_predictions_r + \
                    cost_IC4_r_flops *counter_IC4_predictions_r)*(test_segments_r) + \
                    (cost_IC1_mv_flops *counter_IC1_predictions_mv + \
                    cost_IC2_mv_flops *counter_IC2_predictions_mv + \
                    cost_IC3_mv_flops *counter_IC3_predictions_mv + \
                    cost_IC4_mv_flops *counter_IC4_predictions_mv)*(test_segments_mv) + \
                    (cost_r *counter_IC1_predictions_mv + \
                    cost_r *counter_IC2_predictions_mv + \
                    cost_r *counter_IC3_predictions_mv + \
                    cost_r *counter_IC4_predictions_mv)*(test_segments_r) + \
                    (cost_IC1_i_flops *counter_IC1_predictions_iframe + \
                    cost_IC2_i_flops *counter_IC2_predictions_iframe + \
                    cost_IC3_i_flops *counter_IC3_predictions_iframe + \
                    cost_IC4_i_flops *counter_IC4_predictions_iframe)*(test_segments_i) + \
                    (cost_r *counter_IC1_predictions_iframe + \
                    cost_r *counter_IC2_predictions_iframe + \
                    cost_r *counter_IC3_predictions_iframe + \
                    cost_r *counter_IC4_predictions_iframe)*(test_segments_r) + \
                    (cost_mv *counter_IC1_predictions_iframe + \
                    cost_mv *counter_IC2_predictions_iframe + \
                    cost_mv *counter_IC3_predictions_iframe + \
                    cost_mv *counter_IC4_predictions_iframe)*(test_segments_mv) 
    
    print(counter_IC1_predictions_iframe,counter_IC2_predictions_iframe,counter_IC3_predictions_iframe,counter_IC4_predictions_iframe)
    print(counter_IC1_predictions_r,counter_IC2_predictions_r,counter_IC3_predictions_r,counter_IC4_predictions_r)
    print(counter_IC1_predictions_mv,counter_IC2_predictions_mv,counter_IC3_predictions_mv,counter_IC4_predictions_mv)

    print('COST:', cost_flops)
    print("ACC: ", (correct/num_samples_test)*100)
    acc = (correct/num_samples_test)*100

    return acc, cost_flops

def main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4):
    softmax = torch.nn.Softmax(dim=1)
    # threshold = threshold

    # evaluation
    output1_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC1_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC2_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC3_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC4_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_r = torch.load('./logits/ucf101_kinetics_pretrained/IC1_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_r = torch.load('./logits/ucf101_kinetics_pretrained/IC2_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_r = torch.load('./logits/ucf101_kinetics_pretrained/IC3_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_r = torch.load('./logits/ucf101_kinetics_pretrained/IC4_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC1_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC2_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC3_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC4_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    num_samples_test = len(output1_mv)
    print('Number of Samples: ', num_samples_test)

    pred1_mv = []
    pred2_mv = []
    pred3_mv = []
    pred4_mv = []

    pred1_r = []
    pred2_r = []
    pred3_r = []
    pred4_r = []

    pred1_iframe = []
    pred2_iframe = []
    pred3_iframe = []
    pred4_iframe = []

    labels = []
    for ii in range(0, num_samples_test):
        pred1_mv.append(output1_mv[ii][0])
        pred2_mv.append(output2_mv[ii][0])
        pred3_mv.append(output3_mv[ii][0])
        pred4_mv.append(output4_mv[ii][0])

        pred1_r.append(output1_r[ii][0])
        pred2_r.append(output2_r[ii][0])
        pred3_r.append(output3_r[ii][0])
        pred4_r.append(output4_r[ii][0])

        pred1_iframe.append(output1_iframe[ii][0])
        pred2_iframe.append(output2_iframe[ii][0])
        pred3_iframe.append(output3_iframe[ii][0])
        pred4_iframe.append(output4_iframe[ii][0])

        labels.append(output1_mv[ii][1])


    w_r1_r2 = w_r1_r2.detach().numpy()
    w_r1_r2_r3 = w_r1_r2_r3.detach().numpy()
    w_r1_r2_r3_r4 = w_r1_r2_r3_r4.detach().numpy()
    w_r1_r2_r3_r4_mv1 = w_r1_r2_r3_r4_mv1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2 = w_r1_r2_r3_r4_mv1_mv2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3 = w_r1_r2_r3_r4_mv1_mv2_mv3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4.detach().numpy()

    counter_IC1_predictions_mv = 0
    correct_IC1_predictions_mv = 0

    counter_IC2_predictions_mv = 0
    correct_IC2_predictions_mv = 0

    counter_IC3_predictions_mv = 0
    correct_IC3_predictions_mv = 0

    counter_IC4_predictions_mv = 0
    correct_IC4_predictions_mv = 0

    counter_IC1_predictions_r = 0
    correct_IC1_predictions_r = 0

    counter_IC2_predictions_r = 0
    correct_IC2_predictions_r = 0

    counter_IC3_predictions_r = 0
    correct_IC3_predictions_r = 0

    counter_IC4_predictions_r = 0
    correct_IC4_predictions_r = 0

    counter_IC1_predictions_iframe = 0
    correct_IC1_predictions_iframe = 0

    counter_IC2_predictions_iframe = 0
    correct_IC2_predictions_iframe = 0

    counter_IC3_predictions_iframe = 0
    correct_IC3_predictions_iframe = 0

    counter_IC4_predictions_iframe = 0
    correct_IC4_predictions_iframe = 0

    correct = 0

    for ii in range(0, num_samples_test):
        # residual network
        if softmax(pred1_r[ii]).max() > threshold:
            prediction = np.argmax(pred1_r[ii])
            counter_IC1_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii])/2)).max() > threshold:
            prediction = np.argmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii])/2)
            counter_IC2_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii])/3)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii])/3)
            counter_IC3_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii])/4)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii])/4)
            counter_IC4_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_r += 1
                correct += 1

        # MV network
        elif softmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii])/5).max() > threshold:
            # print(w_r1_r2_r3_r4_mv1)
            prediction = np.argmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii])/5)
            counter_IC1_predictions_mv += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii])/6)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii])/6)
            counter_IC2_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC2_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii])/7)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii])/7)
            counter_IC3_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC3_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii])/8)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii])/8)
            counter_IC4_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC4_predictions_mv += 1
                correct += 1

        # iframe network
        elif softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii])/9).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii])/9)
            counter_IC1_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii])/10)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii])/10)
            counter_IC2_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii])/11)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii])/11)
            counter_IC3_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_iframe += 1
                correct += 1
        else:
            prediction = np.argmax((pred4_iframe[ii]+pred4_r[ii]+pred4_mv[ii])/3)
            counter_IC4_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_iframe += 1
                correct += 1

    cost_IC1_r_flops = 1.09
    cost_IC2_r_flops = 1.09+0.88
    cost_IC3_r_flops = 1.09+0.88+0.52
    cost_IC4_r_flops = 1.09+0.88+0.52+0.41

    cost_IC1_mv_flops = 1.13
    cost_IC2_mv_flops = 1.13+0.87
    cost_IC3_mv_flops = 1.13+0.87+0.53
    cost_IC4_mv_flops = 1.13+0.87+0.53+0.41
    cost_r = 2.9
    cost_mv = 2.94

    cost_IC1_i_flops = 2.97 - 1.71
    cost_IC2_i_flops = 2.97 + 2.88 - 1.71 - 0.92
    cost_IC3_i_flops = 2.97 + 2.88 + 3.32 - 1.71 - 0.92
    cost_IC4_i_flops = 2.97 + 2.88 + 3.32 + 0.81 - 1.71 - 0.92

    cost_flops = (cost_IC1_r_flops *counter_IC1_predictions_r + \
                    cost_IC2_r_flops *counter_IC2_predictions_r + \
                    cost_IC3_r_flops *counter_IC3_predictions_r + \
                    cost_IC4_r_flops *counter_IC4_predictions_r)*(test_segments_r) + \
                    (cost_IC1_mv_flops *counter_IC1_predictions_mv + \
                    cost_IC2_mv_flops *counter_IC2_predictions_mv + \
                    cost_IC3_mv_flops *counter_IC3_predictions_mv + \
                    cost_IC4_mv_flops *counter_IC4_predictions_mv)*(test_segments_mv) + \
                    (cost_r *counter_IC1_predictions_mv + \
                    cost_r *counter_IC2_predictions_mv + \
                    cost_r *counter_IC3_predictions_mv + \
                    cost_r *counter_IC4_predictions_mv)*(test_segments_r) + \
                    (cost_IC1_i_flops *counter_IC1_predictions_iframe + \
                    cost_IC2_i_flops *counter_IC2_predictions_iframe + \
                    cost_IC3_i_flops *counter_IC3_predictions_iframe + \
                    cost_IC4_i_flops *counter_IC4_predictions_iframe)*(test_segments_i) + \
                    (cost_r *counter_IC1_predictions_iframe + \
                    cost_r *counter_IC2_predictions_iframe + \
                    cost_r *counter_IC3_predictions_iframe + \
                    cost_r *counter_IC4_predictions_iframe)*(test_segments_r) + \
                    (cost_mv *counter_IC1_predictions_iframe + \
                    cost_mv *counter_IC2_predictions_iframe + \
                    cost_mv *counter_IC3_predictions_iframe + \
                    cost_mv *counter_IC4_predictions_iframe)*(test_segments_mv) 
    
    print(counter_IC1_predictions_iframe,counter_IC2_predictions_iframe,counter_IC3_predictions_iframe,counter_IC4_predictions_iframe)
    print(counter_IC1_predictions_r,counter_IC2_predictions_r,counter_IC3_predictions_r,counter_IC4_predictions_r)
    print(counter_IC1_predictions_mv,counter_IC2_predictions_mv,counter_IC3_predictions_mv,counter_IC4_predictions_mv)

    print('COST:', cost_flops)
    print("ACC: ", (correct/num_samples_test)*100)
    acc = (correct/num_samples_test)*100

    return acc, cost_flops

def main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4):
    softmax = torch.nn.Softmax(dim=1)

    # evaluation
    output1_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC1_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC2_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC3_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC4_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_r = torch.load('./logits/ucf101_kinetics_pretrained/IC1_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_r = torch.load('./logits/ucf101_kinetics_pretrained/IC2_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_r = torch.load('./logits/ucf101_kinetics_pretrained/IC3_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_r = torch.load('./logits/ucf101_kinetics_pretrained/IC4_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC1_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC2_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC3_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC4_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    num_samples_test = len(output1_mv)
    print('Number of Samples: ', num_samples_test)

    pred1_mv = []
    pred2_mv = []
    pred3_mv = []
    pred4_mv = []

    pred1_r = []
    pred2_r = []
    pred3_r = []
    pred4_r = []

    pred1_iframe = []
    pred2_iframe = []
    pred3_iframe = []
    pred4_iframe = []

    labels = []
    for ii in range(0, num_samples_test):
        pred1_mv.append(output1_mv[ii][0])
        pred2_mv.append(output2_mv[ii][0])
        pred3_mv.append(output3_mv[ii][0])
        pred4_mv.append(output4_mv[ii][0])

        pred1_r.append(output1_r[ii][0])
        pred2_r.append(output2_r[ii][0])
        pred3_r.append(output3_r[ii][0])
        pred4_r.append(output4_r[ii][0])

        pred1_iframe.append(output1_iframe[ii][0])
        pred2_iframe.append(output2_iframe[ii][0])
        pred3_iframe.append(output3_iframe[ii][0])
        pred4_iframe.append(output4_iframe[ii][0])

        labels.append(output1_mv[ii][1])


    w_r1_r2 = w_r1_r2.detach().numpy()
    w_r1_r2_r3 = w_r1_r2_r3.detach().numpy()
    w_r1_r2_r3_r4 = w_r1_r2_r3_r4.detach().numpy()
    w_r1_r2_r3_r4_mv1 = w_r1_r2_r3_r4_mv1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2 = w_r1_r2_r3_r4_mv1_mv2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3 = w_r1_r2_r3_r4_mv1_mv2_mv3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4.detach().numpy()

    counter_IC1_predictions_mv = 0
    correct_IC1_predictions_mv = 0

    counter_IC2_predictions_mv = 0
    correct_IC2_predictions_mv = 0

    counter_IC3_predictions_mv = 0
    correct_IC3_predictions_mv = 0

    counter_IC4_predictions_mv = 0
    correct_IC4_predictions_mv = 0

    counter_IC1_predictions_r = 0
    correct_IC1_predictions_r = 0

    counter_IC2_predictions_r = 0
    correct_IC2_predictions_r = 0

    counter_IC3_predictions_r = 0
    correct_IC3_predictions_r = 0

    counter_IC4_predictions_r = 0
    correct_IC4_predictions_r = 0

    counter_IC1_predictions_iframe = 0
    correct_IC1_predictions_iframe = 0

    counter_IC2_predictions_iframe = 0
    correct_IC2_predictions_iframe = 0

    counter_IC3_predictions_iframe = 0
    correct_IC3_predictions_iframe = 0

    counter_IC4_predictions_iframe = 0
    correct_IC4_predictions_iframe = 0

    correct = 0

    for ii in range(0, num_samples_test):
        # residual network
        if softmax(pred1_r[ii]).max() > threshold:
            prediction = np.argmax(pred1_r[ii])
            counter_IC1_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii]))
            counter_IC2_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii]))
            counter_IC3_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii]))
            counter_IC4_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_r += 1
                correct += 1

        # MV network
        elif softmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii])).max() > threshold:
            # print(w_r1_r2_r3_r4_mv1)
            prediction = np.argmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii]))
            counter_IC1_predictions_mv += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii]))
            counter_IC2_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC2_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii]))
            counter_IC3_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC3_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii]))
            counter_IC4_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC4_predictions_mv += 1
                correct += 1

        # iframe network
        elif softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii])).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii]))
            counter_IC1_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii]))
            counter_IC2_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii]))
            counter_IC3_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_iframe += 1
                correct += 1
        else:
            prediction = np.argmax((pred4_iframe[ii]+pred4_r[ii]+pred4_mv[ii]))
            counter_IC4_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_iframe += 1
                correct += 1

    cost_IC1_r_flops = 1.09
    cost_IC2_r_flops = 1.09+0.88
    cost_IC3_r_flops = 1.09+0.88+0.52
    cost_IC4_r_flops = 1.09+0.88+0.52+0.41

    cost_IC1_mv_flops = 1.13
    cost_IC2_mv_flops = 1.13+0.87
    cost_IC3_mv_flops = 1.13+0.87+0.53
    cost_IC4_mv_flops = 1.13+0.87+0.53+0.41
    cost_r = 2.9
    cost_mv = 2.94

    cost_IC1_i_flops = 2.97 - 1.71
    cost_IC2_i_flops = 2.97 + 2.88 - 1.71 - 0.92
    cost_IC3_i_flops = 2.97 + 2.88 + 3.32 - 1.71 - 0.92
    cost_IC4_i_flops = 2.97 + 2.88 + 3.32 + 0.81 - 1.71 - 0.92

    cost_flops = (cost_IC1_r_flops *counter_IC1_predictions_r + \
                    cost_IC2_r_flops *counter_IC2_predictions_r + \
                    cost_IC3_r_flops *counter_IC3_predictions_r + \
                    cost_IC4_r_flops *counter_IC4_predictions_r)*(test_segments_r) + \
                    (cost_IC1_mv_flops *counter_IC1_predictions_mv + \
                    cost_IC2_mv_flops *counter_IC2_predictions_mv + \
                    cost_IC3_mv_flops *counter_IC3_predictions_mv + \
                    cost_IC4_mv_flops *counter_IC4_predictions_mv)*(test_segments_mv) + \
                    (cost_r *counter_IC1_predictions_mv + \
                    cost_r *counter_IC2_predictions_mv + \
                    cost_r *counter_IC3_predictions_mv + \
                    cost_r *counter_IC4_predictions_mv)*(test_segments_r) + \
                    (cost_IC1_i_flops *counter_IC1_predictions_iframe + \
                    cost_IC2_i_flops *counter_IC2_predictions_iframe + \
                    cost_IC3_i_flops *counter_IC3_predictions_iframe + \
                    cost_IC4_i_flops *counter_IC4_predictions_iframe)*(test_segments_i) + \
                    (cost_r *counter_IC1_predictions_iframe + \
                    cost_r *counter_IC2_predictions_iframe + \
                    cost_r *counter_IC3_predictions_iframe + \
                    cost_r *counter_IC4_predictions_iframe)*(test_segments_r) + \
                    (cost_mv *counter_IC1_predictions_iframe + \
                    cost_mv *counter_IC2_predictions_iframe + \
                    cost_mv *counter_IC3_predictions_iframe + \
                    cost_mv *counter_IC4_predictions_iframe)*(test_segments_mv) 
    
    print(counter_IC1_predictions_iframe,counter_IC2_predictions_iframe,counter_IC3_predictions_iframe,counter_IC4_predictions_iframe)
    print(counter_IC1_predictions_r,counter_IC2_predictions_r,counter_IC3_predictions_r,counter_IC4_predictions_r)
    print(counter_IC1_predictions_mv,counter_IC2_predictions_mv,counter_IC3_predictions_mv,counter_IC4_predictions_mv)

    print('COST:', cost_flops)
    print("ACC: ", (correct/num_samples_test)*100)
    acc = (correct/num_samples_test)*100

    return acc, cost_flops

def main_CE(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4):
    softmax = torch.nn.Softmax(dim=1)
    # threshold = threshold
    learning_type = 'CE'

    # evaluation
    output1_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC1_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC2_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC3_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC4_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_r = torch.load('./logits/ucf101_kinetics_pretrained/IC1_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_r = torch.load('./logits/ucf101_kinetics_pretrained/IC2_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_r = torch.load('./logits/ucf101_kinetics_pretrained/IC3_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_r = torch.load('./logits/ucf101_kinetics_pretrained/IC4_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC1_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC2_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC3_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC4_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    num_samples_test = len(output1_mv)
    print('Number of Samples: ', num_samples_test)

    pred1_mv = []
    pred2_mv = []
    pred3_mv = []
    pred4_mv = []

    pred1_r = []
    pred2_r = []
    pred3_r = []
    pred4_r = []

    pred1_iframe = []
    pred2_iframe = []
    pred3_iframe = []
    pred4_iframe = []

    labels = []
    for ii in range(0, num_samples_test):
        pred1_mv.append(output1_mv[ii][0])
        pred2_mv.append(output2_mv[ii][0])
        pred3_mv.append(output3_mv[ii][0])
        pred4_mv.append(output4_mv[ii][0])

        pred1_r.append(output1_r[ii][0])
        pred2_r.append(output2_r[ii][0])
        pred3_r.append(output3_r[ii][0])
        pred4_r.append(output4_r[ii][0])

        pred1_iframe.append(output1_iframe[ii][0])
        pred2_iframe.append(output2_iframe[ii][0])
        pred3_iframe.append(output3_iframe[ii][0])
        pred4_iframe.append(output4_iframe[ii][0])

        labels.append(output1_mv[ii][1])


    w_r1_r2 = w_r1_r2.detach().numpy()
    w_r1_r2_r3 = w_r1_r2_r3.detach().numpy()
    w_r1_r2_r3_r4 = w_r1_r2_r3_r4.detach().numpy()
    w_r1_r2_r3_r4_mv1 = w_r1_r2_r3_r4_mv1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2 = w_r1_r2_r3_r4_mv1_mv2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3 = w_r1_r2_r3_r4_mv1_mv2_mv3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4.detach().numpy()

    counter_IC1_predictions_mv = 0
    correct_IC1_predictions_mv = 0

    counter_IC2_predictions_mv = 0
    correct_IC2_predictions_mv = 0

    counter_IC3_predictions_mv = 0
    correct_IC3_predictions_mv = 0

    counter_IC4_predictions_mv = 0
    correct_IC4_predictions_mv = 0

    counter_IC1_predictions_r = 0
    correct_IC1_predictions_r = 0

    counter_IC2_predictions_r = 0
    correct_IC2_predictions_r = 0

    counter_IC3_predictions_r = 0
    correct_IC3_predictions_r = 0

    counter_IC4_predictions_r = 0
    correct_IC4_predictions_r = 0

    counter_IC1_predictions_iframe = 0
    correct_IC1_predictions_iframe = 0

    counter_IC2_predictions_iframe = 0
    correct_IC2_predictions_iframe = 0

    counter_IC3_predictions_iframe = 0
    correct_IC3_predictions_iframe = 0

    counter_IC4_predictions_iframe = 0
    correct_IC4_predictions_iframe = 0

    correct = 0

    for ii in range(0, num_samples_test):
        # residual network
        if softmax(pred1_r[ii]).max() > threshold:
            prediction = np.argmax(pred1_r[ii])
            counter_IC1_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii])/2)).max() > threshold:
            prediction = np.argmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii])/2)
            counter_IC2_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii])/3)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii])/3)
            counter_IC3_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii])/4)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii])/4)
            counter_IC4_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_r += 1
                correct += 1

        # MV network
        elif softmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii])/5).max() > threshold:
            # print(w_r1_r2_r3_r4_mv1)
            prediction = np.argmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii])/5)
            counter_IC1_predictions_mv += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii])/6)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii])/6)
            counter_IC2_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC2_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii])/7)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii])/7)
            counter_IC3_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC3_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii])/8)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii])/8)
            counter_IC4_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC4_predictions_mv += 1
                correct += 1

        # iframe network
        elif softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii])/9).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii])/9)
            counter_IC1_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii])/10)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii])/10)
            counter_IC2_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii])/11)).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii])/11)
            counter_IC3_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_iframe += 1
                correct += 1
        else:
            # prediction = np.argmax((2*pred4_iframe[ii]+1*pred4_r[ii]+1*pred4_mv[ii])/3)
            # prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[10]*pred3_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[11]*pred4_iframe[ii])/11)
            # prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[11]*pred4_iframe[ii])/3)
            if args.split == 'split1':
                prediction = np.argmax((1*pred4_iframe[ii]+1*pred4_r[ii]+1*pred4_mv[ii])/3)
            elif args.split == 'split2':
                prediction = np.argmax((1*pred4_iframe[ii]+1*pred4_r[ii]+1*pred4_mv[ii])/3)
            elif args.split == 'split3':
                prediction = np.argmax((1*pred4_iframe[ii]+1*pred4_r[ii]+1*pred4_mv[ii])/3)
            counter_IC4_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_iframe += 1
                correct += 1

    cost_IC1_r_flops = 1.09
    cost_IC2_r_flops = 1.09+0.88
    cost_IC3_r_flops = 1.09+0.88+0.52
    cost_IC4_r_flops = 1.09+0.88+0.52+0.41

    cost_IC1_mv_flops = 1.13
    cost_IC2_mv_flops = 1.13+0.87
    cost_IC3_mv_flops = 1.13+0.87+0.53
    cost_IC4_mv_flops = 1.13+0.87+0.53+0.41
    cost_r = 2.9
    cost_mv = 2.94

    cost_IC1_i_flops = 2.97 - 1.71
    cost_IC2_i_flops = 2.97 + 2.88 - 1.71 - 0.92
    cost_IC3_i_flops = 2.97 + 2.88 + 3.32 - 1.71 - 0.92
    cost_IC4_i_flops = 2.97 + 2.88 + 3.32 + 0.81 - 1.71 - 0.92

    cost_flops = (cost_IC1_r_flops *counter_IC1_predictions_r + \
                    cost_IC2_r_flops *counter_IC2_predictions_r + \
                    cost_IC3_r_flops *counter_IC3_predictions_r + \
                    cost_IC4_r_flops *counter_IC4_predictions_r)*(test_segments_r) + \
                    (cost_IC1_mv_flops *counter_IC1_predictions_mv + \
                    cost_IC2_mv_flops *counter_IC2_predictions_mv + \
                    cost_IC3_mv_flops *counter_IC3_predictions_mv + \
                    cost_IC4_mv_flops *counter_IC4_predictions_mv)*(test_segments_mv) + \
                    (cost_r *counter_IC1_predictions_mv + \
                    cost_r *counter_IC2_predictions_mv + \
                    cost_r *counter_IC3_predictions_mv + \
                    cost_r *counter_IC4_predictions_mv)*(test_segments_r) + \
                    (cost_IC1_i_flops *counter_IC1_predictions_iframe + \
                    cost_IC2_i_flops *counter_IC2_predictions_iframe + \
                    cost_IC3_i_flops *counter_IC3_predictions_iframe + \
                    cost_IC4_i_flops *counter_IC4_predictions_iframe)*(test_segments_i) + \
                    (cost_r *counter_IC1_predictions_iframe + \
                    cost_r *counter_IC2_predictions_iframe + \
                    cost_r *counter_IC3_predictions_iframe + \
                    cost_r *counter_IC4_predictions_iframe)*(test_segments_r) + \
                    (cost_mv *counter_IC1_predictions_iframe + \
                    cost_mv *counter_IC2_predictions_iframe + \
                    cost_mv *counter_IC3_predictions_iframe + \
                    cost_mv *counter_IC4_predictions_iframe)*(test_segments_mv) 
    
    print(counter_IC1_predictions_iframe,counter_IC2_predictions_iframe,counter_IC3_predictions_iframe,counter_IC4_predictions_iframe)
    print(counter_IC1_predictions_r,counter_IC2_predictions_r,counter_IC3_predictions_r,counter_IC4_predictions_r)
    print(counter_IC1_predictions_mv,counter_IC2_predictions_mv,counter_IC3_predictions_mv,counter_IC4_predictions_mv)

    print('COST:', cost_flops)
    print("ACC: ", (correct/num_samples_test)*100)
    acc = (correct/num_samples_test)*100

    return acc, cost_flops

def main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4):
    softmax = torch.nn.Softmax(dim=1)
    # threshold = threshold

    # evaluation
    output1_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC1_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC2_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC3_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC4_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_r = torch.load('./logits/ucf101_kinetics_pretrained/IC1_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_r = torch.load('./logits/ucf101_kinetics_pretrained/IC2_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_r = torch.load('./logits/ucf101_kinetics_pretrained/IC3_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_r = torch.load('./logits/ucf101_kinetics_pretrained/IC4_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC1_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC2_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC3_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC4_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    num_samples_test = len(output1_mv)
    print('Number of Samples: ', num_samples_test)

    pred1_mv = []
    pred2_mv = []
    pred3_mv = []
    pred4_mv = []

    pred1_r = []
    pred2_r = []
    pred3_r = []
    pred4_r = []

    pred1_iframe = []
    pred2_iframe = []
    pred3_iframe = []
    pred4_iframe = []

    labels = []
    for ii in range(0, num_samples_test):
        pred1_mv.append(output1_mv[ii][0])
        pred2_mv.append(output2_mv[ii][0])
        pred3_mv.append(output3_mv[ii][0])
        pred4_mv.append(output4_mv[ii][0])

        pred1_r.append(output1_r[ii][0])
        pred2_r.append(output2_r[ii][0])
        pred3_r.append(output3_r[ii][0])
        pred4_r.append(output4_r[ii][0])

        pred1_iframe.append(output1_iframe[ii][0])
        pred2_iframe.append(output2_iframe[ii][0])
        pred3_iframe.append(output3_iframe[ii][0])
        pred4_iframe.append(output4_iframe[ii][0])

        labels.append(output1_mv[ii][1])


    w_r1_r2 = w_r1_r2.detach().numpy()
    w_r1_r2_r3 = w_r1_r2_r3.detach().numpy()
    w_r1_r2_r3_r4 = w_r1_r2_r3_r4.detach().numpy()
    w_r1_r2_r3_r4_mv1 = w_r1_r2_r3_r4_mv1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2 = w_r1_r2_r3_r4_mv1_mv2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3 = w_r1_r2_r3_r4_mv1_mv2_mv3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4.detach().numpy()

    counter_IC1_predictions_mv = 0
    correct_IC1_predictions_mv = 0

    counter_IC2_predictions_mv = 0
    correct_IC2_predictions_mv = 0

    counter_IC3_predictions_mv = 0
    correct_IC3_predictions_mv = 0

    counter_IC4_predictions_mv = 0
    correct_IC4_predictions_mv = 0

    counter_IC1_predictions_r = 0
    correct_IC1_predictions_r = 0

    counter_IC2_predictions_r = 0
    correct_IC2_predictions_r = 0

    counter_IC3_predictions_r = 0
    correct_IC3_predictions_r = 0

    counter_IC4_predictions_r = 0
    correct_IC4_predictions_r = 0

    counter_IC1_predictions_iframe = 0
    correct_IC1_predictions_iframe = 0

    counter_IC2_predictions_iframe = 0
    correct_IC2_predictions_iframe = 0

    counter_IC3_predictions_iframe = 0
    correct_IC3_predictions_iframe = 0

    counter_IC4_predictions_iframe = 0
    correct_IC4_predictions_iframe = 0

    correct = 0

    for ii in range(0, num_samples_test):
        # residual network
        if softmax(pred1_r[ii]).max() > threshold:
            prediction = np.argmax(pred1_r[ii])
            counter_IC1_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2[0]*pred1_r[ii]+w_r1_r2[1]*pred2_r[ii]))
            counter_IC2_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3[0]*pred1_r[ii]+w_r1_r2_r3[1]*pred2_r[ii]+w_r1_r2_r3[2]*pred3_r[ii]))
            counter_IC3_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_r += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4[0]*pred1_r[ii]+w_r1_r2_r3_r4[1]*pred2_r[ii]+w_r1_r2_r3_r4[2]*pred3_r[ii]+w_r1_r2_r3_r4[3]*pred4_r[ii]))
            counter_IC4_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_r += 1
                correct += 1

        # MV network
        elif softmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii])).max() > threshold:
            # print(w_r1_r2_r3_r4_mv1)
            prediction = np.argmax((w_r1_r2_r3_r4_mv1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1[4]*pred1_mv[ii]))
            counter_IC1_predictions_mv += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2[5]*pred2_mv[ii]))
            counter_IC2_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC2_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3[6]*pred3_mv[ii]))
            counter_IC3_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC3_predictions_mv += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4[7]*pred4_mv[ii]))
            counter_IC4_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC4_predictions_mv += 1
                correct += 1

        # iframe network
        elif softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii])).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1[8]*pred1_iframe[ii]))
            counter_IC1_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2[9]*pred2_iframe[ii]))
            counter_IC2_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_iframe += 1
                correct += 1
        elif (softmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii]))).max() > threshold:
            prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3[10]*pred3_iframe[ii]))
            counter_IC3_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_iframe += 1
                correct += 1
        else:
            # prediction = np.argmax((2*pred4_iframe[ii]+1*pred4_r[ii]+1*pred4_mv[ii])/3)
            # prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[0]*pred1_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[1]*pred2_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[2]*pred3_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[4]*pred1_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[5]*pred2_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[6]*pred3_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[8]*pred1_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[9]*pred2_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[10]*pred3_iframe[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[11]*pred4_iframe[ii])/11)
            # prediction = np.argmax((w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[3]*pred4_r[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[7]*pred4_mv[ii]+w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4[11]*pred4_iframe[ii])/3)
            if args.split == 'split1':
                prediction = np.argmax((1.8532*pred4_iframe[ii]+1.3720*pred4_r[ii]+0.5565*pred4_mv[ii]))
            elif args.split == 'split2':
                prediction = np.argmax((1.8532*pred4_iframe[ii]+1.3720*pred4_r[ii]+0.5565*pred4_mv[ii]))
            elif args.split == 'split3':
                prediction = np.argmax((1.5995*pred4_iframe[ii]+1.4348*pred4_r[ii]+0.9558*pred4_mv[ii]))
            counter_IC4_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_iframe += 1
                correct += 1

    cost_IC1_r_flops = 1.09
    cost_IC2_r_flops = 1.09+0.88
    cost_IC3_r_flops = 1.09+0.88+0.52
    cost_IC4_r_flops = 1.09+0.88+0.52+0.41

    cost_IC1_mv_flops = 1.13
    cost_IC2_mv_flops = 1.13+0.87
    cost_IC3_mv_flops = 1.13+0.87+0.53
    cost_IC4_mv_flops = 1.13+0.87+0.53+0.41
    cost_r = 2.9
    cost_mv = 2.94

    cost_IC1_i_flops = 2.97 - 1.71
    cost_IC2_i_flops = 2.97 + 2.88 - 1.71 - 0.92
    cost_IC3_i_flops = 2.97 + 2.88 + 3.32 - 1.71 - 0.92
    cost_IC4_i_flops = 2.97 + 2.88 + 3.32 + 0.81 - 1.71 - 0.92

    cost_flops = (cost_IC1_r_flops *counter_IC1_predictions_r + \
                    cost_IC2_r_flops *counter_IC2_predictions_r + \
                    cost_IC3_r_flops *counter_IC3_predictions_r + \
                    cost_IC4_r_flops *counter_IC4_predictions_r)*(test_segments_r) + \
                    (cost_IC1_mv_flops *counter_IC1_predictions_mv + \
                    cost_IC2_mv_flops *counter_IC2_predictions_mv + \
                    cost_IC3_mv_flops *counter_IC3_predictions_mv + \
                    cost_IC4_mv_flops *counter_IC4_predictions_mv)*(test_segments_mv) + \
                    (cost_r *counter_IC1_predictions_mv + \
                    cost_r *counter_IC2_predictions_mv + \
                    cost_r *counter_IC3_predictions_mv + \
                    cost_r *counter_IC4_predictions_mv)*(test_segments_r) + \
                    (cost_IC1_i_flops *counter_IC1_predictions_iframe + \
                    cost_IC2_i_flops *counter_IC2_predictions_iframe + \
                    cost_IC3_i_flops *counter_IC3_predictions_iframe + \
                    cost_IC4_i_flops *counter_IC4_predictions_iframe)*(test_segments_i) + \
                    (cost_r *counter_IC1_predictions_iframe + \
                    cost_r *counter_IC2_predictions_iframe + \
                    cost_r *counter_IC3_predictions_iframe + \
                    cost_r *counter_IC4_predictions_iframe)*(test_segments_r) + \
                    (cost_mv *counter_IC1_predictions_iframe + \
                    cost_mv *counter_IC2_predictions_iframe + \
                    cost_mv *counter_IC3_predictions_iframe + \
                    cost_mv *counter_IC4_predictions_iframe)*(test_segments_mv) 
    
    print('COST:', cost_flops)
    print("ACC: ", (correct/num_samples_test)*100)
    acc = (correct/num_samples_test)*100

    return acc, cost_flops

def main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4):
    softmax = torch.nn.Softmax(dim=1)
    # threshold = threshold

    # evaluation
    output1_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC1_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC2_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC3_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_mv = torch.load('./logits/ucf101_kinetics_pretrained/IC4_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_r = torch.load('./logits/ucf101_kinetics_pretrained/IC1_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_r = torch.load('./logits/ucf101_kinetics_pretrained/IC2_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_r = torch.load('./logits/ucf101_kinetics_pretrained/IC3_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_r = torch.load('./logits/ucf101_kinetics_pretrained/IC4_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC1_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC2_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC3_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_iframe = torch.load('./logits/ucf101_kinetics_pretrained/IC4_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    num_samples_test = len(output1_mv)
    print('Number of Samples: ', num_samples_test)

    pred1_mv = []
    pred2_mv = []
    pred3_mv = []
    pred4_mv = []

    pred1_r = []
    pred2_r = []
    pred3_r = []
    pred4_r = []

    pred1_iframe = []
    pred2_iframe = []
    pred3_iframe = []
    pred4_iframe = []

    labels = []
    for ii in range(0, num_samples_test):
        pred1_mv.append(output1_mv[ii][0])
        pred2_mv.append(output2_mv[ii][0])
        pred3_mv.append(output3_mv[ii][0])
        pred4_mv.append(output4_mv[ii][0])

        pred1_r.append(output1_r[ii][0])
        pred2_r.append(output2_r[ii][0])
        pred3_r.append(output3_r[ii][0])
        pred4_r.append(output4_r[ii][0])

        pred1_iframe.append(output1_iframe[ii][0])
        pred2_iframe.append(output2_iframe[ii][0])
        pred3_iframe.append(output3_iframe[ii][0])
        pred4_iframe.append(output4_iframe[ii][0])

        labels.append(output1_mv[ii][1])


    w_r1_r2 = w_r1_r2.detach().numpy()
    w_r1_r2_r3 = w_r1_r2_r3.detach().numpy()
    w_r1_r2_r3_r4 = w_r1_r2_r3_r4.detach().numpy()
    w_r1_r2_r3_r4_mv1 = w_r1_r2_r3_r4_mv1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2 = w_r1_r2_r3_r4_mv1_mv2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3 = w_r1_r2_r3_r4_mv1_mv2_mv3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3.detach().numpy()
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4.detach().numpy()

    counter_IC1_predictions_mv = 0
    correct_IC1_predictions_mv = 0

    counter_IC2_predictions_mv = 0
    correct_IC2_predictions_mv = 0

    counter_IC3_predictions_mv = 0
    correct_IC3_predictions_mv = 0

    counter_IC4_predictions_mv = 0
    correct_IC4_predictions_mv = 0

    counter_IC1_predictions_r = 0
    correct_IC1_predictions_r = 0

    counter_IC2_predictions_r = 0
    correct_IC2_predictions_r = 0

    counter_IC3_predictions_r = 0
    correct_IC3_predictions_r = 0

    counter_IC4_predictions_r = 0
    correct_IC4_predictions_r = 0

    counter_IC1_predictions_iframe = 0
    correct_IC1_predictions_iframe = 0

    counter_IC2_predictions_iframe = 0
    correct_IC2_predictions_iframe = 0

    counter_IC3_predictions_iframe = 0
    correct_IC3_predictions_iframe = 0

    counter_IC4_predictions_iframe = 0
    correct_IC4_predictions_iframe = 0

    correct = 0

    for ii in range(0, num_samples_test):
        # residual network
        if softmax(pred1_r[ii]).max() > threshold:
            prediction = np.argmax(pred1_r[ii])
            counter_IC1_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_r += 1
                correct += 1
        elif (softmax((pred2_r[ii]))).max() > threshold:
            prediction = np.argmax((pred2_r[ii]))
            counter_IC2_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_r += 1
                correct += 1
        elif (softmax((pred3_r[ii]))).max() > threshold:
            prediction = np.argmax((pred3_r[ii]))
            counter_IC3_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_r += 1
                correct += 1
        elif (softmax((pred4_r[ii]))).max() > threshold:
            prediction = np.argmax((pred4_r[ii]))
            counter_IC4_predictions_r += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_r += 1
                correct += 1

        # MV network
        elif softmax((pred1_mv[ii])).max() > threshold:
            prediction = np.argmax((pred1_mv[ii]))
            counter_IC1_predictions_mv += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_mv += 1
                correct += 1
        elif (softmax((pred2_mv[ii]))).max() > threshold:
            prediction = np.argmax((pred2_mv[ii]))
            counter_IC2_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC2_predictions_mv += 1
                correct += 1
        elif (softmax((pred3_mv[ii]))).max() > threshold:
            prediction = np.argmax((pred3_mv[ii]))
            counter_IC3_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC3_predictions_mv += 1
                correct += 1
        elif (softmax((pred4_mv[ii]))).max() > threshold:
            prediction = np.argmax((pred4_mv[ii]))
            counter_IC4_predictions_mv += 1
            if prediction == labels[ii]:
                counter_IC4_predictions_mv += 1
                correct += 1

        # iframe network
        elif softmax((pred1_iframe[ii])).max() > threshold:
            prediction = np.argmax((pred1_iframe[ii]))
            counter_IC1_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC1_predictions_iframe += 1
                correct += 1
        elif (softmax((pred2_iframe[ii]))).max() > threshold:
            prediction = np.argmax((pred2_iframe[ii]))
            counter_IC2_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC2_predictions_iframe += 1
                correct += 1
        elif (softmax((pred3_iframe[ii]))).max() > threshold:
            prediction = np.argmax((pred3_iframe[ii]))
            counter_IC3_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC3_predictions_iframe += 1
                correct += 1
        else:
            prediction = np.argmax((pred4_iframe[ii]))
            counter_IC4_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_iframe += 1
                correct += 1


    cost_IC1_r_flops = 1.09
    cost_IC2_r_flops = 1.09+0.88
    cost_IC3_r_flops = 1.09+0.88+0.52
    cost_IC4_r_flops = 1.09+0.88+0.52+0.41

    cost_IC1_mv_flops = 1.13
    cost_IC2_mv_flops = 1.13+0.87
    cost_IC3_mv_flops = 1.13+0.87+0.53
    cost_IC4_mv_flops = 1.13+0.87+0.53+0.41
    cost_r = 2.9
    cost_mv = 2.94

    cost_IC1_i_flops = 2.97 - 1.71
    cost_IC2_i_flops = 2.97 + 2.88 - 1.71 - 0.92
    cost_IC3_i_flops = 2.97 + 2.88 + 3.32 - 1.71 - 0.92
    cost_IC4_i_flops = 2.97 + 2.88 + 3.32 + 0.81 - 1.71 - 0.92

    cost_flops = (cost_IC1_r_flops *counter_IC1_predictions_r + \
                    cost_IC2_r_flops *counter_IC2_predictions_r + \
                    cost_IC3_r_flops *counter_IC3_predictions_r + \
                    cost_IC4_r_flops *counter_IC4_predictions_r)*(test_segments_r) + \
                    (cost_IC1_mv_flops *counter_IC1_predictions_mv + \
                    cost_IC2_mv_flops *counter_IC2_predictions_mv + \
                    cost_IC3_mv_flops *counter_IC3_predictions_mv + \
                    cost_IC4_mv_flops *counter_IC4_predictions_mv)*(test_segments_mv) + \
                    (cost_r *counter_IC1_predictions_mv + \
                    cost_r *counter_IC2_predictions_mv + \
                    cost_r *counter_IC3_predictions_mv + \
                    cost_r *counter_IC4_predictions_mv)*(test_segments_r) + \
                    (cost_IC1_i_flops *counter_IC1_predictions_iframe + \
                    cost_IC2_i_flops *counter_IC2_predictions_iframe + \
                    cost_IC3_i_flops *counter_IC3_predictions_iframe + \
                    cost_IC4_i_flops *counter_IC4_predictions_iframe)*(test_segments_i) + \
                    (cost_r *counter_IC1_predictions_iframe + \
                    cost_r *counter_IC2_predictions_iframe + \
                    cost_r *counter_IC3_predictions_iframe + \
                    cost_r *counter_IC4_predictions_iframe)*(test_segments_r) + \
                    (cost_mv *counter_IC1_predictions_iframe + \
                    cost_mv *counter_IC2_predictions_iframe + \
                    cost_mv *counter_IC3_predictions_iframe + \
                    cost_mv *counter_IC4_predictions_iframe)*(test_segments_mv) 
    
    print(counter_IC1_predictions_iframe,counter_IC2_predictions_iframe,counter_IC3_predictions_iframe,counter_IC4_predictions_iframe)
    print(counter_IC1_predictions_r,counter_IC2_predictions_r,counter_IC3_predictions_r,counter_IC4_predictions_r)
    print(counter_IC1_predictions_mv,counter_IC2_predictions_mv,counter_IC3_predictions_mv,counter_IC4_predictions_mv)

    print('COST:', cost_flops)
    print("ACC: ", (correct/num_samples_test)*100)
    acc = (correct/num_samples_test)*100

    return acc, cost_flops

if __name__ == '__main__':
    # w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = optimize_scaling_coefficients()

    # torch.save(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3_r4_mv1_mv2_mv3_mv4, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3_r4_mv1_mv2_mv3, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3_r4_mv1_mv2, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3_r4_mv1, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3_r4, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2_r3, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_'+str(args.split)+'.pt')
    # torch.save(w_r1_r2, './scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_'+str(args.split)+'.pt')

    
    if args.split == 'split1':
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([0, 0, 0, 1.3787148, 0, 0, 0, 0.54696816, 0, 0, 0, 1.8505168])
    elif args.split == 'split2':
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([0, 0, 0, 1.4578189, 0, 0, 0, 0.708287, 0, 0, 0, 1.683037])
    elif args.split == 'split3':
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([0, 0, 0, 1.4397587, 0, 0, 0, 0.9563417, 0, 0, 0, 1.5926598])

    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_'+str(args.split)+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_'+str(args.split)+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_'+str(args.split)+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_'+str(args.split)+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_'+str(args.split)+'.pt')
    w_r1_r2_r3_r4_mv1_mv2 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_'+str(args.split)+'.pt')
    w_r1_r2_r3_r4_mv1 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_mv1_'+str(args.split)+'.pt')
    w_r1_r2_r3_r4 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_r4_'+str(args.split)+'.pt')
    w_r1_r2_r3 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_r3_'+str(args.split)+'.pt')
    w_r1_r2 = torch.load('./scaling_coefficients/ucf101_kinetics_pretrained/w_r1_r2_'+str(args.split)+'.pt')

 
    split = args.split

    list_cost = []
    list_acc = []

    list_cost_no_division = []
    list_acc_no_division = []

    list_cost_no_scaling = []
    list_acc_no_scaling = []

    list_cost_no_scaling_no_division = []
    list_acc_no_scaling_no_division = []

    list_cost_no_scaling_no_lateral_connections = []
    list_acc_no_scaling_no_lateral_connections = []

    list_cost_no_scaling_CE = []
    list_acc_no_scaling_CE = []


    #########################
    #########################
    # KD no division
    if args.split == 'split1':
        # KD no division
        print("KD no division")
        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        # # KD with division
        # threshold = 1.0
        # test_segments_mv = 3
        # test_segments_r = 5
        # test_segments_i = 3
        # acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost.append(cost_flops)
        # list_acc.append(acc)
        # print('\n')

        # threshold = 0.999995
        # test_segments_mv = 3
        # test_segments_r = 3
        # test_segments_i = 3
        # acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost.append(cost_flops)
        # list_acc.append(acc)
        # print('\n')

        # threshold = 0.999999
        # test_segments_mv = 1
        # test_segments_r = 1
        # test_segments_i = 1
        # acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost.append(cost_flops)
        # list_acc.append(acc)
        # print('\n')

        # threshold = 0.999999
        # test_segments_mv = 1
        # test_segments_r = 2
        # test_segments_i = 1
        # acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost.append(cost_flops)
        # list_acc.append(acc)
        # print('\n')

        # threshold = 0.999999
        # test_segments_mv = 1
        # test_segments_r = 2
        # test_segments_i = 2
        # acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost.append(cost_flops)
        # list_acc.append(acc)
        # print('\n')


        # KD no scaling
        print("KD no scaling")
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        w_r1_r2 = torch.Tensor([1, 1])

        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        # # KD no scaling no division
        # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        # w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        # w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        # w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        # w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        # w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        # w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        # w_r1_r2 = torch.Tensor([1, 1])

        # threshold = 1.0
        # test_segments_mv = 3
        # test_segments_r = 5
        # test_segments_i = 3
        # acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost_no_scaling_no_division.append(cost_flops)
        # list_acc_no_scaling_no_division.append(acc)
        # print('\n')

        # threshold = 0.999995
        # test_segments_mv = 3
        # test_segments_r = 3
        # test_segments_i = 3
        # acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost_no_scaling_no_division.append(cost_flops)
        # list_acc_no_scaling_no_division.append(acc)
        # print('\n')

        # threshold = 0.999999
        # test_segments_mv = 1
        # test_segments_r = 1
        # test_segments_i = 1
        # acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost_no_scaling_no_division.append(cost_flops)
        # list_acc_no_scaling_no_division.append(acc)
        # print('\n')

        # threshold = 0.999999
        # test_segments_mv = 1
        # test_segments_r = 2
        # test_segments_i = 1
        # acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost_no_scaling_no_division.append(cost_flops)
        # list_acc_no_scaling_no_division.append(acc)
        # print('\n')

        # threshold = 0.999999
        # test_segments_mv = 1
        # test_segments_r = 2
        # test_segments_i = 2
        # acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        # list_cost_no_scaling_no_division.append(cost_flops)
        # list_acc_no_scaling_no_division.append(acc)
        # print('\n')

        # KD no scaling no lateral connections
        print('KD no scaling no lateral connections')
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        w_r1_r2 = torch.Tensor([1, 1])

        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')


    elif args.split == 'split2':
        # KD no division
        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        # KD with division
        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')


        # KD no scaling
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        w_r1_r2 = torch.Tensor([1, 1])

        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        # KD no scaling no division
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        w_r1_r2 = torch.Tensor([1, 1])

        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        # KD no scaling no lateral connections
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        w_r1_r2 = torch.Tensor([1, 1])

        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

    elif args.split == 'split3':
        # KD no division
        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_division.append(cost_flops)
        list_acc_no_division.append(acc)
        print('\n')


        # KD with division
        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost.append(cost_flops)
        list_acc.append(acc)
        print('\n')


        # KD no scaling
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        w_r1_r2 = torch.Tensor([1, 1])

        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_scaling(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling.append(cost_flops)
        list_acc_no_scaling.append(acc)
        print('\n')

        # KD no scaling no division
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        w_r1_r2 = torch.Tensor([1, 1])

        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_scaling_no_division(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_division.append(cost_flops)
        list_acc_no_scaling_no_division.append(acc)
        print('\n')

        # KD no scaling no lateral connections
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1_mv2 = torch.Tensor([1, 1, 1, 1, 1, 1])
        w_r1_r2_r3_r4_mv1 = torch.Tensor([1, 1, 1, 1, 1])
        w_r1_r2_r3_r4 = torch.Tensor([1, 1, 1, 1])
        w_r1_r2_r3 = torch.Tensor([1, 1, 1])
        w_r1_r2 = torch.Tensor([1, 1])

        threshold = 1.0
        test_segments_mv = 3
        test_segments_r = 5
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999995
        test_segments_mv = 3
        test_segments_r = 3
        test_segments_i = 3
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.99999995
        test_segments_mv = 1
        test_segments_r = 1
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 1
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

        threshold = 0.999999
        test_segments_mv = 1
        test_segments_r = 2
        test_segments_i = 2
        acc, cost_flops = main_no_scaling_no_lateral_connections(threshold, test_segments_mv, test_segments_r, test_segments_i,split,w_r1_r2,w_r1_r2_r3,w_r1_r2_r3_r4,w_r1_r2_r3_r4_mv1,w_r1_r2_r3_r4_mv1_mv2,w_r1_r2_r3_r4_mv1_mv2_mv3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2,w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3, w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4)
        list_cost_no_scaling_no_lateral_connections.append(cost_flops)
        list_acc_no_scaling_no_lateral_connections.append(acc)
        print('\n')

    ##########################

    # Specify the folder path you want to check and create
    folder_path = './lists/acc_cost/'+str(args.data_name)+'_kinetics_pretrained/'
        
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    # #########################
    # np.save(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101.npy', list_cost)
    # np.save(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101.npy', list_acc)

    # x1 = np.load(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101.npy')
    # y1 = np.load(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101.npy')

    # idx_sorted = np.argsort(x1)

    # plt.plot(x1[idx_sorted], y1[idx_sorted], marker='o', label = 'KD', markersize=8)


    ###############################

    #########################
    np.save(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_division.npy', list_cost_no_division)
    np.save(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_division.npy', list_acc_no_division)

    x4 = np.load(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_division.npy')
    y4 = np.load(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_division.npy')

    if args.split == 'split2':
        idx_sorted = np.argsort(y4)
    else:
        idx_sorted = np.argsort(x4)

    plt.plot(x4[idx_sorted], y4[idx_sorted], marker='o', label = 'KD-no division', markersize=8)

    #########################

    np.save(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_scaling.npy', list_cost_no_scaling)
    np.save(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_scaling.npy', list_acc_no_scaling)

    x2 = np.load(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_scaling.npy')
    y2 = np.load(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_scaling.npy')

    idx_sorted = np.argsort(x2)

    plt.plot(x2[idx_sorted], y2[idx_sorted], marker='o', label = 'KD-no scaling', markersize=8)

    #########################################

    # np.save(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_division.npy', list_cost_no_scaling_no_division)
    # np.save(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_division.npy', list_acc_no_scaling_no_division)

    # x2 = np.load(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_division.npy')
    # y2 = np.load(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_division.npy')

    # idx_sorted = np.argsort(x2)

    # plt.plot(x2[idx_sorted], y2[idx_sorted], marker='o', label = 'KD-no scaling-no division', markersize=8)


    # #########################
    np.save(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy', list_cost_no_scaling_no_lateral_connections)
    np.save(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy', list_acc_no_scaling_no_lateral_connections)

    x3 = np.load(folder_path +'/list_cost_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')
    y3 = np.load(folder_path +'/list_acc_'+str(split)+'_wise_trade_off_ucf101_no_scaling_no_lateral_connections.npy')

    idx_sorted = np.argsort(x3)

    plt.plot(x3[idx_sorted], y3[idx_sorted], marker='o', label = 'KD-no scaling no lateral connections', markersize=8)


    # SOTA WORKS 
    # plt.plot(282743.18, 93.19, 'rD', label = 'CoViAR', markersize=8)
    # plt.plot(62310.88, 85.8, 'rd', label = 'MIMO', markersize=8)
    # plt.plot(323426.74, 88.5, 'rp', label = 'Wu et al.', markersize=8)

    # plt.plot(543903, 93.19, 'rD', label = 'CoViAR', markersize=8)
    # plt.plot(172587, 85.8, 'rd', label = 'MIMO', markersize=8)
    # plt.plot(4717060, 88.5, 'rp', label = 'Wu et al.', markersize=8)

    plt.legend() 
    plt.xlabel('Cost (GMACs)')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.title('UCF-101')
    plt.show()

    # ########################################################################

   