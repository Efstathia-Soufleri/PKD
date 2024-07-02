"""Run testing given a trained model."""

import argparse
import time
import os
import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from dataset import CoviarDataSet
# from model_unrolled_all_IC import Model_unrolled_IC
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale
from model_unrolled_all_IC import model_unrolled_all_IC

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
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

def main(test_segments_mv, test_segments_r, test_segments_i, split, threshold):

    output1_mv = torch.load('./logits/hmdb51_kinetics_pretrained/IC1_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_mv = torch.load('./logits/hmdb51_kinetics_pretrained/IC2_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_mv = torch.load('./logits/hmdb51_kinetics_pretrained/IC3_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_mv = torch.load('./logits/hmdb51_kinetics_pretrained/IC4_mv_frames_'+str(test_segments_mv)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_r = torch.load('./logits/hmdb51_kinetics_pretrained/IC1_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_r = torch.load('./logits/hmdb51_kinetics_pretrained/IC2_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_r = torch.load('./logits/hmdb51_kinetics_pretrained/IC3_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_r = torch.load('./logits/hmdb51_kinetics_pretrained/IC4_residual_frames_'+str(test_segments_r)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')

    output1_iframe = torch.load('./logits/hmdb51_kinetics_pretrained/IC1_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output2_iframe = torch.load('./logits/hmdb51_kinetics_pretrained/IC2_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output3_iframe = torch.load('./logits/hmdb51_kinetics_pretrained/IC3_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')
    output4_iframe = torch.load('./logits/hmdb51_kinetics_pretrained/IC4_iframe_frames_'+str(test_segments_i)+'_'+str(args.learning_type)+'_'+str(args.split)+'.pt')


    num_samples = len(output1_mv)
    print('Number of Samples: ', num_samples)
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
    for ii in range(0, num_samples):
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

    # threshold = 0.999999 #0.999999 # 0.9999

    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_i4_'+args.split+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_i3_'+args.split+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_i2_'+args.split+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_i1_'+args.split+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3_mv4 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3_mv4_'+args.split+'.pt')
    w_r1_r2_r3_r4_mv1_mv2_mv3 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_mv3'+args.split+'.pt')
    w_r1_r2_r3_r4_mv1_mv2 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_mv1_mv2_'+args.split+'.pt')
    w_r1_r2_r3_r4_mv1 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_mv1_'+args.split+'.pt')
    w_r1_r2_r3_r4 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_r4_'+args.split+'.pt')
    w_r1_r2_r3 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_r3_'+args.split+'.pt')
    w_r1_r2 = torch.load('./scaling_coefficients/hmdb51_kinetics_pretrained/w_r1_r2_'+args.split+'.pt')

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

    ##############################################


    softmax = torch.nn.Softmax(dim=1)
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
    
    for ii in range(0, num_samples):
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
            prediction = np.argmax((2*pred4_iframe[ii]+1*pred4_r[ii]+1*pred4_mv[ii])/3)
            counter_IC4_predictions_iframe += 1
            if prediction == labels[ii]:
                correct_IC4_predictions_iframe += 1
                correct += 1

    print("Accuracy: ", (correct/num_samples)*100)

    cost_IC1_r_flops = 1.09
    cost_IC2_r_flops = 1.09+0.88
    cost_IC3_r_flops = 1.09+0.88+0.52
    cost_IC4_r_flops = 1.09+0.88+0.52+0.41

    cost_IC1_mv_flops = 2.9+1.13
    cost_IC2_mv_flops = 2.9+1.13+0.87
    cost_IC3_mv_flops = 2.9+1.13+0.87+0.53
    cost_IC4_mv_flops = 2.9+1.13+0.87+0.53+0.41

    cost_r = 2.9
    cost_mv = 2.94

    # cost_IC1_i_flops = 2.9+2.94+2.65
    # cost_IC2_i_flops = 2.9+2.94+2.65+3.77
    # cost_IC3_i_flops = 2.9+2.94+2.65+3.77+9.91
    # cost_IC4_i_flops = 2.9+2.94+2.65+3.77+9.91+0.81

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
    

    print('Cost for the proposal:', cost_flops)
    return (correct/num_samples)*100, cost_flops

if __name__ == '__main__':
    split = args.split

    # Specify the folder path you want to check and create
    folder_path = './lists/ablation_study_frame_number/'+args.data_name+'_kinetics_pretrained'

    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


    print("MV")
    ######## MV ####################
    list_acc = []
    list_cost = []
    test_segments_r = 3
    test_segments_i = 3
    threshold = 0.9999 # 0.99999
    frame_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]
    for test_segments_mv in frame_list:
        print("Test Segments:", test_segments_mv)
        acc, cost = main(test_segments_mv, test_segments_r, test_segments_i, split, threshold)
        list_acc.append(acc)
        list_cost.append(cost)
        print('\n')

    np.save(folder_path+'/list_acc_mv_'+str(args.split)+'_hmdb51.npy', list_acc)
    np.save(folder_path+'/list_cost_mv_'+str(args.split)+'_hmdb51.npy', list_cost)
    np.save(folder_path+'/frame_list_mv_'+str(args.split)+'_hmdb51.npy', frame_list)

    print("RESIDUAL")
    ######## RESIDUAL ####################
    list_acc = []
    list_cost = []
    test_segments_i = 3
    test_segments_mv = 3
    threshold = 0.9999 # 999999#999 # 0.99999
    frame_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]
    for test_segments_r in frame_list:
        print("Test Segments:", test_segments_r)
        acc, cost = main(test_segments_mv, test_segments_r, test_segments_i, split, threshold)
        list_acc.append(acc)
        list_cost.append(cost)
        print('\n')

    np.save(folder_path+'/list_acc_residual_'+str(args.split)+'_hmdb51.npy', list_acc)
    np.save(folder_path+'/list_cost_residual_'+str(args.split)+'_hmdb51.npy', list_cost)
    np.save(folder_path+'/frame_list_residual_'+str(args.split)+'_hmdb51.npy', frame_list)

    print("IFRAME")
    ######## IFRAME ####################
    list_acc = []
    list_cost = []
    test_segments_r = 3
    test_segments_mv = 3
    threshold = 0.9999
    frame_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20 ]
    for test_segments_i in frame_list:
        print("Test Segments:", test_segments_i)
        acc, cost = main(test_segments_mv, test_segments_r, test_segments_i, split, threshold)
        list_acc.append(acc)
        list_cost.append(cost)
        print('\n')

    np.save(folder_path+'/list_acc_iframe_'+str(args.split)+'_hmdb51.npy', list_acc)
    np.save(folder_path+'/list_cost_iframe_'+str(args.split)+'_hmdb51.npy', list_cost)
    np.save(folder_path+'/frame_list_iframe_'+str(args.split)+'_hmdb51.npy', frame_list)

    #############################################################################################
    #############################################################################################

    # # plot MV
    # y1 = np.load('list_acc_mv_'+str(args.split)+'_v2_hmdb51.npy')
    # x1 = np.load('list_cost_mv_'+str(args.split)+'_v2_hmdb51.npy')
    # plt.plot(x1, y1, marker='o', label = 'mv')

    # for i in range(len(x1)): 
    #     plt.annotate(frame_list[i], (x1[i], y1[i]), xytext =(x1[i]-0.05, y1[i]-0.05)) 

    # plt.legend() 
    # plt.xlabel('Cost (GMACs)')
    # plt.ylabel('Accuracy (%)')
    # plt.grid()
    # plt.title('HMDB-51')
    # plt.show()

    # # plot RESIDUAL
    # y2 = np.load('list_acc_residual_'+str(args.split)+'_v2_hmdb51.npy')
    # x2 = np.load('list_cost_residual_'+str(args.split)+'_v2_hmdb51.npy')
    # plt.plot(x2, y2, marker='o', label = 'residual')

    # for i in range(len(x2)): 
    #     plt.annotate(frame_list[i], (x2[i], y2[i]), xytext =(x2[i]-0.05, y2[i]-0.05)) 

    # plt.legend() 
    # plt.xlabel('Cost (GMACs)')
    # plt.ylabel('Accuracy (%)')
    # plt.grid()
    # plt.title('HMDB-51')
    # plt.show()

    # # plot IFRAME
    # y3 = np.load('list_acc_iframe_'+str(args.split)+'_v2_hmdb51.npy')
    # x3 = np.load('list_cost_iframe_'+str(args.split)+'_v2_hmdb51.npy')

    # plt.plot(x3, y3, marker='o', label = 'iframe')

    # for i in range(len(x3)): 
    #     plt.annotate(frame_list[i], (x3[i], y3[i]), xytext =(x3[i]-0.05, y3[i]-0.05)) 

    # plt.legend() 
    # plt.xlabel('Cost (GMACs)')
    # plt.ylabel('Accuracy (%)')
    # plt.grid()
    # plt.title('HMDB-51')
    # plt.show()
