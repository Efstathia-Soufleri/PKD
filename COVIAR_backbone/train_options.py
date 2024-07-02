"""Training options."""

import argparse
from str2bool import str2bool

parser = argparse.ArgumentParser(description="CoViAR")

# Seed.
parser.add_argument('--random_seed', default=0, type=int,
                    help='seed.')
# Data.
parser.add_argument('--data-name', type=str, choices=['ucf101', 'hmdb51'],
                    help='dataset name.')
parser.add_argument('--data-root', type=str,
                    help='root of data directory.')
parser.add_argument('--train-list', type=str,
                    help='training example list.')
parser.add_argument('--test-list', type=str,
                    help='testing example list.')
parser.add_argument('--split', type=str, choices=['split1', 'split2', 'split3'],
                    help='split.')

# Model.
parser.add_argument('--representation', type=str, choices=['iframe', 'mv', 'residual'],
                    help='data representation.')
parser.add_argument('--arch', type=str, default="resnet152",
                    help='base architecture.')
parser.add_argument('--num_segments', type=int, default=3,
                    help='number of TSN segments.')
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')

# Training.
parser.add_argument('--epochs', default=500, type=int,
                    help='number of training epochs.')
parser.add_argument('--batch-size', default=40, type=int,
                    help='batch size.')
parser.add_argument('--lr', default=0.001, type=float,
                    help='base learning rate.')
parser.add_argument('--lr-steps', default=[200, 300, 400], type=float, nargs="+",
                    help='epochs to decay learning rate.')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    help='lr decay factor.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay.')
parser.add_argument('--alpha', default=0, type=float,
                    help='coefficient for KD loss.')
parser.add_argument('--beta', default=0, type=float,
                    help='coefficient for beta loss.')
parser.add_argument('--gamma', default=0, type=float,
                    help='coefficient for gamma loss.')
parser.add_argument('--T', default=1, type=float,
                    help='T.')
parser.add_argument('--optimizer', type=str,
                    help='optimizer.')
# Log.
parser.add_argument('--eval-freq', default=5, type=int,
                    help='evaluation frequency (epochs).')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loader workers.')
parser.add_argument('--model-prefix', type=str, default="model",
                    help="prefix of model name.")
parser.add_argument('--gpus', nargs='+', type=int, default=None,
                    help='gpu ids.')
