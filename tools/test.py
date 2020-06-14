# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim

from ..data.dataset import VideoDataset
from ..model.models import TSN
from ..data.transforms import *

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics','mini_kinetics'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--new_length', type=int, default=4)
parser.add_argument('--fast_implementation', type=int, default=1)
parser.add_argument('--slow_testing', type=int, default=0)
parser.add_argument('--train_segments', type=int, default=4)
parser.add_argument('--test_segments', type=int, default=10)
parser.add_argument('--keep_ratio', type=int, default=0)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=3)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
elif args.dataset == 'mini_kinetics':
    num_class = 200
else:
    raise ValueError('Unknown dataset '+args.dataset)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    #print("calculate one accuracy")

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


if args.slow_testing:
    num_segments=args.train_segments
else:
    num_segments=args.test_segments
net = TSN(num_class, num_segments, args.modality,new_length=args.new_length,test_mode=True,slow_testing=args.slow_testing,fast_implementation=args.fast_implementation,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,apply_softmax=True,
          dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 3:
    cropping = torchvision.transforms.Compose([
        GroupFCNSample(256) if args.keep_ratio else GroupFCNSample_0(256)  
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        VideoDataset("", args.test_list, num_segments=args.test_segments,
                   new_length=args.new_length if args.modality == "RGB" else 5,new_step=8,
                   modality=args.modality,
                   image_tmpl="{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,slow_testing=args.slow_testing,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=9, pin_memory=True)

if not args.slow_testing:
    devices = [0]
else:
    devices = list(range(6))

net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []
top1 = AverageMeter()
top5 = AverageMeter()
batch_time = AverageMeter()
end = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
img_size=256
with torch.no_grad():
    for i, (input, target) in data_gen:
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.view((-1,3,args.new_length)+ input.size()[-2:]),
                                            volatile=True)

        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = net(input_var)
        print(output.shape)
        output = output.mean(dim=0, keepdim=True)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        #losses.update(loss.data[0], input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 1 == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   i, total_num, batch_time=batch_time,
                   top1=top1)))


    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1,top5=top5)))
