# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import math

import torch
from torch.nn import ReplicationPad3d

import inflate

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from conv4d import Conv4d


class I3ResNet_18_34(torch.nn.Module):
    def __init__(self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False,num_segments=1,gtsn=False):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet_18_34, self).__init__()
        self.num_segments=num_segments
        self.conv_class = conv_class
        self.gtsn=gtsn

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=1, time_padding=0, center=False)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=1, time_padding=0, time_stride=1)


        self.layer1 = inflate_reslayer_18_34(resnet2d.layer1)
        self.layer2 = inflate_reslayer_18_34(resnet2d.layer2,num_R4D=3,in_channels=128)

        self.layer3 = inflate_reslayer_18_34(resnet2d.layer3,time_dim=3,time_padding=1,num_R4D=3,in_channels=256)
        self.layer4 = inflate_reslayer_18_34(resnet2d.layer4,time_dim=3,time_padding=1)

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True)
        else:
            final_time_dim = int(math.ceil(frame_nb))

            self.avgpool = inflate.inflate_pool(
                resnet2d.avgpool, time_dim=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        main_stream = self.layer3(x)

        if False:#3d convolution for comparison

            x=main_stream.view(-1,self.num_segments,256,4,14,14)
            x=x.permute(0, 2, 1, 3, 4,5).contiguous()
            x=x.view(-1,256,self.num_segments*4,14,14)
            x=self.tsa_cmp(x)
            x=x.view(-1,256,self.num_segments,4,14,14)
            x=x.permute(0, 2, 1, 3, 4,5).contiguous()
            x_3d=x.view(-1,256,4,14,14)
            main_stream=x_3d+main_stream
            
        if False:#4d convolution

            x=main_stream.view(-1,self.num_segments,256,4,14,14)
            x=x.permute(0, 2, 1, 3, 4,5).contiguous()
            x=self.tsa(x)
            x=x.view(-1,256,self.num_segments,4,14,14)
            x=x.permute(0, 2, 1, 3, 4,5).contiguous()
            x_4d=x.view(-1,256,4,14,14)
            main_stream=x_4d+main_stream
      
        x = self.layer4(main_stream)
      
        if self.conv_class:
            x = self.avgpool(x)
            x = self.classifier(x)
            x = x.squeeze(3)
            x = x.squeeze(3)
            x = x.mean(2)
        else:
            x = self.avgpool(x)
            x=F.dropout(x,p=0.5)
            x_reshape = x.view(x.size(0), -1)
        return x_reshape
        
class I3ResNet(torch.nn.Module):
    def __init__(self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False,num_segments=4,test_mode=False,fast_implementation=0):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.conv_class = conv_class
        self.num_segments=num_segments
        self.frame=frame_nb

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=1, time_padding=0, center=False)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=1, time_padding=0, time_stride=1)

        
        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2,num_R4D=2,in_channels=512,fast_implementation=fast_implementation,num_segments=num_segments)
        self.layer3 = inflate_reslayer(resnet2d.layer3,time_dim=3,time_padding=1)
        self.layer4 = inflate_reslayer(resnet2d.layer4,time_dim=3,time_padding=1)

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True)
        else:
            final_time_dim = int(math.ceil(frame_nb ))
            if test_mode:
                self.avgpool=nn.AvgPool3d((frame_nb,8,8))
            else:
                self.avgpool=nn.AvgPool3d((frame_nb,7,7))
            

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)    
        x = self.layer2(x)
        
        # This is where v4d inference goes in the paper. However, to handle for practical memory usage limitations, we move the combination strategy implementation to the data loading         # process to approximate the real v4d inference.
        main_stream = self.layer3(x)
                
        x = self.layer4(main_stream)
        
        if self.conv_class:
            x = self.avgpool(x)
            x = self.classifier(x)
            x = x.squeeze(3)
            x = x.squeeze(3)
            x = x.mean(2)
        else:
            x = self.avgpool(x)
            
            x=F.dropout(x,p=0.5)
            
            x_reshape = x.view(x.size(0), -1)
            
        return x_reshape


def inflate_reslayer(reslayer2d,in_channels=1024,time_dim=1,time_padding=0,nonlocal_channel=0,num_R4D=0,frame=0,num_segments=0,fast_implementation=0):
    reslayers3d = []
    for ilayer,layer2d in enumerate(reslayer2d):
        if frame>0 and ilayer%2==1:
            layer3d = Bottleneck3d(layer2d,time_dim=time_dim,time_padding=time_padding,frame=frame,num_segments=num_segments)
        else:
            layer3d = Bottleneck3d(layer2d,time_dim=time_dim,time_padding=time_padding)
        reslayers3d.append(layer3d)
        if num_R4D>0 and ilayer%2==1:
            layer_R4D=R4D(in_channels,fast_implementation=fast_implementation,num_segments=num_segments)
            reslayers3d.append(layer_R4D)
        if nonlocal_channel>0 and ilayer%2==1:
            layer_nonlocal=NL3d(nonlocal_channel,int(nonlocal_channel/2))
            reslayers3d.append(layer_nonlocal)
    
    return torch.nn.Sequential(*reslayers3d)

def inflate_reslayer_18_34(reslayer2d,in_channels=1024,time_dim=1,time_padding=0,nonlocal_channel=0,num_R4D=0,max_pool_aggr=0,jojo_pool_aggr=0):
    reslayers3d = []
    for ilayer,layer2d in enumerate(reslayer2d):
        layer3d = BasicBlock3d(layer2d,time_dim=time_dim,time_padding=time_padding)
        reslayers3d.append(layer3d)
        if num_R4D>0 and ilayer%2==1:
            layer_R4D=R4D(in_channels)
            reslayers3d.append(layer_R4D)
        if nonlocal_channel>0 and ilayer%2==1:
            layer_nonlocal=NL3d(nonlocal_channel,int(nonlocal_channel/2))
            reslayers3d.append(layer_nonlocal)
        if max_pool_aggr>0 and False:
            layer_max_pool_aggr=nn.MaxPool3d(kernel_size=3,stride=1,padding=1)
            reslayers3d.append(layer_max_pool_aggr)

    if max_pool_aggr>0:
        layer_max_pool_aggr=nn.MaxPool3d(kernel_size=(3,1,1),stride=1,padding=(1,0,0))
        reslayers3d.append(layer_max_pool_aggr)
    if jojo_pool_aggr>0:
        layer_jojo_pool_aggr=JoJoPool_fast()
        reslayers3d.append(layer_jojo_pool_aggr)
    return torch.nn.Sequential(*reslayers3d)

class BasicBlock3d(torch.nn.Module):
    def __init__(self, bottleneck2d,time_dim=1,time_padding=0,frame=0,num_segments=4):
        super(BasicBlock3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]
        self.frame=frame
        self.num_segments=num_segments

        self.conv1 = inflate.inflate_conv(
            bottleneck2d.conv1, time_dim=time_dim,time_padding=time_padding, center=False)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=time_dim,
            time_padding=time_padding,
            time_stride=1,
            center=False)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class R4D(torch.nn.Module):
    def __init__(self, in_channels=1024,num_segments=4,time_dim=1,time_padding=0,fast_implementation=0):
        super(R4D, self).__init__()
        self.num_segments=num_segments
        self.in_channels=in_channels
        self.fast_implementation=fast_implementation
        if fast_implementation:#fast_implementation with 3D conv
          self.tsa=nn.Conv3d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=(3,3,1),padding=(1,1,0),bias=False)
        else:#4D conv is not supported by CUDA
          self.tsa=Conv4d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=(3,3,1,1),padding=(1,1,0,0),bias=False)

    def forward(self, x):
        main_stream = x
        b,c,t,h,w=x.shape
        x=main_stream.view(-1,self.num_segments,c,t,h,w)
        x=x.permute(0, 2, 1, 3, 4,5).contiguous()

        if self.fast_implementation:
          x=x.view(-1,c,self.num_segments,t,h*w)

        x=self.tsa(x)
        x=x.view(-1,c,self.num_segments,t,h,w)

        x=x.permute(0, 2, 1, 3, 4,5).contiguous()
        x_4d=x.view(-1,c,t,h,w)
        
        main_stream=x_4d+main_stream
        return main_stream

class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d,time_dim=1,time_padding=0,frame=0,num_segments=4):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]
        self.frame=frame
        self.num_segments=num_segments

        self.conv1 = inflate.inflate_conv(
            bottleneck2d.conv1, time_dim=time_dim,time_padding=time_padding, center=False)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=1,
            time_padding=0,
            time_stride=1,
            center=False)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(
            bottleneck2d.conv3, time_dim=1, center=False)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.frame>0:
            b,c,t,h,w=x.shape
            x_=x.view(-1,self.num_segments,c,t,h,w)
            x_=x_.permute(0, 2, 1, 3, 4,5).contiguous()
            x_=x_.view(-1,c,self.num_segments,t,h*w)
            x_=self.conv1(x_)
            x_=x_.view(-1,int(c/4),self.num_segments,t,h,w)
            x_=x_.permute(0, 2, 1, 3, 4,5).contiguous()
            x_=x_.view(-1,int(c/4),t,h,w)
            out=out+x_
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=1, center=False),
        inflate.inflate_batch_norm(downsample2d[1]))
    return downsample3d


