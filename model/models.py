# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from i3res import I3ResNet,I3ResNet_18_34
from conv4d import Conv4d


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,test_mode=False,slow_testing=0,fast_implementation=1,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', apply_softmax=False,gtsn=False,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.test_mode=test_mode
        self.fast_implementation=fast_implementation
        self.slow_testing=slow_testing
        self.num_segments = num_segments
        self.gtsn=gtsn
        self.reshape = True
        self.apply_softmax = apply_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        if self.test_mode:
            self.input_size=256
        else:
            self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        if not apply_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 4 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))


        #self._prepare_base_model(base_model)

        #feature_dim = self._prepare_tsn(num_class)
        if base_model=='i3dresnet18':
            resnet18 = torchvision.models.resnet18(pretrained=True)

            self.base_model= I3ResNet_18_34(resnet18, self.new_length,num_class,num_segments=self.num_segments,gtsn=self.gtsn)
            if self.gtsn:
                self.new_fc = nn.Linear(512*self.num_segments, num_class)
            else:
                self.new_fc = nn.Linear(512, num_class)
        else:
            resnet50 = torchvision.models.resnet50(pretrained=True)

            self.base_model= I3ResNet(resnet50, self.new_length,num_class,test_mode=test_mode,num_segments=self.num_segments,fast_implementation=fast_implementation)
            self.new_fc = nn.Linear(2048, num_class)
        '''
        resnet18 = torchvision.models.resnet18(pretrained=True)

        self.base_model= I3ResNet_18_34(resnet18, 4,num_class)#I2ResNet_maxpool(resnet50, 8,num_class)
        '''
        #self.base_model=r3d_34()
        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)
        #self.new_fc = nn.Linear(512, num_class)
        std = 0.001
        normal(self.new_fc.weight, 0, std)
        constant(self.new_fc.bias, 0)

        if  self.apply_softmax:
            self.softmax = nn.Softmax()


        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable



    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d) or isinstance(m,Conv4d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},

            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},

            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]



    def forward(self, input):
        #print "input.shape=",input.shape
        #sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        if not self.slow_testing:#self.test_mode:
            input=input.view((-1,self.new_length,3)+ input.size()[-2:])
            input=input.permute(0,2,1,3,4).contiguous()
        else:
            input=input.view((-1,3,self.new_length)+ input.size()[-2:])
        
        base_out = self.base_model(input)
        #base_out=F.dropout(base_out,p=0.5)
        if self.gtsn:
            base_out=base_out.view(base_out.shape[0]/self.num_segments,-1)



        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        
        
        if not self.gtsn:
            if self.reshape:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

            #print "base out ",base_out.shape
            base_out = self.consensus(base_out).squeeze(1)
            if  self.slow_testing:#apply_softmax:
                base_out = self.softmax(base_out)
            return base_out#.squeeze(1)


        else:
            return base_out
        

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return 256#self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66],fix_crop=True, more_fix_crop=True),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
