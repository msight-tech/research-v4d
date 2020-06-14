# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
from torch.nn import Conv2d

def conv4d(data,filters,padding,bias=None,permute_filters=True,use_half=False):
    b,c,h,w,d,t=data.size()

    data=data.permute(2,0,1,3,4,5).contiguous() # permute to avoid making contiguous inside loop    
        
    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        filters=filters.permute(2,0,1,3,4,5).contiguous() # permute to avoid making contiguous inside loop    

    padding_0=padding[0]
    dimension_4th=h+2*padding_0-filters.shape[0]+1
    #print dimension_4th
    c_out=filters.size(1)
    if use_half:
        output = Variable(torch.HalfTensor(dimension_4th,b,c_out,w,d,t),requires_grad=data.requires_grad)
    else:
        output = Variable(torch.zeros(dimension_4th,b,c_out,w,d,t),requires_grad=data.requires_grad)
    
    
    if use_half:
        Z=Variable(torch.zeros(padding_0,b,c,w,d,t).half())
    else:
        Z=Variable(torch.zeros(padding_0,b,c,w,d,t))
    
    if data.is_cuda:
        Z=Z.cuda(data.get_device())    
        output=output.cuda(data.get_device())
        
    data_padded = torch.cat((Z,data,Z),0)
 
    for i in range(output.size(0)): # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i,:,:,:,:,:]=F.conv3d(data_padded[i+padding_0,:,:,:,:,:], 
                                     filters[padding_0,:,:,:,:,:], bias=bias, stride=1, padding=padding[1:])
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1,padding_0+1):
            output[i,:,:,:,:,:]=output[i,:,:,:,:,:]+F.conv3d(data_padded[i+padding_0-p,:,:,:,:,:], 
                                                             filters[padding_0-p,:,:,:,:,:], bias=None, stride=1, padding=padding[1:])
            output[i,:,:,:,:,:]=output[i,:,:,:,:,:]+F.conv3d(data_padded[i+padding_0+p,:,:,:,:,:], 
                                                             filters[padding_0+p,:,:,:,:,:], bias=None, stride=1, padding=padding[1:])

    output=output.permute(1,2,0,3,4,5).contiguous()
    return output

class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size,padding, bias=True, pre_permuted_filters=True): 
        # stride, dilation and groups !=1 functionality not tested 
        stride=1
        dilation=1
        groups=1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding_for_convnd = 0
        self.padding_for_4d=padding

        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        super(Conv4d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding_for_convnd, dilation,
            False, _quadruple(0), groups, bias,'zeros')  
        
        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor, 
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters=pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data=self.weight.data.permute(2,0,1,3,4,5).contiguous()
        self.use_half=False


    def forward(self, input):
        return conv4d(input, self.weight,self.padding_for_4d, bias=self.bias,permute_filters=not self.pre_permuted_filters,use_half=self.use_half) # filters pre-permuted in constructor

if __name__ == "__main__":
    video=torch.randn(11,1024,3,8,14,14).cuda()
    net=Conv4d(in_channels=1024,out_channels=1024,kernel_size=(3,1,1,1),padding=(1,0,0,0),bias=False)
    net=net.cuda()
    video_4d=net(video)