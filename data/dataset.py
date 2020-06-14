# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from PIL import Image
import torch
import random

try:
    import decord
except ImportError:
    pass


class RawFramesRecord(object):

    def __init__(self, row):
        self._data = row
        self.num_frames = -1

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class VideoDataset(Dataset):

    def __init__(self,
                 img_prefix,
                 ann_file,
                 transform=None,
                 #img_norm_cfg,
                 num_segments=3,
                 new_length=1,
                 new_step=1,
                 slow_testing=0,
                 random_shift=True,
                 temporal_jitter=False,
                 modality='RGB',
                 image_tmpl='img_{}.jpg',
                 
                 
                 
                 test_mode=False,
                 
                 use_decord=True,
                 video_ext='mp4'):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations
        self.video_infos = self.load_annotations(ann_file)
        self.transform=transform
        self.slow_testing=slow_testing
        # normalization config
        #self.img_norm_cfg = img_norm_cfg

        # parameters for frame fetching
        # number of segments
        self.num_segments = num_segments
        # number of consecutive frames
        self.old_length = new_length * new_step
        self.new_length = new_length
        # number of steps (sparse sampling for efficiency of io)
        self.new_step = new_step
        # whether to temporally random shift when training
        self.random_shift = random_shift
        

        # parameters for modalities
        if isinstance(modality, (list, tuple)):
            self.modalities = modality
            num_modality = len(modality)
        else:
            self.modalities = [modality]
            num_modality = 1
        if isinstance(image_tmpl, (list, tuple)):
            self.image_tmpls = image_tmpl
        else:
            self.image_tmpls = [image_tmpl]
        assert len(self.image_tmpls) == num_modality

        
        

        # test mode or not
        self.test_mode = test_mode

        

        
        

        self.use_decord = use_decord
        self.video_ext = video_ext

    def __len__(self):
        return len(self.video_infos)

    def load_annotations(self, ann_file):
        return [RawFramesRecord(x.strip().split(' ')) for x in open(ann_file)]
        # return mmcv.load(ann_file)

    
    def _load_image(self, video_reader, directory, modality, idx):
        if modality in ['RGB', 'RGBDiff']:
            return [video_reader[idx - 1]]
        elif modality == 'Flow':
            raise NotImplementedError
        else:
            raise ValueError('Not implemented yet; modality should be '
                             '["RGB", "RGBDiff", "Flow"]')

    def _sample_indices(self, record):
        '''

        :param record: VideoRawFramesRecord
        :return: list, list
        '''
        average_duration = (record.num_frames - self.old_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments)
        elif record.num_frames > max(self.num_segments, self.old_length):
            offsets = np.sort(
                np.random.randint(
                    record.num_frames - self.old_length + 1,
                    size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))
        
        return offsets, None  

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / \
                float(self.num_segments)
            offsets = np.array(
                [int( tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments, ))
        
        return offsets, None

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.old_length - 1:
            tick = (record.num_frames -  1) / float(self.num_segments)

            offsets = np.array([int( tick * x) for x in range(self.num_segments)])
            #tick = (record.num_frames - self.duration_length + 1) / float(self.num_segments)
            #offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets,None
        
    
    def _get_frames(self, record, video_reader,indices,step_frames=8):

        images = list()
        id_list=[]
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                #frame=video_reader[p]
                #print(frame.shape)
                #print(frame)
                id_list.append(p)                                         
                seg_imgs = [video_reader[p].asnumpy()]#self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames-self.new_step:
                    p += self.new_step#step_frames

        #images=images[::-1]
        #process_data = self.transform(images)
        #print(id_list)
        return images
        #print "process_data.shape=",process_data.shape

        return process_data, record.label
    
    def __getitem__(self, idx):
        record = self.video_infos[idx]
        if self.use_decord:
            video_reader = decord.VideoReader('{}'.format(
                osp.join(self.img_prefix, record.path)))
            record.num_frames = len(video_reader)
            index=idx
            while len(video_reader)<=0:
                index-=1
                if index<0:
                    index=len(video_infos)-1
                record = self.video_infos[index]
                video_reader = decord.VideoReader('{}'.format(
                    osp.join(self.img_prefix, record.path)))
                record.num_frames = len(video_reader)
        else:
            print("using mmcv")
            '''
            video_reader = mmcv.VideoReader('{}.{}'.format(
                osp.join(self.img_prefix, record.path), self.video_ext))
            record.num_frames = len(video_reader)
            '''

        
        if self.test_mode:
            segment_indices, skip_offsets = self._get_val_indices(record)
        else:
            segment_indices, skip_offsets = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)

        '''
        data = dict(
            num_modalities=DC(to_tensor(len(self.modalities))),
            gt_label=DC(to_tensor(record.label), stack=True, pad_dims=None))

        '''
        # handle the first modality
        modality = self.modalities[0]
        image_tmpl = self.image_tmpls[0]
        img_group = self._get_frames(record, video_reader,segment_indices)

        img_group=[Image.fromarray(img) for img in img_group]
        
        img_group=self.transform(img_group)
        if self.test_mode and self.slow_testing:
            img_group=img_group.view((-1,self.new_length,3)+ img_group.size()[-2:])
            img_group=img_group.permute(0,2,1,3,4).contiguous()
            img_group=img_group.view((-1,self.num_segments,3,self.new_length)+ img_group.size()[-2:])
            global_connect_index=[]
            global_connect_representation=[]
            
            
            
            for i0 in range(int(self.num_segments/4)):
                for i1 in range(int(self.num_segments/4),int(self.num_segments/2)):
                    for i2 in range(int(self.num_segments/2),int(3*self.num_segments/4)+1):
                        for i3 in range(int(3*self.num_segments/4)+1,int(self.num_segments)):
                            global_connect_index.append((i0,i1,i2,i3))
            
            
            
            for seg_id in global_connect_index:
                video_group=torch.stack((img_group[:,seg_id[0],:],img_group[:,seg_id[1],:],
                    img_group[:,seg_id[2],:],img_group[:,seg_id[3],:]),dim=1)
                #video_group: [#crop,U_train,C,T,H,W]
                global_connect_representation.append(video_group)
            
            
            img_group=torch.stack(global_connect_representation,dim=1)#[#crop,U_combined,U_train,C,T,H,W]
            img_group=img_group.view((-1,3,self.new_length)+ img_group.size()[-2:])
            #print(img_group.shape)
            #img_group=img_group.permute(0,1,3,2,4,5,6).contiguous()
            #img_group=img_group.view((-1,3,4*self.new_length)+ img_group.size()[-2:])#[#crop,U_combined,C,U_train*T,H,W]
        
        return img_group, record.label        

