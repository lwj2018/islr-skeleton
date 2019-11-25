import torch.utils.data as data

from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import os
import os.path as osp

class SkeletonRecord(object):
    def __init__(self,row):
        self._data = row

    @property
    def filename(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class skeleton_Dataset(data.Dataset):
    
    def __init__(self, skeleton_root, list_file,
                transform=None,
                sample_length=32,
                skeleton_index=[5,6,7,9,10,11,21,22,23,24]):
        self.skeleton_root = skeleton_root
        self.list_file = list_file
        self.transform = transform
        self.sample_length=sample_length
        self.skeleton_index = skeleton_index
        self._prepare_list()
        
    def _load_data(self, filename):
        filename = osp.join(self.skeleton_root, filename)
        f = open(filename,"r")
        content = f.readlines()
        mat = self.content_to_mat(content)
        return mat

    def _prepare_list(self):
        tmp = [x.strip().split('\t') for x in open(self.list_file)]
        self.skeleton_list = [SkeletonRecord(item) for item in tmp]
        print('skeleton number:%d'%(len(self.skeleton_list)))

    def get_sample_indices(self,num_frames):
        indices = np.linspace(0,num_frames-1,self.sample_length).astype(int)
        return indices

    def content_to_mat(self,content):
        mat = []
        for record in content:
            try:
                skeleton = record.rstrip("\n").rstrip(" ").split(" ")
                skeleton = [int(x) for x in skeleton]
                skeleton = np.array(skeleton)
                shape = skeleton.size
                skeleton = np.reshape(skeleton,[shape//2,2])
                skeleton = skeleton[self.skeleton_index]
            except:
                pass
                # print(skeleton.shape)
                # print(skeleton)
            mat.append(skeleton)
        # mat: T,N,D
        mat = np.array(mat)
        t = mat.shape[0]
        indices = self.get_sample_indices(t)
        try:
            mat = mat[indices,:,:].astype(np.float32)
        except:
            print("what",mat.shape)
        return mat   
            
    def __getitem__(self, index):
        record = self.skeleton_list[index]
        data = self._load_data(record.filename)
        if self.transform is not None:
            process_data = self.transform(data)
        else:
            process_data = data
        return process_data, record.label

    def __len__(self):
        return len(self.skeleton_list)