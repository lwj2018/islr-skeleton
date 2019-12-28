import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F
from torch.autograd import Variable
from module.base_cnn import base_cnn, cnn_classifier
from module.resnet_model import ResidualNet
from module.skeleton_model import skeleton_model
from module.cnn_model import cnn_model

class islr_model(nn.Module):

    def __init__(self, num_class, in_channel=2,
                            length=32,num_joint=10,modality='rgb',
                            cnn_type='resnet18',train_mode='simple_fusion'):
        # T N D
        super(islr_model, self).__init__()
        self.num_class = num_class
        self.in_channel = in_channel
        self.length = length
        self.num_joint = num_joint
        self.modality = modality
        self.train_mode = train_mode

        if self.train_mode=="single_skeleton" or "fusion" in self.train_mode:
            self.get_skeleton_model()
        if "rgb" in self.train_mode or "fusion" in self.train_mode:
            self.get_cnn_model(cnn_type)
            self.cnn_classifier = cnn_classifier(
                num_class=num_class,
                length=length)
        if self.train_mode=="late_fusion":
            self.late_fusion = late_fusion(num_class)
        elif self.train_mode=="simple_fusion":
            self.simple_fusion = simple_fusion(num_class)


    def get_cnn_model(self,cnn_type ):
        self.cnn_model = cnn_model(self.num_class,base_model=cnn_type)

    def get_skeleton_model(self):
        self.skeleton_model = skeleton_model(self.num_class,self.in_channel,
                self.length,self.num_joint)
    

    def forward(self, input, image, heatmap):
        '''
            input: N J T D
        '''
        train_mode = self.train_mode
        heatmap = heatmap.view((-1,)+heatmap.size()[-3:])
        if train_mode=="single_skeleton":
            out = self.skeleton_model(input)

        elif train_mode=="single_rgb":
            # f = self.cnn_forward(image,heatmap)
            # out = self.cnn_classifier(f)
            out = self.cnn_model.forward_with_heatmap(image,heatmap)
            # out = self.cnn_model(image)
            self.conv1map = self.cnn_model.conv1map
            self.feature = self.cnn_model.feature

        elif train_mode=="rgb":
            out = self.cnn_model(image)
            self.conv1map = self.cnn_model.conv1map
            self.feature = self.cnn_model.feature
        
        elif train_mode=="late_fusion":
            out = self.skeleton_model.get_feature(input)
            out = out.transpose(1,2).contiguous()
            N,T,J,D = out.size()
            out = out.view(N,T,-1)
            out = l2norm(out,2)
            # print("out {}".format(out))
            # N T(T/16) C1

            f = self.cnn_forward(image,heatmap)
            out_c = self.cnn_classifier.get_feature(f)
            out_c = l2norm(out_c,2)
            # print("out_c {}".format(out_c))
            # N T(T/16) C2

            out = self.late_fusion(out,out_c)
        
        elif train_mode=='simple_fusion':
            out = self.skeleton_model(input)
            out = l2norm(out,1)
            # print("out {}".format(out.argmax(1)))

            out_c = self.cnn_model.forward_with_heatmap(image,heatmap)
            out_c = l2norm(out_c,1)
            # print("out_c {}".format(out_c.argmax(1)))

            out = torch.stack([out,out_c],2)
            out = self.simple_fusion(out)
            # print("final out {}".format(out.argmax(1)))

        return out

    def cnn_forward(self, image, heatmap):
        if self.modality=='rgb':
            sample_len = 3
        N,C,H,W = image.size()
        T = C//sample_len
        image = image.view( (-1, sample_len) + image.size()[-2:])
        conv_out = self.cnn_model.base_model.get_conv_out(image)
        # use heatmap
        _f_list = []
        N,C,K,K = conv_out.size()
        heatmap = F.upsample(heatmap,size=(K,K),mode='bilinear').contiguous()
        for i in range(heatmap.size(1)):
            _f =  conv_out*heatmap[:,i,:,:].unsqueeze(1)
            _f = F.adaptive_avg_pool2d(_f,[1,1]).squeeze()
            _f_list.append(_f)
        f = torch.stack(_f_list,2)
        # NxT C J
        f = F.adaptive_avg_pool1d(f,1)
        # don't use heatmap
        # f = F.adaptive_avg_pool2d(conv_out,[1,1])
        
        # NxT C
        f = f.view(N,T,-1)
        # N T C
        return f

    def get_optim_policies(self):
        if "rgb" in self.train_mode:
            return [
            {'params':self.cnn_model.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"cnn_params"},
        ]
        elif self.train_mode=="single_skeleton":
            return [
            {'params':self.skeleton_model.conv1.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.conv2.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            # {'params':self.skeleton_model.conv3.parameters(),'lr_mult':1,'decay_mult':0,
            # 'name':"skeleton_params"},
            {'params':self.skeleton_model.hconv.parameters(),'lr_mult':10,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.conv4.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.convm1.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.convm2.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            # {'params':self.skeleton_model.convm3.parameters(),'lr_mult':1,'decay_mult':0,
            # 'name':"skeleton_params"},
            {'params':self.skeleton_model.hconvm.parameters(),'lr_mult':10,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.convm4.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.conv5.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.conv6.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.fc7.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            {'params':self.skeleton_model.fc8.parameters(),'lr_mult':1,'decay_mult':0,
            'name':"skeleton_params"},
            # {'params':self.skeleton_model.conv_att.parameters(),'lr_mult':10,'decay_mult':10,
            # 'name':"att_params"},
            # {'params':self.skeleton_model.convm_att.parameters(),'lr_mult':10,'decay_mult':10,
            # 'name':"att_params"},
            ]
        elif self.train_mode=="simple_fusion":
            return [
                {'params':self.cnn_model.parameters(),'lr_mult':1,'decay_mult':0,
                'name':"cnn_params"},
                {'params':self.skeleton_model.parameters(),'lr_mult':1,'decay_mult':0,
                'name':"skeleton_params"},
                {'params':self.simple_fusion.parameters(),'lr_mult':10,'decay_mult':0,
                'name':"normal_params"},
            ]
        elif self.train_mode=="late_fusion":
            return [
                {'params':self.cnn_model.parameters(),'lr_mult':1,'decay_mult':0,
                'name':"cnn_params"},
                {'params':self.skeleton_model.parameters(),'lr_mult':1,'decay_mult':0,
                'name':"skeleton_params"},
                {'params':self.late_fusion.parameters(),'lr_mult':10,'decay_mult':0,
                'name':"normal_params"},
            ]

class late_fusion(nn.Module):
    def __init__(self,num_class):
        self.num_class = num_class
        super(late_fusion,self).__init__()
        self.fusion1 = nn.Sequential(
            nn.Linear(2048,256),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fusion2 = nn.Linear(256,self.num_class)
    
    def forward(self,input,input_c):
        N = input.size(0)
        out = torch.cat([input,input_c],2)
        # TODO what is N?
        out = out.view(N,-1)
        out = self.fusion1(out)
        out = self.fusion2(out)
        return out

class simple_fusion(nn.Module):
    def __init__(self,num_class):
        self.num_class = num_class
        super(simple_fusion,self).__init__()
        self.fusion1 = nn.Sequential(
            nn.Linear(2*self.num_class,500),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fusion2 = nn.Linear(500,self.num_class)

    def forward(self,input):
        N = input.size(0)
        out = input.view(N,-1)
        out = self.fusion1(out)
        out = self.fusion2(out)
        return out

def l2norm(vector,dim):
    l2sum = torch.sum(torch.pow(vector,2),dim)
    # if (l2sum<=0.0001).any():
    #     print("fuck {}".format(l2sum))
    l2norm = torch.pow(l2sum,0.5)
    vector = vector/(l2norm.unsqueeze(dim)+1e-6)
    return vector


        



        