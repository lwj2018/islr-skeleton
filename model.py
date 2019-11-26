import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base_cnn import base_cnn, cnn_classifier

class skeleton_model(nn.Module):

    def __init__(self, num_class, in_channel=2,
                            length=32,num_joint=10,modality='rgb'):
        # T N D
        super(skeleton_model, self).__init__()
        self.num_class = num_class
        self.in_channel = in_channel
        self.length = length
        self.num_joint = num_joint
        self.modality = modality
        self.base_cnn = base_cnn()
        self.cnn_classifier = cnn_classifier(
            num_class=num_class,
            length=length)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.conv2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.num_joint,32,3,1,padding=1),
            nn.MaxPool2d(2)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.convm1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.convm2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.convm3 = nn.Sequential(
            nn.Conv2d(self.num_joint,32,3,1,padding=1),
            nn.MaxPool2d(2)
            )
        self.convm4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
                
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,128,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,256,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(256*(length//16)*(32//16),256),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256,self.num_class)

        self.fusion1 = nn.Sequential(
            nn.Linear(2*640,256),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fusion2 = nn.Linear(256,self.num_class)
    

    def forward(self, input, image, heatmap,train_mode="single_skeleton"):
        '''
            input: N J T D
        '''
        heatmap = heatmap.view((-1,)+heatmap.size()[-3:])
        if train_mode=="single_skeleton":
            out = skeleton_forward(input)
            out = out.view(out.size(0),-1)
            out = self.fc7(out)
            out = self.fc8(out)

            t = out
            assert not ((t != t).any())# find out nan in tensor
            assert not (t.abs().sum() == 0) # find out 0 tensor

        elif train_mode=="single_rgb":
            f = self.cnn_forward(image,heatmap)
            out = self.cnn_classifier(f)
        
        elif train_mode=="late_fusion":
            out = skeleton_forward(input)
            out = out.transpose(1,2).contiguous()
            N,T,J,D = out.size()
            out = out.view(N,T,-1)

            f = self.cnn_forward(image,heatmap)
            out_c = self.cnn_classifier.get_feature(f)
            out = self.late_fusion(out,out_c)

        return out

    def skeleton_forward(self, input):
        # input: N D T J
        input = input.permute(0,3,1,2)
        N, D, T, J = input.size()
        motion = input[:,:,1::,:]-input[:,:,0:-1,:]
        motion = F.upsample(motion,size=(T,V),mode='bilinear').contiguous()

        out = self.conv1(input)
        out = self.conv2(out)
        out = out.permute(0,3,2,1).contiguous()
        out = self.conv3(out)
        out = self.conv4(out)

        outm = self.convm1(motion)
        outm = self.convm2(outm)
        outm = outm.permute(0,3,2,1).contiguous()
        outm = self.convm3(outm)
        outm = self.convm4(outm)

        out = torch.cat((out,outm),dim=1)
        out = self.conv5(out)
        out = self.conv6(out)
        # out:  N J T(T/16) D
        return out

    def cnn_forward(self, image, heatmap):
        if self.modality=='rgb':
            sample_len = 3
        N,C,H,W = image.size()
        T = C//sample_len
        image = image.view( (-1, sample_len) + image.size()[-2:])
        conv_out = self.base_cnn(image)
        _f_list = []
        for i in range(heatmap.size(1)):
            _f =  conv_out*heatmap[:,i,:,:].unsqueeze(1)
            _f = F.adaptive_avg_pool2d(_f,[1,1]).squeeze()
            _f_list.append(_f)
        f = torch.stack(_f_list,2)
        # NxT C J
        f = F.adaptive_avg_pool1d(f,1)
        # NxT C
        f = f.view(N,T,-1)
        # N T C
        return f

    def late_fusion(self,out,out_c):
        out = torch.cat([out,out_c],2)
        out = out.view(N,-1)
        out = self.fusion1(out)
        out = self.fusion2(out)
        return out

        