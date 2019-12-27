import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F
from torch.autograd import Variable

class skeleton_model(nn.Module):
    def __init__(self,num_class, in_channel=2,
                            length=32,num_joint=10):
        super(skeleton_model, self).__init__()
        self.num_class = num_class
        self.in_channel = in_channel
        self.length = length
        self.num_joint = num_joint
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.conv2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.conv_att = nn.Conv1d(32*self.num_joint,1,3,1,padding=1)
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(self.num_joint,32,3,1,padding=1),
        #     nn.MaxPool2d(2)
        #     )
        self.hconv = HierarchyConv()
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            # nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.convm1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.convm2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.convm_att = nn.Conv1d(32*self.num_joint,1,3,1,padding=1)
        # self.convm3 = nn.Sequential(
        #     nn.Conv2d(self.num_joint,32,3,1,padding=1),
        #     nn.MaxPool2d(2)
        #     )
        self.hconvm = HierarchyConv()
        self.convm4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            # nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
                
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,128,3,1,padding=1),
            nn.ReLU(),
            # nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,256,3,1,padding=1),
            nn.ReLU(),
            # nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(256*(length//16)*(32//16),256),
            nn.ReLU(),
            # nn.Dropout2d(p=0.5)
        )
        self.fc8 = nn.Linear(256,self.num_class)

    def forward(self,input):
        output = self.get_feature(input)
        output = self.classify(output)
        return output

    def get_feature(self,input):
        # input: N J T D
        input = input.permute(0,3,1,2)
        N, D, T, J = input.size()
        motion = input[:,:,1::,:]-input[:,:,0:-1,:]
        motion = F.upsample(motion,size=(T,J),mode='bilinear').contiguous()

        out = self.conv1(input)
        out = self.conv2(out)
        out = out.permute(0,3,2,1).contiguous()
        # out: N J T D

        out_for_att = (out.permute(0,1,3,2).contiguous()).view(N,-1,T)
        att = self.conv_att(out_for_att).unsqueeze(3)
        att = torch.sigmoid(att)
        # print(att)
        out = out*att
        
        # out = self.conv3(out)
        out = self.hconv(out)
        out = self.conv4(out)

        outm = self.convm1(motion)
        outm = self.convm2(outm)
        outm = outm.permute(0,3,2,1).contiguous()
        # outm: N J T D

        outm_for_att = (outm.permute(0,1,3,2).contiguous()).view(N,-1,T)
        attm = self.convm_att(outm_for_att).unsqueeze(3)
        attm = torch.sigmoid(attm)
        # print(attm)
        outm = outm*attm

        # outm = self.convm3(outm)
        outm = self.hconvm(outm)
        outm = self.convm4(outm)

        out = torch.cat((out,outm),dim=1)
        out = self.conv5(out)
        out = self.conv6(out)
        # out:  N J T(T/16) D
        return out

    def classify(self,input):
        out = input.view(input.size(0),-1)
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor
        # NxC(num_class)
        return out

class HierarchyConv(nn.Module):
    def __init__(self):
        super(HierarchyConv,self).__init__()
        self.conva1 = nn.Conv2d(2,16,3,1,padding=1)
        self.conva2 = nn.Conv2d(2,16,3,1,padding=1)
        self.convh1 = nn.Conv2d(3,16,3,1,padding=1)
        self.convh2 = nn.Conv2d(3,16,3,1,padding=1)
        self.convl = nn.Conv2d(32,32,3,1,padding=1)
        self.convr = nn.Conv2d(32,32,3,1,padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(64,32,3,1,padding=1),
            nn.MaxPool2d(2)
        )

    def forward(self,input):
        a1 = input[:,[0,1],:,:]
        a2 = input[:,[3,4],:,:]
        h1 = input[:,[2,6,7],:,:]
        h2 = input[:,[5,8,9],:,:]
        l1 = self.conva1(a1) 
        l2 = self.conva2(a2) 
        r1 = self.convh1(h1)
        r2 = self.convh2(h2)
        l = torch.cat([l1,l2],1)
        r = torch.cat([r1,r2],1)
        l = self.convl(l)
        r = self.convr(r)
        out = torch.cat([l,r],1)
        out = self.conv(out)
        return out