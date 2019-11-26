import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F

class base_cnn(nn.Module):

    def __init__(self):
        super(base_cnn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,1,1,padding=0),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,8,(3,1),1,padding=(1,0)),
            nn.Conv2d(8,32,(1,3),1,padding=(0,1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,8,(3,1),1,padding=(1,0)),
            nn.Conv2d(8,32,(1,3),1,padding=(0,1)),
            nn.MaxPool2d(2)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,8,(3,1),1,padding=(1,0)),
            nn.Conv2d(8,64,(1,3),1,padding=(0,1)),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
                
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,8,(3,1),1,padding=(1,0)),
            nn.Conv2d(8,64,(1,3),1,padding=(0,1)),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64,8,(3,1),1,padding=(1,0)),
            nn.Conv2d(8,128,(1,3),1,padding=(0,1)),        
            nn.Tanh(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )


    def forward(self, input):
        '''
            input: N  C H W
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = self.conv5(out)
        out = self.conv6(out)

        return out

class cnn_classifier(nn.Module):
    def __init__(self, num_class, length=32, img_feature_dim=128):
        super(cnn_classifier, self).__init__()
        self.length = length
        self.img_feature_dim = img_feature_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(img_feature_dim,32,3,1,padding=1),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32,img_feature_dim,3,1,padding=1),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(img_feature_dim,32,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32,img_feature_dim,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(img_feature_dim*length//16,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(1024,num_class)

    def forward(self, input):
        N,T,C = input.size()
        feature = self.get_feature(input)

        out = feature.view(N,-1)    
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_feature(self,input):
        input = input.view(-1,self.length,self.img_feature_dim)
        input = input.permute(0,2,1)

        feature = self.conv1(input)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        # N C T/16
        feature = feature.transpose(1,2).contiguous()
        # N T/16 C
        return feature


        