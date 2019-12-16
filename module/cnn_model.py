import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import functional as F
from torch.autograd import Variable
from transforms import *
from module.resnet_model import ResidualNet
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class cnn_model(nn.Module):

    def __init__(self, num_class,
                hidden_unit=1024,
                modality='RGB',
                base_model='resnet18',
                dropout=0.8,
                img_feature_dim=256,
                hidden_size=256,
                partial_bn=True):
        super(cnn_model, self).__init__()
        self.num_class = num_class
        self.hidden_unit = hidden_unit
        self.modality = modality
        self.dropout = dropout
        self.img_feature_dim = img_feature_dim
        self.hidden_size = hidden_size

        self._prepare_base_model(base_model)
        feature_dim = self._prepare_new_fc()
        self._prepare_fc()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_new_fc(self):
        if 'vgg' in self.base_model_name:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name)[0].in_features
        elif 'res' in self.base_model_name:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features            
        elif self.base_model_name == 'BNInception':
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        elif self.base_model_name == 'Resnet_cbam':
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)

    def _prepare_base_model(self, base_model):

        self.base_model_name = base_model
        if 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        elif 'resnet' in base_model:
            depth = int(base_model.lstrip('resnet'))
            self.base_model = ResidualNet("ImageNet",depth,1000,None)
            from torch.utils import model_zoo
            checkpoint = model_zoo.load_url(model_urls[self.base_model_name])
            self.base_model.load_state_dict(checkpoint)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        elif base_model == 'Resnet_cbam':
            self.base_model = ResidualNet("ImageNet",50,1000,'CBAM')
            checkpoint = torch.load("models/RESNET50_CBAM_new_name_wrap.pth")
            # restore_param = {k: v for k, v in checkpoint.items()}
            # self.base_model.state_dict().update(restore_param)
            state_dict = {".".join(k.split(".")[1:]):v
                            for k,v in checkpoint['state_dict'].items()}
            self.base_model.load_state_dict(state_dict)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]            

        elif base_model == 'BNInception':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

        elif base_model == 'InceptionV3':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_mean = [104,117,128]
            self.input_std = [1]

        elif 'inception' in base_model:
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

        # DEBUG
        # print(self.base_model)

    def _prepare_fc(self):
        if self.hidden_unit>0:
            self.first_fc = nn.Linear(16*self.img_feature_dim, self.hidden_unit)
            self.final_fc = nn.Linear(self.hidden_unit, self.num_class)
        else:
            self.final_fc = nn.Linear(16*self.img_feature_dim, self.num_class)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(cnn_model, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        # HAVE_CHANGED origin is both false
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable
    
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        cbam_conv_weight = []
        cbam_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        lstm = []

        conv_cnt = 0
        bn_cnt = 0
        for key in self.state_dict():
            if "conv" in key and "cbam" in key:
                if key.split(".")[-1]=="weight":
                    cbam_conv_weight.append(self.state_dict()[key])
                elif key.split(".")[-1]=="bias":
                    cbam_conv_bias.append(self.state_dict()[key])
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
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

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.modules.rnn.LSTM):
                lstm.extend(list(m.parameters()))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': cbam_conv_weight, 'lr_mult': 1, 'decay_mult': 2,
             'name': "cbam_conv_weight"},
            {'params': cbam_conv_bias, 'lr_mult': 2, 'decay_mult': 2,
             'name': "cbam_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': lstm, 'lr_mult': 1, 'decay_mult': 0,
             'name': "lstm weights/bias"},             
        ]

    def forward(self, input):
        if self.modality == 'RGB':
            sample_len = 3
        base_out = self.base_model(input.view( (-1, sample_len) + input.size()[-2:]) )
        # if self.base_model_name=="Resnet_cbam":
        #     self.attention_map = self.base_model.attention_map
        self.conv1map = self.base_model.conv1map

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        base_out = base_out.view( (input.size(0),-1) )
        if self.first_fc is not None:
            base_out = self.first_fc(base_out)
        self.feature = base_out
        output = self.final_fc(base_out)
        
        return output

    def forward_with_heatmap(self, input, heatmap):
        if self.modality == 'RGB':
            sample_len = 3
        conv_out = self.base_model.get_conv_out(input.view( (-1, sample_len) + input.size()[-2:]) )
        self.conv1map = self.base_model.conv1map

        # use heatmap
        _f_list = []
        N,C,K,K = conv_out.size()
        heatmap = F.upsample(heatmap,size=(K,K),mode='bilinear').contiguous()

        # for i in range(heatmap.size(1)):
        #     plt.subplot(2,5,i+1)
        #     plt.imshow(heatmap.detach().cpu().numpy()[0,i,...])
        # plt.show()

        for i in range(heatmap.size(1)):
            _f =  conv_out*heatmap[:,i,:,:].unsqueeze(1)
            _f_list.append(_f)
        x,_ = torch.max(torch.stack(_f_list,0),0)      

        # pool,flatten and fc
        if self.base_model.network_type == "ImageNet":
            x = self.base_model.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc(x)  
        base_out = x

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        base_out = base_out.view( (input.size(0),-1) )
        if self.first_fc is not None:
            base_out = self.first_fc(base_out)
        self.feature = base_out
        output = self.final_fc(base_out)
        
        return output

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([
            GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
            GroupRandomHorizontalFlip(is_flow=False),
        ])

        