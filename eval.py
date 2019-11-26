import torch
import torch.nn as nn
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from torch.utils import data
import torchvision
import numpy as np
from dataset import skeleton_Dataset
from model import skeleton_model

import argparse
import os
import os.path as osp
import time
from tensorboardX import SummaryWriter

from opts import parser
from train import validate

def create_path(path):
    if not osp.exists(path):
        os.makedirs(path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    
    create_path(args.root_model)

    args.store_name = '_'.join(['eval','iSLR'\
                                'class'+str(args.num_class)])
    
    # get model 
    model = skeleton_model(args.num_class)

    model = torch.nn.DataParallel(model).cuda()
    model_dict = model.state_dict()

    # restore model
    if args.val_resume:
        if osp.isfile(args.val_resume):
            checkpoint = torch.load(args.val_resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # restore_param = {k:v for k,v in model_dict.items()}
            # model_dict.update(restore_param)
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {}) (best prec {})"
                  .format(args.evaluate, checkpoint['epoch'], best_prec1)))

        else:
            print(("=> no checkpoint found at '{}'".format(args.val_resume)))
    
    cudnn.benchmark = True

    # Data loading code

    val_loader = torch.utils.data.DataLoader(
        iSLR_Dataset(args.video_root,args.skeleton_root,args.val_file,
            length=args.length,
            transform=torchvision.transforms.Compose([
                GroupScale((crop_size,crop_size)),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        ),
        batch_size=args.batch_size,shuffle=False,
        num_workers=args.workers,pin_memory=True,
        # collate_fn=collate
    )

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # get writer
    # global writer
    # writer = SummaryWriter(logdir='runs/'+args.store_name)

    prec1 = validate(val_loader, model, criterion, 0)

    # for epoch in range(args.start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch , args.lr_steps)


    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch)

    #     # evaluate on validation set
    #     if (epoch) % args.eval_freq == 0 or epoch == args.epochs-1:
    #         prec1 = validate(val_loader, model, criterion, epoch // args.eval_freq)

if __name__=="__main__":
    main()

        