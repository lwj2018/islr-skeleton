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
from dataset import iSLR_Dataset
from model import islr_model

import argparse
import os
import os.path as osp
import time
from tensorboardX import SummaryWriter

from opts import parser
from transforms import *
from viz_utils import attentionmap_visualize
from train import accuracy

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
best_prec5 = 0


def main():
    global args, best_prec1, best_prec5
    args = parser.parse_args()
    args.store_name = '_'.join(['iSLR',args.train_mode,\
                                'class'+str(args.num_class)])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    create_path(args.root_model)
    # get model 
    model = islr_model(args.num_class,train_mode=args.train_mode)

    model = torch.nn.DataParallel(model).cuda()

    # restore model
    if args.val_resume:
        if osp.isfile(args.val_resume):
            checkpoint = torch.load(args.val_resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec5 = checkpoint['best_prec5']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})\n \
                    best_prec1: {:.3f}\n \
                    best_prec5: {:.3f}"
                  .format(args.evaluate, checkpoint['epoch'],\
                      best_prec1,best_prec5)))

        else:
            print(("=> no checkpoint found at '{}'".format(args.val_resume)))
    
    cudnn.benchmark = True

    # Data loading code
    scale_size = 256
    crop_size = 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]          

    normalize = GroupNormalize(input_mean,input_std)

    val_loader = torch.utils.data.DataLoader(
        iSLR_Dataset(args.video_root,args.skeleton_root,args.val_file,
            length=args.length,
            image_length=args.image_length,
            train_mode=args.train_mode,
            transform=torchvision.transforms.Compose([
                GroupScale((crop_size,crop_size)),
                # GroupScale(int(scale_size)),
                # GroupCenterCrop(crop_size),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        ),
        batch_size=args.batch_size,shuffle=False,
        num_workers=args.workers,pin_memory=True,
        # collate_fn=collate
    )

    # define loss function (criterion)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    prec1,prec5 = validate(val_loader, model, criterion, 0 // args.eval_freq)

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cmat =AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, image, heatmap, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda()
            # input_var = torch.autograd.Variable(input)
            # target_var = torch.autograd.Variable(target)
            input.require_grad = False
            input_var = input
            target.require_grad = False
            target_var = target
            image.require_grad = True
            heatmap.require_grad = True
            input_var = input_var.float()
            heatmap = heatmap.float()

            # compute output
            output = model(input_var,image,heatmap)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            c_matrix = confusion_matrix(output.data,target)

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            cmat.update(c_matrix.detach().cpu().numpy(),1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                    'Prec@5 {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)


    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f Best Prec@5: %.3f'%(best_prec1,best_prec5)
    print(output_best)
    print(cmat.sum)
    save_cmat(cmat.sum)
    print("train mode: %s"%args.train_mode)

    return top1.avg, top5.avg

def confusion_matrix(output,target):
    num_class = output.size(1)
    pred = torch.argmax(output,1)
    cmat = torch.zeros(num_class,num_class)
    for i,j in zip(pred,target):
        cmat[i,j] += 1
    return cmat

def save_cmat(cmat):
    create_path("output")
    file = open("output/cmat_"+args.store_name+".txt","w")
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            file.write("%d "%cmat[i,j])
        file.write("\n")

if __name__=="__main__":
    main()

        