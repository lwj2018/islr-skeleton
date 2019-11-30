import faulthandler;faulthandler.enable()
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.store_name = '_'.join(['iSLR',args.train_mode,\
                                'class'+str(args.num_class)])
    
    create_path(args.root_model)
    # get model 
    model = islr_model(args.num_class)
    policies = model.get_optim_policies()
    # resume model
    model = resume_model(model,args.skeleton_resume,args.cnn_resume)

    model = torch.nn.DataParallel(model).cuda()
    model_dict = model.state_dict()

    # restore model
    if args.resume:
        if osp.isfile(args.resume):
            checkpoint = torch.load(args.resume)
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
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    
    cudnn.benchmark = True

    # Data loading code
    scale_size = 256
    crop_size = 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]          

    normalize = GroupNormalize(input_mean,input_std)

    train_loader = torch.utils.data.DataLoader(
        iSLR_Dataset(args.video_root,args.skeleton_root,args.train_file,
            length=args.length,
            transform=torchvision.transforms.Compose([
                # train_augmentation,
                GroupScale((crop_size,crop_size)),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        ),
        batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers,pin_memory=True,
        # collate_fn=collate
    )

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

    # optimizer = torch.optim.SGD(policies,
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(policies,
                                    args.lr)
    # get writer
    global writer
    writer = SummaryWriter(comment=args.store_name)

    # prec1 = validate(val_loader, model, criterion, 0 // args.eval_freq)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch , args.lr_steps)


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch) % args.eval_freq == 0 or epoch == args.epochs-1:
            prec1, prec5 = validate(val_loader, model, criterion, epoch // args.eval_freq)

            # remember best prec@1 and save checkpoint
            is_best = prec1>best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec5 = max(prec5, best_prec5)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5
            }, is_best)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, image, heatmap, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        input.require_grad = True
        input_var = input
        target.require_grad = True
        target_var = target
        image.require_grad = True
        heatmap.require_grad = True
        heatmap =heatmap.float()

        # visualize heatmap
        # tmp = heatmap.view((-1,)+heatmap.size()[-3:])
        # attentionmap_visualize(image,tmp[[12,14,16,18],4,:,:].unsqueeze(1))
        # compute output
        output = model(input_var,image,heatmap,train_mode=args.train_mode)
        loss = criterion(output, target_var)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        # print(attention_map)

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            # print(total_norm,args.clip_gradient)
            # if total_norm > args.clip_gradient:
                # print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                    'Prec@5 {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, 
                        lr=optimizer.param_groups[-1]['lr']))
            print(output)

        writer.add_scalar('train/loss', losses.avg, epoch*len(train_loader)+i)
        writer.add_scalar('train/acc', top1.avg, epoch*len(train_loader)+i)



def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

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
            heatmap = heatmap.float()

            # compute output
            output = model(input_var,image,heatmap,train_mode=args.train_mode)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

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

            writer.add_scalar('val/acc', top1.avg, epoch*len(val_loader)+i)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)

    return top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),
            '%s/%s_best.pth.tar' % (args.root_model, args.store_name))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # return the k largest elements of the given input Tensor
    # along the given dimension. dim = 1
    # pred is the indices
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def collate(batch):
    '''
        @return:
            fts: [N,L,A]
    '''
    fts = []
    targets = []
    len_list = []
    max_length = 0
    for sample in batch:
        ft = torch.Tensor(sample[0])
        target = torch.LongTensor([sample[1]])
        length = ft.size(0)
        if length > max_length:
            max_length = length
        len_list.append(length)
        fts.append(ft)
        targets.append(target)
    # fill zeros
    full_ft_list = []
    for ft, target, length in zip(fts, targets, len_list):
        pad = torch.zeros(max_length-length,ft.size(1))
        full_ft = torch.cat([ft, pad], 0)
        full_ft_list.append(full_ft)
    fts = full_ft_list
    # sorter
    X = zip(len_list, fts, targets)
    X = sorted(X, key=lambda t:t[0], reverse=True)
    len_list = [l for l,_,_ in X]
    fts = [ft for _,ft,_ in X]
    targets = [target for _,_,target in X]
    return torch.stack(fts, 0), torch.cat(targets, 0), torch.Tensor(len_list)

def resume_model(model, skeleton_resume, cnn_resume):
    if args.train_mode == "late_fusion":
        skeleton_checkpoint = torch.load(skeleton_resume)
        cnn_checkpoint = torch.load(cnn_resume)
        skeleton_state_dict = skeleton_checkpoint['state_dict']
        cnn_state_dict = cnn_checkpoint['state_dict']
        skeleton_restore_params = {".".join(["skeleton_model"]+k.split(".")[1:]):v for k,v in 
                skeleton_state_dict.items() if not "fc" in k}
        cnn_restore_params = {".".join(["cnn_model"]+k.split(".")[2:]):v for k,v in
                cnn_state_dict.items() if not ("fc" in k  )}
        model.state_dict().update(skeleton_restore_params)
        model.state_dict().update(cnn_restore_params)
    return model

if __name__=="__main__":
    main()

        