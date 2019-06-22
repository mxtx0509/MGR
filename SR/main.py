import argparse
import os, sys
import shutil
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc
import os.path as osp
from torch.autograd import Variable
import math
from networks.GRM import GRM
from dataset.dataset import SRDataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Relationship')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('objects', metavar='DIR', help='path to objects (bboxes and categories)')
parser.add_argument('trainlist', metavar='DIR', help='path to test list')
parser.add_argument('testlist', metavar='DIR', help='path to test list')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--start_epoch',  default=0, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--weights', default='', type=str, metavar='PATH',
                    help='path to weights (default: none)')
parser.add_argument('--mode', default='', type=str, metavar='PATH',
                    help='path to weights (default: none)')
parser.add_argument('--scale-size',default=256, type=int,
                    help='input size')
parser.add_argument('-n', '--num_classes', default=16, type=int, metavar='N',
                    help='number of classes / categories')
parser.add_argument('--write-out', dest='write_out', action='store_true',
                    help='write scores')
parser.add_argument('--crop_size',default=224, type=int,
                    help='crop size')
parser.add_argument('--val_step',default=1, type=int,
                    help='val step')
parser.add_argument('--epochs',default=400, type=int,
                    help='epochs')
parser.add_argument('--save_dir',default='./checkpoints/att/', type=str,
                    help='save_dir')
parser.add_argument('--num_gpu', default=4, type=int, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--adjacency_matrix', default='', type=str, metavar='PATH',
                    help='path to adjacency-matrix of graph')
best_prec1 = 0

def get_dataset(data_dir, objects_dir,train_list, test_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size

    train_data_transform = transforms.Compose([
            transforms.Scale((336,336)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((crop_size,crop_size)),
            transforms.ToTensor(),
            normalize])  # what about horizontal flip
    test_data_transform = transforms.Compose([
            transforms.Scale((crop_size, crop_size)),
            transforms.ToTensor(),
            normalize])  # what about horizontal flip
    train_full_transform = transforms.Compose([
            transforms.Scale((448,448)),
            transforms.ToTensor(),
            normalize]) 
    test_full_transform = transforms.Compose([
            transforms.Scale((448, 448)),
            transforms.ToTensor(),
            normalize])  # what about horizontal flip
    train_set = SRDataset(data_dir, objects_dir, train_list, train_data_transform, test_full_transform, is_train= True )
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=True)
    
    test_set = SRDataset(data_dir, objects_dir, test_list, test_data_transform, test_full_transform, is_train= False )
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=False)
    return train_loader,test_loader
    

def main():
    
    global args, best_prec1
    args = parser.parse_args()
    print (args)
    best_acc = 0
    # Create dataloader
    print ('====> Creating dataloader...')
    data_dir = args.data
    train_list = args.trainlist
    test_list = args.testlist
    objects_dir = args.objects
    train_loader,test_loader = get_dataset(data_dir, objects_dir, train_list,test_list)
    # load network
    model = GRM(num_class=args.num_classes)
    
    model = torch.nn.DataParallel(model)
    model.cuda()
    mkdir_if_missing(args.save_dir)
    if args.weights != '':
        try:
            ckpt = torch.load(args.weights)
            model.module.load_state_dict(ckpt['state_dict'])
            print ('!!!load weights success !! path is ',args.weights)
        except Exception as e:
            model_init(args.weights,model)
            
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = 10e-3 )
    criterion = nn.CrossEntropyLoss().cuda()
    
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_lr(optimizer, epoch)
        # train_loader,test_loader = get_dataset(data_dir, objects_dir, train_list,test_list)
        train (train_loader,test_loader, model, criterion,optimizer, epoch)
        if epoch% args.val_step == 0:
            acc = validate(test_loader, model, criterion)
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch,
                }, is_best=is_best,train_batch=60000, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch) + '.pth.tar')


    return

def model_init(weights,model):
    print ('attention!!!!!!! load model fail and go on init!!!')
    ckpt = torch.load(weights)
    pretrained_dict=ckpt['state_dict']
    model_dict = model.module.state_dict()
    model_pre_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(model_pre_dict)
    model.module.load_state_dict(model_dict)
    for v ,val in model_pre_dict.items() :
        print ('update',v)

def adjust_lr(optimizer, ep):
    # if ep < 10:
        # lr = 1e-3
    # elif ep < 20:
    if ep < 20*2 :
        lr = 1e-4 
    elif ep < 30*2:
        lr = 1e-5 
    elif ep < 40*2:
        lr = 1e-6 
    else:
        lr = 1e-7 
    for p in optimizer.param_groups:
        p['lr'] = lr

    print ("lr is ",lr)


def train(train_loader,test_loader, model, criterion,optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    # switch to train mode
    model.train()


    for i, (union, obj1, obj2, bpos, target, full_im, bboxes_14, categories) in enumerate(train_loader):
        
        batch_size = union.shape[0]
        # cur_rois_sum = categories[0,0]
        # bboxes = bboxes_14[0, 0:categories[0,0], :]
        # for b in range(1, batch_size):
            # bboxes = torch.cat((bboxes, bboxes_14[b, 0:categories[b,0], :]), 0)
            # cur_rois_sum += categories[b,0]
        # assert(bboxes.size(0) == cur_rois_sum), 'Bboxes num must equal to categories num'

        target = target.cuda(async=True)
        union_var = torch.autograd.Variable(union, volatile=True).cuda()
        obj1_var = torch.autograd.Variable(obj1, volatile=True).cuda()
        obj2_var = torch.autograd.Variable(obj2, volatile=True).cuda()
        bpos_var = torch.autograd.Variable(bpos, volatile=True).cuda()
        full_im_var = torch.autograd.Variable(full_im, volatile=True).cuda()
        bboxes_var = torch.autograd.Variable(bboxes_14, volatile=True).cuda()
        categories_var = torch.autograd.Variable(categories, volatile=True).cuda()
        
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(union_var, obj1_var, obj2_var, bpos_var, full_im_var, bboxes_var, categories_var)
        
        loss = criterion(output, target)

        prec1= accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), obj1.size(0))
        top1.update(prec1[0], obj1.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 1==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
               

        # val_list_1 = [math.ceil(len(train_loader)/3),math.ceil(len(train_loader)*2/3)]
        # if i in val_list_1:
            # acc = validate(test_loader, model, criterion)
            # save_checkpoint({
                # 'state_dict': model.module.state_dict(),
                # 'epoch': epoch,
            # }, is_best=False,train_batch=i, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch) + '.pth.tar')
            # model.train()



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    lated = 0
    val_label = []
    val_pre = []
    
    model.eval()

    end = time.time()
    tp = {} # precision
    p = {}  # prediction
    r = {}  # recall
    
    for i, (union, obj1, obj2, bpos, target, full_im, bboxes_14, categories) in enumerate(val_loader):
        
        batch_size = union.shape[0]
        # cur_rois_sum = categories[0,0]
        # bboxes = bboxes_14[0, 0:categories[0,0], :]
        # for b in range(1, batch_size):
            # bboxes = torch.cat((bboxes, bboxes_14[b, 0:categories[b,0], :]), 0)
            # cur_rois_sum += categories[b,0]
        # assert(bboxes.size(0) == cur_rois_sum), 'Bboxes num must equal to categories num'

        target = target.cuda(async=True)
        union_var = torch.autograd.Variable(union, volatile=True).cuda()
        obj1_var = torch.autograd.Variable(obj1, volatile=True).cuda()
        obj2_var = torch.autograd.Variable(obj2, volatile=True).cuda()
        bpos_var = torch.autograd.Variable(bpos, volatile=True).cuda()
        full_im_var = torch.autograd.Variable(full_im, volatile=True).cuda()
        bboxes_var = torch.autograd.Variable(bboxes_14, volatile=True).cuda()
        categories_var = torch.autograd.Variable(categories, volatile=True).cuda()
        
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(union_var, obj1_var, obj2_var, bpos_var, full_im_var, bboxes_var, categories_var)
        
        

        # compute output
        loss = criterion(output, target)

        prec1 = accuracy(output, target, topk=(1,))
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        val_label[lated:lated + batch_size] =target
        val_pre [lated:lated+batch_size] = pred.data.cpu().numpy().tolist()[:]
        lated = lated + batch_size
        
        losses.update(loss.item(), obj1.size(0))
        top1.update(prec1[0], obj1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 8==0:
            print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))


    print ('----------------------------------------------------------')
    count = [0]*args.num_classes
    acc = [0]*args.num_classes
    pre_new = []
    for i in val_pre:
        for j in i:
            pre_new.append(j)
    for idx in range(len(val_label)):
        count[val_label[idx]]+=1
        if val_label[idx] == pre_new[idx]:
            acc[val_label[idx]]+=1
    classaccuracys = []
    for i in range(args.num_classes):
        if count[i]!=0:
            classaccuracy = (acc[i]*1.0/count[i])*100.0
        else:
            classaccuracy = 0
        classaccuracys.append(classaccuracy)

    print(('Testing Results: Prec@1 {top1.avg:.3f} classacc {classaccuracys} Loss {loss.avg:.5f}'
          .format(top1=top1, classaccuracys = classaccuracys, loss=losses)))

    return top1.avg[0]


def save_checkpoint(state, is_best, train_batch,save_dir, filename='checkpoint.pth.tar'):
    fpath = '_'.join((str(state['epoch']),str(train_batch), filename))
    fpath = osp.join(save_dir, fpath)
    #mkdir_if_missing(save_dir)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(save_dir, 'model_best.pth.tar'))
def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
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


if __name__=='__main__':
    main()
