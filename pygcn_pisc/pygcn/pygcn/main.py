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
from pygcn.dataset import SRDataset
from torch.autograd import Variable
import math
#from pygcn.models import GCN
from pygcn.models import GCN
import torch.optim as optim

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Relationship')

parser.add_argument('--start_epoch',  default=0, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--weights', default='', type=str, metavar='PATH', #model_best.pth.tar
                    help='path to weights (default: none)')
parser.add_argument('--write_out',default=True, type=int,
                    help='val step')
parser.add_argument('--val_step',default=2, type=int,
                    help='val step')
parser.add_argument('--num_class',default=6, type=int,
                    help='num_class')
parser.add_argument('--save_dir',default='./checkpoints/pose_1128/', type=str, 
                    help='save_dir')
parser.add_argument('--feature_dir',default='./checkpoints/pose_1128/', type=str, 
                    help='save_dir')
parser.add_argument('--graph_mode',default='pose', type=str,
                    help='mode')
parser.add_argument('--num_gpu', default=1, type=int, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--print_freq', default=4, type=int, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--test_mode', default=True, type=bool, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
                    # help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--tag', type=int, default=1,
                    help='Number of hidden units.')
parser.add_argument('--threshold', type=float, default=0.12,
                    help='Number of hidden units.')
parser.add_argument('--workers', type=int, default=4,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.8,
                    help='Dropout rate (1 - keep probability).')
                    
best_prec1 = 0
  
def get_loader(adj_dir, feature_dir,SIZE):
    train_list ='/export/home/zm/test/icme2019/SR_graph/list/PISC_fine_train.txt'
    test_list = '/export/home/zm/test/icme2019/SR_graph/list/PISC_fine_test.txt'

    train_set = SRDataset(adj_dir,feature_dir=feature_dir, file_list = train_list,adj_size=SIZE,graph_mode=args.graph_mode,is_train=True)
    test_set = SRDataset(adj_dir, feature_dir=feature_dir, file_list = test_list,adj_size=SIZE,graph_mode=args.graph_mode,is_train=False)
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=False)
    return train_loader , test_loader

def main():
    SIZE = 8
    global args, best_prec1
    args = parser.parse_args()
    print (args)
    best_acc = 0 
    tag = int(args.threshold * 100)
    # Create dataloader
    print ('====> Creating dataloader...')
    if not os.path.exists(args.save_dir) :
        os.makedirs(args.save_dir)
    if args.graph_mode == 'pose' :
        adj_dir = './%s_graph_1.npy'%args.graph_mode
    elif args.graph_mode == 'pose_obj':
        adj_dir = './%s_graph_1.npy'%args.graph_mode
    else:
        print ('arg.graph_mode input wrong!!!!!')
    feature_dir = args.feature_dir #'/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/'
    train_loader , test_loader = get_loader(adj_dir, feature_dir,SIZE)


    # load network
    print ('====> Loading the network...')
    model = GCN(adj_size=SIZE,nfeat=256,nhid=args.hidden,nclass=6,dropout=args.dropout) 
    print (model)
    print ('adj_dir:',adj_dir)
    print ('feature_dir:',feature_dir)
    model = torch.nn.DataParallel(model)
    model.cuda()
    
    # if args.weights!='':
        # ckpt = torch.load(args.save_dir + args.weights)
        # model.module.load_state_dict(ckpt['state_dict'])
        # print ('!!!load weights success !! path is ',args.weights)
    
    # mkdir_if_missing(args.save_dir)
    if args.weights != '':
        try:
            ckpt = torch.load(args.weights)
            model.module.load_state_dict(ckpt['state_dict'])
            print ('!!!load weights success !! path is ',args.weights)
        except Exception as e:
            model_init(args.weights,model)
            
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                           lr=args.lr,momentum=0.9)# 
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs + 1):
        # acc = validate(test_loader, model, criterion,epoch)
        adjust_lr(optimizer, epoch)
        train_loader,test_loader = get_loader(adj_dir, feature_dir,SIZE)
        train(train_loader, model, criterion,optimizer, epoch)
        if epoch% args.val_step == 0:
            acc = validate(test_loader, model, criterion,epoch)
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch,
                }, is_best=is_best, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch) + '.pth.tar')
        #train_loader , test_loader = get_loader(adj_dir, feature_dir,json_dir)

    return

def model_init(weights,model):
    print ('attention!!!!!!! load model fail and go on init!!!')
    ckpt = torch.load(args.save_dir+weights)
    pretrained_dict=ckpt['state_dict']
    model_dict = model.module.state_dict()
    model_pre_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(model_pre_dict)
    model.module.load_state_dict(model_dict)
    for v ,val in model_pre_dict.items() :
        print ('update',v)

def adjust_lr(optimizer, ep):

    if ep < 30 * 2:
        lr = 1e-2 
    elif ep < 60 * 2:
        lr = 1e-3 
    elif ep < 90 * 2:
        lr = 1e-4 
    elif ep < 120 * 2:
        lr = 1e-5 
    else:
        lr = 1e-6 
    for p in optimizer.param_groups:
        p['lr'] = lr
    print ("lr is ",lr)


def train(train_loader, model, criterion,optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    sub = AverageMeter()
    f1_ma = AverageMeter()
    end = time.time()
    # switch to train mode
    model.train()


    for i, (adj_tensor, feature_tensor,target) in enumerate(train_loader):
        # measure data loading time
        adj_tensor = Variable(adj_tensor.cuda())
        feature_tensor = Variable(feature_tensor.cuda())
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        
        output = model(feature_tensor, adj_tensor)
        #print ('=======',output.size())
        

        loss = criterion(output, target_var)
        #print (output)
        prec1,prec3 = accuracy(output.data, target, topk=(1,3))
        losses.update(loss.data[0], adj_tensor.size(0))
        top1.update(prec1[0], adj_tensor.size(0))
        top3.update(prec3[0], adj_tensor.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3, lr=optimizer.param_groups[-1]['lr'])))


               

        # val_list_1 = [math.ceil(len(train_loader)/3),math.ceil(len(train_loader)*2/3)]
        # if i in val_list_1:
            # acc = validate(test_loader, model, criterion)
            # save_checkpoint({
                # 'state_dict': model.module.state_dict(),
                # 'epoch': epoch,
            # }, is_best=False,train_batch=i, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch) + '.pth.tar')
            # model.train()



def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    sub = AverageMeter()
    tp = {} # precision
    p = {}  # prediction
    r = {}  # recall

    model.eval()

    end = time.time()
    lated = 0
    val_label = []
    val_pre = []

    for i, (adj_tensor, feature_tensor,target) in enumerate(val_loader):
        # measure data loading time
        adj_tensor = Variable(adj_tensor.cuda())
        feature_tensor = Variable(feature_tensor.cuda())
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        output = model(feature_tensor, adj_tensor)

        
                # compute output
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        #prec1, prec5, classacc = accuracy(output.data, target, topk=(1,5))
        prec1, prec3 = accuracy(output.data, target, topk=(1,3))
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        val_label[lated:lated + batch_size] =target
        val_pre [lated:lated+batch_size] = pred.data.cpu().numpy().tolist()[:]
        lated = lated + batch_size

        losses.update(loss.data[0], adj_tensor.size(0))
        top1.update(prec1[0], adj_tensor.size(0))
        top3.update(prec3[0], adj_tensor.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3)))
                    #####################################

    count = [0]*6
    acc = [0]*6
    pre_new = []
    for i in val_pre:
        for j in i:
            pre_new.append(j)
    for idx in range(len(val_label)):
        count[val_label[idx]]+=1
        if val_label[idx] == pre_new[idx]:
            acc[val_label[idx]]+=1
    classaccuracys = []
    for i in range(6):
        if count[i]!=0:
            classaccuracy = (acc[i]*1.0/count[i])*100.0
        else:
            classaccuracy = 0
        classaccuracys.append(classaccuracy)

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} classacc {classaccuracys} Loss {loss.avg:.5f}'
          .format(top1=top1, top3=top3,classaccuracys = classaccuracys, loss=losses)))

    return top1.avg



def save_checkpoint(state, is_best,save_dir, filename='checkpoint.pth.tar'):
    fpath = '_'.join((str(state['epoch']), filename))
    fpath = osp.join(save_dir, fpath)
    #mkdir_if_missing(save_dir)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(save_dir, 'model_best.pth.tar'))




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    count = [0]*6
    acc = [0]*6
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  #zhuanzhi
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for idx in range(batch_size):
        count[target[idx]]+=1
        if target[idx] == pred[0][idx]:
            acc[target[idx]]+=1
    res = []
    classaccuracys = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    for i in range(6):
        if count[i]!=0:
            classaccuracy = (acc[i]*1.0/count[i])*100.0
        else:
            classaccuracy = 0
        classaccuracys.append(classaccuracy)
    return res


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

if __name__=='__main__':
    main()
