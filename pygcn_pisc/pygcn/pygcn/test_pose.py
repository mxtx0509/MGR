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
from pygcn.dataset_pose import SRDataset
from torch.autograd import Variable
import math
#from pygcn.models import GCN
from pygcn.models_pose import GCN
import torch.optim as optim


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Relationship')
parser.add_argument('--start_epoch',  default=0, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--result_path', default='./result/', type=str, metavar='PATH', #model_best.pth.tar
                    help='path to weights (default: none)')
parser.add_argument('--weights', default='./checkpoints/3000_SGD_noshuffle_new/26_checkpoint_ep26.pth.tar', type=str, metavar='PATH', #model_best.pth.tar
                    help='path to weights (default: none)')   
parser.add_argument('--write_out',default=True, type=int,
                    help='val step')
parser.add_argument('--val_step',default=1, type=int,
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
parser.add_argument('--print_freq', default=1000, type=int, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--test_mode', default=True, type=bool, metavar='PATH',
                    help='path for saving result (default: none)')
parser.add_argument('--epochs', type=int, default=150,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
                    # help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of hidden units.')
parser.add_argument('--tag', type=int, default=1,
                    help='Number of hidden units.')
parser.add_argument('--threshold', type=float, default=0.12,
                    help='Number of hidden units.')
parser.add_argument('--workers', type=int, default=4,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
                    

best_prec1 = 0

def get_loader(adj_dir, feature_dir,SIZE):
    test_list = '/export/home/zm/test/icme2019/SR_graph/list/PISC_fine_test.txt'

    test_set = SRDataset(adj_dir, feature_dir=feature_dir, file_list = test_list,adj_size=SIZE,graph_mode=args.graph_mode,is_train= False)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=False)
    return  test_loader
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
    feature_dir = args.feature_dir
    test_loader = get_loader(adj_dir, feature_dir,SIZE)


    # load network
    print ('====> Loading the network...')
    model = GCN(adj_size=SIZE,nfeat=256,nhid=args.hidden,nclass=6,dropout=args.dropout)
    print (model)   
    # print model
    
    # load weight
    if args.weights:
        if os.path.isfile(args.weights):
            print("====> loading model '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            #checkpoint_dict = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("====> no pretrain model at '{}'".format(args.weights))
    
    # model.fg = torch.nn.DataParallel(model.fg)
    # model.full_im_net = torch.nn.DataParallel(model.full_im_net)
    model = torch.nn.DataParallel(model)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    
    cudnn.benchmark = True

    fnames = []
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.write_out:
        print ('------Write out result---')
        for i in range(args.num_class):
            fnames.append(open(args.result_path + str(i) + '.txt', 'w'))
    validate(test_loader, model, criterion, fnames)
    if args.write_out:
        for i in range(args.num_class):
            fnames[i].close()
    return

def validate(val_loader, model, criterion, fnames=[]):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    tp = {} # precision
    p = {}  # prediction
    r = {}  # recall
    output_sum = np.zeros((3961,args.num_class))
    for i, (adj_tensor, per_obj_feature,pose_adj,pose_feature,target) in enumerate(val_loader):
        # measure data loading time
        adj_tensor = Variable(adj_tensor.cuda())
        per_obj_feature = Variable(per_obj_feature.cuda())
        pose_adj = Variable(pose_adj.cuda())
        pose_feature = Variable(pose_feature.cuda())
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        
        output = model(per_obj_feature,adj_tensor,pose_feature, pose_adj)
        
        loss = criterion(output, target_var)
        losses.update(loss.data[0], per_obj_feature.size(0))
        prec1 = accuracy(output.data, target)
        top1.update(prec1[0], per_obj_feature.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1))
        
        #####################################
        ## write scores
        if args.write_out:
            output_f = F.softmax(output, dim=1)
            output_np = output_f.data.cpu().numpy()
            output_sum[i] = output_np
            pre = np.argmax(output_np[0])
            t = target.data.cpu().numpy()[0]
            # if r.has_key(t):
            if t in r.keys():
                r[t] += 1
            else:
                r[t] = 1
            #if p.has_key(pre):
            if pre in p.keys():
                p[pre] += 1
            else:
                p[pre] = 1
            if pre == t:
                if t in tp.keys():
                    tp[t] += 1
                else:
                    tp[t] = 1
            
            for j in range(args.num_class):
                fnames[j].write(str(output_np[0][j]) + '\n')
        #####################################

    np.savetxt('./result/result.txt',output_sum)
    print ('tp: ', tp)
    print ('p: ', p)
    print ('r: ', r)
    precision = {}
    recall = {}
    for k in tp.keys():
        precision[k] = float(tp[k]) / float(p[k])
        recall[k] = float(tp[k]) / float(r[k])
    print ('precision: ', precision)
    print ('recall: ', recall)

    print(' * Prec@1 {top1.avg[0]:.3f}\t * Loss {loss.avg:.4f}'.format(top1=top1, loss=losses))
    return top1.avg[0]

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
