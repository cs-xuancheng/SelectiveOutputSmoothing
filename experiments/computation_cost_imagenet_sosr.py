import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Datasets
parser.add_argument('-d', '--data', default='', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# SOSR parameter
parser.add_argument('--beta', type=float, default=1, help='strength of SOSR')
parser.add_argument('--threshold', type=float, default=0.99,
                    help='probability used to detect the over-confident samples')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0                    # best test accuracy
best_t5 = 0                     # top-5 accuracy
best_epoch = 0                  # best epoch


def main():
    global best_acc, best_t5, best_epoch

    if os.path.exists(args.data):
        traindir = os.path.join(args.data, 'train')
        print('==> Preparing dataset in %s' % args.data)
    else:
        print('Error: no dataset directory found in %s!' % args.data)
        exit()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    model = resnet50(pretrained=False)
    print('===> Preparing ResNet-50')

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # each experiment is performed twice to avoid unnecessary error

    # vanilla
    for _ in range(2):
        train_plain(train_loader, model, criterion, optimizer, use_cuda)
    # 25% SOSR involvement
    for _ in range(2):
        train_SOSRbypercent(train_loader, model, criterion, optimizer, 0.25, use_cuda)
    # 50% SOSR involvement
    for _ in range(2):
        train_SOSRbypercent(train_loader, model, criterion, optimizer, 0.5, use_cuda)
    # 100% SOSR involvement
    for _ in range(2):
        train_SOSRbypercent(train_loader, model, criterion, optimizer, 1, use_cuda)


def train_plain(train_loader, model, criterion, optimizer, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 1000 == 0:
            print('Plain Training {batch}/{size}| Data: {data:.3f}s | Batch: {bt:.3f}s'.format(batch=batch_idx + 1, size=len(train_loader),data=data_time.avg,bt=batch_time.avg))


def train_SOSRbypercent(train_loader, model, criterion, optimizer, percent, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        desired_outputs = output_smooth_by_percent(outputs, targets, percent)
        loss = criterion(outputs, targets) + args.beta * pow((outputs - desired_outputs), 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 1000 == 0:
            print('SOSR involvement {pt}% {batch}/{size} | Data: {data:.3f}s | Batch: {bt:.3f}s'.format(batch=batch_idx + 1, size=len(train_loader),pt=int(percent*100),data=data_time.avg,bt=batch_time.avg))
    

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def output_smooth(outputs, targets, threshold):
    final_output = torch.tensor(outputs)
    outputs_softmax = F.softmax(outputs, dim=1)
    for idx in range(outputs.size(0)):
        pic = final_output[idx]
        t1_value, t1_index = torch.max(pic, dim=0)
        if t1_index.item() == targets[idx].item() and outputs_softmax[idx][t1_index].item() > threshold:
            mean_value = (pic.sum() - t1_value) / (pic.size(0) - 1)
            pic = torch.ones(pic.size()).cuda()
            pic = pic * mean_value
            pic[t1_index] = t1_value
            final_output[idx] = pic

    return final_output


def output_smooth_by_percent(outputs, targets, percent):
    final_output = torch.tensor(outputs)
    outputs_softmax = F.softmax(outputs, dim=1)
    for idx in range(outputs.size(0)):
        pic = final_output[idx]
        t1_value, t1_index = torch.max(pic, dim=0)
        if idx < percent*outputs.size(0):
            mean_value = (pic.sum() - t1_value) / (pic.size(0) - 1)
            pic = torch.ones(pic.size()).cuda()
            pic = pic * mean_value
            pic[t1_index] = t1_value
            final_output[idx] = pic
    return final_output



if __name__ == '__main__':
    a = time.time()
    main()
    b = time.time()
    print('Total time: {}day {}hour'.format(int((b - a) / 86400), int((b - a) % 86400 / 3600)))