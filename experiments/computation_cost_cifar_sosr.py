import argparse
import os
import time
import random
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
# Datasets
parser.add_argument('-d', '--data', default='', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
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

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR100(root= args.data, train=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)


    model = ResNet(depth=56, num_classes=100, block_name='bottleneck')

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
    # # 50% SOSR involvement
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

        if batch_idx % 20 == 0:
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

        if batch_idx % 20 == 0:
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='bottleneck'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



if __name__ == '__main__':
    main()
