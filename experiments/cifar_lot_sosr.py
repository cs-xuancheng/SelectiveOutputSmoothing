"""
Original Code is based on:
Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate.
https://github.com/jincan333/LoT
@misc{jin2024learning,
      title={Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate}, 
      author={Can Jin and Tong Che and Hongwu Peng and Yiyuan Li and Marco Pavone},
      year={2024},
      eprint={2402.02769},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

"""
from __future__ import print_function

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
import models.cifar as models
import torch.nn.functional as F


parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='test/0', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='bottleneck',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# LoT setting, we follow the recommended setting as in https://github.com/jincan333/LoT/blob/main/run/run_image_classification_LoT.sh
parser.add_argument('--student_index', type=int, default=0, help='an independent index for student updating')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--T', type=float, default=1.5)
parser.add_argument('--student_steps_ratio', type=int, default=2)
# SOSR parameter
parser.add_argument('--beta', type=float, default=0.5, help='strength of SOSR')
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

best_acc = 0  # best test accuracy
best_epoch = 0


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    global num_classes
    dataloader = datasets.CIFAR100
    num_classes = 100
    path = './data'

    trainset = dataloader(root=path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model")
    teacher_model = ResNet(num_classes=num_classes, depth=110)
    student_model = ResNet(num_classes=num_classes, depth=110)
    
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    student_model = torch.nn.DataParallel(student_model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in teacher_model.parameters())/1000000.0))
    teacher_optimizer = torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, params=teacher_model.parameters())
    student_optimizer = torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, params=student_model.parameters())

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(teacher_optimizer, student_optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train(trainloader, teacher_model, student_model, teacher_optimizer, student_optimizer, epoch)
        t_testacc, s_testacc = test(testloader, teacher_model, student_model)

        # save model
        is_best = s_testacc > best_acc
        best_acc = max(s_testacc, best_acc)
        if is_best is True:
            best_epoch = epoch + 1

        print('Teacher Acc:{:.4f} | Student Acc:{:.4f}'.format(t_testacc, s_testacc))
        print('Best Acc: {:.4f} | Best Epoch:{}'.format(best_acc, best_epoch))


def train(trainloader, teacher, student, teacher_optimizer, student_optimizer, epoch):
    # switch to train mode
    teacher.train()
    student.train()

    losses = AverageMeter()
    t_top1 = AverageMeter()
    s_top1 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)

        desired_t_outputs = output_smooth(teacher_outputs, targets, threshold=args.threshold)
        desired_s_outputs = output_smooth(student_outputs, targets, threshold=args.threshold)
        teacher_pred=F.log_softmax(teacher_outputs, dim=-1)
        student_pred=F.log_softmax(student_outputs, dim=-1)

        # compute gradient and do SGD step
        teacher_optimizer.zero_grad()
        student_optimizer.zero_grad()
        
        teacher_loss=F.cross_entropy(teacher_pred, targets) + args.alpha*kl_div_logits(teacher_pred, student_pred.detach(), args.T) + args.beta*pow((teacher_outputs - desired_t_outputs), 2).mean()
        student_loss=F.cross_entropy(student_pred, targets) + args.alpha*kl_div_logits(student_pred, teacher_pred.detach(), args.T) + args.beta*pow((student_outputs - desired_s_outputs), 2).mean()
        
        teacher_loss.backward()
        student_loss.backward()
        teacher_optimizer.step()
        student_optimizer.step()

        # measure accuracy and record loss
        t_prec1, _ = accuracy(teacher_outputs.data, targets.data, topk=(1,5))
        s_prec1, _ = accuracy(student_outputs.data, targets.data, topk=(1,5))
        losses.update(teacher_loss.item(), inputs.size(0))
        t_top1.update(t_prec1.item(), inputs.size(0))
        s_top1.update(s_prec1.item(), inputs.size(0))

        # student additional train
        for _ in range(args.student_steps_ratio - 1):
            s_inputs, s_targets = get_batch(trainloader, args.student_index)
            s_inputs, s_targets = s_inputs.cuda(), s_targets.cuda()
            args.student_index = (args.student_index+1) % len(trainloader)
            teacher_pred=F.log_softmax(teacher(s_inputs), dim=-1)
            student_pred=F.log_softmax(student(s_inputs), dim=-1)

            student_loss=F.cross_entropy(student_pred, s_targets) + args.alpha * kl_div_logits(student_pred, teacher_pred.detach(), T)
            student_optimizer.zero_grad()
            student_loss.backward()
            student_optimizer.step()

        # progress
        if batch_idx % 10 == 0 or batch_idx == len(trainloader)-1:
            print('Training | Epoch:{}/{}| Batch: {}/{}| Teacher Top-1:{:.2f} | Student Top-1:{:.2f}'.format(
                epoch+1, args.epochs, batch_idx, len(trainloader), t_top1.avg, s_top1.avg))
    return (t_top1.avg, s_top1.avg)


def test(testloader, teacher, student):
    teacher.eval()
    student.eval()

    t_top1 = AverageMeter()
    s_top1 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)

        # measure accuracy and record loss
        t_prec1, _ = accuracy(teacher_outputs.data, targets.data, topk=(1,5))
        s_prec1, _ = accuracy(student_outputs.data, targets.data, topk=(1,5))
        t_top1.update(t_prec1.item(), inputs.size(0))
        s_top1.update(s_prec1.item(), inputs.size(0))

        # progress
        if batch_idx % 10 == 0 or batch_idx == len(testloader)-1:
            print('Testing  | Batch: {}/{}| Teacher Top-1:{:.2f} | Student Top-1:{:.2f}'.format(
                batch_idx, len(testloader), t_top1.avg, s_top1.avg))
    return (t_top1.avg, s_top1.avg)


def adjust_learning_rate(toptimizer,soptimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in toptimizer.param_groups:
            param_group['lr'] = state['lr']
        for param_group in soptimizer.param_groups:
            param_group['lr'] = state['lr']


def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


def get_batch(data_loader, batch_index):
    start_index = batch_index * data_loader.batch_size
    end_index = start_index + data_loader.batch_size
    batch_data = []
    batch_targets = []
    
    for i in range(start_index, end_index):
        if i >= len(data_loader.dataset):
            break
        data, target = data_loader.dataset[i]
        batch_data.append(data)
        batch_targets.append(target)
    
    return torch.stack(batch_data), torch.tensor(batch_targets)


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
    a = time.time()
    main()
    b = time.time()
    print('Total time: {}hour {}min'.format(int((b - a) / 3600), int((b - a) % 3600 / 60)))
