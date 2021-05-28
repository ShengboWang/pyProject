#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model import *
from utils import progress_bar
import pickle
import json

from datetime import datetime
import logging
import sys

# +
parser = argparse.ArgumentParser(description='Training Capsules using Inverted Dot-Product Attention Routing')

parser.add_argument('--resume_dir', '-r', default='', type=str, help='dir where we resume from checkpoint')
parser.add_argument('--num_routing', default=1, type=int, help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset. CIFAR10 or CIFAR100.')
parser.add_argument('--backbone', default='nas', type=str, help='type of backbone. simple, resnet or nas')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers. 0 or 2')
parser.add_argument('--config_path', default='pretrainingnet.json', type=str,
                    help='path of the config')
parser.add_argument('--save', type=str, default='Pretrain', help='experiment name')
parser.add_argument('--debug', action='store_true',
                    help='use debug mode (without saving to a directory)')
parser.add_argument('--sequential_routing', action='store_true', help='not using concurrent_routing')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate. 0.1 for SGD')
parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# -

args = parser.parse_args()
assert args.num_routing > 0

args.save = '{}-{}'.format(args.save, datetime.today().strftime(('%Y-%m-%d-%H-%M-%S')))

if not os.path.exists(args.save):
    os.mkdir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("args = %s", args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 扩展每张图片为40 * 40，随即裁剪为32 * 32
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transforms.ToTensor(),  # 转化为训练支持的tensor数据
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 数据标准化，需给定三维张量的均值和方差向量
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = getattr(torchvision.datasets, args.dataset)(root='../data', train=True, download=True,
                                                       transform=transform_train)
testset = getattr(torchvision.datasets, args.dataset)(root='../data', train=False, download=True,
                                                      transform=transform_test)
num_class = int(
    args.dataset.split('CIFAR')[1])  # 提取出cifar数据集包含的分类数，cifar10为10，cifar100为100；以cifar为分割符，分成若干部分；取第二个，即10或100

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=args.num_workers)


num_train = len(trainset)
indices = list(range(num_train))  # 建立所有数据集的索引
split = int(np.floor(args.train_portion * num_train))  # 数据集分割成两部分，包括训练部分和验证部分


train_queue = torch.utils.data.DataLoader(
      trainset, batch_size=64,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),  # 自定义取样本的策略，范围为训练部分
      pin_memory=True,  # 是否拷贝tensors到cuda中的固定内存中
      num_workers=2)  # 进程数量，0意味着都被load进主进程

valid_queue = torch.utils.data.DataLoader(
      trainset, batch_size=64,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]), # 范围为测试部分
      pin_memory=True, num_workers=2)


print('==> Building model..')
# Model parameters

image_dim_size = 32

with open(args.config_path, 'rb') as file:
    params = json.load(file)  # 将json中的对象转换成python中的dict


loss_func = nn.CrossEntropyLoss()


net = CapsModel(image_dim_size,
                              params,
                              args.backbone,
                              args.dp,
                              args.num_routing,
                              args.layers,
                              loss_func,
                              sequential_routing=args.sequential_routing)

# +

architect = Architect(net, args)


optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)


# -

def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# if not os.path.isdir('results') and not args.debug:
#     os.mkdir('results')
# if not args.debug:
#     store_dir = os.path.join('results')
#     os.mkdir(store_dir)

net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=[0])
cudnn.benchmark = True

if args.resume_dir and not args.debug:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_queue):

        inputs = inputs.to(device)
        targets = targets.to(device)

        input_search, target_search = next(iter(valid_queue))

        input_search = input_search.to(device)

        target_search = target_search.to(device)

        architect.step(input_search, target_search)

        optimizer.zero_grad()

        v = net(inputs)

        loss = loss_func(v, targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = v.max(dim=1)

        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_queue), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100. * correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)

            targets = targets.to(device)

            v = net(inputs)

            loss = loss_func(v, targets)

            test_loss += loss.item()

            _, predicted = v.max(dim=1)

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc and not args.debug:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.save, 'ckpt.pth'))
        best_acc = acc
    return 100. * correct / total


# +
results = {
    'args': args,
    'params': params,
    'train_acc': [],
    # 'test_acc': [],
}

total_epochs = 15

for epoch in range(start_epoch, start_epoch + total_epochs):
    logging.info('epoch %d', epoch)
    results['train_acc'].append(train(epoch))
    genotype = net.module.genotype()
    logging.info('genotype = %s', genotype)
    # print(F.softmax(net.module.pre_caps.alphas_normal, dim=-1))
    # print(F.softmax(net.module.pre_caps.alphas_reduce, dim=-1))

    lr_decay.step()
    # results['test_acc'].append(test(epoch))
# -

if not args.debug:
    store_file = os.path.join(args.save, 'dataset_' + str(args.dataset) + '_num_routing_' + str(args.num_routing) + \
                              '_backbone_' + args.backbone + '.dct')

    pickle.dump(results, open(store_file, 'wb'))
