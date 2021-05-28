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
import pickle
import json

import parameters
from NASCapsModel import *
from utils import progress_bar
import logging
import sys

# +
parser = argparse.ArgumentParser(description='Training CapsNet with NAS searched architecture')
parser.add_argument('--num_routing', default=1, type=int, help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset. CIFAR10 or CIFAR100.')
parser.add_argument('--backbone', default='nas', type=str, help='type of backbone. simple, resnet or nas')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers. 0 or 2')
parser.add_argument('--config_path', default='network_paras.json', type=str,
                    help='path of the config')
parser.add_argument('--save', type=str, default='train', help='experiment name')
parser.add_argument('--sequential_routing', action='store_true', help='not using concurrent_routing')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate for SGD')
parser.add_argument('--dp', default=0.2, type=float, help='dropout rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# -

args = parser.parse_args()
assert args.num_routing > 0

args.save = '{}-{}'.format(args.save, '10opt-without-pooling')

if not os.path.exists(args.save):
    os.mkdir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
logging.info("device = %s", device)

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

train_queue = torch.utils.data.DataLoader(
      trainset, batch_size=128, shuffle=True, pin_memory=True, num_workers=2)

test_queue = torch.utils.data.DataLoader(
      testset, batch_size=128, shuffle=False, pin_memory=True, num_workers=2)


print('==> Building model..')
# Model parameters

image_dim_size = 32

with open(args.config_path, 'rb') as file:
    params = json.load(file)  # 将json中的对象转换成python中的dict

loss_func = nn.CrossEntropyLoss()

genotype = parameters.DARTS
net = CapsModel(image_dim_size,
                params,
                args.backbone,
                args.dp,
                args.num_routing,
                args.layers,
                genotype,
                sequential_routing=args.sequential_routing)


optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=[2, 3])
cudnn.benchmark = True

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
    torch.save(net, os.path.join(args.save, 'weights.pt'))
    return 100. * correct / total


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_queue):
            inputs = inputs.to(device)

            targets = targets.to(device)

            v = net(inputs)

            loss = loss_func(v, targets)

            test_loss += loss.item()

            _, predicted = v.max(dim=1)

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_queue), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    logging.info('test acc = %d', acc)
    print('Saving..')
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
    }
    torch.save(state, os.path.join(args.save, 'ckpt.pth'))
    return 100. * correct / total


# +
results = {
    'args': args,
    'params': params,
    'train_acc': [],
    'test_acc': [],
}

total_epochs = 350

for epoch in range(total_epochs):
    logging.info('epoch %d', epoch)
    net.pre = args.dp * epoch / total_epochs
    results['train_acc'].append(train(epoch))

    lr_decay.step()
    results['test_acc'].append(test(epoch))
# -

test(total_epochs)

store_file = os.path.join(args.save, 'dataset_' + str(args.dataset) + '_num_routing_' + str(args.num_routing) + \
                              '_backbone_' + args.backbone + '.dct')
pickle.dump(results, open(store_file, 'wb'))