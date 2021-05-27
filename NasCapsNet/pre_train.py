import os
import sys
import time
import glob
import numpy as np

import logging
import argparse

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import json

from model import *
from parameters import *

parser = argparse.ArgumentParser("NAS-CapsNet-cifar10")
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='NASCAPS', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers. 0 or 2')

args = parser.parse_args()

# args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = '{}-search-test'.format(args.save)


if not os.path.exists(args.save):
    os.mkdir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES_NUMBER = 10
CLASS_CIFAR = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    # preparing gpu device preparing
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    logging.info("device is %s", device)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    # preparing the dataset
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

    train_set = dset.CIFAR10(root='../data', train=True, download=True, transform=transform_train)

    test_set = dset.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    # divide the training dataset into two part, i.e. training part and validating part
    num_train = len(train_set)
    indices = list(range(num_train))  # 建立所有数据集的索引
    split = int(np.floor(args.train_portion * num_train))  # 数据集分割成两部分，包括训练部分和验证部分

    # 迭代器将数据分批放入训练网络中；还包括shuffle(不能和sampler共用)，batch_sampler一次只返回一个batch(互斥)
    train_queue = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),  # 自定义取样本的策略，范围为训练部分
        pin_memory=True,  # 是否拷贝tensors到cuda中的固定内存中
        num_workers=2)  # 进程数量，0意味着都被load进主进程

    valid_queue = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),  # 范围为测试部分
        pin_memory=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.num_workers)


    # set models for learning task

    with open('network_paras.json', 'rb') as file:
        params = json.load(file)  # 将json中的对象转换成python中的dict

    model = CapsModel(3, # image_dim_size,
                    params,
                    0.0, #args.dp,
                    1, # args.num_routing,
                    )

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        print("running well, exit!")
        return 0

if __name__ == '__main__':
  main()
