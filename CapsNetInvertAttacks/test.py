import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from src import capsule_model
# from utils import progress_barv-
import pickle
import json

from art.attacks.evasion import FastGradientMethod # test acc: 23.64
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch # test acc: 1.52
from art.estimators.classification import PyTorchClassifier

from datetime import datetime

# +
parser = argparse.ArgumentParser(description='Training Capsules using Inverted Dot-Product Attention Routing')

parser.add_argument('--resume_dir', '-r', default='', type=str, help='dir where we resume from checkpoint')
parser.add_argument('--num_routing', default=1, type=int, help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset. CIFAR10 or CIFAR100.')
parser.add_argument('--backbone', default='resnet', type=str, help='type of backbone. simple or resnet')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers. 0 or 2')
parser.add_argument('--config_path', default='./configs/resnet_backbone_CIFAR10.json', type=str, help='path of the config')
parser.add_argument('--debug', action='store_true',
                    help='use debug mode (without saving to a directory)')
parser.add_argument('--sequential_routing', action='store_true', help='not using concurrent_routing')


parser.add_argument('--lr', default=0.1, type=float, help='learning rate. 0.1 for SGD')
parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
# -

args = parser.parse_args()
'''
class Modclass(nn.Module):
    def __init__(self):
        super(Modclass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x


model = Modclass()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print("Model's state_dict")
for para_tensor in model.state_dict():
    print(para_tensor, "\t", model.state_dict()[para_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

torch.save(model.state_dict(), "results/testtensor.pth")

model1 = Modclass()
model1.load_state_dict(torch.load("results/testtensor.pth"))
model1.eval()
print("Model1's state_dict")
for para_tensor in model1.state_dict():
    print(para_tensor, "\t", model.state_dict()[para_tensor].size())
'''
image_dim_size = 32
with open(args.config_path, 'rb') as file:
    params = json.load(file)
net = capsule_model.CapsModel(image_dim_size,
                    params,
                    args.backbone,
                    args.dp,
                    args.num_routing,
                    sequential_routing=args.sequential_routing)


print(torch.__version__)
print("\n")
print("cuda version needed to be",torch.version.cuda, "\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device is ",device)

checkpoint = torch.load("results/ckpt.pth", map_location=device)
# print(checkpoint['acc'])
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
net.load_state_dict(checkpoint['net'])
net.eval()
# for para_tensor in net.state_dict():
    # print(para_tensor, "\t", net.state_dict()[para_tensor].size())


transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
testset = getattr(torchvision.datasets, args.dataset)(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

loss_func = nn.CrossEntropyLoss()

classifier = PyTorchClassifier(
    model=net.module,
    # clip_values=
    loss=loss_func,
    # optimizer =
    input_shape=(3, 32, 32),
    nb_classes=10,
)

attack = FastGradientMethod(estimator=classifier, eps=0.2)
attack2 = ProjectedGradientDescentPyTorch(estimator=classifier)


'''
for batch_idx, (inputs, targets) in enumerate(testloader):
    x_test_adv = attack.generate(x=inputs)
    print("batch_idx = ", batch_idx, "\t")
    print("convert successfully\n!")

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            print(inputs.size())

            targets = targets.to(device)

            v = net.module(inputs)

            loss = loss_func(v, targets)

            test_loss += loss.item()

            _, predicted = v.max(dim=1)

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total

    return acc
'''


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
            # print(inputs.size())
            inputs = attack2.generate(x=inputs)
            print("batch_idx = ", batch_idx, "\t")
            print("convert successfully!\n")
            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # v = net.module(inputs)
            v = net(inputs)

            loss = loss_func(v, targets)

            test_loss += loss.item()

            _, predicted = v.max(dim=1)

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total

    return acc

# print("Total accuracy under FGSM attack is", test(1))
print("Total accuracy under PGD attack is", test(1))

