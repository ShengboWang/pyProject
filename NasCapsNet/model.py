import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.autograd import Variable
import math
from parameters import *

import numpy as np


class ResNetPreCapsule(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetPreCapsule, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # (b_size,64,32,32)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # (b_size,64,32,32)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # (b_size,128,16,16)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:  # 遍历搜索空间的操作
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:  # 池化后加一个batch操作
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)  # 添加到模型链表中，保留参数

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))  # 操作加权和，通过改变权重来实现变量松弛和搜索空间连续化


class Cell(nn.Module):  # 继承父类的_call_和底层forward函数
    # 一个cell由7个nodes组成，分为两个input nodes，四个intermediate nodes，1个output node
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        # 第一个input nodes的结构，取决于前一个细胞是否为reduction，且输入为k-2个细胞的输出，即通道数为C_prev_prev,输出通道为C
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        # 第二个input结构
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()  # 列表的形式存储网络，同一元素共享parameters，不自带forward函数
        self._bns = nn.ModuleList()

        for i in range(self._steps):  # 遍历四个中间节点，构建混合操作
            for j in range(2 + i):  # 遍历当前节点i的所有前驱节点，即k-2,k-1cell的输出和i个前驱nodes，共2+i
                stride_mix = 2 if reduction and j < 2 else 1  # 步幅，reduction cell的前两个为2，其余为1
                op = MixedOp(C, stride_mix)  # 第i个节点第j个前驱的混合操作类
                self._ops.append(op)  # ，按照节点编号、前驱节点编号构建模型链表

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)  # 第一个输入节点操作
        s1 = self.preprocess1(s1)  # 第二个输入节点操作

        states = [s0, s1]  # 第一个中间节点的前驱节点，即两个输入节点的输出状态
        offset = 0
        for i in range(self._steps):  # 遍历中间节点的操作
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in
                    enumerate(states))  # for j, h in enumerate(states) 的输出为 j=0, h=s0; j=1, h=s1
            offset += len(states)  # 偏移量累加
            states.append(s)  # 前驱节点累加

        return torch.cat(states[-self._multiplier:],
                         dim=1)  # 索引L[-a:]表示从length(L)-a 至末尾的部分，此处为四个中间节点的输出状态通过cat操作作为一个cell的最终输出，dim=1使得输出向量拉长，即concat操作，输出通道数变为原先的4倍


# model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
class NASPreCaps(nn.Module):  # 继承父类nn.Module，主要设计init和上层forward，利用父类内部的_call_和底层forward函数自动搭建网络，调用时传参为上层forward参数

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(NASPreCaps, self).__init__()
        self._C = C  # 初始通道数
        self._num_classes = num_classes  # 类别数量，cifar10为10，cifar100为100
        self._layers = layers  # 细胞总层数,DARTS为8个cell，即8层
        self._criterion = criterion  # 评价指标函数，此处即交叉熵损失函数
        self._steps = steps  # cell的中间节点数，即4个中间连接节点状态需要确定
        self._multiplier = multiplier  # 一个cell的中间节点数，也是扩充通道的倍数

        C_curr = stem_multiplier * C
        # (b_size,128,16,16) nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            # 参数依次为，输入通道数，输出通道数，卷积核大小(边长或高宽)，stride为滑动步长默认为1，padding为增值为padding_mode的边距大小
            nn.BatchNorm2d(C_curr)  # C_curr为(N,C,H,W)的C；其中N代表数量，C代表信道channel，H代表高度，W代表宽度
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  # 初始化通道数，第一个cell的k-2,k-1相同
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:  # //表示整除，取1/3,2/3处的细胞作为reduction cell，其余为normal cell
                C_curr *= 2  # 每过一个reduction cell, 整体网络的通道数*2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr  # 每个cell输出将信道扩充4倍

        self._initialize_alphas()  # 架构参数初始化

    def new(self):
        model_new = NASPreCaps(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = s1
        return out

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        # 存储两种cell（是否reduction）中间节点的所有操作对应的权重矩阵
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


class simple_backbone(nn.Module):
    def __init__(self, cl_input_channels, cl_num_filters, cl_filter_size,
                 cl_stride, cl_padding):
        super(simple_backbone, self).__init__()
        # Conv2d详解：
        # 输入格式 input(N batchsize, C-in channel, H-in height, W-in weight)
        # 输出格式 output(N, C-out, H-out, W-out)
        # 参数:in_channels输入图像的通道数；out_channels输出图像的通道数（即滤波器的个数）；kernel卷积核的大小（即滤波窗口大小）；stride为步长。padding为输入数据的填充

        self.pre_caps = nn.Sequential(
            # 输入为batchsize * 3 * 32 * 32；输出为batchsize * 128 * 16 * 16 (即out_img_size)
            nn.Conv2d(in_channels=cl_input_channels,  # 3
                      out_channels=cl_num_filters,  # 128
                      kernel_size=cl_filter_size,  # 3
                      stride=cl_stride,  # 2
                      padding=cl_padding),  # 1
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.pre_caps(x)  # x is an image
        return out


class CapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """

    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, matrix_pose, dp):
        super(CapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules  # 输入数量
        self.in_d_capsules = in_d_capsules  # 输入维度
        self.out_n_capsules = out_n_capsules  # 输出数量
        self.out_d_capsules = out_d_capsules  # 输出维度
        self.matrix_pose = matrix_pose  # pose是否是矩阵结构

        if matrix_pose:  # 是矩阵结构，则维度是输入输出向量的开方
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))  # 输入维度开方并取整
            self.weight_init_const = np.sqrt(
                out_n_capsules / (self.sqrt_d * in_n_capsules))  # 权重初始化常数 = 开方(输出数量 / (输入维度开方 * 输入数量))
            self.w = nn.Parameter(self.weight_init_const * \
                                  torch.randn(in_n_capsules, self.sqrt_d, self.sqrt_d,
                                              out_n_capsules))  # 初始化参数矩阵，四维输入数量 * 开方 * 开方 * 输出数量

        else:  # 若不是矩阵结构，则维度为向量原长
            self.weight_init_const = np.sqrt(out_n_capsules / (in_d_capsules * in_n_capsules))
            self.w = nn.Parameter(self.weight_init_const * \
                                  torch.randn(in_n_capsules, in_d_capsules, out_n_capsules,
                                              out_d_capsules))  # 初始化参数矩阵，四维输入数量 * 输入维度 * 输出数量 * 输出维度
        self.dropout_rate = dp
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)  # 使用层归一化作为非线性激活函数
        self.drop = nn.Dropout(self.dropout_rate)  # 随机dropout防止过拟合，概率为0.5
        self.scale = 1. / (out_d_capsules ** 0.5)  # 1 / 输出维度开方

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
            weight_init_const={}, dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
            self.weight_init_const, self.dropout_rate
        )

    def forward(self, input, num_iter, next_capsule_value=None):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        if len(input.shape) == 5:
            input = input.permute(0, 4, 1, 2, 3)  # 张量表下标替换
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)  # 重新整合，保留原数据0，4两维度，123整合成一维
            input = input.permute(0, 2, 1)  # 下标替换，原input整合成{0，[123], 4}

        if self.matrix_pose:
            # w:四维输入数量 * 开方 * 开方 * 输出数量
            w = self.w  # nxdm 本层胶囊数 * x * 下一层维度 * 下一层胶囊数
            _input = input.view(input.shape[0], input.shape[1], self.sqrt_d,
                                self.sqrt_d)  # bnax batchsize * 本层胶囊数 * 本层维度(开方) * x(开方)
        else:
            w = self.w

        if next_capsule_value is None:
            query_key = torch.zeros(self.in_n_capsules, self.out_n_capsules).type_as(
                input)  # nm 定义一个输入输出矩阵，数据类型转换为与input相同
            query_key = F.softmax(query_key, dim=1)  # 矩阵softmax

            if self.matrix_pose:
                # query_key 输入n输出m的转换矩阵
                # _input batchsize b * 本层胶囊数 n * 本层维度(开方) a * x(开方)
                # w:四维输入数量n * 开方x * 开方d * 输出数量m
                next_capsule_value = torch.einsum('nm, bnax, nxdm->bmad', query_key, _input,
                                                  w)  # einsum为爱因斯坦求和约定，使得矩阵或张量按照axis表示的方向乘积并输出求和：保留b(batchsize), 下层胶囊数量 * 本层向量维度 (m*a), 下层向量维度(d)
            else:
                next_capsule_value = torch.einsum('nm, bna, namd->bmd', query_key, input, w)
        else:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0],
                                                             next_capsule_value.shape[1], self.sqrt_d, self.sqrt_d)
                _query_key = torch.einsum('bnax, nxdm, bmad->bnm', _input, w, next_capsule_value)
            else:
                _query_key = torch.einsum('bna, namd, bmd->bnm', input, w, next_capsule_value)
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=2)
            query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) + 1e-10)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('bnm, bnax, nxdm->bmad', query_key, _input,
                                                  w)
            else:
                next_capsule_value = torch.einsum('bnm, bna, namd->bmd', query_key, input,
                                                  w)

        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:  # 若capsule_value不是一维的张量，则执行算法
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0],
                                                             next_capsule_value.shape[1],
                                                             self.out_d_capsules)  # 输出维度的开方还原
                next_capsule_value = self.nonlinear_act(next_capsule_value)  # 用非线性激活函数进行层归一化操作
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)
        return next_capsule_value


class CapsModel(nn.Module):
    '''
    net = capsule_model.CapsModel(image_dim_size,
                    params,
                    args.backbone,
                    args.dp,
                    args.num_routing,
                    sequential_routing=args.sequential_routing)
    '''

    def __init__(self,
                 image_dim_size,
                 params,
                 dp,
                 num_routing,
                 sequential_routing=True):

        super(CapsModel, self).__init__()
        #### Parameters
        self.sequential_routing = sequential_routing

        ## Primary Capsule Layer
        self.pc_num_caps = 32  # params['primary_capsules']['num_caps']
        self.pc_caps_dim = 16  # params['primary_capsules']['caps_dim']
        self.pc_output_dim = 16  # params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing  # >3 may cause slow converging

        ## Backbone (before capsule)
        self.pre_caps = simple_backbone(3,  # params['backbone']['input_dim'],
                                        16,  # params['backbone']['output_dim'],
                                        3,  # params['backbone']['kernel_size'],
                                        2,  # params['backbone']['stride'],
                                        1)  # params['backbone']['padding'])

        ## Primary Capsule Layer (a single CNN)
        # 张量为batchsize * 128 * 16 * 16
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],  # 128
                                  out_channels=params['primary_capsules']['num_caps'] * \
                                               params['primary_capsules']['caps_dim'],  # 32.胶囊数 * 16.输出维度 = 512
                                  kernel_size=params['primary_capsules']['kernel_size'],  # 1
                                  stride=params['primary_capsules']['stride'],  # 1
                                  padding=params['primary_capsules']['padding'],  # 0
                                  bias=False)
        # 张量为batchsize * 512 * 16 * 16
        # self.pc_layer = nn.Sequential()

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])  # 每16维度进行层归一化，即各个胶囊输出进行归一化
        # 张量为batchsize * 512 * 16 * 16
        ## Main Capsule Layers
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if i == 0:
                in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] * \
                            params['primary_capsules']['out_img_size']
                in_d_caps = params['primary_capsules']['caps_dim']
            elif params['capsules'][i - 1]['type'] == 'FC':
                in_n_caps = params['capsules'][i - 1]['num_caps']
                in_d_caps = params['capsules'][i - 1]['caps_dim']
            elif params['capsules'][i - 1]['type'] == 'CONV':
                in_n_caps = params['capsules'][i - 1]['num_caps'] * params['capsules'][i - 1]['out_img_size'] * \
                            params['capsules'][i - 1]['out_img_size']
                in_d_caps = params['capsules'][i - 1]['caps_dim']
            self.capsule_layers.append(
                CapsuleFC(in_n_capsules=in_n_caps,
                          in_d_capsules=in_d_caps,
                          out_n_capsules=params['capsules'][i]['num_caps'],
                          out_d_capsules=params['capsules'][i]['caps_dim'],
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp
                          )
            )

        ## Class Capsule Layer
        if not len(params['capsules']) == 0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] * \
                            params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] * \
                        params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            CapsuleFC(in_n_capsules=in_n_caps,
                             in_d_capsules=in_d_caps,
                             out_n_capsules=params['class_capsules']['num_caps'],
                             out_d_capsules=params['class_capsules']['caps_dim'],
                             matrix_pose=params['class_capsules']['matrix_pose'],
                             dp=dp
                             )
        )

        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'],
                                  1)  # 全连接层，y = xA^T + b, in_feature为16， out_features = 1; 默认bias = True
        # different classifier for different capsules
        # self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)

        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c)  # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1)  # 100, 14, 14, 512
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps,
                   self.pc_caps_dim)  # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4)  # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)  # capsule_utils.squash(u) 整体目标：通过变换得到胶囊数为32，16维输出向量，并进行输出归一化，不改变输出维度

        ## Main Capsule Layers
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            # 使用primary层归一化的输出作为capsule_value的初始值，进行第一次迭代
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val)  # get the capsule value for next layer

            # second to t iterations
            # perform the routing between capsule layers
            # 首先利用第一次迭代之后的val进行更新，然后更新val序列进行t-1次路由迭代
            for n in range(self.num_routing - 1):
                _capsule_values = [init_capsule_value]  # 保证第层之前的接口为primary的非线性输出
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n,
                                                          capsule_values[i + 1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing - 1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)

        ## After Capsule
        out = capsule_values[-1]  # 输出最后一次capsule的val
        out = self.final_fc(out)  # fixed classifier for all capsules
        out = out.squeeze()  # fixed classifier for all capsules
        # out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules

        return out
