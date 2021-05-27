#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
'''Capsule in PyTorch
TBD
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from parameters import *


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
                stride = 2 if reduction and j < 2 else 1  # 步幅，reduction cell的前两个为2，其余为1
                op = MixedOp(C, stride)  # 第i个节点第j个前驱的混合操作类
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


class NasPreCaps(nn.Module):  # 继承父类nn.Module，主要设计init和上层forward，利用父类内部的_call_和底层forward函数自动搭建网络，调用时传参为上层forward参数

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3):
        super(NasPreCaps, self).__init__()
        self._C = C  # 初始通道数
        self._layers = layers  # 细胞总层数,DARTS为8个cell，即8层
        self._steps = steps  # cell的中间节点数，即4个中间连接节点状态需要确定
        self._multiplier = multiplier  # 一个cell的中间节点数，也是扩充通道的倍数

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            # 参数依次为，输入通道数，输出通道数，卷积核大小(边长或高宽)，stride为滑动步长默认为1，padding为增值为padding_mode的边距大小
            nn.BatchNorm2d(C_curr)  # C_curr为(N,C,H,W)的C；其中N代表数量，C代表信道channel，H代表高度，W代表宽度
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  # 初始化通道数，第一个cell的k-2,k-1相同
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 2]:
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
        model_new = NasPreCaps(self._C, self._num_classes, self._layers, self._criterion).cuda()
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

    # def _loss(self, input, target):
    #     logits = self(input)
    #     return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        # 存储两种cell（是否reduction）中间节点的所有操作对应的权重矩阵
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)
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

#### Simple Backbone ####
class simple_backbone(nn.Module):
    def __init__(self, cl_input_channels, cl_num_filters, cl_filter_size,
                 cl_stride, cl_padding):
        super(simple_backbone, self).__init__()

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


class BasicBlock(nn.Module):
    expansion = 1  # 残差网络参数

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 输入为batchsize * in_planes * x * x；输出为batchsize * planes * x' * x'
        # 例1：input = batchsize * 64 * 32 * 32；in_planes = planes = 64
        # 例2：in_planes = 64， planes = 128，stride = 2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 例1：tensor = batchsize * 64 * 32 * 32；例2：tensor = batchsize * 64 * 32 * 32；
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 输出为batchsize * planes * x'' * x''，例1为 batchsize * 64 * 32 * 32 例2为batchsize * 128 * 16 * 16
        self.shortcut = nn.Sequential()  # 创建一个空的module操作，表示一个捷径操作
        if stride != 1 or in_planes != self.expansion * planes:  # 注意到在forward函数中使用+=进行调用，因此使用输入数据维度32 * 32，生成16 * 16
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 经过了Relu的卷积+正则化
        out = self.bn2(self.conv2(out))  # 卷积+正则化
        out += self.shortcut(x)  # 增加捷径部分，即残差部分
        out = F.relu(out)  # Relu变换
        return out


class resnet_backbone(nn.Module):

    def __init__(self, cl_input_channels, cl_num_filters,
                 cl_stride):
        super(resnet_backbone, self).__init__()
        self.in_planes = 64

        def _make_layer(block, planes, num_blocks, stride):
            # _make_layer(block=BasicBlock, planes=64, num_blocks=3, stride=1), # num_blocks=2 or 3
            # input = batchsize * 64 * 32 * 32
            # _make_layer(block=BasicBlock, planes=cl_num_filters = 128, num_blocks=4, stride=cl_stride = 2), num_blocks=2 or 4
            #
            strides = [stride] + [1] * (
                    num_blocks - 1)  # strides = [stride, 1, 1...]，第一个stride由用户定义，随后的stride默认为1；用户定义的stride决定了每个信道的数据矩阵大小，为1则不改变，为2则缩小一半
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion  # 更新输入为上一层输出
            return nn.Sequential(*layers)

        self.pre_caps = nn.Sequential(
            # 输入为batchsize * 3 * 32 * 32；输出为batchsize * 64 * 32 * 32
            nn.Conv2d(in_channels=cl_input_channels,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(64),  # 数据通道数为64，对每个通道分别求均值和方差，并进行归一化
            nn.ReLU(),
            # 输入为batchsize * 64 * 32 * 32；输出为 batchsize * 128 * x * x
            _make_layer(block=BasicBlock, planes=64, num_blocks=3, stride=1),  # num_blocks=2 or 3, 可以改变内部block数量
            _make_layer(block=BasicBlock, planes=cl_num_filters, num_blocks=4, stride=cl_stride),  # num_blocks=2 or 4
        )


    def forward(self, x):
        out = self.pre_caps(x)  # x is an image
        return out


#### Capsule Layer ####
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


class CapsuleCONV(nn.Module):
    r"""Applies as a capsule convolutional layer.
    TBD
    """

    # 以第一次迭代为例，张量为batchsize * 512 * 16 * 16
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules,
                 kernel_size, stride, matrix_pose, dp, coordinate_add=False):
        super(CapsuleCONV, self).__init__()
        self.in_n_capsules = in_n_capsules  # 32
        self.in_d_capsules = in_d_capsules  # 16
        self.out_n_capsules = out_n_capsules  # 32
        self.out_d_capsules = out_d_capsules  # 16
        self.kernel_size = kernel_size  # 3
        self.stride = stride  # 2
        self.matrix_pose = matrix_pose  # TRUE
        self.coordinate_add = coordinate_add  # False
        '''
            "type" : "CONV",
            "num_caps": 32,
            "caps_dim": 16,
            "kernel_size": 3,
            "stride": 2,
            "matrix_pose": true,
            "out_img_size": 7
        '''
        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))  # 将输入维度16开方转换成4 * 4矩阵
            self.weight_init_const = np.sqrt(
                out_n_capsules / (self.sqrt_d * in_n_capsules * kernel_size * kernel_size))  # 权重系数初始化
            self.w = nn.Parameter(self.weight_init_const * torch.randn(kernel_size, kernel_size,
                                                                       in_n_capsules, self.sqrt_d, self.sqrt_d,
                                                                       out_n_capsules))  # 3 * 3 * 32.in_n_capsules * (4 * 4).in_d_capsules * 32.out_n_capsules
        else:
            self.weight_init_const = np.sqrt(
                out_n_capsules / (in_d_capsules * in_n_capsules * kernel_size * kernel_size))
            self.w = nn.Parameter(self.weight_init_const * torch.randn(kernel_size, kernel_size,
                                                                       in_n_capsules, in_d_capsules, out_n_capsules,
                                                                       out_d_capsules))  # 3 * 3 * 32.in_n_capsules * 16.in_d_capsules * 32.out_n_capsules
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)  # 对各个胶囊的输出进行归一化
        self.dropout_rate = dp
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, \
                    kernel_size={}, stride={}, coordinate_add={}, matrix_pose={}, weight_init_const={}, \
                    dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules,
            self.kernel_size, self.stride, self.coordinate_add, self.matrix_pose, self.weight_init_const,
            self.dropout_rate
        )

    def input_expansion(self, input):
        # input has size [batch x num_of_capsule x height x width x  x capsule_dimension]
        unfolded_input = input.unfold(2, size=self.kernel_size, step=self.stride).unfold(3, size=self.kernel_size,
                                                                                         step=self.stride)
        # unfold的参数为dim, size, step
        # 第一次在dim=2上进行展开，使得原始维度转化为[batch x num_of_capsule x (height/step).8 x width x capsule_dimension x size.3]
        # 第二次在dim=3上进行展开，使得原始维度转化为[batch x num_of_capsule x (height/step).8 x (width/step).8 x capsule_dimension x size.3 x size.3]
        unfolded_input = unfolded_input.permute([0, 1, 5, 6, 2, 3, 4])
        # output has size [batch x num_of_capsule x kernel_size x kernel_size x h_out x w_out x capsule_dimension]
        return unfolded_input

    def forward(self, input, num_iter, next_capsule_value=None):
        # k,l: kernel size
        # h,w: output width and length
        inputs = self.input_expansion(
            input)  # 张量展开运算，相当于卷积中的“卷”操作，输入为100, 32, 14, 14, 16，输出为100, 32, 3, 3, 7(8), 7(8), 16

        if self.matrix_pose:
            w = self.w  # klnxdm 3 * 3 * 32.in_n_capsules * (4 * 4).in_d_capsules * 32.out_n_capsules
            _inputs = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3], \
                                  inputs.shape[4], inputs.shape[5], self.sqrt_d, self.sqrt_d)  # bnklmhax 将输出维度进行开方，形成矩阵
        else:
            w = self.w

        if next_capsule_value is None:  # 每次前向传递均初始化为none
            query_key = torch.zeros(self.in_n_capsules, self.kernel_size, self.kernel_size,
                                    self.out_n_capsules).type_as(inputs)  # 全零张量维度为 输入胶囊数 * 3 * 3 * 输出胶囊数
            query_key = F.softmax(query_key, dim=3)  # 对第4维数据进行softmax，即对输出胶囊的数个单元进行归一化

            if self.matrix_pose:
                next_capsule_value = torch.einsum('nklm, bnklhwax, klnxdm->bmhwad', query_key,
                                                  _inputs,
                                                  w)  # n,k,l,m,b,h,w,a,x,d化为bmhwad,即batchsize,out_n_capsules,height*,weight*,(4 * 4)
                # query_key：输入胶囊数 * 3 * 3 * 输出胶囊数
                # _inputs：100, 32, 3, 3, 7(8), 7(8), 4, 4(16)
                # w：3 * 3 * 32.in_n_capsules * (4 * 4).in_d_capsules * 32.out_n_capsules
                # 相同输入胶囊数32,相同核大小3*3,输出维度开方矩阵进行乘积
            else:
                next_capsule_value = torch.einsum('nklm, bnklhwa, klnamd->bmhwd', query_key,
                                                  inputs, w)
        else:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0], \
                                                             next_capsule_value.shape[1], next_capsule_value.shape[2], \
                                                             next_capsule_value.shape[3], self.sqrt_d, self.sqrt_d)
                _query_key = torch.einsum('bnklhwax, klnxdm, bmhwad->bnklmhw', _inputs, w,
                                          next_capsule_value)  # b n k l h w a x d m
                # _inputs：100, 32, 3, 3, 7(8), 7(8), 4, 4(16)
                # w：3 * 3 * 32.in_n_capsules * (4 * 4).in_d_capsules * 32.out_n_capsules
                # next_capsule_value：batchsize,out_n_capsules,height*,weight*,(4 * 4)
                # 开方矩阵维度*2，输出维度开放矩阵进行乘积
            else:
                _query_key = torch.einsum('bnklhwa, klnamd, bmhwd->bnklmhw', inputs, w,
                                          next_capsule_value)
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=4)  # 第五维，即输出胶囊个数进行softmax归一化
            query_key = query_key / (torch.sum(query_key, dim=4, keepdim=True) + 1e-10)

            if self.matrix_pose:
                next_capsule_value = torch.einsum('bnklmhw, bnklhwax, klnxdm->bmhwad', query_key,
                                                  _inputs, w)  # 沿输入数量，核矩阵，开方乘积
            else:
                next_capsule_value = torch.einsum('bnklmhw, bnklhwa, klnamd->bmhwd', query_key,
                                                  inputs, w)

        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0], \
                                                             next_capsule_value.shape[1], next_capsule_value.shape[2], \
                                                             next_capsule_value.shape[3],
                                                             self.out_d_capsules)  # 末尾输出维度开方矩阵化为输出向量
                next_capsule_value = self.nonlinear_act(next_capsule_value)  # 输出维度进行归一化
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)

        return next_capsule_value


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    # params(iterable)用于迭代优化的参数;lr(float, optional)学习率，默认为1e-3；betas(Tuple[float, float], optional)用于计算梯度的平均和平方的系数，默认0.9, 0.999；eps提高数字稳定性而增加到分母的一个项，默认维1e-8;weight_decay默认为0
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)


  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()
