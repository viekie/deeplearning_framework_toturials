#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/8/31

import torch
import torch.nn as nn
import torch.nn.functional as F


def num_flat_features(x):
    """
    此函数用于计算多维矩阵进行扁平化，最后一维的数量
    如 x
    :param x: 输入矩阵，如（16, 5, 3, 2)
    :return: 5*3*2
    """
    size = x.size()[1:]
    num_flat = 1
    for s in size:
        num_flat = num_flat * s
    return num_flat


class Net(nn.Module):
    """
    定义LeNet的网络主体
    """

    def __init__(self):
        """
       初始化网络结构
       """
        super(Net, self).__init__()
        # 定义一个接收1个输入通道，采用6个channel卷积核为5*5的卷积操作对象
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        # 定义一个接收6个输入通道，采用16个channel卷积核为5*5的卷积操作对象
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        # 定义一个接收16个输入通道，采用120个channel卷积核为5*5的卷积操作对象
        self.conv3 = nn.Conv2d(16, 120, (5, 5))
        # 定义一个输入120*84的全连接操作
        self.fc1 = nn.Linear(120, 84)
        # 定义一个输入的84*10的全连接操作
        self.fc2 = nn.Linear(84, 10)

    def forward(self, input):
        """
        定义LeNet的前项操作
        :param input:
        :return:
        """

        x = F.max_pool2d(F.relu(self.conv1(input)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义,下面的2 等价于(2,2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        num_features = num_flat_features(x)
        x = x.view(-1, num_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


net = Net()
# print(net)
# params = net.parameters()
# params_list = list(params)
# print(params_list[0])
# print('*' * 70)








