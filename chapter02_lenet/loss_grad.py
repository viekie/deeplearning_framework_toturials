#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/1

import torch
from net_class import net

learning_rate = 0.01

x = torch.randn(1, 1, 32, 32)
output = net(x)
# 随机初始化一个target，模拟
target = torch.rand_like(output)

loss_fn = torch.nn.MSELoss()
loss = loss_fn(output, target)

net.zero_grad()
print(net.conv1.bias.grad)

loss.backward()

for p in net.parameters():
    p.data -= p.grad.data * learning_rate

print(net.conv1.bias.grad)