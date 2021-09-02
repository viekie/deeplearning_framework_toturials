#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/1

import torch
import torch.optim as optim
from net_class import net

x = torch.randn(1, 1, 32, 32)
output = net(x)
target = torch.randn(10)
target = target.view(1, -1)

credition = torch.nn.MSELoss()
loss = credition(target, output)

optimzer = optim.SGD(net.parameters(), lr=0.01)
optimzer.zero_grad()
print(net.conv1.bias.data)
print('*' * 70)
loss.backward()
print(net.conv1.bias.grad.data)
print('*' * 70)
optimzer.step()
print(net.conv1.bias.data)
print('*' * 70)
