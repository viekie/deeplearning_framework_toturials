#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/1

import time
import torch
import dataloader
from cifar_net import net
import torch.optim as optim

start = time.time()
learning_rate = 0.001
epoches = 20

# 定义损失函数
cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# 开始训练

for epoch in range(epoches):
    total_loss = 0.0
    for i, data in enumerate(dataloader.train_data_loader):
        x, y = data
        output = net(x)
        net.zero_grad()
        loss = cross_entropy_loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 2000))
            total_loss = 0.0

end = time.time()
print('total cost:  {} ', end-start)
PATH = './model/cifar_net.pth'
torch.save(net.state_dict(), PATH)

print('*' * 70)
print('*' * 28 + ' train finish ' + '*' * 28)
print('*' * 70)
net.par

