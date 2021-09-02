#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/1

import torch
import dataloader

from cifar_net import net
net.load_state_dict(torch.load('./model/cifar_net.pth'))

total = 0
correct = 0
# print([True, True, False, True].sum())
with torch.no_grad():
    for data in dataloader.test_data_loader:
        images, labels = data
        outputs = net(images)
        _, predict = torch.max(outputs.data, 1)
        total += labels.size()[0]
        correct += (predict == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
