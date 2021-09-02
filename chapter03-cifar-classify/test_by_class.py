#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/1

import torch
import dataloader
from cifar_net import net

class_type = [0 for i in range(10)]
class_total = [0 for i in range(10)]

net.load_state_dict(torch.load('./model/cifar_net.pth'))

with torch.no_grad():
    for data in dataloader.test_data_loader:
        images, labels = data
        output = net(images)
        _, predict = torch.max(output, 1)
        corr = (predict == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_type[label] += corr[i].item()
            class_total[label] += 1


[print('{} avg accuracy is {}'.format(dataloader.classes[i],
                                      class_type[i]/class_total[i])) for i in range(10)]