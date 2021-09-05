#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import utils
import torch
import dataloader
from net import model
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class_pred = []
class_prob = []

train_data_loader, test_data_loader = dataloader.get_data_loader('./data')
model.load_state_dict(torch.load('./model/fashion.pth'))

total = 0.0
correct = 0.0

with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        output = model(images)
        pred_class_prob = [F.softmax(el, dim=0) for el in output]
        _, pred_class = torch.max(output, 1)
        class_pred.append(pred_class)
        class_prob.append(pred_class_prob)

        correct += (pred_class == labels).sum().item()
        total += labels.size()[0]

test_probs = torch.cat([torch.stack(batch) for batch in class_prob])
test_preds = torch.cat(class_pred)

print('avg acc is {}'.format(correct/total))

writer = SummaryWriter('./log/fashion_minist_01')
for i in range(len(utils.get_labels())):
    utils.add_pr_curve_tensorboard(writer, i, test_probs, test_preds)

writer.close()