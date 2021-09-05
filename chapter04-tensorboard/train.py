#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import dataloader
import loss
import utils
import optimizer
from net import model
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

epoches = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataloader, test_dataloader = dataloader.get_data_loader('./data')
entropy_loss_fn = loss.get_cross_entropy_loss_fn()
sgd_optim = optimizer.get_sgd_optimizer(model, lr=0.01)

model.to(device)
model = nn.DataParallel(model)

total_loss = 0.0

writer = SummaryWriter('./log/fashion_minist_01')

for epoch in range(epoches):

    for i, data in enumerate(train_dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        y = model.forward(images)
        model.zero_grad()
        batch_loss = entropy_loss_fn(y, labels)
        batch_loss.backward()
        sgd_optim.step()

        total_loss += batch_loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 2000))
            writer.add_scalar('train loss', total_loss/2000, epoch*len(train_dataloader) + 1)
            writer.add_figure('prediction vs actual',
                              utils.plot_classes_preds(model, images, labels),
                              global_step=epoch * len(train_dataloader) + i)
            total_loss = 0.0

path = './model/fasion.pth'
torch.save(model.state_dict(), path)
print('finish')