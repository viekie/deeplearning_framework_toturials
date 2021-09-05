#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import torch


def get_sgd_optimizer(model, lr=0.01):
    sgd_optim = torch.optim.SGD(model.parameters(), lr=lr)
    return sgd_optim
