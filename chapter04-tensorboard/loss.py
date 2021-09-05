#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import torch.nn as nn


def get_cross_entropy_loss_fn():
    criterion = nn.CrossEntropyLoss()
    return criterion
