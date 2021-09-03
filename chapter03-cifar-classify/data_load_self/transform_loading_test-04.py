#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

data_set = datasets.ImageFolder(root='../data/faces', transform=data_transform)

data_loader = torch.utils.data.DataLoader(data_set, batch=4, shuffle=False, num_workers=0)