#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import torch
from torchvision import transforms, datasets


def get_data_loader(root_dir):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root=root_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root=root_dir, train=False, transform=transform, download=True)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=0)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=4, num_workers=0)

    return train_data_loader, test_data_loader


