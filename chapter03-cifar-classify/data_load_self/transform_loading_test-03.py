#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import os
import torch
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision import transforms, utils
from data_load_self.transform_tools import Rescale, ToTensor, RandomCrop
from data_load_self.FaceLandmarksDataset import FaceLandmarksDataset

data_set = FaceLandmarksDataset('../data/faces',
                                 landmark_csv='../data/faces/face_landmarks.csv',
                                 transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
data_loader = torch.utils.data.DataLoader(data_set, batch_size=4, shuffle=True, num_workers=0)


def show_landmarks_batch(samples):
    batch_images, batch_landmarks = samples['image'], samples['landmarks']
    batch_size = len(batch_images)
    img_size = batch_images.size(2)
    grid_border_size = 2
    grid = utils.make_grid(batch_images)
    plt.imshow(grid.numpy().transpose((1 ,2, 0)))

    for idx in range(batch_size):
        plt.scatter(batch_landmarks[idx, :, 0].numpy() + idx * img_size + (idx + 1) * grid_border_size,
                    batch_landmarks[idx, :, 1].numpy() + grid_border_size,
                    s=10, c='r', marker='.')
        plt.title('batch from dataloader')


for i, samples in enumerate(data_loader):
    print('number {}, image size {}, landmark size {}'.format(i, samples['image'].size(),
                                                              samples['landmarks'].size()))

    if i == 3:
        plt.figure()
        show_landmarks_batch(samples)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
