#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision import transforms
from data_load_self.transform_tools import Rescale, RandomCrop, ToTensor
from data_load_self.FaceLandmarksDataset import FaceLandmarksDataset
import matplotlib.pyplot as plt

face_dataset = FaceLandmarksDataset(root_dir='../data/faces',
                                    landmark_csv='../data/faces/face_landmarks.csv')
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(224)])
fig = plt.figure()
sample = face_dataset[65]

for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)

    plt.imshow(transformed_sample['image'])
    landmarks = transformed_sample['landmarks']
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='.', s=10)
    plt.pause(0.01)
plt.show()








