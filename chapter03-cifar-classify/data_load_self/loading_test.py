#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

from __future__ import print_function, division

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt

from data_load_self.FaceLandmarksDataset import FaceLandmarksDataset


def show_image_landmark(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='.', s=10)
    plt.pause(0.01)


face_dataset = FaceLandmarksDataset(root_dir='../data/faces',
                                    landmark_csv='../data/faces/face_landmarks.csv')
print('total data size: {}'.format(len(face_dataset)))


fig = plt.figure()

for i in range(len(face_dataset)):
    data = face_dataset[i]
    image = data['image']
    landmarks = data['landmarks']
    show_image_landmark(**data)
    plt.show()

