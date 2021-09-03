#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision import transforms
from data_load_self.transform_tools import Rescale, ToTensor, RandomCrop
from data_load_self.FaceLandmarksDataset import FaceLandmarksDataset

data_loader = FaceLandmarksDataset('../data/faces',
                                 landmark_csv='../data/faces/face_landmarks.csv',
                                 transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))

for i in range(len(data_loader)):
    data = data_loader[i]
    print('image size is {}, landmark size is {}'.format(data['image'].size(),
                                                         data['landmarks'].size()))








