#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

from __future__ import print_function, division

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, transform

pd_landmarks = pd.read_csv('../data/faces/face_landmarks.csv')
N = 65
image_name = pd_landmarks.iloc[N, 0]
landmarks = pd_landmarks.iloc[N, 1:]
landmarks = np.asarray(landmarks, dtype=float)
landmarks = landmarks.reshape(-1, 2)


def show_image_with_landmarker(image, landmark):
    plt.imshow(image)
    plt.scatter(landmark[:, 0], landmark[:, 1], c='r', marker='.', s=10)
    plt.pause(0.01)



plt.figure()
image = io.imread(os.path.join('./data/faces', image_name))
show_image_with_landmarker(image, landmarks)
plt.show()
