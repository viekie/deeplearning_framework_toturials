#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import os

import torch
import pandas as pd
import numpy as np

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


class FaceLandmarksDataset(Dataset):
    """
    定义自己的face数据及
    """
    def __init__(self, root_dir, landmark_csv, transform=None):
        self.root_dir = root_dir
        self.pd_landmark = pd.read_csv(landmark_csv)
        self.tranform = transform

    def __len__(self):
        return len(self.pd_landmark)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.pd_landmark.iloc[idx, 0]
        abs_image_path = os.path.join(self.root_dir, image_name)
        image = io.imread(abs_image_path)
        landmark = self.pd_landmark.iloc[idx, 1:]
        landmark = np.asarray(landmark, dtype='float')
        landmark = landmark.reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmark}

        if self.tranform:
            sample = self.tranform(sample)
        return sample
